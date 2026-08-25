from __future__ import annotations

import copy
import datetime as dt
import dataclasses
import hashlib
import json
import subprocess
import sys
import struct
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine

from validation.gps_ground_truth import parse_tenv3
from validation.heldout_temporal_covariance.cohort import (
    build_manifest,
    canonical_digest,
    discover_candidates,
)
from validation.heldout_temporal_covariance.executor import (
    Attrition,
    derive_product_observations,
    exact_common_dates,
    fit_slope_with_covariance,
    read_production_difference_factor,
    select_manifest_cluster,
    station_pixel_from_grid,
    station_pair_gnss_observation,
    write_one_shot,
)
from validation.heldout_temporal_covariance.runner import (
    CohortRunLedger,
    RUN_IDENTITY_FIELDS,
    assemble_heldout_receipt,
    pre_outcome_product_manifest_sha256,
    product_identity_sha256,
    run_production_temporal_estimator,
    validate_product_run_plan,
)
from validation.run_temporal_covariance_holdout_cluster import (
    IMPLEMENTATION_SOURCE_HASH_FIELDS,
    implementation_source_hashes,
)


VALIDATION = Path(__file__).parents[1]
PREREGISTRATION = json.loads(
    (VALIDATION / "temporal_covariance_heldout_preregistration.json").read_text()
)
LINE = "{station} {stamp} 2023.0082 59948 2243 3 -99.1 0 {east} 0 {north} 0 {up} 0.0000 0.001 0.002 0.003 0.25 -0.1 0.2 35.0 -120.0 10.0"


def candidate(index: int) -> dict[str, object]:
    query_digest = PREREGISTRATION["candidate_query"]["query_digest"]
    return {
        "candidate_id": f"t001_{index:06d}_iw1_a{index:03d}_b{index:03d}",
        "source_kind": "catalog_metadata",
        "burst_id": f"T001_{index:06d}_IW1",
        "orbit_id": "ascending-r001",
        "footprint_id": f"sha256-{index:064x}",
        "site_id": f"t001_{index:06d}_iw1_a{index:03d}_b{index:03d}",
        "frame_id": f"ascending-r001-burst-t001_{index:06d}_iw1",
        "station_ids": [f"A{index:03d}", f"B{index:03d}"],
        "date_start": "2023-01-01",
        "date_end": "2023-03-19",
        "epoch_count": 12,
        "metadata_hashes": {
            "catalog_sha256": "0" * 64,
            "burst_metadata_sha256": "1" * 64,
            "gnss_station_metadata_sha256": "2" * 64,
        },
        "query_digest": query_digest,
    }


def frozen_manifest() -> dict[str, object]:
    discovery = discover_candidates([candidate(index) for index in range(116)], PREREGISTRATION)
    return build_manifest(discovery, PREREGISTRATION)


def records(station: str, values: list[float]) -> list[object]:
    stamps = [
        (dt.date(2023, 1, 1) + dt.timedelta(days=7 * index)).strftime("%y%b%d").upper()
        for index in range(len(values))
    ]
    return parse_tenv3(
        "\n".join(
            LINE.format(station=station, stamp=stamp, east=value, north=0.0, up=0.0)
            for stamp, value in zip(stamps, values, strict=True)
        )
    )


def string_dataset(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=np.frombuffer(value.encode(), dtype=np.uint8))


def write_factor(root: Path, candidate_value: dict[str, object]) -> tuple[Path, Path, Path, Path]:
    evidence = {
        "approximation_receipt_digest": "referenced_displacement_covariance_approximation_receipt.json",
        "resource_receipt_digest": "referenced_displacement_covariance_resource_receipt.json",
        "review_receipt_digest": "referenced_displacement_covariance_review_receipt.json",
        "method_manifest_digest": "referenced_displacement_covariance_method_manifest.json",
    }
    for file_name in evidence.values():
        (root / file_name).write_text("{}\n")
    for file_name in (
        "referenced_displacement_covariance_approximation_result.json",
        "referenced_displacement_covariance_preregistration.json",
        "referenced_displacement_covariance_design.md",
        "referenced_displacement_covariance_producer_binary",
    ):
        (root / file_name).write_bytes(b"fixture-evidence")
    factor_path = root / "referenced_displacement_covariance_factor.h5"
    with h5py.File(factor_path, "w") as factor:
        factor.attrs.update(
            schema_version=np.uint16(4),
            method_version=np.uint16(1),
            gauge_date_index=np.uint32(0),
            calibration_scope=np.uint16(1),
            complete=np.uint8(1),
        )
        metadata = factor.create_group("metadata")
        string_dataset(metadata, "method", "reference_specific_influence_v1")
        string_dataset(metadata, "burst_id", str(candidate_value["burst_id"]))
        string_dataset(metadata, "units", "millimeters")
        string_dataset(metadata, "crs", "EPSG:4326")
        metadata.attrs.update(reference_row=0, reference_col=1)
        metadata.create_dataset("ordered_date_indices", data=np.arange(12, dtype=np.uint32))
        metadata.create_dataset("acquisition_days", data=np.arange(12) * 7.0)
        metadata.create_dataset("geotransform", data=[-120.5, 0.1, 0.0, 35.5, 0.0, -0.1])
        for name in (
            "mask_digest",
            "source_replay_digest",
            "l2_map_digest",
            "reference_signature_digest",
            "runtime_resource_receipt_digest",
            "source_model_digest",
            "effective_looks_digest",
            "support_digest",
            "correction_order_digest",
            "unwrap_branch_digest",
            "burst_ownership_digest",
            "calibration_scope_digest",
        ):
            string_dataset(metadata, name, "sha256:" + hashlib.sha256(name.encode()).hexdigest())
        for name, file_name in evidence.items():
            string_dataset(
                metadata,
                name,
                "sha256:" + hashlib.sha256((root / file_name).read_bytes()).hexdigest(),
            )
        full_grid = factor.create_group("full_grid")
        full_grid.attrs.update(row_start=0, col_start=0, rows=1, cols=2, stride_y=1, stride_x=1)
        blocks = factor.create_group("blocks")
        block = blocks.create_group("00000000000000000000")
        target_grid = block.create_group("target_grid")
        target_grid.attrs.update(row_start=0, col_start=0, rows=1, cols=2, stride_y=1, stride_x=1)
        block.create_dataset("status", data=np.zeros(2, dtype=np.uint16))
        block.create_dataset("rank_by_target", data=np.ones(2, dtype=np.uint32))
        values = np.zeros((2, 12, 1), dtype=float)
        values[0, 1:, 0] = 2.0
        values[1, 1:, 0] = 1.0
        block.create_dataset("difference_factor", data=values)
    factor_sha = hashlib.sha256(factor_path.read_bytes()).hexdigest()
    manifest_path = root / "referenced_displacement_covariance_provenance.json"
    manifest = {
        "schema_version": 3,
        "method": "reference_specific_influence_v1",
        "method_version": 1,
        "hdf5_file": factor_path.name,
        "hdf5_bytes": factor_path.stat().st_size,
        "hdf5_sha256": factor_sha,
        "burst_id": candidate_value["burst_id"],
        "calibration_scope": "calibrated_scope_match",
    }
    manifest_path.write_text(json.dumps(manifest))
    operator_path = root / "phase_covariance_operator.h5"
    operator_manifest_path = root / "phase_covariance_provenance.json"
    operator_path.write_bytes(b"synthetic excluded-development operator")
    operator_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "sequential_source_dag_v1",
                "method_version": 1,
                "gauge_date_index": 0,
                "hdf5_file": operator_path.name,
                "hdf5_bytes": operator_path.stat().st_size,
                "hdf5_sha256": hashlib.sha256(operator_path.read_bytes()).hexdigest(),
            }
        )
    )
    return factor_path, manifest_path, operator_path, operator_manifest_path


class HeldoutTemporalCovarianceExecutorContract(unittest.TestCase):
    def test_station_pixel_is_derived_from_factor_crs_and_affine(self) -> None:
        geotransform = [-120.5, 0.5, 0.0, 35.5, 0.0, -0.5]
        self.assertEqual(
            station_pixel_from_grid(-120.25, 35.25, "EPSG:4326", geotransform, 2, 2),
            (0, 0),
        )
        self.assertEqual(
            station_pixel_from_grid(-119.75, 34.75, "EPSG:4326", geotransform, 2, 2),
            (1, 1),
        )
        with self.assertRaisesRegex(Attrition, "outside"):
            station_pixel_from_grid(-118.0, 35.0, "EPSG:4326", geotransform, 2, 2)

    def test_product_builder_derives_pixels_los_and_difference_from_cog_bytes(self) -> None:
        geotransform = [-120.5, 0.5, 0.0, 35.5, 0.0, -0.5]
        transform = Affine.from_gdal(*geotransform)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = {
                "contract_version": "fixed-cube-v1",
                "acquisition_days": [0.0, 7.0, 14.0],
                "rows": 2,
                "cols": 2,
                "geotransform": geotransform,
                "epsg": 4326,
                "geometry_source": "CSLC-S1-STATIC",
                "los_rasters": ["los_east.tif", "los_north.tif", "los_up.tif"],
                "acquisition_days_sha256": "sha256:"
                + hashlib.sha256(
                    b"".join(struct.pack("<d", value) for value in [0.0, 7.0, 14.0])
                ).hexdigest(),
                "velocity_estimator": "linear_post_gauge_unit_precision",
                "inference_status": "conditional_only",
                "corrected_velocity_raster": None,
                "corrected_sigma_raster": None,
                "validity_mask_raster": "velocity_validity_mask.tif",
                "geometry_provenance": "geometry_provenance.json",
                "reference_point": [1, 1],
            }
            (root / "fixed_cube_receipt.json").write_text(json.dumps(receipt))
            (root / "geometry_provenance.json").write_text("{}\n")

            def write_tif(name: str, values: np.ndarray) -> None:
                with rasterio.open(
                    root / name,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=1,
                    dtype="float32",
                    crs="EPSG:4326",
                    transform=transform,
                ) as dataset:
                    dataset.write(values.astype(np.float32), 1)

            write_tif("los_east.tif", np.zeros((2, 2)))
            write_tif("los_north.tif", np.zeros((2, 2)))
            write_tif("los_up.tif", np.ones((2, 2)))
            write_tif("velocity_validity_mask.tif", np.ones((2, 2)))
            write_tif("displacement_00.tif", np.array([[0.001, 0.0], [0.0, 0.0]]))
            write_tif("displacement_01.tif", np.array([[0.002, 0.0], [0.0, 0.0]]))
            first = [dataclasses.replace(value, longitude=-120.25, latitude=35.25) for value in records("A000", [0.0] * 3)]
            second = [dataclasses.replace(value, longitude=-119.75, latitude=34.75) for value in records("B000", [0.0] * 3)]
            result = derive_product_observations(
                root,
                candidate(0),
                {"A000": first, "B000": second},
                {
                    "acquisition_days": [0.0, 7.0, 14.0],
                    "full_grid": {"row_start": 0, "col_start": 0, "rows": 2, "cols": 2, "stride_y": 1, "stride_x": 1},
                    "geotransform": geotransform,
                    "crs": "EPSG:4326",
                    "units": "meters",
                },
                PREREGISTRATION,
            )
            self.assertEqual(result["station_pixels"], {"A000": (0, 0), "B000": (1, 1)})
            np.testing.assert_allclose(result["insar_difference_mm"], [0.0, 1.0, 2.0], atol=1e-6)
            np.testing.assert_array_equal(result["station_los"]["A000"], [0.0, 0.0, 1.0])

    def test_product_identity_binds_every_local_input_and_rejects_external_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "product"
            root.mkdir()
            required = {
                "fixed_cube_receipt.json": b"{}",
                "los_east.tif": b"east",
                "los_north.tif": b"north",
                "los_up.tif": b"up",
                "velocity_validity_mask.tif": b"mask",
                "geometry_provenance.json": b"{}",
                "displacement_00.tif": b"displacement",
                "phase_covariance_operator.h5": b"operator",
                "phase_covariance_provenance.json": b"{}",
                "referenced_displacement_covariance_factor.h5": b"factor",
                "referenced_displacement_covariance_provenance.json": b"{}",
                "referenced_displacement_covariance_approximation_receipt.json": b"{}",
                "referenced_displacement_covariance_resource_receipt.json": b"{}",
                "referenced_displacement_covariance_review_receipt.json": b"{}",
                "referenced_displacement_covariance_method_manifest.json": b"{}",
                "referenced_displacement_covariance_approximation_result.json": b"{}",
                "referenced_displacement_covariance_preregistration.json": b"{}",
                "referenced_displacement_covariance_design.md": b"design",
                "referenced_displacement_covariance_producer_binary": b"binary",
            }
            for name, payload in required.items():
                (root / name).write_bytes(payload)
            first = product_identity_sha256(root, PREREGISTRATION)
            (root / "displacement_00.tif").write_bytes(b"changed")
            self.assertNotEqual(first, product_identity_sha256(root, PREREGISTRATION))
            external = Path(directory) / "external"
            external.write_bytes(b"external")
            (root / "los_up.tif").unlink()
            (root / "los_up.tif").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "external"):
                product_identity_sha256(root, PREREGISTRATION)

    def test_immutable_ledger_allows_only_exact_identity_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            identity = {"generation_id": "f53-06-v1", "run_sha256": "a" * 64}
            ledger = CohortRunLedger.acquire(path, identity)
            self.assertEqual(ledger.payload["state"], "running")
            with self.assertRaisesRegex(PermissionError, "active"):
                CohortRunLedger.acquire(path, identity)
            ledger.close()
            ledger = CohortRunLedger.acquire(path, identity)
            with self.assertRaisesRegex(PermissionError, "identity"):
                ledger.close()
                CohortRunLedger.acquire(
                    path,
                    {"generation_id": "f53-06-v1", "run_sha256": "b" * 64},
                )
            ledger = CohortRunLedger.acquire(path, identity)
            ledger.complete("c" * 64)
            ledger.close()
            with self.assertRaisesRegex(PermissionError, "completed"):
                CohortRunLedger.acquire(path, identity)

    def test_run_plan_requires_exact_unique_96_plus_20_product_roots(self) -> None:
        manifest = frozen_manifest()
        candidates = manifest["frozen_clusters"] + manifest["surplus_clusters"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clusters = []
            for value in candidates:
                product = root / value["candidate_id"]
                product.mkdir()
                clusters.append(
                    {
                        "cluster_id": value["candidate_id"],
                        "product_directory": str(product),
                    }
                )
            plan = {
                "schema": "dolphinrust.temporal_covariance.heldout_run_plan",
                "schema_version": 1,
                "clusters": clusters,
            }
            self.assertEqual(len(validate_product_run_plan(plan, manifest)), 116)
            missing = copy.deepcopy(plan)
            missing["clusters"].pop()
            with self.assertRaisesRegex(ValueError, "exact 96\+20"):
                validate_product_run_plan(missing, manifest)
            reused = copy.deepcopy(plan)
            reused["clusters"][1]["product_directory"] = reused["clusters"][0][
                "product_directory"
            ]
            with self.assertRaisesRegex(ValueError, "reuses"):
                validate_product_run_plan(reused, manifest)

    def test_pre_outcome_product_manifest_matches_pr82_run_identity_contract(self) -> None:
        expected_fields = {
            "generation_id",
            "preregistration_sha256",
            "manifest_sha256",
            "freeze_receipt_sha256",
            "run_plan_sha256",
            "binary_sha256",
            "implementation_source_hashes",
            "product_identities_sha256",
        }
        self.assertEqual(RUN_IDENTITY_FIELDS, expected_fields)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            product_roots = {"cluster-b": second, "cluster-a": first}
            digest = pre_outcome_product_manifest_sha256(
                product_roots, PREREGISTRATION
            )
            self.assertEqual(
                digest,
                pre_outcome_product_manifest_sha256(
                    {"cluster-a": first, "cluster-b": second}, PREREGISTRATION
                ),
            )
            self.assertEqual(len(digest), 64)
            self.assertFalse(any(first.iterdir()))
            self.assertFalse(any(second.iterdir()))
            moved = root / "moved"
            second.rename(moved)
            self.assertNotEqual(
                digest,
                pre_outcome_product_manifest_sha256(
                    {"cluster-a": first, "cluster-b": moved}, PREREGISTRATION
                ),
            )

    def test_source_identity_matches_pr82_exact_seven_entry_contract(self) -> None:
        expected_paths = {
            "executor_sha256": VALIDATION
            / "heldout_temporal_covariance"
            / "executor.py",
            "runner_sha256": VALIDATION
            / "heldout_temporal_covariance"
            / "runner.py",
            "scorer_sha256": VALIDATION
            / "heldout_temporal_covariance"
            / "scorer.py",
            "runner_cli_sha256": VALIDATION
            / "run_temporal_covariance_holdout_cluster.py",
            "scorer_cli_sha256": VALIDATION
            / "score_temporal_covariance_holdout.py",
            "cohort_sha256": VALIDATION
            / "heldout_temporal_covariance"
            / "cohort.py",
            "gps_ground_truth_sha256": VALIDATION / "gps_ground_truth.py",
        }
        self.assertEqual(IMPLEMENTATION_SOURCE_HASH_FIELDS, set(expected_paths))
        self.assertEqual(
            implementation_source_hashes(),
            {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in expected_paths.items()
            },
        )

    def test_attrited_surplus_is_retained_and_next_lexical_surplus_fills(self) -> None:
        manifest = frozen_manifest()
        primary = manifest["frozen_clusters"]
        surplus = sorted(manifest["surplus_clusters"], key=lambda value: value["candidate_id"])
        fragments = []
        for index, value in enumerate(primary):
            fragments.append({
                "cluster_id": value["candidate_id"], "station_ids": value["station_ids"],
                "burst_id": value["burst_id"], "site_id": value["site_id"],
                "status": "not_evaluable" if index == 0 else "pass",
                **({"reason_code": "gnss_solution_missing"} if index == 0 else {}),
            })
        for index, value in enumerate(surplus[:2]):
            fragments.append({
                "cluster_id": value["candidate_id"], "station_ids": value["station_ids"],
                "burst_id": value["burst_id"], "site_id": value["site_id"],
                "status": "not_evaluable" if index == 0 else "pass",
                **({"reason_code": "gnss_solution_missing"} if index == 0 else {}),
            })
        receipt = assemble_heldout_receipt(
            PREREGISTRATION, manifest, fragments,
            {field: "0" * 64 for field in PREREGISTRATION["receipt_hash_fields"]},
        )
        by_id = {value["cluster_id"]: value for value in receipt["clusters"]}
        self.assertEqual(by_id[surplus[0]["candidate_id"]]["status"], "not_evaluable")
        self.assertEqual(receipt["attrition"]["used_surplus_ids"], [surplus[1]["candidate_id"]])

    def test_exact_receipt_assembles_primary_then_lexical_surplus(self) -> None:
        manifest = frozen_manifest()
        primaries = manifest["frozen_clusters"]
        surplus = sorted(manifest["surplus_clusters"], key=lambda value: value["candidate_id"])
        fragments = []
        for index, value in enumerate(primaries):
            fragments.append(
                {
                    "cluster_id": value["candidate_id"],
                    "station_ids": value["station_ids"],
                    "burst_id": value["burst_id"],
                    "site_id": value["site_id"],
                    "status": "not_evaluable" if index == 0 else "pass",
                    **({"reason_code": "gnss_solution_missing"} if index == 0 else {}),
                }
            )
        fragments.append(
            {
                "cluster_id": surplus[0]["candidate_id"],
                "station_ids": surplus[0]["station_ids"],
                "burst_id": surplus[0]["burst_id"],
                "site_id": surplus[0]["site_id"],
                "status": "pass",
            }
        )
        receipt = assemble_heldout_receipt(
            PREREGISTRATION,
            manifest,
            fragments,
            {field: "0" * 64 for field in PREREGISTRATION["receipt_hash_fields"]},
        )
        self.assertEqual(len(receipt["clusters"]), 116)
        self.assertEqual(receipt["attrition"]["used_surplus_ids"], [surplus[0]["candidate_id"]])
        self.assertTrue(
            all(
                value["status"] == "not_used"
                for value in receipt["clusters"]
                if value["cluster_id"] in {item["candidate_id"] for item in surplus[1:]}
            )
        )

    def test_temporal_estimator_uses_complete_refit_output_and_hashes_binary(self) -> None:
        script = """#!/usr/bin/env python3
import json, sys
request=json.loads(sys.stdin.readline())
fit={"status":"evaluated","bootstrap_attempts":200,"bootstrap_successes":200,
"complete_refit_bootstrap":{"status":"evaluated","point_estimate":0.01,
"standard_error_diagnostic":0.002,"attempted_replicates":200,"successful_replicates":200},
"conditional_wls":{"status":"evaluated","standard_error_diagnostic":0.001}}
print(json.dumps({"schema":"dolphinrust-temporal-covariance-batch/4",
"execution_path":"fixed_factor","fixed_factor_status":"evaluated","emitted":True,
"failed":False,"fit":fit,"resource":{"wall_micros":1,"resident_set_bytes_before":1,
"resident_set_bytes_after":1}}))
"""
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "batch"
            binary.write_text(script)
            binary.chmod(0o755)
            result = run_production_temporal_estimator(
                binary,
                "cluster-1",
                [0.0, 7.0, 14.0],
                [0.0, 1.0, 2.0],
                np.diag([0.0, 1.0, 1.0]),
                PREREGISTRATION,
            )
            self.assertEqual(result["method"], "complete_refit_bootstrap")
            self.assertAlmostEqual(result["slope_mm_year"], 3.6525)
            self.assertAlmostEqual(result["slope_variance"], (0.002 * 365.25) ** 2)
            self.assertAlmostEqual(result["baseline_sigma"], 0.001 * 365.25)
            self.assertEqual(result["binary_sha256"], hashlib.sha256(binary.read_bytes()).hexdigest())

    def test_cli_requires_explicit_unblinding_before_any_input_read(self) -> None:
        script = VALIDATION / "run_temporal_covariance_holdout_cluster.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                sys.executable,
                str(script),
                "--run-plan",
                str(root / "missing-plan.json"),
                "--rust-batch",
                str(root / "missing-batch"),
            ]
            gated = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(gated.returncode, 0)
            self.assertIn("--unblind-frozen-outcomes is required", gated.stderr)

            existing = subprocess.run(
                [*command, "--unblind-frozen-outcomes"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(existing.returncode, 0)
            self.assertNotIn("--unblind-frozen-outcomes is required", existing.stderr)

    def test_exact_cluster_and_common_epoch_contract(self) -> None:
        manifest = frozen_manifest()
        selected = select_manifest_cluster(manifest, PREREGISTRATION, candidate(0)["candidate_id"])
        self.assertEqual(selected["candidate_id"], candidate(0)["candidate_id"])
        with self.assertRaisesRegex(ValueError, "frozen"):
            select_manifest_cluster(manifest, PREREGISTRATION, "not-frozen")

        dates = [dt.date(2023, 1, 1) + dt.timedelta(days=7 * index) for index in range(12)]
        station_records = {"A000": records("A000", list(range(12))), "B000": records("B000", list(range(12)))}
        self.assertEqual(exact_common_dates(selected, dates, station_records, PREREGISTRATION), dates)
        with self.assertRaisesRegex(Attrition, "insufficient_dates"):
            exact_common_dates(selected, dates, {"A000": station_records["A000"][:11], "B000": station_records["B000"]}, PREREGISTRATION)

    def test_full_enu_covariance_projects_into_station_pair_slope_variance(self) -> None:
        dates = [dt.date(2023, 1, 1) + dt.timedelta(days=7 * index) for index in range(12)]
        first = records("A000", [index / 1000 for index in range(12)])
        second = records("B000", [0.0] * 12)
        result = station_pair_gnss_observation(first, second, dates, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(result["slope_mm_year"], 52.17857142857143)
        self.assertGreater(result["slope_variance"], 0.0)
        self.assertEqual(result["covariance_projection"], "u_transpose_C_u")

        fitted = fit_slope_with_covariance(dates, np.arange(12, dtype=float), np.eye(12))
        self.assertAlmostEqual(fitted["slope"], 365.25 / 7.0)
        self.assertGreater(fitted["variance"], 0.0)

    def test_production_factor_identity_and_atomic_no_replace(self) -> None:
        candidate_value = candidate(0)
        dates = [dt.date(2023, 1, 1) + dt.timedelta(days=7 * index) for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factor, factor_manifest, operator, operator_manifest = write_factor(root, candidate_value)
            result = read_production_difference_factor(
                factor,
                factor_manifest,
                candidate_value,
                (0, 0),
                (0, 1),
                dates,
                PREREGISTRATION,
                operator,
                operator_manifest,
            )
            self.assertEqual(result["covariance"].shape, (12, 12))
            self.assertAlmostEqual(result["covariance"][1, 1], 4.0)
            self.assertEqual(result["binding"]["scope"]["schema_version"], 4)
            self.assertEqual(result["binding"]["calibrated_scope_match"], "calibrated_scope_match")

            with h5py.File(factor, "r+") as product:
                product.attrs["calibration_scope"] = np.uint16(0)
            changed_hash = hashlib.sha256(factor.read_bytes()).hexdigest()
            factor_manifest_payload = json.loads(factor_manifest.read_text())
            factor_manifest_payload["hdf5_sha256"] = changed_hash
            factor_manifest_payload["hdf5_bytes"] = factor.stat().st_size
            factor_manifest.write_text(json.dumps(factor_manifest_payload))
            with self.assertRaisesRegex(Attrition, "difference_covariance_uncalibrated"):
                read_production_difference_factor(
                    factor,
                    factor_manifest,
                    candidate_value,
                    (0, 0),
                    (0, 1),
                    dates,
                    PREREGISTRATION,
                    operator,
                    operator_manifest,
                )

            output = root / "cluster.json"
            payload = {"schema": "synthetic", "outcomes_present": True}
            write_one_shot(output, payload)
            first = output.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "one-shot"):
                write_one_shot(output, copy.deepcopy(payload))
            self.assertEqual(output.read_bytes(), first)

    def test_production_factor_rejects_tampered_calibration_evidence(self) -> None:
        candidate_value = candidate(0)
        dates = [
            dt.date(2023, 1, 1) + dt.timedelta(days=7 * index)
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factor, factor_manifest, operator, operator_manifest = write_factor(
                root, candidate_value
            )
            (root / "referenced_displacement_covariance_review_receipt.json").write_text(
                '{"tampered":true}\n'
            )
            with self.assertRaisesRegex(Attrition, "evidence hash differs"):
                read_production_difference_factor(
                    factor,
                    factor_manifest,
                    candidate_value,
                    (0, 0),
                    (0, 1),
                    dates,
                    PREREGISTRATION,
                    operator,
                    operator_manifest,
                )

    def test_trusted_single_cluster_executor_is_not_public(self) -> None:
        module = __import__(
            "validation.heldout_temporal_covariance.executor",
            fromlist=["run_one_cluster"],
        )
        self.assertFalse(hasattr(module, "run_one_cluster"))


if __name__ == "__main__":
    unittest.main()
