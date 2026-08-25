from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np
import requests

from validation.gps_ground_truth import parse_tenv3
from validation.heldout_temporal_covariance.cohort import (
    build_manifest,
    canonical_digest,
    discover_candidates,
)
from validation.heldout_temporal_covariance.executor import (
    Attrition,
    exact_common_dates,
    fit_slope_with_covariance,
    read_production_difference_factor,
    run_one_cluster,
    select_manifest_cluster,
    station_pair_gnss_observation,
    write_one_shot,
)
from validation.tests.test_remote_hdf5_crop import TOKEN, range_server


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


def tenv3_text(station: str, values: list[float]) -> str:
    stamps = [
        (dt.date(2023, 1, 1) + dt.timedelta(days=7 * index)).strftime("%y%b%d").upper()
        for index in range(len(values))
    ]
    return "\n".join(
        LINE.format(station=station, stamp=stamp, east=value, north=0.0, up=0.0)
        for stamp, value in zip(stamps, values, strict=True)
    )


@contextmanager
def ngl_server(payloads: dict[str, str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            payload = payloads.get(self.path)
            if payload is None:
                self.send_response(404)
                self.end_headers()
                return
            content = payload.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def string_dataset(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=np.frombuffer(value.encode(), dtype=np.uint8))


def write_factor(root: Path, candidate_value: dict[str, object]) -> tuple[Path, Path, Path, Path]:
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
        ):
            string_dataset(metadata, name, "sha256:" + hashlib.sha256(name.encode()).hexdigest())
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


def write_static(path: Path) -> None:
    with h5py.File(path, "w") as product:
        data = product.create_group("data")
        data.create_dataset("los_east", data=np.full((3, 3), 0.1, dtype=np.float32))
        data.create_dataset("los_north", data=np.full((3, 3), 0.2, dtype=np.float32))
        data.create_dataset("x_coordinates", data=np.array([-120.1, -120.0, -119.9]))
        data.create_dataset("y_coordinates", data=np.array([34.9, 35.0, 35.1]))
        data.create_dataset("projection", data=np.int32(4326))


class HeldoutTemporalCovarianceExecutorContract(unittest.TestCase):
    def test_cli_requires_explicit_unblinding_and_absent_aggregate_before_io(self) -> None:
        script = VALIDATION / "run_temporal_covariance_holdout_cluster.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                sys.executable,
                str(script),
                "--input",
                str(root / "missing-input.json"),
                "--output",
                str(root / "fragment.json"),
                "--aggregate-output",
                str(root / "aggregate.json"),
            ]
            gated = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(gated.returncode, 0)
            self.assertIn("--unblind-frozen-outcomes is required", gated.stderr)

            (root / "aggregate.json").write_text("already unblinded")
            existing = subprocess.run(
                [*command, "--unblind-frozen-outcomes"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn("aggregate outcome artifact already exists", existing.stderr)

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

    def test_local_http_one_shot_executor_and_registered_attrition(self) -> None:
        manifest = frozen_manifest()
        candidate_value = candidate(0)
        freeze_sha = "f" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static_source = root / "static.h5"
            write_static(static_source)
            factor, factor_manifest, operator, operator_manifest = write_factor(
                root, candidate_value
            )
            dates = [
                (dt.date(2023, 1, 1) + dt.timedelta(days=7 * index)).isoformat()
                for index in range(12)
            ]
            static_session = requests.Session()
            static_session.headers["Authorization"] = f"Bearer {TOKEN}"
            with range_server(static_source) as (static_url, _), ngl_server(
                {
                    "/A000.tenv3": tenv3_text("A000", [index / 1000 for index in range(12)]),
                    "/B000.tenv3": tenv3_text("B000", [0.0] * 12),
                }
            ) as base_url:
                spec = {
                    "schema": "dolphinrust.temporal_covariance.heldout_cluster_input",
                    "schema_version": 1,
                    "cluster_id": candidate_value["candidate_id"],
                    "manifest_sha256": canonical_digest(manifest),
                    "freeze_receipt_sha256": freeze_sha,
                    "insar_source_sha256": "a" * 64,
                    "estimator_receipt_sha256": "b" * 64,
                    "acquisition_dates": dates,
                    "station_pixels": {"A000": [0, 0], "B000": [0, 1]},
                    "insar_difference_mm": list(range(12)),
                    "baseline_sigma": {"68": 10.0, "90": 10.0, "95": 10.0},
                    "static_source": {
                        "url": static_url,
                        "file_name": static_source.name,
                        "windows": {
                            "A000": {"row0": 0, "col0": 0, "height": 3, "width": 3},
                            "B000": {"row0": 0, "col0": 0, "height": 3, "width": 3},
                        },
                        "maximum_transfer_bytes_per_station": 1_000_000,
                    },
                    "factor": {
                        "hdf5_path": str(factor),
                        "manifest_path": str(factor_manifest),
                        "operator_path": str(operator),
                        "operator_manifest_path": str(operator_manifest),
                    },
                }
                output = root / "fragment.json"
                aggregate = root / "aggregate.json"
                fragment = run_one_cluster(
                    manifest,
                    PREREGISTRATION,
                    spec,
                    output,
                    aggregate,
                    allow_one_shot_unblinding=True,
                    freeze_receipt_sha256=freeze_sha,
                    static_session=static_session,
                    ngl_base_url=base_url,
                )
                self.assertEqual(fragment["status"], "evaluable")
                self.assertEqual(len(fragment["common_dates"]), 12)
                self.assertEqual(
                    set(fragment["station_pair_provenance"]["station_los_vectors"]),
                    {"A000", "B000"},
                )
                self.assertNotIn(TOKEN, output.read_text())
                with self.assertRaisesRegex(FileExistsError, "one-shot"):
                    run_one_cluster(
                        manifest,
                        PREREGISTRATION,
                        spec,
                        output,
                        aggregate,
                        allow_one_shot_unblinding=True,
                        freeze_receipt_sha256=freeze_sha,
                        static_session=static_session,
                        ngl_base_url=base_url,
                    )

            with ngl_server(
                {
                    "/A000.tenv3": tenv3_text("A000", [index / 1000 for index in range(11)]),
                    "/B000.tenv3": tenv3_text("B000", [0.0] * 12),
                }
            ) as base_url:
                attrition_output = root / "attrition.json"
                attrition = run_one_cluster(
                    manifest,
                    PREREGISTRATION,
                    spec,
                    attrition_output,
                    root / "aggregate-two.json",
                    allow_one_shot_unblinding=True,
                    freeze_receipt_sha256=freeze_sha,
                    static_session=static_session,
                    ngl_base_url=base_url,
                )
                self.assertEqual(attrition["status"], "not_evaluable")
                self.assertEqual(attrition["reason_code"], "insufficient_dates")


if __name__ == "__main__":
    unittest.main()
