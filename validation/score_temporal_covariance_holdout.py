#!/usr/bin/env python3
"""Validate and score one GNSS receipt bundle for EO field acceptance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from validation.heldout_temporal_covariance.cohort import canonical_digest, validate_freeze_receipt, validate_manifest
    from validation.heldout_temporal_covariance.executor import read_production_difference_factor, write_one_shot
    from validation.heldout_temporal_covariance.runner import product_identity_sha256, validate_product_run_plan
    from validation.heldout_temporal_covariance.scorer import score_receipt
else:
    from heldout_temporal_covariance.cohort import canonical_digest, validate_freeze_receipt, validate_manifest
    from heldout_temporal_covariance.executor import read_production_difference_factor, write_one_shot
    from heldout_temporal_covariance.runner import product_identity_sha256, validate_product_run_plan
    from heldout_temporal_covariance.scorer import score_receipt


JSON_CAP = 16 * 1024 * 1024
MANIFEST_CAP = 1024 * 1024
FACTOR_CAP = 1024 * 1024 * 1024
ROOT = Path(__file__).resolve().parent


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_bounded(path: Path, byte_cap: int) -> bytes:
    before = path.stat()
    if before.st_size > byte_cap:
        raise ValueError(f"{path.name} exceeds its byte cap")
    with path.open("rb") as handle:
        opened = path.stat()
        payload = handle.read(byte_cap + 1)
        after = path.stat()
    if len(payload) > byte_cap:
        raise ValueError(f"{path.name} exceeds its byte cap")
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise ValueError(f"{path.name} changed while it was read")
    return payload


def read_json(path: Path, byte_cap: int = JSON_CAP) -> dict[str, Any]:
    value = json.loads(
        read_bounded(path, byte_cap).decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number: {value}")),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def current_implementation_source_hashes() -> dict[str, str]:
    paths = {
        "executor_sha256": ROOT / "heldout_temporal_covariance" / "executor.py",
        "runner_sha256": ROOT / "heldout_temporal_covariance" / "runner.py",
        "scorer_sha256": ROOT / "heldout_temporal_covariance" / "scorer.py",
        "runner_cli_sha256": ROOT / "run_temporal_covariance_holdout_cluster.py",
        "scorer_cli_sha256": Path(__file__).resolve(),
        "cohort_sha256": ROOT / "heldout_temporal_covariance" / "cohort.py",
        "gps_ground_truth_sha256": ROOT / "gps_ground_truth.py",
    }
    return {
        name: hashlib.sha256(read_bounded(path, JSON_CAP)).hexdigest()
        for name, path in paths.items()
    }


def bind_factor_files(receipt: dict[str, Any], factor_path: Path, manifest_path: Path) -> None:
    hashes = receipt.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError("EO field-acceptance receipt is missing artifact hashes")
    factor = read_bounded(factor_path, FACTOR_CAP)
    manifest_bytes = read_bounded(manifest_path, MANIFEST_CAP)
    factor_sha256 = sha256(factor)
    manifest_sha256 = sha256(manifest_bytes)
    if factor_sha256 != hashes.get("persisted_factor_sha256"):
        raise ValueError("EO field-acceptance factor hash differs from supplied HDF5")
    if manifest_sha256 != hashes.get("persisted_factor_manifest_sha256"):
        raise ValueError("EO field-acceptance factor-manifest hash differs from supplied JSON")
    manifest = json.loads(
        manifest_bytes.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number: {value}")),
    )
    required = {
        "schema_version": 3,
        "method": "reference_specific_influence_v1",
        "method_version": 1,
        "hdf5_file": "referenced_displacement_covariance_factor.h5",
        "hdf5_bytes": len(factor),
        "hdf5_sha256": factor_sha256,
        "calibration_scope": "calibrated_scope_match",
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("supplied factor provenance is not the calibrated production v4 artifact")


def bind_product_files(
    receipt: dict[str, Any],
    preregistration: dict[str, Any],
    manifest: dict[str, Any],
    product_roots: dict[str, Path],
) -> None:
    candidates = {
        value["candidate_id"]: value
        for value in manifest["frozen_clusters"] + manifest["surplus_clusters"]
    }
    clusters = receipt.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("EO field-acceptance receipt clusters are invalid")
    run_identity = receipt.get("run_identity")
    run_identity_sha256 = receipt.get("run_identity_sha256")
    if (
        not isinstance(run_identity, dict)
        or run_identity_sha256 != canonical_digest(run_identity)
    ):
        raise ValueError("EO field-acceptance receipt run identity is invalid")
    current_products = {
        cluster_id: product_identity_sha256(root, preregistration)
        for cluster_id, root in sorted(product_roots.items())
    }
    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise ValueError("EO field-acceptance receipt cluster is invalid")
        cluster_id = cluster.get("cluster_id")
        if cluster_id not in product_roots:
            raise ValueError("EO field-acceptance cluster is absent from the run plan")
        if cluster.get("status") == "not_used":
            continue
        actual_product_identity = current_products[cluster_id]
        if (
            cluster.get("run_identity_sha256") != run_identity_sha256
            or cluster.get("product_identity_sha256") != actual_product_identity
        ):
            raise ValueError("EO field-acceptance product identity differs from current bytes")
        if cluster.get("status") not in {"pass", "fail"}:
            continue
        if (
            cluster.get("estimator", {}).get("binary_sha256")
            != run_identity.get("binary_sha256")
        ):
            raise ValueError("EO field-acceptance estimator binary is stale")
        binding = cluster.get("difference_covariance")
        if not isinstance(binding, dict) or not isinstance(binding.get("scope"), dict):
            raise ValueError("evaluable cluster factor binding is missing")
        scope = binding["scope"]
        common_dates = [dt.date.fromisoformat(value) for value in cluster["common_dates"]]
        factor_spec = preregistration["factor_binding"]
        root = product_roots[cluster_id]
        actual = read_production_difference_factor(
            root / factor_spec["output_factor"]["artifact_hdf5"],
            root / factor_spec["output_factor"]["artifact_manifest"],
            candidates[cluster_id],
            tuple(scope["target_station_pixel"]),
            tuple(scope["control_station_pixel"]),
            common_dates,
            preregistration,
            root / factor_spec["input_operator"]["artifact_hdf5"],
            root / factor_spec["input_operator"]["artifact_manifest"],
        )
        if actual["binding"] != binding:
            raise ValueError("EO field-acceptance factor binding differs from current bytes")


def build_scored_result(
    score: dict[str, Any],
    receipt: dict[str, Any],
    preregistration: dict[str, Any],
    manifest: dict[str, Any],
    *,
    receipt_file_sha256: str,
    manifest_file_sha256: str,
) -> dict[str, Any]:
    required_score_fields = {
        "status",
        "errors",
        "levels",
        "evaluated_clusters",
        "emission_rate",
        "attrited_primary_ids",
        "used_surplus_ids",
        "unused_surplus_ids",
        "reasons_by_cluster",
    }
    if set(score) != required_score_fields | {"holm"}:
        raise ValueError("EO field-acceptance scorer output differs from the acceptance schema")
    return {
        "schema": "eo.temporal_covariance.field_acceptance_score",
        "schema_version": 1,
        "cohort_id": preregistration["cohort_id"],
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": canonical_digest(manifest),
        "freeze_receipt_sha256": receipt["run_identity"][
            "freeze_receipt_sha256"
        ],
        "factor_scope_sha256": receipt["hashes"]["factor_scope_sha256"],
        "heldout_receipt_sha256": receipt_file_sha256,
        "primary_cluster_count": len(manifest["frozen_clusters"]),
        "surplus_cluster_count": len(manifest["surplus_clusters"]),
        **{field: score[field] for field in required_score_fields},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        preregistration = read_json(args.prereg)
        expected_output = (
            ROOT.parent
            / preregistration["execution"]["result_directory"]
            / preregistration["execution"]["scored_result"]
        ).resolve()
        if args.output.resolve() != expected_output:
            raise ValueError("EO field-acceptance score path is not the frozen acceptance path")
        manifest = read_json(args.manifest)
        freeze_receipt = read_json(args.freeze_receipt)
        run_plan = read_json(args.run_plan)
        receipt = read_json(args.receipt)
        validate_freeze_receipt(freeze_receipt, args.manifest, args.prereg)
        validate_manifest(manifest, preregistration)
        product_roots = validate_product_run_plan(run_plan, manifest)
        run_identity = receipt.get("run_identity")
        if (
            not isinstance(run_identity, dict)
            or receipt.get("run_identity_sha256") != canonical_digest(run_identity)
            or run_identity.get("run_plan_sha256") != canonical_digest(run_plan)
            or run_identity.get("freeze_receipt_sha256")
            != hashlib.sha256(read_bounded(args.freeze_receipt, JSON_CAP)).hexdigest()
            or run_identity.get("implementation_source_hashes")
            != current_implementation_source_hashes()
        ):
            raise ValueError("EO field-acceptance receipt run identity is stale")
        bind_product_files(receipt, preregistration, manifest, product_roots)
        result = build_scored_result(
            score_receipt(preregistration, manifest, receipt),
            receipt,
            preregistration,
            manifest,
            receipt_file_sha256=hashlib.sha256(
                read_bounded(args.receipt, JSON_CAP)
            ).hexdigest(),
            manifest_file_sha256=hashlib.sha256(
                read_bounded(args.manifest, JSON_CAP)
            ).hexdigest(),
        )
        write_one_shot(args.output, result)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return {"pass": 0, "fail": 2, "not_evaluable": 3}.get(result["status"], 4)


if __name__ == "__main__":
    raise SystemExit(main())
