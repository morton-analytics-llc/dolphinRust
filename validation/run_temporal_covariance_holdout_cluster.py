#!/usr/bin/env python3
"""Execute the frozen GNSS cohort once for EO field acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import requests

if __package__:
    from validation.heldout_temporal_covariance.cohort import canonical_digest, validate_freeze_receipt, validate_manifest
    from validation.heldout_temporal_covariance.executor import run_product_cluster, write_one_shot
    from validation.heldout_temporal_covariance.runner import CohortRunLedger, RUN_IDENTITY_FIELDS, assemble_heldout_receipt, pre_outcome_product_manifest_sha256, product_identity_sha256, validate_product_run_plan
    from validation.heldout_temporal_covariance.scorer import score_receipt
    from validation.score_temporal_covariance_holdout import read_json
else:
    from heldout_temporal_covariance.cohort import canonical_digest, validate_freeze_receipt, validate_manifest
    from heldout_temporal_covariance.executor import run_product_cluster, write_one_shot
    from heldout_temporal_covariance.runner import CohortRunLedger, RUN_IDENTITY_FIELDS, assemble_heldout_receipt, pre_outcome_product_manifest_sha256, product_identity_sha256, validate_product_run_plan
    from heldout_temporal_covariance.scorer import score_receipt
    from score_temporal_covariance_holdout import read_json


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
RUN_PLAN_CAP = 4 * 1024 * 1024
IMPLEMENTATION_SOURCE_HASH_FIELDS = {
    "executor_sha256",
    "runner_sha256",
    "scorer_sha256",
    "runner_cli_sha256",
    "scorer_cli_sha256",
    "cohort_sha256",
    "gps_ground_truth_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=ROOT / "temporal_covariance_heldout_preregistration.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "temporal_covariance_heldout_cohort_manifest.json")
    parser.add_argument("--freeze-receipt", type=Path, default=ROOT / "temporal_covariance_heldout_cohort_freeze_receipt.json")
    parser.add_argument("--run-plan", type=Path)
    parser.add_argument("--rust-batch", type=Path)
    parser.add_argument("--unblind-frozen-outcomes", action="store_true")
    return parser.parse_args()


def validate_run_plan(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Path]:
    return validate_product_run_plan(plan, manifest)


def scorer_source_sha256() -> str:
    scorer_digest = hashlib.sha256()
    for path in (ROOT / "score_temporal_covariance_holdout.py", ROOT / "heldout_temporal_covariance" / "scorer.py"):
        payload = path.read_bytes()
        scorer_digest.update(len(payload).to_bytes(8, "little"))
        scorer_digest.update(payload)
    return scorer_digest.hexdigest()


def implementation_hashes(preregistration: Mapping[str, Any], manifest: Mapping[str, Any], rust_batch: Path, fragments: list[Mapping[str, Any]]) -> dict[str, str]:
    scope_values = {
        name: {
            fragment["cluster_id"]: fragment["difference_covariance"]["scope"][scope]
            for fragment in fragments if fragment["status"] in {"pass", "fail"}
        }
        for name, scope in (
            ("approximation_receipt_sha256", "approximation_receipt_sha256"),
            ("resource_receipt_sha256", "resource_receipt_sha256"),
            ("calibration_scope_receipt_sha256", "method_manifest_sha256"),
            ("review_receipt_sha256", "review_receipt_sha256"),
        )
    }
    values = {
        "binary_sha256": sha256_file(rust_batch),
        "scorer_sha256": scorer_source_sha256(),
        "preregistration_sha256": canonical_digest(preregistration),
        "manifest_sha256": canonical_digest(manifest),
        "factor_scope_sha256": canonical_digest(preregistration["factor_binding"]),
        "gnss_catalog_sha256": canonical_digest({}),
        "operator_sha256": canonical_digest({}),
        "operator_manifest_sha256": canonical_digest({}),
        "persisted_factor_sha256": canonical_digest({}),
        "persisted_factor_manifest_sha256": canonical_digest({}),
        **{name: canonical_digest(value) for name, value in scope_values.items()},
    }
    if set(values) != set(preregistration["receipt_hash_fields"]):
        raise ValueError("EO field-acceptance hashes differ from preregistration")
    return values


def implementation_source_hashes() -> dict[str, str]:
    paths = {
        "executor_sha256": ROOT / "heldout_temporal_covariance" / "executor.py",
        "runner_sha256": ROOT / "heldout_temporal_covariance" / "runner.py",
        "scorer_sha256": ROOT / "heldout_temporal_covariance" / "scorer.py",
        "runner_cli_sha256": Path(__file__).resolve(),
        "scorer_cli_sha256": ROOT / "score_temporal_covariance_holdout.py",
        "cohort_sha256": ROOT / "heldout_temporal_covariance" / "cohort.py",
        "gps_ground_truth_sha256": ROOT / "gps_ground_truth.py",
    }
    if set(paths) != IMPLEMENTATION_SOURCE_HASH_FIELDS:
        raise ValueError("EO field-acceptance source identity fields differ")
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_completed_receipt(
    receipt: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    product_identities: Mapping[str, str],
) -> None:
    if (
        receipt.get("run_identity") != identity
        or receipt.get("run_identity_sha256") != canonical_digest(identity)
    ):
        raise ValueError("persisted receipt has stale run identity")
    clusters = receipt.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("persisted receipt clusters are invalid")
    executable = []
    for fragment in clusters:
        if not isinstance(fragment, Mapping):
            raise ValueError("persisted receipt fragment is invalid")
        if fragment.get("status") == "not_used":
            continue
        cluster_id = fragment.get("cluster_id")
        if (
            cluster_id not in product_identities
            or fragment.get("run_identity_sha256") != canonical_digest(identity)
            or fragment.get("product_identity_sha256")
            != product_identities[cluster_id]
        ):
            raise ValueError("persisted receipt fragment has stale product/run identity")
        if fragment.get("status") in {"pass", "fail"} and (
            fragment.get("estimator", {}).get("binary_sha256")
            != identity["binary_sha256"]
        ):
            raise ValueError("persisted receipt fragment has stale estimator binary")
        executable.append(fragment)
    rebuilt = assemble_heldout_receipt(
        preregistration, manifest, executable, receipt.get("hashes", {})
    )
    actual_core = {
        key: value
        for key, value in receipt.items()
        if key not in {"run_identity", "run_identity_sha256"}
    }
    if rebuilt != actual_core:
        raise ValueError("persisted receipt does not reproduce from cluster fragments")
    hashes = receipt["hashes"]
    exact_hashes = {
        "binary_sha256": identity["binary_sha256"],
        "scorer_sha256": scorer_source_sha256(),
        "preregistration_sha256": canonical_digest(preregistration),
        "manifest_sha256": canonical_digest(manifest),
        "factor_scope_sha256": canonical_digest(preregistration["factor_binding"]),
    }
    if any(hashes.get(field) != value for field, value in exact_hashes.items()):
        raise ValueError("persisted receipt implementation hashes are stale")
    score = score_receipt(preregistration, manifest, receipt)
    if score.get("status") == "fail" and score.get("errors"):
        raise ValueError("persisted receipt fails schema/provenance validation")


def main() -> int:
    args = parse_args()
    if not args.unblind_frozen_outcomes:
        raise SystemExit("--unblind-frozen-outcomes is required")
    if args.run_plan is None or args.rust_batch is None:
        raise SystemExit("--run-plan and --rust-batch are required")
    preregistration = read_json(args.preregistration)
    manifest = read_json(args.manifest)
    freeze_receipt = read_json(args.freeze_receipt)
    validate_freeze_receipt(freeze_receipt, args.manifest, args.preregistration)
    validate_manifest(manifest, preregistration)
    plan = read_json(args.run_plan, RUN_PLAN_CAP)
    product_roots = validate_run_plan(plan, manifest)
    rust_batch = args.rust_batch.resolve(strict=True)
    if not rust_batch.is_file():
        raise SystemExit("Rust batch path is not a file")
    result_directory = REPOSITORY / preregistration["execution"]["result_directory"]
    ledger_path = result_directory / "unblinding_ledger.json"
    receipt_path = result_directory / "temporal_covariance_heldout_result_receipt.json"
    fragment_directory = result_directory / "clusters"
    identity = {
        "generation_id": preregistration["generation_id"],
        "preregistration_sha256": canonical_digest(preregistration),
        "manifest_sha256": canonical_digest(manifest),
        "freeze_receipt_sha256": sha256_file(args.freeze_receipt),
        "run_plan_sha256": canonical_digest(plan),
        "binary_sha256": sha256_file(rust_batch),
        "implementation_source_hashes": implementation_source_hashes(),
        "product_identities_sha256": pre_outcome_product_manifest_sha256(
            product_roots, preregistration
        ),
    }
    if set(identity) != RUN_IDENTITY_FIELDS:
        raise ValueError("EO field-acceptance run identity differs from the acceptance contract")
    ledger = CohortRunLedger.acquire(ledger_path, identity)
    product_identities = {
        cluster_id: product_identity_sha256(root, preregistration)
        for cluster_id, root in sorted(product_roots.items())
    }
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        validate_completed_receipt(
            receipt, preregistration, manifest, identity, product_identities
        )
        ledger.complete(sha256_file(receipt_path))
        ledger.close()
        return 0
    fragment_directory.mkdir(parents=True, exist_ok=True)
    fragments: list[dict[str, Any]] = []

    def execute(candidate: Mapping[str, Any]) -> dict[str, Any]:
        cluster_id = candidate["candidate_id"]
        path = fragment_directory / f"{cluster_id}.json"
        product_identity = product_identity_sha256(
            product_roots[cluster_id], preregistration
        )
        if product_identity != product_identities[cluster_id]:
            raise ValueError("EO field-acceptance product changed after run identity was frozen")
        if path.exists():
            value = read_json(path)
            if value.get("cluster_id") != cluster_id or value.get("manifest_sha256") != identity["manifest_sha256"] or value.get("preregistration_sha256") != identity["preregistration_sha256"] or value.get("freeze_receipt_sha256") != identity["freeze_receipt_sha256"] or value.get("run_identity_sha256") != canonical_digest(identity) or value.get("product_identity_sha256") != product_identity:
                raise ValueError("persisted cluster fragment has stale run identity")
            if value.get("status") in {"pass", "fail"} and value.get("estimator", {}).get("binary_sha256") != identity["binary_sha256"]:
                raise ValueError("persisted cluster fragment has stale estimator binary")
            return value
        value = run_product_cluster(
            manifest, preregistration, cluster_id, product_roots[cluster_id], rust_batch,
            freeze_receipt_sha256=identity["freeze_receipt_sha256"],
            run_identity_sha256=canonical_digest(identity),
            product_identity_sha256=product_identity,
            ngl_session=requests.Session(),
        )
        if product_identity_sha256(product_roots[cluster_id], preregistration) != product_identity:
            raise ValueError("EO field-acceptance product changed during cluster execution")
        write_one_shot(path, value)
        return value

    for candidate in manifest["frozen_clusters"]:
        fragments.append(execute(candidate))
    required_surplus = sum(value["status"] == "not_evaluable" for value in fragments)
    filled = 0
    for candidate in sorted(manifest["surplus_clusters"], key=lambda value: value["candidate_id"]):
        if filled >= required_surplus:
            break
        value = execute(candidate)
        fragments.append(value)
        if value["status"] != "not_evaluable":
            filled += 1
    receipt = assemble_heldout_receipt(
        preregistration, manifest, fragments,
        implementation_hashes(preregistration, manifest, rust_batch, fragments),
    )
    receipt["run_identity"] = identity
    receipt["run_identity_sha256"] = canonical_digest(identity)
    write_one_shot(receipt_path, receipt)
    validate_completed_receipt(
        read_json(receipt_path), preregistration, manifest, identity, product_identities
    )
    ledger.complete(sha256_file(receipt_path))
    ledger.close()
    print(json.dumps({"status": "eo_field_acceptance_receipt_complete", "receipt": str(receipt_path), "executed_clusters": len(fragments)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
