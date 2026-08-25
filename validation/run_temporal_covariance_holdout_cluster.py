#!/usr/bin/env python3
"""Execute one frozen #53 cluster after explicit one-shot unblinding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

if __package__:
    from validation.fetch_real import authenticated_session
    from validation.heldout_temporal_covariance.cohort import validate_freeze_receipt
    from validation.heldout_temporal_covariance.executor import run_one_cluster
    from validation.score_temporal_covariance_holdout import read_json
else:
    from fetch_real import authenticated_session
    from heldout_temporal_covariance.cohort import validate_freeze_receipt
    from heldout_temporal_covariance.executor import run_one_cluster
    from score_temporal_covariance_holdout import read_json


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT / "temporal_covariance_heldout_preregistration.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "temporal_covariance_heldout_cohort_manifest.json",
    )
    parser.add_argument(
        "--freeze-receipt",
        type=Path,
        default=ROOT / "temporal_covariance_heldout_cohort_freeze_receipt.json",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--unblind-frozen-outcomes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.unblind_frozen_outcomes:
        raise SystemExit("--unblind-frozen-outcomes is required")
    if args.output.exists():
        raise SystemExit(f"one-shot output already exists: {args.output}")
    if args.aggregate_output.exists():
        raise SystemExit(
            f"aggregate outcome artifact already exists: {args.aggregate_output}"
        )
    preregistration = read_json(args.preregistration)
    manifest = read_json(args.manifest)
    freeze_receipt = read_json(args.freeze_receipt)
    validate_freeze_receipt(
        freeze_receipt, args.manifest, args.preregistration
    )
    input_spec = read_json(args.input)
    freeze_receipt_sha256 = hashlib.sha256(args.freeze_receipt.read_bytes()).hexdigest()
    fragment = run_one_cluster(
        manifest,
        preregistration,
        input_spec,
        args.output,
        args.aggregate_output,
        allow_one_shot_unblinding=True,
        freeze_receipt_sha256=freeze_receipt_sha256,
        static_session=authenticated_session(),
        ngl_session=requests.Session(),
    )
    print(
        json.dumps(
            {
                "cluster_id": fragment["cluster_id"],
                "status": fragment["status"],
                "reason_code": fragment.get("reason_code"),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if fragment["status"] == "evaluable" else 3


if __name__ == "__main__":
    raise SystemExit(main())
