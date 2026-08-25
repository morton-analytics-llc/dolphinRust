#!/usr/bin/env python3
"""Validate and score one externally supplied #53 held-out receipt bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from validation.heldout_temporal_covariance.scorer import score_receipt
else:
    from heldout_temporal_covariance.scorer import score_receipt


JSON_CAP = 16 * 1024 * 1024
MANIFEST_CAP = 1024 * 1024
FACTOR_CAP = 1024 * 1024 * 1024


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


def bind_factor_files(receipt: dict[str, Any], factor_path: Path, manifest_path: Path) -> None:
    hashes = receipt.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError("held-out receipt is missing artifact hashes")
    factor = read_bounded(factor_path, FACTOR_CAP)
    manifest_bytes = read_bounded(manifest_path, MANIFEST_CAP)
    factor_sha256 = sha256(factor)
    manifest_sha256 = sha256(manifest_bytes)
    if factor_sha256 != hashes.get("persisted_factor_sha256"):
        raise ValueError("held-out receipt factor hash differs from supplied HDF5")
    if manifest_sha256 != hashes.get("persisted_factor_manifest_sha256"):
        raise ValueError("held-out receipt factor-manifest hash differs from supplied JSON")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        preregistration = read_json(args.prereg)
        manifest = read_json(args.manifest)
        receipt = read_json(args.receipt)
        bind_factor_files(receipt, args.factor, args.factor_manifest)
        result = score_receipt(preregistration, manifest, receipt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return {"pass": 0, "fail": 2, "not_evaluable": 3}.get(result["status"], 4)


if __name__ == "__main__":
    raise SystemExit(main())
