#!/usr/bin/env python3
"""Deterministic shard preparation and commit driver for F54-07 v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    from validation.score_spatial_covariance import (
        DIMENSION_NAMES,
        FROZEN_MAX_RECORD_BYTES,
        FROZEN_MAX_SHARD_BYTES,
        FROZEN_PROCESS_RSS_BYTES,
        FROZEN_SEED_COUNT,
        SchemaError,
        ShardSpec,
        _expected_seed_hash,
        iter_shard_specs,
        load_preregistration,
        preregistration_digest,
        result_root_sha256,
        sha256_file,
        sha256_json,
        validate_input_shard,
        validate_preregistration,
        validate_shard_manifest,
    )
except ModuleNotFoundError:
    from score_spatial_covariance import (
        DIMENSION_NAMES,
        FROZEN_MAX_RECORD_BYTES,
        FROZEN_MAX_SHARD_BYTES,
        FROZEN_PROCESS_RSS_BYTES,
        FROZEN_SEED_COUNT,
        SchemaError,
        ShardSpec,
        _expected_seed_hash,
        iter_shard_specs,
        load_preregistration,
        preregistration_digest,
        result_root_sha256,
        sha256_file,
        sha256_json,
        validate_input_shard,
        validate_preregistration,
        validate_shard_manifest,
    )


def compact_json_line(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def iter_attempt_requests(preregistration: Mapping[str, Any], spec: ShardSpec) -> Iterator[dict[str, Any]]:
    validate_preregistration(preregistration)
    for cell_offset, cell_id in enumerate(spec.cell_ids):
        dimensions = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
        for seed_index in range(FROZEN_SEED_COUNT):
            yield {
                "schema": "dolphinrust.spatial-covariance.attempt/3",
                "cell_id": cell_id,
                "cell_ordinal": spec.cell_ordinal_start + cell_offset,
                "seed_index": seed_index,
                "seed_sha256": _expected_seed_hash(preregistration, cell_id, seed_index),
                **dimensions,
            }


def write_jsonl_atomic(records: Iterable[Mapping[str, Any]], destination: Path, byte_limit: int = FROZEN_MAX_SHARD_BYTES) -> dict[str, Any]:
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise SchemaError(f"refusing to overwrite existing shard state at {destination}")
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("xb") as handle:
            for record in records:
                encoded = compact_json_line(record)
                if len(encoded) > FROZEN_MAX_RECORD_BYTES:
                    raise SchemaError("encoded record exceeds the frozen per-record cap")
                if byte_count + len(encoded) > byte_limit:
                    raise SchemaError("JSONL shard exceeds the frozen uncompressed byte cap")
                handle.write(encoded)
                digest.update(encoded)
                byte_count += len(encoded)
                record_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"sha256": digest.hexdigest(), "bytes": byte_count, "records": record_count}


def prepare_input_shard(preregistration: Mapping[str, Any], spec: ShardSpec, destination: Path) -> dict[str, Any]:
    receipt = write_jsonl_atomic(iter_attempt_requests(preregistration, spec), destination)
    if receipt["records"] != spec.expected_attempts:
        raise SchemaError(f"input shard {spec.index} does not contain its exact deterministic seed schedule")
    return receipt


def inspect_one_input_one_output(input_path: Path, output_partial: Path, byte_limit: int = FROZEN_MAX_SHARD_BYTES, require_partial: bool = True) -> dict[str, Any]:
    input_path = Path(input_path)
    output_partial = Path(output_partial)
    if require_partial and not output_partial.name.endswith(".partial"):
        raise SchemaError("uncommitted batch output must use the .partial suffix")
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    with input_path.open("rb") as input_handle, output_partial.open("rb") as output_handle:
        while True:
            input_line = input_handle.readline(FROZEN_MAX_RECORD_BYTES + 2)
            output_line = output_handle.readline(FROZEN_MAX_RECORD_BYTES + 2)
            if not input_line and not output_line:
                break
            if not input_line or not output_line:
                raise SchemaError("batch output violates exact one-input-one-output cardinality")
            if len(output_line) > FROZEN_MAX_RECORD_BYTES or not output_line.endswith(b"\n"):
                raise SchemaError("batch output exceeds the record cap or lacks newline framing")
            try:
                request = json.loads(input_line)
                receipt = json.loads(output_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SchemaError("batch input/output contains malformed JSON") from exc
            identity = ("cell_id", "cell_ordinal", "seed_index", "seed_sha256")
            if any(receipt.get(field_name) != request.get(field_name) for field_name in identity):
                raise SchemaError("batch output order/identity does not match its input record")
            byte_count += len(output_line)
            if byte_count > byte_limit:
                raise SchemaError("batch output exceeds the frozen uncompressed byte cap")
            digest.update(output_line)
            record_count += 1
    return {"sha256": digest.hexdigest(), "bytes": byte_count, "records": record_count}


def commit_output_shard(
    preregistration: Mapping[str, Any],
    spec: ShardSpec,
    run_root: Path,
    input_path: Path,
    output_partial: Path,
    manifest_path: Path,
    code_sha256: str,
    binary_sha256: str,
    elapsed_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    if peak_rss_bytes > FROZEN_PROCESS_RSS_BYTES:
        raise SchemaError("batch process exceeds the frozen 24 GiB RSS cap")
    run_root = Path(run_root).resolve()
    input_path = Path(input_path).resolve()
    output_partial = Path(output_partial).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_name = output_partial.name.removesuffix(".partial")
    output_path = output_partial.with_name(output_name)
    if input_path.name.endswith(".partial") or output_path.exists() or manifest_path.exists():
        raise SchemaError("refusing to overwrite committed shard state")
    input_digest, input_bytes = sha256_file(input_path, FROZEN_MAX_SHARD_BYTES)
    output = inspect_one_input_one_output(input_path, output_partial)
    if output["records"] != spec.expected_attempts:
        raise SchemaError(f"shard {spec.index} is incomplete; top-up is prohibited")
    try:
        relative_input = input_path.relative_to(run_root).as_posix()
        relative_output = output_path.relative_to(run_root).as_posix()
        manifest_path.relative_to(run_root)
    except ValueError as exc:
        raise SchemaError("shard paths must remain below the run root") from exc
    manifest = {
        "schema": "dolphinrust.spatial-covariance.shard-manifest",
        "schema_version": 3,
        "shard_index": spec.index,
        "cell_ordinal_start": spec.cell_ordinal_start,
        "cell_ordinal_end_exclusive": spec.cell_ordinal_end_exclusive,
        "expected_cells": len(spec.cell_ids),
        "expected_attempts": spec.expected_attempts,
        "input_path": relative_input,
        "output_path": relative_output,
        "input_sha256": input_digest,
        "output_sha256": output["sha256"],
        "input_bytes": input_bytes,
        "output_bytes": output["bytes"],
        "input_records": output["records"],
        "output_records": output["records"],
        "preregistration_sha256": preregistration_digest(preregistration),
        "code_sha256": code_sha256,
        "binary_sha256": binary_sha256,
        "generator_protocol_sha256": sha256_json(preregistration["execution_protocol"]),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "committed": True,
    }
    validate_shard_manifest(preregistration, manifest, spec)
    os.replace(output_partial, output_path)
    try:
        write_jsonl_atomic((manifest,), manifest_path)
    except BaseException:
        output_path.rename(output_partial)
        raise
    return manifest


def committed_shard_matches(preregistration: Mapping[str, Any], spec: ShardSpec, run_root: Path, manifest_path: Path) -> bool:
    try:
        if Path(str(manifest_path) + ".partial").exists():
            return False
        raw = Path(manifest_path).read_bytes().splitlines()
        if len(raw) != 1:
            return False
        manifest = json.loads(raw[0])
        validate_shard_manifest(preregistration, manifest, spec)
        root = Path(run_root)
        for path_field, hash_field, byte_field in (("input_path", "input_sha256", "input_bytes"), ("output_path", "output_sha256", "output_bytes")):
            path = root / manifest[path_field]
            if Path(str(path) + ".partial").exists():
                return False
            digest, size = sha256_file(path, FROZEN_MAX_SHARD_BYTES)
            if digest != manifest[hash_field] or size != manifest[byte_field]:
                return False
        validate_input_shard(preregistration, root / manifest["input_path"], manifest, spec)
        output = inspect_one_input_one_output(root / manifest["input_path"], root / manifest["output_path"], require_partial=False)
        if output["sha256"] != manifest["output_sha256"] or output["bytes"] != manifest["output_bytes"] or output["records"] != manifest["output_records"]:
            return False
        return True
    except (OSError, json.JSONDecodeError, SchemaError):
        return False


def derive_concurrency_receipt(projected_serial_seconds: float, target_wall_seconds: float, reserve_fraction: float = 0.25) -> int:
    if projected_serial_seconds <= 0 or target_wall_seconds <= 0 or not 0 <= reserve_fraction < 1:
        raise SchemaError("concurrency inputs must be positive with reserve in [0,1)")
    return math.ceil(projected_serial_seconds / (target_wall_seconds * (1.0 - reserve_fraction)))


def build_run_manifest(
    preregistration: Mapping[str, Any],
    run_root: Path,
    shard_manifest_paths: Iterable[Path],
    code_sha256: str,
    binary_sha256: str,
    performance_probe: Mapping[str, Any],
    resources: list[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    run_root = Path(run_root).resolve()
    shard_manifest_paths = tuple(shard_manifest_paths)
    expected_shards = preregistration["execution_protocol"]["shard_count"]
    if len(shard_manifest_paths) != expected_shards:
        raise SchemaError(f"run manifest requires exactly {expected_shards} committed shards")
    entries = []
    digests = []
    for path in shard_manifest_paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(run_root).as_posix()
        except ValueError as exc:
            raise SchemaError("shard manifest path must remain below the run root") from exc
        digest, _ = sha256_file(resolved)
        entries.append({"path": relative, "sha256": digest})
        digests.append(digest)
    return {
        "schema": "dolphinrust.spatial-covariance.run-manifest",
        "schema_version": 3,
        "preregistration_sha256": preregistration_digest(preregistration),
        "code_sha256": code_sha256,
        "binary_sha256": binary_sha256,
        "generator_protocol_sha256": sha256_json(preregistration["execution_protocol"]),
        "performance_probe": dict(performance_probe),
        "resources": [dict(resource) for resource in resources],
        "shard_manifests": entries,
        "result_root_sha256": result_root_sha256(digests),
    }


def write_run_manifest_atomic(run_manifest: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise SchemaError("refusing to overwrite run-manifest state")
    encoded = compact_json_line(run_manifest)
    if len(encoded) > FROZEN_MAX_SHARD_BYTES:
        raise SchemaError("run manifest exceeds the frozen uncompressed byte cap")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("spatial_covariance_preregistration.json"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prepare-input-shard", type=int, required=True)
    args = parser.parse_args()
    preregistration = load_preregistration(args.preregistration)
    spec = next((item for item in iter_shard_specs(preregistration) if item.index == args.prepare_input_shard), None)
    if spec is None:
        raise SystemExit(f"shard index is outside 0..{preregistration['execution_protocol']['shard_count'] - 1}")
    destination = args.run_root / "shards" / f"input-{spec.index:05d}.jsonl"
    print(json.dumps(prepare_input_shard(preregistration, spec, destination), sort_keys=True))


if __name__ == "__main__":
    main()
