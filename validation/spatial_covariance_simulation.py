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
        INPUT_KEYS,
        CellAccumulator,
        FROZEN_MAX_RECORD_BYTES,
        FROZEN_MAX_SHARD_BYTES,
        FROZEN_PROCESS_RSS_BYTES,
        FROZEN_SEED_COUNT,
        FROZEN_SHARD_COUNT,
        SchemaError,
        ShardSpec,
        _expected_seed_hash,
        _validate_performance_probe,
        _validate_resources,
        iter_shard_specs,
        expected_cell_ids,
        load_preregistration,
        preregistration_digest,
        resolve_below_run_root,
        result_root_sha256,
        sha256_file,
        sha256_json,
        validate_input_shard,
        validate_cell_summary,
        validate_preregistration,
        validate_shard_manifest,
    )
except ModuleNotFoundError:
    from score_spatial_covariance import (
        DIMENSION_NAMES,
        INPUT_KEYS,
        CellAccumulator,
        FROZEN_MAX_RECORD_BYTES,
        FROZEN_MAX_SHARD_BYTES,
        FROZEN_PROCESS_RSS_BYTES,
        FROZEN_SEED_COUNT,
        FROZEN_SHARD_COUNT,
        SchemaError,
        ShardSpec,
        _expected_seed_hash,
        _validate_performance_probe,
        _validate_resources,
        iter_shard_specs,
        expected_cell_ids,
        load_preregistration,
        preregistration_digest,
        resolve_below_run_root,
        result_root_sha256,
        sha256_file,
        sha256_json,
        validate_input_shard,
        validate_cell_summary,
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
                "schema": "dolphinrust.spatial-covariance.attempt/4",
                "cell_id": cell_id,
                "cell_ordinal": spec.cell_ordinal_start + cell_offset,
                "seed_index": seed_index,
                "seed_sha256": _expected_seed_hash(preregistration, cell_id, seed_index),
                **dimensions,
            }


def write_jsonl_atomic(records: Iterable[Mapping[str, Any]], destination: Path, byte_limit: int = FROZEN_MAX_SHARD_BYTES) -> dict[str, Any]:
    destination = Path(destination)
    if destination.name.endswith(".partial"):
        raise SchemaError("final JSONL destination must not use the .partial suffix")
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


def _has_exactly_one_partial_suffix(path: Path) -> bool:
    suffix = ".partial"
    return path.name.endswith(suffix) and not path.name.removesuffix(suffix).endswith(suffix)


def inspect_one_input_one_output(
    preregistration: Mapping[str, Any],
    spec: ShardSpec,
    input_path: Path,
    output_partial: Path,
    byte_limit: int = FROZEN_MAX_SHARD_BYTES,
    require_partial: bool = True,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    input_path = Path(input_path)
    output_partial = Path(output_partial)
    if require_partial and not _has_exactly_one_partial_suffix(output_partial):
        raise SchemaError("uncommitted batch output must use exactly one .partial suffix")
    if not require_partial and output_partial.name.endswith(".partial"):
        raise SchemaError("committed batch output must not use the .partial suffix")
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    with input_path.open("rb") as input_handle, output_partial.open("rb") as output_handle:
        for cell_offset, cell_id in enumerate(spec.cell_ids):
            cell_ordinal = spec.cell_ordinal_start + cell_offset
            dimensions = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
            accumulator = CellAccumulator(preregistration, cell_id, cell_ordinal)
            for seed_index in range(FROZEN_SEED_COUNT):
                input_line = input_handle.readline(FROZEN_MAX_RECORD_BYTES + 2)
                output_line = output_handle.readline(FROZEN_MAX_RECORD_BYTES + 2)
                if not input_line or not output_line:
                    raise SchemaError("batch output violates exact one-input-one-output cardinality")
                if len(input_line) > FROZEN_MAX_RECORD_BYTES or not input_line.endswith(b"\n") or len(output_line) > FROZEN_MAX_RECORD_BYTES or not output_line.endswith(b"\n"):
                    raise SchemaError("batch input/output exceeds the record cap or lacks newline framing")
                try:
                    request = json.loads(input_line)
                    receipt = json.loads(output_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SchemaError("batch input/output contains malformed JSON") from exc
                expected_request = {
                    "schema": "dolphinrust.spatial-covariance.attempt/3",
                    "cell_id": cell_id,
                    "cell_ordinal": cell_ordinal,
                    "seed_index": seed_index,
                    "seed_sha256": _expected_seed_hash(preregistration, cell_id, seed_index),
                    **dimensions,
                }
                if not isinstance(request, dict) or set(request) != INPUT_KEYS or type(request.get("cell_ordinal")) is not int or type(request.get("seed_index")) is not int or request != expected_request:
                    raise SchemaError("batch input has malformed, duplicate, missing, or out-of-order identity")
                identity = ("cell_id", "cell_ordinal", "seed_index", "seed_sha256")
                if not isinstance(receipt, dict) or type(receipt.get("cell_ordinal")) is not int or type(receipt.get("seed_index")) is not int or any(receipt.get(field_name) != request[field_name] for field_name in identity):
                    raise SchemaError("batch output order/identity does not match its input record")
                accumulator.add(receipt)
                byte_count += len(output_line)
                if byte_count > byte_limit:
                    raise SchemaError("batch output exceeds the frozen uncompressed byte cap")
                digest.update(output_line)
                record_count += 1
            accumulator.finalize()
        if input_handle.read(1) or output_handle.read(1):
            raise SchemaError("batch output violates exact one-input-one-output cardinality or contains top-up records")
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
    if type(peak_rss_bytes) is not int or peak_rss_bytes < 0 or peak_rss_bytes > FROZEN_PROCESS_RSS_BYTES:
        raise SchemaError("batch process exceeds the frozen 24 GiB RSS cap")
    lexical_output_partial = Path(output_partial)
    if not _has_exactly_one_partial_suffix(lexical_output_partial):
        raise SchemaError("uncommitted batch output must use exactly one .partial suffix")
    if lexical_output_partial.is_symlink():
        raise SchemaError("uncommitted batch output must not be a symlink")
    if Path(input_path).name.endswith(".partial") or Path(manifest_path).name.endswith(".partial"):
        raise SchemaError("input and manifest destinations must not use the .partial suffix")
    run_root = Path(run_root).resolve()
    input_path = Path(input_path).resolve()
    output_partial = Path(output_partial).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_name = output_partial.name.removesuffix(".partial")
    output_path = output_partial.with_name(output_name)
    if output_path.name.endswith(".partial") or output_path.exists() or manifest_path.exists():
        raise SchemaError("refusing to overwrite committed shard state")
    input_digest, input_bytes = sha256_file(input_path, FROZEN_MAX_SHARD_BYTES)
    output = inspect_one_input_one_output(preregistration, spec, input_path, output_partial)
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


def committed_shard_matches(
    preregistration: Mapping[str, Any],
    spec: ShardSpec,
    run_root: Path,
    manifest_path: Path,
    expected_code_sha256: str,
    expected_binary_sha256: str,
) -> bool:
    try:
        root = Path(run_root).resolve(strict=True)
        manifest_path = Path(manifest_path)
        if manifest_path.name.endswith(".partial"):
            return False
        resolved_manifest = manifest_path.resolve(strict=True)
        resolved_manifest.relative_to(root)
        if Path(str(manifest_path) + ".partial").exists():
            return False
        raw = resolved_manifest.read_bytes().splitlines()
        if len(raw) != 1:
            return False
        manifest = json.loads(raw[0])
        validate_shard_manifest(preregistration, manifest, spec)
        if manifest["code_sha256"] != expected_code_sha256 or manifest["binary_sha256"] != expected_binary_sha256:
            return False
        for path_field, hash_field, byte_field in (("input_path", "input_sha256", "input_bytes"), ("output_path", "output_sha256", "output_bytes")):
            path = resolve_below_run_root(root, manifest[path_field], f"shard {spec.index} {path_field}")
            if Path(str(path) + ".partial").exists():
                return False
            digest, size = sha256_file(path, FROZEN_MAX_SHARD_BYTES)
            if digest != manifest[hash_field] or size != manifest[byte_field]:
                return False
        input_path = resolve_below_run_root(root, manifest["input_path"], f"shard {spec.index} input path")
        output_path = resolve_below_run_root(root, manifest["output_path"], f"shard {spec.index} output path")
        validate_input_shard(preregistration, input_path, manifest, spec)
        output = inspect_one_input_one_output(preregistration, spec, input_path, output_path, require_partial=False)
        if output["sha256"] != manifest["output_sha256"] or output["bytes"] != manifest["output_bytes"] or output["records"] != manifest["output_records"]:
            return False
        return True
    except (OSError, ValueError, json.JSONDecodeError, SchemaError):
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
    _validate_performance_probe(preregistration, performance_probe, code_sha256, binary_sha256)
    _validate_resources(preregistration, resources, binary_sha256)
    entries = []
    digests = []
    for spec, path in zip(iter_shard_specs(preregistration), shard_manifest_paths):
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(run_root).as_posix()
        except ValueError as exc:
            raise SchemaError("shard manifest path must remain below the run root") from exc
        digest, _ = sha256_file(resolved)
        raw = resolved.read_bytes().splitlines()
        if len(raw) != 1:
            raise SchemaError(f"shard {spec.index} manifest is not one canonical record")
        try:
            shard_manifest = json.loads(raw[0])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SchemaError(f"shard {spec.index} manifest is malformed") from exc
        validate_shard_manifest(preregistration, shard_manifest, spec)
        if shard_manifest["code_sha256"] != code_sha256 or shard_manifest["binary_sha256"] != binary_sha256:
            raise SchemaError(f"shard {spec.index} code/binary scope differs from the run manifest")
        if not committed_shard_matches(
            preregistration,
            spec,
            run_root,
            resolved,
            code_sha256,
            binary_sha256,
        ):
            raise SchemaError(f"shard {spec.index} is not an exact committed shard")
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
    if destination.name.endswith(".partial"):
        raise SchemaError("run-manifest destination must not use the .partial suffix")
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


def _request_digest(preregistration: Mapping[str, Any], spec: ShardSpec) -> str:
    digest = hashlib.sha256(b"dolphinrust:spatial-covariance:shard-requests:v4\0")
    for request in iter_attempt_requests(preregistration, spec):
        encoded = compact_json_line(request)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def prepare_input_shard(preregistration: Mapping[str, Any], spec: ShardSpec, destination: Path) -> dict[str, Any]:
    descriptor = {
        "schema": "dolphinrust.spatial-covariance.shard-request/4",
        "shard_index": spec.index,
        "cell_ordinal_start": spec.cell_ordinal_start,
        "cell_ordinal_end_exclusive": spec.cell_ordinal_end_exclusive,
        "cell_ids": list(spec.cell_ids),
        "attempts_per_cell": FROZEN_SEED_COUNT,
        "expected_attempts": spec.expected_attempts,
        "preregistration_sha256": preregistration_digest(preregistration),
        "request_digest": _request_digest(preregistration, spec),
        "retained": False,
    }
    return write_jsonl_atomic((descriptor,), destination, byte_limit=FROZEN_MAX_RECORD_BYTES)


def commit_cell_transport(
    preregistration: Mapping[str, Any],
    cell_id: str,
    cell_ordinal: int,
    transport_path: Path,
    destination: Path,
    code_sha256: str,
    binary_sha256: str,
    expected_seed_count: int = FROZEN_SEED_COUNT,
) -> dict[str, Any]:
    if not _is_digest(code_sha256) or not _is_digest(binary_sha256):
        raise SchemaError("cell commit code/binary identity is invalid")
    transport_path = Path(transport_path)
    accumulator = CellAccumulator(preregistration, cell_id, cell_ordinal, expected_seed_count, code_sha256, binary_sha256)
    with transport_path.open("rb") as handle:
        for line_number in range(expected_seed_count):
            raw = handle.readline(FROZEN_MAX_RECORD_BYTES + 2)
            if not raw or len(raw) > FROZEN_MAX_RECORD_BYTES or not raw.endswith(b"\n"):
                raise SchemaError("ephemeral attempt transport is incomplete or oversized")
            try:
                accumulator.add(json.loads(raw))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SchemaError("ephemeral attempt transport is malformed") from exc
        if handle.read(1):
            raise SchemaError("ephemeral attempt transport contains top-up evidence")
    summary = accumulator.finalize()
    validate_cell_summary(preregistration, summary, cell_id, cell_ordinal, code_sha256, binary_sha256)
    receipt = write_jsonl_atomic((summary,), destination, byte_limit=FROZEN_MAX_RECORD_BYTES)
    transport_path.unlink()
    return receipt


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _summary_root(preregistration: Mapping[str, Any], directory: Path, spec: ShardSpec, code_sha256: str, binary_sha256: str) -> tuple[str, int]:
    digest = hashlib.sha256(b"dolphinrust:spatial-covariance:cell-summary-root:v4\0")
    total_bytes = 0
    for offset, cell_id in enumerate(spec.cell_ids):
        path = directory / f"cell-{spec.cell_ordinal_start + offset:05d}.jsonl"
        raw = path.read_bytes()
        if len(raw.splitlines()) != 1 or len(raw) > FROZEN_MAX_RECORD_BYTES:
            raise SchemaError(f"cell {cell_id} compact summary is malformed")
        summary = json.loads(raw)
        validate_cell_summary(preregistration, summary, cell_id, spec.cell_ordinal_start + offset, code_sha256, binary_sha256)
        digest.update(offset.to_bytes(8, "big"))
        digest.update(hashlib.sha256(raw).digest())
        total_bytes += len(raw)
    return digest.hexdigest(), total_bytes


def commit_output_shard(
    preregistration: Mapping[str, Any],
    spec: ShardSpec,
    run_root: Path,
    summary_directory: Path,
    manifest_path: Path,
    code_sha256: str,
    binary_sha256: str,
    elapsed_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    root = Path(run_root).resolve(strict=True)
    directory = Path(summary_directory).resolve(strict=True)
    manifest_path = Path(manifest_path)
    if manifest_path.name.endswith(".partial") or manifest_path.exists() or not directory.is_dir():
        raise SchemaError("compact shard commit destination/state is invalid")
    try:
        relative = directory.relative_to(root).as_posix()
        manifest_path.resolve().parent.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SchemaError("compact shard paths must remain below run root") from exc
    summary_digest, summary_bytes = _summary_root(preregistration, directory, spec, code_sha256, binary_sha256)
    manifest = {
        "schema": "dolphinrust.spatial-covariance.shard-manifest/4", "schema_version": 4,
        "shard_index": spec.index, "cell_ordinal_start": spec.cell_ordinal_start,
        "cell_ordinal_end_exclusive": spec.cell_ordinal_end_exclusive, "expected_cells": len(spec.cell_ids),
        "expected_attempts": spec.expected_attempts, "summary_path": relative, "summary_sha256": summary_digest,
        "summary_bytes": summary_bytes, "summary_records": len(spec.cell_ids),
        "preregistration_sha256": preregistration_digest(preregistration), "code_sha256": code_sha256,
        "binary_sha256": binary_sha256, "generator_protocol_sha256": sha256_json(preregistration["execution_protocol"]),
        "elapsed_seconds": elapsed_seconds, "peak_rss_bytes": peak_rss_bytes, "committed": True,
    }
    validate_shard_manifest(preregistration, manifest, spec)
    return write_jsonl_atomic((manifest,), manifest_path, byte_limit=preregistration["execution_protocol"]["max_encoded_shard_manifest_bytes"])


def committed_shard_matches(preregistration: Mapping[str, Any], spec: ShardSpec, run_root: Path, manifest_path: Path, expected_code_sha256: str, expected_binary_sha256: str) -> bool:
    try:
        raw = Path(manifest_path).read_bytes().splitlines()
        if len(raw) != 1:
            return False
        manifest = json.loads(raw[0])
        validate_shard_manifest(preregistration, manifest, spec)
        if manifest["code_sha256"] != expected_code_sha256 or manifest["binary_sha256"] != expected_binary_sha256:
            return False
        directory = resolve_below_run_root(Path(run_root), manifest["summary_path"], "compact summary directory")
        digest, size = _summary_root(preregistration, directory, spec, expected_code_sha256, expected_binary_sha256)
        return digest == manifest["summary_sha256"] and size == manifest["summary_bytes"]
    except (OSError, ValueError, json.JSONDecodeError, SchemaError):
        return False


def build_run_manifest(preregistration: Mapping[str, Any], run_root: Path, shard_manifest_paths: Iterable[Path], code_sha256: str, binary_sha256: str, performance_probe: Mapping[str, Any], resources: list[Mapping[str, Any]]) -> dict[str, Any]:
    validate_preregistration(preregistration)
    paths = tuple(shard_manifest_paths)
    if len(paths) != FROZEN_SHARD_COUNT:
        raise SchemaError("run manifest requires exactly 891 compact shards")
    _validate_performance_probe(preregistration, performance_probe, code_sha256, binary_sha256)
    _validate_resources(preregistration, resources, binary_sha256)
    root = Path(run_root).resolve(strict=True)
    entries = []
    digests = []
    for spec, path in zip(iter_shard_specs(preregistration), paths):
        resolved = Path(path).resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        digest, _ = sha256_file(resolved)
        if not committed_shard_matches(preregistration, spec, root, resolved, code_sha256, binary_sha256):
            raise SchemaError(f"shard {spec.index} is not exact compact committed evidence")
        entries.append({"path": relative, "sha256": digest})
        digests.append(digest)
    return {"schema": "dolphinrust.spatial-covariance.run-manifest/4", "schema_version": 4,
            "preregistration_sha256": preregistration_digest(preregistration), "code_sha256": code_sha256,
            "binary_sha256": binary_sha256, "generator_protocol_sha256": sha256_json(preregistration["execution_protocol"]),
            "performance_probe": dict(performance_probe), "resources": [dict(item) for item in resources],
            "shard_manifests": entries, "result_root_sha256": result_root_sha256(digests)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("spatial_covariance_preregistration.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write one compact deterministic shard descriptor")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--shard-index", type=int, required=True)
    reduce_cell = commands.add_parser("reduce-cell", help="independently reduce one ephemeral cell transport")
    reduce_cell.add_argument("--cell-ordinal", type=int, required=True)
    reduce_cell.add_argument("--transport", type=Path, required=True)
    reduce_cell.add_argument("--destination", type=Path, required=True)
    reduce_cell.add_argument("--code-sha256", required=True)
    reduce_cell.add_argument("--binary-sha256", required=True)
    commit = commands.add_parser("commit", help="validate and atomically commit one compact shard")
    commit.add_argument("--run-root", type=Path, required=True)
    commit.add_argument("--shard-index", type=int, required=True)
    commit.add_argument("--summary-directory", type=Path, required=True)
    commit.add_argument("--manifest", type=Path, required=True)
    commit.add_argument("--code-sha256", required=True)
    commit.add_argument("--binary-sha256", required=True)
    commit.add_argument("--elapsed-seconds", type=float, required=True)
    commit.add_argument("--peak-rss-bytes", type=int, required=True)
    resume = commands.add_parser("resume", help="verify whether one committed shard is exactly reusable")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument("--shard-index", type=int, required=True)
    resume.add_argument("--manifest", type=Path, required=True)
    resume.add_argument("--code-sha256", required=True)
    resume.add_argument("--binary-sha256", required=True)
    assemble = commands.add_parser("assemble", help="atomically assemble the final run manifest")
    assemble.add_argument("--run-root", type=Path, required=True)
    assemble.add_argument("--shard-manifest-directory", type=Path, required=True)
    assemble.add_argument("--performance-probe", type=Path, required=True)
    assemble.add_argument("--resources", type=Path, required=True)
    assemble.add_argument("--destination", type=Path, required=True)
    assemble.add_argument("--code-sha256", required=True)
    assemble.add_argument("--binary-sha256", required=True)
    args = parser.parse_args()
    preregistration = load_preregistration(args.preregistration)
    if args.command in {"prepare", "commit", "resume"}:
        spec = next((item for item in iter_shard_specs(preregistration) if item.index == args.shard_index), None)
        if spec is None:
            raise SystemExit(f"shard index is outside 0..{preregistration['execution_protocol']['shard_count'] - 1}")
    if args.command == "prepare":
        destination = args.run_root / "requests" / f"shard-{spec.index:05d}.jsonl"
        result = prepare_input_shard(preregistration, spec, destination)
    elif args.command == "reduce-cell":
        cell_ids = expected_cell_ids(preregistration)
        if args.cell_ordinal < 0 or args.cell_ordinal >= len(cell_ids):
            raise SchemaError("cell ordinal is outside the frozen matrix")
        result = commit_cell_transport(
            preregistration, cell_ids[args.cell_ordinal], args.cell_ordinal, args.transport,
            args.destination, args.code_sha256, args.binary_sha256,
        )
    elif args.command == "commit":
        result = commit_output_shard(
            preregistration, spec, args.run_root, args.summary_directory, args.manifest,
            args.code_sha256, args.binary_sha256, args.elapsed_seconds, args.peak_rss_bytes,
        )
    elif args.command == "resume":
        result = {"reusable": committed_shard_matches(
            preregistration, spec, args.run_root, args.manifest, args.code_sha256, args.binary_sha256,
        )}
    else:
        run_root = args.run_root.resolve(strict=True)
        if args.destination.parent.resolve() != run_root:
            raise SchemaError("run-manifest destination parent must equal the run root")
        manifest_paths = [args.shard_manifest_directory / f"manifest-{index:05d}.jsonl" for index in range(FROZEN_SHARD_COUNT)]
        with args.performance_probe.open(encoding="utf-8") as handle:
            performance_probe = json.load(handle)
        with args.resources.open(encoding="utf-8") as handle:
            resources = json.load(handle)
        run_manifest = build_run_manifest(
            preregistration, run_root, manifest_paths, args.code_sha256, args.binary_sha256,
            performance_probe, resources,
        )
        result = write_run_manifest_atomic(run_manifest, args.destination)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
