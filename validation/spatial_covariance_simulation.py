#!/usr/bin/env python3
"""Deterministic shard preparation and commit driver for F54-07 v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

try:
    from validation.score_spatial_covariance import (
        DIMENSION_NAMES,
        INPUT_KEYS,
        CellAccumulator,
        FROZEN_MAX_RECORD_BYTES,
        FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
        FROZEN_MAX_RUN_MANIFEST_BYTES,
        FROZEN_MAX_SHARD_BYTES,
        FROZEN_PROCESS_RSS_BYTES,
        FROZEN_SEED_COUNT,
        FROZEN_SHARD_COUNT,
        SchemaError,
        ShardSpec,
        _expected_seed_hash,
        _read_bounded_bytes,
        _read_hashed_json_record,
        _read_single_json_record,
        _validate_performance_probe,
        _validate_resources,
        regenerate_frozen_attempt_inputs,
        validate_matched_pair_cohorts,
        iter_shard_specs,
        expected_cell_ids,
        expected_seed_count,
        load_preregistration,
        preregistration_digest,
        producer_identities,
        resolve_below_run_root,
        result_root_sha256,
        sha256_json,
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
        FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
        FROZEN_MAX_RUN_MANIFEST_BYTES,
        FROZEN_MAX_SHARD_BYTES,
        FROZEN_PROCESS_RSS_BYTES,
        FROZEN_SEED_COUNT,
        FROZEN_SHARD_COUNT,
        SchemaError,
        ShardSpec,
        _expected_seed_hash,
        _read_bounded_bytes,
        _read_hashed_json_record,
        _read_single_json_record,
        _validate_performance_probe,
        _validate_resources,
        regenerate_frozen_attempt_inputs,
        validate_matched_pair_cohorts,
        iter_shard_specs,
        expected_cell_ids,
        expected_seed_count,
        load_preregistration,
        preregistration_digest,
        producer_identities,
        resolve_below_run_root,
        result_root_sha256,
        sha256_json,
        validate_cell_summary,
        validate_preregistration,
        validate_shard_manifest,
    )


def compact_json_line(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def _load_bounded_json(path: Path, byte_limit: int, label: str) -> Any:
    raw = _read_bounded_bytes(Path(path), byte_limit, label)
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"{label} is malformed JSON") from exc


def capture_benchmark_stdout(command: list[str], byte_limit: int = 8192) -> dict[str, Any]:
    if not isinstance(command, list) or not command or any(not isinstance(value, str) or not value for value in command):
        raise SchemaError("benchmark command is malformed")
    with tempfile.TemporaryFile() as stdout_file:
        completed = subprocess.run(
            command, check=False, stdout=stdout_file, stderr=subprocess.DEVNULL
        )
        stdout_bytes = stdout_file.tell()
        if completed.returncode != 0 or stdout_bytes > byte_limit:
            raise SchemaError("benchmark command failed or exceeded the stdout cap")
        stdout_file.seek(0)
        stdout = stdout_file.read(byte_limit + 1)
    if len(stdout) != stdout_bytes:
        raise SchemaError("benchmark stdout changed while captured")
    try:
        parsed = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("benchmark stdout is malformed JSON") from exc
    if not isinstance(parsed, dict) or stdout.count(b"\n") > 1:
        raise SchemaError("benchmark stdout must contain one JSON object")
    return {
        "command": list(command),
        "exit_status": completed.returncode,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_json": stdout.decode("utf-8"),
    }


PERFORMANCE_CELL_IDS = {
    "hw_7x14_near_emi_spatial":
        "hw_7x14|stride_4|glrt_frozen|interior|shared_75_positive|four_blocks|emi|well_separated|spatial_correlation_stress",
    "hw_7x14_near_evd_spatial":
        "hw_7x14|stride_4|glrt_frozen|interior|shared_75_positive|four_blocks|evd|well_separated|spatial_correlation_stress",
}
MATCHED_POSITIVE_CELL = (
    "hw_1x1|stride_4|glrt_frozen|interior|shared_75_positive|four_blocks|emi|"
    "well_separated|spatial_correlation_stress"
)
MATCHED_NEGATIVE_CELL = MATCHED_POSITIVE_CELL.replace(
    "shared_75_positive", "shared_75_negative"
)
PARALLEL_BATCH_WORKER_RSS_ADMISSION_BYTES = 2 << 30
PARALLEL_BATCH_WORKER_COUNT = FROZEN_PROCESS_RSS_BYTES // PARALLEL_BATCH_WORKER_RSS_ADMISSION_BYTES
PARALLEL_BATCH_MAX_REQUESTS_PER_CHILD = 3
PARALLEL_BATCH_RSS_SAMPLE_SECONDS = 0.05
MATCHED_COHORT_SEED_COUNT = 512


def _validated_producer_identities(
    preregistration: Mapping[str, Any],
    source_root: Path,
    batch_binary: Path,
    benchmark_binary: Path,
) -> tuple[str, str]:
    code_sha256, binary_sha256 = producer_identities(
        source_root, batch_binary, benchmark_binary
    )
    frozen = preregistration["generator"]["binary"]["source_identity"]["sha256"]
    if code_sha256 != frozen:
        raise SchemaError("checked-out producer source set differs from the frozen source identity")
    return code_sha256, binary_sha256


def _write_bounded_json_atomic(value: Any, destination: Path, byte_limit: int) -> dict[str, Any]:
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise SchemaError(f"refusing to overwrite receipt state at {destination}")
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"
    if len(encoded) > byte_limit:
        raise SchemaError("generated receipt exceeds its frozen byte cap")
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


def _fsync_directory(path: Path) -> None:
    handle = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _tool_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    value = completed.stdout.strip()
    if not value:
        raise SchemaError(f"tool version command emitted no output: {command[0]}")
    return value


def _host_provenance(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise SchemaError("frozen resource receipts require Darwin ru_maxrss byte semantics")
    ram_bytes = int(_tool_output(["sysctl", "-n", "hw.memsize"]))
    brand = _tool_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    hardware_class = "apple-m2-32gb" if brand.startswith("Apple M2") and ram_bytes == 32 << 30 else ""
    provenance = {
        "os": f"{platform.system()} {platform.machine()}",
        "hardware_class": hardware_class,
        "ram_bytes": ram_bytes,
        "tool_versions": {
            "rustc": _tool_output(["rustc", "-Vv"]),
            "cargo": _tool_output(["cargo", "-V"]),
            "uname": _tool_output(["uname", "-a"]),
        },
    }
    sampling = preregistration["resource_sampling"]
    for name in ("os", "hardware_class", "ram_bytes"):
        if provenance[name] != sampling[name]:
            raise SchemaError(f"current host differs from frozen resource sampling field {name}")
    return provenance


def _child_max_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(rss if platform.system() == "Darwin" else rss * 1024)


def run_parallel_batch(
    source_root: Path,
    preregistration_path: Path,
    request_file: Path | None,
    cell_id: str,
    destination: Path | None = None,
    seed_count: int | None = None,
    batch_binary: Path | None = None,
    generation_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_root = Path(source_root).resolve(strict=True)
    preregistration_path = Path(preregistration_path).resolve(strict=True)
    batch_binary = (
        source_root / "target/release/examples/spatial_covariance_batch"
        if batch_binary is None else Path(batch_binary)
    ).resolve(strict=True)
    if not math.isfinite(generation_delay_seconds) or generation_delay_seconds < 0:
        raise SchemaError("parallel batch generation delay is invalid")
    preregistration = load_preregistration(preregistration_path)
    cell_ids = expected_cell_ids(preregistration)
    if cell_id not in cell_ids:
        raise SchemaError("parallel batch cell is outside the frozen matrix")
    cell_ordinal = cell_ids.index(cell_id)
    request_path = None
    if request_file is None:
        if seed_count is None or not 0 < seed_count <= expected_seed_count(cell_id):
            raise SchemaError("parallel generated batch requires an exact seed count")
        request_count = seed_count
    else:
        if seed_count is not None:
            raise SchemaError("parallel batch cannot combine a request file and seed count")
        request_path = Path(request_file).resolve(strict=True)
        request_count = 0
        with request_path.open("rb") as handle:
            while True:
                raw = handle.readline(FROZEN_MAX_RECORD_BYTES + 2)
                if not raw:
                    break
                if len(raw) > FROZEN_MAX_RECORD_BYTES or not raw.endswith(b"\n"):
                    raise SchemaError("parallel batch request is oversized or lacks newline framing")
                try:
                    request = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SchemaError("parallel batch request is malformed") from exc
                dimensions = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
                expected = _cell_request_at(
                    preregistration, cell_id, cell_ordinal, dimensions, request_count
                )
                if request != expected or raw != compact_json_line(expected):
                    raise SchemaError("parallel batch request differs from its exact frozen seed descriptor")
                request_count += 1
    if request_count == 0:
        raise SchemaError("parallel batch requires at least one request")
    chunks = [
        (start, min(PARALLEL_BATCH_MAX_REQUESTS_PER_CHILD, request_count - start))
        for start in range(0, request_count, PARALLEL_BATCH_MAX_REQUESTS_PER_CHILD)
    ]
    worker_count = min(PARALLEL_BATCH_WORKER_COUNT, len(chunks))
    wave_count = math.ceil(len(chunks) / PARALLEL_BATCH_WORKER_COUNT)
    peak_rss_bytes = 0
    output_digest = hashlib.sha256(b"dolphinrust:ordered-parallel-batch:v1\0")
    output_records = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outputs = []
        for wave_start in range(0, len(chunks), PARALLEL_BATCH_WORKER_COUNT):
            processes = []
            for index in range(
                wave_start, min(wave_start + PARALLEL_BATCH_WORKER_COUNT, len(chunks))
            ):
                seed_start, chunk_seed_count = chunks[index]
                output_path = root / f"output-{index:03d}.jsonl"
                output_handle = output_path.open("w+b")
                command = [
                    sys.executable, str(Path(__file__).resolve()), "_batch-chunk-child",
                    "--source-root", str(source_root),
                    "--preregistration", str(preregistration_path),
                    "--batch-binary", str(batch_binary),
                    "--cell-id", cell_id,
                    "--seed-start", str(seed_start),
                    "--seed-count", str(chunk_seed_count),
                    "--generation-delay-seconds", str(generation_delay_seconds),
                ]
                if request_path is not None:
                    command.extend(["--request-file", str(request_path)])
                process = subprocess.Popen(
                    command,
                    cwd=source_root,
                    stdin=subprocess.DEVNULL,
                    stdout=output_handle,
                    stderr=subprocess.DEVNULL,
                )
                outputs.append(output_handle)
                processes.append(process)
            while True:
                live = [process.pid for process in processes if process.poll() is None]
                if not live:
                    break
                sampled = subprocess.run(
                    ["ps", "-axo", "pid=,ppid=,rss="],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                rows = [tuple(map(int, line.split())) for line in sampled.stdout.splitlines()]
                process_tree = set(live)
                changed = True
                while changed:
                    changed = False
                    for pid, ppid, _ in rows:
                        if ppid in process_tree and pid not in process_tree:
                            process_tree.add(pid)
                            changed = True
                rss_bytes = sum(rss for pid, _, rss in rows if pid in process_tree) * 1024
                peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
                if peak_rss_bytes > FROZEN_PROCESS_RSS_BYTES:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                    raise SchemaError("parallel batch aggregate RSS exceeded the frozen process cap")
                time.sleep(PARALLEL_BATCH_RSS_SAMPLE_SECONDS)
            if any(process.wait() != 0 for process in processes):
                for handle in outputs:
                    handle.close()
                raise SchemaError("parallel batch child failed")
        final = None
        partial = None
        destination_handle = None
        try:
            if destination is not None:
                final = Path(destination)
                partial = final.with_name(final.name + ".partial")
                if final.exists() or partial.exists():
                    raise SchemaError("refusing to overwrite ordered parallel output")
                final.parent.mkdir(parents=True, exist_ok=True)
                destination_handle = partial.open("xb")
            else:
                destination_handle = None
            for output, (seed_start, chunk_seed_count) in zip(outputs, chunks):
                output.seek(0)
                for seed_index in range(seed_start, seed_start + chunk_seed_count):
                    raw = output.readline(FROZEN_MAX_RECORD_BYTES + 2)
                    if not raw or len(raw) > FROZEN_MAX_RECORD_BYTES or not raw.endswith(b"\n"):
                        raise SchemaError("parallel batch output is incomplete or oversized")
                    try:
                        attempt = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SchemaError("parallel batch output is malformed") from exc
                    expected_identity = {
                        "cell_id": cell_id,
                        "cell_ordinal": cell_ordinal,
                        "seed_index": seed_index,
                        "seed_sha256": _expected_seed_hash(preregistration, cell_id, seed_index),
                    }
                    if any(attempt.get(name) != value for name, value in expected_identity.items()):
                        raise SchemaError("parallel batch output order/scope differs from requests")
                    output_digest.update(len(raw).to_bytes(8, "big"))
                    output_digest.update(raw)
                    output_records += 1
                    if destination_handle is not None:
                        destination_handle.write(raw)
                if output.read(1):
                    raise SchemaError("parallel batch child emitted top-up evidence")
            if destination_handle is not None:
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
                destination_handle.close()
                os.replace(partial, final)
        except BaseException:
            if destination_handle is not None and not destination_handle.closed:
                destination_handle.close()
            if partial is not None:
                partial.unlink(missing_ok=True)
            raise
        finally:
            for handle in outputs:
                handle.close()
    elapsed = time.perf_counter() - started
    return {
        "worker_count": worker_count,
        "max_requests_per_child": PARALLEL_BATCH_MAX_REQUESTS_PER_CHILD,
        "child_invocation_count": len(chunks),
        "wave_count": wave_count,
        "worker_rss_admission_bytes": PARALLEL_BATCH_WORKER_RSS_ADMISSION_BYTES,
        "aggregate_rss_cap_bytes": FROZEN_PROCESS_RSS_BYTES,
        "peak_rss_bytes": peak_rss_bytes,
        "elapsed_seconds": elapsed,
        "output_records": output_records,
        "output_sha256": output_digest.hexdigest(),
    }


def _batch_chunk_worker(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve(strict=True)
    preregistration_path = args.preregistration.resolve(strict=True)
    batch_binary = args.batch_binary.resolve(strict=True)
    preregistration = load_preregistration(preregistration_path)
    cell_ids = expected_cell_ids(preregistration)
    if args.cell_id not in cell_ids:
        raise SchemaError("parallel child cell is outside the frozen matrix")
    if (
        args.seed_start < 0
        or not 0 < args.seed_count <= PARALLEL_BATCH_MAX_REQUESTS_PER_CHILD
        or args.seed_start + args.seed_count > expected_seed_count(args.cell_id)
        or not math.isfinite(args.generation_delay_seconds)
        or args.generation_delay_seconds < 0
    ):
        raise SchemaError("parallel child seed descriptor is outside the frozen matrix")
    cell_ordinal = cell_ids.index(args.cell_id)
    dimensions = dict(zip(DIMENSION_NAMES, args.cell_id.split("|")))
    with tempfile.TemporaryFile() as stdin:
        if args.request_file is None:
            for seed_index in range(args.seed_start, args.seed_start + args.seed_count):
                if args.generation_delay_seconds:
                    time.sleep(args.generation_delay_seconds)
                stdin.write(compact_json_line(_cell_request_at(
                    preregistration, args.cell_id, cell_ordinal, dimensions, seed_index
                )))
        else:
            request_path = args.request_file.resolve(strict=True)
            with request_path.open("rb") as source:
                for seed_index in range(args.seed_start + args.seed_count):
                    raw = source.readline(FROZEN_MAX_RECORD_BYTES + 2)
                    if not raw:
                        raise SchemaError("parallel child request slice is incomplete")
                    if seed_index >= args.seed_start:
                        stdin.write(raw)
        stdin.seek(0)
        completed = subprocess.run(
            [
                str(batch_binary), "--preregistration", str(preregistration_path),
                "--cell-id", args.cell_id, "--ephemeral-evidence-stdout",
            ],
            cwd=source_root,
            stdin=stdin,
            stdout=sys.stdout.buffer,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if completed.returncode != 0:
        raise SchemaError("parallel batch child failed")


def _measurement_worker(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve(strict=True)
    if args.kind == "batch":
        measured = run_parallel_batch(
            source_root,
            args.preregistration,
            None,
            args.cell_id,
            seed_count=args.seed_count,
        )
        print(json.dumps({
            "exit_status": 0,
            "wall_seconds": measured["elapsed_seconds"],
            "max_rss_bytes": measured["peak_rss_bytes"],
            "worker_count": measured["worker_count"],
            "max_requests_per_child": measured["max_requests_per_child"],
            "child_invocation_count": measured["child_invocation_count"],
            "wave_count": measured["wave_count"],
            "worker_rss_admission_bytes": measured["worker_rss_admission_bytes"],
            "aggregate_rss_cap_bytes": measured["aggregate_rss_cap_bytes"],
            "output_records": measured["output_records"],
            "ordered_output_sha256": measured["output_sha256"],
            "stdout_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_json": "",
        }, sort_keys=True, separators=(",", ":")))
        return
    if args.kind == "benchmark":
        command = [
            "target/release/examples/spatial_covariance_bench",
            "--tile-pixels", str(args.tile_pixels), "--dates", str(args.dates),
        ]
        stdin = subprocess.DEVNULL
    else:
        raise SchemaError("unknown measurement worker kind")
    try:
        with tempfile.TemporaryFile() as stdout:
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=source_root,
                stdin=stdin,
                stdout=stdout if args.kind == "benchmark" else subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            elapsed = time.perf_counter() - started
            stdout_size = stdout.tell() if args.kind == "benchmark" else 0
            if stdout_size > 8192:
                raise SchemaError("measured benchmark stdout exceeded its frozen cap")
            stdout.seek(0)
            stdout_bytes = stdout.read(8193) if args.kind == "benchmark" else b""
    finally:
        if hasattr(stdin, "close"):
            stdin.close()
    result = {
        "exit_status": completed.returncode,
        "wall_seconds": elapsed,
        "max_rss_bytes": _child_max_rss_bytes(),
        "stdout_bytes": len(stdout_bytes),
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stdout_json": stdout_bytes.decode("utf-8"),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


def _fresh_measurement(
    source_root: Path,
    kind: str,
    *,
    preregistration_path: Path | None = None,
    cell_id: str | None = None,
    seed_count: int | None = None,
    tile_pixels: int | None = None,
    dates: int | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "_measure-child",
        "--source-root", str(Path(source_root).resolve(strict=True)),
        "--kind", kind,
    ]
    if kind == "benchmark":
        command.extend(["--tile-pixels", str(tile_pixels), "--dates", str(dates)])
    else:
        command.extend([
            "--preregistration", str(Path(preregistration_path).resolve(strict=True)),
            "--cell-id", str(cell_id),
            "--seed-count", str(seed_count),
        ])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        measurement = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SchemaError("fresh measurement parent emitted malformed JSON") from exc
    if (
        not isinstance(measurement, dict)
        or measurement.get("exit_status") != 0
        or not isinstance(measurement.get("wall_seconds"), (int, float))
        or measurement["wall_seconds"] <= 0
        or type(measurement.get("max_rss_bytes")) is not int
        or measurement["max_rss_bytes"] <= 0
    ):
        raise SchemaError("measured release executable failed or emitted invalid resource evidence")
    return measurement


def iter_attempt_requests(preregistration: Mapping[str, Any], spec: ShardSpec) -> Iterator[dict[str, Any]]:
    validate_preregistration(preregistration)
    for cell_offset, cell_id in enumerate(spec.cell_ids):
        dimensions = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
        for seed_index in range(expected_seed_count(cell_id)):
            yield {
                "schema": "dolphinrust.spatial-covariance.attempt/4",
                "cell_id": cell_id,
                "cell_ordinal": spec.cell_ordinal_start + cell_offset,
                "seed_index": seed_index,
                "seed_sha256": _expected_seed_hash(preregistration, cell_id, seed_index),
                **dimensions,
            }


def _iter_cell_requests(
    preregistration: Mapping[str, Any], cell_id: str, seed_count: int
) -> Iterator[dict[str, Any]]:
    cell_ids = expected_cell_ids(preregistration)
    if cell_id not in cell_ids or not 0 < seed_count <= FROZEN_SEED_COUNT:
        raise SchemaError("measurement cell or seed count is outside the frozen matrix")
    cell_ordinal = cell_ids.index(cell_id)
    dimensions = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    for seed_index in range(seed_count):
        yield _cell_request_at(
            preregistration, cell_id, cell_ordinal, dimensions, seed_index
        )


def _cell_request_at(
    preregistration: Mapping[str, Any],
    cell_id: str,
    cell_ordinal: int,
    dimensions: Mapping[str, str],
    seed_index: int,
) -> dict[str, Any]:
    return {
        "schema": "dolphinrust.spatial-covariance.attempt/4",
        "cell_id": cell_id,
        "cell_ordinal": cell_ordinal,
        "seed_index": seed_index,
        "seed_sha256": _expected_seed_hash(preregistration, cell_id, seed_index),
        **dimensions,
    }


def generate_performance_probe(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    source_root: Path,
    batch_binary: Path,
    benchmark_binary: Path,
    target_wall_seconds: float,
    checkpoint_directory: Path | None = None,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    if not math.isfinite(target_wall_seconds) or target_wall_seconds <= 0:
        raise SchemaError("performance target wall seconds must be finite and positive")
    code_sha256, binary_sha256 = _validated_producer_identities(
        preregistration, source_root, batch_binary, benchmark_binary
    )
    frozen = preregistration["execution_protocol"]["performance_probe"]
    if (
        list(PERFORMANCE_CELL_IDS) != frozen["required_cell_classes"]
        or PERFORMANCE_CELL_IDS != frozen["cell_bindings"]
        or frozen.get("parallel_worker_count") != PARALLEL_BATCH_WORKER_COUNT
        or frozen.get("max_requests_per_child") != PARALLEL_BATCH_MAX_REQUESTS_PER_CHILD
        or frozen.get("worker_rss_admission_bytes") != PARALLEL_BATCH_WORKER_RSS_ADMISSION_BYTES
        or frozen.get("aggregate_rss_cap_bytes") != FROZEN_PROCESS_RSS_BYTES
        or frozen["parallel_worker_count"] * frozen["worker_rss_admission_bytes"]
        != frozen["aggregate_rss_cap_bytes"]
    ):
        raise SchemaError("performance class-to-cell bindings drifted from the frozen order")
    measurements = []
    checkpoint_root = Path(checkpoint_directory) if checkpoint_directory is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_json(preregistration["generator"])
    execution_sha256 = sha256_json(frozen)
    for cell_class in frozen["required_cell_classes"]:
        cell_id = PERFORMANCE_CELL_IDS[cell_class]
        for seed_count in frozen["seed_counts"]:
            checkpoint_path = (
                checkpoint_root / f"{cell_class}-{seed_count}.json"
                if checkpoint_root is not None else None
            )
            measured = None
            if checkpoint_path is not None and checkpoint_path.exists():
                checkpoint = _load_bounded_json(
                    checkpoint_path, FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
                    "performance measurement checkpoint",
                )
                expected_identity = {
                    "schema": "dolphinrust.spatial-covariance.performance-checkpoint/1",
                    "cell_class": cell_class,
                    "cell_id": cell_id,
                    "seed_count": seed_count,
                    "code_sha256": code_sha256,
                    "binary_sha256": binary_sha256,
                    "config_sha256": config_sha256,
                    "execution_sha256": execution_sha256,
                }
                if not isinstance(checkpoint, dict) or any(
                    checkpoint.get(name) != value for name, value in expected_identity.items()
                ):
                    raise SchemaError("performance checkpoint producer/scope identity differs")
                measured = checkpoint.get("measurement")
            if measured is None:
                measured = _fresh_measurement(
                    source_root,
                    "batch",
                    preregistration_path=preregistration_path,
                    cell_id=cell_id,
                    seed_count=seed_count,
                )
                if checkpoint_path is not None:
                    _write_bounded_json_atomic({
                        "schema": "dolphinrust.spatial-covariance.performance-checkpoint/1",
                        "cell_class": cell_class,
                        "cell_id": cell_id,
                        "seed_count": seed_count,
                        "code_sha256": code_sha256,
                        "binary_sha256": binary_sha256,
                        "config_sha256": config_sha256,
                        "execution_sha256": execution_sha256,
                        "measurement": measured,
                    }, checkpoint_path, FROZEN_MAX_RESOURCE_RECEIPT_BYTES)
            measurement = {
                "cell_class": cell_class,
                "seed_count": seed_count,
                "attempt_count": seed_count,
                "elapsed_seconds": measured["wall_seconds"],
                "peak_rss_bytes": measured["max_rss_bytes"],
                "worker_count": measured["worker_count"],
                "max_requests_per_child": measured["max_requests_per_child"],
                "child_invocation_count": measured["child_invocation_count"],
                "wave_count": measured["wave_count"],
                "worker_rss_admission_bytes": measured["worker_rss_admission_bytes"],
                "aggregate_rss_cap_bytes": measured["aggregate_rss_cap_bytes"],
                "output_records": measured["output_records"],
                "ordered_output_sha256": measured["ordered_output_sha256"],
                "outcomes_persisted": False,
            }
            measurements.append(measurement)
    total_attempts = sum(item["attempt_count"] for item in measurements)
    total_elapsed = sum(item["elapsed_seconds"] for item in measurements)
    attempts_per_second = total_attempts / total_elapsed
    projected_serial_seconds = (
        preregistration["matrix_contract"]["expected_attempt_count"] / attempts_per_second
    )
    reserve_fraction = frozen["reserve_fraction"]
    receipt = {
        "schema": "dolphinrust.spatial-covariance.performance-probe",
        "schema_version": 1,
        "outcomes_persisted": False,
        "seed_counts": list(frozen["seed_counts"]),
        "cell_classes": list(frozen["required_cell_classes"]),
        "measurements": measurements,
        "attempts_per_second": attempts_per_second,
        "peak_rss_bytes": max(item["peak_rss_bytes"] for item in measurements),
        "target_wall_seconds": target_wall_seconds,
        "reserve_fraction": reserve_fraction,
        "projected_serial_seconds": projected_serial_seconds,
        "derived_concurrency": math.ceil(
            projected_serial_seconds / (target_wall_seconds * (1.0 - reserve_fraction))
        ),
        "code_sha256": code_sha256,
        "binary_sha256": binary_sha256,
        "config_sha256": config_sha256,
    }
    _validate_performance_probe(preregistration, receipt, code_sha256, binary_sha256)
    return receipt


def _growth_exponent(points: list[tuple[int, int]]) -> float:
    x = [math.log(float(scale)) for scale, _ in points]
    y = [math.log(float(value)) for _, value in points]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0:
        raise SchemaError("resource growth axis is not identifiable")
    return sum(
        (left - x_mean) * (right - y_mean) for left, right in zip(x, y)
    ) / denominator


def _benchmark_receipt_parts(
    allocation: Mapping[str, Any], matrix: Mapping[str, Any]
) -> dict[str, Any]:
    components = allocation.get("allocation_components")
    if not isinstance(components, list):
        raise SchemaError("benchmark omitted its allocation components")
    dependency_cone = {
        "model": "spatial-query-cone-v1",
        "tile_pixels": matrix["tile_pixels"],
        "date_count": matrix["dates"],
        "maximum_sources": allocation.get("maximum_sources_per_block"),
        "block_count": allocation.get("block_count"),
        "maximum_dependency_depth": allocation.get("maximum_dependency_depth"),
        "reference_cone_sources": allocation.get("reference_cone_sources"),
    }
    microbatch_pixels = min(matrix["tile_pixels"], 4096)
    microbatch = {
        "model": "bounded-microbatch-v1",
        "microbatch_pixels": microbatch_pixels,
        "batch_count": math.ceil(matrix["tile_pixels"] / microbatch_pixels),
    }
    allocation_model = {
        "model": "production-runtime-resource-receipt-v1",
        "source": "spatial_covariance_bench captured stdout",
    }
    return {
        "allocation_model": allocation_model,
        "allocation_model_sha256": sha256_json(allocation_model),
        "dependency_cone": dependency_cone,
        "dependency_cone_sha256": sha256_json(dependency_cone),
        "microbatch": microbatch,
        "microbatch_sha256": sha256_json(microbatch),
        "allocation_components": components,
    }


def generate_resource_receipts(
    preregistration: Mapping[str, Any],
    source_root: Path,
    batch_binary: Path,
    benchmark_binary: Path,
) -> list[dict[str, Any]]:
    validate_preregistration(preregistration)
    _, binary_sha256 = _validated_producer_identities(
        preregistration, source_root, batch_binary, benchmark_binary
    )
    sampling = preregistration["resource_sampling"]
    provenance = _host_provenance(preregistration)
    items = []
    matrix_by_id = {item["id"]: item for item in preregistration["resource_matrix"]}
    for matrix in preregistration["resource_matrix"]:
        for _ in range(sampling["warmup_runs"]):
            _fresh_measurement(
                source_root, "benchmark",
                tile_pixels=matrix["tile_pixels"], dates=matrix["dates"],
            )
        observations = []
        receipt_parts = None
        for repetition in range(sampling["measured_repetitions"]):
            measured = _fresh_measurement(
                source_root, "benchmark",
                tile_pixels=matrix["tile_pixels"], dates=matrix["dates"],
            )
            try:
                allocation = json.loads(measured["stdout_json"])
            except json.JSONDecodeError as exc:
                raise SchemaError("measured benchmark emitted malformed JSON") from exc
            current_parts = _benchmark_receipt_parts(allocation, matrix)
            if receipt_parts is None:
                receipt_parts = current_parts
            elif current_parts != receipt_parts:
                raise SchemaError("benchmark allocation receipt changed across repetitions")
            raw_measurement = {
                "command": [
                    "target/release/examples/spatial_covariance_bench",
                    "--tile-pixels", str(matrix["tile_pixels"]),
                    "--dates", str(matrix["dates"]),
                ],
                "exit_status": measured["exit_status"],
                "wall_seconds": measured["wall_seconds"],
                "max_rss_bytes": measured["max_rss_bytes"],
                "rss_sampler": sampling["rss_sampler"],
                "rss_field": sampling["rss_field"],
                "os": provenance["os"],
                "hardware_class": provenance["hardware_class"],
                "ram_bytes": provenance["ram_bytes"],
                "tool_versions": provenance["tool_versions"],
                "stdout_bytes": measured["stdout_bytes"],
                "stdout_sha256": measured["stdout_sha256"],
                "stdout_json": measured["stdout_json"],
            }
            observations.append({
                "repetition": repetition,
                "tile_pixels": matrix["tile_pixels"],
                "date_count": matrix["dates"],
                "peak_rss_bytes": measured["max_rss_bytes"],
                "wall_seconds": measured["wall_seconds"],
                "raw_measurement": raw_measurement,
                "raw_measurement_sha256": sha256_json(raw_measurement),
            })
        assert receipt_parts is not None
        items.append({
            "resource_id": matrix["id"],
            "status": "",
            "rss_bytes": max(item["peak_rss_bytes"] for item in observations),
            "growth_class": "",
            "resource_hash": "",
            "config_hash": sha256_json(preregistration["generator"]),
            "binary_hash": binary_sha256,
            "os": provenance["os"],
            "hardware_class": provenance["hardware_class"],
            "ram_bytes": provenance["ram_bytes"],
            "rss_sampler": sampling["rss_sampler"],
            "rss_field": sampling["rss_field"],
            "warmup_runs": sampling["warmup_runs"],
            "measured_repetitions": sampling["measured_repetitions"],
            "tool_versions": sampling["tool_versions"],
            "growth_observation": observations,
            "area_growth_exponent": 0.0,
            "date_growth_exponent": 0.0,
            "acceptance": sampling["acceptance"],
            **receipt_parts,
        })
    peaks = {item["resource_id"]: item["rss_bytes"] for item in items}
    area_exponent = _growth_exponent([
        (matrix_by_id[name]["tile_pixels"], peaks[name])
        for name in ("area_128_dates_26", "area_256_dates_26", "area_512_dates_26")
    ])
    date_exponent = _growth_exponent([
        (matrix_by_id[name]["dates"], peaks[name])
        for name in ("area_256_dates_13", "area_256_dates_26", "area_256_dates_52")
    ])
    growth_class = "linear" if max(area_exponent, date_exponent) <= 1.25 else "superlinear"
    for item in items:
        item["area_growth_exponent"] = area_exponent
        item["date_growth_exponent"] = date_exponent
        item["growth_class"] = growth_class
        item["status"] = (
            "pass"
            if item["rss_bytes"] <= FROZEN_PROCESS_RSS_BYTES and growth_class == "linear"
            else "fail"
        )
        item["resource_hash"] = sha256_json({
            key: value for key, value in item.items() if key != "resource_hash"
        })
    _validate_resources(preregistration, items, binary_sha256)
    return items


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


def write_run_manifest_atomic(run_manifest: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    destination = Path(destination)
    if destination.name.endswith(".partial"):
        raise SchemaError("run-manifest destination must not use the .partial suffix")
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise SchemaError("refusing to overwrite run-manifest state")
    encoded = compact_json_line(run_manifest)
    if len(encoded) > FROZEN_MAX_RUN_MANIFEST_BYTES:
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
        "seed_counts": list(spec.seed_counts),
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
    expected_seed_count_override: int | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    if not _is_digest(code_sha256) or not _is_digest(binary_sha256):
        raise SchemaError("cell commit code/binary identity is invalid")
    transport_path = Path(transport_path)
    seed_count = expected_seed_count_override if expected_seed_count_override is not None else expected_seed_count(cell_id)
    accumulator = CellAccumulator(
        preregistration, cell_id, cell_ordinal, seed_count, code_sha256, binary_sha256,
        artifact_root=artifact_root,
    )
    with transport_path.open("rb") as handle:
        for line_number in range(seed_count):
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
    _fsync_directory(Path(destination).parent)
    transport_path.unlink()
    _fsync_directory(transport_path.parent)
    return receipt


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


AttemptRegenerator = Callable[[str, int], Iterable[Mapping[str, Any]]]


def rust_attempt_regenerator(
    preregistration: Mapping[str, Any], preregistration_path: Path, batch_binary: Path
) -> AttemptRegenerator:
    preregistration_path = Path(preregistration_path).resolve(strict=True)
    batch_binary = Path(batch_binary).resolve(strict=True)

    def regenerate(cell_id: str, cell_ordinal: int) -> Iterator[Mapping[str, Any]]:
        spec = ShardSpec(
            0,
            cell_ordinal,
            cell_ordinal + 1,
            (cell_id,),
            (expected_seed_count(cell_id),),
        )
        process = subprocess.Popen(
            [
                str(batch_binary),
                "--preregistration", str(preregistration_path),
                "--cell-id", cell_id,
                "--ephemeral-evidence-stdout",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            for request in iter_attempt_requests(preregistration, spec):
                process.stdin.write(compact_json_line(request))
                process.stdin.flush()
                raw = process.stdout.readline(FROZEN_MAX_RECORD_BYTES + 2)
                if not raw or len(raw) > FROZEN_MAX_RECORD_BYTES or not raw.endswith(b"\n"):
                    raise SchemaError("Rust attempt replay is incomplete or exceeds the record cap")
                try:
                    attempt = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SchemaError("Rust attempt replay emitted malformed JSON") from exc
                if not isinstance(attempt, dict):
                    raise SchemaError("Rust attempt replay emitted a non-object record")
                yield attempt
            process.stdin.close()
            if process.stdout.read(1) or process.wait() != 0:
                raise SchemaError("Rust attempt replay failed or emitted top-up evidence")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            process.stdout.close()

    return regenerate


class _TraceAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums: list[float] | None = None
        self.squares: list[float] | None = None

    def add(self, values: list[float]) -> None:
        if self.sums is None:
            self.sums = [0.0] * len(values)
            self.squares = [0.0] * len(values)
        if len(values) != len(self.sums):
            raise SchemaError("matched cohort vector length changed")
        assert self.squares is not None
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise SchemaError("matched cohort contains a nonfinite error")
            self.sums[index] += value
            self.squares[index] += value * value
        self.count += 1

    def covariance_trace(self) -> float:
        if self.count < 2 or self.sums is None or self.squares is None:
            raise SchemaError("matched empirical covariance needs at least two seeds")
        return sum(
            (square - total * total / self.count) / self.count
            for total, square in zip(self.sums, self.squares)
        )


def _digest_update(digest: Any, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def generate_matched_pair_cohorts(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    batch_binary: Path,
    code_sha256: str,
    binary_sha256: str,
    seed_count: int,
) -> list[dict[str, Any]]:
    validate_preregistration(preregistration)
    cell_ids = expected_cell_ids(preregistration)
    frozen_cohort = preregistration["execution_protocol"]["matched_pair_cohort"]
    if (
        frozen_cohort.get("positive_cell") != MATCHED_POSITIVE_CELL
        or frozen_cohort.get("negative_cell") != MATCHED_NEGATIVE_CELL
        or frozen_cohort.get("seed_count") != MATCHED_COHORT_SEED_COUNT
        or seed_count != frozen_cohort.get("seed_count")
        or MATCHED_POSITIVE_CELL not in cell_ids
        or MATCHED_NEGATIVE_CELL not in cell_ids
    ):
        raise SchemaError("matched cohort cells or seed count are outside the frozen matrix")
    preregistration_path = Path(preregistration_path).resolve(strict=True)
    batch_binary = Path(batch_binary).resolve(strict=True)
    processes = []
    for cell_id in (MATCHED_POSITIVE_CELL, MATCHED_NEGATIVE_CELL):
        process = subprocess.Popen(
            [
                str(batch_binary),
                "--preregistration", str(preregistration_path),
                "--cell-id", cell_id,
                "--ephemeral-evidence-stdout",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        processes.append(process)
    accumulators = [
        CellAccumulator(
            preregistration, cell_id, cell_ids.index(cell_id), seed_count,
            code_sha256, binary_sha256,
        )
        for cell_id in (MATCHED_POSITIVE_CELL, MATCHED_NEGATIVE_CELL)
    ]
    target_errors = [_TraceAccumulator(), _TraceAccumulator()]
    reference_errors = [_TraceAccumulator(), _TraceAccumulator()]
    difference_errors = [_TraceAccumulator(), _TraceAccumulator()]
    predicted_difference_totals = [0.0, 0.0]
    predicted_marginal_totals = [0.0, 0.0]
    attempt_digests = [
        hashlib.sha256(b"dolphinrust:matched-positive-attempts:v1\0"),
        hashlib.sha256(b"dolphinrust:matched-negative-attempts:v1\0"),
    ]
    marginal_digest = hashlib.sha256(b"dolphinrust:matched-marginal-dgp:v1\0")
    target_support_digest = hashlib.sha256(b"dolphinrust:matched-target-support:v1\0")
    reference_support_digest = hashlib.sha256(b"dolphinrust:matched-reference-support:v1\0")
    latent_digest = hashlib.sha256(b"dolphinrust:matched-latent-history:v1\0")
    orientation_digest = hashlib.sha256(b"dolphinrust:matched-phase-orientation:v1\0")
    try:
        for seed_index in range(seed_count):
            attempts = []
            regenerated = []
            for index, (process, cell_id) in enumerate(zip(
                processes, (MATCHED_POSITIVE_CELL, MATCHED_NEGATIVE_CELL)
            )):
                assert process.stdin is not None and process.stdout is not None
                request = _cell_request_at(
                    preregistration, cell_id, cell_ids.index(cell_id),
                    dict(zip(DIMENSION_NAMES, cell_id.split("|"))), seed_index,
                )
                process.stdin.write(compact_json_line(request))
                process.stdin.flush()
                raw = process.stdout.readline(FROZEN_MAX_RECORD_BYTES + 2)
                if not raw or len(raw) > FROZEN_MAX_RECORD_BYTES or not raw.endswith(b"\n"):
                    raise SchemaError("matched Rust cohort output is incomplete or oversized")
                try:
                    attempt = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SchemaError("matched Rust cohort output is malformed") from exc
                accumulators[index].add(attempt)
                attempt_digests[index].update(len(raw).to_bytes(8, "big"))
                attempt_digests[index].update(raw)
                attempts.append(attempt)
                regenerated.append(
                    regenerate_frozen_attempt_inputs(preregistration, cell_id, seed_index)
                )
            equality_fields = (
                "target_support_sha256", "reference_support_sha256",
                "latent_history_sha256", "raw_dgp_identity_sha256",
                "target_marginal_oracle_sha256", "reference_marginal_oracle_sha256",
                "target_coordinate", "reference_coordinate", "date_axis_sha256",
                "target_source_count", "reference_source_count", "union_source_count",
                "source_correlation_model", "source_correlation_distance_scale_pixels",
            )
            if any(regenerated[0][name] != regenerated[1][name] for name in equality_fields):
                raise SchemaError("matched signed Rust cohorts do not share exact marginals")
            common = {name: regenerated[0][name] for name in equality_fields}
            common["seed_index"] = seed_index
            _digest_update(marginal_digest, common)
            _digest_update(target_support_digest, {
                "seed_index": seed_index,
                "sha256": regenerated[0]["target_support_sha256"],
            })
            _digest_update(reference_support_digest, {
                "seed_index": seed_index,
                "sha256": regenerated[0]["reference_support_sha256"],
            })
            _digest_update(latent_digest, {
                "seed_index": seed_index,
                "sha256": regenerated[0]["latent_history_sha256"],
            })
            _digest_update(orientation_digest, {
                "seed_index": seed_index,
                "date_axis_sha256": regenerated[0]["date_axis_sha256"],
                "target_coordinate": regenerated[0]["target_coordinate"],
                "reference_coordinate": regenerated[0]["reference_coordinate"],
            })
            for index, (attempt, truth) in enumerate(zip(attempts, regenerated)):
                target_error = [
                    estimate - latent
                    for estimate, latent in zip(
                        attempt["target_estimate_history"], truth["latent_target_history"]
                    )
                ]
                reference_error = [
                    estimate - latent
                    for estimate, latent in zip(
                        attempt["reference_estimate_history"], truth["latent_reference_history"]
                    )
                ]
                target_errors[index].add(target_error)
                reference_errors[index].add(reference_error)
                difference_errors[index].add([
                    target - reference
                    for target, reference in zip(target_error, reference_error)
                ])
                difference = attempt["predicted_difference_covariance"]
                predicted_difference_totals[index] += sum(
                    difference[date][date] for date in range(len(difference))
                )
                joint = attempt["production_operator_matrix"]
                dates = len(joint) // 2
                predicted_marginal_totals[index] += sum(
                    joint[date][date] + joint[dates + date][dates + date]
                    for date in range(dates)
                )
        for process in processes:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.close()
            if process.stdout.read(1) or process.wait() != 0:
                raise SchemaError("matched Rust cohort producer failed or emitted top-up evidence")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
    shared = {
        "marginal_dgp_digest": marginal_digest.hexdigest(),
        "target_support_digest": target_support_digest.hexdigest(),
        "reference_support_digest": reference_support_digest.hexdigest(),
        "latent_history_digest": latent_digest.hexdigest(),
        "phase_orientation_digest": orientation_digest.hexdigest(),
        "positive_cell_id": MATCHED_POSITIVE_CELL,
        "negative_cell_id": MATCHED_NEGATIVE_CELL,
        "seed_count": seed_count,
        "positive_attempt_digest": attempt_digests[0].hexdigest(),
        "negative_attempt_digest": attempt_digests[1].hexdigest(),
        "code_sha256": code_sha256,
        "binary_sha256": binary_sha256,
        "config_sha256": sha256_json(preregistration["generator"]),
    }
    positive_predicted = predicted_difference_totals[0] / seed_count
    negative_predicted = predicted_difference_totals[1] / seed_count
    independent_predicted = sum(predicted_marginal_totals) / (2 * seed_count)
    positive_empirical = difference_errors[0].covariance_trace()
    negative_empirical = difference_errors[1].covariance_trace()
    independent_empirical = 0.5 * sum(
        target.covariance_trace() + reference.covariance_trace()
        for target, reference in zip(target_errors, reference_errors)
    )
    result = [
        {"coupling": "positive", **shared,
         "predicted_covariance_trace": positive_predicted,
         "empirical_error_covariance_trace": positive_empirical},
        {"coupling": "independent", **shared,
         "predicted_covariance_trace": independent_predicted,
         "empirical_error_covariance_trace": independent_empirical},
        {"coupling": "negative", **shared,
         "predicted_covariance_trace": negative_predicted,
         "empirical_error_covariance_trace": negative_empirical},
    ]
    validate_matched_pair_cohorts(
        result, code_sha256, binary_sha256,
        sha256_json(preregistration["generator"]),
    )
    return result


def generate_preoutcome_receipts(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    source_root: Path,
    batch_binary: Path,
    benchmark_binary: Path,
    target_wall_seconds: float,
    destination: Path,
) -> dict[str, Any]:
    destination = Path(destination)
    if destination.exists():
        raise SchemaError("refusing to overwrite pre-outcome receipt directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    code_sha256, binary_sha256 = _validated_producer_identities(
        preregistration, source_root, batch_binary, benchmark_binary
    )
    partial = Path(tempfile.mkdtemp(
        prefix=f"{destination.name}.partial-", dir=destination.parent
    ))
    try:
        performance = generate_performance_probe(
            preregistration, preregistration_path, source_root,
            batch_binary, benchmark_binary, target_wall_seconds,
        )
        resources = generate_resource_receipts(
            preregistration, source_root, batch_binary, benchmark_binary
        )
        matched = generate_matched_pair_cohorts(
            preregistration, preregistration_path, batch_binary,
            code_sha256, binary_sha256,
            preregistration["execution_protocol"]["matched_pair_cohort"]["seed_count"],
        )
        receipts = {}
        for name, value in (
            ("performance.json", performance),
            ("resources.json", resources),
            ("matched-pair-cohorts.json", matched),
        ):
            receipts[name] = _write_bounded_json_atomic(
                value, partial / name, FROZEN_MAX_RESOURCE_RECEIPT_BYTES
            )
        manifest = {
            "schema": "dolphinrust.spatial-covariance.preoutcome-receipts/1",
            "code_sha256": code_sha256,
            "binary_sha256": binary_sha256,
            "config_sha256": sha256_json(preregistration["generator"]),
            "preregistration_sha256": preregistration_digest(preregistration),
            "receipts": receipts,
        }
        manifest_receipt = _write_bounded_json_atomic(
            manifest, partial / "manifest.json", FROZEN_MAX_RESOURCE_RECEIPT_BYTES
        )
        directory_handle = os.open(partial, os.O_RDONLY)
        try:
            os.fsync(directory_handle)
        finally:
            os.close(directory_handle)
        os.replace(partial, destination)
        return {
            "directory": str(destination),
            "manifest_sha256": manifest_receipt["sha256"],
            "code_sha256": code_sha256,
            "binary_sha256": binary_sha256,
        }
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def validate_preoutcome_receipts(
    preregistration: Mapping[str, Any],
    directory: Path,
    code_sha256: str,
    binary_sha256: str,
) -> None:
    directory = Path(directory).resolve(strict=True)
    manifest = _load_bounded_json(
        directory / "manifest.json",
        FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
        "pre-outcome receipt manifest",
    )
    expected_identity = {
        "schema": "dolphinrust.spatial-covariance.preoutcome-receipts/1",
        "code_sha256": code_sha256,
        "binary_sha256": binary_sha256,
        "config_sha256": sha256_json(preregistration["generator"]),
        "preregistration_sha256": preregistration_digest(preregistration),
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(name) != value for name, value in expected_identity.items()
    ):
        raise SchemaError("pre-outcome receipt manifest identity differs")
    expected_names = (
        "performance.json", "resources.json", "matched-pair-cohorts.json"
    )
    if set(manifest.get("receipts", {})) != set(expected_names):
        raise SchemaError("pre-outcome receipt manifest set differs")
    values = {}
    for name in expected_names:
        path = directory / name
        raw = _read_bounded_bytes(path, FROZEN_MAX_RESOURCE_RECEIPT_BYTES, name)
        receipt = manifest["receipts"].get(name)
        if (
            not isinstance(receipt, dict)
            or receipt.get("sha256") != hashlib.sha256(raw).hexdigest()
            or receipt.get("bytes") != len(raw)
        ):
            raise SchemaError(f"pre-outcome receipt {name} differs from its manifest")
        try:
            values[name] = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SchemaError(f"pre-outcome receipt {name} is malformed") from exc
    _validate_performance_probe(
        preregistration, values["performance.json"], code_sha256, binary_sha256
    )
    if _validate_resources(
        preregistration, values["resources.json"], binary_sha256
    ) != ["pass"] * 5:
        raise SchemaError("pre-outcome resource receipts did not pass")
    validate_matched_pair_cohorts(
        values["matched-pair-cohorts.json"],
        code_sha256,
        binary_sha256,
        sha256_json(preregistration["generator"]),
    )


def _summary_root(
    preregistration: Mapping[str, Any],
    directory: Path,
    spec: ShardSpec,
    code_sha256: str,
    binary_sha256: str,
    attempt_regenerator: AttemptRegenerator | None = None,
    artifact_root: Path | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256(b"dolphinrust:spatial-covariance:cell-summary-root:v4\0")
    total_bytes = 0
    for offset, cell_id in enumerate(spec.cell_ids):
        path = directory / f"cell-{spec.cell_ordinal_start + offset:05d}.jsonl"
        summary, raw = _read_single_json_record(
            path,
            preregistration["execution_protocol"]["max_encoded_cell_summary_bytes"],
            f"cell {cell_id} compact summary",
        )
        cell_ordinal = spec.cell_ordinal_start + offset
        validate_cell_summary(preregistration, summary, cell_id, cell_ordinal, code_sha256, binary_sha256)
        if attempt_regenerator is not None:
            accumulator = CellAccumulator(
                preregistration, cell_id, cell_ordinal, expected_seed_count(cell_id), code_sha256, binary_sha256,
                artifact_root=artifact_root,
            )
            for attempt in attempt_regenerator(cell_id, cell_ordinal):
                accumulator.add(attempt)
            regenerated = accumulator.finalize()
            if compact_json_line(regenerated) != raw:
                raise SchemaError(f"cell {cell_id} compact summary does not match deterministic replay")
        digest.update(offset.to_bytes(8, "big"))
        digest.update(hashlib.sha256(raw).digest())
        total_bytes += len(raw)
    return digest.hexdigest(), total_bytes


def run_outcomes(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    source_root: Path,
    batch_binary: Path,
    benchmark_binary: Path,
    preoutcome_directory: Path,
    run_root: Path,
    shard_index: int,
    spec_override: ShardSpec | None = None,
    attempt_regenerator_override: AttemptRegenerator | None = None,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    code_sha256, binary_sha256 = _validated_producer_identities(
        preregistration, source_root, batch_binary, benchmark_binary
    )
    validate_preoutcome_receipts(
        preregistration, preoutcome_directory, code_sha256, binary_sha256
    )
    if spec_override is None:
        spec = next(
            (item for item in iter_shard_specs(preregistration) if item.index == shard_index),
            None,
        )
        if spec is None:
            raise SchemaError("outcome shard index is outside the frozen plan")
    else:
        spec = spec_override
        if spec.index != shard_index:
            raise SchemaError("outcome shard override index differs")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_root = run_root.resolve(strict=True)
    summary_directory = run_root / "cells" / f"shard-{spec.index:05d}"
    summary_directory.mkdir(parents=True, exist_ok=True)
    transport_directory = run_root / "transports"
    transport_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory = run_root / "shards"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_directory / f"manifest-{spec.index:05d}.jsonl"
    attempt_regenerator = (
        rust_attempt_regenerator(preregistration, preregistration_path, batch_binary)
        if attempt_regenerator_override is None else attempt_regenerator_override
    )
    if manifest_path.exists():
        if not committed_shard_matches(
            preregistration, spec, run_root, manifest_path,
            code_sha256, binary_sha256, attempt_regenerator,
        ):
            raise SchemaError("committed outcome shard failed exact resume validation")
        return {
            "schema": "dolphinrust.spatial-covariance.outcome-run/1",
            "shard_index": spec.index,
            "reusable": True,
            "generated_cells": 0,
            "resumed_cells": len(spec.cell_ids),
            "manifest": str(manifest_path),
        }
    started = time.perf_counter()
    peak_rss_bytes = 0
    generated_cells = 0
    resumed_cells = 0
    for offset, cell_id in enumerate(spec.cell_ids):
        cell_ordinal = spec.cell_ordinal_start + offset
        destination = summary_directory / f"cell-{cell_ordinal:05d}.jsonl"
        transport = transport_directory / f"cell-{cell_ordinal:05d}.jsonl"
        cell_spec = ShardSpec(
            spec.index,
            cell_ordinal,
            cell_ordinal + 1,
            (cell_id,),
            (expected_seed_count(cell_id),),
        )
        if destination.exists():
            _summary_root(
                preregistration, summary_directory, cell_spec,
                code_sha256, binary_sha256, attempt_regenerator,
                artifact_root=run_root,
            )
            for residual in (
                transport,
                transport.with_name(transport.name + ".partial"),
            ):
                if residual.exists() or residual.is_symlink():
                    metadata = residual.lstat()
                    if residual.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                        raise SchemaError("owned residual cell transport is not a regular file")
                    residual.unlink()
                    _fsync_directory(residual.parent)
            resumed_cells += 1
            continue
        partial_summary = destination.with_name(destination.name + ".partial")
        partial_summary.unlink(missing_ok=True)
        if not transport.exists():
            transport.with_name(transport.name + ".partial").unlink(missing_ok=True)
            measurement = run_parallel_batch(
                source_root,
                preregistration_path,
                None,
                cell_id,
                transport,
                expected_seed_count(cell_id),
                batch_binary=batch_binary,
            )
            peak_rss_bytes = max(peak_rss_bytes, measurement["peak_rss_bytes"])
        commit_cell_transport(
            preregistration,
            cell_id,
            cell_ordinal,
            transport,
            destination,
            code_sha256,
            binary_sha256,
            artifact_root=run_root,
        )
        generated_cells += 1
    peak_rss_bytes = max(peak_rss_bytes, _child_max_rss_bytes())
    if peak_rss_bytes <= 0:
        _summary_root(
            preregistration, summary_directory, spec,
            code_sha256, binary_sha256, attempt_regenerator,
            artifact_root=run_root,
        )
        peak_rss_bytes = _child_max_rss_bytes()
    receipt = commit_output_shard(
        preregistration,
        spec,
        run_root,
        summary_directory,
        manifest_path,
        code_sha256,
        binary_sha256,
        time.perf_counter() - started,
        peak_rss_bytes,
        attempt_regenerator,
    )
    return {
        "schema": "dolphinrust.spatial-covariance.outcome-run/1",
        "shard_index": spec.index,
        "reusable": False,
        "generated_cells": generated_cells,
        "resumed_cells": resumed_cells,
        "manifest": str(manifest_path),
        "manifest_sha256": receipt["sha256"],
    }


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
    attempt_regenerator: AttemptRegenerator | None = None,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    if attempt_regenerator is None:
        raise SchemaError("exact shard commit requires the Rust spatial_covariance_batch replay executable")
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
    summary_digest, summary_bytes = _summary_root(
        preregistration, directory, spec, code_sha256, binary_sha256,
        attempt_regenerator, artifact_root=root,
    )
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
    receipt = write_jsonl_atomic(
        (manifest,), manifest_path,
        byte_limit=preregistration["execution_protocol"]["max_encoded_shard_manifest_bytes"],
    )
    descriptor = root / "requests" / f"shard-{spec.index:05d}.jsonl"
    descriptor.unlink(missing_ok=True)
    requests_directory = descriptor.parent
    if requests_directory.exists():
        descriptor_directory = os.open(requests_directory, os.O_RDONLY)
        try:
            os.fsync(descriptor_directory)
        finally:
            os.close(descriptor_directory)
    return receipt


def committed_shard_matches(
    preregistration: Mapping[str, Any],
    spec: ShardSpec,
    run_root: Path,
    manifest_path: Path,
    expected_code_sha256: str,
    expected_binary_sha256: str,
    attempt_regenerator: AttemptRegenerator | None = None,
) -> bool:
    try:
        if attempt_regenerator is None:
            return False
        manifest, _, _ = _read_hashed_json_record(
            Path(manifest_path),
            preregistration["execution_protocol"]["max_encoded_shard_manifest_bytes"],
            f"shard {spec.index} manifest",
        )
        return _committed_shard_matches_manifest(
            preregistration, spec, run_root, manifest, expected_code_sha256,
            expected_binary_sha256, attempt_regenerator,
        )
    except (OSError, ValueError, json.JSONDecodeError, SchemaError):
        return False


def _committed_shard_matches_manifest(
    preregistration: Mapping[str, Any],
    spec: ShardSpec,
    run_root: Path,
    manifest: Mapping[str, Any],
    expected_code_sha256: str,
    expected_binary_sha256: str,
    attempt_regenerator: AttemptRegenerator,
) -> bool:
    try:
        validate_shard_manifest(preregistration, manifest, spec)
        if manifest["code_sha256"] != expected_code_sha256 or manifest["binary_sha256"] != expected_binary_sha256:
            return False
        directory = resolve_below_run_root(Path(run_root), manifest["summary_path"], "compact summary directory")
        digest, size = _summary_root(
            preregistration,
            directory,
            spec,
            expected_code_sha256,
            expected_binary_sha256,
            attempt_regenerator,
            artifact_root=run_root,
        )
        return digest == manifest["summary_sha256"] and size == manifest["summary_bytes"]
    except (OSError, ValueError, json.JSONDecodeError, SchemaError):
        return False


def build_run_manifest(
    preregistration: Mapping[str, Any],
    run_root: Path,
    shard_manifest_paths: Iterable[Path],
    code_sha256: str,
    binary_sha256: str,
    performance_probe: Mapping[str, Any],
    resources: list[Mapping[str, Any]],
    attempt_regenerator: AttemptRegenerator | None = None,
    production_parity_fixture: Mapping[str, Any] | None = None,
    matched_pair_cohorts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    if attempt_regenerator is None:
        raise SchemaError(
            "exact shard assembly requires the Rust spatial_covariance_batch replay executable"
        )
    if production_parity_fixture is None or matched_pair_cohorts is None:
        raise SchemaError("run assembly requires production parity and matched cohort evidence")
    paths = tuple(shard_manifest_paths)
    if len(paths) != FROZEN_SHARD_COUNT:
        raise SchemaError("run manifest requires exactly four compact shards")
    _validate_performance_probe(preregistration, performance_probe, code_sha256, binary_sha256)
    _validate_resources(preregistration, resources, binary_sha256)
    validate_matched_pair_cohorts(
        matched_pair_cohorts, code_sha256, binary_sha256,
        sha256_json(preregistration["generator"]),
    )
    root = Path(run_root).resolve(strict=True)
    entries = []
    digests = []
    for spec, path in zip(iter_shard_specs(preregistration), paths):
        resolved = Path(path).resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        manifest, _, digest = _read_hashed_json_record(
            resolved,
            preregistration["execution_protocol"]["max_encoded_shard_manifest_bytes"],
            f"shard {spec.index} manifest",
        )
        if not _committed_shard_matches_manifest(
            preregistration,
            spec,
            root,
            manifest,
            code_sha256,
            binary_sha256,
            attempt_regenerator,
        ):
            raise SchemaError(f"shard {spec.index} is not exact compact committed evidence")
        entries.append({"path": relative, "sha256": digest})
        digests.append(digest)
    return {"schema": "dolphinrust.spatial-covariance.run-manifest/4", "schema_version": 4,
            "preregistration_sha256": preregistration_digest(preregistration), "code_sha256": code_sha256,
            "binary_sha256": binary_sha256, "generator_protocol_sha256": sha256_json(preregistration["execution_protocol"]),
            "performance_probe": dict(performance_probe), "resources": [dict(item) for item in resources],
            "shard_manifests": entries, "result_root_sha256": result_root_sha256(digests),
            "production_parity_fixture": dict(production_parity_fixture),
            "production_parity_fixture_sha256": sha256_json(production_parity_fixture),
            "matched_pair_cohorts": [dict(item) for item in matched_pair_cohorts]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("spatial_covariance_preregistration.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("_measure-child", help=argparse.SUPPRESS)
    worker.add_argument("--source-root", type=Path, required=True)
    worker.add_argument("--kind", choices=("batch", "benchmark"), required=True)
    worker.add_argument("--preregistration", type=Path)
    worker.add_argument("--request-file", type=Path)
    worker.add_argument("--cell-id")
    worker.add_argument("--seed-count", type=int)
    worker.add_argument("--tile-pixels", type=int)
    worker.add_argument("--dates", type=int)
    batch_child = commands.add_parser("_batch-chunk-child", help=argparse.SUPPRESS)
    batch_child.add_argument("--source-root", type=Path, required=True)
    batch_child.add_argument("--preregistration", type=Path, required=True)
    batch_child.add_argument("--batch-binary", type=Path, required=True)
    batch_child.add_argument("--cell-id", required=True)
    batch_child.add_argument("--seed-start", type=int, required=True)
    batch_child.add_argument("--seed-count", type=int, required=True)
    batch_child.add_argument("--generation-delay-seconds", type=float, default=0.0)
    batch_child.add_argument("--request-file", type=Path)
    capture = commands.add_parser("capture-resource", help="capture one benchmark allocation stdout record")
    capture.add_argument("--destination", type=Path, required=True)
    capture.add_argument("benchmark_command", nargs=argparse.REMAINDER)
    prepare = commands.add_parser("prepare", help="write one compact deterministic shard descriptor")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--shard-index", type=int, required=True)
    reduce_cell = commands.add_parser("reduce-cell", help="independently reduce one ephemeral cell transport")
    reduce_cell.add_argument("--run-root", type=Path, required=True)
    reduce_cell.add_argument("--cell-ordinal", type=int, required=True)
    reduce_cell.add_argument("--transport", type=Path, required=True)
    reduce_cell.add_argument("--destination", type=Path, required=True)
    commit = commands.add_parser("commit", help="validate and atomically commit one compact shard")
    commit.add_argument("--run-root", type=Path, required=True)
    commit.add_argument("--shard-index", type=int, required=True)
    commit.add_argument("--summary-directory", type=Path, required=True)
    commit.add_argument("--manifest", type=Path, required=True)
    commit.add_argument("--elapsed-seconds", type=float, required=True)
    commit.add_argument("--peak-rss-bytes", type=int, required=True)
    resume = commands.add_parser("resume", help="verify whether one committed shard is exactly reusable")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument("--shard-index", type=int, required=True)
    resume.add_argument("--manifest", type=Path, required=True)
    assemble = commands.add_parser("assemble", help="atomically assemble the final run manifest")
    assemble.add_argument("--run-root", type=Path, required=True)
    assemble.add_argument("--shard-manifest-directory", type=Path, required=True)
    assemble.add_argument("--performance-probe", type=Path, required=True)
    assemble.add_argument("--resources", type=Path, required=True)
    assemble.add_argument("--production-parity-fixture", type=Path, required=True)
    assemble.add_argument("--matched-pair-cohorts", type=Path, required=True)
    assemble.add_argument("--destination", type=Path, required=True)
    performance = commands.add_parser(
        "generate-performance", help="measure all frozen outcome-discarding performance classes"
    )
    performance.add_argument("--target-wall-seconds", type=float, required=True)
    performance.add_argument("--checkpoint-directory", type=Path)
    performance.add_argument("--destination", type=Path, required=True)
    resources = commands.add_parser(
        "generate-resources", help="measure all five frozen area/date resource cells"
    )
    resources.add_argument("--destination", type=Path, required=True)
    matched = commands.add_parser(
        "generate-matched-cohorts", help="derive matched signed evidence from exact Rust attempts"
    )
    matched.add_argument("--seed-count", type=int, default=MATCHED_COHORT_SEED_COUNT)
    matched.add_argument("--destination", type=Path, required=True)
    preoutcome = commands.add_parser(
        "generate-preoutcome", help="atomically generate all three pre-outcome receipt sets"
    )
    preoutcome.add_argument("--target-wall-seconds", type=float, required=True)
    preoutcome.add_argument("--destination", type=Path, required=True)
    outcomes = commands.add_parser(
        "run-outcomes", help="run or exactly resume one bounded frozen outcome shard"
    )
    outcomes.add_argument("--preoutcome-directory", type=Path, required=True)
    outcomes.add_argument("--run-root", type=Path, required=True)
    outcomes.add_argument("--shard-index", type=int, required=True)
    for identity_command in (
        reduce_cell, commit, resume, assemble, performance, resources, matched, preoutcome,
        outcomes,
    ):
        identity_command.add_argument("--source-root", type=Path, required=True)
        identity_command.add_argument("--batch-binary", type=Path, required=True)
        identity_command.add_argument("--benchmark-binary", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "_measure-child":
        _measurement_worker(args)
        return
    if args.command == "_batch-chunk-child":
        _batch_chunk_worker(args)
        return
    preregistration = load_preregistration(args.preregistration)
    identity_commands = {
        "reduce-cell", "commit", "resume", "assemble",
        "generate-performance", "generate-resources", "generate-matched-cohorts",
        "generate-preoutcome", "run-outcomes",
    }
    if args.command in identity_commands:
        code_sha256, binary_sha256 = producer_identities(
            args.source_root, args.batch_binary, args.benchmark_binary
        )
        if code_sha256 != preregistration["generator"]["binary"]["source_identity"]["sha256"]:
            raise SchemaError("checked-out producer source set differs from the frozen source identity")
        attempt_regenerator = rust_attempt_regenerator(
            preregistration, args.preregistration, args.batch_binary
        )
    if args.command in {"prepare", "commit", "resume"}:
        spec = next((item for item in iter_shard_specs(preregistration) if item.index == args.shard_index), None)
        if spec is None:
            raise SystemExit(f"shard index is outside 0..{preregistration['execution_protocol']['shard_count'] - 1}")
    if args.command == "capture-resource":
        command = args.benchmark_command
        if command and command[0] == "--":
            command = command[1:]
        result = capture_benchmark_stdout(command)
        write_jsonl_atomic((result,), args.destination, byte_limit=8192)
    elif args.command == "generate-performance":
        receipt = generate_performance_probe(
            preregistration, args.preregistration, args.source_root,
            args.batch_binary, args.benchmark_binary, args.target_wall_seconds,
            args.checkpoint_directory,
        )
        result = _write_bounded_json_atomic(
            receipt, args.destination, FROZEN_MAX_RESOURCE_RECEIPT_BYTES
        )
    elif args.command == "generate-resources":
        receipt = generate_resource_receipts(
            preregistration, args.source_root, args.batch_binary, args.benchmark_binary
        )
        result = _write_bounded_json_atomic(
            receipt, args.destination, FROZEN_MAX_RESOURCE_RECEIPT_BYTES
        )
    elif args.command == "generate-matched-cohorts":
        receipt = generate_matched_pair_cohorts(
            preregistration, args.preregistration, args.batch_binary,
            code_sha256, binary_sha256, args.seed_count,
        )
        result = _write_bounded_json_atomic(
            receipt, args.destination, FROZEN_MAX_RESOURCE_RECEIPT_BYTES
        )
    elif args.command == "generate-preoutcome":
        result = generate_preoutcome_receipts(
            preregistration, args.preregistration, args.source_root,
            args.batch_binary, args.benchmark_binary,
            args.target_wall_seconds, args.destination,
        )
    elif args.command == "run-outcomes":
        result = run_outcomes(
            preregistration,
            args.preregistration,
            args.source_root,
            args.batch_binary,
            args.benchmark_binary,
            args.preoutcome_directory,
            args.run_root,
            args.shard_index,
        )
    elif args.command == "prepare":
        destination = args.run_root / "requests" / f"shard-{spec.index:05d}.jsonl"
        result = prepare_input_shard(preregistration, spec, destination)
    elif args.command == "reduce-cell":
        cell_ids = expected_cell_ids(preregistration)
        if args.cell_ordinal < 0 or args.cell_ordinal >= len(cell_ids):
            raise SchemaError("cell ordinal is outside the frozen matrix")
        result = commit_cell_transport(
            preregistration, cell_ids[args.cell_ordinal], args.cell_ordinal, args.transport,
            args.destination, code_sha256, binary_sha256,
            artifact_root=args.run_root.resolve(strict=True),
        )
    elif args.command == "commit":
        result = commit_output_shard(
            preregistration, spec, args.run_root, args.summary_directory, args.manifest,
            code_sha256, binary_sha256, args.elapsed_seconds, args.peak_rss_bytes,
            attempt_regenerator,
        )
    elif args.command == "resume":
        result = {"reusable": committed_shard_matches(
            preregistration, spec, args.run_root, args.manifest, code_sha256, binary_sha256,
            attempt_regenerator,
        )}
    else:
        run_root = args.run_root.resolve(strict=True)
        if args.destination.parent.resolve() != run_root:
            raise SchemaError("run-manifest destination parent must equal the run root")
        manifest_paths = [args.shard_manifest_directory / f"manifest-{index:05d}.jsonl" for index in range(FROZEN_SHARD_COUNT)]
        performance_probe = _load_bounded_json(args.performance_probe, FROZEN_MAX_RESOURCE_RECEIPT_BYTES, "performance probe")
        resources = _load_bounded_json(args.resources, FROZEN_MAX_RESOURCE_RECEIPT_BYTES, "resource receipts")
        production_parity_fixture = _load_bounded_json(
            args.production_parity_fixture,
            FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
            "production parity fixture",
        )
        matched_pair_cohorts = _load_bounded_json(
            args.matched_pair_cohorts,
            FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
            "matched pair cohorts",
        )
        run_manifest = build_run_manifest(
            preregistration, run_root, manifest_paths, code_sha256, binary_sha256,
            performance_probe, resources, attempt_regenerator,
            production_parity_fixture, matched_pair_cohorts,
        )
        result = write_run_manifest_atomic(run_manifest, args.destination)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
