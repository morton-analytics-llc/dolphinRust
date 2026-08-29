#!/usr/bin/env python3
"""Run one frozen spatial shard with cell-parallel exact replay verification."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from validation import score_spatial_covariance as spatial_scorer
    from validation import spatial_covariance_simulation as spatial_simulation
except ModuleNotFoundError:
    import score_spatial_covariance as spatial_scorer
    import spatial_covariance_simulation as spatial_simulation


SchemaError = spatial_simulation.SchemaError
ShardSpec = spatial_simulation.ShardSpec
_ORIGINAL_SUMMARY_ROOT = spatial_simulation._summary_root
_SUMMARY_ROOT_DOMAIN = b"dolphinrust:spatial-covariance:cell-summary-root:v4\0"
_SOURCE_FILE_CAP_BYTES = 16 << 20


def _safe_bytes(path: Path, byte_limit: int, label: str) -> bytes:
    return spatial_scorer._read_bounded_bytes(Path(path), byte_limit, label)


def _sha256_file(path: Path, byte_limit: int, label: str) -> str:
    return hashlib.sha256(_safe_bytes(path, byte_limit, label)).hexdigest()


def _require_exact_validation_sources(source_root: Path) -> None:
    source_root = Path(source_root).resolve(strict=True)
    for imported_path, relative in (
        (
            Path(spatial_simulation.__file__),
            Path("validation/spatial_covariance_simulation.py"),
        ),
        (Path(spatial_scorer.__file__), Path("validation/score_spatial_covariance.py")),
        (Path(__file__), Path("validation/run_spatial_covariance_parallel.py")),
    ):
        imported_path = imported_path.resolve(strict=True)
        exact_path = (source_root / relative).resolve(strict=True)
        imported_digest = _sha256_file(
            imported_path, _SOURCE_FILE_CAP_BYTES, f"imported {relative}"
        )
        exact_digest = _sha256_file(
            exact_path, _SOURCE_FILE_CAP_BYTES, f"exact-checkout {relative}"
        )
        if imported_digest != exact_digest:
            raise SchemaError(
                f"parallel replay imported {relative} from outside the exact source checkout"
            )


def _single_summary_root(summary_sha256: str) -> str:
    digest = hashlib.sha256(_SUMMARY_ROOT_DOMAIN)
    digest.update((0).to_bytes(8, "big"))
    digest.update(bytes.fromhex(summary_sha256))
    return digest.hexdigest()


def _process_tree_sample(root_pids: set[int]) -> tuple[int, set[int]]:
    sampled = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [tuple(map(int, line.split())) for line in sampled.stdout.splitlines()]
    process_tree = set(root_pids)
    changed = True
    while changed:
        changed = False
        for pid, ppid, _rss in rows:
            if ppid in process_tree and pid not in process_tree:
                process_tree.add(pid)
                changed = True
    rss_bytes = sum(rss for pid, _ppid, rss in rows if pid in process_tree) * 1024
    return rss_bytes, process_tree


def _terminate_process_tree(process_tree: set[int]) -> None:
    for pid in sorted(process_tree, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_bounded_pool(
    function: Any, tasks: list[Any], worker_limit: int
) -> tuple[tuple[Any, ...], int]:
    if not tasks:
        return (), 0
    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=min(worker_limit, len(tasks)), mp_context=context
    )
    futures = [executor.submit(function, task) for task in tasks]
    peak_rss_bytes = 0
    try:
        while not all(future.done() for future in futures):
            root_pids = {
                process.pid for process in executor._processes.values() if process.pid
            }
            rss_bytes, process_tree = _process_tree_sample(root_pids)
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            if peak_rss_bytes > spatial_simulation.FROZEN_PROCESS_RSS_BYTES:
                _terminate_process_tree(process_tree)
                for future in futures:
                    future.cancel()
                raise SchemaError(
                    "parallel cell pool aggregate RSS exceeded the frozen process cap"
                )
            done, _pending = concurrent.futures.wait(
                futures,
                timeout=spatial_simulation.PARALLEL_BATCH_RSS_SAMPLE_SECONDS,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )
            failed = next(
                (future for future in done if future.exception() is not None), None
            )
            if failed is not None:
                _terminate_process_tree(process_tree)
                failed.result()
        results = tuple(future.result() for future in futures)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results, peak_rss_bytes


@dataclass(frozen=True)
class _ReplayTask:
    source_root: str
    preregistration_path: str
    batch_binary: str
    benchmark_binary: str
    summary_directory: str
    artifact_root: str | None
    shard_index: int
    cell_id: str
    cell_ordinal: int
    preregistration_sha256: str
    preregistration_file_sha256: str
    code_sha256: str
    binary_sha256: str


@dataclass(frozen=True)
class _ReductionTask:
    source_root: str
    preregistration_path: str
    batch_binary: str
    benchmark_binary: str
    run_root: str
    transport: str
    destination: str
    cell_id: str
    cell_ordinal: int
    preregistration_sha256: str
    preregistration_file_sha256: str
    code_sha256: str
    binary_sha256: str


@dataclass(frozen=True)
class _GenerationTask:
    source_root: str
    preregistration_path: str
    batch_binary: str
    benchmark_binary: str
    run_root: str
    destination: str
    cell_id: str
    cell_ordinal: int
    preregistration_sha256: str
    preregistration_file_sha256: str
    code_sha256: str
    binary_sha256: str


def _load_bound_inputs(task: Any, label: str) -> Mapping[str, Any]:
    source_root = Path(task.source_root)
    _require_exact_validation_sources(source_root)
    preregistration_path = Path(task.preregistration_path)
    preregistration = spatial_simulation.load_preregistration(preregistration_path)
    preregistration_file_sha256 = _sha256_file(
        preregistration_path,
        spatial_scorer.FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
        f"{label} preregistration",
    )
    actual_code_sha256, actual_binary_sha256 = (
        spatial_simulation._validated_producer_identities(
            preregistration,
            source_root,
            Path(task.batch_binary),
            Path(task.benchmark_binary),
        )
    )
    if (
        spatial_simulation.preregistration_digest(preregistration)
        != task.preregistration_sha256
        or preregistration_file_sha256 != task.preregistration_file_sha256
        or (actual_code_sha256, actual_binary_sha256)
        != (task.code_sha256, task.binary_sha256)
    ):
        raise SchemaError(f"{label} input identity changed")
    return preregistration


def _verify_cell(task: _ReplayTask) -> dict[str, Any]:
    preregistration_path = Path(task.preregistration_path)
    preregistration = _load_bound_inputs(task, "parallel replay")
    spec = ShardSpec(
        task.shard_index,
        task.cell_ordinal,
        task.cell_ordinal + 1,
        (task.cell_id,),
        (spatial_simulation.expected_seed_count(task.cell_id),),
    )
    started = time.perf_counter()
    summary_root, summary_bytes = _ORIGINAL_SUMMARY_ROOT(
        preregistration,
        Path(task.summary_directory),
        spec,
        task.code_sha256,
        task.binary_sha256,
        spatial_simulation.rust_attempt_regenerator(
            preregistration, preregistration_path, Path(task.batch_binary)
        ),
        artifact_root=None if task.artifact_root is None else Path(task.artifact_root),
    )
    raw = _safe_bytes(
        Path(task.summary_directory) / f"cell-{task.cell_ordinal:05d}.jsonl",
        preregistration["execution_protocol"]["max_encoded_cell_summary_bytes"],
        f"parallel replay cell {task.cell_id} summary",
    )
    summary_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        summary_bytes != len(raw)
        or summary_root != _single_summary_root(summary_sha256)
    ):
        raise SchemaError("parallel replay result is not bound to the cached summary bytes")
    _load_bound_inputs(task, "parallel replay")
    return {
        "cell_id": task.cell_id,
        "cell_ordinal": task.cell_ordinal,
        "summary_sha256": summary_sha256,
        "summary_bytes": len(raw),
        "exact_replay_root_sha256": summary_root,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _reduce_cell(task: _ReductionTask) -> dict[str, Any]:
    preregistration = _load_bound_inputs(task, "parallel reduction")
    started = time.perf_counter()
    seed_count = spatial_simulation.expected_seed_count(task.cell_id)
    accumulator = spatial_scorer.CellAccumulator(
        preregistration,
        task.cell_id,
        task.cell_ordinal,
        seed_count,
        task.code_sha256,
        task.binary_sha256,
        artifact_root=Path(task.run_root),
    )
    transport = Path(task.transport)
    with transport.open("rb") as handle:
        for _line_number in range(seed_count):
            raw = handle.readline(spatial_simulation.FROZEN_MAX_RECORD_BYTES + 2)
            if (
                not raw
                or len(raw) > spatial_simulation.FROZEN_MAX_RECORD_BYTES
                or not raw.endswith(b"\n")
            ):
                raise SchemaError("ephemeral attempt transport is incomplete or oversized")
            try:
                accumulator.add(json.loads(raw))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SchemaError("ephemeral attempt transport is malformed") from exc
        if handle.read(1):
            raise SchemaError("ephemeral attempt transport contains top-up evidence")
    summary = accumulator.finalize()
    spatial_simulation.validate_cell_summary(
        preregistration,
        summary,
        task.cell_id,
        task.cell_ordinal,
        task.code_sha256,
        task.binary_sha256,
    )
    _load_bound_inputs(task, "parallel reduction")
    receipt = spatial_simulation.write_jsonl_atomic(
        (summary,),
        Path(task.destination),
        byte_limit=spatial_simulation.FROZEN_MAX_RECORD_BYTES,
    )
    spatial_simulation._fsync_directory(Path(task.destination).parent)
    transport.unlink()
    spatial_simulation._fsync_directory(transport.parent)
    return {
        "cell_id": task.cell_id,
        "cell_ordinal": task.cell_ordinal,
        "summary_sha256": receipt["sha256"],
        "summary_bytes": receipt["bytes"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def _generate_and_reduce_cell(task: _GenerationTask) -> dict[str, Any]:
    preregistration = _load_bound_inputs(task, "parallel generation")
    preregistration_path = Path(task.preregistration_path)
    started = time.perf_counter()
    seed_count = spatial_simulation.expected_seed_count(task.cell_id)
    accumulator = spatial_scorer.CellAccumulator(
        preregistration,
        task.cell_id,
        task.cell_ordinal,
        seed_count,
        task.code_sha256,
        task.binary_sha256,
        artifact_root=Path(task.run_root),
    )
    for attempt in spatial_simulation.rust_attempt_regenerator(
        preregistration, preregistration_path, Path(task.batch_binary)
    )(task.cell_id, task.cell_ordinal):
        accumulator.add(attempt)
    summary = accumulator.finalize()
    spatial_simulation.validate_cell_summary(
        preregistration,
        summary,
        task.cell_id,
        task.cell_ordinal,
        task.code_sha256,
        task.binary_sha256,
    )
    _load_bound_inputs(task, "parallel generation")
    receipt = spatial_simulation.write_jsonl_atomic(
        (summary,),
        Path(task.destination),
        byte_limit=spatial_simulation.FROZEN_MAX_RECORD_BYTES,
    )
    spatial_simulation._fsync_directory(Path(task.destination).parent)
    return {
        "cell_id": task.cell_id,
        "cell_ordinal": task.cell_ordinal,
        "summary_sha256": receipt["sha256"],
        "summary_bytes": receipt["bytes"],
        "attempt_transport_retained": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


class ParallelSummaryVerifier:
    """Verify exact cell replays concurrently and reduce their digests in order."""

    def __init__(
        self,
        preregistration: Mapping[str, Any],
        source_root: Path,
        preregistration_path: Path,
        batch_binary: Path,
        benchmark_binary: Path,
        worker_limit: int,
    ) -> None:
        maximum_workers = spatial_simulation.PARALLEL_BATCH_WORKER_COUNT
        if not isinstance(worker_limit, int) or not 0 < worker_limit <= maximum_workers:
            raise SchemaError(
                f"parallel cell worker limit must be within 1..{maximum_workers}"
            )
        self.preregistration = preregistration
        self.source_root = Path(source_root).resolve(strict=True)
        self.preregistration_path = Path(preregistration_path).resolve(strict=True)
        self.batch_binary = Path(batch_binary).resolve(strict=True)
        self.benchmark_binary = Path(benchmark_binary).resolve(strict=True)
        self.worker_limit = worker_limit
        self.full_specs = tuple(spatial_simulation.iter_shard_specs(preregistration))
        self.max_summary_bytes = preregistration["execution_protocol"][
            "max_encoded_cell_summary_bytes"
        ]
        self.preregistration_sha256 = spatial_simulation.preregistration_digest(
            preregistration
        )
        self.preregistration_file_sha256 = _sha256_file(
            self.preregistration_path,
            spatial_scorer.FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
            "parallel replay preregistration",
        )
        self.code_sha256, self.binary_sha256 = (
            spatial_simulation._validated_producer_identities(
                preregistration,
                self.source_root,
                self.batch_binary,
                self.benchmark_binary,
            )
        )
        self._verified: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.measurements: list[dict[str, Any]] = []

    def _raw_fingerprint(self, directory: Path, cell_ordinal: int) -> tuple[str, int]:
        raw = _safe_bytes(
            directory / f"cell-{cell_ordinal:05d}.jsonl",
            self.max_summary_bytes,
            f"parallel replay cell {cell_ordinal} summary",
        )
        return hashlib.sha256(raw).hexdigest(), len(raw)

    def _full_spec_if_ready(self, directory: Path, spec: ShardSpec) -> ShardSpec | None:
        full_spec = next((item for item in self.full_specs if item.index == spec.index), None)
        if full_spec is None:
            return None
        slice_start = spec.cell_ordinal_start - full_spec.cell_ordinal_start
        slice_end = spec.cell_ordinal_end_exclusive - full_spec.cell_ordinal_start
        if (
            slice_start < 0
            or slice_end > len(full_spec.cell_ids)
            or tuple(full_spec.cell_ids[slice_start:slice_end]) != tuple(spec.cell_ids)
            or tuple(full_spec.seed_counts[slice_start:slice_end]) != tuple(spec.seed_counts)
        ):
            return None
        available = range(
            full_spec.cell_ordinal_start, full_spec.cell_ordinal_end_exclusive
        )
        for ordinal in available:
            path = directory / f"cell-{ordinal:05d}.jsonl"
            if not path.is_file() or path.is_symlink():
                return None
        return full_spec

    def _assert_current_identities(
        self, preregistration: Mapping[str, Any], code_sha256: str, binary_sha256: str
    ) -> None:
        current_preregistration = spatial_simulation.load_preregistration(
            self.preregistration_path
        )
        current_preregistration_file_sha256 = _sha256_file(
            self.preregistration_path,
            spatial_scorer.FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
            "parallel replay preregistration",
        )
        current_code_sha256, current_binary_sha256 = (
            spatial_simulation._validated_producer_identities(
                current_preregistration,
                self.source_root,
                self.batch_binary,
                self.benchmark_binary,
            )
        )
        if (
            spatial_simulation.preregistration_digest(preregistration)
            != self.preregistration_sha256
            or spatial_simulation.preregistration_digest(current_preregistration)
            != self.preregistration_sha256
            or current_preregistration_file_sha256
            != self.preregistration_file_sha256
            or (code_sha256, binary_sha256)
            != (self.code_sha256, self.binary_sha256)
            or (current_code_sha256, current_binary_sha256)
            != (self.code_sha256, self.binary_sha256)
        ):
            raise SchemaError("parallel replay source, binary, or preregistration changed")

    def _cache_key(
        self,
        directory: Path,
        artifact_root: Path | None,
        spec: ShardSpec,
        cell_id: str,
        cell_ordinal: int,
    ) -> tuple[Any, ...]:
        return (
            str(directory),
            None if artifact_root is None else str(Path(artifact_root).resolve(strict=True)),
            spec.index,
            cell_ordinal,
            cell_id,
            self.preregistration_sha256,
            self.preregistration_file_sha256,
            self.code_sha256,
            self.binary_sha256,
        )

    def _verify(
        self,
        directory: Path,
        spec: ShardSpec,
        code_sha256: str,
        binary_sha256: str,
        artifact_root: Path | None,
    ) -> None:
        tasks = []
        for offset, cell_id in enumerate(spec.cell_ids):
            cell_ordinal = spec.cell_ordinal_start + offset
            summary_sha256, summary_bytes = self._raw_fingerprint(directory, cell_ordinal)
            cache_key = self._cache_key(
                directory, artifact_root, spec, cell_id, cell_ordinal
            )
            cached = self._verified.get(cache_key)
            if cached is not None and (
                cached["summary_sha256"], cached["summary_bytes"]
            ) == (summary_sha256, summary_bytes):
                continue
            tasks.append(_ReplayTask(
                source_root=str(self.source_root),
                preregistration_path=str(self.preregistration_path),
                batch_binary=str(self.batch_binary),
                benchmark_binary=str(self.benchmark_binary),
                summary_directory=str(directory),
                artifact_root=None if artifact_root is None else str(artifact_root),
                shard_index=spec.index,
                cell_id=cell_id,
                cell_ordinal=cell_ordinal,
                preregistration_sha256=self.preregistration_sha256,
                preregistration_file_sha256=self.preregistration_file_sha256,
                code_sha256=code_sha256,
                binary_sha256=binary_sha256,
            ))
        if not tasks:
            return
        started = time.perf_counter()
        worker_count = min(self.worker_limit, len(tasks))
        results, peak_rss_bytes = _run_bounded_pool(
            _verify_cell, tasks, worker_count
        )
        for task, result in zip(tasks, results):
            current = self._raw_fingerprint(directory, task.cell_ordinal)
            if current != (result["summary_sha256"], result["summary_bytes"]):
                raise SchemaError("parallel replay summary changed during exact verification")
            if result["exact_replay_root_sha256"] != _single_summary_root(
                result["summary_sha256"]
            ):
                raise SchemaError("parallel replay receipt root differs from summary bytes")
            cache_key = self._cache_key(
                directory, artifact_root, spec, task.cell_id, task.cell_ordinal
            )
            self._verified[cache_key] = result
        self.measurements.append({
            "stage_index": len(self.measurements),
            "cell_count": len(tasks),
            "worker_count": worker_count,
            "peak_aggregate_rss_bytes": peak_rss_bytes,
            "aggregate_rss_limit_bytes": spatial_simulation.FROZEN_PROCESS_RSS_BYTES,
            "elapsed_seconds": time.perf_counter() - started,
            "cells": list(results),
        })

    def __call__(
        self,
        preregistration: Mapping[str, Any],
        directory: Path,
        spec: ShardSpec,
        code_sha256: str,
        binary_sha256: str,
        attempt_regenerator: Any = None,
        artifact_root: Path | None = None,
    ) -> tuple[str, int]:
        self._assert_current_identities(preregistration, code_sha256, binary_sha256)
        if attempt_regenerator is None:
            raise SchemaError("parallel replay requires the exact Rust regenerator")
        directory = Path(directory).resolve(strict=True)
        verification_spec = self._full_spec_if_ready(directory, spec) or spec
        self._verify(
            directory, verification_spec, code_sha256, binary_sha256, artifact_root
        )
        digest = hashlib.sha256(_SUMMARY_ROOT_DOMAIN)
        total_bytes = 0
        for offset, cell_id in enumerate(spec.cell_ids):
            cell_ordinal = spec.cell_ordinal_start + offset
            summary_sha256, summary_bytes = self._raw_fingerprint(directory, cell_ordinal)
            cache_key = self._cache_key(
                directory, artifact_root, spec, cell_id, cell_ordinal
            )
            cached = self._verified.get(cache_key)
            if cached is None or (
                cached["summary_sha256"], cached["summary_bytes"]
            ) != (summary_sha256, summary_bytes):
                raise SchemaError("parallel replay summary lacks exact verification")
            digest.update(offset.to_bytes(8, "big"))
            digest.update(bytes.fromhex(summary_sha256))
            total_bytes += summary_bytes
        return digest.hexdigest(), total_bytes

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "dolphinrust.spatial-covariance.parallel-exact-replay/1",
            "worker_limit": self.worker_limit,
            "aggregate_rss_limit_bytes": spatial_simulation.FROZEN_PROCESS_RSS_BYTES,
            "preregistration_sha256": self.preregistration_sha256,
            "preregistration_file_sha256": self.preregistration_file_sha256,
            "code_sha256": self.code_sha256,
            "binary_sha256": self.binary_sha256,
            "verified_cell_count": len(self._verified),
            "measurements": self.measurements,
        }


def _unlink_owned_regular(path: Path) -> None:
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise SchemaError("owned parallel replay residual is not a regular file")
    path.unlink()
    spatial_simulation._fsync_directory(path.parent)


def run_parallel_outcomes(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    source_root: Path,
    batch_binary: Path,
    benchmark_binary: Path,
    preoutcome_directory: Path,
    run_root: Path,
    shard_index: int,
    verifier: ParallelSummaryVerifier,
) -> dict[str, Any]:
    spatial_simulation.validate_preregistration(preregistration)
    code_sha256, binary_sha256 = spatial_simulation._validated_producer_identities(
        preregistration, source_root, batch_binary, benchmark_binary
    )
    spatial_simulation.validate_preoutcome_receipts(
        preregistration, preoutcome_directory, code_sha256, binary_sha256
    )
    spec = next(
        (
            item
            for item in spatial_simulation.iter_shard_specs(preregistration)
            if item.index == shard_index
        ),
        None,
    )
    if spec is None:
        raise SchemaError("outcome shard index is outside the frozen plan")
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
    attempt_regenerator = spatial_simulation.rust_attempt_regenerator(
        preregistration, preregistration_path, batch_binary
    )
    if manifest_path.exists():
        if not spatial_simulation.committed_shard_matches(
            preregistration,
            spec,
            run_root,
            manifest_path,
            code_sha256,
            binary_sha256,
            attempt_regenerator,
        ):
            raise SchemaError("committed outcome shard failed exact resume validation")
        return {
            "schema": "dolphinrust.spatial-covariance.outcome-run/1",
            "shard_index": spec.index,
            "reusable": True,
            "generated_cells": 0,
            "resumed_cells": len(spec.cell_ids),
            "manifest": str(manifest_path),
            "residual_transport_reduction": None,
            "parallel_generation_reduction": {
                "cell_count": 0,
                "worker_count": 0,
                "peak_aggregate_rss_bytes": 0,
                "cells": [],
            },
        }

    started = time.perf_counter()
    peak_rss_bytes = 0
    generated_cells = 0
    resumed_cells = 0
    generation_tasks: list[_GenerationTask] = []
    residual_reduction_task: _ReductionTask | None = None
    residuals: list[Path] = []
    for offset, cell_id in enumerate(spec.cell_ids):
        cell_ordinal = spec.cell_ordinal_start + offset
        destination = summary_directory / f"cell-{cell_ordinal:05d}.jsonl"
        transport = transport_directory / f"cell-{cell_ordinal:05d}.jsonl"
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not stat.S_ISREG(destination.lstat().st_mode):
                raise SchemaError("existing parallel cell summary is not a regular file")
            residuals.extend((transport, transport.with_name(transport.name + ".partial")))
            resumed_cells += 1
            continue
        _unlink_owned_regular(destination.with_name(destination.name + ".partial"))
        common = {
            "source_root": str(Path(source_root).resolve(strict=True)),
            "preregistration_path": str(Path(preregistration_path).resolve(strict=True)),
            "batch_binary": str(Path(batch_binary).resolve(strict=True)),
            "benchmark_binary": str(Path(benchmark_binary).resolve(strict=True)),
            "run_root": str(run_root),
            "destination": str(destination),
            "cell_id": cell_id,
            "cell_ordinal": cell_ordinal,
            "preregistration_sha256": verifier.preregistration_sha256,
            "preregistration_file_sha256": verifier.preregistration_file_sha256,
            "code_sha256": code_sha256,
            "binary_sha256": binary_sha256,
        }
        if transport.exists() or transport.is_symlink():
            if transport.is_symlink() or not stat.S_ISREG(transport.lstat().st_mode):
                raise SchemaError("existing parallel cell transport is not a regular file")
            if residual_reduction_task is not None:
                raise SchemaError(
                    "more than one ephemeral cell transport violates the frozen resume boundary"
                )
            residual_reduction_task = _ReductionTask(
                transport=str(transport), **common
            )
        else:
            _unlink_owned_regular(transport.with_name(transport.name + ".partial"))
            generation_tasks.append(_GenerationTask(**common))

    residual_reduction = None
    if residual_reduction_task is not None:
        residual_reduction = _reduce_cell(residual_reduction_task)
        generated_cells += 1

    generation_started = time.perf_counter()
    generation_worker_count = min(verifier.worker_limit, len(generation_tasks))
    generation_results, generation_peak_rss_bytes = _run_bounded_pool(
        _generate_and_reduce_cell, generation_tasks, verifier.worker_limit
    )
    generation_elapsed_seconds = time.perf_counter() - generation_started
    peak_rss_bytes = max(
        peak_rss_bytes,
        generation_peak_rss_bytes,
        spatial_simulation._child_max_rss_bytes(),
    )
    generated_cells += len(generation_results)

    verifier(
        preregistration,
        summary_directory,
        spec,
        code_sha256,
        binary_sha256,
        attempt_regenerator,
        artifact_root=run_root,
    )
    if verifier.measurements:
        peak_rss_bytes = max(
            peak_rss_bytes,
            verifier.measurements[-1]["peak_aggregate_rss_bytes"],
            spatial_simulation._child_max_rss_bytes(),
        )
    for residual in residuals:
        _unlink_owned_regular(residual)
    receipt = spatial_simulation.commit_output_shard(
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
        "residual_transport_reduction": residual_reduction,
        "parallel_generation_reduction": {
            "cell_count": len(generation_results),
            "worker_count": generation_worker_count,
            "elapsed_seconds": generation_elapsed_seconds,
            "peak_aggregate_rss_bytes": generation_peak_rss_bytes,
            "aggregate_rss_limit_bytes": spatial_simulation.FROZEN_PROCESS_RSS_BYTES,
            "attempt_transport_retained": False,
            "cells": list(generation_results),
        },
    }


def assemble_parallel_run(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    source_root: Path,
    batch_binary: Path,
    benchmark_binary: Path,
    preoutcome_directory: Path,
    run_root: Path,
    shard_manifest_directory: Path,
    performance_probe: Mapping[str, Any],
    resources: list[Mapping[str, Any]],
    production_parity_fixture: Mapping[str, Any],
    destination: Path,
    verifier: ParallelSummaryVerifier,
) -> dict[str, Any]:
    _require_exact_validation_sources(source_root)
    code_sha256, binary_sha256 = spatial_simulation._validated_producer_identities(
        preregistration, source_root, batch_binary, benchmark_binary
    )
    attempt_regenerator = spatial_simulation.rust_attempt_regenerator(
        preregistration, preregistration_path, batch_binary
    )
    manifest_paths = tuple(
        Path(shard_manifest_directory) / f"manifest-{index:05d}.jsonl"
        for index in range(spatial_simulation.FROZEN_SHARD_COUNT)
    )
    original_summary_root = spatial_simulation._summary_root
    spatial_simulation._summary_root = verifier
    try:
        manifest = spatial_simulation.build_run_manifest(
            preregistration,
            run_root,
            manifest_paths,
            code_sha256,
            binary_sha256,
            performance_probe,
            resources,
            preoutcome_directory,
            attempt_regenerator=attempt_regenerator,
            production_parity_fixture=production_parity_fixture,
        )
    finally:
        spatial_simulation._summary_root = original_summary_root
    receipt = spatial_simulation.write_run_manifest_atomic(manifest, destination)
    return {
        "schema": "dolphinrust.spatial-covariance.parallel-run-assembly/1",
        "manifest": str(destination),
        "manifest_sha256": receipt["sha256"],
        "manifest_bytes": receipt["bytes"],
        "parallel_summary_verification": verifier.receipt(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--batch-binary", type=Path, required=True)
    parser.add_argument("--benchmark-binary", type=Path, required=True)
    parser.add_argument("--preoutcome-directory", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument(
        "--parallel-cell-workers",
        type=int,
        default=spatial_simulation.PARALLEL_BATCH_WORKER_COUNT,
    )
    args = parser.parse_args()

    _require_exact_validation_sources(args.source_root)
    preregistration = spatial_simulation.load_preregistration(args.preregistration)
    verifier = ParallelSummaryVerifier(
        preregistration,
        args.source_root,
        args.preregistration,
        args.batch_binary,
        args.benchmark_binary,
        args.parallel_cell_workers,
    )
    original_summary_root = spatial_simulation._summary_root
    spatial_simulation._summary_root = verifier
    try:
        result = run_parallel_outcomes(
            preregistration,
            args.preregistration,
            args.source_root,
            args.batch_binary,
            args.benchmark_binary,
            args.preoutcome_directory,
            args.run_root,
            args.shard_index,
            verifier,
        )
    finally:
        spatial_simulation._summary_root = original_summary_root
    result["parallel_summary_verification"] = verifier.receipt()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
