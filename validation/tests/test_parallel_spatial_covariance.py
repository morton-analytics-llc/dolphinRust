from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from validation.run_spatial_covariance_parallel import (
    ParallelSummaryVerifier,
    SchemaError,
    _GenerationTask,
    _ReplayTask,
    _SUMMARY_ROOT_DOMAIN,
    _generate_and_reduce_cell,
    _reduce_cell,
    _run_bounded_pool,
    _sha256_file,
    _single_summary_root,
    _verify_cell,
    assemble_parallel_run,
    run_parallel_outcomes,
)
from validation.spatial_covariance_simulation import (
    FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
    FROZEN_PROCESS_RSS_BYTES,
    ShardSpec,
)
from validation import spatial_covariance_simulation as spatial_simulation


_POOL_CALLS: list[tuple[int, tuple[object, ...]]] = []


def _synchronous_bounded_pool(function: object, tasks: list[object], worker_limit: int):
    materialized = tuple(tasks)
    _POOL_CALLS.append((worker_limit, materialized))
    results = []
    for task in materialized:
        raw = (
            Path(task.summary_directory) / f"cell-{task.cell_ordinal:05d}.jsonl"
        ).read_bytes()
        summary_sha256 = hashlib.sha256(raw).hexdigest()
        results.append({
            "cell_id": task.cell_id,
            "cell_ordinal": task.cell_ordinal,
            "summary_sha256": summary_sha256,
            "summary_bytes": len(raw),
            "exact_replay_root_sha256": _single_summary_root(summary_sha256),
            "elapsed_seconds": 0.1,
        })
    return tuple(results), 4096


class _PendingFuture:
    def done(self) -> bool:
        return False

    def cancel(self) -> bool:
        return True


class _PendingExecutor:
    last: "_PendingExecutor | None" = None

    def __init__(self, max_workers: int, mp_context: object) -> None:
        self._processes = {100: SimpleNamespace(pid=100)}
        self.shutdown_called = False
        self.__class__.last = self

    def submit(self, function: object, task: object) -> _PendingFuture:
        return _PendingFuture()

    def shutdown(self, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_called = True


class ParallelSpatialCovarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        _POOL_CALLS.clear()

    def _identity_patches(
        self, full_spec: ShardSpec, preregistration: dict[str, object]
    ):
        return (
            patch(
                "validation.run_spatial_covariance_parallel.spatial_simulation.iter_shard_specs",
                return_value=(full_spec,),
            ),
            patch(
                "validation.run_spatial_covariance_parallel.spatial_simulation.preregistration_digest",
                return_value="c" * 64,
            ),
            patch(
                "validation.run_spatial_covariance_parallel.spatial_simulation._validated_producer_identities",
                return_value=("a" * 64, "b" * 64),
            ),
            patch(
                "validation.run_spatial_covariance_parallel.spatial_simulation.load_preregistration",
                return_value=preregistration,
            ),
            patch(
                "validation.run_spatial_covariance_parallel._run_bounded_pool",
                side_effect=_synchronous_bounded_pool,
            ),
        )

    def _fixture(
        self, root: Path
    ) -> tuple[dict[str, object], Path, Path, Path, Path, Path]:
        preregistration = {
            "execution_protocol": {"max_encoded_cell_summary_bytes": 1024}
        }
        source = root / "source"
        source.mkdir()
        preregistration_path = root / "preregistration.json"
        preregistration_path.write_text("{}")
        batch = root / "batch"
        batch.write_bytes(b"batch")
        benchmark = root / "benchmark"
        benchmark.write_bytes(b"benchmark")
        summaries = root / "summaries"
        summaries.mkdir()
        return preregistration, source, preregistration_path, batch, benchmark, summaries

    def test_full_ready_shard_is_verified_once_and_reduced_in_order(self) -> None:
        full_spec = ShardSpec(0, 0, 2, ("cell-a", "cell-b"), (1, 1))
        single_spec = ShardSpec(0, 0, 1, ("cell-a",), (1,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                preregistration,
                source,
                preregistration_path,
                batch,
                benchmark,
                summaries,
            ) = self._fixture(root)
            raw_a = b'{"cell":"a"}\n'
            raw_b = b'{"cell":"b"}\n'
            (summaries / "cell-00000.jsonl").write_bytes(raw_a)
            (summaries / "cell-00001.jsonl").write_bytes(raw_b)
            patches = self._identity_patches(full_spec, preregistration)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                verifier = ParallelSummaryVerifier(
                    preregistration,
                    source,
                    preregistration_path,
                    batch,
                    benchmark,
                    2,
                )
                single_root, single_bytes = verifier(
                    preregistration,
                    summaries,
                    single_spec,
                    "a" * 64,
                    "b" * 64,
                    attempt_regenerator=object(),
                    artifact_root=root,
                )
                full_root, full_bytes = verifier(
                    preregistration,
                    summaries,
                    full_spec,
                    "a" * 64,
                    "b" * 64,
                    attempt_regenerator=object(),
                    artifact_root=root,
                )
            self.assertEqual(len(_POOL_CALLS), 1)
            self.assertEqual(_POOL_CALLS[0][0], 2)
            self.assertEqual(len(_POOL_CALLS[0][1]), 2)
            self.assertEqual(single_root, _single_summary_root(hashlib.sha256(raw_a).hexdigest()))
            self.assertEqual(single_bytes, len(raw_a))
            expected_full = hashlib.sha256(_SUMMARY_ROOT_DOMAIN)
            for offset, raw in enumerate((raw_a, raw_b)):
                expected_full.update(offset.to_bytes(8, "big"))
                expected_full.update(hashlib.sha256(raw).digest())
            self.assertEqual(full_root, expected_full.hexdigest())
            self.assertEqual(full_bytes, len(raw_a) + len(raw_b))
            receipt = verifier.receipt()
            self.assertEqual(receipt["verified_cell_count"], 2)
            self.assertEqual(receipt["measurements"][0]["peak_aggregate_rss_bytes"], 4096)

    def test_changed_summary_is_reverified(self) -> None:
        spec = ShardSpec(0, 0, 1, ("cell-a",), (1,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                preregistration,
                source,
                preregistration_path,
                batch,
                benchmark,
                summaries,
            ) = self._fixture(root)
            path = summaries / "cell-00000.jsonl"
            path.write_bytes(b'{"version":1}\n')
            patches = self._identity_patches(spec, preregistration)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                verifier = ParallelSummaryVerifier(
                    preregistration,
                    source,
                    preregistration_path,
                    batch,
                    benchmark,
                    1,
                )
                verifier(
                    preregistration, summaries, spec, "a" * 64, "b" * 64,
                    attempt_regenerator=object(), artifact_root=root,
                )
                path.write_bytes(b'{"version":2}\n')
                verifier(
                    preregistration, summaries, spec, "a" * 64, "b" * 64,
                    attempt_regenerator=object(), artifact_root=root,
                )
            self.assertEqual(len(_POOL_CALLS), 2)
            self.assertEqual(len(verifier.measurements), 2)

    def test_full_shard_prefetch_rejects_mismatched_cell_slice(self) -> None:
        full_spec = ShardSpec(0, 0, 2, ("cell-a", "cell-b"), (1, 1))
        mismatched = ShardSpec(0, 0, 1, ("cell-x",), (1,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                preregistration,
                source,
                preregistration_path,
                batch,
                benchmark,
                summaries,
            ) = self._fixture(root)
            (summaries / "cell-00000.jsonl").write_bytes(b"{}\n")
            (summaries / "cell-00001.jsonl").write_bytes(b"{}\n")
            patches = self._identity_patches(full_spec, preregistration)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                verifier = ParallelSummaryVerifier(
                    preregistration,
                    source,
                    preregistration_path,
                    batch,
                    benchmark,
                    2,
                )
                self.assertIsNone(verifier._full_spec_if_ready(summaries, mismatched))

    def test_pool_terminates_process_tree_above_aggregate_rss_cap(self) -> None:
        with (
            patch(
                "validation.run_spatial_covariance_parallel.concurrent.futures.ProcessPoolExecutor",
                _PendingExecutor,
            ),
            patch(
                "validation.run_spatial_covariance_parallel._process_tree_sample",
                return_value=(FROZEN_PROCESS_RSS_BYTES + 1, {100, 101}),
            ),
            patch(
                "validation.run_spatial_covariance_parallel._terminate_process_tree"
            ) as terminate,
        ):
            with self.assertRaisesRegex(SchemaError, "aggregate RSS"):
                _run_bounded_pool(object(), [object()], 1)
        terminate.assert_called_once_with({100, 101})
        self.assertTrue(_PendingExecutor.last.shutdown_called)

    def test_outcome_coordinator_resumes_generates_reduces_and_then_verifies(self) -> None:
        spec = ShardSpec(
            0, 0, 3, ("cell-a", "cell-b", "cell-c"), (1, 1, 1)
        )
        preregistration: dict[str, object] = {}

        class FakeVerifier:
            worker_limit = 2
            preregistration_sha256 = "c" * 64
            preregistration_file_sha256 = "d" * 64
            measurements = [{"peak_aggregate_rss_bytes": 8192}]

            def __init__(self) -> None:
                self.calls = 0

            def __call__(
                self,
                _preregistration: object,
                directory: Path,
                _spec: ShardSpec,
                _code: str,
                _binary: str,
                _regenerator: object,
                artifact_root: Path,
            ) -> tuple[str, int]:
                self.calls += 1
                self.assert_all_summaries(directory)
                return "d" * 64, 9

            @staticmethod
            def assert_all_summaries(directory: Path) -> None:
                for ordinal in range(3):
                    if not (directory / f"cell-{ordinal:05d}.jsonl").is_file():
                        raise AssertionError("verification ran before every reduction completed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            preregistration_path = root / "preregistration.json"
            preregistration_path.write_text("{}")
            batch = root / "batch"
            batch.write_bytes(b"batch")
            benchmark = root / "benchmark"
            benchmark.write_bytes(b"benchmark")
            preoutcome = root / "preoutcome"
            preoutcome.mkdir()
            run_root = root / "run"
            summary_directory = run_root / "cells/shard-00000"
            summary_directory.mkdir(parents=True)
            transport_directory = run_root / "transports"
            transport_directory.mkdir(parents=True)
            (summary_directory / "cell-00000.jsonl").write_bytes(b"a\n")
            (transport_directory / "cell-00000.jsonl").write_bytes(b"residual\n")
            (transport_directory / "cell-00001.jsonl").write_bytes(b"b-transport\n")
            generated_cells: list[str] = []

            def fake_residual_reduce(task: object) -> dict[str, object]:
                Path(task.destination).write_bytes(b"cell-b\n")
                Path(task.transport).unlink()
                return {
                    "cell_id": task.cell_id,
                    "cell_ordinal": task.cell_ordinal,
                    "summary_sha256": "d" * 64,
                    "summary_bytes": 7,
                    "elapsed_seconds": 0.1,
                }

            def fake_pool(function: object, tasks: list[object], worker_limit: int):
                self.assertIs(function, _generate_and_reduce_cell)
                self.assertEqual(worker_limit, 2)
                results = []
                for task in tasks:
                    generated_cells.append(task.cell_id)
                    Path(task.destination).write_bytes(
                        f"{task.cell_id}\n".encode()
                    )
                    results.append({
                        "cell_id": task.cell_id,
                        "cell_ordinal": task.cell_ordinal,
                        "summary_sha256": "e" * 64,
                        "summary_bytes": len(task.cell_id) + 1,
                        "attempt_transport_retained": False,
                        "elapsed_seconds": 0.1,
                    })
                return tuple(results), 4096

            def fake_commit(*args: object, **kwargs: object) -> dict[str, object]:
                manifest_path = Path(args[4])
                manifest_path.write_bytes(b"manifest\n")
                return {"sha256": "f" * 64, "bytes": 9}

            verifier = FakeVerifier()
            with (
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.validate_preregistration"
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation._validated_producer_identities",
                    return_value=("a" * 64, "b" * 64),
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.validate_preoutcome_receipts"
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.iter_shard_specs",
                    return_value=(spec,),
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.rust_attempt_regenerator",
                    return_value=object(),
                ),
                patch(
                    "validation.run_spatial_covariance_parallel._reduce_cell",
                    side_effect=fake_residual_reduce,
                ),
                patch(
                    "validation.run_spatial_covariance_parallel._run_bounded_pool",
                    side_effect=fake_pool,
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation._child_max_rss_bytes",
                    return_value=2048,
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.commit_output_shard",
                    side_effect=fake_commit,
                ),
            ):
                result = run_parallel_outcomes(
                    preregistration,
                    preregistration_path,
                    source,
                    batch,
                    benchmark,
                    preoutcome,
                    run_root,
                    0,
                    verifier,
                )
            self.assertEqual(generated_cells, ["cell-c"])
            self.assertEqual(result["resumed_cells"], 1)
            self.assertEqual(result["generated_cells"], 2)
            self.assertEqual(result["parallel_generation_reduction"]["cell_count"], 1)
            self.assertFalse(
                result["parallel_generation_reduction"]["attempt_transport_retained"]
            )
            self.assertEqual(
                result["residual_transport_reduction"]["cell_id"], "cell-b"
            )
            self.assertEqual(verifier.calls, 1)
            self.assertFalse((transport_directory / "cell-00000.jsonl").exists())

    def test_parallel_assembly_verifies_before_atomic_manifest_write(self) -> None:
        preregistration: dict[str, object] = {}

        class FakeVerifier:
            def receipt(self) -> dict[str, object]:
                return {"verified_cell_count": 39}

        verifier = FakeVerifier()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            preregistration_path = root / "preregistration.json"
            preregistration_path.write_text("{}")
            batch = root / "batch"
            batch.write_bytes(b"batch")
            benchmark = root / "benchmark"
            benchmark.write_bytes(b"benchmark")
            preoutcome = root / "preoutcome"
            preoutcome.mkdir()
            run_root = root / "run"
            run_root.mkdir()
            manifests = run_root / "shards"
            manifests.mkdir()
            destination = run_root / "run-manifest.json"
            original_summary_root = spatial_simulation._summary_root

            def fake_build(*args: object, **kwargs: object) -> dict[str, object]:
                self.assertIs(spatial_simulation._summary_root, verifier)
                self.assertEqual(
                    args[2], (manifests / "manifest-00000.jsonl",)
                )
                return {"schema": "manifest"}

            with (
                patch(
                    "validation.run_spatial_covariance_parallel._require_exact_validation_sources"
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation._validated_producer_identities",
                    return_value=("a" * 64, "b" * 64),
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.rust_attempt_regenerator",
                    return_value=object(),
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.build_run_manifest",
                    side_effect=fake_build,
                ),
                patch(
                    "validation.run_spatial_covariance_parallel.spatial_simulation.write_run_manifest_atomic",
                    return_value={"sha256": "c" * 64, "bytes": 123},
                ) as write_manifest,
            ):
                result = assemble_parallel_run(
                    preregistration,
                    preregistration_path,
                    source,
                    batch,
                    benchmark,
                    preoutcome,
                    run_root,
                    manifests,
                    {"schema": "performance"},
                    [{"schema": "resources"}],
                    {"schema": "parity"},
                    destination,
                    verifier,
                )

            self.assertIs(spatial_simulation._summary_root, original_summary_root)
            write_manifest.assert_called_once_with(
                {"schema": "manifest"}, destination
            )
            self.assertEqual(result["manifest_sha256"], "c" * 64)
            self.assertEqual(
                result["parallel_summary_verification"]["verified_cell_count"],
                39,
            )

    @unittest.skipUnless(
        os.environ.get("DOLPHINRUST_SPATIAL_EXACT_ROOT"),
        "exact-main spatial replay smoke is opt-in",
    )
    def test_real_pipe_generation_is_byte_identical_under_exact_replay(self) -> None:
        source_root = Path(os.environ["DOLPHINRUST_SPATIAL_EXACT_ROOT"])
        preregistration_path = source_root / "validation/spatial_covariance_preregistration.json"
        batch = source_root / "target/release/examples/spatial_covariance_batch"
        benchmark = source_root / "target/release/examples/spatial_covariance_bench"
        preregistration = spatial_simulation.load_preregistration(preregistration_path)
        code_sha256, binary_sha256 = spatial_simulation._validated_producer_identities(
            preregistration, source_root, batch, benchmark
        )
        preregistration_sha256 = spatial_simulation.preregistration_digest(preregistration)
        preregistration_file_sha256 = _sha256_file(
            preregistration_path,
            FROZEN_MAX_RESOURCE_RECEIPT_BYTES,
            "integration preregistration",
        )
        spec = next(spatial_simulation.iter_shard_specs(preregistration))
        cell_id = spec.cell_ids[0]
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            summaries = run_root / "cells/shard-00000"
            summaries.mkdir(parents=True)
            destination = summaries / "cell-00000.jsonl"
            common = {
                "source_root": str(source_root),
                "preregistration_path": str(preregistration_path),
                "batch_binary": str(batch),
                "benchmark_binary": str(benchmark),
                "cell_id": cell_id,
                "cell_ordinal": 0,
                "preregistration_sha256": preregistration_sha256,
                "preregistration_file_sha256": preregistration_file_sha256,
                "code_sha256": code_sha256,
                "binary_sha256": binary_sha256,
            }
            generated, generation_peak = _run_bounded_pool(
                _generate_and_reduce_cell,
                [_GenerationTask(
                    run_root=str(run_root),
                    destination=str(destination),
                    **common,
                )],
                1,
            )
            replayed, replay_peak = _run_bounded_pool(
                _verify_cell,
                [_ReplayTask(
                    summary_directory=str(summaries),
                    artifact_root=str(run_root),
                    shard_index=0,
                    **common,
                )],
                1,
            )
        self.assertFalse(generated[0]["attempt_transport_retained"])
        self.assertEqual(generated[0]["summary_sha256"], replayed[0]["summary_sha256"])
        self.assertEqual(
            replayed[0]["exact_replay_root_sha256"],
            _single_summary_root(generated[0]["summary_sha256"]),
        )
        self.assertLessEqual(generation_peak, FROZEN_PROCESS_RSS_BYTES)
        self.assertLessEqual(replay_peak, FROZEN_PROCESS_RSS_BYTES)


if __name__ == "__main__":
    unittest.main()
