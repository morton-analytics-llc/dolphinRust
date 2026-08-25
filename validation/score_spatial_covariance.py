#!/usr/bin/env python3
"""Fail-closed compact scorer for the outcome-free F54-07 v4 protocol."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import stat
import struct
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np

PASS = "pass"
FAIL = "fail"
NOT_EVALUABLE = "not_evaluable"
STATUSES = {PASS, FAIL, NOT_EVALUABLE}
ATTEMPT_STATUSES = {"valid", "masked_target", "tied_eigenvalue"}
HASH_RE = set("0123456789abcdef")
FROZEN_SEED_COUNT = 5000
FROZEN_DETERMINISTIC_SEED_COUNT = 1
FROZEN_CELL_COUNT = 353
FROZEN_ATTEMPT_COUNT = 440265
FROZEN_MAX_CELLS_PER_SHARD = 100
FROZEN_SHARD_COUNT = 4
FROZEN_MAX_SHARD_BYTES = 100 * 8192
FROZEN_MAX_RECORD_BYTES = 262144
FROZEN_MAX_CELL_SUMMARY_BYTES = 8192
FROZEN_MAX_SHARD_MANIFEST_BYTES = 16384
FROZEN_MAX_RUN_MANIFEST_BYTES = 16777216
FROZEN_MAX_RESOURCE_RECEIPT_BYTES = 1 << 20
FROZEN_CELL_SUMMARY_COMPONENT_BYTES = FROZEN_CELL_COUNT * FROZEN_MAX_CELL_SUMMARY_BYTES
FROZEN_RETAINED_SIZE_BOUND_BYTES = 19734528
FROZEN_PROCESS_RSS_BYTES = 24 << 30
FROZEN_GENERATOR_SHA256 = "a017a15f48755f9dbaceb5842709aa74fe214d9025021eeb7665215da969c74e"
FROZEN_SCIENTIFIC_GENERATOR_SHA256 = "38a11589430b5f78add22ed7ad9cb96cddca1af098211e899bc306f4a3a61f0f"
FROZEN_EXECUTION_SHA256 = "6299045b34ce6e9e6c6c6ce87502fe0028cc7d1ae0d0ec4f5f3305e10cf706c6"
FROZEN_REDUCERS_SHA256 = "2dbe430d050e79388bc1484e1f270435dba3031468f4b4abb93315e2ed48cf53"
FROZEN_MATRIX_SHA256 = "eb079cb384196d28bfc3b957697769d186cd1c8fc04155282fb258a99eff2ac0"
FROZEN_RECEIPT_SHA256 = "931f90ef3fa0c97c16926ad96a4ea6661f1d73f7310a22e2351059cf64b02ca1"
FROZEN_HASH_FIELDS_SHA256 = "c9ffda0f207a373e94e4125f81a792c043ffa40cba3a945f9013741eebacade9"
FROZEN_RESOURCE_SAMPLING_SHA256 = "75309d564a3ad7f4fb8765b89af7696a9b995dd2a6a4c65bf8f792ee8ec9847c"
FROZEN_RESOURCE_MATRIX_SHA256 = "2da4e6ab51c72437791b4ae8c225e1df7a4e78da74838dfbade162335e2fdd69"
FROZEN_CELL_POLICY_SHA256 = "2c58266788cc16eadc487ea21bcf90aed9dd0ffe4ccf83807743f30a7d1d448e"
FROZEN_V3_PREREGISTRATION_SHA256 = "d1b29a1dc63a69c952397af1c713e604142b260be7068d37ee0f1a3158b88184"
FROZEN_DETERMINISM_SHA256 = "145cd8d79f58a9290c2e8669b45edd5bcd387f3b4d99e3ac92286f248437bdda"
FROZEN_NUMERIC_SHA256 = "f534593ef3027eb41c15bc40163c030c6eadeab131bfd06922bcb43fbc3e3811"
FROZEN_RESOURCE_IDS = ("area_128_dates_26", "area_256_dates_26", "area_512_dates_26", "area_256_dates_13", "area_256_dates_52")
DIMENSION_NAMES = ("half_window", "stride", "support", "position", "pair_geometry", "block_topology", "estimator", "eigen_stress", "source_process")
FROZEN_DIMENSION_IDS = {
    "half_window": ("hw_1x1", "hw_3x6", "hw_7x14"),
    "stride": ("stride_1", "stride_2", "stride_4"),
    "support": ("rect", "glrt_frozen", "ks_frozen"),
    "position": ("interior", "border_clamped", "tile_edge", "bounded_halo", "masked"),
    "pair_geometry": ("coincident", "shared_75_positive", "shared_75_negative", "shared_50_positive", "shared_50_negative", "shared_25_positive", "shared_25_negative", "disjoint_immediate", "disjoint_after_depth_1", "disjoint_after_depth_2", "disjoint_after_depth_4"),
    "block_topology": ("one_block", "two_blocks", "four_blocks", "two_blocks_cap_eviction", "four_blocks_partial_final"),
    "estimator": ("emi", "evd"),
    "eigen_stress": ("well_separated", "tied_eigenvalue", "near_tie"),
    "source_process": ("independent_complex_looks", "spatial_correlation_stress"),
}
FROZEN_THRESHOLDS = {
    "deterministic_operator_relative_error_max": 1e-10,
    "stochastic_operator_relative_error_max": 0.1,
    "contrast_variance_relative_error_max": 0.1,
    "psd_min_eigenvalue_min": -1e-10,
    "coverage_probability": 0.95,
    "coverage_absolute_error_max": 0.02,
    "emission_rate_min": 0.99,
    "weak_zero_variance_max": 1e-14,
    "resource_rss_bytes_max": FROZEN_PROCESS_RSS_BYTES,
    "resource_growth": "area_or_dates_linear; no quadratic axis",
    "resource_buffer_policy": "no_tile_area_times_s_b_times_d_b_squared_buffer",
}
PAIR_SIGN = {
    "coincident": "zero", "shared_75_positive": "positive", "shared_75_negative": "negative",
    "shared_50_positive": "positive", "shared_50_negative": "negative", "shared_25_positive": "positive",
    "shared_25_negative": "negative", "disjoint_immediate": "none", "disjoint_after_depth_1": "none",
    "disjoint_after_depth_2": "none", "disjoint_after_depth_4": "none",
}
DISJOINT_GEOMETRIES = {geometry for geometry, sign in PAIR_SIGN.items() if sign == "none"}
ATTEMPT_KEYS = {
    "schema", "cell_id", "cell_ordinal", "seed_index", "seed_sha256", "status", "emitted", "factor_emitted",
    "raw_input_sha256", "truth_sha256", "operator_hash", "variance_hash", "emission_hash", "date_axis_sha256",
    "generator_hash", "config_hash", "source_model_hash", "target_coordinate", "reference_coordinate",
    "target_support_sha256", "reference_support_sha256", "target_source_count", "reference_source_count",
    "intersection_source_count", "union_source_count", "realized_overlap_jaccard", "signed_cross_influence",
    "signed_influence_sign", "effective_looks_fraction", "effective_looks_application", "operator_relative_error",
    "contrast_variance_reference", "contrast_variance_relative_error", "psd_min_eigenvalue", "covered_95", "interval_score", "interval_width",
    "operator_matrix", "truth_matrix", "contrast_weights", "estimate_value", "truth_value",
    "raw_input_shape", "raw_input_value_count", "target_raw_input_sha256", "reference_raw_input_sha256",
    "sequential_ancestry_sha256", "raw_dgp_identity_sha256",
}
INPUT_KEYS = {"schema", "cell_id", "cell_ordinal", "seed_index", "seed_sha256", *DIMENSION_NAMES}
ATTEMPT_HASH_FIELDS = (
    "seed_sha256", "raw_input_sha256", "truth_sha256", "operator_hash", "variance_hash", "emission_hash",
    "date_axis_sha256", "generator_hash", "config_hash", "source_model_hash", "target_support_sha256",
    "reference_support_sha256", "target_raw_input_sha256", "reference_raw_input_sha256",
    "sequential_ancestry_sha256", "raw_dgp_identity_sha256",
)
CELL_SUMMARY_KEYS = {
    "schema", "cell_id", "cell_ordinal", "status", "attempted_seeds", "emitted_seeds",
    "status_histogram", "failure_histogram", "request_digest", "attempt_digest", "operator_digest",
    "truth_digest", "variance_digest", "emission_digest", "target_source_count_total", "reference_source_count_total",
    "intersection_source_count_total", "union_source_count_total", "realized_overlap_jaccard_mean",
    "effective_looks_fraction", "operator_relative_error", "contrast_variance_relative_error",
    "variance_evaluable", "psd_min_eigenvalue", "coverage_95", "interval_score_mean",
    "interval_width_mean", "code_sha256", "binary_sha256", "preregistration_sha256",
}
SHARD_MANIFEST_KEYS = {
    "schema", "schema_version", "shard_index", "cell_ordinal_start", "cell_ordinal_end_exclusive",
    "expected_cells", "expected_attempts", "summary_path", "summary_sha256", "summary_bytes",
    "summary_records", "preregistration_sha256", "code_sha256", "binary_sha256",
    "generator_protocol_sha256", "elapsed_seconds", "peak_rss_bytes", "committed",
}
RUN_MANIFEST_KEYS = {
    "schema", "schema_version", "preregistration_sha256", "code_sha256", "binary_sha256",
    "generator_protocol_sha256", "performance_probe", "resources", "shard_manifests", "result_root_sha256",
}
RESOURCE_KEYS = {
    "resource_id", "status", "rss_bytes", "growth_class", "resource_hash", "config_hash", "binary_hash", "os",
    "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "warmup_runs", "measured_repetitions",
    "tool_versions", "growth_observation", "area_growth_exponent", "date_growth_exponent", "acceptance",
}
PERFORMANCE_MEASUREMENT_KEYS = {
    "cell_class", "seed_count", "attempt_count", "elapsed_seconds", "peak_rss_bytes", "outcomes_persisted",
}
RESOURCE_OBSERVATION_KEYS = {
    "repetition", "tile_pixels", "date_count", "peak_rss_bytes", "wall_seconds", "raw_measurement", "raw_measurement_sha256",
}
RESOURCE_RAW_MEASUREMENT_KEYS = {
    "command", "exit_status", "wall_seconds", "max_rss_bytes", "rss_sampler", "rss_field", "os",
    "hardware_class", "ram_bytes", "tool_versions",
}


class SchemaError(ValueError):
    """The preregistration or receipt is not the frozen contract."""


@dataclass(frozen=True)
class ShardSpec:
    index: int
    cell_ordinal_start: int
    cell_ordinal_end_exclusive: int
    cell_ids: tuple[str, ...]
    seed_counts: tuple[int, ...] = ()

    @property
    def expected_attempts(self) -> int:
        return sum(self.seed_counts) if self.seed_counts else len(self.cell_ids) * FROZEN_SEED_COUNT


def load_preregistration(path: Path) -> Dict[str, Any]:
    raw = _read_bounded_bytes(Path(path), 4 * 1024 * 1024, "preregistration")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("preregistration is malformed JSON") from exc
    if not isinstance(value, dict):
        raise SchemaError("preregistration root must be an object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, byte_limit: int | None = None) -> tuple[str, int]:
    path = Path(path)
    admitted_size = path.stat().st_size
    if byte_limit is not None and admitted_size > byte_limit:
        raise SchemaError(f"{path} exceeds the frozen uncompressed byte cap before read")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            if byte_limit is not None and size > byte_limit:
                raise SchemaError(f"{path} exceeds the frozen uncompressed byte cap")
            digest.update(chunk)
    if size != admitted_size:
        raise SchemaError(f"{path} changed while it was being read")
    return digest.hexdigest(), size


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_bounded_bytes(path: Path, byte_limit: int, label: str) -> bytes:
    path = Path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise SchemaError(f"{label} is not a regular file")
    if before.st_size > byte_limit:
        raise SchemaError(f"{label} exceeds its frozen byte cap before read")
    chunks: list[bytes] = []
    byte_count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _file_identity(opened) != _file_identity(before):
            raise SchemaError(f"{label} changed before it was opened")
        while True:
            chunk = handle.read(min(65536, byte_limit + 1 - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > byte_limit:
                raise SchemaError(f"{label} exceeds its frozen byte cap during read")
        after = os.fstat(handle.fileno())
    path_after = path.stat()
    if _file_identity(after) != _file_identity(before) or _file_identity(path_after) != _file_identity(before) or byte_count != before.st_size:
        raise SchemaError(f"{label} changed while it was being read")
    return b"".join(chunks)


def _read_single_json_record(path: Path, byte_limit: int, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_bounded_bytes(Path(path), byte_limit, label)
    if not raw:
        raise SchemaError(f"{label} is empty")
    if not raw.endswith(b"\n"):
        raise SchemaError(f"{label} lacks newline framing")
    if b"\n" in raw[:-1]:
        raise SchemaError(f"{label} must contain exactly one JSON record")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{label} is not an object")
    return value, raw


def resolve_below_run_root(run_root: Path, relative_path: Any, label: str) -> Path:
    root = Path(run_root).resolve(strict=True)
    path = Path(relative_path) if isinstance(relative_path, (str, os.PathLike)) else Path()
    if not isinstance(relative_path, (str, os.PathLike)) or path.is_absolute() or ".." in path.parts:
        raise SchemaError(f"{label} escapes the run root")
    resolved = (root / path).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SchemaError(f"{label} escapes the run root through a symlink") from exc
    return resolved


def preregistration_digest(preregistration: Mapping[str, Any]) -> str:
    return sha256_json(preregistration)


def seed_schedule_digest(preregistration: Mapping[str, Any]) -> str:
    return sha256_json(preregistration.get("seed_schedule"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HASH_RE


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value: Any) -> bool:
    return type(value) is int


def _dimension_values(preregistration: Mapping[str, Any], name: str) -> Sequence[str]:
    values = preregistration.get("dimensions", {}).get(name, [])
    return tuple(item.get("id") for item in values if isinstance(item, dict))


def _planner_rows(num_slc: int, ministack_size: int, max_num_compressed: int) -> list[dict[str, int]]:
    return [{"block_id": block_id, "num_compressed": min(block_id, max_num_compressed), "real_start": real_start, "num_real": min(ministack_size, num_slc - real_start)} for block_id, real_start in enumerate(range(0, num_slc, ministack_size))]


def _scientific_generator(generator: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in generator.items() if key not in {"binary", "identity"}}


def _validate_executable_generator(preregistration: Mapping[str, Any], errors: List[str]) -> None:
    generator = preregistration.get("generator")
    if not isinstance(generator, dict):
        return
    raw = generator.get("raw_proper_complex", {})
    if (
        raw.get("model") != "production_shaped_proper_complex_ar1_v2"
        or raw.get("source_shape") != "realized_support_union_by_acquisition"
        or raw.get("component_order") != ["real", "imag"]
        or "AR(1)" not in raw.get("temporal_signal", "")
        or raw.get("pseudo_covariance") != "E[Z Z^T]=0"
    ):
        errors.append("raw generator must define the frozen full proper-complex source-by-acquisition process")
    replay = generator.get("full_replay_dgp", {})
    if (
        replay.get("native_tile_shape") != [256, 256]
        or replay.get("raw_shape") != "realized support union count by complete topology acquisition count by real/imag"
        or "every expected ministack" not in replay.get("support_generation", "")
        or "2N_by_2N" not in replay.get("truth", "")
        or "exact zero" not in replay.get("coincident", "")
    ):
        errors.append("full replay DGP must bind raw shape, direct joint truth, and coincident zero")
    source = generator.get("source_centered_empirical", {})
    if source.get("covariance_definition") != "E[Z Z*]/n" or source.get("mean") != "zero; no sample-mean subtraction" or source.get("floor_application") != "after_zero_mean_covariance_and_shrinkage":
        errors.append("source model must be zero-mean E[Z Z*]/n with post-shrinkage floor")
    acquisition = generator.get("acquisition", {})
    for topology, spec in acquisition.get("topologies", {}).items():
        expected = _planner_rows(spec.get("acquisition_count", 0), spec.get("ministack_size", 0), spec.get("max_num_compressed", 0))
        if spec.get("expected_blocks") != expected or len(spec.get("date_axis", [])) != spec.get("acquisition_count", -1):
            errors.append(f"topology {topology} does not reproduce the frozen planner")
        try:
            dates = [date.fromisoformat(value) for value in spec.get("date_axis", [])]
        except (TypeError, ValueError):
            errors.append(f"topology {topology} contains a non-ISO date")
            continue
        if any((dates[index + 1] - dates[index]).days != acquisition.get("cadence_days") for index in range(len(dates) - 1)):
            errors.append(f"topology {topology} does not use the frozen cadence")
    coordinates = generator.get("coordinates", {})
    expected_windows = {f"{half_window}|{stride}" for half_window in FROZEN_DIMENSION_IDS["half_window"] for stride in FROZEN_DIMENSION_IDS["stride"]}
    if set(coordinates.get("window_stride", {})) != expected_windows:
        errors.append("coordinates must provide every half-window/stride realization")
    neighbors = generator.get("neighbor_generation", {})
    if neighbors.get("full_half_window") is not True or neighbors.get("glrt", {}).get("alpha") != 0.001 or neighbors.get("ks", {}).get("alpha") != 0.001 or neighbors.get("fixed_support_reuse") is not True:
        errors.append("GLRT/KS support contract does not match the frozen production algorithms")
    supported = generator.get("supported", {})
    if supported.get("stable_attempt_statuses") != ["valid", "masked_target", "tied_eigenvalue"] or supported.get("not_evaluable_if") != ["tied_eigenvalue"] or supported.get("expected_abstention_if") != ["masked_target"]:
        errors.append("attempt status policy drifted")


def validate_preregistration(preregistration: Mapping[str, Any]) -> None:
    errors: List[str] = []
    if preregistration.get("schema") != "dolphinrust.spatial_covariance.preregistration" or not _integer(preregistration.get("schema_version")) or preregistration.get("schema_version") != 4:
        errors.append("preregistration must use the F54-07 v4 schema")
    if preregistration.get("status") != "preregistered" or preregistration.get("outcomes_present") is not False:
        errors.append("preregistration must remain outcome-free and preregistered")
    supersedes = preregistration.get("supersedes")
    if not isinstance(supersedes, dict) or supersedes.get("schema_version") != 3 or supersedes.get("canonical_preregistration_sha256") != FROZEN_V3_PREREGISTRATION_SHA256 or supersedes.get("outcomes_present") is not False:
        errors.append("v4 must bind and outcome-free supersede the exact v3 preregistration")
    dimensions = preregistration.get("dimensions")
    if not isinstance(dimensions, dict) or tuple(dimensions) != DIMENSION_NAMES:
        errors.append("dimensions must contain the nine frozen axes in order")
    else:
        for name in DIMENSION_NAMES:
            if _dimension_values(preregistration, name) != FROZEN_DIMENSION_IDS[name]:
                errors.append(f"dimension {name} does not match the frozen matrix")
    schedule = preregistration.get("seed_schedule")
    if (
        not isinstance(schedule, dict)
        or schedule.get("supported_monte_carlo_seeds") != FROZEN_SEED_COUNT
        or schedule.get("deterministic_contract_seeds") != FROZEN_DETERMINISTIC_SEED_COUNT
        or schedule.get("no_top_up") is not True
        or schedule.get("selection_rule") != "one seed for masked, tied, non-interior, coincident, and disjoint cells; 5000 fixed seeds otherwise"
    ):
        errors.append("seed schedule must freeze supported and deterministic counts without top-up")
    if preregistration.get("thresholds") != FROZEN_THRESHOLDS:
        errors.append("thresholds differ from immutable F54-07 thresholds")
    for field_name, frozen_hash, message in (
        ("matrix_contract", FROZEN_MATRIX_SHA256, "matrix contract must freeze the exact pairwise risk design and attempt count"),
        ("execution_protocol", FROZEN_EXECUTION_SHA256, "execution protocol differs from the frozen v4 compact contract"),
        ("cell_reducers", FROZEN_REDUCERS_SHA256, "cell reducers or denominators differ from the frozen v3 contract"),
        ("receipt_contract", FROZEN_RECEIPT_SHA256, "receipt contract differs from the frozen v3 contract"),
        ("hash_fields", FROZEN_HASH_FIELDS_SHA256, "receipt identity fields differ from the frozen contract"),
        ("resource_sampling", FROZEN_RESOURCE_SAMPLING_SHA256, "resource sampling differs from the frozen contract"),
        ("resource_matrix", FROZEN_RESOURCE_MATRIX_SHA256, "resource matrix differs from the frozen contract"),
        ("cell_policy", FROZEN_CELL_POLICY_SHA256, "cell status policy differs from the frozen contract"),
        ("determinism", FROZEN_DETERMINISM_SHA256, "deterministic numeric generation contract drifted"),
        ("numeric_contract", FROZEN_NUMERIC_SHA256, "independent numeric scoring contract drifted"),
    ):
        value = preregistration.get(field_name)
        if value is None or sha256_json(value) != frozen_hash:
            errors.append(message)
    generator = preregistration.get("generator")
    if not isinstance(generator, dict) or sha256_json(generator) != FROZEN_GENERATOR_SHA256:
        errors.append("generator parameters/protocol differ from the frozen v4 generator")
    elif sha256_json(_scientific_generator(generator)) != FROZEN_SCIENTIFIC_GENERATOR_SHA256:
        errors.append("scientific generator differs from the outcome-free v2 design")
    execution = preregistration.get("execution_protocol", {})
    retained_bound = (
        FROZEN_CELL_COUNT * execution.get("max_encoded_cell_summary_bytes", 0)
        + FROZEN_SHARD_COUNT * execution.get("max_encoded_shard_manifest_bytes", 0)
        + execution.get("max_encoded_run_manifest_bytes", 0)
    )
    if execution.get("retained_attempt_records") is not False or execution.get("request_files_retained") is not False or retained_bound > execution.get("retained_size_bound_bytes", -1) or execution.get("retained_size_bound_bytes") != FROZEN_RETAINED_SIZE_BOUND_BYTES:
        errors.append("v4 retained evidence does not satisfy the frozen compact bound")
    if execution.get("process_rss_bytes_max") != FROZEN_PROCESS_RSS_BYTES:
        errors.append("execution process cap must equal the frozen 24 GiB resource threshold")
    _validate_executable_generator(preregistration, errors)
    if errors:
        raise SchemaError("; ".join(errors))


def iter_expected_cell_ids(preregistration: Mapping[str, Any]) -> Iterator[str]:
    validate_preregistration(preregistration)
    values = [_dimension_values(preregistration, name) for name in DIMENSION_NAMES]
    defaults = [dimension[0] for dimension in values]
    cells: set[tuple[str, ...]] = set()
    for first, second in itertools.combinations(range(len(DIMENSION_NAMES)), 2):
        for first_value in values[first]:
            for second_value in values[second]:
                labels = defaults.copy()
                labels[first] = first_value
                labels[second] = second_value
                cells.add(tuple(labels))
    for cell_id in preregistration["matrix_contract"]["risk_cells"]:
        labels = tuple(cell_id.split("|"))
        if len(labels) != len(DIMENSION_NAMES) or any(labels[index] not in values[index] for index in range(len(labels))):
            raise SchemaError("risk cell is outside the frozen dimensions")
        cells.add(labels)
    if len(cells) != FROZEN_CELL_COUNT:
        raise SchemaError("pairwise risk design does not contain its exact frozen cell count")
    return ("|".join(labels) for labels in sorted(cells))


def expected_cell_ids(preregistration: Mapping[str, Any]) -> List[str]:
    return list(iter_expected_cell_ids(preregistration))


def expected_seed_count(cell_id: str) -> int:
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    deterministic = (
        labels.get("position") != "interior"
        or labels.get("eigen_stress") == "tied_eigenvalue"
        or labels.get("pair_geometry") == "coincident"
        or labels.get("pair_geometry") in DISJOINT_GEOMETRIES
    )
    return FROZEN_DETERMINISTIC_SEED_COUNT if deterministic else FROZEN_SEED_COUNT


def iter_shard_specs(preregistration: Mapping[str, Any]) -> Iterator[ShardSpec]:
    cells = iter_expected_cell_ids(preregistration)
    ordinal = 0
    for shard_index in range(FROZEN_SHARD_COUNT):
        cell_ids = tuple(itertools.islice(cells, FROZEN_MAX_CELLS_PER_SHARD))
        if not cell_ids:
            raise SchemaError("frozen shard count exceeds the matrix")
        end = ordinal + len(cell_ids)
        yield ShardSpec(
            shard_index,
            ordinal,
            end,
            cell_ids,
            tuple(expected_seed_count(cell_id) for cell_id in cell_ids),
        )
        ordinal = end
    if next(cells, None) is not None or ordinal != FROZEN_CELL_COUNT:
        raise SchemaError("frozen shards do not cover exactly 353 cells")


def _expected_seed_hash(preregistration: Mapping[str, Any], cell_id: str, index: int) -> str:
    value = f"{preregistration['seed_schedule']['validation_seed']}||{cell_id}||{index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_ctr_uniforms(seed_digest: str, count: int) -> list[float]:
    if not _is_sha256(seed_digest) or not _integer(count) or count < 0:
        raise SchemaError("SHA-256-CTR request is invalid")
    words: list[float] = []
    counter = 0
    seed = bytes.fromhex(seed_digest)
    while len(words) < count:
        block = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        for offset in range(0, len(block), 8):
            value = int.from_bytes(block[offset : offset + 8], "big")
            words.append(((value >> 11) + 0.5) / 9007199254740992.0)
            if len(words) == count:
                break
        counter += 1
    return words


def deterministic_normals(seed_digest: str, count: int) -> list[float]:
    uniforms = sha256_ctr_uniforms(seed_digest, 2 * ((count + 1) // 2))
    result: list[float] = []
    for offset in range(0, len(uniforms), 2):
        radius = math.sqrt(-2.0 * math.log(uniforms[offset]))
        angle = 2.0 * math.pi * uniforms[offset + 1]
        result.extend((radius * math.cos(angle), radius * math.sin(angle)))
    return result[:count]


def numeric_digest(domain: str, values: Iterable[float]) -> str:
    values = list(values)
    digest = hashlib.sha256(domain.encode("utf-8"))
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        if not _number(value):
            raise SchemaError("numeric digest contains non-finite evidence")
        canonical = 0.0 if value == 0.0 else float(value)
        digest.update(struct.pack(">d", canonical))
    return digest.hexdigest()


def _matrix(value: Any, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{label} is not a binary64 matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise SchemaError(f"{label} is not a finite square matrix")
    return matrix


def _frobenius(matrix: np.ndarray) -> float:
    total = 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            total += float(matrix[row, column]) * float(matrix[row, column])
    return math.sqrt(total)


def _quadratic(weights: np.ndarray, matrix: np.ndarray) -> float:
    total = 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            total += float(weights[row]) * float(matrix[row, column]) * float(weights[column])
    return total


def _minimum_symmetric_eigenvalue(matrix: np.ndarray) -> float:
    values = ((matrix + matrix.T) * 0.5).tolist()
    size = len(values)
    tolerance = 1e-14 * max(1.0, max(abs(values[index][index]) for index in range(size)))
    for _ in range(100 * size * size):
        maximum = max((abs(values[row][column]) for row in range(size) for column in range(row + 1, size)), default=0.0)
        if maximum <= tolerance:
            return min(values[index][index] for index in range(size))
        for row in range(size):
            for column in range(row + 1, size):
                off_diagonal = values[row][column]
                if abs(off_diagonal) <= tolerance:
                    continue
                tau = (values[column][column] - values[row][row]) / (2.0 * off_diagonal)
                tangent = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine
                row_diagonal = values[row][row]
                column_diagonal = values[column][column]
                values[row][row] = row_diagonal - tangent * off_diagonal
                values[column][column] = column_diagonal + tangent * off_diagonal
                values[row][column] = 0.0
                values[column][row] = 0.0
                for index in range(size):
                    if index in (row, column):
                        continue
                    first = values[index][row]
                    second = values[index][column]
                    values[index][row] = cosine * first - sine * second
                    values[row][index] = values[index][row]
                    values[index][column] = sine * first + cosine * second
                    values[column][index] = values[index][column]
    raise SchemaError("independent cyclic Jacobi PSD oracle did not converge")


def independently_recompute_metrics(attempt: Mapping[str, Any]) -> dict[str, Any]:
    operator = _matrix(attempt.get("operator_matrix"), "operator matrix")
    truth = _matrix(attempt.get("truth_matrix"), "truth matrix")
    weights = np.asarray(attempt.get("contrast_weights"), dtype=np.float64)
    if operator.shape != truth.shape or weights.shape != (operator.shape[0],) or not np.isfinite(weights).all():
        raise SchemaError("operator, truth, and contrast dimensions disagree")
    difference = operator - truth
    truth_norm = _frobenius(truth)
    operator_error = _frobenius(difference) / max(truth_norm, 1e-15)
    operator_variance = _quadratic(weights, operator)
    truth_variance = _quadratic(weights, truth)
    variance_error = None if truth_variance <= 1e-14 else abs(operator_variance - truth_variance) / max(abs(truth_variance), 1e-14)
    psd_minimum = _minimum_symmetric_eigenvalue(operator)
    estimate = attempt.get("estimate_value")
    truth_value = attempt.get("truth_value")
    if not _number(estimate) or not _number(truth_value) or operator_variance < -1e-10:
        raise SchemaError("coverage sufficient values are invalid")
    half_width = 1.959963984540054 * math.sqrt(max(operator_variance, 0.0))
    lower = estimate - half_width
    upper = estimate + half_width
    covered = lower <= truth_value <= upper
    interval_score = upper - lower
    if truth_value < lower:
        interval_score += 40.0 * (lower - truth_value)
    elif truth_value > upper:
        interval_score += 40.0 * (truth_value - upper)
    return {
        "operator_relative_error": operator_error,
        "contrast_variance_reference": truth_variance,
        "contrast_variance_relative_error": variance_error,
        "psd_min_eigenvalue": psd_minimum,
        "covered_95": covered,
        "interval_score": interval_score,
        "interval_width": upper - lower,
        "operator_hash": numeric_digest("operator-v4", operator.flat),
        "truth_sha256": numeric_digest("truth-v4", truth.flat),
        "variance_hash": numeric_digest("variance-v4", [operator_variance, truth_variance, *weights]),
        "emission_hash": numeric_digest("emission-v4", [estimate, truth_value]),
    }


def _same_metric(claimed: Any, recomputed: Any) -> bool:
    if isinstance(recomputed, bool):
        return type(claimed) is bool and claimed is recomputed
    if recomputed is None:
        return claimed is None
    return _number(claimed) and math.isclose(claimed, recomputed, rel_tol=1e-12, abs_tol=1e-12)


def _expected_coordinates(preregistration: Mapping[str, Any], cell_id: str) -> tuple[list[int], list[int]]:
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    window = preregistration["generator"]["coordinates"]["window_stride"][f"{labels['half_window']}|{labels['stride']}"]
    target = window["target_by_position"][labels["position"]]
    delta = window["reference_delta_by_pair_geometry"][labels["pair_geometry"]]
    return target, [target[0] + delta[0], target[1] + delta[1]]


def _candidate_support(
    center: Sequence[int], half_window: Sequence[int], native_tile_shape: Sequence[int]
) -> list[tuple[int, int]]:
    window_shape = [2 * half_window[0] + 1, 2 * half_window[1] + 1]
    tile_origin = [
        (center[axis] // native_tile_shape[axis]) * native_tile_shape[axis]
        for axis in range(2)
    ]
    start = [
        min(
            max(tile_origin[axis], center[axis] - half_window[axis]),
            tile_origin[axis] + native_tile_shape[axis] - window_shape[axis],
        )
        for axis in range(2)
    ]
    return [
        (start[0] + row, start[1] + column)
        for row in range(window_shape[0])
        for column in range(window_shape[1])
    ]


def _generate_complex_source(
    preregistration: Mapping[str, Any],
    cell_id: str,
    seed_index: int,
    coordinate: tuple[int, int],
    date_count: int,
    global_normals: Sequence[float],
    spatial: bool,
    eigen_stress: str,
) -> list[complex]:
    raw = preregistration["generator"]["raw_proper_complex"]
    seed = hashlib.sha256(
        f"{preregistration['seed_schedule']['validation_seed']}||{cell_id}||{seed_index}||{coordinate[0]},{coordinate[1]}".encode("utf-8")
    ).hexdigest()
    local = deterministic_normals(seed, 4 * date_count)
    spatial_weight = math.sqrt(0.5) if spatial else 0.0
    local_weight = math.sqrt(1.0 - spatial_weight * spatial_weight)
    temporal_rho = math.exp(-preregistration["generator"]["acquisition"]["cadence_days"] / raw["correlation_days"])
    innovation_weight = math.sqrt(1.0 - temporal_rho * temporal_rho)
    signal_real = 0.0
    signal_imag = 0.0
    values: list[complex] = []
    for acquisition in range(date_count):
        offset = 4 * acquisition
        innovation_real = local_weight * local[offset] + spatial_weight * global_normals[offset]
        innovation_imag = local_weight * local[offset + 1] + spatial_weight * global_normals[offset + 1]
        if acquisition == 0:
            signal_real, signal_imag = innovation_real, innovation_imag
        else:
            signal_real = temporal_rho * signal_real + innovation_weight * innovation_real
            signal_imag = temporal_rho * signal_imag + innovation_weight * innovation_imag
        stress_scale = {
            "well_separated": 1.0 + 0.15 * math.cos(acquisition + 1.0),
            "near_tie": 1.0 + (2.0**-24 if acquisition % 2 else 0.0),
            "tied_eigenvalue": 1.0,
        }[eigen_stress]
        amplitude = raw["signal_amplitude"] * stress_scale * (
            1.0 + 0.05 * math.sin((coordinate[0] + 3.0 * coordinate[1]) / 7.0)
        )
        phase = (
            raw["phase0_rad"]
            + raw["slope_rad_per_day"] * preregistration["generator"]["acquisition"]["cadence_days"] * acquisition
            + 0.125 * math.cos((2.0 * coordinate[0] - coordinate[1]) / 11.0)
        )
        rotation = complex(math.cos(phase), math.sin(phase))
        signal = amplitude * complex(signal_real, signal_imag) * rotation / math.sqrt(2.0)
        noise = math.sqrt(raw["noise_variance"] / 2.0) * complex(local[offset + 2], local[offset + 3])
        value = signal + noise
        values.append(complex(
            struct.unpack(">f", struct.pack(">f", value.real))[0],
            struct.unpack(">f", struct.pack(">f", value.imag))[0],
        ))
    return values


def _select_support(
    method: str,
    candidates: Sequence[tuple[int, int]],
    center: Sequence[int],
    raw_by_source: Mapping[tuple[int, int], Sequence[complex]],
) -> list[tuple[int, int]]:
    center_key = (center[0], center[1])
    if method == "rect":
        return list(candidates)
    center_magnitudes = [abs(value) for value in raw_by_source[center_key]]
    selected: list[tuple[int, int]] = []
    for coordinate in candidates:
        if coordinate == center_key:
            continue
        magnitudes = [abs(value) for value in raw_by_source[coordinate]]
        if method == "glrt_frozen":
            mean = sum(magnitudes) / len(magnitudes)
            variance = sum((value - mean) ** 2 for value in magnitudes) / len(magnitudes)
            center_mean = sum(center_magnitudes) / len(center_magnitudes)
            center_variance = sum((value - center_mean) ** 2 for value in center_magnitudes) / len(center_magnitudes)
            center_scale = (center_variance + center_mean * center_mean) / 2.0
            scale = (variance + mean * mean) / 2.0
            pooled = (center_scale + scale) / 2.0
            statistic = len(magnitudes) * (2.0 * math.log(pooled) - math.log(center_scale) - math.log(scale))
            keep = statistic < 10.827566170662733
        else:
            first = sorted(center_magnitudes)
            second = sorted(magnitudes)
            index_first = index_second = output = 0
            cdf_first = cdf_second = distance = 0.0
            step = 1.0 / len(first)
            while output < 2 * len(first):
                if index_first == len(first):
                    cdf_second += step
                    index_second += 1
                elif index_second == len(second) or first[index_first] < second[index_second]:
                    cdf_first += step
                    index_first += 1
                elif first[index_first] > second[index_second]:
                    cdf_second += step
                    index_second += 1
                else:
                    cdf_first += step
                    cdf_second += step
                    index_first += 1
                    index_second += 1
                    output += 1
                output += 1
                distance = max(distance, abs(cdf_first - cdf_second))
            sqrt_n = math.sqrt(len(first) / 2.0)
            cutoff = 0.01
            while cutoff <= 1.0:
                value = cutoff * (sqrt_n + 0.12 + 0.11 / sqrt_n)
                pvalue = min(1.0, max(0.0, 2.0 * sum((-1.0) ** (term - 1) * math.exp(-2.0 * value * value * term * term) for term in range(1, 101))))
                if pvalue <= 0.001:
                    break
                cutoff += 0.001
            keep = distance < (cutoff if cutoff <= 1.0 else 0.1)
        if keep:
            selected.append(coordinate)
    return selected


def _raw_source_digest(
    domain: str,
    support: Sequence[tuple[int, int]],
    raw_by_source: Mapping[tuple[int, int], Sequence[complex]],
) -> str:
    digest = hashlib.sha256(domain.encode("utf-8"))
    digest.update(len(support).to_bytes(8, "big"))
    for coordinate in support:
        digest.update(struct.pack(">qq", coordinate[0], coordinate[1]))
        values = raw_by_source[coordinate]
        digest.update(len(values).to_bytes(8, "big"))
        for value in values:
            digest.update(struct.pack(">dd", 0.0 if value.real == 0.0 else value.real, 0.0 if value.imag == 0.0 else value.imag))
    return digest.hexdigest()


def _sequential_phase(values: Sequence[complex], blocks: Sequence[Mapping[str, int]]) -> list[float]:
    raw_phase = [math.atan2((value * values[0].conjugate()).imag, (value * values[0].conjugate()).real) for value in values]
    result = [0.0] * len(values)
    for block in blocks:
        start = block["real_start"]
        parent = 0.0 if start == 0 else result[start - 1]
        local_reference = raw_phase[start]
        for acquisition in range(start, start + block["num_real"]):
            delta = math.atan2(math.sin(raw_phase[acquisition] - local_reference), math.cos(raw_phase[acquisition] - local_reference))
            result[acquisition] = parent + delta
    result[0] = 0.0
    return result


def _support_by_date(
    supports: Sequence[Sequence[tuple[int, int]]],
    blocks: Sequence[Mapping[str, int]],
    date_count: int,
) -> list[set[tuple[int, int]]]:
    result = [set() for _ in range(date_count)]
    for support, block in zip(supports, blocks):
        for acquisition in range(block["real_start"], block["real_start"] + block["num_real"]):
            result[acquisition] = set(support)
    return result


def _direct_phase_covariance(
    row_supports: Sequence[set[tuple[int, int]]],
    column_supports: Sequence[set[tuple[int, int]]],
    row_phases: Mapping[tuple[int, int], Sequence[float]],
    column_phases: Mapping[tuple[int, int], Sequence[float]],
    orientation: float = 1.0,
) -> list[list[float]]:
    date_count = len(row_supports)
    result = [[0.0] * date_count for _ in range(date_count)]
    for row in range(date_count):
        for column in range(date_count):
            shared = row_supports[row] & column_supports[column]
            scale = orientation / math.sqrt(len(row_supports[row]) * len(column_supports[column]))
            result[row][column] = scale * sum(
                row_phases[coordinate][row] * column_phases[coordinate][column]
                for coordinate in shared
            )
    return result


def _effective_looks_fraction(support: Sequence[tuple[int, int]], spatial: bool) -> float:
    if not spatial:
        return 1.0
    denominator = sum(
        math.exp(-math.hypot(first[0] - second[0], first[1] - second[1]) / 1.5)
        for first in support
        for second in support
    )
    return len(support) / denominator


def regenerate_frozen_attempt_inputs(
    preregistration: Mapping[str, Any], cell_id: str, seed_index: int
) -> dict[str, Any]:
    if not _integer(seed_index) or seed_index < 0 or seed_index >= expected_seed_count(cell_id):
        raise SchemaError("frozen DGP seed index is outside the preregistered schedule")
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    if set(labels) != set(DIMENSION_NAMES):
        raise SchemaError("frozen DGP cell identity is malformed")
    target, reference = _expected_coordinates(preregistration, cell_id)
    window = preregistration["generator"]["coordinates"]["window_stride"][f"{labels['half_window']}|{labels['stride']}"]
    native_tile_shape = preregistration["generator"]["full_replay_dgp"]["native_tile_shape"]
    target_candidates = _candidate_support(target, window["half_window"], native_tile_shape)
    reference_candidates = _candidate_support(reference, window["half_window"], native_tile_shape)
    candidate_union = sorted(set(target_candidates) | set(reference_candidates))
    topology = preregistration["generator"]["acquisition"]["topologies"][labels["block_topology"]]
    date_count = topology["acquisition_count"]
    global_seed = hashlib.sha256(
        f"{preregistration['seed_schedule']['validation_seed']}||{cell_id}||{seed_index}||global-source-process".encode("utf-8")
    ).hexdigest()
    global_normals = deterministic_normals(global_seed, 4 * date_count)
    spatial = labels["source_process"] == "spatial_correlation_stress"
    raw_by_source = {
        coordinate: _generate_complex_source(
            preregistration, cell_id, seed_index, coordinate, date_count,
            global_normals, spatial, labels["eigen_stress"],
        )
        for coordinate in candidate_union
    }
    target_supports = []
    reference_supports = []
    for block in topology["expected_blocks"]:
        block_dates = range(block["real_start"], block["real_start"] + block["num_real"])
        block_raw = {
            coordinate: [raw_by_source[coordinate][acquisition] for acquisition in block_dates]
            for coordinate in candidate_union
        }
        target_supports.append(_select_support(labels["support"], target_candidates, target, block_raw))
        reference_supports.append(_select_support(labels["support"], reference_candidates, reference, block_raw))
    target_support = sorted(set().union(*map(set, target_supports)))
    reference_support = sorted(set().union(*map(set, reference_supports)))
    union_support = sorted(set(target_support) | set(reference_support))
    target_phases = {coordinate: _sequential_phase(raw_by_source[coordinate], topology["expected_blocks"]) for coordinate in target_support}
    reference_phases = {coordinate: _sequential_phase(raw_by_source[coordinate], topology["expected_blocks"]) for coordinate in reference_support}
    target_by_date = _support_by_date(target_supports, topology["expected_blocks"], date_count)
    reference_by_date = _support_by_date(reference_supports, topology["expected_blocks"], date_count)
    target_covariance = _direct_phase_covariance(target_by_date, target_by_date, target_phases, target_phases)
    reference_covariance = target_covariance if target_supports == reference_supports else _direct_phase_covariance(reference_by_date, reference_by_date, reference_phases, reference_phases)
    shared = sorted(set(target_support) & set(reference_support))
    orientation = -1.0 if labels["pair_geometry"].endswith("_negative") else 1.0
    if target_supports == reference_supports:
        cross_covariance = target_covariance
    else:
        cross_covariance = _direct_phase_covariance(
            target_by_date, reference_by_date, target_phases, reference_phases, orientation
        )
    effective_looks = _effective_looks_fraction(union_support, spatial)
    target_covariance = [[value / effective_looks for value in row] for row in target_covariance]
    reference_covariance = [[value / effective_looks for value in row] for row in reference_covariance]
    cross_covariance = [[value / effective_looks for value in row] for row in cross_covariance]
    truth_matrix = [
        target_covariance[row] + cross_covariance[row]
        for row in range(date_count)
    ] + [
        [cross_covariance[column][row] for column in range(date_count)] + reference_covariance[row]
        for row in range(date_count)
    ]
    weights = [0.0] * (2 * date_count)
    weights[date_count - 1] = 1.0
    weights[2 * date_count - 1] = -1.0
    if target_supports == reference_supports:
        truth_value = 0.0
    else:
        truth_value = (
            sum(vector[-1] for vector in target_phases.values()) / len(target_phases)
            - orientation * sum(vector[-1] for vector in reference_phases.values()) / len(reference_phases)
        )
    target_support_receipt = [
        {"block_id": block["block_id"], "sources": support}
        for block, support in zip(topology["expected_blocks"], target_supports)
    ]
    reference_support_receipt = [
        {"block_id": block["block_id"], "sources": support}
        for block, support in zip(topology["expected_blocks"], reference_supports)
    ]
    target_support_sha256 = sha256_json(target_support_receipt)
    reference_support_sha256 = sha256_json(reference_support_receipt)
    ancestry = {
        "date_axis": topology["date_axis"],
        "expected_blocks": topology["expected_blocks"],
        "max_num_compressed": topology["max_num_compressed"],
        "partial_tail_count": topology["partial_tail_count"],
    }
    sequential_ancestry_sha256 = sha256_json(ancestry)
    raw_identity = {
        "cell_id": cell_id,
        "seed_index": seed_index,
        "shape": [len(union_support), date_count, 2],
        "target_coordinate": target,
        "reference_coordinate": reference,
        "target_support_sha256": target_support_sha256,
        "reference_support_sha256": reference_support_sha256,
        "sequential_ancestry_sha256": sequential_ancestry_sha256,
        "estimator": labels["estimator"],
        "eigen_stress": labels["eigen_stress"],
        "source_process": labels["source_process"],
    }
    return {
        "raw_input_shape": [len(union_support), date_count, 2],
        "raw_input_value_count": 2 * len(union_support) * date_count,
        "raw_input_sha256": _raw_source_digest("raw-input-v4", union_support, raw_by_source),
        "target_raw_input_sha256": _raw_source_digest("source-raw-input-v4", target_support, raw_by_source),
        "reference_raw_input_sha256": _raw_source_digest("source-raw-input-v4", reference_support, raw_by_source),
        "target_support_sha256": target_support_sha256,
        "reference_support_sha256": reference_support_sha256,
        "sequential_ancestry_sha256": sequential_ancestry_sha256,
        "raw_dgp_identity_sha256": sha256_json(raw_identity),
        "target_source_count": len(target_support),
        "reference_source_count": len(reference_support),
        "intersection_source_count": len(shared),
        "union_source_count": len(union_support),
        "effective_looks_fraction": effective_looks,
        "truth_matrix": truth_matrix,
        "truth_sha256": numeric_digest("truth-v4", itertools.chain.from_iterable(truth_matrix)),
        "contrast_weights": weights,
        "truth_value": truth_value,
    }


def realized_overlap_jaccard(target_count: Any, reference_count: Any, intersection_count: Any, union_count: Any) -> float:
    counts = (target_count, reference_count, intersection_count, union_count)
    if any(not _integer(value) or value < 0 for value in counts):
        raise SchemaError("source-key overlap counts must be non-negative integers")
    if intersection_count > min(target_count, reference_count) or union_count != target_count + reference_count - intersection_count or union_count == 0:
        raise SchemaError("source-key intersection/union arithmetic is invalid")
    return intersection_count / union_count


def _update_field_digest(digest: Any, value: Any) -> None:
    encoded = _canonical_bytes(value)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


@dataclass
class CellAccumulator:
    preregistration: Mapping[str, Any]
    cell_id: str
    cell_ordinal: int
    expected_seed_count: int | None = None
    code_sha256: str = "0" * 64
    binary_sha256: str = "0" * 64
    next_seed_index: int = 0
    emitted: int = 0
    covered: int = 0
    coverage_denominator: int = 0
    target_total: int = 0
    reference_total: int = 0
    intersection_total: int = 0
    union_total: int = 0
    overlap_total: float = 0.0
    effective_looks_total: float = 0.0
    max_operator_error: float | None = None
    max_variance_error: float | None = None
    min_psd_eigenvalue: float | None = None
    interval_score_total: float = 0.0
    interval_width_total: float = 0.0
    statuses: dict[str, int] = field(default_factory=lambda: {status: 0 for status in ATTEMPT_STATUSES})
    field_digests: dict[str, Any] = field(default_factory=lambda: {name: hashlib.sha256() for name in ("operator_hash", "variance_hash", "emission_hash", "truth_sha256")})
    request_digest: Any = field(default_factory=lambda: hashlib.sha256(b"dolphinrust:spatial-covariance:requests:v4\0"))
    attempt_digest: Any = field(default_factory=lambda: hashlib.sha256(b"dolphinrust:spatial-covariance:attempts:v4\0"))

    def __post_init__(self) -> None:
        if self.expected_seed_count is None:
            self.expected_seed_count = expected_seed_count(self.cell_id)

    def add(self, attempt: Any) -> None:
        if not isinstance(attempt, dict) or set(attempt) != ATTEMPT_KEYS:
            raise SchemaError(f"cell {self.cell_id} has malformed or unknown per-attempt evidence")
        if attempt.get("schema") != "dolphinrust.spatial-covariance.attempt-evidence/4":
            raise SchemaError(f"cell {self.cell_id} has the wrong attempt schema")
        if attempt.get("cell_id") != self.cell_id or not _integer(attempt.get("cell_ordinal")) or attempt.get("cell_ordinal") != self.cell_ordinal:
            raise SchemaError(f"cell {self.cell_id} has an out-of-order cell identity")
        if not _integer(attempt.get("seed_index")) or attempt.get("seed_index") != self.next_seed_index or self.next_seed_index >= self.expected_seed_count:
            raise SchemaError(f"cell {self.cell_id} has a missing, duplicate, top-up, or out-of-order seed")
        if attempt.get("seed_sha256") != _expected_seed_hash(self.preregistration, self.cell_id, self.next_seed_index):
            raise SchemaError(f"cell {self.cell_id} has a seed derivation mismatch")
        if any(not _is_sha256(attempt.get(field_name)) for field_name in ATTEMPT_HASH_FIELDS):
            raise SchemaError(f"cell {self.cell_id} has an invalid identity hash")
        self._validate_scope(attempt)
        request = {
            "schema": "dolphinrust.spatial-covariance.attempt/4",
            "cell_id": self.cell_id,
            "cell_ordinal": self.cell_ordinal,
            "seed_index": self.next_seed_index,
            "seed_sha256": _expected_seed_hash(self.preregistration, self.cell_id, self.next_seed_index),
            **dict(zip(DIMENSION_NAMES, self.cell_id.split("|"))),
        }
        _update_field_digest(self.request_digest, request)
        _update_field_digest(self.attempt_digest, attempt)
        self._accumulate(attempt)
        self.next_seed_index += 1

    def _validate_scope(self, attempt: Mapping[str, Any]) -> None:
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        target, reference = _expected_coordinates(self.preregistration, self.cell_id)
        generator = self.preregistration["generator"]
        topology = generator["acquisition"]["topologies"][labels["block_topology"]]
        if attempt.get("generator_hash") != sha256_json(generator) or attempt.get("config_hash") != sha256_json(generator):
            raise SchemaError(f"cell {self.cell_id} has a generator/config identity mismatch")
        if attempt.get("source_model_hash") != sha256_json(generator["source_centered_empirical"]):
            raise SchemaError(f"cell {self.cell_id} has a source-model identity mismatch")
        if attempt.get("date_axis_sha256") != sha256_json(topology["date_axis"]):
            raise SchemaError(f"cell {self.cell_id} has a date-axis identity mismatch")
        if any(
            not isinstance(attempt.get(field_name), list)
            or len(attempt[field_name]) != 2
            or any(not _integer(value) for value in attempt[field_name])
            for field_name in ("target_coordinate", "reference_coordinate")
        ) or attempt.get("target_coordinate") != target or attempt.get("reference_coordinate") != reference:
            raise SchemaError(f"cell {self.cell_id} has a coordinate identity mismatch")
        self._validate_regenerated_inputs(attempt)
        status = attempt.get("status")
        if status not in ATTEMPT_STATUSES or not isinstance(attempt.get("emitted"), bool) or not isinstance(attempt.get("factor_emitted"), bool):
            raise SchemaError(f"cell {self.cell_id} has invalid status/emission flags")
        if not _number(attempt.get("effective_looks_fraction")) or attempt["effective_looks_fraction"] <= 0 or attempt.get("effective_looks_application") != "source_factor_divided_by_sqrt_fraction":
            raise SchemaError(f"cell {self.cell_id} has an invalid effective-look realization")
        if labels["position"] == "masked":
            self._validate_masked(attempt)
        elif status == "masked_target":
            raise SchemaError(f"cell {self.cell_id} cannot use masked_target")
        elif status == "tied_eigenvalue" and labels["eigen_stress"] != "tied_eigenvalue":
            raise SchemaError(f"cell {self.cell_id} has an undeclared not-evaluable attempt")
        elif attempt.get("factor_emitted") != attempt.get("emitted"):
            raise SchemaError(f"cell {self.cell_id} has inconsistent factor/emission flags")

    def _validate_regenerated_inputs(self, attempt: Mapping[str, Any]) -> None:
        expected = regenerate_frozen_attempt_inputs(
            self.preregistration, self.cell_id, self.next_seed_index
        )
        raw_fields = (
            "raw_input_shape", "raw_input_value_count", "raw_input_sha256",
            "target_raw_input_sha256", "reference_raw_input_sha256",
            "target_support_sha256", "reference_support_sha256",
            "sequential_ancestry_sha256", "raw_dgp_identity_sha256",
            "target_source_count", "reference_source_count", "intersection_source_count",
            "union_source_count", "effective_looks_fraction",
        )
        if any(attempt.get(field_name) != expected[field_name] for field_name in raw_fields):
            raise SchemaError(f"cell {self.cell_id} raw DGP does not match deterministic regeneration")
        if attempt.get("truth_sha256") != expected["truth_sha256"]:
            raise SchemaError(f"cell {self.cell_id} frozen truth digest does not match regeneration")
        if attempt.get("status") == "masked_target":
            return
        try:
            supplied_truth = _matrix(attempt.get("truth_matrix"), "truth matrix")
        except SchemaError as exc:
            raise SchemaError(f"cell {self.cell_id} frozen truth matrix is invalid") from exc
        expected_truth = np.asarray(expected["truth_matrix"], dtype=np.float64)
        if (
            not np.array_equal(supplied_truth, expected_truth)
            or attempt.get("contrast_weights") != expected["contrast_weights"]
            or attempt.get("truth_value") != expected["truth_value"]
        ):
            raise SchemaError(f"cell {self.cell_id} frozen truth sufficient values were replaced")

    def _validate_masked(self, attempt: Mapping[str, Any]) -> None:
        if attempt.get("status") != "masked_target" or attempt.get("emitted") is not False or attempt.get("factor_emitted") is not False:
            raise SchemaError(f"cell {self.cell_id} masked attempt must abstain")
        metrics = ("operator_relative_error", "contrast_variance_reference", "contrast_variance_relative_error", "psd_min_eigenvalue", "covered_95", "interval_score", "interval_width", "signed_cross_influence", "operator_matrix", "truth_matrix", "contrast_weights", "estimate_value", "truth_value")
        if any(attempt.get(metric) is not None for metric in metrics):
            raise SchemaError(f"cell {self.cell_id} masked attempt must use null numeric evidence")

    def _accumulate(self, attempt: Mapping[str, Any]) -> None:
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        overlap = realized_overlap_jaccard(attempt["target_source_count"], attempt["reference_source_count"], attempt["intersection_source_count"], attempt["union_source_count"])
        if not _number(attempt.get("realized_overlap_jaccard")) or not math.isclose(attempt["realized_overlap_jaccard"], overlap, abs_tol=1e-15, rel_tol=0.0):
            raise SchemaError(f"cell {self.cell_id} has inconsistent realized overlap")
        if labels["pair_geometry"] == "coincident" and (overlap != 1.0 or attempt["target_support_sha256"] != attempt["reference_support_sha256"]):
            raise SchemaError(f"cell {self.cell_id} coincident support mismatch")
        if labels["pair_geometry"] in DISJOINT_GEOMETRIES and overlap != 0.0:
            raise SchemaError(f"cell {self.cell_id} disjoint support overlaps")
        if attempt.get("signed_influence_sign") != PAIR_SIGN[labels["pair_geometry"]]:
            raise SchemaError(f"cell {self.cell_id} has a signed-influence mismatch")
        influence = attempt.get("signed_cross_influence")
        expected_sign = PAIR_SIGN[labels["pair_geometry"]]
        if labels["position"] != "masked" and expected_sign in {"positive", "negative"} and (not _number(influence) or (influence > 0) != (expected_sign == "positive")):
            raise SchemaError(f"cell {self.cell_id} has a signed cross-influence mismatch")
        if expected_sign in {"zero", "none"} and labels["position"] != "masked" and influence != 0.0:
            raise SchemaError(f"cell {self.cell_id} zero/disjoint influence must be exactly zero")
        self.statuses[attempt["status"]] += 1
        self.emitted += int(attempt["emitted"])
        self.target_total += attempt["target_source_count"]
        self.reference_total += attempt["reference_source_count"]
        self.intersection_total += attempt["intersection_source_count"]
        self.union_total += attempt["union_source_count"]
        self.overlap_total += overlap
        self.effective_looks_total += attempt["effective_looks_fraction"]
        if attempt["status"] in {"valid", "tied_eigenvalue"}:
            self._validate_metric_evidence(attempt)
        if attempt["emitted"] and attempt["status"] == "valid":
            self._accumulate_valid_metrics(attempt)
        for field_name, digest in self.field_digests.items():
            _update_field_digest(digest, attempt[field_name])

    def _validate_metric_evidence(self, attempt: Mapping[str, Any]) -> None:
        recomputed = independently_recompute_metrics(attempt)
        for field_name, value in recomputed.items():
            if field_name.endswith("hash") or field_name == "truth_sha256":
                if attempt.get(field_name) != value:
                    raise SchemaError(f"cell {self.cell_id} has a {field_name} canonical digest mismatch")
            elif not _same_metric(attempt.get(field_name), value):
                raise SchemaError(f"cell {self.cell_id} has a fabricated or drifted {field_name}")
        if not _number(attempt.get("operator_relative_error")) or attempt["operator_relative_error"] < 0 or not _number(attempt.get("psd_min_eigenvalue")) or not isinstance(attempt.get("covered_95"), bool):
            raise SchemaError(f"cell {self.cell_id} nonmasked attempt lacks numeric gate evidence")
        if not _number(attempt.get("interval_score")) or not _number(attempt.get("interval_width")) or attempt["interval_width"] < 0:
            raise SchemaError(f"cell {self.cell_id} nonmasked attempt lacks interval evidence")
        variance_reference = attempt.get("contrast_variance_reference")
        variance_error = attempt.get("contrast_variance_relative_error")
        if not _number(variance_reference) or variance_reference < 0:
            raise SchemaError(f"cell {self.cell_id} has invalid contrast-variance reference")
        weak_zero = variance_reference <= self.preregistration["thresholds"]["weak_zero_variance_max"]
        if weak_zero and variance_error is not None:
            raise SchemaError(f"cell {self.cell_id} weak-zero variance attempt must use null relative error")
        if not weak_zero:
            if not _number(variance_error) or variance_error < 0:
                raise SchemaError(f"cell {self.cell_id} evaluable variance attempt lacks relative error")

    def _accumulate_valid_metrics(self, attempt: Mapping[str, Any]) -> None:
        self.max_operator_error = max(self.max_operator_error if self.max_operator_error is not None else -math.inf, attempt["operator_relative_error"])
        self.min_psd_eigenvalue = min(self.min_psd_eigenvalue if self.min_psd_eigenvalue is not None else math.inf, attempt["psd_min_eigenvalue"])
        self.covered += int(attempt["covered_95"])
        self.coverage_denominator += 1
        self.interval_score_total += attempt["interval_score"]
        self.interval_width_total += attempt["interval_width"]
        variance_reference = attempt["contrast_variance_reference"]
        variance_error = attempt["contrast_variance_relative_error"]
        weak_zero = variance_reference <= self.preregistration["thresholds"]["weak_zero_variance_max"]
        if not weak_zero:
            self.max_variance_error = max(self.max_variance_error if self.max_variance_error is not None else -math.inf, variance_error)

    def finalize(self) -> dict[str, Any]:
        if self.next_seed_index != self.expected_seed_count:
            raise SchemaError(f"cell {self.cell_id} is missing one or more seed indices")
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        if labels["position"] == "masked":
            if self.statuses["masked_target"] != self.expected_seed_count:
                raise SchemaError(f"cell {self.cell_id} masked status count drifted")
            status = PASS
        elif labels["eigen_stress"] == "tied_eigenvalue":
            if self.statuses["tied_eigenvalue"] != self.expected_seed_count:
                raise SchemaError(f"cell {self.cell_id} tied-eigen cell is not completely not-evaluable")
            status = NOT_EVALUABLE
        else:
            if self.statuses["valid"] != self.expected_seed_count:
                raise SchemaError(f"cell {self.cell_id} has an unsupported attempt status")
            status = self._numeric_status(labels)
        coverage = self.covered / self.coverage_denominator if self.coverage_denominator else None
        return {
            "schema": "dolphinrust.spatial-covariance.cell-summary/4",
            "cell_id": self.cell_id, "cell_ordinal": self.cell_ordinal, "status": status,
            "attempted_seeds": self.expected_seed_count, "emitted_seeds": self.emitted,
            "status_histogram": dict(self.statuses), "failure_histogram": {},
            "request_digest": self.request_digest.hexdigest(), "attempt_digest": self.attempt_digest.hexdigest(),
            "target_source_count_total": self.target_total, "reference_source_count_total": self.reference_total,
            "intersection_source_count_total": self.intersection_total, "union_source_count_total": self.union_total,
            "realized_overlap_jaccard_mean": self.overlap_total / self.expected_seed_count,
            "effective_looks_fraction": self.effective_looks_total / self.expected_seed_count,
            "operator_relative_error": self.max_operator_error, "contrast_variance_relative_error": self.max_variance_error,
            "variance_evaluable": self.max_variance_error is not None, "psd_min_eigenvalue": self.min_psd_eigenvalue,
            "coverage_95": coverage,
            "interval_score_mean": self.interval_score_total / self.coverage_denominator if self.coverage_denominator else None,
            "interval_width_mean": self.interval_width_total / self.coverage_denominator if self.coverage_denominator else None,
            "operator_digest": self.field_digests["operator_hash"].hexdigest(),
            "variance_digest": self.field_digests["variance_hash"].hexdigest(),
            "emission_digest": self.field_digests["emission_hash"].hexdigest(),
            "truth_digest": self.field_digests["truth_sha256"].hexdigest(),
            "code_sha256": self.code_sha256, "binary_sha256": self.binary_sha256,
            "preregistration_sha256": preregistration_digest(self.preregistration),
        }

    def _numeric_status(self, labels: Mapping[str, str]) -> str:
        if self.max_operator_error is None or self.min_psd_eigenvalue is None or self.coverage_denominator != self.emitted:
            return FAIL
        thresholds = self.preregistration["thresholds"]
        deterministic = self.expected_seed_count == FROZEN_DETERMINISTIC_SEED_COUNT
        operator_limit = thresholds["deterministic_operator_relative_error_max"] if deterministic else thresholds["stochastic_operator_relative_error_max"]
        coverage = self.covered / self.coverage_denominator if self.coverage_denominator else math.nan
        passes = (
            self.max_operator_error <= operator_limit
            and (self.max_variance_error is None or self.max_variance_error <= thresholds["contrast_variance_relative_error_max"])
            and self.min_psd_eigenvalue >= thresholds["psd_min_eigenvalue_min"]
            and (deterministic or abs(coverage - thresholds["coverage_probability"]) <= thresholds["coverage_absolute_error_max"])
            and self.emitted / self.expected_seed_count >= thresholds["emission_rate_min"]
        )
        return PASS if passes else FAIL


def _read_json_line(handle: BinaryIO, path: Path, line_number: int) -> tuple[dict[str, Any] | None, bytes]:
    raw = handle.readline(FROZEN_MAX_RECORD_BYTES + 2)
    if not raw:
        return None, b""
    if len(raw) > FROZEN_MAX_RECORD_BYTES or not raw.endswith(b"\n"):
        raise SchemaError(f"{path}:{line_number} exceeds the frozen record cap or lacks newline framing")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"{path}:{line_number} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{path}:{line_number} is not an object")
    return value, raw


def validate_shard_manifest(preregistration: Mapping[str, Any], manifest: Any, spec: ShardSpec) -> None:
    if not isinstance(manifest, dict) or set(manifest) != SHARD_MANIFEST_KEYS:
        raise SchemaError(f"shard {spec.index} manifest has unknown or missing fields")
    expected = {
        "schema": "dolphinrust.spatial-covariance.shard-manifest", "schema_version": 3,
        "shard_index": spec.index, "cell_ordinal_start": spec.cell_ordinal_start,
        "cell_ordinal_end_exclusive": spec.cell_ordinal_end_exclusive, "expected_cells": len(spec.cell_ids),
        "expected_attempts": spec.expected_attempts, "preregistration_sha256": preregistration_digest(preregistration),
        "generator_protocol_sha256": sha256_json(preregistration["execution_protocol"]), "committed": True,
    }
    integer_fields = (
        "schema_version", "shard_index", "cell_ordinal_start", "cell_ordinal_end_exclusive",
        "expected_cells", "expected_attempts", "input_bytes", "output_bytes", "input_records",
        "output_records", "peak_rss_bytes",
    )
    if any(not _integer(manifest.get(field_name)) for field_name in integer_fields) or manifest.get("committed") is not True:
        raise SchemaError(f"shard {spec.index} manifest has an invalid integer identity")
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SchemaError(f"shard {spec.index} manifest scope/order/count drifted")
    for field_name in ("input_sha256", "output_sha256", "code_sha256", "binary_sha256", "generator_protocol_sha256"):
        if not _is_sha256(manifest.get(field_name)):
            raise SchemaError(f"shard {spec.index} manifest has an invalid {field_name}")
    if manifest.get("input_records") != spec.expected_attempts or manifest.get("output_records") != spec.expected_attempts:
        raise SchemaError(f"shard {spec.index} violates one-input-one-output")
    for field_name in ("input_bytes", "output_bytes"):
        value = manifest.get(field_name)
        if not _integer(value) or value < 0 or value > FROZEN_MAX_SHARD_BYTES:
            raise SchemaError(f"shard {spec.index} exceeds the uncompressed byte cap")
    if not _number(manifest.get("elapsed_seconds")) or manifest["elapsed_seconds"] < 0:
        raise SchemaError(f"shard {spec.index} has invalid elapsed time")
    if type(manifest.get("peak_rss_bytes")) is not int or manifest["peak_rss_bytes"] < 0 or manifest["peak_rss_bytes"] > FROZEN_PROCESS_RSS_BYTES:
        raise SchemaError(f"shard {spec.index} exceeds the process RSS cap")
    if any(
        not isinstance(manifest[field_name], str)
        or Path(manifest[field_name]).is_absolute()
        or ".." in Path(manifest[field_name]).parts
        or Path(manifest[field_name]).name.endswith(preregistration["execution_protocol"]["partial_suffix"])
        for field_name in ("input_path", "output_path")
    ):
        raise SchemaError(f"shard {spec.index} manifest path escapes the run root")


def validate_input_shard(preregistration: Mapping[str, Any], input_path: Path, manifest: Mapping[str, Any], spec: ShardSpec) -> None:
    digest = hashlib.sha256()
    byte_count = 0
    line_number = 0
    with Path(input_path).open("rb") as handle:
        for cell_offset, cell_id in enumerate(spec.cell_ids):
            dimensions = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
            for seed_index in range(FROZEN_SEED_COUNT):
                line_number += 1
                request, raw = _read_json_line(handle, Path(input_path), line_number)
                if request is None:
                    raise SchemaError(f"shard {spec.index} input is missing request {line_number}")
                expected = {
                    "schema": "dolphinrust.spatial-covariance.attempt/3",
                    "cell_id": cell_id,
                    "cell_ordinal": spec.cell_ordinal_start + cell_offset,
                    "seed_index": seed_index,
                    "seed_sha256": _expected_seed_hash(preregistration, cell_id, seed_index),
                    **dimensions,
                }
                if set(request) != INPUT_KEYS or not _integer(request.get("cell_ordinal")) or not _integer(request.get("seed_index")) or request != expected:
                    raise SchemaError(f"shard {spec.index} input has malformed, duplicate, missing, or out-of-order identity")
                byte_count += len(raw)
                if byte_count > FROZEN_MAX_SHARD_BYTES:
                    raise SchemaError(f"shard {spec.index} input exceeds the uncompressed byte cap")
                digest.update(raw)
        extra, _ = _read_json_line(handle, Path(input_path), line_number + 1)
        if extra is not None:
            raise SchemaError(f"shard {spec.index} input contains top-up records")
    if byte_count != manifest["input_bytes"] or digest.hexdigest() != manifest["input_sha256"]:
        raise SchemaError(f"shard {spec.index} input hash/byte count mismatch")


def score_attempt_shard(preregistration: Mapping[str, Any], run_root: Path, manifest: Mapping[str, Any], spec: ShardSpec) -> list[dict[str, Any]]:
    validate_shard_manifest(preregistration, manifest, spec)
    input_path = resolve_below_run_root(run_root, manifest["input_path"], f"shard {spec.index} input path")
    output_path = resolve_below_run_root(run_root, manifest["output_path"], f"shard {spec.index} output path")
    validate_input_shard(preregistration, input_path, manifest, spec)
    if output_path.name.endswith(preregistration["execution_protocol"]["partial_suffix"]):
        raise SchemaError(f"shard {spec.index} references a partial output")
    digest = hashlib.sha256()
    summaries: list[dict[str, Any]] = []
    byte_count = 0
    line_number = 0
    with output_path.open("rb") as handle:
        for cell_offset, cell_id in enumerate(spec.cell_ids):
            accumulator = CellAccumulator(preregistration, cell_id, spec.cell_ordinal_start + cell_offset)
            for _ in range(FROZEN_SEED_COUNT):
                line_number += 1
                attempt, raw = _read_json_line(handle, output_path, line_number)
                if attempt is None:
                    raise SchemaError(f"shard {spec.index} output is missing attempt {line_number}")
                byte_count += len(raw)
                if byte_count > FROZEN_MAX_SHARD_BYTES:
                    raise SchemaError(f"shard {spec.index} output exceeds the uncompressed byte cap")
                digest.update(raw)
                accumulator.add(attempt)
            summaries.append(accumulator.finalize())
        extra, _ = _read_json_line(handle, output_path, line_number + 1)
        if extra is not None:
            raise SchemaError(f"shard {spec.index} output contains top-up records")
    if byte_count != manifest["output_bytes"] or digest.hexdigest() != manifest["output_sha256"]:
        raise SchemaError(f"shard {spec.index} output hash/byte count mismatch")
    return summaries


def result_root_sha256(manifest_digests: Iterable[str]) -> str:
    digest = hashlib.sha256(b"dolphinrust.spatial-covariance.result-root/3\0")
    count = 0
    for index, manifest_digest in enumerate(manifest_digests):
        if not _is_sha256(manifest_digest):
            raise SchemaError("result root contains an invalid shard-manifest digest")
        digest.update(index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(manifest_digest))
        count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def validate_cell_summary(
    preregistration: Mapping[str, Any],
    summary: Any,
    cell_id: str,
    cell_ordinal: int,
    code_sha256: str,
    binary_sha256: str,
) -> None:
    seed_count = expected_seed_count(cell_id)
    if not isinstance(summary, dict) or set(summary) != CELL_SUMMARY_KEYS:
        raise SchemaError(f"cell {cell_id} compact summary has unknown or missing fields")
    if len(_canonical_bytes(summary)) + 1 > FROZEN_MAX_CELL_SUMMARY_BYTES:
        raise SchemaError(f"cell {cell_id} compact summary exceeds its retained byte cap")
    expected = {
        "schema": "dolphinrust.spatial-covariance.cell-summary/4",
        "cell_id": cell_id,
        "cell_ordinal": cell_ordinal,
        "attempted_seeds": seed_count,
        "code_sha256": code_sha256,
        "binary_sha256": binary_sha256,
        "preregistration_sha256": preregistration_digest(preregistration),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise SchemaError(f"cell {cell_id} compact summary scope/count drifted")
    if any(not _is_sha256(summary.get(name)) for name in ("request_digest", "attempt_digest", "operator_digest", "variance_digest", "truth_digest", "emission_digest", "code_sha256", "binary_sha256", "preregistration_sha256")):
        raise SchemaError(f"cell {cell_id} compact summary has an invalid digest")
    statuses = summary.get("status_histogram")
    if not isinstance(statuses, dict) or set(statuses) != ATTEMPT_STATUSES or any(not _integer(value) or value < 0 for value in statuses.values()) or sum(statuses.values()) != seed_count:
        raise SchemaError(f"cell {cell_id} compact status histogram is incomplete")
    if not isinstance(summary.get("failure_histogram"), dict) or any(not isinstance(key, str) or not _integer(value) or value < 0 for key, value in summary["failure_histogram"].items()):
        raise SchemaError(f"cell {cell_id} compact failure histogram is malformed")
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    expected_status = PASS
    if labels["eigen_stress"] == "tied_eigenvalue" and labels["position"] != "masked":
        expected_status = NOT_EVALUABLE
    if labels["position"] == "masked":
        valid_histogram = statuses["masked_target"] == seed_count and summary["emitted_seeds"] == 0
    elif expected_status == NOT_EVALUABLE:
        valid_histogram = statuses["tied_eigenvalue"] == seed_count
    else:
        valid_histogram = statuses["valid"] == seed_count
        deterministic = seed_count == FROZEN_DETERMINISTIC_SEED_COUNT
        operator_limit = preregistration["thresholds"]["deterministic_operator_relative_error_max" if deterministic else "stochastic_operator_relative_error_max"]
        coverage = summary.get("coverage_95")
        passes = (
            _number(summary.get("operator_relative_error")) and summary["operator_relative_error"] <= operator_limit
            and (summary.get("contrast_variance_relative_error") is None or (_number(summary["contrast_variance_relative_error"]) and summary["contrast_variance_relative_error"] <= preregistration["thresholds"]["contrast_variance_relative_error_max"]))
            and _number(summary.get("psd_min_eigenvalue")) and summary["psd_min_eigenvalue"] >= preregistration["thresholds"]["psd_min_eigenvalue_min"]
            and _number(coverage) and (deterministic or abs(coverage - preregistration["thresholds"]["coverage_probability"]) <= preregistration["thresholds"]["coverage_absolute_error_max"])
            and summary["emitted_seeds"] / seed_count >= preregistration["thresholds"]["emission_rate_min"]
        )
        expected_status = PASS if passes else FAIL
    if not valid_histogram or summary.get("status") != expected_status:
        raise SchemaError(f"cell {cell_id} compact status contradicts its independently reduced evidence")


def validate_shard_manifest(preregistration: Mapping[str, Any], manifest: Any, spec: ShardSpec) -> None:
    if not isinstance(manifest, dict) or set(manifest) != SHARD_MANIFEST_KEYS:
        raise SchemaError(f"shard {spec.index} manifest has unknown or missing fields")
    expected = {
        "schema": "dolphinrust.spatial-covariance.shard-manifest/4",
        "schema_version": 4,
        "shard_index": spec.index,
        "cell_ordinal_start": spec.cell_ordinal_start,
        "cell_ordinal_end_exclusive": spec.cell_ordinal_end_exclusive,
        "expected_cells": len(spec.cell_ids),
        "expected_attempts": spec.expected_attempts,
        "summary_records": len(spec.cell_ids),
        "preregistration_sha256": preregistration_digest(preregistration),
        "generator_protocol_sha256": sha256_json(preregistration["execution_protocol"]),
        "committed": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SchemaError(f"shard {spec.index} compact scope/order/count drifted")
    if any(not _is_sha256(manifest.get(name)) for name in ("summary_sha256", "code_sha256", "binary_sha256", "preregistration_sha256", "generator_protocol_sha256")):
        raise SchemaError(f"shard {spec.index} compact manifest has an invalid digest")
    if not _integer(manifest.get("summary_bytes")) or not 0 <= manifest["summary_bytes"] <= FROZEN_MAX_SHARD_BYTES:
        raise SchemaError(f"shard {spec.index} compact summaries exceed the retained cap")
    if not _number(manifest.get("elapsed_seconds")) or manifest["elapsed_seconds"] < 0 or type(manifest.get("peak_rss_bytes")) is not int or not 0 <= manifest["peak_rss_bytes"] <= FROZEN_PROCESS_RSS_BYTES:
        raise SchemaError(f"shard {spec.index} has invalid resource evidence")
    path = manifest.get("summary_path")
    if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts or Path(path).name.endswith(".partial"):
        raise SchemaError(f"shard {spec.index} summary path escapes the run root")


def score_attempt_shard(preregistration: Mapping[str, Any], run_root: Path, manifest: Mapping[str, Any], spec: ShardSpec) -> list[dict[str, Any]]:
    validate_shard_manifest(preregistration, manifest, spec)
    directory = resolve_below_run_root(run_root, manifest["summary_path"], f"shard {spec.index} summary path")
    if not directory.is_dir():
        raise SchemaError(f"shard {spec.index} compact summary path is not a directory")
    digest = hashlib.sha256(b"dolphinrust:spatial-covariance:cell-summary-root:v4\0")
    byte_count = 0
    summaries: list[dict[str, Any]] = []
    for offset, cell_id in enumerate(spec.cell_ids):
        path = directory / f"cell-{spec.cell_ordinal_start + offset:05d}.jsonl"
        if path.is_symlink():
            raise SchemaError(f"shard {spec.index} cell summary must not be a symlink")
        if path.stat().st_size > FROZEN_MAX_CELL_SUMMARY_BYTES:
            raise SchemaError(f"shard {spec.index} cell summary exceeds its cap before read")
        with path.open("rb") as handle:
            summary, raw = _read_json_line(handle, path, 1)
            if summary is None:
                raise SchemaError(f"shard {spec.index} is missing compact cell summary {offset}")
            extra, _ = _read_json_line(handle, path, 2)
            if extra is not None:
                raise SchemaError(f"shard {spec.index} contains duplicate compact summary {offset}")
            validate_cell_summary(preregistration, summary, cell_id, spec.cell_ordinal_start + offset, manifest["code_sha256"], manifest["binary_sha256"])
        item_digest = hashlib.sha256(raw).digest()
        digest.update(offset.to_bytes(8, "big"))
        digest.update(item_digest)
        byte_count += len(raw)
        summaries.append(summary)
    if digest.hexdigest() != manifest["summary_sha256"] or byte_count != manifest["summary_bytes"]:
        raise SchemaError(f"shard {spec.index} compact summary hash/bytes mismatch")
    return summaries


def result_root_sha256(manifest_digests: Iterable[str]) -> str:
    digest = hashlib.sha256(b"dolphinrust.spatial-covariance.result-root/4\0")
    count = 0
    for index, manifest_digest in enumerate(manifest_digests):
        if not _is_sha256(manifest_digest):
            raise SchemaError("result root contains an invalid shard-manifest digest")
        digest.update(index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(manifest_digest))
        count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def _validate_performance_probe(preregistration: Mapping[str, Any], probe: Any, code_sha256: str, binary_sha256: str) -> None:
    required = {"schema", "schema_version", "outcomes_persisted", "seed_counts", "cell_classes", "measurements", "attempts_per_second", "peak_rss_bytes", "target_wall_seconds", "reserve_fraction", "projected_serial_seconds", "derived_concurrency", "code_sha256", "binary_sha256", "config_sha256"}
    if not isinstance(probe, dict) or set(probe) != required:
        raise SchemaError("performance probe receipt has unknown or missing fields")
    frozen = preregistration["execution_protocol"]["performance_probe"]
    if probe["schema"] != "dolphinrust.spatial-covariance.performance-probe" or not _integer(probe["schema_version"]) or probe["schema_version"] != 1 or probe["outcomes_persisted"] is not False:
        raise SchemaError("performance probe must be outcome-discarding")
    if not isinstance(probe["seed_counts"], list) or any(not _integer(value) for value in probe["seed_counts"]) or probe["seed_counts"] != frozen["seed_counts"] or probe["cell_classes"] != frozen["required_cell_classes"]:
        raise SchemaError("performance probe does not cover the frozen classes/seeds")
    if probe["code_sha256"] != code_sha256 or probe["binary_sha256"] != binary_sha256 or probe["config_sha256"] != sha256_json(preregistration["generator"]):
        raise SchemaError("performance probe scope identity mismatch")
    expected_pairs = [(cell_class, seed_count) for cell_class in frozen["required_cell_classes"] for seed_count in frozen["seed_counts"]]
    measurements = probe["measurements"]
    if not isinstance(measurements, list) or len(measurements) != len(expected_pairs):
        raise SchemaError("performance probe measurements do not cover the frozen classes/seeds")
    total_attempts = 0
    total_elapsed = 0.0
    measured_peak_rss = 0
    for measurement, (cell_class, seed_count) in zip(measurements, expected_pairs):
        if not isinstance(measurement, dict) or set(measurement) != PERFORMANCE_MEASUREMENT_KEYS:
            raise SchemaError("performance probe measurement has unknown or missing fields")
        if not _integer(measurement["seed_count"]) or not _integer(measurement["attempt_count"]) or measurement["cell_class"] != cell_class or measurement["seed_count"] != seed_count or measurement["attempt_count"] != seed_count:
            raise SchemaError("performance probe measurement order/count drifted")
        if measurement["outcomes_persisted"] is not False or not _number(measurement["elapsed_seconds"]) or measurement["elapsed_seconds"] <= 0:
            raise SchemaError("performance probe measurement is not outcome-free with positive timing")
        if type(measurement["peak_rss_bytes"]) is not int or measurement["peak_rss_bytes"] <= 0 or measurement["peak_rss_bytes"] > FROZEN_PROCESS_RSS_BYTES:
            raise SchemaError("performance probe measurement has invalid RSS")
        total_attempts += measurement["attempt_count"]
        total_elapsed += measurement["elapsed_seconds"]
        measured_peak_rss = max(measured_peak_rss, measurement["peak_rss_bytes"])
    measured_rate = total_attempts / total_elapsed
    if not _number(probe.get("attempts_per_second")) or not math.isclose(probe["attempts_per_second"], measured_rate, rel_tol=1e-12, abs_tol=1e-12):
        raise SchemaError("performance probe rate is not derived from its measurements")
    if probe.get("peak_rss_bytes") != measured_peak_rss:
        raise SchemaError("performance probe peak RSS is not derived from its measurements")
    if any(not _number(probe.get(field_name)) or probe[field_name] <= 0 for field_name in ("attempts_per_second", "target_wall_seconds", "projected_serial_seconds")):
        raise SchemaError("performance probe rates/timing must be finite and positive")
    projected = FROZEN_ATTEMPT_COUNT / probe["attempts_per_second"]
    if not math.isclose(probe["projected_serial_seconds"], projected, rel_tol=1e-12, abs_tol=1e-9):
        raise SchemaError("performance probe serial projection does not match frozen attempt count/rate")
    if probe["reserve_fraction"] != frozen["reserve_fraction"] or not _integer(probe.get("derived_concurrency")) or probe["derived_concurrency"] < 1:
        raise SchemaError("performance probe concurrency receipt is invalid")
    expected = math.ceil(probe["projected_serial_seconds"] / (probe["target_wall_seconds"] * (1.0 - probe["reserve_fraction"])))
    if probe["derived_concurrency"] != expected:
        raise SchemaError("performance probe derived concurrency does not match the frozen formula")
    if type(probe.get("peak_rss_bytes")) is not int or probe["peak_rss_bytes"] < 0 or probe["peak_rss_bytes"] > FROZEN_PROCESS_RSS_BYTES:
        raise SchemaError("performance probe exceeds the process RSS cap")


def _validate_resources(preregistration: Mapping[str, Any], resources: Any, binary_sha256: str) -> list[str]:
    if not isinstance(resources, list):
        raise SchemaError("resources must be a per-resource list")
    by_id = {item.get("resource_id"): item for item in resources if isinstance(item, dict)}
    if set(by_id) != set(FROZEN_RESOURCE_IDS) or len(resources) != len(FROZEN_RESOURCE_IDS):
        raise SchemaError("resource receipts must contain exactly the three frozen resource cells")
    sampling = preregistration["resource_sampling"]
    matrix_by_id = {item["id"]: item for item in preregistration["resource_matrix"]}
    validated: list[tuple[dict[str, Any], int]] = []
    for resource_id in FROZEN_RESOURCE_IDS:
        item = by_id[resource_id]
        if set(item) != RESOURCE_KEYS:
            raise SchemaError(f"resource {resource_id} has unknown or missing fields")
        if item["status"] not in {PASS, FAIL} or type(item["rss_bytes"]) is not int or item["rss_bytes"] < 0:
            raise SchemaError(f"resource {resource_id} has invalid status/RSS")
        if item["growth_class"] not in {"linear", "superlinear"} or item["binary_hash"] != binary_sha256 or item["config_hash"] != sha256_json(preregistration["generator"]):
            raise SchemaError(f"resource {resource_id} identity/growth mismatch")
        provenance = ("os", "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "sampling_interval_ms", "warmup_runs", "measured_repetitions", "tool_versions", "growth_regression", "acceptance")
        if any(not _integer(item[field_name]) for field_name in ("ram_bytes", "sampling_interval_ms", "warmup_runs", "measured_repetitions")):
            raise SchemaError(f"resource {resource_id} has invalid integer sampling identity")
        if any(item[field_name] != sampling[field_name] for field_name in provenance):
            raise SchemaError(f"resource {resource_id} sampling provenance mismatch")
        observations = item["growth_observation"]
        matrix = matrix_by_id[resource_id]
        if not isinstance(observations, list) or len(observations) != sampling["measured_repetitions"]:
            raise SchemaError(f"resource {resource_id} lacks measured growth observations")
        observed_rss = []
        for repetition, observation in enumerate(observations):
            if not isinstance(observation, dict) or set(observation) != RESOURCE_OBSERVATION_KEYS:
                raise SchemaError(f"resource {resource_id} has malformed growth observations")
            if any(not _integer(observation[field_name]) for field_name in ("repetition", "tile_pixels", "date_count")) or observation["repetition"] != repetition or observation["tile_pixels"] != matrix["tile_pixels"] or observation["date_count"] != matrix["dates"]:
                raise SchemaError(f"resource {resource_id} growth observation scope drifted")
            if type(observation["peak_rss_bytes"]) is not int or observation["peak_rss_bytes"] <= 0 or not _number(observation["wall_seconds"]) or observation["wall_seconds"] <= 0:
                raise SchemaError(f"resource {resource_id} growth observation has invalid measurements")
            observed_rss.append(observation["peak_rss_bytes"])
        peak_rss = max(observed_rss)
        if item["rss_bytes"] != peak_rss:
            raise SchemaError(f"resource {resource_id} RSS is not derived from its repetitions")
        expected_hash = sha256_json({key: value for key, value in item.items() if key != "resource_hash"})
        if item["resource_hash"] != expected_hash:
            raise SchemaError(f"resource {resource_id} hash does not bind its receipt")
        validated.append((item, peak_rss))
    x = [math.log(matrix_by_id[resource_id]["tile_pixels"] * matrix_by_id[resource_id]["dates"]) for resource_id in FROZEN_RESOURCE_IDS]
    y = [math.log(peak_rss) for _, peak_rss in validated]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    growth_exponent = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y)) / denominator
    expected_growth_class = "linear" if growth_exponent <= 1.25 else "superlinear"
    statuses = []
    for item, peak_rss in validated:
        expected_status = PASS if peak_rss <= FROZEN_PROCESS_RSS_BYTES and expected_growth_class == "linear" else FAIL
        if item["growth_class"] != expected_growth_class or item["status"] != expected_status:
            raise SchemaError(f"resource {item['resource_id']} status/growth declaration contradicts measured evidence")
        statuses.append(item["status"])
    return statuses


def _growth_exponent(points: list[tuple[int, int]]) -> float:
    x = [math.log(float(scale)) for scale, _ in points]
    y = [math.log(float(rss)) for _, rss in points]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        raise SchemaError("resource growth axis is not identifiable")
    return sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y)) / denominator


def _validate_resources(preregistration: Mapping[str, Any], resources: Any, binary_sha256: str) -> list[str]:
    if not isinstance(resources, list):
        raise SchemaError("resources must be a per-resource list")
    by_id = {item.get("resource_id"): item for item in resources if isinstance(item, dict)}
    if set(by_id) != set(FROZEN_RESOURCE_IDS) or len(resources) != len(FROZEN_RESOURCE_IDS):
        raise SchemaError("resource receipts must contain the five identifiable area/date cells")
    sampling = preregistration["resource_sampling"]
    matrix = {item["id"]: item for item in preregistration["resource_matrix"]}
    peaks: dict[str, int] = {}
    for resource_id in FROZEN_RESOURCE_IDS:
        item = by_id[resource_id]
        if set(item) != RESOURCE_KEYS:
            raise SchemaError(f"resource {resource_id} has unknown or missing fields")
        if item["binary_hash"] != binary_sha256 or item["config_hash"] != sha256_json(preregistration["generator"]):
            raise SchemaError(f"resource {resource_id} identity mismatch")
        provenance = ("os", "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "warmup_runs", "measured_repetitions", "tool_versions", "acceptance")
        if any(item.get(name) != sampling[name] for name in provenance):
            raise SchemaError(f"resource {resource_id} sampling provenance mismatch")
        observations = item.get("growth_observation")
        if not isinstance(observations, list) or len(observations) != sampling["measured_repetitions"]:
            raise SchemaError(f"resource {resource_id} lacks exact repetitions")
        observed: list[int] = []
        for repetition, observation in enumerate(observations):
            if not isinstance(observation, dict) or set(observation) != RESOURCE_OBSERVATION_KEYS:
                raise SchemaError(f"resource {resource_id} has malformed observations")
            expected = matrix[resource_id]
            if observation["repetition"] != repetition or observation["tile_pixels"] != expected["tile_pixels"] or observation["date_count"] != expected["dates"]:
                raise SchemaError(f"resource {resource_id} observation scope drifted")
            raw_measurement = observation.get("raw_measurement")
            expected_command = [
                "cargo", "run", "--release", "-p", "dolphin-workflows", "--example",
                "spatial_covariance_bench", "--", "--tile-pixels", str(expected["tile_pixels"]),
                "--dates", str(expected["dates"]),
            ]
            if (
                not isinstance(raw_measurement, dict)
                or set(raw_measurement) != RESOURCE_RAW_MEASUREMENT_KEYS
                or raw_measurement.get("command") != expected_command
                or type(raw_measurement.get("exit_status")) is not int
                or raw_measurement["exit_status"] != 0
                or raw_measurement.get("wall_seconds") != observation.get("wall_seconds")
                or raw_measurement.get("max_rss_bytes") != observation.get("peak_rss_bytes")
                or any(raw_measurement.get(name) != sampling[name] for name in ("rss_sampler", "rss_field", "os", "hardware_class", "ram_bytes"))
                or not isinstance(raw_measurement.get("tool_versions"), dict)
                or set(raw_measurement["tool_versions"]) != {"rustc", "cargo", "uname"}
                or any(not isinstance(value, str) or not value for value in raw_measurement["tool_versions"].values())
                or len(_canonical_bytes(raw_measurement)) > sampling["max_encoded_raw_measurement_bytes"]
                or observation.get("raw_measurement_sha256") != sha256_json(raw_measurement)
            ):
                raise SchemaError(f"resource {resource_id} raw resource measurement is invalid")
            if type(observation["peak_rss_bytes"]) is not int or observation["peak_rss_bytes"] <= 0 or not _number(observation["wall_seconds"]) or observation["wall_seconds"] <= 0:
                raise SchemaError(f"resource {resource_id} observation is not bound to raw measurements")
            observed.append(observation["peak_rss_bytes"])
        peaks[resource_id] = max(observed)
        if item.get("rss_bytes") != peaks[resource_id] or item.get("resource_hash") != sha256_json({key: value for key, value in item.items() if key != "resource_hash"}):
            raise SchemaError(f"resource {resource_id} aggregate/hash is not derived")
    area_names = ("area_128_dates_26", "area_256_dates_26", "area_512_dates_26")
    date_names = ("area_256_dates_13", "area_256_dates_26", "area_256_dates_52")
    area_exponent = _growth_exponent([(matrix[name]["tile_pixels"], peaks[name]) for name in area_names])
    date_exponent = _growth_exponent([(matrix[name]["dates"], peaks[name]) for name in date_names])
    growth_class = "linear" if max(area_exponent, date_exponent) <= 1.25 else "superlinear"
    statuses: list[str] = []
    for resource_id in FROZEN_RESOURCE_IDS:
        item = by_id[resource_id]
        expected_status = PASS if peaks[resource_id] <= FROZEN_PROCESS_RSS_BYTES and growth_class == "linear" else FAIL
        if not math.isclose(item.get("area_growth_exponent", math.nan), area_exponent, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(item.get("date_growth_exponent", math.nan), date_exponent, rel_tol=1e-12, abs_tol=1e-12) or item.get("growth_class") != growth_class or item.get("status") != expected_status:
            raise SchemaError(f"resource {resource_id} status/growth contradicts identifiable measurements")
        statuses.append(item["status"])
    return statuses


class _CellSummarySink:
    def __init__(
        self,
        destination: Path | None,
        byte_limit: int = FROZEN_CELL_SUMMARY_COMPONENT_BYTES,
    ):
        self.destination = Path(destination) if destination is not None else None
        self.partial = self.destination.with_name(self.destination.name + ".partial") if self.destination is not None else None
        self.handle = None
        self.digest = hashlib.sha256()
        self.byte_count = 0
        self.record_count = 0
        self.byte_limit = byte_limit

    def open(self) -> None:
        if self.destination is None:
            return
        if self.destination.exists() or self.partial.exists():
            raise SchemaError("refusing to overwrite cell-summary state")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.partial.open("xb")

    def add(self, summary: Mapping[str, Any]) -> None:
        if self.handle is None:
            return
        encoded = _canonical_bytes(summary) + b"\n"
        if self.byte_count + len(encoded) > self.byte_limit:
            raise SchemaError("cell-summary JSONL exceeds the frozen full retained cell-summary cap")
        self.handle.write(encoded)
        self.digest.update(encoded)
        self.byte_count += len(encoded)
        self.record_count += 1

    def commit(self) -> dict[str, Any]:
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None
            os.replace(self.partial, self.destination)
        return {"sha256": self.digest.hexdigest(), "bytes": self.byte_count, "records": self.record_count}

    def abort(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.partial is not None:
            self.partial.unlink(missing_ok=True)


def score_run_manifest(preregistration: Mapping[str, Any], manifest_path: Path, cell_summary_path: Path | None = None) -> dict[str, Any]:
    sink = _CellSummarySink(cell_summary_path)
    try:
        validate_preregistration(preregistration)
        manifest_path = Path(manifest_path)
        if manifest_path.name.endswith(preregistration["execution_protocol"]["partial_suffix"]):
            raise SchemaError("partial run manifests are not admissible")
        run_manifest, _ = _read_single_json_record(
            manifest_path,
            preregistration["execution_protocol"]["max_encoded_run_manifest_bytes"],
            "run manifest",
        )
        if not isinstance(run_manifest, dict) or set(run_manifest) != RUN_MANIFEST_KEYS:
            raise SchemaError("run manifest has unknown or missing fields")
        if run_manifest["schema"] != "dolphinrust.spatial-covariance.run-manifest/4" or not _integer(run_manifest["schema_version"]) or run_manifest["schema_version"] != 4:
            raise SchemaError("run manifest must use schema v4")
        if run_manifest["preregistration_sha256"] != preregistration_digest(preregistration):
            raise SchemaError("run manifest preregistration identity mismatch")
        for field_name in ("code_sha256", "binary_sha256"):
            if not _is_sha256(run_manifest[field_name]):
                raise SchemaError(f"run manifest {field_name} is invalid")
        if run_manifest["generator_protocol_sha256"] != sha256_json(preregistration["execution_protocol"]):
            raise SchemaError("run manifest generator protocol identity mismatch")
        _validate_performance_probe(preregistration, run_manifest["performance_probe"], run_manifest["code_sha256"], run_manifest["binary_sha256"])
        resource_statuses = _validate_resources(preregistration, run_manifest["resources"], run_manifest["binary_sha256"])
        sink.open()
        entries = run_manifest["shard_manifests"]
        if not isinstance(entries, list) or len(entries) != FROZEN_SHARD_COUNT:
            raise SchemaError("run manifest must contain exactly 891 ordered shard manifests")
        run_root = manifest_path.resolve(strict=True).parent
        manifest_digests = []
        cell_count = 0
        any_failed = FAIL in resource_statuses
        any_not_evaluable = NOT_EVALUABLE in resource_statuses
        for spec, entry in zip(iter_shard_specs(preregistration), entries):
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise SchemaError(f"shard {spec.index} run-manifest entry is malformed")
            if not isinstance(entry["path"], str):
                raise SchemaError(f"shard {spec.index} manifest path escapes the run root")
            entry_path = Path(entry["path"])
            if entry_path.is_absolute() or ".." in entry_path.parts or entry_path.name.endswith(preregistration["execution_protocol"]["partial_suffix"]):
                raise SchemaError(f"shard {spec.index} manifest path escapes the run root")
            resolved_entry = resolve_below_run_root(run_root, entry["path"], f"shard {spec.index} manifest path")
            digest, _ = sha256_file(
                resolved_entry,
                preregistration["execution_protocol"]["max_encoded_shard_manifest_bytes"],
            )
            if digest != entry["sha256"]:
                raise SchemaError(f"shard {spec.index} manifest hash mismatch")
            shard_manifest, _ = _read_single_json_record(
                resolved_entry,
                preregistration["execution_protocol"]["max_encoded_shard_manifest_bytes"],
                f"shard {spec.index} manifest",
            )
            if shard_manifest.get("code_sha256") != run_manifest["code_sha256"] or shard_manifest.get("binary_sha256") != run_manifest["binary_sha256"]:
                raise SchemaError(f"shard {spec.index} code/binary scope differs from the run manifest")
            for summary in score_attempt_shard(preregistration, run_root, shard_manifest, spec):
                sink.add(summary)
                cell_count += 1
                any_failed = any_failed or summary["status"] == FAIL
                any_not_evaluable = any_not_evaluable or summary["status"] == NOT_EVALUABLE
            manifest_digests.append(digest)
        if result_root_sha256(manifest_digests) != run_manifest["result_root_sha256"]:
            raise SchemaError("run result root does not bind the ordered shard manifests")
        summary_receipt = sink.commit()
        status = FAIL if any_failed else NOT_EVALUABLE if any_not_evaluable else PASS
        return {"status": status, "errors": [], "cell_count": cell_count, "attempt_count": FROZEN_ATTEMPT_COUNT, "cell_summary": summary_receipt}
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        sink.abort()
        return {"status": FAIL, "errors": [str(exc)], "cell_count": 0, "attempt_count": 0}


def score_receipt(preregistration: Mapping[str, Any], receipt: Any) -> Dict[str, Any]:
    """Reject unbound aggregate receipts; v4 requires compact digested cell evidence."""
    try:
        validate_preregistration(preregistration)
    except SchemaError as exc:
        return {"status": FAIL, "errors": [str(exc)]}
    return {"status": FAIL, "errors": ["aggregate receipts are rejected; provide a v4 run manifest with compact digested cell summaries"]}


validate_receipt = score_receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_manifest", type=Path)
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("spatial_covariance_preregistration.json"))
    parser.add_argument("--cell-summary-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_run_manifest(load_preregistration(args.preregistration), args.run_manifest, args.cell_summary_jsonl), indent=2, sort_keys=True))
