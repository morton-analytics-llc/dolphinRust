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
ATTEMPT_STATUSES = {
    "valid", "masked_target", "empty_support", "singular_local_information",
    "nondifferentiable_node",
}
HASH_RE = set("0123456789abcdef")
FROZEN_SEED_COUNT = 5000
FROZEN_DETERMINISTIC_SEED_COUNT = 1
FROZEN_CELL_COUNT = 353
FROZEN_ATTEMPT_COUNT = 665220
FROZEN_MAX_CELLS_PER_SHARD = 100
FROZEN_SHARD_COUNT = 4
FROZEN_MAX_SHARD_BYTES = 100 * 8192
FROZEN_MAX_RECORD_BYTES = 262144
FROZEN_MAX_CELL_SUMMARY_BYTES = 8192
FROZEN_MAX_SHARD_MANIFEST_BYTES = 16384
FROZEN_MAX_RUN_MANIFEST_BYTES = 16777216
FROZEN_MAX_RESOURCE_RECEIPT_BYTES = 1 << 20
FROZEN_CELL_SUMMARY_COMPONENT_BYTES = FROZEN_CELL_COUNT * FROZEN_MAX_CELL_SUMMARY_BYTES
FROZEN_RETAINED_SIZE_BOUND_BYTES = 21831680
FROZEN_PROCESS_RSS_BYTES = 24 << 30
FROZEN_GENERATOR_SHA256 = "b84b60dd769230e7f0f4f34d50ad1167c5064a5797e08d201634fe9299230a29"
FROZEN_SCIENTIFIC_GENERATOR_SHA256 = "0f10a825c66f3b887f660aab9cf97e4ef08675b4a3e927c5ca27a61e65f14020"
FROZEN_EXECUTION_SHA256 = "465b603c7202f4e9ba6ae0de6e4b8ade99020efef9802d7e41705f3ab34c7048"
FROZEN_REDUCERS_SHA256 = "07c75209502a7ac67078e99960edc88d71616308d9d37b50d9bdc41389713007"
FROZEN_MATRIX_SHA256 = "787f8e83bf8841b8cdfe5492e52b12a41dfb7da759e8425e554a3caefe10a528"
FROZEN_RECEIPT_SHA256 = "747658ad422cef7fa97ee2df13c076f2ab931ac2dbc30e6d12c673b26b64bf9e"
FROZEN_HASH_FIELDS_SHA256 = "80eb7ab1c16f9908e39515d7988f2fe5e2fb6b50f81974f0544e6a1abec98d4c"
FROZEN_RESOURCE_SAMPLING_SHA256 = "cd88a410b60c927b9aebb783d1431cde3b6f3396d70b079db64993f17d93b508"
FROZEN_RESOURCE_MATRIX_SHA256 = "2da4e6ab51c72437791b4ae8c225e1df7a4e78da74838dfbade162335e2fdd69"
FROZEN_CELL_POLICY_SHA256 = "602436d3f59dbce016a3a6489ded3d624a8aa5bf42bc6f782b9e787d50bb15d4"
FROZEN_V3_PREREGISTRATION_SHA256 = "d1b29a1dc63a69c952397af1c713e604142b260be7068d37ee0f1a3158b88184"
FROZEN_DETERMINISM_SHA256 = "001a6ec1f4bad33313e5b1fe72c1e97d15b6c8bd001d471f9cc3b2cb8636acf0"
FROZEN_NUMERIC_SHA256 = "89d5b32eadb562188798fe9231c2ea66933edcbcd85cad650473b08a1370e344"
FROZEN_NORMAL_QUANTILE_SHA256 = "35c91380e9b6c7b388ff195391cc1a16700b3c028e397ea0d940374d821b53a4"
FROZEN_DGP_COEFFICIENT_SHA256 = "2c531b7366441ca255065098e4a3c35a4d9220596c2e9a3e7ad4ae1f8a71abb0"
FROZEN_PORTABLE_DGP_TABLE_SHA256 = "2152045c9eaedaa5fe423178a0acee621deb43fae562dae34f9dd195f1afa299"
FROZEN_PORTABLE_DGP_ASSET_BYTES = 3_120_555
FROZEN_PORTABLE_DGP_ASSET_SHA256 = "d2ea499ad4915903a33c0f64ad0b43d09a78aa46b1fcf2659b69ec60c91c9b12"
FROZEN_PORTABLE_DGP_COORDINATE_COUNT = 29_007
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
    "covariance_calibration_relative_error_max": 0.1,
    "psd_min_eigenvalue_min": -1e-10,
    "coverage_probability": 0.95,
    "coverage_absolute_error_max": 0.02,
    "emission_rate_min": 0.99,
    "resource_rss_bytes_max": FROZEN_PROCESS_RSS_BYTES,
    "resource_growth": "area_or_dates_linear; no quadratic area axis",
    "resource_buffer_policy": "no_allocation_component_with_two_area_scaled_axes",
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
    "raw_input_sha256", "latent_history_sha256", "estimate_sha256", "predicted_covariance_sha256", "date_axis_sha256",
    "generator_hash", "config_hash", "source_model_hash", "target_coordinate", "reference_coordinate",
    "target_support_sha256", "reference_support_sha256", "target_source_count", "reference_source_count",
    "intersection_source_count", "union_source_count", "realized_overlap_jaccard", "signed_cross_influence",
    "signed_influence_sign", "effective_looks_fraction", "effective_looks_application",
    "source_correlation_model", "source_correlation_distance_scale_pixels", "estimator_branch",
    "target_estimate_history", "reference_estimate_history", "predicted_difference_covariance",
    "production_operator_matrix", "contrast_weights", "operator_sha256",
    "raw_input_shape", "raw_input_value_count", "target_raw_input_sha256", "reference_raw_input_sha256",
    "sequential_ancestry_sha256", "raw_dgp_identity_sha256",
}
INPUT_KEYS = {"schema", "cell_id", "cell_ordinal", "seed_index", "seed_sha256", *DIMENSION_NAMES}
ATTEMPT_HASH_FIELDS = (
    "seed_sha256", "raw_input_sha256", "latent_history_sha256", "estimate_sha256", "predicted_covariance_sha256",
    "date_axis_sha256", "generator_hash", "config_hash", "source_model_hash", "target_support_sha256",
    "reference_support_sha256", "target_raw_input_sha256", "reference_raw_input_sha256",
    "sequential_ancestry_sha256", "raw_dgp_identity_sha256", "operator_sha256",
)
CELL_SUMMARY_KEYS = {
    "schema", "cell_id", "cell_ordinal", "status", "attempted_seeds", "emitted_seeds",
    "status_histogram", "failure_histogram", "request_digest", "attempt_digest", "estimate_digest",
    "latent_history_digest", "predicted_covariance_digest", "empirical_error_covariance_digest",
    "target_source_count_total", "reference_source_count_total",
    "intersection_source_count_total", "union_source_count_total", "realized_overlap_jaccard_mean",
    "effective_looks_fraction", "covariance_calibration_relative_error", "error_bias_norm",
    "operator_relative_error", "contrast_variance_relative_error",
    "target_reference_error_covariance_trace",
    "target_support_digest", "reference_support_digest",
    "target_predicted_covariance_trace", "reference_predicted_covariance_trace",
    "target_empirical_error_covariance_trace", "reference_empirical_error_covariance_trace",
    "empirical_error_covariance_trace", "predicted_covariance_trace", "psd_min_eigenvalue",
    "coverage_95_by_date", "interval_score_mean_by_date", "interval_width_mean_by_date",
    "final_date_coverage_95", "final_date_interval_score_mean", "final_date_interval_width_mean",
    "code_sha256", "binary_sha256", "preregistration_sha256",
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
    "production_parity_fixture", "production_parity_fixture_sha256", "matched_pair_cohorts",
}
RESOURCE_KEYS = {
    "resource_id", "status", "rss_bytes", "growth_class", "resource_hash", "config_hash", "binary_hash", "os",
    "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "warmup_runs", "measured_repetitions",
    "tool_versions", "growth_observation", "area_growth_exponent", "date_growth_exponent", "acceptance",
    "allocation_model", "allocation_model_sha256", "dependency_cone", "dependency_cone_sha256",
    "microbatch", "microbatch_sha256", "allocation_components",
}
PERFORMANCE_MEASUREMENT_KEYS = {
    "cell_class", "seed_count", "attempt_count", "elapsed_seconds", "peak_rss_bytes", "outcomes_persisted",
}
RESOURCE_OBSERVATION_KEYS = {
    "repetition", "tile_pixels", "date_count", "peak_rss_bytes", "wall_seconds", "raw_measurement", "raw_measurement_sha256",
}
RESOURCE_RAW_MEASUREMENT_KEYS = {
    "command", "exit_status", "wall_seconds", "max_rss_bytes", "rss_sampler", "rss_field", "os",
    "hardware_class", "ram_bytes", "tool_versions", "stdout_bytes", "stdout_sha256", "stdout_json",
}
ALLOCATION_COMPONENT_NAMES = {
    "factor_block", "serialization", "fixed_l2_workspace", "replay_reservation",
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
    path = Path(path)
    raw = _read_bounded_bytes(path, 4 * 1024 * 1024, "preregistration")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("preregistration is malformed JSON") from exc
    if not isinstance(value, dict):
        raise SchemaError("preregistration root must be an object")
    asset = value.get("portable_dgp_asset")
    if not isinstance(asset, dict):
        raise SchemaError("preregistration portable DGP asset reference is missing")
    relative = asset.get("path")
    relative_path = Path(relative) if isinstance(relative, str) else Path()
    if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
        raise SchemaError("portable DGP asset path must be relative to the preregistration")
    asset_raw = _read_bounded_bytes(
        path.parent / relative_path, 4 * 1024 * 1024, "portable DGP asset"
    )
    if len(asset_raw) != asset.get("byte_count") or hashlib.sha256(asset_raw).hexdigest() != asset.get("sha256"):
        raise SchemaError("portable DGP asset differs from its exact byte identity")
    try:
        tables = json.loads(asset_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("portable DGP asset is malformed JSON") from exc
    if not isinstance(tables, dict):
        raise SchemaError("portable DGP asset root must be an object")
    value["_portable_dgp_tables"] = tables
    return value


def _portable_dgp_tables(preregistration: Mapping[str, Any]) -> Mapping[str, Any]:
    tables = preregistration.get("_portable_dgp_tables")
    if not isinstance(tables, dict):
        raise SchemaError("portable DGP asset was not resolved by the bounded loader")
    return tables


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


def _read_hashed_json_record(
    path: Path, byte_limit: int, label: str
) -> tuple[dict[str, Any], bytes, str]:
    value, raw = _read_single_json_record(path, byte_limit, label)
    return value, raw, hashlib.sha256(raw).hexdigest()


def _hash_bounded_file(path: Path, byte_limit: int, label: str) -> tuple[str, int]:
    path = Path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise SchemaError(f"{label} is not a regular file")
    if before.st_size > byte_limit:
        raise SchemaError(f"{label} exceeds its frozen byte cap before read")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _file_identity(opened) != _file_identity(before):
            raise SchemaError(f"{label} changed before it was opened")
        while True:
            chunk = handle.read(min(1 << 20, byte_limit + 1 - byte_count))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > byte_limit:
                raise SchemaError(f"{label} exceeds its frozen byte cap during read")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    path_after = path.stat()
    if (
        _file_identity(after) != _file_identity(before)
        or _file_identity(path_after) != _file_identity(before)
        or byte_count != before.st_size
    ):
        raise SchemaError(f"{label} changed while it was being read")
    return digest.hexdigest(), byte_count


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
    return sha256_json({
        key: value for key, value in preregistration.items() if key != "_portable_dgp_tables"
    })


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


def _normal_quantile_digest(entries: Sequence[Any]) -> str:
    digest = hashlib.sha256(b"dolphinrust:normal-quantile-lut:v1\0")
    digest.update(struct.pack("<I", len(entries)))
    for bits in entries:
        if not isinstance(bits, str) or len(bits) != 16 or set(bits) - HASH_RE:
            return ""
        digest.update(struct.pack("<Q", int(bits, 16)))
    return digest.hexdigest()


def _validate_portable_dgp_tables(preregistration: Mapping[str, Any], errors: List[str]) -> None:
    asset = preregistration.get("portable_dgp_asset")
    expected_asset = {
        "schema": "dolphinrust.spatial-covariance.portable-dgp-asset/1",
        "path": "spatial_covariance_portable_tables.json",
        "byte_count": FROZEN_PORTABLE_DGP_ASSET_BYTES,
        "sha256": FROZEN_PORTABLE_DGP_ASSET_SHA256,
        "normal_quantile_sha256": FROZEN_NORMAL_QUANTILE_SHA256,
        "coefficient_sha256": FROZEN_DGP_COEFFICIENT_SHA256,
        "table_sha256": FROZEN_PORTABLE_DGP_TABLE_SHA256,
    }
    if asset != expected_asset:
        errors.append("portable DGP asset reference differs from the frozen exact-byte identity")
    try:
        tables = _portable_dgp_tables(preregistration)
    except SchemaError:
        errors.append("portable DGP tables are missing")
        return
    normal = tables.get("normal_quantile")
    entries = normal.get("entries", []) if isinstance(normal, dict) else []
    if (
        not isinstance(normal, dict)
        or normal.get("index_bits") != 12
        or len(entries) != 1 << 12
        or normal.get("sha256") != FROZEN_NORMAL_QUANTILE_SHA256
        or _normal_quantile_digest(entries) != FROZEN_NORMAL_QUANTILE_SHA256
    ):
        errors.append("portable normal-quantile table differs from the frozen binary64 LUT")
    coefficients = tables.get("coefficients")
    if (
        not isinstance(coefficients, dict)
        or tables.get("coefficient_sha256") != FROZEN_DGP_COEFFICIENT_SHA256
        or sha256_json(coefficients) != FROZEN_DGP_COEFFICIENT_SHA256
    ):
        errors.append("portable DGP coefficient tables differ from the frozen binary64 tables")
    table_payload = {key: value for key, value in tables.items() if key != "table_sha256"}
    if (
        tables.get("schema") != "dolphinrust.spatial-covariance.portable-dgp-tables/1"
        or tables.get("fused_multiply_add") is not False
        or tables.get("maximum_date_count") != 20
        or tables.get("coordinate_count") != FROZEN_PORTABLE_DGP_COORDINATE_COUNT
        or tables.get("table_sha256") != FROZEN_PORTABLE_DGP_TABLE_SHA256
        or sha256_json(table_payload) != FROZEN_PORTABLE_DGP_TABLE_SHA256
    ):
        errors.append("portable DGP table contract differs from the frozen v1 contract")


def _validate_executable_generator(preregistration: Mapping[str, Any], errors: List[str]) -> None:
    generator = preregistration.get("generator")
    if not isinstance(generator, dict):
        return
    raw = generator.get("raw_proper_complex", {})
    if (
        raw.get("model") != "production_shaped_proper_complex_portable_lut_v3"
        or raw.get("source_shape") != "realized_support_union_by_acquisition"
        or raw.get("component_order") != ["real", "imag"]
        or "AR(1)" not in raw.get("temporal_signal", "")
        or raw.get("pseudo_covariance") != "E[Z Z^T]=0"
    ):
        errors.append("raw generator must define the frozen full proper-complex source-by-acquisition process")
    replay = generator.get("full_replay_dgp", {})
    if (
        replay.get("native_tile_shape") != [256, 256]
        or replay.get("raw_shape") != "complete transitive replay dependency halo by complete topology acquisition count by real/imag"
        or replay.get("dependency_halo_expansions") != "one phase half-window per expected sequential block, including the final source-factor window"
        or "every expected ministack" not in replay.get("support_generation", "")
        or replay.get("model") != "full_production_shaped_raw_replay_v3"
        or "noiseless" not in replay.get("latent_history", "")
        or "EMI or EVD" not in replay.get("rust_output", "")
        or "exact zero" not in replay.get("coincident", "")
    ):
        errors.append("full replay DGP must bind raw shape, direct joint truth, and coincident zero")
    source = generator.get("source_centered_empirical", {})
    if source.get("covariance_definition") != "E[Z Z*]/n" or source.get("mean") != "zero; no sample-mean subtraction" or source.get("floor_application") != "after_zero_mean_covariance_and_shrinkage" or source.get("half_window") != "cell half_window around every realized primitive source, production inward-clamped inside its 256 by 256 native tile":
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
    if supported.get("stable_attempt_statuses") != ["valid", "masked_target", "empty_support", "singular_local_information", "nondifferentiable_node"] or supported.get("not_evaluable_if") != ["empty_support", "singular_local_information"] or supported.get("expected_abstention_if") != ["masked_target", "nondifferentiable_node"]:
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
        or schedule.get("selection_rule") != "one seed for masked, tied, non-interior, coincident, and intrinsically empty KS-ministack cells; 5000 fixed seeds for every otherwise-supported interior cell including disjoint geometries"
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
        + execution.get("max_production_hdf5_bytes", 0)
        + execution.get("max_production_sidecar_bytes", 0)
    )
    if execution.get("retained_attempt_records") is not False or execution.get("request_files_retained") is not False or retained_bound > execution.get("retained_size_bound_bytes", -1) or execution.get("retained_size_bound_bytes") != FROZEN_RETAINED_SIZE_BOUND_BYTES:
        errors.append("v4 retained evidence does not satisfy the frozen compact bound")
    if execution.get("process_rss_bytes_max") != FROZEN_PROCESS_RSS_BYTES:
        errors.append("execution process cap must equal the frozen 24 GiB resource threshold")
    _validate_portable_dgp_tables(preregistration, errors)
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


def portable_table_coverage(preregistration: Mapping[str, Any]) -> dict[str, int]:
    coordinates: set[tuple[int, int]] = set()
    maximum_date_count = 0
    cells = expected_cell_ids(preregistration)
    for cell_id in cells:
        labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
        target, reference = _expected_coordinates(preregistration, cell_id)
        window = preregistration["generator"]["coordinates"]["window_stride"][f"{labels['half_window']}|{labels['stride']}"]
        native_tile_shape = preregistration["generator"]["full_replay_dgp"]["native_tile_shape"]
        topology = preregistration["generator"]["acquisition"]["topologies"][labels["block_topology"]]
        phase_candidates = set(_candidate_support(target, window["half_window"], native_tile_shape))
        phase_candidates.update(_candidate_support(reference, window["half_window"], native_tile_shape))
        dependency_support = phase_candidates
        for _ in topology["expected_blocks"]:
            dependency_support = set(
                _source_factor_support(
                    dependency_support, window["half_window"], native_tile_shape
                )
            )
        coordinates.update(dependency_support)
        maximum_date_count = max(
            maximum_date_count,
            preregistration["generator"]["acquisition"]["topologies"][labels["block_topology"]]["acquisition_count"],
        )
    amplitude_arguments = {row + 3 * column for row, column in coordinates}
    slope_arguments = {2 * row - column for row, column in coordinates}
    coefficients = _portable_dgp_tables(preregistration)["coefficients"]
    if set(map(str, amplitude_arguments)) - set(coefficients["amplitude_scale_bits"]):
        raise SchemaError("portable amplitude table does not cover the frozen coordinates")
    if set(map(str, slope_arguments)) - set(coefficients["phasor_bits"]):
        raise SchemaError("portable phasor table does not cover the frozen coordinates")
    if set(map(str, slope_arguments)) - set(coefficients["latent_phase_bits"]):
        raise SchemaError("portable latent-phase table does not cover the frozen coordinates")
    return {
        "cell_count": len(cells),
        "coordinate_count": len(coordinates),
        "amplitude_argument_count": len(amplitude_arguments),
        "slope_argument_count": len(slope_arguments),
        "date_count": maximum_date_count,
    }


def expected_seed_count(cell_id: str) -> int:
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    deterministic = (
        labels.get("position") != "interior"
        or labels.get("eigen_stress") == "tied_eigenvalue"
        or labels.get("pair_geometry") == "coincident"
        or labels.get("support") == "ks_frozen"
    )
    return FROZEN_DETERMINISTIC_SEED_COUNT if deterministic else FROZEN_SEED_COUNT


def expected_empty_support(cell_id: str) -> bool:
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    return (
        labels.get("support") == "ks_frozen"
        and labels.get("position") != "masked"
        and labels.get("eigen_stress") != "tied_eigenvalue"
    )


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


_CELL_ORDINAL_CACHE: dict[int, tuple[Mapping[str, Any], dict[str, int]]] = {}


def _dgp_generator_identity(preregistration: Mapping[str, Any]) -> str:
    identity = preregistration.get("determinism", {}).get(
        "dgp_generator_identity_sha256"
    )
    if not _is_sha256(identity):
        raise SchemaError("frozen DGP generator identity is invalid")
    return identity


def _dgp_cell_ordinal(preregistration: Mapping[str, Any], cell_id: str) -> int:
    cached = _CELL_ORDINAL_CACHE.get(id(preregistration))
    if cached is None or cached[0] is not preregistration:
        ordinal_by_cell = {
            expected_cell_id: ordinal
            for ordinal, expected_cell_id in enumerate(expected_cell_ids(preregistration))
        }
        cached = (preregistration, ordinal_by_cell)
        _CELL_ORDINAL_CACHE[id(preregistration)] = cached
    ordinal_by_cell = cached[1]
    matched_cell_id = cell_id
    labels = cell_id.split("|")
    if len(labels) == len(DIMENSION_NAMES):
        geometry_index = DIMENSION_NAMES.index("pair_geometry")
        geometry = labels[geometry_index]
        if geometry.endswith("_negative"):
            labels[geometry_index] = geometry.removesuffix("_negative") + "_positive"
            positive_cell_id = "|".join(labels)
            if positive_cell_id in ordinal_by_cell:
                matched_cell_id = positive_cell_id
    try:
        return ordinal_by_cell[matched_cell_id]
    except KeyError as exc:
        raise SchemaError("frozen DGP cell is outside the preregistered matrix") from exc


def portable_dgp_key_sha256(
    preregistration: Mapping[str, Any], cell_ordinal: int, seed_index: int,
    row: int, column: int, date_index: int, stream: str, counter: int = 0,
) -> str:
    integers = (cell_ordinal, seed_index, row, column, date_index, counter)
    if (
        any(not _integer(value) for value in integers)
        or not 0 <= cell_ordinal < 2**64
        or not 0 <= seed_index < 2**64
        or not -(2**63) <= row < 2**63
        or not -(2**63) <= column < 2**63
        or not 0 <= date_index < 2**32
        or not 0 <= counter < 2**64
        or not isinstance(stream, str)
    ):
        raise SchemaError("portable DGP counter key is invalid")
    try:
        stream_bytes = stream.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SchemaError("portable DGP stream must be ASCII") from exc
    if not stream_bytes or len(stream_bytes) >= 2**16:
        raise SchemaError("portable DGP stream length is invalid")
    digest = hashlib.sha256()
    digest.update(b"dolphinrust:spatial-covariance-dgp:v1\0")
    digest.update(bytes.fromhex(_dgp_generator_identity(preregistration)))
    digest.update(struct.pack("<QQqqI", cell_ordinal, seed_index, row, column, date_index))
    digest.update(struct.pack("<H", len(stream_bytes)))
    digest.update(stream_bytes)
    digest.update(struct.pack("<Q", counter))
    return digest.hexdigest()


def _f64_from_bits(bits: str) -> float:
    if not isinstance(bits, str) or len(bits) != 16 or any(value not in "0123456789abcdef" for value in bits):
        raise SchemaError("portable DGP table contains an invalid IEEE-754 bit string")
    return struct.unpack("<d", struct.pack("<Q", int(bits, 16)))[0]


def portable_normal(
    preregistration: Mapping[str, Any], cell_ordinal: int, seed_index: int,
    row: int, column: int, date_index: int, stream: str, counter: int = 0,
) -> float:
    table = _portable_dgp_tables(preregistration)["normal_quantile"]
    digest = bytes.fromhex(portable_dgp_key_sha256(
        preregistration, cell_ordinal, seed_index, row, column, date_index, stream, counter
    ))
    word = int.from_bytes(digest[:8], "little")
    index = word >> (64 - table["index_bits"])
    return _f64_from_bits(table["entries"][index])


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
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, list) for row in value)
        or any(not _number(item) for row in value for item in row)
    ):
        raise SchemaError(f"{label} is not a binary64 matrix")
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
    return float(weights @ matrix @ weights)


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


def _history(value: Any, date_count: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or any(not _number(item) for item in value):
        raise SchemaError(f"{label} is not a binary64 history")
    try:
        history = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{label} is not a binary64 history") from exc
    if history.shape != (date_count,) or not np.isfinite(history).all() or history[0] != 0.0:
        raise SchemaError(f"{label} must be finite, date ordered, and use the exact acquisition-zero gauge")
    return history


def _difference_covariance_from_joint(joint: np.ndarray) -> np.ndarray:
    date_count = joint.shape[0] // 2
    difference = (
        joint[:date_count, :date_count]
        + joint[date_count:, date_count:]
        - joint[:date_count, date_count:]
        - joint[date_count:, :date_count]
    )
    difference = 0.5 * (difference + difference.T)
    difference[0, :] = 0.0
    difference[:, 0] = 0.0
    return difference


def _frozen_roundoff_matches(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 1e-12 * max(1.0, abs(expected))


def _independent_fixed_l2_psd_covariance(operator: np.ndarray) -> np.ndarray:
    covariance = _difference_covariance_from_joint(operator)
    scale = float(np.max(np.abs(covariance)))
    if scale == 0.0:
        return np.zeros_like(covariance)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    rank_tolerance = scale * 1e-10
    if np.any(values < -rank_tolerance):
        raise SchemaError("independent fixed-L2 covariance is not PSD")
    retained = values > rank_tolerance
    if not np.any(retained):
        return np.zeros_like(covariance)
    factor = vectors[:, retained] * np.sqrt(values[retained])
    reconstructed = factor @ factor.T
    reconstructed[0, :] = 0.0
    reconstructed[:, 0] = 0.0
    return reconstructed


def _fixed_l2_reconstruction_bound(expected: np.ndarray) -> float:
    return (
        np.finfo(np.float64).eps
        * expected.shape[0]
        * 256.0
        * max(1.0, float(np.max(np.abs(expected))))
    )


def independently_recompute_metrics(
    attempt: Mapping[str, Any], latent_target: Sequence[float], latent_reference: Sequence[float],
    dense_oracle: np.ndarray | None = None,
) -> dict[str, Any]:
    date_count = len(latent_target)
    target = _history(attempt.get("target_estimate_history"), date_count, "target estimate history")
    reference = _history(attempt.get("reference_estimate_history"), date_count, "reference estimate history")
    covariance = _matrix(attempt.get("predicted_difference_covariance"), "predicted difference covariance")
    if covariance.shape != (date_count, date_count) or not np.array_equal(covariance, covariance.T):
        raise SchemaError("predicted difference covariance has the wrong date shape or is not exactly symmetric")
    if np.any(covariance[0, :] != 0.0) or np.any(covariance[:, 0] != 0.0):
        raise SchemaError("predicted difference covariance violates the exact acquisition-zero gauge")
    minimum = _minimum_symmetric_eigenvalue(covariance)
    if minimum < -1e-10:
        raise SchemaError("predicted difference covariance is not PSD")
    latent_difference = np.asarray(latent_target, dtype=np.float64) - np.asarray(latent_reference, dtype=np.float64)
    latent_target_array = np.asarray(latent_target, dtype=np.float64)
    latent_reference_array = np.asarray(latent_reference, dtype=np.float64)
    target_error = target - latent_target_array
    reference_error = reference - latent_reference_array
    error = target_error - reference_error
    operator = _matrix(attempt.get("production_operator_matrix"), "production operator matrix")
    weights = np.asarray(attempt.get("contrast_weights"), dtype=np.float64)
    joint_count = 2 * date_count
    if operator.shape != (joint_count, joint_count) or weights.shape != (joint_count,):
        raise SchemaError("production operator and contrast dimensions disagree")
    expected_weights = np.zeros(joint_count, dtype=np.float64)
    expected_weights[date_count - 1] = 1.0
    expected_weights[-1] = -1.0
    if not np.isfinite(weights).all() or not np.array_equal(weights, expected_weights):
        raise SchemaError("contrast weights must be the exact final-date target-minus-reference contrast")
    reconstructed_covariance = _independent_fixed_l2_psd_covariance(operator)
    if np.max(np.abs(covariance - reconstructed_covariance)) > _fixed_l2_reconstruction_bound(
        reconstructed_covariance
    ):
        raise SchemaError("predicted difference covariance differs from the independent fixed-L2 PSD reconstruction")
    operator_error = None
    contrast_variance_error = None
    if dense_oracle is not None:
        oracle = np.asarray(dense_oracle, dtype=np.float64)
        if oracle.shape != operator.shape:
            raise SchemaError("production operator and deterministic dense oracle dimensions disagree")
        operator_error = _frobenius(operator - oracle) / max(_frobenius(oracle), 1e-15)
        operator_variance = _quadratic(weights, operator)
        oracle_variance = _quadratic(weights, oracle)
        contrast_variance_error = abs(operator_variance - oracle_variance) / max(abs(oracle_variance), 1e-15)
    covered = np.zeros(date_count, dtype=np.int64)
    interval_score = np.zeros(date_count, dtype=np.float64)
    interval_width = np.zeros(date_count, dtype=np.float64)
    for index in range(1, date_count):
        half_width = 1.959963984540054 * math.sqrt(max(float(covariance[index, index]), 0.0))
        absolute_error = abs(float(error[index]))
        covered[index] = int(absolute_error <= half_width)
        width = 2.0 * half_width
        interval_width[index] = width
        interval_score[index] = width + (40.0 * (absolute_error - half_width) if absolute_error > half_width else 0.0)
    return {
        "error": error,
        "target_error": target_error,
        "reference_error": reference_error,
        "predicted_covariance": covariance,
        "production_operator": operator,
        "contrast_weights": weights,
        "operator_relative_error": operator_error,
        "contrast_variance_relative_error": contrast_variance_error,
        "psd_min_eigenvalue": minimum,
        "covered": covered,
        "interval_score": interval_score,
        "interval_width": interval_width,
        "estimate_sha256": numeric_digest("estimate-history-v4", [*target, *reference]),
        "predicted_covariance_sha256": numeric_digest("predicted-difference-covariance-v4", covariance.flat),
        "operator_sha256": numeric_digest("production-operator-v4", operator.flat),
        "target_predicted_covariance_trace": float(np.trace(operator[:date_count, :date_count])),
        "reference_predicted_covariance_trace": float(np.trace(operator[date_count:, date_count:])),
    }


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
    if any(window_shape[axis] > native_tile_shape[axis] for axis in range(2)):
        raise SchemaError("frozen support exceeds one native tile")
    start = [
        max(0, center[axis] - half_window[axis])
        for axis in range(2)
    ]
    return [
        (start[0] + row, start[1] + column)
        for row in range(window_shape[0])
        for column in range(window_shape[1])
    ]


def _source_factor_support(
    primitive_sources: Iterable[tuple[int, int]],
    half_window: Sequence[int],
    native_tile_shape: Sequence[int],
) -> list[tuple[int, int]]:
    return sorted({
        coordinate
        for source in primitive_sources
        for coordinate in _candidate_support(source, half_window, native_tile_shape)
    })


def _generate_complex_source(
    preregistration: Mapping[str, Any],
    cell_ordinal: int,
    seed_index: int,
    coordinate: tuple[int, int],
    date_count: int,
    spatial: bool,
    eigen_stress: str,
    global_loading: float,
) -> list[complex]:
    coefficients = _portable_dgp_tables(preregistration)["coefficients"]
    temporal_rho = _f64_from_bits(coefficients["temporal_rho_bits"])
    innovation_weight = _f64_from_bits(coefficients["innovation_weight_bits"])
    if spatial:
        local_weight = _f64_from_bits(coefficients["spatial_local_weight_bits"])
        spatial_weight = _f64_from_bits(coefficients["spatial_global_weight_bits"])
    else:
        local_weight = _f64_from_bits(coefficients["independent_local_weight_bits"])
        spatial_weight = _f64_from_bits(coefficients["independent_spatial_weight_bits"])
    noise_scale = _f64_from_bits(coefficients["noise_scale_bits"])
    amplitude_scales = coefficients["amplitude_scale_bits"][str(coordinate[0] + 3 * coordinate[1])][eigen_stress]
    phasors = coefficients["phasor_bits"][str(2 * coordinate[0] - coordinate[1])]
    signal_real = 0.0
    signal_imag = 0.0
    values: list[complex] = []
    for acquisition in range(date_count):
        local_signal_real = portable_normal(
            preregistration, cell_ordinal, seed_index, coordinate[0], coordinate[1],
            acquisition, "local-signal-real",
        )
        local_signal_imag = portable_normal(
            preregistration, cell_ordinal, seed_index, coordinate[0], coordinate[1],
            acquisition, "local-signal-imag",
        )
        global_signal_real = portable_normal(
            preregistration, cell_ordinal, seed_index, 0, 0,
            acquisition, "global-signal-real",
        )
        global_signal_imag = portable_normal(
            preregistration, cell_ordinal, seed_index, 0, 0,
            acquisition, "global-signal-imag",
        )
        weighted_local_real = local_weight * local_signal_real
        weighted_global_real = spatial_weight * global_signal_real
        innovation_real = weighted_local_real + weighted_global_real
        weighted_local_imag = local_weight * local_signal_imag
        weighted_global_imag = spatial_weight * global_signal_imag
        innovation_imag = weighted_local_imag + weighted_global_imag
        if acquisition == 0:
            signal_real, signal_imag = innovation_real, innovation_imag
        else:
            previous_real = temporal_rho * signal_real
            weighted_innovation_real = innovation_weight * innovation_real
            signal_real = previous_real + weighted_innovation_real
            previous_imag = temporal_rho * signal_imag
            weighted_innovation_imag = innovation_weight * innovation_imag
            signal_imag = previous_imag + weighted_innovation_imag
        amplitude_scale = _f64_from_bits(amplitude_scales[acquisition])
        scaled_signal_real = amplitude_scale * signal_real
        scaled_signal_imag = amplitude_scale * signal_imag
        local_noise_real = portable_normal(
            preregistration, cell_ordinal, seed_index, coordinate[0], coordinate[1],
            acquisition, "local-noise-real",
        )
        local_noise_imag = portable_normal(
            preregistration, cell_ordinal, seed_index, coordinate[0], coordinate[1],
            acquisition, "local-noise-imag",
        )
        global_noise_real = portable_normal(
            preregistration, cell_ordinal, seed_index, 0, 0,
            acquisition, "global-noise-real",
        )
        global_noise_imag = portable_normal(
            preregistration, cell_ordinal, seed_index, 0, 0,
            acquisition, "global-noise-imag",
        )
        weighted_local_noise_real = local_weight * local_noise_real
        weighted_global_noise_real = spatial_weight * global_noise_real
        mixed_noise_real = weighted_local_noise_real + weighted_global_noise_real
        scaled_noise_real = noise_scale * mixed_noise_real
        weighted_local_noise_imag = local_weight * local_noise_imag
        weighted_global_noise_imag = spatial_weight * global_noise_imag
        mixed_noise_imag = weighted_local_noise_imag + weighted_global_noise_imag
        scaled_noise_imag = noise_scale * mixed_noise_imag
        base_real = scaled_signal_real + scaled_noise_real
        base_imag = scaled_signal_imag + scaled_noise_imag
        if global_loading == -1.0:
            base_imag = -base_imag
        elif global_loading != 1.0:
            raise SchemaError("portable DGP loading must be exactly positive or negative one")
        cosine = _f64_from_bits(phasors[acquisition][0])
        sine = _f64_from_bits(phasors[acquisition][1])
        rotate_real_first = base_real * cosine
        rotate_real_second = base_imag * sine
        value_real = rotate_real_first - rotate_real_second
        rotate_imag_first = base_real * sine
        rotate_imag_second = base_imag * cosine
        value_imag = rotate_imag_first + rotate_imag_second
        values.append(complex(
            struct.unpack("<f", struct.pack("<f", value_real))[0],
            struct.unpack("<f", struct.pack("<f", value_imag))[0],
        ))
    return values


def _latent_phase_history(
    preregistration: Mapping[str, Any],
    coordinate: Sequence[int],
    date_count: int,
) -> list[float]:
    values = _portable_dgp_tables(preregistration)["coefficients"]["latent_phase_bits"][
        str(2 * coordinate[0] - coordinate[1])
    ]
    return [_f64_from_bits(bits) for bits in values[:date_count]]


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
    digest.update(struct.pack("<Q", len(support)))
    for coordinate in support:
        digest.update(struct.pack("<qq", coordinate[0], coordinate[1]))
        values = raw_by_source[coordinate]
        digest.update(struct.pack("<Q", len(values)))
        for value in values:
            real = 0.0 if value.real == 0.0 else value.real
            imaginary = 0.0 if value.imag == 0.0 else value.imag
            if not math.isfinite(real) or not math.isfinite(imaginary):
                raise SchemaError("raw source digest contains a non-finite component")
            digest.update(struct.pack("<ff", real, imaginary))
    return digest.hexdigest()


def _tied_probe_attempt_inputs(
    preregistration: Mapping[str, Any], cell_id: str, seed_index: int
) -> dict[str, Any]:
    probe = preregistration["generator"]["singular_local_information_probe"]
    support = [(row, column) for row in range(3) for column in range(3)]
    raw_by_source = {
        coordinate: [
            complex(_f64_from_bits(component[0]), _f64_from_bits(component[1]))
            for component in probe["raw_complex_binary64_bits"][3 * coordinate[0] + coordinate[1]]
        ]
        for coordinate in support
    }
    digest = hashlib.sha256(b"singular-local-information-probe-v1")
    digest.update(struct.pack("<Q", len(support)))
    for coordinate in support:
        digest.update(struct.pack("<qq", *coordinate))
        values = raw_by_source[coordinate]
        digest.update(struct.pack("<Q", len(values)))
        for value in values:
            digest.update(struct.pack("<dd", value.real, value.imag))
    raw_input_sha256 = digest.hexdigest()
    support_sha256 = sha256_json([{"block_id": 0, "sources": support}])
    ancestry_sha256 = sha256_json({
        "probe_schema": probe["schema"],
        "native_shape": probe["native_shape"],
        "date_axis": probe["date_axis"],
        "half_window": probe["half_window"],
        "stride": probe["stride"],
        "ministack_size": probe["ministack_size"],
        "max_num_compressed": probe["max_num_compressed"],
        "estimator": probe["estimator"],
        "branch_tolerance": probe["branch_tolerance"],
    })
    raw_identity = {
        "cell_id": cell_id,
        "seed_index": seed_index,
        "dgp_generator_identity_sha256": _dgp_generator_identity(preregistration),
        "probe_sha256": sha256_json(probe),
        "raw_input_sha256": raw_input_sha256,
        "expected_production_status": "singular_local_information",
        "scientific_numeric_axes_executed": False,
    }
    return {
        "target_coordinate": [0, 0],
        "reference_coordinate": [0, 0],
        "date_axis_sha256": sha256_json(probe["date_axis"]),
        "raw_input_shape": [9, 3, 2],
        "raw_input_value_count": 54,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": raw_input_sha256,
        "reference_raw_input_sha256": raw_input_sha256,
        "target_support_sha256": support_sha256,
        "reference_support_sha256": support_sha256,
        "sequential_ancestry_sha256": ancestry_sha256,
        "raw_dgp_identity_sha256": sha256_json(raw_identity),
        "raw_cube_source_count": 9,
        "target_source_count": 9,
        "reference_source_count": 9,
        "intersection_source_count": 9,
        "union_source_count": 9,
        "effective_looks_fraction": 1.0,
        "source_correlation_model": "identity_v1",
        "source_correlation_distance_scale_pixels": 0.0,
        "target_global_loading_mean": 0.0,
        "reference_global_loading_mean": 0.0,
        "latent_target_history": [],
        "latent_reference_history": [],
        "latent_history_sha256": "0" * 64,
        "dense_joint_oracle": [],
        "oracle_difference_covariance": [],
        "independent_difference_covariance": [],
    }


def _effective_looks_fraction(support: Sequence[tuple[int, int]]) -> float:
    denominator = sum(
        math.exp(-math.hypot(first[0] - second[0], first[1] - second[1]) / 1.5)
        for first in support
        for second in support
    )
    return len(support) / denominator


def derive_dense_joint_oracle(
    preregistration: Mapping[str, Any], cell_id: str, regenerated: Mapping[str, Any]
) -> np.ndarray:
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    topology = preregistration["generator"]["acquisition"]["topologies"][labels["block_topology"]]
    date_count = topology["acquisition_count"]
    raw = preregistration["generator"]["raw_proper_complex"]
    rho = math.exp(-preregistration["generator"]["acquisition"]["cadence_days"] / raw["correlation_days"])
    temporal = np.fromfunction(
        lambda row, column: rho ** np.abs(row - column),
        (date_count, date_count),
        dtype=float,
    )
    temporal = temporal - temporal[:, [0]] - temporal[[0], :] + temporal[0, 0]
    target_scale = raw["noise_variance"] / regenerated["target_source_count"]
    reference_scale = raw["noise_variance"] / regenerated["reference_source_count"]
    sign = PAIR_SIGN[labels["pair_geometry"]]
    signed_influence = (
        regenerated.get("target_global_loading_mean", 0.0)
        * regenerated.get("reference_global_loading_mean", 0.0)
    )
    coupling_scale = math.sqrt(target_scale * reference_scale) if sign == "zero" else (
        0.5 * signed_influence * math.sqrt(target_scale * reference_scale)
        if sign in {"positive", "negative"}
        else 0.0
    )
    target = target_scale * temporal
    reference = reference_scale * temporal
    cross = coupling_scale * temporal
    return np.block([[target, cross], [cross.T, reference]])


def regenerate_frozen_attempt_inputs(
    preregistration: Mapping[str, Any], cell_id: str, seed_index: int
) -> dict[str, Any]:
    if not _integer(seed_index) or seed_index < 0 or seed_index >= expected_seed_count(cell_id):
        raise SchemaError("frozen DGP seed index is outside the preregistered schedule")
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    if set(labels) != set(DIMENSION_NAMES):
        raise SchemaError("frozen DGP cell identity is malformed")
    if labels["eigen_stress"] == "tied_eigenvalue" and labels["position"] != "masked":
        return _tied_probe_attempt_inputs(preregistration, cell_id, seed_index)
    target, reference = _expected_coordinates(preregistration, cell_id)
    window = preregistration["generator"]["coordinates"]["window_stride"][f"{labels['half_window']}|{labels['stride']}"]
    native_tile_shape = preregistration["generator"]["full_replay_dgp"]["native_tile_shape"]
    target_candidates = _candidate_support(target, window["half_window"], native_tile_shape)
    reference_candidates = _candidate_support(reference, window["half_window"], native_tile_shape)
    candidate_union = sorted(set(target_candidates) | set(reference_candidates))
    topology = preregistration["generator"]["acquisition"]["topologies"][labels["block_topology"]]
    date_count = topology["acquisition_count"]
    dgp_cell_ordinal = _dgp_cell_ordinal(preregistration, cell_id)
    negative_pair = labels["pair_geometry"].endswith("_negative")
    pair_has_signed_loading = PAIR_SIGN[labels["pair_geometry"]] in {"positive", "negative"}
    spatial = labels["source_process"] == "spatial_correlation_stress" or pair_has_signed_loading
    candidate_loading = {
        coordinate: (
            1.0
            if not negative_pair
            or (coordinate[0] - target[0]) ** 2 + (coordinate[1] - target[1]) ** 2
            <= (coordinate[0] - reference[0]) ** 2 + (coordinate[1] - reference[1]) ** 2
            else -1.0
        )
        for coordinate in candidate_union
    }
    phase_raw_by_source = {
        coordinate: _generate_complex_source(
            preregistration, dgp_cell_ordinal, seed_index, coordinate, date_count,
            spatial, labels["eigen_stress"], candidate_loading[coordinate],
        )
        for coordinate in candidate_union
    }
    target_supports = []
    reference_supports = []
    for block in topology["expected_blocks"]:
        block_dates = range(block["real_start"], block["real_start"] + block["num_real"])
        block_raw = {
            coordinate: [phase_raw_by_source[coordinate][acquisition] for acquisition in block_dates]
            for coordinate in candidate_union
        }
        target_supports.append(_select_support(labels["support"], target_candidates, target, block_raw))
        reference_supports.append(_select_support(labels["support"], reference_candidates, reference, block_raw))
    target_support = sorted(set().union(*map(set, target_supports)))
    reference_support = sorted(set().union(*map(set, reference_supports)))
    union_support = sorted(set(target_support) | set(reference_support))
    shared = sorted(set(target_support) & set(reference_support))
    target_factor_supports = [
        _source_factor_support(support, window["half_window"], native_tile_shape)
        for support in target_supports
    ]
    reference_factor_supports = [
        _source_factor_support(support, window["half_window"], native_tile_shape)
        for support in reference_supports
    ]
    target_raw_support = sorted(set().union(*map(set, target_factor_supports)))
    reference_raw_support = sorted(set().union(*map(set, reference_factor_supports)))
    raw_cube_support = set(candidate_union)
    for _ in topology["expected_blocks"]:
        raw_cube_support = set(
            _source_factor_support(
                raw_cube_support, window["half_window"], native_tile_shape
            )
        )
    raw_cube_support = sorted(
        raw_cube_support | set(target_raw_support) | set(reference_raw_support)
    )
    global_loading_by_source = {
        coordinate: (
            1.0
            if not negative_pair
            or (coordinate[0] - target[0]) ** 2 + (coordinate[1] - target[1]) ** 2
            <= (coordinate[0] - reference[0]) ** 2 + (coordinate[1] - reference[1]) ** 2
            else -1.0
        )
        for coordinate in raw_cube_support
    }
    raw_by_source = {
        coordinate: phase_raw_by_source.get(coordinate) or _generate_complex_source(
            preregistration, dgp_cell_ordinal, seed_index, coordinate, date_count,
            spatial, labels["eigen_stress"], global_loading_by_source[coordinate],
        )
        for coordinate in raw_cube_support
    }
    effective_support = set(union_support)
    for _ in range(1, len(topology["expected_blocks"])):
        effective_support = set(
            _source_factor_support(
                effective_support, window["half_window"], native_tile_shape
            )
        )
    source_correlation_model = (
        "identity_v1"
        if labels["source_process"] == "independent_complex_looks"
        else "exponential_euclidean_v1"
    )
    source_correlation_distance_scale_pixels = (
        0.0 if source_correlation_model == "identity_v1" else 1.5
    )
    effective_looks = (
        None
        if not effective_support
        else 1.0
        if source_correlation_model == "identity_v1"
        else _effective_looks_fraction(sorted(effective_support))
    )
    target_global_loading_mean = (
        sum(global_loading_by_source[source] for source in target_support) / len(target_support)
        if target_support
        else 0.0
    )
    reference_global_loading_mean = (
        sum(global_loading_by_source[source] for source in reference_support) / len(reference_support)
        if reference_support
        else 0.0
    )
    latent_target = _latent_phase_history(preregistration, target, date_count)
    latent_reference = (
        latent_target
        if target == reference
        else _latent_phase_history(preregistration, reference, date_count)
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
    target_factor_support_sha256 = sha256_json([
        {"block_id": block["block_id"], "sources": support}
        for block, support in zip(topology["expected_blocks"], target_factor_supports)
    ])
    reference_factor_support_sha256 = sha256_json([
        {"block_id": block["block_id"], "sources": support}
        for block, support in zip(topology["expected_blocks"], reference_factor_supports)
    ])
    ancestry = {
        "date_axis": topology["date_axis"],
        "expected_blocks": topology["expected_blocks"],
        "max_num_compressed": topology["max_num_compressed"],
        "partial_tail_count": topology["partial_tail_count"],
    }
    sequential_ancestry_sha256 = sha256_json(ancestry)
    provisional = {
        "target_source_count": len(target_support),
        "reference_source_count": len(reference_support),
        "target_global_loading_mean": target_global_loading_mean,
        "reference_global_loading_mean": reference_global_loading_mean,
    }
    dense_oracle = (
        derive_dense_joint_oracle(preregistration, cell_id, provisional)
        if target_support and reference_support
        else np.zeros((2 * date_count, 2 * date_count), dtype=np.float64)
    )
    oracle_difference = _difference_covariance_from_joint(dense_oracle)
    independent_oracle = dense_oracle.copy()
    independent_oracle[:date_count, date_count:] = 0.0
    independent_oracle[date_count:, :date_count] = 0.0
    independent_difference = _difference_covariance_from_joint(independent_oracle)
    raw_input_sha256 = _raw_source_digest("raw-input-v4", raw_cube_support, raw_by_source)
    target_raw_input_sha256 = _raw_source_digest("source-raw-input-v4", target_raw_support, raw_by_source)
    reference_raw_input_sha256 = _raw_source_digest("source-raw-input-v4", reference_raw_support, raw_by_source)
    raw_identity = {
        "cell_id": cell_id,
        "dgp_cell_ordinal": dgp_cell_ordinal,
        "seed_index": seed_index,
        "dgp_generator_identity_sha256": _dgp_generator_identity(preregistration),
        "shape": [len(raw_cube_support), date_count, 2],
        "target_coordinate": target,
        "reference_coordinate": reference,
        "target_support_sha256": target_support_sha256,
        "reference_support_sha256": reference_support_sha256,
        "target_factor_support_sha256": target_factor_support_sha256,
        "reference_factor_support_sha256": reference_factor_support_sha256,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": target_raw_input_sha256,
        "reference_raw_input_sha256": reference_raw_input_sha256,
        "sequential_ancestry_sha256": sequential_ancestry_sha256,
        "estimator": labels["estimator"],
        "eigen_stress": labels["eigen_stress"],
        "source_process": labels["source_process"],
    }
    return {
        "target_coordinate": target,
        "reference_coordinate": reference,
        "date_axis_sha256": sha256_json(topology["date_axis"]),
        "raw_input_shape": [len(raw_cube_support), date_count, 2],
        "raw_input_value_count": 2 * len(raw_cube_support) * date_count,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": target_raw_input_sha256,
        "reference_raw_input_sha256": reference_raw_input_sha256,
        "target_support_sha256": target_support_sha256,
        "reference_support_sha256": reference_support_sha256,
        "sequential_ancestry_sha256": sequential_ancestry_sha256,
        "raw_dgp_identity_sha256": sha256_json(raw_identity),
        "raw_cube_source_count": len(raw_cube_support),
        "target_source_count": len(target_support),
        "reference_source_count": len(reference_support),
        "intersection_source_count": len(shared),
        "union_source_count": len(union_support),
        "effective_looks_fraction": effective_looks,
        "source_correlation_model": source_correlation_model,
        "source_correlation_distance_scale_pixels": source_correlation_distance_scale_pixels,
        "target_global_loading_mean": target_global_loading_mean,
        "reference_global_loading_mean": reference_global_loading_mean,
        "latent_target_history": latent_target,
        "latent_reference_history": latent_reference,
        "latent_history_sha256": numeric_digest(
            "latent-phase-history-v4", [*latent_target, *latent_reference]
        ),
        "dense_joint_oracle": dense_oracle.tolist(),
        "dense_oracle_sha256": numeric_digest("dense-oracle-v4", dense_oracle.flat),
        "oracle_difference_covariance": oracle_difference.tolist(),
        "oracle_difference_covariance_trace": float(np.trace(oracle_difference)),
        "oracle_independent_covariance_trace": float(np.trace(independent_difference)),
        "target_marginal_oracle_sha256": numeric_digest(
            "target-marginal-oracle-v4", dense_oracle[:date_count, :date_count].flat
        ),
        "reference_marginal_oracle_sha256": numeric_digest(
            "reference-marginal-oracle-v4", dense_oracle[date_count:, date_count:].flat
        ),
    }


def realized_overlap_jaccard(target_count: Any, reference_count: Any, intersection_count: Any, union_count: Any) -> float:
    counts = (target_count, reference_count, intersection_count, union_count)
    if any(not _integer(value) or value < 0 for value in counts):
        raise SchemaError("source-key overlap counts must be non-negative integers")
    if intersection_count > min(target_count, reference_count) or union_count != target_count + reference_count - intersection_count or union_count == 0:
        raise SchemaError("source-key intersection/union arithmetic is invalid")
    return intersection_count / union_count


def expected_production_artifact_provenance(
    preregistration: Mapping[str, Any], cell_id: str, seed_index: int,
    regenerated: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if regenerated is None:
        regenerated = regenerate_frozen_attempt_inputs(preregistration, cell_id, seed_index)
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    return {
        "method": "reference_specific_influence_v1",
        "hdf5_schema_version": preregistration["execution_protocol"]["production_hdf5_schema_version"],
        "reference_coordinate": regenerated["reference_coordinate"],
        "estimator_branch": labels["estimator"],
        "gauge": "acquisition_zero_exact",
        "date_axis_sha256": regenerated["date_axis_sha256"],
        "target_support_sha256": regenerated["target_support_sha256"],
        "reference_support_sha256": regenerated["reference_support_sha256"],
        "sequential_ancestry_sha256": regenerated["sequential_ancestry_sha256"],
    }


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
    artifact_root: Path | None = None
    next_seed_index: int = 0
    emitted: int = 0
    date_count: int = 0
    target_total: int = 0
    reference_total: int = 0
    intersection_total: int = 0
    union_total: int = 0
    overlap_total: float = 0.0
    effective_looks_total: float = 0.0
    min_psd_eigenvalue: float | None = None
    coverage_counts: np.ndarray | None = None
    interval_score_sums: np.ndarray | None = None
    interval_width_sums: np.ndarray | None = None
    statuses: dict[str, int] = field(default_factory=lambda: {status: 0 for status in ATTEMPT_STATUSES})
    error_sum: np.ndarray | None = None
    error_outer_sum: np.ndarray | None = None
    predicted_covariance_sum: np.ndarray | None = None
    target_error_sum: np.ndarray | None = None
    reference_error_sum: np.ndarray | None = None
    target_reference_error_outer_sum: np.ndarray | None = None
    target_error_outer_sum: np.ndarray | None = None
    reference_error_outer_sum: np.ndarray | None = None
    production_operator_sum: np.ndarray | None = None
    target_predicted_trace_sum: float = 0.0
    reference_predicted_trace_sum: float = 0.0
    operator_relative_error_max: float = 0.0
    contrast_variance_relative_error_max: float = 0.0
    field_digests: dict[str, Any] = field(default_factory=lambda: {name: hashlib.sha256() for name in (
        "estimate_sha256", "latent_history_sha256",
        "target_support_sha256", "reference_support_sha256",
    )})
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
        regenerated = self._validate_scope(attempt)
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
        self._accumulate(attempt, regenerated)
        self.next_seed_index += 1

    def _validate_scope(self, attempt: Mapping[str, Any]) -> Mapping[str, Any]:
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        tied_probe = labels["eigen_stress"] == "tied_eigenvalue" and labels["position"] != "masked"
        target, reference = ([0, 0], [0, 0]) if tied_probe else _expected_coordinates(self.preregistration, self.cell_id)
        generator = self.preregistration["generator"]
        topology = generator["acquisition"]["topologies"][labels["block_topology"]]
        if attempt.get("generator_hash") != sha256_json(generator) or attempt.get("config_hash") != sha256_json(generator):
            raise SchemaError(f"cell {self.cell_id} has a generator/config identity mismatch")
        if attempt.get("source_model_hash") != sha256_json(generator["source_centered_empirical"]):
            raise SchemaError(f"cell {self.cell_id} has a source-model identity mismatch")
        expected_date_axis = generator["singular_local_information_probe"]["date_axis"] if tied_probe else topology["date_axis"]
        if attempt.get("date_axis_sha256") != sha256_json(expected_date_axis):
            raise SchemaError(f"cell {self.cell_id} has a date-axis identity mismatch")
        expected_estimator = "evd" if tied_probe else labels["estimator"]
        if attempt.get("estimator_branch") != expected_estimator:
            raise SchemaError(f"cell {self.cell_id} estimator identity differs from the frozen branch")
        regenerated = regenerate_frozen_attempt_inputs(
            self.preregistration, self.cell_id, self.next_seed_index
        )
        if any(
            not isinstance(attempt.get(field_name), list)
            or len(attempt[field_name]) != 2
            or any(not _integer(value) for value in attempt[field_name])
            for field_name in ("target_coordinate", "reference_coordinate")
        ) or attempt.get("target_coordinate") != target or attempt.get("reference_coordinate") != reference:
            raise SchemaError(f"cell {self.cell_id} has a coordinate identity mismatch")
        self._validate_regenerated_inputs(attempt, regenerated)
        status = attempt.get("status")
        if status not in ATTEMPT_STATUSES or not isinstance(attempt.get("emitted"), bool) or not isinstance(attempt.get("factor_emitted"), bool):
            raise SchemaError(f"cell {self.cell_id} has invalid status/emission flags")
        empty = expected_empty_support(self.cell_id)
        effective = attempt.get("effective_looks_fraction")
        if (
            (empty and effective is not None)
            or (not empty and labels["position"] != "masked" and (not _number(effective) or effective <= 0))
            or attempt.get("effective_looks_application") != "source_influence_joint_contraction_v1"
        ):
            raise SchemaError(f"cell {self.cell_id} has an invalid effective-look realization")
        expected_model = (
            "identity_v1"
            if labels["source_process"] == "independent_complex_looks"
            else "exponential_euclidean_v1"
        )
        expected_scale = 0.0 if expected_model == "identity_v1" else 1.5
        if (
            attempt.get("source_correlation_model") != expected_model
            or attempt.get("source_correlation_distance_scale_pixels") != expected_scale
        ):
            raise SchemaError(f"cell {self.cell_id} has an invalid source-correlation identity")
        if labels["position"] == "masked":
            self._validate_masked(attempt)
        elif status == "masked_target":
            raise SchemaError(f"cell {self.cell_id} cannot use masked_target")
        elif status == "empty_support" and not empty:
            raise SchemaError(f"cell {self.cell_id} has an undeclared empty-support attempt")
        elif empty and status != "empty_support":
            raise SchemaError(f"cell {self.cell_id} must preserve the production empty-support status")
        elif status == "singular_local_information" and labels["eigen_stress"] != "tied_eigenvalue":
            raise SchemaError(f"cell {self.cell_id} has an undeclared not-evaluable attempt")
        elif status == "nondifferentiable_node":
            self._validate_nondifferentiable(attempt)
        elif attempt.get("factor_emitted") != attempt.get("emitted"):
            raise SchemaError(f"cell {self.cell_id} has inconsistent factor/emission flags")
        return regenerated

    def _validate_regenerated_inputs(
        self, attempt: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        raw_fields = (
            "raw_input_shape", "raw_input_value_count", "raw_input_sha256",
            "target_raw_input_sha256", "reference_raw_input_sha256",
            "target_support_sha256", "reference_support_sha256",
            "sequential_ancestry_sha256", "raw_dgp_identity_sha256",
            "target_source_count", "reference_source_count", "intersection_source_count",
            "union_source_count",
            "source_correlation_model", "source_correlation_distance_scale_pixels",
        )
        if any(attempt.get(field_name) != expected[field_name] for field_name in raw_fields):
            raise SchemaError(f"cell {self.cell_id} raw DGP does not match deterministic regeneration")
        if (
            expected["effective_looks_fraction"] is None
            and attempt.get("effective_looks_fraction") is not None
        ) or (
            expected["effective_looks_fraction"] is not None
            and not _frozen_roundoff_matches(
                attempt["effective_looks_fraction"], expected["effective_looks_fraction"]
            )
        ):
            raise SchemaError(f"cell {self.cell_id} effective-look realization differs from independent recomputation")
        if attempt.get("latent_history_sha256") != expected["latent_history_sha256"]:
            raise SchemaError(f"cell {self.cell_id} latent history identity does not match regeneration")
        if attempt.get("status") in {"empty_support", "singular_local_information"}:
            return
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        expected_influence = (
            expected["target_global_loading_mean"] * expected["reference_global_loading_mean"]
            if PAIR_SIGN[labels["pair_geometry"]] in {"positive", "negative"}
            else 0.0
        )
        if labels["position"] != "masked" and (
            not _number(attempt.get("signed_cross_influence"))
            or not math.isclose(
                attempt["signed_cross_influence"], expected_influence,
                rel_tol=0.0, abs_tol=1e-15,
            )
        ):
            raise SchemaError(f"cell {self.cell_id} signed influence does not match regenerated raw DGP loadings")

    def _validate_masked(self, attempt: Mapping[str, Any]) -> None:
        if attempt.get("status") != "masked_target" or attempt.get("emitted") is not False or attempt.get("factor_emitted") is not False:
            raise SchemaError(f"cell {self.cell_id} masked attempt must abstain")
        metrics = (
            "signed_cross_influence", "target_estimate_history", "reference_estimate_history",
            "predicted_difference_covariance", "production_operator_matrix",
            "contrast_weights",
        )
        if any(attempt.get(metric) is not None for metric in metrics):
            raise SchemaError(f"cell {self.cell_id} masked attempt must use null numeric evidence")
        if any(attempt.get(name) != "0" * 64 for name in (
            "estimate_sha256", "predicted_covariance_sha256", "operator_sha256",
        )):
            raise SchemaError(f"cell {self.cell_id} masked attempt must use null estimator digests")

    def _validate_nondifferentiable(self, attempt: Mapping[str, Any]) -> None:
        if attempt.get("emitted") is not False or attempt.get("factor_emitted") is not False:
            raise SchemaError(f"cell {self.cell_id} nondifferentiable attempt must not emit")
        if any(attempt.get(name) is not None for name in (
            "target_estimate_history", "reference_estimate_history",
            "predicted_difference_covariance", "production_operator_matrix",
            "contrast_weights",
        )):
            raise SchemaError(f"cell {self.cell_id} nondifferentiable attempt must omit estimator evidence")
        if any(attempt.get(name) != "0" * 64 for name in (
            "estimate_sha256", "predicted_covariance_sha256", "operator_sha256",
        )):
            raise SchemaError(f"cell {self.cell_id} nondifferentiable attempt must use null estimator digests")

    def _accumulate(self, attempt: Mapping[str, Any], regenerated: Mapping[str, Any]) -> None:
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        if attempt["status"] in {"empty_support", "singular_local_information"} or (
            attempt["status"] == "masked_target" and attempt["union_source_count"] == 0
        ):
            if attempt["emitted"] or attempt["factor_emitted"] or any(attempt.get(name) is not None for name in (
                "signed_cross_influence", "target_estimate_history", "reference_estimate_history",
                "predicted_difference_covariance", "production_operator_matrix", "contrast_weights",
            )) or any(attempt.get(name) != "0" * 64 for name in (
                "estimate_sha256", "predicted_covariance_sha256", "operator_sha256",
            )):
                raise SchemaError(f"cell {self.cell_id} non-emitting production status must fail closed")
            self.statuses[attempt["status"]] += 1
            for field_name, digest in self.field_digests.items():
                _update_field_digest(digest, attempt[field_name])
            return
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
        if attempt["emitted"] and attempt["status"] == "valid":
            self._accumulate_valid_metrics(attempt, regenerated)
        for field_name, digest in self.field_digests.items():
            _update_field_digest(digest, attempt[field_name])

    def _accumulate_valid_metrics(
        self, attempt: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        deterministic = self.expected_seed_count == FROZEN_DETERMINISTIC_SEED_COUNT
        metrics = independently_recompute_metrics(
            attempt,
            expected["latent_target_history"],
            expected["latent_reference_history"],
            np.asarray(expected["dense_joint_oracle"], dtype=np.float64) if deterministic else None,
        )
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        if labels["pair_geometry"] == "coincident" and (
            attempt["target_estimate_history"] != attempt["reference_estimate_history"]
            or np.any(metrics["predicted_covariance"] != 0.0)
            or np.any(metrics["error"] != 0.0)
        ):
            raise SchemaError(f"cell {self.cell_id} coincident replay must be exact zero")
        for field_name in ("estimate_sha256", "predicted_covariance_sha256"):
            if attempt.get(field_name) != metrics[field_name]:
                raise SchemaError(f"cell {self.cell_id} has a {field_name} canonical digest mismatch")
        for field_name in ("operator_sha256",):
            if attempt.get(field_name) != metrics[field_name]:
                raise SchemaError(f"cell {self.cell_id} has a {field_name} canonical digest mismatch")
        error = metrics["error"]
        covariance = metrics["predicted_covariance"]
        if self.error_sum is None:
            self.date_count = len(error)
            self.error_sum = np.zeros(self.date_count, dtype=np.float64)
            self.error_outer_sum = np.zeros((self.date_count, self.date_count), dtype=np.float64)
            self.predicted_covariance_sum = np.zeros((self.date_count, self.date_count), dtype=np.float64)
            self.target_error_sum = np.zeros(self.date_count, dtype=np.float64)
            self.reference_error_sum = np.zeros(self.date_count, dtype=np.float64)
            self.target_reference_error_outer_sum = np.zeros((self.date_count, self.date_count), dtype=np.float64)
            self.target_error_outer_sum = np.zeros((self.date_count, self.date_count), dtype=np.float64)
            self.reference_error_outer_sum = np.zeros((self.date_count, self.date_count), dtype=np.float64)
            self.production_operator_sum = np.zeros((2 * self.date_count, 2 * self.date_count), dtype=np.float64)
            self.coverage_counts = np.zeros(self.date_count, dtype=np.int64)
            self.interval_score_sums = np.zeros(self.date_count, dtype=np.float64)
            self.interval_width_sums = np.zeros(self.date_count, dtype=np.float64)
        self.error_sum += error
        self.error_outer_sum += np.outer(error, error)
        self.predicted_covariance_sum += covariance
        assert self.target_error_sum is not None and self.reference_error_sum is not None
        assert self.target_reference_error_outer_sum is not None
        self.target_error_sum += metrics["target_error"]
        self.reference_error_sum += metrics["reference_error"]
        self.target_reference_error_outer_sum += np.outer(metrics["target_error"], metrics["reference_error"])
        assert self.target_error_outer_sum is not None and self.reference_error_outer_sum is not None
        self.target_error_outer_sum += np.outer(metrics["target_error"], metrics["target_error"])
        self.reference_error_outer_sum += np.outer(metrics["reference_error"], metrics["reference_error"])
        assert self.production_operator_sum is not None
        self.production_operator_sum += metrics["production_operator"]
        self.target_predicted_trace_sum += metrics["target_predicted_covariance_trace"]
        self.reference_predicted_trace_sum += metrics["reference_predicted_covariance_trace"]
        if deterministic:
            assert metrics["operator_relative_error"] is not None
            assert metrics["contrast_variance_relative_error"] is not None
            self.operator_relative_error_max = max(self.operator_relative_error_max, metrics["operator_relative_error"])
            self.contrast_variance_relative_error_max = max(
                self.contrast_variance_relative_error_max,
                metrics["contrast_variance_relative_error"],
            )
        self.min_psd_eigenvalue = min(
            self.min_psd_eigenvalue if self.min_psd_eigenvalue is not None else math.inf,
            metrics["psd_min_eigenvalue"],
        )
        assert self.coverage_counts is not None and self.interval_score_sums is not None and self.interval_width_sums is not None
        self.coverage_counts += metrics["covered"]
        self.interval_score_sums += metrics["interval_score"]
        self.interval_width_sums += metrics["interval_width"]

    def finalize(self) -> dict[str, Any]:
        if self.next_seed_index != self.expected_seed_count:
            raise SchemaError(f"cell {self.cell_id} is missing one or more seed indices")
        labels = dict(zip(DIMENSION_NAMES, self.cell_id.split("|")))
        if labels["position"] == "masked":
            if self.statuses["masked_target"] != self.expected_seed_count:
                raise SchemaError(f"cell {self.cell_id} masked status count drifted")
            status = PASS
        elif labels["eigen_stress"] == "tied_eigenvalue":
            if self.statuses["singular_local_information"] != self.expected_seed_count:
                raise SchemaError(f"cell {self.cell_id} tied-eigen cell is not completely not-evaluable")
            status = NOT_EVALUABLE
        elif expected_empty_support(self.cell_id):
            if self.statuses["empty_support"] != self.expected_seed_count:
                raise SchemaError(f"cell {self.cell_id} empty-support status count drifted")
            status = NOT_EVALUABLE
        else:
            if (
                self.statuses["valid"] + self.statuses["nondifferentiable_node"]
                != self.expected_seed_count
            ):
                raise SchemaError(f"cell {self.cell_id} has an unsupported attempt status")
            status = self._numeric_status(labels)
        empirical = predicted = None
        calibration_error = bias_norm = empirical_trace = predicted_trace = None
        target_reference_error_covariance_trace = None
        target_empirical_trace = reference_empirical_trace = None
        target_predicted_trace = reference_predicted_trace = None
        coverage_by_date = interval_score_by_date = interval_width_by_date = None
        final_coverage = final_interval_score = final_interval_width = None
        if self.emitted:
            assert self.error_sum is not None and self.error_outer_sum is not None and self.predicted_covariance_sum is not None
            assert self.coverage_counts is not None and self.interval_score_sums is not None and self.interval_width_sums is not None
            mean_error = self.error_sum / self.emitted
            empirical = self.error_outer_sum / self.emitted - np.outer(mean_error, mean_error)
            predicted = self.predicted_covariance_sum / self.emitted
            calibration_error = _frobenius(predicted - empirical) / max(_frobenius(empirical), 1e-15)
            bias_norm = float(np.linalg.norm(mean_error))
            empirical_trace = float(np.trace(empirical))
            predicted_trace = float(np.trace(predicted))
            assert self.target_error_sum is not None and self.reference_error_sum is not None
            assert self.target_reference_error_outer_sum is not None
            cross_covariance = (
                self.target_reference_error_outer_sum / self.emitted
                - np.outer(self.target_error_sum / self.emitted, self.reference_error_sum / self.emitted)
            )
            target_reference_error_covariance_trace = float(np.trace(cross_covariance))
            assert self.target_error_outer_sum is not None and self.reference_error_outer_sum is not None
            target_empirical = self.target_error_outer_sum / self.emitted - np.outer(
                self.target_error_sum / self.emitted, self.target_error_sum / self.emitted
            )
            reference_empirical = self.reference_error_outer_sum / self.emitted - np.outer(
                self.reference_error_sum / self.emitted, self.reference_error_sum / self.emitted
            )
            target_empirical_trace = float(np.trace(target_empirical))
            reference_empirical_trace = float(np.trace(reference_empirical))
            target_predicted_trace = self.target_predicted_trace_sum / self.emitted
            reference_predicted_trace = self.reference_predicted_trace_sum / self.emitted
            if self.expected_seed_count != FROZEN_DETERMINISTIC_SEED_COUNT:
                coverage_by_date = [None] + [
                    float(value / self.emitted) for value in self.coverage_counts[1:]
                ]
                interval_score_by_date = [None] + [
                    float(value / self.emitted) for value in self.interval_score_sums[1:]
                ]
                interval_width_by_date = [None] + [
                    float(value / self.emitted) for value in self.interval_width_sums[1:]
                ]
                final_coverage = coverage_by_date[-1]
                final_interval_score = interval_score_by_date[-1]
                final_interval_width = interval_width_by_date[-1]
        operator_error, contrast_error = self._aggregate_operator_errors()
        return {
            "schema": "dolphinrust.spatial-covariance.cell-summary/4",
            "cell_id": self.cell_id, "cell_ordinal": self.cell_ordinal, "status": status,
            "attempted_seeds": self.expected_seed_count, "emitted_seeds": self.emitted,
            "status_histogram": dict(self.statuses),
            "failure_histogram": (
                {"nondifferentiable_node": self.statuses["nondifferentiable_node"]}
                if self.statuses["nondifferentiable_node"]
                else {}
            ),
            "request_digest": self.request_digest.hexdigest(), "attempt_digest": self.attempt_digest.hexdigest(),
            "target_source_count_total": self.target_total, "reference_source_count_total": self.reference_total,
            "intersection_source_count_total": self.intersection_total, "union_source_count_total": self.union_total,
            "realized_overlap_jaccard_mean": self.overlap_total / self.expected_seed_count,
            "effective_looks_fraction": self.effective_looks_total / self.expected_seed_count,
            "covariance_calibration_relative_error": calibration_error, "error_bias_norm": bias_norm,
            "operator_relative_error": operator_error,
            "contrast_variance_relative_error": contrast_error,
            "target_reference_error_covariance_trace": target_reference_error_covariance_trace,
            "target_predicted_covariance_trace": target_predicted_trace,
            "reference_predicted_covariance_trace": reference_predicted_trace,
            "target_empirical_error_covariance_trace": target_empirical_trace,
            "reference_empirical_error_covariance_trace": reference_empirical_trace,
            "empirical_error_covariance_trace": empirical_trace, "predicted_covariance_trace": predicted_trace,
            "psd_min_eigenvalue": self.min_psd_eigenvalue,
            "coverage_95_by_date": coverage_by_date,
            "interval_score_mean_by_date": interval_score_by_date,
            "interval_width_mean_by_date": interval_width_by_date,
            "final_date_coverage_95": final_coverage,
            "final_date_interval_score_mean": final_interval_score,
            "final_date_interval_width_mean": final_interval_width,
            "estimate_digest": self.field_digests["estimate_sha256"].hexdigest(),
            "latent_history_digest": self.field_digests["latent_history_sha256"].hexdigest(),
            "target_support_digest": self.field_digests["target_support_sha256"].hexdigest(),
            "reference_support_digest": self.field_digests["reference_support_sha256"].hexdigest(),
            "predicted_covariance_digest": None if predicted is None else numeric_digest("mean-predicted-covariance-v4", predicted.flat),
            "empirical_error_covariance_digest": None if empirical is None else numeric_digest("empirical-error-covariance-v4", empirical.flat),
            "code_sha256": self.code_sha256, "binary_sha256": self.binary_sha256,
            "preregistration_sha256": preregistration_digest(self.preregistration),
        }

    def _numeric_status(self, labels: Mapping[str, str]) -> str:
        if self.min_psd_eigenvalue is None or self.coverage_counts is None or self.emitted == 0:
            return FAIL
        thresholds = self.preregistration["thresholds"]
        deterministic = self.expected_seed_count == FROZEN_DETERMINISTIC_SEED_COUNT
        assert self.error_sum is not None and self.error_outer_sum is not None and self.predicted_covariance_sum is not None
        mean_error = self.error_sum / self.emitted
        empirical = self.error_outer_sum / self.emitted - np.outer(mean_error, mean_error)
        predicted = self.predicted_covariance_sum / self.emitted
        calibration_error = _frobenius(predicted - empirical) / max(_frobenius(empirical), 1e-15)
        coverage_by_date = self.coverage_counts[1:] / self.emitted
        coverage_passes = all(
            abs(float(value) - thresholds["coverage_probability"])
            <= thresholds["coverage_absolute_error_max"]
            for value in coverage_by_date
        )
        final_date_passes = abs(
            float(coverage_by_date[-1]) - thresholds["coverage_probability"]
        ) <= thresholds["coverage_absolute_error_max"]
        operator_limit = (
            thresholds["deterministic_operator_relative_error_max"]
            if deterministic
            else thresholds["stochastic_operator_relative_error_max"]
        )
        operator_error, contrast_error = self._aggregate_operator_errors()
        if operator_error is None or contrast_error is None:
            return FAIL
        cross_sign_passes = True
        if not deterministic and PAIR_SIGN[labels["pair_geometry"]] in {"positive", "negative"}:
            assert self.target_error_sum is not None and self.reference_error_sum is not None
            assert self.target_reference_error_outer_sum is not None
            cross = self.target_reference_error_outer_sum / self.emitted - np.outer(
                self.target_error_sum / self.emitted, self.reference_error_sum / self.emitted
            )
            cross_trace = float(np.trace(cross))
            expected_sign = PAIR_SIGN[labels["pair_geometry"]]
            if expected_sign in {"positive", "negative"}:
                cross_sign_passes = (cross_trace > 0.0) == (expected_sign == "positive")
        passes = (
            self.min_psd_eigenvalue >= thresholds["psd_min_eigenvalue_min"]
            and operator_error <= operator_limit
            and contrast_error <= thresholds["contrast_variance_relative_error_max"]
            and (deterministic or calibration_error <= thresholds["covariance_calibration_relative_error_max"])
            and (deterministic or (coverage_passes and final_date_passes))
            and cross_sign_passes
            and self.emitted / self.expected_seed_count >= thresholds["emission_rate_min"]
        )
        return PASS if passes else FAIL

    def _aggregate_operator_errors(self) -> tuple[float | None, float | None]:
        if self.emitted == 0:
            return None, None
        if self.expected_seed_count == FROZEN_DETERMINISTIC_SEED_COUNT:
            return self.operator_relative_error_max, self.contrast_variance_relative_error_max
        assert self.production_operator_sum is not None
        assert self.target_error_sum is not None and self.reference_error_sum is not None
        assert self.target_error_outer_sum is not None and self.reference_error_outer_sum is not None
        assert self.target_reference_error_outer_sum is not None
        target_mean = self.target_error_sum / self.emitted
        reference_mean = self.reference_error_sum / self.emitted
        target = self.target_error_outer_sum / self.emitted - np.outer(target_mean, target_mean)
        reference = self.reference_error_outer_sum / self.emitted - np.outer(reference_mean, reference_mean)
        cross = self.target_reference_error_outer_sum / self.emitted - np.outer(target_mean, reference_mean)
        empirical_joint = np.block([[target, cross], [cross.T, reference]])
        predicted_joint = self.production_operator_sum / self.emitted
        operator_error = _frobenius(predicted_joint - empirical_joint) / max(
            _frobenius(empirical_joint), 1e-15
        )
        weights = np.zeros(2 * self.date_count, dtype=np.float64)
        weights[self.date_count - 1] = 1.0
        weights[-1] = -1.0
        empirical_variance = _quadratic(weights, empirical_joint)
        predicted_variance = _quadratic(weights, predicted_joint)
        contrast_error = abs(predicted_variance - empirical_variance) / max(
            abs(empirical_variance), 1e-15
        )
        return operator_error, contrast_error


def validate_direct_pair_variance_order(
    positive: Mapping[str, Any], independent: Mapping[str, Any], negative: Mapping[str, Any]
) -> None:
    for identity in (
        "marginal_dgp_digest", "target_support_digest", "reference_support_digest",
        "latent_history_digest", "phase_orientation_digest",
    ):
        if len({positive.get(identity), independent.get(identity), negative.get(identity)}) != 1:
            raise SchemaError(f"direct matched pair {identity} differs across signed coupling")
    for difference_name in ("predicted_covariance_trace", "empirical_error_covariance_trace"):
        values = tuple(item.get(difference_name) for item in (positive, independent, negative))
        if any(not _number(value) for value in values) or not values[0] < values[1] < values[2]:
            raise SchemaError(
                f"direct matched pair contract requires positive < independent < negative {difference_name}"
            )


def validate_matched_pair_cohorts(value: Any) -> None:
    keys = {
        "coupling", "marginal_dgp_digest", "target_support_digest", "reference_support_digest",
        "latent_history_digest", "phase_orientation_digest", "predicted_covariance_trace",
        "empirical_error_covariance_trace",
    }
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, dict) or set(item) != keys for item in value)
        or [item["coupling"] for item in value] != ["positive", "independent", "negative"]
        or any(not _is_sha256(item.get(name)) for item in value for name in (
            "marginal_dgp_digest", "target_support_digest", "reference_support_digest",
            "latent_history_digest", "phase_orientation_digest",
        ))
    ):
        raise SchemaError("matched positive/independent/negative cohort evidence is malformed")
    validate_direct_pair_variance_order(*value)


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
            for seed_index in range(expected_seed_count(cell_id)):
                line_number += 1
                request, raw = _read_json_line(handle, Path(input_path), line_number)
                if request is None:
                    raise SchemaError(f"shard {spec.index} input is missing request {line_number}")
                expected = {
                    "schema": "dolphinrust.spatial-covariance.attempt/4",
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
            accumulator = CellAccumulator(
                preregistration, cell_id, spec.cell_ordinal_start + cell_offset,
                artifact_root=run_root,
            )
            for _ in range(expected_seed_count(cell_id)):
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
    if any(not _is_sha256(summary.get(name)) for name in (
        "request_digest", "attempt_digest", "estimate_digest", "latent_history_digest",
        "code_sha256", "binary_sha256", "preregistration_sha256",
    )):
        raise SchemaError(f"cell {cell_id} compact summary has an invalid digest")
    statuses = summary.get("status_histogram")
    if not isinstance(statuses, dict) or set(statuses) != ATTEMPT_STATUSES or any(not _integer(value) or value < 0 for value in statuses.values()) or sum(statuses.values()) != seed_count:
        raise SchemaError(f"cell {cell_id} compact status histogram is incomplete")
    expected_failures = (
        {"nondifferentiable_node": statuses["nondifferentiable_node"]}
        if statuses["nondifferentiable_node"]
        else {}
    )
    if summary.get("failure_histogram") != expected_failures:
        raise SchemaError(f"cell {cell_id} compact failure histogram is malformed")
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    expected_status = PASS
    if (
        labels["eigen_stress"] == "tied_eigenvalue"
        and labels["position"] != "masked"
    ) or expected_empty_support(cell_id):
        expected_status = NOT_EVALUABLE
    if labels["position"] == "masked":
        valid_histogram = statuses["masked_target"] == seed_count and summary["emitted_seeds"] == 0
    elif expected_empty_support(cell_id):
        valid_histogram = statuses["empty_support"] == seed_count and summary["emitted_seeds"] == 0
    elif expected_status == NOT_EVALUABLE:
        valid_histogram = statuses["singular_local_information"] == seed_count
    else:
        valid_histogram = (
            statuses["valid"] + statuses["nondifferentiable_node"] == seed_count
            and summary["emitted_seeds"] == statuses["valid"]
        )
        deterministic = seed_count == FROZEN_DETERMINISTIC_SEED_COUNT
        coverage = summary.get("coverage_95_by_date")
        interval_score = summary.get("interval_score_mean_by_date")
        interval_width = summary.get("interval_width_mean_by_date")
        date_count = preregistration["generator"]["acquisition"]["topologies"][labels["block_topology"]]["acquisition_count"]
        if deterministic:
            per_date_valid = all(summary.get(name) is None for name in (
                "coverage_95_by_date", "interval_score_mean_by_date", "interval_width_mean_by_date",
                "final_date_coverage_95", "final_date_interval_score_mean", "final_date_interval_width_mean",
            ))
            coverage_passes = True
        else:
            per_date_valid = all(
                isinstance(values, list)
                and len(values) == date_count
                and values[0] is None
                and all(_number(value) and value >= 0.0 for value in values[1:])
                for values in (coverage, interval_score, interval_width)
            )
            if per_date_valid:
                per_date_valid = all(0.0 <= value <= 1.0 for value in coverage[1:]) and (
                    summary.get("final_date_coverage_95") == coverage[-1]
                    and summary.get("final_date_interval_score_mean") == interval_score[-1]
                    and summary.get("final_date_interval_width_mean") == interval_width[-1]
                )
            coverage_passes = per_date_valid and all(
                abs(value - preregistration["thresholds"]["coverage_probability"])
                <= preregistration["thresholds"]["coverage_absolute_error_max"]
                for value in coverage[1:]
            ) and abs(
                summary["final_date_coverage_95"] - preregistration["thresholds"]["coverage_probability"]
            ) <= preregistration["thresholds"]["coverage_absolute_error_max"]
        cross_trace = summary.get("target_reference_error_covariance_trace")
        cross_sign_passes = True
        expected_sign = PAIR_SIGN[labels["pair_geometry"]]
        if not deterministic and expected_sign in {"positive", "negative"}:
            cross_sign_passes = _number(cross_trace) and (
                (cross_trace > 0.0) == (expected_sign == "positive")
            )
        passes = (
            _is_sha256(summary.get("predicted_covariance_digest"))
            and _is_sha256(summary.get("empirical_error_covariance_digest"))
            and _number(summary.get("covariance_calibration_relative_error"))
            and (deterministic or summary["covariance_calibration_relative_error"] <= preregistration["thresholds"]["covariance_calibration_relative_error_max"])
            and _number(summary.get("operator_relative_error"))
            and summary["operator_relative_error"] <= (
                preregistration["thresholds"]["deterministic_operator_relative_error_max"]
                if deterministic else preregistration["thresholds"]["stochastic_operator_relative_error_max"]
            )
            and _number(summary.get("contrast_variance_relative_error"))
            and summary["contrast_variance_relative_error"] <= preregistration["thresholds"]["contrast_variance_relative_error_max"]
            and _number(summary.get("error_bias_norm"))
            and _number(summary.get("target_reference_error_covariance_trace"))
            and cross_sign_passes
            and _number(summary.get("psd_min_eigenvalue")) and summary["psd_min_eigenvalue"] >= preregistration["thresholds"]["psd_min_eigenvalue_min"]
            and per_date_valid
            and (deterministic or coverage_passes)
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


def validate_production_parity_fixture(
    preregistration: Mapping[str, Any], run_root: Path, binding: Any, binding_sha256: Any
) -> None:
    def prefixed_sha256(value: Any) -> bool:
        return isinstance(value, str) and value.startswith("sha256:") and _is_sha256(value[7:])

    keys = {
        "schema", "hdf5_path", "sidecar_path", "hdf5_sha256", "sidecar_sha256",
        "hdf5_schema_version", "manifest_schema_version", "coupling", "seed_index",
        "factor_digest", "persisted_factor_digest", "estimator_branch",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != keys
        or binding.get("schema") != "dolphinrust.spatial-covariance.production-parity-fixture/4"
        or binding_sha256 != sha256_json(binding)
        or binding.get("hdf5_schema_version") != 4
        or binding.get("manifest_schema_version") != 3
        or binding.get("coupling") not in {"positive", "independent", "negative", "coincident", "invalid"}
        or not _integer(binding.get("seed_index"))
        or binding["seed_index"] < 0
        or binding.get("estimator_branch") not in {"emi", "evd"}
        or any(not _is_sha256(binding.get(name)) for name in (
            "hdf5_sha256", "sidecar_sha256", "factor_digest", "persisted_factor_digest",
        ))
        or binding["factor_digest"] != binding["persisted_factor_digest"]
    ):
        raise SchemaError("production parity fixture identity is malformed")
    hdf5_path = resolve_below_run_root(run_root, binding["hdf5_path"], "production parity HDF5")
    sidecar_path = resolve_below_run_root(run_root, binding["sidecar_path"], "production parity sidecar")
    hdf5_sha256, hdf5_bytes = _hash_bounded_file(
        hdf5_path,
        preregistration["execution_protocol"]["max_production_hdf5_bytes"],
        "production parity HDF5",
    )
    sidecar_bytes = _read_bounded_bytes(
        sidecar_path,
        preregistration["execution_protocol"]["max_production_sidecar_bytes"],
        "production parity sidecar",
    )
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    try:
        sidecar = json.loads(sidecar_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("production parity sidecar is malformed JSON") from exc
    if (
        hdf5_bytes <= 0
        or not sidecar_bytes
        or hdf5_sha256 != binding["hdf5_sha256"]
        or sidecar_sha256 != binding["sidecar_sha256"]
        or sidecar.get("schema_version") != binding["manifest_schema_version"]
        or sidecar.get("hdf5_file") != Path(binding["hdf5_path"]).name
        or sidecar.get("hdf5_sha256") != hdf5_sha256
        or sidecar.get("method") != "reference_specific_influence_v1"
        or sidecar.get("method_version") != 1
        or not prefixed_sha256(sidecar.get("reference_signature_digest"))
        or not prefixed_sha256(sidecar.get("l2_map_digest"))
        or not prefixed_sha256(sidecar.get("unwrap_branch_digest"))
    ):
        raise SchemaError("production parity HDF5/sidecar binding mismatch")


def _growth_exponent(points: list[tuple[int, int]]) -> float:
    x = [math.log(float(scale)) for scale, _ in points]
    y = [math.log(float(rss)) for _, rss in points]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        raise SchemaError("resource growth axis is not identifiable")
    return sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y)) / denominator


def _validate_allocation_receipt(item: Mapping[str, Any], matrix: Mapping[str, Any]) -> None:
    allocation_model = item.get("allocation_model")
    if allocation_model != {
        "model": "production-runtime-resource-receipt-v1",
        "source": "spatial_covariance_bench captured stdout",
    } or item.get("allocation_model_sha256") != sha256_json(allocation_model):
        raise SchemaError(f"resource {item.get('resource_id')} allocation model identity mismatch")
    dependency_cone = item.get("dependency_cone")
    dependency_keys = {
        "model", "tile_pixels", "date_count", "maximum_sources", "block_count",
        "maximum_dependency_depth", "reference_cone_sources",
    }
    if (
        not isinstance(dependency_cone, dict)
        or set(dependency_cone) != dependency_keys
        or dependency_cone.get("model") != "spatial-query-cone-v1"
        or dependency_cone.get("tile_pixels") != matrix["tile_pixels"]
        or dependency_cone.get("date_count") != matrix["dates"]
        or any(type(dependency_cone.get(name)) is not int for name in (
            "maximum_sources", "block_count", "maximum_dependency_depth",
            "reference_cone_sources",
        ))
        or not 0 < dependency_cone["maximum_sources"] <= matrix["tile_pixels"]
        or dependency_cone["block_count"] <= 0
        or not 0 <= dependency_cone["maximum_dependency_depth"] < dependency_cone["block_count"]
        or not 0 < dependency_cone["reference_cone_sources"] <= dependency_cone["maximum_sources"]
        or item.get("dependency_cone_sha256") != sha256_json(dependency_cone)
    ):
        raise SchemaError(f"resource {item.get('resource_id')} dependency-cone identity mismatch")
    microbatch_pixels = min(matrix["tile_pixels"], 4096)
    microbatch = item.get("microbatch")
    if microbatch != {
        "model": "bounded-microbatch-v1",
        "microbatch_pixels": microbatch_pixels,
        "batch_count": math.ceil(matrix["tile_pixels"] / microbatch_pixels),
    } or item.get("microbatch_sha256") != sha256_json(microbatch):
        raise SchemaError(f"resource {item.get('resource_id')} microbatch identity mismatch")
    components = item.get("allocation_components")
    if not isinstance(components, list) or {component.get("name") for component in components if isinstance(component, dict)} != ALLOCATION_COMPONENT_NAMES or len(components) != len(ALLOCATION_COMPONENT_NAMES):
        raise SchemaError(f"resource {item.get('resource_id')} lacks exact named allocation components")
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "name", "bytes", "source", "component_sha256"
        }:
            raise SchemaError(f"resource {item.get('resource_id')} has malformed allocation components")
        if (
            type(component["bytes"]) is not int
            or component["bytes"] <= 0
            or not isinstance(component["source"], str)
            or not component["source"]
            or component["component_sha256"]
            != sha256_json({key: value for key, value in component.items() if key != "component_sha256"})
        ):
            raise SchemaError(f"resource {item.get('resource_id')} allocation receipt/digest mismatch")


def _validate_benchmark_allocation_measurement(
    resource_id: str, measurement: Any, item: Mapping[str, Any], matrix: Mapping[str, Any]
) -> None:
    keys = {
        "block_count", "maximum_sources_per_block", "maximum_dependency_depth",
        "reference_cone_sources", "persisted_block_bytes", "scratch_bytes", "final_bytes",
        "allocation_components", "maximum_simultaneously_retained_bytes",
        "dependency_cone_bytes", "replay_reservation_bytes", "source_influence_bytes",
        "source_correlation_workspace_bytes", "source_correlation_model",
        "source_cache_peak_bytes", "admitted_block_targets",
        "tile_pixels", "date_count", "runtime_resource_receipt",
    }
    if not isinstance(measurement, dict) or set(measurement) != keys:
        raise SchemaError(f"resource {resource_id} lacks benchmark-emitted allocation evidence")
    integer_names = keys - {
        "allocation_components", "runtime_resource_receipt", "source_correlation_model"
    }
    if any(type(measurement.get(name)) is not int or measurement[name] < 0 for name in integer_names):
        raise SchemaError(f"resource {resource_id} benchmark allocation evidence is non-numeric")
    if (
        measurement["block_count"] <= 0
        or measurement["maximum_sources_per_block"] <= 0
        or measurement["maximum_sources_per_block"] > matrix["tile_pixels"]
        or measurement["reference_cone_sources"] <= 0
        or measurement["reference_cone_sources"] > matrix["tile_pixels"]
        or measurement["maximum_dependency_depth"] >= measurement["block_count"]
        or measurement["allocation_components"] != item["allocation_components"]
        or measurement["block_count"] != item["dependency_cone"]["block_count"]
        or measurement["maximum_sources_per_block"] != item["dependency_cone"]["maximum_sources"]
        or measurement["maximum_dependency_depth"] != item["dependency_cone"]["maximum_dependency_depth"]
        or measurement["reference_cone_sources"] != item["dependency_cone"]["reference_cone_sources"]
        or measurement["tile_pixels"] != matrix["tile_pixels"]
        or measurement["date_count"] != matrix["dates"]
        or measurement["dependency_cone_bytes"] <= 0
        or measurement["replay_reservation_bytes"] < measurement["dependency_cone_bytes"]
        or measurement["source_influence_bytes"] <= 0
        or measurement["source_correlation_workspace_bytes"] <= 0
        or measurement["source_correlation_model"] != "exponential_euclidean_v1"
        or measurement["admitted_block_targets"] <= 0
        or measurement["admitted_block_targets"] > matrix["tile_pixels"]
    ):
        raise SchemaError(f"resource {resource_id} benchmark allocation scope drifted")
    runtime = measurement["runtime_resource_receipt"]
    runtime_keys = {
        "working_set_byte_cap", "factor_block_high_water_bytes", "serialization_high_water_bytes",
        "fixed_l2_workspace_admission_bytes", "fixed_l2_workspace_observed_high_water_bytes",
        "replay_admission_high_water_bytes", "replay_observed_high_water_bytes",
        "provider_peak_count", "provider_peak_bytes", "preflight_provider_open_count",
        "production_provider_open_count", "operator_block_reads", "operator_block_cache_hits",
        "source_member_window_reads", "source_tile_cache_loads", "source_resolutions",
        "working_set_admission_high_water_bytes", "working_set_observed_high_water_bytes",
    }
    component_bytes = {component["name"]: component["bytes"] for component in measurement["allocation_components"]}
    expected_total = sum(component_bytes.values())
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_keys
        or any(type(runtime.get(name)) is not int or runtime[name] < 0 for name in runtime_keys)
        or component_bytes != {
            "factor_block": runtime["factor_block_high_water_bytes"],
            "serialization": runtime["serialization_high_water_bytes"],
            "fixed_l2_workspace": runtime["fixed_l2_workspace_admission_bytes"],
            "replay_reservation": runtime["replay_admission_high_water_bytes"],
        }
        or measurement["persisted_block_bytes"] != runtime["factor_block_high_water_bytes"]
        or measurement["scratch_bytes"] != runtime["serialization_high_water_bytes"]
        or measurement["replay_reservation_bytes"] != runtime["replay_admission_high_water_bytes"]
        or measurement["maximum_simultaneously_retained_bytes"] != expected_total
        or runtime["working_set_admission_high_water_bytes"] != expected_total
        or expected_total > runtime["working_set_byte_cap"]
        or expected_total > FROZEN_PROCESS_RSS_BYTES
    ):
        raise SchemaError(f"resource {resource_id} benchmark allocation arithmetic mismatch")


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
        _validate_allocation_receipt(item, matrix[resource_id])
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
            stdout_json = raw_measurement["stdout_json"]
            if (
                not isinstance(stdout_json, str)
                or not _integer(raw_measurement["stdout_bytes"])
                or raw_measurement["stdout_bytes"] != len(stdout_json.encode("utf-8"))
                or raw_measurement["stdout_sha256"] != hashlib.sha256(stdout_json.encode("utf-8")).hexdigest()
            ):
                raise SchemaError(f"resource {resource_id} captured benchmark stdout hash mismatch")
            try:
                allocation_measurement = json.loads(stdout_json)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"resource {resource_id} captured benchmark stdout is malformed") from exc
            _validate_benchmark_allocation_measurement(
                resource_id, allocation_measurement, item, expected
            )
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
        run_manifest, _, _ = _read_hashed_json_record(
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
        run_root = manifest_path.resolve(strict=True).parent
        validate_production_parity_fixture(
            preregistration,
            run_root,
            run_manifest["production_parity_fixture"],
            run_manifest["production_parity_fixture_sha256"],
        )
        validate_matched_pair_cohorts(run_manifest["matched_pair_cohorts"])
        sink.open()
        entries = run_manifest["shard_manifests"]
        if not isinstance(entries, list) or len(entries) != FROZEN_SHARD_COUNT:
            raise SchemaError("run manifest must contain exactly four ordered shard manifests")
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
            shard_manifest, _, digest = _read_hashed_json_record(
                resolved_entry,
                preregistration["execution_protocol"]["max_encoded_shard_manifest_bytes"],
                f"shard {spec.index} manifest",
            )
            if digest != entry["sha256"]:
                raise SchemaError(f"shard {spec.index} manifest hash mismatch")
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
