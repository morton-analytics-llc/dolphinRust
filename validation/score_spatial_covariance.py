#!/usr/bin/env python3
"""Fail-closed streaming scorer for the outcome-free F54-07 v3 protocol."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Mapping, Sequence

PASS = "pass"
FAIL = "fail"
NOT_EVALUABLE = "not_evaluable"
STATUSES = {PASS, FAIL, NOT_EVALUABLE}
ATTEMPT_STATUSES = {"valid", "masked_target", "tied_eigenvalue"}
HASH_RE = set("0123456789abcdef")
FROZEN_SEED_COUNT = 5000
FROZEN_CELL_COUNT = 89100
FROZEN_ATTEMPT_COUNT = 445500000
FROZEN_MAX_CELLS_PER_SHARD = 100
FROZEN_SHARD_COUNT = 891
FROZEN_MAX_SHARD_BYTES = 1 << 30
FROZEN_MAX_RECORD_BYTES = 2048
FROZEN_PROCESS_RSS_BYTES = 24 << 30
FROZEN_GENERATOR_SHA256 = "f481b639f9f092064b668a1e0f6a6945c9f5257fe6a208372da5dd2b034c2ead"
FROZEN_SCIENTIFIC_GENERATOR_SHA256 = "2328f908c9b45ea416e6202b028c73df37fb40b2c5f338a4ddce375e5206ef7c"
FROZEN_EXECUTION_SHA256 = "58e2bd28e56955b1c0d8e8e3a5e72adb6d5987461b5e34936ca1f35ca8edd13b"
FROZEN_REDUCERS_SHA256 = "65e45eb254d65a9efd04c379581296f2396ee9166fdae6dad1be4058dad7eb5c"
FROZEN_MATRIX_SHA256 = "9133bbac234fe511df7cce8e154bdf9134a1d2af699b8127c34522135cb50939"
FROZEN_RECEIPT_SHA256 = "f2e53c52b485af7bb425c4da84c4d711de74acb80e6fb19c84850d35bc866f38"
FROZEN_HASH_FIELDS_SHA256 = "ac81a3c151c46a953aa3ad279618addadadc57e9cdec83ae12f2cabfe2f4b12a"
FROZEN_RESOURCE_SAMPLING_SHA256 = "76e75e7dd6fe32c3751c8f230f4aa7e53df44800190f6877a685189eb56bc7d7"
FROZEN_RESOURCE_MATRIX_SHA256 = "acfe4ba22b2fcb39496d5688628246c1be9fb488da4d451d2e2c726cbbc0c4b3"
FROZEN_CELL_POLICY_SHA256 = "f3e8f63462ae82c5dc43cebeecd49f0236b4e0804460ada79cf24c12e8096c90"
FROZEN_V2_PREREGISTRATION_SHA256 = "6b897f038176a7ade6a2e27561b64e96885651db836db85207c9ce4c518f00d8"
FROZEN_RESOURCE_IDS = ("tile_256_dates_13", "tile_256_dates_26", "tile_256_dates_52")
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
}
INPUT_KEYS = {"schema", "cell_id", "cell_ordinal", "seed_index", "seed_sha256", *DIMENSION_NAMES}
ATTEMPT_HASH_FIELDS = (
    "seed_sha256", "raw_input_sha256", "truth_sha256", "operator_hash", "variance_hash", "emission_hash",
    "date_axis_sha256", "generator_hash", "config_hash", "source_model_hash", "target_support_sha256",
    "reference_support_sha256",
)
SHARD_MANIFEST_KEYS = {
    "schema", "schema_version", "shard_index", "cell_ordinal_start", "cell_ordinal_end_exclusive",
    "expected_cells", "expected_attempts", "input_path", "output_path", "input_sha256", "output_sha256",
    "input_bytes", "output_bytes", "input_records", "output_records", "preregistration_sha256", "code_sha256",
    "binary_sha256", "generator_protocol_sha256", "elapsed_seconds", "peak_rss_bytes", "committed",
}
RUN_MANIFEST_KEYS = {
    "schema", "schema_version", "preregistration_sha256", "code_sha256", "binary_sha256",
    "generator_protocol_sha256", "performance_probe", "resources", "shard_manifests", "result_root_sha256",
}
RESOURCE_KEYS = {
    "resource_id", "status", "rss_bytes", "growth_class", "resource_hash", "config_hash", "binary_hash", "os",
    "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "sampling_interval_ms", "warmup_runs",
    "measured_repetitions", "tool_versions", "growth_observation", "growth_regression", "acceptance",
}


class SchemaError(ValueError):
    """The preregistration or receipt is not the frozen contract."""


@dataclass(frozen=True)
class ShardSpec:
    index: int
    cell_ordinal_start: int
    cell_ordinal_end_exclusive: int
    cell_ids: tuple[str, ...]

    @property
    def expected_attempts(self) -> int:
        return len(self.cell_ids) * FROZEN_SEED_COUNT


def load_preregistration(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SchemaError("preregistration root must be an object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, byte_limit: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            if byte_limit is not None and size > byte_limit:
                raise SchemaError(f"{path} exceeds the frozen uncompressed byte cap")
            digest.update(chunk)
    return digest.hexdigest(), size


def preregistration_digest(preregistration: Mapping[str, Any]) -> str:
    return sha256_json(preregistration)


def seed_schedule_digest(preregistration: Mapping[str, Any]) -> str:
    return sha256_json(preregistration.get("seed_schedule"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HASH_RE


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


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
    if raw.get("covariance_shape") != "N_by_N_per_topology" or "C_ab" not in raw.get("covariance_formula", "") or raw.get("sampler") != "Z=mu+L*(U+iV)/sqrt(2), U,V iid N(0,I), C=L L^H lower_hermitian_cholesky":
        errors.append("raw generator must define the frozen proper-complex covariance and sampler")
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
    if preregistration.get("schema") != "dolphinrust.spatial_covariance.preregistration" or preregistration.get("schema_version") != 3:
        errors.append("preregistration must use the F54-07 v3 schema")
    if preregistration.get("status") != "preregistered" or preregistration.get("outcomes_present") is not False:
        errors.append("preregistration must remain outcome-free and preregistered")
    supersedes = preregistration.get("supersedes")
    if supersedes != {"schema_version": 2, "canonical_preregistration_sha256": FROZEN_V2_PREREGISTRATION_SHA256, "outcomes_present": False, "reason": "v2 monolithic receipt cannot satisfy bounded streaming evidence; scientific design unchanged"}:
        errors.append("v3 must bind and outcome-free supersede the exact v2 preregistration")
    dimensions = preregistration.get("dimensions")
    if not isinstance(dimensions, dict) or tuple(dimensions) != DIMENSION_NAMES:
        errors.append("dimensions must contain the nine frozen axes in order")
    else:
        for name in DIMENSION_NAMES:
            if _dimension_values(preregistration, name) != FROZEN_DIMENSION_IDS[name]:
                errors.append(f"dimension {name} does not match the frozen matrix")
    schedule = preregistration.get("seed_schedule")
    if not isinstance(schedule, dict) or schedule.get("attempted_seeds_per_cell") != FROZEN_SEED_COUNT or schedule.get("no_top_up") is not True:
        errors.append("seed schedule must freeze 5000 attempts and prohibit top-up")
    if preregistration.get("thresholds") != FROZEN_THRESHOLDS:
        errors.append("thresholds differ from immutable F54-07 thresholds")
    for field_name, frozen_hash, message in (
        ("matrix_contract", FROZEN_MATRIX_SHA256, "matrix contract must freeze 89100 cells and 445500000 attempts including source_process"),
        ("execution_protocol", FROZEN_EXECUTION_SHA256, "execution protocol differs from the frozen v3 shard contract"),
        ("cell_reducers", FROZEN_REDUCERS_SHA256, "cell reducers or denominators differ from the frozen v3 contract"),
        ("receipt_contract", FROZEN_RECEIPT_SHA256, "receipt contract differs from the frozen v3 contract"),
        ("hash_fields", FROZEN_HASH_FIELDS_SHA256, "receipt identity fields differ from the frozen contract"),
        ("resource_sampling", FROZEN_RESOURCE_SAMPLING_SHA256, "resource sampling differs from the frozen contract"),
        ("resource_matrix", FROZEN_RESOURCE_MATRIX_SHA256, "resource matrix differs from the frozen contract"),
        ("cell_policy", FROZEN_CELL_POLICY_SHA256, "cell status policy differs from the frozen contract"),
    ):
        value = preregistration.get(field_name)
        if value is None or sha256_json(value) != frozen_hash:
            errors.append(message)
    generator = preregistration.get("generator")
    if not isinstance(generator, dict) or sha256_json(generator) != FROZEN_GENERATOR_SHA256:
        errors.append("generator parameters/protocol differ from the frozen v3 generator")
    elif sha256_json(_scientific_generator(generator)) != FROZEN_SCIENTIFIC_GENERATOR_SHA256:
        errors.append("scientific generator differs from the outcome-free v2 design")
    execution = preregistration.get("execution_protocol", {})
    worst_case_output = execution.get("max_cells_per_shard", 0) * FROZEN_SEED_COUNT * execution.get("max_encoded_output_record_bytes", 0)
    if worst_case_output > execution.get("max_uncompressed_output_bytes", -1):
        errors.append("worst-case shard encoding exceeds the frozen output byte cap")
    if execution.get("process_rss_bytes_max") != FROZEN_PROCESS_RSS_BYTES:
        errors.append("execution process cap must equal the frozen 24 GiB resource threshold")
    _validate_executable_generator(preregistration, errors)
    if errors:
        raise SchemaError("; ".join(errors))


def iter_expected_cell_ids(preregistration: Mapping[str, Any]) -> Iterator[str]:
    validate_preregistration(preregistration)
    values = [_dimension_values(preregistration, name) for name in DIMENSION_NAMES]
    return ("|".join(labels) for labels in itertools.product(*values))


def expected_cell_ids(preregistration: Mapping[str, Any]) -> List[str]:
    return list(iter_expected_cell_ids(preregistration))


def iter_shard_specs(preregistration: Mapping[str, Any]) -> Iterator[ShardSpec]:
    cells = iter_expected_cell_ids(preregistration)
    ordinal = 0
    for shard_index in range(FROZEN_SHARD_COUNT):
        cell_ids = tuple(itertools.islice(cells, FROZEN_MAX_CELLS_PER_SHARD))
        if not cell_ids:
            raise SchemaError("frozen shard count exceeds the matrix")
        end = ordinal + len(cell_ids)
        yield ShardSpec(shard_index, ordinal, end, cell_ids)
        ordinal = end
    if next(cells, None) is not None or ordinal != FROZEN_CELL_COUNT:
        raise SchemaError("frozen shards do not cover exactly 89100 cells")


def _expected_seed_hash(preregistration: Mapping[str, Any], cell_id: str, index: int) -> str:
    value = f"{preregistration['seed_schedule']['validation_seed']}||{cell_id}||{index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expected_coordinates(preregistration: Mapping[str, Any], cell_id: str) -> tuple[list[int], list[int]]:
    labels = dict(zip(DIMENSION_NAMES, cell_id.split("|")))
    window = preregistration["generator"]["coordinates"]["window_stride"][f"{labels['half_window']}|{labels['stride']}"]
    target = window["target_by_position"][labels["position"]]
    delta = window["reference_delta_by_pair_geometry"][labels["pair_geometry"]]
    return target, [target[0] + delta[0], target[1] + delta[1]]


def realized_overlap_jaccard(target_count: Any, reference_count: Any, intersection_count: Any, union_count: Any) -> float:
    counts = (target_count, reference_count, intersection_count, union_count)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
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
    expected_seed_count: int = FROZEN_SEED_COUNT
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
    statuses: dict[str, int] = field(default_factory=lambda: {status: 0 for status in ATTEMPT_STATUSES})
    field_digests: dict[str, Any] = field(default_factory=lambda: {name: hashlib.sha256() for name in ("operator_hash", "variance_hash", "emission_hash", "truth_sha256")})

    def add(self, attempt: Any) -> None:
        if not isinstance(attempt, dict) or set(attempt) != ATTEMPT_KEYS:
            raise SchemaError(f"cell {self.cell_id} has malformed or unknown per-attempt evidence")
        if attempt.get("schema") != "dolphinrust.spatial-covariance.attempt-receipt/3":
            raise SchemaError(f"cell {self.cell_id} has the wrong attempt schema")
        if attempt.get("cell_id") != self.cell_id or attempt.get("cell_ordinal") != self.cell_ordinal:
            raise SchemaError(f"cell {self.cell_id} has an out-of-order cell identity")
        if attempt.get("seed_index") != self.next_seed_index or self.next_seed_index >= self.expected_seed_count:
            raise SchemaError(f"cell {self.cell_id} has a missing, duplicate, top-up, or out-of-order seed")
        if attempt.get("seed_sha256") != _expected_seed_hash(self.preregistration, self.cell_id, self.next_seed_index):
            raise SchemaError(f"cell {self.cell_id} has a seed derivation mismatch")
        if any(not _is_sha256(attempt.get(field_name)) for field_name in ATTEMPT_HASH_FIELDS):
            raise SchemaError(f"cell {self.cell_id} has an invalid identity hash")
        self._validate_scope(attempt)
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
        if attempt.get("target_coordinate") != target or attempt.get("reference_coordinate") != reference:
            raise SchemaError(f"cell {self.cell_id} has a coordinate identity mismatch")
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

    def _validate_masked(self, attempt: Mapping[str, Any]) -> None:
        if attempt.get("status") != "masked_target" or attempt.get("emitted") is not False or attempt.get("factor_emitted") is not False:
            raise SchemaError(f"cell {self.cell_id} masked attempt must abstain")
        metrics = ("operator_relative_error", "contrast_variance_reference", "contrast_variance_relative_error", "psd_min_eigenvalue", "covered_95", "interval_score", "interval_width", "signed_cross_influence")
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
        if expected_sign in {"positive", "negative"} and (not _number(influence) or (influence > 0) != (expected_sign == "positive")):
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
            "cell_id": self.cell_id, **labels, "status": status,
            "not_evaluable_reason": "tied_eigenvalue" if status == NOT_EVALUABLE else None,
            "attempted_seeds": self.expected_seed_count, "emitted_seeds": self.emitted, "top_up_seeds": 0,
            "target_source_count_total": self.target_total, "reference_source_count_total": self.reference_total,
            "intersection_source_count_total": self.intersection_total, "union_source_count_total": self.union_total,
            "realized_overlap_jaccard_mean": self.overlap_total / self.expected_seed_count,
            "effective_looks_fraction": self.effective_looks_total / self.expected_seed_count,
            "operator_relative_error": self.max_operator_error, "contrast_variance_relative_error": self.max_variance_error,
            "variance_evaluable": self.max_variance_error is not None, "psd_min_eigenvalue": self.min_psd_eigenvalue,
            "coverage_95": coverage, "emission_rate": self.emitted / self.expected_seed_count,
            "operator_hash": self.field_digests["operator_hash"].hexdigest(),
            "variance_hash": self.field_digests["variance_hash"].hexdigest(),
            "emission_hash": self.field_digests["emission_hash"].hexdigest(),
            "truth_hash": self.field_digests["truth_sha256"].hexdigest(),
        }

    def _numeric_status(self, labels: Mapping[str, str]) -> str:
        if self.max_operator_error is None or self.min_psd_eigenvalue is None or self.coverage_denominator != self.emitted:
            return FAIL
        thresholds = self.preregistration["thresholds"]
        deterministic = labels["pair_geometry"] in {"coincident", *DISJOINT_GEOMETRIES}
        operator_limit = thresholds["deterministic_operator_relative_error_max"] if deterministic else thresholds["stochastic_operator_relative_error_max"]
        coverage = self.covered / self.coverage_denominator if self.coverage_denominator else math.nan
        passes = (
            self.max_operator_error <= operator_limit
            and (self.max_variance_error is None or self.max_variance_error <= thresholds["contrast_variance_relative_error_max"])
            and self.min_psd_eigenvalue >= thresholds["psd_min_eigenvalue_min"]
            and abs(coverage - thresholds["coverage_probability"]) <= thresholds["coverage_absolute_error_max"]
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
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SchemaError(f"shard {spec.index} manifest scope/order/count drifted")
    for field_name in ("input_sha256", "output_sha256", "code_sha256", "binary_sha256", "generator_protocol_sha256"):
        if not _is_sha256(manifest.get(field_name)):
            raise SchemaError(f"shard {spec.index} manifest has an invalid {field_name}")
    if manifest.get("input_records") != spec.expected_attempts or manifest.get("output_records") != spec.expected_attempts:
        raise SchemaError(f"shard {spec.index} violates one-input-one-output")
    for field_name in ("input_bytes", "output_bytes"):
        value = manifest.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > FROZEN_MAX_SHARD_BYTES:
            raise SchemaError(f"shard {spec.index} exceeds the uncompressed byte cap")
    if not _number(manifest.get("elapsed_seconds")) or manifest["elapsed_seconds"] < 0:
        raise SchemaError(f"shard {spec.index} has invalid elapsed time")
    if not isinstance(manifest.get("peak_rss_bytes"), int) or manifest["peak_rss_bytes"] > FROZEN_PROCESS_RSS_BYTES:
        raise SchemaError(f"shard {spec.index} exceeds the process RSS cap")
    if any(Path(manifest[field_name]).is_absolute() or ".." in Path(manifest[field_name]).parts for field_name in ("input_path", "output_path")):
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
                if set(request) != INPUT_KEYS or request != expected:
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
    validate_input_shard(preregistration, Path(run_root) / manifest["input_path"], manifest, spec)
    output_path = Path(run_root) / manifest["output_path"]
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


def _validate_performance_probe(preregistration: Mapping[str, Any], probe: Any, code_sha256: str, binary_sha256: str) -> None:
    required = {"schema", "schema_version", "outcomes_persisted", "seed_counts", "cell_classes", "attempts_per_second", "peak_rss_bytes", "target_wall_seconds", "reserve_fraction", "projected_serial_seconds", "derived_concurrency", "code_sha256", "binary_sha256", "config_sha256"}
    if not isinstance(probe, dict) or set(probe) != required:
        raise SchemaError("performance probe receipt has unknown or missing fields")
    frozen = preregistration["execution_protocol"]["performance_probe"]
    if probe["schema"] != "dolphinrust.spatial-covariance.performance-probe" or probe["schema_version"] != 1 or probe["outcomes_persisted"] is not False:
        raise SchemaError("performance probe must be outcome-discarding")
    if probe["seed_counts"] != frozen["seed_counts"] or probe["cell_classes"] != frozen["required_cell_classes"]:
        raise SchemaError("performance probe does not cover the frozen classes/seeds")
    if probe["code_sha256"] != code_sha256 or probe["binary_sha256"] != binary_sha256 or probe["config_sha256"] != sha256_json(preregistration["generator"]):
        raise SchemaError("performance probe scope identity mismatch")
    if any(not _number(probe.get(field_name)) or probe[field_name] <= 0 for field_name in ("attempts_per_second", "target_wall_seconds", "projected_serial_seconds")):
        raise SchemaError("performance probe rates/timing must be finite and positive")
    projected = FROZEN_ATTEMPT_COUNT / probe["attempts_per_second"]
    if not math.isclose(probe["projected_serial_seconds"], projected, rel_tol=1e-12, abs_tol=1e-9):
        raise SchemaError("performance probe serial projection does not match frozen attempt count/rate")
    if probe["reserve_fraction"] != frozen["reserve_fraction"] or not isinstance(probe.get("derived_concurrency"), int) or probe["derived_concurrency"] < 1:
        raise SchemaError("performance probe concurrency receipt is invalid")
    expected = math.ceil(probe["projected_serial_seconds"] / (probe["target_wall_seconds"] * (1.0 - probe["reserve_fraction"])))
    if probe["derived_concurrency"] != expected:
        raise SchemaError("performance probe derived concurrency does not match the frozen formula")
    if not isinstance(probe.get("peak_rss_bytes"), int) or probe["peak_rss_bytes"] > FROZEN_PROCESS_RSS_BYTES:
        raise SchemaError("performance probe exceeds the process RSS cap")


def _validate_resources(preregistration: Mapping[str, Any], resources: Any, binary_sha256: str) -> list[str]:
    if not isinstance(resources, list):
        raise SchemaError("resources must be a per-resource list")
    by_id = {item.get("resource_id"): item for item in resources if isinstance(item, dict)}
    if set(by_id) != set(FROZEN_RESOURCE_IDS) or len(resources) != len(FROZEN_RESOURCE_IDS):
        raise SchemaError("resource receipts must contain exactly the three frozen resource cells")
    statuses = []
    sampling = preregistration["resource_sampling"]
    for resource_id in FROZEN_RESOURCE_IDS:
        item = by_id[resource_id]
        if set(item) != RESOURCE_KEYS:
            raise SchemaError(f"resource {resource_id} has unknown or missing fields")
        statuses.append(item["status"])
        if item["status"] not in STATUSES or not isinstance(item["rss_bytes"], int) or item["rss_bytes"] > FROZEN_PROCESS_RSS_BYTES:
            raise SchemaError(f"resource {resource_id} has invalid status/RSS")
        if item["growth_class"] != "linear" or item["binary_hash"] != binary_sha256 or item["config_hash"] != sha256_json(preregistration["generator"]):
            raise SchemaError(f"resource {resource_id} identity/growth mismatch")
        provenance = ("os", "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "sampling_interval_ms", "warmup_runs", "measured_repetitions", "tool_versions", "growth_regression", "acceptance")
        if any(item[field_name] != sampling[field_name] for field_name in provenance):
            raise SchemaError(f"resource {resource_id} sampling provenance mismatch")
        if not _is_sha256(item["resource_hash"]):
            raise SchemaError(f"resource {resource_id} hash is invalid")
    return statuses


class _CellSummarySink:
    def __init__(self, destination: Path | None):
        self.destination = Path(destination) if destination is not None else None
        self.partial = self.destination.with_name(self.destination.name + ".partial") if self.destination is not None else None
        self.handle = None
        self.digest = hashlib.sha256()
        self.byte_count = 0
        self.record_count = 0

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
        if self.byte_count + len(encoded) > FROZEN_MAX_SHARD_BYTES:
            raise SchemaError("cell-summary JSONL exceeds the frozen 1 GiB cap")
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
        if Path(manifest_path).name.endswith(preregistration["execution_protocol"]["partial_suffix"]):
            raise SchemaError("partial run manifests are not admissible")
        with Path(manifest_path).open(encoding="utf-8") as handle:
            run_manifest = json.load(handle)
        if not isinstance(run_manifest, dict) or set(run_manifest) != RUN_MANIFEST_KEYS:
            raise SchemaError("run manifest has unknown or missing fields")
        if run_manifest["schema"] != "dolphinrust.spatial-covariance.run-manifest" or run_manifest["schema_version"] != 3:
            raise SchemaError("run manifest must use schema v3")
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
        run_root = Path(manifest_path).parent
        manifest_digests = []
        cell_count = 0
        any_failed = FAIL in resource_statuses
        any_not_evaluable = NOT_EVALUABLE in resource_statuses
        for spec, entry in zip(iter_shard_specs(preregistration), entries):
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise SchemaError(f"shard {spec.index} run-manifest entry is malformed")
            entry_path = Path(entry["path"])
            if entry_path.is_absolute() or ".." in entry_path.parts or entry_path.name.endswith(preregistration["execution_protocol"]["partial_suffix"]):
                raise SchemaError(f"shard {spec.index} manifest path escapes the run root")
            digest, _ = sha256_file(run_root / entry_path)
            if digest != entry["sha256"]:
                raise SchemaError(f"shard {spec.index} manifest hash mismatch")
            with (run_root / entry_path).open(encoding="utf-8") as handle:
                shard_manifest = json.load(handle)
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
    """Reject legacy aggregate receipts; v3 scoring requires a file-backed run manifest."""
    try:
        validate_preregistration(preregistration)
    except SchemaError as exc:
        return {"status": FAIL, "errors": [str(exc)]}
    return {"status": FAIL, "errors": ["aggregate receipts are rejected; provide a v3 run manifest with complete attempt shards"]}


validate_receipt = score_receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_manifest", type=Path)
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("spatial_covariance_preregistration.json"))
    parser.add_argument("--cell-summary-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_run_manifest(load_preregistration(args.preregistration), args.run_manifest, args.cell_summary_jsonl), indent=2, sort_keys=True))
