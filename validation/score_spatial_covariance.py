#!/usr/bin/env python3
"""Fail-closed scorer for the outcome-free F54-07 v2 receipt contract."""

import hashlib
import itertools
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PASS = "pass"
FAIL = "fail"
NOT_EVALUABLE = "not_evaluable"
STATUSES = {PASS, FAIL, NOT_EVALUABLE}
HASH_RE = set("0123456789abcdef")
FROZEN_SEED_COUNT = 5000
FROZEN_GENERATOR_SHA256 = "8332973bb64ff7fbb3211106fb447c373d5dd6000ed0122df57999cd7e1398fe"
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
    "resource_rss_bytes_max": 25769803776,
    "resource_growth": "area_or_dates_linear; no quadratic axis",
    "resource_buffer_policy": "no_tile_area_times_s_b_times_d_b_squared_buffer",
}
RECEIPT_HASHES = {"code_sha256", "fixture_sha256", "operator_sha256", "variance_sha256", "resource_sha256", "generator_protocol_sha256", "config_sha256", "source_model_sha256", "result_sha256", "binary_sha256"}
RECEIPT_KEYS = {"schema", "schema_version", "preregistration_sha256", "seed_schedule_sha256", "protocol", "binary", "hashes", "cells", "resources"}
CELL_KEYS = {"cell_id", *DIMENSION_NAMES, "status", "not_evaluable_reason", "attempted_seeds", "emitted_seeds", "top_up_seeds", "target_coordinate", "reference_coordinate", "acquisition_count", "date_axis_sha256", "realized_overlap_percent", "signed_influence_sign", "effective_looks_fraction", "effective_looks_application", "generator_hash", "truth_hash", "operator_relative_error", "contrast_variance_reference", "variance_evaluable", "contrast_variance_relative_error", "psd_min_eigenvalue", "coverage_95", "emission_rate", "operator_hash", "variance_hash", "psd_hash", "coverage_hash", "emission_hash", "attempts"}
ATTEMPT_KEYS = {"seed_index", "seed_sha256", "status", "emitted", "raw_input_sha256", "truth_sha256", "operator_hash", "variance_hash", "emission_hash", "date_axis_sha256", "generator_hash", "config_hash", "source_model_hash", "target_coordinate", "reference_coordinate", "realized_overlap_percent", "signed_cross_influence", "signed_influence_sign", "effective_looks_fraction", "effective_looks_application", "operator_relative_error", "contrast_variance_relative_error", "psd_min_eigenvalue", "covered_95", "interval_score", "interval_width"}
RESOURCE_KEYS = {"resource_id", "status", "rss_bytes", "growth_class", "resource_hash", "config_hash", "binary_hash", "os", "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "sampling_interval_ms", "warmup_runs", "measured_repetitions", "tool_versions", "growth_observation", "growth_regression", "acceptance"}


class SchemaError(ValueError):
    """The preregistration or receipt is not the frozen contract."""


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


def preregistration_digest(preregistration: Mapping[str, Any]) -> str:
    return sha256_json(preregistration)


def seed_schedule_digest(preregistration: Mapping[str, Any]) -> str:
    return sha256_json(preregistration.get("seed_schedule"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HASH_RE


def _dimension_values(preregistration: Mapping[str, Any], name: str) -> Sequence[str]:
    values = preregistration.get("dimensions", {}).get(name, [])
    return tuple(item.get("id") for item in values if isinstance(item, dict))


def _planner_rows(num_slc: int, ministack_size: int, max_num_compressed: int) -> list[dict[str, int]]:
    rows = []
    for block_id, real_start in enumerate(range(0, num_slc, ministack_size)):
        rows.append({"block_id": block_id, "num_compressed": min(block_id, max_num_compressed), "real_start": real_start, "num_real": min(ministack_size, num_slc - real_start)})
    return rows


def _validate_executable_generator(preregistration: Mapping[str, Any], errors: List[str]) -> None:
    generator = preregistration.get("generator")
    if not isinstance(generator, dict):
        return
    raw = generator.get("raw_proper_complex", {})
    if raw.get("covariance_shape") != "N_by_N_per_topology" or "C_ab" not in raw.get("covariance_formula", "") or raw.get("sampler") != "Z=mu+L*(U+iV)/sqrt(2), U,V iid N(0,I), C=L L^H lower_hermitian_cholesky":
        errors.append("raw generator must define an exact N-by-N proper-complex covariance and sampler")
    source = generator.get("source_centered_empirical", {})
    if source.get("covariance_definition") != "E[Z Z*]/n" or source.get("mean") != "zero; no sample-mean subtraction" or source.get("floor_application") != "after_zero_mean_covariance_and_shrinkage":
        errors.append("source model must be zero-mean E[Z Z*]/n with post-shrinkage floor")
    acquisition = generator.get("acquisition", {})
    planner = acquisition.get("planner", {})
    if planner != {"crate": "dolphin-stack", "planner": "MiniStackPlanner::plan", "ministack_size": 5, "max_num_compressed": 2, "output_reference_idx": 0, "compressed_slc_plan": "always_first", "partial_policy": "min(num_slc-start,ministack_size)"}:
        errors.append("planner identity/parameters are not frozen to MiniStackPlanner")
    for topology, spec in acquisition.get("topologies", {}).items():
        expected = _planner_rows(spec.get("acquisition_count", 0), spec.get("ministack_size", 0), spec.get("max_num_compressed", 0))
        if spec.get("expected_blocks") != expected or len(spec.get("date_axis", [])) != spec.get("acquisition_count", -1):
            errors.append("topology %s does not reproduce the frozen MiniStackPlanner output" % topology)
        dates = spec.get("date_axis", [])
        if any(dates[index] >= dates[index + 1] for index in range(len(dates) - 1)):
            errors.append("topology %s date axis is not strictly ascending" % topology)
        try:
            parsed_dates = [date.fromisoformat(value) for value in dates]
            if parsed_dates and parsed_dates[0].isoformat() != acquisition.get("date_origin"):
                errors.append("topology %s date axis has the wrong origin" % topology)
            if any((parsed_dates[index + 1] - parsed_dates[index]).days != acquisition.get("cadence_days") for index in range(len(parsed_dates) - 1)):
                errors.append("topology %s date axis does not use the frozen cadence" % topology)
        except (TypeError, ValueError):
            errors.append("topology %s date axis contains a non-ISO date" % topology)
        if spec.get("partial_tail_count") != (expected[-1]["num_real"] if expected and expected[-1]["num_real"] < spec.get("ministack_size", 0) else 0):
            errors.append("topology %s partial-tail count is not derived from the planner" % topology)
    coordinates = generator.get("coordinates", {})
    expected_window_stride = {"%s|%s" % (hw, stride) for hw in FROZEN_DIMENSION_IDS["half_window"] for stride in FROZEN_DIMENSION_IDS["stride"]}
    if set(coordinates.get("window_stride", {})) != expected_window_stride:
        errors.append("coordinates must provide every half-window/stride realization")
    for key, spec in coordinates.get("window_stride", {}).items():
        shape = spec.get("support_shape", [])
        if shape != [2 * spec.get("half_window", [0, 0])[0] + 1, 2 * spec.get("half_window", [0, 0])[1] + 1]:
            errors.append("coordinate %s support shape does not match production half-window" % key)
        if set(spec.get("reference_delta_by_pair_geometry", {})) != set(PAIR_OVERLAP):
            errors.append("coordinate %s is missing a pair-geometry realization" % key)
    overlap_fixture = coordinates.get("overlap_fixture", {})
    target_units = set(overlap_fixture.get("target_units", []))
    if target_units != {0, 1, 2, 3}:
        errors.append("overlap fixture target support is not the frozen four-unit support")
    if set(overlap_fixture.get("reference_units_by_geometry", {})) != set(PAIR_OVERLAP):
        errors.append("overlap fixture is missing a pair-geometry realization")
    for geometry, units in overlap_fixture.get("reference_units_by_geometry", {}).items():
        if round(100 * len(target_units.intersection(units)) / 4) != PAIR_OVERLAP.get(geometry, -1):
            errors.append("overlap fixture does not realize declared percentage for %s" % geometry)
    neighbors = generator.get("neighbor_generation", {})
    if neighbors.get("full_half_window") is not True or neighbors.get("offset_order") != "neighbor_grid_row_major_from_clamped_start" or neighbors.get("glrt", {}).get("alpha") != 0.001 or neighbors.get("ks", {}).get("alpha") != 0.001 or neighbors.get("fixed_support_reuse") is not True:
        errors.append("GLRT/KS support contract does not match production algorithms")
    if generator.get("effective_looks", {}).get("application") != "source_factor_divided_by_sqrt_fraction" or generator.get("effective_looks", {}).get("recompute_per_cell") is not True:
        errors.append("effective-look application or support-union recomputation is not frozen")
    supported = generator.get("supported", {})
    if supported.get("not_evaluable_if") != ["tied_eigenvalue"] or "missing_attempt_record" not in supported.get("receipt_failure_if", []):
        errors.append("missing attempts must fail receipt validation; only tied eigenvalue is not-evaluable")
    sampling = preregistration.get("resource_sampling", {})
    if not all(key in sampling for key in ("os", "rss_field", "warmup_runs", "measured_repetitions", "growth_regression", "acceptance")):
        errors.append("resource sampling must freeze OS, RSS source, repetitions, regression, and acceptance")


def validate_preregistration(preregistration: Mapping[str, Any]) -> None:
    errors: List[str] = []
    if preregistration.get("schema") != "dolphinrust.spatial_covariance.preregistration":
        errors.append("schema is not the F54-07 preregistration schema")
    if preregistration.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if preregistration.get("status") != "preregistered" or preregistration.get("outcomes_present") is not False:
        errors.append("preregistration must be outcome-free and preregistered")
    dimensions = preregistration.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSION_NAMES):
        errors.append("dimensions must contain exactly the nine frozen axes")
    else:
        for name in DIMENSION_NAMES:
            actual = _dimension_values(preregistration, name)
            if actual != FROZEN_DIMENSION_IDS[name]:
                errors.append("dimension %s does not match the frozen matrix" % name)
            if len(actual) != len(set(actual)):
                errors.append("dimension %s contains duplicate ids" % name)
    if preregistration.get("thresholds") != FROZEN_THRESHOLDS:
        errors.append("thresholds differ from immutable F54-07 thresholds")
    schedule = preregistration.get("seed_schedule")
    if not isinstance(schedule, dict) or schedule.get("attempted_seeds_per_cell") != FROZEN_SEED_COUNT or schedule.get("no_top_up") is not True:
        errors.append("seed schedule must freeze 5000 attempts and prohibit top-up")
    hashes = preregistration.get("hash_fields")
    if not isinstance(hashes, dict) or hashes.get("algorithm") != "sha256":
        errors.append("hash_fields must use sha256")
    generator = preregistration.get("generator")
    if not isinstance(generator, dict) or sha256_json(generator) != FROZEN_GENERATOR_SHA256:
        errors.append("generator parameters/protocol differ from the frozen v2 generator")
    if not isinstance(preregistration.get("resource_sampling"), dict):
        errors.append("resource_sampling is required")
    _validate_executable_generator(preregistration, errors)
    if errors:
        raise SchemaError("; ".join(errors))


def expected_cell_ids(preregistration: Mapping[str, Any]) -> List[str]:
    validate_preregistration(preregistration)
    return ["|".join(values) for values in itertools.product(*[_dimension_values(preregistration, name) for name in DIMENSION_NAMES])]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_hashes(value: Mapping[str, Any], fields: Iterable[str], prefix: str, errors: List[str]) -> None:
    for field in fields:
        if not _is_sha256(value.get(field)):
            errors.append("%s.%s must be a lowercase sha256 digest" % (prefix, field))


def _expected_coordinates(preregistration: Mapping[str, Any], cell: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    coordinates = preregistration["generator"]["coordinates"]
    key = "%s|%s" % (cell.get("half_window"), cell.get("stride"))
    window_stride = coordinates["window_stride"].get(key, {})
    target = window_stride.get("target_by_position", {}).get(cell.get("position"), [-1, -1])
    delta = window_stride.get("reference_delta_by_pair_geometry", {}).get(cell.get("pair_geometry"), [0, 0])
    return target, [target[0] + delta[0], target[1] + delta[1]]


PAIR_OVERLAP = {
    "coincident": 100,
    "shared_75_positive": 75,
    "shared_75_negative": 75,
    "shared_50_positive": 50,
    "shared_50_negative": 50,
    "shared_25_positive": 25,
    "shared_25_negative": 25,
    "disjoint_immediate": 0,
    "disjoint_after_depth_1": 0,
    "disjoint_after_depth_2": 0,
    "disjoint_after_depth_4": 0,
}
PAIR_SIGN = {
    "coincident": "zero",
    "shared_75_positive": "positive",
    "shared_75_negative": "negative",
    "shared_50_positive": "positive",
    "shared_50_negative": "negative",
    "shared_25_positive": "positive",
    "shared_25_negative": "negative",
    "disjoint_immediate": "none",
    "disjoint_after_depth_1": "none",
    "disjoint_after_depth_2": "none",
    "disjoint_after_depth_4": "none",
}


def _expected_date_axis(preregistration: Mapping[str, Any], topology: str) -> list[str]:
    return preregistration["generator"]["acquisition"]["topologies"].get(topology, {}).get("date_axis", [])


def _expected_seed_hash(preregistration: Mapping[str, Any], cell_id: str, index: int) -> str:
    value = "%s||%s||%d" % (preregistration["seed_schedule"]["validation_seed"], cell_id, index)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_attempts(cell: Mapping[str, Any], expected_id: str, preregistration: Mapping[str, Any], errors: List[str]) -> None:
    attempts = cell.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != FROZEN_SEED_COUNT:
        errors.append("cell %s requires exactly 5000 complete per-attempt records" % expected_id)
        return
    expected_target, expected_reference = _expected_coordinates(preregistration, cell)
    date_digest = sha256_json(_expected_date_axis(preregistration, cell["block_topology"]))
    expected_generator = sha256_json(preregistration["generator"])
    expected_config = sha256_json(preregistration["generator"])
    expected_source_model = sha256_json(preregistration["generator"]["source_centered_empirical"])
    expected_overlap = PAIR_OVERLAP.get(cell.get("pair_geometry"), -1)
    expected_sign = PAIR_SIGN.get(cell.get("pair_geometry"), "invalid")
    seen = set()
    emitted = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            errors.append("cell %s has a non-object attempt" % expected_id)
            continue
        if set(attempt) - ATTEMPT_KEYS or ATTEMPT_KEYS - set(attempt):
            errors.append("cell %s has incomplete or unknown per-attempt evidence" % expected_id)
            continue
        index = attempt.get("seed_index")
        if not isinstance(index, int) or index < 0 or index >= FROZEN_SEED_COUNT or index in seen:
            errors.append("cell %s has duplicate/invalid seed index" % expected_id)
            continue
        seen.add(index)
        if attempt.get("seed_sha256") != _expected_seed_hash(preregistration, expected_id, index):
            errors.append("cell %s has a seed derivation mismatch" % expected_id)
        if attempt.get("status") not in STATUSES or not isinstance(attempt.get("emitted"), bool):
            errors.append("cell %s has invalid attempt status/emission" % expected_id)
        emitted += int(attempt.get("emitted") is True)
        _require_hashes(attempt, ("seed_sha256", "raw_input_sha256", "truth_sha256", "operator_hash", "variance_hash", "emission_hash", "date_axis_sha256", "generator_hash", "config_hash", "source_model_hash"), "attempt", errors)
        if attempt.get("date_axis_sha256") != date_digest:
            errors.append("cell %s has a date-axis identity mismatch" % expected_id)
        if attempt.get("generator_hash") != expected_generator:
            errors.append("cell %s has a generator identity mismatch" % expected_id)
        if attempt.get("config_hash") != expected_config or attempt.get("source_model_hash") != expected_source_model:
            errors.append("cell %s has a configuration/source-model identity mismatch" % expected_id)
        if attempt.get("target_coordinate") != expected_target or attempt.get("reference_coordinate") != expected_reference:
            errors.append("cell %s has a coordinate identity mismatch" % expected_id)
        if attempt.get("realized_overlap_percent") != expected_overlap or attempt.get("signed_influence_sign") != expected_sign:
            errors.append("cell %s has a realized overlap/sign mismatch" % expected_id)
        if not _number(attempt.get("effective_looks_fraction")) or attempt["effective_looks_fraction"] <= 0 or attempt.get("effective_looks_application") != "source_factor_divided_by_sqrt_fraction":
            errors.append("cell %s has an invalid effective-look realization" % expected_id)
        if expected_sign in {"positive", "negative"} and (not _number(attempt.get("signed_cross_influence")) or (attempt["signed_cross_influence"] > 0) != (expected_sign == "positive")):
            errors.append("cell %s has a signed cross-influence mismatch" % expected_id)
        if expected_sign == "zero" and attempt.get("signed_cross_influence") != 0.0:
            errors.append("cell %s coincident influence must be exactly zero" % expected_id)
        if expected_sign == "none" and attempt.get("signed_cross_influence") != 0.0:
            errors.append("cell %s disjoint influence must be exactly zero" % expected_id)
        for metric in ("operator_relative_error", "contrast_variance_relative_error", "psd_min_eigenvalue", "interval_score", "interval_width", "signed_cross_influence"):
            if not _number(attempt.get(metric)):
                errors.append("cell %s attempt %s is missing %s" % (expected_id, index, metric))
        if not isinstance(attempt.get("covered_95"), bool):
            errors.append("cell %s attempt %s is missing covered_95" % (expected_id, index))
    if seen != set(range(FROZEN_SEED_COUNT)):
        errors.append("cell %s is missing one or more seed indices" % expected_id)
    if cell.get("emitted_seeds") != emitted:
        errors.append("cell %s emitted_seeds does not equal per-attempt emission" % expected_id)


def _validate_cell(cell: Any, expected_id: str, preregistration: Mapping[str, Any], errors: List[str]) -> str:
    if not isinstance(cell, dict):
        errors.append("cell is not an object")
        return FAIL
    if cell.get("cell_id") != expected_id:
        errors.append("cell_id does not match its frozen dimension labels")
    if set(cell) - CELL_KEYS or (CELL_KEYS - {"not_evaluable_reason"}) - set(cell):
        errors.append("cell %s has unknown or missing required fields" % expected_id)
    for name, label in zip(DIMENSION_NAMES, expected_id.split("|")):
        if cell.get(name) != label:
            errors.append("cell %s has invalid %s" % (expected_id, name))
    status = cell.get("status")
    if status not in STATUSES:
        errors.append("cell status must be pass, fail, or not_evaluable")
        status = FAIL
    if status == NOT_EVALUABLE and (cell.get("eigen_stress") not in preregistration["cell_policy"]["allowed_not_evaluable"] or not cell.get("not_evaluable_reason")):
        errors.append("not_evaluable is only permitted for a declared tied-eigen reason")
    if cell.get("attempted_seeds") != FROZEN_SEED_COUNT or cell.get("top_up_seeds") != 0:
        errors.append("cell %s must report exactly 5000 attempts and zero top-up" % expected_id)
    expected_target, expected_reference = _expected_coordinates(preregistration, cell)
    if cell.get("target_coordinate") != expected_target or cell.get("reference_coordinate") != expected_reference:
        errors.append("cell %s coordinates do not match frozen position/geometry mapping" % expected_id)
    if cell.get("realized_overlap_percent") != PAIR_OVERLAP.get(cell.get("pair_geometry")) or cell.get("signed_influence_sign") != PAIR_SIGN.get(cell.get("pair_geometry")):
        errors.append("cell %s realized overlap/sign does not match frozen geometry" % expected_id)
    if not _number(cell.get("effective_looks_fraction")) or cell["effective_looks_fraction"] <= 0 or cell.get("effective_looks_application") != "source_factor_divided_by_sqrt_fraction":
        errors.append("cell %s effective-look fraction/application is missing or invalid" % expected_id)
    topology = preregistration["generator"]["acquisition"]["topologies"].get(cell.get("block_topology"), {})
    if cell.get("acquisition_count") != topology.get("acquisition_count") or cell.get("date_axis_sha256") != sha256_json(topology.get("date_axis")):
        errors.append("cell %s acquisition/date-axis identity does not match topology" % expected_id)
    _require_hashes(cell, ("operator_hash", "variance_hash", "psd_hash", "coverage_hash", "emission_hash", "generator_hash", "truth_hash"), "cell", errors)
    if cell.get("generator_hash") != sha256_json(preregistration["generator"]):
        errors.append("cell %s generator hash does not match preregistration" % expected_id)
    _validate_attempts(cell, expected_id, preregistration, errors)
    if status == NOT_EVALUABLE:
        return status
    thresholds = preregistration["thresholds"]
    required_metrics = ("operator_relative_error", "psd_min_eigenvalue", "coverage_95", "emission_rate")
    if any(not _number(cell.get(metric)) for metric in required_metrics):
        errors.append("cell %s is missing numeric outcome metrics" % expected_id)
        return status
    if not _number(cell.get("contrast_variance_reference")):
        errors.append("cell %s is missing contrast variance reference scale" % expected_id)
        return status
    weak_zero = cell["contrast_variance_reference"] <= thresholds["weak_zero_variance_max"]
    if weak_zero:
        if cell.get("variance_evaluable") is not False or cell.get("contrast_variance_relative_error") is not None:
            errors.append("cell %s must report weak-zero variance separately" % expected_id)
    elif cell.get("variance_evaluable") is not True or not _number(cell.get("contrast_variance_relative_error")):
        errors.append("cell %s is missing an evaluable contrast variance error" % expected_id)
    deterministic = cell.get("pair_geometry") in {"coincident", "disjoint_immediate", "disjoint_after_depth_1", "disjoint_after_depth_2", "disjoint_after_depth_4"}
    limit = thresholds["deterministic_operator_relative_error_max"] if deterministic else thresholds["stochastic_operator_relative_error_max"]
    if cell["operator_relative_error"] > limit or (not weak_zero and cell.get("contrast_variance_relative_error", 0) > thresholds["contrast_variance_relative_error_max"]):
        errors.append("cell %s exceeds analytic error threshold" % expected_id)
    if cell["psd_min_eigenvalue"] < thresholds["psd_min_eigenvalue_min"]:
        errors.append("cell %s fails PSD threshold" % expected_id)
    if abs(cell["coverage_95"] - thresholds["coverage_probability"]) > thresholds["coverage_absolute_error_max"]:
        errors.append("cell %s fails absolute coverage threshold" % expected_id)
    if cell["emission_rate"] < thresholds["emission_rate_min"]:
        errors.append("cell %s fails emission threshold" % expected_id)
    return status


def _validate_resources(receipt: Mapping[str, Any], preregistration: Mapping[str, Any], errors: List[str]) -> List[str]:
    resources = receipt.get("resources")
    if not isinstance(resources, list):
        errors.append("resources must be a per-resource list")
        return []
    by_id = {item.get("resource_id"): item for item in resources if isinstance(item, dict)}
    if set(by_id) != set(FROZEN_RESOURCE_IDS) or len(by_id) != len(resources):
        errors.append("resource receipts must contain exactly the three frozen resource cells")
    statuses = []
    sampling = preregistration["resource_sampling"]
    for resource_id in FROZEN_RESOURCE_IDS:
        item = by_id.get(resource_id)
        if item is None:
            continue
        statuses.append(item.get("status"))
        if set(item) - RESOURCE_KEYS or RESOURCE_KEYS - set(item):
            errors.append("resource %s has unknown or missing fields" % resource_id)
        if item.get("status") not in STATUSES or not _number(item.get("rss_bytes")) or item.get("rss_bytes", 0) > preregistration["thresholds"]["resource_rss_bytes_max"]:
            errors.append("resource %s has invalid RSS/status" % resource_id)
        if item.get("growth_class") != "linear" or any(item.get(field) != sampling[field] for field in ("os", "hardware_class", "ram_bytes", "rss_sampler", "rss_field", "sampling_interval_ms", "warmup_runs", "measured_repetitions", "tool_versions", "growth_regression", "acceptance")):
            errors.append("resource %s does not match frozen sampling provenance" % resource_id)
        _require_hashes(item, ("resource_hash", "config_hash", "binary_hash"), "resource %s" % resource_id, errors)
        hashes = receipt.get("hashes")
        if isinstance(hashes, dict) and (item.get("config_hash") != hashes.get("config_sha256") or item.get("binary_hash") != hashes.get("binary_sha256")):
            errors.append("resource %s identity hashes are cross-wired" % resource_id)
    return statuses


def score_receipt(preregistration: Mapping[str, Any], receipt: Any) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        validate_preregistration(preregistration)
    except SchemaError as exc:
        return {"status": FAIL, "errors": [str(exc)]}
    if not isinstance(receipt, dict):
        return {"status": FAIL, "errors": ["receipt root must be an object"]}
    if set(receipt) - RECEIPT_KEYS or RECEIPT_KEYS - set(receipt):
        errors.append("receipt has unknown or missing top-level fields")
    if receipt.get("schema") != "dolphinrust.spatial_covariance.receipt" or receipt.get("schema_version") != 2:
        errors.extend(["receipt schema must be version 2", "aggregate-only receipts are rejected"])
    generator = preregistration["generator"]
    protocol = receipt.get("protocol")
    expected_protocol = {key: generator["binary"][key] for key in ("input_schema", "output_schema", "one_input_one_output")}
    if protocol != expected_protocol:
        errors.append("receipt protocol does not match frozen JSONL protocol")
    binary = receipt.get("binary")
    if not isinstance(binary, dict) or binary.get("release_invocation") != generator["binary"]["release_invocation"] or binary.get("release_only") is not True:
        errors.append("receipt binary does not prove the frozen release invocation")
    if receipt.get("preregistration_sha256") != preregistration_digest(preregistration):
        errors.append("preregistration_sha256 does not match the frozen preregistration")
    hashes = receipt.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != RECEIPT_HASHES:
        errors.append("receipt hashes must contain exactly the frozen identity fields")
    else:
        _require_hashes(hashes, RECEIPT_HASHES, "hashes", errors)
        if hashes.get("generator_protocol_sha256") != sha256_json(generator["binary"]):
            errors.append("generator protocol hash does not match preregistration")
        if hashes.get("config_sha256") != sha256_json(generator):
            errors.append("config hash does not match frozen configuration")
        if hashes.get("source_model_sha256") != sha256_json(generator["source_centered_empirical"]):
            errors.append("source model hash does not match frozen source model")
    if receipt.get("seed_schedule_sha256") != seed_schedule_digest(preregistration):
        errors.append("seed_schedule_sha256 does not match the frozen seed schedule")
    cells = receipt.get("cells")
    expected = expected_cell_ids(preregistration)
    statuses: List[str] = []
    if not isinstance(cells, list):
        errors.append("cells must be a per-cell list; aggregate-only receipts are rejected")
    else:
        actual_ids = [cell.get("cell_id") for cell in cells if isinstance(cell, dict)]
        if len(actual_ids) != len(set(actual_ids)):
            errors.append("duplicate cell ids are not allowed")
        if set(expected) - set(actual_ids):
            errors.append("missing %d required matrix cells" % len(set(expected) - set(actual_ids)))
        if set(actual_ids) - set(expected):
            errors.append("receipt contains cells outside the frozen matrix")
        by_id = {cell.get("cell_id"): cell for cell in cells if isinstance(cell, dict)}
        for expected_id in expected:
            if expected_id in by_id:
                statuses.append(_validate_cell(by_id[expected_id], expected_id, preregistration, errors))
    resource_statuses = _validate_resources(receipt, preregistration, errors)
    if errors or any(status == FAIL for status in statuses + resource_statuses):
        status = FAIL
    elif any(status == NOT_EVALUABLE for status in statuses + resource_statuses):
        status = NOT_EVALUABLE
    else:
        status = PASS
    return {"status": status, "errors": errors, "cell_count": len(cells) if isinstance(cells, list) else 0}


validate_receipt = score_receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("spatial_covariance_preregistration.json"))
    args = parser.parse_args()
    with args.receipt.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    print(json.dumps(score_receipt(load_preregistration(args.preregistration), receipt), indent=2, sort_keys=True))
