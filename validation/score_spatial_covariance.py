#!/usr/bin/env python3
"""Strict receipt validator for the preregistered F54-07 validation matrix.

This module validates receipts; it does not run simulations, top up seeds, or
produce scientific outcomes.
"""

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PASS = "pass"
FAIL = "fail"
NOT_EVALUABLE = "not_evaluable"
STATUSES = {PASS, FAIL, NOT_EVALUABLE}
HASH_RE = set("0123456789abcdef")
REQUIRED_RECEIPT_HASHES = {
    "code_sha256",
    "fixture_sha256",
    "operator_sha256",
    "variance_sha256",
    "resource_sha256",
}
DIMENSION_NAMES = (
    "half_window",
    "stride",
    "support",
    "position",
    "pair_geometry",
    "block_topology",
    "estimator",
    "eigen_stress",
    "source_process",
)
FROZEN_DIMENSION_IDS = {
    "half_window": ("hw_1x1", "hw_3x6", "hw_7x14"),
    "stride": ("stride_1", "stride_2", "stride_4"),
    "support": ("rect", "glrt_frozen", "ks_frozen"),
    "position": ("interior", "border_clamped", "tile_edge", "bounded_halo", "masked"),
    "pair_geometry": (
        "coincident",
        "shared_75_positive",
        "shared_75_negative",
        "shared_50_positive",
        "shared_50_negative",
        "shared_25_positive",
        "shared_25_negative",
        "disjoint_immediate",
        "disjoint_after_depth_1",
        "disjoint_after_depth_2",
        "disjoint_after_depth_4",
    ),
    "block_topology": (
        "one_block",
        "two_blocks",
        "four_blocks",
        "two_blocks_cap_eviction",
        "four_blocks_partial_final",
    ),
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
FROZEN_SEED_COUNT = 5000
FROZEN_RESOURCE_IDS = ("tile_256_dates_13", "tile_256_dates_26", "tile_256_dates_52")
RECEIPT_KEYS = {
    "schema",
    "schema_version",
    "preregistration_sha256",
    "seed_schedule_sha256",
    "hashes",
    "cells",
    "resources",
}
CELL_KEYS = {
    "cell_id",
    *DIMENSION_NAMES,
    "status",
    "not_evaluable_reason",
    "attempted_seeds",
    "emitted_seeds",
    "top_up_seeds",
    "operator_relative_error",
    "contrast_variance_reference",
    "variance_evaluable",
    "contrast_variance_relative_error",
    "psd_min_eigenvalue",
    "coverage_95",
    "emission_rate",
    "operator_hash",
    "variance_hash",
    "psd_hash",
    "coverage_hash",
    "emission_hash",
}
RESOURCE_KEYS = {"resource_id", "status", "rss_bytes", "growth_class", "resource_hash"}


class SchemaError(ValueError):
    """Raised when the preregistration itself is not the frozen contract."""


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
    schedule = preregistration.get("seed_schedule")
    return sha256_json(schedule)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HASH_RE


def _dimension_values(preregistration: Mapping[str, Any], name: str) -> Sequence[str]:
    values = preregistration.get("dimensions", {}).get(name, [])
    return tuple(item.get("id") for item in values if isinstance(item, dict))


def validate_preregistration(preregistration: Mapping[str, Any]) -> None:
    errors: List[str] = []
    if preregistration.get("schema") != "dolphinrust.spatial_covariance.preregistration":
        errors.append("schema is not the F54-07 preregistration schema")
    if preregistration.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if preregistration.get("status") != "preregistered" or preregistration.get("outcomes_present") is not False:
        errors.append("preregistration must be outcome-free and preregistered")
    dimensions = preregistration.get("dimensions")
    if not isinstance(dimensions, dict):
        errors.append("dimensions must be an object")
    else:
        if set(dimensions) != set(DIMENSION_NAMES):
            errors.append("dimensions must contain exactly the nine frozen axes")
        for name in DIMENSION_NAMES:
            actual = _dimension_values(preregistration, name)
            if actual != FROZEN_DIMENSION_IDS[name]:
                errors.append("dimension %s does not match the frozen matrix" % name)
            if len(actual) != len(set(actual)):
                errors.append("dimension %s contains duplicate ids" % name)
    if preregistration.get("thresholds") != FROZEN_THRESHOLDS:
        errors.append("thresholds differ from the immutable F54-07 thresholds")
    schedule = preregistration.get("seed_schedule")
    if not isinstance(schedule, dict) or schedule.get("attempted_seeds_per_cell") != FROZEN_SEED_COUNT:
        errors.append("attempted_seeds_per_cell must be the frozen 5000")
    if not isinstance(schedule, dict) or schedule.get("no_top_up") is not True:
        errors.append("seed schedule must prohibit top-up")
    hashes = preregistration.get("hash_fields")
    if not isinstance(hashes, dict) or hashes.get("algorithm") != "sha256":
        errors.append("hash_fields must use sha256")
    if errors:
        raise SchemaError("; ".join(errors))


def expected_cell_ids(preregistration: Mapping[str, Any]) -> List[str]:
    validate_preregistration(preregistration)
    axes = [_dimension_values(preregistration, name) for name in DIMENSION_NAMES]
    return ["|".join(values) for values in itertools.product(*axes)]


def _cell_status(cell: Mapping[str, Any], preregistration: Mapping[str, Any], errors: List[str]) -> str:
    status = cell.get("status")
    if status not in STATUSES:
        errors.append("cell status must be pass, fail, or not_evaluable")
        return FAIL
    if status == NOT_EVALUABLE:
        allowed = preregistration["cell_policy"]["allowed_not_evaluable"]
        if cell.get("eigen_stress") not in allowed:
            errors.append("not_evaluable is not allowed for this cell")
        if not cell.get("not_evaluable_reason"):
            errors.append("not_evaluable cells require a reason")
    return status


def _require_hashes(value: Mapping[str, Any], fields: Iterable[str], prefix: str, errors: List[str]) -> None:
    for field in fields:
        if not _is_sha256(value.get(field)):
            errors.append("%s.%s must be a lowercase sha256 digest" % (prefix, field))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_cell(cell: Any, expected_id: str, preregistration: Mapping[str, Any], errors: List[str]) -> str:
    if not isinstance(cell, dict):
        errors.append("cell is not an object")
        return FAIL
    if cell.get("cell_id") != expected_id:
        errors.append("cell_id does not match its frozen dimension labels")
    if set(cell) - CELL_KEYS:
        errors.append("cell %s contains unknown fields" % expected_id)
    missing_cell_keys = (CELL_KEYS - {"not_evaluable_reason"}) - set(cell)
    if missing_cell_keys:
        errors.append("cell %s is missing required fields" % expected_id)
    labels = expected_id.split("|")
    for name, label in zip(DIMENSION_NAMES, labels):
        if cell.get(name) != label:
            errors.append("cell %s has invalid %s" % (expected_id, name))
    status = _cell_status(cell, preregistration, errors)
    if cell.get("attempted_seeds") != FROZEN_SEED_COUNT:
        errors.append("cell %s must report exactly 5000 attempted seeds" % expected_id)
    if cell.get("top_up_seeds") != 0:
        errors.append("cell %s contains top-up seeds" % expected_id)
    if not _number(cell.get("emitted_seeds")) or cell["emitted_seeds"] < 0 or cell["emitted_seeds"] > FROZEN_SEED_COUNT:
        errors.append("cell %s has invalid emitted_seeds" % expected_id)
    _require_hashes(cell, ("operator_hash", "variance_hash", "psd_hash", "coverage_hash", "emission_hash"), "cell", errors)
    if status == NOT_EVALUABLE:
        return status
    required_metrics = ("operator_relative_error", "psd_min_eigenvalue", "coverage_95", "emission_rate")
    if any(not _number(cell.get(metric)) for metric in required_metrics):
        errors.append("cell %s is missing numeric outcome metrics" % expected_id)
        return status
    thresholds = preregistration["thresholds"]
    if not _number(cell.get("contrast_variance_reference")):
        errors.append("cell %s is missing contrast variance reference scale" % expected_id)
        return status
    variance_is_weak_zero = cell["contrast_variance_reference"] <= thresholds["weak_zero_variance_max"]
    if variance_is_weak_zero:
        if cell.get("variance_evaluable") is not False or cell.get("contrast_variance_relative_error") is not None:
            errors.append("cell %s must report weak-zero variance separately" % expected_id)
    elif cell.get("variance_evaluable") is not True or not _number(cell.get("contrast_variance_relative_error")):
        errors.append("cell %s is missing an evaluable contrast variance error" % expected_id)
    deterministic = cell.get("pair_geometry") in {"coincident", "disjoint_immediate", "disjoint_after_depth_1", "disjoint_after_depth_2", "disjoint_after_depth_4"}
    operator_limit = thresholds["deterministic_operator_relative_error_max"] if deterministic else thresholds["stochastic_operator_relative_error_max"]
    if cell["operator_relative_error"] > operator_limit:
        errors.append("cell %s exceeds operator error threshold" % expected_id)
    if not variance_is_weak_zero and _number(cell.get("contrast_variance_relative_error")) and cell["contrast_variance_relative_error"] > thresholds["contrast_variance_relative_error_max"]:
        errors.append("cell %s exceeds contrast variance threshold" % expected_id)
    if cell["psd_min_eigenvalue"] < thresholds["psd_min_eigenvalue_min"]:
        errors.append("cell %s fails PSD threshold" % expected_id)
    if abs(cell["coverage_95"] - thresholds["coverage_probability"]) > thresholds["coverage_absolute_error_max"]:
        errors.append("cell %s fails coverage threshold" % expected_id)
    if cell["emission_rate"] < thresholds["emission_rate_min"]:
        errors.append("cell %s fails emission threshold" % expected_id)
    return status


def _validate_resources(receipt: Mapping[str, Any], preregistration: Mapping[str, Any], errors: List[str]) -> List[str]:
    resources = receipt.get("resources")
    if not isinstance(resources, list):
        errors.append("resources must be a per-resource list")
        return []
    by_id = {item.get("resource_id"): item for item in resources if isinstance(item, dict)}
    expected = set(FROZEN_RESOURCE_IDS)
    if set(by_id) != expected or len(by_id) != len(resources):
        errors.append("resource receipts must contain exactly the three frozen resource cells")
    statuses: List[str] = []
    for resource_id in FROZEN_RESOURCE_IDS:
        item = by_id.get(resource_id)
        if item is None:
            continue
        if set(item) - RESOURCE_KEYS:
            errors.append("resource %s contains unknown fields" % resource_id)
        if RESOURCE_KEYS - set(item):
            errors.append("resource %s is missing required fields" % resource_id)
        status = item.get("status")
        statuses.append(status)
        if status not in STATUSES:
            errors.append("resource %s has an invalid status" % resource_id)
        if not _number(item.get("rss_bytes")):
            errors.append("resource %s must report numeric rss_bytes" % resource_id)
        elif item["rss_bytes"] > preregistration["thresholds"]["resource_rss_bytes_max"]:
            errors.append("resource %s exceeds RSS threshold" % resource_id)
        if item.get("growth_class") != "linear":
            errors.append("resource %s must report linear growth" % resource_id)
        _require_hashes(item, ("resource_hash",), "resource %s" % resource_id, errors)
    return statuses


def score_receipt(preregistration: Mapping[str, Any], receipt: Any) -> Dict[str, Any]:
    """Return a strict pass/fail/not_evaluable report without generating outcomes."""

    errors: List[str] = []
    try:
        validate_preregistration(preregistration)
    except SchemaError as exc:
        return {"status": FAIL, "errors": [str(exc)]}
    if not isinstance(receipt, dict):
        return {"status": FAIL, "errors": ["receipt root must be an object"]}
    if set(receipt) - RECEIPT_KEYS:
        errors.append("receipt contains unknown fields")
    if receipt.get("schema") != "dolphinrust.spatial_covariance.receipt" or receipt.get("schema_version") != 1:
        errors.append("receipt schema must be version 1")
    if receipt.get("preregistration_sha256") != preregistration_digest(preregistration):
        errors.append("preregistration_sha256 does not match the frozen preregistration")
    hashes = receipt.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != REQUIRED_RECEIPT_HASHES:
        errors.append("receipt hashes must contain exactly the five required hash fields")
    elif hashes:
        _require_hashes(hashes, REQUIRED_RECEIPT_HASHES, "hashes", errors)
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
        missing = set(expected) - set(actual_ids)
        extra = set(actual_ids) - set(expected)
        if missing:
            errors.append("missing %d required matrix cells" % len(missing))
        if extra:
            errors.append("receipt contains %d cells outside the frozen matrix" % len(extra))
        by_id = {cell.get("cell_id"): cell for cell in cells if isinstance(cell, dict)}
        for expected_id in expected:
            if expected_id in by_id:
                statuses.append(_validate_cell(by_id[expected_id], expected_id, preregistration, errors))
    resource_statuses = _validate_resources(receipt, preregistration, errors)
    if any(status == FAIL for status in statuses + resource_statuses) or errors:
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
    prereg = load_preregistration(args.preregistration)
    with args.receipt.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    print(json.dumps(score_receipt(prereg, receipt), indent=2, sort_keys=True))
