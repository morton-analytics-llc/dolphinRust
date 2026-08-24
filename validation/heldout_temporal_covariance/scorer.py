"""One-shot held-out slope-difference scorer.

The scorer consumes a frozen manifest and an already-produced receipt. It never
discovers stations, downloads data, replaces clusters, or writes outcomes.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from .cohort import CohortValidationError, canonical_digest, validate_manifest


LEVELS = ("68", "90", "95")
Z = {"68": 0.994457883209753, "90": 1.6448536269514722, "95": 1.959963984540054}
RECEIPT_HASH_FIELDS = {
    "binary_sha256",
    "scorer_sha256",
    "preregistration_sha256",
    "manifest_sha256",
    "factor_scope_sha256",
    "gnss_catalog_sha256",
}
OBSERVATION_FIELDS = {
    "insar_slope_difference",
    "gnss_slope_difference",
    "insar_difference_variance",
    "gnss_slope_variance",
    "sensor_cross_covariance",
    "baseline_sigma",
}
PROVENANCE_FIELDS = {
    "solution_source",
    "solution_sha256",
    "coordinate_frame",
    "los_source",
    "los_sha256",
    "los_vector",
    "projection_convention",
    "epoch_zero_reference_sha256",
    "covariance_projection",
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _interval_score(lower: float, upper: float, target: float, nominal: float) -> float:
    alpha = 1.0 - nominal
    score = upper - lower
    if target < lower:
        score += 2.0 * (lower - target) / alpha
    elif target > upper:
        score += 2.0 * (target - upper) / alpha
    return score


def score_slope_difference(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Score one combined independent-sensor slope-difference observation."""

    if set(observation) != OBSERVATION_FIELDS:
        raise CohortValidationError("slope observation fields do not match the frozen schema")
    for field in ("insar_slope_difference", "gnss_slope_difference", "insar_difference_variance", "gnss_slope_variance", "sensor_cross_covariance"):
        if not _finite(observation[field]):
            raise CohortValidationError("slope observation contains a non-finite %s" % field)
    if observation["sensor_cross_covariance"] != 0.0:
        raise CohortValidationError("independent InSAR/GNSS scoring requires zero sensor cross covariance")
    if observation["insar_difference_variance"] < 0 or observation["gnss_slope_variance"] < 0:
        raise CohortValidationError("component slope variances cannot be negative")
    variance = observation["insar_difference_variance"] + observation["gnss_slope_variance"]
    if not math.isfinite(variance) or variance <= 0:
        raise CohortValidationError("combined slope-difference covariance is not positive")
    baseline = observation["baseline_sigma"]
    if not isinstance(baseline, Mapping) or set(baseline) != set(LEVELS) or any(not _finite(baseline[level]) or baseline[level] <= 0 for level in LEVELS):
        raise CohortValidationError("baseline sigma must provide positive 68/90/95 values")
    difference = observation["insar_slope_difference"] - observation["gnss_slope_difference"]
    sigma = math.sqrt(variance)
    levels: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        nominal = float(level) / 100.0
        half_width = Z[level] * sigma
        lower = difference - half_width
        upper = difference + half_width
        baseline_half_width = Z[level] * baseline[level]
        baseline_lower = difference - baseline_half_width
        baseline_upper = difference + baseline_half_width
        levels[level] = {
            "nominal": nominal,
            "difference": difference,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "covered": lower <= 0.0 <= upper,
            "interval_score": _interval_score(lower, upper, 0.0, nominal),
            "baseline_width": baseline_upper - baseline_lower,
            "baseline_interval_score": _interval_score(baseline_lower, baseline_upper, 0.0, nominal),
        }
    return {"difference": difference, "variance": variance, "sigma": sigma, "levels": levels}


def exact_binomial_noninferiority(covered: int, evaluated: int, nominal: float, alpha: float) -> dict[str, Any]:
    if evaluated <= 0 or covered < 0 or covered > evaluated:
        return {"status": "not_evaluable", "p_value": None}
    null_probability = nominal - 0.2
    tail = sum(
        math.comb(evaluated, value) * null_probability**value * (1.0 - null_probability) ** (evaluated - value)
        for value in range(covered, evaluated + 1)
    )
    return {
        "status": "pass" if tail <= alpha else "fail",
        "p_value": tail,
        "null_coverage": null_probability,
        "observed_coverage": covered / evaluated,
        "evaluated": evaluated,
        "covered": covered,
    }


def _validate_factor_binding(cluster: Mapping[str, Any], candidate: Mapping[str, Any], preregistration: Mapping[str, Any]) -> None:
    binding = cluster.get("difference_covariance")
    required = preregistration["factor_binding"]
    if not isinstance(binding, Mapping):
        raise CohortValidationError("cluster is missing direct #54 difference covariance binding")
    for field in ("operation", "schema", "mode", "reference_specific", "stitched_burst_count"):
        if binding.get(field) != required[field]:
            raise CohortValidationError("#54 factor identity or scope mismatch")
    if binding.get("marginal_rss_combination_allowed") is not False:
        raise CohortValidationError("#54 factor must be a direct difference covariance")
    if not _hash(binding.get("factor_sha256")) or not _hash(binding.get("scope_sha256")):
        raise CohortValidationError("#54 factor hashes are missing or invalid")
    if not isinstance(binding.get("scope"), Mapping):
        raise CohortValidationError("#54 factor scope is missing")
    scope = binding["scope"]
    if set(scope) != set(required["scope_fields"]):
        raise CohortValidationError("#54 factor scope fields are incomplete")
    if scope["target_station_id"] != candidate["station_ids"][0] or scope["control_station_id"] != candidate["station_ids"][1]:
        raise CohortValidationError("#54 target/control station scope mismatch")
    if not _hash(scope["common_dates_sha256"]):
        raise CohortValidationError("#54 common-date identity is missing")


def _validate_gnss_provenance(cluster: Mapping[str, Any], preregistration: Mapping[str, Any]) -> None:
    provenance = cluster.get("gnss_provenance")
    required = preregistration["gnss_provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != PROVENANCE_FIELDS:
        raise CohortValidationError("GNSS solution/LOS provenance is incomplete")
    for field in ("solution_sha256", "los_sha256", "epoch_zero_reference_sha256"):
        if not _hash(provenance[field]):
            raise CohortValidationError("GNSS provenance hash is invalid")
    for field in ("coordinate_frame", "projection_convention", "covariance_projection"):
        if provenance[field] != required[field if field != "projection_convention" else "projection"]:
            raise CohortValidationError("GNSS projection provenance mismatch")
    if provenance["solution_source"] == "" or provenance["los_source"] != required["los_source"]:
        raise CohortValidationError("GNSS solution or sourced LOS identity is missing")
    vector = provenance["los_vector"]
    if not isinstance(vector, Sequence) or len(vector) != 3 or any(not _finite(value) for value in vector):
        raise CohortValidationError("GNSS LOS vector is invalid")
    norm = math.sqrt(sum(value * value for value in vector))
    if abs(norm - 1.0) > required["los_norm_tolerance"]:
        raise CohortValidationError("GNSS LOS vector is not unit norm")


def _cluster_metrics(cluster: Mapping[str, Any], candidate: Mapping[str, Any], preregistration: Mapping[str, Any]) -> dict[str, Any]:
    _validate_factor_binding(cluster, candidate, preregistration)
    _validate_gnss_provenance(cluster, preregistration)
    result = score_slope_difference(cluster["observation"])
    return result


def score_receipt(preregistration: Mapping[str, Any], manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Score one frozen receipt, returning distinct pass/fail/not-evaluable states."""

    errors: list[str] = []
    try:
        validate_manifest(manifest, preregistration)
    except CohortValidationError as error:
        return {"status": "not_evaluable", "errors": [str(error)]}
    if manifest["status"] != "frozen_metadata_only":
        return {"status": "not_evaluable", "errors": ["candidate pool did not meet frozen power and surplus counts"]}
    if not isinstance(receipt, Mapping):
        return {"status": "fail", "errors": ["receipt must be an object"]}
    if receipt.get("schema") != "dolphinrust.temporal_covariance.heldout_receipt" or receipt.get("schema_version") != 1:
        errors.append("receipt schema/version mismatch")
    if receipt.get("outcomes_present") is not True or receipt.get("one_shot_unblinding") is not True:
        errors.append("receipt is not an explicit one-shot outcome receipt")
    if receipt.get("generation_id") != preregistration["generation_id"]:
        errors.append("receipt is stale for the frozen generation")
    if receipt.get("preregistration_sha256") != canonical_digest(preregistration):
        errors.append("preregistration identity mismatch")
    if receipt.get("manifest_sha256") != canonical_digest(manifest):
        errors.append("manifest identity mismatch")
    if receipt.get("scope_hash") != canonical_digest(preregistration["field_scope"]):
        errors.append("field scope mismatch")
    if receipt.get("factor_binding") != preregistration["factor_binding"]:
        errors.append("#54 configuration identity mismatch")
    hashes = receipt.get("hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != RECEIPT_HASH_FIELDS or any(not _hash(hashes[field]) for field in RECEIPT_HASH_FIELDS):
        errors.append("receipt hash fields are incomplete or invalid")
    elif hashes["preregistration_sha256"] != receipt["preregistration_sha256"] or hashes["manifest_sha256"] != receipt["manifest_sha256"]:
        errors.append("receipt hash fields disagree with their bound identities")
    elif hashes["factor_scope_sha256"] != canonical_digest(preregistration["factor_binding"]):
        errors.append("#54 factor scope hash does not match the frozen identity")
    if errors:
        return {"status": "fail", "errors": errors}
    clusters = receipt.get("clusters")
    expected = {candidate["candidate_id"] for candidate in manifest["frozen_clusters"] + manifest["surplus_clusters"]}
    if not isinstance(clusters, list):
        return {"status": "fail", "errors": ["receipt must contain per-cluster outcomes"]}
    actual = [cluster.get("cluster_id") for cluster in clusters if isinstance(cluster, Mapping)]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        return {"status": "fail", "errors": ["receipt clusters do not exactly match frozen primary and surplus clusters"]}
    by_id = {cluster["cluster_id"]: cluster for cluster in clusters}
    candidates = {candidate["candidate_id"]: candidate for candidate in manifest["frozen_clusters"] + manifest["surplus_clusters"]}
    level_values = {level: [] for level in LEVELS}
    statuses: list[str] = []
    for cluster_id, candidate in candidates.items():
        cluster = by_id[cluster_id]
        if cluster.get("station_ids") != candidate["station_ids"] or cluster.get("burst_id") != candidate["burst_id"] or cluster.get("site_id") != candidate["site_id"]:
            errors.append("cluster %s has scope metadata different from its frozen candidate" % cluster_id)
            continue
        status = cluster.get("status")
        statuses.append(status)
        if status not in {"pass", "fail", "not_evaluable"}:
            errors.append("cluster %s has an invalid status" % cluster_id)
        if status == "not_evaluable":
            if cluster.get("reason_code") not in preregistration["attrition"]["allowed_codes"]:
                errors.append("cluster %s uses an unregistered attrition reason" % cluster_id)
            continue
        if status == "fail":
            continue
        try:
            metrics = _cluster_metrics(cluster, candidate, preregistration)
        except (CohortValidationError, KeyError) as error:
            errors.append("cluster %s is not evaluable: %s" % (cluster_id, error))
            continue
        for level in LEVELS:
            level_values[level].append(metrics["levels"][level])
    if errors:
        return {"status": "fail", "errors": errors}
    if any(status == "fail" for status in statuses):
        return {"status": "fail", "errors": ["at least one frozen cluster reported fail"]}
    emission_rate = len(level_values["68"]) / len(clusters)
    if emission_rate < preregistration["thresholds"]["minimum_emission_rate"]:
        return {"status": "not_evaluable", "errors": ["field emission rate is below the frozen threshold"], "emission_rate": emission_rate}
    required = preregistration["power"]["required_evaluable_clusters"]
    levels: dict[str, Any] = {}
    for level in LEVELS:
        values = level_values[level]
        if len(values) < required[level]:
            levels[level] = {"status": "not_evaluable", "evaluated": len(values), "required": required[level]}
            continue
        covered = sum(value["covered"] for value in values)
        alpha = preregistration["power"]["per_level_alpha"]
        coverage_test = exact_binomial_noninferiority(covered, len(values), float(level) / 100.0, alpha)
        score_improves = sum(value["interval_score"] for value in values) < sum(value["baseline_interval_score"] for value in values)
        width_ratio = median(value["width"] for value in values) / median(value["baseline_width"] for value in values)
        levels[level] = {
            "status": "pass" if coverage_test["status"] == "pass" and score_improves and width_ratio < preregistration["thresholds"]["median_width_ratio_max"] else "fail",
            "coverage": coverage_test,
            "mean_interval_score": sum(value["interval_score"] for value in values) / len(values),
            "mean_baseline_interval_score": sum(value["baseline_interval_score"] for value in values) / len(values),
            "median_width_ratio": width_ratio,
            "proper_score_improves": score_improves,
        }
    if any(level["status"] == "fail" for level in levels.values()):
        status = "fail"
    elif any(level["status"] == "not_evaluable" for level in levels.values()) or any(status == "not_evaluable" for status in statuses):
        status = "not_evaluable"
    else:
        status = "pass"
    return {"status": status, "errors": [], "levels": levels, "evaluated_clusters": len(level_values["68"]), "emission_rate": emission_rate}
