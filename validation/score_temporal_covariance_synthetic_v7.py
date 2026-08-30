"""Successor temporal-covariance scorer with explicit calibration contracts.

``score_records`` consumes one record per scientific cell and seed. A record's
``methods`` object maps method names to comparator results. Comparators contain
``status``, ``point_estimate``, and ``intervals`` keyed by ``68``, ``90``, and
``95``. The frozen v5 ``fit`` field and comparator names are also accepted for
forensic replay without changing the v5 producer.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Iterable


SCHEMA = "coverage_bias_interval_score/7"
POLICY_SCHEMA = "dolphinrust-temporal-covariance-scorer-policy/1"
MODES = {"oracle_calibration", "candidate_evaluation", "forensic_v5"}
LEVELS = ("68", "90", "95")
FROZEN_V5_PREREGISTRATION_SHA256 = (
    "bf8a0cc92d6f0f4e03bb3c0fea88ea411b897d20373376d021540c55dce77166"
)
FROZEN_V5_SOURCE_SHA256 = (
    "6684130b2b8f596bef67de70ed39f00b8cb65cb1023beb169307f660834f7d56"
)
FROZEN_V5_NO_GO_SHA256 = (
    "0c885ac25f6680a18b1739e7c126c5821bc153c808c00e7b51c0b4e001ef483e"
)
FROZEN_V5_RUN_MANIFEST_SHA256 = (
    "bdab395890265496f1fbba8118f741b33be222647e30e3d27b4d84ad33aef05c"
)
FROZEN_V5_RUN_COMMIT_SHA256 = (
    "db53c284bda9be95010622b77c91f783fe668ac65552605f3460f1484ac8f0d6"
)
CERTIFICATION_POLICY_SHA256 = (
    "48fe684154a399ff8265b89b5e2c6a88f20d00e0794ab139ab58bdbe1828b73a"
)
FROZEN_V5_CELL_COUNT = 24
FROZEN_V5_SEEDS_PER_CELL = 1050
FROZEN_V5_EXECUTION_PATHS = {"fixed_factor", "production_path"}
V5_METHOD_FIELDS = {
    "ols": "ols",
    "oracle_gls": "oracle_gls",
    "legacy_intercept_slope_wls_non_comparable": "conditional_wls",
    "lag_one_scalar_effective_n": "scalar_effective_n",
    "plugin_gls_reml": "plugin_gls",
    "reml_covariance_parameter_adjusted_scalar": "adjusted_scalar",
    "slope_profile_likelihood_ml": "adjusted_profile",
    "complete_refit_bootstrap": "complete_refit_bootstrap",
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _scorer_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _receipt_sha256(receipt: dict) -> str:
    return _sha256({key: value for key, value in receipt.items()
                    if key != "receipt_sha256"})


def _fraction(value: object, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a rational number")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be a rational number") from error
    return result


def _ratio(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x
        coefficient /= (qam + even) * (a + even)
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x
        coefficient /= (a + even) * (qap + even)
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise RuntimeError("incomplete beta evaluation did not converge")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    scale = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return scale * _beta_continued_fraction(a, b, x) / a
    return 1.0 - scale * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_two_sided_tail(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0 or value < 0.0:
        raise ValueError("Student-t tail requires positive degrees of freedom")
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    return _regularized_incomplete_beta(
        x, degrees_of_freedom / 2.0, 0.5
    )


def _minimum_bias_count(
    cell_count: int, tolerance: Fraction, familywise_alpha: Fraction
) -> int:
    def calibrated(count: int) -> bool:
        threshold = float(tolerance) * math.sqrt(count)
        family_tail = cell_count * _student_t_two_sided_tail(
            threshold, count - 1
        )
        return family_tail <= float(familywise_alpha)

    lower = 2
    upper = 2
    while not calibrated(upper):
        upper *= 2
        if upper > 10_000_000:
            raise ValueError("bias calibration count exceeds the supported bound")
    while lower < upper:
        middle = (lower + upper) // 2
        if calibrated(middle):
            upper = middle
        else:
            lower = middle + 1
    return lower


def _normalize_policy(policy: dict) -> dict:
    if not isinstance(policy, dict) or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy schema must be {POLICY_SCHEMA}")
    cell_count = policy.get("scientific_cell_count")
    calibration_count = policy.get("calibration_count_per_cell")
    if type(cell_count) is not int or cell_count <= 0:
        raise ValueError("scientific_cell_count must be positive")
    scientific_cells = policy.get("scientific_cells")
    if scientific_cells is not None and (
        not isinstance(scientific_cells, list)
        or len(scientific_cells) != cell_count
        or len(set(scientific_cells)) != cell_count
        or any(not isinstance(cell, str) or not cell for cell in scientific_cells)
    ):
        raise ValueError("scientific_cells must name each unique policy cell")
    if type(calibration_count) is not int or calibration_count <= 0:
        raise ValueError("calibration_count_per_cell must be positive")
    methods = policy.get("methods")
    baselines = policy.get("baseline_methods")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(method, str) or not method for method in methods)
        or len(methods) != len(set(methods))
    ):
        raise ValueError("methods must be unique nonempty strings")
    if (
        not isinstance(baselines, list)
        or not baselines
        or any(method not in methods for method in baselines)
        or len(baselines) != len(set(baselines))
    ):
        raise ValueError("baseline_methods must be unique policy methods")
    selected = policy.get("selected_method")
    oracle = policy.get("oracle_method")
    if selected not in methods or oracle not in methods:
        raise ValueError("selected_method and oracle_method must be policy methods")
    if selected in baselines:
        raise ValueError("selected_method cannot be its own baseline")
    familywise_alpha = _fraction(policy.get("familywise_alpha"), "familywise_alpha")
    bias_tolerance = _fraction(
        policy.get("standardized_bias_tolerance"),
        "standardized_bias_tolerance",
    )
    method_emission = _fraction(
        policy.get("method_emission_minimum"), "method_emission_minimum"
    )
    pairwise_overlap = _fraction(
        policy.get("pairwise_overlap_minimum"), "pairwise_overlap_minimum"
    )
    if not 0 < familywise_alpha < 1:
        raise ValueError("familywise_alpha must be between zero and one")
    if bias_tolerance <= 0:
        raise ValueError("standardized_bias_tolerance must be positive")
    if not 0 < method_emission <= 1 or not 0 < pairwise_overlap <= 1:
        raise ValueError("emission and overlap minima must be in (0, 1]")
    coverage = {}
    raw_coverage = policy.get("coverage")
    if not isinstance(raw_coverage, dict) or set(raw_coverage) != set(LEVELS):
        raise ValueError("coverage policy must define 68, 90, and 95")
    for level in LEVELS:
        value = raw_coverage[level]
        if not isinstance(value, dict):
            raise ValueError(f"coverage {level} must be an object")
        nominal = _fraction(value.get("nominal"), f"coverage {level} nominal")
        tolerance = _fraction(
            value.get("tolerance"), f"coverage {level} tolerance"
        )
        if not 0 < nominal < 1 or not 0 <= tolerance < 1:
            raise ValueError(f"coverage {level} is outside its rational bounds")
        coverage[level] = {"nominal": nominal, "tolerance": tolerance}
    pairwise_rules = {}
    raw_rules = policy.get("pairwise_rules")
    if not isinstance(raw_rules, dict) or any(
        baseline not in raw_rules for baseline in baselines
    ):
        raise ValueError("pairwise_rules must define every named baseline")
    for baseline in baselines:
        raw_rule = raw_rules[baseline]
        if not isinstance(raw_rule, dict):
            raise ValueError(f"pairwise rule {baseline} must be an object")
        score_ratio = _fraction(
            raw_rule.get("maximum_mean_score_ratio"),
            f"pairwise rule {baseline} maximum_mean_score_ratio",
        )
        if score_ratio <= 0:
            raise ValueError("pairwise score ratios must be positive")
        width_ratio = raw_rule.get("maximum_mean_width_ratio")
        width_ratio = (
            None if width_ratio is None else _fraction(
                width_ratio,
                f"pairwise rule {baseline} maximum_mean_width_ratio",
            )
        )
        if width_ratio is not None and width_ratio <= 0:
            raise ValueError("pairwise width ratios must be positive")
        pairwise_rules[baseline] = {
            "maximum_mean_score_ratio": score_ratio,
            "maximum_mean_width_ratio": width_ratio,
        }
    truth = float(_fraction(policy.get("truth"), "truth"))
    if not math.isfinite(truth):
        raise ValueError("truth must be finite")
    minimum_bias_count = _minimum_bias_count(
        cell_count, bias_tolerance, familywise_alpha
    )
    if calibration_count < minimum_bias_count:
        raise ValueError(
            "calibration_count_per_cell does not satisfy the multiplicity-aware bias gate"
        )
    return {
        "raw": policy,
        "sha256": _sha256(policy),
        "cell_count": cell_count,
        "cells": None if scientific_cells is None else frozenset(scientific_cells),
        "calibration_count": calibration_count,
        "methods": tuple(methods),
        "baselines": tuple(baselines),
        "selected": selected,
        "oracle": oracle,
        "truth": truth,
        "familywise_alpha": familywise_alpha,
        "bias_tolerance": bias_tolerance,
        "minimum_bias_count": minimum_bias_count,
        "method_emission": method_emission,
        "pairwise_overlap": pairwise_overlap,
        "coverage": coverage,
        "pairwise_rules": pairwise_rules,
    }


def _normalize_source_identity(source_identity: dict, mode: str) -> dict:
    if not isinstance(source_identity, dict):
        raise ValueError("source_identity must be an object")
    required_hashes = (
        "source_sha256",
        "source_preregistration_sha256",
        "run_manifest_sha256",
        "run_commit_sha256",
    )
    for field in required_hashes:
        if not _valid_sha256(source_identity.get(field)):
            raise ValueError(f"{field} must be a lowercase SHA-256")
    role = source_identity.get("seed_domain_role")
    domain = source_identity.get("seed_domain")
    if not isinstance(domain, dict) or set(domain) != {"start", "count"}:
        raise ValueError("seed_domain must contain integer start and count")
    start = domain["start"]
    count = domain["count"]
    if type(start) is not int or start < 0 or type(count) is not int or count <= 0:
        raise ValueError("seed_domain must contain integer start and count")
    expected_role = {
        "oracle_calibration": "throwaway_oracle_calibration",
        "candidate_evaluation": "candidate_evaluation",
        "forensic_v5": "frozen_v5",
    }[mode]
    if role != expected_role:
        raise ValueError(f"{mode} requires seed_domain_role={expected_role}")
    if mode == "forensic_v5":
        if (
            source_identity["source_sha256"] != FROZEN_V5_SOURCE_SHA256
            or source_identity["source_preregistration_sha256"]
            != FROZEN_V5_PREREGISTRATION_SHA256
            or source_identity.get("no_go_summary_sha256") != FROZEN_V5_NO_GO_SHA256
            or source_identity["run_manifest_sha256"]
            != FROZEN_V5_RUN_MANIFEST_SHA256
            or source_identity["run_commit_sha256"]
            != FROZEN_V5_RUN_COMMIT_SHA256
            or start != 0
            or count != FROZEN_V5_SEEDS_PER_CELL
        ):
            raise ValueError("forensic_v5 requires the frozen v5 artifact hashes")
    return {
        field: source_identity[field] for field in required_hashes
    } | {
        "seed_domain_role": role,
        "seed_domain": {"start": start, "count": count},
    }


def _interval(comparator: object, level: str) -> tuple[float, float] | None:
    if not isinstance(comparator, dict) or comparator.get("status") != "Evaluated":
        return None
    intervals = comparator.get("intervals")
    value = intervals.get(level) if isinstance(intervals, dict) else None
    if value is None:
        value = comparator.get(f"interval_{level}")
    if isinstance(value, dict):
        lower, upper = value.get("lower"), value.get("upper")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        return None
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, (int, float))
        or not isinstance(upper, (int, float))
    ):
        return None
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        return None
    return lower, upper


def _point_estimate(comparator: object) -> float | None:
    if not isinstance(comparator, dict) or comparator.get("status") != "Evaluated":
        return None
    value = comparator.get("point_estimate")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _interval_score(interval: tuple[float, float], truth: float, level: str) -> float:
    lower, upper = interval
    alpha = 1.0 - int(level) / 100.0
    return (
        upper - lower
        + (2.0 / alpha) * max(lower - truth, 0.0)
        + (2.0 / alpha) * max(truth - upper, 0.0)
    )


def _record_methods(record: dict, policy: dict) -> dict:
    methods = record.get("methods")
    if methods is not None:
        if not isinstance(methods, dict):
            raise ValueError("record methods must be an object")
        unknown = set(methods) - set(policy["methods"])
        if unknown:
            raise ValueError(f"record contains unknown methods: {sorted(unknown)}")
        return methods
    fit = record.get("fit")
    if fit is None:
        return {}
    if not isinstance(fit, dict):
        raise ValueError("record fit must be an object or null")
    return {
        method: fit.get(V5_METHOD_FIELDS.get(method, method))
        for method in policy["methods"]
    }


class _MethodAccumulator:
    def __init__(self) -> None:
        self.attempted = 0
        self.scored = 0
        self.mean_bias = 0.0
        self.bias_m2 = 0.0
        self.interval_emitted = {level: 0 for level in LEVELS}
        self.covered = {level: 0 for level in LEVELS}
        self.width_sums = {level: 0.0 for level in LEVELS}
        self.score_sums = {level: 0.0 for level in LEVELS}

    def update(self, comparator: object, truth: float) -> None:
        self.attempted += 1
        estimate = _point_estimate(comparator)
        if estimate is not None:
            self.scored += 1
            bias = estimate - truth
            delta = bias - self.mean_bias
            self.mean_bias += delta / self.scored
            self.bias_m2 += delta * (bias - self.mean_bias)
        for level in LEVELS:
            interval = _interval(comparator, level)
            if interval is None:
                continue
            self.interval_emitted[level] += 1
            lower, upper = interval
            if lower <= truth <= upper:
                self.covered[level] += 1
            self.width_sums[level] += upper - lower
            self.score_sums[level] += _interval_score(interval, truth, level)


class _PairAccumulator:
    def __init__(self) -> None:
        self.attempted = 0
        self.paired = 0
        self.selected_only = 0
        self.baseline_only = 0
        self.neither = 0
        self.selected_score_sum = 0.0
        self.baseline_score_sum = 0.0
        self.selected_width_sum = 0.0
        self.baseline_width_sum = 0.0

    def update(
        self,
        selected: object,
        baseline: object,
        truth: float,
        level: str,
    ) -> None:
        self.attempted += 1
        selected_interval = _interval(selected, level)
        baseline_interval = _interval(baseline, level)
        if selected_interval is not None and baseline_interval is not None:
            self.paired += 1
            self.selected_width_sum += selected_interval[1] - selected_interval[0]
            self.baseline_width_sum += baseline_interval[1] - baseline_interval[0]
            self.selected_score_sum += _interval_score(
                selected_interval, truth, level
            )
            self.baseline_score_sum += _interval_score(
                baseline_interval, truth, level
            )
        elif selected_interval is not None:
            self.selected_only += 1
        elif baseline_interval is not None:
            self.baseline_only += 1
        else:
            self.neither += 1


def _method_row(
    cell_id: object,
    execution_path: str,
    method: str,
    accumulator: _MethodAccumulator,
    policy: dict,
) -> dict:
    attempted = accumulator.attempted
    scored = accumulator.scored
    bias_sd = (
        math.sqrt(accumulator.bias_m2 / (scored - 1)) if scored > 1 else None
    )
    standardized_bias = (
        abs(accumulator.mean_bias) / bias_sd
        if bias_sd is not None and bias_sd > 0.0
        else None
    )
    gates = {
        "emission": attempted > 0
        and scored * policy["method_emission"].denominator
        >= attempted * policy["method_emission"].numerator,
        "bias_count": scored >= policy["minimum_bias_count"],
        "standardized_bias": standardized_bias is not None
        and Fraction(str(standardized_bias)) <= policy["bias_tolerance"],
    }
    intervals = {}
    for level in LEVELS:
        emitted = accumulator.interval_emitted[level]
        covered = accumulator.covered[level]
        coverage = Fraction(covered, emitted) if emitted else None
        coverage_policy = policy["coverage"][level]
        gates[f"interval_emission_{level}"] = attempted > 0 and (
            emitted * policy["method_emission"].denominator
            >= attempted * policy["method_emission"].numerator
        )
        gates[f"coverage_{level}"] = coverage is not None and (
            abs(coverage - coverage_policy["nominal"])
            <= coverage_policy["tolerance"]
        )
        intervals[level] = {
            "emitted": emitted,
            "covered": covered,
            "coverage": _ratio(coverage) if coverage is not None else None,
            "nominal": _ratio(coverage_policy["nominal"]),
            "tolerance": _ratio(coverage_policy["tolerance"]),
            "width_sum": accumulator.width_sums[level],
            "mean_width": (
                accumulator.width_sums[level] / emitted if emitted else None
            ),
            "interval_score_sum": accumulator.score_sums[level],
            "mean_interval_score": (
                accumulator.score_sums[level] / emitted if emitted else None
            ),
        }
    failing_gates = sorted(gate for gate, passed in gates.items() if not passed)
    return {
        "cell_id": cell_id,
        "execution_path": execution_path,
        "method": method,
        "attempted": attempted,
        "scored": scored,
        "failed": attempted - scored,
        "emission": _ratio(Fraction(scored, attempted)) if attempted else None,
        "bias_moments": {
            "count": scored,
            "mean": accumulator.mean_bias if scored else None,
            "m2": accumulator.bias_m2 if scored else None,
            "sample_sd": bias_sd,
            "standardized_bias": standardized_bias,
        },
        "intervals": intervals,
        "gates": gates,
        "failing_gates": failing_gates,
        "passes_all_gates": not failing_gates,
    }


def _pairwise_row(
    cell_id: object,
    execution_path: str,
    selected_method: str,
    baseline_method: str,
    level: str,
    accumulator: _PairAccumulator,
    policy: dict,
) -> dict:
    paired = accumulator.paired
    selected_score = accumulator.selected_score_sum / paired if paired else None
    baseline_score = accumulator.baseline_score_sum / paired if paired else None
    selected_width = accumulator.selected_width_sum / paired if paired else None
    baseline_width = accumulator.baseline_width_sum / paired if paired else None
    rule = policy["pairwise_rules"][baseline_method]
    overlap = (
        Fraction(paired, accumulator.attempted)
        if accumulator.attempted else None
    )
    selected_emitted = paired + accumulator.selected_only
    baseline_emitted = paired + accumulator.baseline_only
    gates = {
        "selected_emission": accumulator.attempted > 0
        and selected_emitted * policy["method_emission"].denominator
        >= accumulator.attempted * policy["method_emission"].numerator,
        "baseline_emission": accumulator.attempted > 0
        and baseline_emitted * policy["method_emission"].denominator
        >= accumulator.attempted * policy["method_emission"].numerator,
        "overlap": overlap is not None
        and overlap >= policy["pairwise_overlap"],
        "proper_score": paired > 0
        and selected_score <= baseline_score * float(
            rule["maximum_mean_score_ratio"]
        ),
    }
    width_ratio = rule["maximum_mean_width_ratio"]
    if width_ratio is not None:
        gates["interval_width"] = paired > 0 and (
            selected_width <= baseline_width * float(width_ratio)
        )
    failing_gates = sorted(gate for gate, passed in gates.items() if not passed)
    return {
        "cell_id": cell_id,
        "execution_path": execution_path,
        "selected_method": selected_method,
        "baseline_method": baseline_method,
        "level": level,
        "attempted": accumulator.attempted,
        "paired": paired,
        "selected_emitted": selected_emitted,
        "baseline_emitted": baseline_emitted,
        "selected_only": accumulator.selected_only,
        "baseline_only": accumulator.baseline_only,
        "neither": accumulator.neither,
        "overlap": _ratio(overlap) if overlap is not None else None,
        "selected_mean_interval_score": selected_score,
        "baseline_mean_interval_score": baseline_score,
        "selected_mean_width": selected_width,
        "baseline_mean_width": baseline_width,
        "gates": gates,
        "failing_gates": failing_gates,
        "passes_all_gates": not failing_gates,
    }


def _validate_calibration_receipt(
    receipt: object,
    policy: dict,
    source: dict,
    scorer_source_sha256: str,
) -> str:
    if not isinstance(receipt, dict):
        raise ValueError("candidate_evaluation requires a calibration receipt")
    if receipt.get("receipt_sha256") != _receipt_sha256(receipt):
        raise ValueError("calibration receipt hash is stale or invalid")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("mode") != "oracle_calibration"
        or receipt.get("calibration_pass") is not True
    ):
        raise ValueError("candidate_evaluation requires a passing calibration receipt")
    if (
        receipt.get("policy_sha256") != policy["sha256"]
        or receipt.get("scorer_source_sha256") != scorer_source_sha256
        or receipt.get("source_sha256") != source["source_sha256"]
        or receipt.get("source_preregistration_sha256")
        != source["source_preregistration_sha256"]
    ):
        raise ValueError("calibration receipt is not bound to this scorer policy and source")
    calibration_domain = receipt.get("seed_domain")
    if (
        not isinstance(calibration_domain, dict)
        or set(calibration_domain) != {"start", "count"}
        or type(calibration_domain["start"]) is not int
        or type(calibration_domain["count"]) is not int
    ):
        raise ValueError("calibration receipt seed domain is invalid")
    calibration_start = calibration_domain["start"]
    calibration_end = calibration_start + calibration_domain["count"]
    candidate_start = source["seed_domain"]["start"]
    candidate_end = candidate_start + source["seed_domain"]["count"]
    if max(calibration_start, candidate_start) < min(calibration_end, candidate_end):
        raise ValueError("calibration and candidate seed domains must be disjoint")
    return receipt["receipt_sha256"]


def score_records(
    records: Iterable[dict],
    policy: dict,
    source_identity: dict,
    mode: str,
) -> dict:
    """Score one frozen run under the requested v7 mode."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    normalized_policy = _normalize_policy(policy)
    source = _normalize_source_identity(source_identity, mode)
    scorer_source_sha256 = _scorer_source_sha256()
    calibration_receipt_sha256 = None
    if mode == "candidate_evaluation":
        calibration_receipt_sha256 = _validate_calibration_receipt(
            source_identity.get("calibration_receipt"),
            normalized_policy,
            source,
            scorer_source_sha256,
        )

    method_accumulators = {}
    pair_accumulators = {}
    cells = set()
    cell_indices = set()
    cell_id_by_index = {}
    cell_index_by_id = {}
    scopes = set()
    seeds_by_scope = {}
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every record must be an object")
        cell_id = record.get("cell_id")
        if not isinstance(cell_id, (str, int)) or isinstance(cell_id, bool):
            raise ValueError("every record must have a string or integer cell_id")
        seed = record.get("outer_seed_index", record.get("seed"))
        if type(seed) is not int or seed < 0:
            raise ValueError("every record must have a nonnegative integer seed index")
        execution_path = record.get("execution_path", "default")
        if not isinstance(execution_path, str) or not execution_path:
            raise ValueError("execution_path must be a nonempty string when present")
        identity = (cell_id, execution_path, seed)
        if identity in seen:
            raise ValueError("records contain a duplicate cell and seed identity")
        seen.add(identity)
        cells.add(cell_id)
        scope = (cell_id, execution_path)
        scopes.add(scope)
        seeds_by_scope.setdefault(scope, set()).add(seed)
        cell_index = record.get("cell_index")
        if cell_index is not None:
            if type(cell_index) is not int or cell_index < 0:
                raise ValueError("cell_index must be a nonnegative integer when present")
            if (
                cell_id_by_index.setdefault(cell_index, cell_id) != cell_id
                or cell_index_by_id.setdefault(cell_id, cell_index) != cell_index
            ):
                raise ValueError("cell_id and cell_index mapping is inconsistent")
            cell_indices.add(cell_index)
        elif mode == "forensic_v5":
            raise ValueError("forensic_v5 record is outside the frozen v5 cell schedule")
        methods = _record_methods(record, normalized_policy)
        selected = methods.get(normalized_policy["selected"])
        for method in normalized_policy["methods"]:
            key = (cell_id, execution_path, method)
            accumulator = method_accumulators.setdefault(key, _MethodAccumulator())
            accumulator.update(methods.get(method), normalized_policy["truth"])
        for baseline in normalized_policy["baselines"]:
            comparator = methods.get(baseline)
            for level in LEVELS:
                key = (cell_id, execution_path, baseline, level)
                accumulator = pair_accumulators.setdefault(key, _PairAccumulator())
                accumulator.update(
                    selected, comparator, normalized_policy["truth"], level
                )

    domain_start = source["seed_domain"]["start"]
    domain_count = source["seed_domain"]["count"]
    expected_seeds = set(range(domain_start, domain_start + domain_count))
    if not scopes or any(seeds != expected_seeds for seeds in seeds_by_scope.values()):
        raise ValueError("records do not match the source seed domain schedule")
    if mode == "forensic_v5" and (
        cell_indices != set(range(FROZEN_V5_CELL_COUNT))
        or {execution_path for _cell_id, execution_path in scopes}
        != FROZEN_V5_EXECUTION_PATHS
        or len(scopes) != FROZEN_V5_CELL_COUNT * len(FROZEN_V5_EXECUTION_PATHS)
    ):
        raise ValueError("records do not match the frozen v5 cell and path schedule")

    ordered_scopes = sorted(
        scopes,
        key=lambda value: (str(type(value[0])), str(value[0]), value[1]),
    )
    method_rows = [
        _method_row(
            cell_id,
            execution_path,
            method,
            method_accumulators[(cell_id, execution_path, method)],
            normalized_policy,
        )
        for cell_id, execution_path in ordered_scopes
        for method in normalized_policy["methods"]
    ]
    pairwise_rows = [
        _pairwise_row(
            cell_id,
            execution_path,
            normalized_policy["selected"],
            baseline,
            level,
            pair_accumulators[(cell_id, execution_path, baseline, level)],
            normalized_policy,
        )
        for cell_id, execution_path in ordered_scopes
        for baseline in normalized_policy["baselines"]
        for level in LEVELS
    ]
    row_by_method = {
        (row["cell_id"], row["execution_path"], row["method"]): row
        for row in method_rows
    }
    cell_family_complete = (
        cells == normalized_policy["cells"]
        if normalized_policy["cells"] is not None
        else len(cells) == normalized_policy["cell_count"]
    )
    oracle_rows = [
        row_by_method[(cell_id, execution_path, normalized_policy["oracle"])]
        for cell_id, execution_path in ordered_scopes
    ]
    selected_rows = [
        row_by_method[(cell_id, execution_path, normalized_policy["selected"])]
        for cell_id, execution_path in ordered_scopes
    ]
    calibration_count_pass = bool(oracle_rows) and all(
        row["attempted"] == normalized_policy["calibration_count"]
        for row in oracle_rows
    )
    oracle_method_pass = bool(oracle_rows) and all(
        row["passes_all_gates"] for row in oracle_rows
    )
    selected_method_pass = bool(selected_rows) and all(
        row["passes_all_gates"] for row in selected_rows
    )
    pairwise_pass = bool(pairwise_rows) and all(
        row["passes_all_gates"] for row in pairwise_rows
    )

    failing_gates = []
    calibration_pass = False
    candidate_pass = False
    if not cell_family_complete:
        failing_gates.append("scientific_cell_family")
    if mode == "oracle_calibration":
        if not calibration_count_pass:
            failing_gates.append("oracle_calibration_count")
        if not oracle_method_pass:
            failing_gates.append("oracle_method_gates")
        calibration_pass = not failing_gates
    elif mode == "candidate_evaluation":
        if not selected_method_pass:
            failing_gates.append("selected_method_gates")
        if not pairwise_pass:
            failing_gates.append("named_pairwise_gates")
        candidate_pass = not failing_gates

    validation_pass = calibration_pass or candidate_pass
    certification_policy_match = (
        normalized_policy["sha256"] == CERTIFICATION_POLICY_SHA256
    )
    receipt = {
        "schema": SCHEMA,
        "mode": mode,
        "policy_sha256": normalized_policy["sha256"],
        "certification_policy_sha256": CERTIFICATION_POLICY_SHA256,
        "certification_policy_match": certification_policy_match,
        "scorer_source_sha256": scorer_source_sha256,
        "source_sha256": source["source_sha256"],
        "source_preregistration_sha256": source[
            "source_preregistration_sha256"
        ],
        "run_manifest_sha256": source["run_manifest_sha256"],
        "run_commit_sha256": source["run_commit_sha256"],
        "seed_domain_role": source["seed_domain_role"],
        "seed_domain": source["seed_domain"],
        "calibration_receipt_sha256": calibration_receipt_sha256,
        "bias_family": {
            "scientific_cell_count": normalized_policy["cell_count"],
            "observed_cell_count": len(cells),
            "familywise_alpha": _ratio(
                normalized_policy["familywise_alpha"]
            ),
            "standardized_bias_tolerance": _ratio(
                normalized_policy["bias_tolerance"]
            ),
            "minimum_scored_per_cell": normalized_policy[
                "minimum_bias_count"
            ],
            "calibration_count_per_cell": normalized_policy[
                "calibration_count"
            ],
            "criterion": (
                "K*2*P(T[n-1]>standardized_bias_tolerance*sqrt(n))"
                "<=familywise_alpha"
            ),
        },
        "selected_method": normalized_policy["selected"],
        "oracle_method": normalized_policy["oracle"],
        "named_baselines": list(normalized_policy["baselines"]),
        "cell_family_complete": cell_family_complete,
        "oracle_method_pass": oracle_method_pass,
        "selected_method_pass": selected_method_pass,
        "pairwise_pass": pairwise_pass,
        "calibration_pass": calibration_pass,
        "candidate_pass": candidate_pass,
        "validation_pass": validation_pass,
        "certification_eligible": (
            validation_pass
            and certification_policy_match
            and mode != "forensic_v5"
        ),
        "retroactive_v5_certification": False,
        "failing_gates": failing_gates,
        "method_rows": method_rows,
        "pairwise_rows": pairwise_rows,
    }
    receipt["receipt_sha256"] = _receipt_sha256(receipt)
    return receipt
