#!/usr/bin/env python3
"""Generate frozen #53 cells and score them through the Rust batch target."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import stat
import subprocess
from pathlib import Path


PAIRWISE_DIMENSIONS = (
    "date_count",
    "rho_at_12_days",
    "cadence",
    "missingness",
    "variance_ratio",
    "variance_arrangement",
    "reference_contribution_ratio",
    "reference_context",
)


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


def seed_identity(preregistration: dict, cell_index: int, outer_seed_index: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    digest.update(b"dolphinrust:temporal-covariance:outer-seed:v1\0")
    digest.update(int(preregistration["global_seed"]).to_bytes(8, "big"))
    digest.update(int(cell_index).to_bytes(8, "big"))
    digest.update(int(outer_seed_index).to_bytes(8, "big"))
    raw = digest.digest()
    return int.from_bytes(raw[:8], "big"), raw.hex()


def all_supported_cells(preregistration: dict) -> list[dict]:
    result: list[dict] = []
    supported_rho_counts = set(preregistration["rho_085_supported_date_counts"])
    for count in preregistration["date_counts"]:
        for rho in preregistration["rho_at_12_days"]:
            if rho == 0.85 and count not in supported_rho_counts:
                continue
            for cadence in preregistration["cadence"]:
                for missingness in preregistration["missingness"]:
                    for variance_ratio in preregistration["variance_ratios"]:
                        for arrangement in preregistration["variance_arrangements"]:
                            for reference_ratio in preregistration["reference_contribution_ratios"]:
                                for reference in preregistration["reference_contexts"]:
                                    result.append({
                                        "date_count": count,
                                        "rho_at_12_days": rho,
                                        "cadence": cadence["name"],
                                        "missingness": missingness,
                                        "variance_ratio": variance_ratio,
                                        "variance_arrangement": arrangement,
                                        "reference_contribution_ratio": reference_ratio,
                                        "reference_context": reference["name"],
                                        "overlap_fraction": reference["overlap_fraction"],
                                        "distance_pixels": reference["distance_pixels"],
                                        "sequential_depth": reference["sequential_depth"],
                                        "approximation": reference["approximation"],
                                    })
    return result


def _pair_tokens(cell: dict) -> frozenset[tuple[str, str, str, str]]:
    tokens = set()
    for left_index, left in enumerate(PAIRWISE_DIMENSIONS):
        for right in PAIRWISE_DIMENSIONS[left_index + 1:]:
            tokens.add((left, repr(cell[left]), right, repr(cell[right])))
    return frozenset(tokens)


def _cell_order(cell: dict) -> tuple:
    return tuple(repr(cell[dimension]) for dimension in PAIRWISE_DIMENSIONS)


def cells(preregistration: dict) -> list[dict]:
    """Return the deterministic greedy pairwise covering array.

    The candidate universe retains every supported level and the rho/date
    constraint. Selection covers every feasible dimension pair at least once;
    lexical tie-breaking makes the frozen list independent of hash iteration.
    """
    candidates = all_supported_cells(preregistration)
    tokens = [_pair_tokens(cell) for cell in candidates]
    uncovered = set().union(*tokens)
    selected: list[dict] = []
    remaining = set(range(len(candidates)))
    while uncovered:
        best = min(
            remaining,
            key=lambda index: (-len(tokens[index] & uncovered), _cell_order(candidates[index])),
        )
        covered = tokens[best] & uncovered
        if not covered:
            raise RuntimeError("pairwise design cannot cover every feasible pair")
        selected.append(dict(candidates[best]))
        uncovered.difference_update(covered)
        remaining.remove(best)
    for index, cell in enumerate(selected):
        cell["cell_index"] = index
        cell["cell_id"] = "c%05d-%02d-%s-%s-v%s-%s-r%s-%s" % (
            index,
            cell["date_count"],
            cell["cadence"].replace("_", "-"),
            cell["missingness"].replace("_", "-"),
            cell["variance_ratio"],
            cell["variance_arrangement"][:3],
            cell["reference_contribution_ratio"],
            cell["reference_context"].replace("_", "-"),
        )
    return selected


def production_cells(preregistration: dict, frozen_cells: list[dict] | None = None) -> list[dict]:
    """Return every pairwise cell for the actual production path."""
    return list(cells(preregistration) if frozen_cells is None else frozen_cells)


def unsupported_cells(preregistration: dict) -> list[dict]:
    """Frozen fail-closed cells excluded from promotion denominators."""
    return [dict(cell, cell_index=index, cell_id=f"u{index:02d}-{cell['stratum']}")
            for index, cell in enumerate(preregistration["unsupported_strata"])]


def cell_hash(frozen_cells: list[dict]) -> str:
    payload = json.dumps(frozen_cells, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def missing_width(cell: dict) -> int:
    if cell["missingness"] == "none":
        return 0
    fraction = {"mcar_10_percent": 0.10, "mcar_25_percent": 0.25,
                "contiguous_20_percent": 0.20}[cell["missingness"]]
    return max(1, math.ceil(cell["date_count"] * fraction / (1.0 - fraction)))


def days_for(cell: dict) -> list[float]:
    count = cell["date_count"] + missing_width(cell)
    gaps = []
    for index in range(count):
        if cell["cadence"] == "alternating_6_18_day":
            gap = 6.0 if index % 2 == 0 else 18.0
        elif cell["cadence"] == "jitter_4_day":
            gap = (8.0, 12.0, 16.0)[index % 3]
        elif cell["cadence"] == "two_36_day_gaps" and index in (count // 3, 2 * count // 3):
            gap = 36.0
        else:
            gap = 12.0
        gaps.append(gap)
    days = [0.0]
    for gap in gaps:
        days.append(days[-1] + gap)
    return days


def normal_noise(state: int) -> tuple[int, float]:
    state = splitmix64(state)
    u1 = max((state >> 11) / float(1 << 53), 1e-15)
    state = splitmix64(state)
    u2 = (state >> 11) / float(1 << 53)
    return state, math.sqrt(-2.0 * math.log(u1)) * math.cos(math.tau * u2)


PROPER_COMPLEX_MOMENT_SEEDS = 4096
SOURCE_CORRELATION_MODEL = "exponential_euclidean_v1"
SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS = 1.5
OUTER_COVERAGE_DGP = "physical_raw_space_v1"
CONDITIONAL_COVARIANCE_ORACLE = "fixed_capture_common_factor_monte_carlo_v1"
FROZEN_SOURCE_SET_SCHEMA = "dolphinrust.canonical-producer-source-set/2"
FROZEN_SOURCE_SET_SHA256 = "d4358eaf1e3ca6d65da78c61df7058835e2978390edf1e55f2faf8b25c842b55"
FROZEN_SOURCE_SET_ROOTS = ("crates",)
FROZEN_SOURCE_SET_FILES = (
    "Cargo.lock",
    "Cargo.toml",
    "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs",
    "validation/temporal_covariance_simulation.py",
    "crates/dolphin-timeseries/src/temporal_covariance.rs",
)
FROZEN_SOURCE_SET_NORMALIZED_ASSIGNMENTS = ("FROZEN_SOURCE_SET_SHA256",)
MAX_REQUEST_LINE_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_LINE_BYTES = 4 * 1024 * 1024
MAX_SHARD_RECORD_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_COMMIT_BYTES = 64 * 1024
MAX_FINAL_RECEIPT_BYTES = 1024 * 1024
MAX_PROBE_RECORDS = 16
RUN_IDENTITY_SCHEMA = "dolphinrust-temporal-covariance-run-identity/1"
SHARD_MANIFEST_SCHEMA = "dolphinrust-temporal-covariance-shard-manifest/1"
SHARD_COMMIT_SCHEMA = "dolphinrust-temporal-covariance-shard-commit/1"


def _proper_complex_draw(
        cell_index: int, outer_seed_index: int, date_index: int, column: int,
        role: int) -> tuple[float, float]:
    key = splitmix64((cell_index + 1) ^ 0xA0761D6478BD642F)
    key ^= splitmix64((outer_seed_index + 1) ^ 0xE7037ED1A0B428DB)
    key ^= splitmix64((date_index + 1) ^ 0x8EBC6AF09C88C6E3)
    key ^= splitmix64((column + 1) ^ 0x589965CC75374CC3)
    key ^= splitmix64((role + 1) ^ 0x1D8E4E27C47D124F)
    key, real = normal_noise(key)
    _, imaginary = normal_noise(key)
    return real, imaginary


def proper_complex_speckle(
        cell_index: int, outer_seed_index: int, column: int) -> tuple[float, float]:
    return proper_complex_spatial_draw(cell_index, outer_seed_index, 0, 0)[column]


def proper_complex_innovation(
        cell_index: int, outer_seed_index: int, date_index: int, column: int) -> tuple[float, float]:
    return proper_complex_spatial_draw(
        cell_index, outer_seed_index, date_index, 1
    )[column]


def spatial_correlation(left: int, right: int) -> float:
    return math.exp(
        -abs(left - right) / SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
    )


def spatial_correlation_cholesky(width: int) -> list[list[float]]:
    covariance = [
        [spatial_correlation(left, right) for right in range(width)]
        for left in range(width)
    ]
    lower = [[0.0 for _ in range(width)] for _ in range(width)]
    for row in range(width):
        for column in range(row + 1):
            residual = covariance[row][column] - sum(
                lower[row][inner] * lower[column][inner]
                for inner in range(column)
            )
            lower[row][column] = (
                math.sqrt(residual) if row == column
                else residual / lower[column][column]
            )
    return lower


def proper_complex_spatial_draw(
        cell_index: int, outer_seed_index: int, date_index: int,
        role: int, width: int = 7) -> tuple[tuple[float, float], ...]:
    independent = [
        complex(*_proper_complex_draw(
            cell_index, outer_seed_index, date_index, column, role
        ))
        for column in range(width)
    ]
    lower = spatial_correlation_cholesky(width)
    return tuple(
        (
            sum(lower[row][column] * independent[column]
                for column in range(row + 1)).real,
            sum(lower[row][column] * independent[column]
                for column in range(row + 1)).imag,
        )
        for row in range(width)
    )


def support_columns(column: int, width: int) -> tuple[int, ...]:
    if width < 3 or column < 0 or column >= width:
        raise ValueError("three-column support is outside the native width")
    start = min(max(column - 1, 0), width - 3)
    return tuple(range(start, start + 3))


def support_intersection_correlation(target_column: int, reference_column: int, width: int) -> float:
    target = set(support_columns(target_column, width))
    reference = set(support_columns(reference_column, width))
    return len(target & reference) / 3.0


def production_support_correlation(
        target_column: int, reference_column: int, width: int) -> float:
    target = support_columns(target_column, width)
    reference = support_columns(reference_column, width)
    covariance = sum(spatial_correlation(left, right)
                     for left in target for right in reference)
    target_variance = sum(spatial_correlation(left, right)
                          for left in target for right in target)
    reference_variance = sum(spatial_correlation(left, right)
                             for left in reference for right in reference)
    return covariance / math.sqrt(target_variance * reference_variance)


def production_temporal_noise_fraction(
        difference_variance: float,
        support_correlation: float) -> float:
    if difference_variance < 0.0 or not 0.0 <= support_correlation < 1.0:
        raise ValueError("production noise moment is outside the supported contract")
    output_variance = difference_variance / (2.0 * (1.0 - support_correlation))
    return -math.expm1(-output_variance)


def stationary_ar_path(days: list[float], rho: float, state: int) -> tuple[int, list[float]]:
    """Draw the exact stationary irregular continuous-time AR(1) truth."""
    state, current = normal_noise(state)
    values = [current]
    for left, right in zip(days, days[1:]):
        state, innovation = normal_noise(state)
        phi = rho ** ((right - left) / 12.0)
        current = phi * current + math.sqrt(max(0.0, 1.0 - phi * phi)) * innovation
        values.append(current)
    return state, values


def missing_indices(cell: dict, seed: int, scheduled_count: int) -> set[int]:
    width = missing_width(cell)
    if cell["missingness"] == "none":
        return set()
    if cell["missingness"] == "contiguous_20_percent":
        start = 1 + splitmix64(seed) % max(1, scheduled_count - width)
        return set(range(start, start + width))
    candidates = list(range(1, scheduled_count + 1))
    state = seed
    for index in range(len(candidates) - 1, 0, -1):
        state = splitmix64(state)
        other = state % (index + 1)
        candidates[index], candidates[other] = candidates[other], candidates[index]
    selected = []
    for candidate in candidates:
        trial = sorted(selected + [candidate])
        longest_run = max(
            (sum(1 for _ in group) for _, group in itertools.groupby(
                enumerate(trial), lambda pair: pair[1] - pair[0]
            )),
            default=0,
        )
        if longest_run <= 2:
            selected.append(candidate)
            if len(selected) == width:
                break
    if len(selected) != width:
        raise RuntimeError("unable to construct supported seed-varying MCAR mask")
    return set(selected)


def capture_scope_sha256(request: dict) -> str:
    production = request["production_path"]
    payload = {
        "cell_id": request["cell_id"],
        "cell_index": request["cell_index"],
        "days": request["days"],
        "native_shape": production["native_shape"],
        "reference": production["reference"],
        "reference_pixel": production["reference_pixel"],
        "scope": production["scope"],
        "source_seed": production["source_seed"],
        "target": production["target"],
        "source_correlation_model": production["source_correlation_model"],
        "source_correlation_distance_scale_pixels": production[
            "source_correlation_distance_scale_pixels"
        ],
        "outer_coverage_dgp": production["outer_coverage_dgp"],
        "conditional_covariance_oracle": production["conditional_covariance_oracle"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def request_for(
        cell: dict, outer_seed_index: int, preregistration: dict,
        execution_path: str, retain_dense_evidence: bool = False) -> dict:
    seed, seed_sha256 = seed_identity(preregistration, cell["cell_index"], outer_seed_index)
    days = days_for(cell)
    missing = missing_indices(cell, seed, len(days) - 1)
    observations = []
    carrier_values = []
    diagonal = []
    for index in range(len(days)):
        if cell["variance_arrangement"] == "alternating":
            scale = 1.0 if index % 2 == 0 else cell["variance_ratio"]
        else:
            scale = 1.0 if index < len(days) // 2 else cell["variance_ratio"]
        diagonal.append(0.01 * (scale + cell["reference_contribution_ratio"]))
    geometric_mean = math.exp(sum(math.log(value) for value in diagonal[1:]) / (len(diagonal) - 1))
    process_variance = 0.04
    state = seed
    state, ar_path = stationary_ar_path(days, cell["rho_at_12_days"], state)
    for index, day in enumerate(days):
        state, measurement = normal_noise(state)
        shape = math.sqrt(diagonal[index] / geometric_mean)
        carrier = 0.01 * day + math.sqrt(process_variance) * shape * ar_path[index]
        carrier = 0.0 if index == 0 else carrier
        value = carrier + math.sqrt(diagonal[index]) * measurement
        carrier_values.append(carrier)
        observations.append(0.0 if index == 0 else (None if index in missing else value))
    covariance = [[0.0 for _ in days] for _ in days]
    for index in range(1, len(days)):
        covariance[index][index] = diagonal[index]
    options = {
        "reference_lag_days": 12.0,
        "rho_min": 0.0,
        "rho_max": 0.98,
        "oracle_rho": cell["rho_at_12_days"],
        "oracle_process_variance": process_variance,
        "condition_limit": preregistration["condition_number_limit"],
        "minimum_dates": preregistration["minimum_dates"],
        "bootstrap_replicates": preregistration["bootstrap"]["count"],
        "bootstrap_minimum_successes": preregistration["bootstrap"]["minimum_successes"],
        "bootstrap_seed": splitmix64(seed ^ 0xB0057),
        "profile_max_expansions": preregistration["optimizer"]["profile_max_expansions"],
        "profile_max_iterations": preregistration["optimizer"]["profile_max_iterations"],
        "optimizer_max_iterations": preregistration["optimizer"]["max_iterations"],
        "optimizer_tolerance": preregistration["optimizer"]["tolerance"],
        "process_variance_min_ratio": preregistration["optimizer"]["process_variance_min_ratio"],
        "process_variance_max_ratio": preregistration["optimizer"]["process_variance_max_ratio"],
        "minimum_profile_curvature": preregistration["optimizer"]["minimum_profile_curvature"],
        "minimum_gap_days": preregistration["supported_cell_predicate"]["scheduled_acquisition_minimum_gap_days"],
        "maximum_gap_days": preregistration["supported_cell_predicate"]["scheduled_acquisition_maximum_gap_days"],
    }
    request = {"execution_path": execution_path, "cell_id": cell["cell_id"],
               "cell_index": cell["cell_index"], "outer_seed_index": outer_seed_index,
               "seed_sha256": seed_sha256, "seed": seed, "days": days,
               "options": options, "fixed_factor": None, "production_path": None,
               "retain_dense_evidence": retain_dense_evidence,
               "conditional_oracle_replicates": 0}
    if execution_path == "fixed_factor":
        request["fixed_factor"] = {"observations": observations,
                                   "difference_covariance": covariance}
    else:
        raw_complex_stack = []
        carrier_stack = []
        common_speckle = [
            complex(*value) for value in proper_complex_spatial_draw(
                cell["cell_index"], outer_seed_index, 0, 0
            )
        ]
        for date_index, value in enumerate(carrier_values):
            reference_column = {"near_exact": 2, "mid_exact": 3, "far_exact": 5}[
                cell["reference_context"]
            ]
            distance = reference_column - 1
            support_correlation = production_support_correlation(1, reference_column, 7)
            noise_fraction = production_temporal_noise_fraction(
                0.0 if date_index == 0 else diagonal[date_index], support_correlation
            )
            innovation_row = proper_complex_spatial_draw(
                cell["cell_index"], outer_seed_index, date_index, 1
            )
            row = []
            carrier_row = []
            for column in range(7):
                phase = ((reference_column - column) / distance) * value
                carrier = complex(math.cos(phase), math.sin(phase))
                innovation = complex(*innovation_row[column])
                source = (
                    math.sqrt(1.0 - noise_fraction) * common_speckle[column]
                    + math.sqrt(noise_fraction) * innovation
                )
                sample = carrier * source
                row.append([sample.real, sample.imag])
                carrier_row.append([carrier.real, carrier.imag])
            raw_complex_stack.append(row)
            carrier_stack.append(carrier_row)
        realized_overlap = {"near_exact": 0.5, "mid_exact": 0.2, "far_exact": 0.0}[
            cell["reference_context"]
        ]
        reference = {
            "geometry_id": "synthetic_same_frame_reference",
            "window_id": cell["reference_context"],
            "overlap_fraction": realized_overlap,
            "distance_pixels": float(reference_column - 1),
            "sequential_depth": (len(days) - 1) // 3,
            "approximation": cell["approximation"],
        }
        request["production_path"] = {
            "source_seed": seed,
            "native_shape": [1, 7],
            "target": [0, 1],
            "reference_pixel": [0, reference_column],
            "raw_complex_stack": raw_complex_stack,
            "carrier_stack": carrier_stack,
            "intended_difference_variance": [0.0] + diagonal[1:],
            "source_correlation_model": SOURCE_CORRELATION_MODEL,
            "source_correlation_distance_scale_pixels": SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS,
            "outer_coverage_dgp": OUTER_COVERAGE_DGP,
            "conditional_covariance_oracle": CONDITIONAL_COVARIANCE_ORACLE,
            "validity": [value is not None for value in observations],
            "reference": reference,
            "scope": "synthetic_validation",
            "capture_scope_sha256": "",
            "validation_receipt_sha256": "53" * 32,
            "selected_method": "complete_refit_bootstrap",
        }
        request["production_path"]["capture_scope_sha256"] = capture_scope_sha256(request)
    return request


METHOD_FIELDS = {
        "ols": "ols", "oracle_gls": "oracle_gls",
        "legacy_intercept_slope_wls_non_comparable": "conditional_wls",
        "lag_one_scalar_effective_n": "scalar_effective_n",
        "plugin_gls_reml": "plugin_gls",
        "reml_covariance_parameter_adjusted_scalar": "adjusted_scalar",
        "slope_profile_likelihood_ml": "adjusted_profile",
        "complete_refit_bootstrap": "complete_refit_bootstrap",
}
COMPARATOR_METHOD_IDENTITIES = [
    "ols", "oracle_gls", "legacy_intercept_slope_wls_non_comparable",
    "lag_one_scalar_effective_n", "plugin_gls_reml",
    "reml_covariance_parameter_adjusted_scalar", "slope_profile_likelihood_ml",
    "complete_refit_bootstrap",
]
TEMPORAL_STATUSES = {
    "Evaluated", "InsufficientDates", "DatesNotStrictlyIncreasing", "GaugeMissing",
    "GaugeNotZero", "DesignRankDeficient", "DesignIllConditioned",
    "CovarianceNonfinite", "TotalCovarianceNotPositiveDefinite",
    "CovarianceParameterAtBoundary", "RhoLowerBoundary", "RhoUpperBoundary",
    "ProcessVarianceLowerBoundary", "ProcessVarianceUpperBoundary",
    "BootstrapInsufficientSuccess", "UnsupportedCadence", "OptimizerNonconverged",
    "WeakParameterIdentification", "LegacyNonComparable",
}
PRODUCTION_FAIL_CLOSED_STATUSES = {
    "production_inputs_missing", "source_seed_mismatch", "production_contract_mismatch",
    "capture_scope_mismatch", "raw_complex_invalid", "source_model_invalid",
    "carrier_sequential_failed", "topology planning failed",
    "production sequential capture failed", "production replay preflight failed",
    "production replay failed", "production replay exceeded its exact preflight receipt",
    "target fixed-L2 map failed", "reference fixed-L2 map failed",
    "fixed-L2 covariance propagation failed", "reference_context_mismatch",
    "source_correlation_mismatch", "conditional common-factor oracle input is invalid",
    "conditional common-factor covariance is not positive semidefinite",
}


class MethodReducer:
    def __init__(self) -> None:
        self.attempted = 0
        self.scored = 0
        self.bias_mean = 0.0
        self.bias_m2 = 0.0
        self.interval_counts = {label: 0 for label in ("68", "90", "95")}
        self.covered = {label: 0 for label in ("68", "90", "95")}
        self.width_sums = {label: 0.0 for label in ("68", "90", "95")}
        self.score_sums = {label: 0.0 for label in ("68", "90", "95")}

    def update(self, comparator: dict | None, truth: float) -> None:
        self.attempted += 1
        if not comparator or comparator.get("point_estimate") is None:
            return
        bias = comparator["point_estimate"] - truth
        self.scored += 1
        delta = bias - self.bias_mean
        self.bias_mean += delta / self.scored
        self.bias_m2 += delta * (bias - self.bias_mean)
        for label, level in (("68", 0.68), ("90", 0.90), ("95", 0.95)):
            interval = comparator.get(f"interval_{label}")
            if interval is None:
                continue
            lower, upper = interval["lower"], interval["upper"]
            if not all(math.isfinite(value) for value in (lower, upper)) or upper < lower:
                continue
            alpha = 1.0 - level
            penalty = (2.0 / alpha) * max(lower - truth, 0.0)
            penalty += (2.0 / alpha) * max(truth - upper, 0.0)
            self.interval_counts[label] += 1
            self.covered[label] += lower <= truth <= upper
            self.width_sums[label] += upper - lower
            self.score_sums[label] += upper - lower + penalty

    def finalize(self, preregistration: dict) -> dict:
        bias_sd = math.sqrt(self.bias_m2 / (self.scored - 1)) if self.scored > 1 else None
        standardized = abs(self.bias_mean) / bias_sd if bias_sd and bias_sd > 0.0 else None
        aggregate = {
            "attempted": self.attempted,
            "scored": self.scored,
            "failed": self.attempted - self.scored,
            "emission_fraction": self.scored / self.attempted if self.attempted else 0.0,
            "mean_bias": self.bias_mean if self.scored else None,
            "standardized_bias": standardized,
        }
        gates = {
            "emission": aggregate["emission_fraction"]
                >= preregistration["thresholds"]["minimum_successful_emission_fraction"],
            "standardized_bias": standardized is not None
                and standardized <= preregistration["thresholds"]["standardized_bias"],
        }
        for label, nominal in (("68", 0.68), ("90", 0.90), ("95", 0.95)):
            count = self.interval_counts[label]
            coverage = self.covered[label] / count if count else None
            aggregate[f"coverage_{label}"] = coverage
            aggregate[f"mean_width_{label}"] = self.width_sums[label] / count if count else None
            aggregate[f"mean_interval_score_{label}"] = self.score_sums[label] / count if count else None
            gates[f"coverage_{label}"] = coverage is not None and abs(coverage - nominal) <= (
                preregistration["thresholds"]["coverage"][f"0.{label}"])
        return {"aggregate": aggregate, "gates": gates}


class StreamingScores:
    def __init__(self, preregistration: dict) -> None:
        self.preregistration = preregistration
        self.truth = 0.01 * 365.25
        frozen = cells(preregistration)
        self.cells_by_id = {cell["cell_id"]: cell for cell in frozen}
        self.reducers = {
            (cell["cell_id"], path, method): MethodReducer()
            for cell in frozen
            for path in preregistration["execution_paths"]
            for method in METHOD_FIELDS
        }
        self.next_seed = {
            (cell["cell_id"], path): 0
            for cell in frozen
            for path in preregistration["execution_paths"]
        }

    def update(self, record: dict) -> None:
        key = (record.get("cell_id"), record.get("execution_path"))
        if key not in self.next_seed:
            raise RuntimeError("batch returned an unknown cell or execution path")
        expected_index = self.next_seed[key]
        if record.get("outer_seed_index") != expected_index:
            raise RuntimeError("batch returned a duplicate, missing, or reordered outer seed")
        cell = self.cells_by_id[key[0]]
        expected_seed, expected_digest = seed_identity(
            self.preregistration, cell["cell_index"], expected_index)
        if record.get("seed") != expected_seed or record.get("seed_sha256") != expected_digest:
            raise RuntimeError("batch returned a stale or mismatched seed identity")
        self.next_seed[key] += 1
        fit = record.get("fit")
        for method, field in METHOD_FIELDS.items():
            comparator = fit.get(field) if fit else None
            self.reducers[(key[0], key[1], method)].update(comparator, self.truth)

    def finalize(self, require_complete: bool) -> dict:
        expected = self.preregistration["outer_seeds_per_supported_cell"]
        if require_complete and any(count != expected for count in self.next_seed.values()):
            raise RuntimeError("batch did not return the exact frozen seed denominator for every cell")
        summaries = []
        global_methods = {
            method: {"attempted": 0, "scored": 0, "failed": 0}
            for method in METHOD_FIELDS
        }
        all_selected_pass = True
        for cell in sorted(self.cells_by_id.values(), key=lambda value: value["cell_index"]):
            for path in self.preregistration["execution_paths"]:
                methods = {
                    method: self.reducers[(cell["cell_id"], path, method)].finalize(self.preregistration)
                    for method in METHOD_FIELDS
                }
                oracle = methods["oracle_gls"]["aggregate"]
                for method, result in methods.items():
                    aggregate = result["aggregate"]
                    score = aggregate["mean_interval_score_95"]
                    width = aggregate["mean_width_95"]
                    oracle_score = oracle["mean_interval_score_95"]
                    oracle_width = oracle["mean_width_95"]
                    result["gates"]["proper_score"] = (
                        score is not None and oracle_score is not None
                        and score <= oracle_score * (1.0 + self.preregistration["thresholds"]["proper_score"])
                    )
                    result["gates"]["interval_width"] = (
                        width is not None and oracle_width is not None
                        and width <= oracle_width * self.preregistration["thresholds"]["maximum_interval_width_ratio"]
                    )
                    result["passes_all_gates"] = all(result["gates"].values())
                    for field in ("attempted", "scored", "failed"):
                        global_methods[method][field] += aggregate[field]
                all_selected_pass = all_selected_pass and all(
                    methods[method]["passes_all_gates"]
                    for method in self.preregistration["promotion_methods"]
                )
                summaries.append({
                    "cell_id": cell["cell_id"],
                    "cell_index": cell["cell_index"],
                    "execution_path": path,
                    "attempted": self.next_seed[(cell["cell_id"], path)],
                    "methods": methods,
                })
        return {
            "schema": self.preregistration["schemas"]["scorer"],
            "truth_slope_per_year": self.truth,
            "methods": global_methods,
            "cell_summaries": summaries,
            "promotion_methods": self.preregistration["promotion_methods"],
            "all_methods_pass": all_selected_pass,
        }


def score_records(records: list[dict], preregistration: dict) -> dict:
    scorer = StreamingScores(preregistration)
    for record in records:
        scorer.update(record)
    return scorer.finalize(require_complete=False)


def iter_requests(preregistration: dict, seed_count: int):
    for cell in cells(preregistration):
        for outer_seed_index in range(seed_count):
            for execution_path in preregistration["execution_paths"]:
                yield request_for(
                    cell,
                    outer_seed_index,
                    preregistration,
                    execution_path,
                )


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _open_bounded_regular(path: Path, byte_cap: int):
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("descriptor-bound validation requires O_NOFOLLOW")
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
    except OSError as error:
        raise RuntimeError(f"{path.name} is not an openable regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{path.name} is not a regular file")
        if metadata.st_size > byte_cap:
            raise RuntimeError(f"{path.name} exceeds its retained byte cap")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_bounded_regular(path: Path, byte_cap: int) -> bytes:
    with _open_bounded_regular(path, byte_cap) as handle:
        payload = handle.read(byte_cap + 1)
    if len(payload) > byte_cap:
        raise RuntimeError(f"{path.name} exceeds its retained byte cap")
    return payload


def sha256_file(path: Path, byte_cap: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with _open_bounded_regular(path, byte_cap) as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > byte_cap:
                raise RuntimeError(f"{path.name} exceeds its retained byte cap")
            digest.update(chunk)
    return digest.hexdigest(), size


def _producer_source_bytes(path: Path, relative_path: str) -> bytes:
    raw = _read_bounded_regular(path, 64 * 1024 * 1024)
    if relative_path != "validation/temporal_covariance_simulation.py":
        return raw
    try:
        lines = raw.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise RuntimeError("temporal producer source is not UTF-8") from error
    seen = set()
    normalized = []
    for line in lines:
        name = next(
            (candidate for candidate in FROZEN_SOURCE_SET_NORMALIZED_ASSIGNMENTS
             if line.startswith(f'{candidate} = "')),
            None,
        )
        if name is None:
            normalized.append(line)
            continue
        if name in seen or not line.rstrip("\r\n").endswith('"'):
            raise RuntimeError("temporal producer identity assignment is malformed")
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        normalized.append(f'{name} = "<producer-source-set-v2>"{ending}')
        seen.add(name)
    if seen != set(FROZEN_SOURCE_SET_NORMALIZED_ASSIGNMENTS):
        raise RuntimeError("temporal producer identity assignments are missing")
    return "".join(normalized).encode("utf-8")


def _canonical_source_entries(source_root: Path) -> list[dict]:
    source_root = source_root.resolve(strict=True)
    paths = [source_root / name for name in FROZEN_SOURCE_SET_FILES]
    for root_name in FROZEN_SOURCE_SET_ROOTS:
        root = source_root / root_name
        paths.extend(
            path for path in root.rglob("*")
            if path.is_file() and (path.suffix == ".rs" or path.name == "Cargo.toml")
        )
    entries = []
    for path in sorted(set(paths), key=lambda value: value.relative_to(source_root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "source identity contains a missing, non-regular, or symlinked file"
            )
        relative_path = path.relative_to(source_root).as_posix()
        normalized = _producer_source_bytes(path, relative_path)
        entries.append({
            "path": relative_path,
            "byte_count": len(normalized),
            "sha256": hashlib.sha256(normalized).hexdigest(),
        })
    if not entries:
        raise RuntimeError("source identity set is empty")
    return entries


def canonical_source_set_sha256(source_root: Path) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "schema": FROZEN_SOURCE_SET_SCHEMA,
        "roots": list(FROZEN_SOURCE_SET_ROOTS),
        "files": list(FROZEN_SOURCE_SET_FILES),
        "normalized_assignments": list(FROZEN_SOURCE_SET_NORMALIZED_ASSIGNMENTS),
        "entries": _canonical_source_entries(source_root),
    })).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _remove_owned_regular(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not _regular_file(path):
        raise RuntimeError(f"refusing to remove non-regular run artifact {path.name}")
    path.unlink()


def atomic_write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    _remove_owned_regular(partial)
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(partial, path, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def producer_identity(preregistration: dict, binary: Path) -> dict:
    root = Path(__file__).parents[1]
    expected_binary = root / "target/release/examples/temporal_covariance_batch"
    try:
        resolved_binary = Path(binary).resolve(strict=True)
        resolved_expected = expected_binary.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("temporal producer must be the exact prebuilt release executable") from error
    metadata = resolved_binary.stat()
    if (
            resolved_binary != resolved_expected
            or not stat.S_ISREG(metadata.st_mode)
            or not os.access(resolved_binary, os.X_OK)):
        raise RuntimeError("temporal producer must be the exact prebuilt release executable")
    paths = {
        "generator_sha256": Path(__file__),
        "batch_source_sha256": (
            root / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs"
        ),
        "estimator_source_sha256": (
            root / "crates/dolphin-timeseries/src/temporal_covariance.rs"
        ),
    }
    actual_hashes = {
        identity: sha256_file(path, 16 * 1024 * 1024)[0]
        for identity, path in paths.items()
    }
    if actual_hashes != preregistration["file_hashes"]:
        raise RuntimeError("frozen temporal covariance source hashes do not match the run")
    source_set_sha256 = canonical_source_set_sha256(root)
    binary_sha256, binary_bytes = sha256_file(resolved_binary, 1024 * 1024 * 1024)
    frozen_identity = preregistration.get("producer_identity")
    expected_identity = {
        "schema": "dolphinrust.temporal-covariance.producer-identity/1",
        "source_set_schema": FROZEN_SOURCE_SET_SCHEMA,
        "source_set_sha256": source_set_sha256,
        "binary_path": "target/release/examples/temporal_covariance_batch",
        "binary_sha256": binary_sha256,
        "binary_bytes": binary_bytes,
    }
    if frozen_identity != expected_identity or source_set_sha256 != FROZEN_SOURCE_SET_SHA256:
        raise RuntimeError("frozen temporal producer source/binary identity does not match the run")
    return {
        "schema": RUN_IDENTITY_SCHEMA,
        "preregistration_sha256": hashlib.sha256(
            canonical_json_bytes(preregistration)
        ).hexdigest(),
        **actual_hashes,
        "source_set_schema": FROZEN_SOURCE_SET_SCHEMA,
        "source_set_sha256": source_set_sha256,
        "binary_path": "target/release/examples/temporal_covariance_batch",
        "binary_sha256": binary_sha256,
        "binary_bytes": binary_bytes,
        "batch_schema": preregistration["schemas"]["batch"],
        "generator_schema": preregistration["schemas"]["generator"],
        "source_correlation_model": SOURCE_CORRELATION_MODEL,
        "source_correlation_distance_scale_pixels": (
            SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
        ),
        "seed_count": preregistration["outer_seeds_per_supported_cell"],
    }


def initialize_run_root(run_root: Path, identity: dict) -> Path:
    if run_root.exists() and (run_root.is_symlink() or not run_root.is_dir()):
        raise RuntimeError("run root must be a real directory")
    run_root.mkdir(parents=True, exist_ok=True)
    identity_path = run_root / "run_identity.json"
    expected = canonical_json_bytes(identity) + b"\n"
    if len(expected) > MAX_COMMIT_BYTES:
        raise RuntimeError("run identity exceeds its retained byte cap")
    if identity_path.exists():
        if _read_bounded_regular(identity_path, MAX_COMMIT_BYTES) != expected:
            raise RuntimeError("run root identity is stale or malformed")
    else:
        atomic_write_no_replace(identity_path, expected)
    shards = run_root / "shards"
    if shards.exists() and (shards.is_symlink() or not shards.is_dir()):
        raise RuntimeError("run shard root must be a real directory")
    shards.mkdir(exist_ok=True)
    return shards


def _shard_paths(shards: Path, cell: dict, execution_path: str) -> dict[str, Path]:
    stem = f"{cell['cell_index']:05d}.{execution_path}"
    return {
        "records": shards / f"{stem}.jsonl",
        "manifest": shards / f"{stem}.manifest.json",
        "commit": shards / f"{stem}.commit.json",
    }


def _read_bounded_line(handle, cap: int) -> bytes:
    line = handle.readline(cap + 1)
    if len(line) > cap:
        raise RuntimeError("temporal covariance batch line exceeds its byte cap")
    if line and not line.endswith(b"\n"):
        raise RuntimeError("temporal covariance batch returned a partial line")
    return line


def _validate_compact_record(record: dict, request: dict, batch_schema: str) -> None:
    response_keys = {
        "schema", "execution_path", "cell_id", "cell_index", "outer_seed_index",
        "seed_sha256", "seed", "fixed_factor_status", "production_path_status",
        "comparator_methods", "attempted", "emitted", "failed", "fit", "provenance",
        "production_receipts", "resource",
    }
    identity = (
        "execution_path", "cell_id", "cell_index", "outer_seed_index",
        "seed_sha256", "seed",
    )
    if (
            not isinstance(record, dict)
            or set(record) != response_keys
            or record.get("schema") != batch_schema):
        raise RuntimeError("batch returned a malformed schema")
    if any(record.get(field) != request[field] for field in identity):
        raise RuntimeError("batch returned a stale or mismatched attempt identity")
    if record["comparator_methods"] != COMPARATOR_METHOD_IDENTITIES:
        raise RuntimeError("batch returned stale comparator identities")
    if record.get("attempted") is not True:
        raise RuntimeError("batch omitted the attempted disposition")
    if type(record.get("emitted")) is not bool or type(record.get("failed")) is not bool:
        raise RuntimeError("batch returned a malformed emission disposition")
    if record["emitted"] == record["failed"]:
        raise RuntimeError("batch emission and failure dispositions are inconsistent")
    resource = record.get("resource")
    if (
            not isinstance(resource, dict)
            or set(resource) != {
                "wall_micros", "resident_set_bytes_before", "resident_set_bytes_after"
            }
            or any(
                type(resource.get(field)) is not int or resource[field] < 0
                for field in (
                    "wall_micros", "resident_set_bytes_before", "resident_set_bytes_after"
                )
            )
    ):
        raise RuntimeError("batch returned a malformed resource receipt")

    def finite_optional(value, *, nonnegative=False):
        return value is None or (
            type(value) in (int, float)
            and math.isfinite(value)
            and (not nonnegative or value >= 0.0)
        )

    def validate_interval(interval):
        if interval is None:
            return True
        return (
            isinstance(interval, dict)
            and set(interval) == {"lower", "upper", "successful_replicates"}
            and finite_optional(interval["lower"])
            and finite_optional(interval["upper"])
            and interval["lower"] is not None
            and interval["upper"] is not None
            and interval["lower"] <= interval["upper"]
            and type(interval["successful_replicates"]) is int
            and interval["successful_replicates"] >= 0
        )

    def close(left, right):
        return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))

    fit = record["fit"]
    if fit is not None:
        fit_keys = {
            "status", "ols_slope", "oracle_gls_slope", "plugin_gls_slope",
            "adjusted_profile_slope", "bootstrap_slope", "bootstrap_interval",
            "fitted_rho", "fitted_process_variance", "raw_correlation",
            "valid_date_count", "rank", "degrees_of_freedom",
            "covariance_condition_number", "ols", "oracle_gls", "conditional_wls",
            "scalar_effective_n", "plugin_gls", "adjusted_scalar", "adjusted_profile",
            "complete_refit_bootstrap", "bootstrap_attempts", "bootstrap_successes",
        }
        if not isinstance(fit, dict) or set(fit) != fit_keys or fit["status"] not in TEMPORAL_STATUSES:
            raise RuntimeError("batch returned a malformed fit schema or status")
        for field in (
                "ols_slope", "oracle_gls_slope", "plugin_gls_slope",
                "adjusted_profile_slope", "bootstrap_slope", "fitted_rho",
                "fitted_process_variance", "covariance_condition_number"):
            if not finite_optional(fit[field], nonnegative=field in {
                    "fitted_process_variance", "covariance_condition_number"}):
                raise RuntimeError("batch returned a non-finite fit value")
        if not validate_interval(fit["bootstrap_interval"]):
            raise RuntimeError("batch returned a malformed bootstrap interval")
        for field in (
                "valid_date_count", "rank", "degrees_of_freedom",
                "bootstrap_attempts", "bootstrap_successes"):
            if type(fit[field]) is not int or fit[field] < 0:
                raise RuntimeError("batch returned a malformed fit count")
        if fit["bootstrap_successes"] > fit["bootstrap_attempts"]:
            raise RuntimeError("batch returned inconsistent bootstrap accounting")
        raw = fit["raw_correlation"]
        if (
                not isinstance(raw, dict)
                or set(raw) != {
                    "rho", "pair_count", "minimum_gap_days", "median_gap_days",
                    "maximum_gap_days"
                }
                or type(raw["pair_count"]) is not int
                or raw["pair_count"] < 0
                or any(not finite_optional(raw[field]) for field in (
                    "rho", "minimum_gap_days", "median_gap_days", "maximum_gap_days"
                ))):
            raise RuntimeError("batch returned malformed raw-correlation diagnostics")
        comparator_keys = {
            "point_estimate", "standard_error_diagnostic", "interval_68", "interval_90",
            "interval_95", "width_68", "width_90", "width_95", "status",
            "attempted_replicates", "successful_replicates",
        }
        for field in (
                "ols", "oracle_gls", "conditional_wls", "scalar_effective_n", "plugin_gls",
                "adjusted_scalar", "adjusted_profile", "complete_refit_bootstrap"):
            comparator = fit[field]
            if (
                    not isinstance(comparator, dict)
                    or set(comparator) != comparator_keys
                    or comparator["status"] not in TEMPORAL_STATUSES
                    or not finite_optional(comparator["point_estimate"])
                    or not finite_optional(
                        comparator["standard_error_diagnostic"], nonnegative=True
                    )
                    or any(not finite_optional(comparator[name], nonnegative=True)
                           for name in ("width_68", "width_90", "width_95"))
                    or any(not validate_interval(comparator[name])
                           for name in ("interval_68", "interval_90", "interval_95"))
                    or type(comparator["attempted_replicates"]) is not int
                    or type(comparator["successful_replicates"]) is not int
                    or comparator["attempted_replicates"] < 0
                    or comparator["successful_replicates"] < 0
                    or comparator["successful_replicates"]
                    > comparator["attempted_replicates"]):
                raise RuntimeError("batch returned malformed comparator diagnostics")
            present_intervals = [comparator[name] is not None for name in (
                "interval_68", "interval_90", "interval_95"
            )]
            present_widths = [comparator[name] is not None for name in (
                "width_68", "width_90", "width_95"
            )]
            if present_intervals != present_widths:
                raise RuntimeError("batch returned inconsistent comparator intervals")
            for level in ("68", "90", "95"):
                interval = comparator[f"interval_{level}"]
                width = comparator[f"width_{level}"]
                if interval is not None and (
                        interval["successful_replicates"]
                        != comparator["successful_replicates"]
                        or not close(width, interval["upper"] - interval["lower"])):
                    raise RuntimeError("batch returned inconsistent comparator interval width")
            if comparator["standard_error_diagnostic"] is not None \
                    and comparator["point_estimate"] is None:
                raise RuntimeError("batch returned a comparator error without an estimate")
            if any(present_intervals) and (
                    comparator["point_estimate"] is None
                    or comparator["standard_error_diagnostic"] is None):
                raise RuntimeError("batch returned intervals without complete comparator values")
            if comparator["status"] == "Evaluated" and (
                    comparator["point_estimate"] is None
                    or comparator["standard_error_diagnostic"] is None
                    or not all(present_intervals)):
                raise RuntimeError("batch returned incomplete evaluated comparator values")
            if field != "complete_refit_bootstrap" and (
                    comparator["attempted_replicates"] != 0
                    or comparator["successful_replicates"] != 0):
                raise RuntimeError("batch returned resamples for a non-bootstrap comparator")
        scalar_couplings = (
            ("ols_slope", "ols"),
            ("oracle_gls_slope", "oracle_gls"),
            ("adjusted_profile_slope", "adjusted_profile"),
            ("bootstrap_slope", "complete_refit_bootstrap"),
        )
        if any(fit[scalar] != fit[comparator]["point_estimate"]
               for scalar, comparator in scalar_couplings):
            raise RuntimeError("batch returned inconsistent fit and comparator estimates")
        plugin_point = fit["plugin_gls"]["point_estimate"]
        if plugin_point is not None and fit["plugin_gls_slope"] != plugin_point:
            raise RuntimeError("batch returned inconsistent plugin GLS estimates")
        if (
                fit["bootstrap_interval"]
                != fit["complete_refit_bootstrap"]["interval_95"]
                or fit["bootstrap_attempts"]
                != fit["complete_refit_bootstrap"]["attempted_replicates"]
                or fit["bootstrap_successes"]
                != fit["complete_refit_bootstrap"]["successful_replicates"]):
            raise RuntimeError("batch returned inconsistent bootstrap evidence")
        gap_values = [raw[field] for field in (
            "minimum_gap_days", "median_gap_days", "maximum_gap_days"
        )]
        if any(value is None for value in gap_values) != all(
                value is None for value in gap_values):
            raise RuntimeError("batch returned incomplete cadence diagnostics")
        if all(value is not None for value in gap_values) and not (
                0.0 < gap_values[0] <= gap_values[1] <= gap_values[2]):
            raise RuntimeError("batch returned inconsistent cadence diagnostics")
        if fit["status"] == "Evaluated" and (
                any(fit[field] is None for field in (
                    "ols_slope", "oracle_gls_slope", "plugin_gls_slope",
                    "adjusted_profile_slope", "bootstrap_slope", "bootstrap_interval",
                    "fitted_rho", "fitted_process_variance", "covariance_condition_number"
                ))
                or fit["valid_date_count"] == 0
                or fit["rank"] == 0
                or any(fit[field]["status"] != "Evaluated" for field in (
                    "ols", "oracle_gls", "scalar_effective_n", "plugin_gls",
                    "adjusted_profile", "complete_refit_bootstrap",
                ))):
            raise RuntimeError("batch returned an incomplete evaluated fit")

    evaluated = fit is not None and fit["status"] == "Evaluated"
    if record["emitted"] != evaluated:
        raise RuntimeError("batch fit and emission dispositions are inconsistent")
    if request["execution_path"] == "fixed_factor":
        if (
                record["production_path_status"] is not None
                or record["fixed_factor_status"] not in TEMPORAL_STATUSES
                or fit is None
                or record["fixed_factor_status"] != fit["status"]
                or record["provenance"] is not None
                or record["production_receipts"] is not None):
            raise RuntimeError("batch returned inconsistent fixed-factor state")
        return

    if record["fixed_factor_status"] is not None:
        raise RuntimeError("batch returned inconsistent production-path state")
    production_status = record["production_path_status"]
    if production_status == "evaluated":
        if not evaluated or record["provenance"] is None or record["production_receipts"] is None:
            raise RuntimeError("batch returned incomplete evaluated production state")
    elif production_status == "estimator_failed":
        if fit is None or evaluated or record["provenance"] is not None \
                or record["production_receipts"] is None:
            raise RuntimeError("batch returned inconsistent estimator-failure state")
    elif production_status in PRODUCTION_FAIL_CLOSED_STATUSES:
        if fit is not None or record["provenance"] is not None \
                or record["production_receipts"] is not None:
            raise RuntimeError("batch returned evidence for a fail-closed production state")
        return
    else:
        raise RuntimeError("batch returned an unknown production status")

    receipts = record.get("production_receipts")
    if receipts is not None:
        receipt_keys = {
            "capture_scope_sha256", "source_manifest_sha256", "source_model_sha256",
            "evd_operator_sha256", "evd_source_factor_sha256", "fixed_l2_map_sha256",
            "issue52_receipt_sha256", "issue54_receipt_sha256", "numeric_evidence_sha256",
            "source_correlation_model", "source_correlation_distance_scale_pixels",
            "source_correlation_support_union_count", "effective_looks_fraction",
            "source_correlation_receipt_sha256", "outer_coverage_dgp",
            "conditional_covariance_oracle", "conditional_oracle_replicates",
        }
        dense_fields = {
            "fixed_l2_difference_covariance", "fixed_l2_difference_variance",
            "carrier_difference_history", "linked_difference_history",
        }
        if request.get("retain_dense_evidence", False):
            receipt_keys.update(dense_fields)
        oracle_replicates = request.get("conditional_oracle_replicates", 0)
        if oracle_replicates > 0:
            receipt_keys.update({
                "conditional_oracle_covariance", "conditional_oracle_receipt_sha256"
            })
        if not isinstance(receipts, dict) or set(receipts) != receipt_keys:
            raise RuntimeError("batch returned a malformed production receipt schema")
        if (
            receipts.get("capture_scope_sha256")
            != request["production_path"]["capture_scope_sha256"]
            or receipts.get("source_correlation_model") != SOURCE_CORRELATION_MODEL
            or receipts.get("source_correlation_distance_scale_pixels")
            != SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
            or type(receipts.get("source_correlation_support_union_count")) is not int
            or receipts["source_correlation_support_union_count"] <= 0
            or receipts.get("outer_coverage_dgp") != OUTER_COVERAGE_DGP
            or receipts.get("conditional_covariance_oracle")
            != CONDITIONAL_COVARIANCE_ORACLE
        ):
            raise RuntimeError("batch returned stale source-correlation provenance")
        if not request.get("retain_dense_evidence", False) and any(
                field in receipts for field in dense_fields):
            raise RuntimeError("batch retained dense per-attempt evidence")
        for field, value in receipts.items():
            if field.endswith("_sha256") and (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise RuntimeError("batch omitted compact numeric evidence identity")
        if receipts.get("conditional_oracle_replicates") != oracle_replicates:
            raise RuntimeError("batch returned stale conditional-oracle provenance")
        oracle_fields = (
            "conditional_oracle_covariance", "conditional_oracle_receipt_sha256"
        )
        if oracle_replicates == 0 and any(field in receipts for field in oracle_fields):
            raise RuntimeError("batch retained an unrequested conditional oracle")
        if oracle_replicates > 0:
            covariance = receipts.get("conditional_oracle_covariance")
            receipt = receipts.get("conditional_oracle_receipt_sha256")
            if (
                    not isinstance(covariance, list)
                    or len(covariance) != len(request["days"])
                    or any(not isinstance(row, list) or len(row) != len(covariance)
                           for row in covariance)
                    or any(not finite_optional(value) or value is None
                           for row in covariance for value in row)
                    or not isinstance(receipt, str)
                    or len(receipt) != 64):
                raise RuntimeError("batch returned a malformed conditional oracle")
        if (
                not finite_optional(receipts["source_correlation_distance_scale_pixels"])
                or not finite_optional(receipts["effective_looks_fraction"])
                or not 0.0 < receipts["effective_looks_fraction"] <= 1.0):
            raise RuntimeError("batch returned malformed source-correlation measurements")

    provenance = record["provenance"]
    if provenance is not None:
        provenance_keys = {
            "schema", "estimator", "estimator_version", "valid_date_count", "rank",
            "degrees_of_freedom", "cadence_days", "raw_rho", "fitted_rho",
            "fitted_process_variance", "issue52_receipt_sha256", "issue54_receipt_sha256",
            "reference", "condition_number", "scope", "bootstrap_attempts",
            "bootstrap_successes", "validation_receipt_sha256", "estimator_input_sha256",
            "bootstrap_minimum_success_fraction", "selected_method",
        }
        if (
                not isinstance(provenance, dict)
                or set(provenance) != provenance_keys
                or provenance["schema"] != "dolphinrust-temporal-covariance-provenance/1"
                or provenance["estimator"] != "origin_anchored_temporal_covariance_slope"
                or provenance["estimator_version"] != "1.6.0"
                or provenance["valid_date_count"] != fit["valid_date_count"]
                or provenance["rank"] != fit["rank"]
                or provenance["degrees_of_freedom"] != fit["degrees_of_freedom"]
                or provenance["cadence_days"] != [
                    fit["raw_correlation"]["minimum_gap_days"],
                    fit["raw_correlation"]["median_gap_days"],
                    fit["raw_correlation"]["maximum_gap_days"],
                ]
                or provenance["raw_rho"] != fit["raw_correlation"]["rho"]
                or provenance["fitted_rho"] != fit["fitted_rho"]
                or provenance["fitted_process_variance"] != fit["fitted_process_variance"]
                or provenance["condition_number"] != fit["covariance_condition_number"]
                or provenance["bootstrap_attempts"] != fit["bootstrap_attempts"]
                or provenance["bootstrap_successes"] != fit["bootstrap_successes"]
                or provenance["issue52_receipt_sha256"] != receipts["issue52_receipt_sha256"]
                or provenance["issue54_receipt_sha256"] != receipts["issue54_receipt_sha256"]
                or provenance["reference"] != request["production_path"]["reference"]
                or provenance["scope"] != request["production_path"]["scope"]
                or provenance["validation_receipt_sha256"]
                != request["production_path"]["validation_receipt_sha256"]
                or provenance["selected_method"]
                != request["production_path"]["selected_method"]
                or provenance["bootstrap_minimum_success_fraction"] != 0.99):
            raise RuntimeError("batch returned malformed or stale production provenance")
        for field in ("issue52_receipt_sha256", "issue54_receipt_sha256",
                      "validation_receipt_sha256", "estimator_input_sha256"):
            value = provenance[field]
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError("batch returned malformed production provenance identity")


def _response_semantic_bytes(record: dict) -> bytes:
    semantic = {key: value for key, value in record.items() if key != "resource"}
    return canonical_json_bytes(semantic) + b"\n"


def _require_binary_identity(binary: Path, identity: dict) -> None:
    digest, byte_count = sha256_file(binary, 1024 * 1024 * 1024)
    if digest != identity["binary_sha256"] or (
            "binary_bytes" in identity and byte_count != identity["binary_bytes"]):
        raise RuntimeError("temporal covariance batch binary identity is stale")


def _replay_response_semantic_sha256(
        preregistration: dict, cell: dict, execution_path: str, seed_count: int,
        binary: Path, identity: dict) -> str:
    _require_binary_identity(binary, identity)
    process = subprocess.Popen(
        [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("temporal covariance batch pipes are unavailable")
    semantic_digest = hashlib.sha256()
    try:
        for seed_index in range(seed_count):
            request = request_for(
                cell, seed_index, preregistration, execution_path,
                retain_dense_evidence=False,
            )
            encoded = canonical_json_bytes(request) + b"\n"
            if len(encoded) > MAX_REQUEST_LINE_BYTES:
                raise RuntimeError("temporal covariance request exceeds its line cap")
            process.stdin.write(encoded)
            process.stdin.flush()
            line = _read_bounded_line(process.stdout, MAX_RESPONSE_LINE_BYTES)
            if not line:
                raise RuntimeError("temporal covariance batch ended before semantic replay")
            record = json.loads(line)
            _validate_compact_record(record, request, identity["batch_schema"])
            semantic_digest.update(_response_semantic_bytes(record))
        process.stdin.close()
        if _read_bounded_line(process.stdout, MAX_RESPONSE_LINE_BYTES):
            raise RuntimeError("temporal covariance batch returned a top-up semantic record")
        if process.wait() != 0:
            raise RuntimeError("temporal covariance batch semantic replay exited unsuccessfully")
    except BaseException:
        process.kill()
        process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        raise
    process.stdout.close()
    return semantic_digest.hexdigest()


def _cleanup_uncommitted(paths: dict[str, Path]) -> None:
    if paths["commit"].exists() or paths["commit"].is_symlink():
        return
    for path in paths.values():
        _remove_owned_regular(path)
        _remove_owned_regular(path.with_name(path.name + ".partial"))


def _validate_manifest(
        manifest: dict, identity: dict, cell: dict, execution_path: str,
        seed_count: int, records_sha256: str, records_bytes: int,
        response_semantic_sha256: str) -> None:
    expected_keys = {
        "schema", "cell_id", "cell_index", "execution_path", "seed_count",
        "records_sha256", "records_bytes", "request_schedule_sha256",
        "response_semantic_sha256",
        "producer_identity", "attempted", "emitted", "failed",
        "producer_source_set_sha256", "producer_binary_sha256",
        "total_wall_micros", "peak_resident_set_bytes",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RuntimeError("shard manifest schema is malformed")
    expected = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "cell_id": cell["cell_id"],
        "cell_index": cell["cell_index"],
        "execution_path": execution_path,
        "seed_count": seed_count,
        "records_sha256": records_sha256,
        "records_bytes": records_bytes,
        "response_semantic_sha256": response_semantic_sha256,
        "producer_identity": identity,
        "producer_source_set_sha256": identity["source_set_sha256"],
        "producer_binary_sha256": identity["binary_sha256"],
        "attempted": seed_count,
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise RuntimeError("shard manifest identity is stale or malformed")
    for field in ("emitted", "failed", "total_wall_micros", "peak_resident_set_bytes"):
        if type(manifest.get(field)) is not int or manifest[field] < 0:
            raise RuntimeError("shard manifest counters are malformed")
    if manifest["emitted"] + manifest["failed"] != seed_count:
        raise RuntimeError("shard manifest dispositions do not equal the seed denominator")
    schedule = manifest.get("request_schedule_sha256")
    if not isinstance(schedule, str) or len(schedule) != 64:
        raise RuntimeError("shard manifest request schedule is malformed")


def _read_committed_shard(
        preregistration: dict, cell: dict, execution_path: str, seed_count: int,
        paths: dict[str, Path], identity: dict, scorer: StreamingScores | None,
        binary: Path | None = None) -> dict:
    try:
        commit_bytes = _read_bounded_regular(paths["commit"], MAX_COMMIT_BYTES)
        manifest_bytes = _read_bounded_regular(paths["manifest"], MAX_MANIFEST_BYTES)
    except RuntimeError as error:
        raise RuntimeError("committed shard is partial or missing") from error
    commit = json.loads(commit_bytes)
    if not isinstance(commit, dict) or set(commit) != {
        "schema", "manifest_sha256", "records_sha256",
        "response_semantic_sha256",
        "producer_source_set_sha256", "producer_binary_sha256",
    } or commit.get("schema") != SHARD_COMMIT_SCHEMA:
        raise RuntimeError("shard commit schema is malformed")
    if (
            commit["producer_source_set_sha256"] != identity["source_set_sha256"]
            or commit["producer_binary_sha256"] != identity["binary_sha256"]):
        raise RuntimeError("shard commit producer identity is stale or malformed")
    if hashlib.sha256(manifest_bytes).hexdigest() != commit.get("manifest_sha256"):
        raise RuntimeError("shard manifest hash is stale or tampered")
    manifest = json.loads(manifest_bytes)
    schedule = hashlib.sha256()
    count = 0
    try:
        records = _open_bounded_regular(paths["records"], MAX_SHARD_RECORD_BYTES)
    except RuntimeError as error:
        raise RuntimeError("committed shard is partial or missing") from error
    with records:
        records_digest = hashlib.sha256()
        records_bytes = 0
        while chunk := records.read(1024 * 1024):
            records_digest.update(chunk)
            records_bytes += len(chunk)
        records_sha256 = records_digest.hexdigest()
        if records_sha256 != commit.get("records_sha256"):
            raise RuntimeError("shard record hash is stale or tampered")
        records.seek(0)
        semantic_digest = hashlib.sha256()
        while line := _read_bounded_line(records, MAX_RESPONSE_LINE_BYTES):
            if not line.strip():
                raise RuntimeError("shard contains an empty record")
            request = request_for(
                cell, count, preregistration, execution_path,
                retain_dense_evidence=False,
            )
            encoded_request = canonical_json_bytes(request) + b"\n"
            schedule.update(encoded_request)
            record = json.loads(line)
            _validate_compact_record(record, request, identity["batch_schema"])
            semantic_digest.update(_response_semantic_bytes(record))
            if scorer is not None:
                scorer.update(record)
            count += 1
            if count > seed_count:
                raise RuntimeError("shard contains a top-up attempt")
    response_semantic_sha256 = semantic_digest.hexdigest()
    _validate_manifest(
        manifest, identity, cell, execution_path, seed_count,
        records_sha256, records_bytes, response_semantic_sha256,
    )
    if count != seed_count or schedule.hexdigest() != manifest["request_schedule_sha256"]:
        raise RuntimeError("shard seed schedule is missing, duplicated, or reordered")
    if commit["response_semantic_sha256"] != response_semantic_sha256:
        raise RuntimeError("shard response semantic receipt is stale or tampered")
    if binary is not None and _replay_response_semantic_sha256(
            preregistration, cell, execution_path, seed_count, binary, identity
    ) != response_semantic_sha256:
        raise RuntimeError("batch returned response semantics inconsistent with the commit")
    return manifest


def execute_or_resume_shard(
        preregistration: dict, cell: dict, execution_path: str, seed_count: int,
        shards: Path, binary: Path, identity: dict) -> tuple[dict, bool]:
    paths = _shard_paths(shards, cell, execution_path)
    if paths["commit"].exists() or paths["commit"].is_symlink():
        return (
            _read_committed_shard(
                preregistration, cell, execution_path, seed_count, paths, identity, None,
                binary,
            ),
            True,
        )
    _cleanup_uncommitted(paths)
    partial_records = paths["records"].with_name(paths["records"].name + ".partial")
    _require_binary_identity(binary, identity)
    process = subprocess.Popen(
        [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("temporal covariance batch pipes are unavailable")
    records_digest = hashlib.sha256()
    schedule_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    attempted = emitted = failed = total_wall = peak_rss = records_bytes = 0
    try:
        with partial_records.open("xb") as retained:
            for seed_index in range(seed_count):
                request = request_for(
                    cell, seed_index, preregistration, execution_path,
                    retain_dense_evidence=False,
                )
                encoded = canonical_json_bytes(request) + b"\n"
                if len(encoded) > MAX_REQUEST_LINE_BYTES:
                    raise RuntimeError("temporal covariance request exceeds its line cap")
                schedule_digest.update(encoded)
                process.stdin.write(encoded)
                process.stdin.flush()
                line = _read_bounded_line(process.stdout, MAX_RESPONSE_LINE_BYTES)
                if not line:
                    raise RuntimeError("temporal covariance batch ended before its shard")
                record = json.loads(line)
                _validate_compact_record(record, request, identity["batch_schema"])
                semantic_digest.update(_response_semantic_bytes(record))
                records_bytes += len(line)
                if records_bytes > MAX_SHARD_RECORD_BYTES:
                    raise RuntimeError("temporal covariance shard exceeds its byte cap")
                retained.write(line)
                records_digest.update(line)
                attempted += 1
                emitted += int(record["emitted"])
                failed += int(record["failed"])
                resource = record["resource"]
                total_wall += resource["wall_micros"]
                peak_rss = max(
                    peak_rss,
                    resource["resident_set_bytes_before"],
                    resource["resident_set_bytes_after"],
                )
            process.stdin.close()
            if _read_bounded_line(process.stdout, MAX_RESPONSE_LINE_BYTES):
                raise RuntimeError("temporal covariance batch returned a top-up record")
            if process.wait() != 0:
                raise RuntimeError("temporal covariance batch exited unsuccessfully")
            retained.flush()
            os.fsync(retained.fileno())
    except BaseException:
        process.kill()
        process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        raise
    process.stdout.close()
    manifest = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "cell_id": cell["cell_id"],
        "cell_index": cell["cell_index"],
        "execution_path": execution_path,
        "seed_count": seed_count,
        "records_sha256": records_digest.hexdigest(),
        "records_bytes": records_bytes,
        "request_schedule_sha256": schedule_digest.hexdigest(),
        "response_semantic_sha256": semantic_digest.hexdigest(),
        "producer_identity": identity,
        "producer_source_set_sha256": identity["source_set_sha256"],
        "producer_binary_sha256": identity["binary_sha256"],
        "attempted": attempted,
        "emitted": emitted,
        "failed": failed,
        "total_wall_micros": total_wall,
        "peak_resident_set_bytes": peak_rss,
    }
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise RuntimeError("shard manifest exceeds its byte cap")
    commit = {
        "schema": SHARD_COMMIT_SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "records_sha256": records_digest.hexdigest(),
        "response_semantic_sha256": semantic_digest.hexdigest(),
        "producer_source_set_sha256": identity["source_set_sha256"],
        "producer_binary_sha256": identity["binary_sha256"],
    }
    os.link(partial_records, paths["records"], follow_symlinks=False)
    partial_records.unlink()
    atomic_write_no_replace(paths["manifest"], manifest_bytes)
    atomic_write_no_replace(paths["commit"], canonical_json_bytes(commit) + b"\n")
    _fsync_directory(shards)
    _read_committed_shard(
        preregistration, cell, execution_path, seed_count, paths, identity, None
    )
    return manifest, False


def _result_receipt(
        preregistration: dict, seed_count: int, processed: int, emitted: int,
        failed: int, total_wall: int, peak_rss: int, scores: dict,
        records: list[dict], exact_denominator: bool, identity: dict) -> dict:
    expected_attempts = (
        len(cells(preregistration)) * seed_count * len(preregistration["execution_paths"])
    )
    result_payload = canonical_json_bytes({"scores": scores})
    projected_full_minutes = (
        (total_wall / processed) * expected_attempts / 60_000_000
        if processed else float("inf")
    )
    resource_gates = {
        "rss": peak_rss <= preregistration["resource_limits"]["rss_limit_bytes"],
        "artifact_size": len(result_payload)
            <= preregistration["resource_limits"]["artifact_size_limit_bytes"],
        "projected_wall": projected_full_minutes
            <= preregistration["resource_limits"]["projected_full_scene_minutes"],
    }
    complete_execution = processed == expected_attempts
    return {
        "schema": preregistration["schemas"]["generator"],
        "preregistration_schema": preregistration["schema"],
        "pre_outcome_status": preregistration["status"],
        "supported_cell_sha256": preregistration["supported_cell_sha256"],
        "attempted_cells": expected_attempts,
        "batch_attempted_cells": processed,
        "emitted_cells": emitted,
        "failed_cells": failed,
        "skipped_contract_cells": expected_attempts - processed,
        "seed_count": seed_count,
        "unsupported_cell_count": len(unsupported_cells(preregistration)),
        "unsupported_cell_sha256": preregistration["unsupported_cell_sha256"],
        "unsupported_cells": unsupported_cells(preregistration),
        "methods": preregistration["methods"],
        "records": records,
        "scores": scores,
        "execution_paths": preregistration["execution_paths"],
        "producer_identity": identity,
        "corrected_inferential_sigma_emission": False,
        "execution_complete": complete_execution,
        "exact_seed_denominator_complete": exact_denominator,
        "result_records_sha256": hashlib.sha256(result_payload).hexdigest(),
        "result_records_bytes": len(result_payload),
        "promotion_eligible": exact_denominator and scores["all_methods_pass"]
            and all(resource_gates.values()),
        "promotion_status": (
            "eligible_for_external_field_review"
            if exact_denominator and scores["all_methods_pass"]
            and all(resource_gates.values())
            else "blocked_pending_complete_passing_synthetic_execution"
        ),
        "resource": {
            "total_wall_micros": total_wall,
            "peak_resident_set_bytes": peak_rss,
            "result_artifact_bytes": len(result_payload),
            "projected_full_minutes": projected_full_minutes,
        },
        "resource_gates": resource_gates,
        "resource_limits": preregistration["resource_limits"],
    }


def _run_probe(
        preregistration: dict, seed_count: int, limit: int, binary: Path,
        identity: dict) -> dict:
    if limit > MAX_PROBE_RECORDS:
        raise RuntimeError("probe record count exceeds its retained bound")
    selected = itertools.islice(iter_requests(preregistration, seed_count), limit)
    process = subprocess.Popen(
        [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("temporal covariance batch pipes are unavailable")
    scorer = StreamingScores(preregistration)
    records = []
    emitted = failed = total_wall = peak_rss = 0
    try:
        for request in selected:
            encoded = canonical_json_bytes(request) + b"\n"
            if len(encoded) > MAX_REQUEST_LINE_BYTES:
                raise RuntimeError("temporal covariance request exceeds its line cap")
            process.stdin.write(encoded)
            process.stdin.flush()
            line = _read_bounded_line(process.stdout, MAX_RESPONSE_LINE_BYTES)
            if not line:
                raise RuntimeError("temporal covariance batch ended before its probe")
            record = json.loads(line)
            _validate_compact_record(
                record, request, preregistration["schemas"]["batch"]
            )
            scorer.update(record)
            records.append(record)
            emitted += int(record["emitted"])
            failed += int(record["failed"])
            total_wall += record["resource"]["wall_micros"]
            peak_rss = max(
                peak_rss,
                record["resource"]["resident_set_bytes_before"],
                record["resource"]["resident_set_bytes_after"],
            )
        process.stdin.close()
        if _read_bounded_line(process.stdout, MAX_RESPONSE_LINE_BYTES):
            raise RuntimeError("temporal covariance batch returned a top-up record")
        if process.wait() != 0:
            raise RuntimeError("temporal covariance batch exited unsuccessfully")
    except BaseException:
        process.kill()
        process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        raise
    process.stdout.close()
    scores = scorer.finalize(require_complete=False)
    return _result_receipt(
        preregistration, seed_count, len(records), emitted, failed,
        total_wall, peak_rss, scores, records, False, identity,
    )


def run(
        preregistration: dict, seed_count: int, limit: int | None,
        run_root: Path | None = None, binary: Path | None = None) -> dict:
    frozen_cells = cells(preregistration)
    if cell_hash(frozen_cells) != preregistration["supported_cell_sha256"]:
        raise RuntimeError("supported cell construction does not match preregistration hash")
    expected_limits = {
        "request_line_limit_bytes": MAX_REQUEST_LINE_BYTES,
        "response_line_limit_bytes": MAX_RESPONSE_LINE_BYTES,
        "shard_record_limit_bytes": MAX_SHARD_RECORD_BYTES,
        "manifest_limit_bytes": MAX_MANIFEST_BYTES,
        "commit_limit_bytes": MAX_COMMIT_BYTES,
        "final_receipt_limit_bytes": MAX_FINAL_RECEIPT_BYTES,
    }
    if any(
            preregistration["resource_limits"].get(field) != value
            for field, value in expected_limits.items()):
        raise RuntimeError("preregistered resource limits do not match the driver")
    shard_count = len(frozen_cells) * len(preregistration["execution_paths"])
    if preregistration["execution_protocol"]["shard_count"] != shard_count:
        raise RuntimeError("preregistered shard count does not match the frozen scope")
    expected_retained_bound = (
        shard_count * (
            MAX_SHARD_RECORD_BYTES + MAX_MANIFEST_BYTES + MAX_COMMIT_BYTES
        )
        + MAX_COMMIT_BYTES
        + MAX_FINAL_RECEIPT_BYTES
    )
    resource_limits = preregistration["resource_limits"]
    if (
            resource_limits.get("retained_bound_bytes") != expected_retained_bound
            or expected_retained_bound > resource_limits["artifact_size_limit_bytes"]):
        raise RuntimeError("preregistered retained-resource bound is stale")
    root = Path(__file__).parents[1]
    if binary is None:
        subprocess.run([
            "cargo", "build", "--release", "-p", "dolphin-timeseries",
            "--example", "temporal_covariance_batch",
        ], cwd=root, check=True)
        binary = root / "target/release/examples/temporal_covariance_batch"
    identity = producer_identity(preregistration, binary)
    if limit is not None:
        return _run_probe(preregistration, seed_count, limit, binary, identity)
    if seed_count != preregistration["outer_seeds_per_supported_cell"]:
        raise RuntimeError("resumable execution requires the exact frozen seed denominator")
    if run_root is None:
        raise RuntimeError("resumable execution requires a run root")
    shards = initialize_run_root(run_root, identity)
    for cell in frozen_cells:
        for execution_path in preregistration["execution_paths"]:
            execute_or_resume_shard(
                preregistration, cell, execution_path, seed_count,
                shards, binary, identity,
            )
    expected_names = set()
    for cell in frozen_cells:
        for execution_path in preregistration["execution_paths"]:
            expected_names.update(
                path.name
                for path in _shard_paths(shards, cell, execution_path).values()
            )
    if {path.name for path in shards.iterdir()} != expected_names:
        raise RuntimeError("run root contains partial, missing, or out-of-scope shard artifacts")
    scorer = StreamingScores(preregistration)
    processed = emitted = failed = total_wall = peak_rss = 0
    for cell in frozen_cells:
        for execution_path in preregistration["execution_paths"]:
            paths = _shard_paths(shards, cell, execution_path)
            manifest = _read_committed_shard(
                preregistration, cell, execution_path, seed_count,
                paths, identity, scorer,
            )
            processed += manifest["attempted"]
            emitted += manifest["emitted"]
            failed += manifest["failed"]
            total_wall += manifest["total_wall_micros"]
            peak_rss = max(peak_rss, manifest["peak_resident_set_bytes"])
    scores = scorer.finalize(require_complete=True)
    receipt = _result_receipt(
        preregistration, seed_count, processed, emitted, failed,
        total_wall, peak_rss, scores, [], True, identity,
    )
    retained_bytes = sum(
        path.stat().st_size
        for path in shards.iterdir()
        if _regular_file(path)
    )
    retained_bytes += (run_root / "run_identity.json").stat().st_size
    receipt["retained_shard_bytes"] = retained_bytes
    while True:
        final_bytes = len(canonical_json_bytes(receipt)) + 1
        retained_total = retained_bytes + final_bytes
        if receipt["resource"]["result_artifact_bytes"] == retained_total:
            break
        receipt["resource"]["result_artifact_bytes"] = retained_total
    if final_bytes > MAX_FINAL_RECEIPT_BYTES:
        raise RuntimeError("final temporal covariance receipt exceeds its byte cap")
    retained_total = receipt["resource"]["result_artifact_bytes"]
    artifact_limit = resource_limits["artifact_size_limit_bytes"]
    receipt["resource_gates"]["artifact_size"] = retained_total <= artifact_limit
    receipt["resource_gates"]["retained_bound"] = (
        retained_total <= resource_limits["retained_bound_bytes"]
    )
    receipt["promotion_eligible"] = (
        receipt["exact_seed_denominator_complete"]
        and receipt["scores"]["all_methods_pass"]
        and all(receipt["resource_gates"].values())
    )
    receipt["promotion_status"] = (
        "eligible_for_external_field_review"
        if receipt["promotion_eligible"]
        else "blocked_pending_complete_passing_synthetic_execution"
    )
    if not receipt["resource_gates"]["artifact_size"]:
        raise RuntimeError("retained temporal covariance shards exceed their artifact cap")
    if not receipt["resource_gates"]["retained_bound"]:
        raise RuntimeError("retained temporal covariance shards exceed their frozen bound")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.seeds <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--seeds and --limit must be positive")
    preregistration = json.loads(args.prereg.read_text())
    receipt = run(preregistration, args.seeds, args.limit, args.run_root)
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if len(encoded) > min(
            MAX_FINAL_RECEIPT_BYTES,
            preregistration["resource_limits"]["artifact_size_limit_bytes"]):
        raise RuntimeError("final temporal covariance receipt exceeds its artifact cap")
    atomic_write_no_replace(args.output, encoded)


if __name__ == "__main__":
    main()
