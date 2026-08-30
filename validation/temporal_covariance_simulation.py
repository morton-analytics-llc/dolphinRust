#!/usr/bin/env python3
"""Generate frozen #53 synthetic-engine cells through the Rust batch target."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import stat
import struct
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
OUTER_SEED_DOMAIN = b"dolphinrust:temporal-covariance:outer-seed:v2\0"
PROPER_COMPLEX_STREAM_DOMAIN = 0xD1B54A32D192ED03
MISSINGNESS_STREAM_DOMAIN = 0xA0761D6478BD642F
LATENT_AR_STREAM_DOMAIN = 0xE7037ED1A0B428DB
MEASUREMENT_NORMAL_STREAM_DOMAIN = 0x0C54A53D9E3779B9


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


def seed_identity(preregistration: dict, cell_index: int, outer_seed_index: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    digest.update(OUTER_SEED_DOMAIN)
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
OUTER_COVERAGE_DGP = "actual_c54_gaussian_measurement_post_link_ar_v1"
CONDITIONAL_COVARIANCE_ORACLE = "fixed_capture_common_factor_monte_carlo_v1"
FROZEN_SOURCE_SET_SCHEMA = "dolphinrust.canonical-producer-source-set/2"
FROZEN_SOURCE_SET_SHA256 = "f5d71d5ba98e742bef7a3652ac7a6217e52a1a0e1118edf9b7c1b30573e9fc66"
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
MAX_PROBE_REQUESTS = 16
MAX_FRAME_REQUESTS = 32
MAX_RAYON_WORKERS = 12
COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS = 200
COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES = 198
SELECTED_METHOD = "reml_covariance_parameter_adjusted_scalar"
SELECTED_METHOD_VERSION = 2
FROZEN_PROMOTION_METHODS = (
    "oracle_gls",
    "plugin_gls_reml",
    "reml_covariance_parameter_adjusted_scalar",
    "slope_profile_likelihood_ml",
    "complete_refit_bootstrap",
)
RUN_IDENTITY_SCHEMA = "dolphinrust-temporal-covariance-run-identity/2"
SHARD_MANIFEST_SCHEMA = "dolphinrust-temporal-covariance-shard-manifest/2"
SHARD_COMMIT_SCHEMA = "dolphinrust-temporal-covariance-shard-commit/2"
RUN_MANIFEST_SCHEMA = "dolphinrust-temporal-covariance-run-manifest/1"
RUN_COMMIT_SCHEMA = "dolphinrust-temporal-covariance-run-commit/1"
FRAME_RESOURCE_SCHEMA = "dolphinrust-temporal-covariance-batch-frame-resource/1"
TEMPORAL_RESOURCE_SCHEMA = "dolphinrust-temporal-inference-resource/2"
TEMPORAL_METHOD_SELECTION_SCHEMA = (
    "dolphinrust-temporal-covariance-method-selection/1"
)
TEMPORAL_RESOURCE_BENCHMARK_METHOD = "factor_native_direct_issue54_full_tile/2"
TEMPORAL_RESOURCE_RECEIPT_FILENAME = "temporal_inference_resource_receipt.json"
TEMPORAL_CANDIDATE_RESOURCE_RECEIPT_FILENAME = (
    "temporal_inference_candidate_resource_receipt.json"
)
TEMPORAL_METHOD_SELECTION_FILENAME = "temporal_covariance_method_selection.json"
TEMPORAL_BATCH_BINARY_FILENAME = "temporal_covariance_batch"
TEMPORAL_INFERENCE_BENCH_BINARY_FILENAME = "temporal_inference_bench"
TEMPORAL_RESOURCE_RECEIPT_CAP_BYTES = 1024 * 1024
TEMPORAL_RESOURCE_BINARY_CAP_BYTES = 256 * 1024 * 1024
TEMPORAL_RESOURCE_TILE_ROWS = 256
TEMPORAL_RESOURCE_TILE_COLUMNS = 256
TEMPORAL_RESOURCE_TARGET_COUNT = (
    TEMPORAL_RESOURCE_TILE_ROWS * TEMPORAL_RESOURCE_TILE_COLUMNS
)
TEMPORAL_RESOURCE_WORKER_SCRATCH_LIMIT_BYTES = 8 * 1024 * 1024
TEMPORAL_RESOURCE_RSS_LIMIT_BYTES = 24 * 1024 * 1024 * 1024
TEMPORAL_RESOURCE_MAXIMUM_TARGETS_PER_BLOCK = 65_536
TEMPORAL_RESOURCE_BLOCK_ID_READ_CAP_BYTES = 4 * 1024 * 1024
TEMPORAL_RESOURCE_FACTOR_BLOCK_READ_CAP_BYTES = 1024 * 1024 * 1024
TEMPORAL_RESOURCE_COMBINED_WORKING_SET_CAP_BYTES = 2 * 1024 * 1024 * 1024
TEMPORAL_RESOURCE_WALL_RATIO_LIMIT = 2.0
TEMPORAL_RESOURCE_DATE_COUNTS = (12, 48, 96)
TEMPORAL_RESOURCE_KEYS = frozenset({
    "schema", "status", "benchmark_method", "baseline_method",
    "candidate_method", "candidate_method_version", "tile_rows",
    "tile_columns", "target_count", "worker_scratch_limit_bytes",
    "resident_set_limit_bytes", "maximum_targets_per_block",
    "block_id_read_cap_bytes", "factor_block_read_cap_bytes",
    "combined_working_set_cap_bytes", "product_source_sha256",
    "benchmark_source_sha256", "batch_source_sha256",
    "pre_outcome_selection_receipt_sha256", "host",
    "temporal_covariance_batch_binary", "temporal_inference_bench_binary",
    "measurements",
})
TEMPORAL_RESOURCE_HOST_KEYS = frozenset({
    "operating_system", "architecture", "logical_processor_count",
    "rayon_thread_count", "omp_thread_count", "openblas_thread_count",
    "mkl_thread_count", "veclib_thread_count",
})
TEMPORAL_BINARY_IDENTITY_KEYS = frozenset({"sha256", "bytes"})
TEMPORAL_RESOURCE_MEASUREMENT_KEYS = frozenset({
    "post_gauge_date_count", "acquisition_count", "target_count",
    "varied_target_fingerprint_count", "plugin_gls_reml",
    "reml_covariance_parameter_adjusted_scalar",
    "adjusted_to_plugin_wall_ratio",
    "adjusted_to_plugin_full_product_wall_ratio",
})
TEMPORAL_RESOURCE_SCALAR_KEYS = frozenset({
    "method", "factor_sha256", "direct_factor_receipt_sha256",
    "factor_block_reads", "nonreference_realized_rank", "processed_pixels",
    "evaluated_pixels", "profile_fit_count", "bootstrap_attempts",
    "optimizer_rho_lane_evaluations", "optimizer_q_objective_evaluations",
    "optimizer_primary_rho_pass_histogram",
    "covariance_parameter_derivative_lane_evaluations",
    "covariance_parameter_adjustment_count", "rayon_worker_count",
    "maximum_worker_scratch_bytes", "exact_optimizer_fallback_targets",
    "condition_exact_fallbacks", "wall_micros", "wall_micros_trials",
    "full_product_wall_micros", "full_product_wall_micros_trials",
    "peak_resident_set_bytes", "checksum",
})
TEMPORAL_METHOD_SELECTION_KEYS = frozenset({
    "schema", "status", "selected_method", "selected_method_version",
    "candidate_resource_receipt_sha256",
    "canonical_v4_preregistration_sha256", "product_source_sha256",
    "benchmark_source_sha256", "batch_source_sha256",
    "temporal_covariance_batch_binary_sha256",
    "temporal_inference_bench_binary_sha256", "tile_rows", "tile_columns",
    "target_count", "post_gauge_date_counts",
    "adjusted_to_plugin_wall_ratio_limit", "worker_scratch_limit_bytes",
    "resident_set_limit_bytes", "outcomes_present",
})


def _proper_complex_draw(
        cell_index: int, outer_seed_index: int, date_index: int, column: int,
        role: int) -> tuple[float, float]:
    key = splitmix64(PROPER_COMPLEX_STREAM_DOMAIN)
    key ^= splitmix64((cell_index + 1) ^ 0xA0761D6478BD642F)
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


def standard_normal_path(count: int, state: int) -> list[float]:
    values = []
    for _ in range(count):
        state, value = normal_noise(state)
        values.append(value)
    return values


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
    digest = hashlib.sha256()
    digest.update(b"dolphinrust:temporal-capture-scope:v2")

    def update_string(value: str) -> None:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    update_string(request["cell_id"])
    digest.update(request["cell_index"].to_bytes(8, "little"))
    digest.update(len(request["days"]).to_bytes(8, "little"))
    for value in request["days"]:
        digest.update(struct.pack("<d", value))
    for value in production["native_shape"]:
        digest.update(value.to_bytes(8, "little"))
    reference = production["reference"]
    update_string(reference["geometry_id"])
    update_string(reference["window_id"])
    digest.update(struct.pack("<d", reference["overlap_fraction"]))
    digest.update(struct.pack("<d", reference["distance_pixels"]))
    digest.update(reference["sequential_depth"].to_bytes(8, "little"))
    update_string(reference["approximation"])
    for value in production["reference_pixel"]:
        digest.update(value.to_bytes(8, "little"))
    update_string(production["scope"])
    digest.update(production["source_seed"].to_bytes(8, "little"))
    for value in production["target"]:
        digest.update(value.to_bytes(8, "little"))
    update_string(production["source_correlation_model"])
    digest.update(struct.pack(
        "<d", production["source_correlation_distance_scale_pixels"]
    ))
    update_string(production["outer_coverage_dgp"])
    update_string(production["conditional_covariance_oracle"])
    return digest.hexdigest()


def request_for(
        cell: dict, outer_seed_index: int, preregistration: dict,
        retain_dense_evidence: bool = False) -> dict:
    seed, seed_sha256 = seed_identity(preregistration, cell["cell_index"], outer_seed_index)
    days = days_for(cell)
    missing = missing_indices(
        cell, splitmix64(seed ^ MISSINGNESS_STREAM_DOMAIN), len(days) - 1
    )
    validity = []
    carrier_values = []
    diagonal = []
    for index in range(len(days)):
        if cell["variance_arrangement"] == "alternating":
            scale = 1.0 if index % 2 == 0 else cell["variance_ratio"]
        else:
            scale = 1.0 if index < len(days) // 2 else cell["variance_ratio"]
        diagonal.append(0.01 * (scale + cell["reference_contribution_ratio"]))
    process_variance = 0.04
    _, ar_path = stationary_ar_path(
        days, cell["rho_at_12_days"], splitmix64(seed ^ LATENT_AR_STREAM_DOMAIN)
    )
    latent_ar_path = [0.0] + ar_path[1:]
    measurement_normal_path = standard_normal_path(
        len(days), splitmix64(seed ^ MEASUREMENT_NORMAL_STREAM_DOMAIN)
    )
    truth_slope_per_day = 0.01
    for index, day in enumerate(days):
        carrier = truth_slope_per_day * day
        carrier = 0.0 if index == 0 else carrier
        carrier_values.append(carrier)
        validity.append(index == 0 or index not in missing)
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
    request = {
        "cell_id": cell["cell_id"],
        "cell_index": cell["cell_index"],
        "outer_seed_index": outer_seed_index,
        "seed_sha256": seed_sha256,
        "seed": seed,
        "days": days,
        "options": options,
        "production_path": {
            "source_seed": seed,
            "native_shape": [1, 7],
            "target": [0, 1],
            "reference_pixel": [0, reference_column],
            "raw_complex_stack": raw_complex_stack,
            "carrier_stack": carrier_stack,
            "intended_difference_variance": [0.0] + diagonal[1:],
            "latent_ar_path": latent_ar_path,
            "measurement_normal_path": measurement_normal_path,
            "truth_slope_per_day": truth_slope_per_day,
            "source_correlation_model": SOURCE_CORRELATION_MODEL,
            "source_correlation_distance_scale_pixels": (
                SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
            ),
            "outer_coverage_dgp": OUTER_COVERAGE_DGP,
            "conditional_covariance_oracle": CONDITIONAL_COVARIANCE_ORACLE,
            "validity": validity,
            "reference": reference,
            "scope": "synthetic_validation",
            "capture_scope_sha256": "",
            "validation_receipt_sha256": "53" * 32,
            "selected_method": SELECTED_METHOD,
        },
        "retain_dense_evidence": retain_dense_evidence,
        "conditional_oracle_replicates": 0,
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
    "WeakParameterIdentification", "LegacyNonComparable", "DiagnosticNotComputed",
}
TEMPORAL_ACTIVE_SET_STATUSES = {
    "RhoLowerBoundary", "RhoUpperBoundary",
    "ProcessVarianceLowerBoundary", "ProcessVarianceUpperBoundary",
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
    "temporal_dgp_invalid",
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
        if (
                not comparator
                or comparator.get("status") != "Evaluated"
                or comparator.get("point_estimate") is None):
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
            conditional_coverage = self.covered[label] / count if count else None
            unconditional_coverage = (
                self.covered[label] / self.attempted if self.attempted else None
            )
            aggregate[f"interval_emitted_{label}"] = count
            aggregate[f"conditional_coverage_{label}"] = conditional_coverage
            aggregate[f"unconditional_coverage_{label}"] = unconditional_coverage
            aggregate[f"coverage_{label}"] = conditional_coverage
            aggregate[f"mean_width_{label}"] = self.width_sums[label] / count if count else None
            aggregate[f"mean_interval_score_{label}"] = self.score_sums[label] / count if count else None
            tolerance = preregistration["thresholds"]["coverage"][f"0.{label}"]
            gates[f"conditional_coverage_{label}"] = (
                conditional_coverage is not None
                and abs(conditional_coverage - nominal) <= tolerance
            )
            gates[f"unconditional_coverage_{label}"] = (
                unconditional_coverage is not None
                and abs(unconditional_coverage - nominal) <= tolerance
            )
        return {"aggregate": aggregate, "gates": gates}


def _selected_method(preregistration: dict) -> str | None:
    selected_method = preregistration.get("selected_method")
    if selected_method is None:
        promotion_methods = preregistration.get("promotion_methods", [])
        selected_method = promotion_methods[-1] if promotion_methods else None
    return selected_method


def _interval_score(comparator: dict | None, label: str, truth: float) -> tuple[float, float] | None:
    if not comparator or comparator.get("status") != "Evaluated":
        return None
    interval = comparator.get(f"interval_{label}")
    if not isinstance(interval, dict):
        return None
    lower, upper = interval.get("lower"), interval.get("upper")
    if (
            not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or upper < lower):
        return None
    level = {"68": 0.68, "90": 0.90, "95": 0.95}[label]
    alpha = 1.0 - level
    width = upper - lower
    score = width
    score += (2.0 / alpha) * max(lower - truth, 0.0)
    score += (2.0 / alpha) * max(truth - upper, 0.0)
    return score, width


class PairedComparisonReducer:
    """Compare selected and baseline intervals on identical emitted seeds."""

    METHODS = ("selected", "oracle_gls", "ols", "plugin_gls_reml")

    def __init__(self, selected_method: str) -> None:
        self.selected_method = selected_method
        self.paired_counts = {label: 0 for label in ("68", "90", "95")}
        self.score_sums = {
            label: {method: 0.0 for method in self.METHODS}
            for label in ("68", "90", "95")
        }
        self.width_sums = {
            label: {method: 0.0 for method in self.METHODS}
            for label in ("68", "90", "95")
        }

    def update(self, fit: dict | None, truth: float) -> None:
        fit = fit if isinstance(fit, dict) else {}
        comparators = {
            "selected": fit.get(METHOD_FIELDS[self.selected_method]),
            "oracle_gls": fit.get(METHOD_FIELDS["oracle_gls"]),
            "ols": fit.get(METHOD_FIELDS["ols"]),
            "plugin_gls_reml": fit.get(METHOD_FIELDS["plugin_gls_reml"]),
        }
        for label in ("68", "90", "95"):
            values = {
                method: _interval_score(comparator, label, truth)
                for method, comparator in comparators.items()
            }
            if any(value is None for value in values.values()):
                continue
            self.paired_counts[label] += 1
            for method, (score, width) in values.items():
                self.score_sums[label][method] += score
                self.width_sums[label][method] += width

    def finalize(self, methods: dict) -> dict:
        result = {}
        method_names = {
            "selected": self.selected_method,
            "oracle_gls": "oracle_gls",
            "ols": "ols",
            "plugin_gls_reml": "plugin_gls_reml",
        }
        for label in ("68", "90", "95"):
            count = self.paired_counts[label]
            emitted_counts = {
                method: methods[name]["aggregate"][f"interval_emitted_{label}"]
                for method, name in method_names.items()
            }
            result[label] = {
                "paired_count": count,
                "emitted_counts": emitted_counts,
                "same_emission_set": count > 0
                    and all(value == count for value in emitted_counts.values()),
                "mean_scores": {
                    method: self.score_sums[label][method] / count if count else None
                    for method in self.METHODS
                },
                "mean_widths": {
                    method: self.width_sums[label][method] / count if count else None
                    for method in self.METHODS
                },
            }
        return result


class StreamingScores:
    def __init__(self, preregistration: dict) -> None:
        self.preregistration = preregistration
        self.truth = 0.01 * 365.25
        self.selected_method = _selected_method(preregistration)
        if self.selected_method not in METHOD_FIELDS:
            raise RuntimeError("preregistration does not name one supported selected method")
        self.promotion_methods = preregistration.get("promotion_methods", [])
        if self.promotion_methods != list(FROZEN_PROMOTION_METHODS):
            raise RuntimeError("preregistration promotion methods are incomplete or unsupported")
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
        self.paired_comparisons = {
            (cell["cell_id"], path): PairedComparisonReducer(self.selected_method)
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
        self.paired_comparisons[key].update(fit, self.truth)

    def finalize(self, require_complete: bool) -> dict:
        expected = self.preregistration["outer_seeds_per_supported_cell"]
        if require_complete and any(count != expected for count in self.next_seed.values()):
            raise RuntimeError("batch did not return the exact frozen seed denominator for every cell")
        selected_method = self.selected_method
        summaries = []
        global_methods = {
            method: {"attempted": 0, "scored": 0, "failed": 0}
            for method in METHOD_FIELDS
        }
        all_selected_pass = True
        all_oracle_pass = True
        all_promotion_methods_pass = True
        all_comparisons_complete = True
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
                        score is not None
                        and oracle_score is not None
                        and score <= oracle_score * (
                            1.0 + self.preregistration["thresholds"]["proper_score"]
                        )
                    )
                    result["gates"]["interval_width"] = (
                        width is not None
                        and oracle_width is not None
                        and width <= oracle_width * self.preregistration[
                            "thresholds"
                        ]["maximum_interval_width_ratio"]
                    )
                    result["passes_all_gates"] = all(result["gates"].values())
                    for field in ("attempted", "scored", "failed"):
                        global_methods[method][field] += aggregate[field]
                selected = methods[selected_method]
                paired = self.paired_comparisons[(cell["cell_id"], path)].finalize(methods)
                comparison_complete = True
                for label in ("68", "90", "95"):
                    comparison = paired[label]
                    selected_score = comparison["mean_scores"]["selected"]
                    selected_width = comparison["mean_widths"]["selected"]
                    oracle_score = comparison["mean_scores"]["oracle_gls"]
                    oracle_width = comparison["mean_widths"]["oracle_gls"]
                    ols_score = comparison["mean_scores"]["ols"]
                    plugin_score = comparison["mean_scores"]["plugin_gls_reml"]
                    values = (
                        selected_score, selected_width, oracle_score, oracle_width,
                        ols_score, plugin_score,
                    )
                    complete = comparison["same_emission_set"] and all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in values
                    )
                    comparison_complete = comparison_complete and complete
                    selected["gates"][f"proper_score_vs_oracle_{label}"] = (
                        complete
                        and selected_score
                        <= oracle_score
                        * (1.0 + self.preregistration["thresholds"]["proper_score"])
                    )
                    selected["gates"][f"proper_score_vs_ols_{label}"] = (
                        complete and selected_score < ols_score
                    )
                    selected["gates"][f"proper_score_vs_plugin_{label}"] = (
                        complete and selected_score < plugin_score
                    )
                    selected["gates"][f"interval_width_{label}"] = (
                        complete
                        and selected_width
                        <= oracle_width
                        * self.preregistration["thresholds"]["maximum_interval_width_ratio"]
                    )
                selected["passes_all_gates"] = all(selected["gates"].values())
                oracle_pass = methods["oracle_gls"]["passes_all_gates"]
                selected_pass = selected["passes_all_gates"] and comparison_complete
                promotion_methods_pass = all(
                    methods[method]["passes_all_gates"]
                    for method in self.promotion_methods
                )
                all_selected_pass = all_selected_pass and selected_pass
                all_oracle_pass = all_oracle_pass and oracle_pass
                all_promotion_methods_pass = (
                    all_promotion_methods_pass and promotion_methods_pass
                )
                all_comparisons_complete = (
                    all_comparisons_complete and comparison_complete
                )
                summaries.append({
                    "cell_id": cell["cell_id"],
                    "cell_index": cell["cell_index"],
                    "execution_path": path,
                    "attempted": self.next_seed[(cell["cell_id"], path)],
                    "selected_method": selected_method,
                    "selected_method_pass": selected_pass,
                    "oracle_reference_pass": oracle_pass,
                    "promotion_methods_pass": promotion_methods_pass,
                    "comparison_complete": comparison_complete,
                    "paired_comparisons": paired,
                    "methods": methods,
                })
        validation_pass = (
            all_promotion_methods_pass
            and all_selected_pass
            and all_oracle_pass
            and all_comparisons_complete
        )
        return {
            "schema": self.preregistration["schemas"]["scorer"],
            "truth_slope_per_year": self.truth,
            "methods": global_methods,
            "cell_summaries": summaries,
            "selected_method": selected_method,
            "promotion_methods": self.promotion_methods,
            "selected_method_pass": all_selected_pass,
            "oracle_reference_pass": all_oracle_pass,
            "comparison_complete": all_comparisons_complete,
            "all_methods_pass": validation_pass,
        }


def score_records(records: list[dict], preregistration: dict) -> dict:
    scorer = StreamingScores(preregistration)
    for record in records:
        scorer.update(record)
    return scorer.finalize(require_complete=False)


def iter_requests(preregistration: dict, seed_count: int):
    for cell in cells(preregistration):
        for outer_seed_index in range(seed_count):
            yield request_for(cell, outer_seed_index, preregistration)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def record_wire_bytes(record: dict) -> bytes:
    return json.dumps(record, allow_nan=False, separators=(",", ":")).encode()


def compact_record_sha256(record: dict) -> str:
    payload = [
        record["schema"],
        record["execution_path"],
        record["cell_id"],
        record["cell_index"],
        record["outer_seed_index"],
        record["seed_sha256"],
        record["seed"],
        record["factor_sha256"],
        record["realized_factor_rank"],
        record["fixed_factor_status"],
        record["production_path_status"],
        record["comparator_methods"],
        record["fit"],
        record["provenance"],
        record["production_receipts"],
    ]
    def semantic_value(value):
        if type(value) is float:
            return f"f64:{struct.pack('>d', value).hex()}"
        if isinstance(value, list):
            return [semantic_value(item) for item in value]
        if isinstance(value, dict):
            return {key: semantic_value(item) for key, item in value.items()}
        return value

    encoded = json.dumps(
        semantic_value(payload), allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def request_frame(requests: list[dict], batch_schema: str) -> dict:
    if not 1 <= len(requests) <= MAX_FRAME_REQUESTS:
        raise RuntimeError("temporal covariance frame request count is invalid")
    cell_identity = (requests[0]["cell_id"], requests[0]["cell_index"])
    first_seed = requests[0]["outer_seed_index"]
    for offset, request in enumerate(requests):
        if (
                (request["cell_id"], request["cell_index"]) != cell_identity
                or request["outer_seed_index"] != first_seed + offset):
            raise RuntimeError(
                "temporal covariance frame must contain same-cell consecutive seeds"
            )
    frame = {"schema": batch_schema, "requests": requests}
    if len(canonical_json_bytes(frame)) + 1 > MAX_REQUEST_LINE_BYTES:
        raise RuntimeError("temporal covariance request frame exceeds its line cap")
    return frame


def iter_request_frames(
        cell: dict, seed_count: int, preregistration: dict,
        frame_size: int = MAX_FRAME_REQUESTS):
    if not 1 <= frame_size <= MAX_FRAME_REQUESTS:
        raise RuntimeError("temporal covariance frame size is invalid")
    for start in range(0, seed_count, frame_size):
        requests = [
            request_for(cell, seed_index, preregistration)
            for seed_index in range(start, min(start + frame_size, seed_count))
        ]
        yield request_frame(requests, preregistration["schemas"]["batch"])


def batch_environment(preregistration: dict) -> dict[str, str]:
    configured = preregistration["execution_protocol"].get(
        "maximum_rayon_workers", MAX_RAYON_WORKERS
    )
    if configured != MAX_RAYON_WORKERS:
        raise RuntimeError("preregistered Rayon worker bound is stale")
    worker_count = max(1, min(configured, os.cpu_count() or 1, MAX_FRAME_REQUESTS))
    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = str(worker_count)
    return environment


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


def _canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _u64(value: object, *, positive: bool = False) -> bool:
    return (
        type(value) is int
        and (value > 0 if positive else value >= 0)
        and value <= (1 << 64) - 1
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _strict_json_object(raw: bytes, label: str) -> dict:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: object, keys: frozenset[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError(f"{label} has missing or unknown fields")
    return value


def _resource_evidence_bytes(path: Path, byte_cap: int, label: str) -> bytes:
    try:
        return _read_bounded_regular(path, byte_cap)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"{label} is missing or unreadable") from error


def canonical_v4_sha256() -> str:
    path = Path(__file__).with_name(
        "temporal_covariance_synthetic_engine_preregistration_v4.json"
    )
    raw = _resource_evidence_bytes(
        path, TEMPORAL_RESOURCE_RECEIPT_CAP_BYTES, "canonical v4 preregistration"
    )
    return hashlib.sha256(canonical_json_bytes(_strict_json_object(raw, path.name))).hexdigest()


def _current_resource_source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[1]
    return {
        "product_source_sha256": sha256_file(
            root / "crates/dolphin-workflows/src/temporal_covariance_product.rs",
            64 * 1024 * 1024,
        )[0],
        "benchmark_source_sha256": sha256_file(
            root / "crates/dolphin-workflows/examples/temporal_inference_bench.rs",
            16 * 1024 * 1024,
        )[0],
        "batch_source_sha256": sha256_file(
            root / "crates/dolphin-timeseries/examples/temporal_covariance_batch.rs",
            16 * 1024 * 1024,
        )[0],
    }


def _validate_binary_identity(value: object, label: str) -> dict:
    identity = _require_exact_keys(value, TEMPORAL_BINARY_IDENTITY_KEYS, label)
    if not _canonical_sha256(identity["sha256"]) or not _u64(
        identity["bytes"], positive=True
    ):
        raise RuntimeError(f"{label} is malformed")
    return identity


def _validate_scalar_resource_measurement(
        value: object, method: str, post_gauge_dates: int, host_workers: int,
        label: str) -> dict:
    scalar = _require_exact_keys(value, TEMPORAL_RESOURCE_SCALAR_KEYS, label)
    integer_fields = (
        "factor_block_reads", "nonreference_realized_rank", "processed_pixels",
        "evaluated_pixels", "profile_fit_count", "bootstrap_attempts",
        "optimizer_rho_lane_evaluations", "optimizer_q_objective_evaluations",
        "covariance_parameter_derivative_lane_evaluations",
        "covariance_parameter_adjustment_count", "rayon_worker_count",
        "maximum_worker_scratch_bytes", "exact_optimizer_fallback_targets",
        "condition_exact_fallbacks", "wall_micros", "full_product_wall_micros",
        "peak_resident_set_bytes",
    )
    if any(not _u64(scalar[field]) for field in integer_fields):
        raise RuntimeError(f"{label} has a malformed integer counter")
    for field in ("factor_sha256", "direct_factor_receipt_sha256"):
        if not _canonical_sha256(scalar[field]):
            raise RuntimeError(f"{label} has a malformed SHA-256")
    histogram = scalar["optimizer_primary_rho_pass_histogram"]
    wall_trials = scalar["wall_micros_trials"]
    full_trials = scalar["full_product_wall_micros_trials"]
    expected_evaluated = TEMPORAL_RESOURCE_TARGET_COUNT - 1
    if (
            scalar["method"] != method
            or scalar["factor_block_reads"] == 0
            or scalar["nonreference_realized_rank"] != post_gauge_dates
            or scalar["processed_pixels"] != TEMPORAL_RESOURCE_TARGET_COUNT
            or scalar["evaluated_pixels"] != expected_evaluated
            or scalar["profile_fit_count"] != expected_evaluated
            or scalar["bootstrap_attempts"] != 0
            or scalar["optimizer_rho_lane_evaluations"] == 0
            or scalar["optimizer_q_objective_evaluations"] == 0
            or not isinstance(histogram, list)
            or len(histogram) != 21
            or any(not _u64(count) for count in histogram)
            or sum(histogram) != expected_evaluated
            or scalar["rayon_worker_count"] != host_workers
            or scalar["maximum_worker_scratch_bytes"]
            > TEMPORAL_RESOURCE_WORKER_SCRATCH_LIMIT_BYTES
            or scalar["exact_optimizer_fallback_targets"] != 0
            or scalar["condition_exact_fallbacks"] != 0
            or scalar["wall_micros"] == 0
            or not isinstance(wall_trials, list)
            or len(wall_trials) != 2
            or any(not _u64(trial, positive=True) for trial in wall_trials)
            or scalar["wall_micros"] != max(wall_trials)
            or scalar["full_product_wall_micros"] < scalar["wall_micros"]
            or not isinstance(full_trials, list)
            or len(full_trials) != 2
            or any(not _u64(trial, positive=True) for trial in full_trials)
            or scalar["full_product_wall_micros"] != max(full_trials)
            or scalar["peak_resident_set_bytes"] == 0
            or scalar["peak_resident_set_bytes"] > TEMPORAL_RESOURCE_RSS_LIMIT_BYTES
            or not _finite_number(scalar["checksum"])):
        raise RuntimeError(f"{label} failed the frozen scalar resource contract")
    return scalar


def _validate_resource_receipt(
        receipt: dict, observed_batch: dict, observed_benchmark: dict,
        label: str) -> None:
    expected_keys = TEMPORAL_RESOURCE_KEYS
    if "pre_outcome_selection_receipt_sha256" not in receipt:
        expected_keys = expected_keys - {"pre_outcome_selection_receipt_sha256"}
    _require_exact_keys(receipt, expected_keys, label)
    selection_sha256 = receipt.get("pre_outcome_selection_receipt_sha256")
    if (
            receipt["schema"] != TEMPORAL_RESOURCE_SCHEMA
            or receipt["status"] not in ("candidate_evidence_only", "pass")
            or (receipt["status"] == "pass" and not _canonical_sha256(selection_sha256))
            or (receipt["status"] == "candidate_evidence_only" and selection_sha256 is not None)
            or receipt["benchmark_method"] != TEMPORAL_RESOURCE_BENCHMARK_METHOD
            or receipt["baseline_method"] != "plugin_gls_reml"
            or receipt["candidate_method"] != SELECTED_METHOD
            or type(receipt["candidate_method_version"]) is not int
            or receipt["candidate_method_version"] != SELECTED_METHOD_VERSION):
        raise RuntimeError(f"{label} has an unsupported method or schema")
    exact_scope = {
        "tile_rows": TEMPORAL_RESOURCE_TILE_ROWS,
        "tile_columns": TEMPORAL_RESOURCE_TILE_COLUMNS,
        "target_count": TEMPORAL_RESOURCE_TARGET_COUNT,
        "worker_scratch_limit_bytes": TEMPORAL_RESOURCE_WORKER_SCRATCH_LIMIT_BYTES,
        "resident_set_limit_bytes": TEMPORAL_RESOURCE_RSS_LIMIT_BYTES,
        "maximum_targets_per_block": TEMPORAL_RESOURCE_MAXIMUM_TARGETS_PER_BLOCK,
        "block_id_read_cap_bytes": TEMPORAL_RESOURCE_BLOCK_ID_READ_CAP_BYTES,
        "factor_block_read_cap_bytes": TEMPORAL_RESOURCE_FACTOR_BLOCK_READ_CAP_BYTES,
        "combined_working_set_cap_bytes": (
            TEMPORAL_RESOURCE_COMBINED_WORKING_SET_CAP_BYTES
        ),
        **_current_resource_source_hashes(),
    }
    numeric_scope_fields = {
        field for field in exact_scope if field not in (
            "product_source_sha256", "benchmark_source_sha256", "batch_source_sha256"
        )
    }
    if (
            any(receipt.get(field) != expected for field, expected in exact_scope.items())
            or any(type(receipt[field]) is not int for field in numeric_scope_fields)):
        raise RuntimeError(f"{label} differs from the frozen resource scope")
    host = _require_exact_keys(receipt["host"], TEMPORAL_RESOURCE_HOST_KEYS, f"{label} host")
    if (
            not isinstance(host["operating_system"], str)
            or not host["operating_system"]
            or not isinstance(host["architecture"], str)
            or not host["architecture"]
            or not _u64(host["logical_processor_count"], positive=True)
            or not _u64(host["rayon_thread_count"], positive=True)
            or any(type(host[field]) is not int or host[field] != 1 for field in (
                "omp_thread_count", "openblas_thread_count", "mkl_thread_count",
                "veclib_thread_count",
            ))):
        raise RuntimeError(f"{label} host differs from the frozen resource contract")
    batch = _validate_binary_identity(
        receipt["temporal_covariance_batch_binary"], f"{label} batch binary"
    )
    benchmark = _validate_binary_identity(
        receipt["temporal_inference_bench_binary"], f"{label} benchmark binary"
    )
    if batch != observed_batch or benchmark != observed_benchmark:
        raise RuntimeError(f"{label} binary identity is not observed")
    measurements = receipt["measurements"]
    if not isinstance(measurements, list) or len(measurements) != 3:
        raise RuntimeError(f"{label} must contain exactly three resource cases")
    for measurement, post_gauge_dates in zip(
            measurements, TEMPORAL_RESOURCE_DATE_COUNTS):
        case_label = f"{label} resource case {post_gauge_dates}"
        measurement = _require_exact_keys(
            measurement, TEMPORAL_RESOURCE_MEASUREMENT_KEYS, case_label
        )
        plugin = _validate_scalar_resource_measurement(
            measurement["plugin_gls_reml"], "plugin_gls_reml", post_gauge_dates,
            host["rayon_thread_count"], case_label,
        )
        adjusted = _validate_scalar_resource_measurement(
            measurement["reml_covariance_parameter_adjusted_scalar"],
            SELECTED_METHOD, post_gauge_dates, host["rayon_thread_count"], case_label,
        )
        expected_ratio = adjusted["wall_micros"] / plugin["wall_micros"]
        expected_full_ratio = (
            adjusted["full_product_wall_micros"]
            / plugin["full_product_wall_micros"]
        )
        ratio = measurement["adjusted_to_plugin_wall_ratio"]
        full_ratio = measurement["adjusted_to_plugin_full_product_wall_ratio"]
        shared_fields = (
            "factor_sha256", "direct_factor_receipt_sha256", "factor_block_reads",
            "optimizer_rho_lane_evaluations", "optimizer_q_objective_evaluations",
            "optimizer_primary_rho_pass_histogram",
        )
        if (
                measurement["post_gauge_date_count"] != post_gauge_dates
                or type(measurement["post_gauge_date_count"]) is not int
                or measurement["acquisition_count"] != post_gauge_dates + 1
                or type(measurement["acquisition_count"]) is not int
                or measurement["target_count"] != TEMPORAL_RESOURCE_TARGET_COUNT
                or type(measurement["target_count"]) is not int
                or not _u64(measurement["varied_target_fingerprint_count"])
                or not 257 <= measurement["varied_target_fingerprint_count"] <= TEMPORAL_RESOURCE_TARGET_COUNT
                or any(plugin[field] != adjusted[field] for field in shared_fields)
                or plugin["covariance_parameter_derivative_lane_evaluations"] != 0
                or plugin["covariance_parameter_adjustment_count"] != 0
                or adjusted["covariance_parameter_derivative_lane_evaluations"]
                != adjusted["optimizer_q_objective_evaluations"]
                or adjusted["covariance_parameter_adjustment_count"]
                != TEMPORAL_RESOURCE_TARGET_COUNT - 1
                or plugin["wall_micros"] > ((1 << 64) - 1) // 2
                or plugin["full_product_wall_micros"] > ((1 << 64) - 1) // 2
                or adjusted["wall_micros"] > 2 * plugin["wall_micros"]
                or adjusted["full_product_wall_micros"]
                > 2 * plugin["full_product_wall_micros"]
                or not _finite_number(ratio)
                or not _finite_number(full_ratio)
                or abs(ratio - expected_ratio)
                > max(abs(expected_ratio), 1.0) * 1.0e-12
                or abs(full_ratio - expected_full_ratio)
                > max(abs(expected_full_ratio), 1.0) * 1.0e-12
                or ratio > TEMPORAL_RESOURCE_WALL_RATIO_LIMIT
                or full_ratio > TEMPORAL_RESOURCE_WALL_RATIO_LIMIT):
            raise RuntimeError(f"{case_label} failed the frozen resource contract")


def _validate_method_selection(selection: dict) -> None:
    _require_exact_keys(
        selection, TEMPORAL_METHOD_SELECTION_KEYS, "temporal method selection"
    )
    exact_values = {
        "schema": TEMPORAL_METHOD_SELECTION_SCHEMA,
        "status": "pre_outcome_selected",
        "selected_method": SELECTED_METHOD,
        "selected_method_version": SELECTED_METHOD_VERSION,
        "canonical_v4_preregistration_sha256": canonical_v4_sha256(),
        "tile_rows": TEMPORAL_RESOURCE_TILE_ROWS,
        "tile_columns": TEMPORAL_RESOURCE_TILE_COLUMNS,
        "target_count": TEMPORAL_RESOURCE_TARGET_COUNT,
        "post_gauge_date_counts": list(TEMPORAL_RESOURCE_DATE_COUNTS),
        "adjusted_to_plugin_wall_ratio_limit": TEMPORAL_RESOURCE_WALL_RATIO_LIMIT,
        "worker_scratch_limit_bytes": TEMPORAL_RESOURCE_WORKER_SCRATCH_LIMIT_BYTES,
        "resident_set_limit_bytes": TEMPORAL_RESOURCE_RSS_LIMIT_BYTES,
        "outcomes_present": False,
    }
    hashes = (
        "candidate_resource_receipt_sha256", "product_source_sha256",
        "benchmark_source_sha256", "batch_source_sha256",
        "temporal_covariance_batch_binary_sha256",
        "temporal_inference_bench_binary_sha256",
    )
    if (
            any(selection.get(field) != expected for field, expected in exact_values.items())
            or any(type(selection[field]) is not int for field in (
                "selected_method_version", "tile_rows", "tile_columns", "target_count",
                "worker_scratch_limit_bytes", "resident_set_limit_bytes",
            ))
            or selection["outcomes_present"] is not False
            or not _finite_number(selection["adjusted_to_plugin_wall_ratio_limit"])
            or any(not _canonical_sha256(selection.get(field)) for field in hashes)):
        raise RuntimeError("temporal method selection differs from the frozen contract")


def build_method_selection_receipt(candidate_path: Path) -> dict:
    candidate_raw = _resource_evidence_bytes(
        Path(candidate_path),
        TEMPORAL_RESOURCE_RECEIPT_CAP_BYTES,
        "temporal candidate resource receipt",
    )
    candidate = _strict_json_object(
        candidate_raw, "temporal candidate resource receipt"
    )
    _validate_resource_receipt(
        candidate,
        _validate_binary_identity(
            candidate.get("temporal_covariance_batch_binary"),
            "temporal candidate batch binary",
        ),
        _validate_binary_identity(
            candidate.get("temporal_inference_bench_binary"),
            "temporal candidate benchmark binary",
        ),
        "temporal candidate resource receipt",
    )
    if (
            candidate["status"] != "candidate_evidence_only"
            or candidate.get("pre_outcome_selection_receipt_sha256") is not None):
        raise RuntimeError("method selection requires candidate-only resource evidence")
    selection = {
        "schema": TEMPORAL_METHOD_SELECTION_SCHEMA,
        "status": "pre_outcome_selected",
        "selected_method": SELECTED_METHOD,
        "selected_method_version": SELECTED_METHOD_VERSION,
        "candidate_resource_receipt_sha256": hashlib.sha256(
            candidate_raw
        ).hexdigest(),
        "canonical_v4_preregistration_sha256": canonical_v4_sha256(),
        "product_source_sha256": candidate["product_source_sha256"],
        "benchmark_source_sha256": candidate["benchmark_source_sha256"],
        "batch_source_sha256": candidate["batch_source_sha256"],
        "temporal_covariance_batch_binary_sha256": candidate[
            "temporal_covariance_batch_binary"
        ]["sha256"],
        "temporal_inference_bench_binary_sha256": candidate[
            "temporal_inference_bench_binary"
        ]["sha256"],
        "tile_rows": TEMPORAL_RESOURCE_TILE_ROWS,
        "tile_columns": TEMPORAL_RESOURCE_TILE_COLUMNS,
        "target_count": TEMPORAL_RESOURCE_TARGET_COUNT,
        "post_gauge_date_counts": list(TEMPORAL_RESOURCE_DATE_COUNTS),
        "adjusted_to_plugin_wall_ratio_limit": TEMPORAL_RESOURCE_WALL_RATIO_LIMIT,
        "worker_scratch_limit_bytes": TEMPORAL_RESOURCE_WORKER_SCRATCH_LIMIT_BYTES,
        "resident_set_limit_bytes": TEMPORAL_RESOURCE_RSS_LIMIT_BYTES,
        "outcomes_present": False,
    }
    _validate_method_selection(selection)
    return selection


def validate_release_resource_evidence(
        preregistration: dict, evidence_directory: Path) -> dict:
    root = Path(evidence_directory)
    try:
        if root.is_symlink() or not root.resolve(strict=True).is_dir():
            raise RuntimeError
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError("temporal resource evidence directory is missing") from error
    final_raw = _resource_evidence_bytes(
        root / TEMPORAL_RESOURCE_RECEIPT_FILENAME,
        TEMPORAL_RESOURCE_RECEIPT_CAP_BYTES,
        "temporal final resource receipt",
    )
    selection_raw = _resource_evidence_bytes(
        root / TEMPORAL_METHOD_SELECTION_FILENAME,
        TEMPORAL_RESOURCE_RECEIPT_CAP_BYTES,
        "temporal method selection",
    )
    candidate_raw = _resource_evidence_bytes(
        root / TEMPORAL_CANDIDATE_RESOURCE_RECEIPT_FILENAME,
        TEMPORAL_RESOURCE_RECEIPT_CAP_BYTES,
        "temporal candidate resource receipt",
    )
    batch_sha256, batch_bytes = sha256_file(
        root / TEMPORAL_BATCH_BINARY_FILENAME, TEMPORAL_RESOURCE_BINARY_CAP_BYTES
    )
    benchmark_sha256, benchmark_bytes = sha256_file(
        root / TEMPORAL_INFERENCE_BENCH_BINARY_FILENAME,
        TEMPORAL_RESOURCE_BINARY_CAP_BYTES,
    )
    if batch_bytes == 0 or benchmark_bytes == 0:
        raise RuntimeError("temporal resource binary identity is empty")
    observed_batch = {"sha256": batch_sha256, "bytes": batch_bytes}
    observed_benchmark = {"sha256": benchmark_sha256, "bytes": benchmark_bytes}
    final = _strict_json_object(final_raw, "temporal final resource receipt")
    selection = _strict_json_object(selection_raw, "temporal method selection")
    candidate = _strict_json_object(candidate_raw, "temporal candidate resource receipt")
    _validate_resource_receipt(
        final, observed_batch, observed_benchmark, "temporal final resource receipt"
    )
    _validate_resource_receipt(
        candidate,
        _validate_binary_identity(
            candidate.get("temporal_covariance_batch_binary"),
            "temporal candidate batch binary",
        ),
        _validate_binary_identity(
            candidate.get("temporal_inference_bench_binary"),
            "temporal candidate benchmark binary",
        ),
        "temporal candidate resource receipt",
    )
    _validate_method_selection(selection)
    final_sha256 = hashlib.sha256(final_raw).hexdigest()
    selection_sha256 = hashlib.sha256(selection_raw).hexdigest()
    candidate_sha256 = hashlib.sha256(candidate_raw).hexdigest()
    source_fields = (
        "product_source_sha256", "benchmark_source_sha256", "batch_source_sha256"
    )
    if (
            final["status"] != "pass"
            or candidate["status"] != "candidate_evidence_only"
            or final["pre_outcome_selection_receipt_sha256"] != selection_sha256
            or selection["candidate_resource_receipt_sha256"] != candidate_sha256
            or any(candidate[field] != selection[field] for field in source_fields)
            or any(candidate[field] != final[field] for field in source_fields)
            or candidate["temporal_covariance_batch_binary"]["sha256"]
            != selection["temporal_covariance_batch_binary_sha256"]
            or candidate["temporal_inference_bench_binary"]["sha256"]
            != selection["temporal_inference_bench_binary_sha256"]
            or preregistration.get("selected_method") != SELECTED_METHOD
            or preregistration.get("selected_method_version") != SELECTED_METHOD_VERSION
            or preregistration.get("pre_outcome_selection_receipt_sha256")
            != selection_sha256):
        raise RuntimeError(
            "temporal resource evidence differs from the frozen selection chain"
        )
    return {
        "resource_receipt_sha256": final_sha256,
        "method_selection_receipt_sha256": selection_sha256,
        "candidate_resource_receipt_sha256": candidate_sha256,
        "batch_binary": observed_batch,
        "benchmark_binary": observed_benchmark,
    }


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


def atomic_write_or_validate(path: Path, payload: bytes, byte_cap: int) -> None:
    if path.exists() or path.is_symlink():
        if _read_bounded_regular(path, byte_cap) != payload:
            raise RuntimeError(f"{path.name} is stale or tampered")
        return
    atomic_write_no_replace(path, payload)


def _runtime_binary_identity(binary: Path) -> tuple[str, int]:
    return sha256_file(binary, 1024 * 1024 * 1024)


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
    binary_sha256, binary_bytes = _runtime_binary_identity(resolved_binary)
    frozen_identity = preregistration.get("producer_identity")
    expected_identity = {
        "schema": "dolphinrust.temporal-covariance.producer-identity/2",
        "source_set_schema": FROZEN_SOURCE_SET_SCHEMA,
        "source_set_sha256": source_set_sha256,
        "binary_path": "target/release/examples/temporal_covariance_batch",
    }
    if frozen_identity != expected_identity or source_set_sha256 != FROZEN_SOURCE_SET_SHA256:
        raise RuntimeError("frozen temporal producer source identity does not match the run")
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


def _shard_paths(shards: Path, cell: dict) -> dict[str, Path]:
    stem = f"{cell['cell_index']:05d}"
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
        "seed_sha256", "seed", "factor_sha256", "realized_factor_rank",
        "fixed_factor_status", "production_path_status", "comparator_methods",
        "attempted", "emitted", "failed", "fit", "provenance",
        "production_receipts", "record_sha256",
    }
    identity = ("cell_id", "cell_index", "outer_seed_index", "seed_sha256", "seed")
    if (
            not isinstance(record, dict)
            or set(record) != response_keys
            or record.get("schema") != batch_schema
            or record.get("execution_path") not in ("fixed_factor", "production_path")):
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
    for field in ("factor_sha256", "record_sha256"):
        value = record.get(field)
        if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise RuntimeError("batch returned a malformed compact record identity")
    if record["record_sha256"] is None:
        raise RuntimeError("batch omitted its compact record identity")
    if record["record_sha256"] != compact_record_sha256(record):
        raise RuntimeError("batch compact record digest differs from its semantics")
    rank = record.get("realized_factor_rank")
    if rank is not None and (type(rank) is not int or rank < 0):
        raise RuntimeError("batch returned a malformed realized factor rank")
    if (record.get("factor_sha256") is None) != (rank is None):
        raise RuntimeError("batch returned incomplete factor identity")

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
            "fitted_rho", "fitted_process_variance", "fitted_parameter_active_set",
            "raw_correlation",
            "valid_date_count", "rank", "degrees_of_freedom",
            "covariance_condition_number", "ols", "oracle_gls", "conditional_wls",
            "scalar_effective_n", "plugin_gls", "adjusted_scalar", "adjusted_profile",
            "complete_refit_bootstrap", "bootstrap_attempts", "bootstrap_successes",
        }
        if not isinstance(fit, dict) or set(fit) != fit_keys or fit["status"] not in TEMPORAL_STATUSES:
            raise RuntimeError("batch returned a malformed fit schema or status")
        if fit["fitted_parameter_active_set"] is not None \
                and fit["fitted_parameter_active_set"] not in TEMPORAL_ACTIVE_SET_STATUSES:
            raise RuntimeError("batch returned a malformed fitted active set")
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
                    "adjusted_scalar", "adjusted_profile", "complete_refit_bootstrap",
                ))
                or fit["conditional_wls"]["status"] != "LegacyNonComparable"
                or fit["bootstrap_attempts"] != COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS
                or fit["bootstrap_successes"]
                < COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES):
            raise RuntimeError("batch returned an incomplete evaluated fit")

    evaluated = fit is not None and fit["status"] == "Evaluated"
    if record["emitted"] != evaluated:
        raise RuntimeError("batch fit and emission dispositions are inconsistent")
    if record["execution_path"] == "fixed_factor":
        if (
                record["production_path_status"] is not None
                or record["provenance"] is not None
                or record["production_receipts"] is not None):
            raise RuntimeError("batch returned inconsistent fixed-factor state")
        if fit is None:
            if record["fixed_factor_status"] is not None:
                raise RuntimeError("batch returned a fixed-factor status without a fit")
        elif (
                record["fixed_factor_status"] not in TEMPORAL_STATUSES
                or record["fixed_factor_status"] != fit["status"]):
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
            "temporal_dgp_receipt_sha256",
            "fixed_l2_difference_factor_sha256", "fixed_l2_realized_rank",
            "source_correlation_model", "source_correlation_distance_scale_pixels",
            "source_correlation_support_union_count", "effective_looks_fraction",
            "source_correlation_receipt_sha256", "outer_coverage_dgp",
            "conditional_covariance_oracle", "conditional_oracle_replicates",
            "temporal_profile_fit_count", "temporal_bootstrap_attempts",
        }
        dense_fields = {
            "fixed_l2_difference_factor",
            "fixed_l2_difference_covariance", "fixed_l2_difference_variance",
            "carrier_difference_history", "linked_difference_history",
            "source_carrier_difference_history", "source_linked_difference_history",
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
            or receipts.get("fixed_l2_difference_factor_sha256")
            != record["factor_sha256"]
            or receipts.get("fixed_l2_realized_rank")
            != record["realized_factor_rank"]
            or receipts.get("source_correlation_model") != SOURCE_CORRELATION_MODEL
            or receipts.get("source_correlation_distance_scale_pixels")
            != SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
            or type(receipts.get("source_correlation_support_union_count")) is not int
            or receipts["source_correlation_support_union_count"] <= 0
            or receipts.get("outer_coverage_dgp") != OUTER_COVERAGE_DGP
            or receipts.get("conditional_covariance_oracle")
            != CONDITIONAL_COVARIANCE_ORACLE
            or receipts.get("temporal_profile_fit_count") != 1
            or receipts.get("temporal_bootstrap_attempts")
            != (fit["bootstrap_attempts"] if fit is not None else 0)
        ):
            raise RuntimeError("batch returned stale source-correlation provenance")
        factor = receipts.get("fixed_l2_difference_factor")
        if request.get("retain_dense_evidence", False) and (
                not isinstance(factor, list)
                or len(factor) != len(request["days"])
                or any(
                    not isinstance(row, list)
                    or len(row) != receipts["fixed_l2_realized_rank"]
                    or any(not finite_optional(value) or value is None for value in row)
                    for row in factor
                )):
            raise RuntimeError("batch returned a malformed direct fixed-L2 factor")
        if request.get("retain_dense_evidence", False):
            histories = [
                receipts.get("carrier_difference_history"),
                receipts.get("linked_difference_history"),
                receipts.get("source_carrier_difference_history"),
                receipts.get("source_linked_difference_history"),
            ]
            if any(
                    not isinstance(history, list)
                    or len(history) != len(request["days"])
                    or any(not finite_optional(value) or value is None for value in history)
                    for history in histories):
                raise RuntimeError("batch returned malformed temporal DGP histories")
            diagonal = [sum(value * value for value in row) for row in factor]
            validity = request["production_path"]["validity"]
            retained = [
                index for index in range(1, len(diagonal)) if validity[index]
            ]
            if not retained or any(value <= 0.0 for value in diagonal[1:]):
                raise RuntimeError("batch returned a malformed direct fixed-L2 factor")
            scale = math.exp(
                sum(math.log(diagonal[index]) for index in retained) / len(retained)
            )
            production = request["production_path"]
            expected_carrier = [0.0] + [
                production["truth_slope_per_day"] * request["days"][index]
                + math.sqrt(request["options"]["oracle_process_variance"])
                * math.sqrt(diagonal[index] / scale)
                * production["latent_ar_path"][index]
                for index in range(1, len(diagonal))
            ]
            expected_measurement_error = [0.0] + [
                sum(
                    factor[index][component]
                    * production["measurement_normal_path"][component]
                    for component in range(receipts["fixed_l2_realized_rank"])
                )
                for index in range(1, len(diagonal))
            ]
            carrier, linked, _, _ = histories
            if any(
                    not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
                    for observed, expected in zip(carrier, expected_carrier)
            ) or any(
                    not math.isclose(
                        linked[index],
                        carrier[index] + expected_measurement_error[index],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for index in range(len(carrier))
            ):
                raise RuntimeError("batch violated the frozen temporal DGP identity")
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
            "fitted_process_variance", "fitted_parameter_active_set",
            "issue52_receipt_sha256", "issue54_receipt_sha256",
            "reference", "condition_number", "scope", "bootstrap_attempts",
            "bootstrap_successes", "validation_receipt_sha256", "estimator_input_sha256",
            "bootstrap_minimum_success_fraction", "selected_method",
        }
        if (
                not isinstance(provenance, dict)
                or set(provenance) != provenance_keys
                or provenance["schema"] != "dolphinrust-temporal-covariance-provenance/2"
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
                or provenance["fitted_parameter_active_set"]
                != fit["fitted_parameter_active_set"]
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


def _validate_record_pair(fixed: dict, production: dict) -> None:
    if (
            fixed["execution_path"] != "fixed_factor"
            or production["execution_path"] != "production_path"
            or fixed["factor_sha256"] != production["factor_sha256"]
            or fixed["realized_factor_rank"] != production["realized_factor_rank"]
            or fixed["fit"] != production["fit"]
            or fixed["emitted"] != production["emitted"]
            or fixed["failed"] != production["failed"]):
        raise RuntimeError(
            "batch returned a missing, reordered, or inconsistent seed record pair"
        )


def _validate_frame_response(
        response: dict, requests: list[dict], batch_schema: str) -> tuple[list[dict], dict]:
    if (
            not isinstance(response, dict)
            or set(response) != {"schema", "records", "resource"}
            or response.get("schema") != batch_schema
            or not isinstance(response.get("records"), list)
            or len(response["records"]) != 2 * len(requests)):
        raise RuntimeError("batch returned a malformed response frame")
    resource = response.get("resource")
    expected_resource_keys = {
        "schema", "request_count", "record_count", "factor_generation_count",
        "temporal_fit_count", "profile_fit_count", "bootstrap_attempts",
        "attempt_record_count", "rayon_worker_count",
        "wall_micros", "resident_set_bytes_before", "resident_set_bytes_after",
    }
    if (
            not isinstance(resource, dict)
            or set(resource) != expected_resource_keys
            or resource.get("schema") != FRAME_RESOURCE_SCHEMA
            or any(
                type(resource.get(field)) is not int or resource[field] < 0
                for field in expected_resource_keys - {"schema"}
            )
            or resource["request_count"] != len(requests)
            or resource["record_count"] != 2 * len(requests)
            or resource["attempt_record_count"] != 2 * len(requests)
            or resource["factor_generation_count"] != len(requests)
            or resource["temporal_fit_count"] != len(requests)
            or resource["rayon_worker_count"]
            != max(1, min(MAX_RAYON_WORKERS, os.cpu_count() or 1, MAX_FRAME_REQUESTS))):
        raise RuntimeError("batch returned a malformed frame resource receipt")
    expected_profile_fit_count = 0
    expected_bootstrap_attempts = 0
    for index, request in enumerate(requests):
        fixed, production = response["records"][2 * index:2 * index + 2]
        _validate_compact_record(fixed, request, batch_schema)
        _validate_compact_record(production, request, batch_schema)
        _validate_record_pair(fixed, production)
        if production["production_receipts"] is not None:
            expected_profile_fit_count += production["production_receipts"][
                "temporal_profile_fit_count"
            ]
            expected_bootstrap_attempts += production["production_receipts"][
                "temporal_bootstrap_attempts"
            ]
    if (
            resource["profile_fit_count"] != expected_profile_fit_count
            or resource["bootstrap_attempts"] != expected_bootstrap_attempts):
        raise RuntimeError("batch returned inconsistent frame method accounting")
    return response["records"], resource


def _exchange_frame(process: subprocess.Popen, frame: dict) -> tuple[list[dict], dict]:
    encoded = canonical_json_bytes(frame) + b"\n"
    if len(encoded) > MAX_REQUEST_LINE_BYTES:
        raise RuntimeError("temporal covariance request frame exceeds its line cap")
    process.stdin.write(encoded)
    process.stdin.flush()
    line = _read_bounded_line(process.stdout, MAX_RESPONSE_LINE_BYTES)
    if not line:
        raise RuntimeError("temporal covariance batch ended before its response frame")
    return _validate_frame_response(
        json.loads(line), frame["requests"], frame["schema"]
    )


def _response_semantic_bytes(record: dict) -> bytes:
    return record_wire_bytes(record) + b"\n"


def _require_binary_identity(binary: Path, identity: dict) -> None:
    digest, byte_count = _runtime_binary_identity(binary)
    if digest != identity["binary_sha256"] or (
            "binary_bytes" in identity and byte_count != identity["binary_bytes"]):
        raise RuntimeError("temporal covariance batch binary identity is stale")


def _cleanup_uncommitted(paths: dict[str, Path]) -> None:
    if paths["commit"].exists() or paths["commit"].is_symlink():
        return
    for path in paths.values():
        _remove_owned_regular(path)
        _remove_owned_regular(path.with_name(path.name + ".partial"))


def _validate_manifest(
        manifest: dict, identity: dict, cell: dict, seed_count: int,
        records_sha256: str, records_bytes: int,
        response_semantic_sha256: str, profile_fit_count: int,
        bootstrap_attempts: int) -> None:
    expected_keys = {
        "schema", "cell_id", "cell_index", "seed_request_count",
        "attempt_record_count", "frame_count", "factor_generation_count",
        "temporal_fit_count", "profile_fit_count", "bootstrap_attempts",
        "records_sha256", "records_bytes", "request_schedule_sha256",
        "response_semantic_sha256",
        "producer_identity", "attempted", "emitted", "failed",
        "producer_source_set_sha256", "producer_binary_sha256",
        "total_wall_micros", "peak_resident_set_bytes", "max_rayon_worker_count",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RuntimeError("shard manifest schema is malformed")
    expected = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "cell_id": cell["cell_id"],
        "cell_index": cell["cell_index"],
        "seed_request_count": seed_count,
        "attempt_record_count": seed_count * 2,
        "frame_count": math.ceil(seed_count / MAX_FRAME_REQUESTS),
        "factor_generation_count": seed_count,
        "temporal_fit_count": seed_count,
        "profile_fit_count": profile_fit_count,
        "bootstrap_attempts": bootstrap_attempts,
        "records_sha256": records_sha256,
        "records_bytes": records_bytes,
        "response_semantic_sha256": response_semantic_sha256,
        "producer_identity": identity,
        "producer_source_set_sha256": identity["source_set_sha256"],
        "producer_binary_sha256": identity["binary_sha256"],
        "attempted": seed_count * 2,
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise RuntimeError("shard manifest identity is stale or malformed")
    for field in (
            "emitted", "failed", "total_wall_micros", "peak_resident_set_bytes",
            "max_rayon_worker_count"):
        if type(manifest.get(field)) is not int or manifest[field] < 0:
            raise RuntimeError("shard manifest counters are malformed")
    if (
            not 0 <= profile_fit_count <= seed_count
            or not 0 <= bootstrap_attempts
            <= seed_count * COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS
            or manifest["emitted"] + manifest["failed"] != seed_count * 2
            or manifest["max_rayon_worker_count"]
            != max(1, min(MAX_RAYON_WORKERS, os.cpu_count() or 1, MAX_FRAME_REQUESTS))):
        raise RuntimeError("shard manifest dispositions do not equal the record denominator")
    schedule = manifest.get("request_schedule_sha256")
    if not isinstance(schedule, str) or len(schedule) != 64:
        raise RuntimeError("shard manifest request schedule is malformed")


def _read_committed_shard(
        preregistration: dict, cell: dict, seed_count: int,
        paths: dict[str, Path], identity: dict,
        scorer: StreamingScores | None) -> dict:
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
    record_count = 0
    profile_fit_count = 0
    bootstrap_attempts = 0
    pair: list[dict] = []
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
            record = json.loads(line)
            if line != record_wire_bytes(record) + b"\n":
                raise RuntimeError("shard contains a noncanonical record encoding")
            request = request_for(cell, record_count // 2, preregistration)
            if record_count % 2 == 0:
                schedule.update(canonical_json_bytes(request) + b"\n")
            _validate_compact_record(record, request, identity["batch_schema"])
            receipts = record.get("production_receipts")
            if receipts is not None:
                profile_fit_count += receipts["temporal_profile_fit_count"]
                bootstrap_attempts += receipts["temporal_bootstrap_attempts"]
            semantic_digest.update(_response_semantic_bytes(record))
            pair.append(record)
            if len(pair) == 2:
                _validate_record_pair(pair[0], pair[1])
                pair.clear()
            if scorer is not None:
                scorer.update(record)
            record_count += 1
            if record_count > seed_count * 2:
                raise RuntimeError("shard contains a top-up attempt record")
    response_semantic_sha256 = semantic_digest.hexdigest()
    _validate_manifest(
        manifest, identity, cell, seed_count,
        records_sha256, records_bytes, response_semantic_sha256,
        profile_fit_count, bootstrap_attempts,
    )
    if (
            pair
            or record_count != seed_count * 2
            or schedule.hexdigest() != manifest["request_schedule_sha256"]):
        raise RuntimeError("shard seed schedule is missing, duplicated, or reordered")
    if commit["response_semantic_sha256"] != response_semantic_sha256:
        raise RuntimeError("shard response semantic receipt is stale or tampered")
    return manifest


def execute_or_resume_shard(
        preregistration: dict, cell: dict, seed_count: int,
        shards: Path, binary: Path, identity: dict) -> tuple[dict, bool]:
    paths = _shard_paths(shards, cell)
    _require_binary_identity(binary, identity)
    if paths["commit"].exists() or paths["commit"].is_symlink():
        return (
            _read_committed_shard(
                preregistration, cell, seed_count, paths, identity, None,
            ),
            True,
        )
    _cleanup_uncommitted(paths)
    partial_records = paths["records"].with_name(paths["records"].name + ".partial")
    process = subprocess.Popen(
        [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        env=batch_environment(preregistration),
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("temporal covariance batch pipes are unavailable")
    records_digest = hashlib.sha256()
    schedule_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    attempted = emitted = failed = total_wall = peak_rss = records_bytes = 0
    frame_count = factor_generation_count = temporal_fit_count = 0
    profile_fit_count = bootstrap_attempts = 0
    max_rayon_worker_count = 0
    try:
        with partial_records.open("xb") as retained:
            for frame in iter_request_frames(cell, seed_count, preregistration):
                for request in frame["requests"]:
                    schedule_digest.update(canonical_json_bytes(request) + b"\n")
                records, resource = _exchange_frame(process, frame)
                for record in records:
                    line = record_wire_bytes(record) + b"\n"
                    semantic_digest.update(_response_semantic_bytes(record))
                    records_bytes += len(line)
                    if records_bytes > MAX_SHARD_RECORD_BYTES:
                        raise RuntimeError("temporal covariance shard exceeds its byte cap")
                    retained.write(line)
                    records_digest.update(line)
                    attempted += 1
                    emitted += int(record["emitted"])
                    failed += int(record["failed"])
                frame_count += 1
                factor_generation_count += resource["factor_generation_count"]
                temporal_fit_count += resource["temporal_fit_count"]
                profile_fit_count += resource["profile_fit_count"]
                bootstrap_attempts += resource["bootstrap_attempts"]
                total_wall += resource["wall_micros"]
                peak_rss = max(
                    peak_rss,
                    resource["resident_set_bytes_before"],
                    resource["resident_set_bytes_after"],
                )
                max_rayon_worker_count = max(
                    max_rayon_worker_count, resource["rayon_worker_count"]
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
        "seed_request_count": seed_count,
        "attempt_record_count": attempted,
        "frame_count": frame_count,
        "factor_generation_count": factor_generation_count,
        "temporal_fit_count": temporal_fit_count,
        "profile_fit_count": profile_fit_count,
        "bootstrap_attempts": bootstrap_attempts,
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
        "max_rayon_worker_count": max_rayon_worker_count,
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
        preregistration, cell, seed_count, paths, identity, None
    )
    return manifest, False


def _result_receipt(
        preregistration: dict, seed_count: int, processed: int, emitted: int,
        failed: int, total_wall: int, peak_rss: int, scores: dict,
        records: list[dict], exact_denominator: bool, identity: dict,
        resource_evidence_bound: bool) -> dict:
    expected_attempts = (
        len(cells(preregistration)) * seed_count * len(preregistration["execution_paths"])
    )
    expected_seed_requests = len(cells(preregistration)) * seed_count
    result_payload = canonical_json_bytes({"scores": scores})
    resource_gates = {
        "rss": peak_rss <= preregistration["resource_limits"]["rss_limit_bytes"],
        "artifact_size": len(result_payload)
            <= preregistration["resource_limits"]["artifact_size_limit_bytes"],
        "bound_resource_receipt": resource_evidence_bound,
    }
    complete_execution = processed == expected_attempts
    engine_validation_eligible = (
        exact_denominator
        and scores["all_methods_pass"]
        and all(resource_gates.values())
    )
    engine_validation = preregistration["engine_validation"]
    return {
        "schema": preregistration["schemas"]["generator"],
        "preregistration_schema": preregistration["schema"],
        "pre_outcome_status": preregistration["status"],
        "supported_cell_sha256": preregistration["supported_cell_sha256"],
        "expected_attempt_record_count": expected_attempts,
        "processed_attempt_record_count": processed,
        "seed_request_count": processed // 2,
        "expected_seed_request_count": expected_seed_requests,
        "attempt_record_count": processed,
        "emitted_attempt_record_count": emitted,
        "failed_attempt_record_count": failed,
        "skipped_attempt_record_count": expected_attempts - processed,
        "seed_requests_per_cell": seed_count,
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
        "run_committed": False,
        "result_records_sha256": hashlib.sha256(result_payload).hexdigest(),
        "result_records_bytes": len(result_payload),
        "engine_validation_eligible": engine_validation_eligible,
        "engine_validation_status": (
            engine_validation["passing_status"]
            if engine_validation_eligible
            else engine_validation["blocked_status"]
        ),
        "resource": {
            "total_wall_micros": total_wall,
            "peak_resident_set_bytes": peak_rss,
            "result_artifact_bytes": len(result_payload),
        },
        "resource_gates": resource_gates,
        "resource_limits": preregistration["resource_limits"],
    }


def _run_probe(
        preregistration: dict, seed_count: int, limit: int, binary: Path,
        identity: dict) -> dict:
    if limit > MAX_PROBE_REQUESTS:
        raise RuntimeError("probe request count exceeds its retained bound")
    selected = list(itertools.islice(iter_requests(preregistration, seed_count), limit))
    process = subprocess.Popen(
        [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        env=batch_environment(preregistration),
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("temporal covariance batch pipes are unavailable")
    scorer = StreamingScores(preregistration)
    records = []
    emitted = failed = total_wall = peak_rss = 0
    try:
        for _, grouped in itertools.groupby(
                selected, key=lambda request: (request["cell_id"], request["cell_index"])):
            cell_requests = list(grouped)
            for start in range(0, len(cell_requests), MAX_FRAME_REQUESTS):
                frame = request_frame(
                    cell_requests[start:start + MAX_FRAME_REQUESTS],
                    preregistration["schemas"]["batch"],
                )
                frame_records, resource = _exchange_frame(process, frame)
                for record in frame_records:
                    scorer.update(record)
                    records.append(record)
                    emitted += int(record["emitted"])
                    failed += int(record["failed"])
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
        total_wall, peak_rss, scores, records, False, identity, True,
    )


def run(
        preregistration: dict, seed_count: int, limit: int | None,
        run_root: Path | None = None, binary: Path | None = None,
        resource_evidence_directory: Path | None = None) -> dict:
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
    shard_count = len(frozen_cells)
    protocol = preregistration["execution_protocol"]
    if (
            protocol.get("shard_axis") != ["cell_index"]
            or protocol.get("shard_count") != shard_count
            or protocol.get("frame_request_count") != MAX_FRAME_REQUESTS
            or protocol.get("maximum_rayon_workers") != MAX_RAYON_WORKERS
            or protocol.get("seed_requests_per_shard")
            != preregistration["outer_seeds_per_supported_cell"]
            or protocol.get("attempt_records_per_shard")
            != 2 * preregistration["outer_seeds_per_supported_cell"]
            or protocol.get("seed_order") != "strictly_increasing_0_through_1049"
            or protocol.get("commit")
            != "records_then_manifest_then_atomic_commit_marker_no_replace"
            or protocol.get("resume")
            != "full_same_byte_hash_schema_identity_and_seed_schedule_revalidation"
            or protocol.get("partial_policy")
            != "owned_uncommitted_residuals_removed_before_full_shard_restart_from_seed_zero"
            or protocol.get("top_up_allowed") is not False
            or protocol.get("dense_attempt_evidence_retained") is not False):
        raise RuntimeError("preregistered shard count does not match the frozen scope")
    expected_seed_requests = (
        shard_count * preregistration["outer_seeds_per_supported_cell"]
    )
    expected_full_attempts = expected_seed_requests * 2
    engine_validation = preregistration.get("engine_validation")
    if (
            preregistration.get("schema")
            != "dolphinrust-temporal-covariance-preregistration/5"
            or preregistration.get("schemas", {}).get("generator")
            != "dolphinrust-temporal-covariance-simulation/9"
            or preregistration.get("schemas", {}).get("batch")
            != "dolphinrust-temporal-covariance-batch/7"
            or preregistration.get("schemas", {}).get("scorer")
            != "coverage_bias_interval_score/6"
            or preregistration.get("schemas", {}).get("run_identity")
            != RUN_IDENTITY_SCHEMA
            or preregistration.get("schemas", {}).get("shard_manifest")
            != SHARD_MANIFEST_SCHEMA
            or preregistration.get("schemas", {}).get("shard_commit")
            != SHARD_COMMIT_SCHEMA
            or preregistration.get("schemas", {}).get("run_manifest")
            != RUN_MANIFEST_SCHEMA
            or preregistration.get("schemas", {}).get("run_commit")
            != RUN_COMMIT_SCHEMA
            or preregistration.get("selected_method") != SELECTED_METHOD
            or preregistration.get("selected_method_version")
            != SELECTED_METHOD_VERSION
            or not _canonical_sha256(
                preregistration.get("pre_outcome_selection_receipt_sha256")
            )
            or not isinstance(engine_validation, dict)
            or engine_validation.get("attempt_count") != expected_full_attempts
            or engine_validation.get("seed_request_count") != expected_seed_requests
            or engine_validation.get("passing_status")
            != "synthetic_validated_scope_match"
            or engine_validation.get("blocked_status")
            != "blocked_pending_complete_passing_synthetic_execution"
            or engine_validation.get("external_holdout_required") is not False
            or engine_validation.get("independent_review_required") is not False):
        raise RuntimeError("synthetic engine-validation contract is stale")
    expected_retained_bound = (
        shard_count * (
            MAX_SHARD_RECORD_BYTES + MAX_MANIFEST_BYTES + MAX_COMMIT_BYTES
        )
        + MAX_COMMIT_BYTES
        + MAX_MANIFEST_BYTES
        + MAX_COMMIT_BYTES
        + MAX_FINAL_RECEIPT_BYTES
    )
    resource_limits = preregistration["resource_limits"]
    if (
            resource_limits.get("retained_bound_bytes") != expected_retained_bound
            or expected_retained_bound > resource_limits["artifact_size_limit_bytes"]):
        raise RuntimeError("preregistered retained-resource bound is stale")
    if resource_evidence_directory is None:
        raise RuntimeError("temporal execution requires observed resource evidence")
    resource_evidence = validate_release_resource_evidence(
        preregistration, resource_evidence_directory
    )
    root = Path(__file__).parents[1]
    if binary is None:
        subprocess.run([
            "cargo", "build", "--release", "-p", "dolphin-timeseries",
            "--example", "temporal_covariance_batch",
        ], cwd=root, check=True)
        binary = root / "target/release/examples/temporal_covariance_batch"
    identity = producer_identity(preregistration, binary)
    evidence_digest_fields = (
        "candidate_resource_receipt_sha256",
        "method_selection_receipt_sha256",
        "resource_receipt_sha256",
    )
    if any(
            not _canonical_sha256(resource_evidence.get(field))
            for field in evidence_digest_fields):
        raise RuntimeError("temporal resource evidence identity is malformed")
    benchmark_binary = resource_evidence.get("benchmark_binary")
    if (
            not isinstance(benchmark_binary, dict)
            or not _canonical_sha256(benchmark_binary.get("sha256"))):
        raise RuntimeError("temporal resource benchmark identity is malformed")
    identity.update({
        field: resource_evidence[field] for field in evidence_digest_fields
    })
    identity["resource_benchmark_binary_sha256"] = benchmark_binary["sha256"]
    if resource_evidence["batch_binary"] != {
            "sha256": identity["binary_sha256"],
            "bytes": identity["binary_bytes"],
    }:
        raise RuntimeError(
            "temporal producer binary differs from the observed resource evidence"
        )
    if limit is not None:
        return _run_probe(preregistration, seed_count, limit, binary, identity)
    if seed_count != preregistration["outer_seeds_per_supported_cell"]:
        raise RuntimeError("resumable execution requires the exact frozen seed denominator")
    if run_root is None:
        raise RuntimeError("resumable execution requires a run root")
    shards = initialize_run_root(run_root, identity)
    for cell in frozen_cells:
        execute_or_resume_shard(
            preregistration, cell, seed_count, shards, binary, identity,
        )
    expected_names = set()
    for cell in frozen_cells:
        expected_names.update(path.name for path in _shard_paths(shards, cell).values())
    if {path.name for path in shards.iterdir()} != expected_names:
        raise RuntimeError("run root contains partial, missing, or out-of-scope shard artifacts")
    scorer = StreamingScores(preregistration)
    processed = emitted = failed = total_wall = peak_rss = 0
    profile_fit_count = bootstrap_attempts = 0
    max_rayon_worker_count = 0
    ordered_shards = []
    ordered_commit_digest = hashlib.sha256()
    for cell in frozen_cells:
        paths = _shard_paths(shards, cell)
        manifest = _read_committed_shard(
            preregistration, cell, seed_count, paths, identity, scorer,
        )
        manifest_bytes = _read_bounded_regular(paths["manifest"], MAX_MANIFEST_BYTES)
        commit_bytes = _read_bounded_regular(paths["commit"], MAX_COMMIT_BYTES)
        commit = json.loads(commit_bytes)
        ordered_commit_digest.update(commit_bytes)
        ordered_shards.append({
            "cell_id": cell["cell_id"],
            "cell_index": cell["cell_index"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "commit_sha256": hashlib.sha256(commit_bytes).hexdigest(),
            "records_sha256": commit["records_sha256"],
            "seed_request_count": manifest["seed_request_count"],
            "attempt_record_count": manifest["attempt_record_count"],
        })
        processed += manifest["attempted"]
        emitted += manifest["emitted"]
        failed += manifest["failed"]
        total_wall += manifest["total_wall_micros"]
        peak_rss = max(peak_rss, manifest["peak_resident_set_bytes"])
        profile_fit_count += manifest["profile_fit_count"]
        bootstrap_attempts += manifest["bootstrap_attempts"]
        max_rayon_worker_count = max(
            max_rayon_worker_count, manifest["max_rayon_worker_count"]
        )
    scores = scorer.finalize(require_complete=True)
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "producer_identity": identity,
        "preregistration_sha256": identity["preregistration_sha256"],
        "shard_count": shard_count,
        "seed_request_count": expected_seed_requests,
        "attempt_record_count": expected_full_attempts,
        "profile_fit_count": profile_fit_count,
        "bootstrap_attempts": bootstrap_attempts,
        "max_rayon_worker_count": max_rayon_worker_count,
        "emitted": emitted,
        "failed": failed,
        "ordered_shards": ordered_shards,
        "ordered_shard_commits_sha256": ordered_commit_digest.hexdigest(),
        "scores_sha256": hashlib.sha256(canonical_json_bytes(scores)).hexdigest(),
    }
    run_manifest_bytes = canonical_json_bytes(run_manifest) + b"\n"
    if len(run_manifest_bytes) > MAX_MANIFEST_BYTES:
        raise RuntimeError("temporal covariance run manifest exceeds its byte cap")
    run_manifest_path = run_root / "run_manifest.json"
    atomic_write_or_validate(run_manifest_path, run_manifest_bytes, MAX_MANIFEST_BYTES)
    run_commit = {
        "schema": RUN_COMMIT_SCHEMA,
        "run_manifest_sha256": hashlib.sha256(run_manifest_bytes).hexdigest(),
        "ordered_shard_commits_sha256": ordered_commit_digest.hexdigest(),
        "producer_source_set_sha256": identity["source_set_sha256"],
        "producer_binary_sha256": identity["binary_sha256"],
        "seed_request_count": expected_seed_requests,
        "attempt_record_count": expected_full_attempts,
    }
    run_commit_bytes = canonical_json_bytes(run_commit) + b"\n"
    run_commit_path = run_root / "run_commit.json"
    atomic_write_or_validate(run_commit_path, run_commit_bytes, MAX_COMMIT_BYTES)
    receipt = _result_receipt(
        preregistration, seed_count, processed, emitted, failed,
        total_wall, peak_rss, scores, [], True, identity, True,
    )
    retained_bytes = sum(
        path.stat().st_size
        for path in shards.iterdir()
        if _regular_file(path)
    )
    retained_bytes += sum(
        path.stat().st_size
        for path in (run_root / "run_identity.json", run_manifest_path, run_commit_path)
    )
    receipt["retained_shard_bytes"] = retained_bytes
    receipt["run_manifest_sha256"] = hashlib.sha256(run_manifest_bytes).hexdigest()
    receipt["run_commit_sha256"] = hashlib.sha256(run_commit_bytes).hexdigest()
    receipt["run_committed"] = True
    receipt["resource"]["profile_fit_count"] = profile_fit_count
    receipt["resource"]["bootstrap_attempts"] = bootstrap_attempts
    receipt["resource"]["max_rayon_worker_count"] = max_rayon_worker_count
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
    receipt["engine_validation_eligible"] = (
        receipt["exact_seed_denominator_complete"]
        and receipt["scores"]["all_methods_pass"]
        and receipt["run_committed"]
        and all(receipt["resource_gates"].values())
    )
    receipt["engine_validation_status"] = (
        engine_validation["passing_status"]
        if receipt["engine_validation_eligible"]
        else engine_validation["blocked_status"]
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
    parser.add_argument("--resource-evidence-directory", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.seeds <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--seeds and --limit must be positive")
    preregistration = json.loads(args.prereg.read_text())
    receipt = run(
        preregistration,
        args.seeds,
        args.limit,
        args.run_root,
        resource_evidence_directory=args.resource_evidence_directory,
    )
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if len(encoded) > min(
            MAX_FINAL_RECEIPT_BYTES,
            preregistration["resource_limits"]["artifact_size_limit_bytes"]):
        raise RuntimeError("final temporal covariance receipt exceeds its artifact cap")
    atomic_write_no_replace(args.output, encoded)


if __name__ == "__main__":
    main()
