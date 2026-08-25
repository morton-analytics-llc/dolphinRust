#!/usr/bin/env python3
"""Generate frozen #53 cells and score them through the Rust batch target."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
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


def request_for(cell: dict, outer_seed_index: int, preregistration: dict, execution_path: str) -> dict:
    seed, seed_sha256 = seed_identity(preregistration, cell["cell_index"], outer_seed_index)
    days = days_for(cell)
    missing = missing_indices(cell, seed, len(days) - 1)
    observations = []
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
        value = 0.01 * day + math.sqrt(process_variance) * shape * ar_path[index]
        value += math.sqrt(diagonal[index]) * measurement
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
               "options": options, "fixed_factor": None, "production_path": None}
    if execution_path == "fixed_factor":
        request["fixed_factor"] = {"observations": observations,
                                   "difference_covariance": covariance}
    else:
        values = [0.0 if value is None else value for value in observations]
        overlap = cell["overlap_fraction"]
        noise = [1e-12]
        for index in range(1, len(days)):
            denominator = 2.0 * (1.0 - overlap * math.cos(values[index]))
            noise.append(math.sqrt(diagonal[index] / denominator))
        request["production_path"] = {
            "raw_complex_seed": seed, "issue52_seed": seed, "issue54_seed": seed,
            "target_raw_complex": [[math.cos(value), math.sin(value)] for value in values],
            "reference_raw_complex": [[1.0, 0.0] for _ in values],
            "complex_noise_standard_deviation": noise,
            "validity": [value is not None for value in observations],
            "reference": {
                "geometry_id": "synthetic_same_frame_reference",
                "window_id": cell["reference_context"],
                "overlap_fraction": overlap,
                "distance_pixels": cell["distance_pixels"],
                "sequential_depth": cell["sequential_depth"],
                "approximation": cell["approximation"],
            },
            "scope": "synthetic_validation",
            "validation_receipt_sha256": "53" * 32,
            "selected_method": "complete_refit_bootstrap",
        }
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
            "schema": "coverage_bias_interval_score/3",
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


def run(preregistration: dict, seed_count: int, limit: int | None) -> dict:
    frozen_cells = cells(preregistration)
    if cell_hash(frozen_cells) != preregistration["supported_cell_sha256"]:
        raise RuntimeError("supported cell construction does not match preregistration hash")
    expected_attempts = len(frozen_cells) * seed_count * len(preregistration["execution_paths"])
    selected = iter_requests(preregistration, seed_count)
    if limit is not None:
        selected = itertools.islice(selected, limit)
    root = Path(__file__).parents[1]
    subprocess.run(
        ["cargo", "build", "--release", "-p", "dolphin-timeseries", "--example", "temporal_covariance_batch"],
        cwd=root,
        check=True,
    )
    process = subprocess.Popen(
        [root / "target/release/examples/temporal_covariance_batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("temporal covariance batch pipes are unavailable")
    scorer = StreamingScores(preregistration)
    records: list[dict] = []
    processed = 0
    emitted = 0
    failed = 0
    total_wall = 0
    peak_rss = 0
    try:
        for request in selected:
            process.stdin.write(json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("temporal covariance batch ended before returning every request")
            record = json.loads(line)
            scorer.update(record)
            processed += 1
            emitted += bool(record["emitted"])
            failed += bool(record["failed"])
            total_wall += record["resource"]["wall_micros"]
            peak_rss = max(
                peak_rss,
                record["resource"]["resident_set_bytes_before"],
                record["resource"]["resident_set_bytes_after"],
            )
            if limit is not None and limit <= 1024:
                records.append(record)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"temporal covariance batch exited with status {return_code}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    exact_denominator = (
        limit is None
        and seed_count == preregistration["outer_seeds_per_supported_cell"]
        and processed == expected_attempts
    )
    scores = scorer.finalize(require_complete=exact_denominator)
    complete_execution = limit is None and processed == expected_attempts
    result_payload = json.dumps({"scores": scores}, separators=(",", ":"), sort_keys=True).encode()
    projected_full_minutes = ((total_wall / processed) * expected_attempts / 60_000_000
                              if processed else float("inf"))
    resource_gates = {
        "rss": peak_rss <= preregistration["resource_limits"]["rss_limit_bytes"],
        "artifact_size": len(result_payload)
            <= preregistration["resource_limits"]["artifact_size_limit_bytes"],
        "projected_wall": projected_full_minutes
            <= preregistration["resource_limits"]["projected_full_scene_minutes"],
    }
    return {
        "schema": "dolphinrust-temporal-covariance-simulation/4",
        "preregistration_schema": preregistration["schema"],
        "pre_outcome_status": preregistration["status"],
        "supported_cell_sha256": preregistration["supported_cell_sha256"],
        "attempted_cells": expected_attempts, "batch_attempted_cells": processed,
        "emitted_cells": emitted,
        "failed_cells": failed,
        "skipped_contract_cells": expected_attempts - processed, "seed_count": seed_count,
        "unsupported_cell_count": len(unsupported_cells(preregistration)),
        "unsupported_cell_sha256": preregistration["unsupported_cell_sha256"],
        "unsupported_cells": unsupported_cells(preregistration),
        "methods": preregistration["methods"], "records": records,
        "scores": scores,
        "execution_paths": preregistration["execution_paths"],
        "corrected_inferential_sigma_emission": False,
        "execution_complete": complete_execution,
        "exact_seed_denominator_complete": exact_denominator,
        "result_records_sha256": hashlib.sha256(result_payload).hexdigest(),
        "result_records_bytes": len(result_payload),
        "promotion_eligible": exact_denominator and scores["all_methods_pass"]
                              and all(resource_gates.values()),
        "promotion_status": ("eligible_for_external_field_review" if exact_denominator
                             and scores["all_methods_pass"] and all(resource_gates.values()) else
                             "blocked_pending_complete_passing_synthetic_execution"),
        "resource": {"total_wall_micros": total_wall, "peak_resident_set_bytes": peak_rss,
                     "result_artifact_bytes": len(result_payload),
                     "projected_full_minutes": projected_full_minutes},
        "resource_gates": resource_gates,
        "resource_limits": preregistration["resource_limits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.seeds <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--seeds and --limit must be positive")
    preregistration = json.loads(args.prereg.read_text())
    receipt = run(preregistration, args.seeds, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
