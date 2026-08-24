#!/usr/bin/env python3
"""Generate frozen #53 cells and score them through the Rust batch target."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import tempfile
from pathlib import Path


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


def cells(preregistration: dict) -> list[dict]:
    result = []
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
    for index, cell in enumerate(result):
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
    return result


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


def request_for(cell: dict, seed: int, preregistration: dict, execution_path: str) -> dict:
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
               "cell_index": cell["cell_index"], "seed": seed, "days": days,
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


def score_records(records: list[dict], preregistration: dict) -> dict:
    truth = 0.01 * 365.25
    method_fields = {
        "ols": "ols", "oracle_gls": "oracle_gls",
        "legacy_intercept_slope_wls_non_comparable": "conditional_wls",
        "lag_one_scalar_effective_n": "scalar_effective_n",
        "plugin_gls_reml": "plugin_gls",
        "reml_covariance_parameter_adjusted_scalar": "adjusted_scalar",
        "slope_profile_likelihood_ml": "adjusted_profile",
        "complete_refit_bootstrap": "complete_refit_bootstrap",
    }
    scores = {}
    for method, field in method_fields.items():
        rows = []
        for record in records:
            comparator = record.get("fit", {}).get(field) if record.get("fit") else None
            if not comparator or comparator["point_estimate"] is None:
                continue
            row = {"cell_id": record["cell_id"], "seed": record["seed"],
                   "execution_path": record["execution_path"],
                   "bias": comparator["point_estimate"] - truth,
                   "status": comparator["status"]}
            for label, level in (("68", 0.68), ("90", 0.90), ("95", 0.95)):
                interval = comparator[f"interval_{label}"]
                if interval is None:
                    row[f"coverage_{label}"] = None
                    row[f"width_{label}"] = None
                    row[f"interval_score_{label}"] = None
                    continue
                lower, upper = interval["lower"], interval["upper"]
                alpha = 1.0 - level
                penalty = (2.0 / alpha) * max(lower - truth, 0.0)
                penalty += (2.0 / alpha) * max(truth - upper, 0.0)
                row[f"coverage_{label}"] = lower <= truth <= upper
                row[f"width_{label}"] = upper - lower
                row[f"interval_score_{label}"] = upper - lower + penalty
            rows.append(row)
        attempted = len(records)
        biases = [row["bias"] for row in rows]
        bias_mean = sum(biases) / len(biases) if biases else None
        bias_sd = None
        if len(biases) > 1:
            bias_sd = math.sqrt(sum((value - bias_mean) ** 2 for value in biases)
                                / (len(biases) - 1))
        aggregate = {
            "attempted": attempted,
            "scored": len(rows),
            "failed": attempted - len(rows),
            "emission_fraction": len(rows) / attempted if attempted else 0.0,
            "mean_bias": bias_mean,
            "standardized_bias": (abs(bias_mean) / bias_sd
                                  if bias_sd and bias_sd > 0.0 else None),
        }
        gates = {
            "emission": aggregate["emission_fraction"]
                        >= preregistration["thresholds"]["minimum_successful_emission_fraction"],
            "standardized_bias": aggregate["standardized_bias"] is not None
                and aggregate["standardized_bias"]
                <= preregistration["thresholds"]["standardized_bias"],
        }
        for label in ("68", "90", "95"):
            covered = [row[f"coverage_{label}"] for row in rows
                       if row[f"coverage_{label}"] is not None]
            widths = [row[f"width_{label}"] for row in rows
                      if row[f"width_{label}"] is not None]
            interval_scores = [row[f"interval_score_{label}"] for row in rows
                               if row[f"interval_score_{label}"] is not None]
            coverage = sum(covered) / len(covered) if covered else None
            aggregate[f"coverage_{label}"] = coverage
            aggregate[f"mean_width_{label}"] = (sum(widths) / len(widths)
                                                   if widths else None)
            aggregate[f"mean_interval_score_{label}"] = (
                sum(interval_scores) / len(interval_scores) if interval_scores else None)
            nominal = {"68": 0.68, "90": 0.90, "95": 0.95}[label]
            gates[f"coverage_{label}"] = coverage is not None and abs(coverage - nominal) <= (
                preregistration["thresholds"]["coverage"][f"0.{label}"])
        scores[method] = {"attempted": attempted, "scored": len(rows), "rows": rows,
                          "aggregate": aggregate, "gates": gates,
                          "passes_all_gates": all(gates.values())}
    oracle = scores["oracle_gls"]["aggregate"]
    oracle_score = oracle["mean_interval_score_95"]
    oracle_width = oracle["mean_width_95"]
    for result in scores.values():
        aggregate = result["aggregate"]
        score = aggregate["mean_interval_score_95"]
        width = aggregate["mean_width_95"]
        result["gates"]["proper_score"] = (score is not None and oracle_score is not None
            and score <= oracle_score * (1.0 + preregistration["thresholds"]["proper_score"]))
        result["gates"]["interval_width"] = (width is not None and oracle_width is not None
            and width <= oracle_width * preregistration["thresholds"]["maximum_interval_width_ratio"])
        result["passes_all_gates"] = all(result["gates"].values())
    promotion_methods = preregistration["promotion_methods"]
    return {"schema": "coverage_bias_interval_score/2", "truth_slope_per_year": truth,
            "methods": scores, "promotion_methods": promotion_methods,
            "all_methods_pass": all(scores[name]["passes_all_gates"]
                                     for name in promotion_methods)}


def run(preregistration: dict, seed_count: int, limit: int | None) -> dict:
    frozen_cells = cells(preregistration)
    if cell_hash(frozen_cells) != preregistration["supported_cell_sha256"]:
        raise RuntimeError("supported cell construction does not match preregistration hash")
    requests = [
        request_for(
            cell,
            splitmix64(splitmix64(preregistration["global_seed"] ^ seed) ^ cell["cell_index"]),
            preregistration,
            execution_path,
        )
        for cell in frozen_cells
        for seed in range(seed_count)
        for execution_path in preregistration["execution_paths"]
    ]
    selected = requests if limit is None else requests[:limit]
    with tempfile.TemporaryDirectory(prefix="dolphinrust-temporal-", dir="/var/tmp") as directory:
        input_path = Path(directory) / "requests.jsonl"
        output_path = Path(directory) / "responses.jsonl"
        with input_path.open("w") as handle:
            for request in selected:
                handle.write(json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n")
        command = ["cargo", "run", "--release", "-p", "dolphin-timeseries", "--example", "temporal_covariance_batch"]
        with input_path.open() as input_handle, output_path.open("w") as output_handle:
            subprocess.run(command, cwd=Path(__file__).parents[1], stdin=input_handle, stdout=output_handle, check=True)
        records = [json.loads(line) for line in output_path.read_text().splitlines() if line]
    scores = score_records(records, preregistration)
    complete_execution = limit is None and len(records) == len(requests)
    result_payload = json.dumps({"records": records, "scores": scores},
                                separators=(",", ":"), sort_keys=True).encode()
    total_wall = sum(record["resource"]["wall_micros"] for record in records)
    peak_rss = max((max(record["resource"]["resident_set_bytes_before"],
                        record["resource"]["resident_set_bytes_after"])
                    for record in records), default=0)
    projected_full_minutes = ((total_wall / len(records)) * len(requests) / 60_000_000
                              if records else float("inf"))
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
        "attempted_cells": len(requests), "batch_attempted_cells": len(records),
        "emitted_cells": sum(record["emitted"] for record in records),
        "failed_cells": sum(record["failed"] for record in records),
        "skipped_contract_cells": len(requests) - len(records), "seed_count": seed_count,
        "unsupported_cell_count": len(unsupported_cells(preregistration)),
        "unsupported_cell_sha256": preregistration["unsupported_cell_sha256"],
        "unsupported_cells": unsupported_cells(preregistration),
        "methods": preregistration["methods"], "records": records,
        "scores": scores,
        "execution_paths": preregistration["execution_paths"],
        "corrected_inferential_sigma_emission": False,
        "execution_complete": complete_execution,
        "result_records_sha256": hashlib.sha256(result_payload).hexdigest(),
        "result_records_bytes": len(result_payload),
        "promotion_eligible": complete_execution and scores["all_methods_pass"]
                              and all(resource_gates.values()),
        "promotion_status": ("eligible_for_external_field_review" if complete_execution
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
