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
                                result.append({
                                    "date_count": count,
                                    "rho_at_12_days": rho,
                                    "cadence": cadence["name"],
                                    "missingness": missingness,
                                    "variance_ratio": variance_ratio,
                                    "variance_arrangement": arrangement,
                                    "reference_contribution_ratio": reference_ratio,
                                })
    for index, cell in enumerate(result):
        cell["cell_index"] = index
        cell["cell_id"] = "c%04d-%02d-%s-%s-v%s-%s-r%s" % (
            index,
            cell["date_count"],
            cell["cadence"].replace("_", "-"),
            cell["missingness"].replace("_", "-"),
            cell["variance_ratio"],
            cell["variance_arrangement"][:3],
            cell["reference_contribution_ratio"],
        )
    return result


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
    ar_state = 0.0
    state = seed
    for index, day in enumerate(days):
        state, innovation = normal_noise(state)
        rho = cell["rho_at_12_days"]
        gap = 12.0 if index == 0 else days[index] - days[index - 1]
        phi = rho ** (gap / 12.0)
        ar_state = phi * ar_state + math.sqrt(max(0.0, 1.0 - phi * phi)) * innovation
        state, measurement = normal_noise(state)
        shape = math.sqrt(diagonal[index] / geometric_mean)
        value = 0.01 * day + math.sqrt(process_variance) * shape * ar_state
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
        "minimum_gap_days": preregistration["supported_cell_predicate"]["minimum_gap_days"],
        "maximum_gap_days": preregistration["supported_cell_predicate"]["maximum_gap_days"],
    }
    request = {"execution_path": execution_path, "cell_id": cell["cell_id"],
               "cell_index": cell["cell_index"], "seed": seed, "days": days,
               "options": options, "fixed_factor": None, "production_path": None}
    if execution_path == "fixed_factor":
        request["fixed_factor"] = {"observations": observations,
                                   "difference_covariance": covariance}
    else:
        values = [0.0 if value is None else value for value in observations]
        factor = [[0.0 for _ in days] for _ in days]
        issue52_factor = [[0.0 for _ in days] for _ in days]
        for index in range(1, len(days)):
            factor[index][index] = math.sqrt(diagonal[index])
            issue52_factor[index][index] = math.sqrt(diagonal[index] / 2.0)
        request["production_path"] = {
            "raw_complex_seed": seed, "issue52_seed": seed, "issue54_seed": seed,
            "target_raw_complex": [[math.cos(value), math.sin(value)] for value in values],
            "reference_raw_complex": [[1.0, 0.0] for _ in values],
            "validity": [value is not None for value in observations],
            "issue52_target_factor": issue52_factor,
            "issue52_reference_factor": issue52_factor,
            "issue54_difference_factor": factor,
            "provenance": {
                "issue52_receipt_sha256": "52" * 32,
                "issue54_receipt_sha256": "54" * 32,
                "reference_geometry": "synthetic_same_frame_reference",
                "reference_window": "synthetic_window_0",
                "overlap_fraction": 1.0,
                "distance_pixels": 10.0,
                "scope": "synthetic_validation",
                "approximation": None,
                "validation_receipt_sha256": "53" * 32,
            },
        }
    return request


def score_records(records: list[dict]) -> dict:
    truth = 0.01 * 365.25
    method_fields = {
        "ols": "ols", "oracle_gls": "oracle_gls",
        "legacy_intercept_slope_wls_non_comparable": "conditional_wls",
        "lag_one_scalar_effective_n": "scalar_effective_n",
        "plugin_gls_ml": "plugin_gls", "slope_profile_likelihood": "adjusted_profile",
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
        scores[method] = {"attempted": len(records), "scored": len(rows), "rows": rows}
    return {"schema": "coverage_bias_interval_score/1", "truth_slope_per_year": truth,
            "methods": scores}


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
    return {
        "schema": "dolphinrust-temporal-covariance-simulation/3",
        "preregistration_schema": preregistration["schema"],
        "pre_outcome_status": preregistration["status"],
        "supported_cell_sha256": preregistration["supported_cell_sha256"],
        "attempted_cells": len(requests), "batch_attempted_cells": len(records),
        "emitted_cells": sum(record["emitted"] for record in records),
        "failed_cells": sum(record["failed"] for record in records),
        "skipped_contract_cells": len(requests) - len(records), "seed_count": seed_count,
        "methods": preregistration["methods"], "records": records,
        "scores": score_records(records),
        "execution_paths": preregistration["execution_paths"],
        "corrected_inferential_sigma_emission": False,
        "promotion_status": "blocked_pending_synthetic_field_review_and_manifest",
        "resource_fields": preregistration["resource_limits"],
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
