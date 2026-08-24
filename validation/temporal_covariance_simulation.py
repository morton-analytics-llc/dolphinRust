#!/usr/bin/env python3
"""Deterministic, pre-outcome #53 simulation/batch driver.

This driver freezes cell construction and emits a compact receipt. It is a
stdlib-only reference harness for contract runs; the release experiment must
invoke the Rust release-mode batch target for every estimator and seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
    return result


def xorshift(state: int) -> tuple[int, float]:
    state ^= (state << 13) & ((1 << 64) - 1)
    state ^= state >> 7
    state ^= (state << 17) & ((1 << 64) - 1)
    unit = ((state >> 11) / float(1 << 53))
    return state, (unit - 0.5) * 2.0 * math.sqrt(3.0)


def compact_point_estimates(cell: dict, seed: int) -> dict:
    """Return deterministic point-only evidence for a small contract cell."""
    state = seed | 1
    slope = 0.01
    x = [12.0 * index for index in range(cell["date_count"])]
    y = []
    previous = 0.0
    for index, day in enumerate(x):
        state, innovation = xorshift(state + index + cell["date_count"])
        rho = cell["rho_at_12_days"]
        previous = rho * previous + math.sqrt(max(0.0, 1.0 - rho * rho)) * innovation
        y.append(slope * day + previous)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    ols = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / sum(
        (a - mean_x) ** 2 for a in x
    )
    rho = cell["rho_at_12_days"]
    denominator = max(1.0 - rho * rho, 1e-12)
    inverse = [[0.0 for _ in x] for _ in x]
    for row in range(len(x)):
        inverse[row][row] = (1.0 if row in (0, len(x) - 1) else 1.0 + rho * rho) / denominator
        if row + 1 < len(x):
            inverse[row][row + 1] = -rho / denominator
            inverse[row + 1][row] = -rho / denominator
    precision_x = [sum(inverse[row][column] * x[column] for column in range(len(x))) for row in range(len(x))]
    precision_y = [sum(inverse[row][column] * y[column] for column in range(len(x))) for row in range(len(x))]
    oracle = sum(a * b for a, b in zip(x, precision_y)) / sum(a * b for a, b in zip(x, precision_x))
    return {"ols_slope_per_day": ols, "oracle_gls_slope_per_day": oracle}


def run(preregistration: dict, seed_count: int) -> dict:
    frozen_cells = cells(preregistration)
    emitted = 0
    records = []
    for cell_index, cell in enumerate(frozen_cells):
        for seed in range(seed_count):
            record = compact_point_estimates(cell, seed + 0x53_2026 + cell_index * 100_003)
            record.update({"cell_index": cell_index, "seed": seed, "status": "point_only_contract"})
            records.append(record)
            emitted += 1
    return {
        "schema": "dolphinrust-temporal-covariance-simulation/1",
        "preregistration_schema": preregistration["schema"],
        "pre_outcome_status": preregistration["status"],
        "attempted_cells": len(frozen_cells) * seed_count,
        "emitted_cells": emitted,
        "failed_cells": 0,
        "seed_count": seed_count,
        "methods": preregistration["methods"],
        "corrected_inferential_sigma_emission": False,
        "promotion_status": "blocked_pending_release_rust_batch_coverage_field_review_and_manifest",
        "resource_fields": {
            "hardware_class": None,
            "wall_seconds": None,
            "peak_rss_bytes": None,
            "artifact_bytes": None,
        },
        "cells": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    preregistration = json.loads(args.prereg.read_text())
    receipt = run(preregistration, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
