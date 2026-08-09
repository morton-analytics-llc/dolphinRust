#!/usr/bin/env python
"""Leave-one-burst-out scale-calibration gate for GNSS validation receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


Z90 = 1.6448536269514722
NOMINAL = 0.90


def interval_metrics(residual: np.ndarray, sigma: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    finite = np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(finite):
        return {
            "evaluated": 0,
            "abstained": int(residual.size),
            "coverage": None,
            "coverage_error": None,
            "mean_width_mm": None,
            "mean_interval_score": None,
        }
    observed = residual[finite]
    half_width = Z90 * sigma[finite]
    lower, upper = -half_width, half_width
    alpha = 1.0 - NOMINAL
    score = (
        upper
        - lower
        + (2.0 / alpha) * (lower - observed) * (observed < lower)
        + (2.0 / alpha) * (observed - upper) * (observed > upper)
    )
    coverage = float(np.mean(np.abs(observed) <= half_width))
    return {
        "evaluated": int(np.sum(finite)),
        "abstained": int(residual.size - np.sum(finite)),
        "coverage": coverage,
        "coverage_error": abs(coverage - NOMINAL),
        "mean_width_mm": float(np.mean(upper - lower)),
        "mean_interval_score": float(np.mean(score)),
    }


def load_site(path: Path, engine: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") not in {"pass", "fail"}:
        raise ValueError(f"site receipt is not scientifically evaluable: {path}")
    data = payload.get("engines", {}).get(engine)
    if data is None:
        raise ValueError(f"site receipt has no {engine} engine: {path}")
    residual = np.asarray(data["insar_diff_mm"], dtype=float) - np.asarray(
        payload["gnss_diff_mm"], dtype=float
    )
    methods = {
        name: np.asarray(method["sigma_mm"], dtype=float)
        for name, method in data["uncertainty_reliability"].items()
        if name in {"crlb_only", "posterior_only"}
    }
    if set(methods) != {"crlb_only", "posterior_only"}:
        raise ValueError(f"site receipt lacks independent uncertainty alternatives: {path}")
    context = payload.get("context", {})
    return {
        "path": str(path),
        "burst_id": context.get("burst_id") or context.get("fixture") or payload["comparison"],
        "comparison": payload["comparison"],
        "residual": residual,
        "methods": methods,
    }


def burst_scale(residual: np.ndarray, sigma: np.ndarray) -> float:
    finite = np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0)
    ratios = np.abs(residual[finite]) / (Z90 * sigma[finite])
    if ratios.size < 3:
        raise ValueError("a burst needs at least three finite non-reference epochs")
    return float(np.quantile(ratios, NOMINAL, method="higher"))


def score(paths: Sequence[Path], engine: str) -> dict[str, Any]:
    sites = [load_site(path, engine) for path in paths]
    bursts = [site["burst_id"] for site in sites]
    if len(set(bursts)) != len(bursts):
        raise ValueError("cohort receipts must represent distinct held-out bursts")
    site_count = len(sites)
    if site_count < 5:
        return {
            "schema": "dolphinrust-gnss-cohort-score/1",
            "status": "not_evaluable",
            "engine": engine,
            "site_count": site_count,
            "nominal_interval": NOMINAL,
            "folds": [],
            "direction_gate": {
                "minimum_distinct_bursts": {
                    "required": 5,
                    "observed": site_count,
                    "pass": False,
                },
                "mean_interval_score_improves": False,
                "coverage_error_improves_on_80_percent": False,
                "invalid_states_abstain": {
                    "observed_abstentions": None,
                    "pass": False,
                    "note": "not evaluated before the five-burst minimum is met",
                },
            },
            "limitations": [
                "proof-of-direction scale calibration requires five distinct bursts",
                "epochs within a burst remain temporally correlated",
                "burst-level factors receive equal weight",
            ],
        }
    folds: list[dict[str, Any]] = []
    for held_out in sites:
        training = [site for site in sites if site is not held_out]
        training_scores = {
            method: float(
                np.mean(
                    [
                        interval_metrics(site["residual"], site["methods"][method])[
                            "mean_interval_score"
                        ]
                        for site in training
                    ]
                )
            )
            for method in ["crlb_only", "posterior_only"]
        }
        base_method = min(training_scores, key=training_scores.get)
        scales = [
            burst_scale(site["residual"], site["methods"][base_method])
            for site in training
        ]
        scale = float(np.median(scales))
        baselines = {
            method: interval_metrics(held_out["residual"], sigma)
            for method, sigma in held_out["methods"].items()
        }
        calibrated = interval_metrics(
            held_out["residual"], held_out["methods"][base_method] * scale
        )
        best_baseline_method = min(
            baselines,
            key=lambda method: baselines[method]["mean_interval_score"],
        )
        best_baseline = baselines[best_baseline_method]
        folds.append(
            {
                "held_out_burst": held_out["burst_id"],
                "comparison": held_out["comparison"],
                "training_base_method": base_method,
                "training_burst_scales": scales,
                "applied_scale": scale,
                "best_held_out_baseline": best_baseline_method,
                "baseline": best_baseline,
                "calibrated": calibrated,
                "improves_interval_score": calibrated["mean_interval_score"]
                < best_baseline["mean_interval_score"],
                "reduces_coverage_error": calibrated["coverage_error"]
                < best_baseline["coverage_error"],
            }
        )
    interval_score_pass = float(
        np.mean([fold["calibrated"]["mean_interval_score"] for fold in folds])
    ) < float(np.mean([fold["baseline"]["mean_interval_score"] for fold in folds]))
    coverage_pass = sum(
        fold["reduces_coverage_error"] for fold in folds
    ) >= math.ceil(0.8 * site_count)
    invalid_count = sum(
        fold["calibrated"]["abstained"] for fold in folds
    )
    return {
        "schema": "dolphinrust-gnss-cohort-score/1",
        "status": "pass" if interval_score_pass and coverage_pass else "fail",
        "engine": engine,
        "site_count": site_count,
        "nominal_interval": NOMINAL,
        "folds": folds,
        "direction_gate": {
            "minimum_distinct_bursts": {"required": 5, "observed": site_count, "pass": True},
            "mean_interval_score_improves": interval_score_pass,
            "coverage_error_improves_on_80_percent": coverage_pass,
            "invalid_states_abstain": {
                "observed_abstentions": invalid_count,
                "pass": True,
                "note": "receipt-level non-finite or non-positive sigma values are excluded, never converted to finite intervals",
            },
        },
        "limitations": [
            "proof-of-direction scale calibration, not a Phase I uncertainty model",
            "epochs within a burst remain temporally correlated",
            "burst-level factors receive equal weight",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--engine", default="native")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = score(args.receipts, args.engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
    if payload["status"] == "not_evaluable":
        raise SystemExit(2)
    if payload["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
