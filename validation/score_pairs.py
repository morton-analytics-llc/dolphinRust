#!/usr/bin/env python
"""Score every station pair in a cohort against one already-computed run.

`run_gps_ground_truth.py --score` evaluates the single pair named in the recipe's
`comparison` block. Issue #35's design point is that this is the wrong unit for a
calibration cohort: requiring all N stations on one common date collapses the 2018
window to 17 usable dates, whereas scoring each pair over *its own* supported dates
keeps far more. So this driver walks every pair and re-uses
`gps_ground_truth.score_common_frame` unchanged — the pipeline output is shared,
only the station sampling differs, so no interferogram is recomputed.

**N stations give N-1 independent differential series, not C(N,2).** Every other
pair is a linear combination. The per-pair table is a coverage *survey*, not a
count of independent evidence, and the summary here says so rather than letting a
reader multiply.

    oracle/.venv/bin/python validation/score_pairs.py \\
      --recipe validation/gps_mmx1_2018.json --fixture mmx1_2018_common
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gps_ground_truth as gps  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pair_recipe(recipe: dict[str, Any], primary: str, control: str) -> dict[str, Any]:
    """The same recipe with its declared comparison pointed at one pair."""
    variant = copy.deepcopy(recipe)
    variant["comparison"] = {
        "id": f"{primary}_minus_{control}",
        "fixture": recipe["comparison"]["fixture"],
        "primary_station": primary,
        "control_station": control,
    }
    return variant


def summarize(entry: dict[str, Any]) -> dict[str, Any]:
    """Pull the comparable numbers out of one pair's payload, tolerating shape drift."""
    metrics = entry.get("metrics", entry)
    picked = {}
    for key in (
        "rmse_mm",
        "correlation",
        "tls_slope",
        "insar_velocity_mm_yr",
        "gnss_velocity_mm_yr",
        "velocity_difference_mm_yr",
        "evaluated_epochs",
    ):
        if isinstance(metrics, dict) and key in metrics:
            picked[key] = metrics[key]
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--engine", default="native")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    recipe = load_json(args.recipe)
    cohort = recipe["cohort_id"]
    run_root = args.run_root or ROOT / "runs" / cohort / args.fixture
    fixture_root = ROOT / "real_data" / cohort / "cropped" / args.fixture
    fixture_manifest = load_json(fixture_root / "fixture_manifest.json")
    config = gps.load_yaml(run_root / f"config_{args.engine}.yaml") if hasattr(gps, "load_yaml") else None
    if config is None:
        import yaml

        config = yaml.safe_load((run_root / f"config_{args.engine}.yaml").read_text())
    work = {args.engine: Path(config["work_directory"])}
    static_files = sorted((fixture_root / "static").glob("OPERA_L2_CSLC-S1-STATIC_*.h5"))
    primary_static = next(
        path for path in static_files if recipe["burst_filename_id"] in path.name
    )
    wavelength = float(config["input_options"]["wavelength"])
    cache = ROOT / "real_data" / cohort / "gnss"
    cache.mkdir(parents=True, exist_ok=True)

    stations = sorted(recipe["stations"])
    results: dict[str, Any] = {}
    for primary, control in itertools.combinations(stations, 2):
        label = f"{primary}-{control}"
        out = run_root / "pairs" / label
        out.mkdir(parents=True, exist_ok=True)
        try:
            payload = gps.score_common_frame(
                pair_recipe(recipe, primary, control),
                fixture_manifest,
                work,
                primary_static,
                cache,
                out,
                wavelength,
            )
            results[label] = {"status": "scored", **summarize(payload)}
        except gps.NotEvaluable as error:
            results[label] = {"status": "not_evaluable", "reason": str(error)}
        except Exception as error:  # noqa: BLE001 - one bad pair must not sink the survey
            results[label] = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
        print(f"{label:12} {results[label]['status']:14} {results[label].get('rmse_mm', '')}")

    scored = [v for v in results.values() if v["status"] == "scored"]
    summary = {
        "cohort_id": cohort,
        "fixture": args.fixture,
        "engine": args.engine,
        "stations": stations,
        "pairs_attempted": len(results),
        "pairs_scored": len(scored),
        # The honest ceiling from issue #35: pairs are not independent evidence.
        "independent_differential_series": len(stations) - 1,
        "pairs": results,
    }
    destination = args.output or run_root / "pair_scores.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        f"\n{len(scored)}/{len(results)} pairs scored across {len(stations)} stations "
        f"({len(stations) - 1} independent differential series) -> {destination}"
    )


if __name__ == "__main__":
    main()
