#!/usr/bin/env python
"""Issue #142 real before/after: loop-closure off, gate, and gate + integer
correction on the cropped Houston burst of the #192 recipe.

    PY=oracle/.venv/bin/python
    $PY validation/issue192_reference_rule.py plan  --root RUN --env-file .env
    $PY validation/issue192_reference_rule.py fetch --root RUN --env-file .env
    $PY validation/issue142_houston_loop_closure.py run   --root RUN --bin target/release/dolphin
    $PY validation/issue142_houston_loop_closure.py score --root RUN \
        --receipt validation/results/issue142_houston_loop_closure/receipt.json

The three arms share the #192 recipe's engine (bandwidth-2 network, native
unwrapper, `always_first` reference plan) and differ only in
`timeseries_options.mask_unwrap_loop_errors` / `correct_unwrap_loop_errors`.
Scored per arm on its own published grid: the fraction of pixels finite on
every date, the valid pixels in each GNSS station window, and the station-pair
LOS rate error against GNSS (station minus THSU, so THSU cancels in a pair).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

from issue192_reference_rule import (
    RECIPE,
    arm_config,
    load_cube_mm,
    load_recipe,
    ols_rate,
    sha256_path,
    station_series,
)

ARMS = {
    "off": {"mask_unwrap_loop_errors": False, "correct_unwrap_loop_errors": False},
    "gate": {"mask_unwrap_loop_errors": True, "correct_unwrap_loop_errors": False},
    "correct": {"mask_unwrap_loop_errors": True, "correct_unwrap_loop_errors": True},
}


def run(args: argparse.Namespace) -> None:
    from ruamel.yaml import YAML

    recipe = load_recipe()
    planned = json.loads((args.root / "plan.json").read_text())
    files = [args.root / "cslc" / granule["file_name"] for granule in planned["granules"]]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} cropped granules are missing; run fetch first")
    identity = {"binary": str(args.bin), "binary_sha256": sha256_path(args.bin)}
    for arm, flags in ARMS.items():
        work = args.root / f"work142_{arm}"
        config = arm_config(recipe, files, work, recipe["arms"]["new"])
        config["timeseries_options"].update(flags)
        config_path = args.root / f"config142_{arm}.yaml"
        with config_path.open("w") as handle:
            YAML().dump(config, handle)
        started = dt.datetime.now(dt.UTC)
        with (args.root / f"run142_{arm}.log").open("w") as log:
            subprocess.run(
                [str(args.bin), "run", "--config", str(config_path)],
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        identity[arm] = {
            "flags": flags,
            "config_sha256": sha256_path(config_path),
            "elapsed_seconds": (dt.datetime.now(dt.UTC) - started).total_seconds(),
        }
        print(arm, identity[arm], flush=True)
    (args.root / "run142_identity.json").write_text(json.dumps(identity, indent=2) + "\n")


def score(args: argparse.Namespace) -> None:
    recipe = load_recipe()
    dates = [dt.date.fromisoformat(value) for value in recipe["expected_dates"]]
    years = np.array([(date - dates[0]).days / 365.25 for date in dates])
    arms = {}
    for arm in ARMS:
        cube, transform, _ = load_cube_mm(args.root / f"work142_{arm}", len(dates))
        valid = np.isfinite(cube).all(axis=0)
        stations = {
            name: station_series(cube, valid, transform, station, recipe)
            for name, station in recipe["stations"].items()
        }
        pairs = {}
        for a, b in itertools.combinations(sorted(stations), 2):
            if stations[a][0] is None or stations[b][0] is None:
                continue
            insar = float(ols_rate(years, stations[a][0] - stations[b][0]))
            gnss = (
                recipe["stations"][a]["gnss_los_rate_vs_thsu_mm_yr"]
                - recipe["stations"][b]["gnss_los_rate_vs_thsu_mm_yr"]
            )
            pairs[f"{a}-{b}"] = {"insar_mm_yr": insar, "gnss_mm_yr": gnss, "error_mm_yr": insar - gnss}
        arms[arm] = {
            "valid_fraction": float(valid.mean()),
            "valid_pixels": int(valid.sum()),
            "station_pixels": {name: value[1] for name, value in stations.items()},
            "gnss_pairs": pairs,
            "gnss_mean_abs_mm_yr": (
                float(np.mean([abs(p["error_mm_yr"]) for p in pairs.values()])) if pairs else None
            ),
        }
    gate, correct = arms["gate"]["valid_pixels"], arms["correct"]["valid_pixels"]
    dropped = arms["off"]["valid_pixels"] - gate
    receipt = {
        "schema": "dolphinrust.issue142-houston-loop-closure-receipt/1",
        "recipe_sha256": sha256_path(RECIPE),
        "scorer_sha256": sha256_path(Path(__file__)),
        "run_identity": json.loads((args.root / "run142_identity.json").read_text()),
        "dates": len(dates),
        "arms": arms,
        "recovered_fraction_of_gate_drop": (correct - gate) / dropped if dropped > 0 else None,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--bin", type=Path, required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--root", type=Path, required=True)
    score_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    {"run": run, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
