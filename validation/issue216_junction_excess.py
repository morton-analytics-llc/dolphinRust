#!/usr/bin/env python
"""Issue #216: why every ministack junction on Houston T034 carries 2.5-5x the
half-cycle local steps of ordinary dates.

Test design (frozen before any arm was run or scored):
`md/research/issue-216-junction-excess-test-design-2026-11.md`. Inputs are the #192 ones
(`issue192_reference_rule.py plan` and `fetch`); the statistic is the #207 local one
(`issue207_junction_steps.local_residuals`).

    PY=oracle/.venv/bin/python
    $PY validation/issue192_reference_rule.py plan  --root RUN
    $PY validation/issue192_reference_rule.py fetch --root RUN
    $PY validation/issue216_junction_excess.py arms  --root RUN --bin target/release/dolphin --only a0
    $PY validation/issue216_junction_excess.py score --root RUN --receipt <out.json>

Arms run one at a time (`--only`), in the design's order, and `score` scores whichever
arms exist. For M5, `issue207_junction_steps.py opera --root RUN` writes the OPERA steps
onto A0's grid, and `score` adds them when present.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from issue207_junction_steps import (  # noqa: E402
    local_residuals,
    network_pairs,
    provenance,
    published_series_cycles,
    recipe,
    sha256_path,
    step_labels,
)

DESIGN = "md/research/issue-216-junction-excess-test-design-2026-11.md"
WORK = "work_new"  # the #192 driver's `always_first` arm
J0 = [15, 30, 45, 60, 75, 90]
HALF = 6
MIN_NEIGHBOURS = 4
ARMS: dict[str, dict[str, Any]] = {
    "a0": {},
    "r": {"phase_linking": {"compressed_slc_plan": "first_per_ministack"}},
    "a1": {"unwrap_options": {"unwrap_method": "snaphu"}},
    "b": {"phase_linking": {"ministack_size": 91}},
    "c": {"phase_linking": {"ministack_size": 13}},
}
# #207 `residual_vs_184_flags` cells beyond half a cycle, and the like-for-like p95.
M0_CELLS = {15: 77, 30: 24, 45: 64, 60: 101, 75: 48, 90: 57}
M0_P95 = 0.00015310277856091207
TILE = 48


def arms(args: argparse.Namespace) -> None:
    for name in args.only:
        arm_root = args.root / name
        arm_root.mkdir(exist_ok=True)
        for shared in ("cslc", "plan.json"):
            link = arm_root / shared
            if not link.exists():
                link.symlink_to((args.root / shared).resolve())
        command = [sys.executable, str(ROOT / "issue192_reference_rule.py"), "run", "--root", str(arm_root), "--bin", str(args.bin)]
        if ARMS[name]:
            command += ["--config-overrides", json.dumps(ARMS[name])]
        subprocess.run(command, check=True)


def residuals(step: np.ndarray, labels: np.ndarray, half: int) -> np.ndarray:
    """`local_residuals` over row tiles with a `half`-row overlap: identical values,
    bounded memory at wide windows."""
    out = np.full(step.shape, np.nan)
    rows = step.shape[0]
    for start in range(0, rows, TILE):
        lo, hi = max(0, start - half), min(rows, start + TILE + half)
        tile = local_residuals(step[lo:hi], labels[lo:hi], half, MIN_NEIGHBOURS)
        out[start : min(rows, start + TILE)] = tile[start - lo : start - lo + min(TILE, rows - start)]
    return out


def arm_counts(work: Path, n_dates: int, halves: list[int]) -> dict[str, Any]:
    """Per step and half-window: cells over half a cycle (`n`) and judged (`e`)."""
    frozen = recipe()
    series = published_series_cycles(work, frozen["engine"]["wavelength_m"], n_dates)
    pairs = network_pairs(n_dates, frozen["engine"]["max_bandwidth"])
    counts: dict[int, dict[int, tuple[int, int]]] = {h: {} for h in halves}
    over: dict[int, np.ndarray] = {}
    steps: dict[int, np.ndarray] = {}
    for k in range(1, n_dates):
        step = series[k] - series[k - 1]
        steps[k] = step
        labels = step_labels(work, pairs, k, np.isfinite(step))
        for h in halves:
            r = residuals(step, labels, h)
            judged = np.isfinite(r)
            beyond = judged & (np.abs(np.where(judged, r, 0.0)) > 0.5)
            counts[h][k] = (int(beyond.sum()), int(judged.sum()))
            if h == HALF:
                over[k] = beyond
    record = provenance(work)
    own = [item["epoch"] for item in record["ministack_junctions"]["junctions"]]
    return {"counts": counts, "own_junctions": own, "over": over, "steps": steps}


def excess(counts: dict[int, tuple[int, int]], at: list[int], ordinary: list[int]) -> dict[str, Any]:
    fractions = np.array([counts[k][0] / counts[k][1] for k in ordinary if counts[k][1]])
    p95 = float(np.percentile(fractions, 95))
    per = {str(k): {"cells": counts[k][0], "judged": counts[k][1], "fraction": counts[k][0] / counts[k][1]} for k in at}
    total = float(sum(counts[k][0] - p95 * counts[k][1] for k in at))
    return {
        "excess_cells": total,
        "ordinary": {"steps": int(fractions.size), "median": float(np.median(fractions)), "p95": p95, "max": float(fractions.max())},
        "at": per,
    }


def ordinary_steps(n_dates: int, own: list[int]) -> list[int]:
    return [k for k in range(1, n_dates) if k not in J0 and k not in own]


def opera_m5(root: Path, a0: dict[str, Any], b: dict[str, Any], n_dates: int, wavelength: float) -> dict[str, Any]:
    """M5: at A0's over-half junction cells, how often A0 (and B) disagree with OPERA
    by more than half a cycle, and OPERA's own local statistic."""
    stacks = [np.load(root / "opera" / f"{index:02d}.npz") for index in range(n_dates)]
    references = [str(stack["reference"]) for stack in stacks]
    cycles = [np.where(stack["recommended_mask"] == 1, stack["displacement_m"], np.nan) / (wavelength / 2.0) for stack in stacks]
    comparable = [k for k in range(1, n_dates) if references[k] == references[k - 1]]
    correlation = {}
    for k in comparable:
        theirs = cycles[k] - cycles[k - 1]
        common = np.isfinite(a0["steps"][k]) & np.isfinite(theirs)
        correlation[k] = float(np.corrcoef(a0["steps"][k][common], theirs[common])[0, 1])
    sign = float(np.sign(np.median([correlation[k] for k in comparable if k not in J0])))

    def disagreement(ours: np.ndarray, theirs: np.ndarray, cells: np.ndarray) -> tuple[int, int]:
        common = np.isfinite(ours) & np.isfinite(theirs)
        difference = ours - theirs
        centred = difference - np.median(difference[common])
        inside = cells & common
        return int((np.abs(centred[inside]) > 0.5).sum()), int(inside.sum())

    per, s_o = {}, {}
    for k in comparable:
        theirs = sign * (cycles[k] - cycles[k - 1])
        finite = np.isfinite(theirs)
        r = residuals(theirs, finite.astype(np.uint32), HALF)
        judged = np.isfinite(r)
        s_o[k] = (int((judged & (np.abs(np.where(judged, r, 0.0)) > 0.5)).sum()), int(judged.sum()))
        if k not in J0:
            continue
        cells = a0["over"][k]
        d_a0 = disagreement(a0["steps"][k], theirs, cells)
        d_b = disagreement(b["steps"][k], theirs, cells) if b else None
        per[str(k)] = {
            "a0_over_half_cells": int(cells.sum()),
            "covered_by_opera": d_a0[1],
            "D_A0": d_a0[0] / d_a0[1] if d_a0[1] else None,
            "D_B": (d_b[0] / d_b[1] if d_b[1] else None) if d_b else None,
            "S_O": s_o[k][0] / s_o[k][1] if s_o[k][1] else None,
        }
    scored = [item for item in per.values() if item["covered_by_opera"]]
    covered = sum(item["covered_by_opera"] for item in scored)
    d_a0_pooled = sum(item["D_A0"] * item["covered_by_opera"] for item in scored) / covered if covered else None
    d_b_pooled = sum(item["D_B"] * item["covered_by_opera"] for item in scored) / covered if covered and b else None
    ordinary_o = [s_o[k][0] / s_o[k][1] for k in comparable if k not in J0 and s_o[k][1]]
    return {
        "comparable_steps": comparable,
        "sign": sign,
        "per_junction": per,
        "pooled": {"covered_cells": covered, "D_A0": d_a0_pooled, "D_B": d_b_pooled},
        "S_O_ordinary": {"median": float(np.median(ordinary_o)), "p95": float(np.percentile(ordinary_o, 95)), "max": float(max(ordinary_o))},
        "correlation_junctions": {str(k): correlation[k] for k in J0 if k in correlation},
    }


def wrap_cycles(values: np.ndarray) -> np.ndarray:
    return values - np.round(values)


def dolphin_m6(root: Path, a0: dict[str, Any], n_dates: int) -> dict[str, Any]:
    """M6: the A0 wrapped step against dolphin v0.42.9's linked-phase step, per cell."""
    import rasterio

    identity = json.loads((root / "dolphin" / "identity.json").read_text())
    linked = []
    for path in identity["linked"]:
        with rasterio.open(path) as dataset:
            linked.append(dataset.read(1).astype(np.complex128))
    if len(linked) != n_dates:
        raise SystemExit(f"dolphin wrote {len(linked)} linked-phase rasters, expected {n_dates}")
    ordinary = [k for k in range(1, n_dates) if k not in J0]

    def difference(k: int, sign: float) -> tuple[np.ndarray, np.ndarray]:
        ours = wrap_cycles(a0["steps"][k])
        theirs = np.angle(linked[k] * np.conj(linked[k - 1])) / (2 * np.pi)
        common = np.isfinite(ours) & (np.abs(linked[k]) > 0) & (np.abs(linked[k - 1]) > 0)
        raw = np.where(common, ours - sign * theirs, 0.0)
        offset = np.angle(np.mean(np.exp(2j * np.pi * raw[common]))) / (2 * np.pi)
        return wrap_cycles(raw - offset), common

    def agreement(sign: float) -> float:
        return float(np.median([np.abs(np.mean(np.exp(2j * np.pi * d[c]))) for d, c in (difference(k, sign) for k in ordinary)]))

    sign = max((1.0, -1.0), key=agreement)
    per, over_all = {}, {}
    for k in range(1, n_dates):
        delta, common = difference(k, sign)
        over_all[k] = float(np.mean(np.abs(delta[common]) > 0.25))
        if k in J0:
            cells = a0["over"][k] & common
            per[str(k)] = {
                "a0_over_half_cells": int(a0["over"][k].sum()),
                "compared": int(cells.sum()),
                "disagree_over_quarter": int((np.abs(delta[cells]) > 0.25).sum()),
                "frame_fraction_over_quarter": over_all[k],
            }
    compared = sum(item["compared"] for item in per.values())
    disagree = sum(item["disagree_over_quarter"] for item in per.values())
    ordinary_fractions = np.array([over_all[k] for k in ordinary])
    pooled = disagree / compared if compared else None
    verdict = None if pooled is None else ("estimator property" if pooled <= 0.2 else "port defect" if pooled >= 0.5 else "between")
    return {
        "dolphin": identity["dolphin"],
        "sign": sign,
        "M6a": {"per_junction": per, "pooled_fraction": pooled, "verdict": verdict},
        "M6b": {
            "junction_frame_fraction": {str(k): over_all[k] for k in J0},
            "ordinary_frame_fraction": {"median": float(np.median(ordinary_fractions)), "p95": float(np.percentile(ordinary_fractions, 95))},
        },
    }


def score(args: argparse.Namespace) -> None:
    frozen = recipe()
    n_dates = len(frozen["expected_dates"])
    present = [name for name in ARMS if (args.root / name / WORK / "geometry_provenance.json").exists()]
    if "a0" not in present:
        raise SystemExit("arm a0 has not been run")
    a0 = arm_counts(args.root / "a0" / WORK, n_dates, [3, HALF, 9, 12])
    ordinary = ordinary_steps(n_dates, a0["own_junctions"])
    window = {str(h): excess(a0["counts"][h], J0, ordinary) for h in (3, HALF, 9, 12)}
    baseline = window[str(HALF)]["excess_cells"]
    half_baseline = 0.5 * baseline

    m0 = {
        "cells": {str(k): {"now": a0["counts"][HALF][k][0], "issue207": M0_CELLS[k]} for k in J0},
        "p95": {"now": window[str(HALF)]["ordinary"]["p95"], "issue207": M0_P95},
    }
    m0["pass"] = all(abs(a0["counts"][HALF][k][0] - M0_CELLS[k]) <= 2 for k in J0) and abs(m0["p95"]["now"] / M0_P95 - 1) <= 0.10

    results: dict[str, Any] = {"a0": {"own_junctions": a0["own_junctions"], "J0": window[str(HALF)]}}
    a0l = arm_counts(args.root / "a0" / "work_old", n_dates, [HALF])
    results["a0l"] = {"own_junctions": a0l["own_junctions"], "J0": excess(a0l["counts"][HALF], J0, ordinary_steps(n_dates, a0l["own_junctions"]))}
    loaded = {"a0": a0}
    for name in present[1:]:
        arm = arm_counts(args.root / name / WORK, n_dates, [HALF])
        loaded[name] = arm
        own = arm["own_junctions"]
        entry = {"own_junctions": own, "J0": excess(arm["counts"][HALF], J0, ordinary_steps(n_dates, own))}
        if own and own != J0:
            entry["own"] = excess(arm["counts"][HALF], own, ordinary_steps(n_dates, own))
        results[name] = entry

    e = lambda name, key="J0": results[name][key]["excess_cells"] if name in results else None  # noqa: E731
    checks = {
        "C1_window": {
            "excess_by_half_window": {h: window[h]["excess_cells"] for h in window},
            "explains": window["9"]["excess_cells"] <= half_baseline and window["12"]["excess_cells"] <= half_baseline,
        },
        "C2_reference": {
            "excess_a0l": e("a0l"),
            "excess_r": e("r"),
            "explains": (e("a0l") <= half_baseline) or (e("r") is not None and e("r") <= half_baseline),
            "run": "r" in results,
        },
        "C3_unwrapper": {
            "excess_a1": e("a1"),
            "explains": e("a1") is not None and e("a1") <= half_baseline,
            "run": "a1" in results,
        },
        "C4_handoff": {
            "excess_b_at_J0": e("b"),
            "excess_c_at_own": e("c", "own") if "c" in results else None,
            "excess_c_at_J0": e("c"),
            "explains": (
                "b" in results and "c" in results
                and e("b") <= half_baseline and e("c", "own") >= half_baseline and e("c") <= half_baseline
            ),
            "run": "b" in results and "c" in results,
        },
    }
    finding = "no finding: M0 reproduction failed" if not m0["pass"] else next(
        (name for name in ("C1_window", "C2_reference", "C3_unwrapper", "C4_handoff") if checks[name]["explains"]),
        "no finding",
    )
    receipt: dict[str, Any] = {
        "schema": "dolphinrust.issue216-junction-excess-receipt/1",
        "design": DESIGN,
        "scorer_sha256": sha256_path(Path(__file__)),
        "recipe_sha256": sha256_path(ROOT / "issue192_houston_recipe.json"),
        "plan": {k: v for k, v in json.loads((args.root / "plan.json").read_text()).items() if k != "granules"},
        "run_identity": {name: json.loads((args.root / name / "run_identity.json").read_text()) for name in present},
        "arm_overrides": {name: ARMS[name] for name in present},
        "statistic": {"half_window": HALF, "min_neighbours": MIN_NEIGHBOURS, "threshold_cycles": 0.5},
        "baseline_excess_cells": baseline,
        "M0_reproduction": m0,
        "arms": results,
        "checks": checks,
        "finding": finding,
    }
    if (args.root / "opera" / f"{n_dates - 1:02d}.npz").exists():
        receipt["M5_opera"] = opera_m5(args.root, a0, loaded.get("b"), n_dates, frozen["engine"]["wavelength_m"])
    if (args.root / "dolphin" / "identity.json").exists():
        receipt["M6_dolphin"] = dolphin_m6(args.root, a0, n_dates)
        print(json.dumps(receipt["M6_dolphin"], indent=2))
    receipt["scored_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("baseline_excess_cells", "M0_reproduction", "checks", "finding")}, indent=2))
    if "M5_opera" in receipt:
        print(json.dumps({k: receipt["M5_opera"][k] for k in ("per_junction", "pooled", "S_O_ordinary")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("arms", arms), ("score", score)):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--root", type=Path, required=True, help="run directory (kept out of git)")
        if name == "arms":
            command.add_argument("--bin", type=Path, required=True)
            command.add_argument("--only", nargs="+", choices=list(ARMS), required=True)
        if name == "score":
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args.handler(args)


if __name__ == "__main__":
    main()
