#!/usr/bin/env python
"""Compare dolphinRust vs the dolphin oracle end-to-end, emit a pass/fail table.

Both engines run the SAME dolphin config; dolphin writes ``timeseries/<ref>_<sec>.tif``
+ ``timeseries/velocity.tif``, dolphinRust writes ``displacement_NN.tif`` + ``velocity.tif``.

Comparison follows PLAYBOOK §Correctness: physical tolerances, not bit-identity. dolphin
masks low-coherence/edge pixels to nodata and references every date to an auto-picked
spatial point; dolphinRust fills those pixels and references only temporally. So we compare
on the common finite mask, after removing a per-date constant (demean) — i.e. up to the
global phase reference the spec allows — and report sign, correlation, RMS and max deviation.

Usage:
  oracle/.venv/bin/python validation/compare.py --oracle <work> --rust <work> --label <tag>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# Stated physical tolerances for an end-to-end cross-implementation comparison
# (two independent pipelines, different eigensolvers + different SNAPHU builds).
CORR_MIN = 0.95          # dominant-signal date pattern agreement
RMS_MAX_RAD = 0.10       # demeaned per-pixel residual (< 0.016 cycle)
VELOCITY_SCALE_TOL = 0.02  # |affine slope - 1| for velocity absolute scale


def read(path: Path) -> np.ndarray:
    ds = gdal.Open(str(path))
    arr = ds.ReadAsArray().astype(np.float64)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = np.nan
    arr[arr == 0.0] = np.nan  # dolphin nodata sentinel for masked pixels
    return arr


def secondary_dates(oracle: Path) -> list[str]:
    ts = sorted((oracle / "timeseries").glob("*_*.tif"))
    ts = [p for p in ts if "velocity" not in p.name and "conncomp" not in p.name]
    return [p.stem.split("_")[1] for p in ts]


def compare_field(o: np.ndarray, r: np.ndarray) -> dict:
    m = np.isfinite(o) & np.isfinite(r)
    oo = o[m] - o[m].mean()
    rr = r[m] - r[m].mean()
    corr = float(np.corrcoef(oo, rr)[0, 1])
    sign = 1.0 if corr >= 0 else -1.0
    resid = rr - sign * oo
    return {
        "n": int(m.sum()),
        "corr": corr,
        "sign": sign,
        "rms": float(np.sqrt(np.mean(resid**2))),
        "max": float(np.max(np.abs(resid))),
        "signal_range": float(np.ptp(oo)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--rust", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    dates = secondary_dates(args.oracle)
    rows = []
    for i, d in enumerate(dates):
        o = read(args.oracle / "timeseries" / f"{dates_ref(args.oracle)}_{d}.tif")
        r = read(args.rust / f"displacement_{i:02d}.tif")
        st = compare_field(o, r)
        st["stage"] = f"displacement[{i}] {d}"
        st["pass"] = st["corr"] >= CORR_MIN and st["rms"] <= RMS_MAX_RAD
        rows.append(st)

    # Velocity absolute scale. Both engines now parse the real acquisition dates, so the
    # rate is a true physical /year. dolphin additionally subtracts a spatial reference
    # pixel (a per-field additive constant), so the honest scale metric is the slope of the
    # affine fit oracle = a*rust + b — a==1 means the absolute scale matches. (A raw median
    # ratio is dominated by the additive offset b and is meaningless here.)
    ov = read(args.oracle / "timeseries" / "velocity.tif")
    rv = read(args.rust / "velocity.tif")
    vst = compare_field(ov, rv)
    vst["stage"] = "velocity"
    m = np.isfinite(ov) & np.isfinite(rv)
    a_mat = np.vstack([rv[m], np.ones(int(m.sum()))]).T
    (slope, offset), *_ = np.linalg.lstsq(a_mat, ov[m], rcond=None)
    vst["velocity_scale_slope"] = float(slope)
    vst["velocity_offset"] = float(offset)
    vst["pass"] = bool(vst["corr"] >= CORR_MIN and abs(float(slope) - 1.0) <= VELOCITY_SCALE_TOL)
    rows.append(vst)

    print(f"\n=== end-to-end comparison: {args.label} ===")
    hdr = f"{'stage':22s} {'n':>5s} {'corr':>8s} {'sign':>5s} {'rms_rad':>9s} {'max_rad':>9s} {'pass':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        print(
            f"{s['stage']:22s} {s['n']:5d} {s['corr']:8.4f} {s['sign']:+5.0f} "
            f"{s['rms']:9.4e} {s['max']:9.4e} {'PASS' if s['pass'] else 'FAIL':>5s}"
        )
    if "velocity_scale_slope" in rows[-1]:
        print(
            f"\nvelocity absolute scale (affine slope oracle=a*rust+b): "
            f"a={rows[-1]['velocity_scale_slope']:.4f} b={rows[-1]['velocity_offset']:.3f}"
        )
    print(
        f"\ntolerances: corr >= {CORR_MIN}, demeaned RMS <= {RMS_MAX_RAD} rad, "
        f"|velocity slope-1| <= {VELOCITY_SCALE_TOL}"
    )

    if args.json_out:
        args.json_out.write_text(json.dumps({"label": args.label, "rows": rows}, indent=2))


def dates_ref(oracle: Path) -> str:
    ts = sorted((oracle / "timeseries").glob("*_*.tif"))
    ts = [p for p in ts if "velocity" not in p.name and "conncomp" not in p.name]
    return ts[0].stem.split("_")[0]


if __name__ == "__main__":
    main()
