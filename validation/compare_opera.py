#!/usr/bin/env python
"""Gate 2a: compare dolphinRust displacement against a RELEASED OPERA L3 DISP-S1 COG.

This is the premier safety oracle for the GroundPulse dolphin flag: dolphinRust's
own output vs the reference implementation's *published* product (not a local
Python re-run). Per docs/project/dolphin-safety-validation-loop.md §2a the compare
is CORRECTION-AWARE: OPERA applies solid-earth-tide / iono / tropo corrections, a
specific reference network and a coherence mask that the default dolphinRust config
does NOT. A constant + planar-ramp offset between the two is therefore EXPECTED and
is NOT a bug; only a wrong deformation PATTERN, wrong SCALE, or wrong SIGN is a real
defect.

Method, on the common high-coherence finite mask over a chosen UTM window:
  * resample the OPERA COG (30 m UTM) onto the dolphinRust output grid (gdal.Warp);
  * solve the affine model  opera = a*rust + bx*x + by*y + c  (least squares):
      - a            -> velocity/displacement SCALE         (target |a-1| < 0.05)
      - sign(a)      -> sign agreement
      - (bx,by,c)    -> the expected missing-correction plane (documented, not failed)
  * detrend BOTH fields by their own best-fit plane and report the spatial pattern
    correlation of the residual deformation field (target corr > 0.9 on a deforming
    window) plus the post-fit residual RMS.

Usage:
  oracle/.venv/bin/python validation/compare_opera.py \
      --rust validation/runs/<tag>/work_rust \
      --opera-cog /vsis3/.../F07079_..._20240429..._.tif \
      --rust-index 19 --label F07079_20240429 --json-out result_opera.json

`--rust-index` selects the dolphinRust displacement_NN.tif whose secondary date
matches the OPERA COG's secondary date. `--opera-cog` may be a local path or a
/vsis3/ URL (set AWS_PROFILE=groundpulse).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# Gate 2a tolerances (deformation pattern + absolute scale; offset/ramp are expected).
CORR_MIN = 0.90          # spatial pattern agreement on the deforming window
SCALE_TOL = 0.05         # |affine slope a - 1| for absolute displacement scale


def _to_nan(arr: np.ndarray, nodata) -> np.ndarray:
    arr = arr.astype(np.float64)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def read_rust(path: Path):
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    arr = _to_nan(ds.ReadAsArray(), band.GetNoDataValue())
    arr[arr == 0.0] = np.nan  # dolphin masked-pixel sentinel
    gt = ds.GetGeoTransform()
    return arr, gt, ds.GetProjection(), ds.RasterXSize, ds.RasterYSize


def opera_on_rust_grid(opera_cog: str, gt, proj, nx, ny) -> np.ndarray:
    """Warp the OPERA COG onto the exact dolphinRust output grid (bilinear)."""
    xmin, xres, _, ymax, _, yres = gt
    xmax = xmin + xres * nx
    ymin = ymax + yres * ny
    warped = gdal.Warp(
        "", opera_cog, format="MEM", dstSRS=proj,
        outputBounds=(xmin, min(ymin, ymax), xmax, max(ymin, ymax)),
        width=nx, height=ny, resampleAlg="bilinear", dstNodata=float("nan"),
    )
    arr = _to_nan(warped.ReadAsArray(), warped.GetRasterBand(1).GetNoDataValue())
    return arr


def affine_fit(opera: np.ndarray, rust: np.ndarray, gt):
    """opera = a*rust + bx*x + by*y + c  on the common finite mask."""
    ny, nx = rust.shape
    xs = gt[0] + gt[1] * (np.arange(nx) + 0.5)
    ys = gt[3] + gt[5] * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(xs - xs.mean(), ys - ys.mean())  # center to condition the fit
    m = np.isfinite(opera) & np.isfinite(rust)
    n = int(m.sum())
    if n < 200:
        return None, m, n
    A = np.vstack([rust[m], X[m], Y[m], np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, opera[m], rcond=None)
    a, bx, by, c = (float(v) for v in coef)
    # Detrend each field by its OWN best-fit plane, then correlate the residual pattern.
    def detrend(field):
        Ap = np.vstack([X[m], Y[m], np.ones(n)]).T
        cp, *_ = np.linalg.lstsq(Ap, field[m], rcond=None)
        return field[m] - Ap @ cp
    od, rd = detrend(opera), detrend(rust)
    corr = float(np.corrcoef(od, rd)[0, 1])
    resid = opera[m] - A @ coef
    return {
        "n": n, "scale_a": a, "ramp_bx": bx, "ramp_by": by, "offset_c": c,
        "sign": 1.0 if a >= 0 else -1.0,
        "pattern_corr": corr,
        "resid_rms_m": float(np.sqrt(np.mean(resid**2))),
        "opera_signal_range_m": float(np.ptp(opera[m])),
        "rust_signal_range_m": float(np.ptp(rust[m])),
    }, m, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", type=Path, required=True, help="dolphinRust work dir")
    ap.add_argument("--opera-cog", required=True, help="released OPERA COG path or /vsis3/ URL")
    ap.add_argument("--rust-index", type=int, required=True,
                    help="displacement_NN.tif index matching the OPERA secondary date")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    rust_path = args.rust / f"displacement_{args.rust_index:02d}.tif"
    if not rust_path.exists():
        raise SystemExit(f"missing {rust_path}")
    rust, gt, proj, nx, ny = read_rust(rust_path)
    opera = opera_on_rust_grid(args.opera_cog, gt, proj, nx, ny)

    stat, mask, n = affine_fit(opera, rust, gt)
    if stat is None:
        raise SystemExit(f"too few co-finite pixels ({n}) — window not co-located or all-masked")

    verdict_pattern = stat["pattern_corr"] >= CORR_MIN
    verdict_scale = abs(stat["scale_a"] - 1.0) <= SCALE_TOL
    passed = bool(verdict_pattern and verdict_scale)

    print(f"\n=== Gate 2a: dolphinRust vs released OPERA COG — {args.label} ===")
    print(f"co-finite pixels        : {stat['n']}")
    print(f"OPERA signal range (m)  : {stat['opera_signal_range_m']:.4f}")
    print(f"rust  signal range (m)  : {stat['rust_signal_range_m']:.4f}")
    print(f"pattern corr (detrended): {stat['pattern_corr']:.4f}   (target > {CORR_MIN})  {'PASS' if verdict_pattern else 'FAIL'}")
    print(f"scale a (opera=a*rust+..): {stat['scale_a']:.4f}    (|a-1|<{SCALE_TOL})    {'PASS' if verdict_scale else 'FAIL'}")
    print(f"sign                    : {stat['sign']:+.0f}")
    print(f"expected ramp/offset    : bx={stat['ramp_bx']:.2e}/m by={stat['ramp_by']:.2e}/m c={stat['offset_c']:.4f} m  (missing-correction delta, documented)")
    print(f"post-fit residual RMS(m): {stat['resid_rms_m']:.4f}")
    print(f"\nGATE 2a: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"label": args.label, "pass": passed,
             "pattern_pass": verdict_pattern, "scale_pass": verdict_scale, **stat}, indent=2))


if __name__ == "__main__":
    main()
