#!/usr/bin/env python
"""Velocity absolute-scale check between dolphinRust and the dolphin oracle.

The end-to-end `compare.py` reports an OLS slope of `oracle = a*rust + b`. OLS is
biased toward zero when the regressor (rust) carries its own noise (classic
errors-in-variables attenuation): with cross-engine noise present, OLS understates
the true scale. This tool adds the **total-least-squares (orthogonal) slope**,
which is symmetric in the two engines and unbiased under comparable noise — the
honest absolute-scale metric — plus coherence-gated variants and the magnitude in
mm/yr. Use it to confirm rust tracks the oracle at real magnitude on a deforming
scene.

    oracle/.venv/bin/python validation/velocity_scale.py --run validation/runs/real_mexico_T005
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

SENTINEL1_WAVELENGTH_M = 0.05546576


def read(path: Path) -> np.ndarray:
    arr = gdal.Open(str(path)).ReadAsArray().astype(np.float64)
    arr[arr == 0.0] = np.nan
    return arr


def tls_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Orthogonal-regression slope of y on x (first principal axis)."""
    xc, yc = x - x.mean(), y - y.mean()
    m = np.vstack([xc, yc])
    _, _, vt = np.linalg.svd(m @ m.T)
    return float(vt[0, 1] / vt[0, 0])


def coherence_file(work_oracle: Path) -> Path | None:
    hits = sorted(work_oracle.glob("linked_phase/temporal_coherence_average_*.tif"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--wavelength", type=float, default=SENTINEL1_WAVELENGTH_M)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    ov = read(args.run / "work_oracle" / "timeseries" / "velocity.tif")
    rv = read(args.run / "work_rust" / "velocity.tif")
    cf = coherence_file(args.run / "work_oracle")
    oc = read(cf) if cf else np.ones_like(ov)
    mm = abs(-args.wavelength / (4.0 * np.pi) * 1000.0)  # rad/yr -> mm/yr

    rows = []
    for thr in (0.0, 0.6, 0.8, 0.9):
        m = np.isfinite(ov) & np.isfinite(rv) & (oc > thr)
        if int(m.sum()) < 300:
            continue
        o, r = ov[m], rv[m]
        rows.append({
            "coh_gate": thr,
            "n": int(m.sum()),
            "corr": float(np.corrcoef(o, r)[0, 1]),
            "ols_slope": float(np.polyfit(r, o, 1)[0]),
            "tls_slope": tls_slope(r, o),
            "oracle_std_mm_yr": float(o.std() * mm),
            "rust_std_mm_yr": float(r.std() * mm),
        })

    print(f"{'coh>':>5s} {'n':>7s} {'corr':>6s} {'OLS':>6s} {'TLS':>6s} {'o_mm/yr':>8s} {'r_mm/yr':>8s}")
    for s in rows:
        print(f"{s['coh_gate']:>5.2f} {s['n']:7d} {s['corr']:6.3f} {s['ols_slope']:6.3f} "
              f"{s['tls_slope']:6.3f} {s['oracle_std_mm_yr']:8.3f} {s['rust_std_mm_yr']:8.3f}")
    print("\nTLS (orthogonal) slope is the attenuation-free absolute-scale metric; "
          "TLS ~= 1 confirms rust tracks the oracle at real magnitude.")

    if args.json_out:
        args.json_out.write_text(json.dumps({"run": str(args.run), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
