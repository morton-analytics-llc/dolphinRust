"""Issue #230 acceptance: in-house raw-SLC CSLC against OPERA CSLC-S1.

Both stacks cover the same burst, dates and grid (ours is geocoded onto the
OPERA crop's own grid). OPERA is the backtest reference only.

Registration, per date (same acquisition, so the two should be the same
image up to interpolation):
- median 11x11 windowed complex correlation |gamma| between our CSLC and
  OPERA's, both *unflattened* (each multiplied by exp(-j*flattening_phase),
  its own layer), over pixels whose OPERA consecutive-date interferometric
  coherence (11x11) is >= 0.7 (stable scatterers); pass >= 0.95. A
  flattened CSLC carries 4*pi*R(h)/lambda, ~160 rad per metre of DEM height
  at C band, so two processors whose DEMs differ by centimetres per pixel
  (GLO-30 is a surface model with building-scale relief) decorrelate in the
  flattened product even when their sampling is identical; COMPASS on the
  same burst with a different GLO-30 resampling correlates 0.13 with OPERA
  flattened and 0.987 unflattened. Unflattened, |gamma| measures only where
  each processor sampled the SLC. The flattened value is reported as well;
- amplitude offset tracking on 128x128 patches (FFT cross-correlation,
  16x upsampled peak); pass when |median offset| <= 0.1 px in rows and cols.

Phase, per consecutive date pair (i, i+1): Delta = arg(I_ours * conj(I_opera)),
I = s_i * conj(s_{i+1}). Timing corrections the two processors may apply
differently (ionosphere, solid-earth tide) are smooth over a burst, so a
best-fit plane is removed first; pass when the circular std of Delta is
<= 0.1 rad over pixels with OPERA interferometric coherence (11x11) >= 0.9.

    oracle/.venv/bin/python -I validation/issue230_raw_front_end.py \
        --ours DIR --opera DIR --label T100-213486-IW2 --out results/...json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter

WINDOW = 11
PATCH = 128
UPSAMPLE = 16
DATE = re.compile(r"_(\d{8})T\d{6}Z")


def date_of(path: Path) -> str:
    match = DATE.search(path.name)
    if not match:
        raise ValueError(f"no acquisition date in {path.name}")
    return match.group(1)


def read(path: Path, dataset: str = "data/VV") -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f[dataset][()].astype(np.complex64)


def read_unflattened(path: Path) -> np.ndarray | None:
    """CSLC times exp(-j*flattening_phase), or None without that layer."""
    with h5py.File(path, "r") as f:
        if "data/flattening_phase" not in f:
            return None
        return (f["data/VV"][()] * np.exp(-1j * f["data/flattening_phase"][()])).astype(np.complex64)


def boxsum(values: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(values):
        return uniform_filter(values.real, WINDOW) + 1j * uniform_filter(values.imag, WINDOW)
    return uniform_filter(values, WINDOW)


def windowed_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid = np.isfinite(a) & np.isfinite(b)
    a = np.where(valid, a, 0)
    b = np.where(valid, b, 0)
    num = np.abs(boxsum(a * np.conj(b)))
    den = np.sqrt(boxsum(np.abs(a) ** 2) * boxsum(np.abs(b) ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = num / den
    full = boxsum(valid.astype(np.float64)) > 0.999
    return np.where(full & valid, gamma, np.nan)


def patch_offset(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    """Sub-pixel (row, col) shift of b relative to a from amplitude patches."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    fa = np.abs(a) - np.abs(a).mean()
    fb = np.abs(b) - np.abs(b).mean()
    spectrum = np.fft.fft2(fa) * np.conj(np.fft.fft2(fb))
    n = PATCH * UPSAMPLE
    padded = np.zeros((n, n), dtype=complex)
    half = PATCH // 2
    shifted = np.fft.fftshift(spectrum)
    padded[n // 2 - half : n // 2 + half, n // 2 - half : n // 2 + half] = shifted
    surface = np.abs(np.fft.ifft2(np.fft.ifftshift(padded)))
    r, c = np.unravel_index(np.argmax(surface), surface.shape)
    wrap = lambda v: (v + n // 2) % n - n // 2  # noqa: E731
    return -wrap(r) / UPSAMPLE, -wrap(c) / UPSAMPLE


def offsets(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    found = []
    for r in range(0, a.shape[0] - PATCH + 1, PATCH):
        for c in range(0, a.shape[1] - PATCH + 1, PATCH):
            shift = patch_offset(a[r : r + PATCH, c : c + PATCH], b[r : r + PATCH, c : c + PATCH])
            if shift is not None:
                found.append(shift)
    if not found:
        return float("nan"), float("nan"), 0
    arr = np.array(found)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), len(found)


def remove_plane(delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove a ramp + constant from a wrapped phase field over `mask`."""
    z = np.exp(1j * delta)
    gy = np.angle(np.nansum((z[1:, :] * np.conj(z[:-1, :]))[mask[1:, :] & mask[:-1, :]]))
    gx = np.angle(np.nansum((z[:, 1:] * np.conj(z[:, :-1]))[mask[:, 1:] & mask[:, :-1]]))
    rows, cols = np.indices(delta.shape)
    z = z * np.exp(-1j * (gy * rows + gx * cols))
    z = z * np.exp(-1j * np.angle(np.nansum(z[mask])))
    residual = np.angle(z)
    design = np.column_stack([np.ones(mask.sum()), rows[mask], cols[mask]])
    coef, *_ = np.linalg.lstsq(design, residual[mask], rcond=None)
    return np.angle(np.exp(1j * (residual - (coef[0] + coef[1] * rows + coef[2] * cols))))


def circular_std(values: np.ndarray) -> float:
    resultant = np.abs(np.mean(np.exp(1j * values)))
    return float(np.sqrt(-2.0 * np.log(resultant)))


def pair_files(ours_dir: Path, opera_dir: Path) -> list[tuple[str, Path, Path]]:
    ours = {date_of(p): p for p in ours_dir.glob("CSLC-S1_*.h5")}
    opera = {date_of(p): p for p in opera_dir.glob("OPERA_L2_CSLC-S1_*.h5")}
    return [(d, ours[d], opera[d]) for d in sorted(set(ours) & set(opera))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--opera", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pairs = pair_files(args.ours, args.opera)
    if len(pairs) < 2:
        raise SystemExit(f"need >= 2 common dates, found {len(pairs)}")
    stacks = [(d, read(o), read(r)) for d, o, r in pairs]
    unflat = {d: (read_unflattened(o), read_unflattened(r)) for d, o, r in pairs}
    dates = []
    for i, (day, ours, opera) in enumerate(stacks):
        partner = stacks[i + 1][2] if i + 1 < len(stacks) else stacks[i - 1][2]
        stable = windowed_correlation(opera, partner) >= 0.7
        flattened = windowed_correlation(ours, opera)
        ours_u, opera_u = unflat[day]
        if ours_u is None or opera_u is None:
            raise SystemExit(f"{day}: both stacks need data/flattening_phase for the registration check")
        gamma = windowed_correlation(ours_u, opera_u)
        d_row, d_col, patches = offsets(opera, ours)
        dates.append(
            {
                "date": day,
                "valid_fraction": float(np.isfinite(ours).mean()),
                "median_correlation_stable": float(np.nanmedian(gamma[stable])),
                "median_correlation_all": float(np.nanmedian(gamma)),
                "median_correlation_flattened_stable": float(np.nanmedian(flattened[stable])),
                "stable_pixels": int(np.isfinite(gamma[stable]).sum()),
                "offset_rows_px": d_row,
                "offset_cols_px": d_col,
                "offset_patches": patches,
            }
        )
    phase = []
    for (d1, o1, r1), (d2, o2, r2) in zip(stacks, stacks[1:]):
        ours_ifg = o1 * np.conj(o2)
        opera_ifg = r1 * np.conj(r2)
        coherence = windowed_correlation(r1, r2)
        mask = (coherence >= 0.9) & np.isfinite(ours_ifg) & np.isfinite(opera_ifg)
        if mask.sum() < 100:
            phase.append({"pair": f"{d1}_{d2}", "pixels": int(mask.sum()), "circular_std_rad": None})
            continue
        delta = np.angle(np.where(mask, ours_ifg * np.conj(opera_ifg), 1))
        raw = circular_std(delta[mask])
        flat = remove_plane(delta, mask)
        phase.append(
            {
                "pair": f"{d1}_{d2}",
                "pixels": int(mask.sum()),
                "circular_std_raw_rad": raw,
                "circular_std_rad": circular_std(flat[mask]),
            }
        )
    scored = [p["circular_std_rad"] for p in phase if p["circular_std_rad"] is not None]
    summary = {
        "dates": len(dates),
        "pairs_scored": len(scored),
        "worst_median_correlation": min(d["median_correlation_stable"] for d in dates),
        "worst_abs_offset_rows_px": max(abs(d["offset_rows_px"]) for d in dates),
        "worst_abs_offset_cols_px": max(abs(d["offset_cols_px"]) for d in dates),
        "worst_phase_circular_std_rad": max(scored) if scored else None,
        "median_phase_circular_std_rad": float(np.median(scored)) if scored else None,
    }
    summary["pass_registration"] = bool(
        summary["worst_median_correlation"] >= 0.95
        and summary["worst_abs_offset_rows_px"] <= 0.1
        and summary["worst_abs_offset_cols_px"] <= 0.1
    )
    summary["pass_phase"] = bool(scored) and summary["worst_phase_circular_std_rad"] <= 0.1
    report = {"label": args.label, "summary": summary, "dates": dates, "phase_pairs": phase}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
