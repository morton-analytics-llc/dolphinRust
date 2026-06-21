#!/usr/bin/env python
"""Adversarial seam-robustness scenes for the native tiled unwrapper.

PUSH-1 of the native-unwrap mission: the modal-offset inter-tile stitch is
parity-sensitive to seam placement at high residue density — when a seam bisects
a coherent region a single per-tile integer offset cannot reconcile the two
sides. This generator builds a FAMILY of scenes specifically designed to break
that stitch, spanning:

  - residue density (via multilook -> CRLB phase noise),
  - coherence structure (where the low-coherence moats fall, if any), and
  - implicitly, tile-count interaction (the Rust sweep tiles each scene 2..8).

Structures (env STRUCT):
  multimoat  - central cross + extra bands: many components, even tilings align.
  offcenter  - moat at 37%/63%: even tile counts do NOT align -> stresses stitch.
  layover    - rectangular full-decorrelation patches (layover-like).
  multicomp  - many small coherent islands in a decorrelated sea.
  diag       - a diagonal coherent strip; seams bisect it at every angle.
  nomoat     - NO moats: every seam bisects a fully-coherent noisy region. The
               maximal straddling-dipole case (worst case for a per-tile offset).

The native GLOBAL solve (tile=None) is the oracle for the tiled-vs-global sweep
(it is SNAPHU-parity at 0.011% per the dense gate), so this generator does NOT
invoke SNAPHU per scene — it only emits ifg + corr. A separate flag (--snaphu)
adds a SNAPHU golden for headline confirmation scenes. SNAPHU, when used, is a
black-box oracle only: never linked, never read.

Run:  oracle/.venv/bin/python oracle/gen_seam_sweep.py
Env:  SWEEP_SIZE (default 256), SWEEP_OUT (override dir),
      STRUCT (one structure) or omitted (emit the whole family),
      SEEDS (count of seeds per structure, default 3),
      LOOKS (comma list of multilooks, default '3,6').
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("SWEEP_OUT", Path(__file__).resolve().parent / "fixtures" / "seam"))
TWO_PI = 2.0 * np.pi
STRUCTS = ("multimoat", "offcenter", "layover", "multicomp", "diag", "nomoat")


def wrap(d: np.ndarray) -> np.ndarray:
    return d - TWO_PI * np.round(d / TWO_PI)


def residue_count(phase: np.ndarray) -> int:
    d_top = wrap(phase[:-1, 1:] - phase[:-1, :-1])
    d_right = wrap(phase[1:, 1:] - phase[:-1, 1:])
    d_bot = wrap(phase[1:, :-1] - phase[1:, 1:])
    d_left = wrap(phase[:-1, :-1] - phase[1:, :-1])
    res = np.round((d_top + d_right + d_bot + d_left) / TWO_PI).astype(np.int32)
    return int(np.count_nonzero(res))


def smooth_background(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    lf = rng.normal(0.0, 1.0, size=(8, 8))
    bg = np.kron(lf, np.ones((rows // 8 + 1, cols // 8 + 1)))[:rows, :cols]
    bg = 0.5 * bg + 0.5 * np.roll(bg, 1, axis=0)
    bg = (bg - bg.min()) / (np.ptp(bg) + 1e-9)
    return 0.45 + 0.25 * bg


def coherence_field(struct: str, rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    corr = smooth_background(rows, cols, rng)
    # Two stable high-coherence islands common to all structures.
    for cy, cx, r in ((rows * 0.30, cols * 0.30, rows * 0.18),
                      (rows * 0.72, cols * 0.68, rows * 0.16)):
        corr[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = 0.90
    low = _low_coh_mask(struct, yy, xx, rows, cols, rng)
    corr[low] = 0.02
    return np.clip(corr, 0.02, 0.95).astype(np.float32)


def _low_coh_mask(struct, yy, xx, rows, cols, rng):  # noqa: PLR0911
    if struct == "nomoat":
        return np.zeros_like(yy, dtype=bool)
    if struct == "multimoat":
        cross = (np.abs(xx - cols * 0.5) < cols * 0.018) | (np.abs(yy - rows * 0.5) < rows * 0.018)
        band = np.abs((yy - rows * 0.25) - 0.6 * (xx - cols * 0.5)) < rows * 0.02
        return cross | band
    if struct == "offcenter":
        return (np.abs(xx - cols * 0.37) < cols * 0.015) | (np.abs(yy - rows * 0.63) < rows * 0.015)
    if struct == "layover":
        mask = np.zeros_like(yy, dtype=bool)
        for _ in range(5):
            y0, x0 = rng.integers(0, rows - 1), rng.integers(0, cols - 1)
            h, w = rng.integers(rows // 12, rows // 5), rng.integers(cols // 12, cols // 5)
            mask[y0:y0 + h, x0:x0 + w] = True
        return mask
    if struct == "multicomp":
        # A lattice of thin decorrelation lines splitting the scene into a grid
        # of small coherent cells (many components, off the tile grid).
        gx = (np.abs(((xx + cols * 0.07) % (cols / 5.0)) - cols / 10.0) < cols * 0.012)
        gy = (np.abs(((yy + rows * 0.11) % (rows / 5.0)) - rows / 10.0) < rows * 0.012)
        return gx | gy
    if struct == "diag":
        # Decorrelated everywhere EXCEPT a diagonal coherent strip.
        strip = np.abs((yy - rows * 0.5) - (xx - cols * 0.5)) < rows * 0.16
        return ~strip
    raise SystemExit(f"unknown STRUCT {struct!r}")


def deformation(rows: int, cols: int) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    ramp = 0.06 * xx + 0.04 * yy
    cy, cx = rows * 0.55, cols * 0.45
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    cone = 9.0 * TWO_PI * np.exp(-rr / (rows * 0.16))
    return ramp + cone


def build_scene(struct: str, rows: int, cols: int, looks: int, seed: int):
    rng = np.random.default_rng(seed)
    true = deformation(rows, cols)
    corr = coherence_field(struct, rows, cols, rng)
    g = np.clip(corr, 0.05, 0.999)
    sigma = np.sqrt((1.0 - g ** 2) / (2.0 * g ** 2)) / np.sqrt(max(looks, 1))
    noisy = true + rng.normal(0.0, 1.0, size=true.shape) * sigma
    ifg = np.exp(1j * noisy).astype(np.complex64)
    return ifg, corr, noisy


def run_snaphu(ifg, corr):
    rows, cols = ifg.shape
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "ifg.c8").write_bytes(ifg.astype(np.complex64).tobytes())
        (tmp / "corr.f4").write_bytes(corr.astype(np.float32).tobytes())
        unw_path, cc_path = tmp / "unw.f4", tmp / "cc.u4"
        cmd = ["snaphu", "-s", "--mcf",
               "-C", "CONNCOMPOUTTYPE UINT", "-C", "OUTFILEFORMAT FLOAT_DATA",
               "-C", "CORRFILEFORMAT FLOAT_DATA", "-c", str(tmp / "corr.f4"),
               "-o", str(unw_path), "-g", str(cc_path), str(tmp / "ifg.c8"), str(cols)]
        subprocess.run(cmd, check=True, capture_output=True)
        unw = np.frombuffer(unw_path.read_bytes(), dtype=np.float32).reshape(rows, cols)
        cc = np.frombuffer(cc_path.read_bytes(), dtype=np.uint32).reshape(rows, cols)
    return unw.copy(), cc.copy()


def emit(struct: str, size: int, looks: int, seed: int, with_snaphu: bool) -> None:
    ifg, corr, noisy = build_scene(struct, size, size, looks, seed)
    nres = residue_count(noisy)
    tag = f"{struct}_l{looks}_s{seed}"
    np.save(OUT / f"{tag}_ifg.npy", ifg)
    np.save(OUT / f"{tag}_corr.npy", corr)
    extra = ""
    if with_snaphu:
        unw, cc = run_snaphu(ifg, corr)
        np.save(OUT / f"{tag}_oracle.npy", unw)
        np.save(OUT / f"{tag}_conncomp.npy", cc)
        ncomp = int((np.unique(cc) > 0).sum())
        extra = f"  snaphu_comps={ncomp}"
    print(f"[{tag}] {size}x{size} residues={nres} ({100.0 * nres / size / size:.2f}%)"
          f"  corr_min={corr.min():.2f}{extra}")


def main() -> None:
    size = int(os.environ.get("SWEEP_SIZE", 256))
    seeds = int(os.environ.get("SEEDS", 3))
    looks = [int(x) for x in os.environ.get("LOOKS", "3,6").split(",")]
    with_snaphu = os.environ.get("SNAPHU", "0") == "1"
    only = os.environ.get("STRUCT")
    structs = (only,) if only else STRUCTS
    OUT.mkdir(parents=True, exist_ok=True)
    base = 20260620
    for struct in structs:
        for li in looks:
            for s in range(seeds):
                emit(struct, size, li, base + 1000 * STRUCTS.index(struct) + 17 * li + s, with_snaphu)


if __name__ == "__main__":
    main()
