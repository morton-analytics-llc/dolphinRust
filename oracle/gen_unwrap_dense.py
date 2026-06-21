#!/usr/bin/env python
"""Residue-DENSE native-unwrap parity scene + SNAPHU goldens.

The Tier-2 synthetic suite (gen_unwrap_suite.py) was 4/5 residue-free; only
`lowcoh` (95 residues) exercised the MCF solver — orders of magnitude below real
Sentinel-1 residue density (1e4-1e6). This generator builds a realistic-synthetic
high-residue scene that mirrors real burst statistics so the native clean-room
unwrapper is gated where it actually matters:

  - decorrelation-driven phase noise (CRLB sigma from a spatially varying
    coherence field) yielding 1e4+ residues,
  - MULTIPLE disconnected coherent regions (near-zero-correlation moats that
    SNAPHU's conncomp regrow drops to component 0, splitting the scene),
  - masked low-coherence bands,
  - steep deformation gradients (a subsidence cone with near-aliasing fringe rate).

NOT a real scene: no real CSLC burst was available on disk (only 48x64 toy
fixtures). Flagged synthetic-but-realistic per the parity-gate mission.

SNAPHU is invoked EXACTLY as the dolphinRust default config does (smooth cost,
MCF init, UINT conncomp, FLOAT formats) and used ONLY as a black-box oracle —
never linked, never read.

Run:  oracle/.venv/bin/python oracle/gen_unwrap_dense.py
Env:  DENSE_ROWS, DENSE_COLS (default 1024), DENSE_LOOKS (multilook, default 6),
      DENSE_TAG (fixture prefix, default 'unwdense'), DENSE_OUT (override dir).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("DENSE_OUT", Path(__file__).resolve().parent / "fixtures"))
SEED = 20260620
TWO_PI = 2.0 * np.pi


def wrap(d: np.ndarray) -> np.ndarray:
    """Principal-value wrap to (-pi, pi]."""
    return d - TWO_PI * np.round(d / TWO_PI)


def residue_count(phase: np.ndarray) -> tuple[int, np.ndarray]:
    """Discrete curl of wrapped gradients around every 2x2 loop (native's defn)."""
    d_top = wrap(phase[:-1, 1:] - phase[:-1, :-1])
    d_right = wrap(phase[1:, 1:] - phase[:-1, 1:])
    d_bot = wrap(phase[1:, :-1] - phase[1:, 1:])
    d_left = wrap(phase[:-1, :-1] - phase[1:, :-1])
    res = np.round((d_top + d_right + d_bot + d_left) / TWO_PI).astype(np.int32)
    return int(np.count_nonzero(res)), res


def coherence_field(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """High-coherence coherent patches, broad moderate-decorrelation speckle,
    near-zero-corr moats (component splitters) + a masked low-coh band."""
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    # Smooth moderate background (0.45-0.7) from low-freq random field.
    lf = rng.normal(0.0, 1.0, size=(8, 8))
    bg = np.kron(lf, np.ones((rows // 8 + 1, cols // 8 + 1)))[:rows, :cols]
    # box-smooth the blocky kron a little
    bg = 0.5 * bg + 0.5 * np.roll(bg, 1, axis=0)
    bg = (bg - bg.min()) / (np.ptp(bg) + 1e-9)
    corr = 0.45 + 0.25 * bg
    # Two high-coherence islands (stable scatterer clusters).
    for cy, cx, r in ((rows * 0.30, cols * 0.30, rows * 0.18),
                      (rows * 0.72, cols * 0.68, rows * 0.16)):
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        corr[m] = 0.90
    # Near-zero-corr moats: a cross that splits the scene into 4 quadrants,
    # plus a diagonal masked low-coh band. SNAPHU drops these to conncomp 0,
    # disconnecting the coherent regions.
    moat = (np.abs(xx - cols * 0.5) < cols * 0.018) | (np.abs(yy - rows * 0.5) < rows * 0.018)
    band = np.abs((yy - rows * 0.25) - 0.6 * (xx - cols * 0.5)) < rows * 0.02
    corr[moat | band] = 0.02
    return np.clip(corr, 0.02, 0.95).astype(np.float32)


def deformation(rows: int, cols: int) -> np.ndarray:
    """Multi-fringe ramp + steep subsidence cone (near-aliasing gradient)."""
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    ramp = 0.06 * xx + 0.04 * yy
    cy, cx = rows * 0.55, cols * 0.45
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    cone = 9.0 * TWO_PI * np.exp(-rr / (rows * 0.16))  # ~9 fringes, steep core
    return ramp + cone


def build_scene(rows: int, cols: int, looks: int):
    rng = np.random.default_rng(SEED)
    true = deformation(rows, cols)
    corr = coherence_field(rows, cols, rng)
    # CRLB interferometric phase std (single-look), reduced by sqrt(looks).
    g = np.clip(corr, 0.05, 0.999)
    sigma = np.sqrt((1.0 - g ** 2) / (2.0 * g ** 2)) / np.sqrt(max(looks, 1))
    noisy = true + rng.normal(0.0, 1.0, size=true.shape) * sigma
    ifg = np.exp(1j * noisy).astype(np.complex64)
    return ifg, corr, true, noisy


def run_snaphu(ifg: np.ndarray, corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = ifg.shape
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "ifg.c8").write_bytes(ifg.astype(np.complex64).tobytes())
        (tmp / "corr.f4").write_bytes(corr.astype(np.float32).tobytes())
        unw_path, cc_path = tmp / "unw.f4", tmp / "cc.u4"
        cmd = [
            "snaphu", "-s", "--mcf",
            "-C", "CONNCOMPOUTTYPE UINT",
            "-C", "OUTFILEFORMAT FLOAT_DATA",
            "-C", "CORRFILEFORMAT FLOAT_DATA",
            "-c", str(tmp / "corr.f4"),
            "-o", str(unw_path),
            "-g", str(cc_path),
            str(tmp / "ifg.c8"), str(cols),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        unw = np.frombuffer(unw_path.read_bytes(), dtype=np.float32).reshape(rows, cols)
        cc = np.frombuffer(cc_path.read_bytes(), dtype=np.uint32).reshape(rows, cols)
    return unw.copy(), cc.copy()


def main() -> None:
    rows = int(os.environ.get("DENSE_ROWS", 1024))
    cols = int(os.environ.get("DENSE_COLS", 1024))
    looks = int(os.environ.get("DENSE_LOOKS", 6))
    tag = os.environ.get("DENSE_TAG", "unwdense")
    OUT.mkdir(parents=True, exist_ok=True)

    ifg, corr, true, noisy = build_scene(rows, cols, looks)
    nres, _ = residue_count(noisy)
    print(f"[scene] {rows}x{cols} looks={looks}  residues={nres} "
          f"({100.0 * nres / (rows * cols):.3f}% of px)  "
          f"corr[min={corr.min():.2f} mean={corr.mean():.2f}]")

    unw, cc = run_snaphu(ifg, corr)
    labels, counts = np.unique(cc, return_counts=True)
    ncomp = int((labels > 0).sum())
    masked = int(counts[labels == 0].sum()) if 0 in labels else 0
    print(f"[snaphu] conncomps={ncomp} (>0)  masked_px={masked} "
          f"({100.0 * masked / (rows * cols):.2f}%)  "
          f"unw=[{unw.min():.1f},{unw.max():.1f}]")
    for lab, cnt in sorted(zip(labels.tolist(), counts.tolist()), key=lambda t: -t[1])[:8]:
        print(f"    comp {lab:3d}: {cnt} px")

    np.save(OUT / f"{tag}_ifg.npy", ifg)
    np.save(OUT / f"{tag}_corr.npy", corr)
    np.save(OUT / f"{tag}_oracle.npy", unw)
    np.save(OUT / f"{tag}_conncomp.npy", cc)
    print(f"wrote {tag}_* ({rows}x{cols}) to {OUT}")


if __name__ == "__main__":
    main()
