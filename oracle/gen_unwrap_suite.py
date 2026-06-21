#!/usr/bin/env python
"""Generate the native-unwrap golden suite via direct SNAPHU invocations.

The native (clean-room) unwrapping backend must match SNAPHU's integer-cycle
field (equal up to a global constant) on every hard case, not just smooth ramps.
This writes a fixture set that SPANS those cases:

  smooth     — gentle deformation ramp (several fringes), high coherence
  steep      — near-aliasing fringe rate (gradient approaching pi/pixel)
  discont    — fault-like 2-pi+ step across a line (true discontinuity)
  lowcoh     — noisy phase with a low-correlation band
  multitile  — larger grid that exercises tiling

For each class we synthesize a unit-magnitude wrapped interferogram + a
correlation raster, run the SNAPHU binary exactly as the dolphinRust default
config invokes it (smooth cost, MCF init, single tile, UINT conncomp, FLOAT
formats), and save inputs + SNAPHU's unwrapped phase + connected components.
SNAPHU is used here ONLY as a black-box oracle — never linked, never read.

Run inside the pinned env (only needs numpy + snaphu on PATH):
  oracle/.venv/bin/python oracle/gen_unwrap_suite.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"
SEED = 20260620


def run_snaphu(ifg: np.ndarray, corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Black-box SNAPHU oracle: wrapped ifg + corr -> (unwrapped, conncomp)."""
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


def wrap_unit(true: np.ndarray) -> np.ndarray:
    """Unit-magnitude wrapped ifg — matches what the backend forms from linked phase."""
    return np.exp(1j * true).astype(np.complex64)


def smooth(rows=64, cols=64):
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    true = 0.10 * xx + 0.06 * yy + 0.0015 * (xx - cols / 2) ** 2
    corr = np.full((rows, cols), 0.9, dtype=np.float32)
    return wrap_unit(true), corr, true


def steep(rows=64, cols=64):
    # Fringe rate ~0.9 rad/px in x and ~0.7 rad/px in y -> near-aliasing.
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    true = 0.90 * xx + 0.70 * yy
    corr = np.full((rows, cols), 0.9, dtype=np.float32)
    return wrap_unit(true), corr, true


def discont(rows=64, cols=64):
    # Two smooth halves separated by a fault. A NON-integer-cycle step (2.7
    # cycles) is genuinely visible in wrapped phase (an integer-cycle step would
    # be congruent / invisible); the unwrapper must choose the integer offset
    # that reconnects the halves across the seam.
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    base = 0.08 * xx + 0.05 * yy
    step = np.where(xx >= cols // 2, 2.7 * 2.0 * np.pi, 0.0)
    true = base + step
    corr = np.full((rows, cols), 0.9, dtype=np.float32)
    return wrap_unit(true), corr, true


def lowcoh(rows=64, cols=64):
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    true = 0.10 * xx + 0.06 * yy
    # Correlation: high background, a low band through the middle rows.
    corr = np.full((rows, cols), 0.85, dtype=np.float32)
    band = (yy >= rows * 0.40) & (yy <= rows * 0.60)
    corr[band] = 0.25
    # Phase noise scaled by decorrelation (sigma grows as coherence drops).
    sigma = np.sqrt(np.maximum(1.0 / np.maximum(corr, 0.05) ** 2 - 1.0, 0.0)) * 0.30
    noisy = true + rng.normal(0.0, 1.0, size=true.shape) * sigma
    return wrap_unit(noisy), corr.astype(np.float32), true


def multitile(rows=160, cols=160):
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    # Several fringes both axes + a broad bowl, high coherence.
    true = (
        0.12 * xx + 0.09 * yy
        + 0.0008 * (xx - cols / 2) ** 2
        + 0.0008 * (yy - rows / 2) ** 2
    )
    corr = np.full((rows, cols), 0.9, dtype=np.float32)
    return wrap_unit(true), corr, true


CLASSES = {
    "smooth": smooth,
    "steep": steep,
    "discont": discont,
    "lowcoh": lowcoh,
    "multitile": multitile,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, gen in CLASSES.items():
        ifg, corr, _true = gen()
        unw, cc = run_snaphu(ifg, corr)
        np.save(OUT / f"unwsuite_{name}_ifg.npy", ifg)
        np.save(OUT / f"unwsuite_{name}_corr.npy", corr)
        np.save(OUT / f"unwsuite_{name}_oracle.npy", unw)
        np.save(OUT / f"unwsuite_{name}_conncomp.npy", cc)
        print(
            f"{name:10s} {ifg.shape}  unw=[{unw.min():7.2f},{unw.max():7.2f}]  "
            f"cc_labels={np.unique(cc).size}"
        )
    print(f"wrote native-unwrap golden suite to {OUT}")


if __name__ == "__main__":
    main()
