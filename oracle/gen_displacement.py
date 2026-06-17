#!/usr/bin/env python
"""Generate Phase-10 end-to-end displacement oracle fixtures.

Chains dolphin primitives on a synthetic single-burst CSLC stack:
phase-linking -> single-reference network -> ifg formation -> SNAPHU unwrap ->
SBAS L2 inversion -> velocity. Writes the CSLC HDF5 stack + a workflow YAML the
Rust CLI consumes, and the displacement series + velocity for comparison.

The synthetic phase is smooth and small (|ifg phase| < ~1.2 rad) so unwrapping is
cycle-free and robust to the faer-vs-jax ~1e-6 phase-linking divergence.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_displacement.py
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link._core import run_phase_linking
from dolphin.timeseries import estimate_velocity, get_incidence_matrix, invert_stack

OUT = Path(__file__).resolve().parent / "fixtures"
DISP_DIR = OUT / "disp"
N, ROWS, COLS = 5, 48, 64
HALF = HalfWindow(1, 1)
STRIDES = Strides(1, 1)
DATASET = "/data/VV"
DT_DAYS = 12.0
BASE_DATE = date(2022, 11, 19)  # real 12-day Sentinel-1 cadence; names carry the dates
REF_POINT = (24, 32)  # explicit spatial reference pixel (row, col); see write_config


def synth_stack() -> np.ndarray:
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[0:ROWS, 0:COLS].astype(np.float64)
    ramp = xx / COLS  # in [0, 1)
    phases = np.stack([0.3 * t * ramp for t in range(N)])  # smooth, small
    speckle = rng.standard_normal((N, ROWS, COLS)) + 1j * rng.standard_normal((N, ROWS, COLS))
    return (np.exp(1j * phases) + 0.05 * speckle / np.sqrt(2)).astype(np.complex64)


def snaphu_unwrap(ifg: np.ndarray) -> np.ndarray:
    corr = np.ones((ROWS, COLS), dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "i.c8").write_bytes(ifg.astype(np.complex64).tobytes())
        (t / "c.f4").write_bytes(corr.tobytes())
        subprocess.run(
            ["snaphu", "-s", "--mcf",
             "-C", "CONNCOMPOUTTYPE UINT", "-C", "OUTFILEFORMAT FLOAT_DATA",
             "-C", "CORRFILEFORMAT FLOAT_DATA",
             "-c", str(t / "c.f4"), "-o", str(t / "u.f4"), "-g", str(t / "g.u4"),
             str(t / "i.c8"), str(COLS)],
            check=True, capture_output=True,
        )
        return np.frombuffer((t / "u.f4").read_bytes(), np.float32).reshape(ROWS, COLS).astype(np.float64)


def main() -> None:
    DISP_DIR.mkdir(parents=True, exist_ok=True)
    stack = synth_stack()

    # Write per-date CSLC HDF5 files + the workflow YAML.
    files = []
    for t in range(N):
        stamp = (BASE_DATE + timedelta(days=int(t * DT_DAYS))).strftime("%Y%m%d")
        path = DISP_DIR / f"cslc_{stamp}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(DATASET, data=stack[t])
        files.append(path)
    write_config(files)

    # Phase linking (single ministack), referenced to date 0.
    pl = np.asarray(run_phase_linking(stack, HALF, STRIDES, use_evd=False, reference_idx=0).cpx_phase)

    # Single-reference network + ifgs + unwrap.
    pairs = [(0, j) for j in range(1, N)]
    dphi = np.stack([snaphu_unwrap(np.exp(1j * np.angle(pl[j] * np.conj(pl[i])))) for i, j in pairs])

    a = get_incidence_matrix(pairs).astype(np.float64)
    disp, _ = invert_stack(a, dphi)
    disp = np.asarray(disp)  # (N-1, rows, cols)

    days = np.array([t * DT_DAYS for t in range(N)], dtype=float)
    series = np.concatenate([np.zeros((1, ROWS, COLS)), disp], axis=0)
    velocity = np.asarray(estimate_velocity(days, series, None))

    # Spatially reference to REF_POINT (matches dolphinRust's reference_to_point with
    # the explicit timeseries_options.reference_point set in the config below).
    rr, cc = REF_POINT
    disp = disp - disp[:, rr, cc][:, None, None]
    velocity = velocity - velocity[rr, cc]

    np.save(OUT / "disp_displacement.npy", disp.astype(np.float64))
    np.save(OUT / "disp_velocity.npy", velocity.astype(np.float64))
    print(f"wrote displacement fixtures to {OUT}")
    print(f"  displacement {disp.shape} range=[{disp.min():.3f},{disp.max():.3f}]")
    print(f"  velocity {velocity.shape} range=[{velocity.min():.3f},{velocity.max():.3f}]")


def write_config(files: list[Path]) -> None:
    lines = ["cslc_file_list:"]
    lines += [f"  - {f}" for f in files]
    lines += [
        "input_options:",
        f"  subdataset: {DATASET}",
        "phase_linking:",
        "  ministack_size: 15",
        "  half_window:",
        "    y: 1",
        "    x: 1",
        "output_options:",
        "  strides:",
        "    y: 1",
        "    x: 1",
        "interferogram_network:",
        "  reference_idx: 0",
        "timeseries_options:",
        "  reference_point:",
        f"    - {REF_POINT[0]}",
        f"    - {REF_POINT[1]}",
        f"work_directory: {DISP_DIR}",
    ]
    (DISP_DIR / "config.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
