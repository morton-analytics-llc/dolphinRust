#!/usr/bin/env python
"""Generate Phase-9 (unwrapping) oracle fixtures via a direct SNAPHU invocation.

Writes a synthetic wrapped interferogram + correlation, runs the SNAPHU binary
directly (smooth cost, MCF init, single tile, UINT conncomp), and saves the
inputs and SNAPHU's unwrapped + connected-component outputs. The Rust wrapper
must reproduce these by invoking the same binary the same way.

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_unwrap.py
(only needs numpy + snaphu on PATH)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"
ROWS, COLS = 48, 64


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:ROWS, 0:COLS].astype(np.float64)
    # Smooth true phase (several cycles) -> wrapped ifg; high correlation.
    true = 0.10 * xx + 0.06 * yy + 0.0015 * (xx - COLS / 2) ** 2
    ifg = np.exp(1j * true).astype(np.complex64)
    corr = np.full((ROWS, COLS), 0.9, dtype=np.float32)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "ifg.c8").write_bytes(ifg.tobytes())
        (tmp / "corr.f4").write_bytes(corr.tobytes())
        unw_path, cc_path = tmp / "unw.f4", tmp / "cc.u4"
        cmd = [
            "snaphu", "-s", "--mcf",
            "-C", "CONNCOMPOUTTYPE UINT",
            "-C", "OUTFILEFORMAT FLOAT_DATA",
            "-C", "CORRFILEFORMAT FLOAT_DATA",
            "-c", str(tmp / "corr.f4"),
            "-o", str(unw_path),
            "-g", str(cc_path),
            str(tmp / "ifg.c8"), str(COLS),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        unw = np.frombuffer(unw_path.read_bytes(), dtype=np.float32).reshape(ROWS, COLS)
        cc = np.frombuffer(cc_path.read_bytes(), dtype=np.uint32).reshape(ROWS, COLS)

    np.save(OUT / "unw_ifg.npy", ifg)
    np.save(OUT / "unw_corr.npy", corr)
    np.save(OUT / "unw_oracle.npy", unw.copy())
    np.save(OUT / "unw_conncomp.npy", cc.copy())

    print(f"wrote unwrap fixtures to {OUT}")
    print(f"  ifg {ifg.shape}  unw range=[{unw.min():.2f},{unw.max():.2f}]")
    print(f"  conncomp labels = {np.unique(cc)}")


if __name__ == "__main__":
    main()
