#!/usr/bin/env python
"""Form a REAL wrapped interferogram + REAL empirical coherence from the real
OPERA CSLC-S1 burst stack (Mexico T005, `oracle/fixtures/real_slc_stack.npy`,
13x384x384 complex64 captured SAR). Long-temporal-baseline pair (epoch 0 vs 12)
for realistic temporal/geometric decorrelation; coherence is the standard boxcar
sample coherence. Emits flat binary `ifg.c8`/`corr.f4` (SNAPHU layout) for the
native-vs-SNAPHU validation harness. NOT synthetic — real captured Sentinel-1.
"""
import numpy as np
from pathlib import Path


def boxcar(a, w):
    """Mean over a w x w window (reflect-padded) via a separable integral image."""
    pad = w // 2
    x = np.pad(a, pad, mode="reflect").astype(np.float64)
    c = np.cumsum(np.cumsum(x, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    s = c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]
    return s / (w * w)


import os

NAME = os.environ.get("PAIR_NAME", "real_ifg")
REF = int(os.environ.get("REF", "0"))
SEC = int(os.environ.get("SEC", "12"))
OUT = Path(__file__).resolve().parent / "fixtures" / NAME
OUT.mkdir(parents=True, exist_ok=True)
stack = np.load(Path(__file__).resolve().parent / "fixtures" / "real_slc_stack.npy")
ref, sec = stack[REF], stack[SEC]
ifg_full = ref * np.conj(sec)
W = 5
num = boxcar(ifg_full.real, W) + 1j * boxcar(ifg_full.imag, W)
p1 = boxcar((ref * np.conj(ref)).real, W)
p2 = boxcar((sec * np.conj(sec)).real, W)
coh = np.abs(num) / np.sqrt(np.maximum(p1 * p2, 1e-20))
coh = np.clip(coh, 0.0, 1.0).astype(np.float32)
ifg = np.exp(1j * np.angle(ifg_full)).astype(np.complex64)  # unit-magnitude wrapped phase
ifg.tofile(OUT / "ifg.c8")
coh.tofile(OUT / "corr.f4")

TWO_PI = 2 * np.pi


def wrap(d):
    return d - TWO_PI * np.round(d / TWO_PI)


ph = np.angle(ifg)
dt = wrap(ph[:-1, 1:] - ph[:-1, :-1])
dr = wrap(ph[1:, 1:] - ph[:-1, 1:])
db = wrap(ph[1:, :-1] - ph[1:, 1:])
dl = wrap(ph[:-1, :-1] - ph[1:, :-1])
res = np.round((dt + dr + db + dl) / TWO_PI).astype(int)
print(f"shape={ifg.shape} dtype={ifg.dtype}")
print(f"coh: min={coh.min():.3f} max={coh.max():.3f} mean={coh.mean():.3f} "
      f"frac<0.15={np.mean(coh < 0.15):.3f} frac<0.3={np.mean(coh < 0.3):.3f}")
print(f"residues={np.count_nonzero(res)} ({100 * np.count_nonzero(res) / res.size:.3f}%)")
print(f"wrote {OUT}/ifg.c8 {OUT}/corr.f4")
