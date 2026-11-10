"""Controlled measurement: sequential phase linking with the ALWAYS_FIRST plan,
referencing each ministack to the OLDEST carried compressed SLC (dolphin v0.35,
and dolphinRust today) against the NEWEST (dolphin >= v0.39, PR #588).

Same simulated stack, same dolphin kernels (run_phase_linking + compress), only
the two reference indices differ. Truth is known, so the error is measured
directly. Run in either oracle env; the kernels used exist in both:

    oracle/.venv-v042/bin/python oracle/ref_idx_experiment.py TAU_DAYS GAMMA_INF

e.g. `90 0.0`. Results and reading: md/research/compressed-reference-index-audit-2026-11.md.
"""

from __future__ import annotations

import sys

import numpy as np

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link._compress import compress
from dolphin.phase_link._core import run_phase_linking

N, ROWS, COLS = 60, 44, 44
SIZE, MAXC = 15, 5
HALF = HalfWindow(5, 5)
STRIDES = Strides(1, 1)
TAU_DAYS, GAMMA_INF, REPEAT = float(sys.argv[1]), float(sys.argv[2]), 12.0


def simulate(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous distributed scatterers: exponential decorrelation to a floor,
    one true phase history shared by the scene (a 3 rad/yr trend plus a seasonal)."""
    rng = np.random.default_rng(seed)
    days = REPEAT * np.arange(N)
    truth = 3.0 * days / 365.25 + 0.8 * np.sin(2 * np.pi * days / 365.25)
    lag = np.abs(days[:, None] - days[None, :])
    coherence = (1 - GAMMA_INF) * np.exp(-lag / TAU_DAYS) + GAMMA_INF
    covariance = coherence * np.exp(1j * (truth[:, None] - truth[None, :]))
    chol = np.linalg.cholesky(covariance + 1e-10 * np.eye(N))
    white = (rng.standard_normal((N, ROWS * COLS)) + 1j * rng.standard_normal((N, ROWS * COLS))) / np.sqrt(2)
    return (chol @ white).reshape(N, ROWS, COLS).astype(np.complex64), truth


def sequential(stack: np.ndarray, newest: bool) -> np.ndarray:
    comp: list[np.ndarray] = []
    phases = []
    for start in range(0, N, SIZE):
        carried = comp[-MAXC:]
        ncc = len(carried)
        reference = max(0, ncc - 1) if newest else 0
        combined = np.concatenate(
            [np.stack(carried), stack[start : start + SIZE]] if carried else [stack[start : start + SIZE]]
        ).astype(np.complex64)
        linked = np.asarray(
            run_phase_linking(combined, HALF, STRIDES, use_evd=False, reference_idx=reference).cpx_phase
        )
        comp.append(compress(combined, linked, first_real_slc_idx=ncc, reference_idx=reference))
        phases.append(linked[ncc:])
    return np.concatenate(phases)


def error_by_ministack(linked: np.ndarray, truth: np.ndarray) -> list[float]:
    """RMS of the wrapped error against truth, interior pixels, per ministack."""
    interior = linked[:, HALF.y : -HALF.y, HALF.x : -HALF.x]
    # dolphin's linked phase is conj-referenced; sign is fixed by date 1 below.
    estimate = np.angle(interior * np.conj(interior[0:1]))
    sign = np.sign(np.nanmedian(estimate[1])) * np.sign(truth[1] - truth[0])
    wrapped = np.angle(np.exp(1j * (sign * estimate - (truth - truth[0])[:, None, None])))
    return [float(np.sqrt(np.nanmean(wrapped[s : s + SIZE] ** 2))) for s in range(0, N, SIZE)]


def junction_step(linked: np.ndarray, truth: np.ndarray) -> list[float]:
    """Per-pixel spread (std, rad) of each ministack's mean error: the datum
    offset one ministack carries relative to the truth, which shows as a step at
    the junction."""
    interior = linked[:, HALF.y : -HALF.y, HALF.x : -HALF.x]
    estimate = np.angle(interior * np.conj(interior[0:1]))
    sign = np.sign(np.nanmedian(estimate[1])) * np.sign(truth[1] - truth[0])
    wrapped = np.angle(np.exp(1j * (sign * estimate - (truth - truth[0])[:, None, None])))
    offsets = [np.angle(np.exp(1j * wrapped[s : s + SIZE]).mean(axis=0)) for s in range(0, N, SIZE)]
    return [float(np.nanstd(offset)) for offset in offsets]


def main() -> None:
    rms = {False: [], True: []}
    step = {False: [], True: []}
    for seed in range(5):
        stack, truth = simulate(seed)
        for newest in (False, True):
            linked = sequential(stack, newest)
            rms[newest].append(error_by_ministack(linked, truth))
            step[newest].append(junction_step(linked, truth))
    fmt = lambda rows: " ".join(f"{v:.4f}" for v in np.mean(rows, axis=0))
    print(f"tau={TAU_DAYS} d, gamma_inf={GAMMA_INF}, {N} dates, ministack {SIZE}, half-window {HALF.y}, 5 seeds")
    print("RMS phase error vs truth by ministack (rad):")
    print("  oldest compressed (v0.35 / dolphinRust):", fmt(rms[False]))
    print("  newest compressed (dolphin >= 0.39)    :", fmt(rms[True]))
    print("Spread across pixels of each ministack's datum offset (rad):")
    print("  oldest compressed (v0.35 / dolphinRust):", fmt(step[False]))
    print("  newest compressed (dolphin >= 0.39)    :", fmt(step[True]))


if __name__ == "__main__":
    main()
