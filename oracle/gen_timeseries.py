#!/usr/bin/env python
"""Generate Phase-6 (timeseries / SBAS L2) oracle fixtures from dolphin v0.35.0.

Emits, for several network modes, the interferogram index pairs; and for one
bandwidth network: the incidence matrix, synthetic weighted unwrapped ifgs, the
dolphin L2-inverted displacement series, and the fitted velocity.

Validates both inversion paths: the L2 weighted least squares (`invert_stack`)
and dolphin's default **L1/ADMM** (`invert_stack_l1`, Phase 6b). Run inside the
pinned env:
    oracle/.venv/bin/python oracle/gen_timeseries.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dolphin.interferogram import Network
from dolphin.timeseries import (
    estimate_velocity,
    get_incidence_matrix,
    invert_stack,
    invert_stack_l1,
)

OUT = Path(__file__).resolve().parent / "fixtures"
N_DATES = 6
ROWS = COLS = 4
DT_DAYS = 12.0  # even temporal sampling


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    slc = list(range(N_DATES))  # integer date identities (== indices)
    dts = [datetime(2020, 1, 1) + timedelta(days=DT_DAYS * i) for i in slc]
    days = [i * DT_DAYS for i in slc]

    # --- network modes (index pairs, sorted+deduped as dolphin does) ---
    nets = {
        "single_ref": sorted(set(Network._single_reference_network(slc, 0))),
        "bandwidth2": sorted(set(Network.limit_by_bandwidth(slc, 2))),
        "temporal30": sorted(
            set(Network.limit_by_temporal_baseline(slc, dates=dts, max_temporal_baseline=30.0))
        ),
        "indexes": sorted({(0, 1), (0, 3), (2, 5)}),
    }
    for name, pairs in nets.items():
        np.save(OUT / f"net_{name}.npy", np.array(pairs, dtype=np.int64))

    # --- L2 inversion on the bandwidth-2 network ---
    pairs = nets["bandwidth2"]
    A = get_incidence_matrix(pairs).astype(np.float64)  # (M, N_DATES-1)
    m = A.shape[0]

    rng = np.random.default_rng(11)
    # True displacement series (date 0 = 0), shape (N_DATES, rows, cols)
    disp = np.cumsum(
        np.concatenate([np.zeros((1, ROWS, COLS)), rng.standard_normal((N_DATES - 1, ROWS, COLS))]),
        axis=0,
    )
    # Unwrapped ifgs dphi = later - earlier, plus small noise
    dphi = np.empty((m, ROWS, COLS))
    for k, (a, b) in enumerate(pairs):
        dphi[k] = disp[b] - disp[a]
    dphi += 0.02 * rng.standard_normal(dphi.shape)
    weights = 0.5 + 0.5 * rng.random((m, ROWS, COLS))

    phase, _ = invert_stack(A, dphi, weights=weights)
    phase = np.asarray(phase)  # (N_DATES-1, rows, cols)

    # L1/ADMM inversion (unweighted, dolphin default) on the same network + dphi.
    phase_l1, _ = invert_stack_l1(A, dphi)
    phase_l1 = np.asarray(phase_l1)  # (N_DATES-1, rows, cols)

    full_series = np.concatenate([np.zeros((1, ROWS, COLS)), phase], axis=0)
    velocity = np.asarray(estimate_velocity(np.array(days, dtype=float), full_series, None))

    np.save(OUT / "ts_incidence.npy", A.astype(np.int64))
    np.save(OUT / "ts_dphi.npy", dphi.astype(np.float64))
    np.save(OUT / "ts_weights.npy", weights.astype(np.float64))
    np.save(OUT / "ts_phase.npy", phase.astype(np.float64))
    np.save(OUT / "ts_phase_l1.npy", phase_l1.astype(np.float64))
    np.save(OUT / "ts_velocity.npy", velocity.astype(np.float64))

    print(f"wrote timeseries fixtures to {OUT}")
    for name, pairs in nets.items():
        print(f"  net_{name}: {pairs}")
    print(f"  incidence {A.shape}  dphi {dphi.shape}  phase {phase.shape}  vel {velocity.shape}")


if __name__ == "__main__":
    main()
