#!/usr/bin/env python
"""Generate Phase-5 (ministack planner) oracle fixtures from dolphin v0.35.0.

For several (N, ministack_size, max_num_compressed, plan) combos, dump dolphin's
MiniStackPlanner.plan() structure as an int64 array of one row per ministack:
[num_compressed, num_real, output_reference_idx, compressed_reference_idx].

Run inside the pinned env:  oracle/.venv/bin/python oracle/gen_stack.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dolphin.stack import CompressedSlcPlan, MiniStackPlanner

OUT = Path(__file__).resolve().parent / "fixtures"

# (N, ministack_size, max_num_compressed, plan-name)  -- mirrored in the Rust test.
COMBOS = [
    (10, 5, 10, "always_first"),
    (12, 5, 2, "always_first"),
    (7, 3, 10, "always_first"),
    (10, 4, 10, "first_per_ministack"),
    (10, 4, 10, "last_per_ministack"),
]
PLANS = {p.value: p for p in CompressedSlcPlan}


def plan_rows(n: int, size: int, maxc: int, plan_name: str) -> np.ndarray:
    files = [f"slc_{i}.h5" for i in range(n)]
    dates = [[datetime(2020, 1, 1) + timedelta(days=12 * i)] for i in range(n)]
    planner = MiniStackPlanner(
        file_list=files,
        dates=dates,
        is_compressed=[False] * n,
        max_num_compressed=maxc,
        output_reference_idx=0,
        compressed_slc_plan=PLANS[plan_name],
    )
    rows = []
    for ms in planner.plan(size):
        num_comp = int(sum(ms.is_compressed))
        num_real = len(ms.file_list) - num_comp
        rows.append([num_comp, num_real, ms.output_reference_idx, ms.compressed_reference_idx])
    return np.array(rows, dtype=np.int64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n, size, maxc, plan_name in COMBOS:
        rows = plan_rows(n, size, maxc, plan_name)
        name = f"plan_{n}_{size}_{maxc}_{plan_name}.npy"
        np.save(OUT / name, rows)
        print(f"  {name}: {rows.tolist()}")
    print(f"wrote planner fixtures to {OUT}")


if __name__ == "__main__":
    main()
