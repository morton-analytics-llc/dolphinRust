#!/usr/bin/env python
"""Generate ministack-planner oracle fixtures from a pinned dolphin release.

For several (N, ministack_size, max_num_compressed, plan, output_reference_idx)
combos, dump dolphin's MiniStackPlanner.plan() structure as an int64 array of
one row per ministack:
[num_compressed, num_real, output_reference_idx, compressed_reference_idx].

The dolphin version is part of every file name (`plan_v0429_...`), and the
script refuses to run under a release it has no combos for, so a fixture cannot
be regenerated under the wrong rule:

- v0.42.9 pins `always_first` as dolphinRust implements it (issue #192): both
  indices on the newest carried compressed SLC unless the output is configured.
  It also pins the single ministack `last_per_ministack` allows (issue #206):
  output on index 0 unless configured, compressed on -1.
- v0.35.0 pins `always_first_legacy`, dolphin's rule before PR #588 (v0.39.0):
  both indices on the configured output index, 0 by default.

Run once per env:
    oracle/.venv/bin/python oracle/gen_stack.py              # v0.35.0
    PYTHONPATH=<dolphin v0.42.9> oracle/.venv-v042/bin/python oracle/gen_stack.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import dolphin
from dolphin.stack import CompressedSlcPlan, MiniStackPlanner

OUT = Path(__file__).resolve().parent / "fixtures"

# (N, ministack_size, max_num_compressed, plan-name, output_reference_idx)
# -- mirrored in crates/dolphin-stack/tests/planner_contract.rs.
COMBOS = {
    "0.42.9": [
        (10, 5, 10, "always_first", None),
        (12, 5, 2, "always_first", None),
        (7, 3, 10, "always_first", None),
        (60, 15, 5, "always_first", None),
        (20, 5, 3, "always_first", None),
        (20, 5, 10, "always_first", 2),
        (4, 4, 10, "last_per_ministack", None),
        (4, 4, 10, "last_per_ministack", 2),
    ],
    "0.35.0": [
        (10, 5, 10, "always_first", 0),
        (12, 5, 2, "always_first", 0),
        (7, 3, 10, "always_first", 0),
        (60, 15, 5, "always_first", 0),
        (20, 5, 10, "always_first", 2),
        (10, 4, 10, "first_per_ministack", 0),
    ],
}
PLANS = {p.value: p for p in CompressedSlcPlan}


def plan_rows(n: int, size: int, maxc: int, plan_name: str, reference: int | None) -> np.ndarray:
    files = [f"slc_{i}.h5" for i in range(n)]
    dates = [[datetime(2020, 1, 1) + timedelta(days=12 * i)] for i in range(n)]
    planner = MiniStackPlanner(
        file_list=files,
        dates=dates,
        is_compressed=[False] * n,
        max_num_compressed=maxc,
        output_reference_idx=reference,
        compressed_slc_plan=PLANS[plan_name],
    )
    rows = []
    for ms in planner.plan(size):
        num_comp = int(sum(ms.is_compressed))
        num_real = len(ms.file_list) - num_comp
        rows.append([num_comp, num_real, ms.output_reference_idx, ms.compressed_reference_idx])
    return np.array(rows, dtype=np.int64)


def main() -> None:
    version = dolphin.__version__
    if version not in COMBOS:
        raise SystemExit(f"no planner combos pinned for dolphin {version}; expected one of {sorted(COMBOS)}")
    tag = "v" + version.replace(".", "")
    OUT.mkdir(parents=True, exist_ok=True)
    for n, size, maxc, plan_name, reference in COMBOS[version]:
        rows = plan_rows(n, size, maxc, plan_name, reference)
        ref = "none" if reference is None else str(reference)
        name = f"plan_{tag}_{n}_{size}_{maxc}_{plan_name}_ref{ref}.npy"
        np.save(OUT / name, rows)
        print(f"  {name}: {rows.tolist()}")
    print(f"wrote dolphin {version} planner fixtures to {OUT}")


if __name__ == "__main__":
    main()
