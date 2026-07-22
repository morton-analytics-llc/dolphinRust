#!/usr/bin/env python
"""Pin dolphin v0.35.0's contract for a fully non-finite SLC acquisition.

Run with `oracle/.venv/bin/python oracle/check_all_nan_v035.py`. Exit 0 means
the pinned oracle rejected the stack with `PhaseLinkRuntimeError`; any returned
quality output or different failure is an evidence-changing error.
"""

from __future__ import annotations

import json

import numpy as np

import dolphin
from dolphin._types import HalfWindow, Strides
from dolphin.phase_link._core import PhaseLinkRuntimeError, run_phase_linking


def main() -> None:
    stack = np.ones((4, 5, 5), dtype=np.complex64)
    stack[2] = np.nan + 1j * np.nan
    try:
        run_phase_linking(
            stack,
            HalfWindow(1, 1),
            Strides(1, 1),
            calc_average_coh=True,
        )
    except PhaseLinkRuntimeError as error:
        message = str(error)
        if "slc_stack[[2]]" not in message or "all NaNs" not in message:
            raise SystemExit(f"unexpected PhaseLinkRuntimeError: {message}") from error
        print(
            json.dumps(
                {
                    "dolphin_version": dolphin.__version__,
                    "status": "rejected",
                    "exception": type(error).__name__,
                    "message": message,
                },
                sort_keys=True,
            )
        )
        return
    raise SystemExit("pinned dolphin returned output for an all-NaN acquisition")


if __name__ == "__main__":
    main()
