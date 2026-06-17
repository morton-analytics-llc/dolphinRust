#!/usr/bin/env python
"""dolphinRust-vs-Python-dolphin speed benchmark (the pre-R1 baseline).

Measures, on identical inputs and one shared `dolphin config` YAML per stack:

  * per-frame end-to-end wall-clock for a full `dolphin run` (Python oracle) vs
    `target/release/dolphin run` (Rust), cold (first invocation) and warm
    (median of repeats);
  * the phase-linking stage time for each engine, pulled from each engine's own
    logs (oracle: `wrapped_phase` total; Rust: the `stage=phase_linking`
    `elapsed_s` event emitted under `RUST_LOG=info`);
  * Python dolphin's JAX cost decomposed in-process — interpreter+import, JIT
    compile (first `run_phase_linking` call), and warm compute (second call) —
    so the JIT warm-up the compiled Rust binary never pays is stated separately.

Honest by construction: every number here is measured in this run. Nothing is
assumed. The two engines use different SNAPHU backends (snaphu-py wheel vs the
x86_64 Stanford binary under Rosetta on arm64), so the unwrap stage is NOT
apples-to-apples and is reported but excluded from the phase-linking comparison.

Run inside the pinned oracle env:
  oracle/.venv/bin/python bench/bench.py --reps 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "oracle" / ".venv" / "bin"
RUST_BIN = ROOT / "target" / "release" / "dolphin"
DATASET = "/data/VV"
HALF_WINDOW = (11, 5)  # (x, y) — matches the generated config
MINISTACK = 15

# Oracle "Total elapsed time for dolphin.workflows.wrapped_phase.run : ... (N seconds)"
_ORACLE_PL = re.compile(r"wrapped_phase\.run\s*:.*?\(([\d.]+)\s*seconds?\)")
# Rust tracing event (after ANSI strip): stage="phase_linking" ... elapsed_s=2.0204
_RUST_STAGE = re.compile(r'stage="(\w+)".*?elapsed_s=([\d.]+)')
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_stack(files: list[Path]) -> np.ndarray:
    return np.stack([h5py.File(f, "r")[DATASET][:] for f in files]).astype(np.complex64)


def gen_config(files: list[Path], work_dir: Path, out: Path) -> None:
    subprocess.run(
        [str(VENV / "dolphin"), "config", "--slc-files", *map(str, files),
         "-sds", DATASET, "--work-directory", str(work_dir), "-ms", str(MINISTACK), "-o", str(out)],
        check=True, stdout=subprocess.DEVNULL,
    )


def time_run(cmd: list[str], env: dict | None = None) -> tuple[float, str]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return elapsed, p.stdout + p.stderr


def cold_warm(times: list[float]) -> dict:
    warm = times[1:] or times
    return {"cold_s": times[0], "warm_s": statistics.median(warm), "reps": len(times)}


def fresh(*dirs: Path) -> None:
    """Wipe + recreate work dirs so each rep does the FULL pipeline — dolphin
    skips stages whose outputs already exist, which would make warm runs no-ops."""
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)


def bench_endtoend(
    label: str, oracle_cfg: Path, rust_cfg: Path, oracle_work: Path, rust_work: Path, reps: int
) -> dict:
    oracle, rust = [], []
    o_pl = []
    r_stages: dict[str, list[float]] = {}
    for _ in range(reps):
        fresh(oracle_work)
        dt, log = time_run([str(VENV / "dolphin"), "run", str(oracle_cfg)])
        oracle.append(dt)
        m = _ORACLE_PL.search(log)
        if m:
            o_pl.append(float(m.group(1)))
        fresh(rust_work)
        env = {**os.environ, "RUST_LOG": "info", "NO_COLOR": "1"}
        dt, log = time_run([str(RUST_BIN), "run", "--config", str(rust_cfg)], env=env)
        rust.append(dt)
        for stage, secs in _RUST_STAGE.findall(_ANSI.sub("", log)):
            r_stages.setdefault(stage, []).append(float(secs))
    return {
        "label": label,
        "oracle_total": cold_warm(oracle),
        "rust_total": cold_warm(rust),
        "oracle_phase_linking": cold_warm(o_pl) if o_pl else None,
        "rust_phase_linking": cold_warm(r_stages["phase_linking"]) if r_stages.get("phase_linking") else None,
        "rust_stages_warm_s": {k: round(statistics.median(v[1:] or v), 4) for k, v in r_stages.items()},
    }


def bench_jax_decompose(stack: np.ndarray) -> dict:
    """In-process: import cost, JIT-compile cost, warm compute, for run_phase_linking."""
    t0 = time.perf_counter()
    import jax  # noqa: F401
    from dolphin import HalfWindow, Strides
    from dolphin.phase_link import run_phase_linking
    import_s = time.perf_counter() - t0

    hw, st = HalfWindow(x=HALF_WINDOW[0], y=HALF_WINDOW[1]), Strides(x=1, y=1)

    t0 = time.perf_counter()
    run_phase_linking(stack, half_window=hw, strides=st)  # first call: JIT + compute
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_phase_linking(stack, half_window=hw, strides=st)  # second call: warm compute
    warm_s = time.perf_counter() - t0

    return {
        "import_s": import_s,
        "first_call_s": cold_s,
        "warm_call_s": warm_s,
        "jit_warmup_s": max(cold_s - warm_s, 0.0),
    }


def synth_stack_dir(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(VENV / "python"), str(ROOT / "validation" / "gen_stack.py"),
         "--outdir", str(out), "--speckle", "0.05"],
        check=True, stdout=subprocess.DEVNULL,
    )
    return out


def run_stack(label: str, data_dir: Path, pattern: str, work: Path, reps: int) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise SystemExit(f"no stack files matching {pattern} in {data_dir}")
    oracle_cfg, rust_cfg = work / "config_oracle.yaml", work / "config_rust.yaml"
    gen_config(files, work / "work_oracle", oracle_cfg)
    gen_config(files, work / "work_rust", rust_cfg)

    stack = load_stack(files)
    n, rows, cols = stack.shape
    result = {"shape": {"n": n, "rows": rows, "cols": cols, "pixels": rows * cols}}
    result["endtoend"] = bench_endtoend(
        label, oracle_cfg, rust_cfg, work / "work_oracle", work / "work_rust", reps)
    result["jax"] = bench_jax_decompose(stack)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results.json")
    args = ap.parse_args()

    bench_dir = ROOT / "bench" / "runs"
    results = {}

    synth = synth_stack_dir(bench_dir / "synthetic" / "data")
    results["synthetic_48x64x5"] = run_stack(
        "synthetic_48x64x5", synth, "cslc_*.h5", bench_dir / "synthetic", args.reps)

    real = ROOT / "validation" / "real_data" / "cropped"
    if list(real.glob("OPERA_*.h5")):
        results["real_T144_384x384x9"] = run_stack(
            "real_T144_384x384x9", real, "OPERA_*.h5", bench_dir / "real", args.reps)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
