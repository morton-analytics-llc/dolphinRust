"""Issue #192 simulation parity: the experiment of `ref_idx_experiment.py`, with
dolphinRust's production sequential driver on the same stacks.

For each of five seeds the simulated stack goes through dolphin's kernels and
through `run_sequential` (the `ref_idx_parity` example), each under both
`always_first` reference rules. Reported per ministack:

- RMS phase error against the known truth, for the four runs;
- the largest wrapped phase difference between dolphinRust and dolphin under
  the same rule, over interior pixels, after referencing both to date 0.

Build the example first, then run in a dolphin >= 0.39 env (the kernels are the
same in both oracle envs; the measurement of record used v0.42.9):

    cargo build --release -p dolphin-workflows --example ref_idx_parity
    PYTHONPATH=<dolphin v0.42.9> oracle/.venv-v042/bin/python oracle/ref_idx_parity.py \\
        TAU_DAYS GAMMA_INF target/release/examples/ref_idx_parity [out.json]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import dolphin

# `ref_idx_experiment` reads TAU_DAYS and GAMMA_INF from argv[1:3] at import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref_idx_experiment as experiment  # noqa: E402

RUST_BIN = Path(sys.argv[3])
RULES = {"oldest": False, "newest": True}


def rust_sequential(stack: np.ndarray, rule: str, workdir: Path) -> np.ndarray:
    stack_path, out_path = workdir / "stack.npy", workdir / f"rust_{rule}.npy"
    np.save(stack_path, stack)
    subprocess.run([str(RUST_BIN), str(stack_path), str(out_path), rule], check=True)
    return np.load(out_path)


def largest_difference(rust: np.ndarray, oracle: np.ndarray) -> list[float]:
    """Largest wrapped phase difference (rad) per ministack over interior
    pixels, both histories referenced to their own date 0."""
    half = experiment.HALF
    interior = lambda a: a[:, half.y : -half.y, half.x : -half.x]
    rust, oracle = interior(rust), interior(oracle)
    difference = np.angle(rust * np.conj(rust[0:1]) * np.conj(oracle * np.conj(oracle[0:1])))
    size = experiment.SIZE
    return [float(np.nanmax(np.abs(difference[s : s + size]))) for s in range(0, experiment.N, size)]


def main() -> None:
    rms = {(engine, rule): [] for engine in ("dolphin", "rust") for rule in RULES}
    difference = {rule: [] for rule in RULES}
    for seed in range(5):
        stack, truth = experiment.simulate(seed)
        with tempfile.TemporaryDirectory() as workdir:
            for rule, newest in RULES.items():
                oracle = experiment.sequential(stack, newest)
                rust = rust_sequential(stack, rule, Path(workdir))
                rms[("dolphin", rule)].append(experiment.error_by_ministack(oracle, truth))
                rms[("rust", rule)].append(experiment.error_by_ministack(rust, truth))
                difference[rule].append(largest_difference(rust, oracle))
    result = {
        "dolphin_version": dolphin.__version__,
        "tau_days": experiment.TAU_DAYS,
        "gamma_inf": experiment.GAMMA_INF,
        "dates": experiment.N,
        "ministack_size": experiment.SIZE,
        "max_num_compressed": experiment.MAXC,
        "half_window": experiment.HALF.y,
        "seeds": 5,
        "rms_phase_error_rad_by_ministack": {
            f"{engine}_{rule}": np.mean(rows, axis=0).round(4).tolist() for (engine, rule), rows in rms.items()
        },
        "largest_rust_minus_dolphin_rad_by_ministack": {
            rule: np.max(rows, axis=0).tolist() for rule, rows in difference.items()
        },
    }
    print(json.dumps(result, indent=2))
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
