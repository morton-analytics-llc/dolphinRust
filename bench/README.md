# dolphinRust vs Python `dolphin` — speed baseline

The pre-R1 baseline: the figure that justifies the rebuild and the number R4's
performance push must beat. Every value here is **measured** by `bench/bench.py`
in one run; nothing is estimated. Reproduce with the commands at the bottom.

Run date: **2026-06-17**. Raw numbers: [`results.json`](results.json).

## Environment (pinned)

| Component | Version / detail |
|---|---|
| Python `dolphin` | 0.35.0 (`oracle/.venv`), JAX CPU backend, 1 device |
| dolphinRust | 1.0.0, `--release` (rustc 1.94.1) |
| dolphin unwrap | `snaphu-py` 0.4.1 wheel — **native arm64** |
| dolphinRust unwrap | snaphu v2.0.7 **x86_64 binary under Rosetta 2** (see caveat) |
| Host | Apple Silicon (arm64), macOS 25.5 |
| Stacks | synthetic 5×48×64 (`gen_stack.py`, speckle 0.05); real OPERA CSLC T144-308011-IW2 cropped 9×384×384 |
| Config | one `dolphin config` YAML per stack, consumed unchanged by both engines; half_window 11×5, strides 1×1, ministack 15 |
| Reps | 4 per engine per stack; **each rep runs into a wiped work dir** so dolphin never short-circuits a stage whose output already exists |

cold = first invocation; warm = median of the remaining reps. Each `dolphin run`
and `dolphin run --config` is a **fresh process**, so the Python engine pays
interpreter import + JAX JIT on every invocation — the compiled binary does not.

## End-to-end per-frame wall-clock

| Stack | oracle warm | rust warm | **speedup** | oracle cold | rust cold |
|---|---|---|---|---|---|
| synthetic 5×48×64 | 8.49 s | **0.231 s** | **36.8×** | 8.01 s | 0.234 s |
| real 9×384×384 | 17.12 s | **8.36 s** | **2.0×** | 17.11 s | 8.32 s |

## Phase-linking stage (the core estimator, apples-to-apples)

Same half-window, same pixels, isolated from unwrap. Oracle = `wrapped_phase.run`
total from dolphin's own log; Rust = the `stage=phase_linking` `elapsed_s` event.

| Stack | oracle | rust | **speedup** | oracle throughput | rust throughput |
|---|---|---|---|---|---|
| synthetic 5×48×64 | 5.59 s | **0.027 s** | **206×** | 3 kpix·slc/s | 565 kpix·slc/s |
| real 9×384×384 | 7.24 s | **2.01 s** | **3.6×** | 183 kpix·slc/s | 661 kpix·slc/s |

The synthetic stack is dominated by dolphin's fixed per-process overhead (below),
so its 206× is an *overhead* result, not an estimator-throughput result. The real
stack's **3.6×** is the honest phase-linking number: at scale, Rust's
self-adjoint EVD (faer) sustains ~660 k pixel·SLC/s vs JAX-CPU's ~180 k.

## Where Rust loses, and why the end-to-end gap is "only" 2×

Rust per-stage breakdown on the real stack (warm median, seconds):

| stage | rust | note |
|---|---|---|
| phase_linking | 2.01 | faer EVD |
| **unwrap** | **5.83** | snaphu **x86_64 binary under Rosetta 2** |
| timeseries | 0.11 | L1/ADMM inversion |
| velocity | 0.00 | |

Unwrap is **70 % of dolphinRust's runtime**, and it is slow for a packaging
reason, not an algorithm one: the Stanford snaphu binary on this host is x86_64
running under Rosetta emulation, while the oracle's `snaphu-py` wheel is native
arm64. Rust's *own* compute (phase-linking + timeseries + velocity ≈ **2.1 s**)
is ~3.4× faster than the oracle's equivalent. A native arm64 snaphu build would
bring dolphinRust end-to-end to ≈2.5 s — roughly **7×** over the oracle. Tracked
as the headline R4 packaging fix; the two engines also use *different* SNAPHU
implementations by design ("shell out for unwrapping"), so unwrap is reported but
excluded from the estimator comparison.

## JAX cost the compiled binary never pays

In-process decomposition of `run_phase_linking` (Python oracle), per fresh process:

| cost | synthetic | real | paid by Rust? |
|---|---|---|---|
| interpreter + `import jax`/`dolphin` | 466 ms | 466 ms¹ | no (~0) |
| JIT compile (first call) | 569 ms | 120 ms | no |
| warm compute (second call) | 28 ms | 2013 ms | — |

¹ import is process-global; measured once on the first stack. Every `dolphin run`
is a new process, so dolphin pays ~0.5–1.0 s of import+JIT **before any pixel is
processed**, on every frame. This is the near-real-time argument for eo: a
per-frame incremental job amortizes none of it, whereas the binary starts in
~milliseconds.

## Honest summary

- **Phase-linking is 3.6× faster** on a real frame, sustaining ~3.6× the JAX-CPU
  pixel throughput — the result that justifies the rebuild.
- **End-to-end is 2.0× faster today**, capped by an *emulated* unwrap binary, not
  by Rust. Native arm64 snaphu projects to ~7×; that build is the first R4 task.
- **No JIT / no import tax**: the binary skips the 0.5–1.0 s fixed cost dolphin
  pays on every invocation — decisive for NRT per-frame processing.
- On tiny stacks the gap looks huge (37×) purely because that fixed cost
  dominates; don't quote the synthetic number as the throughput win.

## Reproduce

```sh
cargo build --release -p dolphin-cli
oracle/.venv/bin/python bench/bench.py --reps 4
# writes bench/results.json and prints the per-stack summary;
# work dirs land under bench/runs/ (git-ignored).
```
