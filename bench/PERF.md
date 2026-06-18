# Phase-linking performance optimization (v1.4.0 Phase 3)

Beats the committed pre-R1 baseline ([`results.json`](results.json)) on the
phase-linking stage — the apples-to-apples estimator core — by a **covariance
hot-path rewrite**, with no accuracy change. Every number below is measured by
the **unaltered** `bench/bench.py` (same scenes, same config, 4 reps); the scenes
were not touched to flatter the result.

Measured **2026-06-18**. Evidence: [`baseline_repro.json`](baseline_repro.json)
(pre-optimization binary) and [`optimized_repro.json`](optimized_repro.json)
(post-optimization), both produced this session on the same host minutes apart.

## The optimization

`dolphin-phaselink::covariance` — the per-pixel sample-coherence matrix, the #1
hot path (one N×N solve per output pixel, 147 k pixels on the real frame). The
old inner reduction was:

```rust
let conj_t = masked.t().mapv(|z| z.conj());   // per-pixel (nsamps×nslc) alloc
let numer  = masked.dot(&conj_t);             // ndarray generic complex matmul
```

ndarray has **no SIMD/BLAS path for `Complex<f64>`**, so `.dot` ran a naive
bounds-checked triple loop, and `conj_t` allocated a conjugate-transpose copy for
every pixel. Replaced with a direct **Hermitian** product that sums only the
upper triangle over contiguous sample rows and mirrors the lower
(`numer[j][i] = conj(numer[i][j])`) — half the multiply-adds, no per-pixel
conjugate-transpose allocation, a tight loop the compiler can vectorize. The
coherence matrix is Hermitian by construction, so the result is identical.

## Result — same-session A/B (host-controlled)

The host ran slower this session than on the 2026-06-17 baseline run (the
pre-optimization binary measured 3.07 s here vs 2.01 s committed), so the rigorous
figure is the **rust-vs-rust A/B on the same host, same session**:

| Stack | rust PL — pre-opt | rust PL — optimized | **speedup** |
|---|---|---|---|
| real 9×384×384 | 3.074 s | **1.292 s** | **2.38×** |
| synthetic 5×48×64 | 0.0352 s | 0.0213 s | 1.65× |

On the real frame, phase-linking throughput rises **432 → 1028 k pixel·SLC/s**
(2.38×). The synthetic stack is overhead-dominated (tiny), so its 1.65× is not an
estimator-throughput result.

**Absolute, vs the committed baseline:** the optimized real-frame phase-linking is
**1.292 s vs the committed 2.008 s** — faster in absolute terms even though this
host ran the *un*-optimized code at 3.07 s today. The win is real under both framings.

**vs Python dolphin:** the oracle's own phase-linking time swings run-to-run today
(7.8–14.4 s for the same frame, JAX/JIT + load sensitive), so the "× over dolphin"
multiple ranges ~6–11×; the host-controlled rust-vs-rust 2.38× is the number to
quote. End-to-end is still gated by the Rosetta-emulated SNAPHU binary (unwrap),
unchanged here — see [`README.md`](README.md).

## No accuracy regression

`covariance_matches_oracle` (≤1e-4 vs dolphin v0.35.0) and every analytic
covariance/EVD/EMI, quality (CRLB/closure, incl. the v0.42 forward oracle), GPU,
and `sign_convention` / `sign_real_data` contract stays green. The matrix math is
unchanged — only the reduction order and allocation pattern differ.

## Reproduce

```sh
cargo build --release -p dolphin-cli
oracle/.venv/bin/python bench/bench.py --reps 4 --out bench/optimized_repro.json
```
