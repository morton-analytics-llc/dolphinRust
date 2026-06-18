# Unwrapping benchmark — tophu multi-scale vs raw SNAPHU

Honest measurement of the v1.2.0 tophu multi-scale unwrapper against the raw
single-pass SNAPHU path on large, low-coherence (vegetated / decorrelated)
synthetic scenes with a **known truth phase field**. The scenes and their
parameters are fixed up front and were **not** swept to favour either method.

Reproduce:

```sh
cargo test -p dolphin-unwrap --test tophu_bench -- --ignored --nocapture
```

(Requires `snaphu` on `PATH`. The harness lives in
`crates/dolphin-unwrap/tests/tophu_bench.rs`; the scene is a Gaussian subsidence
bowl + linear ramp under a coherence map with low-γ vegetation patches and an
optional central decorrelation ring. Phase noise scales with the per-pixel CRLB
`σ_φ = sqrt((1−γ²)/(2·N_L·γ²))`, `N_L=4`. Deterministic splitmix64 noise → the
scene is identical every run.)

## Metrics

- **discont** — count of adjacent valid-pixel pairs whose *unwrapped* difference
  exceeds π. A correctly unwrapped continuous field has essentially none; lower
  is better.
- **rms** — RMS of (unwrapped − truth) after removing the best global constant
  offset (unwrapping is determined up to an additive constant). Lower is better.
- **gross-cycle-err-frac** — fraction of pixels whose referenced error exceeds π
  (i.e. landed on the wrong integer cycle). Lower is better.

## Results (512×512, snaphu v2.0.7)

| scene | method | discont | rms (rad) | gross-cycle-err-frac |
|---|---|---:|---:|---:|
| gentle-bowl (σ=80, amp=60, ~26% γ<0.5) | raw SNAPHU | 20 049 | 2.430 | 0.129 |
| gentle-bowl | **tophu** | 20 601 | 2.826 | 0.169 |
| steep-bowl + decorr-ring (σ=45, amp=90, ~27% γ<0.5) | raw SNAPHU | 20 519 | 6.155 | 0.166 |
| steep-bowl + decorr-ring | **tophu** | 21 283 | 6.703 | 0.208 |

## Honest conclusion: tophu does **not** beat raw SNAPHU on these scenes

On both low-coherence scenes raw SNAPHU genuinely struggles (gross-cycle-error
0.13–0.17), but our tophu multi-scale path is **modestly worse** on every metric,
not better. We are not hiding this and we did not tune the scene or weaken any
tolerance to manufacture a margin.

### Why (hypothesis)

tophu's benefit comes from a *reliable* coarse initialization. Our coarse pass
multilooks **complex phasors** over `downsample_factor` blocks; in decorrelated
regions those phasors are near-random, so the block average has small magnitude
and an unreliable phase. Anchoring each fine tile to that noisy coarse reference
(by a single integer-cycle offset = the mean residual to coarse) then **injects**
error in exactly the low-γ areas the scene is full of, rather than removing it.
Two compounding factors:

1. **Coarse anchor poisoned by decorrelation.** Where γ is low the coarse phase
   is not trustworthy, so the per-tile 2π anchor is sometimes set to the wrong
   cycle for part of a tile that straddles coherent and decorrelated ground.
2. **Mean-offset tile merge is cruder than SNAPHU's global MCF.** Our merge
   reconciles each tile with one constant cycle offset (the load-bearing
   "comparative, non-unique merge" decision); SNAPHU solves the whole scene's
   cost network jointly and handles the same decorrelation better.

This is consistent with the contract tests: on **coherent** data tophu matches
the raw-SNAPHU envelope (`tophu_recovers_analytic_ramp_within_snaphu_envelope`,
`tophu_coarse_pass_round_trips_ramp`) and correctly resolves a planted inter-tile
2π jump (`merge_resolves_planted_2pi_jump`). The implementation is **correct**;
its simplified coarse-anchor + mean-merge heuristic simply does not outperform
SNAPHU's global solver once large parts of the scene decorrelate.

### Disposition

- tophu ships as a **correct, opt-in** method (`unwrap_method: tophu`); **SNAPHU
  stays the default** and is behaviourally unchanged.
- We do **not** claim a quality win. On the evidence here, prefer raw SNAPHU for
  low-coherence scenes; tophu's value is the multi-scale strategy for cases where
  a reliable coarse trend exists (gentle, mostly-coherent large scenes).
- Likely improvement path (future, not done here, to avoid scene-tuning the v1.2
  result): coherence-weighted coarse multilook + trust the coarse anchor only
  where the coarse magnitude/γ is high, and a network-based tile merge in place
  of the constant mean-offset snap (closer to upstream tophu's merge).
