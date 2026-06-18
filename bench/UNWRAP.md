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
| gentle-bowl | **tophu** | **18 195** | **2.345** | **0.124** |
| steep-bowl + decorr-ring (σ=45, amp=90, ~27% γ<0.5) | raw SNAPHU | 20 519 | 6.155 | 0.166 |
| steep-bowl + decorr-ring | **tophu** | **18 563** | **6.112** | **0.150** |

## Conclusion: tophu now beats raw SNAPHU on both scenes

tophu is **≤ raw SNAPHU on all three metrics on both scenes** — the scenes,
parameters, noise model, seeds and metric definitions are unchanged from the
earlier honest-loss measurement; only the *algorithm* changed.

| scene | discont | rms | gross-cycle-err |
|---|---:|---:|---:|
| gentle-bowl | −9.2 % | −3.5 % | −3.4 % |
| steep-bowl + decorr-ring | −9.5 % | −0.7 % | −9.8 % |

The clear, non-noise wins are **discontinuities (−9 % on both scenes)** and
**gross-cycle-error (−10 % on the steep+decorr-ring scene)**. The rms margin is
narrow (−0.7 %) on the steep scene but is still a win, not a regression.

### What changed (the two fixes)

The earlier loss had two named causes; both are now addressed in
`crates/dolphin-unwrap/src/tophu.rs`:

1. **Coherence-weighted coarse pass (was: complex-phasor mean).** The coarse
   multilook now weights each phasor by its correlation, so decorrelated pixels
   no longer drag the block phase toward noise; the block's resulting *vector
   coherence* `|Σ w·z|/Σ w` is the trust map. Coarse blocks below the trust floor
   are masked and filled from trusted neighbours instead of anchoring downstream
   work to garbage. (Unit contract: `coherence_weighted_coarse_tracks_truth_better`.)
2. **Overlap-region inter-tile reconciliation + graph solve (was: per-tile
   snap-to-coarse).** Each adjacent tile pair's integer-cycle offset is estimated
   from the robust median of `phase_b − phase_a` over their *coherent overlap*
   (the difference is an exact multiple of 2π there, since both tiles unwrap the
   same wrapped samples). A **maximum-reliability spanning forest** (Kruskal,
   weight = count of agreeing coherent overlap pixels) propagates offsets only
   through trustworthy overlaps, so a single decorrelated seam cannot shift a
   whole subtree by a cycle. Each connected component is then anchored to the
   coarse reference by one global integer cycle (metric-neutral; all three metrics
   remove a global constant). (Unit contracts: `merge_resolves_planted_2pi_jump`,
   `merge_reconciles_2x2_grid_consistently`.)
3. **Feathered tile merge (was: hard core paste).** Tiles are blended across their
   offset-aligned overlap halos with a weight that is 1 over each tile's core and
   ramps to ~0 at the halo fringe. This is what turned the corner: it removes the
   tile-seam discontinuities (any residual seam step is spread over the halo into
   sub-π increments) and downweights each tile's least-reliable fringe. Without
   it, the inter-tile graph solve alone was still marginally *worse* than raw
   SNAPHU on every metric; with it, tophu wins.

The path the earlier writeup sketched (coherence-weighted coarse + network-based
merge) was the right diagnosis; the feathered seam merge was the additional piece
needed to actually clear SNAPHU.

### Disposition

- tophu now beats raw SNAPHU on these low-coherence scenes and ships as an opt-in
  method (`unwrap_method: tophu`). SNAPHU remains the default (behaviourally
  unchanged) for compatibility and as the correctness reference.
- **When to use which:** prefer **tophu** for large, partly-decorrelated scenes
  (vegetated / fast-subsidence centres) where inter-tile cycle consistency and
  seam continuity matter — it cuts discontinuities ~9 % and gross-cycle errors up
  to ~10 % vs a single global SNAPHU solve. Raw **SNAPHU** is the simpler default
  for small or mostly-coherent scenes where one global MCF already suffices.
