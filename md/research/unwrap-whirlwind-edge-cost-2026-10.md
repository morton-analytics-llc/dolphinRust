# Whirlwind's joint phase-gradient + coherence edge cost for the native unwrapper (issue #149)

**Date:** 2026-10-31
**Gate (from #149):** (a) read the full Whirlwind paper and extract the exact cost; (b) show a
residue-dense scene where the CRLB-only cost misroutes cuts through steep-but-coherent ground
that a joint model would fix.
**Conclusion:** DEFER. (a) is done (§1). (b) wasn't met: the phase-gradient term bought nothing
measurable in native's solver (§3).

## 1. The cost, from the paper

Staniewicz, Gunter, Mirzaee, Govorcin, Oliver-Cabrera, Fattahi, "A Probabilistic-Cost Algorithm
for Large-Scale 2D Phase Unwrapping," arXiv:2609.36267 (v1, 2026-09-28). #149 credits
"Huang et al."; the authors are as listed here. Full PDF read this session. §II-B:

- **Inputs per pixel edge.** α̂ is the box-filtered wrapped phase gradient along the edge
  (7×7, SNAPHU's window), γ̂ is the *minimum* of the two endpoint sample coherences, and L is
  the effective number of looks.
- **Arc cost.** One +2π correction costs `c(α̂, γ̂, L) = max{0, −ln(P₊/P₀)}` (eq. 4). P₀ and P₊
  are the probabilities that the noisy unwrapped edge gradient lies in (−π, π] or above +π.
- **Noise model.** η is the L-look phase-noise pdf of Lee et al. 1994 (IEEE TGRS 32(5)) at
  true γ. That pdf is marginalized over the true γ with a uniform prior and the sample-coherence
  density of Touzi et al. 1999 (IEEE TGRS 37(1)) (eq. 5). The edge noise is the self-convolution
  of that pdf on [−2π, 2π].
- **Slope marginalization.** The conditional tail and centre integrals q₀, q₊ (eq. 6) are
  marginalized over the true slope α. This uses a flat prior and Gaussian slope error, following
  Carballo & Fieguth 2000 (IEEE TGRS 38(5):2192–2201) eqs. (8), (9), (14), (15) (eq. 7).
- **Directional.** The −1 correction is the +1 cost evaluated at −α̂, so a pixel edge's two arcs
  are priced differently.
- **Quantization and guard.** Costs are integerized as ⌊100c⌋. An empirical slope guard
  zero-costs edges whose raw gradient exceeds 1 rad, capped at 3% of valid edges. Masked edges
  cost 0.

The implementation is `github.com/scottstanie/whirlwind-insar` @ `0ff1df6` (Rust,
BSD-3-Clause OR Apache-2.0). Its `crates/whirlwind-core/src/cost/spline_lut.rs` stores P₀/P₁
on a 31(α) × 21(γ) × 21(L ∈ [1, 300]) grid. The older `isce-framework/whirlwind` (Python, last
commit 2024-10-29, no license) is superseded. dolphin wraps the new one
(`dolphin/unwrap/_whirlwind.py`, `isce-framework/dolphin` @ `6edd7e1`).

**Contrast with native.** `crates/dolphin-unwrap/src/native/cost.rs` prices an edge as the CRLB
precision γ²/(1−γ²) of the *mean* endpoint coherence, clamped at 0.99. The cost is symmetric
and has no gradient or look-count input.

## 2. How it was tested

This isolates the cost from the solver. A scratch crate (not in repo) reproduces native's
global unwrap: `native.rs: unwrap_grid` plus `mcf.rs: Grid`, compiling
`crates/dolphin-unwrap/src/native/simplex.rs` in place via `#[path]`, with per-arc fwd/bwd
costs. With the CRLB cost, the replica's output equals `unwrap_native` (global)
**bit-for-bit** (max |Δ| = 0 on all 9 synthetic runs). Directional costs then map straight
onto native's existing graph. `mcf.rs: dual_arc` already adds separate forward and backward
arcs per edge; it just gives them the same weight. Forward (+1 on `ax`/`ay`) gets c(+α̂) and
backward gets c(−α̂). The c(·) values come from whirlwind-core's embedded table.

Arms:

| arm | cost | solver |
|---|---|---|
| native_global / native_tiled64 | CRLB, shipped | shipped (untiled / production 64-px auto-tiling) |
| replica_carballo_dir | Whirlwind directional, L = 8 | native network simplex |
| replica_carballo_noslope | same table at α̂ = 0 (coherence + looks only) | native network simplex |
| whirlwind_linear | Whirlwind, end to end | Whirlwind SSP, capacity-1 |

**Scenes.** 512², circular-Gaussian 8-look simulation: each pixel forms
`Σ z₁z₂*·e^{iφ}` from L pairs at true γ, with ML sample coherence from the same looks, so the
cost model's own noise assumptions hold. Truth φ is known. Error = % of truth-coherent pixels
(γ ≥ 0.3) whose integer cycle differs from their truth region's modal cycle. 3 seeds each.

- `bench`: `unwrap_bench.rs` geometry, 0.7 coherence with 0.02 moats, 9-cycle cone, ~7.2k
  residues.
- `steep`: γ = 0.6 everywhere except one decorrelated disc. A Gaussian bowl peaking at
  2.4 rad/px puts steep, coherent ground beside a residue source, the case #149 describes.
  ~3.4k residues.
- `patchy`: γ = 0.45 with patchy 0.08 decorrelation and a range slope ramping 0.2→1.6 rad/px.
  ~17k residues.

Command: `N=512 SEEDS=3 RAYON_NUM_THREADS=4 unwrap-proto cost`. Machine: Apple M2 Pro
(Mac14,10), 12 cores, macOS 26.6.2, release build of `25ab07a`. The host was heavily contended
(load ~50), so no wall times are reported for this experiment. Accuracy is deterministic.

## 3. Results

**Synthetic, vs truth (mean of 3 seeds; per-seed in parentheses):**

| scene | native (CRLB) | Whirlwind cost, directional | Whirlwind cost, α̂ = 0 | Whirlwind end-to-end |
|---|---:|---:|---:|---:|
| bench | 0.0009% (.0008 .0008 .0012) | 0.0011% (.0004 .0004 .0025) | 0.0008% | 0.0008% |
| steep | 0.0244% (.0230 .0234 .0269) | 0.0233% (.0214 .0210 .0276) | 0.0241% | 0.0159% |
| patchy | 0.2288% (.2236 .2285 .2343) | 0.2114% (.2018 .2183 .2141) | 0.2073% | 0.1765% |

Native tiled64 matched native global within 0.001 pp on every run.

**Real data, vs SNAPHU (not truth).** Fixture: MMX1/ICMX common frame, Sentinel-1, 352×2217,
final-epoch pair `pair_0011`, from the `native_mmx1_live_contract.rs` fixture in the main
checkout's `validation/runs/gps_mmx1/mmx1_icmx_common/work_snaphu/scratch`. Four 352×554 crops,
global solves. Metric: per-SNAPHU-component cycle disagreement, the contract's own metric.
Command: `RAYON_NUM_THREADS=4 PAIRS=11 LOOKSET=8,30 unwrap-proto mmx1`.

| crop (col₀) | residues | native CRLB | Whirlwind cost L=8 | Whirlwind cost L=30 | Whirlwind end-to-end |
|---|---:|---:|---:|---:|---:|
| 0 | 216 | 0.0944% | 0.0575% | 0.0893% | 0.1098% |
| 554 | 28 | 0.0031% | 0.0031% | 0.0031% | 0.0005% |
| 1108 | 87 | 0.0087% | 0.0072% | 0.0067% | 0.0015% |
| 1662 | 181 | 0.0421% | 0.0385% | 0.0385% | 0.0467% |

**Reading:**

1. **The gradient term, the thing #149 is about, adds nothing measurable.** The scene means of
   directional cost and the same table at α̂ = 0 differ by at most 0.004 pp (per seed,
   ≤0.0033 pp). The sign flips across scenes, including `steep`, the scene built for #149's
   hypothesis.
2. **The coherence-statistics shape gives a small gain on the hard scene.** On `patchy`,
   −ln(P₊/P₀) from the Lee/Touzi pdf beats CRLB precision by 0.017–0.022 pp (mean), with or
   without the slope.
3. **The solver matters more than the cost.** Whirlwind end-to-end beats native-with-its-cost
   by a further 0.035 pp on `patchy` and 0.007 pp on `steep`. That gap is the solver and network
   (capacity-1 SSP, boundary handling), not the cost.
4. **Real data is low-residue and the differences are noise-scale and L-dependent.** All arms
   sit at ≤0.11% against SNAPHU, inside the 0.5% parity gate. SNAPHU isn't truth, so "closer to
   SNAPHU" isn't "more correct".
5. **Every arm is far inside the existing 0.5% parity bar.** None of these scenes shows
   misrouting at a scale anyone would see in a displacement product.

## 4. Two adoption hazards found along the way

- **The look count.** The cost needs L. dolphinRust's production correlation (since #137) is a
  Gaussian-weighted unit-phasor average (σ = 11/3 px, 31-tap,
  `unwrap_backend.rs: interferogram_correlation`). Its independent-sample equivalent is
  1/Σw² = 168.9 looks, but neighbouring phase-linked pixels are correlated, so the true L is
  lower and unknown. MMX1 crop 0 moves from 0.0575% to 0.0893% between L = 8 and L = 30. A
  wrong L shifts every cost, so adoption would need an L-calibration step.
- **The configured cost mode is silently ignored.** `edge_costs(corr, _mode)`
  (`cost.rs:27`) discards `CostMode`. `displacement.rs:4966` maps
  `unwrap_options.snaphu_options.cost` into `NativeConfig.cost`, so `cost: defo`/`topo` with
  the default native backend runs the same CRLB smooth cost with no warning. Any new cost model
  needs this fixed first, as an explicit native cost selection that rejects modes it doesn't
  implement.

## 5. Conclusion: DEFER

**Reason.** In native's own solver, on three synthetic scenes × 3 seeds with known truth, the
Whirlwind cost changes mean per-region cycle error by ≤0.022 pp. The no-slope ablation matches or
beats the directional version, so the joint phase-gradient term #149 proposes showed no effect.
On real MMX1 data the differences are ≤0.04 pp against SNAPHU and depend on an uncalibrated L.
Gate (b) wasn't met. The larger measured gain is the solver, which #147's follow-up
(`unwrap-gpu-solve-2026-10.md` §5, Arm B: `WhirlwindBackend`) would bring along with this cost.

**What would reopen it (any one):**

1. A real scene shows CRLB-cost native misrouting at product scale (≥0.5% per-component
   disagreement vs SNAPHU, or ≥1 cycle vs GNSS). The Whirlwind cost in the native solver must
   fix it **and** α̂ = 0 must not. Candidates: Houston (#138), or an aliased-gradient scene
   (coseismic, glacier).
2. #147 Arm B is adopted. The cost then arrives with the Whirlwind solver and is evaluated as
   part of that backend, not grafted onto network simplex.

**If reopened, the change is small and already mapped:** `cost.rs` returns four arrays
`(wx_fwd, wx_bwd, wy_fwd, wy_bwd)`, and `mcf.rs: dual_arc` takes `(w_fwd, w_bwd)` instead of
one `w`. Add a `NativeCost::{Crlb, Carballo { nlooks }}` selection on `NativeConfig`. The cost
table needs a clean-room derivation from Lee 1994 / Touzi 1999 / Carballo 2000 per §1, or the
BSD/Apache whirlwind-core table with attribution. Contract tests: CRLB stays the default and
bit-identical; the Carballo arm passes `native_dense_parity` / `native_tiling_contract` /
`native_mmx1_live_contract` at ≤0.5%; and a directional unit test checks that an edge with
α̂ = +2.8 rad prices +1 below −1.

## Sources

- arXiv:2609.36267 (full text, §II-A to §II-D, §III, §IV-B, refs [14], [15], [23]–[25]).
- `github.com/scottstanie/whirlwind-insar` @ `0ff1df6`: `crates/whirlwind-core/src/cost/{mod,lut,spline_lut}.rs`,
  `src/lib.rs` (`unwrap_linear`), `LICENSE`.
- `github.com/isce-framework/whirlwind` @ `dce837b` (`src/whirlwind/_cost.py`, superseded).
- `github.com/isce-framework/dolphin` @ `6edd7e1`: `src/dolphin/unwrap/_whirlwind.py`.
- Carballo & Fieguth, IEEE TGRS 38(5):2192–2201, 2000; Lee et al., IEEE TGRS 32(5):1017–1028,
  1994; Touzi et al., IEEE TGRS 37(1):135–149, 1999. Bibliographic data as cited in
  arXiv:2609.36267; not read directly.
- This repo: `crates/dolphin-unwrap/src/native/{cost,mcf,simplex}.rs`,
  `crates/dolphin-unwrap/tests/native_mmx1_live_contract.rs`,
  `crates/dolphin-workflows/src/{unwrap_backend,displacement}.rs`.
