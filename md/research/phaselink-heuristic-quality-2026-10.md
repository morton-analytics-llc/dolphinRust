# Estimator-agnostic heuristic quality coefficients (#145)

**Date:** 2026-10-31 · **Base commit:** `25ab07a` · **Issue:** #145
**Question:** should `dolphin-phaselink` add Heimpel, Hajnsek & Frey's goodness-of-fit,
closure-phase and ambiguity coefficients next to temporal coherence, CRLB and closure phase?

## Result

**Defer.** The full paper (v2, 2026-04-21) is now read, so the blocker named in the issue is
cleared. dolphinRust already computes almost all of the inputs: the EMI eigenvalue and
eigengap come out of `estimator.rs`. The noise-floor fit in the paper reproduces against
dolphinRust's EMI, within 0.005. But the coefficients barely track phase error better than
the temporal coherence dolphinRust already writes. For EMI, the goodness-of-fit coefficient's
correlation with error is 0.016–0.026 higher. The ambiguity coefficient is *anti*-correlated
with error for EMI, dolphinRust's default. The closure coefficient roughly ties temporal
coherence. That gain does not pay for a new output layer.

## What the paper defines, mapped onto dolphinRust

Source, fetched and read in full: Heimpel, Hajnsek, Frey, "Heuristic Quality Coefficients for
Interferometric Phase Linking", arXiv:2510.24512v2 (2026-04-21; v1 2025-10-28).

Phase linking is cast as `argmax ⟨vvᴴ, W∘Φ⟩` over the torus (PT) or the unit sphere (ED). Its
Table 1 maps the methods: **EMI = ML-ED** (`W = −|T|⁻¹∘|T|`). dolphin's **EVD** diagonalises
`C ⊙ |C|`, i.e. ED with weights `|T|∘|T|`. That weighting is not in the paper's Table 3; its
bounds follow Prop. B.8's CW-ED argument (`fmax = λmax(|T|∘|T|)`, `fmin = 1`).

- **γ_GOF** (Eq. 3.10–3.12) normalises the achieved objective between `fmin` and `fmax`, then
  against a noise floor `F_noise(N, W)`. The floor is a fitted sigmoid (Eq. 3.13, Table 3). For
  ML-ED, `f = −λmin(|C|⁻¹∘C)`, `fmax = −1`, `fmin = −mean diag(|C|⁻¹)`.
- **γ_CPw** (Eq. 3.16–3.17) is `Σ Re(TijTjkTki) / Σ|TijTjkTki|` over all triplets. dolphinRust's
  `closure.rs` only does nearest triplets.
- **γ_A** (Eq. 3.22) compares the best and second-best solution, with damping μ = 1/3. For ED the
  second eigenvalue stands in, which is already in `PixelEstimate::eigengap`.
- The paper's own simulations (Table 5) report ML-ED ρ_A = −0.104 (exponential scenario). So
  the ambiguity coefficient failing for EMI is the authors' finding too.

## Synthetic benchmark

**Setup.** Heimpel's exponential scenario (η0 = 1, η∞ = .025, ζ = .85) at N = 15. Noise mixing
`(1−ν)C + νI` for ν = 0, .1, …, 1, 300 pixels per level. Sample sizes L = 64 and 435. EMI and EVD
phases, eigenvalues, eigengaps, `estimate_temp_coh` and `estimate_crlb` (num_looks = L) are
**dolphinRust's** outputs from a scratch binary on this worktree. The coefficients are computed
in numpy from those outputs. `F_noise` is a 500-draw Σ = I Monte Carlo at the same (N, L).
The error metric is the paper's `nAE = sqrt(3/(Nπ²))·‖ε‖₂`. Seed `20261031`.
Run: `python3 -I hqc_sim.py`.

Noise-floor replication (ML-ED): MC 0.7479 vs paper sigmoid 0.7524 at L = 64; 0.6610 vs 0.6620
at L = 435.

| L | est | predictor of nAE | Pearson | Spearman | Spearman, nAE < 0.5 | mean abs calibration error |
|---|---|---|---|---|---|---|
| 64 | EMI | 1 − temp_coh (existing) | 0.868 | 0.786 | 0.638 | 0.116 |
| 64 | EMI | CRLB-implied nAE (existing) | 0.801 | 0.783 | 0.640 | 0.180 |
| 64 | EMI | 1 − γ_GOF | 0.884 | 0.789 | 0.645 | 0.120 |
| 64 | EMI | 1 − γ_A | 0.508 | 0.193 | −0.254 | 0.363 |
| 64 | EMI | 1 − γ_CPw | 0.872 | 0.804 | 0.668 | 0.113 |
| 435 | EMI | 1 − temp_coh (existing) | 0.941 | 0.795 | 0.727 | 0.065 |
| 435 | EMI | CRLB-implied nAE (existing) | 0.844 | 0.818 | 0.758 | 0.097 |
| 435 | EMI | 1 − γ_GOF | 0.967 | 0.813 | 0.752 | 0.055 |
| 435 | EMI | 1 − γ_A | 0.313 | −0.321 | −0.719 | 0.475 |
| 435 | EMI | 1 − γ_CPw | 0.934 | 0.816 | 0.755 | 0.061 |
| 435 | EVD | 1 − temp_coh (existing) | 0.942 | 0.775 | 0.701 | 0.067 |
| 435 | EVD | 1 − γ_GOF | 0.958 | 0.789 | 0.719 | 0.057 |
| 435 | EVD | 1 − γ_A | 0.955 | 0.578 | 0.436 | 0.067 |

(Full table, EVD at L = 64 included, in `hqc_out.md`.)

**The normalisation's selling point is a noise floor that holds as N and W change.** Temporal
coherence already behaves nearly that way here. Under Σ = I its mean drifts only 0.353 → 0.324
as L goes 30 → 435 (`hqc_floor.py`, 400 reps, seed 7).

**The [0, 1] bound does not hold for ML-ED on sample matrices.** `F > 1` (λmin(|C|⁻¹∘C) < 1)
occurred on 3.8% / 5.0% / 26.5% of mid-coherence pixels at L = 30 / 64 / 435. At L = 30 the
largest value was F = 50, where `fmax − fmin` nearly vanishes. Any implementation must clamp
and guard that denominator. The paper hints at this ("for ML-ED … the lower bound must be
chosen adaptively", §3.1) but gives no procedure.

## Real-data check

Same crops as #106 (Mexico City T005, 13 dates; rural POMM T137, 15 dates), GLRT SHP sets, EMI.
Run: `python3 -I mexico_real.py [scene_dir] [max_dates]`.

| | Mexico (2024 centers) | POMM (578 centers) |
|---|---|---|
| median temp_coh / γ_GOF / γ_CPw | 0.997 / 0.996 / 0.993 | 0.661 / 0.853 / 0.725 |
| Spearman(temp_coh, γ_GOF) | 0.765 | 0.782 |
| Spearman(temp_coh, γ_CPw) | 0.969 | 0.765 |
| Spearman(temp_coh, −mean CRLB) | 0.840 | 0.476 |
| fraction F > 1 (clamped) | 0.018 | 0.000 |

On real data the coefficients rank pixels differently from temporal coherence (Spearman
0.77–0.78 for γ_GOF). Without truth this cannot show which ranking is right, and in simulation
the two are close.

## Conclusion: DEFER

**Reason.** For EMI, γ_GOF beats `1 − temp_coh` by 0.016–0.026 in Pearson correlation with phase error,
and by 0.007–0.025 in Spearman correlation inside the usable range (nAE < 0.5). γ_A is
anti-correlated with error for EMI, the default estimator. γ_CPw ties or beats temporal
coherence by up to 0.03 Spearman, and dolphinRust has no step that would use its advantage
(screening pixels before phase linking). A new raster layer, a fitted noise-floor table and
clamping rules buy too little. This is a DEFER, not a reject: the inputs are nearly free
(eigenvalue, eigengap, `diag(|C|⁻¹)`), so the cost of building it later stays low.

**Evidence that would reopen it.** (a) A GNSS-scored scene where thresholding on γ_GOF or γ_CPw
selects pixels with measurably lower station residuals than the same-count selection by
`temp_coh`. (b) dolphinRust adding a PT-class estimator (for example LS-MM from #106), where
`temp_coh` and the estimator's own objective diverge more than they do for ED. (c) A consumer
that thresholds quality across stacks of very different N or SHP count, where a fixed
`temp_coh` cut misclassifies. If reopened, the spec is a `quality_gof` raster for EMI and EVD
built in `fused.rs` from the existing eigenvalue and eigengap plus `mean diag(Γ⁻¹)` (EMI) or
`λmax(|C|∘|C|)` (EVD). Clamp it to [0, 1] and return NaN when `|fmax − fmin| < 1e-6`. Ship a
noise-floor table computed by dolphinRust itself for N ∈ 4…30, W ∈ 9…435. The contract test
reproduces the ML-ED Table 3 floor to ±0.01 at (15, 64) and (15, 435). Leave γ_A out for EMI.

## Sources

- arXiv:2510.24512v2, Heimpel, Hajnsek, Frey: Eq. 2.11–2.15, Table 1–3, §3.1–3.3, Table 5,
  Prop. B.8.
- dolphinRust `crates/dolphin-phaselink/src/estimator.rs` (`PixelEstimate::eigenvalue`,
  `eigengap`), `quality.rs` (`temp_coh_single`), `crlb.rs`, `closure.rs`.

Scratch (not kept):
`/private/tmp/claude-501/-Users-ryanemorton-Documents-GitHub-morton-command-center/f61921b6-8b7b-44be-8f6e-b073da472ff2/scratchpad/research-pl/`
holds `hqc.py`, `hqc_sim.py`, `hqc_floor.py`, `mexico_real.py`, the Rust `harness/`, and the
`*_out.md` outputs.
