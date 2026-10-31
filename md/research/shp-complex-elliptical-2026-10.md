# Complex-elliptical (shape-based) SHP selection (#109)

**Date:** 2026-10-31 · **Base commit:** `25ab07a` · **Issue:** #109
**Question:** should `dolphin-shp` add the Yao/Frery/Balz angular-consistency filter (ACAF,
a complex angular central Gaussian test on unit-normalised pixel vectors) as a third
`shp_method`?

## Result

**Defer.** The blind spot the paper targets is real and easy to show. dolphinRust's GLRT
keeps 95% of neighbors whose coherence and phase history differ from the center's but whose
amplitude statistics are the same. A CACG shape test rejects 95% of them, and in the
worst-case synthetic it cuts center-pixel phase RMSE from 0.402 to 0.078 rad. On two real crops,
though, the shape test removes only about 8 percentage points of GLRT neighbors beyond its
nominal rate. Removing them changes the EMI phase by a median 0.013 rad (Mexico City) and
0.099 rad (rural POMM). Both are well below the 0.037 / 0.562 rad spread between the two
estimators dolphinRust already ships. The method costs about 150 ms per center pixel in numpy,
against GLRT's few hundred scalar operations, and it carries several hand-tuned constants
with no public reference code.

## What the paper does

Source, fetched and read: Yao, Frery, Balz, "Shape-to-Scale InSAR Adaptive Filtering and
Phase Linking under Complex Elliptical Models", arXiv:2509.12700 (2025-09-16), submitted to
IEEE TGRS.

- Pixels are projected to the unit sphere, `z̃ = z/‖z‖`. Every complex elliptically symmetric
  model then reduces to CACG, and the shape matrix `Σ` is the Tyler fixed point (Eq. 3).
- The statistic is `t = z̃ᴴ Σ̂⁻¹ z̃`. Thresholds come from a per-pixel Monte-Carlo parametric
  bootstrap (Alg. 1, line 3), α = 0.05.
- Selection is layered (§III-A.3). Seed with the N+1 pixels of highest mean temporal
  autocorrelation, iterate a right-sided test up to 10 times, then apply a "mask reversal"
  rule with constants 3, 5, 0.2 and a 0.15 mean-coherence floor (Eq. 6), keep the 4-connected
  component, and finish with one two-sided pass.
- Phase linking is a complex generalised Gaussian MLE or a covariance-fitting MM. That part
  overlaps #106.
- Evaluation: the 2-D simulation classes differ in *both* mean power and coherence (§IV-B). The
  baselines are FaSHPS and CCM-MTF, not GLRT or KS. Real data (TerraSAR-X, Iceland, 37 dates) is
  qualitative. The paper's own power study (Fig. 4) shows the test rejects only about half of
  the heterogeneous pixels when the center pixel has low coherence.
- No code was found (GitHub repository search for "Shape-to-Scale InSAR", "ACAF InSAR",
  "angular central gaussian InSAR": 0 results). dolphin v0.42.9 ships no such method.

## Synthetic test

**Setup.** One 15×29 window (the default `half_window` 7×14) per trial, N = 15, 150 trials per
row. The center pixel is class A (ζ = .95, η∞ = .2). Half of the neighbors are class B
(ζ = .6, η∞ = .05). Both classes have unit mean power, so their amplitude distributions match.
In the *shared* case B has A's phase history. In the *offset* case B carries an extra ramp to
3 rad at the last date. Clutter is Gaussian or K (Gamma ν = 1 texture per pixel). **GLRT and KS
are dolphinRust's `estimate_neighbors_glrt` / `_ks`** (α = 0.001); EMI is dolphinRust's. The
CACG test is a simplified Alg. 1 in numpy: Tyler shape, right-sided, 2000-draw bootstrap,
α = 0.05, at most 10 iterations, started from all pixels with no seeding, reversal or
connectivity step. Seed `20261031`. Run: `python3 -I shp_sim.py`.

| phase | clutter | method | A kept | B kept | center RMSE (rad) |
|---|---|---|---|---|---|
| shared | Gaussian | rect (all) | 1.000 | 1.000 | 0.054 |
| shared | Gaussian | GLRT | 0.876 | 0.944 | 0.072 |
| shared | Gaussian | KS | 0.597 | 0.821 | 0.103 |
| shared | Gaussian | CACG | 0.949 | 0.068 | 0.044 |
| shared | Gaussian | oracle A | 1.000 | 0.000 | 0.038 |
| shared | K ν=1 | GLRT | 0.621 | 0.661 | 0.123 |
| shared | K ν=1 | CACG | 0.946 | 0.067 | 0.064 |
| offset | Gaussian | rect (all) | 1.000 | 1.000 | 0.371 |
| offset | Gaussian | GLRT | 0.875 | 0.950 | 0.402 |
| offset | Gaussian | KS | 0.600 | 0.821 | 0.586 |
| offset | Gaussian | CACG | 0.946 | 0.044 | 0.078 |
| offset | Gaussian | oracle A | 1.000 | 0.000 | 0.038 |
| offset | K ν=1 | GLRT | 0.643 | 0.686 | 0.388 |
| offset | K ν=1 | CACG | 0.949 | 0.047 | 0.087 |

(Full table, KS/K rows included, in `shp_out.md`.)

**Side finding: GLRT and KS over-reject coherent DS.** In the table above GLRT drops 12% of
*identical* class-A neighbors. The isolated check (`glrt_iid_check.py`, 200 windows, seed 3)
shows why:

| temporal coherence of both pixels | GLRT reject | KS reject | nominal |
|---|---|---|---|
| none (Σ = I, iid over time) | 0.0000 | 0.0003 | 0.001 |
| ζ = .6, η∞ = .05 | 0.0007 | 0.0090 | 0.001 |
| ζ = .95, η∞ = .2 | 0.1240 | 0.4042 | 0.001 |

Both tests treat a pixel's N amplitudes as iid. For a stable DS they are strongly correlated
across dates, so the effective sample size is far below N and the chi-squared / Kolmogorov
thresholds are too tight. The implementations match dolphin (`glrt.rs`, `ks.rs` are direct
ports), so this is a property of the reference method, not a dolphinRust bug. In the
shared-phase case it makes GLRT (0.072) and KS (0.103) worse than no SHP selection at all
(0.054).

## Real-data check

Same crops and settings as #106 (`phaselink-cofi-pl-2026-10.md`). GLRT at strides 8×8, then
the CACG test applied *inside* each GLRT set for 300 random centers. Run:
`python3 -I mexico_real.py [scene_dir] [max_dates]`.

| | Mexico City T005 (13 dates) | POMM T137 (15 dates) |
|---|---|---|
| median GLRT SHPs (of 434) | 405 | 427 |
| intensity `E[I²]/E[I]²` in SHP sets, median (Gaussian = 2) | 2.55 | 2.50 |
| implied K ν, median / fraction ν < 5 | 3.65 / 0.68 | 4.03 / 0.67 |
| CACG reject inside GLRT sets, mean / p90 (nominal 0.05) | 0.132 / 0.198 | 0.135 / 0.202 |
| EMI phase change, GLRT vs GLRT ∩ CACG, median / p90 (rad) | 0.013 / 0.030 | 0.099 / 0.296 |
| EMI vs EVD on GLRT sets, RMS (rad), for scale | 0.037 | 0.562 |

Real DS clutter inside GLRT sets is moderately non-Gaussian (ν ≈ 4). There is about 8
percentage points of shape inconsistency in those sets, but excluding it moves the linked
phase by a fifth to a third of the spread between EVD and EMI (0.099/0.562 and
0.013/0.037). Without truth, this cannot
say which set is better. It can only say the change is small.

## Conclusion: DEFER

**Reason.** The test does detect what GLRT and KS cannot, and in the worst-case synthetic the
effect is large. On both real crops the effect on linked phase is below the existing
estimator spread. The cost is two to three orders of magnitude above GLRT per center pixel
(Tyler iterations plus a per-pixel bootstrap), it would sit on the SHP → covariance hot path,
and the published method depends on hand-tuned reversal constants with no reference code to
validate against. The real-data effect does not justify that.

**Evidence that would reopen it.** A GNSS-scored scene (the MMX1/ICMX harness, POMM or another
cohort station) where restricting GLRT sets by a CACG test reduces the station-pixel residual
by more than the station error floor. Or a production AOI where CACG rejection inside GLRT sets
exceeds about 30% over a contiguous area *and* the phase change there exceeds the local
EMI–EVD spread. If reopened, start with the simplified test used here as an opt-in refinement
of the GLRT mask (`shp_method: glrt_shape`). Replace the bootstrap with a tabulated or
asymptotic `t` quantile before any production use. Do not adopt Eq. 6's reversal heuristics
until they are shown to matter.

**Separate, smaller follow-up worth filing.** The temporal-correlation over-rejection above is
measurable and affects coherent DS today: 12% (GLRT) and 40% (KS) false rejections at
ζ = .95. The fix belongs with dolphin parity, as an effective-N correction or a documented
limitation. It is not part of #109.

## Sources

- arXiv:2509.12700, Yao, Frery, Balz 2025: §II-B (CACG), §III-A (Eq. 3–6, Alg. 1), §IV-A
  (Fig. 4 power), §IV-B (simulation design), §V (TerraSAR-X Iceland).
- dolphinRust `crates/dolphin-shp/src/glrt.rs`, `ks.rs`;
  `crates/dolphin-workflows/src/sequential.rs` `shp_neighbors`.

Scratch (not kept):
`/private/tmp/claude-501/-Users-ryanemorton-Documents-GitHub-morton-command-center/f61921b6-8b7b-44be-8f6e-b073da472ff2/scratchpad/research-pl/`
holds `shp_sim.py`, `glrt_iid_check.py`, `mexico_real.py`, the Rust `harness/`, and the
`*_out.md` outputs.
