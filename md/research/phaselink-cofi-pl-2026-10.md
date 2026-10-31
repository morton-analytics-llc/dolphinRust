# COFI-PL covariance-fitting phase linking as a third estimator (#106)

**Date:** 2026-10-31 · **Base commit:** `25ab07a` · **Issue:** #106
**Question:** should `dolphin-phaselink` add COFI-PL (Vu et al. 2024) or its sequential form
S-COFI-PL (El Hajjar et al. 2025) alongside EVD and EMI?

## Result

**Reject as an estimator.** At dolphinRust's default window (435 samples) COFI-PL's
majorization-minimization (MM) refinement does not change accuracy: KL-MM matches EMI and
LS-MM matches EVD to within 0.003 rad RMSE in every simulated case. The method only moves
results with small sample sets (30 to 64 samples), and there it cuts both ways. LS-MM is 6 to 11%
better than EVD at low coherence, but KL-MM is up to 2.3× worse than EMI. On two real crops,
switching estimators moves the phase by less than the existing EMI-vs-EVD spread. The one
thing in the COFI-PL papers that changes answers more than estimator choice is the
**phase-only plug-in**. But at the clutter non-Gaussianity measured on real scenes it is
*worse* than dolphin's coherence, so it is not a follow-up either.

## What COFI-PL is, mapped onto the existing code

Sources, fetched and read in full: Vu, Breloy, Brigui, Yan, Ginolhac, "Covariance Fitting
Interferometric Phase Linking: Modular Framework and Optimization Algorithms",
arXiv:2403.08646 (2024-03-13); El Hajjar, Ginolhac, Yan, El Korso, "Sequential Covariance
Fitting for InSAR Phase Linking", arXiv:2502.09248 (2025-02-13).

COFI-PL fits `|Σ̃| ∘ wwᴴ` to a plug-in estimate `Σ̃`, with `w` on the torus (unit-modulus
entries). It has three modules: plug-in (SCM, Tyler, phase-only "PO" `mean(yyᴴ)`, `y = z/|z|`),
regularization (shrinkage, low rank, banding), and distance (KL, LS = Frobenius, WLS). Vu et
al. §VII gives two MM algorithms:

- **KL-MM** (Alg. 1): `M = |Σ̃|⁻¹ ∘ Σ̃`, iterate `w ← φ((λmax I − M) w)`. This is the
  torus-constrained version of EMI's objective. EMI is the `‖w‖₂` relaxation of the same
  problem (Vu §VII-D).
- **LS-MM** (Alg. 2): `M = |Σ̃| ∘ Σ̃`, iterate `w ← φ(M w)`. `M` is exactly the matrix that
  dolphin's EVD diagonalises (`C ⊙ |C|`, `estimator.rs` `hadamard_abs`), so LS-MM is the
  torus-constrained EVD.

So COFI-PL as an estimator is a projected power iteration started from the EMI/EVD
eigenvector. It is not a new family. The sequential paper keeps the same MM step and only
splits the problem into a past block and a new block.

Corrections to the issue text:
- There **is** a reference implementation: `gginolhac/InSAR_COFI-PL` (Python, MIT;
  `src/optimization.py` `MM_KL_IPL`, `MM_LS_IPL`) and `DanaElhajjar/S-COFI-PL` (Python, no
  licence file). Missing upstream code is therefore not the blocker.
- "Validated by the authors on Sentinel-1-scale stacks" overstates it. Both papers' real-data
  sections (Mexico City, 31 and 40 S1 dates, 8×8 window) are visual comparisons with no
  error numbers. The quantitative results are simulations at Toeplitz `ρ = 0.98`
  (El Hajjar §IV), which is far more coherent than DS pixels.
- dolphin v0.42.9 (2026-10-27) has no covariance-fitting, Tyler or phase-only code (GitHub
  code search for `cofi`/`tyler` in `isce-framework/dolphin`: 0 hits).

## Synthetic benchmark

**Setup.** N = 15 (default `ministack_size`), 1000 Monte-Carlo pixels per row, phases
U(−π, π], `|Σ|ij = (1−η∞)ζ^|i−j| + η∞`. The clutter is Gaussian, or K with per-pixel
Gamma(ν=1) texture shared across dates (the El Hajjar §IV model). The plug-in is
dolphin's coherence `C` (`dolphin-phaselink/CLAUDE.md`). **EVD and EMI are dolphinRust's
own `process_coherence_matrices`** (β = 0, threshold 0), run from a scratch binary built
against this worktree. KL-MM and LS-MM follow Vu Alg. 1 and 2 in numpy, initialised at the Rust
EMI/EVD phase, up to 200 iterations. The metric is wrapped phase RMSE (rad) over dates 1–14,
referenced to date 0. Seed `20261031`.
Run: `python3 -I cofi_sim.py` (scratch dir, below). Rust-vs-numpy eigenvector check on
one fixture: `|⟨v_rust, v_py⟩|/n = 1.0` for both EVD and EMI.

| coherence | L | clutter | EVD | LS-MM | EMI | KL-MM | EMI on PO plug-in |
|---|---|---|---|---|---|---|---|
| ζ=.95, η∞=.2 | 30 | Gaussian | 0.112 | 0.113 | 0.113 | 0.113 | 0.170 |
| ζ=.95, η∞=.2 | 435 | Gaussian | 0.028 | 0.028 | 0.026 | 0.026 | 0.042 |
| ζ=.95, η∞=.2 | 435 | K ν=1 | 0.041 | 0.041 | 0.037 | 0.037 | 0.042 |
| ζ=.85, η∞=.05 | 30 | Gaussian | 0.279 | 0.281 | 0.301 | 0.407 | 0.379 |
| ζ=.85, η∞=.05 | 30 | K ν=1 | 0.444 | 0.427 | 0.462 | 1.051 | 0.409 |
| ζ=.85, η∞=.05 | 64 | Gaussian | 0.187 | 0.188 | 0.182 | 0.184 | 0.235 |
| ζ=.85, η∞=.05 | 435 | Gaussian | 0.066 | 0.067 | 0.055 | 0.055 | 0.076 |
| ζ=.85, η∞=.05 | 435 | K ν=1 | 0.099 | 0.100 | 0.085 | 0.085 | 0.076 |
| ζ=.6, η∞=.05 | 30 | Gaussian | 0.879 | 0.791 | 0.922 | 0.912 | 0.999 |
| ζ=.6, η∞=.05 | 64 | Gaussian | 0.507 | 0.466 | 0.506 | 0.490 | 0.616 |
| ζ=.6, η∞=.05 | 64 | K ν=1 | 0.837 | 0.741 | 0.817 | 0.788 | 0.584 |
| ζ=.6, η∞=.05 | 435 | Gaussian | 0.142 | 0.145 | 0.126 | 0.127 | 0.161 |
| ζ=.6, η∞=.05 | 435 | K ν=1 | 0.221 | 0.221 | 0.211 | 0.213 | 0.168 |

(Full 18-row table in `cofi_out.md`, which also has LS/KL-MM on the PO plug-in. They track
EVD/EMI on PO the same way they track them on `C`.) KL-MM did not reach `‖Δw‖ < 1e-9` in 200
iterations in any cell; the `λmax` shift makes its steps short. LS-MM converged in a median of
10 to 172 iterations.

**PO plug-in at the measured texture.** The real crops below have implied K-texture
ν ≈ 3.6–4.0 inside GLRT SHP sets. The rerun at L = 405, ν ∈ {4, 10}, 600 pixels, seed
`20261101` (`cofi_nu.py`) gives:

| coherence | clutter | EMI on C | EMI on PO | EVD on C | EVD on PO |
|---|---|---|---|---|---|
| ζ=.95, η∞=.2 | K ν=4 | 0.033 | 0.045 | 0.035 | 0.041 |
| ζ=.95, η∞=.2 | K ν=10 | 0.029 | 0.041 | 0.032 | 0.039 |
| ζ=.6, η∞=.05 | K ν=4 | 0.160 | 0.171 | 0.180 | 0.180 |
| ζ=.6, η∞=.05 | K ν=10 | 0.149 | 0.179 | 0.169 | 0.191 |

PO only wins at ν = 1, a heavier texture than either real crop shows.

## Real-data check

Crops already on disk under `dolphinRust/validation/real_data/`:
- Mexico City `T005-008704-IW1` 384×384, 13 dates 2023-01-04…06-09.
- Rural Parkfield `t137_292329_iw1_land_pomm` 287×169, first 15 dates 2023-01-01…06-06.

dolphinRust GLRT SHP (half window 7×14, α = 0.001) at output strides 8×8; EMI/EVD from the
harness. Run: `python3 -I mexico_real.py [scene_dir] [max_dates]`. There is no truth, so the
table reports how far each change moves the phase, with the EMI–EVD spread as the reference:

| RMS wrapped phase difference (rad) | Mexico (2024 centers, median temp_coh 0.997) | POMM (578 centers, median temp_coh 0.661) |
|---|---|---|
| EMI vs EVD (two shipped estimators) | 0.037 | 0.562 |
| EMI vs KL-MM | 0.005 | 0.202 |
| EVD vs LS-MM | 0.009 | 0.501 |
| EMI on C vs EMI on PO | 0.078 | 0.748 |

Median GLRT SHP count is 405 (Mexico) and 427 (POMM). Only 0.4% / 0% of centers have fewer
than 64 SHPs, the regime where LS-MM helped in simulation.

## Conclusion: REJECT

**Reason.** As an estimator, COFI-PL adds nothing at production sample sizes: the RMSE
difference is ≤ 0.003 rad at L = 435 in every simulated case. The small-L gain belongs to LS-MM
only, applies to under 0.5% of real centers, and comes with KL-MM's 2.3× loss at L = 30 if the
"ML" variant is picked. The plug-in option is what moves results on real data. At the measured
texture it raises RMSE by 7–41% relative to EMI on dolphin's coherence, so it does not justify
a follow-up either. S-COFI-PL's sequential form solves a problem dolphinRust's ministack
compression already solves, and its accuracy evidence is a single ρ = 0.98 simulation.

**Evidence that would reopen it.** GNSS-scored real data (the MMX1/ICMX harness, or a POMM-style
rural station) where EVD/EMI residuals exceed the station error floor in pixels with fewer
than 64 SHPs, and where LS-MM on the same pixels reduces the residual. Alternatively, a
production configuration that drives median SHP counts below about 64 (much smaller
`half_window`, or a stricter SHP test). If reopened, the minimal design is an opt-in LS-MM
refinement pass after EVD (`w ← φ((|C|∘C) w)`, start at the EVD vector, stop at
`‖Δw‖ < 1e-6` or 50 iterations). Its oracle is `MM_LS_IPL` from `gginolhac/InSAR_COFI-PL`. KL-MM
should not be built without regularization: unregularized `|C|⁻¹` is what produced the
L = 30 failures.

## Notes for the code

- `crates/dolphin-phaselink/CLAUDE.md:21` says EMI does "Cholesky-invert with 1e-6 jitter".
  Neither `estimator.rs` `invert_spd` nor dolphin v0.35 `_core.py:441` adds jitter; only CRLB
  does (`crlb.rs` `GAMMA_JITTER`). The code matches dolphin, so the doc line is the error.
- dolphin selects the EMI eigenvalue nearest μ = 0.99 by shifted inverse iteration
  (`_eigenvalues.py` `eigh_smallest_stack`). dolphinRust takes the true minimum (`argmin`). In
  500-draw simulations the two disagree on 2.0–2.2% of matrices at L = 30 and on 0% at
  L ≥ 64 (`emi_shift_check.py`). This is a known, sanctioned difference (CLAUDE.md allows direct
  eigensolvers), and it only shows up at very small SHP counts.

## Sources

- arXiv:2403.08646, Vu et al. 2024: §IV (PO plug-in, Eq. 19), §VII Alg. 1–2, §X (visual-only
  real data).
- arXiv:2502.09248, El Hajjar et al. 2025: §IV simulation (ρ = 0.98, Gamma ν = 1); code
  link on p. 1.
- `github.com/gginolhac/InSAR_COFI-PL` `src/optimization.py`, `src/covariance_estimators.py`
  (fetched 2026-10-31).
- dolphin v0.35.0 `src/dolphin/phase_link/_core.py`, `_eigenvalues.py`; dolphin latest
  release v0.42.9.

Scratch (not kept in the repo):
`/private/tmp/claude-501/-Users-ryanemorton-Documents-GitHub-morton-command-center/f61921b6-8b7b-44be-8f6e-b073da472ff2/scratchpad/research-pl/`.
It holds `harness/` (the Rust binary: `process_coherence_matrices`, `estimate_temp_coh`,
`estimate_crlb`, GLRT/KS, `calc_ps`), `common.py`, `cofi_sim.py`, `cofi_nu.py`,
`mexico_real.py`, `emi_shift_check.py`, and the `*_out.md` outputs.
