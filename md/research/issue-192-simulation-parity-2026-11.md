# Issue #192: dolphinRust against dolphin v0.42.9 on the simulated stacks

**Date:** 2026-11-10 · **Script:** `oracle/ref_idx_parity.py` with the `ref_idx_parity` example
(`crates/dolphin-workflows/examples/`) · **Oracle:** dolphin v0.42.9 kernels

The experiment of `compressed-reference-index-audit-2026-11.md` (60 dates at 12 days, ministack
15, five compressed SLCs carried, half-window 5, 34×34 interior pixels, 5 seeds), with the
production driver `run_sequential` run end to end on the same stacks under both rules:
`newest` is `CompressedSlcPlan::AlwaysFirst`, `oldest` is `AlwaysFirstLegacy`.

RMS phase error against truth, rad, by ministack, mean of 5 seeds:

| τ (days), γ∞ | rule | engine | ms 1 | ms 2 | ms 3 | ms 4 |
|---|---|---|---:|---:|---:|---:|
| 180, 0 | newest | dolphin | 0.0655 | 0.1178 | 0.1341 | 0.1613 |
| | newest | dolphinRust | 0.0655 | 0.1178 | 0.1341 | 0.1613 |
| | oldest | dolphin | 0.0655 | 0.1178 | 0.1851 | 0.3540 |
| | oldest | dolphinRust | 0.0655 | 0.1178 | 0.1851 | 0.3540 |
| 90, 0 | newest | dolphin | 0.1037 | 0.2118 | 0.2537 | 0.3161 |
| | newest | dolphinRust | 0.1037 | 0.2118 | 0.2537 | 0.3161 |
| | oldest | dolphin | 0.1037 | 0.2118 | 0.5288 | 1.2696 |
| | oldest | dolphinRust | 0.1037 | 0.2118 | 0.5288 | 1.2695 |
| 45, 0.05 | newest | dolphin | 0.1731 | 0.3206 | 0.3347 | 0.3708 |
| | newest | dolphinRust | 0.1731 | 0.3206 | 0.3347 | 0.3708 |
| | oldest | dolphin | 0.1731 | 0.3206 | 0.4646 | 0.5948 |
| | oldest | dolphinRust | 0.1731 | 0.3206 | 0.4646 | 0.5948 |
| 30, 0.1 | newest | dolphin | 0.1886 | 0.2713 | 0.2472 | 0.2690 |
| | newest | dolphinRust | 0.1886 | 0.2713 | 0.2472 | 0.2690 |
| | oldest | dolphin | 0.1886 | 0.2713 | 0.2528 | 0.2742 |
| | oldest | dolphinRust | 0.1886 | 0.2713 | 0.2528 | 0.2742 |
| 20, 0 | newest | dolphin | 0.6327 | 1.4175 | 1.7317 | 1.7660 |
| | newest | dolphinRust | 0.6328 | 1.4177 | 1.7332 | 1.7608 |
| | oldest | dolphin | 0.6327 | 1.4175 | 1.7218 | 1.8538 |
| | oldest | dolphinRust | 0.6328 | 1.4177 | 1.7216 | 1.8536 |

Largest wrapped difference between the two engines under the same rule, over interior pixels
and all seeds, rad:

| τ (days), γ∞ | newest: ms 1–4 | oldest: ms 1–4 |
|---|---|---|
| 180, 0 | 6e-6, 6e-6, 1e-5, 2e-5 | 6e-6, 6e-6, 1e-5, 4e-4 |
| 90, 0 | 5e-6, 9e-6, 2e-5, 4e-5 | 5e-6, 9e-6, 2e-3, 2e-3 |
| 45, 0.05 | 2e-5, 4e-5, 2e-5, 5e-5 | 2e-5, 4e-5, 2e-4, 9e-4 |
| 30, 0.1 | 1e-5, 2e-5, 1e-5, 1e-5 | 1e-5, 2e-5, 2e-5, 2e-5 |
| 20, 0 | 0.28, 1.30, 0.69, 2.33 | 0.28, 1.30, 0.76, 2.50 |

In the four regimes with any coherence left, dolphinRust under the new rule is dolphin v0.42.9
to 5e-5 rad at every pixel, and its error against truth is the same to four decimals. Under the
old rule the two engines agree to 2e-3 rad.

At τ = 20 d with no floor both engines fail under both rules, and individual pixels differ by
radians between engines from the first ministack, where the reference rule plays no part: the
estimate there is an eigenvector of a matrix with no signal, and dolphin's single-precision and
dolphinRust's double-precision solves pick differently. The RMS error still agrees to 0.005 rad.

## Real-data validation: run 2026-11-11, pass, no reduction measured

Protocol `issue-192-validation-protocol-2026-11.md`, frozen in commit `3ac9bd7` and run
unchanged. Receipt:
`validation/results/issue192_reference_rule/houston_t034_071054_iw3_20261110/receipt.json`.
Houston burst T034-071054-IW3, 91 dates, 126,950 output cells valid on every date in both arms.

| Frozen metric | old (`always_first_legacy`) | new (`always_first`) | rule | |
|---|---:|---:|---|---|
| J, junction-step roughness, mm | 1.919 | 1.926 | ≤ 1.05 × old = 2.015 | pass |
| F_int, integer-cycle fraction | 0.1849 | 0.1842 | ≤ old + 0.002 | pass |
| F_link, linked phase over a quarter cycle | 0.2951 | 0.2941 | ≤ old + 0.002 | pass |
| O_mad vs OPERA DISP-S1, mm/yr | 0.414 | 0.438 | ≤ old + 0.1 | pass |
| O_frac within 1.5 mm/yr of OPERA | 0.9354 | 0.9298 | ≥ old − 0.01 | pass |
| G vs GNSS, mean abs error, mm/yr (3 pairs) | 0.825 | 0.720 | ≤ old + 0.25 | pass |

The new rule passes. **Junction steps are not reduced** (J(new) is 0.3% above J(old); the
protocol's bar for a reduction is 5% below). On this urban burst the two rules are the same to
within the tolerances on every metric, which is what the simulation gives where coherence has
a floor.

Not criteria, recorded:

- The control junction (date index 15) is identical in both arms (1.574 mm), and the median
  roughness of non-junction steps is 1.391 mm in both.
- Issue #184 flags (`cycle_steps.flagged_pixels`): 428 old, 391 new, of 155,178 cells.
- The junctions at date indices 45 and 75 carry an integer-cycle disagreement on 51% and 37%
  of evaluable cells **under both rules** (old 0.5067 / 0.3753, new 0.5070 / 0.3740). The
  reference rule does not cause it and does not remove it.
