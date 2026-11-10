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

## Real-data validation: frozen, not run

The protocol is `issue-192-validation-protocol-2026-11.md`, frozen in commit `3ac9bd7`. It has
not been run. The Earthdata bearer token available to this session was rejected (HTTP 401 on a
one-byte authenticated range read of the first granule, 2026-11-10), so no sample of the site
was fetched and neither arm was run. No metric in the protocol has a value.
