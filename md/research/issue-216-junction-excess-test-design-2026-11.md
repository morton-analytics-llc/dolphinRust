# Issue #216 test design: why every ministack junction on Houston T034 carries 2.5–5× the half-cycle local steps of ordinary dates

**Frozen 2026-11-12**, in the commit that adds this file, after the #192 inputs were
re-fetched and before any arm below was run or scored. It freezes the arms, the statistic
and the decision rules. Engine configuration is that of #192
(`validation/issue192_houston_recipe.json`) except for the one variable each arm changes.
Nothing below is changed after results are seen; anything learned while running goes in a
dated addendum.

## Finding under test

#207 (receipt `validation/results/issue207_junction_steps/houston_t034_071054_iw3_20261112/receipt.json`,
key `fix`) measured, on the published series of the recipe run, the fraction of cells whose
12-day step is more than half a cycle from the median of its same-component neighbours
within 6 px. At the six ministack junctions (acquisition indices 15, 30, 45, 60, 75, 90) it
is 0.000188–0.000791, 24–101 cells each, 371 in all. Over the 84 other steps it is median
0.000047, p95 0.000153, max 0.000258. The #207 acceptance bound fails on this population.

## Held fixed

Same 91 granules (plan `catalog_sha256` 9226039c…, window row0 2704, col0 1266, 699 × 3996),
same recipe engine configuration, binary built from main 014be98, the #192 driver
(`validation/issue192_reference_rule.py run`), its `always_first` arm (`work_new`) unless
stated.

## The statistic

For arm X and step k (1..90), `S_X(k)` is the fraction of judged cells with
`|r| > 0.5` cycle, where `r` is the cell's step `phi_k − phi_{k−1}` (published
`displacement_*.tif`, cycles at `−λ/2`, acquisition 0 as zero) minus the upper median of
its finite same-component neighbours within half-window `h` (labels from the `(k−1, k)`
interferogram's `conncomp_*.tif`), judged where at least 4 such neighbours exist. This is
`local_residuals` in `validation/issue207_junction_steps.py`, the statistic of the #207
`like_for_like_junction_window_all_steps`. `h = 6` unless stated. `n_X(k)` and `e_X(k)` are
the over-half and judged cell counts.

`J0 = {15, 30, 45, 60, 75, 90}` (the recipe's junctions). `J_X` is arm X's own junction
set (`ministack_junctions` in its provenance). The ordinary steps of X are
`O_X = {1..90} − J0 − J_X`, and `p95_X` is the 95th percentile of `S_X` over `O_X`.

The **excess** of X at a step set K, in cells:
`E_X(K) = Σ_{k∈K} (n_X(k) − p95_X · e_X(k))`. Baseline `E_A0 = E_A0(J0)`.

A check **explains most of the excess** when its arm brings the excess at the junction
epochs to at most half the baseline, `E ≤ 0.5 · E_A0`.

## Arms

| arm | the one change from the recipe | junctions |
|---|---|---|
| A0 | none (`always_first`, native unwrapper, ministack 15) | J0 |
| A0L | `always_first_legacy` (the driver's `work_old`, same run as A0) | J0 |
| R | `phase_linking.compressed_slc_plan: last_per_ministack` | J0 |
| A1 | `unwrap_options.unwrap_method: snaphu` | J0 |
| B | `phase_linking.ministack_size: 91` (one ministack, no compressed SLC) | none |
| C | `phase_linking.ministack_size: 13` | 13, 26, 39, 52, 65, 78 |

## Measurements, in order

**M0 — reproduction.** A0 on the re-fetched inputs. Pass: `n_A0(k)` within 2 cells of the
#207 `residual_vs_184_flags` counts (77, 24, 64, 101, 48, 57) at every junction and `p95_A0`
within 10% of 0.000153. If not, stop and report; nothing else is scored.

Then the four candidate causes, cheapest first. **Stop at the first that explains most of
the excess**; the remaining arms are reported if they were run, but do not change the
finding.

**C1 — the diagnostic's window.** Offline on A0, `S` recomputed at `h ∈ {3, 9, 12}`, each
with its own `p95`. The window is the cause if `E_A0^{h=9}(J0) ≤ 0.5 · E_A0` and
`E_A0^{h=12}(J0) ≤ 0.5 · E_A0`. (A wider window cannot hide a patch that a 6 px window sees;
it can only absorb a reference the 6 px median gets wrong. `h = 3` is reported only: a 7 × 7
median moves with a patch about a look window wide, which would hide a real excess.)

**C2 — the reference date carried across ministacks.** The reference is the cause if
`E_R(J0) ≤ 0.5 · E_A0` or `E_A0L(J0) ≤ 0.5 · E_A0`.

**C3 — unwrapping at the junction interferograms.** The native unwrapper is the cause if
`E_A1(J0) ≤ 0.5 · E_A0`.

**C4 — the compressed-SLC handoff.** The handoff is the cause if both
`E_B(J0) ≤ 0.5 · E_A0` (with no handoff, the same dates are ordinary) and
`E_C(J_C) ≥ 0.5 · E_A0` with `E_C(J0) ≤ 0.5 · E_A0` (the excess moves with the boundary,
not with the dates).

None of C1–C4 → no finding; the measurements are reported as they stand.

## If C4 holds: estimator property or port defect

The sequential estimator puts the compressed SLC's phase error into every date of the next
ministack, so a datum step at each junction is expected of dolphin itself (#184 module notes
cite v0.35.0 on Fresno; no measurement on this site). Two measurements, both frozen here:

**M5 — OPERA DISP-S1 at the same cells.** The #207 OPERA steps (frame F08882 v1.0, newest
production per pair, `recommended_mask = 1`, nearest-neighbour onto the output grid, sign
fixed by the median Pearson correlation over ordinary comparable steps). OPERA is dolphin
in production, run on its own ministack schedule. At each junction `k ∈ J0` whose two
acquisitions share an OPERA reference date, for the A0 over-half cells `F_k` that OPERA
covers:

- `D_A0(k)` — fraction of `F_k` with `|step_A0 − step_O − median_frame(step_A0 − step_O)| > 0.5`;
- `D_B(k)` — the same at the same cells with B in place of A0;
- `S_O(k)` and `S_O` over OPERA's ordinary comparable steps — the statistic on the OPERA step,
  one component, `h = 6`.

If `D_A0 ≥ 0.5` while `D_B ≤ 0.1` pooled over the comparable junctions, the flagged cells are
the handoff's datum error, absent from an independent estimate of the same steps; they are
not deformation. If `D_A0 ≤ 0.1`, OPERA carries the same steps at the same cells: they are in
the data, or produced by dolphin's estimator alike. Anything between is reported as measured.

**M6 — dolphin's own estimator on these inputs.** A dolphin-parity check of the handoff on
real data requires dolphin v0.42.9 on these 91 granules. The oracle environments in this
repository are v0.35.0 and v0.42.0. If neither the installed v0.42.0 nor a v0.42.9
environment can run the recipe configuration, M6 is recorded as not run, and the finding
rests on C4 and M5.

A port defect is established only by a measured difference from dolphin (M6), or by a
located deviation in `sequential.rs` / `compress` from dolphin's `_compress.py` found while
reading the code for this issue. If one is found: fix it, add a test pinned to the
behavior, and Houston must be not worse than #192 on the frozen velocity metrics (median
`|velocity − OPERA|` ≤ 0.414 + 0.1 mm/yr, fraction within 1.5 mm/yr ≥ 0.935 − 0.01) with
`E_A0(J0)` at most half the baseline. If none is found, the excess is recorded as a
property of the sequential estimator with the C4 and M5 numbers, and published displacement
is unchanged.

Receipt: `validation/results/issue216_junction_excess/houston_t034_071054_iw3_<date>/receipt.json`.
Scorer: `validation/issue216_junction_excess.py`. Eo #758 (Fresno T137 junction patches) is
the same population on another site; Fresno is closed as a validation site and is not run.
