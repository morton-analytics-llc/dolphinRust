# StaMPS-style iterative PS refinement and proximity weeding (#148)

**Date:** 2026-10-31 · **Base commit:** `25ab07a` · **Issue:** #148
**Question:** should `dolphin-ps` start from a relaxed amplitude-dispersion threshold, refine
candidates by phase stability, and weed PS pixels that are too close together, as StaMPS does?

## Result

**Reject as scoped.** The issue assumes dolphinRust runs single-pass D_A < 0.25 PS selection.
It does not. `dolphin-ps` exists and passes its contract tests, but no workflow calls it, and
the config refuses any PS setting. Refining a selection that is never used has nothing to
improve. Three findings also undercut the method on its own terms:

- StaMPS's own current default turns proximity weeding *off*.
- dolphin's PS-fill design already keeps one PS per output cell.
- StaMPS's "phase-stability evidence" is a spatial-filter and look-angle-error model that
  belongs to a different pipeline, not a refinement step for this one.

## Where PS stands in dolphinRust

- `crates/dolphin-core/src/config.rs:227-230` declares `ps_options.amp_dispersion_threshold`
  as `compatibility_only!` with the reason "persistent-scatterer selection and update inputs
  are not implemented". `config.rs:1629` rejects non-default values.
  `phase_linking.mask_input_ps` is likewise compatibility-only (`config.rs:727-729`).
- `crates/dolphin-workflows/Cargo.toml:19` depends on `dolphin-ps`, but no `.rs` file outside
  `crates/dolphin-ps` references `dolphin_ps` (repo-wide grep). `calc_ps` and `fill_ps_pixels`
  are reached only by `crates/dolphin-ps/tests/ps_contract.rs`.
- So dolphinRust's DS-only output differs from dolphin's whenever dolphin would PS-fill.
  That parity gap comes before any StaMPS extension.

## What StaMPS actually does

Source: `github.com/dbekaert/StaMPS` (GPL-3.0, last push 2023-02-03), files fetched
2026-10-31.

- **Candidate threshold:** `bin/mt_prep_snap:47`: "typical values: 0.4 for PS, 0.6 for SB". The
  issue's D_A < 0.6 is the small-baseline value; the PS value is 0.4.
- **Phase-stability refinement:** `ps_est_gamma_quick.m` iterates γ (`gamma_max_iterations` = 3,
  `gamma_change_convergence` = 0.005; `ps_parms_default.m:78-83`). The model behind γ is a
  spatially band-pass filtered phase (`clap_*`, `filter_grid_size` = 50) plus a per-pixel
  look-angle (DEM) error fitted against perpendicular baselines (`max_topo_err` = 20). It is
  not an "a-priori deformation model". `ps_select.m` then sets the γ threshold from a
  random-phase reference population (`density_rand` = 20 per km² for PS,
  `ps_parms_default.m:110-114`), binned by D_A.
- **Proximity weeding:** `ps_weed.m:40-48` drops adjacent PS only when `weed_neighbours` = 'y'.
  The default has been 'n' since 09/2015 (`ps_parms_default.m:35`, `:155`). Duplicate lon/lat
  pixels are always reduced to the highest-γ one (`ps_weed.m:264-270`).

## Measurements on real crops

dolphinRust `calc_ps` (`min_count` = 1) on the amplitude stacks of the two crops used for
#106/#109. "Cells" are 3(y)×6(x) native-pixel blocks, the DISP-S1-style output strides at
which `fill_ps_pixels` keeps only the brightest PS. Run: `python3 -I mexico_real.py [scene_dir] [max_dates]`.

| scene | D_A threshold | PS pixels (% of crop) | PS with an 8-neighbour PS | cells with ≥ 1 PS | cells with ≥ 2 PS |
|---|---|---|---|---|---|
| Mexico City T005, 13 dates, 384×384 | 0.25 | 9019 (6.12%) | 0.774 | 4062 | 2206 |
| | 0.4 | 52547 (35.64%) | 0.966 | 8134 | 7863 |
| | 0.6 | 129893 (88.09%) | 1.000 | 8192 | 8192 |
| POMM T137, 15 dates, 287×169 | 0.25 | 76 (0.16%) | 0.158 | 69 | 6 |
| | 0.4 | 5639 (11.63%) | 0.717 | 2254 | 1609 |
| | 0.6 | 37707 (77.74%) | 1.000 | 2660 | 2659 |

PS clustering is heavy in the urban crop: 77% of PS have a PS neighbour, as expected from
oversampled bright targets. But fill already collapses each cell to its single brightest PS,
so neighbour weeding at native resolution would mostly remove pixels that fill ignores.
With 13–15 dates, the relaxed thresholds select a third to nearly all of the crop. At that
point any refinement step becomes the real selector. D_A from 13–15 samples is itself a noisy
statistic, and a 15-date ministack is all a single PS pass would see.

## Conclusion: REJECT

**Reason.** There is no PS selection in the dolphinRust workflow to refine. The weeding
StaMPS once did is off by default upstream and is redundant with dolphin's per-cell fill. The
"phase-stability refinement" is StaMPS's γ pipeline (spatial filtering plus perpendicular-baseline
look-angle fitting). That is a separate PS-InSAR method, not an extension of amplitude
dispersion in a DS phase-linking chain, and neither dolphin nor OPERA DISP-S1 uses it.

**What should happen instead (separate issue, not #148).** Wire dolphin's own PS path for
parity: call `calc_ps` per ministack, use the PS mask when filling (`fill_ps_pixels`), and
accept non-default `ps_options.amp_dispersion_threshold`. Contract: on the existing oracle
fixtures, the PS mask and filled `temporal_coherence = 1` cells match dolphin v0.35 exactly.

**Evidence that would reopen #148.** After PS wiring lands: a GNSS-scored or corner-reflector
scene where PS-filled cells carry larger residuals than adjacent DS cells, or a PS-dominated
AOI where the D_A < 0.25 mask has a false-PS rate (cells whose PS phase disagrees with the
DS estimate by > 1 rad RMS) high enough that a γ-style test would plausibly fix it. Reopening
also requires perpendicular baselines per interferogram, which StaMPS's look-angle error term
needs.

## Sources

- StaMPS `bin/mt_prep_snap`, `matlab/ps_parms_default.m`, `ps_select.m`,
  `ps_est_gamma_quick.m`, `ps_weed.m` (`github.com/dbekaert/StaMPS`, master).
- dolphinRust `crates/dolphin-ps/src/lib.rs`, `crates/dolphin-core/src/config.rs`,
  `crates/dolphin-workflows/Cargo.toml`.

Scratch (not kept):
`/private/tmp/claude-501/-Users-ryanemorton-Documents-GitHub-morton-command-center/f61921b6-8b7b-44be-8f6e-b073da472ff2/scratchpad/research-pl/`
holds `mexico_real.py` (PS block), the Rust `harness/` (`calc_ps` mode), `stamps/`, and
`real_*_out.md`.
