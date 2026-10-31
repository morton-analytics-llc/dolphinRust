# Adaptive (data-driven) ministack partitioning (issue #108)

**Date:** 2026-10-31
**Question:** Should `MiniStackPlanner` support coherence-driven ministack boundaries (ASE,
Wang et al. 2025) instead of a fixed `ministack_size`?
**Answer:** Defer. On simulated stacks, ministack size barely matters where the signal is
recoverable. Where it does matter, the effect is set by how phases are referenced across
ministacks, and dolphinRust's referencing differs from current dolphin. Settle that first.

## 1. Sources (verified)

- **ASE:** Wang, Li, Gao, Hu, Jiang, Ren, Zhang, "Adaptive sequential estimator for InSAR time
  series phase estimation", *Int. J. Appl. Earth Obs. Geoinf.* 139, 104552 (2025),
  doi:10.1016/j.jag.2025.104552 (Crossref metadata). The full text is gold OA, but ScienceDirect
  returned HTTP 403 to every fetch, so only the abstract was read (OpenAlex). The abstract says:
  the coherence-weighted matrix is partitioned with Otsu thresholding plus local subset merging;
  a modified SE "selects the optimal subsets from the list with multiple merging degrees"; it was
  tested on 30 Radarsat-2 VV images; and it reports "+7%" DS point density vs SE. **Not
  verifiable from the abstract:** whether partitions are per pixel or per scene, any RMSE, and
  runtime.
- **A scene-wide adaptive variant with numbers:** Zhang et al., S-JSInSAR, *Remote Sens.*
  18(2):329 (2026), doi:10.3390/rs18020329 (PDF from mdpi-res.com). It builds one average
  coherence matrix from bright patches and splits 145 S1 images into 4 subsets of 35/34/49/38
  scenes, overlapping by 4–6 (§2.2, §3.2). It does not use compressed SLCs. See
  `joint-scatterer-sequential-2026-10.md`.
- **dolphin's own sensitivity test:** Staniewicz et al., arXiv:2511.12051v2 §VII-B. Setup: 60
  synthetic SLCs, ρ0=1, ρ∞=0, τ=60 d. "For mini-stack sizes between M=6 and M=30, the mean RMSE
  is relatively stable." It degrades for M>50, and run time is lowest at M≈12–25. Production
  uses M=15.

## 2. Measurement (this session)

Scratch crate `…/scratchpad/research-seq/bench` (not in repo). It generates circular-Gaussian
SLCs with the NRT paper's coherence model, ρ(Δt)=(ρ0−ρ∞)e^{−Δt/τ}+ρ∞ at 12-day sampling,
via a Cholesky factor, times a known phase of 0.15 rad/date. Each stack goes through
`run_sequential` with EMI, rect window, `AlwaysFirst` and `max_num_compressed=10`. The score is
the circular RMSE of the linked phase against truth, both referenced to date 0, over interior
pixels and all dates. The result is mean ± SD across independent realizations. dolphinRust
cannot express variable boundaries today, so this sweep is an upper bound: it shows what *any*
per-regime choice of fixed M could gain.

`research-bench ministack --nslc 60 --rows 48 --cols 72 --reps 8 --hy 5 --hx 10`
(11×21 window):

| regime (ρ0, τ, ρ∞) | M=6 | M=10 | M=15 | M=20 | M=30 | M=60 |
|---|---|---|---|---|---|---|
| urban-like (0.9, 365 d, 0.5) | 0.045±.004 | 0.044±.004 | 0.043±.004 | 0.043±.003 | 0.044±.003 | 0.049±.003 |
| seasonal-low (0.5, 36 d, 0.15) | 0.148±.017 | 0.148±.020 | 0.146±.016 | 0.149±.013 | 0.149±.013 | 0.158±.012 |
| NRT paper (1.0, 60 d, 0) | **0.525**±.045 | 0.704±.057 | 0.825±.070 | 0.771±.068 | 0.550±.048 | 1.122±.145 |
| fast-decor (0.7, 60 d, 0) | 0.625±.047 | 0.792±.064 | 0.877±.078 | 0.813±.065 | 0.635±.054 | 1.095±.148 |
| vegetated (0.6, 24 d, 0) | 1.477±.055 | 1.450±.092 | 1.419±.095 | 1.469±.080 | 1.583±.085 | 1.620±.093 |

The 7×11 window run (`--rows 48 --cols 64 --reps 8 --hy 3 --hx 5`) has the same shape.

- **Where ρ∞>0:** M from 6 to 30 is flat to within one SD of M=15, and M=60 is worse. There is
  nothing for adaptivity to win.
- **Where ρ∞=0:** the best fixed M depends on the regime (M=6 for τ=60 d, M=15 for τ=24 d), and
  the gap from M=15 is large: 0.825 → 0.525 rad on the paper's own parameters. The dependence on
  M is **non-monotonic** (6 good, 15 bad, 30 good). That contradicts the paper's "stable for
  6≤M≤30" on the same model.
- **Where the error comes from:** `research-bench profile --reps 4`, run under the NRT-paper
  regime. RMSE by date for M=15 is 0.26 rad through date 27, then jumps to 0.73 at date 30 and to
  1.31 at date 45. Those are the first dates of ministacks 3 and 4, the first ministacks carrying
  ≥2 compressed SLCs. M=30 has one junction and grows smoothly to 0.80. The jumps at junctions are
  measured. Attributing them to which carried compressed SLC serves as the reference, and how
  coherent it still is, is inference. It has not been tested, because the public API cannot
  change the reference rule.
- **The reference convention differs from current dolphin.** dolphinRust's planner implements
  dolphin v0.35.0's `ALWAYS_FIRST`: output and compressed reference = `output_reference_idx`,
  and `None` maps to `0`, the oldest carried compressed SLC (`dolphin-stack/src/lib.rs:160-169`;
  `dolphin-workflows/src/displacement.rs:1145,2679`). dolphin v0.42.0 `stack.py` (fetched via the
  GitHub API) changed `ALWAYS_FIRST` to `compressed_reference_idx = max(0, num_ccslc-1)`, and with
  `output_reference_idx=None` it uses "the *latest* compressed SLC". PLAYBOOK §Elevated Q3 lists
  `output_reference_idx` as "None (v0.42's)". The field's value matches v0.42, but its meaning is
  v0.35's. The planner contract oracle is v0.35.0 (`dolphin-stack/tests/planner_contract.rs:5-7`),
  which is why the contract never caught it.

## 3. Architecture note

The issue ties this to PLAYBOOK §Elevated Q2 (vertical incremental covariance accumulation).
They are not coupled. A **scene-wide** boundary list keeps every tile on the same plan.
`phase_link_tiled_impl` already sizes the halo from the ministack count
(`displacement.rs:2982`), so the tiled/whole bit-identity holds once `depth` is the planned
ministack count rather than `div_ceil(nslc, ministack_size)`. A **per-pixel** partition, which
ASE may be, is incompatible with compressed SLCs as rasters and with the `depth·(h+s)` halo.
It would mean a different estimator, not a planner option.

## 4. Conclusion — DEFER

Simulation evidence, with no real AOI: adaptivity has nothing to gain where ρ∞>0. Where ρ∞=0 it
competes with an unresolved referencing divergence that produces the same M-sensitivity. Reopen
when **both**:
1. the planner's `AlwaysFirst` reference has been brought to v0.42 semantics, or kept on purpose
   with a recorded reason, and this sweep has been re-run; **and**
2. in a regime present on a validated AOI (MMX1-style GNSS harness), the best fixed M still beats
   M=15 by >10% RMSE and by more than 2 standard errors.

If reopened, build only the scene-wide form: `MiniStackPlanner::plan_boundaries(&[usize])`
(a validated, strictly increasing list of real-SLC starts, each ministack ≥2), and a
`phase_linking.ministack_boundaries: Option<Vec<usize>>` config field. The fixed-size `plan`
becomes the case `plan_boundaries(0, M, 2M, …)`. Contract: the existing planner oracle combos
reproduce bit-for-bit through `plan_boundaries`. `tiled_phase_link_is_bit_identical_to_whole_burst`
must pass with unequal ministacks, which means `depth` comes from the plan. Otsu-style choice of
boundaries is a separate, later step.
