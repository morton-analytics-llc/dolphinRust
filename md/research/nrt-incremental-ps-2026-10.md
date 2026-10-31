# Incremental PS amplitude statistics and small-stack quality metrics for NRT (issue #107)

**Date:** 2026-10-31
**Question:** Should PS amplitude-dispersion statistics update incrementally across ministacks
(as Staniewicz et al. describe), and does dolphinRust need new small-stack quality metrics?
**Answer:** Adopt (a), reframed. The issue's two premises are wrong. PS is not recomputed on
each update: it is not run at all. And the method is not unspecified: the paper gives it and
OPERA ships it. (b) is already built.

## 1. Correcting the issue's premises

- **PS selection is not wired into the pipeline.** `dolphin_ps::calc_ps` and `fill_ps_pixels`
  are called only by `crates/dolphin-ps/tests/ps_contract.rs`. The config disposition marks
  `ps_options.amp_dispersion_threshold` as `compatibility_only`, with the reason "persistent-scatterer
  selection and update inputs are not implemented" (`crates/dolphin-core/src/config.rs:227-230`).
  So "PS recomputing from scratch on every update" is not what happens. README.md:8 ("persistent /
  distributed scatterer phase linking") overstates this.
- **The algorithm is specified, and an oracle exists.** Staniewicz et al., arXiv:2511.12051v2
  §III-A (fetched from arxiv.org/html/2511.12051v2) store "mini-stack amplitude means and
  variances as ancillary data products along with each newly formed compressed SLC". They merge
  the stored and new statistics with weighted group-mean/variance formulas (Eqs. 6–7), using
  either even weights or "an exponential decay". The same statistics feed the GLRT SHP test,
  which "requires only the amplitude means and variances". Shipped code:
  - dolphin `ps.combine_amplitude_dispersions` / `combine_means` (PRs #331/#332, July 2024),
    using `E[X²]−E[X]²` with sample-count weights.
  - OPERA `disp-s1` `src/disp_s1/_ps.py::run_combine` calls it with
    `WeightScheme.EXPONENTIAL`, α=0.5: `N = round(exp(0.5·k)/max)`. `disp_s1/main.py`
    triggers it whenever compressed SLCs are present. With no compressed SLCs it drops the
    threshold to 0.15.
- **SHP is the bigger parity gap.** dolphin `workflows/single.py:94,238-243` passes the PS
  amplitude mean/variance rasters into `shp.estimate_neighbors` (GLRT). Those rasters cover the
  full stack in dolphin, or the combined history in disp-s1. dolphinRust's GLRT (the default
  `shp_method`) uses each ministack's own ~15 real dates: `amplitude.mean_axis` / `var_axis` and
  `nslc = real.dim().0` (`crates/dolphin-workflows/src/sequential.rs:2149-2161`). That is the
  small-sample noise the paper's merged statistics exist to fix.

## 2. Measurements (this session)

Scratch crate `…/scratchpad/research-seq/bench` (not in repo). The batch reference is
`dolphin_ps::calc_ps(stack, 0.25, min_count = N)`.

Synthetic amplitudes (deterministic SplitMix64 seeds):
- 40% Rayleigh DS;
- 50% stationary Rician PS, scale 10^U(0,4), noise-to-signal U(0.02, 0.40);
- 10% PS that decorrelate to Rayleigh at mid-stack.

Commands: `research-bench ad --rows 400 --cols 500 --nslc 60 --ministack 15` (200k px, 4
merges) and `--rows 200 --cols 250 --nslc 300` (50k px, 20 merges).

| incremental variant | worst max\|ΔD_A\| vs batch (N=60 / N=300) | worst rel. | PS flips at 0.25 |
|---|---|---|---|
| Chan pairwise merge, f64 state `(n, mean, M2)` | 0 / 0 (identical at the f32 output) | 0 | 0 / 0 |
| Chan merge, f32 state rounded after every merge | 1.2e-7 / 2.4e-7 | 2.3e-6 / 7.7e-7 | 0 / 0 |
| dolphin `combine_amplitude_dispersions`, f32 group rasters, count weights | 1.29e-5 / 1.78e-5 | 9.1e-4 / 1.2e-3 | 0 / 0 |
| OPERA exponential weights (α=0.5, rounded) | 0.106 / 0.109 (stationary PS) | 0.42 / 0.48 | 4697 of 100k / 1598 of 25k |

- The exact merge matches batch to f32 output precision. dolphin's `E[X²]−E[X]²` form in f32
  loses about 3 digits on low-D_A PS (cancellation). No flips at this threshold, but it is the
  worse formula.
- OPERA's rounded exponential weights are `[0,0,1,1]` for 4 groups and `[0,…,0,1,1]` for 20.
  In effect this is a two-ministack window, not cumulative statistics. That is a deliberate
  adaptivity choice, not an equivalence. 4.7% of stationary PS change classification against
  the 60-date batch.
- **Cost:** batch `calc_ps` on 200k px × 60 dates took 0.029 s, and folding one 15-date
  ministack into the state took 0.006 s. Compute is not the reason to do this. The reasons are
  (1) NRT keeps only `open_real_slcs` of the trailing ministack
  (`sequential.rs:124-145`), so a from-scratch D_A after an update would mean re-reading every
  historical SLC; and (2) SHP and PS quality from more samples.

## 3. (b) Small-stack quality metrics — already present

The paper's "multiple complementary metrics" are γ_t (Eq. 9) plus the phase-similarity metric
γ_s (Eqs. 11–12, the median of neighbour cosine similarity). The paper recommends masking
γ_t<0.6 or γ_s<0.5 (§V-A). dolphinRust has γ_t, CRLB and closure phase. It also has γ_s as
`phase_linking.write_phase_similarity` (`dolphin-phaselink/src/similarity.rs`, Wang et al. 2022
eq. 5, a port of dolphin `similarity.median_similarity`), off by default. Nothing new is needed.
Turning on γ_s or a γ_t/γ_s mask by default is a product-masking decision for GroundPulse.

## 4. Conclusion — ADOPT (a), with exact cumulative statistics; (b) no work

**Assumption everything rests on:** statistics are causal. Ministack k uses dates [0, end of k].
That keeps the v1.4 NRT contract that incremental output is bit-identical to a full rerun.
dolphin's batch mode uses whole-stack statistics, which are non-causal. Matching that would
break the contract.

**Spec**
1. `dolphin-ps`: add `AmplitudeStats { count: Array2<u32>, mean: Array2<f64>, m2: Array2<f64> }`
   with `from_stack(ArrayView3<f64>)`, `merge(&self, &Self) -> Self` (Chan pairwise:
   `δ=μb−μa; n=na+nb; μ=μa+δ·nb/n; M2=M2a+M2b+δ²·na·nb/n`), `variance()`, `dispersion()`, and
   `to_ps(threshold) -> PsResult`. That last one reuses the existing nodata rule: 255 where D_A==0,
   or where `count` is below the number of dates seen, as `calc_ps` does with
   `min_count = N`. Keep `calc_ps` as the batch reference.
2. `dolphin-workflows/src/sequential.rs`: `drive` carries a running `AmplitudeStats`. Per
   ministack, merge in the real dates (`assemble`'s real slice). Pass the cumulative mean,
   variance and count to `estimate_neighbors_glrt` instead of the per-ministack values (in
   `shp_neighbors`, `:2149`). Then run `fill_ps_pixels` on the linked output with the
   cumulative PS mask and the cumulative mean as `avg_mag`. `SequentialState` gains
   `sealed_amplitude: AmplitudeStats` (the stats through the last sealed ministack), and
   `update_sequential` resumes from it.
3. `dolphin-core/src/config.rs`: move `ps_options.amp_dispersion_threshold` from
   `compatibility_only` to `consumed`. Add `phase_linking.shp_amplitude_statistics:
   {ministack, cumulative}`, defaulting to `ministack` until test 4 below is green against a
   dolphin oracle. PS fill is on when `ps_options` is present, which matches dolphin.
4. `displacement.rs`: write `ps_mask.tif`, `amp_dispersion.tif` and `amp_mean.tif` per burst,
   then stitch.
5. Not adopted: OPERA's exponential weights and the first-ministack 0.15 threshold. Both are
   OPERA-SAS heuristics. Reopen if a validated AOI shows non-stationary PS that cumulative
   statistics misclassify.

**Contract tests**
1. `amplitude_stats_merge_matches_batch`: the synthetic mix above at N=60, M=15 and N=300,
   M=15. `max|ΔD_A| ≤ 1e-6` and zero PS flips against `calc_ps`.
2. `nrt_ps_and_shp_bit_identical`: `run_sequential_resumable` + `update_sequential`
   (one-at-a-time and block arrivals, sealed-boundary case) against a full run. Requires
   max|Δ| = 0 on PS mask, D_A, SHP neighbour arrays, cpx_phase and temp_coh.
3. `cumulative_glrt_equals_batch_glrt_on_prefix`: neighbours for ministack k equal
   `estimate_neighbors_glrt(batch mean/var over dates [0,end_k], nslc=end_k)`, exactly.
4. `ps_fill_end_to_end`: a planted bright, stable pixel (D_A=0.05) in a Rayleigh field comes
   out with `temporal_coherence = 1` and its own referenced phase in `run_displacement` output.
   A planted D_A=0.40 pixel does not.
5. Existing `ps_contract` and the `nrt_*` contracts stay green.
