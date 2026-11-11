# Issue #192 real-data validation protocol: the two `always_first` reference rules

**Frozen 2026-11-10**, in the commit that adds this file, before any CSLC sample of this
site was fetched for this test and before either arm was run. It freezes **the site, the
inputs, the metrics, the tolerances and the pass rule**. It does not freeze engine
configuration: a configuration may be swept in-session, always in both arms at once, and
every sweep is recorded in the receipt. A failure is a finding, reported with the controlled
measurement that isolates its cause. Nothing below is changed after results are seen.

Machine-readable form of everything the scorer uses:
[`validation/issue192_houston_recipe.json`](../../validation/issue192_houston_recipe.json).
Driver and scorer: [`validation/issue192_reference_rule.py`](../../validation/issue192_reference_rule.py).

## Question

`CompressedSlcPlan::AlwaysFirst` now references each ministack to the newest carried
compressed SLC (dolphin v0.42.9), where it used the oldest (dolphin v0.35.0, kept as
`always_first_legacy`). In simulation the old rule is the noisier from the third ministack on
(`compressed-reference-index-audit-2026-11.md`). On real data: is the new rule at least as good
as the old on every metric below, and does it reduce the steps at ministack junctions?

## Arms

One binary, built from this branch, run twice on identical inputs. The two configs differ in
one line: `phase_linking.compressed_slc_plan` is `always_first_legacy` (**old**) or
`always_first` (**new**).

## Site and inputs

Houston, the west–east site GroundPulse selected on data completeness on 2026-11-03 (OPERA
DISP-S1, a complete Sentinel-1A series, NGL GNSS, urban coherence). Fresno is closed as a
validation site and is not used.

| | |
|---|---|
| Burst | OPERA L2 CSLC-S1 `T034-071054-IW3` (ascending track 34) |
| Dates | the 91 secondary dates of OPERA DISP-S1 frame F08882, 2022-01-11 to 2025-01-31, listed in the recipe; one granule per date, the latest production where a date has several |
| Extent | UTM 15N, E 255000–275000 m, N 3296500–3303500 m (20 km × 7 km), snapped to the burst grid and trimmed to a whole number of output cells. One burst, because 38 GiB of disk was free; the crop is range-read (`remote_hdf5_crop.py`), capped at 64 MiB per granule |
| Ministacks | 15 dates each, so new ministacks start at date indices 15, 30, 45, 60, 75 and 90. The two rules are the same function through the second ministack, so index 15 is the **control** junction and 30, 45, 60, 75, 90 are the **affected** junctions |
| Starting configuration | GroundPulse's adopted one: half-window (6, 12), strides (3, 6) (30 m), rectangular window, `max_num_compressed` 10, nearest-2 interferogram network, native unwrapper, `coherent_support` reference with floor 0.85, loop-closure masking on, no atmospheric or tidal correction, no water mask |
| OPERA reference | GroundPulse's F08882 velocity raster over this site: DISP-S1 v1.0, the same 91 dates, chained slope over reference groups, `recommended_mask` = 1 in every granule, mm/yr positive toward the sensor, 30 m. sha256 in the recipe |
| GNSS reference | NGL IGS20 LOS rates of HSMN, RDCT and SPBH relative to THSU at the run dates, from GroundPulse's scored receipt for this track (sha256 in the recipe). They are the only admitted stations inside this burst's part of the site. THSU itself lies in the neighbouring burst, so the rates are used as station-to-station differences |

The reference products are read from GroundPulse's existing artifacts and are not
re-downloaded; both were produced before this test and neither depends on either arm.

## Scoring

All per-pixel metrics use one **common mask**: the output cells whose published displacement
is finite on all 91 dates in **both** arms. Displacement is the published `displacement_NN.tif`
series in mm with the zero first date prepended.

### J — junction-step roughness (mm)

A reference-rule error is a per-pixel offset of a whole ministack, which shows as a step at
the junction that neighbouring pixels do not share. Atmosphere and deformation are smooth at
this scale.

For each date step `k` (date `k−1` to date `k`): take the step, subtract its Gaussian-weighted
local mean (σ = 10 output cells = 300 m, normalised by the common mask), and take
1.4826 × MAD of the residual over the common mask. **J** is the mean of that over the five
affected junctions. Reported alongside, not criteria: the control junction (index 15) and the
median over all non-junction steps.

### F_int, F_link — the engine's own junction diagnostic

From each arm's `geometry_provenance.json`, `ministack_junctions`: the mean over the five
affected junctions of `fraction_integer_cycle` (the unwrapper's whole-cycle disagreement with
the connected component) and of `fraction_linked_over_quarter_cycle` (the linked phase itself
disagreeing by more than a quarter cycle).

### O_mad, O_frac — velocity against OPERA DISP-S1 (mm/yr)

Per-pixel OLS rate of the displacement series over the 91 dates, minus the OPERA rate
(nearest-neighbour onto the output grid), over common-mask cells with a finite OPERA rate. The
median difference is removed (the two products have different reference points). **O_mad** is
the median absolute remaining difference; **O_frac** is the fraction of cells within
1.5 mm/yr.

### G — velocity against GNSS (mm/yr)

Station value per date: the median of the 5 × 5 output cells centred on the station, keeping
common-mask cells; a station needs at least 13 of 25, otherwise it is "not measured" and its
pairs are dropped. For every pair of measured stations, the OLS rate of the difference of the
two series, against the difference of the two stations' GNSS rates. **G** is the mean absolute
error over scored pairs. With fewer than two measured stations G is "not evaluable" and is not
a criterion; that outcome is recorded.

## Pass rule

The new rule **passes** when it is not worse than the old on every metric, beyond the stated
tolerance:

| Metric | New must satisfy | Tolerance stands for |
|---|---|---|
| J | ≤ 1.05 × old | 5% of the old value |
| F_int | ≤ old + 0.002 | 0.2% of evaluable cells |
| F_link | ≤ old + 0.002 | 0.2% of evaluable cells |
| O_mad | ≤ old + 0.1 mm/yr | |
| O_frac | ≥ old − 0.01 | one percentage point |
| G (if evaluable) | ≤ old + 0.25 mm/yr | the allowance GroundPulse's frozen troposphere criterion uses |

**Junction steps are reduced** when J(new) ≤ 0.95 × J(old). That is reported as a separate
statement. It is expected, not required: where coherence has a floor the simulation puts the
two rules within a few percent of each other, and Houston is urban.

Either arm failing to run, or the common mask being empty, is a failed validation.

## What this does and does not establish

- It compares the two rules on one burst crop at one site. It does not re-validate the engine
  against OPERA or GNSS in absolute terms; GroundPulse's full-site receipts do that.
- G uses three stations within 5 km of each other, whose rates differ by at most 1.3 mm/yr. It
  can show that a rule made a station pair worse; it is a weak test of absolute accuracy.
- The issue #184 junction flags (`cycle_steps.flagged_pixels`) are recorded for both arms as an
  observation. A drop is not attributed to the reference rule by this protocol.

## Receipt

`validation/results/issue192_reference_rule/houston_t034_071054_iw3_20261110/receipt.json`,
written by the scorer: recipe and scorer hashes, binary hash, the two config hashes, the crop
window, pixel counts, every metric for both arms, each not-worse check and the verdict.
