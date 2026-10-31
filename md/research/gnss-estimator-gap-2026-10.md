# The GNSS harness's two rate estimators, decomposed

**Date:** 2026-10-31 · **Issues:** #129 (estimator gap), #126 (pair sweep, items 2, 3, 5, 6) ·
**Run analysed:** `064ab0f`, `gps_mmx1_2018`, fixture `mmx1_2018_los_common`, seasonal model,
native backend, no troposphere — the 2026-09-10 run behind
[`gnss-truth-set-capability.md`](gnss-truth-set-capability.md) · **Script:**
`validation/gnss_estimator_gap.py`

No pipeline was re-run. Everything below is computed from the 09-10 receipts
(`validation/results/gps_mmx1_2018/pair_sweep_20260910/`), the cached tenv3 files the scorer
read, and the burst's CSLC-S1-STATIC LOS. Before decomposing anything, the script reproduces
the receipts: the series InSAR rate and the GNSS rate for all six pairs to 2×10⁻¹³ mm/yr, and
the `velocity.tif` pair difference to 8×10⁻⁶ mm/yr (float32 cast). So the decomposition is of
the estimators that actually produced the receipts, not a re-implementation of them.

## Result

1. **The 3.4–7.1 mm/yr gap is the temporal model.** `velocity.tif` fits rate + annual
   sin/cos; the series path fits a rate only. That one axis accounts for 101–114% of the gap
   on every recipe-gate pair. Epoch support, the cause the issue led with, is −27% to +23% and
   partly offsets it. The spatial window (1×1 pixel against 5×5 mean) is the fourth term and
   is under 0.9 mm/yr.
2. **`difference_raster_mm_yr` compares two different fits.** Its InSAR rate has the annual
   term removed and its GNSS rate does not. The gap mostly measures that mismatch, not the
   InSAR. The harness headline is now declared as `difference_mm_yr`, the matched design, and
   the gap is emitted as its own field (`estimator_gap_mm_yr`).
3. **Fitted with the same design on both sides, the residual is
   +22.36 ± 5.21 mm/yr (ICMX−MMX1, matched linear).** On the three recipe-gate pairs the
   residual series has no annual term (F test p = 0.56–0.66), so the linear fit is the right
   one there, and adding the seasonal term costs 1.7× in standard error without moving the
   rate significantly (z = 0.49–0.89).
4. **Two conclusions in the capability note depended on the mismatched estimator.** Under the
   matched estimator, the 2.92 km pair has the *smallest* residual, not one as large as the
   11 km pairs, and a plane explains 90% of the four-station field (p = 0.31), not 9%
   (p = 0.95). Neither result supports a gradient: no estimator gives p < 0.05 with
   dof = 1. But "station-local" was never established either.
5. **No a/b/c verdict for #126.** The evidence does not support any of the three options. The
   measurements that would decide it need data this burst does not have; see
   [Verdict](#126-verdict-not-reachable-from-this-cohort).

## #129 — what the gap is made of

The two InSAR rates differ on four axes, not the two the issue named. The run's own code
fixes each setting (`fit_velocity` in `crates/dolphin-workflows/src/displacement.rs`, and
`score_common_frame`), and refitting the station series reproduces both receipts:

| axis | `velocity.tif` (raster) | series (`insar_velocity_mm_yr`) |
|---|---|---|
| temporal model | rate + annual sin/cos (`--velocity-seasonal`) | rate only |
| gauge epoch (2018-01-06, zero by construction) | excluded (post-gauge fit) | included |
| GNSS-gap epochs | kept (all 51 post-gauge dates) | dropped (common GNSS only) |
| spatial sample | 1×1 station pixel | 5×5 window mean |

All 16 combinations were fitted per pair. The gap is split with exact Shapley shares (mean
contribution of each axis over all 24 switching orders), which sum to the gap exactly. The
single-axis switches don't add up, because the model axis interacts with the epoch axes:
dropping dates moves a four-parameter fit more than a two-parameter one.

Series minus raster InSAR rate, mm/yr:

| pair | common epochs | raster | series | gap | temporal model | gauge epoch | GNSS gaps | spatial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ICMX−MMX1 | 48 | +259.77 | +263.19 | **+3.42** | **+3.47** | −1.04 | +0.13 | +0.86 |
| ICMX−MXMX | 48 | −4.29 | −8.19 | **−3.90** | **−4.36** | −0.42 | +0.59 | +0.28 |
| MMX1−MXMX | 47 | −264.07 | −271.12 | **−7.05** | **−8.05** | +0.61 | +0.98 | −0.59 |
| *0.80 diagnostic gate only:* | | | | | | | | |
| ICMX−MXTM | 42 | +215.74 | +239.72 | +23.98 | +36.81 | −3.77 | −7.48 | −1.58 |
| MMX1−MXTM | 42 | −44.03 | −24.21 | +19.82 | +33.78 | −2.67 | −8.72 | −2.57 |
| MXMX−MXTM | 44 | +220.03 | +252.05 | +32.02 | +41.14 | −3.30 | −3.81 | −2.01 |

As a share of the gap on the recipe-gate pairs: the temporal model explains 101%, 112%, 114%.
Epoch support (gauge + GNSS gaps together) explains −27%, −4%, −23%. The spatial window
explains +25%, −7%, +8%. **(2) in the issue dominates; (1) is a minority term of mixed
sign.**

## Matched designs: which residual the harness should headline

Removing the gap means fitting the InSAR and the GNSS with *the same* design on *the same*
epochs, then fitting the InSAR−GNSS residual series directly. The table compares both
matched temporal models (5×5 window, common GNSS epochs) with the raster residual as it
appears in the receipts. SEs are OLS IID. The residual's lag-1 autocorrelation is near zero
on the recipe-gate pairs, so the IID SE is not obviously optimistic there.

| pair | raster (mismatched) | **matched linear (headline)** | lag-1 ρ | matched seasonal | residual annual amp. (mm) | annual term in residual? | z (lin − seas) |
|---|---:|---:|---:|---:|---:|---|---:|
| ICMX−MMX1 | +18.94 | **+22.36 ± 5.21** | −0.04 | +16.02 ± 8.85 | 3.0 | F(2,44) = 0.41, p = 0.66 | +0.89 |
| ICMX−MXMX | +11.23 | **+7.33 ± 3.49** | −0.14 | +5.03 ± 5.88 | 1.7 | F(2,44) = 0.59, p = 0.56 | +0.49 |
| MMX1−MXMX | −9.51 | **−16.56 ± 5.78** | +0.01 | −10.05 ± 9.64 | 3.3 | F(2,43) = 0.46, p = 0.63 | −0.84 |
| *0.80 gate:* ICMX−MXTM | +0.68 | +24.66 ± 16.61 | +0.35 | −9.31 ± 21.61 | 27.5 | F(2,38) = 10.76, p = 0.0002 | +2.46 |
| MMX1−MXTM | −17.58 | +2.25 ± 15.54 | +0.30 | −27.29 ± 20.17 | 25.3 | F(2,38) = 10.74, p = 0.0002 | +2.30 |
| MXMX−MXTM | −6.47 | +25.55 ± 16.68 | +0.38 | −10.00 ± 21.53 | 28.4 | F(2,40) = 11.17, p = 0.0001 | +2.61 |

z is the linear-minus-seasonal rate difference over √(SE²_seas − SE²_lin), its standard
error under the null of no annual term.

**Declared headline: `difference_mm_yr`, matched linear.** Reasons, from the table:

- It is the only harness residual whose two rates come from one design matrix, so a nonzero
  value is a disagreement between the measurements and not between two fits. The raster
  residual subtracts a linear GNSS rate from a seasonal InSAR rate, and the first table shows
  that this mismatch is most of its distance from the headline.
- On every recipe-gate pair the residual series carries no annual term (p = 0.56–0.66, with
  amplitudes of 1.7–3.3 mm). The linear fit is then unbiased and the more precise of the two.
  The seasonal fit's SE is 1.7× larger because a rate and a sinusoid are collinear over a
  354-day window, and its rate is within 0.9 SE of the linear one.
- The condition is checkable and fails where it should. At MXTM the residual has a 25–28 mm
  annual term (p ≤ 0.0002 on all three pairs, so it belongs to MXTM), and the linear headline
  aliases it: z = 2.3–2.6 between models. MXTM pairs clear only the 0.80 diagnostic gate, so
  this does not touch a recipe-gate score. It does mean a matched *seasonal* residual is the
  right one for any MXTM pair, and that the scorer's docstring states the assumption the
  headline rests on.

What the scorer now does (acceptance 2–3): `score_common_frame`'s docstring declares the
headline and why. `velocity_comparison` carries `estimator_gap_mm_yr` (raster minus series
InSAR rate, identically `difference_raster_mm_yr − difference_mm_yr`) and
`estimators.headline = "difference_mm_yr"`. `score_pairs.py` carries the gap into every pair
row. Test: `test_scorer_declares_the_headline_and_emits_the_estimator_gap`. The matched
seasonal residual and the annual-term test live only in the analysis script, not in the
scorer. Adding them to the scorer is a separate decision.

## #126 measurement 2 — residual against baseline and bearing

**Independence first.** N stations give N−1 independent differential series. At the recipe
gate that is 3 stations, 3 pairs, **2 independent series**. At the 0.80 gate it is 4 stations,
6 pairs, **3 independent series**. Every pair-level fit below reports its nominal dof (pairs −
parameters) and its independent dof (series − parameters). The nominal SEs assume independent
pairs and overstate precision. Where the independent dof is 0, the fit has no error estimate
at all, however small its nominal SE.

Fits, per estimator. Baseline and offsets come from the fixture manifest's EPSG:32614
coordinates. A ramp flips sign with pair orientation, so the baseline fit uses |R|. The tilt
fit is R = g_E·ΔE + g_N·ΔN, with no intercept, which is the bearing dependence written
without circular regression.

| gate / estimator | \|R\| vs baseline slope (mm/yr per km) | tilt g_E | tilt g_N | dof nominal / independent | station-plane test |
|---|---:|---:|---:|---|---|
| 0.90 / matched linear | +1.46 ± 0.62 | −1.43 ± 0.06 | −2.61 ± 0.25 | 1 / **0** | untestable (3 stations) |
| 0.90 / matched seasonal | +0.96 ± 0.63 | −0.92 ± 0.04 | −2.01 ± 0.15 | 1 / **0** | untestable |
| 0.90 / raster | +0.35 ± 0.99 | −0.79 ± 0.08 | −3.84 ± 0.29 | 1 / **0** | untestable |
| 0.80 / matched linear | +0.98 ± 0.45 | −1.43 ± 0.47 | +0.51 ± 1.27 | 4 / 1 | R² = 0.90, F(2,1) = 4.62, p = 0.31 |
| 0.80 / matched seasonal | −0.02 ± 0.54 | −0.76 ± 0.89 | +3.28 ± 2.39 | 4 / 1 | R² = 0.39, F(2,1) = 0.32, p = 0.78 |
| 0.80 / raster | −0.60 ± 0.37 | −0.51 ± 0.90 | +1.51 ± 2.43 | 4 / 1 | R² = 0.09, F(2,1) = 0.05, p = 0.95 |

The station-plane test is the gauge-free version: solve the pair table for per-station
residuals (zero-mean), then fit a plane. Its proper gradient SEs at the 0.80 gate are
0.88–1.80 mm/yr per km east and 2.38–4.85 north, depending on estimator.

What this supports:

- **At the recipe gate the baseline and bearing tests cannot be run.** The independent dof is
  0. The table's slopes describe three points and test nothing.
- **At the 0.80 gate no estimator finds a gradient at p < 0.05.** That is the only negative
  result here. With dof = 1 it has almost no power: a one-SE tilt over the 11 km baselines
  is 9.7–53 mm/yr, the same size as the residual.
- **The p-value depends on the estimator: 0.31, 0.78, 0.95.** The 0.95 in the capability
  note is the raster value. MXTM's annual term (above) makes matched seasonal the defensible
  estimator at this gate, so p = 0.78. That still means no detected gradient on a test that
  could not have detected one.
- **The baseline argument in the capability note reverses under the headline estimator.**
  That note found the 2.92 km pair "carrying as much residual as the 11 km pairs" (raster:
  +11.23 against +18.94 and −9.51). Matched linear gives +7.33 against +22.36 and −16.56, so
  the shortest pair has the smallest residual. With 0 independent dof neither ordering is
  evidence. The note's ramp argument rested on the mismatched estimator.

Per-station residuals (zero-mean gauge, recipe gate) under the headline: ICMX +9.90,
MMX1 −12.97, MXMX +3.08 mm/yr. The pair table fits them to R² = 0.9990, misfit RMS
0.51 mm/yr. Under matched seasonal: +7.02, −8.69, +1.67. Under raster: +10.06, −9.48, −0.58,
which reproduces the capability note.

## #126 measurement 3 — window sweep in mm/yr

Rate at each window size per station, under the linear all-epoch fit the capability note used
and under the `velocity.tif` estimator (post-gauge, seasonal). The spread is max − min across
1/3/5/7 px:

| station | spread, linear all epochs | spread, `velocity.tif` estimator | mean 5×5 within-window sd |
|---|---:|---:|---:|
| ICMX | 0.23 | 0.21 | 0.41 mm |
| MMX1 | 0.70 | 1.10 | 0.48 mm |
| MXMX | 0.08 | 0.13 | 0.20 mm |
| MXTM | 2.62 | 4.39 | 2.04 mm |

The linear column reproduces the capability note's numbers. Under the raster estimator the
spread grows at three of four stations, most at MXTM (4.39 mm/yr).

Per pair, under the headline estimator (common epochs, linear): ICMX−MMX1 0.76, ICMX−MXMX
0.21, MMX1−MXMX 0.53 mm/yr. MXTM pairs (0.80 gate): 2.79–3.60 mm/yr. On the recipe-gate pairs,
representativeness is about 3% of the headline residual (2.9–3.4%), so the conclusion stands:
point-vs-pixel sampling does not explain it.

## #126 item 6 — the MMX1–ICMX baseline

The run receipts do not record station coordinates; `headline_receipt.json` and
`window_samples.json` carry pixels only. The script verifies the baseline from three sources
the run itself consumed:

| source | MMX1–ICMX |
|---|---:|
| recipe coordinates, great circle | 11.134 km |
| tenv3 reference-epoch coordinates (2018-01-06), great circle | 11.128 km |
| fixture manifest EPSG:32614 (grid distance, includes UTM scale) | 11.142 km |

**11.13 km is confirmed.** The 13 km figure survived in four places in
`gps-mmx1-2018-troposphere-result.md`. All four are corrected. `gnss-truth-set-capability.md`
already carried 11.13 km.

## #126 item 2 — SSNX–TNGF, still blocked on data

This analysis cannot produce the 80 m control. The capability note established that burst
`t005_008704_iw1` has no CSLC data at either station. What the receipts do settle is the
geometry question the issue asked:

- Baseline: 80.7 m (recipe), 80.0 m (tenv3), 80.7 m (projected).
- At this burst's output posting (5 m × 10 m, from the run grid's geotransform) the stations
  are **14.7 px apart in x and 3.3 px in y**. Two 5×5 windows overlap only if both offsets are
  under 5 px, so **the windows would not overlap**. The pair would measure GNSS + InSAR, not
  the GNSS floor alone. This assumes the neighbouring burst's product has the same posting,
  which is true of OPERA CSLC-S1 by specification but has to be checked against that run's
  geotransform.

What is missing, and what would produce it: the neighbouring burst's own 52-granule OPERA
CSLC-S1 stack for 2018 (the burst whose footprint, not bounding box, contains SSNX and TNGF),
its CSLC-S1-STATIC, a `gps_mmx1_2018`-style recipe and LOS fixture around SSNX/TNGF, one
pipeline run, then `score_pairs.py`. Fetching needs an Earthdata login. Coverage at the 0.90
gate also has to be checked for that pair. It is not in the existing receipts.

## #126 verdict: not reachable from this cohort

Issue #126 asks whether the residual is (a) below the comparison's own error floor, (b) an
engine error with a baseline signature, or (c) real, local, and worth explaining. The evidence
supports none of them:

- **Not (a), as far as it can be measured.** The headline residual is +22.36 ± 5.21 mm/yr,
  4.3 IID SEs, with lag-1 ρ = −0.04. Window representativeness is 0.2–0.8 mm/yr per pair. The
  comparison's random floor is well under the residual. Its *systematic* GNSS-side floor
  (monument, multipath, the ENU→LOS projection) is unmeasured, and measuring it is exactly
  what SSNX–TNGF is for.
- **(b) against (c) is undecided.** At the recipe gate the baseline and bearing tests have 0
  independent dof. At the 0.80 gate the plane test has 1 dof, and its p-value runs from 0.31
  to 0.95 depending on the estimator. The capability note's (c) leaned on two raster-estimator
  readings, the 2.92 km pair and the 9% plane R². Under the matched estimator both readings
  move toward (b). Neither move is significant.

Two measurements are missing:

1. **The GNSS-side floor:** SSNX–TNGF on the neighbouring burst (above).
2. **A plane test with at least 2 real dof:** at least 5 stations that clear the recipe gate on
   one InSAR frame. This burst cannot supply them in any year. Daily tenv3 solution coverage
   (calendar-year fraction, from the cached files) puts all five LOS stations at ≥0.8 in no
   year from 2015 to 2024. MXMX and MXTM have no solutions after 2019, and UNVA's best years
   (2019–2021, 0.94–0.99) are years when MXMX and MXTM are at 0.37 or below. More stations
   means a multi-burst frame, the same new work the 80 m control needs.

Until one of those exists, any residual hypothesis has to be sized against the headline
residual (+22.36 ± 5.21 mm/yr, matched linear, ICMX−MMX1) and has to name its estimator.

## Reproduce

```sh
M=<main checkout>/validation
oracle/.venv/bin/python validation/gnss_estimator_gap.py \
  --recipe validation/gps_mmx1_2018.json \
  --sweep $M/results/gps_mmx1_2018/pair_sweep_20260910 \
  --fixture-root $M/real_data/gps_mmx1_2018/cropped/mmx1_2018_los_common \
  --gnss-cache $M/real_data/gps_mmx1_2018/gnss \
  --grid $M/runs/gps_mmx1_2018/mmx1_2018_los_common/work_native/displacement_01.tif \
  --output estimator_gap.json
```

`--grid` supplies only the output geotransform and CRS. The raster used is from an earlier
run on the same fixture, and the script refuses to run unless every station pixel it computes
from that grid equals the pixel recorded in `window_samples.json`. It also exits if the
receipts do not reproduce. `validation/results/` and `validation/real_data/` are gitignored,
so these inputs exist only on the machine that ran the 09-10 sweep.
