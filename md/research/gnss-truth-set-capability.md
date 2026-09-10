# Can the GNSS truth set settle the 17–19 mm/yr residual?

**Date:** 2026-09-10 · **Commit:** `064ab0f` (post-#123) · **Cohort:** `gps_mmx1_2018`,
burst `T005_008704_IW1`, 52 epochs, fixture `mmx1_2018_los_common`, seasonal model, native
backend, no troposphere · **Continues:**
[`gps-mmx1-2018-troposphere-result.md`](gps-mmx1-2018-troposphere-result.md)

> **Result (2026-09-10, same day):** the sweep was run. The residual is a **station-local
> additive term of order 10 mm/yr**. It is not point-vs-pixel sampling (0.08–0.70 mm/yr), not
> a smooth ramp (a 2.9 km pair disagrees *more* than an 11 km pair), and not a scale error
> (the residual is flat at 9–19 mm/yr against signals from 15 to 255 mm/yr). Whether the
> station-local term sits on the InSAR side or the GNSS side is still open and needs the 80 m
> control pair. Full numbers in [Results](#results) below; the sections before it are the
> design and still describe what was run.

## The question

Troposphere is ruled out (≤0.83 mm/yr available against the residual, and it degrades the
epoch-wise fit). Rigid plate motion is ruled out on the same geometry. Both failed for the
same structural reason rather than because the effect is absent: a differential pair cancels
any smooth regional field. Before spending more hypotheses against a 17–19 mm/yr number, the
prior question is whether this comparison can resolve 17–19 mm/yr at all.

**Answer: yes, but not with the pair currently scored, and the capability is measurable
rather than something to argue about. Two of the three measurements needed already exist in
the cohort and cost no pipeline re-run.**

## What the harness actually scores

`run_gps_ground_truth.py --score` evaluates the single pair named in the recipe's
`comparison` block: `MMX1_minus_ICMX_2018_common_frame`. That is one pair out of the ten
available in the LOS fixture and one of thirty-six in the nine-station cohort.

Station geometry from `validation/gps_mmx1_2018.json` (great-circle, recipe coordinates):

| pair | baseline | in LOS fixture |
|---|---:|:--:|
| SSNX–TNGF | **0.08 km** | no |
| ICMX–UTAC | 2.55 km | no |
| ICMX–MXMX | 2.92 km | yes |
| MXMX–UNVA | 6.50 km | yes |
| ICMX–UNVA | 9.02 km | yes |
| MMX1–UNVA | 10.08 km | yes |
| MMX1–MXTM | 11.10 km | yes |
| **MMX1–ICMX** | **11.13 km** | **yes — the scored pair** |
| MMX1–MXMX | 11.21 km | yes |
| MXTM–UNVA | 17.68 km | yes |
| MXMX–MXTM | 21.46 km | yes |
| ICMX–MXTM | 21.99 km | yes |
| MXTM–UJAL | 32.21 km | no |

(Thirteen of thirty-six shown; the full set spans 0.08–32.21 km.)

The scored pair sits in the middle of the available baseline range, and nothing about the
17–19 mm/yr number tells you whether it is a property of the engine, of the ground, or of
the comparison itself. Note also that MMX1–ICMX is **11.13 km**, not the 13 km carried in
earlier notes.

`score_pairs.py` already walks every pair and reuses `score_common_frame` unchanged against
an existing run — "the pipeline output is shared, only the station sampling differs, so no
interferogram is recomputed." The sweep is built. It has not been pointed at this question.

Its own docstring carries the constraint that matters for reading the output: **N stations
give N−1 independent differential series, not C(N,2).** Every other pair is a linear
combination. The sweep is a coverage survey, not thirty-six independent tests.

## Three measurements that separate the causes

### 1. SSNX–TNGF, an 80 m pair — the error floor

SSNX (19.3272, −99.1768) and TNGF (19.3269, −99.1761) are **80 m apart**. Real differential
ground motion over 80 m is negligible against 17–19 mm/yr, even in a basin whose rate varies
2–3× over 6–9 km. Whatever this pair reports is therefore the comparison's own error, not
signal.

What it bounds honestly: at 80 m the two GNSS series are nearly the same measurement, so
their differential is dominated by **GNSS-side** error — monument stability, multipath,
solution noise. That is exactly the term worth isolating first, because it is the one term
no InSAR change can fix. If SSNX–TNGF returns ~0 mm/yr, GNSS noise is not carrying the
residual and the 17–19 is InSAR or real ground motion. If it returns 10+ mm/yr, a large part
of the residual is untestable with this truth set at this scale.

Whether the two stations' InSAR windows overlap depends on the run's output posting — at
OPERA CSLC-S1's posting a 5×5 window is tens of metres per side, so they should be distinct,
but this must be read off the run's geotransform rather than assumed. If the windows do
overlap, the InSAR half of the differential is suppressed by construction and the pair
measures the GNSS floor alone — still the right control, but say which it is.

Both stations are in `mmx1_2018_common` and **not** in `mmx1_2018_los_common`. Adding them
to the LOS fixture is a recipe change, subject to their falling inside the LOS frame with
the declared 32-pixel margin.

### 2. Residual versus baseline — engine error or ground?

Ten LOS pairs from 2.92 to 21.99 km, at bearings spanning 16°–351°. The two candidate causes
separate cleanly:

- An **engine** error of the kind that survives differencing — a reference-frame tilt, a rate
  bias with a spatial gradient, an unwrapping ramp — grows with baseline, roughly linearly,
  and correlates with bearing if it is a tilt.
- **Local ground motion** or point-vs-pixel representativeness scatters with location and
  shows no baseline trend.

Fit residual against baseline and against bearing across the ten pairs. This is the test that
tropo and plate motion could not be given, because both were evaluated on the single pair
where they were guaranteed to cancel.

### 3. The window sweep — representativeness, already computed

`window_stats` already returns the **mean and standard deviation** over 1×1, 3×3, 5×5 and
7×7 windows at every station and every epoch (`sample_windows: [1, 3, 5, 7]`,
`primary_window: 5`). Only the 5×5 mean is consumed. The unused parts are a direct
measurement of the local spatial gradient at each station: the spread across window sizes,
converted to a rate, is the representativeness error of comparing a point monument against a
pixel patch.

This matters here specifically because the scoping note records Mexico City rates varying
**2–3× over 6–9 km** and **3–4× between adjacent sites within 20 km**, against a signal of
**26–31 cm/yr** at MMX1. A residual of 17–19 mm/yr is 6–7% of that signal. A field with those
gradients can produce a residual of that size from a modest error in effective sampling
location alone — and the window sweep already holds the data to say whether it does.

## Why this is not a call for a new truth set

The reflex answer to a truth set that cannot resolve a residual is to get a better one:
longer baselines, more stations, a different network. That is premature here. The cohort
already carries nine stations, thirty-six pairs, an 80 m control pair, and a four-point
window sweep at every station. The gap is not data. It is that one pair of one fixture is
being scored and the rest was collected and never read.

## What it costs

`score_pairs.py` needs an existing run's work directory and recomputes no interferograms.
The 2018 cohort output is not on disk — `validation/results/` is gitignored and scratchpads
are wiped between sessions — so one pipeline run is needed before the sweep, on the refetched
52-granule cohort (64,765,952 B) plus the Copernicus GLO-30 DEM.

That run should happen **after #123 merges**, not before: #123 changes which estimator
`velocity.tif` carries, so a sweep on current main would be stale on arrival. #123's own
acceptance requires re-running the GNSS A/B anyway. One run serves both.

## Results

Run on `064ab0f`, `mmx1_2018_los_common`, seasonal, native, no troposphere. 26 minutes at
full resolution over 52 epochs. The cropped fixture was already on disk, so no refetch was
needed for this half.

### #123's acceptance check: the harness numbers did not move

`write_velocity_uncertainty` was already set by the harness, so decoupling the estimator must
leave its output untouched. It does:

| | 2026-09-08 (pre-#123) | now (post-#123) |
|---|---:|---:|
| `difference_raster_mm_yr` | −18.9444 | **−18.94442** |
| MAE / RMSE (mm) | 9.034 / 11.244 | 9.0339 / 11.2444 |
| correlation / TLS slope | 0.9908 / 1.0997 | 0.99077 / 1.09972 |
| seasonal amplitude MMX1 / ICMX (mm) | 33.66 / 29.88 | 33.656 / 29.880 |

That is the check that #123 landed as a **default** fix and not as a silent change to the
harness. Issue #123 acceptance item 4 is satisfied.

### The truth set is thinner than the recipe suggests

Of the LOS fixture's five stations, only three pass the recipe's own
`minimum_gnss_fraction: 0.9` gate in 2018. Seven of the ten pairs are `not_evaluable`, and
every failure is GNSS-side coverage, not InSAR:

| station | 2018 common-GNSS fraction | usable |
|---|---:|:--:|
| MMX1, ICMX, MXMX | 0.90–0.92 | yes |
| MXTM | 0.808 | no |
| UNVA | 0.442 | no |

**Three stations means two independent differential series.** A tilt/ramp plane has three
parameters, so with three stations the plane fits by construction and tests nothing. That
alone answers part of the original question: the LOS fixture as scored *cannot* run the
baseline test. The nine-station frame is not a nice-to-have.

### Point-vs-pixel sampling: ruled out

`window_stats` fitted at each window size, converted to a rate:

| station | 1×1 | 3×3 | 5×5 | 7×7 | spread | 5×5 within-window sd |
|---|---:|---:|---:|---:|---:|---:|
| ICMX | 168.73 | 168.95 | 168.96 | 168.95 | **0.23** | 0.41 mm |
| MMX1 | −92.23 | −92.82 | −92.93 | −92.80 | **0.70** | 0.48 mm |
| MXMX | 178.19 | 178.13 | 178.12 | 178.11 | **0.08** | 0.20 mm |

(mm/yr unless noted.) The InSAR field is flat to well under 1 mm/yr across a 7×7 patch at
every station. Representativeness error is **2–4% of the residual**, not an explanation for
it. This was the hypothesis the Mexico City gradient literature made most plausible, and the
data rejects it.

### The residual is station-local, not a field

Solving the pair table for per-station residuals (zero-mean gauge, since only differences are
observable):

| station | residual | east | north |
|---|---:|---:|---:|
| ICMX | **+10.06** | −3.43 km | −1.93 km |
| MMX1 | **−9.48** | +7.32 km | +0.97 km |
| MXMX | **−0.58** | −3.89 km | +0.95 km |

The pair table is explained by these per-station numbers to R² = 0.9975, misfit RMS
**0.598 mm/yr** — so these are genuine per-station quantities and not pair-specific noise
from each pair's differing epoch support. Per-station spread is 8.0 mm/yr.

Two structural facts follow, and both are negative results:

**It does not scale with baseline.** A smooth ramp would grow with distance. It does not:

| pair | baseline | residual (raster) |
|---|---:|---:|
| ICMX–MXMX | **2.92 km** | **+11.23** |
| ICMX–MMX1 | 11.13 km | +18.94 |
| MMX1–MXMX | 11.21 km | −9.51 |

The shortest pair carries a residual comparable to the longest. A ramp is not the shape of
this.

**It does not scale with signal.** The residual is roughly flat in absolute terms while the
signal it sits on varies by 16×:

| pair | GNSS signal | residual (raster) | as % of signal |
|---|---:|---:|---:|
| ICMX–MMX1 | 240.83 | +18.94 | 7.9% |
| MMX1–MXMX | −254.56 | −9.51 | 3.7% |
| ICMX–MXMX | −15.53 | +11.23 | **72%** |

A wavelength or LOS-projection scale error would be a constant *percentage*. This is closer
to a constant *offset* of order 10 mm/yr per station.

### The estimator choice is itself worth 3–7 mm/yr

The harness reports two InSAR rates per pair — OLS on the displacement series over common
GNSS epochs, and the difference of `velocity.tif` at the two station pixels. They disagree
substantially on the same run:

| pair | raster | series | gap |
|---|---:|---:|---:|
| ICMX–MMX1 | +18.94 | +22.36 | 3.42 |
| ICMX–MXMX | +11.23 | +7.33 | 3.90 |
| MMX1–MXMX | −9.51 | −16.56 | **7.05** |

Up to 7 mm/yr of the 17–19 mm/yr is internal to how the rate is fitted from the *same*
displacement product — a third of the number under investigation, before any physics. Any
future statement of the residual has to name which estimator it means. This is the same class
of problem #123 just fixed one level down.

## Verdict

The residual is a **station-local additive term of order 10 mm/yr**, plus up to 7 mm/yr of
estimator choice. Ruled out by this run: point-vs-pixel sampling, a smooth spatial ramp, and
a multiplicative scale error. Troposphere and plate motion were already out.

What remains, and what separates them:

- **Real local ground motion** differing between monuments — entirely plausible in a basin
  where rates vary 3–4× between adjacent sites.
- **A station-local error on the InSAR side** — unwrapping or reference handling at specific
  pixels.
- **A station-local error on the GNSS side** — monument motion, or the ENU→LOS projection.

The 80 m SSNX–TNGF pair separates the GNSS side from the rest, because at that separation
real differential ground motion is negligible. The nine-station frame supplies both that pair
and enough degrees of freedom for the plane test that three stations cannot support.

## Disposition

- **Done:** pair sweep and window-sweep readout on the LOS fixture; #123 acceptance item 4.
- **In progress:** the nine-station frame, rebuilt by bounded remote range-read (the parent
  crop was deleted but its manifest and the acquisition catalogue both survive; ~31 MB read
  per 260 MB product).
- **Scheduled:** issue #126 — 80 m control pair, plane test with real degrees of freedom.
- **Not scheduled:** any further hypothesis tested against a single number for the residual
  until the estimator gap above is stated alongside it.
- **Corrected:** MMX1–ICMX is 11.13 km, not 13 km. The SSNX/TNGF 5×5 windows do **not**
  overlap (cols 1068–1072 vs 1083–1087 at 5 m posting), so that pair measures GNSS *and*
  InSAR error together, not the GNSS floor alone.
