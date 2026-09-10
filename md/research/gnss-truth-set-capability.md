# Can the GNSS truth set settle the 17–19 mm/yr residual?

**Date:** 2026-09-10 · **Commit:** `b1e11e5` · **Cohort:** `gps_mmx1_2018`, burst
`T005_008704_IW1`, 52 epochs · **Continues:**
[`gps-mmx1-2018-troposphere-result.md`](gps-mmx1-2018-troposphere-result.md)

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

## Disposition

- **Scheduled:** issue #126 — pair sweep, 80 m control, window-sweep readout.
- **Not scheduled:** any further hypothesis tested against 17–19 mm/yr until the sweep says
  what that number is made of.
- **Corrected:** MMX1–ICMX is 11.13 km, not 13 km.
