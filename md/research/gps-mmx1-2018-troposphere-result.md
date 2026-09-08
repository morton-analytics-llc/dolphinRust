# 2018 troposphere cohort — result

**Intake:** DR-TROPO-2018 / T06 (continues
[`gps-mmx1-2018-troposphere-fetch-strategy.md`](gps-mmx1-2018-troposphere-fetch-strategy.md))
**Run date:** 2026-09-03 · **Commit:** `c12bc07` · **Cohort:** `gps_mmx1_2018`,
fixture `mmx1_2018_los_common`, burst `T005_008704_IW1`, 52 epochs

## Result

**Troposphere does not explain the InSAR-minus-GNSS velocity residual.** It can move the
comparison by at most **0.83 mm/yr** against a **22.36 mm/yr** discrepancy — under 4 % — and
applying it makes epoch-wise agreement *worse*, not better.

The lead is closed. This was the only remaining hypothesis on the residual after the
plate-motion check was ruled out the same day.

## A/B, native backend, linear temporal model

Identical configs but for `correction_options.troposphere_files` + `dem_file`.

| metric | baseline | + troposphere | change |
|---|---:|---:|---|
| velocity difference (polyfit) | −22.3625 mm/yr | −22.3820 mm/yr | −0.020 |
| velocity difference (raster) | −20.2730 mm/yr | −19.3974 mm/yr | **+0.876** |
| MAE | 9.034 mm | 10.182 mm | **+1.148 worse** |
| RMSE | 11.244 mm | 13.335 mm | **+2.091 worse** |
| correlation | 0.9908 | 0.9858 | −0.0050 worse |
| TLS slope | 1.0997 | 1.1077 | +0.008 |

Both runs score `pass` against the provisional thresholds; the correction changes nothing
about that verdict.

## Why it cannot help, quantitatively

The delay itself is large, but almost all of it is common mode between two stations 13 km
apart at nearly equal elevation, and the harness scores `MMX1_minus_ICMX`, so the common
part cancels before anything is compared.

| quantity | value |
|---|---:|
| slant delay at MMX1 (epoch 0) | 2.1675 m |
| slant delay at ICMX (epoch 0) | 2.1449 m |
| MMX1−ICMX differential delay, mean | +22.513 mm |
| differential, standard deviation | 6.038 mm |
| differential, range | +4.139 to +34.774 mm |
| **linear trend of the differential** | **−0.828 mm/yr** |

That trend is the entire budget available to the velocity comparison, and the measured shift
in the raster-based difference (+0.876 mm/yr) matches it. The correction did exactly what
the geometry says it should — it is simply an order of magnitude too small.

The mean +22.5 mm offset is constant and cannot affect a rate. What the correction *does*
inject is the 6 mm epoch-to-epoch scatter of the HRES model differential, which is why MAE
and RMSE degrade: model noise added, no comparable real signal removed.

This is the same cancellation that ruled out rigid plate motion for this comparison on the
same day. A 13 km differential pair is insensitive to any smooth, regional field — which is
a property of the truth set, not a defect in either correction.

## Seasonal-model A/B (2026-09-08)

**Commit:** `f3e6a36` (#115 fixed, so `--velocity-seasonal --score` emits `velocity_sigma.tif`
and scores). Same recipe, fixture, and 52-granule cohort, refetched (64,765,952 B, the 09-03
byte count exactly; first granule `G3752930776-ASF`). DEM rebuilt from Copernicus GLO-30 tiles
N19W100 + N19W099 via `gdalwarp`. Results committed under
`validation/results/gps_mmx1_2018/seasonal_{base,tropo}/`.

| metric | baseline | + troposphere | change |
|---|---:|---:|---|
| velocity difference (polyfit) | −22.3625 mm/yr | −22.3721 mm/yr | −0.010 |
| velocity difference (raster, seasonal) | −18.9444 mm/yr | −17.0195 mm/yr | **+1.925** |
| RSS velocity sigma (new) | 20.78 mm/yr | 21.78 mm/yr | +1.00 |
| MAE | 9.034 mm | 10.180 mm | **+1.146 worse** |
| RMSE | 11.244 mm | 13.332 mm | **+2.088 worse** |
| correlation | 0.9908 | 0.9858 | −0.0050 worse |
| TLS slope | 1.0997 | 1.1076 | +0.008 |
| seasonal amplitude MMX1 / ICMX | 33.66 / 29.88 mm | 33.23 / 29.35 mm | |
| seasonal peak day MMX1 / ICMX | 172.8 / 169.5 | 171.5 / 166.9 | |

The displacement-series metrics (MAE, RMSE, correlation, TLS, polyfit) are identical to the
linear run to the fourth digit — the temporal model changes only the velocity raster, as it
should. Both runs `pass`. The two stations carry nearly the same seasonal signal (30–34 mm,
peaking ~3 days apart), so most of it cancels in the differential and the seasonal term buys
1.3 mm/yr on the raster residual under the harness estimator.

### The −5.74 mm/yr figure does not reproduce, and the reason is the estimator

The August number (−11.51 linear → −5.74 seasonal) was produced before v1.6.0 changed what
`write_velocity_uncertainty: true` selects. The harness sets that flag, so today's
`velocity.tif` is the **post-gauge, unit-precision** fit
(`time_function_post_gauge_unit_precision`); the August raster was the **full-series,
stitched-CRLB-weighted** fit. Re-fitting the two station pixels offline from today's
`displacement_NN.tif` + `crlb_sigma_NN.tif` under each estimator reproduces the harness
numbers exactly and gives, MMX1−ICMX residual vs GNSS (mm/yr):

| estimator | linear, base | seasonal, base | linear, +tropo | seasonal, +tropo |
|---|---:|---:|---:|---:|
| post-gauge, unit weights (harness) | −20.27 | −18.94 | −19.40 | −17.02 |
| full series, unit weights | −20.13 | −17.15 | −19.31 | −15.42 |
| full series, CRLB weights (pre-v1.6.0) | −20.01 | −11.91 | −18.95 | −8.52 |

Three things follow:

1. **The seasonal term's benefit is estimator-dependent**: 1.3 mm/yr under unit weights,
   8.1 mm/yr under CRLB weights. The linear rate barely moves (0.3 mm/yr across estimators);
   the seasonal *amplitude* fit is what the CRLB weights change. That is a sign the seasonal
   estimate is being carried by a few epochs the CRLB rates as precise, not a robust
   property of the series — treat the 8 mm/yr with suspicion, not the 1.3.
2. **−5.74 is not reachable on current main** under any estimator; the closest is −11.91.
   The remaining gap to the August figure is engine change between v1.5.0 and `f3e6a36`
   (#117 acquisition/UTC contracts, v1.6.0 velocity gauge), not a scoring difference.
3. **Troposphere's contribution is estimator-independent in sign and small in size**:
   +0.9 to +3.4 mm/yr across every row, against residuals of 8.5–20.3. It does not close the
   gap under any framing and always degrades the epoch-wise fit.

The residual to explain is therefore **17–19 mm/yr** in the seasonal model under the
production estimator, not 5.74.

## Transfer

Fetched by byte range under the probe gate, never as whole objects:

| | |
|---|---:|
| projected (probe gate) | 68,794,097 B |
| **actual, 52 epochs** | **64,765,952 B** |
| whole-object fallback avoided | 111,638,814,943 B |

The projection was conservative by 6 %. Staged via
`validation/fetch_l4_tropo_cohort.py`; each file is a netCDF-4 window subset of ~15 KB.

## Caveats

- **Seasonal model confirmed 2026-09-08.** The 09-03 A/B ran in the linear model because
  `--velocity-seasonal` could not score (issue #115). With #115 fixed (`f3e6a36`) the A/B was
  rerun in the seasonal model; see [Seasonal-model A/B](#seasonal-model-ab-2026-09-08). The
  conclusion holds: troposphere moves the seasonal raster residual by 1.9 mm/yr against
  18.9, and the epoch-wise degradation is identical.
- One cohort, one station pair, one burst. The cancellation argument generalizes to any
  short-baseline differential pair, not to absolute single-station comparison.
- Delays come from HRES via OPERA L4 `TROPO-ZENITH`; no other weather model was tried.

## Reproduce

```sh
source validation/creds.sh
<venv>/bin/python validation/fetch_l4_tropo_cohort.py --out <cohort_dir>
oracle/.venv/bin/python validation/run_gps_ground_truth.py \
  --recipe validation/gps_mmx1_2018.json --fixture mmx1_2018_los_common \
  --native-only --score --run-root <base_dir>
oracle/.venv/bin/python validation/run_gps_ground_truth.py \
  --recipe validation/gps_mmx1_2018.json --fixture mmx1_2018_los_common \
  --native-only --score --run-root <tropo_dir> \
  --troposphere-dir <cohort_dir> --dem <frame_dem.tif>
```

Each pipeline run is ~26 minutes at full resolution over 52 epochs (~30–34 with
`--velocity-seasonal`). Add `--velocity-seasonal` to both runs for the seasonal A/B.

## What this leaves

The residual is unexplained. Troposphere and plate motion are both ruled out for this
comparison, and both for the same structural reason: the truth set is a 13 km differential
pair. The next hypotheses worth separating are ones that do *not* cancel over 13 km —
per-station local motion, reference-pixel selection, or a systematic in the phase-linking
or inversion chain itself. Testing absolute (single-station) agreement would need a
different truth construction than this cohort provides.
