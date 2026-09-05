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

- **Linear temporal model, not seasonal.** The −5.74 mm/yr figure that motivated this
  investigation is a *seasonal-fit* residual, and `--velocity-seasonal` cannot currently
  produce a score — the seasonal model cannot emit `velocity_sigma.tif`, which the scorer
  requires (issue #115). So the A/B ran in the linear model, where the residual is
  −22.36 mm/yr. The conclusion is unchanged under either framing: 0.83 mm/yr is 3.7 % of
  22.36 and still only ~15 % of 5.74, and the sign of the epoch-wise degradation does not
  depend on the model.
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

Each pipeline run is ~26 minutes at full resolution over 52 epochs.

## What this leaves

The residual is unexplained. Troposphere and plate motion are both ruled out for this
comparison, and both for the same structural reason: the truth set is a 13 km differential
pair. The next hypotheses worth separating are ones that do *not* cancel over 13 km —
per-station local motion, reference-pixel selection, or a systematic in the phase-linking
or inversion chain itself. Testing absolute (single-station) agreement would need a
different truth construction than this cohort provides.
