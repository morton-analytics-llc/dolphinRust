# Tropospheric correction audited against eo #757

**Date:** 2026-11-09 · **Issue:** #186 · **Source:** `morton-analytics-llc/eo` #757 (the
`opera_hres_v1` correction raises InSAR-minus-GNSS residuals: t035_073260 goes from 12.8 mm
RMS uncorrected to 20.0 mm corrected)

## Result

1. **No defect was found in dolphinRust's application of the correction.** Reference-date
   differencing, the six-hourly time weight, height interpolation, zenith-to-LOS projection,
   spatial registration and row order are each pinned by a closed-form test, and the engine
   reproduces that closed form to 4×10⁻¹⁵ m.
2. **The cause is the HRES product at 12–17 km baselines**, by eo's controlled measurement:
   an independent recomputation of the HRES delay reproduces the engine's applied correction
   to 0.03 mm on all 27 epochs of t035_073260, and HRES's station-to-station delay error
   against GNSS-measured zenith delay (8.8 mm RMS) is larger than the true differential
   signal (5.4 mm).
3. **One gap in the test coverage was real and is closed.** Both timed-TROPO fixtures were
   spatially uniform, and the only spatially varying test (the legacy per-date path) allowed
   5 mm, which is what a half-pixel shift of its 0.1° grid produces. A row flip or a
   half-pixel registration error in the timed path would have passed. The existing timed test
   also cannot see a reversed time weight (see below). The legacy test's tolerance is now
   1×10⁻⁶ m, which it meets.

The three kinds of evidence are kept apart below.

## Local inspection (this audit)

Read against eo's suspect list and the two items #186 left unverified:

| Suspect | Where | Finding |
|---|---|---|
| Differenced against the wrong epoch | `dolphin-corrections/src/apply.rs::subtract_delay` | `delay[t+1] − delay[0]`, the series' own reference date. |
| Not re-referenced to the spatial reference | `displacement.rs`, corrections precede the final spatial reference (v1.6.0) | A frame-wide offset cancels; primary minus control cancels the reference pixel as well. |
| Time interpolation of the six-hourly grid | `corrections.rs::troposphere_bracket`, `build_troposphere` | `weight = (t − t_lower)/(t_upper − t_lower)`, `a·(1 − w) + b·w`, no snapping to an epoch. |
| Zenith to slant | `corrections.rs::slant_grid` | `1/up` per pixel from the LOS vector. |
| Units | `tropo_at_terrain`, `subtract_delay` | Metres throughout; `−4π/λ` to phase. |
| Height | `tropo_at_terrain`, `interpolate_to_terrain` | Linear between the two bracketing levels at each pixel's DEM height. The DEM must be ellipsoidal, as the product's axis is; that is the caller's input (`timed TROPO requires ellipsoidal terrain`). |
| Row orientation / registration | `troposphere.rs::read_dataset_for_grid`, `warp_to_frame` | GDAL presents both latitude orders north-up; the warp is `GDALReprojectImage` with `GRA_Bilinear` and a zero error threshold (gdal 0.19 `reproject`), not nearest-neighbour. |
| Legacy per-date path selected by mistake (#186's own candidate) | `build_troposphere` | It refuses a config that sets both `troposphere_files` and `troposphere_epochs`. Which one eo configures is an eo fact: `opera_hres_v1` goes through `configure_epochs`, the timed path (eo's receipt). |

## Local test (this audit)

`corrections::tests::timed_troposphere_reproduces_a_hand_computed_station_differential`, with
fixtures `crates/dolphin-workflows/tests/fixtures/tropo_plane_{lower,upper}.{cdl,nc}`:

- two epochs six hours apart, each a different plane in longitude, latitude and height for
  both `hydrostatic_delay` and `wet_delay`, three unequally spaced height levels, one file
  stored south-up and the other north-up;
- a UTM 32N frame (the reprojecting path production uses), one row of 30 m pixels 15 km
  long, with the two "stations" at its ends;
- a sloping DEM (624 m and 1066 m at the stations), a different LOS at each station
  (`up` 0.80 and 0.74), three acquisitions at +1 h, +2.5 h and +5 h.

Every step is exact on planes, so the expected value is a closed form evaluated at each
station's true longitude, latitude and terrain height.

| Quantity | Engine | Closed form | Difference |
|---|---:|---:|---:|
| Layer, largest error over 2 stations × 3 dates | | | 4.0×10⁻¹⁵ m |
| Primary − control, date 1 relative to date 0 | −13.887172 mm | −13.887172 mm | 7×10⁻¹⁸ m |
| Primary − control, date 2 relative to date 0 | −37.032459 mm | −37.032459 mm | 3×10⁻¹⁷ m |

The test fails at 116 mm when the time weight is reversed in `build_troposphere`. The
pre-existing `timed_troposphere_combines_height_time_and_los` still passes with that change,
because its three acquisitions sit at weights 0, 0.5 and 1.

## GitHub evidence (eo, not re-run here)

`md/reports/real-run-2026-11/tropo/issue-757-diagnosis.md` on eo's
`origin/plan/real-science-asset-risk` (commits `655eb3ef`, `0b6f7152`, `85ca4296`), dated
2026-11-02:

- eo's `scripts/tropo_gnss_ztd.py` recomputes the HRES delay at each station without the
  engine (bilinear in latitude/longitude, linear in ellipsoidal height, linear in time to the
  CSLC sensing time, `1/up`, primary minus control, relative to the reference date). It
  matches the engine's recorded correction on t035_073260 to within 0.03 mm on all 27 epochs.
- OPERA HRES against NGL GNSS zenith delay, differential change in mm:

| Block (baseline) | HRES differential RMS | GNSS differential RMS | HRES − GNSS RMS |
|---|---:|---:|---:|
| t035_073260 P190/P192 (12.1 km) | 8.28 | 5.41 | 8.76 |
| t115_245711 P196/P198 (12.6 km) | 2.90 | 4.30 | 3.72 |
| t144_307992 P097/P098 (17.1 km) | 2.63 | 3.83 | 3.68 |

- Residual RMS, mm, same engine path with each correction source:

| Block | Uncorrected | HRES | HRRR | Measured GNSS ZTD |
|---|---:|---:|---:|---:|
| t035 | 12.83 | 20.02 | 12.53 | 12.09 |
| t115 | 14.14 | 14.69 | 14.44 | 14.38 |
| t144 | 7.77 | 8.25 | 6.90 | 8.10 |

On t035 the reference date alone carries a −7.0 mm HRES-minus-GNSS differential bias, which
time differencing copies into every epoch.

## Not done

eo's measurement was not repeated in this audit. It needs the OPERA L4 TROPO-ZENITH HRES
granules (Earthdata login) and the calibration-block station files, which eo's own receipt
records as deleted from `.worktrees/calibration/target/`. The numbers in the previous section
are eo's.

## Inference

The engine applies whatever delay field it is given exactly, on the path eo uses (local
test). eo's independent recomputation agrees with the engine on the real product (eo's
measurement). So the 12.8 → 20.0 mm inflation on t035 is the product's differential error at
that baseline, not the application.
