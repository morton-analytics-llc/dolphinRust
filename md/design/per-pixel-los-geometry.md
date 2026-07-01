# Design — Per-pixel LOS geometry ingest from OPERA CSLC-S1-STATIC

**Status:** Phase 2 complete (design validated; blockers folded). **Date:** 2026-06-30.
**Follow-up gate for:** `md/research/gps-feasibility-spike.md` §3 step 1 — the hard
prerequisite for the MMX1-colocated GPS ground-truth harness.

## What / why

`dolphin-workflows/src/corrections.rs` projects zenith→line-of-sight with a single
scalar `CorrectionOptions::incidence_angle_deg` (default 37°), uniform over the frame:
iono `vtec_to_range_delay(vtec, incidence_angle_deg, freq)` filled uniformly
(`corrections.rs:130`); tropo `slant = 1/cos(incidence_angle_deg)` uniform
(`corrections.rs:155`).

That scalar is not precise enough for a GNSS ENU→LOS projection at mm tolerance, and it
discards the LOS **unit vector** (east/north/up) the GPS harness needs; it also leaves the
asc/desc ambiguity unresolved (no heading is read anywhere — `gps-feasibility-spike.md` §3).

This task ingests **per-pixel LOS geometry** from the OPERA **CSLC-S1-STATIC** companion
product, mosaics + reprojects it onto the frame grid, (a) replaces the scalar incidence in
the correction slant/iono math with per-pixel incidence, and (b) exposes a `LosGeometry`
struct (east/north/up unit components) plumbed onto `DisplacementOutput` as the public
front door for the GPS harness.

## Load-bearing assumptions (one line each)

1. **Source is the CSLC-S1-STATIC companion**, not the per-acquisition CSLC-S1 granule
   (which has no LOS rasters). *(user-confirmed)*
2. **Layer names & units — spec-verified**, not provisional: `/data/los_east`,
   `/data/los_north` are `float32`, dimensionless, the East/North components of the
   **ground→sensor** LOS unit vector (OPERA CSLC-S1-STATIC ProductSpec §4.3/§5.3). No
   `los_up`/`z` layer exists; the product also carries `local_incidence_angle` (terrain,
   deg), `layover_shadow_mask` (int8), `x/y_coordinates`, `projection`.
3. **Up is derived; incidence formula matches dolphin bit-for-bit:**
   `los_up = +sqrt(max(0, 1 − e² − n²))`, `incidence_deg = acos(los_up)·180/π` — character-
   identical to dolphin `atmosphere/ionosphere.py:93`. This is the **ellipsoidal** incidence
   (ellipsoid-normal vertical), the correct angle for a zenith→slant `1/cos` mapping.
4. **`local_incidence_angle` (terrain) is NOT used** for atmosphere or ENU→LOS (spec §4.3:
   angle to the *local surface* normal — radiometric-terrain geometry). Not read.
5. **Geometry is time-invariant but per-burst in space:** CSLC-S1-STATIC is one HDF5 per
   burst ID. A multi-burst stitched frame therefore needs **N** per-burst STATIC granules
   (`geometry_files` = a list, mosaicked), **not** one-per-date (unlike iono/tropo). For a
   single-burst frame (the immediate GPS-harness crop) N=1. *(load-bearing; dolphin
   stitches a per-burst list — `workflows/corrections.py`.)*
6. **Uniform-nodata resolve:** every component is reprojected onto the frame via
   `warp_to_frame` (STATIC always carries an EPSG via `/data/projection`), whose GDAL
   reproject fills out-of-coverage with **exact 0** — matching dolphin's `nodata=0` for
   los_east/north. So `(e==0 && n==0)` is the single, uniform invalid/nodata test covering
   both source-nodata and off-frame fill. *(load-bearing)*
7. **Sign/convention reconciliation is out of scope.** The reader ingests the ground→sensor
   convention faithfully and documents it; matching it to the pipeline's `−λ/4π·φ`
   displacement sign is the GPS harness's job (`ifg-sign-inversion` memory).

## Data layout & geometry math

```
LosGeometry {                    // on the frame grid (rows, cols); no redundant fields
    east:  Array2<f64>,          // ground→sensor LOS unit E-component
    north: Array2<f64>,          // ground→sensor LOS unit N-component
    up:    Array2<f64>,          // +sqrt(max(0, 1 - e² - n²))
}
```

- Incidence (derived at call sites, not stored): `incidence_deg = acos(up)·180/π`.
- Atmospheric **tropo** slant per pixel: `slant = 1/up` (== `1/cos(incidence)`). Exact.
- Atmospheric **iono**: `vtec_to_range_delay(vtec, acos(up)·180/π, freq)` per pixel —
  this reproduces the *current scalar* behavior (ground incidence into the shell-refraction
  fn). NOTE: dolphin additionally maps ground→ionospheric-shell (450 km) incidence *before*
  the refraction; dolphinRust does not (pre-existing divergence, see Deferred). So the iono
  term is **not** claimed "exactly 1/cos" — it is exactly *today's* behavior, per-pixel.
- ENU→LOS (for the harness, documented not wired here):
  `d_los = d_e·east + d_n·north + d_u·up` (positive = motion toward sensor).
- **Nodata guard is for fill, not layover:** `e²+n² = sin²(incidence)`; for S1
  (incidence ≈ 30–46°) `e²+n² ≤ 0.52`, so `up` is never near 0 for a *valid* pixel — the
  `max(0,·)` clamp only ever fires on nodata/roundoff. Invalid pixels are detected by
  `(e==0 && n==0)` and rejected, never fed through `acos`.

## Affected files / component boundaries

| Layer | File | Change |
|---|---|---|
| IO (h5 read) | `dolphin-io/src/geometry.rs` (new) | `read_los_layers(path, subdataset) -> Result<LosLayers{east,north,geo}>`: reads `/data/los_east`, `/data/los_north` (f32→f64) + `read_geotransform`. `?`-propagation, **no unwrap/expect on happy path**. Re-export from `lib.rs`. |
| IO (fixture) | `dolphin-io/src/geometry.rs` (test mod) | `write_static_fixture(...)` writing a STATIC-layout h5 (los_east/los_north/x_/y_/projection), mirroring `write_vv_fixture`. |
| corrections (math) | `dolphin-corrections/src/geometry.rs` (new) | `LosGeometry` struct + `resolve_los_geometry(layers: &[LosLayers], dst_gt, dst_epsg, shape) -> Result<LosGeometry>`: per-burst `warp_to_frame` of east/north, mosaic **first-valid-wins** (`e==0&&n==0` = invalid), derive `up`, `ensure!` full coverage else `CorrectionError` naming the uncovered fraction. Re-export from `lib.rs`. |
| workflow (wire) | `dolphin-workflows/src/corrections.rs` | Decouple gate: resolve geometry whenever `geometry_files` non-empty, **independent of `is_enabled()`**; require `wavelength` only when a delay is actually subtracted (iono/tropo present). Per-pixel iono + tropo when geometry present; scalar path **unchanged** when absent. Add `los_geometry: Option<LosGeometry>` to `CorrectionLayers`. |
| workflow (front door) | `dolphin-workflows/src/displacement.rs` | Add `los_geometry: Option<LosGeometry>` to `DisplacementOutput` (additive) and its literal (~`displacement.rs:213-226`), fed from `CorrectionLayers`. |
| config | *(none)* | `geometry_files` already exists (`config.rs:413`). `is_enabled()` is **not** changed; the early-return guard in `apply_corrections` is broadened to `!is_enabled() && geometry_files.is_empty()`. |

No new crate deps. No config/YAML schema change.

## Correctness bar (contract tests, red first)

1. **Reader round-trip (analytic):** STATIC fixture, spatially-constant geometry
   (incidence θ=34°, chosen az ⇒ known e,n), read → `resolve_los_geometry` onto a same-EPSG
   frame → `acos(up)·180/π ≈ 34°` (±0.01°), `up ≈ cos34°`, unit-norm `|e²+n²+up²−1|<1e-9`,
   e/n preserved.
2. **Cross-CRS warp:** STATIC on EPSG:4326, frame on UTM 32610 (mirrors the tropo warp
   test) → per-pixel incidence lands the analytic value at interior frame pixels (±0.02°).
3. **Two distinct fallback invariants:** (a) the **`None`/scalar-unchanged path is exactly
   bit-identical** to the pre-geometry code — `from_elem(scalar)`+`assign` == the old
   `fill(scalar)`, and `&band * &slant` (uniform slant) == the old `band * scalar` bit-for-bit
   — covered by `disabled_is_noop` / `build_troposphere_warps_*`; (b) the **geometry-derived
   path** with a *uniform*-incidence STATIC product reproduces the scalar result only to the
   **f32-quantization floor of los_east/north (~5e-8)** — the product stores those layers as
   `float32`, so the honest test bar is `< 1e-6`, NOT 1e-9. `uniform_geometry_matches_scalar_path`
   asserts (b); a future "simplify" that breaks the derivation blows 1e-6.
4. **Coverage is fail-loud:** a frame extending beyond the STATIC footprint (or a nodata
   hole) → `resolve_los_geometry` returns `Err` naming the uncovered fraction — **never** a
   silent 0°/nadir pixel. Explicit reject-path test (the nodata guard exercised).
5. **Geometry-only config (no iono/tropo, no wavelength):** `geometry_files` set, both delay
   lists empty, `wavelength=None` → `apply_corrections` succeeds, returns
   `los_geometry: Some(_)`, leaves `disp` untouched. Proves the gate decoupling.
6. **Multi-burst mosaic:** two adjacent per-burst STATIC fixtures → frame spanning both →
   full coverage, correct incidence in each burst's extent (first-valid-wins).
7. **Oracle/real (best-effort, deferred-with-receipts):** if a real CSLC-S1-STATIC granule
   is reachable, assert incidence ∈ [30°,46°] + unit-norm; else skip with eprintln receipt.
   Not a merge gate.

Tolerances are physically-meaningful (angles 0.01–0.02°, unit-norm 1e-9), except bar #3
(< 1e-9 numeric) which must not perturb the existing opt-out contract.

## Deferred (explicit disposition)

- **iono ground→ionospheric-shell (450 km) incidence mapping** — Deferred (pre-existing
  divergence from dolphin `ionosphere.py:98`, not introduced here). Destination: a
  dedicated iono-parity task. Re-entry gate: iono absolute-accuracy validation vs dolphin.
  This task passes *ground* incidence per-pixel, exactly reproducing today's scalar path.
- **`local_incidence_angle` ingest** — Deferred: radiometric/terrain work; no consumer today.
- **`layover_shadow_mask` ingest** — Out of scope: masking is `layover_shadow_mask_files`
  (config.rs:542), a separate existing channel.
- **Heading/orbit_pass_direction read** — Out of scope: the signed LOS unit vector already
  encodes asc/desc; no separate heading read needed for ENU→LOS.
- **Nearest-neighbor (vs bilinear) resample at burst seams** — Deferred: dolphin uses
  `resample_alg="nearest"` to avoid blending real e/n against the 0-fill ring at burst
  edges. This task uses GDAL warp (bilinear) → a ≤1-pixel edge ring may be blended toward 0.
  Immediate consumer is a **single-burst interior crop** (MMX1), unaffected. Re-entry gate:
  multi-burst seam accuracy becomes a validation target. Mitigation already present: the
  coverage `ensure!` + first-valid-wins mosaic; the residual is sub-pixel edge softening.
- **GPS ENU→LOS projection + double-difference harness** — Deferred to the follow-up
  (`gps-feasibility-spike.md` §3 step 2); this task delivers the `LosGeometry` front door.

## Elevated questions

None open. The contract-changing fork (LOS source) was resolved with the user before Phase 1
(**CSLC-S1-STATIC companion**, **reader + wire into corrections**). Phase-2 blockers
(gate decoupling, per-burst coverage/nodata, front-door plumbing) are folded above; no new
decision requires the user.
