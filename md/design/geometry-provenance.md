# Geometry provenance artifact — dolphinRust #1 / eo #120

Machine-readable geometry provenance emitted alongside every displacement run, so
GroundPulse can gate asc/desc decomposition (`gp-displacement::decompose_dual_geometry`)
on *sourced* geometry instead of guessed defaults. Absent provenance is represented
explicitly; nothing is ever defaulted from Sentinel-1 nominals or track number.

## Contract (cross-repo, pinned by eo migration `20260703000003`)

Two surfaces, one type:

1. `DisplacementOutput.geometry_provenance: GeometryProvenance` — the in-process
   surface gp-dolphin `run_frame` reads (eo Path B, the real consumer).
2. `<work_directory>/geometry_provenance.json` — the on-disk artifact, written in the
   same `timed("write", …)` stage as the rasters (`displacement.rs:206-217`), for eo
   Path A (`ParseDolphinOutputTask`), the CLI, and audit.

### JSON schema (serialization of `GeometryProvenance`)

```json
{
  "schema": "dolphinrust-geometry-provenance/1",
  "method_version": "1.0.0",
  "orbit_direction": "descending",
  "incidence_angle_deg": 39.27,
  "incidence_angle_spread_deg": 1.57,
  "incidence_angle_min_deg": 36.36,
  "incidence_angle_max_deg": 41.95,
  "heading_deg": 189.98,
  "native_range_spacing_m": 2.329562114715323,
  "native_azimuth_spacing_m": 14.06,
  "acquisition_time_of_day_utc_s": 50428.5,
  "phase_linking_coherence": "temporal_coherence.tif",
  "decomposition_geometry_complete": true,
  "geometry_provenance": {
    "fields": {
      "orbit_direction": {
        "status": "sourced",
        "source_files": ["OPERA_L2_CSLC-S1_T144-308011-IW2_...h5", "..."],
        "source_keys": ["/identification/orbit_pass_direction"],
        "method": "read_scalar_consistent",
        "raw_value": "Descending"
      },
      "incidence_angle_deg": { "...": "..." },
      "heading_deg": { "...": "..." },
      "native_range_spacing_m": { "...": "..." },
      "native_azimuth_spacing_m": { "...": "..." }
    }
  }
}
```

- Scalar fields are `null` when not sourced; the matching `fields` entry is then
  `{"status": "absent", "reason": "<why>"}`. A `null` scalar with a `sourced` entry
  (or vice versa) is a bug.
- `orbit_direction` is normalized to lowercase `ascending`/`descending` (eo CHECK
  constraint); the raw product string is preserved in `raw_value`.
- `incidence_angle_deg` is the spatial **mean**; `incidence_angle_spread_deg` (stddev),
  `_min_deg`, `_max_deg` expose representativeness over the same finite-pixel
  population. Incidence varies ~36–42° across one IW2 burst (sin(θ) varies ~12%
  end-to-end) and ~30–46° across a multi-subswath frame — a mean-only scalar would
  silently hide that.
- `decomposition_geometry_complete` is `true` iff `orbit_direction`,
  `incidence_angle_deg`, and `heading_deg` are all `sourced` **and**
  `incidence_angle_spread_deg <= 3.0` (single-subswath spreads are ~1.6°;
  cross-subswath mixes exceed 4° — a frame-level scalar incidence is not a safe
  decomposition input there, and the gate records reason
  `incidence spread > 3 deg (multi-subswath frame?)` on the incidence field while
  leaving the raw stats populated for the consumer's own policy). This is the
  unambiguous "safe to decompose (this side)" bit; eo keeps decomposition disabled
  unless both geometry sides carry it.
- `phase_linking_coherence` is the artifact key of the phase-linking temporal
  coherence raster, relative to `work_directory` (always `temporal_coherence.tif`
  today; eo uploads it and stores the S3 key as `phase_linking_coherence_s3_key`).
- `heading_deg` convention: platform velocity azimuth **in the scene-center ENU
  frame**, degrees clockwise from geographic north, normalized to `[0, 360)`.
  Measured on the real T144 descending burst: 189.98°. Note this differs from the
  nominal ground-track (sub-satellite) heading by ~3° at S1 look geometry — do not
  compare against textbook track headings. Reconstruction check for eo
  (`look_direction = Right`): target→sensor LOS azimuth = `heading_deg − 90°`, i.e.
  `los_east = sin(inc)·sin(heading−90°)`, `los_north = sin(inc)·cos(heading−90°)`;
  verified against real STATIC LOS to 0.12°. (eo CHECK `-360..360` accepts; eo must
  apply this convention in `decompose_dual_geometry`.)

## Per-field ingest sources (all real product metadata, verified against
`OPERA_L2_CSLC-S1_T144-308011-IW2_20221102T140026Z_..._v1.1.h5`)

Sourcing rule (all fields): `sourced` requires **every** granule in `cslc_file_list`
to be readable for that field's keys AND the per-granule values to pass the field's
consistency gate. Any unreadable granule, any missing key, or any gate failure →
absent with reason. Every absent field also emits a WARN-level `tracing` event.

| field | source | method / aggregation | absent when |
|---|---|---|---|
| `orbit_direction` | `/identification/orbit_pass_direction` in every CSLC granule of `cslc_file_list` | case-insensitive match to `ascending`/`descending`, normalized lowercase | any granule unreadable, unrecognized value, or granules disagree |
| `incidence_angle_deg` (+ spread/min/max) | resolved per-pixel LOS on the output grid (CSLC-S1-STATIC `los_east`/`los_north` → derived `up`) | mean/std/min/max over finite pixels of `degrees(acos(up))` — same derivation as opera_utils (`degrees(arccos(los_up))`). Full frame coverage is already a hard error in `resolve_los_geometry` (`geometry.rs:120-129`), so no fill pixels reach the stats; "finite pixels" is not the load-bearing guard | `correction_options.geometry_files` empty. (A resolve *failure* remains a fatal run error, as today — it is never downgraded to absent, so a corrupt STATIC cannot silently degrade corrections to the 37° scalar) |
| `heading_deg` | `/metadata/orbit/{time,velocity_x,velocity_y,velocity_z,reference_epoch}` + `/identification/zero_doppler_{start,end}_time` + `/metadata/processing_information/input_burst_metadata/center` (lon/lat), per CSLC granule | linear-interpolate ECEF velocity at mid zero-doppler time, rotate to ENU at scene center (geodetic lat), `atan2(v_e, v_n)`; vector-sum circular mean across granules | any granule unreadable, or max angular deviation from the circular mean > 1° |
| `native_range_spacing_m` | `/metadata/processing_information/input_burst_metadata/range_pixel_spacing` (slant-range spacing) | read per granule (a constant of the acquisition mode) | any granule unreadable, or max−min > 1e-6 m |
| `native_azimuth_spacing_m` | `input_burst_metadata/azimuth_time_interval` + orbit state vectors + `center` lat | per granule `azimuth_time_interval × |v| × r_earth(lat)/|r_platform|` (ground-projected azimuth line spacing; ECEF `|v|`, geocentric radius at scene latitude — validated 14.0638 m vs 14.0618 m independent geodesic check on the real granule); arithmetic mean across granules | any granule unreadable, or max deviation from mean > 0.1 m |
| `acquisition_time_of_day_utc_s` | `/identification/zero_doppler_{start,end}_time` per CSLC granule | mid-time seconds-of-day (UTC), mean across granules. Same-track acquisitions repeat time-of-day to within seconds; eo needs hour-resolution acquisition time for `acquisition_pair_delta_hours` (its decomposition CHECK) and has no other source than filename parsing | any granule unreadable, or max deviation from mean > 60 s |

All numeric gates additionally require the value to be finite — `serde_json`
serializes NaN/∞ as `null`, which would forge the "null scalar with sourced entry"
invariant violation and fail eo's CHECK constraints. Spacings must additionally be
positive (eo CHECK `> 0`). The heading derivation rejects a mid zero-doppler time
outside the orbit state-vector span (clamped interpolation would otherwise source a
plausible-but-wrong heading).

**STATIC↔CSLC consistency (adversarial-review finding):** `up = √(1−e²−n²)` is
sign-insensitive, so a STATIC granule from the wrong pass — or an adjacent track in
an overlap zone — yields perfectly plausible sourced incidence. Before sourcing
incidence, each `geometry_files` granule's `/identification` (STATIC products carry
the same group) is cross-checked against the CSLC stack: `orbit_pass_direction`
must match and its normalized `burst_id` must be in the stack's burst set; any
mismatch or unreadable STATIC identification → incidence absent with the mismatch
as reason (fail-safe). When the CSLC identifications themselves are unreadable the
check cannot run — incidence stays sourced with an "unverified" note, and the
decomposition gate is already closed via the absent CSLC-derived fields.

`decomposition_geometry_complete` additionally requires
`acquisition_time_of_day_utc_s` sourced: with readable granules it can only be
absent through its 60 s consistency gate, which signals a mixed stack (e.g. an
adjacent relative orbit whose heading/spacing still pass their gates).

Notes:

- The CSLC **acquisition** granule is the primary metadata source; the STATIC granule
  carries the same `/identification` + `/metadata/orbit` groups and could serve as a
  fallback, but v1 reads acquisition granules only (one source per field — simpler
  provenance story).
- `correction_options.incidence_angle_deg` (the 37° scalar knob) is **never** a
  provenance source. It remains what it is: an atmospheric-projection fallback. The
  provenance incidence comes only from per-pixel LOS.
- NISAR GSLC runs (`input_options.input_type == NisarGslc`): all geometry fields
  absent with reason `nisar gslc metadata mapping not implemented` — explicit, not
  defaulted. (S1-only is fine for #1/#120.)
- Cropped fixtures / any granule missing the metadata groups: fields absent with the
  underlying read error as reason. **Metadata read failure never fails the run** —
  fail-safe (absence disables decomposition, understating rather than overstating).
- Compressed SLCs (NRT incremental) never appear in `cslc_file_list` — they are
  in-memory arrays (`sequential.rs`); metadata reads only ever touch real granule
  paths. If NRT ever persists compressed SLCs to disk, exclude them here.
- The **JSON artifact write failure fails the run** (same `?`-propagation as
  `write_outputs`), and the file is written unconditionally on every run — including
  all-absent runs — overwriting any previous artifact. A stale
  `geometry_provenance.json` from an earlier run in a reused `work_directory` would
  otherwise be a sourced-looking lie to eo Path A.

## Placement

- `dolphin-io/src/cslc_metadata.rs` — HDF5 readers: identification scalars, orbit
  state vectors, burst metadata scalars. Per-field `Result`s; no interpretation.
- `dolphin-corrections` — `LosGeometry::mean_incidence_deg()` (mean over finite
  pixels of existing per-pixel `incidence_deg`).
- `dolphin-workflows/src/provenance.rs` — `GeometryProvenance` (Serialize),
  assembly (consistency checks, heading geodesy, absence reasons), JSON writer.
  Called from `finish_displacement`'s write stage; struct added to
  `DisplacementOutput`.
- deps: `serde_json` added to workspace + dolphin-workflows.

## Contract tests (red first)

Oracle constants below were computed by `validation/make_geomprov_fixture.py`
(independent Python implementation) on the real T144-308011-IW2 granules; the same
script produces the committed fixtures `oracle/fixtures/geomprov_ci_{cslc,static}.h5`.

1. **Real-metadata fixture test** — `geomprov_ci_cslc.h5` (metadata-only crop of the
   real granule, 17 kB). Asserts: `orbit_direction == "descending"` (raw value
   `"Descending"` recorded), `native_range_spacing_m == 2.329562114715323` (exact —
   pure read), `heading_deg == 189.981317 ± 0.1°`,
   `native_azimuth_spacing_m == 14.063791 ± 0.02 m`, and the provenance block names
   the exact source keys.
2. **Absence test** — existing `/data`-only cropped fixture → every scalar `null`,
   every `fields` entry `absent` with reason, `decomposition_geometry_complete ==
   false`. Plus the adversarial variant: `correction_options.incidence_angle_deg =
   55.0` (distinctive non-default) with `geometry_files` empty → incidence still
   `null`/absent — proves the atmospheric knob can never leak into provenance.
3. **Incidence test** — synthetic STATIC fixture (existing `write_static_fixture`)
   with constant LOS → exact mean incidence, zero spread; real STATIC crop
   `geomprov_ci_static.h5` → mean ellipsoidal incidence `== 39.329207 ± 0.05°`
   (oracle computed on the **identical** 64×64 crop — incidence is a strong monotone
   function of range, so full-burst means don't transfer to crops).
4. **e2e** — `run_displacement` on the cropped validation stack writes
   `geometry_provenance.json`; it parses; all geometry fields absent (cropped
   inputs); `phase_linking_coherence == "temporal_coherence.tif"`.
5. **Cross-derivation check** — heading from orbit velocity (189.98°) vs heading
   implied by real STATIC LOS signs + `look_direction` (190.09°) agree within 2°
   (measured 0.12°; the gate would catch a ground-track-convention error at 3.2° and
   any ±90°/sign mistake).

Existing numeric paths untouched → bit-identity holds by construction; full
workspace test suite must stay green.

## eo-side follow-up (tracked in eo #120, not this repo)

- Bump `vendor/dolphinRust` submodule; read `DisplacementOutput.geometry_provenance`
  in `run_frame`; extend `DolphinRunRow`/`upsert_run`/`mark_run_ready` to persist the
  seven columns (migration already applied); upload `temporal_coherence.tif` →
  `phase_linking_coherence_s3_key`; regenerate sqlx offline cache.
- `ParseDolphinOutputTask` (Path A) can read `geometry_provenance.json` from
  `work_dir`.
- Gate: set `los_geometry_set = asc_desc` only when both sides'
  `decomposition_geometry_complete` are true.
- The incidence spread/min/max scalars have no dedicated `dolphin_run_state` columns —
  persist them inside the `geometry_provenance` JSONB (they ride along in the
  artifact's provenance block verbatim).
- **REQUIRED eo fix — east-west sign inversion.** `decompose_dual_geometry`
  (`gp-displacement/src/decompose.rs:88-97`) builds the look-vector east column with
  `heading + 90°`; under this contract's (physically verified) convention the
  target→sensor LOS azimuth is `heading − 90°`. Worked on the real geometries
  (asc 348°/desc 189.98°, θ=39.3°): eo's matrix is `A_true · diag(1, −1)` — vertical
  exact, **east-west sign inverted** (pure +10 mm east decomposes to −10 mm). The
  docstring at `decompose.rs:74-76` states a third, also-inconsistent formula. eo's
  only dual test (`test_dual_geometry_pure_vertical`) uses equal incidence + pure
  vertical motion, which is provably insensitive to the east-column sign. eo must
  (1) change `heading + 90.0` → `heading - 90.0` at `decompose.rs:89-90` and fix the
  docstring, (2) add a pure-east dual-geometry test with the contract headings
  asserting the east-west **sign**. (`decompose_los` uses yet another convention with
  `sin_phi.abs()` — out of contract scope, flagged in eo #120.)
- A work_dir with no `geometry_provenance.json` (e.g. Python-dolphin container runs)
  must read as all-absent — gate closed, fail-safe.
- Coherence flavors: for dolphinRust output, `temporal_coherence.tif` **is** the
  phase-linking quality raster — both `temporal_coherence_mean` and
  `phase_linking_coherence_mean` derive from this one artifact; there is no second
  raster to hunt for.
- eo's two exhaustive `DisplacementOutput` struct literals
  (`gp-dolphin/src/lib.rs:651-664, 678-690`) gain a `geometry_provenance` field at
  the submodule bump.
- `REAL` columns truncate f64 → f32; don't assert f64 exactness against DB
  round-trips.
