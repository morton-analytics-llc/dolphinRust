# Implementation plan: MMX1/ICMX GNSS ground-truth harness

**Status:** implemented 2026-07-09; synthetic contracts and live acquisition
preflight pass. Full CSLC acquisition and the two real pipeline runs remain
unexecuted because the host has insufficient safe free-disk headroom.
**Source decisions:** `md/research/gps-feasibility-spike.md` and
`md/research/gps-ground-truth-scoping.md`.
**Primary burst:** OPERA CSLC-S1 `T005-008704-IW1`.
**Validation interval:** 2023-01-04 through 2023-06-09 (13 acquisitions in the
existing T005 series).
**Stations:** MMX1 is the high-SNR primary; ICMX is the near-null control.

## Objective

Build a reproducible, offline validation harness that acquires and crops a new
MMX1-centered OPERA stack, runs the displacement pipeline with both the native
MCF and SNAPHU unwrappers, projects NGL GNSS ENU motion into the exact
ground-to-sensor LOS at each station, and reports sign, magnitude, and
time-series agreement in millimeters.

The deliverable must provide independent evidence about the pipeline and the
two unwrap backends. It must not use the Python dolphin output as its truth
target, silently compare different physical locations, or describe a
spatially-referenced InSAR value as absolute station motion.

## Design summary

### Current state

- The feasibility verdict is **GO**, but only for a new station-colocated AOI;
  retrofitting the 1.92 km T005 crop is permanently limited to a weak regional
  sign/order-of-magnitude check.
- Per-pixel LOS geometry is on `main` in `67db149`. `LosGeometry` contains the
  OPERA CSLC-S1-STATIC ground-to-sensor east/north/up unit vector, and
  `DisplacementOutput.los_geometry` exposes it to the workflow.
- `validation/fetch_real.py` currently fetches CSLC products only.
- `validation/crop_real.py` assumes every input HDF5 contains `/data/VV`; it
  cannot crop a CSLC-S1-STATIC product and accepts only pixel offsets.
- `validation/run_real.sh` does not populate
  `correction_options.geometry_files`. Its dolphin-generated YAML also writes
  `unwrap_method: snaphu`, so an explicit native configuration is required.
- `source validation/creds.sh` currently loads `GP_EARTHDATA_TOKEN` in this
  sandbox, but token validity and a full-burst authenticated download have not
  been verified. Acquisition remains the first external gate.
- The pipeline subtracts a spatial reference pixel from every displacement
  epoch. A small MMX1-only crop therefore cannot establish MMX1's absolute
  magnitude unless the reference pixel's true motion is independently known.

### Proposed validation shape

Use two related crops from the same full burst stack:

1. **MMX1 core crop:** a 384×384 crop centered on MMX1. This is the first
   acquisition, geometry, coherence, sign, and runtime gate requested by the
   spike. It proves the strong station signal is inside a processable crop, but
   it is not the final absolute-magnitude gate.
2. **MMX1/ICMX common frame:** the smallest practical crop that includes both
   station sample windows plus a declared margin. This is the ground-truth
   comparison frame. At every epoch compare the spatial differential

   `InSAR(MMX1) - InSAR(ICMX)`

   with

   `GNSS_LOS(MMX1) - GNSS_LOS(ICMX)`.

The station-pair difference cancels the pipeline's selected spatial reference
pixel. ICMX is not treated as exactly zero: its measured ENU series is projected
and subtracted. If a common frame is rejected because of runtime or scope, the
harness may report the MMX1 core run as exploratory evidence only; it must not
claim an absolute magnitude pass.

For each station `s` and acquisition date `t`, with the first date `t0`:

```text
GNSS_LOS_s(t) = (ENU_s(t) - ENU_s(t0)) dot LOS_s
GNSS_DIFF(t)  = GNSS_LOS_MMX1(t) - GNSS_LOS_ICMX(t)
INSAR_DIFF(t) = mean_window(D_t, MMX1) - mean_window(D_t, ICMX)
```

`LOS_s` is sampled from the CSLC-S1-STATIC ground-to-sensor unit vector at the
same output-grid pixel as `D_t`. Negative LOS means motion away from the sensor;
for this descending scene, MMX1 subsidence is expected to be robustly negative.

## Technical requirements

### R1. Reproducible acquisition and immutable source manifest

- Extend `validation/fetch_real.py` without changing its existing defaults.
- Add an explicit recipe for burst `T005_008704_IW1`, start `2023-01-04`, end
  `2023-06-10`, and the expected 13 CSLC acquisition dates.
- Fetch the 13 full CSLC granules and the matching per-burst CSLC-S1-STATIC
  companion into separate ignored source directories.
- Authenticate only through `GP_EARTHDATA_TOKEN`; never print or persist the
  token and never fall back to the known-stale `.netrc` credential.
- Fail before processing if the token is absent, any expected epoch is missing
  or duplicated, an HDF5 signature/required dataset is invalid, or the STATIC
  product's burst/pass identity does not match the CSLC stack.
- Write a machine-readable acquisition manifest containing product names,
  burst, dates, source URLs/identifiers, byte sizes, and SHA-256 hashes. Raw
  products and credentials remain ignored.
- Support an inspection/dry-run mode that resolves the expected product set
  without downloading it.

### R2. Coordinate-driven, paired CSLC/STATIC cropping

- Preserve the existing `--row0`, `--col0`, and `--size` interface.
- Add WGS84 coordinate/bounds input so crop windows are derived from source
  geotransforms rather than copied as unexplained row/column constants.
- Produce the 384×384 MMX1 core crop centered on the authoritative station
  coordinates from the NGL station metadata, not the rounded research values.
- Produce a common frame containing MMX1 and ICMX, both station sample windows,
  and an explicit edge margin. Record its dimensions and estimated pipeline
  cost before running it.
- Crop CSLC and STATIC products by the same geographic bounds even when their
  pixel sizes or grids differ. Do not reuse CSLC pixel indexes for STATIC.
- Preserve the datasets required by existing readers plus the
  `/identification` metadata needed to prove burst and orbit-pass consistency.
- Verify every acquisition has the same grid, the MMX1 core center is within one
  output pixel of the station, and both common-frame station windows are fully
  in bounds.
- Emit a fixture manifest with station lon/lat, projected coordinates,
  source/crop geotransforms, EPSG, pixel indexes, bounds, and source hashes.

### R3. Controlled native/SNAPHU run matrix

- Add a dedicated validation runner rather than changing
  `validation/run_real.sh` semantics.
- Generate two Rust configs from one base config. They may differ only in
  `work_directory` and `unwrap_options.unwrap_method` (`native` vs `snaphu`).
- Set `correction_options.geometry_files` to the cropped STATIC product in both
  configs and verify `geometry_provenance.json` identifies sourced geometry.
- Pin the same input files, network, phase-linking settings, masks, wavelength,
  spatial reference configuration, and output grid for both backends.
- Run the MMX1 core first. Do not spend the common-frame runtime until the core
  crop passes HDF5/grid checks, yields finite displacement at MMX1, and has a
  usable station window.
- The Python dolphin run is optional diagnostic context only. It is never a
  ground-truth input and never determines pass/fail.
- A missing SNAPHU binary blocks the SNAPHU comparison with an explicit receipt;
  it must not silently collapse the matrix to native-only.

### R4. NGL IGS20 tenv3 ingest

- Add a validation-only GNSS module/script using the corrected NGL IGS20 tenv3
  layout documented by the spike.
- Fetch and cache MMX1 and ICMX station metadata and daily final series. Preserve
  source identifiers, retrieval time, and hashes in the run manifest.
- Parse fields by the documented tenv3 schema; do not infer ENU column positions
  from numeric appearance.
- Convert ENU to meters internally and emit comparison values in millimeters.
- Match the exact 13 SAR acquisition dates. Permit interpolation only between
  bracketing daily solutions across a configured maximum gap; flag every
  interpolated epoch and never extrapolate.
- Reject duplicate dates, non-finite ENU, missing reference epoch, unsupported
  frame/schema, or gaps beyond the declared limit.
- Preserve raw daily values. Any detrending, outlier rejection, or smoothing is
  diagnostic-only and must not replace the primary comparison series.

### R5. Co-located LOS projection and raster sampling

- Transform station WGS84 coordinates with an explicit `always_xy` path and
  invert the displacement affine transform to compute pixel indexes.
- Sample LOS east/north/up on the same resolved output grid and assert unit norm
  within the existing geometry tolerance.
- Use the documented ground-to-sensor dot product. Do not reconstruct heading
  from a scalar incidence angle and do not apply an extra asc/desc sign flip.
- Prepend the zero-valued reference epoch to Rust's `n_dates - 1` displacement
  rasters before date matching.
- Use a 5×5 finite-pixel mean as the primary station estimator. Also report 1×1,
  3×3, and 7×7 sensitivity, valid-pixel counts, within-window standard deviation,
  and temporal coherence. Never choose the window size after seeing which one
  produces the best agreement.
- If fewer than half the pixels in the primary window are finite or the station
  falls outside the connected usable region, mark the run invalid rather than
  moving the station sample to a more favorable pixel.

### R6. Metrics, decisions, and artifacts

- Score native and SNAPHU independently against the same GNSS differential.
- Report, at minimum: endpoint sign, endpoint residual in mm, MAE, RMSE, Pearson
  correlation, OLS intercept/slope, TLS slope, and the 1×1/3×3/5×5/7×7 window
  sensitivity.
- Report native-minus-SNAPHU differences separately so the evidence can say
  which backend is closer to GNSS without treating either backend as truth.
- Proposed initial acceptance bars, taken from the scoping brief, are:
  final displacement sign agrees; absolute endpoint residual is at most 20 mm;
  TLS slope is within 0.85–1.15; and time-series correlation is at least 0.90.
  These are provisional until the user accepts them as release gates.
- Emit a versioned JSON result for automation, a CSV containing every epoch and
  both station/backend series, and a plot for human review. Every result must
  record source hashes, configs, station pixels, reference pixel, window size,
  geometry components, units, sign convention, software commit, and threshold
  version.
- Distinguish `pass`, `fail`, and `not_evaluable`. Missing data, invalid geometry,
  insufficient finite pixels, or unavailable SNAPHU are `not_evaluable`, not
  scientific failures.

### R7. Documentation and claim boundary

- Add a `VALIDATION.md` section with exact reproduction commands and committed
  result summaries after the real run.
- State separately what was validated locally, what required live Earthdata/NGL
  access, and what remains unrun.
- Describe the common-frame result as independent GNSS-referenced differential
  validation. Do not claim absolute single-station displacement unless a stable
  spatial anchor or equivalent absolute phase calibration is added later.
- Do not change the production default unwrapper based on one station pair. A
  backend change requires a separate decision using this result plus the existing
  real-burst, analytic, performance, and regression evidence.

## Constraints and guardrails

- Keep all network/data work under `validation/`; no AWS or GroundPulse changes.
- Do not modify the T005 fixture or its existing results. The new fixture gets
  distinct source, crop, and run names.
- Do not commit full CSLC, STATIC, GNSS cache, COG, log, or run directories.
- Keep the existing fetch/crop commands backward-compatible.
- Do not duplicate per-pixel geometry math with a scalar incidence shortcut.
- Do not compare separately processed MMX1 and ICMX crops as if their spatial
  references were shared. The magnitude gate requires one common processed frame.
- Do not tune crop bounds, reference point, coherence cutoff, interpolation, or
  sample window independently for native and SNAPHU.
- Hold atmospheric settings identical across the backends. If ionosphere/tropo
  corrections remain off, say explicitly that the GNSS residual contains
  atmospheric as well as unwrap/pipeline error and cannot be assigned solely to
  the unwrapper.
- Treat acquisition as idempotent. A rerun over valid, hash-matching files must
  skip downloads; a hash mismatch must fail loudly rather than overwrite evidence.
- Python additions should use the existing oracle environment and avoid adding a
  dependency unless the current environment cannot provide the required function.
- If Rust is touched, follow crate `CLAUDE.md` rules, write the contract first,
  serialize HDF5 tests, and keep workspace fmt/clippy/tests green.

## Test contract

Tests are written red before implementation.

| Contract | Location | Behavior proved |
|---|---|---|
| Expected-product selection | `validation/tests/test_gps_acquisition.py` | Unsorted/extra search results resolve to exactly the declared 13 dates and one matching STATIC; gaps, duplicates, or wrong burst/pass fail. |
| Credential secrecy/error | same | Missing token produces an actionable error and manifests/logs never contain the token. |
| Coordinate-to-window | `validation/tests/test_gps_crop.py` | A synthetic georeferenced CSLC grid centers MMX1 within one pixel and a bounds crop contains both stations plus margin. |
| Paired-grid crop | same | CSLC and coarser/different-grid STATIC crops cover identical geographic bounds while retaining their own affine transforms and `/identification`. |
| Out-of-bounds rejection | same | A station or sample window outside the source footprint fails before files are written. |
| tenv3 parser | `validation/tests/test_gps_ground_truth.py` | A documented fixture line maps date/E/N/U correctly; malformed schema, duplicate/non-finite data, or absent reference date fails. |
| Date alignment | same | Exact dates pass; bounded interpolation is labeled; extrapolation and excessive gaps return `not_evaluable`. |
| ENU→LOS analytic sign | same | Pure vertical subsidence with a positive LOS-up component projects negative; east/north terms and unit conversion match a closed-form vector dot product. |
| Pixel co-location | same | WGS84→projected→affine inversion selects the known synthetic pixel and rejects axis-order reversal. |
| Raster sampling | same | 5×5 finite mean, valid count, standard deviation, and sensitivity windows are correct; fewer than 50% valid is invalid. |
| Spatial-reference cancellation | same | Adding any common per-epoch reference offset to both station pixels leaves `MMX1 - ICMX` unchanged. Separately referenced crops are rejected. |
| Epoch reconstruction | same | `n_dates - 1` displacement rasters become a 13-date series with an exact zero reference epoch and deterministic filename/date order. |
| Metrics | same | Synthetic identical, biased, sign-inverted, and noisy series produce expected residuals, correlation, OLS/TLS slopes, and pass/fail states. |
| Run-matrix identity | `validation/tests/test_gps_runner.py` | Native and SNAPHU configs differ only in backend/work directory and both carry the same STATIC file and scientific settings. |
| Artifact schema | same | JSON is versioned, finite values are serialized with units/provenance, `not_evaluable` is distinct from `fail`, and CSV epochs match JSON. |
| Real core smoke gate | live/ignored data | Both backends produce finite, georeferenced displacement and sourced LOS geometry at MMX1 on the 384×384 crop. |
| Real common-frame gate | live/ignored data | Both station windows are valid in one frame and the JSON reports each backend against the same GNSS differential. |

## Implementation plan

### Phase 0 — Freeze the validation contract

1. Add a tracked recipe such as `validation/gps_mmx1.json` containing the burst,
   expected dates, station IDs, authoritative metadata endpoints, crop rules,
   window sizes, interpolation limit, metrics, and threshold version.
2. Encode the two-crop design and spatial-differential equation in the recipe
   documentation.
3. Add the synthetic tests for reference cancellation and config identity first.

**Exit:** a reviewer can tell exactly which data, dates, geometry, configs, and
thresholds will be used before any live result exists.

### Phase 1 — Make acquisition complete and fail-loud

1. Refactor product selection/validation into testable functions in
   `validation/fetch_real.py`.
2. Add recipe/dry-run/output options and paired STATIC acquisition while
   preserving the old CLI defaults.
3. Source credentials, validate the authenticated path with one idempotent
   product transfer, then acquire the complete declared source set.
4. Write and verify the source manifest before cropping.

**Dependency:** valid Earthdata token and remote product availability.
**Exit:** 13 valid full CSLCs plus one matching STATIC exist locally with a
complete hash manifest, or the phase stops with an external-dependency receipt.

### Phase 2 — Build the station-aware fixtures

1. Extract reusable affine/CRS/window logic from `validation/crop_real.py`.
2. Add coordinate/bounds modes and product-type dispatch for CSLC vs STATIC.
3. Generate `mmx1_core` and the proposed `mmx1_icmx_common` fixtures.
4. Validate grids, identity metadata, station indexes, edge margins, and hashes;
   write the fixture manifest.
5. Estimate common-frame memory/runtime from dimensions before launching the
   full pipeline. If it is impractical, stop and elevate the reference strategy;
   do not fall back to separate absolute claims.

**Exit:** the MMX1 core is centered and both stations are co-located in one
common-frame coordinate system with usable sample windows.

### Phase 3 — Run both unwrap backends under one contract

1. Add the dedicated runner and config-equivalence assertion.
2. Run native then SNAPHU on `mmx1_core`; validate outputs, dates, geometry
   provenance, and station-window coherence/finite coverage.
3. If the core gate passes, run both backends on the common frame.
4. Preserve stdout/stderr, configs, commit, elapsed time, and output hashes under
   separate ignored run directories.

**Exit:** four valid runs exist (two backends × two fixtures), or a named backend
or common-frame gate is explicitly `not_evaluable`.

### Phase 4 — Add GNSS ingest and LOS projection

1. Implement the documented IGS20 tenv3 parser and cache layer.
2. Resolve authoritative station coordinates and match the 13 acquisition dates.
3. Sample each station's per-pixel LOS from the common output grid and project
   the reference-epoch ENU differences.
4. Write the epoch-by-epoch GNSS series and provenance before reading backend
   displacement, so backend output cannot influence GNSS preprocessing.

**Exit:** MMX1 and ICMX have auditable 13-epoch LOS series in mm with explicit
date-quality flags and geometry.

### Phase 5 — Score and render the independent comparison

1. Sample fixed 1×1/3×3/5×5/7×7 windows for every backend/epoch.
2. Form the InSAR and GNSS MMX1-minus-ICMX series.
3. Compute metrics and threshold status independently for native and SNAPHU.
4. Emit versioned JSON, CSV, plot, and a concise human-readable summary.
5. Confirm the result is invariant to the configured pipeline reference pixel by
   rerunning or analytically checking a second valid reference choice.

**Exit:** the evidence states whether each backend agrees with GNSS on sign,
magnitude, and shape, and which residuals are shared pipeline/atmosphere effects.

### Phase 6 — Integrate the evidence without overclaiming

1. Add exact reproduction commands and result hashes to `VALIDATION.md`.
2. Update the two research docs with a short implementation/result pointer; keep
   their original historical verdict intact.
3. Run the complete local quality gate and review the generated plot/JSON.
4. Make any decision about default-unwrapper policy in a separate change.

**Exit:** another engineer can reproduce the result from credentials plus the
tracked recipe, and validation claims match the evidence actually run.

## Validation

Minimum local checks before claiming the harness is implemented:

```sh
oracle/.venv/bin/python -m unittest discover -s validation/tests -p 'test_gps_*.py'
oracle/.venv/bin/python -m compileall validation
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Live data gates, using the final CLI defined by the implementation:

```sh
source validation/creds.sh
oracle/.venv/bin/python validation/fetch_real.py --recipe validation/gps_mmx1.json --dry-run
oracle/.venv/bin/python validation/fetch_real.py --recipe validation/gps_mmx1.json --with-static
oracle/.venv/bin/python validation/crop_real.py --recipe validation/gps_mmx1.json --fixture mmx1_core
oracle/.venv/bin/python validation/crop_real.py --recipe validation/gps_mmx1.json --fixture mmx1_icmx_common
oracle/.venv/bin/python validation/run_gps_ground_truth.py --recipe validation/gps_mmx1.json --fixture mmx1_core
oracle/.venv/bin/python validation/run_gps_ground_truth.py --recipe validation/gps_mmx1.json --fixture mmx1_icmx_common --score
```

The exact flags may be adjusted during implementation, but one documented
top-level command must reproduce acquisition validation, both backend runs, and
the final artifacts without manual YAML edits.

Final review checklist:

- Manifest has exactly 13 declared dates and one burst/pass-matched STATIC.
- MMX1 core center and common-frame station pixels match the fixture manifest.
- Native and SNAPHU configs pass the allowed-difference assertion.
- Geometry provenance is sourced and LOS unit norms are valid at both stations.
- GNSS and InSAR series share dates, units, sign convention, and temporal epoch.
- The magnitude score uses a common frame and spatial differential.
- JSON, CSV, plot, logs, configs, commit, and hashes agree.
- `VALIDATION.md` distinguishes pass/fail from unrun or unavailable external gates.

## Open questions

1. **Spatial-reference contract — resolved for this harness:** the implementation
   uses the two-fixture design and requires the common MMX1/ICMX differential for
   magnitude scoring. A future absolute MMX1 claim still requires a stable
   co-processed spatial anchor or another absolute phase-calibration method; the
   384×384 MMX1 crop alone cannot support that claim.
2. **Release thresholds:** confirm whether endpoint error ≤20 mm, TLS slope
   0.85–1.15, and correlation ≥0.90 are hard gates or report-only initial bars.
3. **Atmospheric scope:** the recommended first pass holds existing correction
   settings fixed and treats atmosphere as part of the total pipeline residual.
   If the goal is to attribute error specifically to native vs SNAPHU, decide
   whether acquiring 13-date ionosphere/troposphere inputs belongs in this item
   or a follow-up controlled experiment.

## Coding agent prompt

```text
Implement md/plans/gps-mmx1-aoi-harness.md in dolphinRust.

Objective:
Build the independent MMX1/ICMX GNSS validation harness, including paired
CSLC/STATIC acquisition, coordinate-driven crops, native and SNAPHU runs, NGL
IGS20 tenv3 ingest, exact per-pixel ENU-to-LOS projection, and versioned metrics.

Required context:
Read CLAUDE.md, the CLAUDE.md files for any Rust crates touched,
md/research/gps-feasibility-spike.md, md/research/gps-ground-truth-scoping.md,
md/design/per-pixel-los-geometry.md, and this plan. The pipeline spatially
references each epoch, so the magnitude gate must use MMX1 minus ICMX from one
common processed frame unless the user approves a different absolute anchor.

Write scope:
validation/fetch_real.py, validation/crop_real.py, new validation-only recipe,
runner/scorer/tests, VALIDATION.md, and short research-doc result pointers. Avoid
production Rust changes unless the existing validation surfaces cannot express
the contract; if Rust is needed, stop and explain the missing interface first.

Guardrails:
Keep old fetch/crop commands compatible. Never print or commit credentials or
large data. Do not use dolphin, native, or SNAPHU as truth. Do not compare
separately referenced station crops for magnitude. Keep both backend configs
scientifically identical except backend/work directory. Use the signed
ground-to-sensor LOS vector directly. Preserve unfiltered GNSS as primary.

Test contract:
Write the tests in §Test contract red first, then implement phase by phase. Live
data tests must be fail-loud or not_evaluable with receipts; they must not pass
on skipped or synthetic data.

Validation:
Run the Python test/compile checks and the workspace check/clippy/test commands
listed in §Validation. Then run the MMX1 core gate before the common frame. Report
local, live, and still-unrun evidence separately.

Escalate to the user if:
The common frame is operationally impractical; the token or matching STATIC is
unavailable; authoritative station metadata conflicts with the recipe; a stable
absolute anchor is required instead of the MMX1-ICMX differential; or real data
suggests changing the declared thresholds, crop, reference, or window after the
results are visible.
```
