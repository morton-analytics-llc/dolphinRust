# Changelog

All notable changes to dolphinRust are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Per-pixel LOS geometry ingest from OPERA CSLC-S1-STATIC.** The atmospheric-correction
  stage no longer projects zenith→line-of-sight with a single scalar `incidence_angle_deg`:
  when `correction_options.geometry_files` are supplied (per-burst CSLC-S1-STATIC granules),
  `dolphin-io::read_los_layers` reads the ground→sensor LOS unit-vector components
  (`/data/los_east`, `/data/los_north`), and `dolphin-corrections::resolve_los_geometry`
  reprojects + mosaics them onto the frame grid (first-covered-burst wins), deriving
  `up = sqrt(max(0, 1−e²−n²))` and per-pixel incidence `acos(up)` — character-identical to
  dolphin `atmosphere/ionosphere.py`. The iono/tropo slant then uses per-pixel `1/up`; with
  no geometry files the scalar path is **exactly bit-identical** to before. The resolved
  `LosGeometry{east,north,up}` is exposed on `DisplacementOutput.los_geometry` as the front
  door for the MMX1-colocated GPS ground-truth harness. Coverage is **fail-loud**: a frame
  extending beyond the supplied STATIC footprint (or a NaN/nodata hole) is a hard error, never
  a silent 0°/nadir pixel. Design: `md/design/per-pixel-los-geometry.md`; follow-up harness:
  `md/research/gps-feasibility-spike.md` §3.

### Fixed
- **Flaky HDF5 unit tests in `dolphin-io`.** `hdf5-metno` links a non-thread-safe HDF5, so
  parallel test threads creating/opening HDF5 fixtures raced and corrupted global library
  state (intermittent `geo`/`nisar`/geometry failures). HDF5-touching unit tests now serialize
  through a shared test lock.

### Deferred
- iono **ground→ionospheric-shell (450 km) incidence mapping** (pre-existing divergence from
  dolphin, not introduced here; the per-pixel path reproduces today's *ground*-incidence
  behavior); `local_incidence_angle` ingest; nearest-vs-bilinear resample at burst seams
  (single-burst consumer unaffected). See the design doc's Deferred section.

## [v1.4.0] — 2026-06-18

### Changed
- **Phase-linking covariance — ~2.4× faster** (Phase 3), no accuracy change. The per-pixel
  sample-coherence matrix (`dolphin-phaselink::covariance`, the #1 hot path) now reduces via a
  direct **Hermitian** product — summing only the upper triangle over contiguous sample rows and
  mirroring the lower — instead of ndarray's generic complex `dot` (which has no SIMD/BLAS path for
  `Complex<f64>`) plus a per-pixel conjugate-transpose allocation. Real-frame phase-linking is
  **2.38× faster** (host-controlled same-session A/B: 3.07 → 1.29 s; throughput 432 → 1028
  kpix·slc/s) and beats the committed pre-R1 baseline (2.01 s) absolutely. The coherence matrix is
  Hermitian by construction so the result is identical; `covariance_matches_oracle` (≤1e-4) and all
  analytic/quality/GPU/sign contracts stay green. Measurements + methodology in `bench/PERF.md`.

### Added
- **3D-unwrap-ready dispatch interface** (Phase 5), `dolphin-workflows::unwrap_backend`. The unwrap
  backend is now behind a **network-level** `UnwrapBackend` trait — it receives the linked phase +
  date pairs (not pre-formed independent 2D interferograms), so a future spurt-style 3D
  spatiotemporal solver can implement the trait and unwrap the whole stack jointly without any
  pipeline change. The shipped backends `SnaphuBackend` and `TophuBackend` implement it via the
  unchanged per-ifg loop; **output is bit-identical** (the end-to-end oracle contract still passes
  through the new dispatch; trait seam covered by `unwrap_backend_contract.rs`). No spurt port — the
  interface only. The `ref·conj(sec)` ifg sign convention is preserved (guarded by
  `sign_convention.rs`).
- **Phase-bias / non-closure correction** (Phase 4), `dolphin-phaselink::phasebias` — Michaelides
  et al. (RSE 2022). **Not in Python dolphin** (leads the oracle). The nearest-neighbour closure of
  the coherence matrix satisfies `Ξ_k = β_k + β_{k+1}`; a per-pixel first-order constant
  bias-velocity `β̄ = mean_k(Ξ_k)/2` is subtracted from the linked series (`θ_n ← θ_n·e^{−j n β̄}`)
  before the interferogram network. Opt-in via `phase_linking.correct_phase_bias` (**off by
  default** → default output and the oracle/sign contracts are unchanged; forces closure
  computation when on). Validated by an analytic fixture (constant bias recovered to <1e-9, zero
  residual) and a **measured non-closure reduction 0.800 → 0.095 rad (8.4×)** on a 100-date series;
  wired end-to-end (`run_displacement`). Numbers in VALIDATION.md.
- **NRT incremental displacement — end-to-end front door** (Phase 2b). `run_displacement_resumable`
  returns a `DisplacementState` (per-burst resumable phase-linking state + the files consumed);
  `update_displacement` folds newly-arrived acquisitions into the series — re-phase-linking only
  each burst's open trailing ministack + new ones via the carried compressed SLC, then recomputing
  the non-causal downstream (ifg network → SNAPHU unwrap → SBAS → velocity) from the updated phase
  history. The result is **bit-identical to a full `run_displacement`** of the extended stack
  (max|Δ| = 0 through unwrap + inversion; `nrt_displacement_contract.rs`). Exposed as a
  `dolphin stream --config <yaml> --initial <N>` CLI subcommand (process an initial window, then
  fold each later acquisition in, rewriting outputs). An update must extend every burst (a SAR pass
  yields one CSLC per burst) and the prior files must be a date-ordered prefix.
- **NRT incremental ministack updates** (Phase 2), in `dolphin-workflows::sequential`. Sequential
  phase-linking is feed-forward — a ministack reads only the compressed SLCs of prior ministacks
  and its own real SLCs — so a ministack that has filled to `ministack_size` ("sealed") never
  changes when later acquisitions arrive. `run_sequential_resumable` returns a `SequentialState`
  (sealed ministacks' products + the open trailing ministack's raw SLCs); `update_sequential`
  folds in newly-arrived acquisitions by re-phase-linking **only** the open ministack and any new
  ones, carrying the sealed compressed SLCs. The result is **bit-identical** to a full rerun of
  the extended stack — `cpx_phase`, compressed SLCs, stitched temporal coherence, CRLB, and
  closure all match with max|Δ| = 0 (`tests/nrt_incremental_contract.rs`: block update,
  one-at-a-time streaming, and the sealed-boundary edge case). `MiniStackPlanner::plan_with_offset`
  resumes the carry-forward batch accounting for the tail. The non-causal downstream (ifg network
  → unwrap → timeseries → velocity) recomputes from the updated phase history; the operational
  speedup is in skipping re-phase-linking the sealed history of a long stack.

## [v1.3.0] — 2026-06-17

### Added
- **Atmospheric corrections — ionospheric + tropospheric** (second half of v1.3.0), in the new
  `dolphin-corrections` crate. Both produce a per-acquisition range delay (meters) on the frame
  grid; the apply stage subtracts the per-date delay (relative to acquisition 0) from the
  inverted LOS-phase series **before velocity**. **Off by default** (opt-in via correction
  files, matching dolphin) — with none configured, `run_displacement` output is unchanged.
  - **Ionosphere (`dolphin-corrections::ionosphere`)** — IONEX GNSS TEC maps → L-band range
    delay via the closed-form `delay = TEC_LOS·K/f²` (`K = 40.31`; Yunjun et al. 2022 / Chen &
    Zebker 2012), **scaled to the configured carrier** (`1/f²`). The dominant L-band term:
    `(f_C/f_L)² ≈ 18×` C-band for the same TEC. Closed-form contract green; **validated on a
    real IGS final GIM from CDDIS** — 56.5 TECU → 14.4 m L-band delay (18.5× C-band).
  - **Troposphere (`dolphin-corrections::troposphere`)** — OPERA L4 (`OPERA_L4_TROPO-ZENITH_V1`)
    netCDF ingest via GDAL's `NETCDF:` driver, then a **reprojecting resample**: same-CRS grids
    take the bilinear path, cross-CRS grids (global EPSG:4326 product → UTM frame) take the new
    `warp_to_frame` (GDAL bilinear `reproject`), zenith→slant by `1/cos(inc)`. Synthesized-fixture
    and 4326→UTM warp contracts green (analytic delay recovered at known frame pixels `< 5e-3 m`,
    bare-warp + end-to-end through `build_troposphere`); the old CRS-mismatch `warn!` path is gone.
    **Real granule validated end-to-end on a real UTM frame** — the global EPSG:4326
    `OPERA_L4_TROPO-ZENITH_V1` granule warps onto the real Mexico City UTM 32614 384² frame:
    applied zenith mean **2.553 m** (slant@39° ≈ 3.285 m), physically consistent with the city's
    ~2.2 km altitude vs the 2.79 m sea-level centre. `DelayGrid` now carries the source CRS WKT;
    a CRS-less L4 grid spanning geographic-degree ranges is assigned EPSG:4326 (the plate-carrée
    product spec). See `VALIDATION.md`.
  - **RAiDER fallback (`dolphin-corrections::raider`)** — subprocess + GDAL ingest, **gated
    behind a `raider_available()` check like SNAPHU**; returns `RaiderUnavailable` rather than
    being stubbed when RAiDER is absent. The L4 path is primary.
  - `correction_options` config mirrors dolphin's `ionosphere_files` / `geometry_files` /
    `dem_file` (a dolphin YAML round-trips); `troposphere_files` (direct OPERA-L4 ingest),
    `incidence_angle_deg`, and `troposphere_variable` (default `"total"` = hydrostatic + wet)
    are **forward divergences** — dolphin derives troposphere from a DEM via RAiDER and has no
    `troposphere_files`. Layers surface on `DisplacementOutput.{ionosphere_delay,
    troposphere_delay}` and as `ionosphere_NN.tif` / `troposphere_NN.tif` COGs.
  - `dolphin-io::grid_centroid_lonlat` — frame-centre (lon, lat) via a CRS transform, to sample
    the coarse global IONEX grid at the frame.
- **NISAR / L-band geocoded-SLC ingest path** (first half of v1.3.0) — reads a NISAR L-band
  GSLC stack end-to-end into a displacement product.
  - `dolphin-io::nisar` — `read_nisar_rslc` / `read_nisar_stack` read the NISAR complex-`f32`
    `{r, i}` compound grid as `Cf32`; `read_nisar_geotransform` derives the affine transform
    from the NISAR `xCoordinates`/`yCoordinates` arrays and the `projection.epsg_code`
    attribute (GDAL returns identity for this layout). Contract test vs a synthesized
    NISAR-layout fixture (pixel values, grid shape, geotransform, EPSG).
  - **De-risk correction:** the prompt assumed NISAR was a *complex-int16* compound; the real
    `NISAR_L2_GSLC_BETA_V1` granule is **complex-`f32` `{r, i}`** (same layout as OPERA), so
    the only NISAR-specific code is the geocoding metadata reader. Validated end-to-end on a
    real 7.2 GB granule (reader + geotransform/EPSG) — see `VALIDATION.md`.
  - `input_options.input_type: InputType { opera_cslc (default) | nisar_gslc }` selects the
    reader. **Forward divergence** — dolphin v0.35.0 has no product-type field (it dispatches
    by workflow entrypoint); legacy YAML round-trips to `opera_cslc`.
  - L-band wavelength (≈0.2384 m) threads through `input_options.wavelength` to the `−λ/4π`
    velocity scaling (`velocity_uses_nisar_wavelength` proves the NISAR λ is used, not the S1
    default). No new solver — L-band is a parameter change.
  - End-to-end contract (`nisar_e2e_contract`): a multi-acquisition synthesized NISAR stack
    runs through `run_displacement` → typed output + COGs, grid/EPSG/geotransform correct.
  - **Limitation:** geometrically correct but **atmospherically uncorrected**. Ionospheric
    (~16× the C-band effect) + tropospheric corrections are a separate later v1.3.0 loop.

### Fixed
- **Interferogram sign convention — inverted LOS sign in v1.0.0–v1.2.0, now corrected.**
  `displacement.rs::unwrap_pair` formed the ifg as `sec·conj(ref)`; dolphin **production**
  (`interferogram.py`) forms `ref·conj(sec)`. The reversed order **globally inverted the LOS
  displacement *and* velocity sign of every release v1.0.0–v1.2.0** — subsidence read as uplift
  and vice-versa. It was invisible because the oracle generator (`oracle/gen_displacement.py`)
  carried the *same* inversion, so the sign-sensitive contracts proved Rust agreed with a
  flipped oracle, not with production. **Impact for eo:** the `velocity_mm_yr` sign (subsidence
  vs uplift) that drives GroundPulse risk tiers was inverted in v1.0–v1.2 and is now correct.
  Fixed in `e1db05a`; the oracle was corrected in lockstep (`2c85a79`). Backfilled this release
  with an **always-on analytic sign guard** (`sign_convention`, proven to go red if `unwrap_pair`
  is reverted) and a **gated real-data test** (`sign_real_data`, `SIGN_REF_PROD_IFG`) confirming
  dolphinRust matches a full production `dolphin run` on the F38502/Corcoran subsidence bowl —
  displacement correlation **−0.97 → +0.99** before/after the fix. See `VALIDATION.md`
  §"Interferogram sign convention".

## [Unreleased] — v1.2.0

### Added
- **CRLB uncertainty + sequential closure-phase quality layers** (`dolphin-phaselink`),
  validated against a **forward dolphin oracle v0.42.0** used *only* for these two layers
  (existing kernels stay validated at v0.35.0).
  - `crlb::estimate_crlb` — per-date Cramér–Rao σ from the Fisher information of the
    coherence model (`X = 2L·(Γ⊙Γ⁻¹−I)`, σ = `sqrt(diag(inv(ΘᵀXΘ+εI)))`), CPU `faer`/f64.
    Singular / fully-decorrelated Γ → `NaN` past the reference date (the v0.42 fix). This is
    the physical per-pixel uncertainty that feeds GroundPulse's `confidence_score`.
  - `closure::estimate_closure_phases` — nearest-neighbour triplet non-closure
    `∠(C[k,k+1]·C[k+1,k+2]·conj(C[k,k+2]))`; the prerequisite signal for phase-bias work.
  - Surfaced on `DisplacementOutput` (`crlb_sigma`, `closure_phase`, both `Option<Array3<f64>>`)
    and written as per-band COGs (`crlb_sigma_NN.tif`, `closure_phase_NN.tif`), sharing the
    grid CRS/geotransform; produced end-to-end by `run_displacement`.
  - Config flags match dolphin: `phase_linking.write_crlb` (default **on**),
    `phase_linking.write_closure_phase` (default **off**) — a real dolphin YAML round-trips.
  - Contracts: `quality_v042_contract` (CRLB σ + closure max |Δ| < 1e-4 vs v0.42.0;
    singular-Γ NaN matches; analytic consistency checks). GPU CRLB is a later follow-up.
- **tophu-style multi-scale unwrapping** (`dolphin-unwrap::unwrap_multiscale`) — OPERA's
  production multi-scale strategy driven over the existing SNAPHU wrapper: **coherence-weighted**
  coarse multilook (low-trust blocks masked + filled from trusted neighbours) → single SNAPHU
  unwrap → nearest upsample → overlapping tiled SNAPHU (rayon) → **overlap-based inter-tile
  cycle reconciliation** (maximum-reliability spanning forest over the coherent overlaps) →
  **feathered tile merge**. **Opt-in** via `unwrap_method: tophu`; **SNAPHU stays the default
  and the default build is behaviourally unchanged.**
  - Config: dolphin's `tophu_options` block (`ntiles`, `downsample_factor`, `init_method`,
    `cost`) is now modeled, so a real dolphin YAML round-trips it; new `UnwrapMethod::Tophu`
    routes the unwrap network through it (dolphin reserves its `multiscale_unwrap` for
    ICU/PHASS — we expose it driving the SNAPHU solver we ship).
  - Contracts: ramp recovery within the raw-SNAPHU envelope, coarse-pass round-trip, planted
    inter-tile 2π jump resolved, 2×2-grid loop-consistency, coherence-weighted-coarse-tracks-
    truth, fill, tile-cover, and up-sample unit tests.
  - **Measured win** (`bench/UNWRAP.md`): on the frozen large low-coherence scenes tophu now
    **beats** raw SNAPHU on all three metrics on both scenes — discontinuities −9 % on both,
    gross-cycle-error −10 % on the steep+decorr-ring scene, rms ≤ raw on both. The scenes,
    noise model, seeds and metrics are unchanged from the earlier honest-loss measurement;
    only the algorithm changed (coherence-weighted coarse + overlap-graph merge + feathered
    seams replacing the per-tile snap-to-coarse). Prefer tophu for large partly-decorrelated
    scenes; SNAPHU stays the simpler default for small/coherent scenes.
- **Per-ministack temporal-coherence stitching** (`dolphin-workflows::sequential`) — the
  cross-ministack temporal-coherence reduction is now dolphin's NaN-aware mean
  (`numpy.nanmean`, `_average_or_rename`) rather than a plain mean. Equal on all-finite
  layers (parity preserved), but a pixel masked/decorrelated in some ministacks now averages
  only the finite ones instead of being diluted toward zero — matching dolphin on
  many-ministack frames and closing the per-band CRLB/closure concatenation caveat. Contract
  `stitching_and_quality_match_oracle_multiministack` vs v0.42 oracle (`gen_stitch_v042.py`)
  on a 2-ministack stack: stitched temp_coh + concatenated CRLB + closure all < 1e-3.

## [Unreleased] — v1.1.0

### Added
- **GPU compute backend — first-class** (`wgpu`/Metal, f32; compiled into the **default
  build**). Runtime-selected via `worker_settings.compute_backend` (`auto` / `cpu` / `gpu`):
  `auto` uses the GPU at/above the ~128² crossover and the CPU below; **no GPU adapter,
  unsupported `nslc`, or a `no-gpu` build → automatic CPU fallback with a warning, never a
  panic.** The CPU (`faer`, f64) path stays the correctness reference. Covariance + EVD/EMI
  run in-shader (one thread per pixel); GPU covariance supports the SHP neighbor mask and the
  EMI β regularization. EMI uses an **all-pixel-accurate hybrid**: the kernel flags
  ill-conditioned / near-degenerate / borderline-PD pixels (bottom eigengap, Rayleigh
  wrong-mode guard, coherence floor, min Cholesky pivot) and the host recomputes that minority
  on f64 `faer`. Real Mexico 384² stack: **max Δφ 0.607 mm across every pixel, no π-rad tail**
  (EVD 0.176 mm). `MAX_NSLC` lifted 16→32 via deterministic threadgroup scratch (bit-identical
  run-to-run). Wired through `run_displacement` (`dolphin_phaselink::ComputeEngine`). Build
  CPU-only with `--no-default-features --features no-gpu`. Honest speed: end-to-end on an
  *integrated* M2 Pro the GPU is ~0.66× on the real stack (slower) and ~1.09× on synthetic
  stacks above ~192² — the value is correctness + portability to discrete NVIDIA/AMD (same WGSL,
  unchanged). See `bench/GPU.md` and `VALIDATION.md`.
- **Auto spatial reference-point selection** (dolphin v0.36 center-of-mass): the displacement
  series is referenced to a stable pixel — `timeseries_options.reference_point` if set, else
  the quality-weighted center of mass of the largest high-coherence region
  (`dolphin_timeseries::select_reference_point` / `reference_to_point`). The chosen point is
  exposed on `DisplacementOutput::reference_point`. The pinned v0.35.0 oracle uses `argmin`
  (no center-of-mass), so selection is contract-tested analytically.
- **Speed baseline** (`bench/`): reproducible dolphinRust-vs-dolphin benchmark with per-stage
  `tracing` timing in `run_displacement` (`RUST_LOG=info`). Real-frame phase-linking 3.6×,
  end-to-end 2.0× (unwrap-bound by an emulated snaphu binary). See `bench/README.md`.

### Validated
- **Velocity absolute scale on a real deforming scene** (B4): Mexico City burst
  T005-008704-IW1 — velocity TLS (orthogonal) slope ≈1.03 vs the oracle with matching
  magnitude, closing the documented real-data scale gap. See `VALIDATION.md`.

### Integration
- **GroundPulse (eo) adoption**: a `gp-dolphin` crate + standalone worker in `../eo`
  (branch `feature/gp-dolphin-rust`) calls `run_displacement` in-process via
  `spawn_blocking`, lands a velocity COG via `gp-storage`, and writes
  `displacement_aoi_summary` + `aoi_raster_products` rows in PostGIS. One real OPERA
  frame ran end-to-end. Isolated as its own Cargo workspace because dolphinRust's
  `hdf5-metno` (system HDF5 2.x) cannot share a binary graph with eo's static
  `hdf5-sys` (HDF5 1.x). Unpushed, pending review.

## [1.0.0] — 2026-06-16

First complete build: an end-to-end, library-first Rust rebuild of the OPERA / DISP-S1
displacement pipeline, validated against Python `dolphin` v0.35.0 as a reference oracle to
physically-meaningful tolerances.

### Added
- **End-to-end displacement pipeline** (`dolphin_workflows::run_displacement`): read CSLC
  stack → sequential phase linking (EVD/EMI) → interferogram network → SNAPHU unwrap →
  SBAS inversion → velocity. Synchronous and runtime-agnostic (no tokio) for `spawn_blocking`.
- **Typed public result** (`DisplacementOutput`): displacement cube, velocity (raster units),
  `velocity_mm_yr`, temporal coherence, acquisition days, EPSG, and geotransform — returned
  in memory and mirrored to disk.
- **L1/ADMM inversion** (dolphin's default least-absolute-deviations) alongside L2 weighted
  least squares; config-driven via `timeseries_options.method` (default L1). Matches the
  dolphin oracle to < 1.5e-6 on a redundant network.
- **Physical velocity** in mm/yr: acquisition dates are parsed from CSLC filenames
  (`input_options.cslc_date_fmt`) to derive real temporal baselines, and LOS phase is
  converted via `−λ/4π` (`input_options.wavelength`, else the Sentinel-1 default).
- **Temporal coherence** quality layer (ministack-averaged, dolphin's
  `temporal_coherence_average`), surfaced in the result and written as a raster.
- **Cloud-Optimized GeoTIFF outputs** (tiled, DEFLATE, overviews) for velocity, temporal
  coherence, and per-date displacement, sharing the CSLC grid's CRS + geotransform
  (`dolphin_io::read_geotransform` reads OPERA coordinate arrays + EPSG).
- **`dolphin` CLI** — a thin wrapper over `run_displacement` consuming a genuine dolphin
  `DisplacementWorkflow` YAML unchanged.
- **Real-data validation harness** (`validation/run.sh`, `compare.py`) and per-kernel oracle
  contract tests for every numerical crate.
- **Docs**: README quickstart (CLI + library), `docs/usage.md` integration guide (incl. the
  `spawn_blocking` pattern and output schema), and a runnable
  `crates/dolphin-workflows/examples/run_synthetic.rs`.
- `#![warn(missing_docs)]` on every crate; `cargo doc --no-deps` is clean.

### Validation
- Per-kernel contracts vs dolphin v0.35.0 `.npy` fixtures all pass (phase-link eigenvector
  overlap > 0.999, coherence < 1e-4, L1 < 1.5e-6).
- End-to-end synthetic single-burst equivalence: displacement corr 1.0000 / demeaned
  RMS ≤ 0.05 rad; velocity absolute scale a = 1.0000 (noise-free) → 0.9997 (realistic speckle).
- Real OPERA tier (4 bursts incl. Central Valley): config compatibility PASS; engine
  agreement PASS (displacement RMS residual ≤ 0.008 rad, matching velocity magnitude +
  temporal coherence). Reproducer: `validation/{fetch_real,crop_real,scan_coherence}.py`,
  `run_real.sh`.

### Known limitations / deferred
- **Real-data velocity absolute scale under strong signal** not independently pinned (sampled
  coherent scenes were tectonically stable); scale confirmed on the synthetic tier.
- Multi-burst stitching is implemented but not yet exercised on a real multi-burst frame.
- CRLB / closure-phase rasters, complex-GeoTIFF (CFloat32) writer, NISAR custom geotransform,
  `EagerLoader` prefetch, and tophu/spurt/whirlwind unwrappers are deferred (see STATUS.md).

[1.0.0]: https://github.com/morton-analytics-llc/dolphinRust/releases/tag/v1.0.0
