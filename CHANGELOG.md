# Changelog

All notable changes to dolphinRust are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v1.1.0

### Added
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
