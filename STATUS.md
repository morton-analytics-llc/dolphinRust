# dolphinRust — build status

Target: **v1.0.0** (first complete build). Update this file as phases land — it is the
single source of truth for build progress across sessions. Phase details in PLAYBOOK.md.

## Phases (build in dependency order, per PLAYBOOK.md DAG)
- [x] 0 — Foundation (`dolphin-core`): types, `StridedBlockManager`, config, error
- [x] 1 — Covariance + EMI/EVD phase linking (`dolphin-phaselink`) ★
- [x] 2 — SHP selection (`dolphin-shp`)
- [x] 3 — PS selection (`dolphin-ps`)
- [x] 4 — Quality layers (`dolphin-phaselink`): temp_coh + compressed SLC done;
      **CRLB/closure deferred** — absent in pinned dolphin v0.35.0 (off the v1.0.0 critical path)
- [x] 5 — Ministack sequencing (`dolphin-stack` + `workflows::sequential`)
- [x] 6 — Interferogram network + SBAS L2 inversion (`dolphin-timeseries`)
      (L2 only; L1/ADMM = Phase 6b, the documented temporary divergence from the L1-default oracle)
- [x] 7 — Filters (`dolphin-filtering`): long-wavelength high-pass + Goldstein
      (GDAL gap-fill for bad pixels deferred to Phase 8 I/O)
- [x] 8 — I/O layer + S3 read-staging (`dolphin-io` + `dolphin-ingest`)
      GeoTIFF r/w (gdal 0.19) + CSLC HDF5 read (hdf5-metno 0.12) + CSLC stack + S3 stage().
      Deferred: `EagerLoader` prefetch, complex-GeoTIFF writer (CFloat32), NISAR custom
      geotransform — not on the v1.0.0 local-run critical path.
- [x] 9 — Unwrapping dispatch (`dolphin-unwrap`) — SNAPHU subprocess wrapper
      (tophu/spurt/whirlwind = documented gaps, not built)
- [x] 10 — Pipeline orchestration + CLI (`dolphin-workflows` + `dolphin-cli`)
      `dolphin run --config <yaml>`: read CSLC → sequential phase-link → ifg network →
      SNAPHU unwrap → SBAS L2 invert → velocity → GeoTIFF outputs. Single-burst
      (multi-burst stitching deferred); end-to-end matches the dolphin oracle.

## ✅ v1.0.0 — first complete build
All phases green. `dolphin run --config <yaml>` produces a displacement time series +
velocity from a CSLC stack, matching the dolphin v0.35.0 oracle within §Correctness
tolerances (displacement <1e-3, velocity <1e-2). Workspace: clippy/fmt clean, 37 test
groups pass. Deferred (off the v1.0.0 critical path, tracked above): CRLB/closure phase,
L1/ADMM (6b), EagerLoader, complex-GeoTIFF writer, NISAR geotransform, multi-burst
stitch, tophu/spurt/whirlwind unwrappers.

## Awaiting input (see PLAYBOOK.md questions)
- ~~Pin the dolphin reference version~~ — **pinned: `v0.35.0` (`e567e55`)**.
- ~~SNAPHU binary MISSING~~ — **built v2.0.7 from Stanford source, at
  `/opt/homebrew/bin/snaphu`**. GDAL 3.12.2 / HDF5 2.1.1 / OpenBLAS present.
- Packaging: workspace member of `eo` vs. separate crate dependency — before Phase 10.

## Scaffold (done)
- [x] Workspace, 12 crates, builds clean (`cargo check`, `clippy`, `fmt`)
- [x] Claude Code setup: root + per-crate `CLAUDE.md`, PostToolUse hook, workspace lints
- [x] PLAYBOOK.md, README.md
