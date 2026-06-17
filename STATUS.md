# dolphinRust — build status

Target: **v1.0.0** (first complete build). Update this file as phases land — it is the
single source of truth for build progress across sessions. Phase details in PLAYBOOK.md.

## Ready-to-ship v1 progress (branch `v1-ready-to-ship`, per V1_PROMPT.md)

| Workstream | State |
|---|---|
| A1 velocity mm/yr via real temporal baselines + wavelength | ✅ done (oracle scale a=1.0000) |
| A2 L1/ADMM inversion, config-driven default | ✅ done (oracle <1.5e-6) |
| A3 multi-burst frame stitching | ✅ done (2-burst frame stitch contract) |
| B4 real OPERA CSLC validation tier | ✅ tier built; engine agreement on real OPERA confirmed (RMS ≤0.008 rad, velocity magnitude + temp_coh match). Strong-signal velocity *scale* now **confirmed on real data** (Mexico City T005-008704-IW1, TLS slope ≈1.03, v1.1.0) — see VALIDATION.md |
| C5 typed sync public API (+temp coh, CRS/geotransform) | ✅ done |
| C6 COG outputs + documented schema | ✅ done (LAYOUT=COG verified) |
| C7 `#![warn(missing_docs)]` all crates, doc clean | ✅ done |
| D README + docs/usage.md + runnable example | ✅ done |
| E11 release metadata + CHANGELOG + packaging | ✅ done (core dry-run clean; see RELEASING.md) |

Gates green throughout: fmt, clippy -D warnings, test (37 groups), cargo doc --no-deps.
**Nothing pushed** — all on branch `v1-ready-to-ship`, awaiting sign-off.

## v1.1.0 progress (branch `v1.1`, per R1_PROMPT.md / ROADMAP.md)

| Item | State |
|---|---|
| Baseline speed benchmark (`bench/`) | ✅ done — PL 3.6× / e2e 2.0× on a real frame; per-stage `tracing` timing; honest unwrap caveat (Rosetta snaphu) |
| Close velocity-scale residual (B4) | ✅ done — Mexico City T005, TLS slope ≈1.03, magnitudes match; VALIDATION.md updated |
| Auto reference-point selection (center-of-mass) | ✅ done — `select_reference_point`/`reference_to_point`, wired via `timeseries_options.reference_point`, 5 analytic contracts + e2e green |
| eo integration | ✅ done (signed off) — `gp-dolphin` crate+worker in `../eo` (branch `feature/gp-dolphin-rust`, unpushed): in-process `run_displacement` via `spawn_blocking`, COG → gp-storage + summary rows → PostGIS. One real frame ran end-to-end (T144, COG in MinIO + `displacement_aoi_summary`/`aoi_raster_products` ready). Isolated as its own workspace to avoid the hdf5-metno vs hdf5-sys link clash |

Gates green (fmt, clippy -D warnings, test, doc). **Nothing pushed** — branch `v1.1`.

## GPU first-class progress (branch `gpu-first-class`, per GPU_FIRSTCLASS_PROMPT.md)

Promoting the R4 GPU phase-linking spike to a production, runtime-selected backend.
CPU (faer, f64) stays the correctness reference and automatic fallback.

| Item | State |
|---|---|
| 1. EMI GPU↔CPU hybrid — no π-rad tail | ✅ done — kernel emits a per-pixel `reliable` flag (bottom eigengap via Hotelling deflation + Rayleigh wrong-mode guard + coherence floor); host recomputes the flagged minority on f64 faer. Real Mexico stack (384², 13 acqs): **max Δφ 0.61 mm over ALL 147,456 px** (was 13.9 mm / π-rad), 5.6% CPU-recomputed. Contract `gpu_emi_hybrid_no_pi_tail_on_real_stack` green |
| 2. MAX_NSLC ≥ 32, deterministic scratch | ⏳ next |
| 3. GPU covariance SHP mask + β | ⏳ |
| 4. Runtime backend selection (default build) + no-adapter fallback + `no-gpu` | ⏳ |
| 5. Wire selected backend through `run_displacement` | ⏳ |
| 6. End-to-end validation + honest speedup/crossover | ⏳ |
| 7. Docs (README/usage/CHANGELOG/ROADMAP) | ⏳ |

Gates green so far (fmt, clippy -D warnings, gpu_contract). **Nothing pushed** — branch `gpu-first-class`.

## Phases (build in dependency order, per PLAYBOOK.md DAG)
- [x] 0 — Foundation (`dolphin-core`): types, `StridedBlockManager`, config, error
- [x] 1 — Covariance + EMI/EVD phase linking (`dolphin-phaselink`) ★
- [x] 2 — SHP selection (`dolphin-shp`)
- [x] 3 — PS selection (`dolphin-ps`)
- [x] 4 — Quality layers (`dolphin-phaselink`): temp_coh + compressed SLC done;
      **CRLB/closure deferred** — absent in pinned dolphin v0.35.0 (off the v1.0.0 critical path)
- [x] 5 — Ministack sequencing (`dolphin-stack` + `workflows::sequential`)
- [x] 6 — Interferogram network + SBAS inversion (`dolphin-timeseries`)
      L2 weighted least squares **and** L1/ADMM (Phase 6b, dolphin's default `least_absolute_deviations`).
      Method is config-driven (`timeseries_options.method`, default L1); L1 matches the dolphin
      oracle to 1.5e-6 on a redundant bandwidth-2 network (`l1_inversion_matches_oracle`).
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
      SNAPHU unwrap → SBAS L2 invert → velocity → GeoTIFF outputs. Multi-burst
      frame stitching now supported (A3); end-to-end matches the dolphin oracle.

## ✅ v1.0.0 — first complete build
All phases green. `dolphin run --config <yaml>` produces a displacement time series +
velocity from a CSLC stack, matching the dolphin v0.35.0 oracle within §Correctness
tolerances (displacement <1e-3, velocity <1e-2). Workspace: clippy/fmt clean, 37 test
groups pass. Deferred (off the v1.0.0 critical path, tracked above): CRLB/closure phase,
L1/ADMM (6b), EagerLoader, complex-GeoTIFF writer, NISAR geotransform, multi-burst
stitch, tophu/spurt/whirlwind unwrappers.

## ✅ End-to-end validation (2026-06-16) — see VALIDATION.md
Full `dolphin run` (Python, pinned v0.35.0, snaphu-py 0.4.1) vs `dolphinRust` (snaphu
binary v2.0.7) on **one** genuine `dolphin config` YAML, synthetic single-burst stack.
- **Config compatibility: PASS** — dolphinRust runs a real dolphin DisplacementWorkflow YAML unchanged.
- **Displacement: PASS** — noise-free agreement max 1.1e-3 rad (corr 1.0000); residual
  scales linearly with speckle ⇒ sanctioned faer-vs-jax eigensolver divergence, not a bug.
- **Velocity: FIXED (A1)** — acquisition dates are now parsed from CSLC filenames
  (`dolphin-workflows::dates`), so velocity carries a true physical rate. Affine scale vs
  oracle a=1.0000 (noise-free) → 0.9997 (speckle 0.05), within ±0.02 all tiers. Typed API
  exposes `velocity_mm_yr` (`−λ/4π`, config wavelength or S1 default).
- **Pending:** real OPERA CSLC validation tier (B4); L1/ADMM default (A2); multi-burst (A3).

## Awaiting input (see PLAYBOOK.md questions)
- ~~Pin the dolphin reference version~~ — **pinned: `v0.35.0` (`e567e55`)**.
- ~~SNAPHU binary MISSING~~ — **built v2.0.7 from Stanford source, at
  `/opt/homebrew/bin/snaphu`**. GDAL 3.12.2 / HDF5 2.1.1 / OpenBLAS present.
- Packaging: workspace member of `eo` vs. separate crate dependency — before Phase 10.

## Scaffold (done)
- [x] Workspace, 12 crates, builds clean (`cargo check`, `clippy`, `fmt`)
- [x] Claude Code setup: root + per-crate `CLAUDE.md`, PostToolUse hook, workspace lints
- [x] PLAYBOOK.md, README.md
