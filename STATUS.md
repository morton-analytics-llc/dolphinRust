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
| 2. MAX_NSLC ≥ 32, deterministic scratch | ✅ done — EMI nslc² scratch (Γ, Γ⁻¹) moved from per-thread private (spilled → nondeterministic at nslc 32 / 384²) to **threadgroup memory**, sized by pipeline overrides (`WG`, `GAM_LEN`) so 2·WG·nslc²·4 ≤ 24 KiB budget (WG≈18 at nslc 13, 3 at nslc 32). `MAX_NSLC` 16→32. Determinism contracts green (bit-identical run-to-run) at 384²/nslc13 and nslc32; accuracy at nslc32 sub-mm. Suite shares one locked GpuContext (concurrent contexts were the flakiness, not the kernel) |
| 3. GPU covariance SHP mask + β | ✅ done — GPU covariance gained the SHP neighbor-array mask (per-pixel win_h×win_w keep-factor on the window reduction); GPU EMI Γ construction gained `beta` regularization + `zero_correlation_threshold`, threaded through `process_coherence_matrices_gpu`/hybrid. Contracts: `gpu_covariance_shp_matches_oracle` vs dolphin SHP oracle (`cov_C_shp` / `glrt_neighbors`, max |Δ| 5.4e-7), `gpu_emi_beta_matches_cpu` (β=0.1, sub-mm) |
| 4. Runtime backend selection (default build) + no-adapter fallback + `no-gpu` | ✅ done — `gpu` is now a **default** feature (cli/workflows pull wgpu in the default build); `--no-default-features --features no-gpu` is the CPU-only build (verified: no wgpu linked, clippy clean). New `ComputeBackend` enum (`auto`/`cpu`/`gpu`) in `WorkerSettings` (+ kept `gpu_enabled` for dolphin YAML compat). `ComputeEngine` resolves the backend once, acquiring a GPU context if appropriate: Auto uses GPU ≥128² else CPU; no adapter / nslc>32 / no-gpu build → **automatic CPU fallback with a warning, never a panic**. Contracts in `engine_contract.rs` (run in both builds), incl. `no_adapter_falls_back_to_cpu_without_panic` (simulated missing adapter == exact CPU result) |
| 5. Wire selected backend through `run_displacement` | ✅ done — `ComputeEngine` threaded `run_displacement → link_one_burst → phase_link → run_sequential → link_and_compress`, replacing the direct CPU calls with `engine.covariance`/`engine.estimate`. One engine per run (single GPU context reused across bursts + ministacks). Contract `gpu_e2e_contract::gpu_backend_matches_cpu_end_to_end`: GPU vs CPU through the real sequential pipeline — median exact (5e-8 rad), **p99 sub-mm (0.001 mm)**; a tiny 0.13% of near-degenerate pixels (ambiguous EMI optima, masked downstream) differ — reported honestly. **Also generalized item 1's detector**: added a min-Cholesky-pivot flag so the f32/f64 PD-decision mismatch (EMI-vs-EVD on borderline Γ) is recomputed on CPU; Mexico stays 0.61 mm at 5.9% recompute |
| 6. End-to-end validation + honest speedup/crossover | ✅ done — real Mexico stack (384², 13 acqs), GPU vs CPU vs dolphin oracle, ALL pixels. **Accuracy:** EVD 0.176 mm; EMI raw 13.85 mm (π-tail) → **EMI hybrid 0.607 mm** (sub-mm, no tail), 5.9% CPU-recomputed; hybrid-vs-oracle max = CPU-vs-oracle max (tracks CPU exactly). **Speed (honest, end-to-end incl. covariance + readback + recompute):** real stack **0.66× (GPU slower)** on this integrated M2 Pro; synthetic crossover ≥192² (~1.09×). First-class value = correctness + discrete-GPU portability, not integrated speed — stated plainly. `bench/GPU.md` + `VALIDATION.md` updated; `gpu_bench` example rewritten for end-to-end |
| 7. Docs (README/usage/CHANGELOG/ROADMAP) | ✅ done — README §System requirements + "GPU backend (first-class, default-on)"; `docs/usage.md` §3 "Compute backend (CPU/GPU)" (selection, fallback, f32-vs-f64 accuracy, platform/speed, `no-gpu` build); CHANGELOG Added entry; ROADMAP "GPU acceleration — SHIPPED as a first-class backend" (was R4-deferred) |

**All 7 items done.** Gates green (default build *and* `no-gpu`): fmt, clippy -D warnings, test, `cargo doc --no-deps`. **Nothing pushed** — committed on branch `gpu-first-class`, awaiting sign-off.

## v1.2.0 quality-layers progress (branch `v1.2-quality`, per QUALITY_LAYERS_PROMPT.md)

The quality half of v1.2.0 (CRLB + closure phase). tophu unwrapping + per-ministack
coherence stitching are the *other* half — a separate later loop.

| Item | State |
|---|---|
| v0.42.0 forward oracle stood up | ✅ done — `oracle/.venv-v042` (dolphin 0.42.0), used **only** for the two new layers; existing kernels stay at v0.35.0 (reuses the committed `cov_C.npy`, no existing-kernel re-tune). Pin recorded in VALIDATION.md |
| CRLB σ raster | ✅ done — `crlb::estimate_crlb`, Fisher-information σ (CPU faer/f64), singular-Γ → NaN (v0.42 fix); contract `quality_v042_contract` σ max \|Δ\| <1e-4 incl. singular case |
| Closure-phase raster | ✅ done — `closure::estimate_closure_phases`, nearest-neighbour triplet non-closure; contract closure max \|Δ\| <1e-4 vs v0.42.0 |
| Typed API + COG + config + e2e | ✅ done — `DisplacementOutput.{crlb_sigma,closure_phase}` (`Option<Array3>`), per-band COGs, `phase_linking.write_crlb`(on)/`write_closure_phase`(off) match dolphin, produced through `run_displacement`; real dolphin YAML round-trips (`config_contract`, `displacement_contract`) |
| Docs (README/usage/CHANGELOG/ROADMAP) | ✅ done — incl. the CRLB→`confidence_score` note |

Gates green (default == gpu build, *and* `no-gpu`): fmt, clippy -D warnings, test (default 42
groups), `cargo doc --no-deps`. **Nothing pushed** — committed on branch `v1.2-quality`,
awaiting sign-off.

## v1.2.0 unwrap + stitching progress (branch `v1.2-unwrap`, per UNWRAP_STITCH_PROMPT.md)

The *other* half of v1.2.0 — tophu multi-scale unwrapping + per-ministack temporal-coherence
stitching. With this the v1.2.0 scope is complete.

| Item | State |
|---|---|
| tophu multi-scale unwrap | ✅ done — `dolphin-unwrap::unwrap_multiscale` (coherence-weighted coarse multilook + mask/fill → SNAPHU → upsample → overlapping tiled SNAPHU via rayon → overlap-based inter-tile cycle reconciliation, max-reliability spanning forest → feathered merge). Contracts: ramp within SNAPHU envelope, coarse round-trip, planted inter-tile 2π jump, 2×2 loop-consistency, weighted-coarse-tracks-truth, + fill/tile-cover/upsample unit tests |
| tophu-vs-SNAPHU measurement | ✅ **measured win** — on the frozen low-coherence scenes tophu now beats raw SNAPHU on all three metrics on both scenes (discont −9 % both, gross-cycle-err −10 % steep, rms ≤ raw both); numbers + margins in `bench/UNWRAP.md`. Scenes/metrics unchanged from the earlier honest-loss run — only the algorithm changed. SNAPHU stays the default |
| Per-ministack temp-coh stitching | ✅ done — `sequential.rs::stitch_temp_coh` = dolphin's NaN-aware mean (`numpy.nanmean`); contract `stitching_and_quality_match_oracle_multiministack` vs v0.42 oracle (`gen_stitch_v042.py`), temp_coh + CRLB + closure <1e-3 on a 2-ministack stack. CRLB/closure concatenation caveat closed |
| Config + wiring | ✅ done — `TophuOptions` + `UnwrapMethod::Tophu` (dolphin YAML round-trips `tophu_options`); `run_displacement` routes to tophu when selected, SNAPHU default behaviourally unchanged |
| Docs (README/usage/CHANGELOG/ROADMAP/VALIDATION) | ✅ done — incl. the honest tophu caveat + the nanmean-stitching clarification |

Gates green (default == gpu build, *and* `no-gpu`): fmt, clippy -D warnings, test, `cargo doc
--no-deps`. **v1.2.0 complete.** **Nothing pushed** — committed on branch `v1.2-unwrap`,
awaiting sign-off.

## v1.3.0 NISAR / L-band ingest progress (branch `v1.3-nisar`, per NISAR_INGEST_PROMPT.md)

First half of v1.3.0 — a NISAR L-band GSLC stack read end-to-end into displacement.
Atmospheric (ionospheric/tropospheric) corrections are the *other* half, a separate later loop.

| Item | State |
|---|---|
| NISAR reader (complex-f32 compound → Cf32) + custom geotransform/EPSG | ✅ done — `dolphin-io::nisar`: `read_nisar_rslc`/`read_nisar_stack` read the `{r,i}` **f32** compound as `Cf32`; `read_nisar_geotransform` from NISAR `xCoordinates`/`yCoordinates` + `projection.epsg_code` attribute. Contract `reads_synthesized_nisar_fixture` (pixels, shape, geotransform, EPSG). **⚠️ De-risk correction:** the prompt assumed complex-int16; the real granule is **complex-f32 `{r,i}`** (same layout as OPERA — only the geocoding metadata is NISAR-specific). hdf5-metno reads it cleanly |
| Config + product detection | ✅ done — `input_options.input_type: {opera_cslc | nisar_gslc}` (forward divergence; legacy YAML → opera_cslc), NISAR subdataset/granule-date parse; reader + geotransform dispatch in `run_displacement`. Contracts: `nisar_input_type_round_trips_and_defaults_to_opera`, `parses_nisar_granule_name` |
| L-band λ end-to-end | ✅ done — NISAR λ ≈ 0.2384 m threads via `input_options.wavelength` to `−λ/4π`; `velocity_uses_nisar_wavelength` proves the NISAR λ is used (not the S1 default). No new solver |
| End-to-end on a synthesized NISAR stack | ✅ done — `nisar_e2e_contract`: multi-acquisition NISAR fixture → `run_displacement` → typed output + COGs, grid/EPSG/geotransform correct |
| Real/sample NISAR granule | ✅ reader validated on real data / ⏳ full stack deferred — one real 7.2 GB `NISAR_L2_GSLC_BETA_V1` granule fetched via Earthdata/ASF; `nisar_real_data` test reads a center HH block (65536/65536 finite f32 samples) + geotransform (EPSG 32736, 10×5 m posting). **Full multi-date displacement deferred** — a real velocity needs ≥2 co-located repeat-pass dates (~15 GB+); single granule = single acquisition. See VALIDATION.md |
| Atmospheric correction (ionosphere/troposphere) | ✅ done (v1.3 part 2, below) |

Gates green (default == gpu build, *and* `no-gpu`): fmt, clippy -D warnings, test, `cargo doc
--no-deps`. **Nothing pushed** — committed on branch `v1.3-nisar`, awaiting sign-off.

## v1.3.0 atmospheric corrections progress (branch `v1.3-atmo`, per ATMO_CORRECTIONS_PROMPT.md)

Second half of v1.3.0 — ionospheric + tropospheric corrections that make L-band *usable*. New
crate `dolphin-corrections` (per-crate CLAUDE.md with the delay math). Corrections subtract a
per-date range delay (relative to date 0) from the inverted LOS series before velocity; **off by
default** (output unchanged when no correction files configured).

| Item | State |
|---|---|
| Apply stage + typed API + COGs | ✅ done — `subtract_delay` (φ = d·(−4π/λ)); `DisplacementOutput.{ionosphere_delay,troposphere_delay}` + `ionosphere_NN.tif`/`troposphere_NN.tif` COGs. Contracts: zero-delay identity, exact subtraction, constant-delay cancels |
| Ionosphere (IONEX → L-band `1/f²`) | ✅ done + **real-data validated** — closed-form `delay=vtec·1e16·K/f²` (K=40.31); IONEX parser. Real IGS GIM from CDDIS: 56.5 TECU → **14.4 m** L-band delay (**18.5×** C-band). `closed_form_vertical_delay`, `l_band_dwarfs_c_band_by_freq_squared`, `real_ionex_parses_to_physical_delay` (gated `IONEX_REAL`) |
| Troposphere (OPERA L4 netCDF + 4326→UTM warp) | ✅ done + **real-data validated end-to-end** — GDAL `NETCDF:` read + reprojecting resample (`warp_to_frame`, GDAL bilinear `reproject`, for cross-CRS; bilinear for same-CRS) + zenith→slant. Synthesized + **4326→UTM warp contracts** (`warps_4326_field_onto_utm_frame`, `build_troposphere_warps_4326_onto_utm_frame`, analytic delay `<5e-3 m`). Real `OPERA_L4_TROPO-ZENITH_V1` (ASF, 2 GB) ingest = **2.79 m** centre; **warped onto the real Mexico City UTM 32614 384² frame: zenith mean 2.553 m, slant@39° ≈ 3.285 m** (`real_l4_warps_onto_real_utm_frame`). CRS-mismatch `warn!` path gone |
| RAiDER fallback | ✅ wired, gated like SNAPHU — `raider_available()` check; `RaiderUnavailable` (not stubbed) when absent. Deferred this run (RAiDER not installed). L4 is primary |
| Config (dolphin parity + round-trip) | ✅ done — `correction_options` mirrors dolphin `ionosphere_files`/`geometry_files`/`dem_file`; `troposphere_files`/`incidence_angle_deg`/`troposphere_variable` forward divergence. `dolphin_correction_options_round_trips` |
| **Ifg sign-convention fix + backfilled evidence** | ✅ done + **real-data validated** — `unwrap_pair` forms `ref·conj(sec)` (production `interferogram.py`); the old `sec·conj(ref)` **inverted LOS displacement + velocity sign of v1.0.0–v1.2.0** (oracle was inverted in lockstep, so contracts were blind). Fixed `e1db05a`/`2c85a79`. Always-on analytic guard `sign_convention` (proven red on revert); gated real-data test `sign_real_data` (`SIGN_REF_PROD_IFG`) vs a full production `dolphin run` on the F38502/Corcoran bowl: displacement corr **−0.97 → +0.99**. eo `velocity_mm_yr` (subsidence vs uplift) sign now correct. See VALIDATION.md §"Interferogram sign convention" |

Gates green (default, `no-gpu`): fmt, clippy -D warnings, test, `cargo doc --no-deps`.
**Merged to `main` (`--no-ff`) and pushed once the real-data sign gate confirmed green.**

**Tropo 4326→UTM warp (branch `v1.3-tropo-warp`, per REMAINING_WORK_PROMPT.md Phase 1)** —
the deferred step is done: `warp_to_frame` reprojects the global EPSG:4326 L4 product onto a
UTM frame; `build_troposphere` dispatches to it on CRS mismatch (bilinear when same-CRS).
`DelayGrid` carries the source CRS WKT; a CRS-less L4 grid spanning geographic-degree ranges
is assigned EPSG:4326 (plate-carrée product spec). Fixture + real-frame contracts green; gates
green (default + `no-gpu`); sign guard green. **Awaiting sign-off to merge `--no-ff` + tag v1.3.0.**

## Phases (build in dependency order, per PLAYBOOK.md DAG)
- [x] 0 — Foundation (`dolphin-core`): types, `StridedBlockManager`, config, error
- [x] 1 — Covariance + EMI/EVD phase linking (`dolphin-phaselink`) ★
- [x] 2 — SHP selection (`dolphin-shp`)
- [x] 3 — PS selection (`dolphin-ps`)
- [x] 4 — Quality layers (`dolphin-phaselink`): temp_coh + compressed SLC done;
      **CRLB + closure phase landed in v1.2.0** (branch `v1.2-quality`, validated vs forward
      oracle dolphin v0.42.0) — see the v1.2.0 progress section above
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
