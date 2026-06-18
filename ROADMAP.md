# dolphinRust — 12-month roadmap (2026-07 → 2027-06)

Four releases on a **~2-month cadence** (compressed from the original quarterly plan). The
throughline: **get adopted in GroundPulse, reach parity with Python dolphin v0.4x, then lead
it.** dolphinRust's only real competitor is dolphin itself, so correctness parity is table
stakes — the differentiation is throughput (a compiled binary with no JAX JIT warm-up) and
being *ahead* on phase-bias and near-real-time.

Three strategic bets drive the sequencing:
- **Benchmark now, not as a milestone.** Establish the dolphinRust-vs-Python-dolphin speed
  baseline immediately — both engines and the validation stacks are already in place. It
  gates nothing and shouldn't occupy a release; *optimizing to beat it* is R4.
- **CRLB is a product feature.** GroundPulse scores asset risk from velocity + a
  `confidence_score`; per-pixel CRLB uncertainty is the missing physical input to that. → R2.
- **NISAR is the growth surface.** L-band DISP-NI penetrates canopy where C-band fails —
  exactly eo's forested pipeline/dam assets. Calibrated NISAR data lands mid-2026. → R3.

Validate every release against the pinned dolphin oracle; keep fmt/clippy/test/doc green;
semver throughout.

| Release | Date | Theme |
|---|---|---|
| baseline | now (pre-R1) | dolphinRust-vs-dolphin speed benchmark committed under `bench/` |
| v1.1.0 | 2026-07-01 | Close v1.0 residual + eo adoption + parity quick wins |
| v1.2.0 | 2026-09-01 | Quality layers (CRLB/closure) + production unwrapping (tophu) |
| v1.3.0 | 2026-11-01 | NISAR / L-band + atmospheric corrections |
| v1.4.0 | 2027-01-01 | Performance push + phase-bias + NRT incremental updates |

---

## Baseline — now (before R1)

**Speed benchmark** (~2 days). Per-frame wall-clock and phase-linking throughput,
dolphinRust vs Python dolphin v0.35.0, on the existing validation stacks; commit a
reproducible `bench/` with the numbers. This is the figure that justifies the rebuild and
sells NRT/throughput to eo — it's a measurement we already have everything to take, so it
runs ahead of the release train rather than inside it. It also sets the target R4 must beat.

---

## v1.1.0 — 2026-07-01 · "Close out v1.0, get into eo"

Short runway (~2 weeks) — with the benchmark already taken, scope is the v1.0 loose ends
plus the work that gets dolphinRust running inside GroundPulse.

- **Close the velocity-scale residual** (~2 days). One PS-rich, high-coherence subsidence
  scene (Mexico City or Las Vegas Valley) run through both engines to pin velocity absolute
  scale at real magnitude. Closes the one documented B4 gap in VALIDATION.md.
- **eo integration** (~1 wk). On an `eo` branch: a `gp-dolphin` crate calling
  `run_displacement` via `spawn_blocking`, wired as a `gp-tasks` task that lands a COG via
  `gp-storage` and summary rows in PostGIS. Aim for a working frame run, not just a skeleton
  (the benchmark moving out frees the time R1 would have spent on it).
- **Auto reference-point selection** (center-of-mass, dolphin v0.36.0) (~2–3 days). Cheap
  correctness/parity fix.

**Exit:** velocity scale confirmed on a deforming scene; ref-point matches oracle;
dolphinRust produces a displacement product end-to-end from a `gp-tasks` job on one frame.

---

## v1.2.0 — 2026-09-01 · "Quality layers + production-grade unwrapping"

- **CRLB uncertainty rasters** (dolphin v0.40) — **SHIPPED** (branch `v1.2-quality`,
  validated vs forward oracle dolphin v0.42.0, σ max |Δ| < 1e-4, singular-Γ NaN matches).
  Per-pixel per-date phase-estimate σ from the Fisher information of the coherence model,
  CPU `faer`/f64; on `DisplacementOutput.crlb_sigma` + per-band COGs, default on. *Feeds
  GroundPulse `confidence_score`/risk tiers a real physical uncertainty — a product
  capability, not just parity.* GPU CRLB is a later follow-up.
- **Sequential closure-phase rasters** (dolphin v0.41) — **SHIPPED** (same branch/oracle,
  closure max |Δ| < 1e-4). Nearest-neighbour triplet non-closure on
  `DisplacementOutput.closure_phase` + per-band COGs (default off, matching dolphin); the
  prerequisite signal for phase-bias work in R4.
- **tophu-style multi-scale tiled unwrapping** — **SHIPPED** (branch `v1.2-unwrap`,
  `dolphin-unwrap::unwrap_multiscale`): coherence-weighted coarse multilook (low-trust blocks
  masked + filled) → single SNAPHU → nearest upsample → overlapping tiled SNAPHU (rayon) →
  overlap-based inter-tile cycle reconciliation (max-reliability spanning forest) → feathered
  merge. Opt-in via `unwrap_method: tophu`; **SNAPHU stays the default path.** Correct
  (contract tests: ramp recovery within the SNAPHU envelope, planted 2π jump, 2×2
  loop-consistency, weighted-coarse-tracks-truth). **Now beats raw SNAPHU on the frozen
  low-coherence scenes on all three metrics on both scenes** (discont −9 % both,
  gross-cycle-err −10 % steep, rms ≤ raw both) — scenes/metrics unchanged from the earlier
  honest-loss run, only the algorithm changed. Numbers + margins in `bench/UNWRAP.md`.
- **Per-ministack temporal-coherence stitching** (dolphin v0.41) — **SHIPPED** (same branch):
  cross-ministack reduction is now dolphin's NaN-aware mean (`numpy.nanmean`) instead of a
  zero-diluting plain mean, matching the layer the per-band CRLB/closure concatenate against
  (caveat closed). Contract vs v0.42 oracle on a 2-ministack stack (`gen_stitch_v042.py`):
  stitched temp_coh + concatenated CRLB + closure all < 1e-3.
- **Finish eo integration** to production (PostGIS summary rows + COG, behind a gp-tasks
  task) if not completed in R1.

**Exit:** CRLB + closure rasters match the v0.4x oracle ✅; tophu implemented + correct and
**now beats SNAPHU on the frozen low-coherence scenes on all three metrics** (measured,
reported in `bench/UNWRAP.md`; SNAPHU stays the default for small/coherent scenes) ✅;
per-ministack stitching matches the oracle ✅; dolphinRust running in eo's worker (R1).

---

## v1.3.0 — 2026-11-01 · "NISAR / L-band + atmosphere"

- **NISAR RSLC reader** ✅ (v1.3 part 1). HDF5 + complex-int16 compound types — hdf5-metno
  reads the `{r,i}` int16 compound via a derived `H5Type` (de-risk cleared). NISAR product
  group structure + custom geotransform/EPSG (`epsg_code` attribute). `input_type: nisar_gslc`
  selector (forward divergence); L-band λ threads to mm/yr. End-to-end on a synthesized NISAR
  stack (typed output + COGs). **Atmospheric corrections still pending** (below) — the product
  is geometrically correct but atmospherically uncorrected.
- **L-band ionospheric correction** ✅ (v1.3 part 2). IONEX TEC → `1/f²`-scaled L-band range
  delay (`K=40.31`, Yunjun 2022), in the new `dolphin-corrections` crate. Closed-form contract
  green; **validated on a real IGS GIM from CDDIS — 56.5 TECU → 14.4 m L-band delay (18.5×
  C-band)**. Mandatory for usable L-band, confirmed on real data.
- **OPERA L4 tropospheric product ingest** ✅ (v1.3 part 2). GDAL `NETCDF:` ingest + bilinear
  resample + zenith→slant. Synthesized-fixture contract green; **real `OPERA_L4_TROPO-ZENITH_V1`
  granule ingested (ASF, 2 GB): total ZTD = hydrostatic + wet ≈ 2.79 m centre.** Full real-frame
  application (global 4326 → UTM warp) deferred-with-receipts; see VALIDATION.md.
- **RAiDER dispatch** ✅ wired behind an availability check (subprocess + GDAL ingest), gated
  like SNAPHU; deferred this run (RAiDER not installed) — never stubbed. L4 is the primary path.
- L-band spectral parameters in covariance estimation. *(deferred to v1.4 — not required for the
  atmospheric-correction exit.)*

**Exit:** dolphinRust ingests a NISAR RSLC stack and produces displacement ✅; tropospheric +
ionospheric corrections applied and **validated against real OPERA/IGS atmospheric layers** ✅
(real IONEX + real OPERA L4 both reachable and validated; full real-frame tropo warp deferred).
**v1.3.0 complete** (both corrections fixture-proven, wired, real-source-validated).

---

## v1.4.0 — 2027-01-01 · "Performance + near-real-time + lead dolphin"

- **NRT incremental ministack updates** (NEW) (~3–4 wks). A streaming mode that folds a newly
  arrived acquisition into an existing time series using the carried compressed SLC — no full
  reprocessing. *Rationale: this is the capability that turns the compiled-binary / no-JIT
  speed edge into an operational one. dolphin's model is batch; eo monitors continuously
  (new Sentinel-1/NISAR frame → updated displacement in minutes, not a full rerun). It's both
  a lead over dolphin and the natural payoff of the baseline benchmark.*
- **Performance optimization** (beat the pre-R1 baseline) (~3–4 wks): faer small-matrix
  (N×N covariance, N≈10–30) tuning, `EagerLoader`-style block prefetch, streaming I/O,
  thread-pool/BLAS contention. Target a documented multiple over Python dolphin/CPU.
- **Phase-bias / non-closure correction** (Michaelides et al., RSE 2022) (~4–6 wks). *Not in
  Python dolphin* — puts dolphinRust ahead of the oracle on correctness. Uses the
  closure-phase layer from R2. High value for slow long-baseline signals (the regime eo
  cares about).
- **3D-unwrap-ready dispatch interface** (~1 wk). Abstract the unwrap backend so a spurt-style
  3D spatiotemporal solver can drop in later without a refactor. Monitor spurt maturity; do
  not port it yet.

**Exit:** an incoming frame updates an existing time series incrementally (validated against
a full rerun); published speedup figure vs the pre-R1 baseline; phase-bias correction reduces
non-closure on a long series; unwrap interface ready for a 3D backend.

---

## Deferred (with rationale)

- **spurt 3D unwrapping port** — spurt is v0.1.x, pre-production, and dolphin hasn't adopted
  it. Design the interface (R4); port only once it stabilizes or dolphin integrates it.
- **GPU acceleration — SHIPPED as a first-class backend** (branch `gpu-first-class`, wgpu/Metal,
  compiled into the **default build**; see `bench/GPU.md`, `VALIDATION.md`). Runtime-selected via
  `worker_settings.compute_backend` (`auto`/`cpu`/`gpu`); no adapter / unsupported / `no-gpu`
  build → automatic CPU fallback, never a panic. The CPU (faer, f64) path stays the correctness
  reference. Closed every spike gap:
  - **EMI is now all-pixel-accurate** via a hybrid — the GPU kernel flags ill-conditioned /
    near-degenerate / borderline-PD pixels and the host recomputes that 5.9% minority on f64
    faer. Real Mexico 384² stack: **max Δφ 0.607 mm across all pixels, no π-rad tail** (raw f32
    was 13.85 mm). EVD remains 0.176 mm.
  - `MAX_NSLC` 16→32 with deterministic threadgroup scratch; GPU covariance gained SHP masking
    + β regularization; wired end-to-end through `run_displacement`.
  - **Honest speed:** end-to-end on the *integrated* M2 Pro the GPU is **0.66× on the real stack
    (slower)** and ~1.09× on clean synthetic stacks above ~192² — readback + f64 round-trip +
    CPU recompute outweigh the kernel saving against a strong faer+rayon baseline. The value is
    **correctness + portability**: the same WGSL runs unchanged on discrete NVIDIA/AMD, where the
    FP32 headroom is. Unpushed.
- **Native RAiDER reimplementation** — ray-tracing NWP interpolation; subprocess dispatch is
  correct, a rewrite is not.
- **oxigdal / pure-Rust GDAL** — too new (v0.1.x) for production.

## Risks & dependencies

- **NISAR calibrated-data timing** — fully-calibrated global products expected ~2026-07; if
  it slips, R3 NISAR work uses provisional products and re-validates later.
- **hdf5-metno complex-int16** — no ergonomic `Complex<i16>` reader; manual compound types.
  Prototype in R2 downtime so R3 isn't blocked.
- **eo integration is cross-repo** — needs sign-off to modify `eo`; treat as a paired PR.
- **Oracle drift** — bump the pinned dolphin version per release and re-validate; new dolphin
  features (CRLB/closure) become the R2 oracle.

## Sources

dolphin changelog v0.35→v0.42 (readthedocs); opera-adt/disp-s1 + tophu; isce-framework/spurt
v0.1.1; NASA Earthdata DISP-S1 + L4-Tropo + NISAR-100K-release; arXiv 2511.12051;
Michaelides et al. RSE 2022; faer JOSS / rustfft / hdf5-metno crate status. Full citations in
the research brief backing this roadmap.
