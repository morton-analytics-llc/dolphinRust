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

- **CRLB uncertainty rasters** (dolphin v0.40) (~1–2 wks). Per-pixel phase-estimate σ from
  the Fisher information of the covariance model. *Rationale: feeds GroundPulse
  `confidence_score`/risk tiers a real physical uncertainty — a product capability, not just
  parity.*
- **Sequential closure-phase rasters** (dolphin v0.41) (~1 wk). Triplet non-closure
  diagnostic; also the prerequisite signal for phase-bias work in R4.
- **tophu-style multi-scale tiled unwrapping** (~2–3 wks). OPERA's *production* unwrapper —
  coarse-resolution init feeding tiled SNAPHU, merged. Materially better than raw SNAPHU on
  large, low-coherence (vegetated) scenes. Keep SNAPHU as the simple path.
- **Per-ministack temporal-coherence stitching** (dolphin v0.41) (~3–5 days). Full ministack
  correctness for frame mosaics.
- **Finish eo integration** to production (PostGIS summary rows + COG, behind a gp-tasks
  task) if not completed in R1.

**Exit:** CRLB + closure rasters match the v0.4x oracle; tophu beats SNAPHU on a
low-coherence scene (fewer unwrap errors, measured); dolphinRust running in eo's worker.

---

## v1.3.0 — 2026-11-01 · "NISAR / L-band + atmosphere"

- **NISAR RSLC reader** (~1–2 wks). HDF5 + complex-int16 compound types (de-risk the
  hdf5-metno ergonomics early — it blocks everything L-band). NISAR product group structure.
- **L-band ionospheric correction** (~1–2 wks). TEC/IONEx — ionosphere is ~16× the C-band
  effect, so this is mandatory for usable L-band, not optional.
- **OPERA L4 tropospheric product ingest** (~3–5 days). Free public netCDF aligned to
  DISP-S1 frames; lowest-effort, user-expected atmospheric path.
- **RAiDER dispatch** (subprocess + GDAL ingest) (~1–2 wks) for scenes without an L4 product.
- L-band spectral parameters in covariance estimation.

**Exit:** dolphinRust ingests a NISAR RSLC stack and produces displacement; tropospheric +
ionospheric corrections applied and validated against an OPERA correction layer.

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
