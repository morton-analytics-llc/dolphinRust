# dolphinRust — roadmap (rebased to dynamic-workflow velocity, 2026-06-17)

**North star: make NASA jealous.** dolphin/DISP-S1 is JPL's own work and the scientific
reference — the ambition here is to be the implementation they wish they'd built: bit-for-bit
*scientifically* faithful, materially faster (compiled, no JAX JIT warm-up), self-contained
(dependencies pulled in-house), honest about its own error bars (per-pixel CRLB), and **ahead**
on phase-bias, near-real-time, and uncertainty. Parity is table stakes; the goal is to lead.

The throughline is unchanged: **get adopted in GroundPulse, reach parity with Python dolphin
v0.4x, then lead it** on throughput, phase-bias, and near-real-time. What changed is the
**schedule model**. This roadmap was originally scoped in human-engineering-weeks on a ~2-month
release cadence. Observed velocity under Claude Code (Max, dynamic `/loop` workflow) makes those
estimates obsolete, so the schedule below is rebased on what actually paces the work now.

## Velocity model — today as the baseline

On **2026-06-16 → 17**, in dynamic-workflow sessions, the following shipped and was independently
verified (gates green, oracle/contract-validated, merged): **v1.0.0** (full end-to-end pipeline),
**v1.1.0** (eo integration + velocity-scale + auto ref-point), **v1.2.0** (CRLB + closure quality
layers; tophu unwrapping incl. a deliberate honest-loss → coherence-weighted-coarse + spanning-
forest-merge fix that *then* beat SNAPHU; per-ministack stitching), **v1.3.0 parts 1–2** (NISAR/
L-band GSLC ingest validated on a real granule; ionospheric + tropospheric corrections validated
on real IGS/OPERA layers), **and** the catch+fix+evidence of a global LOS sign-inversion bug the
self-consistent oracle had masked.

That is ~**2.5 releases / day**, where the original plan budgeted ~2 months each. The unit of
progress is now **one dynamic-workflow loop ≈ one release-half or substantial feature**, landing
in tens of minutes of autonomous work plus verification and a sign-off pause.

**Implication: engineering effort is no longer the schedule driver.** The full original four-
release year (v1.1–v1.4) completes inside **June 2026 — days, not quarters.** What genuinely
takes calendar time from here is *external*:

1. **External data arrival** — NISAR calibrated repeat-pass stacks (≥2 co-located dates) for real
   multi-date displacement validation; expected ~2026-07+, the hard gate on real NISAR claims.
2. **Real-world validation campaigns** — deforming scenes with ground truth to confirm
   phase-bias, NRT, and absolute-scale claims against reality, not just the oracle.
3. **Hardware access** — discrete NVIDIA/AMD GPU (e.g. rented) to validate the GPU path where the
   FP32 headroom actually pays off; the integrated-M2 result is marginal by design.
4. **Upstream dolphin & ecosystem cadence** — new dolphin releases to parity/lead against; spurt
   3D-unwrap maturing enough to adopt.
5. **eo operational feedback** — real frames flowing through the worker, surfacing what production
   actually needs.
6. **Human review/sign-off cadence** — the one in-loop throttle, deliberately kept (each merge/tag
   waits on sign-off).

So the schedule has two regimes: a **near-term build sprint** (effort-bound, days) that finishes
the planned feature set, then a **capability-gated phase** (calendar-bound, the rest of the year)
driven by the six factors above.

| Release | Date (rebased) | Status | Theme |
|---|---|---|---|
| v1.0.0 | 2026-06-16 | ✅ shipped | Full end-to-end displacement pipeline |
| v1.1.0 | 2026-06-16 | ✅ shipped | eo adoption + velocity scale + auto ref-point |
| v1.2.0 | 2026-06-17 | ✅ shipped | Quality layers (CRLB/closure) + tophu unwrapping + stitching |
| v1.3.0 | 2026-06-17 | ✅ complete, pending tag | NISAR/L-band ingest + atmospheric corrections (incl. tropo 4326→UTM warp) |
| v1.4.0 | 2026-06-17 | 🔄 in flight today | Performance + NRT incremental + phase-bias + 3D-unwrap interface |
| v1.5.0+ | 2026-H2 → 2027 | ⏳ gated | Capability-gated: real-data validation, discrete-GPU, lead-dolphin |

The original speed **baseline** is committed (`bench/results.json`, `bench/runs/`) — it gates
nothing and is the target v1.4.0 performance work must beat.

---

## v1.0.0–v1.2.0 — ✅ SHIPPED (2026-06-16/17)

Tagged on `main`. Details (kept as record):
- **v1.0.0** — read CSLC HDF5 → sequential phase-linking → ifg network → SNAPHU → SBAS → velocity
  → COGs. Validated vs dolphin v0.35.0 oracle. *(Note: v1.0–v1.2 carried an inverted LOS sign vs
  dolphin, masked by a lockstep-inverted oracle; caught and fixed 2026-06-17 — see VALIDATION.md /
  CHANGELOG. Re-run any pre-fix eo output.)*
- **v1.1.0** — eo `gp-dolphin` integration (one real frame end-to-end), velocity-scale residual
  closed, auto center-of-mass reference point.
- **v1.2.0** — CRLB σ (feeds eo `confidence_score`) + closure-phase rasters vs forward v0.42
  oracle (<1e-4); **tophu multi-scale unwrapping that beats raw SNAPHU on the frozen low-coherence
  scenes** (discont −9%, gross-cycle −10%; `bench/UNWRAP.md`), SNAPHU still default; per-ministack
  NaN-aware stitching (closes the CRLB/closure many-ministack caveat).

---

## v1.3.0 — ✅ complete, pending tag sign-off · "NISAR / L-band + atmosphere"

Both parts landed and merged; **the deferred tropo warp now lands — ready to tag.**
- **NISAR/L-band GSLC reader** ✅ — real NISAR GSLC is complex-**f32** `{r,i}` (the prompt's int16
  assumption was wrong; corrected against a real granule). Custom geocoding geotransform/EPSG;
  `input_type: nisar_gslc`; L-band λ end-to-end. Reader validated on a real
  `NISAR_L2_GSLC_BETA_V1` granule.
- **Ionospheric correction** ✅ — IONEX TEC → `1/f²` L-band delay; validated on a real IGS GIM
  (56.5 TECU → 14.4 m, 18.5× C-band).
- **Tropospheric correction** ✅ — OPERA L4 netCDF ingest + **4326→UTM warp** (`warp_to_frame`,
  GDAL bilinear `reproject`). Synthesized + 4326→UTM warp contracts green; **real granule applied
  end-to-end on the real Mexico City UTM 32614 frame: zenith mean 2.553 m (slant@39° ≈ 3.285 m).**
  The CRS-mismatch `warn!` path is gone. **v1.3.0 is ready to tag.**
- **Deferred (data-gated):** NISAR multi-date real displacement — needs ≥2 co-located repeat-pass
  dates; moves to the capability-gated phase below.

---

## v1.4.0 — ▶ next build sprint (target ~2026-06-20) · "Performance + NRT + lead dolphin"

Effort-bound; expect it within days, landing as units with sign-off. Per-phase scope and bars are
in `REMAINING_WORK_PROMPT.md`.
- **NRT incremental ministack updates** ✅ — folds new acquisitions into an existing series via the
  carried compressed SLC, re-phase-linking only the open trailing ministack + new ones.
  - *Phase 2 (core):* `run_sequential_resumable` + `update_sequential`, **bit-identical to a full
    rerun** (max|Δ| = 0 across phase/compressed/temp-coh/CRLB/closure; block + one-at-a-time
    streaming + boundary cases) — exact because sequential phase-linking is feed-forward.
  - *Phase 2b (front door):* `run_displacement_resumable` + `update_displacement` carry per-burst
    state through the whole pipeline; the downstream (network→unwrap→timeseries→velocity) is
    non-causal and recomputes from the updated phase history. End-to-end incremental update is
    **bit-identical to a full `run_displacement`** of the extended stack (max|Δ| = 0 through SNAPHU
    + SBAS). Exposed as a `dolphin stream` CLI subcommand (process an initial window, fold each
    later acquisition in). The speed win is skipping re-phase-linking the sealed history.
- **Performance optimization** ✅ — beat the committed baseline (`bench/results.json`). Covariance
  hot-path rewrite (the #1 phase-linking cost): direct **Hermitian** product (upper triangle +
  mirror) over contiguous rows, replacing ndarray's generic complex `dot` (no SIMD/BLAS for
  `Complex<f64>`) and its per-pixel conjugate-transpose alloc. **Real-frame phase-linking 2.38×
  faster** (host-controlled same-session A/B, 3.07→1.29 s; throughput 432→1028 kpix·slc/s), also
  beating the committed 2.01 s absolutely. No accuracy change (all oracle/analytic/sign contracts
  green). Numbers in `bench/PERF.md`. (faer micro-tuning / EagerLoader prefetch not pursued — the
  covariance win cleared the bar; end-to-end stays gated by the Rosetta SNAPHU binary, a packaging
  issue.)
- **Phase-bias / non-closure correction** (Michaelides et al., RSE 2022) — *not in dolphin*, so it
  leads the oracle; validated by measured non-closure reduction on the v1.2 closure layer.
- **3D-unwrap-ready dispatch interface** — trait behind which SNAPHU/tophu sit; no spurt port.

**Exit:** NRT == full rerun within tolerance; published speedup vs baseline; phase-bias reduces
non-closure on a long series; unwrap interface ready for a 3D backend.

---

## After v1.4.0 — the job is done; keep it accurate and fast

dolphinRust's job is narrow and, once v1.4.0 lands (today), substantially complete: **do the
InSAR processing, provide per-pixel uncertainty (CRLB), and produce accurate output — fast.**
There is no feature backlog to invent. The only ongoing work serves that mission, and it is
gated by external events, not coding effort:

- **Accuracy on real data** *(gate: data as it arrives)* — validate against reality, not just the
  oracle: NISAR multi-date displacement when calibrated repeat-pass stacks land (~2026-07+), and
  real deforming-scene/ground-truth checks of velocity scale, phase-bias non-closure, and
  NRT-vs-rerun equivalence. Standing lesson from the sign-inversion bug: a self-consistent oracle
  can't catch a shared-convention error — real-production comparison is the check.
- **Hold parity with upstream dolphin** *(gate: dolphin releases)* — bump the pinned oracle per
  cycle and re-validate. Cheap now.
- **Performance** *(gate: real bottlenecks / hardware)* — continue beating the baseline only where
  profiling shows a genuine hotspot; discrete-GPU validation when that hardware is available (the
  wgpu path already runs unchanged — a measurement, not a build). No speculative tuning.
- **Bring dependencies in-house** *(gate: payoff vs. risk per dependency)* — reduce reliance on
  external C libraries and subprocesses where it makes the core job more self-contained, portable,
  and accurate. This is hardening, not feature work. Highest-value candidates, in order:
  - **Native phase unwrapper** — replace the **SNAPHU** subprocess (PATH/binary fragility, flat-
    binary round-trips). A pure-Rust MCF/network-flow unwrapper removes the only runtime external
    binary and lets tophu tile in-process. Validate against current SNAPHU output before switching
    the default.
  - **Native HDF5/CSLC reader** — the `hdf5-metno` system-lib link is the exact constraint that
    forced `gp-dolphin` into its own workspace + worker in eo (two `links="hdf5"` crates can't
    coexist). A pure-Rust reader of the CSLC/GSLC layout we actually consume would dissolve that
    and simplify eo integration. Scope to the layouts we read, not all of HDF5.
  - **Native raster I/O (GDAL)** — track pure-Rust GeoTIFF/COG read+write to drop `gdal`/`gdal-sys`
    eventually; gated on maturity (oxigdal is still v0.1.x). Lower priority than the above two.
  - RAiDER stays a subprocess fallback — a native NWP ray-tracer is the lowest-payoff in-housing
    and only if the subprocess proves insufficient.

That's it. No fixed dates: each item's Claude-cost is small but cannot start until its gate opens
or a dependency's in-house payoff clears its risk. *(An R interface over `run_displacement` is
possible later, but it's consumer reach, not the processing job — out of scope unless a real R
consumer needs it.)*

---

## Deferred (with rationale)

- **GPU acceleration — SHIPPED first-class** (wgpu/Metal, default build; `bench/GPU.md`). Runtime-
  selected `compute_backend` (`auto`/`cpu`/`gpu`), automatic CPU fallback. EMI all-pixel-accurate
  via the f64-recompute hybrid (max Δφ 0.607 mm, no π-rad tail; EVD 0.176 mm). Honest speed: on the
  *integrated* M2 Pro the GPU is ~0.66× on the real stack — the value is correctness + portability
  to discrete hardware. Discrete-GPU validation is in the capability-gated phase.
- **GPU CRLB** — CRLB is CPU/faor f64 only; a GPU port is a later follow-up, not required.
- **spurt 3D unwrapping** — external project still v0.1.x; design the interface (v1.4), port only
  once it stabilizes or dolphin adopts it.
- **Native GDAL (oxigdal) / native RAiDER** — moved from "rejected" to **in-housing candidates**
  (see "Bring dependencies in-house" above): gated on maturity/payoff, not off the table.

## Risks & dependencies

- **NISAR calibrated-data timing** — the binding external gate (~2026-07); the reader + L-band path
  are ready and validated on a single real granule, so this is purely a multi-date-validation wait.
- **Oracle drift** — bump the pinned dolphin version per cycle and re-validate; CRLB/closure use a
  forward v0.42 oracle while existing kernels stay pinned at v0.35.0.
- **eo integration is cross-repo** — paired-PR with sign-off; the sign-fix re-run is a live action
  item there.
- **Self-consistent-oracle blind spots** — the sign-inversion bug proved a self-generated oracle
  can't catch a shared-convention error; prefer real-production comparison for convention/sign
  claims (now guarded by `tests/sign_convention.rs`).

## Sources

dolphin changelog v0.35→v0.42; opera-adt/disp-s1 + tophu; isce-framework/spurt v0.1.1; NASA
Earthdata DISP-S1 + L4-Tropo + NISAR; Michaelides et al. RSE 2022; faer / rustfft / hdf5-metno.
Schedule model rebased on observed 2026-06-16/17 dynamic-workflow throughput (this repo's git
history + VALIDATION.md).
