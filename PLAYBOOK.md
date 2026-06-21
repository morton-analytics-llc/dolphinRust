# dolphinRust Implementation Playbook

Phased plan for a ground-up Rust **rebuild** of the DISP-S1 wrapped-phase → displacement
pipeline, optimized for performance. This is **not a port**:
[dolphin](https://github.com/isce-framework/dolphin) is the algorithm reference (the
scientific spec), and we are free to choose the fastest correct Rust realization of each
algorithm. The goal is **scientific correctness**, validated at each phase against analytic
fixtures and dolphin outputs used as a reference oracle.

**Pinned dolphin reference: `v0.35.0`** (commit `e567e554300f9bb2c6c4c49358d41876ce81e5a7`,
`isce-framework/dolphin`). All oracle data is generated from this version so validation is
reproducible. (Phase 0 types/config were initially mirrored from `main`; reconcile any
default drift against v0.35.0 if a Phase-0 contract ever depends on it.)

---

## Architecture decisions

1. **`rayon` over pixels, not `jax.vmap`.** dolphin's hot kernels are `jax`
   `vmap(vmap(f))` over the `(rows, cols)` grid, each cell solving one `N×N` complex
   matrix. The Rust equivalent is a `rayon` parallel iterator over flattened pixel
   indices; each closure owns one matrix and calls `faer`. This removes JAX JIT
   cold-start and per-static-arg recompilation.
2. **`faer` for dense complex linear algebra** (Cholesky, LU/shift-invert, eigen),
   `ndarray` for array scaffolding and block slicing, `num-complex` for elements,
   `rustfft` for filters. `faer` is preferred over `nalgebra` for throughput on the
   small dense `N×N` systems (N = ministack size, typically 10–30).
3. **Shell out for unwrapping.** SNAPHU/tophu/spurt/whirlwind are external solvers;
   dolphin contains no unwrapping math. `dolphin-unwrap` orchestrates the SNAPHU binary
   via subprocess. Not a reimplementation target.
4. **f64 inside kernels, f32 at the I/O boundary.** SLCs are `complex64` (f32) on disk;
   covariance/eigensolver math runs in `Cf64` to match NumPy/JAX default accumulation
   precision, then casts back. This matters for hitting the correctness tolerances.
5. **System-lib deps deferred to Phase 8.** GDAL/HDF5/LAPACK bindings are introduced
   only when the I/O layer lands, so the numerical core builds and tests on any machine
   with synthetic in-memory arrays.
6. **Stage, don't stream (S3).** Raw CSLC stacks live in S3 (the host app puts them
   there; dolphinRust only *reads*). Do not read processing blocks over `/vsis3/` —
   phase linking is sliding-window, so every pixel is read many times across overlapping
   covariance windows, and OPERA CSLC HDF5 is not cloud-optimized (random access over S3
   is pathological). Download each granule **once** to local scratch/tmpfs, open with
   GDAL/HDF5 locally, delete after. COG GeoTIFFs are the sole exception — those may be
   read via GDAL `/vsis3/` directly. The concurrent download is the *only* async stage.
7. **Runtime-agnostic public API.** Compute crates stay sync + `rayon` and own no
   runtime. S3 read-staging lives in a feature-gated `dolphin-ingest` crate
   (`object_store` + `tokio`, off by default) that downloads concurrently and returns
   local paths. The library's public entry points are synchronous (`fn run(cfg) -> …`);
   the host app — which already has a tokio runtime — bridges via `spawn_blocking` / a
   dedicated thread so a long CPU-bound burst run never blocks its reactor.

---

## Dependency / environment setup (do once, before Phase 8)

The numerical phases (1–7) need no system libraries. The I/O phase does:

```sh
# macOS
brew install gdal hdf5 openblas
# Debian/Ubuntu
apt-get install libgdal-dev libhdf5-dev liblapack-dev libopenblas-dev
```

Then enable in `dolphin-io` / `dolphin-timeseries`:
- `gdal = "0.17"` (raster I/O; links system GDAL ≥ 3.4)
- `hdf5 = "0.8"` (CSLC subdataset reading)
- `ndarray-linalg = { version = "0.16", features = ["openblas-system"] }` *only if*
  `faer`'s least-squares proves insufficient for the SBAS solve (it should not).

Run `command -v gdal-config h5cc || echo missing` at the top of Phase 8 to fail fast.

---

## Correctness & validation strategy

Since this is a rebuild, not a port, validation proves the Rust kernels are *scientifically
correct* — it does not chase bit-exactness with the Python. Two complementary checks per
numerical phase, contract test written FIRST (red):

1. **Analytic fixtures (primary).** A synthetic input with a known closed-form answer
   (e.g. a coherence matrix whose dominant eigenvector is known by construction; a PS-like
   stable point with `D_A → 0`). These are Rust-native fixtures — no Python dependency —
   and are the real correctness contract.
2. **Reference oracle (secondary).** In a scratch Python env, install the pinned dolphin
   and emit outputs for the same synthetic stack (`oracle/gen_<module>.py`, data not
   committed; fixed seed; dump to `.npy`, load via `ndarray-npy` dev-dependency). Confirms
   we agree with the established implementation where the algorithm has no closed form.

**Tolerances are physical, not numerical-identity:** phase compared modulo `2π` and up to a
global phase reference; coherence to `atol≈1e-4`; eigenvectors as `|⟨v_rust, v_oracle⟩|`
(sign / global-phase ambiguity). Where an optimized Rust algorithm choice diverges from
dolphin's numerics (different eigensolver, accumulation order), that is fine as long as it
stays inside these tolerances — note the choice and why.

A kernel is "done" only when its contract test is green. Code existence is not done.

---

## Phase 0 — Foundation (`dolphin-core`)

**Scope.** No numerics; everything downstream depends on these types.

- `types`: `Cf32`/`Cf64` (done), `HalfWindow { y, x }`, `Strides { y, x }`,
  acquisition date wrappers.
- `blocks`: build `StridedBlockManager` / `BlockIndices` (algorithm from `io/_blocks.py`) — the
  5-tuple (out block, out trim, in block, in-no-pad, in trim) halo scheme. This is the
  single most reused struct; get it exactly right with property tests (every input pixel
  covered exactly once after trimming; output strides honored).
- `config`: `serde` structs mirroring the pydantic `DisplacementWorkflow` tree
  (`PhaseLinkingOptions`, `PsOptions`, `InterferogramNetwork`, `UnwrapOptions`,
  `TimeseriesOptions`, `WorkerSettings`) with identical defaults and YAML field names so
  existing dolphin configs deserialize unchanged. Add a round-trip test against a real
  dolphin YAML.
- `error`: `thiserror` enum.

**Done when:** block manager property tests pass; a sample dolphin displacement YAML
deserializes with all defaults matching the Python brief (§6).

---

## Phase 1 — Covariance + EMI/EVD phase linking (`dolphin-phaselink`) ★ highest value

**Scope.** `phase_link/covariance.py`, `_core.py`, `_eigenvalues.py`.

1. **Covariance estimation.** Sliding `(2·half_y+1)×(2·half_x+1)` window over the
   `N×rows×cols` stack; per output pixel build the normalized coherence matrix
   `C_ij = Σ(z_i z_j*) / sqrt(Σ|z_i|² · Σ|z_j|²)`, optionally masked by the SHP neighbor
   array (Phase 2). Parallelize over output pixels with `rayon`. Respect `strides` for
   output decimation. This is the #1 hot path.
2. **Eigensolvers.** Iterative power / inverse iteration is dolphin's approach; faer's
   direct dense eigensolver is a candidate too. The N×N systems are small — pick whichever
   is faster as long as it converges to the correct eigenvector within tolerance.
3. **EVD estimator.** Largest eigenvector of `C ⊙ |C|`.
4. **EMI estimator (default).** `Γ = |C|`; regularize `Γ ← (1-β)Γ + βI`; threshold
   near-zero entries (`zero_correlation_threshold`); Cholesky-invert with `1e-6` jitter;
   smallest eigenvector of `Γ⁻¹ ⊙ C`. **Fallback to EVD on singular `Γ⁻¹` (NaN)** — match
   dolphin's `lax.select` behavior exactly.
5. **Phase referencing.** `θ ← θ · exp(-j·∠θ[ref_idx])`.

**Done when:** EVD and EMI eigenvector contract tests pass on the synthetic DS fixture
(compare `|⟨v,v_py⟩|` and referenced phase); singular-matrix fallback verified.

---

## Phase 2 — SHP selection (`dolphin-shp`)

**Scope.** `shp/_glrt.py`, `shp/_ks.py`. Feeds the neighbor mask into Phase 1.

- **GLRT (default).** Rayleigh amplitude model; `σ² = (var+mean²)/2`;
  `T = N·(2·log σ_pooled − log σ_1 − log σ_2)`; threshold `χ²(1, 1−α)`, α=0.001 via
  `statrs`. Parallel over center pixels.
- **KS test.** Per-pixel-pair sorted-amplitude ECDF max distance vs. critical value;
  the numba `njit(parallel=True)` loop → `rayon`.
- Output: boolean `(rows, cols, win_h, win_w)` neighbor array.

**Done when:** GLRT and KS neighbor arrays match the oracle's boolean decision on the
fixture; wire into Phase 1 covariance and re-run Phase 1 validation with SHP weighting on.

---

## Phase 3 — PS selection (`dolphin-ps`)

**Scope.** `ps.py`. `D_A = std(|z|)/mean(|z|)` over time, threshold 0.25 → uint8 mask
(1=PS, 255=nodata), tiled. PS-fill rule: PS pixels take phase from the brightest PS in
the look window and `temp_coh = 1.0`, bypassing covariance.

**Done when:** PS mask + amp_dispersion/amp_mean rasters match the fixture; PS-fill
integrates with Phase 1 output.

---

## Phase 4 — Quality layers (`dolphin-phaselink`)

**Scope.** `metrics.py`, `crlb.py`, `_closure_phase.py`, `_compress.py`.

- **Temporal coherence** `|Σ_{i>j} C_ij exp(-j(θ_i−θ_j)) W_ij| / Σ W_ij`.
- **CRLB**: Fisher matrix from `Γ`, `Γ⁻¹`; invert; per-pixel phase σ. (`write_crlb`)
- **Closure phase**: nearest triple `angle(C_{i,i+1} C_{i+1,i+2} C*_{i,i+2})`.
- **Compressed SLC**: `Σ_k z_k exp(-j θ_k)/N` projection; magnitude from mean amplitude.
  Needed by Phase 5.

**Done when:** temp_coh, CRLB, closure, and compressed-SLC contract tests pass.

> **Pin note (v0.35.0):** the pinned reference has **no `crlb.py` or `_closure_phase.py`**
> (those are `main`-only, as are the `write_crlb`/`write_closure_phase` config flags).
> Phase 4 shipped temp_coh + compressed SLC with oracle validation; **CRLB and closure
> phase are deferred** — they are optional quality side-outputs, off the v1.0.0 displacement
> critical path (1→5→6→10). Revisit if a newer dolphin pin or those rasters are required.

---

## Phase 5 — Ministack sequencing (`dolphin-stack` + `dolphin-workflows::sequential`)

**Scope.** `stack.py`, `workflows/sequential.py`. Ansari et al. (2017) sequential
estimator.

- `MiniStackPlanner`: partition N dates into `ministack_size` (15) batches; plan
  compressed-SLC carry-forward (`ALWAYS_FIRST`); enforce `max_num_compressed` (10).
- Sequential loop: ministack → SHP → covariance → phase-link → compress → next ministack
  prepends the compressed SLC. `_average_or_rename` merges temp_coh across ministacks.

**Done when:** planner output (batch composition, compressed-SLC placement) matches
dolphin for several N/size combinations; full sequential run on a multi-ministack
synthetic stack matches phase history end-to-end.

---

## Phase 6 — Interferogram network + SBAS inversion (`dolphin-timeseries`)

**Scope.** `interferogram.py` (network construction), `timeseries.py` (inversion).

- **Network**: from phase-linked SLCs, build the ifg set per `reference_idx` /
  `max_bandwidth` / `max_temporal_baseline` / explicit `indexes`.
- **SBAS L2 first**: incidence matrix `A (n_ifgs × n_dates−1)` of ±1; solve `Aφ = Δφ`
  weighted least squares via `faer`, block-parallel (256×256). Optional correlation
  weighting; `correlation_threshold` censoring.
- **Velocity**: linear regression of phase series → rate.
- **L1/ADMM deferred** to Phase 6b — build only after L2 is validated. Note: dolphin
  defaults to L1, so the L2-only interim is a known temporary divergence from the oracle.

**Done when:** L2 displacement series + velocity match the dolphin oracle (L2 mode) on
synthetic unwrapped ifgs; network construction matches for each network mode.

---

## Phase 7 — Filters (`dolphin-filtering`)

**Scope.** `filtering.py`, `goldstein.py`. Long-wavelength FFT Gaussian high-pass and
Goldstein adaptive filter via `rustfft`. Optional pre-unwrap stages.

**Done when:** filtered rasters match dolphin to `atol=1e-4`.

---

## Phase 8 — I/O layer (`dolphin-io` + `dolphin-ingest`) — introduces system libs

**Scope.** `io/_readers.py`, writers, and S3 read-staging. **Run the environment
preflight first.**

- `dolphin-ingest` (feature `s3`): given S3 URIs for a CSLC stack, download granules
  concurrently (`object_store` + bounded `tokio` runtime) to a local scratch dir, return
  local paths, clean up on drop. Read-only — dolphinRust never writes raw data to S3.
  Synchronous `stage(uris, scratch) -> Vec<PathBuf>` facade hides the runtime so callers
  stay sync. Off by default; local-path callers pull zero async deps.

- `gdal` crate: GeoTIFF block read/write; multi-band VRT construction for the SLC stack
  (`VRTStack` — auto-sort by date, NumPy-like 3D indexing).
- `hdf5` crate: OPERA/NISAR CSLC subdataset reading
  (`HDF5:"f.h5"://science/SENTINEL1/CSLC/grids/VV`); custom geotransform reader for NISAR
  (GDAL HDF5 driver returns identity).
- `EagerLoader`: background block prefetch (thread pool) wrapping any stack reader.
- Output: complex-f32 phase SLCs, f32 quality, uint8 PS, compressed SLCs.

**Done when:** round-trip read/write of a real CSLC HDF5 + GeoTIFF matches GDAL/h5py
byte-for-byte on geotransform, CRS, and pixel values.

---

## Phase 9 — Unwrapping dispatch (`dolphin-unwrap`)

**Scope.** `unwrap/`. Subprocess wrapper around the SNAPHU binary: tiling, cost model /
init method config, NPROC parallelism, nodata propagation, connected-component regrow.
tophu/spurt/whirlwind left as documented gaps unless required.

**Done when:** a wrapped ifg + correlation produces an unwrapped + conncomp raster
matching a direct SNAPHU invocation; `run_unwrap=false` path skips cleanly.

---

## Phase 10 — Pipeline orchestration + CLI (`dolphin-workflows`, `dolphin-cli`)

**Scope.** `workflows/displacement.py` order (brief §7): prepare/group inputs by burst →
per-burst wrapped_phase (mask → PS → SHP → covariance → phase-link → compress → ifg net)
→ stitch bursts → unwrap → timeseries → velocity. Burst-parallel executor (`rayon` /
process pool equivalent). `dolphin run --config <yaml>` drives it.

**Done when:** an end-to-end run on a small real OPERA CSLC burst stack produces a
displacement time series matching the dolphin oracle within tolerance; CLI config matches
dolphin's YAML.

---

## Build priority (critical path)

```
0 core ─► 1 phaselink ─► 2 shp ─┐
                 │              ├─► 5 sequencing ─► 6 timeseries ─► 10 pipeline
                 └─► 3 ps ──────┘                         ▲
                 └─► 4 quality                            │
8 io ───────────────────────────────────────────────────┤
9 unwrap ────────────────────────────────────────────────┘
7 filtering (optional, parallel)
```

Phases 1–5 are the differentiated value (the JAX/numba kernels). Phases 8–9 are
integration glue (bindings + subprocess) and can proceed in parallel once core types
(Phase 0) exist. Do **not** start Phase 10 until 1–6, 8, 9 each carry green contract tests.

---

## GroundPulse integration (host app: `../eo`)

GroundPulse is the consumer — a Rust monorepo (Axum/tokio/sqlx, PostGIS + TimescaleDB,
Postgres `SKIP LOCKED` task queue; S3 via `gp-storage`/aws-sdk-s3; GDAL + HDF5).

**GroundPulse is adopting the Python `dolphin` now.** dolphinRust is its **optimized Rust
drop-in replacement** — same algorithms, same workflow surface, faster. This sets the bar:

- **Match the Python dolphin GP runs, end to end.** Since GP runs Python dolphin in
  production, the cleanest oracle is GP's own production output: dolphinRust must reproduce
  it within the §Correctness tolerances. Migration = "swap `dolphin run` for dolphinRust,
  confirm equivalent displacement."
- **Full scope, not just the front half.** Mirror dolphin's whole pipeline including
  timeseries/SBAS (Phase 6). GP's existing `gp-displacement` SBAS (`sbas.rs`, Berardino
  2002) becomes legacy once it moves to dolphin; dolphinRust replaces *dolphin's*
  timeseries, not gp-displacement's. (Resolves the earlier SBAS-overlap question.)
- **Compatible config.** Accept dolphin's displacement-workflow YAML unchanged, so a GP
  task can point either implementation at the same config.
- **Consumed as a synchronous library** by a `gp-dolphin` / `gp-phase-linking` crate (or
  inside `gp-displacement`), called from a `gp-tasks` `Task` via `spawn_blocking` on its
  bounded worker runtime — replacing whatever subprocess/binding GP uses to invoke Python
  dolphin.
- **S3:** GroundPulse's `gp-storage` already stages S3 → local (staging-key + lifecycle
  pattern) and runs blocking work via `spawn_blocking`. On the GroundPulse path, GP stages
  and hands dolphinRust **local paths**; `dolphin-ingest` is for the standalone CLI only.
- Reuse GP conventions: EPSG:4326 geometry with native UTM in COG metadata; COG 512×512,
  DEFLATE+predictor3, overviews [2,4,8,16,32]; outputs as COG via `gp-storage`, summary
  stats to PostGIS.

---

## Optimization log

### Unwrap-network parallelization (Tier-1, 2026-06-20)

Unwrap was ~76% of full-res compute and ran the ifg network serially. Shipped on
`feat/unwrap-parallel` (3 commits, one per unit):

- **#1 parallelize + per-pair scratch isolation.** `unwrap_each_ifg` `.iter()` →
  `.par_iter()`; each pair solves into its own `pair_NNNN` scratch subdir so
  SNAPHU's fixed-name files (`ifg.c8`/`unw.f4`/`conncomp.u4`) never collide.
  `par_iter().collect()` is order-stable → output matches `pairs` order. **Bit-identical**
  to the serial golden (`tests/unwrap_parallel_contract.rs`, red→green). Concurrency
  is bounded by the existing `unwrap_options.n_parallel_jobs` knob (≤0 = all cores)
  via a pinned rayon pool.
- **#3 hoist shared correlation write.** corr.f4 is identical across pairs; written
  once into shared scratch + reused (`write_correlation` + `unwrap_with_corr`).
  Bit-identical.
- **#2 opt-in auto-tiling** (`snaphu_options.auto_tile`, default **off**). Changes
  SNAPHU numerics; **held opt-in** — smooth-ramp deviation 7.06e-5 rad (~3e-4 mm),
  but noisy-scene tiling has no large oracle fixture. #1 already saturates cores on
  deep networks, so #2's marginal value is low.

**Measured** (512×512 single-ref network, macOS 12-core, smooth synthetic):

| epochs | ifgs | 1T | 2T | 4T | 8T |
|--------|------|----|----|----|----|
| 12 | 11 | 5.89 s | 3.44 s (1.71×) | 2.00 s (2.95×) | 1.50 s (3.92×) |
| 30 | 29 | 15.33 s | 8.96 s (1.71×) | 5.07 s (3.02×) | 3.23 s (4.75×) |

Speedup flattens past 4 threads (~3×), reaching **3.9–4.75× at 8** (deeper networks
scale better; ceiling = per-ifg rust/I-O + SNAPHU process overhead). **RSS flat across
thread counts** (125→132 MB @12ep; 269→272 MB @30ep) — parallelism adds ~7 MB, no
regression vs the block-tiled win. Reproduce: `EPOCHS=12 RAYON_NUM_THREADS=4
cargo run --release --example unwrap_bench`.

**Next (measured ceiling → next lever):** the inter-process/scratch-I/O ceiling caps
2D per-ifg parallelism at ~4–5×. The next win is **Tier 2 in-process unwrapping**
(eliminate SNAPHU subprocess + flat-binary scratch I/O per ifg) or **Tier 3 3D
spatiotemporal backend** (the `UnwrapBackend` trait seam already exists).

---

## Out of scope (initial)

- `atmosphere/` tropospheric corrections (wraps external delay models) — defer.
- DISP-S1 HDF5 product schema — owned by `disp-s1` SAS, not dolphin; build only if needed.
- GPU paths — `rayon` CPU first; revisit GPU after the CPU rebuild is correct and profiled.
- L1/ADMM inversion until L2 lands (Phase 6b).

---

## Elevated questions (need your input)

Strategic decisions surfaced by the `../eo` review — answer before the affected phase:

1. **Packaging (before Phase 10).** Does dolphinRust ship as a workspace member of `eo` or
   as a separately versioned crate dependency?

## Open questions (technical, resolve before Phase 1)

1. ~~Pin the exact dolphin reference version/commit for oracle generation.~~ **Resolved:
   `v0.35.0` (`e567e55`).**
2. ~~Confirm `faer`'s complex Cholesky + shift-invert (or its direct dense eigensolver) hit
   the correctness tolerances~~ **Resolved (Phase 1):** Rust uses faer's direct
   `selfadjoint_eigendecomposition` for both EVD (largest) and EMI (least) eigenvectors —
   the sanctioned divergence from dolphin's JAX power/inverse iteration. EMI inverts
   `Γ=|C|` via faer real Cholesky (`Side::Lower`), falling back to EVD on Cholesky failure
   or a non-finite inverse (the dolphin NaN fallback). Validated against the v0.35.0 oracle:
   covariance max-err < 1e-4, eigenvector `|⟨v_rust,v_oracle⟩|` > 0.999. No LAPACK needed.
3. Decide the oracle-fixture generation env (containerized Python + pinned dolphin) so
   reference data is reproducible.
