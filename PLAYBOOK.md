# dolphinRust Implementation Playbook

Phased plan to port [dolphin](https://github.com/isce-framework/dolphin) to Rust. The
goal is **algorithmic parity** with the Python reference on the DISP-S1 wrapped-phase →
displacement pipeline, validated numerically against dolphin outputs at each phase.

Reference commit/version: pin a specific dolphin release (e.g. `v0.x`) before starting and
record it in [PARITY.md](#parity-strategy). All parity claims are against that pinned ref.

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
   precision, then casts back. This matters for parity tolerances.
5. **System-lib deps deferred to Phase 8.** GDAL/HDF5/LAPACK bindings are introduced
   only when the I/O layer lands, so the numerical core builds and tests on any machine
   with synthetic in-memory arrays.

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

## Parity strategy

Each numerical phase ships with a **golden-data parity test** before it is called done:

1. In a scratch Python env, install the pinned dolphin and emit reference outputs for a
   small synthetic stack (script lives in `parity/gen_<module>.py`, not committed data).
   Use a fixed seed; dump inputs + outputs to `.npy`.
2. Load the `.npy` in a Rust integration test (`ndarray-npy` dev-dependency), run the
   Rust kernel, assert closeness.
3. **Tolerances:** phase quantities compared modulo `2π` and up to a global phase
   reference; `temp_coh`/coherence to `atol=1e-4`; eigenvector phase to `1e-3` rad after
   referencing. Document any kernel that cannot meet these and why (e.g. eigenvector sign
   ambiguity → compare `|⟨v_rust, v_py⟩|`).
4. A kernel is "ported" only when its parity test is green against the pinned ref. Code
   existence is not done.

Cross-cutting test data: one synthetic DS region (known coherence matrix → simulated
SLC samples) + one PS-like region (high-amplitude stable point). Build these as Rust
fixtures so unit tests don't depend on Python.

---

## Phase 0 — Foundation (`dolphin-core`)

**Scope.** No numerics; everything downstream depends on these types.

- `types`: `Cf32`/`Cf64` (done), `HalfWindow { y, x }`, `Strides { y, x }`,
  acquisition date wrappers.
- `blocks`: port `StridedBlockManager` / `BlockIndices` from `io/_blocks.py` — the
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
2. **Eigensolvers.** `power_iteration(A, tol=1e-5, max_iters=50)` (dominant pair) and
   `inverse_iteration(A, mu=0.99, ...)` (shift-invert via LU, smallest pair). Match
   dolphin's iteration counts/tolerances for parity.
3. **EVD estimator.** Largest eigenvector of `C ⊙ |C|`.
4. **EMI estimator (default).** `Γ = |C|`; regularize `Γ ← (1-β)Γ + βI`; threshold
   near-zero entries (`zero_correlation_threshold`); Cholesky-invert with `1e-6` jitter;
   smallest eigenvector of `Γ⁻¹ ⊙ C`. **Fallback to EVD on singular `Γ⁻¹` (NaN)** — match
   dolphin's `lax.select` behavior exactly.
5. **Phase referencing.** `θ ← θ · exp(-j·∠θ[ref_idx])`.

**Done when:** EVD and EMI eigenvector parity tests pass on the synthetic DS fixture
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

**Done when:** GLRT and KS neighbor arrays match dolphin bit-for-bit (boolean) on the
fixture; wire into Phase 1 covariance and re-run Phase 1 parity with SHP weighting on.

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

**Done when:** temp_coh, CRLB, closure, and compressed-SLC parity tests pass.

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
- **L1/ADMM deferred** to Phase 6b — port only after L2 parity, since dolphin defaults to
  L1; document the temporary divergence in PARITY.md.

**Done when:** L2 displacement series + velocity match dolphin (L2 mode) on synthetic
unwrapped ifgs; network construction matches for each network mode.

---

## Phase 7 — Filters (`dolphin-filtering`)

**Scope.** `filtering.py`, `goldstein.py`. Long-wavelength FFT Gaussian high-pass and
Goldstein adaptive filter via `rustfft`. Optional pre-unwrap stages.

**Done when:** filtered rasters match dolphin to `atol=1e-4`.

---

## Phase 8 — I/O layer (`dolphin-io`) — introduces system libs

**Scope.** `io/_readers.py`, writers. **Run the environment preflight first.**

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
displacement time series matching dolphin within parity tolerances; CLI config matches
dolphin's YAML.

---

## Port priority (critical path)

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
(Phase 0) exist. Do **not** start Phase 10 until 1–6, 8, 9 each carry green parity tests.

---

## Out of scope (initial)

- `atmosphere/` tropospheric corrections (wraps external delay models) — defer.
- DISP-S1 HDF5 product schema — owned by `disp-s1` SAS, not dolphin; port only if needed.
- GPU paths (JAX GPU, CUDA KS) — `rayon` CPU first; revisit GPU after parity.
- L1/ADMM inversion until L2 lands (Phase 6b).

---

## Open questions to resolve before Phase 1

1. Pin the exact dolphin reference version/commit for all parity tests.
2. Confirm `faer`'s complex Cholesky + shift-invert match JAX's numerics within
   tolerance, or whether a thin LAPACK path (`ndarray-linalg`) is needed for the EMI
   inverse.
3. Decide the parity-fixture generation env (containerized Python + pinned dolphin) so
   golden data is reproducible.
