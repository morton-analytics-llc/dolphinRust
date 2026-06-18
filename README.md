# dolphinRust

A ground-up Rust **rebuild** of the OPERA InSAR surface-displacement pipeline that produces
the DISP-S1 product — optimized for performance. The Python
[dolphin](https://github.com/isce-framework/dolphin) library is the algorithm reference
(the scientific spec), **not** a line-by-line port target.

The pipeline estimates surface displacement from Sentinel-1 CSLC stacks via persistent /
distributed scatterer phase linking (EVD/EMI), sequential ministack processing, phase
unwrapping, and SBAS network inversion. The rebuild targets the numerically hot paths —
covariance estimation, eigensolver-based phase linking, SHP selection — where Rust's
`rayon` + `faer` stack replaces the Python `jax`/`numba` JIT kernels without dispatch or
cold-start overhead, while delegating mature external solvers (SNAPHU) via subprocess.

## Status

**v1.0.0 — first complete build.** A single synchronous entry point,
[`dolphin_workflows::run_displacement`](crates/dolphin-workflows/src/displacement.rs), runs
the end-to-end pipeline (read CSLC → sequential phase-linking → interferogram network →
SNAPHU unwrap → SBAS inversion → velocity) and returns a typed result — displacement cube,
velocity (mm/yr), temporal coherence, acquisition dates, CRS + geotransform — while also
writing Cloud-Optimized GeoTIFFs. The `dolphin` CLI is a thin wrapper over it. The library
path is runtime-agnostic (no tokio), so a host app bridges it via `spawn_blocking`.

**Validated against Python dolphin v0.35.0** as a reference oracle, to physically-meaningful
tolerances (not bit-exactness):

- **Per-kernel contract tests** — every numerical kernel matches a dolphin `.npy` fixture
  (phase linking eigenvector overlap > 0.999, coherence < 1e-4, **L1/ADMM inversion < 1.5e-6**).
- **End-to-end, synthetic single-burst stack** — full `dolphin run` vs `dolphinRust` on a
  genuine `dolphin config` YAML: displacement corr 1.0000 / demeaned RMS ≤ 0.05 rad, and
  **velocity absolute scale matches** (affine slope a = 1.0000 noise-free → 0.9997 at
  realistic speckle). See [VALIDATION.md](VALIDATION.md).
- **Real OPERA CSLC stack** — both engines run the full pipeline on genuine OPERA L2 CSLC-S1
  granules (config compatibility on real data: PASS) across four bursts. Engine agreement is
  confirmed: displacement RMS residual ≤ 0.008 rad (within the sanctioned envelope), matching
  velocity magnitude, and matching temporal coherence. See [VALIDATION.md](VALIDATION.md).

**Honest scope / deferred:**
- **Real-data velocity *absolute scale* under strong signal** is not independently pinned: the
  coherent windows sampled were tectonically stable (deformation at the cross-engine noise
  floor), so the scale regression is signal-limited there. Absolute scale is confirmed on the
  synthetic tier (a = 1.0000); a high-coherence *deforming* scene is the narrow follow-up.
- **Multi-burst frame stitching** is implemented (frame mosaic by burst geotransform); not yet
  exercised on a real multi-burst OPERA frame.
- **tophu multi-scale unwrapping** is implemented and opt-in (`unwrap_method: tophu`), but on
  low-coherence scenes it does **not** beat the default SNAPHU path — see
  [bench/UNWRAP.md](bench/UNWRAP.md). SNAPHU remains the recommended default.
- `EagerLoader` prefetch, complex-GeoTIFF (CFloat32) writer, NISAR custom geotransform, and the
  spurt/whirlwind unwrappers are deferred.

See [STATUS.md](STATUS.md) and [PLAYBOOK.md](PLAYBOOK.md) for the full roadmap.

## System requirements

| Dependency | Version | Needed for |
|---|---|---|
| Rust | ≥ 1.94 | build |
| GDAL | ≥ 3.4 (tested 3.12) | GeoTIFF/COG I/O (`gdal` 0.19) |
| HDF5 | 1.10+ (tested 2.x) | CSLC reading (`hdf5-metno` 0.12) |
| SNAPHU | binary on `PATH` (tested 2.0.7) | phase unwrapping |
| GPU (optional) | any `wgpu` adapter — Metal / Vulkan / DX12 | GPU phase-linking backend (opt-in; CPU is the default) |

The numerical crates (`dolphin-core/-phaselink/-shp/-ps/-stack/-timeseries/-filtering`) build
with a pure-Rust dependency set; only the I/O, unwrap, and workflow layers need the system
libraries above. `cargo test` runs analytic contracts always; oracle/SNAPHU-dependent tests
skip cleanly when fixtures or the binary are absent.

### GPU backend (first-class, opt-in)

Phase linking has a `wgpu` GPU backend compiled into the **default build**, but it is **off
by default**: `worker_settings.compute_backend` defaults to `cpu`. Opt in with `gpu`, or
`auto` (GPU at/above the ~128² kernel crossover, CPU below). With **no GPU adapter** the run
falls back to the CPU automatically
with a warning — never a crash. The CPU (`faer`, f64) path is the correctness reference; the
GPU runs single precision (`f32`) and recomputes the small set of ill-conditioned EMI pixels
on the f64 CPU, so EMI matches the CPU reference **sub-mm on every pixel** (EVD too). On an
integrated Apple GPU the end-to-end win is marginal (often CPU-favoured) — the payoff is
portability to discrete NVIDIA/AMD, where the same shaders run unchanged. Build CPU-only (no
wgpu link) with `cargo build --no-default-features --features no-gpu`. See
[bench/GPU.md](bench/GPU.md).

## Quickstart — CLI

```sh
cargo build --release
# accepts a genuine dolphin `DisplacementWorkflow` YAML unchanged
./target/release/dolphin run --config workflow.yaml
# writes velocity.tif, temporal_coherence.tif, displacement_NN.tif (COGs) to work_directory
# plus crlb_sigma_NN.tif (CRLB σ uncertainty, on by default) and — when enabled —
# closure_phase_NN.tif
```

The **CRLB σ** layer (`crlb_sigma`) is the per-pixel, per-date physical uncertainty (radians)
of the phase-linking estimate, from the Fisher information of the coherence model — the input
GroundPulse's `confidence_score` needs to weight a velocity by how well-determined it is. The
**closure-phase** layer (`closure_phase`, off by default) is the nearest-neighbour triplet
non-closure diagnostic. Both are validated against a forward dolphin v0.42.0 oracle; see
[docs/usage.md](docs/usage.md) §5 and [VALIDATION.md](VALIDATION.md).

## Quickstart — library

Add the workflow crate (path or git) and call the one entry point:

```rust
use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::run_displacement;

fn main() -> anyhow::Result<()> {
    let cfg = DisplacementWorkflow::from_yaml(&std::fs::read_to_string("workflow.yaml")?)?;
    let out = run_displacement(&cfg)?; // synchronous; ~no tokio in the library path
    println!(
        "{} dates, velocity {}x{} px in mm/yr, EPSG {:?}",
        out.acquisition_days.len(),
        out.velocity_mm_yr.nrows(),
        out.velocity_mm_yr.ncols(),
        out.epsg,
    );
    Ok(())
}
```

A host app on a tokio runtime bridges the blocking call:

```rust
let out = tokio::task::spawn_blocking(move || dolphin_workflows::run_displacement(&cfg)).await??;
```

A runnable version is in [`crates/dolphin-workflows/examples/run_synthetic.rs`](crates/dolphin-workflows/examples/run_synthetic.rs)
(generates a synthetic stack and produces output in one command). Full integration guide,
config reference, and output schema: **[docs/usage.md](docs/usage.md)**.

## Workspace layout

| Crate | Reference (dolphin) | Responsibility |
|---|---|---|
| `dolphin-core` | cross-cutting | Types, block/tiling geometry, config models, errors |
| `dolphin-io` | `dolphin/io/` | HDF5 CSLC reading, geotransform/CRS, GeoTIFF/COG I/O |
| `dolphin-phaselink` | `dolphin/phase_link/` | Covariance, EVD/EMI, compression, temporal coherence |
| `dolphin-shp` | `dolphin/shp/` | GLRT / KS homogeneous-pixel selection |
| `dolphin-ps` | `dolphin/ps.py` | Amplitude-dispersion PS selection |
| `dolphin-stack` | `dolphin/stack.py` | Ministack planning, compressed-SLC sequencing |
| `dolphin-timeseries` | `dolphin/timeseries.py` | SBAS L1/L2 network inversion, velocity |
| `dolphin-filtering` | `dolphin/filtering.py` | Long-wavelength / Goldstein FFT filters |
| `dolphin-unwrap` | `dolphin/unwrap/` | Dispatch to external unwrappers (SNAPHU) |
| `dolphin-ingest` | — | Concurrent S3 read-staging (feature `s3`, off by default) |
| `dolphin-workflows` | `dolphin/workflows/` | Displacement pipeline orchestration + config |
| `dolphin-cli` | `dolphin` CLI | `dolphin run --config <yaml>` |

## License

MIT © Morton Analytics LLC. An independent Rust implementation; algorithms are referenced
from the upstream dolphin project (Apache-2.0, isce-framework / Caltech), no code is copied.
