# dolphinRust — usage & integration guide

How to install, configure, run, and consume dolphinRust from another Rust application
(e.g. GroundPulse). For the project overview and validation status see
[../README.md](../README.md) and [../VALIDATION.md](../VALIDATION.md).

## 1. Install & system dependencies

dolphinRust is a Cargo workspace. The numerical crates are pure Rust; the I/O, unwrap, and
workflow layers need three system libraries:

| Dependency | Min | Tested | Provides |
|---|---|---|---|
| Rust toolchain | 1.94 | 1.94.1 | build |
| GDAL | 3.4 | 3.12.2 | GeoTIFF/COG read+write (`gdal` 0.19) |
| HDF5 | 1.10 | 2.1.1 | CSLC reading (`hdf5-metno` 0.12) |
| SNAPHU | — | 2.0.7 | phase unwrapping (subprocess on `PATH`) |

macOS (Homebrew): `brew install gdal hdf5 snaphu`. SNAPHU may need building from the
[Stanford source](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) if no
package is available; put the resulting `snaphu` binary on `PATH`.

```sh
cargo build --release      # builds the workspace incl. the `dolphin` CLI
cargo test                 # analytic contracts always run; oracle/SNAPHU tests skip if absent
```

## 2. Input requirements

The pipeline consumes a **coregistered stack of OPERA Sentinel-1 CSLC granules** (one
acquisition per HDF5 file), all on the same grid:

- **Format:** HDF5, complex-`float32` grid at a subdataset path (OPERA: `/data/VV`).
- **Naming:** the acquisition date must be embedded in each filename and parseable by
  `input_options.cslc_date_fmt` (default `%Y%m%d`). The first matching date substring is
  used, so OPERA granule names (`..._20221119T232411Z_...`) and short names
  (`cslc_20221119.h5`) both work. **Real temporal baselines are derived from these dates** —
  they drive the velocity rate, so correct filenames matter.
- **Georeferencing:** for projected output, the CSLC group should carry `x_coordinates`,
  `y_coordinates`, and a `projection` (EPSG) dataset (OPERA layout). When absent, output
  falls back to an identity geotransform and `output_options.epsg`.
- **Ordering:** list files in acquisition order in `cslc_file_list`.

Single-burst only in v1.0.0 — multi-burst frame mosaics are not yet stitched.

## 3. Configuration

dolphinRust deserializes a genuine dolphin `DisplacementWorkflow` YAML unchanged (generate
one with `dolphin config ...`); unknown solver blocks (tophu/spurt/whirlwind) are ignored.
Key parameters (defaults match dolphin):

```yaml
cslc_file_list:
  - /data/cslc_20221119.h5
  - /data/cslc_20221201.h5     # 12-day cadence drives the mm/yr velocity
input_options:
  subdataset: /data/VV         # HDF5 path to the complex grid (required)
  cslc_date_fmt: "%Y%m%d"      # date parser for the filenames
  wavelength: 0.05546576       # radar wavelength (m). Set it to get meters / mm-yr output;
                               # omit and rasters stay in radians (velocity_mm_yr still uses
                               # the Sentinel-1 default)
phase_linking:
  ministack_size: 15           # SLCs per ministack (sequential estimator)
  half_window: { y: 7, x: 14 } # covariance/SHP window half-extent
  use_evd: false               # false = EMI (default), true = EVD
interferogram_network:
  reference_idx: 0             # single-reference network; or set max_bandwidth / max_temporal_baseline
timeseries_options:
  method: L1                   # L1 (dolphin default, ADMM/LAD) or L2 (weighted least squares)
unwrap_options:
  snaphu_options: { cost: smooth, init_method: mcf, ntiles: [1, 1] }
output_options:
  strides: { y: 1, x: 1 }      # output multilooking
  epsg: 32611                  # fallback CRS when the CSLC carries none
worker_settings:
  compute_backend: cpu         # default; phase-linking backend: cpu | auto | gpu (see below)
work_directory: /out           # outputs are written here
```

The complete config tree, with every field documented, is the rustdoc for
`dolphin_core::config::DisplacementWorkflow` (`cargo doc --no-deps -p dolphin-core --open`).

### Compute backend (CPU / GPU)

Phase linking (covariance + EVD/EMI) has a first-class `wgpu` GPU backend compiled into the
default build, but it is **off by default**. `worker_settings.compute_backend` selects it:

- **`cpu`** (default) — always the `faer` f64 reference path.
- **`auto`** — GPU at/above the ~128² output-pixel kernel crossover, CPU below it.
- **`gpu`** — GPU wherever supported.

**Fallback is automatic and safe.** With no GPU adapter, an `nslc` above the kernel cap (32),
or a CPU-only (`no-gpu`) build, any mode runs on the CPU with a `tracing` warning — never a
panic. The CPU path is the correctness reference.

**Accuracy (f32 vs f64).** The GPU runs single precision. EVD matches the f64 CPU sub-mm on
every pixel; EMI's f32 least-eigenvector is ill-conditioned on near-degenerate pixels, so the
GPU flags those and the host recomputes them on f64 `faer` — EMI then matches the CPU
reference **sub-mm on every pixel** (real 384² stack: ≤ 0.61 mm). One config gives the same
result on CPU or GPU.

**Platform & speed.** Any `wgpu` adapter works (Metal on macOS, Vulkan/DX12 elsewhere). On an
*integrated* Apple GPU the end-to-end speed is marginal and often CPU-favoured; the payoff is
discrete NVIDIA/AMD, where the same shaders run unchanged. Honest numbers: `bench/GPU.md`. To
build without linking wgpu: `cargo build --no-default-features --features no-gpu`.

## 4. Running

### CLI

```sh
./target/release/dolphin run --config workflow.yaml
# logs honor RUST_LOG (e.g. RUST_LOG=info)
```

### Library (synchronous)

```rust
use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::{run_displacement, DisplacementOutput};

let cfg = DisplacementWorkflow::from_yaml(&std::fs::read_to_string("workflow.yaml")?)?;
let out: DisplacementOutput = run_displacement(&cfg)?;
```

### From an async host (the GroundPulse pattern)

`run_displacement` is synchronous and CPU/IO-bound (GDAL, HDF5, SNAPHU subprocess). Bridge
it onto a blocking thread so it never stalls the tokio reactor:

```rust
use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::{run_displacement, DisplacementOutput};

async fn run_displacement_job(cfg: DisplacementWorkflow) -> anyhow::Result<DisplacementOutput> {
    // cfg is moved into the blocking pool; the library itself spawns no runtime.
    let out = tokio::task::spawn_blocking(move || run_displacement(&cfg)).await??;
    Ok(out)
}
```

`DisplacementOutput` owns its arrays (`ndarray`), so it crosses the `spawn_blocking`
boundary freely. eo can persist `velocity_mm_yr` for risk scoring and serve the COGs directly.

## 5. Output schema

`run_displacement` returns everything in memory **and** writes COGs to `work_directory`.

### In-memory (`DisplacementOutput`)

| Field | Type | Units | Notes |
|---|---|---|---|
| `displacement` | `Array3<f64>` `(n_dates-1, rows, cols)` | meters (if `wavelength`) else radians | cumulative LOS vs acquisition 0 |
| `velocity` | `Array2<f64>` `(rows, cols)` | m/yr (if `wavelength`) else rad/yr | raster-unit linear rate |
| `velocity_mm_yr` | `Array2<f64>` `(rows, cols)` | **mm/yr** | LOS rate via `−λ/4π`; config λ or Sentinel-1 default |
| `temporal_coherence` | `Array2<f64>` `(rows, cols)` | `[0, 1]` | ministack-averaged phase quality (unmasked) |
| `acquisition_days` | `Vec<f64>` length `n_dates` | days | decimal days from acquisition 0 |
| `epsg` | `Option<u32>` | — | output CRS (CSLC metadata, else config) |
| `geotransform` | `[f64; 6]` | — | GDAL `[origin_x, dx, 0, origin_y, 0, dy]` |

### On-disk rasters (`work_directory`)

All are single-band **Cloud-Optimized GeoTIFFs** (`float32`, internally tiled 256×256,
DEFLATE-compressed, overviews) sharing `epsg` + `geotransform`:

| File | Band | Units |
|---|---|---|
| `velocity.tif` | linear velocity | raster units/yr (m/yr or rad/yr) |
| `temporal_coherence.tif` | temporal coherence | `[0, 1]` |
| `displacement_NN.tif` | cumulative displacement at date `NN+1` | meters or radians |

No-data is unset (the typed layers are filled, not masked); threshold on
`temporal_coherence` to mask low-quality pixels downstream.

## 6. Known limitations (v1.0.0)

- **Single-burst only** — multi-burst stitching pending.
- **Real-OPERA validation pending** — equivalence to dolphin is established on synthetic
  inputs (see VALIDATION.md); the real-data tier is the open gate.
- CRLB/closure-phase rasters, complex-GeoTIFF (CFloat32) output, NISAR geotransform,
  `EagerLoader` prefetch, and non-SNAPHU unwrappers are deferred.

## 7. Running validation

```sh
# rebuild the pinned dolphin v0.35.0 oracle venv (recipe in oracle/ + auto-memory)
oracle/.venv/bin/python validation/gen_stack.py --outdir /tmp/d --speckle 0.05
validation/run.sh 0.0     # noise-free pure-algorithm agreement
validation/run.sh 0.05    # realistic speckle
```

`validation/run.sh <speckle>` synthesizes a CSLC stack, generates one dolphin config,
runs both engines, and diffs displacement + velocity (incl. absolute mm/yr scale). The
per-kernel oracle contracts run under `cargo test`.
