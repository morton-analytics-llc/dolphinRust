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

### NISAR / L-band input (`input_type: nisar_gslc`)

dolphinRust also reads **NISAR L-band geocoded SLC (GSLC)** stacks. NISAR penetrates
canopy where C-band fails (forested/vegetated terrain). Differences from the OPERA path,
all handled by the NISAR reader:

- **Complex samples are a complex-`float32` `{r, i}` compound** — the same h5py-style layout
  as OPERA (verified against a real `NISAR_L2_GSLC_BETA_V1` granule; the "complex-int16" found
  in some NISAR notes does **not** apply to GSLC). So NISAR differs from OPERA only in the
  grid metadata below.
- **Grid lives in the NISAR product group**, e.g.
  `/science/LSAR/GSLC/grids/frequencyA/HH` (set this as `subdataset`), with camelCase
  `xCoordinates`/`yCoordinates` arrays and the EPSG carried as the `epsg_code` **attribute**
  of the `projection` dataset. GDAL returns an identity geotransform for this layout, so it
  is read directly.
- **Wavelength is L-band** ≈ `0.238403545` m (vs S1 C-band `0.05546576`); set
  `input_options.wavelength` so velocity comes out in mm/yr.
- **Granule names** (`NISAR_L2_PR_GSLC_..._20240601T120000_...h5`) parse with the default
  `cslc_date_fmt: "%Y%m%d"`.

Select it with `input_options.input_type: nisar_gslc` (default `opera_cslc`). This is a
forward divergence from dolphin v0.35.0, which has no product-type field.

> **Atmospheric correction is NOT applied (v1.3.0).** This is a geometrically-correct but
> **atmospherically-uncorrected** L-band product. Ionospheric delay is ~16× the C-band
> effect and is mandatory for a *usable* L-band displacement product; ionospheric +
> tropospheric corrections land in a later v1.3.0 loop. Treat NISAR displacement as
> provisional until then.

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
  write_crlb: true             # per-date CRLB σ uncertainty layer (default on)
  write_closure_phase: false   # per-triplet closure-phase layer (default off)
interferogram_network:
  reference_idx: 0             # single-reference network; or set max_bandwidth / max_temporal_baseline
timeseries_options:
  method: L1                   # L1 (dolphin default, ADMM/LAD) or L2 (weighted least squares)
unwrap_options:
  unwrap_method: snaphu        # default; or `tophu` for multi-scale (see below)
  snaphu_options: { cost: smooth, init_method: mcf, ntiles: [1, 1] }
  tophu_options: { ntiles: [4, 4], downsample_factor: [3, 3], init_method: mcf, cost: smooth }
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
| `temporal_coherence` | `Array2<f64>` `(rows, cols)` | `[0, 1]` | per-ministack-stitched phase quality (dolphin's NaN-aware mean across ministacks; unmasked) |
| `crlb_sigma` | `Option<Array3<f64>>` `(n_dates, rows, cols)` | radians | per-date Cramér–Rao σ lower bound; band 0 = reference (σ=0), singular-Γ pixels `NaN`. `Some` by default (`write_crlb`) |
| `closure_phase` | `Option<Array3<f64>>` `(n_dates-2, rows, cols)` | radians | per-triplet nearest-neighbour non-closure; `Some` only when `write_closure_phase` |
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
| `crlb_sigma_NN.tif` | CRLB σ at date `NN` (band 0 = reference) | radians |
| `closure_phase_NN.tif` | nearest-neighbour closure of triplet `NN` (only if `write_closure_phase`) | radians |

No-data is unset (the typed layers are filled, not masked); threshold on
`temporal_coherence` to mask low-quality pixels downstream.

**CRLB → GroundPulse `confidence_score`.** `crlb_sigma` is the per-pixel, per-date
*physical* uncertainty (radians) of the phase-linking estimate — the Cramér–Rao lower bound
from the Fisher information of the coherence model. It is the missing input to GroundPulse's
asset-risk `confidence_score`: a velocity is only as trustworthy as the σ of the phases it
was fit from. A pixel with low CRLB σ (and high temporal coherence) carries a high-confidence
velocity; a high-σ or `NaN` (singular-Γ, fully decorrelated) pixel should be down-weighted or
masked. To reduce the per-date layer to one scalar per pixel for scoring, take a
baseline-appropriate summary (e.g. the last date's σ, or the RMS across dates); the choice is
the consumer's, so dolphinRust surfaces the full per-date bound rather than pre-collapsing it.

## 5b. Phase unwrapping: SNAPHU (default) vs tophu multi-scale

`unwrap_options.unwrap_method` selects the unwrapper. Both drive the SNAPHU binary
(`snaphu` on `PATH`); only the strategy differs.

- **`snaphu`** (default) — one SNAPHU solve over the whole interferogram, with SNAPHU's
  own internal tiling controlled by `snaphu_options` (`ntiles`, `tile_overlap`,
  `n_parallel_tiles`, `cost`, `init_method`). This is the recommended path.
- **`tophu`** — OPERA's multi-scale strategy: a **coherence-weighted** coarse multilook
  (`tophu_options.downsample_factor`) is unwrapped once and used as the absolute-phase
  reference (low-trust blocks masked + filled from trusted neighbours), the full-res grid is
  unwrapped in `tophu_options.ntiles` overlapping tiles in parallel, and the tiles are merged
  by estimating each adjacent pair's integer-cycle offset from their *coherent overlap* and
  solving a maximum-reliability spanning forest for globally consistent per-tile cycles, then
  feather-blending the tiles across their overlap halos.

**When to use tophu:** large, partly-decorrelated scenes (vegetated / fast-subsidence
centres). On the low-coherence scenes we benchmark, tophu now **beats** raw SNAPHU on all
three metrics on both scenes — discontinuities ~9 % lower on both and gross-cycle errors up
to ~10 % lower — by keeping the tiled solves inter-tile consistent and the seams continuous.
See [`bench/UNWRAP.md`](../bench/UNWRAP.md) for the measured numbers and reproduction. For
small or mostly-coherent scenes the default **SNAPHU** path (one global MCF solve) is simpler
and sufficient.

```yaml
unwrap_options:
  unwrap_method: tophu
  tophu_options: { ntiles: [4, 4], downsample_factor: [3, 3], init_method: mcf, cost: smooth }
```

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
