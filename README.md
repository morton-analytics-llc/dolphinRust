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
Correctness is validated against analytic fixtures and dolphin as a reference oracle.

## Status

**v1.0.0 — first complete build.** `dolphin run --config <yaml>` produces an end-to-end
displacement time series + velocity from a CSLC stack (read → sequential phase-linking →
interferogram network → SNAPHU unwrap → SBAS L2 inversion → velocity → GeoTIFF outputs),
validated against Python dolphin **v0.35.0** as a reference oracle within physically
meaningful tolerances. All numerical phases carry green analytic + oracle contract tests.

Known deferrals (off the v1.0.0 critical path): CRLB / closure-phase rasters and L1/ADMM
inversion (not in the pinned v0.35.0 / deferred to 6b), `EagerLoader` prefetch,
complex-GeoTIFF (CFloat32) writer, NISAR custom geotransform, multi-burst stitching, and
the tophu/spurt/whirlwind unwrappers. See [STATUS.md](STATUS.md) and
[PLAYBOOK.md](PLAYBOOK.md).

## Workspace layout

| Crate | Reference (dolphin) | Responsibility |
|---|---|---|
| `dolphin-core` | cross-cutting | Types, block/tiling geometry, config models, errors |
| `dolphin-io` | `dolphin/io/` | VRT stack, HDF5 CSLC reading, GeoTIFF block I/O |
| `dolphin-phaselink` | `dolphin/phase_link/` | Covariance, EVD/EMI, compression, CRLB, metrics |
| `dolphin-shp` | `dolphin/shp/` | GLRT / KS homogeneous-pixel selection |
| `dolphin-ps` | `dolphin/ps.py` | Amplitude-dispersion PS selection |
| `dolphin-stack` | `dolphin/stack.py` | Ministack planning, compressed-SLC sequencing |
| `dolphin-timeseries` | `dolphin/timeseries.py` | SBAS network inversion, velocity |
| `dolphin-filtering` | `dolphin/filtering.py` | Long-wavelength / Goldstein FFT filters |
| `dolphin-unwrap` | `dolphin/unwrap/` | Dispatch to external unwrappers (SNAPHU) |
| `dolphin-ingest` | — | Concurrent S3 read-staging (feature `s3`, off by default) |
| `dolphin-workflows` | `dolphin/workflows/` | Displacement pipeline orchestration + config |
| `dolphin-cli` | `dolphin` CLI | `dolphin run --config <yaml>` |

## Build

```sh
cargo build
cargo test
```

The numerical crates build with a pure-Rust dependency set. The I/O and unwrap layers
need system libraries: **GDAL ≥ 3.4** (`gdal` 0.19), **HDF5** (`hdf5-metno` 0.12), and the
**SNAPHU** binary on `PATH` for unwrapping. `cargo test` runs analytic contracts always;
oracle/SNAPHU-dependent tests skip cleanly when fixtures or the binary are absent.

Run the pipeline:

```sh
dolphin run --config workflow.yaml   # accepts dolphin's displacement-workflow YAML
```

## License

MIT © Morton Analytics LLC. An independent Rust implementation; algorithms are referenced
from the upstream dolphin project (Apache-2.0, isce-framework / Caltech), no code is copied.
