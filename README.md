# dolphinRust

A Rust reimplementation of [dolphin](https://github.com/isce-framework/dolphin) — the
OPERA InSAR surface-displacement processing library that produces the DISP-S1 product.

dolphin estimates surface displacement from Sentinel-1 CSLC stacks via persistent /
distributed scatterer phase linking (EVD/EMI), sequential ministack processing, phase
unwrapping, and SBAS network inversion. This port targets the numerically hot paths —
covariance estimation, eigensolver-based phase linking, SHP selection — where Rust's
`rayon` + `faer` stack replaces the Python `jax`/`numba` JIT kernels without dispatch
or cold-start overhead, while delegating mature external solvers (SNAPHU) via subprocess.

## Status

Scaffold. No pipeline stages implemented yet. See [PLAYBOOK.md](PLAYBOOK.md) for the
phased implementation plan and the parity strategy against the Python reference.

## Workspace layout

| Crate | Mirrors (Python) | Responsibility |
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

The scaffold builds with a pure-Rust dependency set. System-library bindings
(GDAL, HDF5, LAPACK) are introduced in Phase 8 — see the playbook for setup.

## License

Apache-2.0. Algorithmic parity with the upstream dolphin project
(also Apache-2.0, isce-framework / Caltech).
