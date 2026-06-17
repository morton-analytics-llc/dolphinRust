# dolphin-core — shared types & geometry

Cross-cutting primitives every crate depends on. No numerics, no I/O.

## Domain
- `Cf32` / `Cf64`: complex SLC samples. SLCs are `complex64` (f32) on disk; numerical
  kernels accumulate in `Cf64` to match NumPy/JAX precision, then cast back at the boundary.
- **Look geometry:** `HalfWindow { y, x }` (default y=7, x=14) defines the covariance/SHP
  window `(2y+1)×(2x+1)`. `Strides { y, x }` decimate the output grid (multilooking).
- **`StridedBlockManager`** (port of `io/_blocks.py`): tiles the raster for block
  processing. Each tile yields five `BlockIndices` — output block, output trim, input
  block, input-without-halo, input trim. The halo equals `half_window`; "trim" strips it
  before writing. Reused everywhere — property-test it: every pixel covered exactly once,
  strides honored.
- **Config** mirrors dolphin's pydantic `DisplacementWorkflow` tree with identical YAML
  field names and defaults, so existing dolphin configs deserialize unchanged.

## Conventions
- Keep this crate dependency-light (no rayon/faer/gdal). Types only.
