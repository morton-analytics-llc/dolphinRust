# dolphin-io — block raster & HDF5 I/O (port of `dolphin/io/`)

## Domain
- **`VRTStack`**: N coregistered SLC files presented as one `N×rows×cols` virtual stack;
  auto-sorted by acquisition date; NumPy-like 3D block indexing.
- **CSLC reading:** OPERA S1 / NISAR CSLC live in HDF5. GDAL path form
  `HDF5:"file.h5"://science/SENTINEL1/CSLC/grids/VV`. NISAR needs a custom geotransform
  reader (GDAL's HDF5 driver returns identity).
- **`EagerLoader`**: background-thread prefetch of upcoming blocks — a dedicated thread
  plus a bounded channel, NOT async/tokio.
- Outputs: complex-f32 phase SLCs, f32 quality layers, uint8 PS mask, compressed SLCs
  (GeoTIFF via GDAL).

- **CSLC-S1-STATIC geometry (`geometry.rs`):** `read_los_layers` reads the per-burst
  static-layer companion product's `/data/los_east` + `/data/los_north` (f32 ground→sensor LOS
  unit-vector components) plus its `GeoInfo` (via `geo::read_geotransform`). Raw IO only — the
  reproject/mosaic/up-derivation lives in `dolphin-corrections::geometry`.

## Conventions
- GDAL/HDF5 are blocking C libraries (HDF5 is not thread-safe without the threadsafe
  build) — keep all access synchronous; parallelize across tiles, not within a reader.
  **HDF5-touching unit tests must serialize** through `test_hdf5_lock::guard()` (`lib.rs`,
  `#[cfg(test)]`): parallel `File::create`/`open` on the non-thread-safe build corrupt global
  library state and flake. Cross-crate test binaries need their own local lock (the guard is
  `pub(crate)`).
- System libs (`gdal`, `hdf5` crates) are added in Phase 8 only. Do not pull them earlier.
