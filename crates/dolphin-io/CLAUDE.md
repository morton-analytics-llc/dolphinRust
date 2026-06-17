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

## Conventions
- GDAL/HDF5 are blocking C libraries (HDF5 is not thread-safe without the threadsafe
  build) — keep all access synchronous; parallelize across tiles, not within a reader.
- System libs (`gdal`, `hdf5` crates) are added in Phase 8 only. Do not pull them earlier.
