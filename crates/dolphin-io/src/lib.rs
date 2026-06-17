//! Block-based raster/HDF5 I/O — port of `dolphin/io/`.
//!
//! Targets: the `VRTStack` SLC stack abstraction, `EagerLoader` background
//! prefetch, HDF5 CSLC subdataset reading (`HDF5:"f.h5"://path/VV`), and
//! GeoTIFF block writing. GDAL/HDF5 bindings are wired in Phase 8.
