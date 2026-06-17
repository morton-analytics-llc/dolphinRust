//! Block raster & HDF5 I/O — port of `dolphin/io/`.
//!
//! GeoTIFF block read/write via GDAL ([`geotiff`]) and OPERA/NISAR CSLC reading
//! from HDF5 ([`cslc`]). GDAL/HDF5 are blocking C libraries; access is kept
//! synchronous and parallelism happens across tiles, not within a reader.
//!
//! Bindings: `gdal` 0.19 (system GDAL 3.12) and `hdf5-metno` 0.12 (system HDF5
//! 2.x). The `EagerLoader` prefetch and complex-GeoTIFF writer are follow-ups
//! (see STATUS.md); S3 read-staging lives in the feature-gated `dolphin-ingest`.
#![warn(missing_docs)]

pub mod cslc;
pub mod error;
pub mod geo;
pub mod geotiff;

pub use cslc::{read_cslc, read_cslc_stack};
pub use error::{IoError, Result};
pub use geo::{read_geotransform, GeoInfo};
pub use geotiff::{read_raster, write_raster, RasterData};
