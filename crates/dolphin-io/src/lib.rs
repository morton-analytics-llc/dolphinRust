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
pub mod nisar;
#[cfg(any(test, feature = "nisar-fixture"))]
pub mod nisar_fixture;

pub use cslc::{read_cslc, read_cslc_stack};
pub use error::{IoError, Result};
pub use geo::{read_geotransform, GeoInfo};
pub use geotiff::{grid_centroid_lonlat, read_raster, write_raster, RasterData};
pub use nisar::{read_nisar_geotransform, read_nisar_rslc, read_nisar_stack};
