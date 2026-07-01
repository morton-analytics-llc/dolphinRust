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
pub mod geometry;
pub mod geotiff;
pub mod nisar;
#[cfg(any(test, feature = "nisar-fixture"))]
pub mod nisar_fixture;

pub use cslc::{read_cslc, read_cslc_shape, read_cslc_stack, read_cslc_window};
pub use error::{IoError, Result};
pub use geo::{read_geotransform, GeoInfo};
pub use geometry::{read_los_layers, LosLayers};
pub use geotiff::{
    grid_centroid_lonlat, read_raster, read_raster_window, write_raster, RasterData,
};
pub use nisar::{read_nisar_geotransform, read_nisar_rslc, read_nisar_stack, read_nisar_window};

#[cfg(test)]
pub(crate) mod test_hdf5_lock {
    use std::sync::{Mutex, MutexGuard, PoisonError};

    static LOCK: Mutex<()> = Mutex::new(());

    /// Serialize HDF5 access across parallel unit tests. `hdf5-metno` links a
    /// non-thread-safe HDF5, so concurrent `File::create`/`open` in different test
    /// threads corrupts global library state (flaky, data-dependent failures). Every
    /// HDF5-touching unit test takes this guard first; a panic while held only
    /// poisons the mutex, which we recover from so the next test still runs.
    pub(crate) fn guard() -> MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
