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

pub mod covariance;
pub mod cslc;
pub mod cslc_metadata;
pub mod error;
pub mod geo;
pub mod geometry;
pub mod geotiff;
pub mod nisar;
#[cfg(any(test, feature = "nisar-fixture"))]
pub mod nisar_fixture;

pub use covariance::{
    covariance_content_bound_source_id, covariance_identified_id,
    covariance_identity_index_peak_bytes, covariance_record_block_id,
    covariance_source_model_identity_digest, read_covariance_operator,
    read_covariance_operator_block, read_covariance_operator_block_with_receipt,
    read_covariance_operator_header_with_byte_cap, read_covariance_operator_metadata,
    read_covariance_operator_metadata_with_byte_cap, read_covariance_operator_with_byte_cap,
    read_spatial_reference_covariance_block, read_spatial_reference_covariance_header,
    recover_incomplete_covariance_operator, spatial_reference_calibration_scope_digest,
    write_spatial_reference_covariance, CovarianceBurstPlan, CovarianceCalibrationStatus,
    CovarianceEstimatorBranch, CovarianceOperatorArtifact, CovarianceOperatorBlock,
    CovarianceOperatorBlockRead, CovarianceOperatorBlockReader, CovarianceOperatorGrid,
    CovarianceOperatorMetadata, CovarianceOperatorPlan, CovarianceOperatorStatus,
    CovarianceOperatorWriteReceipt, CovarianceOperatorWriter, CovariancePhaseComponent,
    CovariancePhaseComponentKind, CovarianceRectSupport, CovarianceRegistryEntry,
    CovarianceReplayStatus, CovarianceSupportOrdering, CovarianceTilePlan,
    DownstreamInferenceStatus, SourceReplayIdentity, SpatialReferenceCalibrationScope,
    SpatialReferenceCovarianceBlock, SpatialReferenceCovarianceBlockRead,
    SpatialReferenceCovarianceMetadata, SpatialReferenceCovarianceStatus,
    SpatialReferenceCovarianceWriteReceipt, SpatialReferenceCovarianceWriter,
    StitchedCovarianceStatus, COVARIANCE_CALIBRATION_STATUS_REGISTRY,
    COVARIANCE_ESTIMATOR_BRANCH_REGISTRY, COVARIANCE_METHOD_REGISTRY, COVARIANCE_OPERATOR_METHOD,
    COVARIANCE_OPERATOR_METHOD_VERSION, COVARIANCE_OPERATOR_SCHEMA_VERSION,
    COVARIANCE_OPERATOR_STATUS_REGISTRY, COVARIANCE_PHASE_COMPONENT_KIND_REGISTRY,
    COVARIANCE_REPLAY_STATUS_REGISTRY, COVARIANCE_SUPPORT_ORDERING_REGISTRY,
    DOWNSTREAM_INFERENCE_STATUS_REGISTRY, SPATIAL_REFERENCE_COVARIANCE_METHOD,
    SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION, STITCHED_COVARIANCE_STATUS_REGISTRY,
};
pub use cslc::{read_cslc, read_cslc_shape, read_cslc_stack, read_cslc_window};
pub use cslc_metadata::{
    read_cslc_burst_metadata, read_cslc_identification, read_cslc_orbit, read_cslc_orbit_type,
    CslcBurstMetadata, CslcIdentification, CslcOrbit,
};
pub use error::{IoError, Result};
pub use geo::{read_geotransform, transform_bounds, GeoInfo};
pub use geometry::{read_los_layers, LosLayers};
pub use geotiff::{
    grid_centroid_lonlat, grid_corner_lonlat, read_aligned_raster_window, read_raster,
    read_raster_window, write_raster, write_raster_with_metadata, RasterData,
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
