//! Fixed-cube output contract for downstream scientific validation.

use anyhow::{ensure, Result};
use dolphin_core::config::DisplacementWorkflow;
use dolphin_io::write_raster_with_metadata;
use serde::Serialize;
use sha2::{Digest, Sha256};

use dolphin_corrections::LosGeometry;
use ndarray::ArrayView2;

/// Stable receipt for the fixed-cube rasters consumed by EO.
#[derive(Debug, Serialize)]
pub struct FixedCubeReceipt {
    /// Contract schema identifier.
    pub contract_version: &'static str,
    /// Exact decimal-day axis used by the run.
    pub acquisition_days: Vec<f64>,
    /// SHA-256 over the little-endian epoch bytes.
    pub acquisition_days_sha256: String,
    /// Stable estimator identity copied from `velocity.tif`.
    pub velocity_estimator: String,
    /// Unit recorded by the velocity raster metadata.
    pub velocity_unit: &'static str,
    /// Inferential serving status; corrected temporal covariance is not present yet.
    pub inference_status: &'static str,
    /// Corrected velocity raster, absent until issue #53 promotion gates pass.
    pub corrected_velocity_raster: Option<&'static str>,
    /// Corrected sigma raster, absent until issue #53 promotion gates pass.
    pub corrected_sigma_raster: Option<&'static str>,
    /// Fixed-cube validity mask filename.
    pub validity_mask_raster: &'static str,
    /// Velocity raster filename.
    pub velocity_raster: &'static str,
    /// Optional conditional sigma raster filename.
    pub velocity_sigma_raster: Option<&'static str>,
    /// East, north, and up signed LOS raster filenames.
    pub los_rasters: [&'static str; 3],
    /// Geometry provenance filename.
    pub geometry_provenance: &'static str,
    /// Source identity for the signed LOS vectors.
    pub geometry_source: &'static str,
    /// Spatial reference pixel selected for the run.
    pub reference_point: Option<(usize, usize)>,
    /// Output EPSG code.
    pub epsg: Option<u32>,
    /// Output GDAL geotransform.
    pub geotransform: [f64; 6],
    /// Raster row count.
    pub rows: usize,
    /// Raster column count.
    pub cols: usize,
    /// Count of valid mask pixels.
    pub valid_pixels: usize,
}

/// Emit the fixed-cube mask, sourced signed LOS vectors, and receipt.
///
/// This is deliberately separate from the legacy `Full` writer: callers that
/// require the science contract can use this function as a hard gate, while
/// compatibility runs may continue without CSLC-S1-STATIC geometry.
#[allow(clippy::too_many_arguments)]
pub fn write_fixed_cube_bundle(
    cfg: &DisplacementWorkflow,
    acquisition_days: &[f64],
    velocity_estimator: crate::displacement::VelocityEstimator,
    velocity_sigma_present: bool,
    validity_mask: ArrayView2<'_, bool>,
    geometry: &LosGeometry,
    reference_point: Option<(usize, usize)>,
    epsg: Option<u32>,
    geotransform: [f64; 6],
) -> Result<FixedCubeReceipt> {
    validate_geometry(geometry, validity_mask.dim())?;

    let dir = &cfg.work_directory;
    std::fs::create_dir_all(dir)?;
    let mask = validity_mask.mapv(|valid| if valid { 1u8 } else { 0u8 });
    write_raster_with_metadata(
        &dir.join("velocity_validity_mask.tif"),
        mask.view(),
        geotransform,
        epsg,
        Some(0.0),
        &[
            ("MASK_ROLE", "velocity_support"),
            ("MASK_VALUES", "0=invalid;1=valid"),
            ("MASK_POLICY", "common_epoch_complete_support"),
        ],
    )?;
    let geometry_tags = [
        ("GEOMETRY_SOURCE", "CSLC-S1-STATIC"),
        (
            "LOS_SIGN_CONVENTION",
            "ground_to_sensor_positive_toward_sensor",
        ),
        ("LOS_COMPONENTS", "east,north,up"),
        ("RASTER_ROLE", "fixed_cube_run_geometry"),
    ];
    for (name, component) in [
        ("los_east.tif", &geometry.east),
        ("los_north.tif", &geometry.north),
        ("los_up.tif", &geometry.up),
    ] {
        write_raster_with_metadata(
            &dir.join(name),
            component.mapv(|value| value as f32).view(),
            geotransform,
            epsg,
            None,
            &geometry_tags,
        )?;
    }
    let days_sha256 = sha256_days(acquisition_days);
    let (rows, cols) = validity_mask.dim();
    let receipt = FixedCubeReceipt {
        contract_version: "fixed-cube-v1",
        acquisition_days: acquisition_days.to_vec(),
        acquisition_days_sha256: days_sha256,
        velocity_estimator: velocity_estimator.metadata_value().to_owned(),
        velocity_unit: "see velocity.tif UNITTYPE metadata",
        inference_status: "conditional_only",
        corrected_velocity_raster: None,
        corrected_sigma_raster: None,
        validity_mask_raster: "velocity_validity_mask.tif",
        velocity_raster: "velocity.tif",
        velocity_sigma_raster: velocity_sigma_present.then_some("velocity_sigma.tif"),
        los_rasters: ["los_east.tif", "los_north.tif", "los_up.tif"],
        geometry_provenance: "geometry_provenance.json",
        geometry_source: "CSLC-S1-STATIC",
        reference_point,
        epsg,
        geotransform,
        rows,
        cols,
        valid_pixels: validity_mask.iter().filter(|&&v| v).count(),
    };
    let receipt_path = dir.join("fixed_cube_receipt.json");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

fn validate_geometry(geometry: &LosGeometry, shape: (usize, usize)) -> Result<()> {
    ensure!(
        geometry.east.dim() == shape,
        "LOS east shape does not match fixed cube"
    );
    ensure!(
        geometry.north.dim() == shape,
        "LOS north shape does not match fixed cube"
    );
    ensure!(
        geometry.up.dim() == shape,
        "LOS up shape does not match fixed cube"
    );
    ensure!(
        geometry
            .east
            .iter()
            .chain(geometry.north.iter())
            .chain(geometry.up.iter())
            .all(|value| value.is_finite()),
        "fixed-cube LOS geometry contains non-finite values"
    );
    Ok(())
}

fn sha256_days(days: &[f64]) -> String {
    let mut digest = Sha256::new();
    for day in days {
        digest.update(day.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::sha256_days;

    #[test]
    fn epoch_digest_is_stable_for_exact_float_bytes() {
        assert_eq!(sha256_days(&[0.0, 12.0]), sha256_days(&[0.0, 12.0]));
        assert_ne!(sha256_days(&[0.0, 12.0]), sha256_days(&[0.0, 13.0]));
    }
}
