//! Fixed-cube output contract for downstream scientific validation.

use anyhow::{ensure, Result};
use dolphin_core::config::DisplacementWorkflow;
use dolphin_io::write_raster_with_metadata;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use dolphin_corrections::LosGeometry;
use ndarray::ArrayView2;

/// Stable receipt for the fixed-cube rasters consumed by EO.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedCubeReceipt {
    /// Contract schema identifier.
    pub contract_version: String,
    /// Exact decimal-day axis used by the run.
    pub acquisition_days: Vec<f64>,
    /// SHA-256 over the little-endian epoch bytes.
    pub acquisition_days_sha256: String,
    /// Stable estimator identity copied from `velocity.tif`.
    pub velocity_estimator: String,
    /// Unit recorded by the velocity raster metadata.
    pub velocity_unit: String,
    /// Inferential serving status; corrected temporal covariance is not present yet.
    pub inference_status: String,
    /// Corrected velocity raster, absent until issue #53 promotion gates pass.
    pub corrected_velocity_raster: Option<String>,
    /// Corrected sigma raster, absent until issue #53 promotion gates pass.
    pub corrected_sigma_raster: Option<String>,
    /// SHA-256 of the corrected velocity COG, absent before promotion.
    #[serde(default)]
    pub corrected_velocity_sha256: Option<String>,
    /// SHA-256 of the corrected sigma COG, absent before promotion.
    #[serde(default)]
    pub corrected_sigma_sha256: Option<String>,
    /// Corrected-inference provenance filename, absent before promotion.
    #[serde(default)]
    pub inference_provenance: Option<String>,
    /// SHA-256 of the corrected-inference provenance.
    #[serde(default)]
    pub inference_provenance_sha256: Option<String>,
    /// SHA-256 of the immutable #53 promotion manifest.
    #[serde(default)]
    pub temporal_promotion_manifest_sha256: Option<String>,
    /// Streamed validation of the exact fixed-cube mask and LOS rasters.
    #[serde(default)]
    pub semantic_validation: Option<FixedCubeSemanticValidation>,
    /// Fixed-cube validity mask filename.
    pub validity_mask_raster: String,
    /// Velocity raster filename.
    pub velocity_raster: String,
    /// Optional conditional sigma raster filename.
    pub velocity_sigma_raster: Option<String>,
    /// East, north, and up signed LOS raster filenames.
    pub los_rasters: [String; 3],
    /// Geometry provenance filename.
    pub geometry_provenance: String,
    /// Source identity for the signed LOS vectors.
    pub geometry_source: String,
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

/// Observed fixed-cube semantics bound into the promoted receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedCubeSemanticValidation {
    /// Valid mask pixels observed by the bounded raster scan.
    pub observed_valid_pixels: usize,
    /// Maximum absolute error in squared LOS-vector norm.
    pub maximum_los_norm_error: f32,
    /// Minimum observed upward LOS component.
    pub minimum_los_up: f32,
    /// Validated signed-LOS convention.
    pub los_sign_convention: String,
    /// Validated geometry source.
    pub geometry_source: String,
    /// Validation status of the sourced geometry provenance.
    pub geometry_provenance_status: String,
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
        ("UNITTYPE", "unitless"),
        ("RASTER_ROLE", "fixed_cube_run_geometry"),
    ];
    for (name, component) in [
        ("los_east.tif", &geometry.east),
        ("los_north.tif", &geometry.north),
        ("los_up.tif", &geometry.up),
    ] {
        let masked = ndarray::Array2::from_shape_fn(validity_mask.dim(), |index| {
            if validity_mask[index] {
                component[index] as f32
            } else {
                f32::NAN
            }
        });
        write_raster_with_metadata(
            &dir.join(name),
            masked.view(),
            geotransform,
            epsg,
            Some(f64::NAN),
            &geometry_tags,
        )?;
    }
    let days_sha256 = sha256_days(acquisition_days);
    let (rows, cols) = validity_mask.dim();
    let receipt = FixedCubeReceipt {
        contract_version: "fixed-cube-v1".to_owned(),
        acquisition_days: acquisition_days.to_vec(),
        acquisition_days_sha256: days_sha256,
        velocity_estimator: velocity_estimator.metadata_value().to_owned(),
        velocity_unit: "see velocity.tif UNITTYPE metadata".to_owned(),
        inference_status: "conditional_only".to_owned(),
        corrected_velocity_raster: None,
        corrected_sigma_raster: None,
        corrected_velocity_sha256: None,
        corrected_sigma_sha256: None,
        inference_provenance: None,
        inference_provenance_sha256: None,
        temporal_promotion_manifest_sha256: None,
        semantic_validation: None,
        validity_mask_raster: "velocity_validity_mask.tif".to_owned(),
        velocity_raster: "velocity.tif".to_owned(),
        velocity_sigma_raster: velocity_sigma_present.then(|| "velocity_sigma.tif".to_owned()),
        los_rasters: [
            "los_east.tif".to_owned(),
            "los_north.tif".to_owned(),
            "los_up.tif".to_owned(),
        ],
        geometry_provenance: "geometry_provenance.json".to_owned(),
        geometry_source: "CSLC-S1-STATIC".to_owned(),
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

/// Add corrected-product identities only after every COG and provenance file exists.
#[allow(clippy::too_many_arguments)]
pub(crate) fn promote_fixed_cube_receipt(
    directory: &std::path::Path,
    provenance_source: &std::path::Path,
    scratch: &std::path::Path,
    semantic_validation: FixedCubeSemanticValidation,
    corrected_velocity_sha256: String,
    corrected_sigma_sha256: String,
    provenance_sha256: String,
    promotion_manifest_sha256: String,
) -> Result<FixedCubeReceipt> {
    let path = directory.join("fixed_cube_receipt.json");
    let bytes = std::fs::read(&path)?;
    ensure!(
        bytes.len() <= 1024 * 1024,
        "fixed-cube receipt exceeds byte cap"
    );
    let mut receipt: FixedCubeReceipt = serde_json::from_slice(&bytes)?;
    ensure!(
        receipt.contract_version == "fixed-cube-v1"
            && receipt.inference_status == "conditional_only"
            && receipt.corrected_velocity_raster.is_none()
            && receipt.corrected_sigma_raster.is_none()
            && receipt.semantic_validation.is_none(),
        "fixed-cube receipt is not eligible for temporal-inference promotion"
    );
    for (name, expected_sha256) in [
        ("velocity_temporal_gls.tif", &corrected_velocity_sha256),
        ("velocity_sigma_corrected.tif", &corrected_sigma_sha256),
    ] {
        let product_path = directory.join(name);
        ensure!(
            sha256_file(&product_path)? == *expected_sha256,
            "{name} hash does not match completed COG"
        );
        let header = dolphin_io::read_raster_header(&product_path)?;
        ensure!(
            header.shape == (receipt.rows, receipt.cols)
                && header.geotransform == receipt.geotransform
                && header.epsg == receipt.epsg,
            "{name} grid differs from the fixed cube"
        );
    }
    ensure!(
        sha256_file(provenance_source)? == provenance_sha256,
        "temporal provenance hash does not match completion marker"
    );
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(provenance_source)?)?;
    ensure!(
        provenance.get("schema").and_then(|value| value.as_str())
            == Some("dolphinrust-temporal-inference-product/1"),
        "temporal provenance completion marker has the wrong schema"
    );
    receipt.inference_status = "calibrated_scope_match".to_owned();
    receipt.corrected_velocity_raster = Some("velocity_temporal_gls.tif".to_owned());
    receipt.corrected_sigma_raster = Some("velocity_sigma_corrected.tif".to_owned());
    receipt.corrected_velocity_sha256 = Some(corrected_velocity_sha256);
    receipt.corrected_sigma_sha256 = Some(corrected_sigma_sha256);
    receipt.inference_provenance = Some("velocity_inference_provenance.json".to_owned());
    receipt.inference_provenance_sha256 = Some(provenance_sha256);
    receipt.temporal_promotion_manifest_sha256 = Some(promotion_manifest_sha256);
    receipt.semantic_validation = Some(semantic_validation);
    ensure!(
        scratch.parent() == Some(directory) && !scratch.exists(),
        "fixed-cube receipt scratch is not a fresh owned path"
    );
    let mut scratch_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(scratch)?;
    std::io::Write::write_all(&mut scratch_file, &serde_json::to_vec_pretty(&receipt)?)?;
    scratch_file.sync_all()?;
    drop(scratch_file);
    if let Err(error) = std::fs::rename(scratch, path) {
        return Err(error.into());
    }
    std::fs::File::open(directory)?.sync_all()?;
    Ok(receipt)
}

fn sha256_file(path: &std::path::Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
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
    use dolphin_core::config::DisplacementWorkflow;
    use ndarray::array;

    #[test]
    fn epoch_digest_is_stable_for_exact_float_bytes() {
        assert_eq!(sha256_days(&[0.0, 12.0]), sha256_days(&[0.0, 12.0]));
        assert_ne!(sha256_days(&[0.0, 12.0]), sha256_days(&[0.0, 13.0]));
    }

    #[test]
    fn los_rasters_are_nodata_outside_exact_common_support() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_fixed_cube_masked_los_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let workflow = DisplacementWorkflow {
            work_directory: directory.clone(),
            ..DisplacementWorkflow::default()
        };
        let geometry = dolphin_corrections::LosGeometry {
            east: array![[0.5], [0.5]],
            north: array![[0.0], [0.0]],
            up: array![[3.0_f64.sqrt() / 2.0], [3.0_f64.sqrt() / 2.0]],
        };
        super::write_fixed_cube_bundle(
            &workflow,
            &[0.0, 12.0],
            crate::displacement::VelocityEstimator::LinearPostGaugeUnitPrecision,
            false,
            array![[true], [false]].view(),
            &geometry,
            Some((0, 0)),
            Some(32611),
            [500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0],
        )
        .unwrap();
        for name in ["los_east.tif", "los_north.tif", "los_up.tif"] {
            let raster = dolphin_io::read_raster::<f32>(&directory.join(name)).unwrap();
            assert!(raster.data[(0, 0)].is_finite());
            assert!(raster.data[(1, 0)].is_nan());
            assert!(dolphin_io::read_raster_header(&directory.join(name))
                .unwrap()
                .nodata
                .is_some_and(f64::is_nan));
            assert_eq!(
                dolphin_io::read_raster_header(&directory.join(name))
                    .unwrap()
                    .metadata
                    .get("UNITTYPE")
                    .map(String::as_str),
                Some("unitless")
            );
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
