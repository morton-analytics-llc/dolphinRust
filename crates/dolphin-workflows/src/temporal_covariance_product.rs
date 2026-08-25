//! Calibrated, fail-closed temporal-GLS raster products for issue #53.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Read;
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{ensure, Context, Result};
use dolphin_core::config::{TemporalUncertaintyMethod, TemporalUncertaintyOptions};
use dolphin_core::BlockIndices;
use dolphin_io::{
    read_raster_header, read_raster_window, read_spatial_reference_covariance_block,
    read_spatial_reference_covariance_block_ids, read_spatial_reference_covariance_header,
    BoundedCogWriter, SpatialReferenceCalibrationScope, SpatialReferenceCovarianceStatus,
};
use dolphin_timeseries::{
    complete_refit_bootstrap_estimate, fit_temporal_covariance,
    CompleteRefitBootstrapCadenceStatus, CompleteRefitBootstrapEstimate,
    CompleteRefitBootstrapEstimateStatus, TemporalCovarianceOptions, TemporalInferenceStatus,
    COMPLETE_REFIT_BOOTSTRAP_METHOD, COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::fixed_cube::promote_fixed_cube_receipt;
use crate::spatial_covariance_artifact::{
    read_spatial_reference_covariance_artifact_manifest, SPATIAL_REFERENCE_COVARIANCE_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME,
};

/// Synthetic result filename required by the promotion validator.
pub const TEMPORAL_SYNTHETIC_RESULT_FILENAME: &str = "temporal_covariance_synthetic_result.json";
/// Externally supplied non-Fresno holdout result filename.
pub const TEMPORAL_HELDOUT_RESULT_FILENAME: &str = "temporal_covariance_heldout_result.json";
/// Independent scientific-review receipt filename.
pub const TEMPORAL_REVIEW_RECEIPT_FILENAME: &str = "temporal_covariance_review_receipt.json";
/// Immutable completion/promotion manifest filename.
pub const TEMPORAL_PROMOTION_MANIFEST_FILENAME: &str =
    "temporal_covariance_promotion_manifest.json";
/// Product provenance completion marker, published after every COG.
pub const TEMPORAL_INFERENCE_PROVENANCE_FILENAME: &str = "velocity_inference_provenance.json";

const JSON_CAP: u64 = 64 * 1024 * 1024;
const PRODUCT_SCHEMA: &str = "dolphinrust-temporal-inference-product/1";
const PROMOTION_SCHEMA: &str = "dolphinrust-temporal-covariance-promotion/1";
const REVIEW_SCHEMA: &str = "dolphinrust-temporal-covariance-review/1";
const SYNTHETIC_SCHEMA: &str = "dolphinrust-temporal-covariance-simulation/5";
const LAYER_COUNT: usize = 14;
const COMBINED_WORKING_SET_CAP_BYTES: u64 = 512 * 1024 * 1024;
const TRANSACTION_LOCK_FILENAME: &str = ".temporal-covariance-product.lock";
const ROLLBACK_JOURNAL_FILENAME: &str = ".temporal-covariance-product.rollback.json";
const ROLLBACK_JOURNAL_SCHEMA: &str = "dolphinrust-temporal-product-rollback/1";
static NEXT_TRANSACTION_FILE_ID: AtomicU64 = AtomicU64::new(0);

const TEMPORAL_PREREGISTRATION_BYTES: &[u8] =
    include_bytes!("../../../validation/temporal_covariance_preregistration.json");
const HELDOUT_PREREGISTRATION_BYTES: &[u8] =
    include_bytes!("../../../validation/temporal_covariance_heldout_preregistration.json");
const SYNTHETIC_SCORER_BYTES: &[u8] =
    include_bytes!("../../../validation/temporal_covariance_simulation.py");
const HELDOUT_SCORER_BYTES: &[u8] =
    include_bytes!("../../../validation/score_temporal_covariance_holdout.py");
const HELDOUT_LIBRARY_BYTES: &[u8] =
    include_bytes!("../../../validation/heldout_temporal_covariance/scorer.py");
const ESTIMATOR_SOURCE_BYTES: &[u8] =
    include_bytes!("../../dolphin-timeseries/src/temporal_covariance.rs");
const BATCH_SOURCE_BYTES: &[u8] =
    include_bytes!("../../dolphin-timeseries/examples/temporal_covariance_batch.rs");
const PRODUCT_SOURCE_BYTES: &[u8] = include_bytes!("temporal_covariance_product.rs");
const FIXED_CUBE_SOURCE_BYTES: &[u8] = include_bytes!("fixed_cube.rs");

/// A promotion authorization that cannot be constructed outside this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalCovariancePromotion {
    manifest_sha256: String,
    review_sha256: String,
    synthetic_sha256: String,
    heldout_sha256: String,
    spatial_manifest_sha256: String,
    spatial_factor_sha256: String,
}

/// Persisted identities of one completed corrected product bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalCovarianceProductReceipt {
    /// SHA-256 of `velocity_temporal_gls.tif`.
    pub corrected_velocity_sha256: String,
    /// SHA-256 of `velocity_sigma_corrected.tif`.
    pub corrected_sigma_sha256: String,
    /// SHA-256 of the provenance completion marker.
    pub provenance_sha256: String,
    /// SHA-256 of the promotion manifest that authorized emission.
    pub promotion_manifest_sha256: String,
}

struct TemporalProductTransaction {
    directory: PathBuf,
    _lock: File,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProductRollbackJournal {
    schema: String,
    original_fixed_cube_receipt: Vec<u8>,
}

impl TemporalProductTransaction {
    #[cfg(unix)]
    fn acquire(directory: &Path) -> Result<Self> {
        let lock_path = directory.join(TRANSACTION_LOCK_FILENAME);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("opening temporal product lock {}", lock_path.display()))?;
        // SAFETY: `lock` owns a live descriptor for the lifetime of this guard.
        let result = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        ensure!(result == 0, "temporal covariance product is already locked");
        let transaction = Self {
            directory: directory.to_owned(),
            _lock: lock,
        };
        recover_incomplete_product(&transaction.directory)?;
        Ok(transaction)
    }

    #[cfg(not(unix))]
    fn acquire(_directory: &Path) -> Result<Self> {
        anyhow::bail!("temporal covariance product locking is unsupported on this platform")
    }
}

#[derive(Deserialize)]
struct SyntheticScores {
    all_methods_pass: bool,
}

#[derive(Deserialize)]
struct SyntheticResult {
    schema: String,
    attempted_cells: u64,
    batch_attempted_cells: u64,
    seed_count: u64,
    execution_complete: bool,
    exact_seed_denominator_complete: bool,
    corrected_inferential_sigma_emission: bool,
    promotion_eligible: bool,
    promotion_status: String,
    scores: SyntheticScores,
    resource_gates: BTreeMap<String, bool>,
}

#[derive(Deserialize)]
struct HeldoutLevel {
    status: String,
}

#[derive(Deserialize)]
struct HeldoutResult {
    status: String,
    errors: Vec<Value>,
    levels: BTreeMap<String, HeldoutLevel>,
    evaluated_clusters: usize,
    emission_rate: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalReviewReceipt {
    schema: String,
    review_status: String,
    reviewer: String,
    independent: bool,
    unresolved_findings: u32,
    synthetic_result_sha256: String,
    heldout_result_sha256: String,
    spatial_manifest_sha256: String,
    temporal_preregistration_sha256: String,
    heldout_preregistration_sha256: String,
    scorer_sha256: String,
    source_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalPromotionManifest {
    schema: String,
    promotion_status: String,
    calibration_scope: String,
    selected_method: String,
    selected_method_version: u16,
    synthetic_result_sha256: String,
    heldout_result_sha256: String,
    review_receipt_sha256: String,
    spatial_factor_sha256: String,
    spatial_manifest_sha256: String,
    temporal_preregistration_sha256: String,
    heldout_preregistration_sha256: String,
    scorer_sha256: String,
    source_sha256: String,
}

/// Validate the full immutable #54/#53 evidence chain and return an
/// unforgeable authorization token.
pub fn validate_temporal_covariance_promotion(
    evidence_directory: &Path,
    factor_directory: &Path,
) -> Result<TemporalCovariancePromotion> {
    let spatial = read_spatial_reference_covariance_artifact_manifest(factor_directory)?;
    ensure!(
        spatial.calibration_scope == "calibrated_scope_match",
        "#54 factor is not calibrated for its exact scope"
    );
    let spatial_manifest_bytes = read_bounded(
        &factor_directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME),
        JSON_CAP,
    )?;
    let spatial_manifest_sha256 = sha256(&spatial_manifest_bytes);
    let synthetic_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_SYNTHETIC_RESULT_FILENAME),
        JSON_CAP,
    )?;
    let synthetic: SyntheticResult = serde_json::from_slice(&synthetic_bytes)?;
    validate_synthetic_result(&synthetic)?;
    let heldout_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_HELDOUT_RESULT_FILENAME),
        JSON_CAP,
    )?;
    let heldout: HeldoutResult = serde_json::from_slice(&heldout_bytes)?;
    validate_heldout_result(&heldout)?;
    let review_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_REVIEW_RECEIPT_FILENAME),
        JSON_CAP,
    )?;
    let review: TemporalReviewReceipt = serde_json::from_slice(&review_bytes)?;
    let expected = EvidenceDigests::current(
        sha256(&synthetic_bytes),
        sha256(&heldout_bytes),
        spatial.hdf5_sha256,
        spatial_manifest_sha256.clone(),
    );
    validate_review(&review, &expected)?;
    let review_sha256 = sha256(&review_bytes);
    let manifest_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_PROMOTION_MANIFEST_FILENAME),
        JSON_CAP,
    )?;
    let manifest: TemporalPromotionManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&manifest, &expected, &review_sha256)?;
    Ok(TemporalCovariancePromotion {
        manifest_sha256: sha256(&manifest_bytes),
        review_sha256,
        synthetic_sha256: expected.synthetic_result_sha256,
        heldout_sha256: expected.heldout_result_sha256,
        spatial_manifest_sha256,
        spatial_factor_sha256: expected.spatial_factor_sha256,
    })
}

struct EvidenceDigests {
    synthetic_result_sha256: String,
    heldout_result_sha256: String,
    spatial_factor_sha256: String,
    spatial_manifest_sha256: String,
    temporal_preregistration_sha256: String,
    heldout_preregistration_sha256: String,
    scorer_sha256: String,
    source_sha256: String,
}

impl EvidenceDigests {
    fn current(
        synthetic_result_sha256: String,
        heldout_result_sha256: String,
        spatial_factor_sha256: String,
        spatial_manifest_sha256: String,
    ) -> Self {
        let mut scorer = Sha256::new();
        scorer.update(SYNTHETIC_SCORER_BYTES);
        scorer.update(HELDOUT_SCORER_BYTES);
        scorer.update(HELDOUT_LIBRARY_BYTES);
        let mut source = Sha256::new();
        source.update(ESTIMATOR_SOURCE_BYTES);
        source.update(BATCH_SOURCE_BYTES);
        source.update(PRODUCT_SOURCE_BYTES);
        source.update(FIXED_CUBE_SOURCE_BYTES);
        Self {
            synthetic_result_sha256,
            heldout_result_sha256,
            spatial_factor_sha256,
            spatial_manifest_sha256,
            temporal_preregistration_sha256: sha256(TEMPORAL_PREREGISTRATION_BYTES),
            heldout_preregistration_sha256: sha256(HELDOUT_PREREGISTRATION_BYTES),
            scorer_sha256: format!("{:x}", scorer.finalize()),
            source_sha256: format!("{:x}", source.finalize()),
        }
    }
}

fn validate_synthetic_result(result: &SyntheticResult) -> Result<()> {
    ensure!(
        result.schema == SYNTHETIC_SCHEMA,
        "unsupported synthetic result schema"
    );
    ensure!(
        result.execution_complete
            && result.exact_seed_denominator_complete
            && result.promotion_eligible
            && result.promotion_status == "eligible_for_external_field_review"
            && result.scores.all_methods_pass
            && !result.corrected_inferential_sigma_emission,
        "synthetic temporal-covariance result is incomplete or failed"
    );
    ensure!(
        result.seed_count == 5_000
            && result.attempted_cells == 240_000
            && result.batch_attempted_cells == result.attempted_cells,
        "synthetic temporal-covariance denominator is not the exact frozen matrix"
    );
    ensure!(
        !result.resource_gates.is_empty() && result.resource_gates.values().all(|passed| *passed),
        "synthetic temporal-covariance resource gates did not all pass"
    );
    Ok(())
}

fn validate_heldout_result(result: &HeldoutResult) -> Result<()> {
    ensure!(
        result.status == "pass"
            && result.errors.is_empty()
            && result.evaluated_clusters >= 96
            && result.emission_rate.is_finite()
            && result.emission_rate >= 0.99,
        "held-out temporal-covariance result did not pass the frozen cohort"
    );
    ensure!(
        result.levels.len() == 3
            && ["68", "90", "95"].iter().all(|level| {
                result
                    .levels
                    .get(*level)
                    .is_some_and(|value| value.status == "pass")
            }),
        "held-out temporal-covariance level gates are incomplete"
    );
    Ok(())
}

fn validate_review(review: &TemporalReviewReceipt, expected: &EvidenceDigests) -> Result<()> {
    ensure!(
        review.schema == REVIEW_SCHEMA
            && review.review_status == "approved"
            && !review.reviewer.trim().is_empty()
            && review.independent
            && review.unresolved_findings == 0,
        "independent temporal-covariance review is not approved"
    );
    ensure!(
        review.synthetic_result_sha256 == expected.synthetic_result_sha256
            && review.heldout_result_sha256 == expected.heldout_result_sha256
            && review.spatial_manifest_sha256 == expected.spatial_manifest_sha256
            && review.temporal_preregistration_sha256 == expected.temporal_preregistration_sha256
            && review.heldout_preregistration_sha256 == expected.heldout_preregistration_sha256
            && review.scorer_sha256 == expected.scorer_sha256
            && review.source_sha256 == expected.source_sha256,
        "independent temporal-covariance review hashes are stale"
    );
    Ok(())
}

fn validate_manifest(
    manifest: &TemporalPromotionManifest,
    expected: &EvidenceDigests,
    review_sha256: &str,
) -> Result<()> {
    ensure!(
        manifest.schema == PROMOTION_SCHEMA
            && manifest.promotion_status == "approved"
            && manifest.calibration_scope == "calibrated_scope_match"
            && manifest.selected_method == COMPLETE_REFIT_BOOTSTRAP_METHOD
            && manifest.selected_method_version == COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
        "temporal-covariance promotion manifest is not approved for the selected method"
    );
    ensure!(
        manifest.synthetic_result_sha256 == expected.synthetic_result_sha256
            && manifest.heldout_result_sha256 == expected.heldout_result_sha256
            && manifest.review_receipt_sha256 == review_sha256
            && manifest.spatial_factor_sha256 == expected.spatial_factor_sha256
            && manifest.spatial_manifest_sha256 == expected.spatial_manifest_sha256
            && manifest.temporal_preregistration_sha256 == expected.temporal_preregistration_sha256
            && manifest.heldout_preregistration_sha256 == expected.heldout_preregistration_sha256
            && manifest.scorer_sha256 == expected.scorer_sha256
            && manifest.source_sha256 == expected.source_sha256,
        "temporal-covariance promotion manifest hashes are stale or scope-mismatched"
    );
    Ok(())
}

/// Write bounded corrected temporal-GLS products from post-gauge displacement COGs.
///
/// The provenance JSON is the completion marker and is published after every
/// COG. Legacy `velocity.tif` and `velocity_sigma.tif` are hashed before and
/// after the transaction and are never opened for writing.
pub fn write_temporal_covariance_products(
    output_directory: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    config: &TemporalUncertaintyOptions,
) -> Result<TemporalCovarianceProductReceipt> {
    ensure!(
        config.method == TemporalUncertaintyMethod::CompleteRefitBootstrap,
        "corrected temporal inference is disabled"
    );
    let transaction = TemporalProductTransaction::acquire(output_directory)?;
    let evidence_directory = config
        .evidence_directory
        .as_deref()
        .context("temporal uncertainty evidence directory is missing")?;
    let factor_directory = config
        .factor_directory
        .as_deref()
        .context("temporal uncertainty factor directory is missing")?;
    ensure_same_run_factor_directory(output_directory, factor_directory)?;
    let promotion = validate_temporal_covariance_promotion(evidence_directory, factor_directory)?;
    validate_no_existing_products(output_directory)?;
    ensure!(
        displacement_rasters
            .len()
            .checked_add(1)
            .is_some_and(|count| count == acquisition_days.len()),
        "post-gauge displacement raster count must equal acquisition-day count minus one"
    );
    let velocity_path = output_directory.join("velocity.tif");
    let legacy_velocity_before = sha256_file(&velocity_path)?;
    let legacy_sigma_path = output_directory.join("velocity_sigma.tif");
    let legacy_sigma_before = legacy_sigma_path
        .exists()
        .then(|| sha256_file(&legacy_sigma_path))
        .transpose()?;
    let fixed_cube_receipt_before = read_bounded(
        &output_directory.join("fixed_cube_receipt.json"),
        1024 * 1024,
    )?;
    let result = write_product_transaction(
        output_directory,
        displacement_rasters,
        acquisition_days,
        config,
        factor_directory,
        &promotion,
        &transaction,
    );
    let receipt = match result {
        Ok(receipt) => receipt,
        Err(error) => {
            remove_published_products(output_directory)?;
            return Err(error);
        }
    };
    complete_publication_after_legacy_check(
        output_directory,
        receipt,
        &legacy_velocity_before,
        legacy_sigma_before.as_deref(),
        &fixed_cube_receipt_before,
    )
}

fn complete_publication_after_legacy_check(
    output_directory: &Path,
    receipt: TemporalCovarianceProductReceipt,
    legacy_velocity_before: &str,
    legacy_sigma_before: Option<&str>,
    fixed_cube_receipt_before: &[u8],
) -> Result<TemporalCovarianceProductReceipt> {
    let legacy_check = (|| {
        let velocity_path = output_directory.join("velocity.tif");
        let sigma_path = output_directory.join("velocity_sigma.tif");
        ensure!(
            sha256_file(&velocity_path)? == legacy_velocity_before
                && sigma_path
                    .exists()
                    .then(|| sha256_file(&sigma_path))
                    .transpose()?
                    .as_deref()
                    == legacy_sigma_before,
            "legacy velocity products changed during corrected inference"
        );
        Ok(())
    })();
    if let Err(error) = legacy_check {
        let removal = remove_published_products(output_directory);
        let restoration = restore_fixed_cube_receipt(output_directory, fixed_cube_receipt_before);
        removal.context("removing corrected products after legacy-product mutation")?;
        restoration.context("restoring fixed-cube receipt after legacy-product mutation")?;
        return Err(error);
    }
    Ok(receipt)
}

fn write_product_transaction(
    output_directory: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    config: &TemporalUncertaintyOptions,
    factor_directory: &Path,
    promotion: &TemporalCovariancePromotion,
    transaction: &TemporalProductTransaction,
) -> Result<TemporalCovarianceProductReceipt> {
    ensure_same_run_factor_directory(output_directory, factor_directory)?;
    let evidence_directory = config
        .evidence_directory
        .as_deref()
        .context("temporal uncertainty evidence directory is missing")?;
    write_product_transaction_with_validator(
        output_directory,
        displacement_rasters,
        acquisition_days,
        config,
        factor_directory,
        promotion,
        transaction,
        || validate_temporal_covariance_promotion(evidence_directory, factor_directory),
    )
}

fn ensure_same_run_factor_directory(
    output_directory: &Path,
    factor_directory: &Path,
) -> Result<()> {
    ensure!(
        std::fs::canonicalize(output_directory)? == std::fs::canonicalize(factor_directory)?,
        "temporal inference requires the run-specific #54 factor from the output directory"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_product_transaction_with_validator(
    output_directory: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    config: &TemporalUncertaintyOptions,
    factor_directory: &Path,
    promotion: &TemporalCovariancePromotion,
    transaction: &TemporalProductTransaction,
    mut revalidate: impl FnMut() -> Result<TemporalCovariancePromotion>,
) -> Result<TemporalCovarianceProductReceipt> {
    ensure!(
        transaction.directory == output_directory,
        "temporal product transaction owns a different output directory"
    );
    let scope = prepare_product_scope(
        output_directory,
        displacement_rasters,
        acquisition_days,
        config,
        factor_directory,
    )?;
    let stage = create_stage_directory(output_directory)?;
    let transaction = (|| {
        admit_combined_working_set(config, acquisition_days.len())?;
        let mut layers = create_layer_writers(&stage, &scope.velocity_header)?;
        process_factor_blocks(
            &scope.factor_path,
            displacement_rasters,
            acquisition_days,
            config,
            scope.factor_metadata.full_grid,
            &mut layers,
        )?;
        finalize_layers(&stage, &mut layers)?;
        ensure!(
            input_raster_receipts(displacement_rasters)? == scope.input_receipts,
            "displacement rasters changed during temporal inference"
        );
        ensure!(
            revalidate()? == *promotion,
            "temporal promotion or factor evidence changed during product generation"
        );
        ensure!(
            input_raster_receipts(&scope.fixed_cube_paths)? == scope.fixed_cube_inputs,
            "fixed-cube inputs changed during temporal inference"
        );
        publish_product_receipt(
            output_directory,
            &stage,
            displacement_rasters,
            acquisition_days,
            scope,
            promotion,
            &mut revalidate,
        )
    })();
    let _ = std::fs::remove_dir_all(&stage);
    if transaction.is_err() && output_directory.join(ROLLBACK_JOURNAL_FILENAME).exists() {
        recover_incomplete_product(output_directory)?;
    }
    transaction
}

struct ProductScope {
    factor_path: PathBuf,
    factor_metadata: dolphin_io::SpatialReferenceCovarianceMetadata,
    velocity_header: dolphin_io::RasterHeader,
    velocity_unit: String,
    input_receipts: Vec<InputRasterReceipt>,
    fixed_cube_paths: Vec<PathBuf>,
    fixed_cube_inputs: Vec<InputRasterReceipt>,
    fixed_cube_semantics: crate::fixed_cube::FixedCubeSemanticValidation,
}

fn prepare_product_scope(
    output_directory: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    config: &TemporalUncertaintyOptions,
    factor_directory: &Path,
) -> Result<ProductScope> {
    let factor_path = factor_directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let factor_metadata =
        read_spatial_reference_covariance_header(&factor_path, config.factor_block_read_cap_bytes)?;
    ensure!(
        factor_metadata.calibration_scope == SpatialReferenceCalibrationScope::CalibratedScopeMatch,
        "factor header is not calibrated_scope_match"
    );
    ensure!(
        factor_metadata.acquisition_days.as_deref() == Some(acquisition_days),
        "factor acquisition days differ from corrected product"
    );
    ensure!(
        factor_metadata.gauge_date_index == 0,
        "corrected temporal inference requires acquisition zero as gauge"
    );
    let velocity_header = read_raster_header(&output_directory.join("velocity.tif"))?;
    let velocity_unit = velocity_header
        .metadata
        .get("UNITTYPE")
        .cloned()
        .context("legacy velocity raster is missing UNITTYPE")?;
    let expected_velocity_unit = match factor_metadata.units.as_str() {
        "radians" => "rad/yr",
        "meters" => "m/yr",
        "millimeters" => "mm/yr",
        _ => anyhow::bail!("unsupported factor units"),
    };
    ensure!(
        velocity_unit == expected_velocity_unit,
        "factor units differ from the legacy velocity product"
    );
    validate_input_rasters(displacement_rasters, &velocity_header, acquisition_days)?;
    let full_grid = factor_metadata.full_grid;
    ensure!(
        usize::try_from(full_grid.rows).ok() == Some(velocity_header.shape.0)
            && usize::try_from(full_grid.cols).ok() == Some(velocity_header.shape.1),
        "factor full grid differs from displacement raster shape"
    );
    let (fixed_cube, fixed_cube_semantics) = validate_fixed_cube_scope(
        output_directory,
        acquisition_days,
        &velocity_header,
        &factor_metadata,
    )?;
    let fixed_cube_paths = fixed_cube_input_paths(output_directory, &fixed_cube);
    let fixed_cube_inputs = input_raster_receipts(&fixed_cube_paths)?;
    Ok(ProductScope {
        factor_path,
        factor_metadata,
        velocity_header,
        velocity_unit,
        input_receipts: input_raster_receipts(displacement_rasters)?,
        fixed_cube_paths,
        fixed_cube_inputs,
        fixed_cube_semantics,
    })
}

fn process_factor_blocks(
    factor_path: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    config: &TemporalUncertaintyOptions,
    full_grid: dolphin_io::CovarianceOperatorGrid,
    layers: &mut [ProductLayer],
) -> Result<()> {
    let block_ids =
        read_spatial_reference_covariance_block_ids(factor_path, config.block_id_read_cap_bytes)?;
    ensure!(
        !block_ids.is_empty(),
        "factor artifact contains no target blocks"
    );
    let options = TemporalCovarianceOptions::default();
    for block_id in block_ids {
        let read = read_spatial_reference_covariance_block(
            factor_path,
            block_id,
            config.factor_block_read_cap_bytes,
        )?;
        let target_count = usize::try_from(
            u64::from(read.block.target_grid.rows)
                .checked_mul(u64::from(read.block.target_grid.cols))
                .context("factor target count overflow")?,
        )?;
        ensure!(
            target_count <= config.maximum_targets_per_block,
            "factor block exceeds configured target microbatch cap"
        );
        let output_window = output_window(read.block.target_grid, full_grid)?;
        let observations = displacement_rasters
            .iter()
            .map(|path| read_raster_window::<f32>(path, output_window))
            .collect::<dolphin_io::Result<Vec<_>>>()?;
        let values = evaluate_block(&read.block, &observations, acquisition_days, &options)?;
        for (layer, value) in layers.iter_mut().zip(values.iter()) {
            layer
                .writer
                .as_mut()
                .context("product writer already finalized")?
                .write_window(output_window, value.view())?;
        }
    }
    Ok(())
}

fn admit_combined_working_set(
    config: &TemporalUncertaintyOptions,
    acquisition_count: usize,
) -> Result<u64> {
    let targets = u64::try_from(config.maximum_targets_per_block)?;
    let dates = u64::try_from(acquisition_count)?;
    let post_gauge_dates = dates
        .checked_sub(1)
        .context("temporal working-set admission requires a gauge acquisition")?;
    let displacement_windows = targets
        .checked_mul(post_gauge_dates)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .context("displacement-window working-set size overflow")?;
    let output_windows = targets
        .checked_mul(LAYER_COUNT as u64)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .context("output-window working-set size overflow")?;
    let covariance_scratch = dates
        .checked_mul(dates)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() as u64))
        .and_then(|value| {
            dates
                .checked_mul(std::mem::size_of::<f64>() as u64)
                .and_then(|series| value.checked_add(series))
        })
        .context("temporal fit scratch working-set size overflow")?;
    let combined = config
        .factor_block_read_cap_bytes
        .checked_add(displacement_windows)
        .and_then(|value| value.checked_add(output_windows))
        .and_then(|value| value.checked_add(covariance_scratch))
        .context("combined temporal working-set size overflow")?;
    ensure!(
        combined <= COMBINED_WORKING_SET_CAP_BYTES,
        "combined temporal working set {combined} exceeds cap {COMBINED_WORKING_SET_CAP_BYTES}"
    );
    Ok(combined)
}

fn publish_product_receipt(
    output_directory: &Path,
    stage: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    scope: ProductScope,
    promotion: &TemporalCovariancePromotion,
    revalidate: &mut impl FnMut() -> Result<TemporalCovariancePromotion>,
) -> Result<TemporalCovarianceProductReceipt> {
    let original_fixed_cube_receipt = read_bounded(
        &output_directory.join("fixed_cube_receipt.json"),
        1024 * 1024,
    )?;
    write_rollback_journal(output_directory, &original_fixed_cube_receipt)?;
    let staged_products = product_receipts(stage)?;
    publish_layers_no_replace(output_directory, stage)?;
    verify_final_cogs(output_directory, &staged_products, &scope.velocity_header)?;
    let corrected_velocity_sha256 =
        sha256_file(&output_directory.join("velocity_temporal_gls.tif"))?;
    let corrected_sigma_sha256 =
        sha256_file(&output_directory.join("velocity_sigma_corrected.tif"))?;
    let final_product_receipts = product_receipts(output_directory)?;
    let provenance = TemporalInferenceProvenance::new(
        acquisition_days,
        &scope,
        promotion,
        &corrected_velocity_sha256,
        &corrected_sigma_sha256,
        final_product_receipts,
    );
    let provenance_path = output_directory.join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME);
    let provenance_scratch = stage.join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME);
    std::fs::write(&provenance_scratch, serde_json::to_vec_pretty(&provenance)?)?;
    File::open(&provenance_scratch)?.sync_all()?;
    let provenance_sha256 = sha256_file(&provenance_scratch)?;
    promote_fixed_cube_receipt(
        output_directory,
        &provenance_scratch,
        scope.fixed_cube_semantics.clone(),
        corrected_velocity_sha256.clone(),
        corrected_sigma_sha256.clone(),
        provenance_sha256.clone(),
        promotion.manifest_sha256.clone(),
    )?;
    verify_final_cogs(output_directory, &staged_products, &scope.velocity_header)?;
    verify_promoted_fixed_cube_receipt(
        output_directory,
        &corrected_velocity_sha256,
        &corrected_sigma_sha256,
        &provenance_sha256,
        &promotion.manifest_sha256,
        &scope.fixed_cube_semantics,
    )?;
    ensure!(
        revalidate()? == *promotion,
        "temporal promotion or factor evidence changed before completion marker"
    );
    ensure!(
        input_raster_receipts(displacement_rasters)? == scope.input_receipts,
        "displacement rasters changed before temporal completion marker"
    );
    ensure!(
        input_raster_receipts(&scope.fixed_cube_paths[1..])? == scope.fixed_cube_inputs[1..],
        "fixed-cube inputs changed before temporal completion marker"
    );
    install_no_replace(&provenance_scratch, &provenance_path)?;
    if let Err(error) = remove_rollback_journal(output_directory) {
        tracing::warn!(%error, "committed temporal product left a recovery journal");
    }
    Ok(TemporalCovarianceProductReceipt {
        corrected_velocity_sha256,
        corrected_sigma_sha256,
        provenance_sha256,
        promotion_manifest_sha256: promotion.manifest_sha256.clone(),
    })
}

fn validate_fixed_cube_scope(
    directory: &Path,
    acquisition_days: &[f64],
    header: &dolphin_io::RasterHeader,
    factor: &dolphin_io::SpatialReferenceCovarianceMetadata,
) -> Result<(
    crate::fixed_cube::FixedCubeReceipt,
    crate::fixed_cube::FixedCubeSemanticValidation,
)> {
    let bytes = read_bounded(&directory.join("fixed_cube_receipt.json"), 1024 * 1024)?;
    let receipt: crate::fixed_cube::FixedCubeReceipt = serde_json::from_slice(&bytes)?;
    let reference_row = factor
        .reference_row
        .checked_sub(factor.full_grid.row_start)
        .context("factor reference row precedes full grid")?;
    let reference_col = factor
        .reference_col
        .checked_sub(factor.full_grid.col_start)
        .context("factor reference column precedes full grid")?;
    ensure!(
        reference_row % u64::from(factor.full_grid.stride_y) == 0
            && reference_col % u64::from(factor.full_grid.stride_x) == 0,
        "factor reference is not aligned to the output grid"
    );
    let reference = (
        usize::try_from(reference_row / u64::from(factor.full_grid.stride_y))?,
        usize::try_from(reference_col / u64::from(factor.full_grid.stride_x))?,
    );
    ensure!(
        receipt.contract_version == "fixed-cube-v1"
            && receipt.inference_status == "conditional_only"
            && receipt.corrected_velocity_raster.is_none()
            && receipt.corrected_sigma_raster.is_none()
            && receipt.semantic_validation.is_none()
            && receipt.acquisition_days == acquisition_days
            && receipt.rows == header.shape.0
            && receipt.cols == header.shape.1
            && receipt.geotransform == header.geotransform
            && receipt.epsg == header.epsg
            && receipt.reference_point == Some(reference)
            && receipt.velocity_raster == "velocity.tif"
            && receipt
                .velocity_sigma_raster
                .as_deref()
                .is_none_or(|name| name == "velocity_sigma.tif")
            && receipt.validity_mask_raster == "velocity_validity_mask.tif"
            && receipt.geometry_provenance == "geometry_provenance.json"
            && receipt.geometry_source == "CSLC-S1-STATIC"
            && receipt.los_rasters
                == [
                    "los_east.tif".to_owned(),
                    "los_north.tif".to_owned(),
                    "los_up.tif".to_owned(),
                ],
        "fixed-cube receipt does not match the factor and displacement scope"
    );
    let semantics = validate_fixed_cube_semantics(directory, header, &receipt)?;
    Ok((receipt, semantics))
}

fn validate_fixed_cube_semantics(
    directory: &Path,
    velocity_header: &dolphin_io::RasterHeader,
    receipt: &crate::fixed_cube::FixedCubeReceipt,
) -> Result<crate::fixed_cube::FixedCubeSemanticValidation> {
    ensure!(
        velocity_header
            .metadata
            .get("VELOCITY_ESTIMATOR")
            .map(String::as_str)
            == Some(receipt.velocity_estimator.as_str()),
        "fixed-cube estimator identity differs from velocity.tif"
    );
    validate_single_band_type(
        &directory.join(&receipt.velocity_raster),
        gdal::raster::GdalDataType::Float32,
    )?;
    if let Some(sigma) = &receipt.velocity_sigma_raster {
        validate_fixed_cube_grid(&directory.join(sigma), velocity_header)?;
        validate_single_band_type(&directory.join(sigma), gdal::raster::GdalDataType::Float32)?;
    }
    let mask_path = directory.join(&receipt.validity_mask_raster);
    let mask_header = validate_fixed_cube_grid(&mask_path, velocity_header)?;
    validate_single_band_type(&mask_path, gdal::raster::GdalDataType::UInt8)?;
    ensure!(
        mask_header.metadata.get("MASK_ROLE").map(String::as_str) == Some("velocity_support")
            && mask_header.metadata.get("MASK_VALUES").map(String::as_str)
                == Some("0=invalid;1=valid")
            && mask_header.metadata.get("MASK_POLICY").map(String::as_str)
                == Some("common_epoch_complete_support")
            && mask_header.nodata == Some(0.0),
        "fixed-cube validity mask metadata is invalid"
    );
    let los_paths = receipt
        .los_rasters
        .each_ref()
        .map(|name| directory.join(name));
    for path in &los_paths {
        let los_header = validate_fixed_cube_grid(path, velocity_header)?;
        validate_single_band_type(path, gdal::raster::GdalDataType::Float32)?;
        ensure!(
            los_header
                .metadata
                .get("GEOMETRY_SOURCE")
                .map(String::as_str)
                == Some("CSLC-S1-STATIC")
                && los_header
                    .metadata
                    .get("LOS_SIGN_CONVENTION")
                    .map(String::as_str)
                    == Some("ground_to_sensor_positive_toward_sensor")
                && los_header
                    .metadata
                    .get("LOS_COMPONENTS")
                    .map(String::as_str)
                    == Some("east,north,up")
                && los_header.metadata.get("RASTER_ROLE").map(String::as_str)
                    == Some("fixed_cube_run_geometry"),
            "fixed-cube LOS metadata is invalid: {}",
            path.display()
        );
    }
    let mut semantics = validate_fixed_cube_pixels(
        &mask_path,
        &los_paths,
        velocity_header.shape,
        receipt.valid_pixels,
    )?;
    validate_geometry_provenance(&directory.join(&receipt.geometry_provenance))?;
    semantics.geometry_provenance_status = "sourced_no_fallback".to_owned();
    Ok(semantics)
}

fn validate_fixed_cube_grid(
    path: &Path,
    expected: &dolphin_io::RasterHeader,
) -> Result<dolphin_io::RasterHeader> {
    let header = read_raster_header(path)?;
    ensure!(
        header.shape == expected.shape
            && header.geotransform == expected.geotransform
            && header.epsg == expected.epsg,
        "fixed-cube raster grid differs from velocity.tif: {}",
        path.display()
    );
    Ok(header)
}

fn validate_single_band_type(path: &Path, expected: gdal::raster::GdalDataType) -> Result<()> {
    let dataset = gdal::Dataset::open(path)?;
    ensure!(
        dataset.raster_count() == 1 && dataset.rasterband(1)?.band_type() == expected,
        "fixed-cube raster has invalid band count or type: {}",
        path.display()
    );
    Ok(())
}

fn validate_fixed_cube_pixels(
    mask_path: &Path,
    los_paths: &[PathBuf; 3],
    shape: (usize, usize),
    expected_valid_pixels: usize,
) -> Result<crate::fixed_cube::FixedCubeSemanticValidation> {
    let mut valid_pixels = 0usize;
    let mut maximum_los_norm_error = 0.0_f32;
    let mut minimum_los_up = f32::INFINITY;
    let bytes_per_pixel = std::mem::size_of::<u8>() + 3 * std::mem::size_of::<f32>();
    let maximum_pixels = usize::try_from(COMBINED_WORKING_SET_CAP_BYTES)?
        .checked_div(bytes_per_pixel)
        .context("fixed-cube semantic byte cap is too small")?
        .min(65_536);
    for row_start in (0..shape.0).step_by(256) {
        let row_stop = (row_start + 256).min(shape.0);
        let rows = row_stop - row_start;
        let columns_per_window = (maximum_pixels / rows).max(1);
        for col_start in (0..shape.1).step_by(columns_per_window) {
            let col_stop = (col_start + columns_per_window).min(shape.1);
            let window = BlockIndices {
                row_start,
                row_stop,
                col_start,
                col_stop,
            };
            let mask = read_raster_window::<u8>(mask_path, window)?;
            let east = read_raster_window::<f32>(&los_paths[0], window)?;
            let north = read_raster_window::<f32>(&los_paths[1], window)?;
            let up = read_raster_window::<f32>(&los_paths[2], window)?;
            for (((mask, east), north), up) in mask.iter().zip(&east).zip(&north).zip(&up) {
                ensure!(
                    *mask <= 1,
                    "fixed-cube validity mask contains a value other than 0 or 1"
                );
                valid_pixels = valid_pixels
                    .checked_add(usize::from(*mask))
                    .context("fixed-cube valid-pixel count overflow")?;
                let norm_error = (east.mul_add(*east, north.mul_add(*north, up * up)) - 1.0).abs();
                ensure!(
                    east.is_finite()
                        && north.is_finite()
                        && up.is_finite()
                        && *up > 0.0
                        && norm_error <= 5e-4,
                    "fixed-cube LOS vector is nonfinite, non-unit, or sign-inconsistent"
                );
                maximum_los_norm_error = maximum_los_norm_error.max(norm_error);
                minimum_los_up = minimum_los_up.min(*up);
            }
        }
    }
    ensure!(
        valid_pixels == expected_valid_pixels && minimum_los_up.is_finite(),
        "fixed-cube validity count differs from receipt"
    );
    Ok(crate::fixed_cube::FixedCubeSemanticValidation {
        observed_valid_pixels: valid_pixels,
        maximum_los_norm_error,
        minimum_los_up,
        los_sign_convention: "ground_to_sensor_positive_toward_sensor".to_owned(),
        geometry_source: "CSLC-S1-STATIC".to_owned(),
        geometry_provenance_status: "pending".to_owned(),
    })
}

fn validate_geometry_provenance(path: &Path) -> Result<()> {
    let provenance: Value = serde_json::from_slice(&read_bounded(path, 1024 * 1024)?)?;
    let incidence = &provenance["geometry_provenance"]["fields"]["incidence_angle_deg"];
    ensure!(
        provenance["schema"].as_str() == Some("dolphinrust-geometry-provenance/4")
            && provenance["incidence_angle_deg"].as_f64().is_some()
            && incidence["status"].as_str() == Some("sourced")
            && incidence["source_files"]
                .as_array()
                .is_some_and(|values| !values.is_empty())
            && incidence["source_keys"].as_array().is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some("/data/los_east"))
                    && values
                        .iter()
                        .any(|value| value.as_str() == Some("/data/los_north"))
            }),
        "geometry provenance is incomplete, unsourced, or fallback-derived"
    );
    Ok(())
}

fn fixed_cube_input_paths(
    directory: &Path,
    receipt: &crate::fixed_cube::FixedCubeReceipt,
) -> Vec<PathBuf> {
    let mut paths = vec![
        directory.join("fixed_cube_receipt.json"),
        directory.join(&receipt.velocity_raster),
        directory.join(&receipt.validity_mask_raster),
        directory.join(&receipt.geometry_provenance),
    ];
    if let Some(sigma) = &receipt.velocity_sigma_raster {
        paths.push(directory.join(sigma));
    }
    paths.extend(receipt.los_rasters.iter().map(|name| directory.join(name)));
    paths
}

struct ProductLayer {
    name: &'static str,
    writer: Option<BoundedCogWriter<f32>>,
}

const PRODUCT_LAYERS: [(&str, &str); LAYER_COUNT] = [
    ("velocity_temporal_gls.tif", "selected_velocity"),
    ("velocity_sigma_corrected.tif", "corrected_standard_error"),
    ("velocity_temporal_inference_status.tif", "selection_status"),
    ("velocity_temporal_fit_status.tif", "fit_status"),
    ("velocity_temporal_cadence_status.tif", "cadence_status"),
    ("velocity_temporal_valid_date_count.tif", "valid_date_count"),
    ("velocity_temporal_rank.tif", "rank"),
    ("velocity_temporal_dof.tif", "degrees_of_freedom"),
    ("velocity_temporal_raw_rho.tif", "raw_adjacent_residual_rho"),
    ("velocity_temporal_fitted_rho.tif", "fitted_rho_12_days"),
    (
        "velocity_temporal_process_variance.tif",
        "fitted_process_variance",
    ),
    ("velocity_temporal_condition_number.tif", "condition_number"),
    (
        "velocity_temporal_bootstrap_attempts.tif",
        "bootstrap_attempts",
    ),
    (
        "velocity_temporal_bootstrap_successes.tif",
        "bootstrap_successes",
    ),
];

fn create_layer_writers(
    stage: &Path,
    header: &dolphin_io::RasterHeader,
) -> Result<Vec<ProductLayer>> {
    let velocity_unit = header
        .metadata
        .get("UNITTYPE")
        .context("legacy velocity raster is missing UNITTYPE")?;
    PRODUCT_LAYERS
        .iter()
        .enumerate()
        .map(|(index, (name, role))| {
            let scratch = stage.join(format!("{name}.scratch.tif"));
            let unit = if index < 2 {
                velocity_unit.as_str()
            } else {
                "1"
            };
            let writer = BoundedCogWriter::create(
                &scratch,
                header.shape,
                header.geotransform,
                header.epsg,
                Some(f64::NAN),
                &[
                    ("PRODUCT_ROLE", *role),
                    ("UNITTYPE", unit),
                    ("TEMPORAL_ESTIMATOR", COMPLETE_REFIT_BOOTSTRAP_METHOD),
                    ("CALIBRATION_STATUS", "calibrated_scope_match"),
                    ("NODATA_POLICY", "per_pixel_abstention"),
                ],
            )?;
            Ok(ProductLayer {
                name,
                writer: Some(writer),
            })
        })
        .collect()
}

fn finalize_layers(stage: &Path, layers: &mut [ProductLayer]) -> Result<()> {
    for layer in layers {
        let writer = layer
            .writer
            .take()
            .context("product writer already finalized")?;
        writer.finalize(&stage.join(layer.name))?;
    }
    Ok(())
}

fn publish_layers_no_replace(output_directory: &Path, stage: &Path) -> Result<()> {
    for (name, _) in PRODUCT_LAYERS {
        install_no_replace(&stage.join(name), &output_directory.join(name))?;
    }
    Ok(())
}

fn install_no_replace(source: &Path, destination: &Path) -> Result<()> {
    ensure!(
        !destination.exists(),
        "refusing to replace existing temporal product {}",
        destination.display()
    );
    std::fs::hard_link(source, destination)?;
    std::fs::remove_file(source)?;
    File::open(
        destination
            .parent()
            .context("temporal product destination has no parent")?,
    )?
    .sync_all()?;
    Ok(())
}

fn evaluate_block(
    block: &dolphin_io::SpatialReferenceCovarianceBlock,
    observations: &[Array2<f32>],
    acquisition_days: &[f64],
    options: &TemporalCovarianceOptions,
) -> Result<[Array2<f32>; LAYER_COUNT]> {
    let shape = (
        usize::try_from(block.target_grid.rows)?,
        usize::try_from(block.target_grid.cols)?,
    );
    let target_count = shape.0 * shape.1;
    ensure!(
        observations
            .len()
            .checked_add(1)
            .is_some_and(|count| count == acquisition_days.len())
            && observations.iter().all(|values| values.dim() == shape),
        "displacement windows differ from factor block"
    );
    let mut output: [Array2<f32>; LAYER_COUNT] =
        std::array::from_fn(|_| Array2::from_elem(shape, f32::NAN));
    for target in 0..target_count {
        if block.status[target] != SpatialReferenceCovarianceStatus::Valid {
            output[2]
                .as_slice_mut()
                .context("status layer is not contiguous")?[target] =
                1_000.0 + block.status[target] as u16 as f32;
            continue;
        }
        let rank = usize::try_from(block.rank_by_target[target])?;
        let maximum_rank = usize::try_from(block.maximum_rank)?;
        let covariance = reconstruct_covariance(
            &block.difference_factor,
            target,
            acquisition_days.len(),
            maximum_rank,
            rank,
        )?;
        let row = target / shape.1;
        let col = target % shape.1;
        let mut series = Vec::with_capacity(acquisition_days.len());
        series.push(0.0);
        series.extend(
            observations
                .iter()
                .map(|values| f64::from(values[(row, col)])),
        );
        let fit = fit_temporal_covariance(acquisition_days, &series, &covariance, options);
        let selected = complete_refit_bootstrap_estimate(&fit, options);
        write_target_diagnostics(&mut output, target, &selected)?;
    }
    Ok(output)
}

fn reconstruct_covariance(
    factor: &[f64],
    target: usize,
    date_count: usize,
    maximum_rank: usize,
    rank: usize,
) -> Result<Vec<Vec<f64>>> {
    ensure!(rank <= maximum_rank, "valid factor target has invalid rank");
    let target_offset = target
        .checked_mul(date_count)
        .and_then(|value| value.checked_mul(maximum_rank))
        .context("factor target offset overflow")?;
    ensure!(
        target_offset + date_count * maximum_rank <= factor.len(),
        "factor target offset exceeds payload"
    );
    Ok((0..date_count)
        .map(|left| {
            (0..date_count)
                .map(|right| {
                    (0..rank)
                        .map(|component| {
                            factor[target_offset + left * maximum_rank + component]
                                * factor[target_offset + right * maximum_rank + component]
                        })
                        .sum()
                })
                .collect()
        })
        .collect())
}

fn write_target_diagnostics(
    layers: &mut [Array2<f32>; LAYER_COUNT],
    target: usize,
    selected: &CompleteRefitBootstrapEstimate,
) -> Result<()> {
    let set = |layer: &mut Array2<f32>, value: f32| -> Result<()> {
        layer
            .as_slice_mut()
            .context("temporal product layer is not contiguous")?[target] = value;
        Ok(())
    };
    if selected.status == CompleteRefitBootstrapEstimateStatus::Evaluated {
        set(
            &mut layers[0],
            selected
                .slope_per_year
                .context("evaluated slope is absent")? as f32,
        )?;
        set(
            &mut layers[1],
            selected
                .standard_error_per_year
                .context("evaluated standard error is absent")? as f32,
        )?;
    }
    set(&mut layers[2], estimate_status_code(selected.status) as f32)?;
    set(
        &mut layers[3],
        inference_status_code(selected.fit_status) as f32,
    )?;
    set(
        &mut layers[4],
        cadence_status_code(selected.cadence_status) as f32,
    )?;
    set(&mut layers[5], selected.valid_date_count as f32)?;
    set(&mut layers[6], selected.rank as f32)?;
    set(&mut layers[7], selected.degrees_of_freedom as f32)?;
    set_optional(&mut layers[8], target, selected.raw_rho)?;
    set_optional(&mut layers[9], target, selected.fitted_rho)?;
    set_optional(&mut layers[10], target, selected.fitted_process_variance)?;
    set_optional(&mut layers[11], target, selected.condition_number)?;
    set(&mut layers[12], selected.bootstrap_attempts as f32)?;
    set(&mut layers[13], selected.bootstrap_successes as f32)?;
    Ok(())
}

fn set_optional(layer: &mut Array2<f32>, target: usize, value: Option<f64>) -> Result<()> {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        layer
            .as_slice_mut()
            .context("temporal diagnostic layer is not contiguous")?[target] = value as f32;
    }
    Ok(())
}

fn estimate_status_code(status: CompleteRefitBootstrapEstimateStatus) -> u16 {
    match status {
        CompleteRefitBootstrapEstimateStatus::Evaluated => 0,
        CompleteRefitBootstrapEstimateStatus::FitNotEvaluated => 1,
        CompleteRefitBootstrapEstimateStatus::ComparatorNotEvaluated => 2,
        CompleteRefitBootstrapEstimateStatus::FrozenConfigurationMismatch => 3,
        CompleteRefitBootstrapEstimateStatus::BootstrapAccountingMismatch => 4,
        CompleteRefitBootstrapEstimateStatus::BootstrapInsufficientSuccess => 5,
        CompleteRefitBootstrapEstimateStatus::InvalidEstimate => 6,
    }
}

fn cadence_status_code(status: CompleteRefitBootstrapCadenceStatus) -> u16 {
    match status {
        CompleteRefitBootstrapCadenceStatus::Supported => 0,
        CompleteRefitBootstrapCadenceStatus::Unsupported => 1,
        CompleteRefitBootstrapCadenceStatus::Unavailable => 2,
    }
}

fn inference_status_code(status: TemporalInferenceStatus) -> u16 {
    match status {
        TemporalInferenceStatus::Evaluated => 0,
        TemporalInferenceStatus::InsufficientDates => 1,
        TemporalInferenceStatus::DatesNotStrictlyIncreasing => 2,
        TemporalInferenceStatus::GaugeMissing => 3,
        TemporalInferenceStatus::GaugeNotZero => 4,
        TemporalInferenceStatus::DesignRankDeficient => 5,
        TemporalInferenceStatus::DesignIllConditioned => 6,
        TemporalInferenceStatus::CovarianceNonfinite => 7,
        TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite => 8,
        TemporalInferenceStatus::CovarianceParameterAtBoundary => 9,
        TemporalInferenceStatus::RhoLowerBoundary => 10,
        TemporalInferenceStatus::RhoUpperBoundary => 11,
        TemporalInferenceStatus::ProcessVarianceLowerBoundary => 12,
        TemporalInferenceStatus::ProcessVarianceUpperBoundary => 13,
        TemporalInferenceStatus::BootstrapInsufficientSuccess => 14,
        TemporalInferenceStatus::UnsupportedCadence => 15,
        TemporalInferenceStatus::OptimizerNonconverged => 16,
        TemporalInferenceStatus::WeakParameterIdentification => 17,
        TemporalInferenceStatus::LegacyNonComparable => 18,
    }
}

fn output_window(
    target: dolphin_io::CovarianceOperatorGrid,
    full: dolphin_io::CovarianceOperatorGrid,
) -> Result<BlockIndices> {
    ensure!(
        target.stride_y == full.stride_y && target.stride_x == full.stride_x,
        "factor target block stride differs from full grid"
    );
    let row_delta = target
        .row_start
        .checked_sub(full.row_start)
        .context("factor target row precedes full grid")?;
    let col_delta = target
        .col_start
        .checked_sub(full.col_start)
        .context("factor target column precedes full grid")?;
    ensure!(
        row_delta % u64::from(full.stride_y) == 0 && col_delta % u64::from(full.stride_x) == 0,
        "factor target block is not aligned to the full grid"
    );
    let row_start = usize::try_from(row_delta / u64::from(full.stride_y))?;
    let col_start = usize::try_from(col_delta / u64::from(full.stride_x))?;
    let row_stop = row_start + usize::try_from(target.rows)?;
    let col_stop = col_start + usize::try_from(target.cols)?;
    ensure!(
        row_stop <= usize::try_from(full.rows)? && col_stop <= usize::try_from(full.cols)?,
        "factor target block exceeds full output grid"
    );
    Ok(BlockIndices {
        row_start,
        row_stop,
        col_start,
        col_stop,
    })
}

fn validate_input_rasters(
    paths: &[PathBuf],
    expected: &dolphin_io::RasterHeader,
    acquisition_days: &[f64],
) -> Result<()> {
    ensure!(
        !paths.is_empty() && acquisition_days.first().copied() == Some(0.0),
        "corrected temporal inference requires acquisition-zero displacement"
    );
    for path in paths {
        let header = read_raster_header(path)?;
        ensure!(
            header.shape == expected.shape
                && header.geotransform == expected.geotransform
                && header.epsg == expected.epsg
                && header.nodata.is_none_or(f64::is_nan),
            "displacement raster grid differs from velocity/factor grid: {}",
            path.display()
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct TemporalInferenceProvenance<'a> {
    schema: &'static str,
    calibration_scope: &'static str,
    estimator: &'static str,
    estimator_version: u16,
    acquisition_days: &'a [f64],
    acquisition_days_sha256: String,
    displacement_rasters: Vec<InputRasterReceipt>,
    fixed_cube_inputs: Vec<InputRasterReceipt>,
    fixed_cube_semantics: crate::fixed_cube::FixedCubeSemanticValidation,
    rows: usize,
    cols: usize,
    geotransform: [f64; 6],
    epsg: Option<u32>,
    velocity_unit: String,
    spatial_burst_id: &'a str,
    spatial_units: &'a str,
    spatial_reference_row: u64,
    spatial_reference_col: u64,
    spatial_method: &'a str,
    spatial_method_version: u16,
    spatial_source_replay_sha256: &'a str,
    spatial_l2_map_sha256: &'a str,
    spatial_reference_signature_sha256: &'a str,
    spatial_mask_sha256: &'a str,
    spatial_approximation_receipt_sha256: &'a str,
    spatial_resource_receipt_sha256: &'a str,
    spatial_review_receipt_sha256: &'a str,
    spatial_method_manifest_sha256: &'a str,
    spatial_support_method: &'a str,
    spatial_support_sha256: &'a str,
    spatial_correction_order_sha256: &'a str,
    spatial_unwrap_branch_sha256: &'a str,
    spatial_burst_ownership_sha256: &'a str,
    spatial_manifest_sha256: &'a str,
    synthetic_result_sha256: &'a str,
    heldout_result_sha256: &'a str,
    review_receipt_sha256: &'a str,
    promotion_manifest_sha256: &'a str,
    corrected_velocity_sha256: &'a str,
    corrected_sigma_sha256: &'a str,
    bootstrap_attempts: usize,
    bootstrap_minimum_successes: usize,
    nodata_policy: &'static str,
    inference_status_map: &'static str,
    cadence_status_map: &'static str,
    product_files: Vec<&'static str>,
    product_receipts: Vec<InputRasterReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct InputRasterReceipt {
    path: String,
    sha256: String,
}

impl<'a> TemporalInferenceProvenance<'a> {
    fn new(
        days: &'a [f64],
        scope: &'a ProductScope,
        promotion: &'a TemporalCovariancePromotion,
        velocity_sha256: &'a str,
        sigma_sha256: &'a str,
        product_receipts: Vec<InputRasterReceipt>,
    ) -> Self {
        let header = &scope.velocity_header;
        let factor = &scope.factor_metadata;
        let day_bytes = days
            .iter()
            .flat_map(|day| day.to_le_bytes())
            .collect::<Vec<_>>();
        Self {
            schema: PRODUCT_SCHEMA,
            calibration_scope: "calibrated_scope_match",
            estimator: COMPLETE_REFIT_BOOTSTRAP_METHOD,
            estimator_version: COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
            acquisition_days: days,
            acquisition_days_sha256: sha256(&day_bytes),
            displacement_rasters: scope.input_receipts.clone(),
            fixed_cube_inputs: scope.fixed_cube_inputs.clone(),
            fixed_cube_semantics: scope.fixed_cube_semantics.clone(),
            rows: header.shape.0,
            cols: header.shape.1,
            geotransform: header.geotransform,
            epsg: header.epsg,
            velocity_unit: scope.velocity_unit.clone(),
            spatial_burst_id: &factor.burst_id,
            spatial_units: &factor.units,
            spatial_reference_row: factor.reference_row,
            spatial_reference_col: factor.reference_col,
            spatial_method: &factor.method,
            spatial_method_version: factor.method_version,
            spatial_source_replay_sha256: &factor.source_replay_digest,
            spatial_l2_map_sha256: &factor.l2_map_digest,
            spatial_reference_signature_sha256: &factor.reference_signature_digest,
            spatial_mask_sha256: &factor.mask_digest,
            spatial_approximation_receipt_sha256: &factor.approximation_receipt_digest,
            spatial_resource_receipt_sha256: &factor.resource_receipt_digest,
            spatial_review_receipt_sha256: &factor.review_receipt_digest,
            spatial_method_manifest_sha256: &factor.method_manifest_digest,
            spatial_support_method: &factor.support_method,
            spatial_support_sha256: &factor.support_digest,
            spatial_correction_order_sha256: &factor.correction_order_digest,
            spatial_unwrap_branch_sha256: &factor.unwrap_branch_digest,
            spatial_burst_ownership_sha256: &factor.burst_ownership_digest,
            spatial_manifest_sha256: &promotion.spatial_manifest_sha256,
            synthetic_result_sha256: &promotion.synthetic_sha256,
            heldout_result_sha256: &promotion.heldout_sha256,
            review_receipt_sha256: &promotion.review_sha256,
            promotion_manifest_sha256: &promotion.manifest_sha256,
            corrected_velocity_sha256: velocity_sha256,
            corrected_sigma_sha256: sigma_sha256,
            bootstrap_attempts: 200,
            bootstrap_minimum_successes: 198,
            nodata_policy: "per_pixel_abstention_no_fallback",
            inference_status_map: "0=evaluated;1=fit_not_evaluated;2=comparator_not_evaluated;3=frozen_configuration_mismatch;4=bootstrap_accounting_mismatch;5=bootstrap_insufficient_success;6=invalid_estimate;factor_failures=1000+spatial_status_code",
            cadence_status_map: "0=supported;1=unsupported;2=unavailable",
            product_files: PRODUCT_LAYERS.iter().map(|(name, _)| *name).collect(),
            product_receipts,
        }
    }
}

fn input_raster_receipts(paths: &[PathBuf]) -> Result<Vec<InputRasterReceipt>> {
    paths
        .iter()
        .map(|path| {
            Ok(InputRasterReceipt {
                path: path.display().to_string(),
                sha256: sha256_file(path)?,
            })
        })
        .collect()
}

fn product_receipts(directory: &Path) -> Result<Vec<InputRasterReceipt>> {
    input_raster_receipts(
        &PRODUCT_LAYERS
            .iter()
            .map(|(name, _)| directory.join(name))
            .collect::<Vec<_>>(),
    )
}

fn verify_final_cogs(
    directory: &Path,
    expected: &[InputRasterReceipt],
    grid: &dolphin_io::RasterHeader,
) -> Result<()> {
    ensure!(
        expected.len() == PRODUCT_LAYERS.len(),
        "temporal product receipt count is incomplete"
    );
    for ((name, role), expected_receipt) in PRODUCT_LAYERS.iter().zip(expected) {
        let path = directory.join(name);
        ensure!(
            sha256_file(&path)? == expected_receipt.sha256,
            "published temporal product hash changed: {name}"
        );
        let header = read_raster_header(&path)?;
        ensure!(
            header.shape == grid.shape
                && header.geotransform == grid.geotransform
                && header.epsg == grid.epsg
                && header.nodata.is_some_and(f64::is_nan)
                && header.metadata.get("PRODUCT_ROLE").map(String::as_str) == Some(*role)
                && header
                    .metadata
                    .get("TEMPORAL_ESTIMATOR")
                    .map(String::as_str)
                    == Some(COMPLETE_REFIT_BOOTSTRAP_METHOD),
            "published temporal product header changed: {name}"
        );
    }
    Ok(())
}

fn verify_promoted_fixed_cube_receipt(
    directory: &Path,
    velocity_sha256: &str,
    sigma_sha256: &str,
    provenance_sha256: &str,
    promotion_sha256: &str,
    semantic_validation: &crate::fixed_cube::FixedCubeSemanticValidation,
) -> Result<()> {
    let receipt: crate::fixed_cube::FixedCubeReceipt = serde_json::from_slice(&read_bounded(
        &directory.join("fixed_cube_receipt.json"),
        1024 * 1024,
    )?)?;
    ensure!(
        receipt.inference_status == "calibrated_scope_match"
            && receipt.corrected_velocity_raster.as_deref() == Some("velocity_temporal_gls.tif")
            && receipt.corrected_sigma_raster.as_deref() == Some("velocity_sigma_corrected.tif")
            && receipt.corrected_velocity_sha256.as_deref() == Some(velocity_sha256)
            && receipt.corrected_sigma_sha256.as_deref() == Some(sigma_sha256)
            && receipt.inference_provenance.as_deref()
                == Some(TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            && receipt.inference_provenance_sha256.as_deref() == Some(provenance_sha256)
            && receipt.temporal_promotion_manifest_sha256.as_deref() == Some(promotion_sha256)
            && receipt.semantic_validation.as_ref() == Some(semantic_validation),
        "fixed-cube receipt changed before temporal completion marker"
    );
    Ok(())
}

fn write_rollback_journal(directory: &Path, original_receipt: &[u8]) -> Result<()> {
    let journal = ProductRollbackJournal {
        schema: ROLLBACK_JOURNAL_SCHEMA.to_owned(),
        original_fixed_cube_receipt: original_receipt.to_vec(),
    };
    let scratch = directory.join(format!(
        ".temporal-product-journal-{}-{}",
        std::process::id(),
        NEXT_TRANSACTION_FILE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&scratch, serde_json::to_vec(&journal)?)?;
    File::open(&scratch)?.sync_all()?;
    install_no_replace(&scratch, &directory.join(ROLLBACK_JOURNAL_FILENAME))
}

fn remove_rollback_journal(directory: &Path) -> Result<()> {
    let path = directory.join(ROLLBACK_JOURNAL_FILENAME);
    if path.exists() {
        std::fs::remove_file(path)?;
        File::open(directory)?.sync_all()?;
    }
    Ok(())
}

fn recover_incomplete_product(directory: &Path) -> Result<()> {
    let journal_path = directory.join(ROLLBACK_JOURNAL_FILENAME);
    if journal_path.exists() {
        let journal: ProductRollbackJournal =
            serde_json::from_slice(&read_bounded(&journal_path, JSON_CAP)?)?;
        ensure!(
            journal.schema == ROLLBACK_JOURNAL_SCHEMA,
            "temporal product rollback journal schema is unsupported"
        );
        if completed_bundle_marker_is_valid(directory)? {
            remove_rollback_journal(directory)?;
        } else {
            remove_published_products(directory)?;
            restore_fixed_cube_receipt(directory, &journal.original_fixed_cube_receipt)?;
            remove_rollback_journal(directory)?;
        }
    }
    remove_stale_stage_directories(directory)
}

fn completed_bundle_marker_is_valid(directory: &Path) -> Result<bool> {
    let marker = directory.join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME);
    if !marker.exists()
        || PRODUCT_LAYERS
            .iter()
            .any(|(name, _)| !directory.join(name).exists())
    {
        return Ok(false);
    }
    let provenance: Value = match serde_json::from_slice(&read_bounded(&marker, JSON_CAP)?) {
        Ok(value) => value,
        Err(_) => return Ok(false),
    };
    let receipt: Value = match serde_json::from_slice(&read_bounded(
        &directory.join("fixed_cube_receipt.json"),
        1024 * 1024,
    )?) {
        Ok(value) => value,
        Err(_) => return Ok(false),
    };
    Ok(provenance["schema"].as_str() == Some(PRODUCT_SCHEMA)
        && receipt["inference_status"].as_str() == Some("calibrated_scope_match"))
}

fn remove_stale_stage_directories(directory: &Path) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(".temporal-inference-stage-"))
        {
            std::fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

fn validate_no_existing_products(directory: &Path) -> Result<()> {
    for (name, _) in PRODUCT_LAYERS {
        ensure!(
            !directory.join(name).exists(),
            "corrected temporal product already exists: {name}"
        );
    }
    ensure!(
        !directory
            .join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            .exists(),
        "temporal inference provenance already exists"
    );
    Ok(())
}

fn remove_published_products(directory: &Path) -> Result<()> {
    for (name, _) in PRODUCT_LAYERS {
        let path = directory.join(name);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
    }
    let provenance = directory.join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME);
    if provenance.exists() {
        std::fs::remove_file(&provenance)?;
    }
    validate_no_existing_products(directory)
}

fn restore_fixed_cube_receipt(directory: &Path, receipt: &[u8]) -> Result<()> {
    static NEXT_ROLLBACK_ID: AtomicU64 = AtomicU64::new(0);
    let scratch = directory.join(format!(
        ".fixed-cube-receipt-rollback-{}-{}",
        std::process::id(),
        NEXT_ROLLBACK_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let restore = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&scratch)?;
        std::io::Write::write_all(&mut file, receipt)?;
        file.sync_all()?;
        std::fs::rename(&scratch, directory.join("fixed_cube_receipt.json"))?;
        File::open(directory)?.sync_all()?;
        Ok(())
    })();
    if restore.is_err() {
        let _ = std::fs::remove_file(&scratch);
    }
    restore
}

fn create_stage_directory(directory: &Path) -> Result<PathBuf> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let stage = directory.join(format!(
        ".temporal-inference-stage-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&stage)?;
    Ok(stage)
}

fn read_bounded(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let before = std::fs::metadata(path)?;
    ensure!(before.len() <= cap, "{} exceeds byte cap", path.display());
    let mut file = File::open(path)?;
    let opened = file.metadata()?;
    let mut bytes = Vec::with_capacity(usize::try_from(before.len())?);
    let read_cap = cap.checked_add(1).context("JSON read cap overflow")?;
    file.by_ref().take(read_cap).read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    ensure!(
        before.len() == opened.len()
            && opened.len() == after.len()
            && before.modified()? == opened.modified()?
            && opened.modified()? == after.modified()?,
        "{} changed while it was read",
        path.display()
    );
    ensure!(
        bytes.len() as u64 <= cap,
        "{} exceeds byte cap",
        path.display()
    );
    Ok(bytes)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        admit_combined_working_set, complete_publication_after_legacy_check,
        ensure_same_run_factor_directory, install_no_replace, output_window,
        reconstruct_covariance, validate_heldout_result, validate_manifest, validate_review,
        validate_synthetic_result, write_product_transaction_with_validator,
        write_rollback_journal, EvidenceDigests, HeldoutLevel, HeldoutResult, SyntheticResult,
        SyntheticScores, TemporalCovariancePromotion, TemporalProductTransaction,
        TemporalPromotionManifest, TemporalReviewReceipt, PRODUCT_LAYERS, PROMOTION_SCHEMA,
        REVIEW_SCHEMA, ROLLBACK_JOURNAL_FILENAME, SYNTHETIC_SCHEMA,
    };
    use dolphin_core::config::{
        DisplacementWorkflow, TemporalUncertaintyMethod, TemporalUncertaintyOptions,
    };
    use dolphin_io::{
        spatial_reference_calibration_scope_digest, spatial_reference_effective_looks_digest,
        spatial_reference_runtime_resource_receipt_digest, write_raster,
        write_spatial_reference_covariance, CovarianceOperatorGrid,
        SpatialReferenceCalibrationScope, SpatialReferenceCovarianceBlock,
        SpatialReferenceCovarianceMetadata, SpatialReferenceCovarianceStatus,
        SpatialReferenceRuntimeResourceReceipt, SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE,
        SPATIAL_REFERENCE_COVARIANCE_METHOD, SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
        SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
    };
    use dolphin_timeseries::{
        COMPLETE_REFIT_BOOTSTRAP_METHOD, COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
    };
    use ndarray::{array, Array2};
    use serde_json::Value;

    #[test]
    fn reconstructs_covariance_from_target_major_factor() {
        let factor = vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let covariance = reconstruct_covariance(&factor, 0, 3, 2, 2).unwrap();
        assert_eq!(covariance[0], vec![0.0, 0.0, 0.0]);
        assert_eq!(covariance[1], vec![0.0, 5.0, 11.0]);
        assert_eq!(covariance[2], vec![0.0, 11.0, 25.0]);
    }

    #[test]
    fn maps_native_factor_coordinates_to_output_window() {
        let full = CovarianceOperatorGrid {
            row_start: 100,
            col_start: 200,
            rows: 10,
            cols: 12,
            stride_y: 2,
            stride_x: 3,
        };
        let target = CovarianceOperatorGrid {
            row_start: 104,
            col_start: 209,
            rows: 3,
            cols: 4,
            stride_y: 2,
            stride_x: 3,
        };
        let window = output_window(target, full).unwrap();
        assert_eq!((window.row_start, window.row_stop), (2, 5));
        assert_eq!((window.col_start, window.col_stop), (3, 7));
    }

    #[test]
    fn rejects_cross_run_factor_directory() {
        let root = std::env::temp_dir().join(format!(
            "dolphin_temporal_factor_scope_{}",
            std::process::id()
        ));
        let output = root.join("output");
        let foreign = root.join("foreign");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&output).unwrap();
        std::fs::create_dir_all(&foreign).unwrap();
        ensure_same_run_factor_directory(&output, &output).unwrap();
        assert!(ensure_same_run_factor_directory(&output, &foreign).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn combined_working_set_rejects_large_date_stacks_before_block_read() {
        let config = TemporalUncertaintyOptions::default();
        assert!(admit_combined_working_set(&config, 12).is_ok());
        assert!(admit_combined_working_set(&config, 100_000).is_err());
    }

    #[test]
    fn transaction_lock_is_exclusive_and_publication_never_replaces() {
        let directory =
            std::env::temp_dir().join(format!("dolphin_temporal_lock_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let first = TemporalProductTransaction::acquire(&directory).unwrap();
        assert!(TemporalProductTransaction::acquire(&directory).is_err());
        drop(first);
        let source = directory.join("source");
        let destination = directory.join("destination");
        std::fs::write(&source, b"new").unwrap();
        std::fs::write(&destination, b"existing").unwrap();
        assert!(install_no_replace(&source, &destination).is_err());
        assert_eq!(std::fs::read(destination).unwrap(), b"existing");
        let malformed = directory.join("malformed");
        std::fs::create_dir(&malformed).unwrap();
        std::fs::write(malformed.join(ROLLBACK_JOURNAL_FILENAME), b"{malformed").unwrap();
        std::fs::write(
            malformed.join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME),
            b"marker",
        )
        .unwrap();
        assert!(TemporalProductTransaction::acquire(&malformed).is_err());
        assert!(malformed.join(ROLLBACK_JOURNAL_FILENAME).exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_recovers_crash_after_each_product_install() {
        let root =
            std::env::temp_dir().join(format!("dolphin_temporal_recovery_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        let original = br#"{"inference_status":"conditional_only"}"#;
        for installed in 0..=PRODUCT_LAYERS.len() {
            let directory = root.join(installed.to_string());
            std::fs::create_dir(&directory).unwrap();
            std::fs::write(directory.join("fixed_cube_receipt.json"), original).unwrap();
            write_rollback_journal(&directory, original).unwrap();
            std::fs::write(
                directory.join("fixed_cube_receipt.json"),
                br#"{"inference_status":"calibrated_scope_match"}"#,
            )
            .unwrap();
            for (name, _) in PRODUCT_LAYERS.iter().take(installed) {
                std::fs::write(directory.join(name), b"partial").unwrap();
            }
            let transaction = TemporalProductTransaction::acquire(&directory).unwrap();
            assert!(PRODUCT_LAYERS
                .iter()
                .all(|(name, _)| !directory.join(name).exists()));
            assert_eq!(
                std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
                original
            );
            assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
            drop(transaction);
        }
        let completed = root.join("completed_marker");
        std::fs::create_dir(&completed).unwrap();
        std::fs::write(completed.join("fixed_cube_receipt.json"), original).unwrap();
        write_rollback_journal(&completed, original).unwrap();
        let promoted = br#"{"inference_status":"calibrated_scope_match"}"#;
        std::fs::write(completed.join("fixed_cube_receipt.json"), promoted).unwrap();
        for (name, _) in PRODUCT_LAYERS {
            std::fs::write(completed.join(name), b"complete").unwrap();
        }
        std::fs::write(
            completed.join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME),
            br#"{"schema":"dolphinrust-temporal-inference-product/1"}"#,
        )
        .unwrap();
        let transaction = TemporalProductTransaction::acquire(&completed).unwrap();
        assert!(PRODUCT_LAYERS
            .iter()
            .all(|(name, _)| completed.join(name).exists()));
        assert_eq!(
            std::fs::read(completed.join("fixed_cube_receipt.json")).unwrap(),
            promoted
        );
        assert!(!completed.join(ROLLBACK_JOURNAL_FILENAME).exists());
        drop(transaction);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn complete_evidence_chain_rejects_tamper_and_scope_mismatch() {
        let synthetic = SyntheticResult {
            schema: SYNTHETIC_SCHEMA.to_owned(),
            attempted_cells: 240_000,
            batch_attempted_cells: 240_000,
            seed_count: 5_000,
            execution_complete: true,
            exact_seed_denominator_complete: true,
            corrected_inferential_sigma_emission: false,
            promotion_eligible: true,
            promotion_status: "eligible_for_external_field_review".to_owned(),
            scores: SyntheticScores {
                all_methods_pass: true,
            },
            resource_gates: BTreeMap::from([("rss".to_owned(), true)]),
        };
        validate_synthetic_result(&synthetic).unwrap();
        let heldout = HeldoutResult {
            status: "pass".to_owned(),
            errors: Vec::<Value>::new(),
            levels: ["68", "90", "95"]
                .map(|level| {
                    (
                        level.to_owned(),
                        HeldoutLevel {
                            status: "pass".to_owned(),
                        },
                    )
                })
                .into_iter()
                .collect(),
            evaluated_clusters: 96,
            emission_rate: 0.99,
        };
        validate_heldout_result(&heldout).unwrap();
        let expected = EvidenceDigests {
            synthetic_result_sha256: "11".repeat(32),
            heldout_result_sha256: "22".repeat(32),
            spatial_factor_sha256: "33".repeat(32),
            spatial_manifest_sha256: "44".repeat(32),
            temporal_preregistration_sha256: "55".repeat(32),
            heldout_preregistration_sha256: "66".repeat(32),
            scorer_sha256: "77".repeat(32),
            source_sha256: "88".repeat(32),
        };
        let review = TemporalReviewReceipt {
            schema: REVIEW_SCHEMA.to_owned(),
            review_status: "approved".to_owned(),
            reviewer: "independent-reviewer".to_owned(),
            independent: true,
            unresolved_findings: 0,
            synthetic_result_sha256: expected.synthetic_result_sha256.clone(),
            heldout_result_sha256: expected.heldout_result_sha256.clone(),
            spatial_manifest_sha256: expected.spatial_manifest_sha256.clone(),
            temporal_preregistration_sha256: expected.temporal_preregistration_sha256.clone(),
            heldout_preregistration_sha256: expected.heldout_preregistration_sha256.clone(),
            scorer_sha256: expected.scorer_sha256.clone(),
            source_sha256: expected.source_sha256.clone(),
        };
        validate_review(&review, &expected).unwrap();
        let mut manifest = TemporalPromotionManifest {
            schema: PROMOTION_SCHEMA.to_owned(),
            promotion_status: "approved".to_owned(),
            calibration_scope: "calibrated_scope_match".to_owned(),
            selected_method: COMPLETE_REFIT_BOOTSTRAP_METHOD.to_owned(),
            selected_method_version: COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
            synthetic_result_sha256: expected.synthetic_result_sha256.clone(),
            heldout_result_sha256: expected.heldout_result_sha256.clone(),
            review_receipt_sha256: "99".repeat(32),
            spatial_factor_sha256: expected.spatial_factor_sha256.clone(),
            spatial_manifest_sha256: expected.spatial_manifest_sha256.clone(),
            temporal_preregistration_sha256: expected.temporal_preregistration_sha256.clone(),
            heldout_preregistration_sha256: expected.heldout_preregistration_sha256.clone(),
            scorer_sha256: expected.scorer_sha256.clone(),
            source_sha256: expected.source_sha256.clone(),
        };
        validate_manifest(&manifest, &expected, &"99".repeat(32)).unwrap();
        manifest.calibration_scope = "scope_mismatch".to_owned();
        assert!(validate_manifest(&manifest, &expected, &"99".repeat(32)).is_err());
        manifest.calibration_scope = "calibrated_scope_match".to_owned();
        manifest.synthetic_result_sha256 = "aa".repeat(32);
        assert!(validate_manifest(&manifest, &expected, &"99".repeat(32)).is_err());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cognitive_complexity)]
    fn bounded_transaction_abstains_and_promotes_receipt_without_touching_legacy() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_product_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("contract")
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let days = (0..12).map(|date| date as f64 * 12.0).collect::<Vec<_>>();
        let block = masked_block(days.len());
        let metadata = calibrated_metadata(&days, &block);
        write_spatial_reference_covariance(
            directory.join(super::SPATIAL_REFERENCE_COVARIANCE_FILENAME),
            &metadata,
            &[block],
        )
        .unwrap();
        let geotransform = metadata.geotransform.unwrap();
        let velocity = array![[1.0_f32], [2.0_f32]];
        dolphin_io::write_raster_with_metadata(
            &directory.join("velocity.tif"),
            velocity.view(),
            geotransform,
            Some(32611),
            None,
            &[
                ("UNITTYPE", "rad/yr"),
                ("VELOCITY_ESTIMATOR", "linear_post_gauge_unit_precision"),
            ],
        )
        .unwrap();
        let legacy_before = std::fs::read(directory.join("velocity.tif")).unwrap();
        write_raster(
            &directory.join("velocity_sigma.tif"),
            Array2::from_elem((2, 1), 0.25_f32).view(),
            geotransform,
            Some(32611),
            None,
        )
        .unwrap();
        let legacy_sigma_before = std::fs::read(directory.join("velocity_sigma.tif")).unwrap();
        let displacement_rasters = days
            .iter()
            .skip(1)
            .enumerate()
            .map(|(date, _)| {
                let path = directory.join(format!("displacement_{date:02}.tif"));
                write_raster(
                    &path,
                    Array2::from_elem((2, 1), date as f32).view(),
                    geotransform,
                    Some(32611),
                    None,
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let workflow = DisplacementWorkflow {
            work_directory: directory.clone(),
            ..DisplacementWorkflow::default()
        };
        let geometry = dolphin_corrections::LosGeometry {
            east: Array2::from_elem((2, 1), 0.5),
            north: Array2::from_elem((2, 1), 0.0),
            up: Array2::from_elem((2, 1), 3.0_f64.sqrt() / 2.0),
        };
        let fixed_cube = crate::fixed_cube::write_fixed_cube_bundle(
            &workflow,
            &days,
            crate::displacement::VelocityEstimator::LinearPostGaugeUnitPrecision,
            true,
            Array2::from_elem((2, 1), true).view(),
            &geometry,
            Some((0, 0)),
            Some(32611),
            geotransform,
        )
        .unwrap();
        let valid_geometry_provenance = br#"{
            "schema":"dolphinrust-geometry-provenance/4",
            "orbit_direction":"ascending",
            "incidence_angle_deg":30.0,
            "heading_deg":10.0,
            "decomposition_geometry_complete":true,
            "geometry_provenance":{"fields":{"incidence_angle_deg":{
                "status":"sourced",
                "source_files":["static.h5"],
                "source_keys":["/data/los_east","/data/los_north"],
                "method":"test"
            }}}
        }"#;
        std::fs::write(
            directory.join("geometry_provenance.json"),
            valid_geometry_provenance,
        )
        .unwrap();
        let velocity_header =
            dolphin_io::read_raster_header(&directory.join("velocity.tif")).unwrap();
        super::validate_fixed_cube_semantics(&directory, &velocity_header, &fixed_cube).unwrap();
        let mut wrong_count = fixed_cube.clone();
        wrong_count.valid_pixels += 1;
        assert!(
            super::validate_fixed_cube_semantics(&directory, &velocity_header, &wrong_count)
                .is_err()
        );
        let mut wrong_estimator = fixed_cube.clone();
        wrong_estimator.velocity_estimator = "different_estimator".to_owned();
        assert!(super::validate_fixed_cube_semantics(
            &directory,
            &velocity_header,
            &wrong_estimator
        )
        .is_err());
        std::fs::write(
            directory.join("geometry_provenance.json"),
            br#"{"schema":"dolphinrust-geometry-provenance/4","decomposition_geometry_complete":false}"#,
        )
        .unwrap();
        assert!(
            super::validate_fixed_cube_semantics(&directory, &velocity_header, &fixed_cube)
                .is_err()
        );
        std::fs::write(
            directory.join("geometry_provenance.json"),
            valid_geometry_provenance,
        )
        .unwrap();
        let mask_tags = [
            ("MASK_ROLE", "velocity_support"),
            ("MASK_VALUES", "0=invalid;1=valid"),
            ("MASK_POLICY", "common_epoch_complete_support"),
        ];
        dolphin_io::write_raster_with_metadata(
            &directory.join("velocity_validity_mask.tif"),
            Array2::from_elem((2, 1), 1.0_f32).view(),
            geotransform,
            Some(32611),
            Some(0.0),
            &mask_tags,
        )
        .unwrap();
        assert!(
            super::validate_fixed_cube_semantics(&directory, &velocity_header, &fixed_cube)
                .is_err()
        );
        dolphin_io::write_raster_with_metadata(
            &directory.join("velocity_validity_mask.tif"),
            Array2::from_elem((2, 1), 1_u8).view(),
            geotransform,
            Some(32611),
            Some(0.0),
            &mask_tags,
        )
        .unwrap();
        let los_tags = [
            ("GEOMETRY_SOURCE", "CSLC-S1-STATIC"),
            (
                "LOS_SIGN_CONVENTION",
                "ground_to_sensor_positive_toward_sensor",
            ),
            ("LOS_COMPONENTS", "east,north,up"),
            ("RASTER_ROLE", "fixed_cube_run_geometry"),
        ];
        dolphin_io::write_raster_with_metadata(
            &directory.join("los_east.tif"),
            Array2::from_elem((2, 1), 0.9_f32).view(),
            geotransform,
            Some(32611),
            None,
            &los_tags,
        )
        .unwrap();
        assert!(
            super::validate_fixed_cube_semantics(&directory, &velocity_header, &fixed_cube)
                .is_err()
        );
        dolphin_io::write_raster_with_metadata(
            &directory.join("los_east.tif"),
            geometry.east.mapv(|value| value as f32).view(),
            geotransform,
            Some(32611),
            None,
            &los_tags,
        )
        .unwrap();
        super::validate_fixed_cube_semantics(&directory, &velocity_header, &fixed_cube).unwrap();
        let fixed_cube_receipt_before =
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap();
        let legacy_velocity_sha256 = super::sha256_file(&directory.join("velocity.tif")).unwrap();
        let legacy_sigma_sha256 =
            super::sha256_file(&directory.join("velocity_sigma.tif")).unwrap();
        let promotion = TemporalCovariancePromotion {
            manifest_sha256: "11".repeat(32),
            review_sha256: "22".repeat(32),
            synthetic_sha256: "33".repeat(32),
            heldout_sha256: "44".repeat(32),
            spatial_manifest_sha256: "55".repeat(32),
            spatial_factor_sha256: "66".repeat(32),
        };
        let config = TemporalUncertaintyOptions {
            method: TemporalUncertaintyMethod::CompleteRefitBootstrap,
            evidence_directory: Some(directory.clone()),
            factor_directory: Some(directory.clone()),
            maximum_targets_per_block: 2,
            block_id_read_cap_bytes: 1024 * 1024,
            factor_block_read_cap_bytes: 1024 * 1024,
        };
        let product_transaction = TemporalProductTransaction::acquire(&directory).unwrap();
        let mut rejected = config.clone();
        rejected.maximum_targets_per_block = 1;
        assert!(write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &rejected,
            &directory,
            &promotion,
            &product_transaction,
            || Ok(promotion.clone()),
        )
        .is_err());
        assert!(!directory
            .join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            .exists());
        assert!(PRODUCT_LAYERS
            .iter()
            .all(|(name, _)| !directory.join(name).exists()));
        let conditional: crate::fixed_cube::FixedCubeReceipt = serde_json::from_slice(
            &std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(conditional.inference_status, "conditional_only");
        let mut changed_promotion = promotion.clone();
        changed_promotion.spatial_factor_sha256 = "77".repeat(32);
        assert!(write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            || {
                let finalized_stage_exists = std::fs::read_dir(&directory)?.any(|entry| {
                    entry.is_ok_and(|entry| {
                        entry
                            .file_name()
                            .to_str()
                            .is_some_and(|name| name.starts_with(".temporal-inference-stage-"))
                            && entry.path().join("velocity_temporal_gls.tif").exists()
                    })
                });
                anyhow::ensure!(
                    finalized_stage_exists,
                    "promotion revalidation ran before COG finalization"
                );
                Ok(changed_promotion.clone())
            },
        )
        .is_err());
        assert!(!directory
            .join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            .exists());
        assert!(PRODUCT_LAYERS
            .iter()
            .all(|(name, _)| !directory.join(name).exists()));
        let geometry_path = directory.join("geometry_provenance.json");
        let geometry_before = std::fs::read(&geometry_path).unwrap();
        let mut revalidation_calls = 0usize;
        assert!(write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            || {
                revalidation_calls += 1;
                if revalidation_calls == 2 {
                    std::fs::write(
                        &geometry_path,
                        br#"{"schema":"tampered_geometry_provenance"}"#,
                    )?;
                }
                Ok(promotion.clone())
            },
        )
        .is_err());
        assert_eq!(revalidation_calls, 2);
        assert!(PRODUCT_LAYERS
            .iter()
            .all(|(name, _)| !directory.join(name).exists()));
        assert!(!directory
            .join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            .exists());
        std::fs::write(&geometry_path, geometry_before).unwrap();
        let published_before_legacy_mutation = write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            || Ok(promotion.clone()),
        )
        .unwrap();
        std::fs::write(directory.join("velocity.tif"), b"mutated legacy velocity").unwrap();
        assert!(complete_publication_after_legacy_check(
            &directory,
            published_before_legacy_mutation,
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
            &fixed_cube_receipt_before,
        )
        .is_err());
        assert!(PRODUCT_LAYERS
            .iter()
            .all(|(name, _)| !directory.join(name).exists()));
        assert!(!directory
            .join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            .exists());
        assert_eq!(
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
            fixed_cube_receipt_before
        );
        std::fs::write(directory.join("velocity.tif"), &legacy_before).unwrap();
        let receipt = write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            || Ok(promotion.clone()),
        )
        .unwrap();
        assert_eq!(
            std::fs::read(directory.join("velocity.tif")).unwrap(),
            legacy_before
        );
        assert_eq!(
            std::fs::read(directory.join("velocity_sigma.tif")).unwrap(),
            legacy_sigma_before
        );
        for (name, _) in PRODUCT_LAYERS {
            assert!(directory.join(name).exists(), "missing {name}");
        }
        assert!(directory
            .join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            .exists());
        let provenance: Value = serde_json::from_slice(
            &std::fs::read(directory.join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            provenance["fixed_cube_semantics"]["observed_valid_pixels"],
            2
        );
        assert_eq!(
            provenance["fixed_cube_semantics"]["geometry_provenance_status"],
            "sourced_no_fallback"
        );
        assert_eq!(provenance["product_receipts"].as_array().unwrap().len(), 14);
        assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
        let fixed_cube_inputs = provenance["fixed_cube_inputs"].as_array().unwrap();
        assert_eq!(fixed_cube_inputs.len(), 8);
        let fixed_cube_names = fixed_cube_inputs
            .iter()
            .map(|entry| {
                std::path::Path::new(entry["path"].as_str().unwrap())
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            fixed_cube_names,
            [
                "fixed_cube_receipt.json",
                "velocity.tif",
                "velocity_validity_mask.tif",
                "geometry_provenance.json",
                "velocity_sigma.tif",
                "los_east.tif",
                "los_north.tif",
                "los_up.tif",
            ]
        );
        let fixed: crate::fixed_cube::FixedCubeReceipt = serde_json::from_slice(
            &std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(fixed.inference_status, "calibrated_scope_match");
        assert_eq!(
            fixed
                .semantic_validation
                .as_ref()
                .unwrap()
                .observed_valid_pixels,
            2
        );
        assert_eq!(
            fixed.corrected_velocity_sha256,
            Some(receipt.corrected_velocity_sha256)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn masked_block(date_count: usize) -> SpatialReferenceCovarianceBlock {
        SpatialReferenceCovarianceBlock {
            block_id: 1,
            target_grid: CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: 2,
                cols: 1,
                stride_y: 1,
                stride_x: 1,
            },
            maximum_rank: 1,
            rank_by_target: vec![0, 0],
            status: vec![SpatialReferenceCovarianceStatus::MaskedTarget; 2],
            source_burst_index_by_target: vec![SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE; 2],
            difference_factor: vec![0.0; 2 * date_count],
            approximation_error_bound: vec![SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE; 2],
            effective_looks_fraction: Some(vec![f64::NAN; 2]),
            support_union_count: Some(vec![0; 2]),
            effective_looks_receipt: Some(vec![0; 64]),
            resource_high_water_bytes: Some(vec![0; 2]),
            condition_number: Some(vec![f64::NAN; 2]),
            source_factor_digest: strong_digest(0x71),
        }
    }

    fn calibrated_metadata(
        days: &[f64],
        block: &SpatialReferenceCovarianceBlock,
    ) -> SpatialReferenceCovarianceMetadata {
        let runtime = SpatialReferenceRuntimeResourceReceipt {
            working_set_byte_cap: 32_768,
            factor_block_high_water_bytes: 8_192,
            serialization_high_water_bytes: 2_048,
            fixed_l2_workspace_admission_bytes: 2_048,
            fixed_l2_workspace_observed_high_water_bytes: 1_024,
            replay_admission_high_water_bytes: 4_096,
            replay_observed_high_water_bytes: 1_536,
            provider_peak_count: 1,
            provider_peak_bytes: 1_024,
            preflight_provider_open_count: 1,
            production_provider_open_count: 1,
            operator_block_reads: 1,
            operator_block_cache_hits: 0,
            source_member_window_reads: 1,
            source_tile_cache_loads: 1,
            source_resolutions: 1,
            working_set_admission_high_water_bytes: 16_384,
            working_set_observed_high_water_bytes: 12_800,
        };
        let mut metadata = SpatialReferenceCovarianceMetadata {
            schema_version: SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
            method: SPATIAL_REFERENCE_COVARIANCE_METHOD.to_owned(),
            method_version: 1,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            producer_commit: Some("ab".repeat(20)),
            burst_id: "T078-165482-IW1".to_owned(),
            crs: "EPSG:32611".to_owned(),
            units: "radians".to_owned(),
            geotransform: Some([500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0]),
            full_grid: block.target_grid,
            reference_row: 0,
            reference_col: 0,
            gauge_date_index: 0,
            ordered_date_indices: (0..days.len() as u32).collect(),
            acquisition_days: Some(days.to_vec()),
            mask_digest: strong_digest(0x11),
            source_replay_digest: strong_digest(0x22),
            l2_map_digest: strong_digest(0x33),
            reference_signature_digest: strong_digest(0x44),
            approximation_receipt_digest: strong_digest(0x55),
            resource_receipt_digest: strong_digest(0x66),
            runtime_resource_receipt_digest: spatial_reference_runtime_resource_receipt_digest(
                runtime,
            ),
            runtime_resource_receipt: Some(runtime),
            review_receipt_digest: strong_digest(0x77),
            method_manifest_digest: strong_digest(0x88),
            calibration_scope_digest: String::new(),
            source_model_digest: strong_digest(0x99),
            effective_looks_digest: spatial_reference_effective_looks_digest(std::slice::from_ref(
                block,
            ))
            .unwrap(),
            support_method: "rect".to_owned(),
            support_digest: strong_digest(0xaa),
            correction_order_digest: strong_digest(0xbb),
            unwrap_branch_digest: strong_digest(0xcc),
            burst_ownership_digest: strong_digest(0xdd),
            source_burst_ids: vec!["T078-165482-IW1".to_owned()],
            reference_source_burst_index: 0,
            calibration_scope: SpatialReferenceCalibrationScope::CalibratedScopeMatch,
            maximum_block_bytes: 32_768,
        };
        metadata.calibration_scope_digest = spatial_reference_calibration_scope_digest(&metadata);
        metadata
    }

    fn strong_digest(byte: u8) -> String {
        format!("sha256:{}", format!("{byte:02x}").repeat(32))
    }
}
