//! Calibrated, fail-closed temporal-GLS raster products for issue #53.

use std::collections::BTreeMap;
#[cfg(unix)]
use std::ffi::{CStr, CString};
use std::fs::{File, OpenOptions};
use std::io::Read;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};

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
    temporal_covariance_workspace_composition, CompleteRefitBootstrapCadenceStatus,
    CompleteRefitBootstrapEstimate, CompleteRefitBootstrapEstimateStatus,
    TemporalCovarianceOptions, TemporalInferenceStatus, COMPLETE_REFIT_BOOTSTRAP_METHOD,
    COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
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
/// Exact one-shot 96+20 raw receipt consumed by the held-out scorer.
pub const TEMPORAL_HELDOUT_RECEIPT_FILENAME: &str =
    "temporal_covariance_heldout_result_receipt.json";
/// Frozen outcome-blind cohort manifest supplied with the held-out receipt.
pub const TEMPORAL_HELDOUT_MANIFEST_FILENAME: &str =
    "temporal_covariance_heldout_cohort_manifest.json";
/// Freeze receipt binding the manifest to the preregistration before unblinding.
pub const TEMPORAL_HELDOUT_FREEZE_RECEIPT_FILENAME: &str =
    "temporal_covariance_heldout_cohort_freeze_receipt.json";
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
const SYNTHETIC_SCHEMA: &str = "dolphinrust-temporal-covariance-simulation/7";
const TEMPORAL_BATCH_SCHEMA: &str = "dolphinrust-temporal-covariance-batch/6";
const TEMPORAL_PRODUCER_IDENTITY_SCHEMA: &str =
    "dolphinrust-temporal-covariance-run-identity/1";
const TEMPORAL_PRODUCER_SOURCE_SET_SCHEMA: &str =
    "dolphinrust.canonical-producer-source-set/2";
const HELDOUT_SCORE_SCHEMA: &str = "dolphinrust.temporal_covariance.heldout_score";
const HELDOUT_RECEIPT_SCHEMA: &str = "dolphinrust.temporal_covariance.heldout_receipt";
const LAYER_COUNT: usize = 14;
const COMBINED_WORKING_SET_CAP_BYTES: u64 = 512 * 1024 * 1024;
const TRANSACTION_LOCK_FILENAME: &str = ".temporal-covariance-product.lock";
const ROLLBACK_JOURNAL_FILENAME: &str = ".temporal-covariance-product.rollback.json";
const ROLLBACK_JOURNAL_SCHEMA: &str = "dolphinrust-temporal-product-rollback/2";
const TRANSACTION_ARTIFACT_MARKER_SCHEMA: &str = "dolphinrust-temporal-transaction-artifact/1";
const TRANSACTION_ARTIFACT_MARKER_FILENAME: &str = ".temporal-transaction-owner.json";
const TRANSACTION_STAGE_CLEANUP_PREFIX: &str = ".temporal-inference-stage-cleanup-";
#[cfg(unix)]
const CLEANUP_QUARANTINE_MARKER_SCHEMA: &str = "dolphinrust-temporal-cleanup-quarantine/1";
const HELDOUT_COHORT_ID: &str = "f53-06-outer-nonfresno-v1";
const HELDOUT_PREREGISTRATION_FILE_SHA256: &str =
    "4e960161f149330c57561b15afa2525e6cb608ce9e9b820816c3733749f41d4e";
const HELDOUT_MANIFEST_FILE_SHA256: &str =
    "016230fca1f3e4381d24effdd14e1839b603a71d6da3696ff561edc433a1970e";
const HELDOUT_MANIFEST_CANONICAL_SHA256: &str =
    "2b9208eaf1f54f10f971544062df348062d4a9b9eb518dd1b373bd8ebe561050";
const HELDOUT_FREEZE_FILE_SHA256: &str =
    "a10505a0e7ba3afd370c44eb9747e2021265626eb80aced72433096ce51c873e";
const HELDOUT_RECEIPT_SCORER_SHA256: &str =
    "942bda853ce55b28b7f92698f28ba9982cce370adfedde42da9e0b0e14426671";
const HELDOUT_REVIEW_SCORER_SHA256: &str =
    "7303e0f75437618afc47099cae05e729a7180603737b38c804f0fc7d30ddc2e8";
static NEXT_TRANSACTION_FILE_ID: AtomicU64 = AtomicU64::new(0);
static GDAL_CACHE_LIMIT_LOCK: Mutex<()> = Mutex::new(());

const TEMPORAL_PREREGISTRATION_BYTES: &[u8] =
    include_bytes!("../../../validation/temporal_covariance_preregistration.json");
const GENERATOR_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../validation/temporal_covariance_simulation.py");
const HELDOUT_PREREGISTRATION_BYTES: &[u8] =
    include_bytes!("../../../validation/temporal_covariance_heldout_preregistration.json");
const ESTIMATOR_SOURCE_BYTES: &[u8] =
    include_bytes!("../../dolphin-timeseries/src/temporal_covariance.rs");
const BATCH_SOURCE_BYTES: &[u8] =
    include_bytes!("../../dolphin-timeseries/examples/temporal_covariance_batch.rs");
const PRODUCT_SOURCE_BYTES: &[u8] = include_bytes!("temporal_covariance_product.rs");
const FIXED_CUBE_SOURCE_BYTES: &[u8] = include_bytes!("fixed_cube.rs");
const DISPLACEMENT_SOURCE_BYTES: &[u8] = include_bytes!("displacement.rs");
const SPATIAL_ARTIFACT_SOURCE_BYTES: &[u8] = include_bytes!("spatial_covariance_artifact.rs");
const GEOTIFF_SOURCE_BYTES: &[u8] = include_bytes!("../../dolphin-io/src/geotiff.rs");
const COVARIANCE_IO_SOURCE_BYTES: &[u8] = include_bytes!("../../dolphin-io/src/covariance.rs");
const CONFIG_SOURCE_BYTES: &[u8] = include_bytes!("../../dolphin-core/src/config.rs");
const PROVENANCE_SOURCE_BYTES: &[u8] = include_bytes!("provenance.rs");
const GEOMETRY_SOURCE_BYTES: &[u8] = include_bytes!("../../dolphin-corrections/src/geometry.rs");
const CARGO_LOCK_BYTES: &[u8] = include_bytes!("../../../Cargo.lock");

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

#[derive(Debug)]
struct TemporalProductTransaction {
    directory: PathBuf,
    ownership_token: String,
    _lock: File,
}

/// An unowned path collides with a reserved temporal-transaction prefix.
#[derive(Debug)]
pub struct TemporalTransactionCollision {
    /// Preserved collision paths, relative to the product directory.
    pub paths: Vec<String>,
}

impl std::fmt::Display for TemporalTransactionCollision {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unowned temporal transaction artifacts use reserved names: {}",
            self.paths.join(", ")
        )
    }
}

impl std::error::Error for TemporalTransactionCollision {}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransactionArtifactMarker {
    schema: String,
    ownership_token: String,
    product_directory_sha256: String,
    artifact_name: String,
}

#[cfg(unix)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CleanupQuarantineMarker {
    schema: String,
    ownership_token: String,
    product_directory_sha256: String,
    artifact_name: String,
    inner_artifact_name: String,
    outer_identity: DirectoryIdentity,
    inner_identity: DirectoryIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OwnedArtifactReceipt {
    name: String,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ProductGridReceipt {
    rows: usize,
    cols: usize,
    geotransform: [f64; 6],
    epsg: Option<u32>,
    velocity_unit: String,
    process_variance_unit: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProductRollbackState {
    Active,
    BlockedUnownedCollision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProductRollbackJournal {
    schema: String,
    ownership_token: String,
    original_fixed_cube_receipt: Vec<u8>,
    legacy_velocity_sha256: String,
    legacy_sigma_sha256: Option<String>,
    promotion_manifest_sha256: String,
    semantic_validation: crate::fixed_cube::FixedCubeSemanticValidation,
    product_grid: ProductGridReceipt,
    expected_products: Vec<OwnedArtifactReceipt>,
    installed_artifacts: Vec<OwnedArtifactReceipt>,
    stage_directory: String,
    expected_provenance_sha256: Option<String>,
    expected_fixed_receipt_sha256: Option<String>,
    rollback_state: ProductRollbackState,
    collision_artifacts: Vec<String>,
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
}

#[cfg(not(unix))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectoryIdentity;

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
            ownership_token: transaction_ownership_token(directory),
            _lock: lock,
        };
        recover_incomplete_product(&transaction.directory)?;
        cleanup_orphan_transaction_files(&transaction.directory)?;
        Ok(transaction)
    }

    #[cfg(not(unix))]
    fn acquire(_directory: &Path) -> Result<Self> {
        anyhow::bail!("temporal covariance product locking is unsupported on this platform")
    }
}

fn transaction_ownership_token(directory: &Path) -> String {
    let sequence = NEXT_TRANSACTION_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    sha256(
        format!(
            "{}\0{}\0{sequence}\0{elapsed}",
            directory.display(),
            std::process::id()
        )
        .as_bytes(),
    )
}

#[derive(Deserialize)]
struct SyntheticScores {
    all_methods_pass: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticProducerIdentity {
    schema: String,
    preregistration_sha256: String,
    generator_sha256: String,
    batch_source_sha256: String,
    estimator_source_sha256: String,
    source_set_schema: String,
    source_set_sha256: String,
    binary_path: String,
    binary_sha256: String,
    binary_bytes: u64,
    batch_schema: String,
    generator_schema: String,
    source_correlation_model: String,
    source_correlation_distance_scale_pixels: f64,
    seed_count: u64,
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
    producer_identity: SyntheticProducerIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldoutCoverage {
    status: String,
    p_value: f64,
    null_coverage: f64,
    observed_coverage: f64,
    evaluated: usize,
    covered: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldoutLevel {
    status: String,
    coverage: HeldoutCoverage,
    coverage_absolute_error: f64,
    coverage_absolute_gate: bool,
    mean_interval_score: f64,
    mean_baseline_interval_score: f64,
    median_width_ratio: f64,
    proper_score_improves: bool,
    holm_reject: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldoutResult {
    schema: String,
    schema_version: u16,
    cohort_id: String,
    manifest_file_sha256: String,
    manifest_sha256: String,
    freeze_receipt_sha256: String,
    factor_scope_sha256: String,
    heldout_receipt_sha256: String,
    primary_cluster_count: usize,
    surplus_cluster_count: usize,
    status: String,
    errors: Vec<Value>,
    levels: BTreeMap<String, HeldoutLevel>,
    evaluated_clusters: usize,
    emission_rate: f64,
    attrited_primary_ids: Vec<String>,
    used_surplus_ids: Vec<String>,
    unused_surplus_ids: Vec<String>,
    reasons_by_cluster: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HeldoutAttrition {
    attrited_primary_ids: Vec<String>,
    used_surplus_ids: Vec<String>,
    unused_surplus_ids: Vec<String>,
    reasons_by_cluster: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldoutRunIdentity {
    generation_id: String,
    preregistration_sha256: String,
    manifest_sha256: String,
    freeze_receipt_sha256: String,
    run_plan_sha256: String,
    binary_sha256: String,
    implementation_source_hashes: BTreeMap<String, String>,
    product_identities_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldoutClusterCounts {
    primary: usize,
    surplus: usize,
    executed: usize,
    evaluable: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldoutReceipt {
    schema: String,
    schema_version: u16,
    outcomes_present: bool,
    one_shot_unblinding: bool,
    cohort_id: String,
    generation_id: String,
    preregistration_sha256: String,
    manifest_sha256: String,
    scope_hash: String,
    calibrated_scope_match: bool,
    factor_binding: Value,
    hashes: BTreeMap<String, String>,
    cluster_counts: HeldoutClusterCounts,
    attrition: HeldoutAttrition,
    clusters: Vec<Value>,
    run_identity: HeldoutRunIdentity,
    run_identity_sha256: String,
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
    heldout_receipt_sha256: String,
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
    heldout_receipt_sha256: String,
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
    let heldout_receipt_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_HELDOUT_RECEIPT_FILENAME),
        JSON_CAP,
    )?;
    let heldout_receipt: HeldoutReceipt = serde_json::from_slice(&heldout_receipt_bytes)?;
    let heldout_manifest_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_HELDOUT_MANIFEST_FILENAME),
        JSON_CAP,
    )?;
    let heldout_freeze_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_HELDOUT_FREEZE_RECEIPT_FILENAME),
        JSON_CAP,
    )?;
    let heldout_identity = validate_heldout_receipt(
        &heldout_receipt,
        &heldout_manifest_bytes,
        &heldout_freeze_bytes,
        sha256(&heldout_receipt_bytes),
    )?;
    validate_heldout_result(&heldout, &heldout_identity)?;
    let review_bytes = read_bounded(
        &evidence_directory.join(TEMPORAL_REVIEW_RECEIPT_FILENAME),
        JSON_CAP,
    )?;
    let review: TemporalReviewReceipt = serde_json::from_slice(&review_bytes)?;
    let expected = EvidenceDigests::current(
        sha256(&synthetic_bytes),
        sha256(&heldout_bytes),
        sha256(&heldout_receipt_bytes),
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
    heldout_receipt_sha256: String,
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
        heldout_receipt_sha256: String,
        spatial_factor_sha256: String,
        spatial_manifest_sha256: String,
    ) -> Self {
        Self {
            synthetic_result_sha256,
            heldout_result_sha256,
            heldout_receipt_sha256,
            spatial_factor_sha256,
            spatial_manifest_sha256,
            temporal_preregistration_sha256: sha256(TEMPORAL_PREREGISTRATION_BYTES),
            heldout_preregistration_sha256: sha256(HELDOUT_PREREGISTRATION_BYTES),
            scorer_sha256: HELDOUT_REVIEW_SCORER_SHA256.to_owned(),
            source_sha256: canonical_named_sources_sha256(&[
                (
                    "dolphin-timeseries/src/temporal_covariance.rs",
                    ESTIMATOR_SOURCE_BYTES,
                ),
                (
                    "dolphin-timeseries/examples/temporal_covariance_batch.rs",
                    BATCH_SOURCE_BYTES,
                ),
                (
                    "dolphin-workflows/src/temporal_covariance_product.rs",
                    PRODUCT_SOURCE_BYTES,
                ),
                (
                    "dolphin-workflows/src/fixed_cube.rs",
                    FIXED_CUBE_SOURCE_BYTES,
                ),
                (
                    "dolphin-workflows/src/displacement.rs",
                    DISPLACEMENT_SOURCE_BYTES,
                ),
                (
                    "dolphin-workflows/src/spatial_covariance_artifact.rs",
                    SPATIAL_ARTIFACT_SOURCE_BYTES,
                ),
                ("dolphin-io/src/geotiff.rs", GEOTIFF_SOURCE_BYTES),
                ("dolphin-io/src/covariance.rs", COVARIANCE_IO_SOURCE_BYTES),
                ("dolphin-core/src/config.rs", CONFIG_SOURCE_BYTES),
                (
                    "dolphin-workflows/src/provenance.rs",
                    PROVENANCE_SOURCE_BYTES,
                ),
                ("dolphin-corrections/src/geometry.rs", GEOMETRY_SOURCE_BYTES),
                ("Cargo.lock", CARGO_LOCK_BYTES),
            ]),
        }
    }
}

fn canonical_named_sources_sha256(sources: &[(&str, &[u8])]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust-canonical-named-sources-v1\0");
    for (name, bytes) in sources {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

struct HeldoutReceiptIdentity {
    receipt_sha256: String,
    manifest_file_sha256: String,
    manifest_sha256: String,
    freeze_receipt_sha256: String,
    factor_scope_sha256: String,
    attrition: HeldoutAttrition,
    evaluated_clusters: usize,
    required_cluster_failed: bool,
    raw_levels: BTreeMap<String, RawHeldoutLevel>,
}

#[derive(Debug)]
struct RawHeldoutLevel {
    covered: usize,
    mean_interval_score: f64,
    mean_baseline_interval_score: f64,
    median_width_ratio: f64,
}

#[derive(Debug)]
struct RawHeldoutObservationLevel {
    covered: bool,
    interval_score: f64,
    baseline_interval_score: f64,
    width: f64,
    baseline_width: f64,
}

fn canonical_json_sha256(bytes: &[u8]) -> Result<String> {
    let value: Value = serde_json::from_slice(bytes)?;
    Ok(sha256(&serde_json::to_vec(&value)?))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn expected_heldout_implementation_source_hashes() -> BTreeMap<String, String> {
    [
        (
            "executor_sha256",
            "e0185db2618ea6d2ef1b0f7300cbac63955a0e4f4d2d68922907ba22ff70a18c",
        ),
        (
            "cohort_sha256",
            "2a19bbceda51be62134944a8af1b6a2db32d469bbc828581c5cb8e97ea3b9183",
        ),
        (
            "gps_ground_truth_sha256",
            "87205de6626aa4687b99b904ad0e429ad49c5aedde82db536390119f3b36c81f",
        ),
        (
            "runner_sha256",
            "ae8ffa1cd31e97f1ccccc3dd31b66aa36a4859c8d0347d5187fb827cb1920e71",
        ),
        (
            "scorer_sha256",
            "a0ba776261cc9e774696ccebd62404804403c7b64f6146d52959942814dfa8ab",
        ),
        (
            "runner_cli_sha256",
            "cb67b8956872ee2525e0f64af18a68de8b57b57b44a08aecdde480ba59ad6211",
        ),
        (
            "scorer_cli_sha256",
            "d69735f7f2ac36e6c1ebace5467d7e152ef2d083ae2872e43fedea175ca8dea0",
        ),
    ]
    .into_iter()
    .map(|(name, digest)| (name.to_owned(), digest.to_owned()))
    .collect()
}

#[allow(clippy::too_many_lines)]
fn validate_heldout_receipt(
    receipt: &HeldoutReceipt,
    manifest_bytes: &[u8],
    freeze_bytes: &[u8],
    receipt_sha256: String,
) -> Result<HeldoutReceiptIdentity> {
    let preregistration: Value = serde_json::from_slice(HELDOUT_PREREGISTRATION_BYTES)?;
    let manifest: Value = serde_json::from_slice(manifest_bytes)?;
    let freeze: Value = serde_json::from_slice(freeze_bytes)?;
    let preregistration_sha256 = canonical_json_sha256(HELDOUT_PREREGISTRATION_BYTES)?;
    let manifest_sha256 = canonical_json_sha256(manifest_bytes)?;
    let manifest_file_sha256 = sha256(manifest_bytes);
    let freeze_receipt_sha256 = sha256(freeze_bytes);
    let factor_binding = preregistration["factor_binding"].clone();
    let factor_scope_sha256 = sha256(&serde_json::to_vec(&factor_binding)?);
    let scope_hash = sha256(&serde_json::to_vec(&preregistration["field_scope"])?);
    ensure!(
        sha256(HELDOUT_PREREGISTRATION_BYTES) == HELDOUT_PREREGISTRATION_FILE_SHA256
            && manifest_file_sha256 == HELDOUT_MANIFEST_FILE_SHA256
            && manifest_sha256 == HELDOUT_MANIFEST_CANONICAL_SHA256
            && freeze_receipt_sha256 == HELDOUT_FREEZE_FILE_SHA256,
        "held-out preregistration, manifest, or freeze bytes differ from the frozen cohort"
    );
    ensure!(
        manifest["schema"].as_str() == Some("dolphinrust.temporal_covariance.heldout_cohort")
            && manifest["schema_version"].as_u64() == Some(2)
            && manifest["cohort_id"].as_str() == Some(HELDOUT_COHORT_ID)
            && manifest["status"].as_str() == Some("frozen_metadata_only")
            && manifest["outcomes_present"].as_bool() == Some(false)
            && manifest["frozen_clusters"].as_array().map(Vec::len) == Some(96)
            && manifest["surplus_clusters"].as_array().map(Vec::len) == Some(20),
        "held-out manifest is not the exact outcome-blind 96+20 schema"
    );
    ensure!(
        freeze["schema"].as_str()
            == Some("dolphinrust.temporal_covariance.heldout_cohort_freeze_receipt")
            && freeze["schema_version"].as_u64() == Some(1)
            && freeze["cohort_id"].as_str() == Some(HELDOUT_COHORT_ID)
            && freeze["outcomes_present"].as_bool() == Some(false)
            && freeze["selection_outcome_blind"].as_bool() == Some(true)
            && freeze["counts"]["frozen_clusters"].as_u64() == Some(96)
            && freeze["counts"]["surplus_clusters"].as_u64() == Some(20)
            && freeze["hashes"]["manifest_file_sha256"].as_str()
                == Some(manifest_file_sha256.as_str())
            && freeze["hashes"]["manifest_canonical_sha256"].as_str()
                == Some(manifest_sha256.as_str())
            && freeze["hashes"]["preregistration_file_sha256"].as_str()
                == Some(HELDOUT_PREREGISTRATION_FILE_SHA256)
            && freeze["hashes"]["preregistration_canonical_sha256"].as_str()
                == Some(preregistration_sha256.as_str()),
        "held-out freeze receipt does not bind the supplied 96+20 manifest"
    );
    ensure!(
        receipt.schema == HELDOUT_RECEIPT_SCHEMA
            && receipt.schema_version == 1
            && receipt.outcomes_present
            && receipt.one_shot_unblinding
            && receipt.cohort_id == HELDOUT_COHORT_ID
            && receipt.generation_id == "f53-06-v1"
            && receipt.preregistration_sha256 == preregistration_sha256
            && receipt.manifest_sha256 == manifest_sha256
            && receipt.scope_hash == scope_hash
            && receipt.calibrated_scope_match
            && receipt.factor_binding == factor_binding,
        "held-out raw receipt schema or frozen scope identity differs"
    );
    let expected_hash_fields = preregistration["receipt_hash_fields"]
        .as_array()
        .context("held-out preregistration lacks receipt hash fields")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .context("held-out receipt hash field is not a string")
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    ensure!(
        expected_hash_fields.contains("binary_sha256")
            && expected_hash_fields.contains("scorer_sha256")
            && expected_hash_fields.contains("gnss_catalog_sha256")
            && expected_hash_fields.contains("factor_scope_sha256")
            && receipt
                .hashes
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                == expected_hash_fields
            && receipt.hashes.values().all(|value| is_sha256(value))
            && receipt.hashes.get("preregistration_sha256") == Some(&preregistration_sha256)
            && receipt.hashes.get("manifest_sha256") == Some(&manifest_sha256)
            && receipt.hashes.get("factor_scope_sha256") == Some(&factor_scope_sha256),
        "held-out raw receipt implementation/factor/GNSS hash inventory is incomplete"
    );
    ensure!(
        receipt.run_identity.generation_id == receipt.generation_id
            && receipt.run_identity.preregistration_sha256 == preregistration_sha256
            && receipt.run_identity.manifest_sha256 == manifest_sha256
            && receipt.run_identity.freeze_receipt_sha256 == freeze_receipt_sha256
            && receipt.hashes.get("binary_sha256").map(String::as_str)
                == Some(receipt.run_identity.binary_sha256.as_str())
            && is_sha256(&receipt.run_identity.run_plan_sha256)
            && receipt.run_identity.implementation_source_hashes
                == expected_heldout_implementation_source_hashes()
            && is_sha256(&receipt.run_identity.product_identities_sha256)
            && receipt.run_identity_sha256 == canonical_run_identity_sha256(&receipt.run_identity)?,
        "held-out raw receipt run identity is stale or internally inconsistent"
    );
    validate_heldout_accounting(receipt, &manifest)?;
    let raw_levels = validate_heldout_cluster_bindings(receipt, &preregistration, &manifest)?;
    ensure!(
        receipt.hashes.get("scorer_sha256").map(String::as_str)
            == Some(HELDOUT_RECEIPT_SCORER_SHA256),
        "held-out raw receipt was not produced for the frozen scorer"
    );
    let primary_ids = manifest_cluster_ids(&manifest, "frozen_clusters")?
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let required_cluster_failed = receipt.clusters.iter().any(|cluster| {
        cluster["cluster_id"]
            .as_str()
            .is_some_and(|id| primary_ids.contains(id))
            && cluster["status"].as_str() == Some("fail")
    }) || receipt.attrition.used_surplus_ids.iter().any(|used| {
        receipt.clusters.iter().any(|cluster| {
            cluster["cluster_id"].as_str() == Some(used.as_str())
                && cluster["status"].as_str() == Some("fail")
        })
    });
    Ok(HeldoutReceiptIdentity {
        receipt_sha256,
        manifest_file_sha256,
        manifest_sha256,
        freeze_receipt_sha256,
        factor_scope_sha256,
        attrition: receipt.attrition.clone(),
        evaluated_clusters: receipt.cluster_counts.evaluable,
        required_cluster_failed,
        raw_levels,
    })
}

fn canonical_run_identity_sha256(identity: &HeldoutRunIdentity) -> Result<String> {
    let fields = BTreeMap::from([
        (
            "binary_sha256",
            Value::String(identity.binary_sha256.clone()),
        ),
        (
            "freeze_receipt_sha256",
            Value::String(identity.freeze_receipt_sha256.clone()),
        ),
        (
            "generation_id",
            Value::String(identity.generation_id.clone()),
        ),
        (
            "implementation_source_hashes",
            serde_json::to_value(&identity.implementation_source_hashes)?,
        ),
        (
            "manifest_sha256",
            Value::String(identity.manifest_sha256.clone()),
        ),
        (
            "preregistration_sha256",
            Value::String(identity.preregistration_sha256.clone()),
        ),
        (
            "product_identities_sha256",
            Value::String(identity.product_identities_sha256.clone()),
        ),
        (
            "run_plan_sha256",
            Value::String(identity.run_plan_sha256.clone()),
        ),
    ]);
    Ok(sha256(&serde_json::to_vec(&fields)?))
}

fn manifest_cluster_ids(manifest: &Value, field: &str) -> Result<Vec<String>> {
    manifest[field]
        .as_array()
        .context("held-out manifest cluster list is missing")?
        .iter()
        .map(|candidate| {
            candidate["candidate_id"]
                .as_str()
                .map(str::to_owned)
                .context("held-out manifest candidate lacks an identity")
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn validate_heldout_cluster_bindings(
    receipt: &HeldoutReceipt,
    preregistration: &Value,
    manifest: &Value,
) -> Result<BTreeMap<String, RawHeldoutLevel>> {
    let candidates = manifest["frozen_clusters"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(
            manifest["surplus_clusters"]
                .as_array()
                .into_iter()
                .flatten(),
        )
        .map(|candidate| {
            Ok((
                candidate["candidate_id"]
                    .as_str()
                    .context("held-out candidate lacks an identity")?
                    .to_owned(),
                candidate,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let required = &preregistration["factor_binding"];
    let scope_fields = required["scope_fields"]
        .as_array()
        .context("held-out factor binding lacks scope fields")?
        .iter()
        .map(|field| {
            field
                .as_str()
                .map(str::to_owned)
                .context("held-out factor scope field is not a string")
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    let mut aggregate: BTreeMap<&str, BTreeMap<String, String>> = [
        "operator_sha256",
        "operator_manifest_sha256",
        "persisted_factor_sha256",
        "persisted_factor_manifest_sha256",
        "gnss_catalog_sha256",
        "approximation_receipt_sha256",
        "resource_receipt_sha256",
        "calibration_scope_receipt_sha256",
        "review_receipt_sha256",
    ]
    .into_iter()
    .map(|field| (field, BTreeMap::new()))
    .collect();
    let mut cluster_levels = BTreeMap::new();
    for cluster in &receipt.clusters {
        let cluster_id = cluster["cluster_id"]
            .as_str()
            .context("held-out cluster lacks an identity")?;
        let candidate = candidates
            .get(cluster_id)
            .context("held-out cluster is not frozen")?;
        ensure!(
            cluster["station_ids"] == candidate["station_ids"]
                && cluster["burst_id"] == candidate["burst_id"]
                && cluster["site_id"] == candidate["site_id"],
            "held-out cluster scope metadata differs from its frozen candidate"
        );
        if !matches!(cluster["status"].as_str(), Some("pass" | "fail")) {
            continue;
        }
        let binding = cluster["difference_covariance"]
            .as_object()
            .context("evaluable held-out cluster lacks a direct factor binding")?;
        ensure!(
            binding.get("operation") == Some(&required["operation"])
                && binding.get("input_operator") == Some(&required["input_operator"])
                && binding.get("output_factor") == Some(&required["output_factor"])
                && binding.get("mode") == Some(&required["mode"])
                && binding.get("reference_specific") == Some(&required["reference_specific"])
                && binding.get("stitched_burst_count") == Some(&required["stitched_burst_count"])
                && binding.get("marginal_rss_combination_allowed") == Some(&Value::Bool(false))
                && binding
                    .get("calibrated_scope_match")
                    .and_then(Value::as_str)
                    == Some("calibrated_scope_match"),
            "held-out cluster factor identity differs from the frozen direct-factor contract"
        );
        for field in ["factor_sha256", "scope_sha256"] {
            ensure!(
                binding
                    .get(field)
                    .and_then(Value::as_str)
                    .is_some_and(is_sha256),
                "held-out cluster factor digest is invalid: {field}"
            );
        }
        let scope = binding
            .get("scope")
            .and_then(Value::as_object)
            .context("evaluable held-out cluster factor scope is missing")?;
        ensure!(
            scope
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                == scope_fields
                && scope.get("target_station_id") == candidate["station_ids"].get(0)
                && scope.get("control_station_id") == candidate["station_ids"].get(1)
                && scope.get("burst_id") == Some(&candidate["burst_id"])
                && scope.get("schema_version") == required["output_factor"].get("schema_version")
                && scope.get("method_version") == required["output_factor"].get("method_version")
                && scope.get("method") == required["output_factor"].get("method")
                && scope.get("calibration_scope")
                    == required["output_factor"].get("calibration_status")
                && matches!(
                    scope.get("units").and_then(Value::as_str),
                    Some("meters" | "millimeters")
                ),
            "held-out cluster factor scope differs from its frozen station/burst identity"
        );
        for field in scope_fields
            .iter()
            .filter(|field| field.ends_with("_sha256"))
        {
            ensure!(
                scope
                    .get(field)
                    .and_then(Value::as_str)
                    .is_some_and(is_sha256),
                "held-out cluster factor scope digest is missing: {field}"
            );
        }
        let scope_sha256 = sha256(&serde_json::to_vec(scope)?);
        ensure!(
            binding.get("scope_sha256").and_then(Value::as_str) == Some(scope_sha256.as_str()),
            "held-out cluster factor scope digest differs from its exact fields"
        );
        for field in [
            "operator_sha256",
            "operator_manifest_sha256",
            "persisted_factor_sha256",
            "persisted_factor_manifest_sha256",
        ] {
            let digest = binding
                .get(field)
                .and_then(Value::as_str)
                .filter(|value| is_sha256(value))
                .context("held-out cluster persisted factor digest is invalid")?;
            aggregate
                .get_mut(field)
                .expect("frozen aggregate field")
                .insert(cluster_id.to_owned(), digest.to_owned());
        }
        let gnss = cluster["gnss_provenance"]["solution_sha256"]
            .as_str()
            .filter(|value| is_sha256(value))
            .context("held-out cluster GNSS digest is invalid")?;
        aggregate
            .get_mut("gnss_catalog_sha256")
            .expect("frozen aggregate field")
            .insert(cluster_id.to_owned(), gnss.to_owned());
        validate_heldout_gnss(cluster, candidate, preregistration)?;
        cluster_levels.insert(
            cluster_id.to_owned(),
            score_heldout_observation(&cluster["observation"])?,
        );
        for (output, source) in [
            (
                "approximation_receipt_sha256",
                "approximation_receipt_sha256",
            ),
            ("resource_receipt_sha256", "resource_receipt_sha256"),
            ("calibration_scope_receipt_sha256", "method_manifest_sha256"),
            ("review_receipt_sha256", "review_receipt_sha256"),
        ] {
            let digest = scope
                .get(source)
                .and_then(Value::as_str)
                .filter(|value| is_sha256(value))
                .context("held-out cluster factor-evidence digest is invalid")?;
            aggregate
                .get_mut(output)
                .expect("frozen aggregate field")
                .insert(cluster_id.to_owned(), digest.to_owned());
        }
    }
    for (field, values) in aggregate {
        let digest = sha256(&serde_json::to_vec(&values)?);
        ensure!(
            receipt.hashes.get(field) == Some(&digest),
            "held-out raw receipt factor/GNSS aggregate hash is cross-wired: {field}"
        );
    }
    aggregate_selected_heldout_levels(receipt, manifest, &cluster_levels)
}

fn validate_heldout_gnss(
    cluster: &Value,
    candidate: &Value,
    preregistration: &Value,
) -> Result<()> {
    let provenance = cluster["gnss_provenance"]
        .as_object()
        .context("held-out cluster GNSS provenance is missing")?;
    let expected_fields = std::collections::BTreeSet::from([
        "solution_sources",
        "solution_sha256",
        "coordinate_frame",
        "los_source",
        "los_sha256",
        "station_los_vectors",
        "projection_convention",
        "epoch_zero_reference_sha256",
        "covariance_projection",
    ]);
    ensure!(
        provenance
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == expected_fields,
        "held-out cluster GNSS provenance fields are incomplete"
    );
    for field in [
        "solution_sha256",
        "los_sha256",
        "epoch_zero_reference_sha256",
    ] {
        ensure!(
            provenance
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(is_sha256),
            "held-out cluster GNSS provenance digest is invalid: {field}"
        );
    }
    let required = &preregistration["gnss_provenance"];
    ensure!(
        provenance.get("coordinate_frame") == Some(&required["coordinate_frame"])
            && provenance.get("projection_convention") == Some(&required["projection"])
            && provenance.get("covariance_projection") == Some(&required["covariance_projection"])
            && provenance.get("los_source") == Some(&required["los_source"]),
        "held-out cluster GNSS projection provenance differs from preregistration"
    );
    let station_ids = candidate["station_ids"]
        .as_array()
        .context("held-out candidate station identities are missing")?
        .iter()
        .map(|station| {
            station
                .as_str()
                .context("held-out candidate station identity is invalid")
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    let sources = provenance
        .get("solution_sources")
        .and_then(Value::as_object)
        .context("held-out cluster GNSS solution sources are missing")?;
    let vectors = provenance
        .get("station_los_vectors")
        .and_then(Value::as_object)
        .context("held-out cluster GNSS LOS vectors are missing")?;
    ensure!(
        sources
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == station_ids
            && vectors
                .keys()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>()
                == station_ids,
        "held-out cluster GNSS station identities differ from its frozen candidate"
    );
    let tolerance = required["los_norm_tolerance"]
        .as_f64()
        .context("held-out GNSS LOS tolerance is invalid")?;
    for vector in vectors.values() {
        let components = vector
            .as_array()
            .filter(|values| values.len() == 3)
            .context("held-out cluster GNSS LOS vector is invalid")?;
        let values = components
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .context("held-out cluster GNSS LOS vector is nonfinite")
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
        ensure!(
            (norm - 1.0).abs() <= tolerance,
            "held-out cluster GNSS LOS vector is not unit norm"
        );
    }
    Ok(())
}

fn score_heldout_observation(
    observation: &Value,
) -> Result<BTreeMap<String, RawHeldoutObservationLevel>> {
    let observation = observation
        .as_object()
        .context("held-out cluster slope observation is missing")?;
    let expected_fields = std::collections::BTreeSet::from([
        "insar_slope_difference",
        "gnss_slope_difference",
        "insar_difference_variance",
        "gnss_slope_variance",
        "sensor_cross_covariance",
        "baseline_sigma",
    ]);
    ensure!(
        observation
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == expected_fields,
        "held-out cluster slope observation fields differ from the frozen schema"
    );
    let finite = |field: &str| -> Result<f64> {
        observation[field]
            .as_f64()
            .filter(|value| value.is_finite())
            .with_context(|| format!("held-out cluster slope observation is invalid: {field}"))
    };
    let insar_difference = finite("insar_slope_difference")?;
    let gnss_difference = finite("gnss_slope_difference")?;
    let insar_variance = finite("insar_difference_variance")?;
    let gnss_variance = finite("gnss_slope_variance")?;
    let cross_covariance = finite("sensor_cross_covariance")?;
    ensure!(
        insar_variance >= 0.0 && gnss_variance >= 0.0 && cross_covariance == 0.0,
        "held-out cluster slope covariance is invalid"
    );
    let variance = insar_variance + gnss_variance;
    ensure!(
        variance.is_finite() && variance > 0.0,
        "held-out cluster combined slope covariance is not positive"
    );
    let baseline = observation["baseline_sigma"]
        .as_object()
        .context("held-out cluster baseline sigma is missing")?;
    ensure!(
        baseline
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == std::collections::BTreeSet::from(["68", "90", "95"]),
        "held-out cluster baseline sigma levels are incomplete"
    );
    let difference = insar_difference - gnss_difference;
    let sigma = variance.sqrt();
    let mut levels = BTreeMap::new();
    for (level, z) in [
        ("68", 0.994_457_883_209_753),
        ("90", 1.644_853_626_951_472_2),
        ("95", 1.959_963_984_540_054),
    ] {
        let nominal = level.parse::<f64>()? / 100.0;
        let baseline_sigma = baseline[level]
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .context("held-out cluster baseline sigma is invalid")?;
        let half_width = z * sigma;
        let baseline_half_width = z * baseline_sigma;
        levels.insert(
            level.to_owned(),
            RawHeldoutObservationLevel {
                covered: difference.abs() <= half_width,
                interval_score: interval_score(
                    difference - half_width,
                    difference + half_width,
                    nominal,
                ),
                baseline_interval_score: interval_score(
                    difference - baseline_half_width,
                    difference + baseline_half_width,
                    nominal,
                ),
                width: 2.0 * half_width,
                baseline_width: 2.0 * baseline_half_width,
            },
        );
    }
    Ok(levels)
}

fn interval_score(lower: f64, upper: f64, nominal: f64) -> f64 {
    let mut score = upper - lower;
    if 0.0 < lower {
        score += 2.0 * lower / (1.0 - nominal);
    } else if 0.0 > upper {
        score += 2.0 * -upper / (1.0 - nominal);
    }
    score
}

fn aggregate_selected_heldout_levels(
    receipt: &HeldoutReceipt,
    manifest: &Value,
    cluster_levels: &BTreeMap<String, BTreeMap<String, RawHeldoutObservationLevel>>,
) -> Result<BTreeMap<String, RawHeldoutLevel>> {
    let statuses = receipt
        .clusters
        .iter()
        .map(|cluster| {
            Ok((
                cluster["cluster_id"]
                    .as_str()
                    .context("held-out cluster identity is missing")?,
                cluster["status"]
                    .as_str()
                    .context("held-out cluster status is missing")?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let selected = manifest_cluster_ids(manifest, "frozen_clusters")?
        .into_iter()
        .filter(|id| statuses.get(id.as_str()) == Some(&"pass"))
        .chain(receipt.attrition.used_surplus_ids.iter().cloned())
        .collect::<Vec<_>>();
    ensure!(
        selected.len() == 96,
        "held-out selected raw cluster denominator is not 96"
    );
    let mut result = BTreeMap::new();
    for level in ["68", "90", "95"] {
        let values = selected
            .iter()
            .map(|id| {
                cluster_levels
                    .get(id)
                    .and_then(|levels| levels.get(level))
                    .context("selected held-out cluster lacks raw level metrics")
            })
            .collect::<Result<Vec<_>>>()?;
        let divisor = values.len() as f64;
        let mut widths = values.iter().map(|value| value.width).collect::<Vec<_>>();
        let mut baseline_widths = values
            .iter()
            .map(|value| value.baseline_width)
            .collect::<Vec<_>>();
        result.insert(
            level.to_owned(),
            RawHeldoutLevel {
                covered: values.iter().filter(|value| value.covered).count(),
                mean_interval_score: values.iter().map(|value| value.interval_score).sum::<f64>()
                    / divisor,
                mean_baseline_interval_score: values
                    .iter()
                    .map(|value| value.baseline_interval_score)
                    .sum::<f64>()
                    / divisor,
                median_width_ratio: median(&mut widths)? / median(&mut baseline_widths)?,
            },
        );
    }
    Ok(result)
}

fn median(values: &mut [f64]) -> Result<f64> {
    ensure!(!values.is_empty(), "held-out median is undefined");
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

#[allow(clippy::too_many_lines)]
fn validate_heldout_accounting(receipt: &HeldoutReceipt, manifest: &Value) -> Result<()> {
    let primary_ids = manifest_cluster_ids(manifest, "frozen_clusters")?;
    let surplus_ids = manifest_cluster_ids(manifest, "surplus_clusters")?;
    let primary = primary_ids
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let surplus = surplus_ids
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut statuses = BTreeMap::new();
    let mut receipt_ids = Vec::with_capacity(receipt.clusters.len());
    for cluster in &receipt.clusters {
        let id = cluster["cluster_id"]
            .as_str()
            .context("held-out cluster lacks an identity")?;
        let status = cluster["status"]
            .as_str()
            .context("held-out cluster lacks a status")?;
        ensure!(
            matches!(status, "pass" | "fail" | "not_evaluable" | "not_used")
                && statuses.insert(id.to_owned(), status.to_owned()).is_none(),
            "held-out cluster accounting has an invalid or duplicate entry"
        );
        receipt_ids.push(id.to_owned());
    }
    let attrition = &receipt.attrition;
    let used = attrition
        .used_surplus_ids
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let unused = attrition
        .unused_surplus_ids
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let attrited = attrition
        .attrited_primary_ids
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let expected_attrited = primary_ids
        .iter()
        .filter(|id| statuses.get(*id).map(String::as_str) == Some("not_evaluable"))
        .cloned()
        .collect::<Vec<_>>();
    let expected_used = surplus_ids
        .iter()
        .filter(|id| matches!(statuses.get(*id).map(String::as_str), Some("pass" | "fail")))
        .take(expected_attrited.len())
        .cloned()
        .collect::<Vec<_>>();
    let expected_unused = surplus_ids
        .iter()
        .filter(|id| !expected_used.contains(id))
        .cloned()
        .collect::<Vec<_>>();
    let expected_reasons = statuses
        .iter()
        .filter_map(|(id, status)| (status == "not_evaluable").then_some(id))
        .collect::<std::collections::BTreeSet<_>>();
    let executed = statuses
        .values()
        .filter(|status| *status != "not_used")
        .count();
    let evaluable = statuses
        .values()
        .filter(|status| matches!(status.as_str(), "pass" | "fail"))
        .count();
    ensure!(
        primary.len() == 96
            && surplus.len() == 20
            && receipt.cluster_counts.primary == primary.len()
            && receipt.cluster_counts.surplus == surplus.len()
            && receipt.cluster_counts.executed == executed
            && receipt.cluster_counts.evaluable == evaluable
            && primary.is_disjoint(&surplus)
            && receipt_ids
                == primary_ids
                    .iter()
                    .chain(&surplus_ids)
                    .cloned()
                    .collect::<Vec<_>>()
            && used.len() == attrition.used_surplus_ids.len()
            && unused.len() == attrition.unused_surplus_ids.len()
            && attrited.len() == attrition.attrited_primary_ids.len()
            && used.is_disjoint(&unused)
            && attrited.is_subset(&primary)
            && used.is_subset(&surplus)
            && unused.is_subset(&surplus)
            && used.len() == attrited.len()
            && used.len() + unused.len() == 20
            && attrition.attrited_primary_ids == expected_attrited
            && attrition.used_surplus_ids == expected_used
            && attrition.unused_surplus_ids == expected_unused
            && primary.iter().all(|id| matches!(
                statuses.get(id).map(String::as_str),
                Some("pass" | "fail" | "not_evaluable")
            ))
            && used
                .iter()
                .all(|id| matches!(statuses.get(id).map(String::as_str), Some("pass" | "fail")))
            && unused.iter().all(|id| matches!(
                statuses.get(id).map(String::as_str),
                Some("not_used" | "not_evaluable")
            ))
            && attrition
                .reasons_by_cluster
                .keys()
                .collect::<std::collections::BTreeSet<_>>()
                == expected_reasons
            && attrition.reasons_by_cluster.iter().all(|(id, reason)| {
                !reason.is_empty() && statuses.get(id).map(String::as_str) == Some("not_evaluable")
            }),
        "held-out raw receipt does not exactly account for 96 primary and 20 surplus clusters"
    );
    Ok(())
}

fn validate_synthetic_result(result: &SyntheticResult) -> Result<()> {
    let preregistration: Value = serde_json::from_slice(TEMPORAL_PREREGISTRATION_BYTES)?;
    let producer = &result.producer_identity;
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
        result.seed_count == 1_050
            && result.attempted_cells == 50_400
            && result.batch_attempted_cells == result.attempted_cells,
        "synthetic temporal-covariance denominator is not the exact frozen matrix"
    );
    ensure!(
        producer.schema == TEMPORAL_PRODUCER_IDENTITY_SCHEMA
            && producer.preregistration_sha256
                == canonical_json_sha256(TEMPORAL_PREREGISTRATION_BYTES)?
            && producer.generator_sha256 == sha256(GENERATOR_SOURCE_BYTES)
            && producer.batch_source_sha256 == sha256(BATCH_SOURCE_BYTES)
            && producer.estimator_source_sha256 == sha256(ESTIMATOR_SOURCE_BYTES)
            && producer.source_set_schema == TEMPORAL_PRODUCER_SOURCE_SET_SCHEMA
            && preregistration["producer_identity"]["source_set_schema"].as_str()
                == Some(producer.source_set_schema.as_str())
            && preregistration["producer_identity"]["source_set_sha256"].as_str()
                == Some(producer.source_set_sha256.as_str())
            && preregistration["producer_identity"]["binary_path"].as_str()
                == Some(producer.binary_path.as_str())
            && producer.binary_path == "target/release/examples/temporal_covariance_batch"
            && is_sha256(&producer.binary_sha256)
            && producer.binary_bytes > 0
            && producer.batch_schema == TEMPORAL_BATCH_SCHEMA
            && producer.generator_schema == SYNTHETIC_SCHEMA
            && producer.source_correlation_model == "exponential_euclidean_v1"
            && producer.source_correlation_distance_scale_pixels == 1.5
            && producer.seed_count == result.seed_count,
        "synthetic temporal-covariance producer identity is stale or malformed"
    );
    ensure!(
        !result.resource_gates.is_empty() && result.resource_gates.values().all(|passed| *passed),
        "synthetic temporal-covariance resource gates did not all pass"
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn validate_heldout_result(
    result: &HeldoutResult,
    expected: &HeldoutReceiptIdentity,
) -> Result<()> {
    let unique = |values: &[String]| {
        values
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == values.len()
    };
    let attrition_count = result.attrited_primary_ids.len();
    ensure!(
        result.schema == HELDOUT_SCORE_SCHEMA
            && result.schema_version == 1
            && result.cohort_id == HELDOUT_COHORT_ID
            && result.manifest_file_sha256 == expected.manifest_file_sha256
            && result.manifest_sha256 == expected.manifest_sha256
            && result.freeze_receipt_sha256 == expected.freeze_receipt_sha256
            && result.factor_scope_sha256 == expected.factor_scope_sha256
            && result.heldout_receipt_sha256 == expected.receipt_sha256,
        "held-out temporal-covariance result is not bound to the exact frozen cohort"
    );
    ensure!(
        result.primary_cluster_count == 96
            && result.surplus_cluster_count == 20
            && result.evaluated_clusters == expected.evaluated_clusters
            && expected.evaluated_clusters == 96
            && !expected.required_cluster_failed
            && attrition_count <= result.surplus_cluster_count
            && result.used_surplus_ids.len() == attrition_count
            && result.unused_surplus_ids.len() + result.used_surplus_ids.len()
                == result.surplus_cluster_count
            && result.attrited_primary_ids.iter().all(|id| {
                result.reasons_by_cluster.contains_key(id)
                    && !result.used_surplus_ids.contains(id)
                    && !result.unused_surplus_ids.contains(id)
            })
            && result
                .used_surplus_ids
                .iter()
                .all(|id| !result.unused_surplus_ids.contains(id))
            && result.reasons_by_cluster.keys().all(|id| {
                result.attrited_primary_ids.contains(id) || result.unused_surplus_ids.contains(id)
            })
            && unique(&result.attrited_primary_ids)
            && unique(&result.used_surplus_ids)
            && unique(&result.unused_surplus_ids)
            && result.attrited_primary_ids == expected.attrition.attrited_primary_ids
            && result.used_surplus_ids == expected.attrition.used_surplus_ids
            && result.unused_surplus_ids == expected.attrition.unused_surplus_ids
            && result.reasons_by_cluster == expected.attrition.reasons_by_cluster,
        "held-out temporal-covariance result does not account for exact 96+20 frozen slots"
    );
    ensure!(
        result.status == "pass"
            && result.errors.is_empty()
            && result.emission_rate.is_finite()
            && approximately_equal(result.emission_rate, 1.0),
        "held-out temporal-covariance result did not pass the frozen cohort"
    );
    let mut p_values = Vec::with_capacity(3);
    ensure!(
        result.levels.len() == 3
            && ["68", "90", "95"]
                .iter()
                .all(|level| result.levels.contains_key(*level)),
        "held-out temporal-covariance level gates are incomplete"
    );
    for level in ["68", "90", "95"] {
        let nominal = level.parse::<f64>()? / 100.0;
        let value = &result.levels[level];
        let coverage = &value.coverage;
        let raw = &expected.raw_levels[level];
        let expected_p =
            exact_binomial_upper_tail(coverage.covered, coverage.evaluated, nominal - 0.2)?;
        let observed = coverage.covered as f64 / coverage.evaluated as f64;
        let coverage_error = (observed - nominal).abs();
        ensure!(
            value.status == "pass"
                && coverage.status == "pass"
                && coverage.evaluated == expected.evaluated_clusters
                && coverage.covered == raw.covered
                && coverage.covered <= coverage.evaluated
                && approximately_equal(coverage.null_coverage, nominal - 0.2)
                && approximately_equal(coverage.observed_coverage, observed)
                && approximately_equal(coverage.p_value, expected_p)
                && coverage.p_value <= 0.05
                && approximately_equal(value.coverage_absolute_error, coverage_error)
                && value.coverage_absolute_gate == (coverage_error <= 0.2)
                && value.coverage_absolute_gate
                && value.mean_interval_score.is_finite()
                && value.mean_interval_score >= 0.0
                && approximately_equal(value.mean_interval_score, raw.mean_interval_score)
                && value.mean_baseline_interval_score.is_finite()
                && value.mean_baseline_interval_score >= 0.0
                && approximately_equal(
                    value.mean_baseline_interval_score,
                    raw.mean_baseline_interval_score,
                )
                && value.mean_interval_score < value.mean_baseline_interval_score
                && value.proper_score_improves
                && value.median_width_ratio.is_finite()
                && value.median_width_ratio >= 0.0
                && approximately_equal(value.median_width_ratio, raw.median_width_ratio)
                && value.median_width_ratio < 2.0
                && value.holm_reject,
            "held-out temporal-covariance level {level} metrics differ from the frozen scorer"
        );
        p_values.push((level, coverage.p_value));
    }
    p_values.sort_by(|left, right| left.1.total_cmp(&right.1).then_with(|| left.0.cmp(right.0)));
    ensure!(
        p_values
            .iter()
            .enumerate()
            .all(|(rank, (_, p_value))| *p_value <= 0.05 / (3 - rank) as f64),
        "held-out temporal-covariance Holm decisions are inconsistent"
    );
    Ok(())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-12_f64.max(1e-12 * left.abs().max(right.abs()))
}

fn exact_binomial_upper_tail(covered: usize, evaluated: usize, probability: f64) -> Result<f64> {
    ensure!(
        evaluated > 0 && covered <= evaluated && probability > 0.0 && probability < 1.0,
        "held-out binomial inputs are invalid"
    );
    let mut coefficient = 1.0;
    for index in 0..covered {
        coefficient *= (evaluated - index) as f64 / (index + 1) as f64;
    }
    let mut term = coefficient
        * probability.powi(i32::try_from(covered)?)
        * (1.0 - probability).powi(i32::try_from(evaluated - covered)?);
    let mut tail = term;
    for successes in covered..evaluated {
        term *= (evaluated - successes) as f64 / (successes + 1) as f64;
        term *= probability / (1.0 - probability);
        tail += term;
    }
    Ok(tail.min(1.0))
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
            && review.heldout_receipt_sha256 == expected.heldout_receipt_sha256
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
            && manifest.heldout_receipt_sha256 == expected.heldout_receipt_sha256
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
    let result = write_product_transaction(
        output_directory,
        displacement_rasters,
        acquisition_days,
        config,
        factor_directory,
        &promotion,
        &transaction,
        &legacy_velocity_before,
        legacy_sigma_before.as_deref(),
    );
    let receipt = match result {
        Ok(receipt) => receipt,
        Err(error) => {
            if output_directory.join(ROLLBACK_JOURNAL_FILENAME).exists() {
                rollback_incomplete_product(output_directory)?;
            }
            return Err(error);
        }
    };
    complete_publication_after_legacy_check(output_directory, receipt)
}

fn complete_publication_after_legacy_check(
    output_directory: &Path,
    receipt: TemporalCovarianceProductReceipt,
) -> Result<TemporalCovarianceProductReceipt> {
    let journal = read_rollback_journal(output_directory)?;
    if let Err(error) = validate_completed_bundle(output_directory, &journal) {
        rollback_incomplete_product_with_journal(output_directory, &journal)?;
        return Err(error);
    }
    remove_rollback_journal(output_directory)?;
    Ok(receipt)
}

#[allow(clippy::too_many_arguments)]
fn write_product_transaction(
    output_directory: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    config: &TemporalUncertaintyOptions,
    factor_directory: &Path,
    promotion: &TemporalCovariancePromotion,
    transaction: &TemporalProductTransaction,
    legacy_velocity_sha256: &str,
    legacy_sigma_sha256: Option<&str>,
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
        legacy_velocity_sha256,
        legacy_sigma_sha256,
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
    legacy_velocity_sha256: &str,
    legacy_sigma_sha256: Option<&str>,
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
    let stage = create_stage_directory(output_directory, &transaction.ownership_token)?;
    let transaction_result = (|| {
        let admission = admit_combined_working_set(config, acquisition_days.len())?;
        let gdal_cache = ScopedGdalCacheLimit::acquire(admission.gdal_cache_budget_bytes)?;
        let mut working_set = WorkingSetMonitor::new(admission);
        let mut layers = create_layer_writers(
            &stage,
            &scope.velocity_header,
            &scope.factor_metadata.units,
            &transaction.ownership_token,
        )?;
        working_set.observe_gdal_cache(&gdal_cache)?;
        process_factor_blocks(
            &scope.factor_path,
            displacement_rasters,
            &scope.fixed_cube_mask_path,
            acquisition_days,
            config,
            scope.factor_metadata.full_grid,
            &mut layers,
            &mut working_set,
            &gdal_cache,
        )?;
        finalize_layers(&stage, &mut layers, &mut working_set, &gdal_cache)?;
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
            transaction,
            legacy_velocity_sha256,
            legacy_sigma_sha256,
            &mut revalidate,
        )
    })();
    let stage_cleanup = if stage.exists() {
        remove_owned_stage_directory(
            output_directory,
            &stage,
            Some(&transaction.ownership_token),
            || Ok(()),
            |_| Ok(()),
        )
    } else {
        Ok(())
    };
    if transaction_result.is_err() && output_directory.join(ROLLBACK_JOURNAL_FILENAME).exists() {
        rollback_incomplete_product(output_directory)?;
    }
    stage_cleanup?;
    transaction_result
}

struct ProductScope {
    factor_path: PathBuf,
    factor_metadata: dolphin_io::SpatialReferenceCovarianceMetadata,
    velocity_header: dolphin_io::RasterHeader,
    velocity_unit: String,
    input_receipts: Vec<InputRasterReceipt>,
    fixed_cube_paths: Vec<PathBuf>,
    fixed_cube_mask_path: PathBuf,
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
    let fixed_cube_mask_path = output_directory.join(&fixed_cube.validity_mask_raster);
    let fixed_cube_inputs = input_raster_receipts(&fixed_cube_paths)?;
    Ok(ProductScope {
        factor_path,
        factor_metadata,
        velocity_header,
        velocity_unit,
        input_receipts: input_raster_receipts(displacement_rasters)?,
        fixed_cube_paths,
        fixed_cube_mask_path,
        fixed_cube_inputs,
        fixed_cube_semantics,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_factor_blocks(
    factor_path: &Path,
    displacement_rasters: &[PathBuf],
    fixed_cube_mask_path: &Path,
    acquisition_days: &[f64],
    config: &TemporalUncertaintyOptions,
    full_grid: dolphin_io::CovarianceOperatorGrid,
    layers: &mut [ProductLayer],
    working_set: &mut WorkingSetMonitor,
    gdal_cache: &ScopedGdalCacheLimit,
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
        let common_support = read_raster_window::<u8>(fixed_cube_mask_path, output_window)?;
        let values = evaluate_block(
            &read.block,
            &observations,
            common_support.view(),
            acquisition_days,
            &options,
        )?;
        validate_product_value_semantics(&values)?;
        for (layer, value) in layers.iter_mut().zip(values.iter()) {
            layer
                .writer
                .as_mut()
                .context("product writer already finalized")?
                .write_window(output_window, value.view())?;
        }
        working_set.observe_gdal_cache(gdal_cache)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkingSetAdmission {
    factor_block_bytes: u64,
    block_id_bytes: u64,
    displacement_window_bytes: u64,
    output_window_bytes: u64,
    output_write_copy_bytes: u64,
    writer_bookkeeping_bytes: u64,
    temporal_solver_workspace_bytes: u64,
    gdal_cache_budget_bytes: u64,
    total_bytes: u64,
}

struct WorkingSetMonitor {
    admission: WorkingSetAdmission,
    observed_gdal_cache_high_water_bytes: u64,
}

impl WorkingSetMonitor {
    fn new(admission: WorkingSetAdmission) -> Self {
        Self {
            observed_gdal_cache_high_water_bytes: 0,
            admission,
        }
    }

    fn observe_gdal_cache(&mut self, cache: &ScopedGdalCacheLimit) -> Result<()> {
        cache.validate()?;
        let observed = observed_gdal_cache_bytes()?;
        validate_working_set_high_water(&self.admission, observed)?;
        self.observed_gdal_cache_high_water_bytes =
            self.observed_gdal_cache_high_water_bytes.max(observed);
        Ok(())
    }
}

fn admit_combined_working_set(
    config: &TemporalUncertaintyOptions,
    acquisition_count: usize,
) -> Result<WorkingSetAdmission> {
    compose_working_set_admission(config, acquisition_count)
}

fn compose_working_set_admission(
    config: &TemporalUncertaintyOptions,
    acquisition_count: usize,
) -> Result<WorkingSetAdmission> {
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
    let temporal_solver_workspace_bytes = temporal_covariance_workspace_composition(
        acquisition_count,
        TemporalCovarianceOptions::default().bootstrap_replicates,
    )
    .context("temporal-fit workspace composition overflow")?
    .total_bytes;
    let output_write_copy_bytes = output_windows;
    let maximum_block_ids = config
        .block_id_read_cap_bytes
        .checked_div(std::mem::size_of::<u64>() as u64)
        .context("block-ID element size is zero")?;
    ensure!(
        maximum_block_ids > 0,
        "block-ID cap cannot hold one identifier"
    );
    let writer_block_capacity = maximum_block_ids
        .checked_next_power_of_two()
        .map(|capacity| capacity.max(4))
        .context("COG writer block capacity overflow")?;
    let writer_bookkeeping_bytes = writer_block_capacity
        .checked_mul(LAYER_COUNT as u64)
        .and_then(|value| value.checked_mul(std::mem::size_of::<BlockIndices>() as u64))
        .context("COG writer bookkeeping size overflow")?;
    let non_gdal = config
        .factor_block_read_cap_bytes
        .checked_add(config.block_id_read_cap_bytes)
        .and_then(|value| value.checked_add(displacement_windows))
        .and_then(|value| value.checked_add(output_windows))
        .and_then(|value| value.checked_add(output_write_copy_bytes))
        .and_then(|value| value.checked_add(writer_bookkeeping_bytes))
        .and_then(|value| value.checked_add(temporal_solver_workspace_bytes))
        .context("combined temporal working-set size overflow")?;
    let gdal_cache_budget_bytes = COMBINED_WORKING_SET_CAP_BYTES
        .checked_sub(non_gdal)
        .context("non-GDAL temporal working set exceeds the combined cap")?;
    ensure!(
        gdal_cache_budget_bytes > 0,
        "temporal working set leaves no GDAL cache budget"
    );
    Ok(WorkingSetAdmission {
        factor_block_bytes: config.factor_block_read_cap_bytes,
        block_id_bytes: config.block_id_read_cap_bytes,
        displacement_window_bytes: displacement_windows,
        output_window_bytes: output_windows,
        output_write_copy_bytes,
        writer_bookkeeping_bytes,
        temporal_solver_workspace_bytes,
        gdal_cache_budget_bytes,
        total_bytes: COMBINED_WORKING_SET_CAP_BYTES,
    })
}

struct ScopedGdalCacheLimit {
    _lock: MutexGuard<'static, ()>,
    previous_max_bytes: i64,
    configured_max_bytes: i64,
}

impl ScopedGdalCacheLimit {
    fn acquire(max_bytes: u64) -> Result<Self> {
        let lock = match GDAL_CACHE_LIMIT_LOCK.try_lock() {
            Ok(lock) => lock,
            Err(TryLockError::WouldBlock) => {
                anyhow::bail!("another temporal product owns the process-global GDAL cache limit")
            }
            Err(TryLockError::Poisoned(_)) => {
                anyhow::bail!("process-global GDAL cache limit lock is poisoned")
            }
        };
        let configured_max_bytes = i64::try_from(max_bytes)?;
        ensure!(configured_max_bytes > 0, "GDAL cache budget is zero");
        // SAFETY: the process-global mutation is serialized by `GDAL_CACHE_LIMIT_LOCK`.
        let previous_max_bytes = unsafe { gdal_sys::GDALGetCacheMax64() };
        // SAFETY: the positive limit is representable by GDAL's signed byte API.
        unsafe { gdal_sys::GDALSetCacheMax64(configured_max_bytes) };
        let result = Self {
            _lock: lock,
            previous_max_bytes,
            configured_max_bytes,
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<()> {
        // SAFETY: these are read-only process-global cache counters.
        let configured = unsafe { gdal_sys::GDALGetCacheMax64() };
        let used = unsafe { gdal_sys::GDALGetCacheUsed64() };
        ensure!(
            configured == self.configured_max_bytes && used >= 0 && used <= configured,
            "GDAL cache limit changed or current usage exceeds its temporal-product budget"
        );
        Ok(())
    }
}

impl Drop for ScopedGdalCacheLimit {
    fn drop(&mut self) {
        // SAFETY: this guard still owns the process-global cache-limit lock.
        unsafe { gdal_sys::GDALSetCacheMax64(self.previous_max_bytes) };
    }
}

fn observed_gdal_cache_bytes() -> Result<u64> {
    // SAFETY: GDAL exposes this as a read-only process-global cache counter.
    let bytes = unsafe { gdal_sys::GDALGetCacheUsed64() };
    ensure!(bytes >= 0, "GDAL reported a negative cache byte count");
    Ok(u64::try_from(bytes)?)
}

fn validate_working_set_high_water(
    admission: &WorkingSetAdmission,
    observed_gdal_cache_bytes: u64,
) -> Result<()> {
    let non_gdal = admission
        .total_bytes
        .checked_sub(admission.gdal_cache_budget_bytes)
        .context("working-set admission GDAL composition underflow")?;
    let observed = non_gdal
        .checked_add(observed_gdal_cache_bytes)
        .context("observed temporal working-set size overflow")?;
    ensure!(
        observed_gdal_cache_bytes <= admission.gdal_cache_budget_bytes
            && observed <= COMBINED_WORKING_SET_CAP_BYTES,
        "observed temporal working set {observed} exceeds cap {COMBINED_WORKING_SET_CAP_BYTES}"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn publish_product_receipt(
    output_directory: &Path,
    stage: &Path,
    displacement_rasters: &[PathBuf],
    acquisition_days: &[f64],
    scope: ProductScope,
    promotion: &TemporalCovariancePromotion,
    transaction: &TemporalProductTransaction,
    legacy_velocity_sha256: &str,
    legacy_sigma_sha256: Option<&str>,
    revalidate: &mut impl FnMut() -> Result<TemporalCovariancePromotion>,
) -> Result<TemporalCovarianceProductReceipt> {
    let original_fixed_cube_receipt = read_bounded(
        &output_directory.join("fixed_cube_receipt.json"),
        1024 * 1024,
    )?;
    let staged_products = product_receipts(stage)?;
    let expected_products = owned_artifact_receipts(&staged_products)?;
    let mut journal = ProductRollbackJournal {
        schema: ROLLBACK_JOURNAL_SCHEMA.to_owned(),
        ownership_token: transaction.ownership_token.clone(),
        original_fixed_cube_receipt,
        legacy_velocity_sha256: legacy_velocity_sha256.to_owned(),
        legacy_sigma_sha256: legacy_sigma_sha256.map(str::to_owned),
        promotion_manifest_sha256: promotion.manifest_sha256.clone(),
        semantic_validation: scope.fixed_cube_semantics.clone(),
        product_grid: ProductGridReceipt {
            rows: scope.velocity_header.shape.0,
            cols: scope.velocity_header.shape.1,
            geotransform: scope.velocity_header.geotransform,
            epsg: scope.velocity_header.epsg,
            velocity_unit: scope.velocity_unit.clone(),
            process_variance_unit: squared_displacement_unit(&scope.factor_metadata.units)?
                .to_owned(),
        },
        expected_products,
        installed_artifacts: Vec::new(),
        stage_directory: stage
            .file_name()
            .context("temporal product stage has no filename")?
            .to_string_lossy()
            .into_owned(),
        expected_provenance_sha256: None,
        expected_fixed_receipt_sha256: None,
        rollback_state: ProductRollbackState::Active,
        collision_artifacts: Vec::new(),
    };
    persist_rollback_journal(output_directory, &journal, true)?;
    publish_layers_no_replace(output_directory, stage, &mut journal)?;
    verify_final_cogs(
        output_directory,
        &staged_products,
        &scope.velocity_header,
        &scope.factor_metadata.units,
        &transaction.ownership_token,
    )?;
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
        &transaction.ownership_token,
    );
    let provenance_path = output_directory.join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME);
    let provenance_scratch = stage.join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME);
    std::fs::write(&provenance_scratch, serde_json::to_vec_pretty(&provenance)?)?;
    File::open(&provenance_scratch)?.sync_all()?;
    let provenance_sha256 = sha256_file(&provenance_scratch)?;
    let fixed_scratch_name = format!(
        ".fixed-cube-receipt-temporal-{}",
        &transaction.ownership_token[..32]
    );
    let fixed_scratch = output_directory.join(&fixed_scratch_name);
    let fixed_scratch_marker = write_transaction_artifact_marker(
        output_directory,
        &fixed_scratch_name,
        &transaction.ownership_token,
    )?;
    let expected_fixed_receipt = promoted_fixed_cube_receipt_bytes(
        &journal.original_fixed_cube_receipt,
        &scope.fixed_cube_semantics,
        &corrected_velocity_sha256,
        &corrected_sigma_sha256,
        &provenance_sha256,
        &promotion.manifest_sha256,
    )?;
    journal.expected_provenance_sha256 = Some(provenance_sha256.clone());
    journal.installed_artifacts.push(OwnedArtifactReceipt {
        name: TEMPORAL_INFERENCE_PROVENANCE_FILENAME.to_owned(),
        sha256: provenance_sha256.clone(),
    });
    journal_expected_fixed_receipt_before_replace(
        output_directory,
        &mut journal,
        &expected_fixed_receipt,
    )?;
    let promote = promote_fixed_cube_receipt(
        output_directory,
        &provenance_scratch,
        &fixed_scratch,
        scope.fixed_cube_semantics.clone(),
        corrected_velocity_sha256.clone(),
        corrected_sigma_sha256.clone(),
        provenance_sha256.clone(),
        promotion.manifest_sha256.clone(),
    );
    ensure!(
        fixed_scratch_marker.exists()
            && transaction_marker_is_owned(
                &fixed_scratch_marker,
                output_directory,
                &fixed_scratch_name,
                Some(&transaction.ownership_token),
            )?,
        "fixed-cube receipt scratch ownership changed before cleanup"
    );
    if promote.is_err() && fixed_scratch.exists() {
        std::fs::remove_file(&fixed_scratch)?;
    }
    std::fs::remove_file(&fixed_scratch_marker)?;
    File::open(output_directory)?.sync_all()?;
    promote?;
    let promoted_fixed_receipt_sha256 =
        sha256_file(&output_directory.join("fixed_cube_receipt.json"))?;
    ensure!(
        journal.expected_fixed_receipt_sha256.as_deref()
            == Some(promoted_fixed_receipt_sha256.as_str()),
        "promoted fixed-cube receipt differs from its durable rollback identity"
    );
    verify_final_cogs(
        output_directory,
        &staged_products,
        &scope.velocity_header,
        &scope.factor_metadata.units,
        &transaction.ownership_token,
    )?;
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
    let receipt = TemporalCovarianceProductReceipt {
        corrected_velocity_sha256,
        corrected_sigma_sha256,
        provenance_sha256,
        promotion_manifest_sha256: promotion.manifest_sha256.clone(),
    };
    validate_completed_bundle(output_directory, &journal)?;
    Ok(receipt)
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
            && receipt.acquisition_days_sha256 == fixed_cube_days_sha256(acquisition_days)
            && receipt.rows == header.shape.0
            && receipt.cols == header.shape.1
            && receipt.geotransform == header.geotransform
            && receipt.epsg == header.epsg
            && receipt.reference_point == Some(reference)
            && receipt.velocity_estimator == "linear_post_gauge_unit_precision"
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
            los_header.nodata.is_some_and(f64::is_nan)
                && los_header
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
                && los_header.metadata.get("UNITTYPE").map(String::as_str) == Some("unitless")
                && los_header.metadata.get("RASTER_ROLE").map(String::as_str)
                    == Some("fixed_cube_run_geometry"),
            "fixed-cube LOS metadata is invalid: {}",
            path.display()
        );
    }
    let mut semantics = validate_fixed_cube_pixels(
        &directory.join(&receipt.velocity_raster),
        &mask_path,
        &los_paths,
        velocity_header.shape,
        receipt.valid_pixels,
    )?;
    validate_geometry_provenance(&directory.join(&receipt.geometry_provenance), receipt)?;
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
    velocity_path: &Path,
    mask_path: &Path,
    los_paths: &[PathBuf; 3],
    shape: (usize, usize),
    expected_valid_pixels: usize,
) -> Result<crate::fixed_cube::FixedCubeSemanticValidation> {
    let mut valid_pixels = 0usize;
    let mut maximum_los_norm_error = 0.0_f32;
    let mut minimum_los_up = f32::INFINITY;
    let bytes_per_pixel = std::mem::size_of::<u8>() + 4 * std::mem::size_of::<f32>();
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
            let velocity = read_raster_window::<f32>(velocity_path, window)?;
            let east = read_raster_window::<f32>(&los_paths[0], window)?;
            let north = read_raster_window::<f32>(&los_paths[1], window)?;
            let up = read_raster_window::<f32>(&los_paths[2], window)?;
            for ((((mask, velocity), east), north), up) in
                mask.iter().zip(&velocity).zip(&east).zip(&north).zip(&up)
            {
                ensure!(
                    *mask <= 1,
                    "fixed-cube validity mask contains a value other than 0 or 1"
                );
                valid_pixels = valid_pixels
                    .checked_add(usize::from(*mask))
                    .context("fixed-cube valid-pixel count overflow")?;
                ensure!(
                    (*mask == 1) == velocity.is_finite(),
                    "fixed-cube validity mask disagrees with finite velocity support"
                );
                if *mask == 1 {
                    let norm_error =
                        (east.mul_add(*east, north.mul_add(*north, up * up)) - 1.0).abs();
                    ensure!(
                        east.is_finite()
                            && north.is_finite()
                            && up.is_finite()
                            && *up > 0.0
                            && norm_error <= 5e-4,
                        "valid fixed-cube LOS vector is nonfinite, non-unit, or sign-inconsistent"
                    );
                    maximum_los_norm_error = maximum_los_norm_error.max(norm_error);
                    minimum_los_up = minimum_los_up.min(*up);
                } else {
                    ensure!(
                        east.is_nan() && north.is_nan() && up.is_nan(),
                        "invalid fixed-cube pixel retains unmasked LOS geometry"
                    );
                }
            }
        }
    }
    ensure!(
        valid_pixels == expected_valid_pixels && (valid_pixels == 0 || minimum_los_up.is_finite()),
        "fixed-cube validity count differs from receipt"
    );
    if valid_pixels == 0 {
        minimum_los_up = 0.0;
    }
    Ok(crate::fixed_cube::FixedCubeSemanticValidation {
        observed_valid_pixels: valid_pixels,
        maximum_los_norm_error,
        minimum_los_up,
        los_sign_convention: "ground_to_sensor_positive_toward_sensor".to_owned(),
        geometry_source: "CSLC-S1-STATIC".to_owned(),
        geometry_provenance_status: "pending".to_owned(),
    })
}

fn validate_geometry_provenance(
    path: &Path,
    receipt: &crate::fixed_cube::FixedCubeReceipt,
) -> Result<()> {
    let provenance: crate::provenance::GeometryProvenance =
        serde_json::from_slice(&read_bounded(path, 1024 * 1024)?)?;
    let (orbit_files, orbit_keys, _) = sourced_geometry_field(&provenance, "orbit_direction")?;
    let (heading_files, heading_keys, _) = sourced_geometry_field(&provenance, "heading_deg")?;
    let (incidence_files, incidence_keys, incidence_method) =
        sourced_geometry_field(&provenance, "incidence_angle_deg")?;
    let coverage = provenance
        .input_coverage
        .as_ref()
        .context("geometry provenance lacks fixed-cube input coverage")?;
    let output_pixels = receipt
        .rows
        .checked_mul(receipt.cols)
        .context("fixed-cube output-pixel count overflow")?;
    validate_input_coverage(
        coverage,
        receipt.acquisition_days.len(),
        output_pixels,
        receipt.valid_pixels,
    )?;
    ensure!(
        provenance.schema == "dolphinrust-geometry-provenance/4"
            && provenance.method_version == "4.0.0"
            && provenance.decomposition_geometry_complete
            && provenance
                .orbit_direction
                .as_deref()
                .is_some_and(|direction| matches!(direction, "ascending" | "descending"))
            && provenance.heading_deg.is_some_and(f64::is_finite)
            && provenance.incidence_angle_deg.is_some_and(f64::is_finite)
            && !orbit_files.is_empty()
            && orbit_keys
                .iter()
                .any(|key| key == "/identification/orbit_pass_direction")
            && !heading_files.is_empty()
            && heading_keys
                .iter()
                .any(|key| key.starts_with("/metadata/orbit/"))
            && !incidence_files.is_empty()
            && incidence_keys.iter().any(|key| key == "/data/los_east")
            && incidence_keys.iter().any(|key| key == "/data/los_north")
            && incidence_method.contains("los_up")
            && provenance.geometry_provenance.method_version == provenance.method_version,
        "geometry provenance is incomplete, unsourced, or fallback-derived"
    );
    Ok(())
}

fn validate_input_coverage(
    coverage: &crate::provenance::InputCoverageProvenance,
    acquisition_count: usize,
    output_pixels: usize,
    valid_pixels: usize,
) -> Result<()> {
    ensure!(
        output_pixels > 0
            && valid_pixels <= output_pixels
            && coverage.policy_version == crate::provenance::INPUT_COVERAGE_POLICY_VERSION
            && coverage.output_pixels == output_pixels
            && coverage.valid_pixels == valid_pixels
            && !coverage.bursts.is_empty(),
        "fixed-cube input coverage is empty or scope-mismatched"
    );
    let mut total_tiles = 0usize;
    let mut linked_tiles = 0usize;
    let mut nodata_tiles = 0usize;
    for (expected_ordinal, burst) in coverage.bursts.iter().enumerate() {
        let burst_total = burst
            .linked_tiles
            .checked_add(burst.nodata_tiles)
            .context("fixed-cube burst tile count overflow")?;
        ensure!(
            burst.burst_index == expected_ordinal
                && burst.acquisition_count == acquisition_count
                && burst.total_tiles == burst_total,
            "fixed-cube burst coverage is not ordinal-complete or internally consistent"
        );
        total_tiles = total_tiles
            .checked_add(burst.total_tiles)
            .context("fixed-cube total tile count overflow")?;
        linked_tiles = linked_tiles
            .checked_add(burst.linked_tiles)
            .context("fixed-cube linked tile count overflow")?;
        nodata_tiles = nodata_tiles
            .checked_add(burst.nodata_tiles)
            .context("fixed-cube nodata tile count overflow")?;
    }
    let declared_total = coverage
        .linked_tiles
        .checked_add(coverage.nodata_tiles)
        .context("fixed-cube aggregate tile count overflow")?;
    ensure!(
        coverage.total_tiles > 0
            && coverage.linked_tiles > 0
            && coverage.total_tiles == declared_total
            && total_tiles == coverage.total_tiles
            && linked_tiles == coverage.linked_tiles
            && nodata_tiles == coverage.nodata_tiles
            && (coverage.valid_fraction - valid_pixels as f64 / output_pixels as f64).abs()
                <= f64::EPSILON,
        "fixed-cube input coverage totals are invalid"
    );
    Ok(())
}

fn sourced_geometry_field<'a>(
    provenance: &'a crate::provenance::GeometryProvenance,
    field: &str,
) -> Result<(&'a [String], &'a [String], &'a str)> {
    match provenance.geometry_provenance.fields.get(field) {
        Some(crate::provenance::FieldProvenance::Sourced {
            source_files,
            source_keys,
            method,
            ..
        }) => Ok((source_files, source_keys, method)),
        _ => anyhow::bail!("geometry provenance lacks sourced {field}"),
    }
}

fn fixed_cube_days_sha256(days: &[f64]) -> String {
    let bytes = days
        .iter()
        .flat_map(|day| day.to_le_bytes())
        .collect::<Vec<_>>();
    format!("sha256:{}", sha256(&bytes))
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
    spatial_units: &str,
    ownership_token: &str,
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
            let unit = product_layer_unit(index, velocity_unit, spatial_units)?;
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
                    ("TRANSACTION_OWNERSHIP_TOKEN", ownership_token),
                ],
            )?;
            Ok(ProductLayer {
                name,
                writer: Some(writer),
            })
        })
        .collect()
}

fn squared_displacement_unit(spatial_units: &str) -> Result<&'static str> {
    match spatial_units {
        "radians" => Ok("rad^2"),
        "meters" => Ok("m^2"),
        "millimeters" => Ok("mm^2"),
        _ => anyhow::bail!("unsupported spatial covariance unit"),
    }
}

fn product_layer_unit<'a>(
    index: usize,
    velocity_unit: &'a str,
    spatial_units: &str,
) -> Result<&'a str> {
    if index < 2 {
        Ok(velocity_unit)
    } else if index == 10 {
        Ok(squared_displacement_unit(spatial_units)?)
    } else {
        Ok("1")
    }
}

fn finalize_layers(
    stage: &Path,
    layers: &mut [ProductLayer],
    working_set: &mut WorkingSetMonitor,
    gdal_cache: &ScopedGdalCacheLimit,
) -> Result<()> {
    for layer in layers {
        let writer = layer
            .writer
            .take()
            .context("product writer already finalized")?;
        writer.finalize(&stage.join(layer.name))?;
        working_set.observe_gdal_cache(gdal_cache)?;
    }
    Ok(())
}

fn publish_layers_no_replace(
    output_directory: &Path,
    stage: &Path,
    journal: &mut ProductRollbackJournal,
) -> Result<()> {
    for (name, _) in PRODUCT_LAYERS {
        let expected = journal
            .expected_products
            .iter()
            .find(|receipt| receipt.name == name)
            .cloned()
            .with_context(|| format!("journal is missing expected product {name}"))?;
        journal.installed_artifacts.push(expected);
        persist_rollback_journal(output_directory, journal, false)?;
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
    common_support: ndarray::ArrayView2<'_, u8>,
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
            && observations.iter().all(|values| values.dim() == shape)
            && common_support.dim() == shape,
        "displacement windows differ from factor block"
    );
    let mut output: [Array2<f32>; LAYER_COUNT] =
        std::array::from_fn(|_| Array2::from_elem(shape, f32::NAN));
    for target in 0..target_count {
        let support = common_support
            .as_slice()
            .context("fixed-cube common-support mask is not contiguous")?[target];
        ensure!(support <= 1, "fixed-cube common-support mask is not binary");
        if support == 0 {
            output[2]
                .as_slice_mut()
                .context("status layer is not contiguous")?[target] = 2_000.0;
            continue;
        }
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
        ensure!(
            observations
                .iter()
                .all(|values| values[(row, col)].is_finite()),
            "fixed-cube common support contains a missing displacement epoch"
        );
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
        set_checked(
            &mut layers[0],
            target,
            selected
                .slope_per_year
                .context("evaluated slope is absent")?,
            "selected velocity",
        )?;
        set_checked(
            &mut layers[1],
            target,
            selected
                .standard_error_per_year
                .context("evaluated standard error is absent")?,
            "corrected standard error",
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
    set_checked(
        &mut layers[5],
        target,
        selected.valid_date_count as f64,
        "valid date count",
    )?;
    set_checked(&mut layers[6], target, selected.rank as f64, "design rank")?;
    set_checked(
        &mut layers[7],
        target,
        selected.degrees_of_freedom as f64,
        "degrees of freedom",
    )?;
    set_optional(&mut layers[8], target, selected.raw_rho, "raw rho")?;
    set_optional(&mut layers[9], target, selected.fitted_rho, "fitted rho")?;
    set_optional(
        &mut layers[10],
        target,
        selected.fitted_process_variance,
        "fitted process variance",
    )?;
    set_optional(
        &mut layers[11],
        target,
        selected.condition_number,
        "condition number",
    )?;
    set_checked(
        &mut layers[12],
        target,
        selected.bootstrap_attempts as f64,
        "bootstrap attempts",
    )?;
    set_checked(
        &mut layers[13],
        target,
        selected.bootstrap_successes as f64,
        "bootstrap successes",
    )?;
    Ok(())
}

fn set_optional(
    layer: &mut Array2<f32>,
    target: usize,
    value: Option<f64>,
    field: &str,
) -> Result<()> {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        layer
            .as_slice_mut()
            .context("temporal diagnostic layer is not contiguous")?[target] =
            checked_f32(value, field)?;
    }
    Ok(())
}

fn set_checked(layer: &mut Array2<f32>, target: usize, value: f64, field: &str) -> Result<()> {
    layer
        .as_slice_mut()
        .context("temporal product layer is not contiguous")?[target] = checked_f32(value, field)?;
    Ok(())
}

fn checked_f32(value: f64, field: &str) -> Result<f32> {
    ensure!(
        value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX),
        "{field} cannot be represented as finite f32"
    );
    Ok(value as f32)
}

fn validate_product_value_semantics(layers: &[Array2<f32>; LAYER_COUNT]) -> Result<()> {
    let shape = layers[0].dim();
    ensure!(
        layers.iter().all(|layer| layer.dim() == shape),
        "temporal product layers have different shapes"
    );
    let slices = layers
        .iter()
        .map(|layer| {
            layer
                .as_slice()
                .context("temporal product semantic layer is not contiguous")
        })
        .collect::<Result<Vec<_>>>()?;
    for target in 0..shape
        .0
        .checked_mul(shape.1)
        .context("product shape overflow")?
    {
        ensure!(
            slices.iter().all(|layer| !layer[target].is_infinite()),
            "temporal product contains an infinite f32 value"
        );
        let selection = slices[2][target];
        ensure!(
            selection.is_finite()
                && selection.fract() == 0.0
                && ((0.0..=6.0).contains(&selection)
                    || (1_000.0..2_000.0).contains(&selection)
                    || selection == 2_000.0),
            "temporal selection status is invalid"
        );
        if selection == 0.0 {
            ensure!(
                slices[0][target].is_finite()
                    && slices[1][target].is_finite()
                    && slices[1][target] > 0.0,
                "evaluated temporal pixel lacks finite velocity and positive standard error"
            );
        } else {
            ensure!(
                slices[0][target].is_nan() && slices[1][target].is_nan(),
                "abstained temporal pixel retains an inferential value"
            );
        }
        for &index in &[3_usize, 4, 5, 6, 7, 12, 13] {
            let value = slices[index][target];
            ensure!(
                value.is_nan() || (value.is_finite() && value >= 0.0 && value.fract() == 0.0),
                "temporal status/count diagnostic is not a nonnegative integer or nodata"
            );
        }
        if selection < 1_000.0 {
            ensure!(
                slices[3][target].is_finite()
                    && slices[4][target].is_finite()
                    && slices[5][target].is_finite()
                    && slices[6][target].is_finite()
                    && slices[7][target].is_finite()
                    && slices[12][target].is_finite()
                    && slices[13][target].is_finite()
                    && slices[13][target] <= slices[12][target],
                "temporal estimator status/count diagnostics are incomplete"
            );
        } else {
            ensure!(
                (3..LAYER_COUNT).all(|index| slices[index][target].is_nan()),
                "support/factor abstention retains estimator diagnostics"
            );
        }
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
    transaction_ownership_token: &'a str,
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
        transaction_ownership_token: &'a str,
    ) -> Self {
        let header = &scope.velocity_header;
        let factor = &scope.factor_metadata;
        let day_bytes = days
            .iter()
            .flat_map(|day| day.to_le_bytes())
            .collect::<Vec<_>>();
        Self {
            schema: PRODUCT_SCHEMA,
            transaction_ownership_token,
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
    spatial_units: &str,
    ownership_token: &str,
) -> Result<()> {
    ensure!(
        expected.len() == PRODUCT_LAYERS.len(),
        "temporal product receipt count is incomplete"
    );
    let velocity_unit = grid
        .metadata
        .get("UNITTYPE")
        .context("legacy velocity raster is missing UNITTYPE")?;
    for (index, ((name, role), expected_receipt)) in PRODUCT_LAYERS.iter().zip(expected).enumerate()
    {
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
                && header.metadata.get("UNITTYPE").map(String::as_str)
                    == Some(product_layer_unit(index, velocity_unit, spatial_units)?)
                && header
                    .metadata
                    .get("TEMPORAL_ESTIMATOR")
                    .map(String::as_str)
                    == Some(COMPLETE_REFIT_BOOTSTRAP_METHOD)
                && header
                    .metadata
                    .get("TRANSACTION_OWNERSHIP_TOKEN")
                    .map(String::as_str)
                    == Some(ownership_token),
            "published temporal product header changed: {name}"
        );
    }
    verify_persisted_product_value_semantics(directory, grid.shape)?;
    Ok(())
}

fn verify_persisted_product_value_semantics(directory: &Path, shape: (usize, usize)) -> Result<()> {
    for row_start in (0..shape.0).step_by(256) {
        for col_start in (0..shape.1).step_by(256) {
            let window = BlockIndices {
                row_start,
                row_stop: (row_start + 256).min(shape.0),
                col_start,
                col_stop: (col_start + 256).min(shape.1),
            };
            let layers = PRODUCT_LAYERS
                .each_ref()
                .map(|(name, _)| read_raster_window::<f32>(&directory.join(name), window))
                .into_iter()
                .collect::<dolphin_io::Result<Vec<_>>>()?;
            let layers: [Array2<f32>; LAYER_COUNT] = layers
                .try_into()
                .map_err(|_| anyhow::anyhow!("temporal product layer count changed"))?;
            validate_product_value_semantics(&layers)?;
        }
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

fn owned_artifact_receipts(receipts: &[InputRasterReceipt]) -> Result<Vec<OwnedArtifactReceipt>> {
    receipts
        .iter()
        .map(|receipt| {
            let name = Path::new(&receipt.path)
                .file_name()
                .context("product receipt path has no filename")?
                .to_string_lossy()
                .into_owned();
            Ok(OwnedArtifactReceipt {
                name,
                sha256: receipt.sha256.clone(),
            })
        })
        .collect()
}

fn persist_rollback_journal(
    directory: &Path,
    journal: &ProductRollbackJournal,
    create: bool,
) -> Result<()> {
    let scratch = directory.join(format!(
        ".temporal-product-journal-{}-{}",
        std::process::id(),
        NEXT_TRANSACTION_FILE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let scratch_name = scratch
        .file_name()
        .context("temporal journal scratch has no filename")?
        .to_string_lossy();
    let marker =
        write_transaction_artifact_marker(directory, &scratch_name, &journal.ownership_token)?;
    let persist = (|| {
        std::fs::write(&scratch, serde_json::to_vec(&journal)?)?;
        File::open(&scratch)?.sync_all()?;
        let path = directory.join(ROLLBACK_JOURNAL_FILENAME);
        if create {
            install_no_replace(&scratch, &path)
        } else {
            std::fs::rename(&scratch, &path)?;
            File::open(directory)?.sync_all()?;
            Ok(())
        }
    })();
    if persist.is_ok() {
        std::fs::remove_file(marker)?;
        File::open(directory)?.sync_all()?;
    }
    persist
}

fn journal_expected_fixed_receipt_before_replace(
    directory: &Path,
    journal: &mut ProductRollbackJournal,
    expected_receipt_bytes: &[u8],
) -> Result<()> {
    journal.expected_fixed_receipt_sha256 = Some(sha256(expected_receipt_bytes));
    persist_rollback_journal(directory, journal, false)
}

fn promoted_fixed_cube_receipt_bytes(
    original_receipt_bytes: &[u8],
    semantic_validation: &crate::fixed_cube::FixedCubeSemanticValidation,
    corrected_velocity_sha256: &str,
    corrected_sigma_sha256: &str,
    provenance_sha256: &str,
    promotion_manifest_sha256: &str,
) -> Result<Vec<u8>> {
    let mut receipt: crate::fixed_cube::FixedCubeReceipt =
        serde_json::from_slice(original_receipt_bytes)?;
    ensure!(
        receipt.contract_version == "fixed-cube-v1"
            && receipt.inference_status == "conditional_only"
            && receipt.corrected_velocity_raster.is_none()
            && receipt.corrected_sigma_raster.is_none()
            && receipt.semantic_validation.is_none(),
        "fixed-cube receipt is not eligible for temporal-inference promotion"
    );
    receipt.inference_status = "calibrated_scope_match".to_owned();
    receipt.corrected_velocity_raster = Some("velocity_temporal_gls.tif".to_owned());
    receipt.corrected_sigma_raster = Some("velocity_sigma_corrected.tif".to_owned());
    receipt.corrected_velocity_sha256 = Some(corrected_velocity_sha256.to_owned());
    receipt.corrected_sigma_sha256 = Some(corrected_sigma_sha256.to_owned());
    receipt.inference_provenance = Some(TEMPORAL_INFERENCE_PROVENANCE_FILENAME.to_owned());
    receipt.inference_provenance_sha256 = Some(provenance_sha256.to_owned());
    receipt.temporal_promotion_manifest_sha256 = Some(promotion_manifest_sha256.to_owned());
    receipt.semantic_validation = Some(semantic_validation.clone());
    Ok(serde_json::to_vec_pretty(&receipt)?)
}

fn read_rollback_journal(directory: &Path) -> Result<ProductRollbackJournal> {
    let journal: ProductRollbackJournal = serde_json::from_slice(&read_bounded(
        &directory.join(ROLLBACK_JOURNAL_FILENAME),
        JSON_CAP,
    )?)?;
    ensure!(
        journal.schema == ROLLBACK_JOURNAL_SCHEMA
            && journal.expected_products.len() == PRODUCT_LAYERS.len()
            && journal
                .stage_directory
                .starts_with(".temporal-inference-stage-")
            && Path::new(&journal.stage_directory)
                .file_name()
                .is_some_and(|name| { name.to_string_lossy() == journal.stage_directory })
            && match journal.rollback_state {
                ProductRollbackState::Active => journal.collision_artifacts.is_empty(),
                ProductRollbackState::BlockedUnownedCollision => {
                    !journal.collision_artifacts.is_empty()
                }
            },
        "temporal product rollback journal is malformed or unsupported"
    );
    Ok(journal)
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
    if directory.join(ROLLBACK_JOURNAL_FILENAME).exists() {
        let journal = read_rollback_journal(directory)?;
        if validate_completed_bundle(directory, &journal).is_ok() {
            remove_rollback_journal(directory)?;
        } else {
            rollback_incomplete_product_with_journal(directory, &journal)?;
        }
    }
    Ok(())
}

fn product_directory_sha256(directory: &Path) -> Result<String> {
    let canonical = std::fs::canonicalize(directory)?;
    Ok(sha256(canonical.to_string_lossy().as_bytes()))
}

fn scratch_marker_path(directory: &Path, artifact_name: &str) -> PathBuf {
    directory.join(format!("{artifact_name}.owner.json"))
}

fn write_transaction_artifact_marker(
    directory: &Path,
    artifact_name: &str,
    ownership_token: &str,
) -> Result<PathBuf> {
    let path = scratch_marker_path(directory, artifact_name);
    write_transaction_marker(&path, directory, artifact_name, ownership_token)?;
    File::open(directory)?.sync_all()?;
    Ok(path)
}

fn write_transaction_marker(
    path: &Path,
    directory: &Path,
    artifact_name: &str,
    ownership_token: &str,
) -> Result<()> {
    ensure!(
        !ownership_token.is_empty()
            && ownership_token.len() <= 256
            && Path::new(artifact_name)
                .file_name()
                .is_some_and(|name| name == artifact_name),
        "temporal transaction ownership identity is invalid"
    );
    let marker = TransactionArtifactMarker {
        schema: TRANSACTION_ARTIFACT_MARKER_SCHEMA.to_owned(),
        ownership_token: ownership_token.to_owned(),
        product_directory_sha256: product_directory_sha256(directory)?,
        artifact_name: artifact_name.to_owned(),
    };
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    std::io::Write::write_all(&mut file, &serde_json::to_vec(&marker)?)?;
    file.sync_all()?;
    Ok(())
}

fn transaction_marker_is_owned(
    path: &Path,
    directory: &Path,
    artifact_name: &str,
    expected_ownership_token: Option<&str>,
) -> Result<bool> {
    let bytes = match read_bounded(path, 64 * 1024) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(false),
    };
    let marker: TransactionArtifactMarker = match serde_json::from_slice(&bytes) {
        Ok(marker) => marker,
        Err(_) => return Ok(false),
    };
    Ok(marker.schema == TRANSACTION_ARTIFACT_MARKER_SCHEMA
        && !marker.ownership_token.is_empty()
        && marker.ownership_token.len() <= 256
        && expected_ownership_token.is_none_or(|token| marker.ownership_token == token)
        && marker.product_directory_sha256 == product_directory_sha256(directory)?
        && marker.artifact_name == artifact_name)
}

#[cfg(unix)]
fn cstring_component(name: &str) -> Result<CString> {
    ensure!(
        Path::new(name)
            .file_name()
            .is_some_and(|component| component == name),
        "temporal cleanup name is not one path component"
    );
    Ok(CString::new(name.as_bytes())?)
}

#[cfg(unix)]
fn open_directory_path(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?)
}

#[cfg(unix)]
fn open_directory_at(parent: RawFd, name: &CStr) -> Result<File> {
    // SAFETY: `parent` is a live directory descriptor and `name` is NUL-terminated.
    let fd = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    ensure!(
        fd >= 0,
        "opening cleanup directory failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: `fd` was returned as a new owned descriptor by `openat`.
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(unix)]
#[allow(clippy::unnecessary_cast)]
fn stat_directory_identity(stat: &libc::stat) -> DirectoryIdentity {
    DirectoryIdentity {
        device: stat.st_dev as u64,
        inode: stat.st_ino as u64,
    }
}

#[cfg(unix)]
fn descriptor_identity(file: &File) -> Result<DirectoryIdentity> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `stat` points to writable storage and `file` owns a live descriptor.
    let result = unsafe { libc::fstat(file.as_raw_fd(), stat.as_mut_ptr()) };
    ensure!(
        result == 0,
        "reading cleanup directory identity failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: successful `fstat` initialized `stat`.
    let stat = unsafe { stat.assume_init() };
    ensure!(
        stat.st_mode & libc::S_IFMT == libc::S_IFDIR,
        "cleanup descriptor is not a directory"
    );
    Ok(stat_directory_identity(&stat))
}

#[cfg(unix)]
fn read_bounded_at(parent: RawFd, name: &CStr, cap: u64) -> Result<Vec<u8>> {
    // SAFETY: `parent` is live and `name` is NUL-terminated.
    let fd = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    ensure!(
        fd >= 0,
        "opening cleanup receipt failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: `fd` is newly owned.
    let mut file = unsafe { File::from_raw_fd(fd) };
    let before = file.metadata()?;
    ensure!(
        before.is_file() && before.len() <= cap,
        "cleanup receipt exceeds byte cap"
    );
    let mut bytes = Vec::with_capacity(usize::try_from(before.len())?);
    file.by_ref()
        .take(cap.checked_add(1).context("cleanup receipt cap overflow")?)
        .read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    ensure!(
        bytes.len() as u64 <= cap
            && before.len() == after.len()
            && before.modified()? == after.modified()?,
        "cleanup receipt changed while read"
    );
    Ok(bytes)
}

#[cfg(unix)]
fn write_new_at(parent: RawFd, name: &CStr, bytes: &[u8]) -> Result<()> {
    // SAFETY: `parent` is live and `name` is NUL-terminated.
    let fd = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    ensure!(
        fd >= 0,
        "creating cleanup receipt failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: `fd` is newly owned.
    let mut file = unsafe { File::from_raw_fd(fd) };
    std::io::Write::write_all(&mut file, bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(unix)]
fn directory_entry_names(directory: RawFd) -> Result<Vec<CString>> {
    let current = CString::new(".")?;
    // SAFETY: `directory` is live; `openat` creates an independent file description.
    let duplicated = unsafe {
        libc::openat(
            directory,
            current.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    ensure!(
        duplicated >= 0,
        "duplicating cleanup descriptor failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: `duplicated` is a directory descriptor owned by `fdopendir` on success.
    let stream = unsafe { libc::fdopendir(duplicated) };
    if stream.is_null() {
        // SAFETY: `fdopendir` failed and did not take ownership.
        unsafe { libc::close(duplicated) };
        anyhow::bail!(
            "opening cleanup directory stream failed: {}",
            std::io::Error::last_os_error()
        );
    }
    let mut names = Vec::new();
    loop {
        // SAFETY: `stream` remains valid until `closedir` below.
        let entry = unsafe { libc::readdir(stream) };
        if entry.is_null() {
            break;
        }
        // SAFETY: POSIX guarantees a NUL-terminated `d_name` in a live dirent.
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
        if name.to_bytes() != b"." && name.to_bytes() != b".." {
            names.push(CString::new(name.to_bytes())?);
        }
    }
    // SAFETY: `stream` is live and owns `duplicated`.
    ensure!(
        unsafe { libc::closedir(stream) } == 0,
        "closing cleanup directory stream failed"
    );
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    Ok(names)
}

#[cfg(unix)]
fn transaction_marker_bytes_are_owned(
    bytes: &[u8],
    directory_sha256: &str,
    artifact_name: &str,
    ownership_token: &str,
) -> bool {
    serde_json::from_slice::<TransactionArtifactMarker>(bytes).is_ok_and(|marker| {
        marker.schema == TRANSACTION_ARTIFACT_MARKER_SCHEMA
            && marker.ownership_token == ownership_token
            && marker.product_directory_sha256 == directory_sha256
            && marker.artifact_name == artifact_name
    })
}

#[cfg(unix)]
struct VerifiedCleanupQuarantine {
    product_directory: File,
    outer: File,
    inner: File,
    marker: CleanupQuarantineMarker,
}

#[cfg(unix)]
fn open_verified_cleanup_quarantine(
    directory: &Path,
    cleanup_name: &str,
    expected_ownership_token: Option<&str>,
) -> Result<VerifiedCleanupQuarantine> {
    let product_directory = open_directory_path(directory)?;
    let cleanup_name_c = cstring_component(cleanup_name)?;
    let outer = open_directory_at(product_directory.as_raw_fd(), &cleanup_name_c)?;
    let marker_name = CString::new(TRANSACTION_ARTIFACT_MARKER_FILENAME)?;
    let marker: CleanupQuarantineMarker = serde_json::from_slice(&read_bounded_at(
        outer.as_raw_fd(),
        &marker_name,
        64 * 1024,
    )?)?;
    ensure!(
        marker.schema == CLEANUP_QUARANTINE_MARKER_SCHEMA
            && marker.artifact_name == cleanup_name
            && marker.product_directory_sha256 == product_directory_sha256(directory)?
            && !marker.ownership_token.is_empty()
            && marker.ownership_token.len() <= 256
            && expected_ownership_token.is_none_or(|expected| marker.ownership_token == expected)
            && marker.outer_identity == descriptor_identity(&outer)?,
        "cleanup quarantine outer identity differs from its durable receipt"
    );
    let names = directory_entry_names(outer.as_raw_fd())?;
    ensure!(
        names.len() == 2
            && names.iter().any(|name| name.as_bytes() == b"owned-stage")
            && names
                .iter()
                .any(|name| name.as_bytes() == TRANSACTION_ARTIFACT_MARKER_FILENAME.as_bytes()),
        "cleanup quarantine has an unexpected inventory"
    );
    let inner_name = CString::new("owned-stage")?;
    let inner = open_directory_at(outer.as_raw_fd(), &inner_name)?;
    ensure!(
        marker.inner_identity == descriptor_identity(&inner)?
            && marker
                .inner_artifact_name
                .starts_with(".temporal-inference-stage-")
            && !marker
                .inner_artifact_name
                .starts_with(TRANSACTION_STAGE_CLEANUP_PREFIX)
            && transaction_marker_bytes_are_owned(
                &read_bounded_at(inner.as_raw_fd(), &marker_name, 64 * 1024)?,
                &marker.product_directory_sha256,
                &marker.inner_artifact_name,
                &marker.ownership_token,
            ),
        "cleanup quarantine inner identity differs from its durable receipt"
    );
    Ok(VerifiedCleanupQuarantine {
        product_directory,
        outer,
        inner,
        marker,
    })
}

#[cfg(unix)]
fn remove_directory_contents_at(directory: RawFd) -> Result<()> {
    for name in directory_entry_names(directory)? {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: arguments are valid and `stat` points to writable storage.
        let status = unsafe {
            libc::fstatat(
                directory,
                name.as_ptr(),
                stat.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        ensure!(
            status == 0,
            "reading cleanup entry identity failed: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: successful `fstatat` initialized `stat`.
        let stat = unsafe { stat.assume_init() };
        if stat.st_mode & libc::S_IFMT == libc::S_IFDIR {
            let child = open_directory_at(directory, &name)?;
            ensure!(
                descriptor_identity(&child)? == stat_directory_identity(&stat),
                "cleanup child identity changed before traversal"
            );
            remove_directory_contents_at(child.as_raw_fd())?;
            child.sync_all()?;
            drop(child);
            // SAFETY: deletion is relative to the verified parent descriptor.
            ensure!(
                unsafe { libc::unlinkat(directory, name.as_ptr(), libc::AT_REMOVEDIR) } == 0,
                "removing cleanup directory failed: {}",
                std::io::Error::last_os_error()
            );
        } else {
            // SAFETY: deletion is relative to the verified parent and never follows links.
            ensure!(
                unsafe { libc::unlinkat(directory, name.as_ptr(), 0) } == 0,
                "removing cleanup entry failed: {}",
                std::io::Error::last_os_error()
            );
        }
    }
    Ok(())
}

#[cfg(unix)]
fn remove_verified_cleanup_quarantine(
    directory: &Path,
    cleanup_name: &str,
    expected_ownership_token: Option<&str>,
    after_verification: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let verified =
        open_verified_cleanup_quarantine(directory, cleanup_name, expected_ownership_token)?;
    after_verification()?;
    remove_directory_contents_at(verified.inner.as_raw_fd())?;
    verified.inner.sync_all()?;
    let inner_name = CString::new("owned-stage")?;
    let mut inner_stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: arguments are valid and `inner_stat` points to writable storage.
    let inner_status = unsafe {
        libc::fstatat(
            verified.outer.as_raw_fd(),
            inner_name.as_ptr(),
            inner_stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    ensure!(
        inner_status == 0,
        "cleanup inner stage disappeared before removal"
    );
    // SAFETY: successful `fstatat` initialized `inner_stat`.
    let inner_stat = unsafe { inner_stat.assume_init() };
    ensure!(
        verified.marker.inner_identity == stat_directory_identity(&inner_stat),
        "cleanup inner stage identity changed before removal"
    );
    drop(verified.inner);
    // SAFETY: removal is relative to the retained verified outer descriptor.
    ensure!(
        unsafe {
            libc::unlinkat(
                verified.outer.as_raw_fd(),
                inner_name.as_ptr(),
                libc::AT_REMOVEDIR,
            )
        } == 0,
        "removing cleanup inner stage failed: {}",
        std::io::Error::last_os_error()
    );
    let marker_name = CString::new(TRANSACTION_ARTIFACT_MARKER_FILENAME)?;
    // SAFETY: removal is relative to the retained verified outer descriptor.
    ensure!(
        unsafe { libc::unlinkat(verified.outer.as_raw_fd(), marker_name.as_ptr(), 0) } == 0,
        "removing cleanup receipt failed: {}",
        std::io::Error::last_os_error()
    );
    verified.outer.sync_all()?;
    let cleanup_name_c = cstring_component(cleanup_name)?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: arguments are valid and `stat` points to writable storage.
    let status = unsafe {
        libc::fstatat(
            verified.product_directory.as_raw_fd(),
            cleanup_name_c.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    ensure!(status == 0, "cleanup quarantine disappeared before removal");
    // SAFETY: successful `fstatat` initialized `stat`.
    let stat = unsafe { stat.assume_init() };
    ensure!(
        verified.marker.outer_identity == stat_directory_identity(&stat),
        "cleanup quarantine root identity changed before removal"
    );
    // SAFETY: removal is relative to the opened product directory.
    ensure!(
        unsafe {
            libc::unlinkat(
                verified.product_directory.as_raw_fd(),
                cleanup_name_c.as_ptr(),
                libc::AT_REMOVEDIR,
            )
        } == 0,
        "removing cleanup quarantine failed: {}",
        std::io::Error::last_os_error()
    );
    verified.product_directory.sync_all()?;
    Ok(())
}

fn is_reserved_transaction_artifact(name: &str) -> bool {
    name.starts_with(".temporal-inference-stage-")
        || name.starts_with(".temporal-product-journal-")
        || name.starts_with(".fixed-cube-receipt-rollback-")
        || name.starts_with(".fixed-cube-receipt-temporal-")
}

#[allow(clippy::too_many_lines)]
fn cleanup_orphan_transaction_files(directory: &Path) -> Result<()> {
    let mut entries = std::fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    let mut owned = Vec::new();
    let mut owned_cleanup_quarantines = Vec::new();
    let mut collisions = Vec::new();
    for entry in &entries {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !is_reserved_transaction_artifact(name) || name.ends_with(".owner.json") {
            continue;
        }
        if entry.file_type()?.is_dir() && name.starts_with(TRANSACTION_STAGE_CLEANUP_PREFIX) {
            #[cfg(unix)]
            if open_verified_cleanup_quarantine(directory, name, None).is_ok() {
                owned_cleanup_quarantines.push(name.to_owned());
            } else {
                collisions.push(name.to_owned());
            }
            #[cfg(not(unix))]
            collisions.push(name.to_owned());
            continue;
        }
        let marker =
            if entry.file_type()?.is_dir() && name.starts_with(".temporal-inference-stage-") {
                entry.path().join(TRANSACTION_ARTIFACT_MARKER_FILENAME)
            } else if entry.file_type()?.is_file() {
                scratch_marker_path(directory, name)
            } else {
                collisions.push(name.to_owned());
                continue;
            };
        if marker.exists() && transaction_marker_is_owned(&marker, directory, name, None)? {
            owned.push((entry.path(), marker));
        } else {
            collisions.push(name.to_owned());
        }
    }
    for entry in &entries {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(artifact_name) = name.strip_suffix(".owner.json") else {
            continue;
        };
        if !is_reserved_transaction_artifact(artifact_name)
            || directory.join(artifact_name).exists()
        {
            continue;
        }
        if transaction_marker_is_owned(&entry.path(), directory, artifact_name, None)? {
            owned.push((directory.join(artifact_name), entry.path()));
        } else {
            collisions.push(name.to_owned());
        }
    }
    if !collisions.is_empty() {
        collisions.sort();
        collisions.dedup();
        return Err(TemporalTransactionCollision { paths: collisions }.into());
    }
    #[cfg(unix)]
    for cleanup_name in owned_cleanup_quarantines {
        remove_verified_cleanup_quarantine(directory, &cleanup_name, None, || Ok(()))?;
    }
    for (artifact, marker) in owned {
        if artifact.is_dir() {
            let name = artifact
                .file_name()
                .context("owned temporal stage has no filename")?
                .to_string_lossy();
            ensure!(
                marker == artifact.join(TRANSACTION_ARTIFACT_MARKER_FILENAME)
                    && transaction_marker_is_owned(&marker, directory, &name, None)?,
                "temporal stage ownership changed before orphan cleanup"
            );
            remove_owned_stage_directory(directory, &artifact, None, || Ok(()), |_| Ok(()))?;
        } else if artifact.exists() {
            let name = artifact
                .file_name()
                .context("owned temporal scratch has no filename")?
                .to_string_lossy();
            ensure!(
                transaction_marker_is_owned(&marker, directory, &name, None)?,
                "temporal scratch ownership changed before orphan cleanup"
            );
            std::fs::remove_file(&artifact)?;
        }
        if marker.exists() {
            std::fs::remove_file(marker)?;
        }
    }
    File::open(directory)?.sync_all()?;
    Ok(())
}

fn rollback_incomplete_product(directory: &Path) -> Result<()> {
    let journal = read_rollback_journal(directory)?;
    rollback_incomplete_product_with_journal(directory, &journal)
}

fn rollback_incomplete_product_with_journal(
    directory: &Path,
    journal: &ProductRollbackJournal,
) -> Result<()> {
    let mut collisions = Vec::new();
    for artifact in journal.installed_artifacts.iter().rev() {
        let path = directory.join(&artifact.name);
        if path.exists() {
            if installed_artifact_is_owned(&path, artifact, &journal.ownership_token)? {
                std::fs::remove_file(&path)?;
            } else {
                collisions.push(artifact.name.clone());
                tracing::warn!(
                    path = %path.display(),
                    "preserving unowned artifact encountered during temporal rollback"
                );
            }
        }
    }
    let fixed_receipt_path = directory.join("fixed_cube_receipt.json");
    let original_fixed_receipt_sha256 = sha256(&journal.original_fixed_cube_receipt);
    let restore_fixed_receipt = if fixed_receipt_path.exists() {
        let current = sha256_file(&fixed_receipt_path)?;
        if current == original_fixed_receipt_sha256 {
            false
        } else if journal.expected_fixed_receipt_sha256.as_deref() == Some(current.as_str()) {
            true
        } else {
            collisions.push("fixed_cube_receipt.json".to_owned());
            false
        }
    } else {
        true
    };
    if restore_fixed_receipt {
        restore_fixed_cube_receipt(
            directory,
            &journal.original_fixed_cube_receipt,
            &journal.ownership_token,
        )?;
    }
    let stage = directory.join(&journal.stage_directory);
    if stage.exists()
        && remove_owned_stage_directory(
            directory,
            &stage,
            Some(&journal.ownership_token),
            || Ok(()),
            |_| Ok(()),
        )
        .is_err()
    {
        collisions.push(journal.stage_directory.clone());
    }
    File::open(directory)?.sync_all()?;
    if !collisions.is_empty() {
        collisions.sort();
        collisions.dedup();
        let mut blocked = journal.clone();
        blocked.rollback_state = ProductRollbackState::BlockedUnownedCollision;
        blocked.collision_artifacts = collisions.clone();
        persist_rollback_journal(directory, &blocked, false)?;
        anyhow::bail!(
            "temporal rollback is blocked by unowned artifacts: {}",
            collisions.join(", ")
        );
    }
    remove_rollback_journal(directory)?;
    Ok(())
}

fn installed_artifact_is_owned(
    path: &Path,
    expected: &OwnedArtifactReceipt,
    ownership_token: &str,
) -> Result<bool> {
    if sha256_file(path)? != expected.sha256 {
        return Ok(false);
    }
    if expected.name == TEMPORAL_INFERENCE_PROVENANCE_FILENAME {
        let provenance: Value = serde_json::from_slice(&read_bounded(path, JSON_CAP)?)?;
        return Ok(provenance["transaction_ownership_token"].as_str() == Some(ownership_token));
    }
    let header = read_raster_header(path)?;
    Ok(header
        .metadata
        .get("TRANSACTION_OWNERSHIP_TOKEN")
        .map(String::as_str)
        == Some(ownership_token))
}

#[allow(clippy::too_many_lines)]
fn validate_completed_bundle(directory: &Path, journal: &ProductRollbackJournal) -> Result<()> {
    let expected_names = PRODUCT_LAYERS
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    ensure!(
        journal
            .expected_products
            .iter()
            .map(|receipt| receipt.name.as_str())
            .eq(expected_names.iter().copied())
            && journal.installed_artifacts.len() == PRODUCT_LAYERS.len() + 1
            && journal.installed_artifacts[..PRODUCT_LAYERS.len()] == journal.expected_products
            && journal
                .installed_artifacts
                .last()
                .is_some_and(|receipt| receipt.name == TEMPORAL_INFERENCE_PROVENANCE_FILENAME),
        "completed temporal journal has an invalid artifact inventory"
    );
    for (index, ((name, role), expected)) in PRODUCT_LAYERS
        .iter()
        .zip(&journal.expected_products)
        .enumerate()
    {
        let path = directory.join(name);
        ensure!(
            expected.name == *name && sha256_file(&path)? == expected.sha256,
            "completed temporal product hash differs: {name}"
        );
        let header = read_raster_header(&path)?;
        ensure!(
            header.shape == (journal.product_grid.rows, journal.product_grid.cols)
                && header.geotransform == journal.product_grid.geotransform
                && header.epsg == journal.product_grid.epsg
                && header.nodata.is_some_and(f64::is_nan)
                && header.metadata.get("PRODUCT_ROLE").map(String::as_str) == Some(*role)
                && header.metadata.get("UNITTYPE").map(String::as_str)
                    == Some(if index < 2 {
                        journal.product_grid.velocity_unit.as_str()
                    } else if index == 10 {
                        journal.product_grid.process_variance_unit.as_str()
                    } else {
                        "1"
                    })
                && header
                    .metadata
                    .get("TEMPORAL_ESTIMATOR")
                    .map(String::as_str)
                    == Some(COMPLETE_REFIT_BOOTSTRAP_METHOD)
                && header
                    .metadata
                    .get("TRANSACTION_OWNERSHIP_TOKEN")
                    .map(String::as_str)
                    == Some(journal.ownership_token.as_str()),
            "completed temporal product header differs: {name}"
        );
    }
    verify_persisted_product_value_semantics(
        directory,
        (journal.product_grid.rows, journal.product_grid.cols),
    )?;
    ensure!(
        sha256_file(&directory.join("velocity.tif"))? == journal.legacy_velocity_sha256
            && directory
                .join("velocity_sigma.tif")
                .exists()
                .then(|| sha256_file(&directory.join("velocity_sigma.tif")))
                .transpose()?
                == journal.legacy_sigma_sha256,
        "legacy velocity products differ from the transaction journal"
    );
    let provenance_sha256 = journal
        .expected_provenance_sha256
        .as_deref()
        .context("completed journal lacks provenance hash")?;
    ensure!(
        journal
            .installed_artifacts
            .last()
            .is_some_and(|receipt| receipt.sha256 == provenance_sha256),
        "completed temporal journal has a different installed marker hash"
    );
    let provenance_path = directory.join(TEMPORAL_INFERENCE_PROVENANCE_FILENAME);
    ensure!(
        sha256_file(&provenance_path)? == provenance_sha256,
        "temporal completion marker hash differs from journal"
    );
    let provenance: Value = serde_json::from_slice(&read_bounded(&provenance_path, JSON_CAP)?)?;
    ensure!(
        provenance["schema"].as_str() == Some(PRODUCT_SCHEMA)
            && provenance["transaction_ownership_token"].as_str()
                == Some(journal.ownership_token.as_str())
            && provenance["promotion_manifest_sha256"].as_str()
                == Some(journal.promotion_manifest_sha256.as_str())
            && provenance["corrected_velocity_sha256"].as_str()
                == Some(journal.expected_products[0].sha256.as_str())
            && provenance["corrected_sigma_sha256"].as_str()
                == Some(journal.expected_products[1].sha256.as_str()),
        "temporal completion marker identity differs from journal"
    );
    let provenance_semantics: crate::fixed_cube::FixedCubeSemanticValidation =
        serde_json::from_value(provenance["fixed_cube_semantics"].clone())?;
    ensure!(
        provenance_semantics == journal.semantic_validation,
        "temporal completion marker semantic receipt differs from journal"
    );
    let provenance_products = provenance["product_receipts"]
        .as_array()
        .context("temporal completion marker lacks product receipts")?;
    ensure!(
        provenance_products.len() == journal.expected_products.len()
            && provenance_products
                .iter()
                .zip(&journal.expected_products)
                .all(|(actual, expected)| {
                    Path::new(actual["path"].as_str().unwrap_or_default())
                        .file_name()
                        .is_some_and(|name| name == expected.name.as_str())
                        && actual["sha256"].as_str() == Some(expected.sha256.as_str())
                }),
        "temporal completion marker product receipts differ from journal"
    );
    let fixed_receipt_path = directory.join("fixed_cube_receipt.json");
    ensure!(
        sha256_file(&fixed_receipt_path)?
            == journal
                .expected_fixed_receipt_sha256
                .as_deref()
                .context("completed journal lacks fixed-receipt hash")?,
        "promoted fixed-cube receipt hash differs from journal"
    );
    let fixed: crate::fixed_cube::FixedCubeReceipt =
        serde_json::from_slice(&read_bounded(&fixed_receipt_path, 1024 * 1024)?)?;
    ensure!(
        fixed.inference_status == "calibrated_scope_match"
            && fixed.corrected_velocity_raster.as_deref() == Some(PRODUCT_LAYERS[0].0)
            && fixed.corrected_sigma_raster.as_deref() == Some(PRODUCT_LAYERS[1].0)
            && fixed.corrected_velocity_sha256.as_deref()
                == Some(journal.expected_products[0].sha256.as_str())
            && fixed.corrected_sigma_sha256.as_deref()
                == Some(journal.expected_products[1].sha256.as_str())
            && fixed.inference_provenance_sha256.as_deref() == Some(provenance_sha256)
            && fixed.inference_provenance.as_deref()
                == Some(TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
            && fixed.temporal_promotion_manifest_sha256.as_deref()
                == Some(journal.promotion_manifest_sha256.as_str())
            && fixed.semantic_validation.as_ref() == Some(&journal.semantic_validation),
        "promoted fixed-cube receipt state differs from journal"
    );
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

fn restore_fixed_cube_receipt(
    directory: &Path,
    receipt: &[u8],
    ownership_token: &str,
) -> Result<()> {
    static NEXT_ROLLBACK_ID: AtomicU64 = AtomicU64::new(0);
    let scratch = directory.join(format!(
        ".fixed-cube-receipt-rollback-{}-{}",
        std::process::id(),
        NEXT_ROLLBACK_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let scratch_name = scratch
        .file_name()
        .context("fixed-cube rollback scratch has no filename")?
        .to_string_lossy();
    let marker = write_transaction_artifact_marker(directory, &scratch_name, ownership_token)?;
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
    if restore.is_ok() {
        std::fs::remove_file(marker)?;
        File::open(directory)?.sync_all()?;
    }
    restore
}

fn create_stage_directory(directory: &Path, ownership_token: &str) -> Result<PathBuf> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let stage = directory.join(format!(
        ".temporal-inference-stage-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&stage)?;
    let initialize = (|| {
        let stage_name = stage
            .file_name()
            .context("temporal product stage has no filename")?
            .to_string_lossy();
        write_transaction_marker(
            &stage.join(TRANSACTION_ARTIFACT_MARKER_FILENAME),
            directory,
            &stage_name,
            ownership_token,
        )?;
        File::open(&stage)?.sync_all()?;
        File::open(directory)?.sync_all()?;
        Ok(stage.clone())
    })();
    if initialize.is_err() && stage.exists() {
        let stage_name = stage
            .file_name()
            .context("temporal product stage has no filename")?
            .to_string_lossy();
        let marker = stage.join(TRANSACTION_ARTIFACT_MARKER_FILENAME);
        if marker.exists()
            && transaction_marker_is_owned(&marker, directory, &stage_name, Some(ownership_token))?
        {
            remove_owned_stage_directory(
                directory,
                &stage,
                Some(ownership_token),
                || Ok(()),
                |_| Ok(()),
            )?;
        }
    }
    initialize
}

#[cfg(unix)]
fn remove_owned_stage_directory(
    directory: &Path,
    stage: &Path,
    ownership_token: Option<&str>,
    before_isolate: impl FnOnce() -> Result<()>,
    after_quarantine: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    static NEXT_CLEANUP_ID: AtomicU64 = AtomicU64::new(0);
    let stage_name = stage
        .file_name()
        .context("temporal product stage has no filename")?
        .to_str()
        .context("temporal product stage filename is not UTF-8")?
        .to_owned();
    ensure!(
        stage.parent() == Some(directory)
            && !stage_name.starts_with(TRANSACTION_STAGE_CLEANUP_PREFIX),
        "temporal stage path is outside the product directory"
    );
    let initial_stage = open_directory_path(stage)?;
    let initial_stage_identity = descriptor_identity(&initial_stage)?;
    let marker_name = CString::new(TRANSACTION_ARTIFACT_MARKER_FILENAME)?;
    let directory_sha256 = product_directory_sha256(directory)?;
    let marker_bytes = read_bounded_at(initial_stage.as_raw_fd(), &marker_name, 64 * 1024)?;
    let marker: TransactionArtifactMarker = serde_json::from_slice(&marker_bytes)?;
    ensure!(
        transaction_marker_bytes_are_owned(
            &marker_bytes,
            &directory_sha256,
            &stage_name,
            &marker.ownership_token,
        ) && ownership_token.is_none_or(|expected| marker.ownership_token == expected),
        "temporal stage ownership changed before deletion"
    );
    before_isolate()?;
    let private_name = format!(
        "{TRANSACTION_STAGE_CLEANUP_PREFIX}{}-{}",
        std::process::id(),
        NEXT_CLEANUP_ID.fetch_add(1, Ordering::Relaxed)
    );
    let private_parent = directory.join(&private_name);
    let mut private_builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        private_builder.mode(0o700);
    }
    private_builder.create(&private_parent)?;
    let quarantined = private_parent.join("owned-stage");
    std::fs::rename(stage, &quarantined)?;
    let product_directory = open_directory_path(directory)?;
    let private_name_c = cstring_component(&private_name)?;
    let outer = open_directory_at(product_directory.as_raw_fd(), &private_name_c)?;
    let inner_name = CString::new("owned-stage")?;
    let inner = open_directory_at(outer.as_raw_fd(), &inner_name)?;
    let inner_identity = descriptor_identity(&inner)?;
    ensure!(
        inner_identity == initial_stage_identity
            && transaction_marker_bytes_are_owned(
                &read_bounded_at(inner.as_raw_fd(), &marker_name, 64 * 1024)?,
                &directory_sha256,
                &stage_name,
                &marker.ownership_token,
            ),
        "temporal stage identity changed while quarantining for deletion"
    );
    let cleanup_marker = CleanupQuarantineMarker {
        schema: CLEANUP_QUARANTINE_MARKER_SCHEMA.to_owned(),
        ownership_token: marker.ownership_token.clone(),
        product_directory_sha256: directory_sha256,
        artifact_name: private_name.clone(),
        inner_artifact_name: stage_name,
        outer_identity: descriptor_identity(&outer)?,
        inner_identity,
    };
    write_new_at(
        outer.as_raw_fd(),
        &marker_name,
        &serde_json::to_vec(&cleanup_marker)?,
    )?;
    outer.sync_all()?;
    product_directory.sync_all()?;
    drop(inner);
    drop(outer);
    drop(product_directory);
    after_quarantine(&quarantined)?;
    remove_verified_cleanup_quarantine(
        directory,
        &private_name,
        Some(&cleanup_marker.ownership_token),
        || Ok(()),
    )
}

#[cfg(not(unix))]
fn remove_owned_stage_directory(
    _directory: &Path,
    _stage: &Path,
    _ownership_token: Option<&str>,
    _before_isolate: impl FnOnce() -> Result<()>,
    _after_quarantine: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    anyhow::bail!("temporal covariance product cleanup is unsupported on this platform")
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
        compose_working_set_admission, ensure_same_run_factor_directory, install_no_replace,
        output_window, reconstruct_covariance, validate_heldout_result, validate_input_coverage,
        validate_manifest, validate_product_value_semantics, validate_review,
        validate_synthetic_result, validate_working_set_high_water,
        write_product_transaction_with_validator, EvidenceDigests, HeldoutAttrition,
        HeldoutCoverage, HeldoutLevel, HeldoutReceiptIdentity, HeldoutResult, RawHeldoutLevel,
        SyntheticResult, SyntheticScores, TemporalCovariancePromotion, TemporalProductTransaction,
        TemporalPromotionManifest, TemporalReviewReceipt, HELDOUT_SCORE_SCHEMA, PRODUCT_LAYERS,
        PROMOTION_SCHEMA, REVIEW_SCHEMA, ROLLBACK_JOURNAL_FILENAME, SYNTHETIC_SCHEMA,
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
        let days: Vec<f64> = (0..13).map(|index| index as f64 * 12.0).collect();
        let mut observations: Vec<f64> = days
            .iter()
            .enumerate()
            .map(|(index, day)| 0.01 * day + (index as f64 * 0.7).sin() * 2.0)
            .collect();
        observations[0] = 0.0;
        let mut covariance = vec![vec![0.0; days.len()]; days.len()];
        for (index, row) in covariance.iter_mut().enumerate().skip(1) {
            row[index] = 1.0;
        }
        let temporal_options = dolphin_timeseries::TemporalCovarianceOptions::default();
        let fit = dolphin_timeseries::fit_temporal_covariance(
            &days,
            &observations,
            &covariance,
            &temporal_options,
        );
        assert_eq!(
            fit.bootstrap_attempts,
            temporal_options.bootstrap_replicates
        );
        let admitted = admit_combined_working_set(&config, days.len()).unwrap();
        let composition = dolphin_timeseries::temporal_covariance_workspace_composition(
            days.len(),
            temporal_options.bootstrap_replicates,
        )
        .unwrap();
        assert_eq!(
            admitted.temporal_solver_workspace_bytes,
            composition.total_bytes
        );
        assert!(admitted.temporal_solver_workspace_bytes > 12 * 12 * 8);
        assert_eq!(
            admitted.total_bytes,
            admitted.factor_block_bytes
                + admitted.block_id_bytes
                + admitted.displacement_window_bytes
                + admitted.output_window_bytes
                + admitted.output_write_copy_bytes
                + admitted.writer_bookkeeping_bytes
                + admitted.temporal_solver_workspace_bytes
                + admitted.gdal_cache_budget_bytes
        );
        assert!(admitted.total_bytes <= super::COMBINED_WORKING_SET_CAP_BYTES);
        assert!(admit_combined_working_set(&config, 100_000).is_err());
        let mut non_power_of_two_block_cap = config.clone();
        non_power_of_two_block_cap.block_id_read_cap_bytes = 4 * 1024 * 1024 + 8;
        assert!(
            admit_combined_working_set(&non_power_of_two_block_cap, days.len()).is_err(),
            "writer Vec capacity growth must be admitted before block reads"
        );
        for block_ids in [1_u64, 2] {
            let mut small_block_cap = config.clone();
            small_block_cap.block_id_read_cap_bytes = block_ids * std::mem::size_of::<u64>() as u64;
            let admission = compose_working_set_admission(&small_block_cap, days.len()).unwrap();
            assert_eq!(
                admission.writer_bookkeeping_bytes,
                4 * super::LAYER_COUNT as u64
                    * std::mem::size_of::<dolphin_core::BlockIndices>() as u64,
                "Vec must reserve its minimum nonzero capacity for {block_ids} block IDs"
            );
        }
        let boundary = compose_working_set_admission(&config, days.len()).unwrap();
        assert!(
            validate_working_set_high_water(&boundary, boundary.gdal_cache_budget_bytes).is_ok()
        );
        assert!(
            validate_working_set_high_water(&boundary, boundary.gdal_cache_budget_bytes + 1)
                .is_err()
        );
    }

    #[test]
    fn scoped_gdal_cache_limit_is_exclusive_and_restored_on_error() {
        // SAFETY: the test only reads GDAL's process-global configured limit.
        let previous = unsafe { gdal_sys::GDALGetCacheMax64() };
        let result: anyhow::Result<()> = (|| {
            let guard = super::ScopedGdalCacheLimit::acquire(8 * 1024 * 1024)?;
            guard.validate()?;
            assert!(super::ScopedGdalCacheLimit::acquire(8 * 1024 * 1024).is_err());
            anyhow::bail!("exercise restoration through the error path")
        })();
        assert!(result.is_err());
        // SAFETY: the scoped guard has restored GDAL's process-global limit.
        assert_eq!(unsafe { gdal_sys::GDALGetCacheMax64() }, previous);
    }

    #[test]
    fn startup_preserves_unowned_prefix_collisions_and_cleans_owned_stages() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_orphan_stage_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let collision = directory.join(".temporal-inference-stage-foreign");
        std::fs::create_dir(&collision).unwrap();
        std::fs::write(collision.join("user-data"), b"preserve").unwrap();
        let scratch_collision = directory.join(".temporal-product-journal-foreign");
        std::fs::write(&scratch_collision, b"preserve scratch").unwrap();
        let error = TemporalProductTransaction::acquire(&directory).unwrap_err();
        let collision_error = error
            .downcast_ref::<super::TemporalTransactionCollision>()
            .unwrap();
        assert_eq!(
            collision_error.paths,
            vec![
                ".temporal-inference-stage-foreign",
                ".temporal-product-journal-foreign"
            ]
        );
        assert_eq!(
            std::fs::read(collision.join("user-data")).unwrap(),
            b"preserve"
        );
        assert_eq!(
            std::fs::read(&scratch_collision).unwrap(),
            b"preserve scratch"
        );

        std::fs::remove_dir_all(&collision).unwrap();
        std::fs::remove_file(&scratch_collision).unwrap();
        let owned = super::create_stage_directory(&directory, &"53".repeat(32)).unwrap();
        std::fs::write(owned.join("partial.tif"), b"partial").unwrap();
        let owned_scratch_name = ".temporal-product-journal-owned";
        let owned_scratch = directory.join(owned_scratch_name);
        super::write_transaction_artifact_marker(&directory, owned_scratch_name, &"54".repeat(32))
            .unwrap();
        std::fs::write(&owned_scratch, b"partial journal").unwrap();
        let transaction = TemporalProductTransaction::acquire(&directory).unwrap();
        assert!(!owned.exists());
        assert!(!owned_scratch.exists());
        assert!(!super::scratch_marker_path(&directory, owned_scratch_name).exists());
        drop(transaction);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rollback_preserves_a_replaced_unowned_stage() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_replaced_stage_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("fixed_cube_receipt.json"), b"changed").unwrap();
        let stage_name = ".temporal-inference-stage-replaced";
        let stage = directory.join(stage_name);
        std::fs::create_dir(&stage).unwrap();
        std::fs::write(stage.join("user-data"), b"preserve").unwrap();
        let journal = super::ProductRollbackJournal {
            schema: super::ROLLBACK_JOURNAL_SCHEMA.to_owned(),
            ownership_token: "owner-token".to_owned(),
            original_fixed_cube_receipt: b"original".to_vec(),
            legacy_velocity_sha256: String::new(),
            legacy_sigma_sha256: None,
            promotion_manifest_sha256: String::new(),
            semantic_validation: crate::fixed_cube::FixedCubeSemanticValidation {
                observed_valid_pixels: 0,
                maximum_los_norm_error: 0.0,
                minimum_los_up: 1.0,
                los_sign_convention: String::new(),
                geometry_source: String::new(),
                geometry_provenance_status: String::new(),
            },
            product_grid: super::ProductGridReceipt {
                rows: 1,
                cols: 1,
                geotransform: [0.0; 6],
                epsg: None,
                velocity_unit: "rad/yr".to_owned(),
                process_variance_unit: "rad^2".to_owned(),
            },
            expected_products: PRODUCT_LAYERS
                .iter()
                .map(|(name, _)| super::OwnedArtifactReceipt {
                    name: (*name).to_owned(),
                    sha256: "00".repeat(32),
                })
                .collect(),
            installed_artifacts: Vec::new(),
            stage_directory: stage_name.to_owned(),
            expected_provenance_sha256: None,
            expected_fixed_receipt_sha256: None,
            rollback_state: super::ProductRollbackState::Active,
            collision_artifacts: Vec::new(),
        };
        assert!(super::rollback_incomplete_product_with_journal(&directory, &journal).is_err());
        assert_eq!(
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
            b"changed",
            "rollback must preserve a replaced unowned fixed-cube receipt"
        );
        assert_eq!(std::fs::read(stage.join("user-data")).unwrap(), b"preserve");
        assert_eq!(
            super::read_rollback_journal(&directory)
                .unwrap()
                .collision_artifacts,
            vec![stage_name, "fixed_cube_receipt.json"]
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rollback_recognizes_a_fixed_receipt_replaced_after_durable_identity() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_fixed_receipt_crash_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let original = b"original fixed receipt".to_vec();
        let promoted = b"owned promoted fixed receipt".to_vec();
        std::fs::write(directory.join("fixed_cube_receipt.json"), &original).unwrap();
        let mut journal = super::ProductRollbackJournal {
            schema: super::ROLLBACK_JOURNAL_SCHEMA.to_owned(),
            ownership_token: "fixed-receipt-owner".to_owned(),
            original_fixed_cube_receipt: original.clone(),
            legacy_velocity_sha256: String::new(),
            legacy_sigma_sha256: None,
            promotion_manifest_sha256: String::new(),
            semantic_validation: crate::fixed_cube::FixedCubeSemanticValidation {
                observed_valid_pixels: 0,
                maximum_los_norm_error: 0.0,
                minimum_los_up: 1.0,
                los_sign_convention: String::new(),
                geometry_source: String::new(),
                geometry_provenance_status: String::new(),
            },
            product_grid: super::ProductGridReceipt {
                rows: 1,
                cols: 1,
                geotransform: [0.0; 6],
                epsg: None,
                velocity_unit: "rad/yr".to_owned(),
                process_variance_unit: "rad^2".to_owned(),
            },
            expected_products: Vec::new(),
            installed_artifacts: Vec::new(),
            stage_directory: ".temporal-inference-stage-fixed-receipt".to_owned(),
            expected_provenance_sha256: None,
            expected_fixed_receipt_sha256: None,
            rollback_state: super::ProductRollbackState::Active,
            collision_artifacts: Vec::new(),
        };
        super::persist_rollback_journal(&directory, &journal, true).unwrap();
        super::journal_expected_fixed_receipt_before_replace(&directory, &mut journal, &promoted)
            .unwrap();
        std::fs::write(directory.join("fixed_cube_receipt.json"), &promoted).unwrap();

        super::rollback_incomplete_product_with_journal(&directory, &journal).unwrap();
        assert_eq!(
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
            original
        );
        assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn normal_cleanup_preserves_a_stage_replaced_after_ownership_check() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_cleanup_stage_swap_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let ownership_token = "cleanup-owner";
        let stage = super::create_stage_directory(&directory, ownership_token).unwrap();
        let error = super::remove_owned_stage_directory(
            &directory,
            &stage,
            Some(ownership_token),
            || {
                std::fs::remove_dir_all(&stage)?;
                std::fs::create_dir(&stage)?;
                std::fs::write(stage.join("user-data"), b"preserve")?;
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap_err();
        let error_text = error.to_string();
        assert!(error_text.contains("identity"), "{error_text}");
        let preserved = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path().join("owned-stage/user-data"))
            .find(|path| path.exists())
            .unwrap();
        assert_eq!(std::fs::read(preserved).unwrap(), b"preserve");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cleanup_preserves_a_quarantined_stage_replaced_after_isolation() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_quarantine_stage_swap_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let ownership_token = "quarantine-owner";
        let stage = super::create_stage_directory(&directory, ownership_token).unwrap();
        std::fs::write(stage.join("owned-data"), b"owned").unwrap();
        let mut replacement = None;
        let error = super::remove_owned_stage_directory(
            &directory,
            &stage,
            Some(ownership_token),
            || Ok(()),
            |quarantined| {
                let marker =
                    std::fs::read(quarantined.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME))?;
                std::fs::remove_dir_all(quarantined)?;
                std::fs::create_dir(quarantined)?;
                std::fs::write(
                    quarantined.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME),
                    marker,
                )?;
                std::fs::write(quarantined.join("user-data"), b"preserve")?;
                replacement = Some(quarantined.to_owned());
                Ok(())
            },
        )
        .unwrap_err();
        let error_text = error.to_string();
        assert!(error_text.contains("identity"), "{error_text}");
        let replacement = replacement.unwrap();
        assert_eq!(
            std::fs::read(replacement.join("user-data")).unwrap(),
            b"preserve"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cleanup_retains_verified_inner_descriptor_across_final_swap_hook() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_descriptor_stage_swap_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let ownership_token = "descriptor-owner";
        let stage = super::create_stage_directory(&directory, ownership_token).unwrap();
        let mut cleanup = None;
        super::remove_owned_stage_directory(
            &directory,
            &stage,
            Some(ownership_token),
            || Ok(()),
            |inner| {
                cleanup = inner.parent().map(std::path::Path::to_owned);
                anyhow::bail!("retain durable cleanup quarantine")
            },
        )
        .unwrap_err();
        let cleanup = cleanup.unwrap();
        let cleanup_name = cleanup.file_name().unwrap().to_str().unwrap().to_owned();
        let inner = cleanup.join("owned-stage");
        let error = super::remove_verified_cleanup_quarantine(
            &directory,
            &cleanup_name,
            Some(ownership_token),
            || {
                let marker =
                    std::fs::read(inner.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME))?;
                std::fs::rename(&inner, cleanup.join("held-owned-stage"))?;
                std::fs::create_dir(&inner)?;
                std::fs::write(
                    inner.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME),
                    marker,
                )?;
                std::fs::write(inner.join("user-data"), b"preserve")?;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("identity changed"));
        assert_eq!(std::fs::read(inner.join("user-data")).unwrap(), b"preserve");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_validates_the_inner_stage_before_cleaning_a_crash_quarantine() {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_temporal_quarantine_recovery_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let ownership_token = "quarantine-recovery-owner";

        let owned_stage = super::create_stage_directory(&directory, ownership_token).unwrap();
        let mut owned_cleanup = None;
        let crash = super::remove_owned_stage_directory(
            &directory,
            &owned_stage,
            Some(ownership_token),
            || Ok(()),
            |inner| {
                owned_cleanup = inner.parent().map(std::path::Path::to_owned);
                anyhow::bail!("simulated crash after durable cleanup receipt")
            },
        )
        .unwrap_err();
        assert!(crash.to_string().contains("simulated crash"));
        let owned_cleanup = owned_cleanup.unwrap();
        let transaction = TemporalProductTransaction::acquire(&directory).unwrap();
        assert!(!owned_cleanup.exists());
        drop(transaction);

        let swapped_stage = super::create_stage_directory(&directory, ownership_token).unwrap();
        let mut swapped_cleanup = None;
        super::remove_owned_stage_directory(
            &directory,
            &swapped_stage,
            Some(ownership_token),
            || Ok(()),
            |inner| {
                swapped_cleanup = inner.parent().map(std::path::Path::to_owned);
                anyhow::bail!("simulated crash before restart substitution")
            },
        )
        .unwrap_err();
        let swapped_cleanup = swapped_cleanup.unwrap();
        let swapped_cleanup_name = swapped_cleanup
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let swapped_inner = swapped_cleanup.join("owned-stage");
        let outer_marker =
            std::fs::read(swapped_cleanup.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME))
                .unwrap();
        let inner_marker =
            std::fs::read(swapped_inner.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME)).unwrap();
        std::fs::remove_dir_all(&swapped_cleanup).unwrap();
        std::fs::create_dir(&swapped_cleanup).unwrap();
        std::fs::write(
            swapped_cleanup.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME),
            outer_marker,
        )
        .unwrap();
        std::fs::create_dir(&swapped_inner).unwrap();
        std::fs::write(
            swapped_inner.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME),
            inner_marker,
        )
        .unwrap();
        std::fs::write(swapped_inner.join("user-data"), b"preserve").unwrap();

        let error = TemporalProductTransaction::acquire(&directory).unwrap_err();
        let collision = error
            .downcast_ref::<super::TemporalTransactionCollision>()
            .unwrap();
        assert_eq!(collision.paths, vec![swapped_cleanup_name]);
        assert_eq!(
            std::fs::read(swapped_inner.join("user-data")).unwrap(),
            b"preserve"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn fixed_cube_coverage_requires_checked_ordinal_complete_bursts() {
        let mut coverage = crate::provenance::InputCoverageProvenance {
            policy_version: crate::provenance::INPUT_COVERAGE_POLICY_VERSION.to_owned(),
            total_tiles: 0,
            linked_tiles: 0,
            nodata_tiles: 0,
            bursts: Vec::new(),
            output_pixels: 2,
            valid_pixels: 2,
            valid_fraction: 1.0,
        };
        assert!(validate_input_coverage(&coverage, 12, 2, 2).is_err());
        coverage.bursts = vec![
            crate::provenance::BurstCoverageProvenance {
                burst_index: 1,
                acquisition_count: 12,
                total_tiles: 1,
                linked_tiles: 1,
                nodata_tiles: 0,
            },
            crate::provenance::BurstCoverageProvenance {
                burst_index: 1,
                acquisition_count: 12,
                total_tiles: 1,
                linked_tiles: 1,
                nodata_tiles: 0,
            },
        ];
        coverage.total_tiles = 2;
        coverage.linked_tiles = 2;
        assert!(validate_input_coverage(&coverage, 12, 2, 2).is_err());
        coverage.bursts[0].burst_index = 0;
        assert!(validate_input_coverage(&coverage, 12, 2, 2).is_ok());
        coverage.bursts[0].total_tiles = usize::MAX;
        coverage.bursts[0].linked_tiles = usize::MAX;
        coverage.total_tiles = usize::MAX;
        coverage.linked_tiles = usize::MAX;
        coverage.nodata_tiles = 0;
        assert!(validate_input_coverage(&coverage, 12, 2, 2).is_err());
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
    #[allow(clippy::too_many_lines)]
    fn startup_rolls_back_each_durable_install_prefix_with_owned_cogs() {
        let root = std::env::temp_dir().join(format!(
            "dolphin_temporal_prefix_recovery_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let source = root.join("source");
        std::fs::create_dir_all(&source).unwrap();
        let token = "owned-prefix-transaction";
        let geotransform = [500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0];
        let mut expected = Vec::new();
        for (name, role) in PRODUCT_LAYERS {
            let path = source.join(name);
            dolphin_io::write_raster_with_metadata(
                &path,
                Array2::from_elem((1, 1), 1.0_f32).view(),
                geotransform,
                Some(32611),
                Some(f64::NAN),
                &[
                    ("PRODUCT_ROLE", role),
                    ("TEMPORAL_ESTIMATOR", COMPLETE_REFIT_BOOTSTRAP_METHOD),
                    ("TRANSACTION_OWNERSHIP_TOKEN", token),
                ],
            )
            .unwrap();
            expected.push(super::OwnedArtifactReceipt {
                name: name.to_owned(),
                sha256: super::sha256_file(&path).unwrap(),
            });
        }
        let marker = serde_json::json!({"transaction_ownership_token": token});
        let marker_bytes = serde_json::to_vec(&marker).unwrap();
        let marker_receipt = super::OwnedArtifactReceipt {
            name: super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME.to_owned(),
            sha256: super::sha256(&marker_bytes),
        };
        let original_receipt = br#"{"inference_status":"conditional_only"}"#;
        for prefix in 0..=PRODUCT_LAYERS.len() + 1 {
            let directory = root.join(format!("prefix-{prefix}"));
            std::fs::create_dir(&directory).unwrap();
            std::fs::write(directory.join("fixed_cube_receipt.json"), original_receipt).unwrap();
            let stage_name = format!(".temporal-inference-stage-prefix-{prefix}");
            let stage = directory.join(&stage_name);
            std::fs::create_dir(&stage).unwrap();
            super::write_transaction_marker(
                &stage.join(super::TRANSACTION_ARTIFACT_MARKER_FILENAME),
                &directory,
                &stage_name,
                token,
            )
            .unwrap();
            let cog_count = prefix.min(PRODUCT_LAYERS.len());
            for artifact in expected.iter().take(cog_count) {
                std::fs::copy(source.join(&artifact.name), directory.join(&artifact.name)).unwrap();
            }
            if prefix > PRODUCT_LAYERS.len() {
                std::fs::write(
                    directory.join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME),
                    &marker_bytes,
                )
                .unwrap();
            }
            let mut installed = expected[..cog_count].to_vec();
            if prefix > PRODUCT_LAYERS.len() {
                installed.push(marker_receipt.clone());
            }
            let journal = super::ProductRollbackJournal {
                schema: super::ROLLBACK_JOURNAL_SCHEMA.to_owned(),
                ownership_token: token.to_owned(),
                original_fixed_cube_receipt: original_receipt.to_vec(),
                legacy_velocity_sha256: "unused-in-incomplete-prefix".to_owned(),
                legacy_sigma_sha256: None,
                promotion_manifest_sha256: "promotion".to_owned(),
                semantic_validation: crate::fixed_cube::FixedCubeSemanticValidation {
                    observed_valid_pixels: 1,
                    maximum_los_norm_error: 0.0,
                    minimum_los_up: 1.0,
                    los_sign_convention: "ground_to_sensor_positive_toward_sensor".to_owned(),
                    geometry_source: "CSLC-S1-STATIC".to_owned(),
                    geometry_provenance_status: "sourced_no_fallback".to_owned(),
                },
                product_grid: super::ProductGridReceipt {
                    rows: 1,
                    cols: 1,
                    geotransform,
                    epsg: Some(32611),
                    velocity_unit: "rad/yr".to_owned(),
                    process_variance_unit: "rad^2".to_owned(),
                },
                expected_products: expected.clone(),
                installed_artifacts: installed,
                stage_directory: stage_name,
                expected_provenance_sha256: Some(marker_receipt.sha256.clone()),
                expected_fixed_receipt_sha256: None,
                rollback_state: super::ProductRollbackState::Active,
                collision_artifacts: Vec::new(),
            };
            super::persist_rollback_journal(&directory, &journal, true).unwrap();
            let transaction = TemporalProductTransaction::acquire(&directory).unwrap();
            assert!(PRODUCT_LAYERS
                .iter()
                .all(|(name, _)| !directory.join(name).exists()));
            assert!(!directory
                .join(super::TEMPORAL_INFERENCE_PROVENANCE_FILENAME)
                .exists());
            assert_eq!(
                std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
                original_receipt
            );
            assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
            drop(transaction);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn complete_evidence_chain_rejects_tamper_and_scope_mismatch() {
        fn passing_level(level: &str, covered: usize) -> HeldoutLevel {
            let nominal = level.parse::<f64>().unwrap() / 100.0;
            let observed = covered as f64 / 96.0;
            HeldoutLevel {
                status: "pass".to_owned(),
                coverage: HeldoutCoverage {
                    status: "pass".to_owned(),
                    p_value: super::exact_binomial_upper_tail(covered, 96, nominal - 0.2).unwrap(),
                    null_coverage: nominal - 0.2,
                    observed_coverage: observed,
                    evaluated: 96,
                    covered,
                },
                coverage_absolute_error: (observed - nominal).abs(),
                coverage_absolute_gate: true,
                mean_interval_score: 1.0,
                mean_baseline_interval_score: 2.0,
                median_width_ratio: 0.5,
                proper_score_improves: true,
                holm_reject: true,
            }
        }

        let preregistration: Value =
            serde_json::from_slice(super::TEMPORAL_PREREGISTRATION_BYTES).unwrap();
        let producer_identity = super::SyntheticProducerIdentity {
            schema: super::TEMPORAL_PRODUCER_IDENTITY_SCHEMA.to_owned(),
            preregistration_sha256: super::canonical_json_sha256(
                super::TEMPORAL_PREREGISTRATION_BYTES,
            )
            .unwrap(),
            generator_sha256: super::sha256(super::GENERATOR_SOURCE_BYTES),
            batch_source_sha256: super::sha256(super::BATCH_SOURCE_BYTES),
            estimator_source_sha256: super::sha256(super::ESTIMATOR_SOURCE_BYTES),
            source_set_schema: super::TEMPORAL_PRODUCER_SOURCE_SET_SCHEMA.to_owned(),
            source_set_sha256: preregistration["producer_identity"]["source_set_sha256"]
                .as_str()
                .unwrap()
                .to_owned(),
            binary_path: "target/release/examples/temporal_covariance_batch".to_owned(),
            binary_sha256: "ab".repeat(32),
            binary_bytes: 1,
            batch_schema: super::TEMPORAL_BATCH_SCHEMA.to_owned(),
            generator_schema: SYNTHETIC_SCHEMA.to_owned(),
            source_correlation_model: "exponential_euclidean_v1".to_owned(),
            source_correlation_distance_scale_pixels: 1.5,
            seed_count: 1_050,
        };
        let mut synthetic = SyntheticResult {
            schema: SYNTHETIC_SCHEMA.to_owned(),
            attempted_cells: 50_400,
            batch_attempted_cells: 50_400,
            seed_count: 1_050,
            execution_complete: true,
            exact_seed_denominator_complete: true,
            corrected_inferential_sigma_emission: false,
            promotion_eligible: true,
            promotion_status: "eligible_for_external_field_review".to_owned(),
            scores: SyntheticScores {
                all_methods_pass: true,
            },
            resource_gates: BTreeMap::from([("rss".to_owned(), true)]),
            producer_identity,
        };
        validate_synthetic_result(&synthetic).unwrap();
        synthetic.seed_count = 5_000;
        synthetic.attempted_cells = 240_000;
        synthetic.batch_attempted_cells = 240_000;
        assert!(validate_synthetic_result(&synthetic).is_err());
        synthetic.seed_count = 1_050;
        synthetic.attempted_cells = 50_400;
        synthetic.batch_attempted_cells = 50_400;
        synthetic.producer_identity.seed_count = 1_050;
        synthetic.producer_identity.source_set_sha256 = "cd".repeat(32);
        assert!(validate_synthetic_result(&synthetic).is_err());
        synthetic.producer_identity.source_set_sha256 = preregistration["producer_identity"]
            ["source_set_sha256"]
            .as_str()
            .unwrap()
            .to_owned();
        let heldout_identity = HeldoutReceiptIdentity {
            receipt_sha256: "aa".repeat(32),
            manifest_file_sha256:
                "75ce4e149aa8ed49ddb22cfedcac8056201a18c75af73ebb4fa22e7f9290ab1d".to_owned(),
            manifest_sha256: "534b9c534255404a71d3e2f53cc55cccc41ef4ef372bc18bb16e3a842826ee7e"
                .to_owned(),
            freeze_receipt_sha256:
                "cd3a0c4bdd5517fb310a55c2b34abbf78ff4bd7ccfe1aa6c1a5f5d61bae99f1c".to_owned(),
            factor_scope_sha256: "ee35862925aa7e88e0c18e0f09633aa5c02ed5781c68c9c2afb3aed3168eb833"
                .to_owned(),
            attrition: HeldoutAttrition {
                attrited_primary_ids: Vec::new(),
                used_surplus_ids: Vec::new(),
                unused_surplus_ids: (0..20).map(|index| format!("s{index:02}")).collect(),
                reasons_by_cluster: BTreeMap::new(),
            },
            evaluated_clusters: 96,
            required_cluster_failed: false,
            raw_levels: ["68", "90", "95"]
                .into_iter()
                .zip([62, 84, 90])
                .map(|(level, covered)| {
                    (
                        level.to_owned(),
                        RawHeldoutLevel {
                            covered,
                            mean_interval_score: 1.0,
                            mean_baseline_interval_score: 2.0,
                            median_width_ratio: 0.5,
                        },
                    )
                })
                .collect(),
        };
        let mut heldout = HeldoutResult {
            schema: HELDOUT_SCORE_SCHEMA.to_owned(),
            schema_version: 1,
            cohort_id: "f53-06-outer-nonfresno-v1".to_owned(),
            manifest_file_sha256: heldout_identity.manifest_file_sha256.clone(),
            manifest_sha256: heldout_identity.manifest_sha256.clone(),
            freeze_receipt_sha256: heldout_identity.freeze_receipt_sha256.clone(),
            factor_scope_sha256: heldout_identity.factor_scope_sha256.clone(),
            heldout_receipt_sha256: heldout_identity.receipt_sha256.clone(),
            primary_cluster_count: 96,
            surplus_cluster_count: 20,
            status: "pass".to_owned(),
            errors: Vec::<Value>::new(),
            levels: ["68", "90", "95"]
                .into_iter()
                .zip([62, 84, 90])
                .map(|(level, covered)| (level.to_owned(), passing_level(level, covered)))
                .collect(),
            evaluated_clusters: 96,
            emission_rate: 1.0,
            attrited_primary_ids: Vec::new(),
            used_surplus_ids: Vec::new(),
            unused_surplus_ids: (0..20).map(|index| format!("s{index:02}")).collect(),
            reasons_by_cluster: BTreeMap::new(),
        };
        validate_heldout_result(&heldout, &heldout_identity).unwrap();
        heldout.levels.get_mut("90").unwrap().coverage.p_value += 0.01;
        assert!(validate_heldout_result(&heldout, &heldout_identity).is_err());
        heldout.levels.get_mut("90").unwrap().coverage.p_value =
            super::exact_binomial_upper_tail(84, 96, 0.7).unwrap();
        heldout.levels.get_mut("90").unwrap().mean_interval_score = 1.25;
        assert!(validate_heldout_result(&heldout, &heldout_identity).is_err());
        heldout.levels.get_mut("90").unwrap().mean_interval_score = 1.0;
        let mut stale_freeze = heldout;
        stale_freeze.freeze_receipt_sha256 = "00".repeat(32);
        assert!(validate_heldout_result(&stale_freeze, &heldout_identity).is_err());
        let expected = EvidenceDigests {
            synthetic_result_sha256: "11".repeat(32),
            heldout_result_sha256: "22".repeat(32),
            heldout_receipt_sha256: "2a".repeat(32),
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
            heldout_receipt_sha256: expected.heldout_receipt_sha256.clone(),
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
            heldout_receipt_sha256: expected.heldout_receipt_sha256.clone(),
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
    fn heldout_receipt_requires_exact_frozen_slot_accounting() {
        let primary = (0..96)
            .map(|index| serde_json::json!({"candidate_id": format!("p{index:02}")}))
            .collect::<Vec<_>>();
        let surplus = (0..20)
            .map(|index| serde_json::json!({"candidate_id": format!("s{index:02}")}))
            .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "frozen_clusters": primary,
            "surplus_clusters": surplus,
        });
        let mut clusters = (0..96)
            .map(
                |index| serde_json::json!({"cluster_id": format!("p{index:02}"), "status": "pass"}),
            )
            .collect::<Vec<_>>();
        clusters.extend((0..20).map(
            |index| serde_json::json!({"cluster_id": format!("s{index:02}"), "status": "not_used"}),
        ));
        let mut receipt = super::HeldoutReceipt {
            schema: super::HELDOUT_RECEIPT_SCHEMA.to_owned(),
            schema_version: 1,
            outcomes_present: true,
            one_shot_unblinding: true,
            cohort_id: "f53-06-outer-nonfresno-v1".to_owned(),
            generation_id: "f53-06-v1".to_owned(),
            preregistration_sha256: "11".repeat(32),
            manifest_sha256: "22".repeat(32),
            scope_hash: "33".repeat(32),
            calibrated_scope_match: true,
            factor_binding: serde_json::json!({}),
            hashes: BTreeMap::new(),
            cluster_counts: super::HeldoutClusterCounts {
                primary: 96,
                surplus: 20,
                executed: 96,
                evaluable: 96,
            },
            attrition: super::HeldoutAttrition {
                attrited_primary_ids: Vec::new(),
                used_surplus_ids: Vec::new(),
                unused_surplus_ids: (0..20).map(|index| format!("s{index:02}")).collect(),
                reasons_by_cluster: BTreeMap::new(),
            },
            clusters,
            run_identity: super::HeldoutRunIdentity {
                generation_id: "f53-06-v1".to_owned(),
                preregistration_sha256: "11".repeat(32),
                manifest_sha256: "22".repeat(32),
                freeze_receipt_sha256: "44".repeat(32),
                run_plan_sha256: "55".repeat(32),
                binary_sha256: "66".repeat(32),
                implementation_source_hashes: BTreeMap::new(),
                product_identities_sha256: "88".repeat(32),
            },
            run_identity_sha256: "77".repeat(32),
        };
        super::validate_heldout_accounting(&receipt, &manifest).unwrap();

        receipt.clusters.swap(0, 1);
        assert!(super::validate_heldout_accounting(&receipt, &manifest).is_err());
        receipt.clusters.swap(0, 1);
        receipt.attrition.used_surplus_ids.push("p00".to_owned());
        receipt.attrition.unused_surplus_ids.pop();
        assert!(super::validate_heldout_accounting(&receipt, &manifest).is_err());
    }

    #[test]
    fn heldout_raw_observation_metrics_are_recomputed() {
        let mut observation = serde_json::json!({
            "insar_slope_difference": 0.0,
            "gnss_slope_difference": 0.0,
            "insar_difference_variance": 0.25,
            "gnss_slope_variance": 0.75,
            "sensor_cross_covariance": 0.0,
            "baseline_sigma": {"68": 2.0, "90": 2.0, "95": 2.0},
        });
        let levels = super::score_heldout_observation(&observation).unwrap();
        assert!(levels.values().all(|level| level.covered));
        assert!(levels.values().all(|level| {
            (level.width / level.baseline_width - 0.5).abs() <= f64::EPSILON
                && level.interval_score < level.baseline_interval_score
        }));

        observation["sensor_cross_covariance"] = serde_json::json!(0.01);
        assert!(super::score_heldout_observation(&observation).is_err());
    }

    #[test]
    fn output_values_must_match_status_semantics_and_fit_f32() {
        assert!(super::checked_f32(f64::MAX, "overflow contract").is_err());
        assert!(super::checked_f32(f64::from(f32::MAX), "maximum f32").is_ok());
        let mut layers: [Array2<f32>; super::LAYER_COUNT] =
            std::array::from_fn(|_| Array2::from_elem((1, 1), f32::NAN));
        layers[0][(0, 0)] = 1.0;
        layers[1][(0, 0)] = 0.5;
        layers[2][(0, 0)] = 0.0;
        layers[3][(0, 0)] = 0.0;
        layers[4][(0, 0)] = 0.0;
        layers[5][(0, 0)] = 12.0;
        layers[6][(0, 0)] = 1.0;
        layers[7][(0, 0)] = 11.0;
        layers[8][(0, 0)] = 0.1;
        layers[9][(0, 0)] = 0.2;
        layers[10][(0, 0)] = 1.0;
        layers[11][(0, 0)] = 10.0;
        layers[12][(0, 0)] = 200.0;
        layers[13][(0, 0)] = 198.0;
        validate_product_value_semantics(&layers).unwrap();
        layers[1][(0, 0)] = f32::INFINITY;
        assert!(validate_product_value_semantics(&layers).is_err());
        layers[1][(0, 0)] = f32::NAN;
        assert!(validate_product_value_semantics(&layers).is_err());
        layers[0][(0, 0)] = f32::NAN;
        layers[2][(0, 0)] = 1.0;
        validate_product_value_semantics(&layers).unwrap();
    }

    #[test]
    fn canonical_source_identity_binds_names_order_and_lengths() {
        let first = super::canonical_named_sources_sha256(&[("a", b"bc"), ("ab", b"c")]);
        let second = super::canonical_named_sources_sha256(&[("ab", b"c"), ("a", b"bc")]);
        let concatenation_collision =
            super::canonical_named_sources_sha256(&[("a", b"b"), ("c", b"")]);
        assert_ne!(first, second);
        assert_ne!(first, concatenation_collision);
    }

    #[test]
    fn common_support_rejects_partial_epoch_series() {
        let mut block = masked_block(13);
        block.status[0] = SpatialReferenceCovarianceStatus::Valid;
        block.rank_by_target[0] = 1;
        for date in 1..13 {
            block.difference_factor[date] = 1.0;
        }
        let mut observations = (1..13)
            .map(|date| Array2::from_elem((2, 1), date as f32))
            .collect::<Vec<_>>();
        observations[4][(0, 0)] = f32::NAN;
        let mask = Array2::from_shape_vec((2, 1), vec![1_u8, 0_u8]).unwrap();
        assert!(super::evaluate_block(
            &block,
            &observations,
            mask.view(),
            &(0..13).map(|date| date as f64 * 12.0).collect::<Vec<_>>(),
            &dolphin_timeseries::TemporalCovarianceOptions::default(),
        )
        .is_err());
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
            Some(f64::NAN),
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
            Some(f64::NAN),
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
            "method_version":"4.0.0",
            "orbit_direction":"ascending",
            "incidence_angle_deg":30.0,
            "incidence_angle_spread_deg":0.0,
            "incidence_angle_min_deg":30.0,
            "incidence_angle_max_deg":30.0,
            "heading_deg":10.0,
            "native_range_spacing_m":2.3,
            "native_azimuth_spacing_m":14.0,
            "acquisition_time_of_day_utc_s":36000.0,
            "phase_linking_coherence":null,
            "decomposition_geometry_complete":true,
            "input_coverage":{
                "policy_version":"complete-temporal-tile/1",
                "total_tiles":1,
                "linked_tiles":1,
                "nodata_tiles":0,
                "bursts":[{"burst_index":0,"acquisition_count":12,"total_tiles":1,"linked_tiles":1,"nodata_tiles":0}],
                "output_pixels":2,
                "valid_pixels":2,
                "valid_fraction":1.0
            },
            "geometry_provenance":{
                "method_version":"4.0.0",
                "fields":{
                    "orbit_direction":{
                        "status":"sourced",
                        "source_files":["cslc.h5"],
                        "source_keys":["/identification/orbit_pass_direction"],
                        "method":"read scalar per granule",
                        "raw_value":"ASCENDING"
                    },
                    "heading_deg":{
                        "status":"sourced",
                        "source_files":["cslc.h5"],
                        "source_keys":["/metadata/orbit/velocity_x"],
                        "method":"ECEF orbit velocity to ENU"
                    },
                    "incidence_angle_deg":{
                        "status":"sourced",
                        "source_files":["static.h5"],
                        "source_keys":["/data/los_east","/data/los_north"],
                        "method":"statistics over los_up on resolved output grid"
                    }
                }
            }
        }"#;
        std::fs::write(
            directory.join("geometry_provenance.json"),
            valid_geometry_provenance,
        )
        .unwrap();
        let velocity_header =
            dolphin_io::read_raster_header(&directory.join("velocity.tif")).unwrap();
        super::validate_fixed_cube_semantics(&directory, &velocity_header, &fixed_cube).unwrap();
        let original_fixed_receipt =
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap();
        let mut wrong_days_digest = fixed_cube.clone();
        wrong_days_digest.acquisition_days_sha256 = "sha256:00".to_owned();
        std::fs::write(
            directory.join("fixed_cube_receipt.json"),
            serde_json::to_vec_pretty(&wrong_days_digest).unwrap(),
        )
        .unwrap();
        assert!(
            super::validate_fixed_cube_scope(&directory, &days, &velocity_header, &metadata)
                .is_err()
        );
        std::fs::write(
            directory.join("fixed_cube_receipt.json"),
            &original_fixed_receipt,
        )
        .unwrap();
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
        let invalid_heading_source = String::from_utf8(valid_geometry_provenance.to_vec())
            .unwrap()
            .replace(
                "/metadata/orbit/velocity_x",
                "/metadata/processing/unbound_heading",
            );
        std::fs::write(
            directory.join("geometry_provenance.json"),
            invalid_heading_source,
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
        dolphin_io::write_raster_with_metadata(
            &directory.join("velocity_validity_mask.tif"),
            array![[1_u8], [0_u8]].view(),
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
            ("UNITTYPE", "unitless"),
            ("RASTER_ROLE", "fixed_cube_run_geometry"),
        ];
        dolphin_io::write_raster_with_metadata(
            &directory.join("los_east.tif"),
            Array2::from_elem((2, 1), 0.9_f32).view(),
            geotransform,
            Some(32611),
            Some(f64::NAN),
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
            Some(f64::NAN),
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
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
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
        let foreign_collision = directory.join(PRODUCT_LAYERS[0].0);
        std::fs::write(&foreign_collision, b"foreign product").unwrap();
        assert!(write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
            || Ok(promotion.clone()),
        )
        .is_err());
        assert_eq!(
            std::fs::read(&foreign_collision).unwrap(),
            b"foreign product"
        );
        let blocked = super::read_rollback_journal(&directory).unwrap();
        assert_eq!(
            blocked.rollback_state,
            super::ProductRollbackState::BlockedUnownedCollision
        );
        assert_eq!(blocked.collision_artifacts, vec![PRODUCT_LAYERS[0].0]);
        std::fs::remove_file(foreign_collision).unwrap();
        super::rollback_incomplete_product(&directory).unwrap();
        assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
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
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
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
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
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
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
            || Ok(promotion.clone()),
        )
        .unwrap();
        std::fs::write(directory.join("velocity.tif"), b"mutated legacy velocity").unwrap();
        assert!(complete_publication_after_legacy_check(
            &directory,
            published_before_legacy_mutation,
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
        let _ = write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
            || Ok(promotion.clone()),
        )
        .unwrap();
        std::fs::write(directory.join(PRODUCT_LAYERS[0].0), b"tampered product").unwrap();
        drop(product_transaction);
        assert!(TemporalProductTransaction::acquire(&directory).is_err());
        let blocked = super::read_rollback_journal(&directory).unwrap();
        assert_eq!(
            blocked.rollback_state,
            super::ProductRollbackState::BlockedUnownedCollision
        );
        assert_eq!(blocked.collision_artifacts, vec![PRODUCT_LAYERS[0].0]);
        assert!(directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
        assert_eq!(
            std::fs::read(directory.join(PRODUCT_LAYERS[0].0)).unwrap(),
            b"tampered product"
        );
        assert!(PRODUCT_LAYERS
            .iter()
            .skip(1)
            .all(|(name, _)| !directory.join(name).exists()));
        assert_eq!(
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
            fixed_cube_receipt_before
        );
        std::fs::remove_file(directory.join(PRODUCT_LAYERS[0].0)).unwrap();
        let product_transaction = TemporalProductTransaction::acquire(&directory).unwrap();
        assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
        let _ = write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
            || Ok(promotion.clone()),
        )
        .unwrap();
        assert!(directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
        std::fs::remove_file(directory.join("fixed_cube_receipt.json")).unwrap();
        drop(product_transaction);
        let product_transaction = TemporalProductTransaction::acquire(&directory).unwrap();
        assert_eq!(
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
            fixed_cube_receipt_before
        );
        assert!(PRODUCT_LAYERS
            .iter()
            .all(|(name, _)| !directory.join(name).exists()));
        assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
        let _ = write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
            || Ok(promotion.clone()),
        )
        .unwrap();
        std::fs::write(
            directory.join("fixed_cube_receipt.json"),
            b"corrupt receipt",
        )
        .unwrap();
        drop(product_transaction);
        let collision = TemporalProductTransaction::acquire(&directory).unwrap_err();
        assert!(collision.to_string().contains("fixed_cube_receipt.json"));
        assert_eq!(
            std::fs::read(directory.join("fixed_cube_receipt.json")).unwrap(),
            b"corrupt receipt"
        );
        assert!(PRODUCT_LAYERS
            .iter()
            .all(|(name, _)| !directory.join(name).exists()));
        std::fs::write(
            directory.join("fixed_cube_receipt.json"),
            &fixed_cube_receipt_before,
        )
        .unwrap();
        let product_transaction = TemporalProductTransaction::acquire(&directory).unwrap();
        let receipt = write_product_transaction_with_validator(
            &directory,
            &displacement_rasters,
            &days,
            &config,
            &directory,
            &promotion,
            &product_transaction,
            &legacy_velocity_sha256,
            Some(&legacy_sigma_sha256),
            || Ok(promotion.clone()),
        )
        .unwrap();
        assert!(directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
        let journal = super::read_rollback_journal(&directory).unwrap();
        super::validate_completed_bundle(&directory, &journal).unwrap();
        let mut mismatched_promotion = journal.clone();
        mismatched_promotion.promotion_manifest_sha256 = "different".to_owned();
        assert!(super::validate_completed_bundle(&directory, &mismatched_promotion).is_err());
        let mut mismatched_semantics = journal.clone();
        mismatched_semantics
            .semantic_validation
            .observed_valid_pixels += 1;
        assert!(super::validate_completed_bundle(&directory, &mismatched_semantics).is_err());
        drop(product_transaction);
        let recovered_transaction = TemporalProductTransaction::acquire(&directory).unwrap();
        assert!(!directory.join(ROLLBACK_JOURNAL_FILENAME).exists());
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
        assert_eq!(
            dolphin_io::read_raster_header(
                &directory.join("velocity_temporal_process_variance.tif")
            )
            .unwrap()
            .metadata
            .get("UNITTYPE")
            .map(String::as_str),
            Some("rad^2")
        );
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
        drop(recovered_transaction);
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
