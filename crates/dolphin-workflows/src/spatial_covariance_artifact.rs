//! Transactional sidecar for bounded reference-specific covariance factors.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dolphin_io::{
    read_spatial_reference_covariance_header, SpatialReferenceCalibrationScope,
    SpatialReferenceCovarianceMetadata, SpatialReferenceCovarianceWriteReceipt,
    SPATIAL_REFERENCE_COVARIANCE_METHOD,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Final reference-specific factor filename.
pub const SPATIAL_REFERENCE_COVARIANCE_FILENAME: &str =
    "referenced_displacement_covariance_factor.h5";
/// JSON completion marker written after the HDF5 rename.
pub const SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME: &str =
    "referenced_displacement_covariance_provenance.json";
/// Canonical HDF5 scratch filename admitted for finalization.
pub const SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME: &str =
    "referenced_displacement_covariance_factor.h5.scratch";
/// Canonical provenance scratch filename committed last.
pub const SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME: &str =
    "referenced_displacement_covariance_provenance.json.scratch";
/// Reader/writer lock filename for the complete factor/provenance pair.
pub const SPATIAL_REFERENCE_COVARIANCE_LOCK_FILENAME: &str =
    "referenced_displacement_covariance.capture.lock";
/// Independent review receipt required for calibrated promotion.
pub const SPATIAL_REFERENCE_COVARIANCE_REVIEW_RECEIPT_FILENAME: &str =
    "referenced_displacement_covariance_review_receipt.json";
/// Reviewed method manifest required for calibrated promotion.
pub const SPATIAL_REFERENCE_COVARIANCE_METHOD_MANIFEST_FILENAME: &str =
    "referenced_displacement_covariance_method_manifest.json";
/// Approximation-validation receipt required for calibrated promotion.
pub const SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RECEIPT_FILENAME: &str =
    "referenced_displacement_covariance_approximation_receipt.json";
/// Numeric approximation result bound by the approximation receipt.
pub const SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME: &str =
    "referenced_displacement_covariance_approximation_result.json";
/// Resource receipt required for calibrated promotion.
pub const SPATIAL_REFERENCE_COVARIANCE_RESOURCE_RECEIPT_FILENAME: &str =
    "referenced_displacement_covariance_resource_receipt.json";
/// Frozen preregistration copied into the artifact evidence directory.
pub const SPATIAL_REFERENCE_COVARIANCE_PREREGISTRATION_FILENAME: &str =
    "referenced_displacement_covariance_preregistration.json";
/// Frozen design copied into the artifact evidence directory.
pub const SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME: &str =
    "referenced_displacement_covariance_design.md";
/// Exact producer executable copied or linked into the artifact evidence directory.
pub const SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME: &str =
    "referenced_displacement_covariance_producer_binary";
const MANIFEST_SCHEMA_VERSION: u16 = 3;
const METADATA_READ_CAP: u64 = 1024 * 1024;
const BINARY_READ_CAP: u64 = 512 * 1024 * 1024;
const PREREGISTRATION_BYTES: &[u8] =
    include_bytes!("../../../validation/spatial_covariance_preregistration.json");
const DESIGN_BYTES: &[u8] = include_bytes!("../../../md/design/spatial-reference-covariance.md");
const CODE_BYTES: &[&[u8]] = &[
    include_bytes!("spatial_covariance_artifact.rs"),
    include_bytes!("sequential.rs"),
    include_bytes!("sequential_covariance.rs"),
    include_bytes!("../../dolphin-timeseries/src/inversion.rs"),
    include_bytes!("../../dolphin-timeseries/src/spatial_covariance.rs"),
    include_bytes!("../../dolphin-phaselink/src/engine.rs"),
    include_bytes!("../../dolphin-phaselink/src/fused.rs"),
    include_bytes!("../../dolphin-phaselink/src/source_model.rs"),
    include_bytes!("../../dolphin-phaselink/src/spatial_covariance.rs"),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialCovarianceApproximationReceipt {
    schema_version: u16,
    method: String,
    method_version: u16,
    crate_version: String,
    producer_commit: String,
    status: String,
    code_sha256: String,
    producer_binary_file: String,
    producer_binary_sha256: String,
    preregistration_file: String,
    preregistration_sha256: String,
    design_file: String,
    design_sha256: String,
    result_file: String,
    result_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialCovarianceApproximationResult {
    schema_version: u16,
    method: String,
    method_version: u16,
    crate_version: String,
    producer_commit: String,
    status: String,
    scope: SpatialCovarianceResultScope,
    evaluated_cases: u64,
    maximum_absolute_error: f64,
    tolerance: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialCovarianceResourceReceipt {
    schema_version: u16,
    method: String,
    method_version: u16,
    crate_version: String,
    producer_commit: String,
    status: String,
    code_sha256: String,
    producer_binary_sha256: String,
    preregistration_sha256: String,
    design_sha256: String,
    result_sha256: String,
    peak_resident_set_bytes: u64,
    wall_micros: u64,
    maximum_block_bytes: u64,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SpatialCovarianceResultScope {
    burst_id: String,
    crs: String,
    units: String,
    geotransform: [f64; 6],
    acquisition_days: Vec<f64>,
    grid_row_start: u64,
    grid_col_start: u64,
    grid_rows: u32,
    grid_cols: u32,
    grid_stride_y: u32,
    grid_stride_x: u32,
    reference_row: u64,
    reference_col: u64,
    gauge_date_index: u32,
    ordered_date_indices: Vec<u32>,
    mask_digest: String,
    source_replay_digest: String,
    l2_map_digest: String,
    reference_signature_digest: String,
    source_model_digest: String,
    effective_looks_digest: String,
    support_method: String,
    support_digest: String,
    correction_order_digest: String,
    unwrap_branch_digest: String,
    burst_ownership_digest: String,
    source_burst_ids: Vec<String>,
    reference_source_burst_index: u32,
    maximum_block_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialCovarianceReviewReceipt {
    schema_version: u16,
    method: String,
    method_version: u16,
    crate_version: String,
    producer_commit: String,
    reviewer: String,
    review_status: String,
    unresolved_findings: u32,
    code_sha256: String,
    producer_binary_sha256: String,
    preregistration_sha256: String,
    design_sha256: String,
    result_sha256: String,
    approximation_receipt_digest: String,
    resource_receipt_digest: String,
    calibration_scope_digest: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialCovarianceMethodManifest {
    schema_version: u16,
    method: String,
    method_version: u16,
    crate_version: String,
    producer_commit: String,
    manifest_status: String,
    code_sha256: String,
    producer_binary_sha256: String,
    preregistration_sha256: String,
    design_sha256: String,
    result_sha256: String,
    approximation_receipt_digest: String,
    resource_receipt_digest: String,
    review_receipt_digest: String,
    calibration_scope_digest: String,
}

#[derive(Clone, Copy)]
enum EvidenceValidationMode {
    Finalize,
    Read,
}

/// Durable receipt binding factor bytes to every scope identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialReferenceCovarianceArtifactManifest {
    /// Manifest schema version.
    pub schema_version: u16,
    /// Persisted factor method.
    pub method: String,
    /// Numeric method version.
    pub method_version: u16,
    /// Producing crate version.
    pub crate_version: String,
    /// Producing Git commit, when supplied by the build.
    pub producer_commit: Option<String>,
    /// Final HDF5 filename relative to this manifest.
    pub hdf5_file: String,
    /// Final HDF5 byte count.
    pub hdf5_bytes: u64,
    /// Lowercase SHA-256 of the final HDF5 bytes.
    pub hdf5_sha256: String,
    /// Source burst identity.
    pub burst_id: String,
    /// Exact CRS identity.
    pub crs: String,
    /// Factor units.
    pub units: String,
    /// Exact GDAL affine geotransform of the factor grid.
    pub geotransform: [f64; 6],
    /// Exact acquisition day coordinates relative to the gauge acquisition.
    pub acquisition_days: Vec<f64>,
    /// Selected reference/grid signature.
    pub reference_signature_digest: String,
    /// Native/output mask identity.
    pub mask_digest: String,
    /// Persisted #52 replay identity.
    pub source_replay_digest: String,
    /// Fixed-valid-observation L2 map identity.
    pub l2_map_digest: String,
    /// Frozen approximation receipt identity.
    pub approximation_receipt_digest: String,
    /// Frozen resource receipt identity.
    pub resource_receipt_digest: String,
    /// Independent-review receipt identity.
    pub review_receipt_digest: String,
    /// Immutable reviewed method-manifest identity.
    pub method_manifest_digest: String,
    /// Exact calibrated scope identity, empty while uncalibrated.
    pub calibration_scope_digest: String,
    /// Proper-complex primitive source-model identity.
    pub source_model_digest: String,
    /// Effective-look rule identity.
    pub effective_looks_digest: String,
    /// HDF5 block dataset containing per-target effective-look fractions.
    pub effective_looks_fraction_dataset: String,
    /// HDF5 block dataset containing per-target support-union counts.
    pub support_union_count_dataset: String,
    /// HDF5 block dataset containing per-target exact effective-look receipts.
    pub effective_looks_receipt_dataset: String,
    /// HDF5 block dataset containing per-target replay resource high-water bounds.
    pub resource_high_water_bytes_dataset: String,
    /// Realized fixed-support method.
    pub support_method: String,
    /// Realized fixed-support identity.
    pub support_digest: String,
    /// Corrections-before-reference ordering identity.
    pub correction_order_digest: String,
    /// Fixed unwrap/estimator branch identity.
    pub unwrap_branch_digest: String,
    /// Source-burst ownership and seam lineage identity.
    pub burst_ownership_digest: String,
    /// Ordered source-burst registry represented by the factor blocks.
    pub source_burst_ids: Vec<String>,
    /// Source-burst registry index of the selected reference.
    pub reference_source_burst_index: u32,
    /// Exact calibration scope; never inferred from file presence.
    pub calibration_scope: String,
    /// Maximum logical numeric bytes admitted for one block.
    pub maximum_block_bytes: u64,
}

/// Exclusive lock spanning scratch validation through manifest commit.
pub struct SpatialReferenceCovarianceArtifactTransaction {
    directory: PathBuf,
    _lock: File,
}

impl SpatialReferenceCovarianceArtifactTransaction {
    /// Acquire the artifact writer lock without waiting.
    ///
    /// # Errors
    /// Returns an error while another reader/writer owns the artifact.
    #[cfg(unix)]
    pub fn acquire(directory: &Path) -> Result<Self> {
        let lock_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_LOCK_FILENAME);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("opening spatial covariance lock {}", lock_path.display()))?;
        // SAFETY: `lock` owns this descriptor for the transaction lifetime.
        let result = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        anyhow::ensure!(result == 0, "spatial covariance artifact is already locked");
        let transaction = Self {
            directory: directory.to_owned(),
            _lock: lock,
        };
        recover_incomplete_artifact(&transaction.directory)?;
        Ok(transaction)
    }

    /// Non-Unix targets cannot provide the required durable lock.
    #[cfg(not(unix))]
    pub fn acquire(_directory: &Path) -> Result<Self> {
        anyhow::bail!("spatial covariance artifact locking is unsupported on this platform")
    }
}

struct SpatialReferenceCovarianceArtifactReadLock {
    _lock: File,
}

#[cfg(unix)]
fn acquire_read_lock(directory: &Path) -> Result<SpatialReferenceCovarianceArtifactReadLock> {
    let lock_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_LOCK_FILENAME);
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening spatial covariance lock {}", lock_path.display()))?;
    // SAFETY: `lock` owns this descriptor for the read-lock lifetime.
    let result = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    anyhow::ensure!(result == 0, "spatial covariance artifact is being replaced");
    Ok(SpatialReferenceCovarianceArtifactReadLock { _lock: lock })
}

#[cfg(not(unix))]
fn acquire_read_lock(_directory: &Path) -> Result<SpatialReferenceCovarianceArtifactReadLock> {
    anyhow::bail!("spatial covariance artifact read locking is unsupported on this platform")
}

/// Validate, atomically install, and commit a reference-specific factor artifact.
///
/// The JSON manifest is the only completion marker and is renamed last.
///
/// # Errors
/// Returns an error for mismatched metadata/bytes, out-of-directory scratch
/// files, or I/O failure.
pub fn finalize_spatial_reference_covariance_artifact(
    transaction: &SpatialReferenceCovarianceArtifactTransaction,
    hdf5_scratch: &Path,
    metadata: &SpatialReferenceCovarianceMetadata,
    write_receipt: &SpatialReferenceCovarianceWriteReceipt,
) -> Result<SpatialReferenceCovarianceArtifactManifest> {
    let directory = &transaction.directory;
    anyhow::ensure!(
        hdf5_scratch == directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME),
        "spatial covariance scratch file must use the canonical transaction path"
    );
    let embedded = read_spatial_reference_covariance_header(hdf5_scratch, METADATA_READ_CAP)
        .context("validating spatial covariance HDF5 metadata")?;
    anyhow::ensure!(
        embedded == *metadata,
        "spatial covariance HDF5 metadata differs from finalization metadata"
    );
    validate_calibration_evidence(directory, metadata, EvidenceValidationMode::Finalize)?;
    File::open(hdf5_scratch)?.sync_all()?;
    let (hdf5_sha256, hdf5_bytes) = sha256_file(hdf5_scratch)?;
    anyhow::ensure!(
        hdf5_sha256 == write_receipt.hdf5_sha256
            && hdf5_bytes == write_receipt.hdf5_bytes,
        "spatial covariance HDF5 changed after sealing: expected {} bytes {} but observed {} bytes {}",
        write_receipt.hdf5_bytes,
        write_receipt.hdf5_sha256,
        hdf5_bytes,
        hdf5_sha256
    );
    let manifest = manifest(metadata, hdf5_sha256, hdf5_bytes);
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let final_manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let final_hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    match inspect_final_artifact(directory) {
        FinalArtifactState::Absent => {}
        FinalArtifactState::Valid => {
            anyhow::bail!("a valid spatial covariance artifact already exists")
        }
        FinalArtifactState::Invalid(reason) => {
            quarantine_final_artifact(directory)
                .with_context(|| format!("quarantining invalid artifact: {reason}"))?;
        }
        FinalArtifactState::Unverifiable(reason) => {
            anyhow::bail!("spatial covariance final artifact is unverifiable: {reason}")
        }
    }
    let scratch_manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME);
    write_synced(&scratch_manifest, &manifest_bytes)?;
    fs::rename(hdf5_scratch, &final_hdf5).context("finalizing spatial covariance HDF5")?;
    sync_directory(directory)?;
    fs::rename(&scratch_manifest, &final_manifest)
        .context("committing spatial covariance manifest")?;
    sync_directory(directory)?;
    Ok(manifest)
}

/// Read a completed manifest and verify both its HDF5 digest and embedded scope.
///
/// # Errors
/// Returns an error for missing, malformed, tampered, stale, or scope-mismatched
/// artifacts.
pub fn read_spatial_reference_covariance_artifact_manifest(
    directory: &Path,
) -> Result<SpatialReferenceCovarianceArtifactManifest> {
    let _lock = acquire_read_lock(directory)?;
    read_manifest_unlocked(directory)
}

fn read_manifest_unlocked(directory: &Path) -> Result<SpatialReferenceCovarianceArtifactManifest> {
    let manifest_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let bytes = read_file_with_cap(&manifest_path, "spatial covariance manifest")?;
    let parsed: SpatialReferenceCovarianceArtifactManifest = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        parsed.schema_version == MANIFEST_SCHEMA_VERSION
            && parsed.method == SPATIAL_REFERENCE_COVARIANCE_METHOD
            && parsed.method_version == 1
            && parsed.hdf5_file == SPATIAL_REFERENCE_COVARIANCE_FILENAME,
        "unsupported spatial covariance manifest"
    );
    let hdf5_path = directory.join(&parsed.hdf5_file);
    let (digest, byte_count) = sha256_file(&hdf5_path)?;
    anyhow::ensure!(
        digest == parsed.hdf5_sha256 && byte_count == parsed.hdf5_bytes,
        "spatial covariance HDF5 does not match its manifest"
    );
    let embedded = read_spatial_reference_covariance_header(&hdf5_path, METADATA_READ_CAP)?;
    validate_calibration_evidence(directory, &embedded, EvidenceValidationMode::Read)?;
    anyhow::ensure!(
        manifest(&embedded, digest, byte_count) == parsed,
        "spatial covariance embedded scope does not match its manifest"
    );
    Ok(parsed)
}

fn recover_incomplete_artifact(directory: &Path) -> Result<()> {
    match inspect_final_artifact(directory) {
        FinalArtifactState::Absent | FinalArtifactState::Valid => {}
        FinalArtifactState::Invalid(reason) => {
            quarantine_final_artifact(directory)
                .with_context(|| format!("quarantining invalid artifact: {reason}"))?;
        }
        FinalArtifactState::Unverifiable(reason) => {
            anyhow::bail!("spatial covariance final artifact is unverifiable: {reason}")
        }
    }
    let mut changed = false;
    changed |=
        remove_if_exists(&directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME))?;
    changed |=
        remove_if_exists(&directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME))?;
    if changed {
        sync_directory(directory)?;
    }
    Ok(())
}

enum FinalArtifactState {
    Absent,
    Valid,
    Invalid(String),
    Unverifiable(String),
}

fn inspect_final_artifact(directory: &Path) -> FinalArtifactState {
    let final_manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let final_hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let manifest_exists = match final_manifest.try_exists() {
        Ok(value) => value,
        Err(error) => {
            return FinalArtifactState::Unverifiable(format!(
                "checking {} failed: {error}",
                final_manifest.display()
            ));
        }
    };
    let hdf5_exists = match final_hdf5.try_exists() {
        Ok(value) => value,
        Err(error) => {
            return FinalArtifactState::Unverifiable(format!(
                "checking {} failed: {error}",
                final_hdf5.display()
            ));
        }
    };
    match (manifest_exists, hdf5_exists) {
        (false, false) => FinalArtifactState::Absent,
        (true, false) | (false, true) => {
            FinalArtifactState::Invalid("factor/provenance pair is incomplete".to_owned())
        }
        (true, true) => match read_manifest_unlocked(directory) {
            Ok(_) => FinalArtifactState::Valid,
            Err(error) if verification_error_is_deterministic(&error) => {
                FinalArtifactState::Invalid(error.to_string())
            }
            Err(error) => FinalArtifactState::Unverifiable(error.to_string()),
        },
    }
}

fn verification_error_is_deterministic(error: &anyhow::Error) -> bool {
    if error.chain().any(|source| source.is::<serde_json::Error>()) {
        return true;
    }
    for source in error.chain() {
        if source.is::<std::collections::TryReserveError>() {
            return false;
        }
        if let Some(io_error) = source.downcast_ref::<std::io::Error>() {
            return io_error.kind() == std::io::ErrorKind::NotFound;
        }
    }
    let message = error.to_string();
    [
        "exceeds byte cap",
        "changed while it was read",
        "unsupported spatial covariance manifest",
        "does not match its manifest",
        "does not match metadata",
        "does not bind the current",
        "stale or mismatched",
        "requires exact nonzero scope identities",
        "cannot carry promotion receipts",
        "embedded scope does not match",
        "calibration evidence",
        "approximation result",
        "resource receipt",
        "producer commit",
    ]
    .iter()
    .any(|fragment| message.contains(fragment))
}

fn quarantine_final_artifact(directory: &Path) -> Result<()> {
    let final_manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let final_hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let mut slot = None;
    for index in 0_u32..1000 {
        let quarantined_manifest = directory.join(format!(
            "{SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME}.quarantine.{index}"
        ));
        let quarantined_hdf5 = directory.join(format!(
            "{SPATIAL_REFERENCE_COVARIANCE_FILENAME}.quarantine.{index}"
        ));
        if !quarantined_manifest.try_exists()? && !quarantined_hdf5.try_exists()? {
            slot = Some(index);
            break;
        }
    }
    let slot = slot.context("no spatial covariance quarantine slot is available")?;
    if final_manifest.try_exists()? {
        fs::rename(
            &final_manifest,
            directory.join(format!(
                "{SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME}.quarantine.{slot}"
            )),
        )?;
    }
    if final_hdf5.try_exists()? {
        fs::rename(
            &final_hdf5,
            directory.join(format!(
                "{SPATIAL_REFERENCE_COVARIANCE_FILENAME}.quarantine.{slot}"
            )),
        )?;
    }
    sync_directory(directory)
}

fn remove_if_exists(path: &Path) -> Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("removing {}", path.display())),
    }
}

fn validate_calibration_evidence(
    directory: &Path,
    metadata: &SpatialReferenceCovarianceMetadata,
    mode: EvidenceValidationMode,
) -> Result<()> {
    match metadata.calibration_scope {
        SpatialReferenceCalibrationScope::Uncalibrated => {
            anyhow::ensure!(
                metadata.review_receipt_digest.is_empty()
                    && metadata.method_manifest_digest.is_empty()
                    && metadata.calibration_scope_digest.is_empty(),
                "uncalibrated spatial covariance cannot carry promotion receipts"
            );
            Ok(())
        }
        SpatialReferenceCalibrationScope::CalibratedScopeMatch => {
            let producer_commit = metadata.producer_commit.as_deref().unwrap_or_default();
            anyhow::ensure!(
                is_git_commit(producer_commit),
                "calibrated spatial covariance requires an exact producer commit"
            );
            if let Some(compiled_commit) = option_env!("DOLPHIN_GIT_COMMIT") {
                anyhow::ensure!(
                    producer_commit == compiled_commit,
                    "calibrated spatial covariance producer commit does not bind the current code"
                );
            }
            for digest in [
                &metadata.mask_digest,
                &metadata.reference_signature_digest,
                &metadata.source_replay_digest,
                &metadata.l2_map_digest,
                &metadata.source_model_digest,
                &metadata.effective_looks_digest,
                &metadata.support_digest,
                &metadata.correction_order_digest,
                &metadata.unwrap_branch_digest,
                &metadata.burst_ownership_digest,
                &metadata.approximation_receipt_digest,
                &metadata.resource_receipt_digest,
                &metadata.review_receipt_digest,
                &metadata.method_manifest_digest,
                &metadata.calibration_scope_digest,
            ] {
                anyhow::ensure!(
                    is_nonzero_sha256(digest),
                    "calibrated spatial covariance requires exact nonzero scope identities"
                );
            }
            validate_promotion_files(directory, metadata, mode)
        }
    }
}

#[allow(clippy::too_many_lines)]
fn validate_promotion_files(
    directory: &Path,
    metadata: &SpatialReferenceCovarianceMetadata,
    mode: EvidenceValidationMode,
) -> Result<()> {
    let producer_commit = metadata.producer_commit.as_deref().unwrap_or_default();
    let code_sha256 = spatial_reference_covariance_code_digest();
    let preregistration_bytes = read_file_with_cap(
        &directory.join(SPATIAL_REFERENCE_COVARIANCE_PREREGISTRATION_FILENAME),
        "spatial covariance preregistration",
    )?;
    let preregistration_sha256 = spatial_reference_covariance_preregistration_digest();
    anyhow::ensure!(
        content_digest(&preregistration_bytes) == preregistration_sha256,
        "spatial covariance calibration evidence preregistration does not bind the current design"
    );
    let design_bytes = read_file_with_cap(
        &directory.join(SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME),
        "spatial covariance design",
    )?;
    let design_sha256 = spatial_reference_covariance_design_digest();
    anyhow::ensure!(
        content_digest(&design_bytes) == design_sha256,
        "spatial covariance calibration evidence design does not bind the current design"
    );
    let result_bytes = read_file_with_cap(
        &directory.join(SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME),
        "spatial covariance approximation result",
    )?;
    let result_sha256 = content_digest(&result_bytes);
    let result: SpatialCovarianceApproximationResult = serde_json::from_slice(&result_bytes)
        .context("parsing spatial covariance approximation result")?;
    anyhow::ensure!(
        result.schema_version == 1
            && result.method == metadata.method
            && result.method_version == metadata.method_version
            && result.crate_version == metadata.crate_version
            && result.producer_commit == producer_commit
            && result.status == "passed"
            && result.scope == result_scope(metadata)
            && result.evaluated_cases > 0
            && result.maximum_absolute_error.is_finite()
            && result.maximum_absolute_error >= 0.0
            && result.tolerance.is_finite()
            && result.tolerance >= 0.0
            && result.maximum_absolute_error <= result.tolerance,
        "spatial covariance approximation result does not bind the current scope or passed result"
    );
    let binary_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME);
    let (producer_binary_digest, _) = sha256_file_with_cap(
        &binary_path,
        BINARY_READ_CAP,
        "spatial covariance producer binary",
    )?;
    let producer_binary_sha256 = format!("sha256:{producer_binary_digest}");
    if matches!(mode, EvidenceValidationMode::Finalize) {
        let current_exe = std::env::current_exe().context("locating current producer binary")?;
        let (current_binary_digest, _) = sha256_file_with_cap(
            &current_exe,
            BINARY_READ_CAP,
            "current spatial covariance producer binary",
        )?;
        let current_binary_sha256 = format!("sha256:{current_binary_digest}");
        anyhow::ensure!(
            producer_binary_sha256 == current_binary_sha256,
            "spatial covariance calibration evidence producer binary does not bind the current binary"
        );
    }
    let approximation_bytes = read_file_with_cap(
        &directory.join(SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RECEIPT_FILENAME),
        "spatial covariance approximation receipt",
    )?;
    anyhow::ensure!(
        content_digest_matches(&metadata.approximation_receipt_digest, &approximation_bytes),
        "spatial covariance approximation receipt content hash does not match metadata"
    );
    let approximation: SpatialCovarianceApproximationReceipt =
        serde_json::from_slice(&approximation_bytes)
            .context("parsing spatial covariance approximation receipt")?;
    anyhow::ensure!(
        approximation.schema_version == 1
            && approximation.method == metadata.method
            && approximation.method_version == metadata.method_version
            && approximation.crate_version == metadata.crate_version
            && approximation.producer_commit == producer_commit
            && approximation.status == "passed"
            && approximation.code_sha256 == code_sha256
            && approximation.producer_binary_file
                == SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME
            && approximation.producer_binary_sha256 == producer_binary_sha256
            && approximation.preregistration_file
                == SPATIAL_REFERENCE_COVARIANCE_PREREGISTRATION_FILENAME
            && approximation.preregistration_sha256 == preregistration_sha256
            && approximation.design_file == SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME
            && approximation.design_sha256 == design_sha256
            && approximation.result_file
                == SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME
            && approximation.result_sha256 == result_sha256,
        "spatial covariance approximation receipt does not bind current code, design, binary, and result"
    );
    let resource_bytes = read_file_with_cap(
        &directory.join(SPATIAL_REFERENCE_COVARIANCE_RESOURCE_RECEIPT_FILENAME),
        "spatial covariance resource receipt",
    )?;
    anyhow::ensure!(
        content_digest_matches(&metadata.resource_receipt_digest, &resource_bytes),
        "spatial covariance resource receipt content hash does not match metadata"
    );
    let resource: SpatialCovarianceResourceReceipt = serde_json::from_slice(&resource_bytes)
        .context("parsing spatial covariance resource receipt")?;
    anyhow::ensure!(
        resource.schema_version == 1
            && resource.method == metadata.method
            && resource.method_version == metadata.method_version
            && resource.crate_version == metadata.crate_version
            && resource.producer_commit == producer_commit
            && resource.status == "passed"
            && resource.code_sha256 == code_sha256
            && resource.producer_binary_sha256 == producer_binary_sha256
            && resource.preregistration_sha256 == preregistration_sha256
            && resource.design_sha256 == design_sha256
            && resource.result_sha256 == result_sha256
            && resource.peak_resident_set_bytes > 0
            && resource.wall_micros > 0
            && resource.maximum_block_bytes == metadata.maximum_block_bytes,
        "spatial covariance resource receipt does not bind current code, result, and resource scope"
    );
    let review_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_REVIEW_RECEIPT_FILENAME);
    let review_bytes = read_file_with_cap(&review_path, "spatial covariance review receipt")?;
    anyhow::ensure!(
        content_digest_matches(&metadata.review_receipt_digest, &review_bytes),
        "spatial covariance review receipt content hash does not match metadata"
    );
    let review: SpatialCovarianceReviewReceipt = serde_json::from_slice(&review_bytes)
        .context("parsing spatial covariance review receipt")?;
    anyhow::ensure!(
        review.schema_version == 1
            && review.method == metadata.method
            && review.method_version == metadata.method_version
            && review.crate_version == metadata.crate_version
            && review.producer_commit == producer_commit
            && !review.reviewer.trim().is_empty()
            && review.review_status == "approved_no_unresolved_findings"
            && review.unresolved_findings == 0
            && review.code_sha256 == code_sha256
            && review.producer_binary_sha256 == producer_binary_sha256
            && review.preregistration_sha256 == preregistration_sha256
            && review.design_sha256 == design_sha256
            && review.result_sha256 == result_sha256,
        "spatial covariance review receipt does not bind the current code and analytic result"
    );
    anyhow::ensure!(
        review.approximation_receipt_digest == metadata.approximation_receipt_digest,
        "spatial covariance review receipt approximation receipt is stale or mismatched"
    );
    anyhow::ensure!(
        review.resource_receipt_digest == metadata.resource_receipt_digest,
        "spatial covariance review receipt resource receipt is stale or mismatched"
    );
    anyhow::ensure!(
        review.calibration_scope_digest == metadata.calibration_scope_digest,
        "spatial covariance review receipt scope is stale or mismatched"
    );

    let method_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_METHOD_MANIFEST_FILENAME);
    let method_bytes = read_file_with_cap(&method_path, "spatial covariance method manifest")?;
    anyhow::ensure!(
        content_digest_matches(&metadata.method_manifest_digest, &method_bytes),
        "spatial covariance method manifest content hash does not match metadata"
    );
    let method: SpatialCovarianceMethodManifest = serde_json::from_slice(&method_bytes)
        .context("parsing spatial covariance method manifest")?;
    anyhow::ensure!(
        method.schema_version == 1
            && method.method == metadata.method
            && method.method_version == metadata.method_version
            && method.crate_version == metadata.crate_version
            && method.producer_commit == producer_commit
            && method.manifest_status == "reviewed_scope_match"
            && method.code_sha256 == code_sha256
            && method.producer_binary_sha256 == producer_binary_sha256
            && method.preregistration_sha256 == preregistration_sha256
            && method.design_sha256 == design_sha256
            && method.result_sha256 == result_sha256,
        "spatial covariance method manifest does not bind the current code and analytic result"
    );
    anyhow::ensure!(
        method.approximation_receipt_digest == metadata.approximation_receipt_digest,
        "spatial covariance method manifest approximation receipt is stale or mismatched"
    );
    anyhow::ensure!(
        method.resource_receipt_digest == metadata.resource_receipt_digest,
        "spatial covariance method manifest resource receipt is stale or mismatched"
    );
    anyhow::ensure!(
        method.review_receipt_digest == metadata.review_receipt_digest,
        "spatial covariance method manifest review receipt is stale or mismatched"
    );
    anyhow::ensure!(
        method.calibration_scope_digest == metadata.calibration_scope_digest,
        "spatial covariance method manifest scope is stale or mismatched"
    );
    Ok(())
}

/// Derive the exact production-code identity required by calibrated evidence.
#[must_use]
pub fn spatial_reference_covariance_code_digest() -> String {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:spatial-reference-production-code:v1");
    for bytes in CODE_BYTES {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("sha256:{:x}", digest.finalize())
}

/// Derive the embedded preregistration identity required by calibrated evidence.
#[must_use]
pub fn spatial_reference_covariance_preregistration_digest() -> String {
    content_digest(PREREGISTRATION_BYTES)
}

/// Derive the embedded design identity required by calibrated evidence.
#[must_use]
pub fn spatial_reference_covariance_design_digest() -> String {
    content_digest(DESIGN_BYTES)
}

fn result_scope(metadata: &SpatialReferenceCovarianceMetadata) -> SpatialCovarianceResultScope {
    SpatialCovarianceResultScope {
        burst_id: metadata.burst_id.clone(),
        crs: metadata.crs.clone(),
        units: metadata.units.clone(),
        geotransform: metadata.geotransform,
        acquisition_days: metadata.acquisition_days.clone(),
        grid_row_start: metadata.full_grid.row_start,
        grid_col_start: metadata.full_grid.col_start,
        grid_rows: metadata.full_grid.rows,
        grid_cols: metadata.full_grid.cols,
        grid_stride_y: metadata.full_grid.stride_y,
        grid_stride_x: metadata.full_grid.stride_x,
        reference_row: metadata.reference_row,
        reference_col: metadata.reference_col,
        gauge_date_index: metadata.gauge_date_index,
        ordered_date_indices: metadata.ordered_date_indices.clone(),
        mask_digest: metadata.mask_digest.clone(),
        source_replay_digest: metadata.source_replay_digest.clone(),
        l2_map_digest: metadata.l2_map_digest.clone(),
        reference_signature_digest: metadata.reference_signature_digest.clone(),
        source_model_digest: metadata.source_model_digest.clone(),
        effective_looks_digest: metadata.effective_looks_digest.clone(),
        support_method: metadata.support_method.clone(),
        support_digest: metadata.support_digest.clone(),
        correction_order_digest: metadata.correction_order_digest.clone(),
        unwrap_branch_digest: metadata.unwrap_branch_digest.clone(),
        burst_ownership_digest: metadata.burst_ownership_digest.clone(),
        source_burst_ids: metadata.source_burst_ids.clone(),
        reference_source_burst_index: metadata.reference_source_burst_index,
        maximum_block_bytes: metadata.maximum_block_bytes,
    }
}

fn content_digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn content_digest_matches(expected: &str, bytes: &[u8]) -> bool {
    expected == content_digest(bytes)
}

fn read_file_with_cap(path: &Path, label: &str) -> Result<Vec<u8>> {
    let expected_bytes = fs::metadata(path)
        .with_context(|| format!("reading {label} metadata {}", path.display()))?
        .len();
    anyhow::ensure!(
        expected_bytes <= METADATA_READ_CAP,
        "{label} byte count {expected_bytes} exceeds byte cap {METADATA_READ_CAP}"
    );
    let capacity = usize::try_from(expected_bytes).context("bounded file size exceeds usize")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .with_context(|| format!("reserving {label} buffer"))?;
    File::open(path)
        .with_context(|| format!("opening {label} {}", path.display()))?
        .take(METADATA_READ_CAP + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {label} {}", path.display()))?;
    let actual_bytes = u64::try_from(bytes.len()).context("bounded file size exceeds u64")?;
    anyhow::ensure!(
        actual_bytes == expected_bytes && actual_bytes <= METADATA_READ_CAP,
        "{label} changed while it was read"
    );
    Ok(bytes)
}

fn is_nonzero_sha256(value: &str) -> bool {
    let hex = value.strip_prefix("sha256:").unwrap_or(value);
    hex.len() == 64
        && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
        && hex.bytes().any(|byte| byte != b'0')
}

fn is_git_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn manifest(
    metadata: &SpatialReferenceCovarianceMetadata,
    hdf5_sha256: String,
    hdf5_bytes: u64,
) -> SpatialReferenceCovarianceArtifactManifest {
    let calibration_scope = match metadata.calibration_scope {
        SpatialReferenceCalibrationScope::Uncalibrated => "uncalibrated",
        SpatialReferenceCalibrationScope::CalibratedScopeMatch => "calibrated_scope_match",
    };
    SpatialReferenceCovarianceArtifactManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        method: metadata.method.clone(),
        method_version: metadata.method_version,
        crate_version: metadata.crate_version.clone(),
        producer_commit: metadata.producer_commit.clone(),
        hdf5_file: SPATIAL_REFERENCE_COVARIANCE_FILENAME.to_owned(),
        hdf5_bytes,
        hdf5_sha256,
        burst_id: metadata.burst_id.clone(),
        crs: metadata.crs.clone(),
        units: metadata.units.clone(),
        geotransform: metadata.geotransform,
        acquisition_days: metadata.acquisition_days.clone(),
        reference_signature_digest: metadata.reference_signature_digest.clone(),
        mask_digest: metadata.mask_digest.clone(),
        source_replay_digest: metadata.source_replay_digest.clone(),
        l2_map_digest: metadata.l2_map_digest.clone(),
        approximation_receipt_digest: metadata.approximation_receipt_digest.clone(),
        resource_receipt_digest: metadata.resource_receipt_digest.clone(),
        review_receipt_digest: metadata.review_receipt_digest.clone(),
        method_manifest_digest: metadata.method_manifest_digest.clone(),
        calibration_scope_digest: metadata.calibration_scope_digest.clone(),
        source_model_digest: metadata.source_model_digest.clone(),
        effective_looks_digest: metadata.effective_looks_digest.clone(),
        effective_looks_fraction_dataset: "blocks/{block_id:020}/effective_looks_fraction"
            .to_owned(),
        support_union_count_dataset: "blocks/{block_id:020}/support_union_count".to_owned(),
        effective_looks_receipt_dataset: "blocks/{block_id:020}/effective_looks_receipt".to_owned(),
        resource_high_water_bytes_dataset: "blocks/{block_id:020}/resource_high_water_bytes"
            .to_owned(),
        support_method: metadata.support_method.clone(),
        support_digest: metadata.support_digest.clone(),
        correction_order_digest: metadata.correction_order_digest.clone(),
        unwrap_branch_digest: metadata.unwrap_branch_digest.clone(),
        burst_ownership_digest: metadata.burst_ownership_digest.clone(),
        source_burst_ids: metadata.source_burst_ids.clone(),
        reference_source_burst_index: metadata.reference_source_burst_index,
        calibration_scope: calibration_scope.to_owned(),
        maximum_block_bytes: metadata.maximum_block_bytes,
    }
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        bytes = bytes
            .checked_add(count as u64)
            .context("spatial covariance byte count overflow")?;
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

fn sha256_file_with_cap(path: &Path, cap: u64, label: &str) -> Result<(String, u64)> {
    let expected_bytes = fs::metadata(path)
        .with_context(|| format!("reading {label} metadata {}", path.display()))?
        .len();
    anyhow::ensure!(
        expected_bytes <= cap,
        "{label} byte count {expected_bytes} exceeds byte cap {cap}"
    );
    let result =
        sha256_file(path).with_context(|| format!("hashing {label} {}", path.display()))?;
    anyhow::ensure!(
        result.1 == expected_bytes,
        "{label} changed while it was read"
    );
    Ok(result)
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
