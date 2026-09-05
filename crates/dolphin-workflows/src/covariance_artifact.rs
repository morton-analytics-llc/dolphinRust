//! Transactional provenance for the persisted sequential covariance operator.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

use anyhow::{Context, Result};
use dolphin_io::{
    read_covariance_operator_header_with_byte_cap, CovarianceCalibrationStatus,
    CovarianceOperatorMetadata, CovarianceOperatorWriteReceipt, CovarianceReplayStatus,
    DownstreamInferenceStatus, StitchedCovarianceStatus, COVARIANCE_OPERATOR_METHOD,
    COVARIANCE_OPERATOR_METHOD_VERSION, COVARIANCE_OPERATOR_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Final HDF5 operator filename.
pub const COVARIANCE_OPERATOR_FILENAME: &str = "phase_covariance_operator.h5";
/// JSON commit marker written only after the HDF5 file is finalized.
pub const COVARIANCE_OPERATOR_MANIFEST_FILENAME: &str = "phase_covariance_provenance.json";
pub(crate) const COVARIANCE_OPERATOR_LOCK_FILENAME: &str = "phase_covariance_operator.capture.lock";
const MANIFEST_SCHEMA_VERSION: u16 = 1;
const CALIBRATION_STATUS: &str = "uncalibrated";
const DOWNSTREAM_INFERENCE_STATUS: &str = "blocked_pending_issue_54_and_53";

pub(crate) struct CovarianceArtifactReadLock {
    _file: File,
}

#[cfg(unix)]
pub(crate) fn acquire_covariance_artifact_read_lock(
    directory: &Path,
) -> Result<CovarianceArtifactReadLock> {
    let path = directory.join(COVARIANCE_OPERATOR_LOCK_FILENAME);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("opening covariance artifact lock {}", path.display()))?;
    // SAFETY: `file` owns a live descriptor for the duration of this call.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    anyhow::ensure!(
        result == 0,
        "covariance artifact is being replaced by an active capture"
    );
    Ok(CovarianceArtifactReadLock { _file: file })
}

#[cfg(not(unix))]
pub(crate) fn acquire_covariance_artifact_read_lock(
    _directory: &Path,
) -> Result<CovarianceArtifactReadLock> {
    anyhow::bail!("covariance artifact read locking is unsupported on this platform")
}

/// Exclusive lock held from scratch recovery through artifact commit.
pub struct CovarianceArtifactTransaction {
    directory: PathBuf,
    _file: File,
}

impl CovarianceArtifactTransaction {
    /// Acquire the artifact's exclusive writer lock without waiting.
    ///
    /// # Errors
    /// Returns an error while a replay reader or another writer owns the artifact.
    #[cfg(unix)]
    pub fn acquire(directory: &Path) -> Result<Self> {
        let path = directory.join(COVARIANCE_OPERATOR_LOCK_FILENAME);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("opening covariance artifact lock {}", path.display()))?;
        // SAFETY: `file` owns a live descriptor for the lifetime of the guard.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        anyhow::ensure!(
            result == 0,
            "covariance artifact has an active replay reader or writer"
        );
        Ok(Self {
            directory: directory.to_owned(),
            _file: file,
        })
    }

    /// Acquire the artifact's exclusive writer lock without waiting.
    ///
    /// # Errors
    /// Always returns an error because the transaction lock is unsupported.
    #[cfg(not(unix))]
    pub fn acquire(_directory: &Path) -> Result<Self> {
        anyhow::bail!("covariance artifact transaction locking is unsupported on this platform")
    }
}

/// Disk-space admission captured before the HDF5 scratch writer allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CovarianceArtifactDiskAdmission {
    /// Conservative projected final HDF5 size.
    pub projected_final_bytes: u64,
    /// Conservative peak temporary disk for the global identity index.
    pub projected_identity_index_peak_bytes: u64,
    /// Final plus scratch bytes with a 25 percent free-space margin.
    pub required_free_bytes: u64,
    /// Filesystem free bytes observed at admission.
    pub available_free_bytes: u64,
}

/// Durable receipt binding the operator bytes to their source/model provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CovarianceArtifactManifest {
    /// Manifest schema version.
    pub schema_version: u16,
    /// Persisted operator method.
    pub method: String,
    /// Persisted operator method version.
    pub method_version: u16,
    /// Producing crate version.
    pub crate_version: String,
    /// Producing Git commit when supplied by the build.
    pub producer_commit: Option<String>,
    /// Exact gauge date, fixed to acquisition 0 for version 1.
    pub gauge_date_index: u32,
    /// Final HDF5 filename, relative to the manifest.
    pub hdf5_file: String,
    /// Final HDF5 byte count.
    pub hdf5_bytes: u64,
    /// Lowercase SHA-256 digest of the final HDF5 bytes.
    pub hdf5_sha256: String,
    /// Disk-space admission checked before the scratch artifact was created.
    pub disk_admission: CovarianceArtifactDiskAdmission,
    /// Observed peak temporary bytes used by the global identity index.
    pub identity_index_peak_bytes: u64,
    /// Digest of the normalized producer configuration.
    pub normalized_config_digest: String,
    /// Digest of the derivative/replay kernel.
    pub kernel_digest: String,
    /// Ordered source-manifest digest, when the caller supplied one.
    pub source_manifest_digest: Option<String>,
    /// External source resolver identity.
    pub source_provider: Option<String>,
    /// External source resolver version.
    pub source_provider_version: Option<String>,
    /// Caller-supplied proper-complex source model.
    pub source_model: Option<String>,
    /// Caller-supplied source-model version.
    pub source_model_version: Option<String>,
    /// Digest of the ordered source provider/model names and versions.
    pub source_model_version_digest: Option<String>,
    /// Digest of the caller's source-model receipt.
    pub source_model_receipt_digest: Option<String>,
    /// Whether immutable raw/model inputs can be resolved and replayed.
    pub replay_status: String,
    /// Whether any stitched seam covariance is represented.
    pub stitched_covariance_status: String,
    /// Explicitly uncalibrated producer status.
    pub calibration_status: String,
    /// Downstream inference remains blocked until issues #54 and #53 pass.
    pub downstream_inference_status: String,
}

/// Required free space for a projected final artifact plus its scratch copy and
/// a 25 percent margin over their combined size.
///
/// # Errors
/// Returns an error when the multiplication exceeds `u64`.
pub fn covariance_artifact_disk_bytes(projected_final_bytes: u64) -> Result<u64> {
    covariance_artifact_disk_bytes_with_identity_index(projected_final_bytes, 0)
}

/// Required free space including final HDF5, scratch HDF5, the bounded
/// identity-index workspace, and a 25 percent margin.
///
/// # Errors
/// Returns an error when the projection exceeds `u64`.
pub fn covariance_artifact_disk_bytes_with_identity_index(
    projected_final_bytes: u64,
    projected_identity_index_peak_bytes: u64,
) -> Result<u64> {
    projected_final_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(projected_identity_index_peak_bytes))
        .and_then(|bytes| bytes.checked_mul(5))
        .and_then(|bytes| bytes.checked_add(3))
        .map(|bytes| bytes / 4)
        .context("covariance artifact disk preflight overflow")
}

/// Validate a supplied free-space observation against the artifact disk bound.
///
/// # Errors
/// Returns an error for arithmetic overflow or insufficient free space.
pub fn admit_covariance_artifact_disk(
    projected_final_bytes: u64,
    available_free_bytes: u64,
) -> Result<CovarianceArtifactDiskAdmission> {
    admit_covariance_artifact_disk_with_identity_index(
        projected_final_bytes,
        0,
        available_free_bytes,
    )
}

/// Validate free space against final, scratch, and identity-index projections.
///
/// # Errors
/// Returns an error for a zero HDF5 projection, arithmetic overflow, or insufficient space.
pub fn admit_covariance_artifact_disk_with_identity_index(
    projected_final_bytes: u64,
    projected_identity_index_peak_bytes: u64,
    available_free_bytes: u64,
) -> Result<CovarianceArtifactDiskAdmission> {
    anyhow::ensure!(
        projected_final_bytes > 0,
        "covariance artifact projected size must be positive"
    );
    let required_free_bytes = covariance_artifact_disk_bytes_with_identity_index(
        projected_final_bytes,
        projected_identity_index_peak_bytes,
    )?;
    anyhow::ensure!(
        available_free_bytes >= required_free_bytes,
        "covariance artifact requires {required_free_bytes} free bytes but only {available_free_bytes} are available"
    );
    Ok(CovarianceArtifactDiskAdmission {
        projected_final_bytes,
        projected_identity_index_peak_bytes,
        required_free_bytes,
        available_free_bytes,
    })
}

/// Query the target filesystem and enforce artifact disk admission before
/// scratch-file creation.
///
/// # Errors
/// Returns an error when free space cannot be queried or is below the required
/// final-plus-scratch margin.
#[cfg(unix)]
pub fn preflight_covariance_artifact_disk(
    directory: &Path,
    projected_final_bytes: u64,
) -> Result<CovarianceArtifactDiskAdmission> {
    preflight_covariance_artifact_disk_with_identity_index(directory, projected_final_bytes, 0)
}

/// Query the target filesystem and admit the HDF5 plus identity workspace.
///
/// # Errors
/// Returns an error when free space cannot be queried or is below the bound.
#[cfg(unix)]
pub fn preflight_covariance_artifact_disk_with_identity_index(
    directory: &Path,
    projected_final_bytes: u64,
    projected_identity_index_peak_bytes: u64,
) -> Result<CovarianceArtifactDiskAdmission> {
    let path = CString::new(directory.as_os_str().as_bytes())
        .context("covariance artifact directory contains a NUL byte")?;
    // SAFETY: `stats` is initialized for `statvfs`, and `path` is a live,
    // NUL-terminated CString for the duration of the call.
    let stats = unsafe {
        let mut stats = std::mem::zeroed::<libc::statvfs>();
        if libc::statvfs(path.as_ptr(), &mut stats) != 0 {
            return Err(std::io::Error::last_os_error())
                .context("querying covariance artifact free space");
        }
        stats
    };
    let available = u128::from(stats.f_bavail)
        .checked_mul(u128::from(stats.f_frsize))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("covariance artifact free-space count overflow")?;
    admit_covariance_artifact_disk_with_identity_index(
        projected_final_bytes,
        projected_identity_index_peak_bytes,
        available,
    )
}

/// Non-Unix targets do not currently expose the required durable disk receipt.
#[cfg(not(unix))]
pub fn preflight_covariance_artifact_disk(
    _directory: &Path,
    _projected_final_bytes: u64,
) -> Result<CovarianceArtifactDiskAdmission> {
    anyhow::bail!("covariance artifact disk preflight is unsupported on this platform")
}

/// Non-Unix targets do not currently expose the required durable disk receipt.
#[cfg(not(unix))]
pub fn preflight_covariance_artifact_disk_with_identity_index(
    _directory: &Path,
    _projected_final_bytes: u64,
    _projected_identity_index_peak_bytes: u64,
) -> Result<CovarianceArtifactDiskAdmission> {
    anyhow::bail!("covariance artifact disk preflight is unsupported on this platform")
}

/// Atomically finalize an already-closed HDF5 scratch file and write its JSON
/// manifest last. The manifest is the only completion marker.
///
/// # Errors
/// Returns an error for invalid method/status metadata, an out-of-directory
/// scratch path, I/O failure, or digest failure. A failed transaction does not
/// leave a new completion marker.
pub fn finalize_covariance_artifact(
    transaction: &CovarianceArtifactTransaction,
    hdf5_scratch: &Path,
    metadata: &CovarianceOperatorMetadata,
    disk_admission: CovarianceArtifactDiskAdmission,
    write_receipt: &CovarianceOperatorWriteReceipt,
) -> Result<CovarianceArtifactManifest> {
    let directory = &transaction.directory;
    validate_metadata(metadata)?;
    anyhow::ensure!(
        hdf5_scratch.parent() == Some(directory),
        "covariance HDF5 scratch file must be inside the work directory"
    );
    let embedded = read_covariance_operator_header_with_byte_cap(
        hdf5_scratch,
        write_receipt.metadata_validation_bytes,
    )
    .context("validating completed covariance HDF5 metadata")?;
    anyhow::ensure!(
        embedded == *metadata,
        "covariance HDF5 metadata does not match finalization metadata"
    );
    File::open(hdf5_scratch)
        .context("opening completed covariance HDF5 scratch file")?
        .sync_all()
        .context("syncing completed covariance HDF5 scratch file")?;
    let (hdf5_sha256, hdf5_bytes) = sha256_file(hdf5_scratch)?;
    anyhow::ensure!(
        hdf5_bytes == write_receipt.sealed_hdf5_bytes
            && hdf5_sha256 == write_receipt.sealed_hdf5_sha256,
        "covariance HDF5 changed after the writer sealed it"
    );
    anyhow::ensure!(
        hdf5_bytes <= disk_admission.projected_final_bytes,
        "covariance HDF5 exceeded its admitted projected size"
    );
    anyhow::ensure!(
        write_receipt.peak_identity_index_disk_bytes
            <= disk_admission.projected_identity_index_peak_bytes,
        "covariance identity-index peak exceeded its admitted projection"
    );
    let manifest = manifest(
        metadata,
        hdf5_sha256,
        hdf5_bytes,
        disk_admission,
        write_receipt.peak_identity_index_disk_bytes,
    );
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;

    let final_hdf5 = directory.join(COVARIANCE_OPERATOR_FILENAME);
    let final_manifest = directory.join(COVARIANCE_OPERATOR_MANIFEST_FILENAME);
    let scratch_manifest = manifest_scratch_path(directory);
    write_synced(&scratch_manifest, &manifest_bytes)?;

    // An old marker must not name bytes while the fixed HDF5 path is replaced.
    match fs::remove_file(&final_manifest) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("removing prior covariance manifest"),
    }
    sync_directory(directory)?;
    fs::rename(hdf5_scratch, &final_hdf5).context("finalizing covariance HDF5")?;
    sync_directory(directory)?;
    fs::rename(&scratch_manifest, &final_manifest)
        .context("committing covariance provenance manifest")?;
    sync_directory(directory)?;
    Ok(manifest)
}

/// Read and verify a completed covariance artifact manifest.
///
/// # Errors
/// Returns an error when the manifest or HDF5 file is missing, malformed, or
/// differs in size/digest from the committed receipt.
pub fn read_covariance_artifact_manifest(directory: &Path) -> Result<CovarianceArtifactManifest> {
    read_covariance_artifact_manifest_with_byte_cap(directory, u64::MAX)
}

/// Read and verify a completed artifact while capping embedded HDF5 metadata
/// and topology allocations.
///
/// The streaming SHA-256 pass uses a fixed-size buffer; `byte_cap` bounds the
/// subsequent HDF5 validation workspace.
///
/// # Errors
/// Returns an error when manifest verification fails or embedded HDF5
/// validation exceeds `byte_cap`.
pub fn read_covariance_artifact_manifest_with_byte_cap(
    directory: &Path,
    byte_cap: u64,
) -> Result<CovarianceArtifactManifest> {
    let manifest_path = directory.join(COVARIANCE_OPERATOR_MANIFEST_FILENAME);
    let bytes =
        read_file_with_byte_cap(&manifest_path, byte_cap, "covariance provenance manifest")?;
    let manifest: CovarianceArtifactManifest =
        serde_json::from_slice(&bytes).context("parsing covariance provenance manifest")?;
    anyhow::ensure!(
        manifest.schema_version == MANIFEST_SCHEMA_VERSION,
        "unsupported covariance provenance schema version {}",
        manifest.schema_version
    );
    anyhow::ensure!(
        manifest.method == COVARIANCE_OPERATOR_METHOD
            && manifest.method_version == COVARIANCE_OPERATOR_METHOD_VERSION
            && manifest.gauge_date_index == 0,
        "unsupported covariance artifact method {}",
        manifest.method
    );
    anyhow::ensure!(
        manifest.hdf5_file == COVARIANCE_OPERATOR_FILENAME,
        "covariance manifest names an unsupported HDF5 path"
    );
    anyhow::ensure!(
        manifest.calibration_status == CALIBRATION_STATUS
            && manifest.downstream_inference_status == DOWNSTREAM_INFERENCE_STATUS,
        "covariance manifest cannot authorize calibrated downstream inference"
    );
    let hdf5_path = directory.join(&manifest.hdf5_file);
    let byte_count = fs::metadata(&hdf5_path)
        .context("reading covariance HDF5 metadata")?
        .len();
    anyhow::ensure!(
        byte_count == manifest.hdf5_bytes,
        "covariance HDF5 byte count does not match its manifest"
    );
    anyhow::ensure!(
        byte_count <= manifest.disk_admission.projected_final_bytes,
        "covariance HDF5 exceeds the admitted projected size"
    );
    let (digest, hashed_byte_count) = sha256_file(&hdf5_path)?;
    anyhow::ensure!(
        hashed_byte_count == byte_count,
        "covariance HDF5 changed while its digest was computed"
    );
    anyhow::ensure!(
        digest == manifest.hdf5_sha256,
        "covariance HDF5 digest does not match its manifest"
    );
    let checked_disk = admit_covariance_artifact_disk_with_identity_index(
        manifest.disk_admission.projected_final_bytes,
        manifest.disk_admission.projected_identity_index_peak_bytes,
        manifest.disk_admission.available_free_bytes,
    )?;
    anyhow::ensure!(
        checked_disk == manifest.disk_admission,
        "covariance disk admission receipt is inconsistent"
    );
    let embedded = read_covariance_operator_header_with_byte_cap(&hdf5_path, byte_cap)
        .context("reading committed covariance HDF5 metadata")?;
    validate_metadata(&embedded)?;
    anyhow::ensure!(
        manifest.identity_index_peak_bytes
            <= manifest.disk_admission.projected_identity_index_peak_bytes
            && self::manifest(
                &embedded,
                digest,
                byte_count,
                checked_disk,
                manifest.identity_index_peak_bytes,
            ) == manifest,
        "covariance manifest metadata does not match the committed HDF5"
    );
    Ok(manifest)
}

fn read_file_with_byte_cap(path: &Path, byte_cap: u64, label: &str) -> Result<Vec<u8>> {
    let expected_bytes = fs::metadata(path)
        .with_context(|| format!("reading {label} metadata"))?
        .len();
    anyhow::ensure!(
        expected_bytes <= byte_cap,
        "{label} byte count {expected_bytes} exceeds byte cap {byte_cap}"
    );
    let limit = byte_cap.saturating_add(1);
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("opening {label}"))?
        .take(limit)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {label}"))?;
    let actual_bytes = u64::try_from(bytes.len()).context("manifest byte count exceeds u64")?;
    anyhow::ensure!(
        actual_bytes <= byte_cap,
        "{label} changed while it was read or exceeds byte cap {byte_cap}"
    );
    anyhow::ensure!(
        actual_bytes == expected_bytes,
        "{label} changed while it was read"
    );
    Ok(bytes)
}

fn validate_metadata(metadata: &CovarianceOperatorMetadata) -> Result<()> {
    anyhow::ensure!(
        metadata.schema_version == COVARIANCE_OPERATOR_SCHEMA_VERSION
            && metadata.method == COVARIANCE_OPERATOR_METHOD
            && metadata.method_version == COVARIANCE_OPERATOR_METHOD_VERSION,
        "unsupported covariance operator method {}",
        metadata.method
    );
    anyhow::ensure!(
        metadata.gauge_date_index == 0,
        "covariance operator version 1 requires acquisition-0 gauge"
    );
    anyhow::ensure!(
        metadata.calibration_status == CovarianceCalibrationStatus::Uncalibrated,
        "covariance operator cannot claim calibrated source uncertainty"
    );
    anyhow::ensure!(
        metadata.downstream_inference_status
            == DownstreamInferenceStatus::BlockedPendingIssue54And53,
        "covariance operator cannot authorize downstream inference"
    );
    Ok(())
}

fn manifest(
    metadata: &CovarianceOperatorMetadata,
    hdf5_sha256: String,
    hdf5_bytes: u64,
    disk_admission: CovarianceArtifactDiskAdmission,
    identity_index_peak_bytes: u64,
) -> CovarianceArtifactManifest {
    CovarianceArtifactManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        method: metadata.method.clone(),
        method_version: metadata.method_version,
        crate_version: metadata.crate_version.clone(),
        producer_commit: metadata.producer_commit.clone(),
        gauge_date_index: metadata.gauge_date_index,
        hdf5_file: COVARIANCE_OPERATOR_FILENAME.to_owned(),
        hdf5_bytes,
        hdf5_sha256,
        disk_admission,
        identity_index_peak_bytes,
        normalized_config_digest: metadata.normalized_config_digest.clone(),
        kernel_digest: metadata.kernel_digest.clone(),
        source_manifest_digest: metadata.source.manifest_digest.clone(),
        source_provider: metadata.source.provider.clone(),
        source_provider_version: metadata.source.provider_version.clone(),
        source_model: metadata.source.model.clone(),
        source_model_version: metadata.source.model_version.clone(),
        source_model_version_digest: metadata.source.model_version_digest.clone(),
        source_model_receipt_digest: metadata.source.model_receipt_digest.clone(),
        replay_status: replay_status_name(metadata.replay_status).to_owned(),
        stitched_covariance_status: stitched_status_name(metadata.stitched_status).to_owned(),
        calibration_status: CALIBRATION_STATUS.to_owned(),
        downstream_inference_status: DOWNSTREAM_INFERENCE_STATUS.to_owned(),
    }
}

const fn replay_status_name(status: CovarianceReplayStatus) -> &'static str {
    match status {
        CovarianceReplayStatus::Replayable => "replayable",
        CovarianceReplayStatus::SourceManifestMissing => "source_manifest_missing",
        CovarianceReplayStatus::SourceManifestMismatch => "source_manifest_mismatch",
        CovarianceReplayStatus::SupportNotFrozen => "support_not_frozen",
        CovarianceReplayStatus::UnsupportedBackend => "unsupported_backend",
        CovarianceReplayStatus::SourceUnavailable => "source_unavailable",
        CovarianceReplayStatus::SourceModelUnavailable => "source_model_unavailable",
    }
}

const fn stitched_status_name(status: StitchedCovarianceStatus) -> &'static str {
    match status {
        StitchedCovarianceStatus::NotStitched => "not_stitched",
        StitchedCovarianceStatus::UnsupportedSeamCovariance => "unsupported_seam_covariance",
    }
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let file = File::open(path)
        .with_context(|| format!("opening covariance artifact {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut byte_count = 0_u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        byte_count = byte_count
            .checked_add(read as u64)
            .context("covariance artifact byte count overflow")?;
    }
    Ok((format!("{:x}", digest.finalize()), byte_count))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("creating covariance manifest {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(directory: &Path) -> Result<()> {
    let directory_file = File::open(directory).with_context(|| {
        format!(
            "opening covariance artifact directory {}",
            directory.display()
        )
    })?;
    directory_file
        .sync_all()
        .context("syncing covariance artifact directory")
}

fn manifest_scratch_path(directory: &Path) -> PathBuf {
    directory.join(format!("{COVARIANCE_OPERATOR_MANIFEST_FILENAME}.scratch"))
}
