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
pub const SPATIAL_REFERENCE_COVARIANCE_FILENAME: &str = "spatial_reference_covariance.h5";
/// JSON completion marker written after the HDF5 rename.
pub const SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME: &str =
    "spatial_reference_covariance.json";
const LOCK_FILENAME: &str = "spatial_reference_covariance.capture.lock";
const MANIFEST_SCHEMA_VERSION: u16 = 1;
const METADATA_READ_CAP: u64 = 1024 * 1024;

/// Durable receipt binding factor bytes to every scope identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
        let lock_path = directory.join(LOCK_FILENAME);
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
        Ok(Self {
            directory: directory.to_owned(),
            _lock: lock,
        })
    }

    /// Non-Unix targets cannot provide the required durable lock.
    #[cfg(not(unix))]
    pub fn acquire(_directory: &Path) -> Result<Self> {
        anyhow::bail!("spatial covariance artifact locking is unsupported on this platform")
    }
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
        hdf5_scratch.parent() == Some(directory.as_path()),
        "spatial covariance scratch file must be inside the work directory"
    );
    let embedded = read_spatial_reference_covariance_header(hdf5_scratch, METADATA_READ_CAP)
        .context("validating spatial covariance HDF5 metadata")?;
    anyhow::ensure!(
        embedded == *metadata,
        "spatial covariance HDF5 metadata differs from finalization metadata"
    );
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
    let scratch_manifest = directory.join(format!(
        ".{SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME}.scratch"
    ));
    write_synced(&scratch_manifest, &manifest_bytes)?;

    let final_manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let final_hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    match fs::remove_file(&final_manifest) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("removing prior spatial covariance manifest"),
    }
    sync_directory(directory)?;
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
    let manifest_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "reading spatial covariance manifest {}",
            manifest_path.display()
        )
    })?;
    anyhow::ensure!(
        bytes.len() <= METADATA_READ_CAP as usize,
        "spatial covariance manifest exceeds byte cap"
    );
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
    anyhow::ensure!(
        manifest(&embedded, digest, byte_count) == parsed,
        "spatial covariance embedded scope does not match its manifest"
    );
    Ok(parsed)
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
        reference_signature_digest: metadata.reference_signature_digest.clone(),
        mask_digest: metadata.mask_digest.clone(),
        source_replay_digest: metadata.source_replay_digest.clone(),
        l2_map_digest: metadata.l2_map_digest.clone(),
        approximation_receipt_digest: metadata.approximation_receipt_digest.clone(),
        resource_receipt_digest: metadata.resource_receipt_digest.clone(),
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
