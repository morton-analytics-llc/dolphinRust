//! Immutable CSLC-member identity and empirical primitive-source resolution.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use dolphin_core::config::{EmpiricalSourceFactorOptions, InputType};
use dolphin_core::{BlockIndices, Cf64};
use dolphin_io::{
    covariance_content_bound_source_id, covariance_identified_id, read_cslc_shape,
    read_cslc_window, read_nisar_window, CovarianceOperatorGrid,
};
use dolphin_phaselink::{
    estimate_empirical_proper_complex_factor, EmpiricalProperComplexConfig, SourceId,
    EMPIRICAL_PROPER_COMPLEX_METHOD,
};
use ndarray::{s, Array1, Array2, Array3, Axis};
use sha2::{Digest, Sha256};

use crate::sequential_covariance::{
    primitive_source_content_digest, ReplayStatus, ResolvedPrimitiveSource,
    SequentialPrimitiveSourceResolver, SequentialReplayBlock, SequentialReplayError,
    SequentialSourceProviderIdentity,
};

/// Production raw-source provider persisted with covariance replay artifacts.
pub const CSLC_COVARIANCE_SOURCE_PROVIDER: &str =
    "dolphin_workflows_cslc_member_bytes_exact_empirical_receipt";
/// Version of the ordered decoded-member identity and canonical-window reader.
pub const CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION: &str = "2";
/// Production proper-complex source-factor method.
pub const CSLC_COVARIANCE_SOURCE_MODEL: &str = EMPIRICAL_PROPER_COMPLEX_METHOD;
/// Production proper-complex source-factor method version.
pub const CSLC_COVARIANCE_SOURCE_MODEL_VERSION: &str = "1";

const MEMBER_DIGEST_STRIPE_ROWS: usize = 256;
const MANIFEST_IDENTITY_PASSES: u64 = 3;

#[derive(Debug, Clone)]
struct CslcMemberIdentity {
    path: PathBuf,
    shape: (usize, usize),
    content_digest: [u8; 32],
    file_fingerprint: CslcFileFingerprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CslcFileFingerprint {
    length: u64,
    modified: Option<SystemTime>,
}

#[derive(Debug)]
struct CslcSourceTileCache {
    block: BlockIndices,
    values: Array3<Cf64>,
    validity: Array2<bool>,
}

/// Bounded physical-read and resident-cache evidence for one resolver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CslcCovarianceResolverMetrics {
    /// Physical CSLC member hyperslab reads used to populate tile caches.
    pub member_window_reads: u64,
    /// Expanded tile-cache loads.
    pub tile_cache_loads: u64,
    /// Primitive factors actually resolved; masked sources are excluded.
    pub source_resolutions: u64,
    /// Peak resident decoded tile-cache bytes.
    pub peak_cached_bytes: u64,
}

/// Ordered immutable identity of every configured CSLC member.
#[derive(Debug, Clone)]
pub struct CslcCovarianceManifest {
    input_type: InputType,
    subdataset: String,
    members: Vec<CslcMemberIdentity>,
    digest: [u8; 32],
    resource_estimate: CslcManifestResourceEstimate,
}

/// Conservative decoded-content hashing preflight for one ordered manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CslcManifestResourceEstimate {
    /// Ordered member count.
    pub member_count: u64,
    /// Conservative decoded complex-f32 bytes for capture plus two final checks.
    pub decoded_content_bytes: u64,
    /// Physical stripe reads for capture plus two final checks.
    pub identity_window_reads: u64,
    /// Maximum resident decoded stripe bytes.
    pub maximum_resident_bytes: u64,
}

/// Canonical validity reader paired with immutable CSLC source support.
pub trait CslcCovarianceValidityReader {
    /// Read validity for one absolute native-grid window.
    ///
    /// # Errors
    /// Returns an error if the exact support cannot be reproduced.
    fn read_validity(&self, block: BlockIndices) -> Result<Array2<bool>, SequentialReplayError>;
}

impl CslcCovarianceManifest {
    /// Read every configured member and bind its ordered decoded complex-f32 bytes.
    ///
    /// # Errors
    /// Returns an error if any member or subdataset is missing, malformed, or unreadable.
    pub fn capture(
        input_type: InputType,
        subdataset: impl Into<String>,
        paths: &[PathBuf],
    ) -> Result<Self> {
        let subdataset = subdataset.into();
        anyhow::ensure!(!subdataset.is_empty(), "CSLC subdataset identity is empty");
        anyhow::ensure!(!paths.is_empty(), "CSLC source manifest is empty");
        let mut members = Vec::with_capacity(paths.len());
        for path in paths {
            let shape = read_cslc_shape(path, &subdataset)
                .with_context(|| format!("reading CSLC member shape from {}", path.display()))?;
            members.push(CslcMemberIdentity {
                path: path.clone(),
                shape,
                content_digest: [0; 32],
                file_fingerprint: file_fingerprint(path)?,
            });
        }
        let resource_estimate = manifest_resource_estimate(&members)?;
        for member in &mut members {
            member.content_digest =
                member_content_digest(input_type, &member.path, &subdataset, member.shape)
                    .with_context(|| format!("hashing CSLC member {}", member.path.display()))?;
            anyhow::ensure!(
                member.content_digest.iter().any(|byte| *byte != 0),
                "CSLC member digest is missing"
            );
            anyhow::ensure!(
                file_fingerprint(&member.path)? == member.file_fingerprint,
                "CSLC member changed during manifest capture"
            );
        }
        let digest = manifest_digest(input_type, &subdataset, &members);
        anyhow::ensure!(
            digest.iter().any(|byte| *byte != 0),
            "CSLC source manifest digest is missing"
        );
        Ok(Self {
            input_type,
            subdataset,
            members,
            digest,
            resource_estimate,
        })
    }

    /// Ordered source-manifest digest.
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Decoded-content identity I/O and peak resident preflight.
    #[must_use]
    pub const fn resource_estimate(&self) -> CslcManifestResourceEstimate {
        self.resource_estimate
    }

    /// Re-read every decoded member and verify the exact captured manifest.
    ///
    /// # Errors
    /// Returns an error if any path, subdataset, shape, or decoded value changed.
    pub fn verify_unchanged(&self) -> Result<()> {
        for member in &self.members {
            anyhow::ensure!(
                file_fingerprint(&member.path)? == member.file_fingerprint,
                "CSLC member metadata changed"
            );
            let shape = read_cslc_shape(&member.path, &self.subdataset)
                .with_context(|| format!("re-reading CSLC member {}", member.path.display()))?;
            anyhow::ensure!(shape == member.shape, "CSLC member shape changed");
            let digest = member_content_digest(
                self.input_type,
                &member.path,
                &self.subdataset,
                member.shape,
            )?;
            anyhow::ensure!(
                digest == member.content_digest,
                "CSLC member content changed"
            );
            anyhow::ensure!(
                file_fingerprint(&member.path)? == member.file_fingerprint,
                "CSLC member changed during manifest verification"
            );
        }
        Ok(())
    }

    /// Verify that this manifest is the exact ordered prefix of `paths`, rehash
    /// every prior decoded member, then capture only the appended members.
    ///
    /// Prior members are verified both before and after extension capture so a
    /// concurrent mutation cannot be admitted into the returned revision.
    ///
    /// # Errors
    /// Returns an error before reading any appended member when the path prefix
    /// differs or a prior member's metadata, shape, or decoded values changed.
    pub fn verify_prefix_and_extend(&self, paths: &[PathBuf]) -> Result<Self> {
        anyhow::ensure!(
            paths.len() > self.members.len()
                && paths[..self.members.len()]
                    .iter()
                    .zip(&self.members)
                    .all(|(path, member)| *path == member.path),
            "CSLC extension does not preserve the exact ordered prefix"
        );
        self.verify_unchanged()?;
        let mut members = self.members.clone();
        for path in &paths[self.members.len()..] {
            let shape = read_cslc_shape(path, &self.subdataset)
                .with_context(|| format!("reading CSLC extension shape from {}", path.display()))?;
            let fingerprint = file_fingerprint(path)?;
            let content_digest =
                member_content_digest(self.input_type, path, &self.subdataset, shape)
                    .with_context(|| format!("hashing CSLC extension member {}", path.display()))?;
            anyhow::ensure!(
                content_digest.iter().any(|byte| *byte != 0),
                "CSLC extension member digest is missing"
            );
            anyhow::ensure!(
                file_fingerprint(path)? == fingerprint,
                "CSLC extension member changed during manifest capture"
            );
            members.push(CslcMemberIdentity {
                path: path.clone(),
                shape,
                content_digest,
                file_fingerprint: fingerprint,
            });
        }
        self.verify_unchanged()?;
        let digest = manifest_digest(self.input_type, &self.subdataset, &members);
        let resource_estimate = manifest_resource_estimate(&members)?;
        let extended = Self {
            input_type: self.input_type,
            subdataset: self.subdataset.clone(),
            members,
            digest,
            resource_estimate,
        };
        extended.verify_unchanged()?;
        Ok(extended)
    }

    /// Exact ordered member receipt for one burst-local sequential generation.
    ///
    /// # Errors
    /// Returns an error for an empty burst, empty generation, repeated/out-of-order
    /// member indices, or an index outside this revision.
    pub fn generation_member_manifest_digest(
        &self,
        member_indices: &[usize],
        burst_id: &str,
        generation: u32,
    ) -> Result<[u8; 32]> {
        anyhow::ensure!(
            !burst_id.is_empty(),
            "CSLC generation burst identity is empty"
        );
        anyhow::ensure!(
            !member_indices.is_empty()
                && member_indices
                    .windows(2)
                    .all(|pair| pair[0].checked_add(1) == Some(pair[1])),
            "CSLC generation member indices must be nonempty and consecutive"
        );
        let mut digest = Sha256::new();
        digest.update(b"dolphinrust:cslc_generation_member_manifest:v1");
        digest.update(input_type_tag(self.input_type));
        digest.update((self.subdataset.len() as u64).to_le_bytes());
        digest.update(self.subdataset.as_bytes());
        digest.update((burst_id.len() as u64).to_le_bytes());
        digest.update(burst_id.as_bytes());
        digest.update(generation.to_le_bytes());
        digest.update((member_indices.len() as u64).to_le_bytes());
        for &index in member_indices {
            let member = self.members.get(index).with_context(|| {
                format!("CSLC generation member index {index} is outside the manifest")
            })?;
            let path = member.path.as_os_str().as_encoded_bytes();
            digest.update((index as u64).to_le_bytes());
            digest.update((path.len() as u64).to_le_bytes());
            digest.update(path);
            digest.update((member.shape.0 as u64).to_le_bytes());
            digest.update((member.shape.1 as u64).to_le_bytes());
            digest.update(member.content_digest);
        }
        Ok(digest.finalize().into())
    }

    /// Build a resolver for one burst and one captured native tile.
    ///
    /// `member_indices` are burst-local date order into this immutable manifest.
    ///
    /// # Errors
    /// Returns an error for an invalid member index, grid, or factor configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn resolver<'a>(
        &self,
        member_indices: &[usize],
        burst_id: impl Into<String>,
        processed_origin: (usize, usize),
        processed_shape: (usize, usize),
        tile_grid: CovarianceOperatorGrid,
        options: &EmpiricalSourceFactorOptions,
        source_model_version_digest: [u8; 32],
        validity_reader: Option<&'a dyn CslcCovarianceValidityReader>,
    ) -> Result<CslcCovarianceSourceResolver<'a>> {
        self.resolver_with_manifest_digest(
            member_indices,
            burst_id.into(),
            self.digest,
            processed_origin,
            processed_shape,
            tile_grid,
            options,
            source_model_version_digest,
            validity_reader,
        )
    }

    /// Build a resolver over the full burst-local `member_indices` whose replay
    /// namespace is the exact `generation_member_indices` receipt while retaining
    /// the complete revision identity. Keeping the full burst-local member list
    /// preserves global acquisition offsets for later generations.
    ///
    /// # Errors
    /// Returns an error for an invalid generation receipt, member, grid, or
    /// factor configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn resolver_for_generation<'a>(
        &self,
        member_indices: &[usize],
        generation_member_indices: &[usize],
        burst_id: impl Into<String>,
        generation: u32,
        processed_origin: (usize, usize),
        processed_shape: (usize, usize),
        tile_grid: CovarianceOperatorGrid,
        options: &EmpiricalSourceFactorOptions,
        source_model_version_digest: [u8; 32],
        validity_reader: Option<&'a dyn CslcCovarianceValidityReader>,
    ) -> Result<CslcCovarianceSourceResolver<'a>> {
        let burst_id = burst_id.into();
        let generation_digest = self.generation_member_manifest_digest(
            generation_member_indices,
            &burst_id,
            generation,
        )?;
        self.resolver_with_manifest_digest(
            member_indices,
            burst_id,
            generation_digest,
            processed_origin,
            processed_shape,
            tile_grid,
            options,
            source_model_version_digest,
            validity_reader,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn resolver_with_manifest_digest<'a>(
        &self,
        member_indices: &[usize],
        burst_id: String,
        source_manifest_digest: [u8; 32],
        processed_origin: (usize, usize),
        processed_shape: (usize, usize),
        tile_grid: CovarianceOperatorGrid,
        options: &EmpiricalSourceFactorOptions,
        source_model_version_digest: [u8; 32],
        validity_reader: Option<&'a dyn CslcCovarianceValidityReader>,
    ) -> Result<CslcCovarianceSourceResolver<'a>> {
        let members = member_indices
            .iter()
            .map(|&index| {
                self.members
                    .get(index)
                    .cloned()
                    .with_context(|| format!("CSLC member index {index} is outside the manifest"))
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(!members.is_empty(), "burst CSLC member list is empty");
        let processed_row_stop = processed_origin
            .0
            .checked_add(processed_shape.0)
            .context("processed source row extent overflows usize")?;
        let processed_col_stop = processed_origin
            .1
            .checked_add(processed_shape.1)
            .context("processed source column extent overflows usize")?;
        let native_shape = members[0].shape;
        anyhow::ensure!(
            members.iter().all(|member| member.shape == native_shape),
            "burst CSLC members do not share one canonical source grid"
        );
        for member in &members {
            anyhow::ensure!(
                processed_row_stop <= member.shape.0 && processed_col_stop <= member.shape.1,
                "processed source grid exceeds CSLC member {}",
                member.path.display()
            );
        }
        let factor_config = empirical_factor_config(options)?;
        let source_model_hash = *factor_config.config_digest();
        let identity = SequentialSourceProviderIdentity {
            source_manifest_digest,
            provider: CSLC_COVARIANCE_SOURCE_PROVIDER.to_owned(),
            provider_version: CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION.to_owned(),
            model: CSLC_COVARIANCE_SOURCE_MODEL.to_owned(),
            model_version: CSLC_COVARIANCE_SOURCE_MODEL_VERSION.to_owned(),
            source_model_version_digest,
            source_model_hash,
        };
        Ok(CslcCovarianceSourceResolver {
            input_type: self.input_type,
            subdataset: self.subdataset.clone(),
            members,
            burst_id,
            native_origin: (0, 0),
            native_shape,
            tile_grid,
            factor_config,
            identity,
            full_revision_manifest_digest: self.digest,
            validity_reader,
            tile_cache: None,
            metrics: CslcCovarianceResolverMetrics::default(),
            last_factor_receipt: None,
        })
    }
}

/// Build and validate the fixed empirical factor configuration.
///
/// # Errors
/// Returns an error for unsupported support, shrinkage, or rank-floor values.
pub fn empirical_factor_config(
    options: &EmpiricalSourceFactorOptions,
) -> Result<EmpiricalProperComplexConfig> {
    let mut model_identity = Sha256::new();
    model_identity.update(b"dolphinrust:cslc_empirical_proper_complex:model:v1");
    model_identity.update(CSLC_COVARIANCE_SOURCE_PROVIDER.as_bytes());
    model_identity.update(CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION.as_bytes());
    EmpiricalProperComplexConfig::new(
        options.half_window.y,
        options.half_window.x,
        options.shrinkage_alpha,
        options.relative_diagonal_floor,
        model_identity.finalize().into(),
    )
    .map_err(anyhow::Error::new)
}

/// CSLC-backed raw-source resolver used both during capture and artifact replay.
pub struct CslcCovarianceSourceResolver<'a> {
    input_type: InputType,
    subdataset: String,
    members: Vec<CslcMemberIdentity>,
    burst_id: String,
    native_origin: (usize, usize),
    native_shape: (usize, usize),
    tile_grid: CovarianceOperatorGrid,
    factor_config: EmpiricalProperComplexConfig,
    identity: SequentialSourceProviderIdentity,
    full_revision_manifest_digest: [u8; 32],
    validity_reader: Option<&'a dyn CslcCovarianceValidityReader>,
    tile_cache: Option<CslcSourceTileCache>,
    metrics: CslcCovarianceResolverMetrics,
    last_factor_receipt: Option<(SourceId, [u8; 32])>,
}

impl CslcCovarianceSourceResolver<'_> {
    /// Verified provider, manifest, and empirical model identity.
    #[must_use]
    pub const fn source_identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    /// Generation-member manifest digest used by replay IDs.
    #[must_use]
    pub const fn generation_manifest_digest(&self) -> [u8; 32] {
        self.identity.source_manifest_digest
    }

    /// Full ordered manifest digest for the complete artifact revision.
    #[must_use]
    pub const fn full_revision_manifest_digest(&self) -> [u8; 32] {
        self.full_revision_manifest_digest
    }

    /// Physical-read and cache high-water evidence.
    #[must_use]
    pub const fn metrics(&self) -> CslcCovarianceResolverMetrics {
        self.metrics
    }

    pub(crate) fn set_tile_grid(&mut self, tile_grid: CovarianceOperatorGrid) {
        if self.tile_grid != tile_grid {
            self.tile_grid = tile_grid;
            self.tile_cache = None;
        }
    }

    fn provider_error(status: ReplayStatus, message: &'static str) -> SequentialReplayError {
        SequentialReplayError::Provider(status, message)
    }

    fn source_pixel(&self, native_index: usize) -> Result<(usize, usize), SequentialReplayError> {
        let cols = usize::try_from(self.tile_grid.cols)
            .map_err(|_| SequentialReplayError::Invalid("source tile columns exceed usize"))?;
        let rows = usize::try_from(self.tile_grid.rows)
            .map_err(|_| SequentialReplayError::Invalid("source tile rows exceed usize"))?;
        let area = rows
            .checked_mul(cols)
            .ok_or(SequentialReplayError::Invalid(
                "source tile area overflows usize",
            ))?;
        if native_index >= area {
            return Err(SequentialReplayError::Invalid(
                "source native index is outside the captured tile",
            ));
        }
        let row = usize::try_from(self.tile_grid.row_start)
            .ok()
            .and_then(|start| start.checked_add(native_index / cols))
            .ok_or(SequentialReplayError::Invalid("source row exceeds usize"))?;
        let column = usize::try_from(self.tile_grid.col_start)
            .ok()
            .and_then(|start| start.checked_add(native_index % cols))
            .ok_or(SequentialReplayError::Invalid(
                "source column exceeds usize",
            ))?;
        Ok((row, column))
    }

    fn canonical_window(
        &self,
        source_pixel: (usize, usize),
    ) -> Result<BlockIndices, SequentialReplayError> {
        let support = self.factor_config.support_shape();
        if self.native_shape.0 < support.0 || self.native_shape.1 < support.1 {
            return Err(Self::provider_error(
                ReplayStatus::SourceModelUnavailable,
                "canonical source grid cannot supply the empirical factor window",
            ));
        }
        let local_row = source_pixel
            .0
            .checked_sub(self.native_origin.0)
            .ok_or_else(|| {
                Self::provider_error(
                    ReplayStatus::SourceIdentityMismatch,
                    "source row precedes the canonical source grid",
                )
            })?;
        let local_col = source_pixel
            .1
            .checked_sub(self.native_origin.1)
            .ok_or_else(|| {
                Self::provider_error(
                    ReplayStatus::SourceIdentityMismatch,
                    "source column precedes the canonical source grid",
                )
            })?;
        if local_row >= self.native_shape.0 || local_col >= self.native_shape.1 {
            return Err(Self::provider_error(
                ReplayStatus::SourceIdentityMismatch,
                "source pixel is outside the canonical source grid",
            ));
        }
        let row = local_row
            .saturating_sub((support.0 - 1) / 2)
            .min(self.native_shape.0 - support.0)
            + self.native_origin.0;
        let col = local_col
            .saturating_sub((support.1 - 1) / 2)
            .min(self.native_shape.1 - support.1)
            + self.native_origin.1;
        Ok(BlockIndices {
            row_start: row,
            row_stop: row + support.0,
            col_start: col,
            col_stop: col + support.1,
        })
    }

    fn verify_member_fingerprints(&self) -> Result<(), SequentialReplayError> {
        for member in &self.members {
            let current = file_fingerprint(&member.path).map_err(|_| {
                Self::provider_error(
                    ReplayStatus::SourceUnavailable,
                    "reading immutable CSLC member identity failed",
                )
            })?;
            if current != member.file_fingerprint {
                return Err(Self::provider_error(
                    ReplayStatus::SourceIdentityMismatch,
                    "CSLC member changed after manifest capture",
                ));
            }
        }
        Ok(())
    }

    fn tile_cache_block(&self) -> Result<BlockIndices, SequentialReplayError> {
        let rows = usize::try_from(self.tile_grid.rows)
            .map_err(|_| SequentialReplayError::Invalid("source tile rows exceed usize"))?;
        let cols = usize::try_from(self.tile_grid.cols)
            .map_err(|_| SequentialReplayError::Invalid("source tile columns exceed usize"))?;
        if rows == 0 || cols == 0 {
            return Err(SequentialReplayError::Invalid("source tile is empty"));
        }
        let first = self.source_pixel(0)?;
        let last = self.source_pixel(rows * cols - 1)?;
        let first_window = self.canonical_window(first)?;
        let last_window = self.canonical_window(last)?;
        Ok(BlockIndices {
            row_start: first_window.row_start.min(last_window.row_start),
            row_stop: first_window.row_stop.max(last_window.row_stop),
            col_start: first_window.col_start.min(last_window.col_start),
            col_stop: first_window.col_stop.max(last_window.col_stop),
        })
    }

    fn load_tile_cache(&mut self) -> Result<(), SequentialReplayError> {
        self.verify_member_fingerprints()?;
        let window = self.tile_cache_block()?;
        let mut values = Array3::zeros((self.members.len(), window.height(), window.width()));
        for (component, member) in self.members.iter().enumerate() {
            let read = match self.input_type {
                InputType::OperaCslc => read_cslc_window(&member.path, &self.subdataset, window),
                InputType::NisarGslc => read_nisar_window(&member.path, &self.subdataset, window),
            }
            .map_err(|_| {
                Self::provider_error(
                    ReplayStatus::SourceUnavailable,
                    "reading immutable CSLC source support failed",
                )
            })?;
            for (target, source) in values
                .index_axis_mut(Axis(0), component)
                .iter_mut()
                .zip(read.iter())
            {
                *target = Cf64::new(f64::from(source.re), f64::from(source.im));
            }
            self.metrics.member_window_reads = self.metrics.member_window_reads.saturating_add(1);
        }
        let validity = match self.validity_reader {
            Some(reader) => reader.read_validity(window)?,
            None => Array2::from_elem((window.height(), window.width()), true),
        };
        if validity.dim() != (window.height(), window.width()) {
            return Err(Self::provider_error(
                ReplayStatus::SourceIdentityMismatch,
                "source validity support differs from the cached factor tile",
            ));
        }
        self.verify_member_fingerprints()?;
        self.metrics.tile_cache_loads = self.metrics.tile_cache_loads.saturating_add(1);
        let cached_bytes = values
            .len()
            .saturating_mul(std::mem::size_of::<Cf64>())
            .saturating_add(validity.len());
        self.metrics.peak_cached_bytes = self
            .metrics
            .peak_cached_bytes
            .max(u64::try_from(cached_bytes).unwrap_or(u64::MAX));
        self.tile_cache = Some(CslcSourceTileCache {
            block: window,
            values,
            validity,
        });
        Ok(())
    }
}

impl SequentialPrimitiveSourceResolver for CslcCovarianceSourceResolver<'_> {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        let support = self.factor_config.support_shape();
        let components = self.members.len();
        let samples = components
            .saturating_mul(support.0)
            .saturating_mul(support.1);
        let matrices = components
            .saturating_mul(components)
            .saturating_mul(2)
            .saturating_mul(16);
        let factor_bytes = u64::try_from(
            samples
                .saturating_mul(24)
                .saturating_add(support.0.saturating_mul(support.1))
                .saturating_add(matrices)
                .saturating_add(components.saturating_mul(16)),
        )
        .unwrap_or(u64::MAX);
        let cache_bytes = self.tile_cache_block().map_or(u64::MAX, |block| {
            u64::try_from(
                components
                    .saturating_mul(block.height())
                    .saturating_mul(block.width())
                    .saturating_mul(std::mem::size_of::<Cf64>())
                    .saturating_add(block.height().saturating_mul(block.width()))
                    .saturating_add(
                        block
                            .height()
                            .saturating_mul(block.width())
                            .saturating_mul(std::mem::size_of::<dolphin_core::Cf32>()),
                    ),
            )
            .unwrap_or(u64::MAX)
        });
        let identity_stripe_bytes = self.members.iter().fold(0_u64, |maximum, member| {
            let bytes = member
                .shape
                .0
                .min(MEMBER_DIGEST_STRIPE_ROWS)
                .saturating_mul(member.shape.1)
                .saturating_mul(std::mem::size_of::<dolphin_core::Cf32>());
            maximum.max(u64::try_from(bytes).unwrap_or(u64::MAX))
        });
        factor_bytes
            .saturating_add(cache_bytes)
            .saturating_add(identity_stripe_bytes)
    }

    fn factor_receipt_digest(
        &self,
        source: &ResolvedPrimitiveSource,
    ) -> Result<[u8; 32], SequentialReplayError> {
        self.last_factor_receipt
            .filter(|(id, _)| *id == source.id)
            .map(|(_, digest)| digest)
            .ok_or_else(|| {
                Self::provider_error(
                    ReplayStatus::SourceIdentityMismatch,
                    "empirical factor receipt does not match the last resolved source",
                )
            })
    }

    #[allow(clippy::too_many_lines)]
    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        let start = usize::try_from(block.real_date_start.get())
            .map_err(|_| SequentialReplayError::Invalid("source date exceeds usize"))?;
        let stop =
            start
                .checked_add(block.num_real_dates)
                .ok_or(SequentialReplayError::Invalid(
                    "source date range overflows usize",
                ))?;
        if stop > self.members.len() {
            return Err(Self::provider_error(
                ReplayStatus::SourceIdentityMismatch,
                "replay block dates differ from the ordered CSLC source members",
            ));
        }
        let source_pixel = self.source_pixel(native_index)?;
        let window = self.canonical_window(source_pixel)?;
        if self.tile_cache.is_none() {
            self.load_tile_cache()?;
        }
        let cache = self.tile_cache.as_ref().ok_or_else(|| {
            Self::provider_error(
                ReplayStatus::SourceUnavailable,
                "canonical source tile cache is unavailable",
            )
        })?;
        let members = &self.members[start..stop];
        let local_window = BlockIndices {
            row_start: window.row_start - cache.block.row_start,
            row_stop: window.row_stop - cache.block.row_start,
            col_start: window.col_start - cache.block.col_start,
            col_stop: window.col_stop - cache.block.col_start,
        };
        let values = cache
            .values
            .slice(s![start..stop, local_window.rows(), local_window.cols()]);
        let valid = cache
            .validity
            .slice(s![local_window.rows(), local_window.cols()]);
        let local_source = (
            source_pixel.0 - cache.block.row_start,
            source_pixel.1 - cache.block.col_start,
        );
        let samples = Array1::from_iter(
            (start..stop)
                .map(|component| cache.values[(component, local_source.0, local_source.1)]),
        );
        let content_digest = primitive_source_content_digest(samples.iter().copied());
        let secondary = (u64::from(block.real_date_start.get()) << 32)
            | u64::try_from(block.num_real_dates)
                .map_err(|_| SequentialReplayError::Invalid("source date count exceeds u64"))?;
        let locator = covariance_identified_id(
            b"source",
            &self.burst_id,
            self.identity.source_manifest_digest,
            self.identity.source_model_version_digest,
            u64::from(block.generation),
            secondary,
            self.tile_grid,
            native_index,
        )
        .map_err(|_| SequentialReplayError::Invalid("deriving source locator failed"))?;
        let id = SourceId::new(
            covariance_content_bound_source_id(locator, &content_digest)
                .map_err(|_| SequentialReplayError::Invalid("binding source content failed"))?,
        );
        let component_ids = (start..stop)
            .map(|index| {
                u64::try_from(index)
                    .map_err(|_| SequentialReplayError::Invalid("component index exceeds u64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut data_identity = Sha256::new();
        data_identity.update(b"dolphinrust:cslc_source_block_identity:v1");
        data_identity.update(self.identity.source_manifest_digest);
        data_identity.update((members.len() as u64).to_le_bytes());
        for (index, member) in (start..stop).zip(members) {
            data_identity.update((index as u64).to_le_bytes());
            data_identity.update(member.content_digest);
        }
        let estimate = estimate_empirical_proper_complex_factor(
            id,
            &component_ids,
            values,
            valid,
            (window.row_start, window.col_start),
            self.native_origin,
            self.native_shape,
            source_pixel,
            data_identity.finalize().into(),
            &self.factor_config,
        )
        .map_err(|_| {
            Self::provider_error(
                ReplayStatus::SourceModelUnavailable,
                "empirical proper-complex source factor could not be reproduced",
            )
        })?;
        let (factor, receipt) = estimate.into_parts();
        if receipt.source() != id
            || receipt.config_digest() != self.factor_config.config_digest()
            || receipt.digest().iter().all(|byte| *byte == 0)
        {
            return Err(Self::provider_error(
                ReplayStatus::SourceIdentityMismatch,
                "empirical source-factor receipt identity is inconsistent",
            ));
        }
        self.last_factor_receipt = Some((id, *receipt.digest()));
        self.metrics.source_resolutions = self.metrics.source_resolutions.saturating_add(1);
        Ok(ResolvedPrimitiveSource {
            id,
            samples,
            factor,
            content_digest,
        })
    }
}

fn input_type_tag(input_type: InputType) -> &'static [u8] {
    match input_type {
        InputType::OperaCslc => b"opera_cslc",
        InputType::NisarGslc => b"nisar_gslc",
    }
}

fn file_fingerprint(path: &Path) -> Result<CslcFileFingerprint> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("reading CSLC member metadata from {}", path.display()))?;
    Ok(CslcFileFingerprint {
        length: metadata.len(),
        modified: metadata.modified().ok(),
    })
}

fn manifest_resource_estimate(
    members: &[CslcMemberIdentity],
) -> Result<CslcManifestResourceEstimate> {
    let mut decoded_content_bytes = 0_u64;
    let mut identity_window_reads = 0_u64;
    let mut maximum_resident_bytes = 0_u64;
    for member in members {
        let pixels = u64::try_from(member.shape.0)?
            .checked_mul(u64::try_from(member.shape.1)?)
            .context("CSLC identity pixel count overflows u64")?;
        decoded_content_bytes = decoded_content_bytes
            .checked_add(
                pixels
                    .checked_mul(std::mem::size_of::<dolphin_core::Cf32>() as u64)
                    .context("CSLC identity byte count overflows u64")?,
            )
            .context("CSLC manifest identity byte count overflows u64")?;
        identity_window_reads = identity_window_reads
            .checked_add(u64::try_from(
                member.shape.0.div_ceil(MEMBER_DIGEST_STRIPE_ROWS),
            )?)
            .context("CSLC manifest identity read count overflows u64")?;
        let stripe_pixels = u64::try_from(member.shape.0.min(MEMBER_DIGEST_STRIPE_ROWS))?
            .checked_mul(u64::try_from(member.shape.1)?)
            .context("CSLC identity stripe pixels overflow u64")?;
        maximum_resident_bytes = maximum_resident_bytes.max(
            stripe_pixels
                .checked_mul(std::mem::size_of::<dolphin_core::Cf32>() as u64)
                .context("CSLC identity stripe bytes overflow u64")?,
        );
    }
    Ok(CslcManifestResourceEstimate {
        member_count: u64::try_from(members.len())?,
        decoded_content_bytes: decoded_content_bytes
            .checked_mul(MANIFEST_IDENTITY_PASSES)
            .context("CSLC manifest identity pass bytes overflow u64")?,
        identity_window_reads: identity_window_reads
            .checked_mul(MANIFEST_IDENTITY_PASSES)
            .context("CSLC manifest identity pass reads overflow u64")?,
        maximum_resident_bytes,
    })
}

fn member_content_digest(
    input_type: InputType,
    path: &Path,
    subdataset: &str,
    shape: (usize, usize),
) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:cslc_member_decoded_cf32:v1");
    digest.update((shape.0 as u64).to_le_bytes());
    digest.update((shape.1 as u64).to_le_bytes());
    for row_start in (0..shape.0).step_by(MEMBER_DIGEST_STRIPE_ROWS) {
        let block = BlockIndices {
            row_start,
            row_stop: (row_start + MEMBER_DIGEST_STRIPE_ROWS).min(shape.0),
            col_start: 0,
            col_stop: shape.1,
        };
        let values = match input_type {
            InputType::OperaCslc => read_cslc_window(path, subdataset, block),
            InputType::NisarGslc => read_nisar_window(path, subdataset, block),
        }?;
        for value in values {
            digest.update(value.re.to_bits().to_le_bytes());
            digest.update(value.im.to_bits().to_le_bytes());
        }
    }
    Ok(digest.finalize().into())
}

fn manifest_digest(
    input_type: InputType,
    subdataset: &str,
    members: &[CslcMemberIdentity],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:cslc_member_manifest:v1");
    digest.update(input_type_tag(input_type));
    digest.update((subdataset.len() as u64).to_le_bytes());
    digest.update(subdataset.as_bytes());
    digest.update((members.len() as u64).to_le_bytes());
    for (index, member) in members.iter().enumerate() {
        let path = member.path.as_os_str().as_encoded_bytes();
        digest.update((index as u64).to_le_bytes());
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path);
        digest.update((member.shape.0 as u64).to_le_bytes());
        digest.update((member.shape.1 as u64).to_le_bytes());
        digest.update(member.content_digest);
    }
    digest.finalize().into()
}
