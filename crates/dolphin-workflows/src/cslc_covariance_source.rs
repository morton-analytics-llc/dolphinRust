//! Immutable CSLC-member identity and empirical primitive-source resolution.

use std::path::{Path, PathBuf};

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
use ndarray::{Array1, Array2, Array3, Axis};
use sha2::{Digest, Sha256};

use crate::sequential_covariance::{
    primitive_source_content_digest, ReplayStatus, ResolvedPrimitiveSource,
    SequentialPrimitiveSourceResolver, SequentialReplayBlock, SequentialReplayError,
    SequentialSourceProviderIdentity,
};

/// Production raw-source provider persisted with covariance replay artifacts.
pub const CSLC_COVARIANCE_SOURCE_PROVIDER: &str = "dolphin_workflows_cslc_member_bytes";
/// Version of the ordered decoded-member identity and canonical-window reader.
pub const CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION: &str = "1";
/// Production proper-complex source-factor method.
pub const CSLC_COVARIANCE_SOURCE_MODEL: &str = EMPIRICAL_PROPER_COMPLEX_METHOD;
/// Production proper-complex source-factor method version.
pub const CSLC_COVARIANCE_SOURCE_MODEL_VERSION: &str = "1";

const MEMBER_DIGEST_STRIPE_ROWS: usize = 256;

#[derive(Debug, Clone)]
struct CslcMemberIdentity {
    path: PathBuf,
    shape: (usize, usize),
    content_digest: [u8; 32],
}

/// Ordered immutable identity of every configured CSLC member.
#[derive(Debug, Clone)]
pub struct CslcCovarianceManifest {
    input_type: InputType,
    subdataset: String,
    members: Vec<CslcMemberIdentity>,
    digest: [u8; 32],
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
            let content_digest = member_content_digest(input_type, path, &subdataset, shape)
                .with_context(|| format!("hashing CSLC member {}", path.display()))?;
            anyhow::ensure!(
                content_digest.iter().any(|byte| *byte != 0),
                "CSLC member digest is missing"
            );
            members.push(CslcMemberIdentity {
                path: path.clone(),
                shape,
                content_digest,
            });
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
        })
    }

    /// Ordered source-manifest digest.
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
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
        native_origin: (usize, usize),
        native_shape: (usize, usize),
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
        let native_row_stop = native_origin
            .0
            .checked_add(native_shape.0)
            .context("canonical source row extent overflows usize")?;
        let native_col_stop = native_origin
            .1
            .checked_add(native_shape.1)
            .context("canonical source column extent overflows usize")?;
        for member in &members {
            anyhow::ensure!(
                native_row_stop <= member.shape.0 && native_col_stop <= member.shape.1,
                "canonical source grid exceeds CSLC member {}",
                member.path.display()
            );
        }
        let factor_config = empirical_factor_config(options)?;
        let source_model_hash = *factor_config.config_digest();
        let identity = SequentialSourceProviderIdentity {
            source_manifest_digest: self.digest,
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
            burst_id: burst_id.into(),
            native_origin,
            native_shape,
            tile_grid,
            factor_config,
            identity,
            validity_reader,
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
    validity_reader: Option<&'a dyn CslcCovarianceValidityReader>,
}

impl CslcCovarianceSourceResolver<'_> {
    /// Verified provider, manifest, and empirical model identity.
    #[must_use]
    pub const fn source_identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    pub(crate) fn set_tile_grid(&mut self, tile_grid: CovarianceOperatorGrid) {
        self.tile_grid = tile_grid;
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

    fn read_window(
        &self,
        members: &[CslcMemberIdentity],
        window: BlockIndices,
    ) -> Result<Array3<Cf64>, SequentialReplayError> {
        let mut values = Array3::zeros((members.len(), window.height(), window.width()));
        for (component, member) in members.iter().enumerate() {
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
        }
        Ok(values)
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
        u64::try_from(
            samples
                .saturating_mul(24)
                .saturating_add(support.0.saturating_mul(support.1))
                .saturating_add(matrices)
                .saturating_add(components.saturating_mul(16)),
        )
        .unwrap_or(u64::MAX)
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
        let members = self.members.get(start..stop).ok_or_else(|| {
            Self::provider_error(
                ReplayStatus::SourceIdentityMismatch,
                "replay block dates differ from the ordered CSLC source members",
            )
        })?;
        let source_pixel = self.source_pixel(native_index)?;
        let window = self.canonical_window(source_pixel)?;
        let values = self.read_window(members, window)?;
        let local_source = (
            source_pixel.0 - window.row_start,
            source_pixel.1 - window.col_start,
        );
        let samples = Array1::from_iter(
            (0..members.len()).map(|component| values[(component, local_source.0, local_source.1)]),
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
        let valid = match self.validity_reader {
            Some(reader) => reader.read_validity(window)?,
            None => Array2::from_elem((window.height(), window.width()), true),
        };
        if valid.dim() != (window.height(), window.width()) {
            return Err(Self::provider_error(
                ReplayStatus::SourceIdentityMismatch,
                "source validity support differs from the canonical factor window",
            ));
        }
        let estimate = estimate_empirical_proper_complex_factor(
            id,
            &component_ids,
            values.view(),
            valid.view(),
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
