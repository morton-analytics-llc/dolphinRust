//! Production identities and fixed-L2 state for reference-specific covariance output.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result};
use dolphin_core::config::{CorrectionOptions, DisplacementWorkflow, ShpMethod, UnwrapMethod};
use dolphin_io::{
    spatial_reference_runtime_resource_receipt_digest, CovarianceOperatorGrid,
    SpatialReferenceCalibrationScope, SpatialReferenceCovarianceBlock,
    SpatialReferenceCovarianceMetadata, SpatialReferenceCovarianceStatus,
    SpatialReferenceCovarianceWriter, SpatialReferenceRuntimeResourceReceipt,
    SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE, SPATIAL_REFERENCE_COVARIANCE_METHOD,
    SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION, SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
};
use dolphin_timeseries::inversion::{fixed_l2_pixel_map, PixelL2MapError, PixelL2ObservationMap};
use dolphin_timeseries::spatial_covariance::{
    fixed_l2_difference_workspace_composition, propagate_fixed_l2_difference_covariance,
    FixedL2DifferenceCovariance, FixedL2WorkspaceComposition, SpatialL2Branch, SpatialL2Error,
    SpatialL2Status,
};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use sha2::{Digest, Sha256};

use crate::corrections::CorrectionLayers;
use crate::covariance_artifact::{
    read_covariance_artifact_manifest_with_byte_cap, CovarianceArtifactManifest,
};
use crate::cslc_covariance_source::{CslcCovarianceManifest, CslcCovarianceSourceResolver};
use crate::displacement::PreparedBurstMask;
use crate::sequential::SequentialConfig;
use crate::sequential_covariance::{
    estimate_global_reference_difference_covariance_from_provider_bundle,
    replay_global_reference_difference_covariance_from_provider_bundle,
    sequential_replay_config_digest, sequential_replay_kernel_digest,
    CovarianceArtifactReplayProvider, EffectiveLooksReplay, GlobalDateId,
    GlobalReferenceCovarianceQuery, ReplayBackend, ReplayExecutionScope, ReplayStatus,
    SequentialReplayBuildIdentity, SequentialReplayError, SequentialReplayTopology,
    SequentialSourceReplayProvider, SequentialTileReplayProvider,
};
use crate::spatial_covariance_artifact::{
    finalize_spatial_reference_covariance_artifact, SpatialReferenceCovarianceArtifactTransaction,
    SPATIAL_REFERENCE_COVARIANCE_FILENAME, SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME,
};

const PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP: u64 = 1_073_741_824;
const EFFECTIVE_LOOKS_MODEL: &str = "source_factor_declared_v1";
const EFFECTIVE_LOOKS_DISTANCE_SCALE_PIXELS: f64 = 1.5;

pub(crate) const NO_BURST_OWNER: u32 = u32::MAX;

pub(crate) struct CapturedReplayTile {
    pub(crate) request: crate::sequential_covariance::SequentialCovarianceCaptureRequest,
    pub(crate) member_indices: Vec<usize>,
    pub(crate) processed_origin: (usize, usize),
    pub(crate) processed_shape: (usize, usize),
    pub(crate) native_validity: Array2<bool>,
    pub(crate) num_real_dates: usize,
}

pub(crate) struct ProductionCovarianceReplayContext {
    pub(crate) source_manifest: CslcCovarianceManifest,
    pub(crate) operator_manifest: CovarianceArtifactManifest,
    pub(crate) tiles: Vec<CapturedReplayTile>,
    pub(crate) masks: BTreeMap<String, Option<PreparedBurstMask>>,
    pub(crate) operator_block_byte_cap: u64,
}

#[derive(Clone)]
pub(crate) struct TargetFactor {
    pub(crate) status: SpatialReferenceCovarianceStatus,
    pub(crate) source_burst_index: u32,
    pub(crate) date_factor: Option<Array2<f64>>,
    pub(crate) source_factor_receipt: [u8; 32],
    pub(crate) effective_looks_fraction: f64,
    pub(crate) support_union_count: u64,
    pub(crate) effective_looks_receipt: [u8; 32],
    pub(crate) resource_high_water_bytes: u64,
    pub(crate) condition_number: f64,
}

pub(crate) fn fixed_l2_status(status: SpatialL2Status) -> SpatialReferenceCovarianceStatus {
    match status {
        SpatialL2Status::Valid => SpatialReferenceCovarianceStatus::Valid,
        SpatialL2Status::UnsupportedL1 | SpatialL2Status::UnsupportedChangedBranch => {
            SpatialReferenceCovarianceStatus::UnsupportedL1
        }
        SpatialL2Status::RankDeficient => SpatialReferenceCovarianceStatus::L2RankDeficient,
        SpatialL2Status::IllConditioned => SpatialReferenceCovarianceStatus::IllConditioned,
        SpatialL2Status::InvalidInput | SpatialL2Status::NonFinite => {
            SpatialReferenceCovarianceStatus::TemporalFactorInvalid
        }
    }
}

pub(crate) fn replay_status(status: ReplayStatus) -> SpatialReferenceCovarianceStatus {
    match status {
        ReplayStatus::Valid => SpatialReferenceCovarianceStatus::Valid,
        ReplayStatus::InvalidReference => SpatialReferenceCovarianceStatus::InvalidReference,
        ReplayStatus::SourceUnavailable | ReplayStatus::DependencyConeExceedsBudget => {
            SpatialReferenceCovarianceStatus::ReplayUnavailable
        }
        ReplayStatus::SourceIdentityMismatch | ReplayStatus::ReplayStateMismatch => {
            SpatialReferenceCovarianceStatus::ReplayMismatch
        }
        ReplayStatus::NondifferentiableNode => {
            SpatialReferenceCovarianceStatus::NondifferentiableEstimator
        }
        ReplayStatus::MaskedNode => SpatialReferenceCovarianceStatus::EmptySupport,
        ReplayStatus::UnsupportedPhaseBiasCorrection => {
            SpatialReferenceCovarianceStatus::UnsupportedPhaseBias
        }
        ReplayStatus::SourceModelUnavailable => SpatialReferenceCovarianceStatus::UnsupportedModel,
        ReplayStatus::NonFiniteReplayState => SpatialReferenceCovarianceStatus::InfluenceInvalid,
        ReplayStatus::InvalidTopology
        | ReplayStatus::InvalidReplayGraph
        | ReplayStatus::SingularLocalInformation
        | ReplayStatus::InvalidCompression => SpatialReferenceCovarianceStatus::InfluenceInvalid,
        ReplayStatus::Disabled
        | ReplayStatus::UnsupportedReferencePlan
        | ReplayStatus::UnsupportedOutputReference
        | ReplayStatus::UnsupportedBackend
        | ReplayStatus::UnsupportedShpMethod
        | ReplayStatus::UnsupportedEstimatorFallback
        | ReplayStatus::UnsupportedSourceIdentity
        | ReplayStatus::UnsupportedSeamCovariance => {
            SpatialReferenceCovarianceStatus::ReplayUnsupported
        }
    }
}

pub(crate) fn production_mask_digest(
    validity: ArrayView2<'_, bool>,
    ownership: ArrayView3<'_, u32>,
) -> Result<[u8; 32]> {
    anyhow::ensure!(
        ownership.dim().1 == validity.dim().0 && ownership.dim().2 == validity.dim().1,
        "production mask and ownership grids differ"
    );
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:production-spatial-covariance-mask:v1");
    hash_shape(&mut digest, validity.shape());
    for &valid in validity {
        digest.update([u8::from(valid)]);
    }
    hash_shape(&mut digest, ownership.shape());
    for &owner in ownership {
        digest.update(owner.to_le_bytes());
    }
    Ok(digest.finalize().into())
}

pub(crate) fn fixed_l2_frame_digest(inputs: Option<&FixedL2WorkflowInputs>) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:production-fixed-l2-frame-map:v1");
    let Some(inputs) = inputs else {
        digest.update(b"unsupported_l1");
        return Ok(digest.finalize().into());
    };
    let shape = inputs.dphi_rad.dim();
    hash_shape(&mut digest, &[shape.1, shape.2]);
    for row in 0..shape.1 {
        for column in 0..shape.2 {
            digest.update((row as u64).to_le_bytes());
            digest.update((column as u64).to_le_bytes());
            match inputs.pixel_map_digest((row, column)) {
                Ok(pixel) => {
                    digest.update(b"valid");
                    digest.update(pixel);
                }
                Err(error) => {
                    digest.update(b"invalid");
                    let message = error.to_string();
                    digest.update((message.len() as u64).to_le_bytes());
                    digest.update(message.as_bytes());
                }
            }
        }
    }
    Ok(digest.finalize().into())
}

pub(crate) fn final_reference_signature_digest(
    full_grid: CovarianceOperatorGrid,
    reference: (u64, u64),
    owner: u32,
    ordered_dates: &[u32],
    geotransform: [f64; 6],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:production-final-reference:v1");
    hash_grid(&mut digest, full_grid);
    digest.update(reference.0.to_le_bytes());
    digest.update(reference.1.to_le_bytes());
    digest.update(owner.to_le_bytes());
    for &date in ordered_dates {
        digest.update(date.to_le_bytes());
    }
    for value in geotransform {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

#[allow(clippy::too_many_arguments)]
fn production_target_receipt(
    source_factor_receipt: [u8; 32],
    support_receipt: [u8; 32],
    reference_signature: [u8; 32],
    effective_looks_model: &str,
    effective_looks_distance_scale_pixels: f64,
    support_union_count: usize,
    effective_looks_fraction: f64,
    effective_looks_receipt: [u8; 32],
    resource_high_water_bytes: u64,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:production-target-source-support-reference-resource:v1");
    digest.update(source_factor_receipt);
    digest.update(support_receipt);
    digest.update(reference_signature);
    digest.update((effective_looks_model.len() as u64).to_le_bytes());
    digest.update(effective_looks_model.as_bytes());
    digest.update(
        effective_looks_distance_scale_pixels
            .to_bits()
            .to_le_bytes(),
    );
    digest.update((support_union_count as u64).to_le_bytes());
    digest.update(effective_looks_fraction.to_bits().to_le_bytes());
    digest.update(effective_looks_receipt);
    digest.update(resource_high_water_bytes.to_le_bytes());
    digest.finalize().into()
}

#[cfg(test)]
pub(crate) fn build_factor_block(
    block_id: u64,
    target_grid: CovarianceOperatorGrid,
    dates: usize,
    phase_to_displacement: f64,
    outcomes: &[TargetFactor],
) -> Result<SpatialReferenceCovarianceBlock> {
    let mut builder = FactorBlockBuilder::new(block_id, target_grid, dates, phase_to_displacement)?;
    for outcome in outcomes.iter().cloned() {
        builder.push(outcome)?;
    }
    builder.finish()
}

struct FactorBlockBuilder {
    block_id: u64,
    target_grid: CovarianceOperatorGrid,
    dates: usize,
    phase_to_displacement: f64,
    targets: usize,
    difference_factor: Vec<f64>,
    rank_by_target: Vec<u32>,
    status: Vec<SpatialReferenceCovarianceStatus>,
    source_burst_index_by_target: Vec<u32>,
    effective_looks_fraction: Vec<f64>,
    support_union_count: Vec<u64>,
    effective_looks_receipt: Vec<u8>,
    resource_high_water_bytes: Vec<u64>,
    condition_number: Vec<f64>,
    receipt: Sha256,
}

impl FactorBlockBuilder {
    fn new(
        block_id: u64,
        target_grid: CovarianceOperatorGrid,
        dates: usize,
        phase_to_displacement: f64,
    ) -> Result<Self> {
        let targets = usize::try_from(target_grid.rows)
            .ok()
            .and_then(|rows| {
                usize::try_from(target_grid.cols)
                    .ok()
                    .and_then(|cols| rows.checked_mul(cols))
            })
            .context("spatial covariance target grid area exceeds usize")?;
        anyhow::ensure!(
            dates > 0 && phase_to_displacement.is_finite(),
            "spatial covariance block inputs disagree"
        );
        let factor_len = targets
            .checked_mul(dates)
            .and_then(|value| value.checked_mul(dates))
            .context("spatial covariance factor block dimensions overflow")?;
        let receipt_capacity = targets
            .checked_mul(32)
            .context("spatial covariance receipt block dimensions overflow")?;
        let mut receipt = Sha256::new();
        receipt.update(b"dolphinrust:production-source-factor-block:v1");
        receipt.update(block_id.to_le_bytes());
        hash_grid(&mut receipt, target_grid);
        Ok(Self {
            block_id,
            target_grid,
            dates,
            phase_to_displacement,
            targets,
            difference_factor: vec![0.0; factor_len],
            rank_by_target: Vec::with_capacity(targets),
            status: Vec::with_capacity(targets),
            source_burst_index_by_target: Vec::with_capacity(targets),
            effective_looks_fraction: Vec::with_capacity(targets),
            support_union_count: Vec::with_capacity(targets),
            effective_looks_receipt: Vec::with_capacity(receipt_capacity),
            resource_high_water_bytes: Vec::with_capacity(targets),
            condition_number: Vec::with_capacity(targets),
            receipt,
        })
    }

    fn push(&mut self, outcome: TargetFactor) -> Result<()> {
        let target = self.status.len();
        anyhow::ensure!(
            target < self.targets,
            "spatial covariance block is already full"
        );
        let rank = match (outcome.date_factor.as_ref(), outcome.status) {
            (Some(factor), SpatialReferenceCovarianceStatus::Valid) => {
                anyhow::ensure!(
                    factor.nrows() == self.dates
                        && factor.ncols() <= self.dates
                        && factor.iter().all(|value| value.is_finite())
                        && (0..factor.ncols())
                            .all(|column| factor.column(column).iter().any(|value| *value != 0.0)),
                    "valid spatial covariance target factor is malformed"
                );
                for date in 0..self.dates {
                    let start = (target * self.dates + date) * self.dates;
                    for component in 0..factor.ncols() {
                        self.difference_factor[start + component] =
                            self.phase_to_displacement * factor[(date, component)];
                    }
                }
                u32::try_from(factor.ncols()).context("target factor rank exceeds u32")?
            }
            (None, SpatialReferenceCovarianceStatus::Valid) => {
                anyhow::bail!("valid spatial covariance target is missing its factor")
            }
            (Some(_), _) => {
                anyhow::bail!("non-valid spatial covariance target carries a factor")
            }
            (None, _) => 0,
        };
        anyhow::ensure!(
            outcome.status != SpatialReferenceCovarianceStatus::Valid
                || outcome.source_burst_index != SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
            "valid spatial covariance target is missing burst ownership"
        );
        self.rank_by_target.push(rank);
        self.status.push(outcome.status);
        self.source_burst_index_by_target
            .push(outcome.source_burst_index);
        self.effective_looks_fraction
            .push(outcome.effective_looks_fraction);
        self.support_union_count.push(outcome.support_union_count);
        self.effective_looks_receipt
            .extend_from_slice(&outcome.effective_looks_receipt);
        self.resource_high_water_bytes
            .push(outcome.resource_high_water_bytes);
        self.condition_number.push(outcome.condition_number);
        self.receipt.update((target as u64).to_le_bytes());
        self.receipt.update((outcome.status as u16).to_le_bytes());
        self.receipt
            .update(outcome.source_burst_index.to_le_bytes());
        self.receipt.update(rank.to_le_bytes());
        self.receipt.update(outcome.source_factor_receipt);
        Ok(())
    }

    fn finish(self) -> Result<SpatialReferenceCovarianceBlock> {
        anyhow::ensure!(
            self.status.len() == self.targets,
            "spatial covariance block is incomplete"
        );
        Ok(SpatialReferenceCovarianceBlock {
            block_id: self.block_id,
            target_grid: self.target_grid,
            maximum_rank: u32::try_from(self.dates).context("maximum factor rank exceeds u32")?,
            rank_by_target: self.rank_by_target,
            status: self.status,
            source_burst_index_by_target: self.source_burst_index_by_target,
            difference_factor: self.difference_factor,
            approximation_error_bound: vec![
                SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE;
                self.targets
            ],
            effective_looks_fraction: Some(self.effective_looks_fraction),
            support_union_count: Some(self.support_union_count),
            effective_looks_receipt: Some(self.effective_looks_receipt),
            resource_high_water_bytes: Some(self.resource_high_water_bytes),
            condition_number: Some(self.condition_number),
            source_factor_digest: format!("sha256:{:x}", self.receipt.finalize()),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BurstOutputMapping {
    pub(crate) owner: u32,
    pub(crate) frame_origin: (usize, usize),
    pub(crate) output_origin: (u64, u64),
    pub(crate) shape: (usize, usize),
}

/// Production state retained until bounded reference relocation is complete.
pub(crate) struct ProductionCovarianceState {
    pub(crate) replay_context: Option<ProductionCovarianceReplayContext>,
    pub(crate) fixed_l2_inputs: Option<FixedL2WorkflowInputs>,
    pub(crate) ownership: Array3<u32>,
    pub(crate) seam_rotations: Vec<(u32, Vec<dolphin_core::Cf64>)>,
    pub(crate) source_burst_ids: Vec<String>,
    pub(crate) burst_output_mappings: Vec<BurstOutputMapping>,
    pub(crate) analysis_origin: (usize, usize),
    pub(crate) correction_order_digest: [u8; 32],
    pub(crate) unwrap_branch_digest: [u8; 32],
}

impl ProductionCovarianceState {
    pub(crate) fn trim(&mut self, target: dolphin_core::BlockIndices) {
        if let Some(inputs) = self.fixed_l2_inputs.as_mut() {
            inputs.trim(target);
        }
        self.ownership = self
            .ownership
            .slice(ndarray::s![
                ..,
                target.row_start..target.row_stop,
                target.col_start..target.col_stop
            ])
            .to_owned();
        self.analysis_origin.0 += target.row_start;
        self.analysis_origin.1 += target.col_start;
    }

    pub(crate) fn owner_output_coordinate(
        &self,
        owner: u32,
        point: (usize, usize),
    ) -> Option<(u64, u64)> {
        let point = (
            self.analysis_origin.0.checked_add(point.0)?,
            self.analysis_origin.1.checked_add(point.1)?,
        );
        let mapping = self
            .burst_output_mappings
            .iter()
            .find(|mapping| mapping.owner == owner)?;
        let local = (
            point.0.checked_sub(mapping.frame_origin.0)?,
            point.1.checked_sub(mapping.frame_origin.1)?,
        );
        if local.0 >= mapping.shape.0 || local.1 >= mapping.shape.1 {
            return None;
        }
        Some((
            mapping.output_origin.0.checked_add(local.0 as u64)?,
            mapping.output_origin.1.checked_add(local.1 as u64)?,
        ))
    }

    pub(crate) fn burst_ownership_digest(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"dolphinrust:production-burst-ownership-and-seams:v1");
        for burst in &self.source_burst_ids {
            digest.update((burst.len() as u64).to_le_bytes());
            digest.update(burst.as_bytes());
        }
        hash_shape(&mut digest, self.ownership.shape());
        for &owner in &self.ownership {
            digest.update(owner.to_le_bytes());
        }
        for (owner, rotations) in &self.seam_rotations {
            digest.update(owner.to_le_bytes());
            digest.update((rotations.len() as u64).to_le_bytes());
            for rotation in rotations {
                digest.update(rotation.re.to_bits().to_le_bytes());
                digest.update(rotation.im.to_bits().to_le_bytes());
            }
        }
        digest.update((self.analysis_origin.0 as u64).to_le_bytes());
        digest.update((self.analysis_origin.1 as u64).to_le_bytes());
        for mapping in &self.burst_output_mappings {
            digest.update(mapping.owner.to_le_bytes());
            digest.update((mapping.frame_origin.0 as u64).to_le_bytes());
            digest.update((mapping.frame_origin.1 as u64).to_le_bytes());
            digest.update(mapping.output_origin.0.to_le_bytes());
            digest.update(mapping.output_origin.1.to_le_bytes());
            digest.update((mapping.shape.0 as u64).to_le_bytes());
            digest.update((mapping.shape.1 as u64).to_le_bytes());
        }
        digest.finalize().into()
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn emit(
        self,
        cfg: &DisplacementWorkflow,
        sequential_cfg: &SequentialConfig,
        validity: ArrayView2<'_, bool>,
        reference: (usize, usize),
        epsg: u32,
        geotransform: [f64; 6],
        acquisition_days: &[f64],
    ) -> Result<()> {
        let fixed_l2 = self
            .fixed_l2_inputs
            .as_ref()
            .context("production spatial covariance requires the retained fixed-L2 map")?;
        anyhow::ensure!(
            validity.dim() == (self.ownership.dim().1, self.ownership.dim().2)
                && reference.0 < validity.nrows()
                && reference.1 < validity.ncols()
                && validity[reference],
            "production spatial covariance reference or validity grid is invalid"
        );
        let reference_owner = same_constant_owner(self.ownership.view(), reference, reference)
            .context(
                "production spatial covariance reference has mixed or missing burst ownership",
            )?;
        let reference_owner_index =
            usize::try_from(reference_owner).context("reference burst owner exceeds usize")?;
        let reference_burst = self
            .source_burst_ids
            .get(reference_owner_index)
            .context("reference burst owner is outside the source registry")?;
        let reference_output = self
            .owner_output_coordinate(reference_owner, reference)
            .context("selected reference is outside its captured burst output")?;
        let replay_context = self
            .replay_context
            .as_ref()
            .context("production spatial covariance requires captured replay context")?;
        anyhow::ensure!(
            !replay_context.tiles.is_empty()
                && replay_context.operator_block_byte_cap > 0
                && replay_context
                    .tiles
                    .iter()
                    .all(|tile| tile.request.branch_tolerance
                        == replay_context.tiles[0].request.branch_tolerance),
            "captured replay tiles do not share one positive branch contract"
        );

        preflight_existing_output(&cfg.work_directory)?;
        let live_operator = read_covariance_artifact_manifest_with_byte_cap(
            &cfg.work_directory,
            replay_context.operator_block_byte_cap,
        )
        .context("validating the committed covariance operator before production replay")?;
        anyhow::ensure!(
            live_operator == replay_context.operator_manifest,
            "committed covariance operator changed after production capture"
        );
        replay_context
            .source_manifest
            .verify_unchanged()
            .context("verifying immutable CSLC members before production replay")?;

        let topologies = replay_context
            .tiles
            .iter()
            .map(|tile| topology_for_tile(tile, sequential_cfg))
            .collect::<Result<Vec<_>>>()?;
        let build_identity = SequentialReplayBuildIdentity {
            normalized_config_digest: sequential_replay_config_digest(sequential_cfg),
            kernel_digest: sequential_replay_kernel_digest(),
            branch_tolerance: replay_context.tiles[0].request.branch_tolerance,
        };
        let preflight_provider_residency = ProviderResidencyTracker::default();
        let dates = fixed_l2
            .pixel_map(reference)
            .map_err(anyhow::Error::new)?
            .date_count();
        anyhow::ensure!(
            dates > 0
                && acquisition_days.len() == dates
                && dates == replay_context.tiles[0].num_real_dates
                && replay_context
                    .tiles
                    .iter()
                    .all(|tile| tile.num_real_dates == dates),
            "production L2 dates differ from captured replay dates"
        );
        let ordered_date_indices = (0..dates)
            .map(|date| u32::try_from(date).context("date index exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        let ordered_dates = ordered_date_indices
            .iter()
            .copied()
            .map(GlobalDateId::new)
            .collect::<Vec<_>>();
        let source_rank = 2_usize
            .checked_mul(dates.min(sequential_cfg.ministack_size))
            .context("production source rank overflows usize")?;
        let fixed_l2_workspace =
            fixed_l2_difference_workspace_composition(dates).map_err(anyhow::Error::new)?;
        let replay_reservation = preflight_production_replay_reservation(
            reference_burst,
            reference_output,
            &ordered_dates,
            source_rank,
            build_identity.branch_tolerance,
            cfg,
            replay_context,
            build_identity,
            &preflight_provider_residency,
            &topologies,
        )?;
        let block_shape = factor_block_shape(
            validity.dim(),
            dates,
            PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP,
            fixed_l2_workspace,
            replay_reservation,
        )?;
        let planned_block_targets = block_shape
            .0
            .checked_mul(block_shape.1)
            .context("planned factor block area overflows usize")?;
        let preflight_resource_receipt = production_resource_admission(
            planned_block_targets,
            dates,
            fixed_l2_workspace,
            replay_reservation,
            PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP,
        )?;
        let full_grid = CovarianceOperatorGrid {
            row_start: u64::try_from(self.analysis_origin.0)?,
            col_start: u64::try_from(self.analysis_origin.1)?,
            rows: u32::try_from(validity.nrows())?,
            cols: u32::try_from(validity.ncols())?,
            stride_y: u32::try_from(sequential_cfg.strides.y)?,
            stride_x: u32::try_from(sequential_cfg.strides.x)?,
        };
        let reference_global = (
            full_grid.row_start + u64::try_from(reference.0)?,
            full_grid.col_start + u64::try_from(reference.1)?,
        );
        let mask_digest = production_mask_digest(validity, self.ownership.view())?;
        let l2_map_digest = fixed_l2_frame_digest(Some(fixed_l2))?;
        let reference_signature = final_reference_signature_digest(
            full_grid,
            reference_global,
            reference_owner,
            &ordered_date_indices,
            geotransform,
        );
        let support_method = support_method(cfg.phase_linking.shp_method);
        let source_replay_digest = hash_serialized_identity(
            b"dolphinrust:production-source-replay-artifact:v1",
            &replay_context.operator_manifest,
        )?;
        let source_model_digest = replay_context
            .operator_manifest
            .source_model_receipt_digest
            .clone()
            .context("production operator is missing its source-model receipt")?;
        let effective_looks_digest = digest_string(
            b"dolphinrust:production-effective-looks:v1",
            &[
                EFFECTIVE_LOOKS_MODEL.as_bytes(),
                &EFFECTIVE_LOOKS_DISTANCE_SCALE_PIXELS
                    .to_bits()
                    .to_le_bytes(),
            ],
        );
        let support_digest = digest_string(
            b"dolphinrust:production-realized-support:v1",
            &[
                support_method.as_bytes(),
                replay_context.operator_manifest.hdf5_sha256.as_bytes(),
                &mask_digest,
            ],
        );
        let approximation_receipt_digest = digest_string(
            b"dolphinrust:production-approximation-unavailable:v1",
            &[b"uncalibrated"],
        );
        let resource_receipt_digest = digest_string(
            b"dolphinrust:production-resource-admission:v1",
            &[
                &PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP.to_le_bytes(),
                &replay_context.operator_block_byte_cap.to_le_bytes(),
                &2_u64.to_le_bytes(),
            ],
        );
        let mut metadata = SpatialReferenceCovarianceMetadata {
            schema_version: SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
            method: SPATIAL_REFERENCE_COVARIANCE_METHOD.to_owned(),
            method_version: 1,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            producer_commit: option_env!("DOLPHIN_GIT_COMMIT").map(str::to_owned),
            burst_id: reference_burst.clone(),
            crs: format!("EPSG:{epsg}"),
            units: match cfg.input_options.wavelength {
                Some(_) => "meters".to_owned(),
                None => "radians".to_owned(),
            },
            geotransform: Some(geotransform),
            full_grid,
            reference_row: reference_global.0,
            reference_col: reference_global.1,
            gauge_date_index: 0,
            ordered_date_indices,
            acquisition_days: Some(acquisition_days.to_vec()),
            mask_digest: sha256_string(mask_digest),
            source_replay_digest,
            l2_map_digest: sha256_string(l2_map_digest),
            reference_signature_digest: sha256_string(reference_signature),
            approximation_receipt_digest,
            resource_receipt_digest,
            runtime_resource_receipt_digest: spatial_reference_runtime_resource_receipt_digest(
                preflight_resource_receipt,
            ),
            runtime_resource_receipt: Some(preflight_resource_receipt),
            review_receipt_digest: String::new(),
            method_manifest_digest: String::new(),
            calibration_scope_digest: String::new(),
            source_model_digest,
            effective_looks_digest,
            support_method: support_method.to_owned(),
            support_digest,
            correction_order_digest: sha256_string(self.correction_order_digest),
            unwrap_branch_digest: sha256_string(self.unwrap_branch_digest),
            burst_ownership_digest: sha256_string(self.burst_ownership_digest()),
            source_burst_ids: self.source_burst_ids.clone(),
            reference_source_burst_index: reference_owner,
            calibration_scope: SpatialReferenceCalibrationScope::Uncalibrated,
            maximum_block_bytes: PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP,
        };
        let transaction =
            SpatialReferenceCovarianceArtifactTransaction::acquire(&cfg.work_directory)?;
        let scratch = cfg
            .work_directory
            .join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
        let mut writer = SpatialReferenceCovarianceWriter::create(&scratch, &metadata)?;
        let provider_residency = ProviderResidencyTracker::default();
        let phase_to_displacement = cfg
            .input_options
            .wavelength
            .map_or(1.0, |wavelength| -wavelength / (4.0 * std::f64::consts::PI));
        let mut block_id = 0_u64;
        let mut replay_observed_high_water_bytes = 0_u64;
        let mut fixed_l2_was_used = false;
        let plans = production_block_plans(
            &self,
            replay_context,
            reference_owner,
            reference_burst,
            validity.dim(),
            block_shape,
        )?;
        let reference_tile = owning_replay_tile(replay_context, reference_burst, reference_output)?;
        let mut production_metrics = ProductionProviderMetrics::default();
        let mut plan_start = 0;
        while plan_start < plans.len() {
            let target_tile = plans[plan_start].target_tile;
            let plan_stop = plans[plan_start..]
                .iter()
                .position(|plan| plan.target_tile != target_tile)
                .map_or(plans.len(), |offset| plan_start + offset);
            let mut providers = target_tile
                .map(|target_tile| {
                    ProductionProviderBundle::open(
                        cfg,
                        replay_context,
                        &topologies,
                        build_identity,
                        &provider_residency,
                        target_tile,
                        reference_tile,
                    )
                })
                .transpose()?;
            for plan in &plans[plan_start..plan_stop] {
                let target_grid = CovarianceOperatorGrid {
                    row_start: full_grid.row_start + u64::try_from(plan.row_start)?,
                    col_start: full_grid.col_start + u64::try_from(plan.col_start)?,
                    rows: u32::try_from(plan.rows)?,
                    cols: u32::try_from(plan.cols)?,
                    stride_y: full_grid.stride_y,
                    stride_x: full_grid.stride_x,
                };
                let mut builder =
                    FactorBlockBuilder::new(block_id, target_grid, dates, phase_to_displacement)?;
                for row in plan.row_start..plan.row_start + plan.rows {
                    for column in plan.col_start..plan.col_start + plan.cols {
                        let target = (row, column);
                        let outcome = if let Some(providers) = providers.as_mut() {
                            production_target_factor(
                                &self,
                                fixed_l2,
                                validity,
                                target,
                                reference,
                                reference_output,
                                &ordered_dates,
                                source_rank,
                                build_identity.branch_tolerance,
                                replay_context,
                                providers,
                                preflight_resource_receipt.replay_admission_high_water_bytes,
                            )?
                        } else {
                            production_nonreplay_target(
                                &self,
                                validity,
                                target,
                                reference,
                                reference_owner,
                            )?
                        };
                        builder.push(outcome)?;
                    }
                }
                let block = builder.finish()?;
                if let Some(resource) = &block.resource_high_water_bytes {
                    replay_observed_high_water_bytes = replay_observed_high_water_bytes
                        .max(resource.iter().copied().max().unwrap_or(0));
                }
                fixed_l2_was_used |= block.status.iter().any(|status| {
                    matches!(
                        status,
                        SpatialReferenceCovarianceStatus::Valid
                            | SpatialReferenceCovarianceStatus::L2RankDeficient
                            | SpatialReferenceCovarianceStatus::IllConditioned
                            | SpatialReferenceCovarianceStatus::TemporalFactorInvalid
                            | SpatialReferenceCovarianceStatus::UnsupportedL1
                    )
                });
                writer.write_block(&block)?;
                block_id = block_id
                    .checked_add(1)
                    .context("factor block ID overflow")?;
            }
            if let Some(providers) = providers.as_ref() {
                let metrics = providers.metrics();
                production_metrics.operator_block_reads += metrics.operator_block_reads;
                production_metrics.operator_block_cache_hits += metrics.operator_block_cache_hits;
                production_metrics.source_member_window_reads += metrics.source_member_window_reads;
                production_metrics.source_tile_cache_loads += metrics.source_tile_cache_loads;
                production_metrics.source_resolutions += metrics.source_resolutions;
            }
            drop(providers);
            plan_start = plan_stop;
        }
        let provider_receipt = provider_residency.receipt();
        anyhow::ensure!(
            provider_receipt.current_count == 0
                && provider_receipt.current_bytes == 0
                && provider_receipt.peak_count <= 2
                && provider_receipt.peak_bytes
                    <= preflight_resource_receipt.replay_admission_high_water_bytes,
            "production replay provider residency exceeded its two-provider bound"
        );
        let fixed_l2_workspace_observed_high_water_bytes = if fixed_l2_was_used {
            fixed_l2_workspace.total_bytes
        } else {
            0
        };
        let working_set_observed_high_water_bytes = preflight_resource_receipt
            .factor_block_high_water_bytes
            .checked_add(preflight_resource_receipt.serialization_high_water_bytes)
            .and_then(|bytes| bytes.checked_add(fixed_l2_workspace_observed_high_water_bytes))
            .and_then(|bytes| bytes.checked_add(replay_observed_high_water_bytes))
            .context("observed covariance working set overflows u64")?;
        let runtime_resource_receipt = SpatialReferenceRuntimeResourceReceipt {
            fixed_l2_workspace_observed_high_water_bytes,
            replay_observed_high_water_bytes,
            provider_peak_count: u64::try_from(provider_receipt.peak_count)?,
            provider_peak_bytes: provider_receipt.peak_bytes,
            preflight_provider_open_count: preflight_provider_residency.receipt().open_count,
            production_provider_open_count: provider_receipt.open_count,
            operator_block_reads: production_metrics.operator_block_reads,
            operator_block_cache_hits: production_metrics.operator_block_cache_hits,
            source_member_window_reads: production_metrics.source_member_window_reads,
            source_tile_cache_loads: production_metrics.source_tile_cache_loads,
            source_resolutions: production_metrics.source_resolutions,
            working_set_observed_high_water_bytes,
            ..preflight_resource_receipt
        };
        writer.seal_runtime_resource_receipt(runtime_resource_receipt)?;
        metadata = writer.seal_effective_looks_digest()?;
        let write_receipt = writer.finish()?;
        replay_context
            .source_manifest
            .verify_unchanged()
            .context("verifying immutable CSLC members before factor commit")?;
        let live_operator_after = read_covariance_artifact_manifest_with_byte_cap(
            &cfg.work_directory,
            replay_context.operator_block_byte_cap,
        )?;
        anyhow::ensure!(
            live_operator_after == replay_context.operator_manifest,
            "committed covariance operator changed during factor replay"
        );
        finalize_spatial_reference_covariance_artifact(
            &transaction,
            &scratch,
            &metadata,
            &write_receipt,
        )?;
        Ok(())
    }
}

fn preflight_existing_output(directory: &Path) -> Result<()> {
    let hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    if hdf5.try_exists()? && manifest.try_exists()? {
        crate::spatial_covariance_artifact::read_spatial_reference_covariance_artifact_manifest(
            directory,
        )?;
        anyhow::bail!("a valid spatial covariance artifact already exists");
    }
    Ok(())
}

fn topology_for_tile(
    tile: &CapturedReplayTile,
    cfg: &SequentialConfig,
) -> Result<SequentialReplayTopology> {
    let native_shape = (
        usize::try_from(tile.request.native_grid.rows)?,
        usize::try_from(tile.request.native_grid.cols)?,
    );
    let output_shape = (
        usize::try_from(tile.request.output_grid.rows)?,
        usize::try_from(tile.request.output_grid.cols)?,
    );
    anyhow::ensure!(
        tile.native_validity.dim() == native_shape,
        "captured replay tile validity differs from its native grid"
    );
    let support_rows = cfg
        .half_window
        .y
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("support rows overflow usize")?;
    let support_cols = cfg
        .half_window
        .x
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("support columns overflow usize")?;
    let namespace = tile
        .request
        .namespace_for(native_shape, output_shape, cfg.strides)
        .map_err(anyhow::Error::new)?;
    SequentialReplayTopology::plan_identified(
        tile.num_real_dates,
        native_shape,
        output_shape,
        support_rows
            .checked_mul(support_cols)
            .context("support area overflows usize")?,
        tile.native_validity.view(),
        cfg,
        ReplayExecutionScope {
            enabled: true,
            backend: ReplayBackend::CpuF64,
            estimator_fallback: false,
            phase_bias_correction: false,
            strong_source_identity: true,
            stitched_burst_count: 1,
        },
        namespace,
    )
    .map_err(anyhow::Error::new)
}

fn owning_replay_tile(
    replay_context: &ProductionCovarianceReplayContext,
    burst_id: &str,
    coordinate: (u64, u64),
) -> Result<usize> {
    let mut owner = None;
    for (index, tile) in replay_context.tiles.iter().enumerate() {
        let grid = tile.request.owned_output_grid;
        let row_stop = grid
            .row_start
            .checked_add(u64::from(grid.rows))
            .context("owned replay row extent overflows u64")?;
        let col_stop = grid
            .col_start
            .checked_add(u64::from(grid.cols))
            .context("owned replay column extent overflows u64")?;
        if tile.request.burst_id == burst_id
            && coordinate.0 >= grid.row_start
            && coordinate.0 < row_stop
            && coordinate.1 >= grid.col_start
            && coordinate.1 < col_stop
        {
            anyhow::ensure!(
                owner.replace(index).is_none(),
                "global output has multiple owning replay tiles"
            );
        }
    }
    owner.context("global output has no owning replay tile")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveProviderPlan {
    target_tile: usize,
    reference_tile: Option<usize>,
}

impl ActiveProviderPlan {
    const fn new(target_tile: usize, reference_tile: usize) -> Self {
        Self {
            target_tile,
            reference_tile: if target_tile == reference_tile {
                None
            } else {
                Some(reference_tile)
            },
        }
    }

    #[cfg(test)]
    const fn provider_count(self) -> usize {
        1 + self.reference_tile.is_some() as usize
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ProviderResidencyReceipt {
    current_count: usize,
    peak_count: usize,
    current_bytes: u64,
    peak_bytes: u64,
    open_count: u64,
}

#[derive(Clone, Default)]
struct ProviderResidencyTracker {
    receipt: Rc<Cell<ProviderResidencyReceipt>>,
}

impl ProviderResidencyTracker {
    fn track<P>(&self, provider: P, reservation_bytes: u64) -> Result<ResidentProvider<P>> {
        let mut receipt = self.receipt.get();
        receipt.open_count = receipt
            .open_count
            .checked_add(1)
            .context("provider open count overflow")?;
        receipt.current_count = receipt
            .current_count
            .checked_add(1)
            .context("active provider count overflow")?;
        receipt.current_bytes = receipt
            .current_bytes
            .checked_add(reservation_bytes)
            .context("active provider reservation overflow")?;
        receipt.peak_count = receipt.peak_count.max(receipt.current_count);
        receipt.peak_bytes = receipt.peak_bytes.max(receipt.current_bytes);
        self.receipt.set(receipt);
        Ok(ResidentProvider {
            provider: Some(provider),
            tracker: self.clone(),
            reservation_bytes,
        })
    }

    fn release(&self, reservation_bytes: u64) {
        let mut receipt = self.receipt.get();
        receipt.current_count = receipt
            .current_count
            .checked_sub(1)
            .expect("provider residency count underflow");
        receipt.current_bytes = receipt
            .current_bytes
            .checked_sub(reservation_bytes)
            .expect("provider residency bytes underflow");
        self.receipt.set(receipt);
    }

    fn receipt(&self) -> ProviderResidencyReceipt {
        self.receipt.get()
    }
}

struct ResidentProvider<P> {
    provider: Option<P>,
    tracker: ProviderResidencyTracker,
    reservation_bytes: u64,
}

impl<P> ResidentProvider<P> {
    fn provider_mut(&mut self) -> &mut P {
        self.provider
            .as_mut()
            .expect("resident provider is available until drop")
    }
}

impl<P> Drop for ResidentProvider<P> {
    fn drop(&mut self) {
        drop(self.provider.take());
        self.tracker.release(self.reservation_bytes);
    }
}

fn open_production_provider<'a>(
    cfg: &DisplacementWorkflow,
    replay_context: &'a ProductionCovarianceReplayContext,
    topologies: &'a [SequentialReplayTopology],
    build_identity: SequentialReplayBuildIdentity,
    tile_index: usize,
) -> Result<CovarianceArtifactReplayProvider<'a, CslcCovarianceSourceResolver<'a>>> {
    let tile = replay_context
        .tiles
        .get(tile_index)
        .context("production replay tile index is out of range")?;
    let topology = topologies
        .get(tile_index)
        .context("production replay topology index is out of range")?;
    let resolver = replay_context.source_manifest.resolver(
        &tile.member_indices,
        tile.request.burst_id.clone(),
        tile.processed_origin,
        tile.processed_shape,
        tile.request.native_grid,
        &cfg.phase_linking.empirical_source_factor,
        tile.request.source_model_version_digest,
        replay_context
            .masks
            .get(&tile.request.burst_id)
            .and_then(Option::as_ref)
            .map(|mask| mask as &dyn crate::cslc_covariance_source::CslcCovarianceValidityReader),
    )?;
    CovarianceArtifactReplayProvider::open(
        &cfg.work_directory,
        replay_context.operator_block_byte_cap,
        topology,
        build_identity,
        resolver,
    )
    .map_err(anyhow::Error::new)
}

type ProductionReplayProvider<'a> =
    CovarianceArtifactReplayProvider<'a, CslcCovarianceSourceResolver<'a>>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ProductionProviderMetrics {
    operator_block_reads: u64,
    operator_block_cache_hits: u64,
    source_member_window_reads: u64,
    source_tile_cache_loads: u64,
    source_resolutions: u64,
}

struct ProductionProviderBundle<'a> {
    plan: ActiveProviderPlan,
    topologies: &'a [SequentialReplayTopology],
    target: ResidentProvider<ProductionReplayProvider<'a>>,
    reference: Option<ResidentProvider<ProductionReplayProvider<'a>>>,
}

impl<'a> ProductionProviderBundle<'a> {
    fn open(
        cfg: &DisplacementWorkflow,
        replay_context: &'a ProductionCovarianceReplayContext,
        topologies: &'a [SequentialReplayTopology],
        build_identity: SequentialReplayBuildIdentity,
        tracker: &ProviderResidencyTracker,
        target_tile: usize,
        reference_tile: usize,
    ) -> Result<Self> {
        let plan = ActiveProviderPlan::new(target_tile, reference_tile);
        let target =
            open_production_provider(cfg, replay_context, topologies, build_identity, target_tile)?;
        let target_reservation = target.maximum_resident_bytes();
        let target = tracker.track(target, target_reservation)?;
        let reference = if let Some(reference_tile) = plan.reference_tile {
            let reference = open_production_provider(
                cfg,
                replay_context,
                topologies,
                build_identity,
                reference_tile,
            )?;
            let reservation = reference.maximum_resident_bytes();
            Some(tracker.track(reference, reservation)?)
        } else {
            None
        };
        Ok(Self {
            plan,
            topologies,
            target,
            reference,
        })
    }

    fn replay(
        &mut self,
        query: GlobalReferenceCovarianceQuery<'_>,
    ) -> std::result::Result<
        crate::sequential_covariance::GlobalReferenceDifferenceCovarianceReplay,
        SequentialReplayError,
    > {
        if let Some(reference) = self.reference.as_mut() {
            let reference_tile = self
                .plan
                .reference_tile
                .expect("reference provider has a tile");
            let mut bundle = [
                SequentialTileReplayProvider::new(
                    &self.topologies[self.plan.target_tile],
                    self.target.provider_mut(),
                ),
                SequentialTileReplayProvider::new(
                    &self.topologies[reference_tile],
                    reference.provider_mut(),
                ),
            ];
            replay_global_reference_difference_covariance_from_provider_bundle(&mut bundle, query)
        } else {
            let mut bundle = [SequentialTileReplayProvider::new(
                &self.topologies[self.plan.target_tile],
                self.target.provider_mut(),
            )];
            replay_global_reference_difference_covariance_from_provider_bundle(&mut bundle, query)
        }
    }

    fn metrics(&self) -> ProductionProviderMetrics {
        let mut metrics = ProductionProviderMetrics::default();
        for provider in std::iter::once(
            self.target
                .provider
                .as_ref()
                .expect("target provider remains live"),
        )
        .chain(
            self.reference
                .as_ref()
                .and_then(|provider| provider.provider.as_ref()),
        ) {
            let operator = provider.metrics();
            let source = provider.source_resolver().metrics();
            metrics.operator_block_reads += operator.operator_block_reads;
            metrics.operator_block_cache_hits += operator.operator_block_cache_hits;
            metrics.source_member_window_reads += source.member_window_reads;
            metrics.source_tile_cache_loads += source.tile_cache_loads;
            metrics.source_resolutions += source.source_resolutions;
        }
        metrics
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn preflight_production_replay_reservation(
    reference_burst: &str,
    reference_output: (u64, u64),
    ordered_dates: &[GlobalDateId],
    source_rank: usize,
    branch_tolerance: f64,
    cfg: &DisplacementWorkflow,
    replay_context: &ProductionCovarianceReplayContext,
    build_identity: SequentialReplayBuildIdentity,
    provider_residency: &ProviderResidencyTracker,
    topologies: &[SequentialReplayTopology],
) -> Result<u64> {
    let mut maximum = 0_u64;
    for tile_index in 0..replay_context.tiles.len() {
        let provider =
            open_production_provider(cfg, replay_context, topologies, build_identity, tile_index)?;
        let reservation = provider.maximum_resident_bytes();
        maximum = maximum.max(reservation);
        drop(provider_residency.track(provider, reservation)?);
    }
    let reference_tile = owning_replay_tile(replay_context, reference_burst, reference_output)?;
    for (target_tile, tile) in replay_context.tiles.iter().enumerate() {
        if tile.request.burst_id != reference_burst {
            continue;
        }
        let provider_plan = ActiveProviderPlan::new(target_tile, reference_tile);
        if provider_plan.reference_tile.is_none() {
            let provider = open_production_provider(
                cfg,
                replay_context,
                topologies,
                build_identity,
                target_tile,
            )?;
            let reservation = provider.maximum_resident_bytes();
            let mut provider = provider_residency.track(provider, reservation)?;
            for row in 0..tile.request.owned_output_grid.rows {
                for column in 0..tile.request.owned_output_grid.cols {
                    let target = (
                        tile.request.owned_output_grid.row_start + u64::from(row),
                        tile.request.owned_output_grid.col_start + u64::from(column),
                    );
                    let query = GlobalReferenceCovarianceQuery {
                        burst_id: reference_burst,
                        target,
                        reference: reference_output,
                        ordered_dates,
                        source_rank,
                        byte_cap: u64::MAX,
                        branch_tolerance,
                    };
                    let mut bundle = [SequentialTileReplayProvider::new(
                        &topologies[target_tile],
                        provider.provider_mut(),
                    )];
                    maximum = maximum.max(
                        estimate_global_reference_difference_covariance_from_provider_bundle(
                            &mut bundle,
                            query,
                        )?
                        .total_bytes,
                    );
                }
            }
        } else {
            let target_provider = open_production_provider(
                cfg,
                replay_context,
                topologies,
                build_identity,
                target_tile,
            )?;
            let target_reservation = target_provider.maximum_resident_bytes();
            let mut target_provider =
                provider_residency.track(target_provider, target_reservation)?;
            let reference_tile = provider_plan
                .reference_tile
                .context("two-provider replay plan lost its reference tile")?;
            let reference_provider = open_production_provider(
                cfg,
                replay_context,
                topologies,
                build_identity,
                reference_tile,
            )?;
            let reference_reservation = reference_provider.maximum_resident_bytes();
            let mut reference_provider =
                provider_residency.track(reference_provider, reference_reservation)?;
            for row in 0..tile.request.owned_output_grid.rows {
                for column in 0..tile.request.owned_output_grid.cols {
                    let target = (
                        tile.request.owned_output_grid.row_start + u64::from(row),
                        tile.request.owned_output_grid.col_start + u64::from(column),
                    );
                    let query = GlobalReferenceCovarianceQuery {
                        burst_id: reference_burst,
                        target,
                        reference: reference_output,
                        ordered_dates,
                        source_rank,
                        byte_cap: u64::MAX,
                        branch_tolerance,
                    };
                    let mut bundle = [
                        SequentialTileReplayProvider::new(
                            &topologies[target_tile],
                            target_provider.provider_mut(),
                        ),
                        SequentialTileReplayProvider::new(
                            &topologies[reference_tile],
                            reference_provider.provider_mut(),
                        ),
                    ];
                    maximum = maximum.max(
                        estimate_global_reference_difference_covariance_from_provider_bundle(
                            &mut bundle,
                            query,
                        )?
                        .total_bytes,
                    );
                }
            }
        }
    }
    Ok(maximum)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProductionBlockPlan {
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    target_tile: Option<usize>,
}

fn production_block_plans(
    state: &ProductionCovarianceState,
    replay_context: &ProductionCovarianceReplayContext,
    reference_owner: u32,
    reference_burst: &str,
    shape: (usize, usize),
    block_shape: (usize, usize),
) -> Result<Vec<ProductionBlockPlan>> {
    let mapping = state
        .burst_output_mappings
        .iter()
        .find(|mapping| mapping.owner == reference_owner)
        .context("reference burst has no output mapping")?;
    let mut row_bounds = BTreeSet::from([0, shape.0]);
    let mut col_bounds = BTreeSet::from([0, shape.1]);
    for tile in &replay_context.tiles {
        if tile.request.burst_id != reference_burst {
            continue;
        }
        let grid = tile.request.owned_output_grid;
        let frame_row = i128::try_from(mapping.frame_origin.0)? + i128::from(grid.row_start)
            - i128::from(mapping.output_origin.0);
        let frame_col = i128::try_from(mapping.frame_origin.1)? + i128::from(grid.col_start)
            - i128::from(mapping.output_origin.1);
        let local_row = frame_row - i128::try_from(state.analysis_origin.0)?;
        let local_col = frame_col - i128::try_from(state.analysis_origin.1)?;
        let row_start = usize::try_from(local_row.max(0))?.min(shape.0);
        let col_start = usize::try_from(local_col.max(0))?.min(shape.1);
        let row_stop = usize::try_from((local_row + i128::from(grid.rows)).max(0))?.min(shape.0);
        let col_stop = usize::try_from((local_col + i128::from(grid.cols)).max(0))?.min(shape.1);
        row_bounds.extend([row_start, row_stop]);
        col_bounds.extend([col_start, col_stop]);
    }
    let rows = row_bounds.into_iter().collect::<Vec<_>>();
    let cols = col_bounds.into_iter().collect::<Vec<_>>();
    let mut plans = Vec::new();
    for row_window in rows.windows(2) {
        for col_window in cols.windows(2) {
            if row_window[0] == row_window[1] || col_window[0] == col_window[1] {
                continue;
            }
            let target_tile = state
                .owner_output_coordinate(reference_owner, (row_window[0], col_window[0]))
                .map(|coordinate| owning_replay_tile(replay_context, reference_burst, coordinate))
                .transpose()?;
            for row_start in (row_window[0]..row_window[1]).step_by(block_shape.0) {
                for col_start in (col_window[0]..col_window[1]).step_by(block_shape.1) {
                    plans.push(ProductionBlockPlan {
                        row_start,
                        col_start,
                        rows: block_shape.0.min(row_window[1] - row_start),
                        cols: block_shape.1.min(col_window[1] - col_start),
                        target_tile,
                    });
                }
            }
        }
    }
    plans.sort_by_key(|plan| (plan.target_tile, plan.row_start, plan.col_start));
    Ok(plans)
}

fn production_nonreplay_target(
    state: &ProductionCovarianceState,
    validity: ArrayView2<'_, bool>,
    target: (usize, usize),
    reference: (usize, usize),
    reference_owner: u32,
) -> Result<TargetFactor> {
    if !validity[target] {
        return Ok(nonvalid_target(
            SpatialReferenceCovarianceStatus::MaskedTarget,
        ));
    }
    let Some(owner) = same_constant_owner(state.ownership.view(), target, reference) else {
        return Ok(nonvalid_target(
            SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference,
        ));
    };
    anyhow::ensure!(
        owner != reference_owner,
        "valid same-owner production target has no retained replay tile"
    );
    Ok(nonvalid_target(
        SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference,
    ))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn production_target_factor(
    state: &ProductionCovarianceState,
    fixed_l2: &FixedL2WorkflowInputs,
    validity: ArrayView2<'_, bool>,
    target: (usize, usize),
    reference: (usize, usize),
    reference_output: (u64, u64),
    ordered_dates: &[GlobalDateId],
    source_rank: usize,
    branch_tolerance: f64,
    replay_context: &ProductionCovarianceReplayContext,
    providers: &mut ProductionProviderBundle<'_>,
    query_byte_cap: u64,
) -> Result<TargetFactor> {
    anyhow::ensure!(query_byte_cap > 0, "production replay byte cap is zero");
    if !validity[target] {
        return Ok(nonvalid_target(
            SpatialReferenceCovarianceStatus::MaskedTarget,
        ));
    }
    let Some(owner) = same_constant_owner(state.ownership.view(), target, reference) else {
        return Ok(nonvalid_target(
            SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference,
        ));
    };
    let Some(target_output) = state.owner_output_coordinate(owner, target) else {
        return Ok(nonvalid_target(
            SpatialReferenceCovarianceStatus::InvalidReference,
        ));
    };
    let burst = state
        .source_burst_ids
        .get(usize::try_from(owner)?)
        .context("target burst owner is outside the source registry")?;
    let target_tile = owning_replay_tile(replay_context, burst, target_output)?;
    anyhow::ensure!(
        target_tile == providers.plan.target_tile,
        "target was routed through the wrong retained replay provider"
    );
    let query = GlobalReferenceCovarianceQuery {
        burst_id: burst,
        target: target_output,
        reference: reference_output,
        ordered_dates,
        source_rank,
        byte_cap: query_byte_cap,
        branch_tolerance,
    };
    let active_provider_bytes = providers
        .target
        .reservation_bytes
        .checked_add(
            providers
                .reference
                .as_ref()
                .map_or(0, |provider| provider.reservation_bytes),
        )
        .context("active provider reservation overflows u64")?;
    let replay_result = providers.replay(query);
    let replay = match replay_result {
        Ok(replay) => replay,
        Err(error) if replay_failure_aborts(&error) => return Err(anyhow::Error::new(error)),
        Err(error) => {
            return Ok(nonvalid_target_with_resource(
                replay_status(error.status()),
                active_provider_bytes,
            ));
        }
    };
    let effective = replay
        .replay
        .effective_looks
        .as_ref()
        .context("production replay did not apply the declared effective-look source factor")?;
    validate_effective_looks(effective)?;
    let propagated = match fixed_l2.propagate_joint_phase_covariance(
        target,
        reference,
        replay.joint_phase_covariance.view(),
    ) {
        Ok(propagated) => propagated,
        Err(error) => {
            return Ok(nonvalid_target_with_resource(
                fixed_l2_status(error.status),
                replay.resource_high_water_bytes,
            ));
        }
    };
    anyhow::ensure!(
        propagated.status == SpatialL2Status::Valid,
        "successful production L2 propagation returned a non-valid status"
    );
    let condition_number = if propagated.date_factor.ncols() == 0 {
        f64::NAN
    } else {
        propagated.covariance_condition_number
    };
    Ok(TargetFactor {
        status: SpatialReferenceCovarianceStatus::Valid,
        source_burst_index: owner,
        date_factor: Some(propagated.date_factor),
        source_factor_receipt: production_target_receipt(
            replay.replay.source_factor_receipt,
            replay.replay.support_receipt,
            replay.replay.reference_signature,
            effective.model,
            effective.distance_scale_pixels,
            effective.support_union_count,
            effective.fraction,
            effective.receipt,
            replay.resource_high_water_bytes,
        ),
        effective_looks_fraction: effective.fraction,
        support_union_count: u64::try_from(effective.support_union_count)
            .context("effective-look support union exceeds u64")?,
        effective_looks_receipt: effective.receipt,
        resource_high_water_bytes: replay.resource_high_water_bytes,
        condition_number,
    })
}

fn replay_failure_aborts(error: &SequentialReplayError) -> bool {
    matches!(
        error.status(),
        ReplayStatus::SourceUnavailable
            | ReplayStatus::SourceIdentityMismatch
            | ReplayStatus::ReplayStateMismatch
    )
}

fn validate_effective_looks(effective: &EffectiveLooksReplay) -> Result<()> {
    anyhow::ensure!(
        effective.model == EFFECTIVE_LOOKS_MODEL
            && effective.distance_scale_pixels == EFFECTIVE_LOOKS_DISTANCE_SCALE_PIXELS
            && effective.support_union_count > 0
            && effective.fraction.is_finite()
            && effective.fraction > 0.0
            && effective.fraction <= 1.0
            && effective.receipt.iter().any(|byte| *byte != 0),
        "production replay effective-look receipt differs from source_factor_declared_v1"
    );
    Ok(())
}

fn nonvalid_target(status: SpatialReferenceCovarianceStatus) -> TargetFactor {
    nonvalid_target_with_resource(status, 0)
}

fn nonvalid_target_with_resource(
    status: SpatialReferenceCovarianceStatus,
    resource_high_water_bytes: u64,
) -> TargetFactor {
    TargetFactor {
        status,
        source_burst_index: SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
        date_factor: None,
        source_factor_receipt: [0; 32],
        effective_looks_fraction: f64::NAN,
        support_union_count: 0,
        effective_looks_receipt: [0; 32],
        resource_high_water_bytes,
        condition_number: f64::NAN,
    }
}

fn factor_block_shape(
    shape: (usize, usize),
    dates: usize,
    working_set_byte_cap: u64,
    fixed_l2_workspace: FixedL2WorkspaceComposition,
    replay_reservation_bytes: u64,
) -> Result<(usize, usize)> {
    let per_target = factor_target_payload_bytes(dates)?;
    let available_for_blocks = working_set_byte_cap
        .checked_sub(fixed_l2_workspace.total_bytes)
        .and_then(|bytes| bytes.checked_sub(replay_reservation_bytes))
        .context("covariance working-set cap cannot hold fixed-L2 workspace and replay")?;
    let targets = usize::try_from(available_for_blocks / per_target / 2)
        .context("factor target block capacity exceeds usize")?;
    anyhow::ensure!(
        targets > 0,
        "covariance working-set cap cannot hold one target"
    );
    let cols = shape.1.min(targets).max(1);
    let rows = shape.0.min(targets / cols).max(1);
    Ok((rows, cols))
}

fn factor_target_payload_bytes(dates: usize) -> Result<u64> {
    u64::try_from(dates)?
        .checked_mul(u64::try_from(dates)?)
        .and_then(|value| value.checked_mul(8))
        .and_then(|value| value.checked_add(82))
        .context("factor target payload bytes overflow u64")
}

fn production_resource_admission(
    targets: usize,
    dates: usize,
    fixed_l2_workspace: FixedL2WorkspaceComposition,
    replay_admission_high_water_bytes: u64,
    working_set_byte_cap: u64,
) -> Result<SpatialReferenceRuntimeResourceReceipt> {
    let factor_block_high_water_bytes = u64::try_from(targets)?
        .checked_mul(factor_target_payload_bytes(dates)?)
        .context("factor block payload bytes overflow u64")?;
    let serialization_high_water_bytes = factor_block_high_water_bytes;
    let working_set_admission_high_water_bytes = factor_block_high_water_bytes
        .checked_add(serialization_high_water_bytes)
        .and_then(|bytes| bytes.checked_add(fixed_l2_workspace.total_bytes))
        .and_then(|bytes| bytes.checked_add(replay_admission_high_water_bytes))
        .context("covariance working-set admission overflows u64")?;
    anyhow::ensure!(
        working_set_admission_high_water_bytes <= working_set_byte_cap,
        "covariance working-set admission exceeds its byte cap"
    );
    Ok(SpatialReferenceRuntimeResourceReceipt {
        working_set_byte_cap,
        factor_block_high_water_bytes,
        serialization_high_water_bytes,
        fixed_l2_workspace_admission_bytes: fixed_l2_workspace.total_bytes,
        fixed_l2_workspace_observed_high_water_bytes: 0,
        replay_admission_high_water_bytes,
        replay_observed_high_water_bytes: 0,
        provider_peak_count: 2,
        provider_peak_bytes: replay_admission_high_water_bytes,
        preflight_provider_open_count: 0,
        production_provider_open_count: 0,
        operator_block_reads: 0,
        operator_block_cache_hits: 0,
        source_member_window_reads: 0,
        source_tile_cache_loads: 0,
        source_resolutions: 0,
        working_set_admission_high_water_bytes,
        working_set_observed_high_water_bytes: factor_block_high_water_bytes
            .checked_add(serialization_high_water_bytes)
            .context("covariance observed working-set admission overflows u64")?,
    })
}

fn support_method(method: ShpMethod) -> &'static str {
    match method {
        ShpMethod::Rect => "rect",
        ShpMethod::Glrt => "glrt_frozen",
        ShpMethod::Ks => "ks_frozen",
    }
}

fn hash_serialized_identity<T: serde::Serialize>(domain: &[u8], value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(digest_string(domain, &[&bytes]))
}

fn digest_string(domain: &[u8], values: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    for value in values {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value);
    }
    format!("sha256:{:x}", digest.finalize())
}

fn sha256_string(value: [u8; 32]) -> String {
    value
        .iter()
        .fold(String::from("sha256:"), |mut output, byte| {
            use std::fmt::Write as _;
            write!(output, "{byte:02x}").expect("writing to String cannot fail");
            output
        })
}

/// Exact post-loop-QC observations and weights used by the production L2 solve.
pub(crate) struct FixedL2WorkflowInputs {
    incidence: Array2<f64>,
    dphi_rad: Array3<f64>,
    precision: Option<Array3<f64>>,
}

impl FixedL2WorkflowInputs {
    pub(crate) fn new(
        incidence: Array2<f64>,
        dphi_rad: Array3<f64>,
        precision: Option<Array3<f64>>,
    ) -> Result<Self> {
        anyhow::ensure!(
            incidence.nrows() == dphi_rad.dim().0
                && incidence.ncols() > 0
                && precision
                    .as_ref()
                    .is_none_or(|values| values.dim() == dphi_rad.dim()),
            "fixed L2 workflow inputs disagree"
        );
        Ok(Self {
            incidence,
            dphi_rad,
            precision,
        })
    }

    pub(crate) fn pixel_map(
        &self,
        pixel: (usize, usize),
    ) -> std::result::Result<PixelL2ObservationMap, PixelL2MapError> {
        fixed_l2_pixel_map(
            self.incidence.view(),
            self.dphi_rad.view(),
            self.precision.as_ref().map(Array3::view),
            pixel,
            self.incidence.nrows(),
        )
    }

    pub(crate) fn pixel_map_digest(&self, pixel: (usize, usize)) -> Result<[u8; 32]> {
        let map = self.pixel_map(pixel).map_err(anyhow::Error::new)?;
        let mut digest = Sha256::new();
        digest.update(b"dolphinrust:production-fixed-l2-pixel-map:v1");
        for &index in map.valid_observation_indices() {
            digest.update((index as u64).to_le_bytes());
        }
        hash_f64_values(&mut digest, map.precisions().iter().copied());
        hash_f64_values(&mut digest, map.observation_phase_map().iter().copied());
        hash_f64_values(&mut digest, map.h_map().iter().copied());
        digest.update(map.condition_number().to_bits().to_le_bytes());
        Ok(digest.finalize().into())
    }

    pub(crate) fn propagate_joint_phase_covariance(
        &self,
        target: (usize, usize),
        reference: (usize, usize),
        joint_phase_covariance: ArrayView2<'_, f64>,
    ) -> std::result::Result<FixedL2DifferenceCovariance, SpatialL2Error> {
        let target = self.pixel_map(target).map_err(|error| SpatialL2Error {
            status: match error.status {
                dolphin_timeseries::inversion::PixelL2MapStatus::RankDeficient
                | dolphin_timeseries::inversion::PixelL2MapStatus::InsufficientObservations => {
                    dolphin_timeseries::spatial_covariance::SpatialL2Status::RankDeficient
                }
                dolphin_timeseries::inversion::PixelL2MapStatus::NonFinite => {
                    dolphin_timeseries::spatial_covariance::SpatialL2Status::NonFinite
                }
                dolphin_timeseries::inversion::PixelL2MapStatus::InvalidInput => {
                    dolphin_timeseries::spatial_covariance::SpatialL2Status::InvalidInput
                }
            },
            message: error.message,
        })?;
        let reference = self.pixel_map(reference).map_err(|error| SpatialL2Error {
            status: match error.status {
                dolphin_timeseries::inversion::PixelL2MapStatus::RankDeficient
                | dolphin_timeseries::inversion::PixelL2MapStatus::InsufficientObservations => {
                    dolphin_timeseries::spatial_covariance::SpatialL2Status::RankDeficient
                }
                dolphin_timeseries::inversion::PixelL2MapStatus::NonFinite => {
                    dolphin_timeseries::spatial_covariance::SpatialL2Status::NonFinite
                }
                dolphin_timeseries::inversion::PixelL2MapStatus::InvalidInput => {
                    dolphin_timeseries::spatial_covariance::SpatialL2Status::InvalidInput
                }
            },
            message: error.message,
        })?;
        propagate_fixed_l2_difference_covariance(
            &target,
            &reference,
            joint_phase_covariance,
            SpatialL2Branch::FixedL2,
        )
    }

    pub(crate) fn trim(&mut self, target: dolphin_core::BlockIndices) {
        self.dphi_rad = self
            .dphi_rad
            .slice(ndarray::s![
                ..,
                target.row_start..target.row_stop,
                target.col_start..target.col_stop
            ])
            .to_owned();
        self.precision = self.precision.take().map(|values| {
            values
                .slice(ndarray::s![
                    ..,
                    target.row_start..target.row_stop,
                    target.col_start..target.col_stop
                ])
                .to_owned()
        });
    }
}

pub(crate) fn same_constant_owner(
    ownership: ArrayView3<'_, u32>,
    target: (usize, usize),
    reference: (usize, usize),
) -> Option<u32> {
    let shape = ownership.dim();
    if target.0 >= shape.1
        || target.1 >= shape.2
        || reference.0 >= shape.1
        || reference.1 >= shape.2
    {
        return None;
    }
    let owner = ownership[(0, target.0, target.1)];
    (owner != NO_BURST_OWNER
        && (0..shape.0).all(|date| {
            ownership[(date, target.0, target.1)] == owner
                && ownership[(date, reference.0, reference.1)] == owner
        }))
    .then_some(owner)
}

pub(crate) fn correction_order_digest(
    options: &CorrectionOptions,
    wavelength: Option<f64>,
    layers: &CorrectionLayers,
) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:corrections-before-spatial-reference:v1");
    let config = serde_json::to_vec(options).context("serializing correction configuration")?;
    digest.update((config.len() as u64).to_le_bytes());
    digest.update(config);
    digest.update(
        wavelength
            .map(f64::to_bits)
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    hash_optional_array(&mut digest, layers.ionosphere.as_ref());
    hash_optional_array(&mut digest, layers.troposphere.as_ref());
    hash_optional_array(&mut digest, layers.solid_earth_tide.as_ref());
    Ok(digest.finalize().into())
}

pub(crate) fn unwrap_branch_digest(
    method: UnwrapMethod,
    backend_config: &[u8],
    pairs: &[(usize, usize)],
    validity: ArrayView2<'_, bool>,
    connected_components: ArrayView3<'_, u32>,
    loop_qc_enabled: bool,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:production-unwrap-branch:v1");
    digest.update([unwrap_method_code(method), u8::from(loop_qc_enabled)]);
    digest.update((backend_config.len() as u64).to_le_bytes());
    digest.update(backend_config);
    for &(reference, secondary) in pairs {
        digest.update((reference as u64).to_le_bytes());
        digest.update((secondary as u64).to_le_bytes());
    }
    hash_shape(&mut digest, validity.shape());
    for &value in validity {
        digest.update([u8::from(value)]);
    }
    hash_shape(&mut digest, connected_components.shape());
    for &value in connected_components {
        digest.update(value.to_le_bytes());
    }
    digest.finalize().into()
}

fn unwrap_method_code(method: UnwrapMethod) -> u8 {
    match method {
        UnwrapMethod::Snaphu => 0,
        UnwrapMethod::Tophu => 1,
        UnwrapMethod::Icu => 2,
        UnwrapMethod::Phass => 3,
        UnwrapMethod::Spurt => 4,
        UnwrapMethod::Whirlwind => 5,
        UnwrapMethod::Native => 6,
    }
}

fn hash_optional_array(digest: &mut Sha256, values: Option<&Array3<f64>>) {
    digest.update([u8::from(values.is_some())]);
    if let Some(values) = values {
        hash_shape(digest, values.shape());
        hash_f64_values(digest, values.iter().copied());
    }
}

fn hash_shape(digest: &mut Sha256, shape: &[usize]) {
    digest.update((shape.len() as u64).to_le_bytes());
    for &value in shape {
        digest.update((value as u64).to_le_bytes());
    }
}

fn hash_grid(digest: &mut Sha256, grid: CovarianceOperatorGrid) {
    digest.update(grid.row_start.to_le_bytes());
    digest.update(grid.col_start.to_le_bytes());
    digest.update(grid.rows.to_le_bytes());
    digest.update(grid.cols.to_le_bytes());
    digest.update(grid.stride_y.to_le_bytes());
    digest.update(grid.stride_x.to_le_bytes());
}

fn hash_f64_values(digest: &mut Sha256, values: impl Iterator<Item = f64>) {
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dolphin_core::config::{CorrectionOptions, UnwrapMethod};
    use ndarray::{array, Array2, Array3};

    #[test]
    fn fixed_l2_state_preserves_differing_validity_and_precision_maps() {
        let incidence = array![[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]];
        let mut dphi =
            Array3::from_shape_vec((3, 1, 2), vec![1.0, 1.0, 2.0, 2.0, f64::NAN, 3.0]).unwrap();
        let precision =
            Array3::from_shape_vec((3, 1, 2), vec![4.0, 8.0, 5.0, 0.0, 6.0, 9.0]).unwrap();
        // Keep both pixels full-rank while making their admitted IFG sets differ.
        dphi[(2, 0, 0)] = 3.0;
        let retained = FixedL2WorkflowInputs::new(incidence, dphi, Some(precision)).unwrap();
        let target = retained.pixel_map((0, 0)).unwrap();
        let reference = retained.pixel_map((0, 1)).unwrap();
        assert_eq!(target.valid_observation_indices(), &[0, 1, 2]);
        assert_eq!(target.precisions(), &[4.0, 5.0, 6.0]);
        assert_eq!(reference.valid_observation_indices(), &[0, 2]);
        assert_eq!(reference.precisions(), &[8.0, 9.0]);
        assert_ne!(
            retained.pixel_map_digest((0, 0)).unwrap(),
            retained.pixel_map_digest((0, 1)).unwrap()
        );

        let mut joint = Array2::<f64>::zeros((6, 6));
        for index in [1, 2, 4, 5] {
            joint[(index, index)] = 1.0;
        }
        joint[(1, 4)] = 0.2;
        joint[(4, 1)] = 0.2;
        joint[(2, 5)] = 0.2;
        joint[(5, 2)] = 0.2;
        let propagated = retained
            .propagate_joint_phase_covariance((0, 0), (0, 1), joint.view())
            .unwrap();
        assert_eq!(
            propagated.date_factor.row(0),
            ndarray::ArrayView1::from(&[0.0, 0.0][..])
        );
        assert_eq!(propagated.propagation_map.dim(), (3, 6));
    }

    #[test]
    fn ownership_requires_one_owner_for_every_date_and_both_pixels() {
        let owners = Array3::from_shape_vec(
            (3, 1, 4),
            vec![
                0,
                0,
                0,
                NO_BURST_OWNER,
                0,
                0,
                1,
                NO_BURST_OWNER,
                0,
                0,
                0,
                NO_BURST_OWNER,
            ],
        )
        .unwrap();
        assert_eq!(same_constant_owner(owners.view(), (0, 0), (0, 1)), Some(0));
        assert_eq!(same_constant_owner(owners.view(), (0, 0), (0, 2)), None);
        assert_eq!(same_constant_owner(owners.view(), (0, 0), (0, 3)), None);
    }

    #[test]
    fn production_receipt_binds_effective_looks_support_reference_and_resource() {
        let base = production_target_receipt(
            [1; 32],
            [2; 32],
            [3; 32],
            "source_factor_declared_v1",
            1.5,
            17,
            0.25,
            [4; 32],
            4096,
        );
        assert_ne!(
            base,
            production_target_receipt(
                [1; 32],
                [2; 32],
                [3; 32],
                "source_factor_declared_v1",
                1.5,
                18,
                0.25,
                [4; 32],
                4096,
            )
        );
        assert_ne!(
            base,
            production_target_receipt(
                [1; 32],
                [2; 32],
                [3; 32],
                "source_factor_declared_v1",
                1.5,
                17,
                0.25,
                [4; 32],
                4097,
            )
        );
    }

    #[test]
    fn effective_look_realization_identity_binds_each_ordered_target_value() {
        let grid = CovarianceOperatorGrid {
            row_start: 4,
            col_start: 7,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 1,
        };
        let outcome = TargetFactor {
            status: SpatialReferenceCovarianceStatus::Valid,
            source_burst_index: 0,
            date_factor: Some(array![[0.0], [1.0]]),
            source_factor_receipt: [1; 32],
            effective_looks_fraction: 0.5,
            support_union_count: 9,
            effective_looks_receipt: [2; 32],
            resource_high_water_bytes: 1024,
            condition_number: 1.0,
        };
        let base = build_factor_block(0, grid, 2, 1.0, std::slice::from_ref(&outcome)).unwrap();
        let mut changed_fraction = outcome.clone();
        changed_fraction.effective_looks_fraction = 0.25;
        let changed_fraction = build_factor_block(0, grid, 2, 1.0, &[changed_fraction]).unwrap();
        let mut changed_receipt = outcome;
        changed_receipt.effective_looks_receipt = [3; 32];
        let changed_receipt = build_factor_block(0, grid, 2, 1.0, &[changed_receipt]).unwrap();
        assert_ne!(
            dolphin_io::spatial_reference_effective_looks_digest(std::slice::from_ref(&base))
                .unwrap(),
            dolphin_io::spatial_reference_effective_looks_digest(std::slice::from_ref(
                &changed_fraction,
            ))
            .unwrap()
        );
        assert_ne!(
            dolphin_io::spatial_reference_effective_looks_digest(std::slice::from_ref(&base))
                .unwrap(),
            dolphin_io::spatial_reference_effective_looks_digest(std::slice::from_ref(
                &changed_receipt,
            ))
            .unwrap()
        );
    }

    #[test]
    fn bounded_trim_preserves_global_captured_output_coordinates() {
        let retained = FixedL2WorkflowInputs::new(
            array![[-1.0, 0.0], [0.0, 1.0]],
            Array3::ones((2, 4, 5)),
            None,
        )
        .unwrap();
        let mut state = ProductionCovarianceState {
            replay_context: None,
            fixed_l2_inputs: Some(retained),
            ownership: Array3::from_elem((3, 4, 5), 0),
            seam_rotations: vec![(0, vec![dolphin_core::Cf64::new(1.0, 0.0); 3])],
            source_burst_ids: vec!["burst-a".to_owned()],
            burst_output_mappings: vec![BurstOutputMapping {
                owner: 0,
                frame_origin: (0, 0),
                output_origin: (100, 200),
                shape: (4, 5),
            }],
            analysis_origin: (0, 0),
            correction_order_digest: [1; 32],
            unwrap_branch_digest: [2; 32],
        };
        state.trim(dolphin_core::BlockIndices {
            row_start: 1,
            row_stop: 4,
            col_start: 2,
            col_stop: 5,
        });
        assert_eq!(state.owner_output_coordinate(0, (0, 0)), Some((101, 202)));
        assert_eq!(state.owner_output_coordinate(0, (2, 2)), Some((103, 204)));
    }

    #[test]
    fn correction_identity_hashes_actual_arrays_config_wavelength_and_order() {
        let options = CorrectionOptions::default();
        let layers = crate::corrections::CorrectionLayers {
            ionosphere: Some(Array3::from_elem((2, 1, 1), 0.25)),
            troposphere: None,
            los_geometry: None,
            solid_earth_tide: None,
        };
        let base = correction_order_digest(&options, Some(0.056), &layers).unwrap();
        let changed_wavelength = correction_order_digest(&options, Some(0.055), &layers).unwrap();
        let changed_arrays = crate::corrections::CorrectionLayers {
            ionosphere: Some(Array3::from_elem((2, 1, 1), 0.5)),
            ..layers
        };
        assert_ne!(base, changed_wavelength);
        assert_ne!(
            base,
            correction_order_digest(&options, Some(0.056), &changed_arrays).unwrap()
        );
    }

    #[test]
    fn unwrap_identity_hashes_backend_pairs_and_actual_validity_branch() {
        let validity = array![[true, false], [true, true]];
        let components = Array3::from_shape_vec((1, 2, 2), vec![1_u32, 0, 1, 1]).unwrap();
        let pairs = vec![(0, 1), (1, 2)];
        let native = unwrap_branch_digest(
            UnwrapMethod::Native,
            b"native-config",
            &pairs,
            validity.view(),
            components.view(),
            true,
        );
        let snaphu = unwrap_branch_digest(
            UnwrapMethod::Snaphu,
            b"native-config",
            &pairs,
            validity.view(),
            components.view(),
            true,
        );
        let changed_validity = unwrap_branch_digest(
            UnwrapMethod::Native,
            b"native-config",
            &pairs,
            Array2::from_elem((2, 2), true).view(),
            components.view(),
            true,
        );
        assert_ne!(native, snaphu);
        assert_ne!(native, changed_validity);
    }

    #[test]
    fn uncalibrated_block_pads_differing_positive_ranks() {
        let grid = CovarianceOperatorGrid {
            row_start: 7,
            col_start: 11,
            rows: 1,
            cols: 3,
            stride_y: 2,
            stride_x: 2,
        };
        let block = build_factor_block(
            4,
            grid,
            3,
            -2.0,
            &[
                TargetFactor {
                    status: SpatialReferenceCovarianceStatus::Valid,
                    source_burst_index: 0,
                    date_factor: Some(array![[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]),
                    source_factor_receipt: [1; 32],
                    effective_looks_fraction: 0.75,
                    support_union_count: 9,
                    effective_looks_receipt: [0x71; 32],
                    resource_high_water_bytes: 2048,
                    condition_number: 4.0,
                },
                TargetFactor {
                    status: SpatialReferenceCovarianceStatus::Valid,
                    source_burst_index: 0,
                    date_factor: Some(array![[0.0], [0.0], [1.0]]),
                    source_factor_receipt: [2; 32],
                    effective_looks_fraction: 0.5,
                    support_union_count: 12,
                    effective_looks_receipt: [0x72; 32],
                    resource_high_water_bytes: 3072,
                    condition_number: 1.0,
                },
                TargetFactor {
                    status: SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference,
                    source_burst_index: SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
                    date_factor: None,
                    source_factor_receipt: [0; 32],
                    effective_looks_fraction: f64::NAN,
                    support_union_count: 0,
                    effective_looks_receipt: [0; 32],
                    resource_high_water_bytes: 0,
                    condition_number: f64::NAN,
                },
            ],
        )
        .unwrap();
        assert_eq!(block.maximum_rank, 3);
        assert_eq!(block.rank_by_target, vec![2, 1, 0]);
        assert_eq!(
            block.status,
            vec![
                SpatialReferenceCovarianceStatus::Valid,
                SpatialReferenceCovarianceStatus::Valid,
                SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference,
            ]
        );
        assert!(block
            .approximation_error_bound
            .iter()
            .all(|bound| bound.is_nan()));
        assert_eq!(
            &block.difference_factor[0..9],
            &[0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -4.0, 0.0]
        );
        assert_eq!(block.difference_factor[9 + 6], -2.0);
        assert!(block.difference_factor[18..]
            .iter()
            .all(|value| *value == 0.0));
        assert!(block.source_factor_digest.starts_with("sha256:"));
    }

    #[test]
    fn block_rejects_false_valid_or_nonvalid_factors() {
        let grid = CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 1,
        };
        let false_valid = TargetFactor {
            status: SpatialReferenceCovarianceStatus::Valid,
            source_burst_index: SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
            date_factor: None,
            source_factor_receipt: [0; 32],
            effective_looks_fraction: f64::NAN,
            support_union_count: 0,
            effective_looks_receipt: [0; 32],
            resource_high_water_bytes: 0,
            condition_number: f64::NAN,
        };
        assert!(build_factor_block(0, grid, 2, 1.0, &[false_valid]).is_err());
        let false_failure = TargetFactor {
            status: SpatialReferenceCovarianceStatus::MaskedTarget,
            source_burst_index: 0,
            date_factor: Some(Array2::zeros((2, 0))),
            source_factor_receipt: [0; 32],
            effective_looks_fraction: f64::NAN,
            support_union_count: 0,
            effective_looks_receipt: [0; 32],
            resource_high_water_bytes: 0,
            condition_number: f64::NAN,
        };
        assert!(build_factor_block(0, grid, 2, 1.0, &[false_failure]).is_err());
    }

    #[test]
    fn fixed_l2_statuses_are_not_collapsed_to_replay_failure() {
        assert_eq!(
            fixed_l2_status(SpatialL2Status::RankDeficient),
            SpatialReferenceCovarianceStatus::L2RankDeficient
        );
        assert_eq!(
            fixed_l2_status(SpatialL2Status::IllConditioned),
            SpatialReferenceCovarianceStatus::IllConditioned
        );
        assert_eq!(
            fixed_l2_status(SpatialL2Status::UnsupportedL1),
            SpatialReferenceCovarianceStatus::UnsupportedL1
        );
        assert_eq!(
            replay_status(ReplayStatus::SourceIdentityMismatch),
            SpatialReferenceCovarianceStatus::ReplayMismatch
        );
        assert_eq!(
            replay_status(ReplayStatus::DependencyConeExceedsBudget),
            SpatialReferenceCovarianceStatus::ReplayUnavailable
        );
        assert_eq!(
            replay_status(ReplayStatus::NondifferentiableNode),
            SpatialReferenceCovarianceStatus::NondifferentiableEstimator
        );
    }

    #[test]
    fn frame_identities_bind_masks_l2_maps_and_relocated_reference() {
        let validity = array![[true, false]];
        let owners = Array3::from_shape_vec((2, 1, 2), vec![0, NO_BURST_OWNER, 0, 1]).unwrap();
        let mask = production_mask_digest(validity.view(), owners.view()).unwrap();
        let mut changed = owners.clone();
        changed[(1, 0, 1)] = 0;
        assert_ne!(
            mask,
            production_mask_digest(validity.view(), changed.view()).unwrap()
        );

        let inputs = FixedL2WorkflowInputs::new(
            array![[-1.0], [1.0]],
            Array3::from_shape_vec((2, 1, 2), vec![1.0, 2.0, 3.0, f64::NAN]).unwrap(),
            None,
        )
        .unwrap();
        assert_ne!(
            fixed_l2_frame_digest(Some(&inputs)).unwrap(),
            fixed_l2_frame_digest(None).unwrap()
        );
        let grid = CovarianceOperatorGrid {
            row_start: 5,
            col_start: 9,
            rows: 1,
            cols: 2,
            stride_y: 2,
            stride_x: 3,
        };
        assert_ne!(
            final_reference_signature_digest(
                grid,
                (5, 9),
                0,
                &[0, 1, 2],
                [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            ),
            final_reference_signature_digest(
                grid,
                (5, 10),
                0,
                &[0, 1, 2],
                [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            )
        );
        assert_ne!(
            final_reference_signature_digest(
                grid,
                (5, 9),
                0,
                &[0, 1, 2],
                [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            ),
            final_reference_signature_digest(
                grid,
                (5, 9),
                0,
                &[0, 1, 2],
                [10.0, 1.0, 0.0, 20.0, 0.0, -1.0],
            )
        );
    }

    #[test]
    fn factor_block_plan_is_rectangular_bounded_and_covers_narrow_caps() {
        let dates = 52;
        let per_target = (dates * dates * 8 + 82) as u64;
        let workspace = fixed_l2_difference_workspace_composition(dates).unwrap();
        let one_target_cap = workspace.total_bytes + 2 * per_target + 1;
        assert_eq!(
            factor_block_shape((4, 7), dates, one_target_cap, workspace, 1).unwrap(),
            (1, 1)
        );
        let fourteen_target_cap = workspace.total_bytes + 2 * per_target * 14 + 1;
        assert_eq!(
            factor_block_shape((4, 7), dates, fourteen_target_cap, workspace, 1).unwrap(),
            (2, 7)
        );
        assert!(factor_block_shape((4, 7), dates, one_target_cap - 1, workspace, 1).is_err());
        let admission =
            production_resource_admission(14, dates, workspace, 1, fourteen_target_cap).unwrap();
        assert_eq!(admission.factor_block_high_water_bytes, per_target * 14);
        assert_eq!(
            admission.working_set_admission_high_water_bytes,
            admission.factor_block_high_water_bytes
                + admission.serialization_high_water_bytes
                + admission.fixed_l2_workspace_admission_bytes
                + admission.replay_admission_high_water_bytes
        );
        assert!(
            production_resource_admission(14, dates, workspace, 1, fourteen_target_cap - 1)
                .is_err()
        );
    }

    #[test]
    fn production_provider_residency_never_accumulates_across_visited_tiles() {
        struct DropProbe {
            tracker: ProviderResidencyTracker,
            drops: Rc<Cell<usize>>,
        }

        impl Drop for DropProbe {
            fn drop(&mut self) {
                assert!(self.tracker.receipt().current_count > 0);
                self.drops.set(self.drops.get() + 1);
            }
        }

        let reference_tile = 17;
        let provider_reservation = 4096_u64;
        let tracker = ProviderResidencyTracker::default();
        let drops = Rc::new(Cell::new(0));
        for target_tile in 0..10_000 {
            let plan = ActiveProviderPlan::new(target_tile, reference_tile);
            let target = tracker
                .track(
                    DropProbe {
                        tracker: tracker.clone(),
                        drops: Rc::clone(&drops),
                    },
                    provider_reservation,
                )
                .unwrap();
            let reference = plan
                .reference_tile
                .map(|_| {
                    tracker.track(
                        DropProbe {
                            tracker: tracker.clone(),
                            drops: Rc::clone(&drops),
                        },
                        provider_reservation,
                    )
                })
                .transpose()
                .unwrap();
            assert_eq!(tracker.receipt().current_count, plan.provider_count());
            drop(reference);
            drop(target);
            assert_eq!(tracker.receipt().current_count, 0);
            assert_eq!(tracker.receipt().current_bytes, 0);
        }
        let receipt = tracker.receipt();
        assert_eq!(receipt.peak_count, 2);
        assert_eq!(receipt.peak_bytes, 2 * provider_reservation);
        assert_eq!(drops.get(), 19_999);
    }
}
