//! Production identities and fixed-L2 state for reference-specific covariance output.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use dolphin_core::config::{CorrectionOptions, UnwrapMethod};
use dolphin_io::{
    CovarianceOperatorGrid, SpatialReferenceCovarianceBlock, SpatialReferenceCovarianceStatus,
    SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE, SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
};
use dolphin_timeseries::inversion::{fixed_l2_pixel_map, PixelL2MapError, PixelL2ObservationMap};
use dolphin_timeseries::spatial_covariance::{
    propagate_fixed_l2_difference_covariance, FixedL2DifferenceCovariance, SpatialL2Branch,
    SpatialL2Error, SpatialL2Status,
};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use sha2::{Digest, Sha256};

use crate::corrections::CorrectionLayers;
use crate::covariance_artifact::CovarianceArtifactManifest;
use crate::cslc_covariance_source::CslcCovarianceManifest;
use crate::displacement::PreparedBurstMask;
use crate::sequential_covariance::ReplayStatus;

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

pub(crate) struct TargetFactor {
    pub(crate) status: SpatialReferenceCovarianceStatus,
    pub(crate) source_burst_index: u32,
    pub(crate) date_factor: Option<Array2<f64>>,
    pub(crate) source_factor_receipt: [u8; 32],
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
    digest.finalize().into()
}

pub(crate) fn build_factor_block(
    block_id: u64,
    target_grid: CovarianceOperatorGrid,
    dates: usize,
    phase_to_displacement: f64,
    outcomes: &[TargetFactor],
) -> Result<SpatialReferenceCovarianceBlock> {
    let targets = usize::try_from(target_grid.rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(target_grid.cols)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .context("spatial covariance target grid area exceeds usize")?;
    anyhow::ensure!(
        dates > 0 && targets == outcomes.len() && phase_to_displacement.is_finite(),
        "spatial covariance block inputs disagree"
    );
    let maximum_rank = dates;
    let factor_len = targets
        .checked_mul(dates)
        .and_then(|value| value.checked_mul(maximum_rank))
        .context("spatial covariance factor block dimensions overflow")?;
    let mut difference_factor = vec![0.0; factor_len];
    let mut rank_by_target = Vec::with_capacity(targets);
    let mut status = Vec::with_capacity(targets);
    let mut source_burst_index_by_target = Vec::with_capacity(targets);
    let mut receipt = Sha256::new();
    receipt.update(b"dolphinrust:production-source-factor-block:v1");
    receipt.update(block_id.to_le_bytes());
    hash_grid(&mut receipt, target_grid);
    for (target, outcome) in outcomes.iter().enumerate() {
        let rank = match (&outcome.date_factor, outcome.status) {
            (Some(factor), SpatialReferenceCovarianceStatus::Valid) => {
                anyhow::ensure!(
                    factor.nrows() == dates
                        && factor.ncols() <= maximum_rank
                        && factor.iter().all(|value| value.is_finite()),
                    "valid spatial covariance target factor is malformed"
                );
                for date in 0..dates {
                    let start = (target * dates + date) * maximum_rank;
                    for component in 0..factor.ncols() {
                        difference_factor[start + component] =
                            phase_to_displacement * factor[(date, component)];
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
        rank_by_target.push(rank);
        status.push(outcome.status);
        source_burst_index_by_target.push(outcome.source_burst_index);
        receipt.update((target as u64).to_le_bytes());
        receipt.update((outcome.status as u16).to_le_bytes());
        receipt.update(outcome.source_burst_index.to_le_bytes());
        receipt.update(rank.to_le_bytes());
        receipt.update(outcome.source_factor_receipt);
    }
    Ok(SpatialReferenceCovarianceBlock {
        block_id,
        target_grid,
        maximum_rank: u32::try_from(maximum_rank).context("maximum factor rank exceeds u32")?,
        rank_by_target,
        status,
        source_burst_index_by_target,
        difference_factor,
        approximation_error_bound: vec![SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE; targets],
        source_factor_digest: format!("sha256:{:x}", receipt.finalize()),
    })
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
    fn uncalibrated_block_pads_differing_ranks_and_preserves_exact_zero_factor() {
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
                    date_factor: Some(array![[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]),
                    source_factor_receipt: [1; 32],
                },
                TargetFactor {
                    status: SpatialReferenceCovarianceStatus::Valid,
                    source_burst_index: 0,
                    date_factor: Some(Array2::zeros((3, 0))),
                    source_factor_receipt: [2; 32],
                },
                TargetFactor {
                    status: SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference,
                    source_burst_index: SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
                    date_factor: None,
                    source_factor_receipt: [0; 32],
                },
            ],
        )
        .unwrap();
        assert_eq!(block.maximum_rank, 3);
        assert_eq!(block.rank_by_target, vec![2, 0, 0]);
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
            &[0.0, 0.0, 0.0, -2.0, -4.0, 0.0, -6.0, -8.0, 0.0]
        );
        assert!(block.difference_factor[9..]
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
        };
        assert!(build_factor_block(0, grid, 2, 1.0, &[false_valid]).is_err());
        let false_failure = TargetFactor {
            status: SpatialReferenceCovarianceStatus::MaskedTarget,
            source_burst_index: 0,
            date_factor: Some(Array2::zeros((2, 0))),
            source_factor_receipt: [0; 32],
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
            final_reference_signature_digest(grid, (5, 9), 0, &[0, 1, 2]),
            final_reference_signature_digest(grid, (5, 10), 0, &[0, 1, 2])
        );
    }
}
