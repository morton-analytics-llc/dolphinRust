//! Fixed-valid-observation L2 covariance for displacement differences.
//!
//! This module is deliberately separate from the legacy independent-IFG
//! uncertainty approximation in [`crate::inversion`].  The caller supplies the
//! exact valid observation set and its joint covariance; the returned `H` map
//! therefore preserves shared-source covariance through the date inversion.

use faer::{Mat, Side};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::inversion::PixelL2ObservationMap;

/// Stable method identity for the fixed-valid-observation L2 map.
pub const FIXED_L2_SPATIAL_COVARIANCE_METHOD: &str =
    "fixed_valid_observation_l2_spatial_covariance_v1";
/// Maximum retained-spectrum condition number accepted for phase or date covariance.
pub const FIXED_L2_MAX_COVARIANCE_CONDITION_NUMBER: f64 = 1.0e12;
const RANK_TOLERANCE: f64 = 1.0e-10;

/// Estimator branch requested by a spatial covariance query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialL2Branch {
    /// Supported fixed-weight L2 branch.
    FixedL2,
    /// Dolphin's robust LAD branch; not represented by this covariance map.
    L1,
    /// Any branch whose unwrap/estimator decisions differ from the captured run.
    ChangedBranch,
}

/// Stable fail-closed status for the fixed L2 covariance contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpatialL2Status {
    /// The fixed-valid-observation L2 map is defined.
    Valid = 0,
    /// L1/ADMM has no fixed linear E/H map in this contract.
    UnsupportedL1 = 1,
    /// Changed estimator or unwrap branches invalidate the replay.
    UnsupportedChangedBranch = 2,
    /// A supplied matrix, vector, or unit scale is invalid.
    InvalidInput = 3,
    /// The design or observation covariance has insufficient numerical rank.
    RankDeficient = 4,
    /// A non-finite intermediate or result was encountered.
    NonFinite = 5,
    /// A fixed production map exceeds the supported condition bound.
    IllConditioned = 6,
}

impl SpatialL2Status {
    /// Stable serialized status name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::UnsupportedL1 => "unsupported_l1",
            Self::UnsupportedChangedBranch => "unsupported_changed_branch",
            Self::InvalidInput => "invalid_input",
            Self::RankDeficient => "rank_deficient",
            Self::NonFinite => "non_finite",
            Self::IllConditioned => "ill_conditioned",
        }
    }
}

/// Date-space covariance/factor for target-minus-reference displacement.
#[derive(Debug, Clone)]
pub struct FixedL2DifferenceCovariance {
    /// Stable method identity.
    pub method: &'static str,
    /// Successful fixed-L2 propagation status.
    pub status: SpatialL2Status,
    /// Exact `E H B` map from joint target/reference phase to date differences.
    pub propagation_map: Array2<f64>,
    /// Date-space target-minus-reference covariance with an exact zero gauge.
    pub date_covariance: Array2<f64>,
    /// Deterministic rank-revealing factor of [`Self::date_covariance`].
    pub date_factor: Array2<f64>,
    /// Numerical rank of the supplied joint phase covariance.
    pub phase_covariance_rank: usize,
    /// Numerical rank of the propagated date covariance.
    pub covariance_rank: usize,
    /// Condition number of the retained joint phase covariance spectrum.
    pub phase_covariance_condition_number: f64,
    /// Condition number of the retained propagated covariance spectrum.
    pub covariance_condition_number: f64,
}

/// Return the stable disposition for a requested estimator branch.
#[must_use]
pub const fn spatial_l2_branch_status(branch: SpatialL2Branch) -> SpatialL2Status {
    match branch {
        SpatialL2Branch::FixedL2 => SpatialL2Status::Valid,
        SpatialL2Branch::L1 => SpatialL2Status::UnsupportedL1,
        SpatialL2Branch::ChangedBranch => SpatialL2Status::UnsupportedChangedBranch,
    }
}

/// Error returned when a fixed L2 covariance map cannot be constructed.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialL2Error {
    /// Stable fail-closed disposition.
    pub status: SpatialL2Status,
    /// Short diagnostic suitable for a validation receipt.
    pub message: &'static str,
}

impl std::fmt::Display for SpatialL2Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.status.as_str(), self.message)
    }
}

impl std::error::Error for SpatialL2Error {}

/// Fixed-valid-observation L2 estimate and its spatial covariance maps.
#[derive(Debug, Clone)]
pub struct SpatialL2Covariance {
    /// Stable method identity.
    pub method: &'static str,
    /// Successful status; unsuccessful construction returns [`SpatialL2Error`].
    pub status: SpatialL2Status,
    /// Estimated non-gauge date parameters.
    pub parameters: Array1<f64>,
    /// E map selecting target and reference displacement parameters.
    pub e_map: Array2<f64>,
    /// H map from fixed valid observations to non-gauge parameters.
    pub h_map: Array2<f64>,
    /// Parameter covariance after the joint observation covariance is applied.
    pub parameter_covariance: Array2<f64>,
    /// Joint target/reference covariance in the requested units.
    pub target_reference_covariance: Array2<f64>,
    /// Variance of target minus reference displacement.
    pub difference_covariance: f64,
    /// Numerical rank of the whitened design.
    pub design_rank: usize,
    /// Numerical rank of the supplied observation covariance.
    pub observation_rank: usize,
    /// Log pseudo-determinant of the observation covariance.
    pub observation_log_pseudodeterminant: f64,
    /// Log pseudo-determinant of the whitened normal matrix.
    pub normal_log_pseudodeterminant: f64,
}

impl SpatialL2Covariance {
    /// Return the parameter-covariance diagonal without allocating a dense block.
    #[must_use]
    pub fn covariance_diagonal(&self) -> Array1<f64> {
        Array1::from_iter(
            (0..self.parameter_covariance.nrows())
                .map(|index| self.parameter_covariance[(index, index)]),
        )
    }

    /// Select a covariance block in deterministic caller-supplied parameter order.
    pub fn covariance_block(&self, indices: &[usize]) -> Result<Array2<f64>, SpatialL2Error> {
        if indices
            .iter()
            .any(|&index| index >= self.parameter_covariance.nrows())
        {
            return Err(error(
                SpatialL2Status::InvalidInput,
                "selected covariance index is out of range",
            ));
        }
        Ok(Array2::from_shape_fn(
            (indices.len(), indices.len()),
            |(row, column)| self.parameter_covariance[(indices[row], indices[column])],
        ))
    }
}

/// Fixed-L2 covariance retained as bounded source and target/reference factors.
///
/// `parameter_factor` is `H L`, where `L` is the persisted observation/source
/// factor and `C = L Lᵀ`.  The covariance fields are reconstructed only by
/// factor congruence, so persistence does not require a dense observation or
/// parameter covariance matrix.
#[derive(Debug, Clone)]
pub struct SpatialL2FactorCovariance {
    /// Stable method identity.
    pub method: &'static str,
    /// Successful status.
    pub status: SpatialL2Status,
    /// Estimated non-gauge date parameters.
    pub parameters: Array1<f64>,
    /// E map selecting target and reference displacement parameters.
    pub e_map: Array2<f64>,
    /// H map from fixed valid observations to non-gauge parameters.
    pub h_map: Array2<f64>,
    /// Bounded factor for non-gauge parameter covariance.
    pub parameter_factor: Array2<f64>,
    /// Bounded factor for the two target/reference parameters.
    pub target_reference_factor: Array2<f64>,
    /// Bounded factor for target minus reference displacement.
    pub difference_factor: Array2<f64>,
    /// Parameter covariance reconstructed as `parameter_factor * factorᵀ`.
    pub parameter_covariance: Array2<f64>,
    /// Target/reference covariance reconstructed by factor congruence.
    pub target_reference_covariance: Array2<f64>,
    /// Variance of target minus reference displacement.
    pub difference_covariance: f64,
    /// Numerical rank of the whitened design.
    pub design_rank: usize,
    /// Numerical rank of the supplied observation/source factor covariance.
    pub observation_rank: usize,
    /// Log pseudo-determinant of the observation covariance.
    pub observation_log_pseudodeterminant: f64,
    /// Log pseudo-determinant of the whitened normal matrix.
    pub normal_log_pseudodeterminant: f64,
}

impl SpatialL2FactorCovariance {
    /// Return the parameter-covariance diagonal from the retained factor.
    #[must_use]
    pub fn covariance_diagonal(&self) -> Array1<f64> {
        Array1::from_iter((0..self.parameter_factor.nrows()).map(|row| {
            self.parameter_factor
                .row(row)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
        }))
    }

    /// Select a parameter covariance block without materializing the full matrix.
    pub fn covariance_block(&self, indices: &[usize]) -> Result<Array2<f64>, SpatialL2Error> {
        if indices
            .iter()
            .any(|&index| index >= self.parameter_factor.nrows())
        {
            return Err(error(
                SpatialL2Status::InvalidInput,
                "selected covariance index is out of range",
            ));
        }
        Ok(Array2::from_shape_fn(
            (indices.len(), indices.len()),
            |(row, column)| {
                self.parameter_factor
                    .row(indices[row])
                    .dot(&self.parameter_factor.row(indices[column]))
            },
        ))
    }
}

/// Build the exact non-gauge contrast for one date, where date zero is gauge.
///
/// `date` is an index in a full date series and `n_dates` includes the gauge
/// date.  The returned vector has length `n_dates - 1`.
pub fn date_contrast(date: usize, n_dates: usize) -> Result<Array1<f64>, SpatialL2Error> {
    if n_dates < 2 || date >= n_dates {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "date contrast is out of range",
        ));
    }
    let mut result = Array1::zeros(n_dates - 1);
    if date > 0 {
        result[date - 1] = 1.0;
    }
    Ok(result)
}

/// Convert a covariance matrix by an exact scalar unit conversion.
pub fn convert_covariance_units(
    covariance: ArrayView2<f64>,
    unit_scale: f64,
) -> Result<Array2<f64>, SpatialL2Error> {
    if !unit_scale.is_finite() || unit_scale == 0.0 || covariance.nrows() != covariance.ncols() {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "invalid covariance unit conversion",
        ));
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(error(
            SpatialL2Status::NonFinite,
            "covariance contains a non-finite value",
        ));
    }
    Ok(covariance.mapv(|value| value * unit_scale * unit_scale))
}

/// Propagate a joint target/reference phase covariance through two fixed
/// production L2 maps without recomputing their estimator weights.
pub fn propagate_fixed_l2_difference_covariance(
    target: &PixelL2ObservationMap,
    reference: &PixelL2ObservationMap,
    joint_phase_covariance: ArrayView2<f64>,
    branch: SpatialL2Branch,
) -> Result<FixedL2DifferenceCovariance, SpatialL2Error> {
    let branch_status = spatial_l2_branch_status(branch);
    if branch_status != SpatialL2Status::Valid {
        return Err(error(
            branch_status,
            "requested branch has no fixed production L2 map",
        ));
    }
    let n_dates = target.date_count();
    let joint_dates = n_dates.checked_mul(2).ok_or_else(|| {
        error(
            SpatialL2Status::InvalidInput,
            "joint phase dimension overflows usize",
        )
    })?;
    if reference.date_count() != n_dates
        || joint_phase_covariance.dim() != (joint_dates, joint_dates)
    {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "production L2 maps and joint phase covariance dimensions disagree",
        ));
    }
    if !target.condition_number().is_finite() || !reference.condition_number().is_finite() {
        return Err(error(
            SpatialL2Status::IllConditioned,
            "production L2 map condition receipt is invalid",
        ));
    }
    let phase_factor = rank_revealing_psd_factor(joint_phase_covariance, true)?;
    let target_map = full_date_l2_map(target);
    let reference_map = full_date_l2_map(reference);
    let propagation_map =
        Array2::from_shape_fn((n_dates, joint_dates), |(date, phase)| {
            match phase < n_dates {
                true => target_map[(date, phase)],
                false => -reference_map[(date, phase - n_dates)],
            }
        });
    let mut date_covariance =
        propagation_map.dot(&joint_phase_covariance.dot(&propagation_map.t()));
    for row in 0..n_dates {
        for column in row + 1..n_dates {
            let symmetric = 0.5 * (date_covariance[(row, column)] + date_covariance[(column, row)]);
            date_covariance[(row, column)] = symmetric;
            date_covariance[(column, row)] = symmetric;
        }
    }
    let phase_scale = joint_phase_covariance
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let propagation_scale = propagation_map
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let date_scale = date_covariance
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let cancellation_tolerance = phase_scale
        * propagation_scale
        * propagation_scale
        * f64::EPSILON
        * joint_dates as f64
        * 32.0;
    if date_scale <= cancellation_tolerance {
        date_covariance.fill(0.0);
    }
    for index in 0..n_dates {
        date_covariance[(0, index)] = 0.0;
        date_covariance[(index, 0)] = 0.0;
    }
    let covariance_factor = rank_revealing_psd_factor(date_covariance.view(), true)?;
    Ok(FixedL2DifferenceCovariance {
        method: FIXED_L2_SPATIAL_COVARIANCE_METHOD,
        status: SpatialL2Status::Valid,
        propagation_map,
        date_covariance,
        date_factor: covariance_factor.factor,
        phase_covariance_rank: phase_factor.rank,
        covariance_rank: covariance_factor.rank,
        phase_covariance_condition_number: phase_factor.condition_number,
        covariance_condition_number: covariance_factor.condition_number,
    })
}

fn full_date_l2_map(map: &PixelL2ObservationMap) -> Array2<f64> {
    let n_dates = map.date_count();
    let compact = map.h_map().dot(&map.observation_phase_map());
    Array2::from_shape_fn((n_dates, n_dates), |(row, column)| match row {
        0 => 0.0,
        _ => compact[(row - 1, column)],
    })
}

/// Solve a fixed-valid-observation weighted L2 inversion and contract a target/reference E map.
pub fn solve_fixed_l2_spatial_covariance(
    design: ArrayView2<f64>,
    observations: ArrayView1<f64>,
    observation_covariance: ArrayView2<f64>,
    target_contrast: ArrayView1<f64>,
    reference_contrast: ArrayView1<f64>,
) -> Result<SpatialL2Covariance, SpatialL2Error> {
    let (n_observations, n_parameters) = design.dim();
    if n_observations == 0
        || n_parameters == 0
        || observations.len() != n_observations
        || observation_covariance.dim() != (n_observations, n_observations)
        || target_contrast.len() != n_parameters
        || reference_contrast.len() != n_parameters
    {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "fixed L2 dimensions do not agree",
        ));
    }
    if design.iter().any(|value| !value.is_finite())
        || observations.iter().any(|value| !value.is_finite())
        || observation_covariance
            .iter()
            .any(|value| !value.is_finite())
        || target_contrast.iter().any(|value| !value.is_finite())
        || reference_contrast.iter().any(|value| !value.is_finite())
    {
        return Err(error(
            SpatialL2Status::NonFinite,
            "fixed L2 input contains a non-finite value",
        ));
    }
    let covariance_factor = spectral_factor(observation_covariance)?;
    let whitened_design = matmul(covariance_factor.inverse_sqrt.view(), design);
    let normal = whitened_design.t().dot(&whitened_design);
    let normal_factor = spectral_factor(normal.view())?;
    if normal_factor.rank < n_parameters {
        return Err(error(
            SpatialL2Status::RankDeficient,
            "fixed L2 design is rank deficient",
        ));
    }
    let h_map = normal_factor
        .pseudo_inverse
        .dot(&whitened_design.t().dot(&covariance_factor.inverse_sqrt));
    let parameters = h_map.dot(&observations);
    let parameter_covariance = h_map.dot(&observation_covariance.dot(&h_map.t()));
    let e_map = Array2::from_shape_fn((2, n_parameters), |(row, column)| {
        if row == 0 {
            target_contrast[column]
        } else {
            reference_contrast[column]
        }
    });
    let target_reference_covariance = e_map.dot(&parameter_covariance.dot(&e_map.t()));
    let difference = &target_contrast - &reference_contrast;
    let difference_covariance = difference.dot(&parameter_covariance.dot(&difference));
    if parameters
        .iter()
        .chain(parameter_covariance.iter())
        .chain(target_reference_covariance.iter())
        .any(|value| !value.is_finite())
        || !difference_covariance.is_finite()
    {
        return Err(error(
            SpatialL2Status::NonFinite,
            "fixed L2 result is non-finite",
        ));
    }
    Ok(SpatialL2Covariance {
        method: FIXED_L2_SPATIAL_COVARIANCE_METHOD,
        status: SpatialL2Status::Valid,
        parameters,
        e_map,
        h_map,
        parameter_covariance,
        target_reference_covariance,
        difference_covariance,
        design_rank: normal_factor.rank,
        observation_rank: covariance_factor.rank,
        observation_log_pseudodeterminant: covariance_factor.log_pseudodeterminant,
        normal_log_pseudodeterminant: normal_factor.log_pseudodeterminant,
    })
}

/// Solve fixed-valid-observation L2 while retaining the bounded source factor.
///
/// This is the persistence form of [`solve_fixed_l2_spatial_covariance`]. The
/// source factor has one row per fixed valid observation and may be rectangular
/// (`m × rank`); its covariance is never replaced by an independence model.
pub fn solve_fixed_l2_spatial_covariance_from_factor(
    design: ArrayView2<f64>,
    observations: ArrayView1<f64>,
    source_factor: ArrayView2<f64>,
    target_contrast: ArrayView1<f64>,
    reference_contrast: ArrayView1<f64>,
) -> Result<SpatialL2FactorCovariance, SpatialL2Error> {
    if source_factor.nrows() != design.nrows() || source_factor.ncols() == 0 {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "source factor dimensions do not match fixed observations",
        ));
    }
    if source_factor.iter().any(|value| !value.is_finite()) {
        return Err(error(
            SpatialL2Status::NonFinite,
            "source factor contains a non-finite value",
        ));
    }
    let observation_covariance = source_factor.dot(&source_factor.t());
    let base = solve_fixed_l2_spatial_covariance(
        design,
        observations,
        observation_covariance.view(),
        target_contrast,
        reference_contrast,
    )?;
    let parameter_factor = base.h_map.dot(&source_factor);
    let target_reference_factor = base.e_map.dot(&parameter_factor);
    let difference = &target_contrast - &reference_contrast;
    let difference_factor = difference
        .insert_axis(ndarray::Axis(0))
        .dot(&parameter_factor);
    let parameter_covariance = parameter_factor.dot(&parameter_factor.t());
    let target_reference_covariance = target_reference_factor.dot(&target_reference_factor.t());
    let difference_covariance = difference_factor
        .row(0)
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    if parameter_factor
        .iter()
        .chain(target_reference_factor.iter())
        .chain(difference_factor.iter())
        .any(|value| !value.is_finite())
        || !difference_covariance.is_finite()
    {
        return Err(error(
            SpatialL2Status::NonFinite,
            "factor-congruent fixed L2 result is non-finite",
        ));
    }
    Ok(SpatialL2FactorCovariance {
        method: base.method,
        status: base.status,
        parameters: base.parameters,
        e_map: base.e_map,
        h_map: base.h_map,
        parameter_factor,
        target_reference_factor,
        difference_factor,
        parameter_covariance,
        target_reference_covariance,
        difference_covariance,
        design_rank: base.design_rank,
        observation_rank: base.observation_rank,
        observation_log_pseudodeterminant: base.observation_log_pseudodeterminant,
        normal_log_pseudodeterminant: base.normal_log_pseudodeterminant,
    })
}

struct SpectralFactor {
    inverse_sqrt: Array2<f64>,
    pseudo_inverse: Array2<f64>,
    rank: usize,
    log_pseudodeterminant: f64,
}

struct RankRevealingFactor {
    factor: Array2<f64>,
    rank: usize,
    condition_number: f64,
}

fn rank_revealing_psd_factor(
    matrix: ArrayView2<f64>,
    allow_zero_rank: bool,
) -> Result<RankRevealingFactor, SpatialL2Error> {
    if matrix.nrows() != matrix.ncols() || matrix.nrows() == 0 {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "rank-revealing covariance factor is not square",
        ));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(error(
            SpatialL2Status::NonFinite,
            "rank-revealing covariance input is non-finite",
        ));
    }
    let size = matrix.nrows();
    let scale = matrix.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if scale == 0.0 && allow_zero_rank {
        return Ok(RankRevealingFactor {
            factor: Array2::zeros((size, 0)),
            rank: 0,
            condition_number: 1.0,
        });
    }
    let psd_tolerance = scale * RANK_TOLERANCE;
    let rank_tolerance = scale * f64::EPSILON * size as f64 * 16.0;
    for row in 0..size {
        for column in row + 1..size {
            if (matrix[(row, column)] - matrix[(column, row)]).abs() > psd_tolerance {
                return Err(error(
                    SpatialL2Status::InvalidInput,
                    "covariance is not symmetric",
                ));
            }
        }
    }
    let symmetric = Mat::from_fn(size, size, |row, column| {
        0.5 * (matrix[(row, column)] + matrix[(column, row)])
    });
    let eigen = symmetric.selfadjoint_eigendecomposition(Side::Lower);
    let values = (0..size)
        .map(|index| eigen.s().column_vector()[index])
        .collect::<Vec<_>>();
    if values.iter().any(|&value| value < -psd_tolerance) {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "covariance is not positive semidefinite",
        ));
    }
    let mut retained = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value > rank_tolerance)
        .collect::<Vec<_>>();
    retained.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    if retained.is_empty() {
        if allow_zero_rank {
            return Ok(RankRevealingFactor {
                factor: Array2::zeros((size, 0)),
                rank: 0,
                condition_number: 1.0,
            });
        }
        return Err(error(
            SpatialL2Status::RankDeficient,
            "covariance factor has zero rank",
        ));
    }
    let condition_number = retained[0].1 / retained[retained.len() - 1].1;
    if !condition_number.is_finite() || condition_number > FIXED_L2_MAX_COVARIANCE_CONDITION_NUMBER
    {
        return Err(error(
            SpatialL2Status::IllConditioned,
            "covariance retained spectrum exceeds the supported condition bound",
        ));
    }
    let factor = deterministic_pivoted_factor(matrix, rank_tolerance)?;
    if factor.ncols() != retained.len() {
        return Err(error(
            SpatialL2Status::RankDeficient,
            "spectral and deterministic covariance ranks disagree",
        ));
    }
    Ok(RankRevealingFactor {
        factor,
        rank: retained.len(),
        condition_number,
    })
}

fn deterministic_pivoted_factor(
    matrix: ArrayView2<f64>,
    tolerance: f64,
) -> Result<Array2<f64>, SpatialL2Error> {
    let size = matrix.nrows();
    let mut factor = Array2::zeros((size, size));
    let mut residual_diagonal = (0..size)
        .map(|index| matrix[(index, index)])
        .collect::<Vec<_>>();
    let mut selected = vec![false; size];
    let mut rank = 0;
    while rank < size {
        let pivot = (0..size)
            .filter(|&index| !selected[index])
            .fold(None, |best, index| match best {
                None => Some(index),
                Some(current)
                    if residual_diagonal[index] > residual_diagonal[current]
                        || (residual_diagonal[index] == residual_diagonal[current]
                            && index < current) =>
                {
                    Some(index)
                }
                Some(current) => Some(current),
            })
            .ok_or_else(|| {
                error(
                    SpatialL2Status::RankDeficient,
                    "deterministic covariance pivot is unavailable",
                )
            })?;
        let pivot_value = residual_diagonal[pivot];
        if pivot_value <= tolerance {
            break;
        }
        if !pivot_value.is_finite() {
            return Err(error(
                SpatialL2Status::NonFinite,
                "deterministic covariance pivot is non-finite",
            ));
        }
        factor[(pivot, rank)] = pivot_value.sqrt();
        for row in 0..size {
            if selected[row] || row == pivot {
                continue;
            }
            let previous = (0..rank)
                .map(|column| factor[(row, column)] * factor[(pivot, column)])
                .sum::<f64>();
            let symmetric = 0.5 * (matrix[(row, pivot)] + matrix[(pivot, row)]);
            let value = (symmetric - previous) / factor[(pivot, rank)];
            factor[(row, rank)] = value;
            residual_diagonal[row] -= value * value;
            if residual_diagonal[row] < -tolerance {
                return Err(error(
                    SpatialL2Status::InvalidInput,
                    "covariance has a negative pivot residual",
                ));
            }
        }
        selected[pivot] = true;
        residual_diagonal[pivot] = 0.0;
        rank += 1;
    }
    Ok(factor.slice(ndarray::s![.., ..rank]).to_owned())
}

fn spectral_factor(matrix: ArrayView2<f64>) -> Result<SpectralFactor, SpatialL2Error> {
    if matrix.nrows() != matrix.ncols() || matrix.nrows() == 0 {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "spectral factor is not square",
        ));
    }
    let size = matrix.nrows();
    let symmetric = Mat::from_fn(size, size, |row, column| matrix[(row, column)]);
    let eig = symmetric.selfadjoint_eigendecomposition(Side::Lower);
    let values = (0..size)
        .map(|index| eig.s().column_vector()[index])
        .collect::<Vec<_>>();
    let scale = values.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return Err(error(
            SpatialL2Status::RankDeficient,
            "spectral factor has zero scale",
        ));
    }
    let tolerance = scale * RANK_TOLERANCE;
    if values.iter().any(|&value| value < -tolerance) {
        return Err(error(
            SpatialL2Status::InvalidInput,
            "covariance is not positive semidefinite",
        ));
    }
    let vectors = eig.u();
    let mut inverse_sqrt = Array2::zeros((size, size));
    let mut pseudo_inverse = Array2::zeros((size, size));
    let mut rank = 0;
    let mut log_pseudodeterminant = 0.0;
    for (component, &value) in values.iter().enumerate() {
        if value <= tolerance {
            continue;
        }
        rank += 1;
        log_pseudodeterminant += value.ln();
        for row in 0..size {
            for column in 0..size {
                let basis = vectors[(row, component)] * vectors[(column, component)];
                inverse_sqrt[(row, column)] += basis / value.sqrt();
                pseudo_inverse[(row, column)] += basis / value;
            }
        }
    }
    if rank == 0 {
        return Err(error(
            SpatialL2Status::RankDeficient,
            "spectral factor has zero rank",
        ));
    }
    Ok(SpectralFactor {
        inverse_sqrt,
        pseudo_inverse,
        rank,
        log_pseudodeterminant,
    })
}

fn matmul(left: ArrayView2<f64>, right: ArrayView2<f64>) -> Array2<f64> {
    left.dot(&right)
}

const fn error(status: SpatialL2Status, message: &'static str) -> SpatialL2Error {
    SpatialL2Error { status, message }
}

#[cfg(test)]
mod production_l2_propagation_contract {
    use super::*;
    use crate::inversion::fixed_l2_pixel_map;
    use ndarray::{array, s, Array2, Array3};

    fn maps() -> (
        crate::inversion::PixelL2ObservationMap,
        crate::inversion::PixelL2ObservationMap,
    ) {
        let design = array![[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]];
        let target = Array3::from_shape_vec((3, 1, 1), vec![0.2, 0.3, 0.5]).unwrap();
        let reference = Array3::from_shape_vec((3, 1, 1), vec![0.1, f64::NAN, 0.4]).unwrap();
        (
            fixed_l2_pixel_map(design.view(), target.view(), None, (0, 0), 3).unwrap(),
            fixed_l2_pixel_map(design.view(), reference.view(), None, (0, 0), 3).unwrap(),
        )
    }

    #[test]
    fn differing_valid_sets_match_direct_e_h_b_c_congruence() {
        let (target, reference) = maps();
        assert_eq!(target.valid_observation_indices(), &[0, 1, 2]);
        assert_eq!(reference.valid_observation_indices(), &[0, 2]);
        let covariance = Array2::from_shape_fn((6, 6), |(row, column)| match row == column {
            true => 2.0 + row as f64 * 0.1,
            false => 0.03 * (1 + row.min(column)) as f64,
        });
        let result = propagate_fixed_l2_difference_covariance(
            &target,
            &reference,
            covariance.view(),
            SpatialL2Branch::FixedL2,
        )
        .unwrap();
        let repeated = propagate_fixed_l2_difference_covariance(
            &target,
            &reference,
            covariance.view(),
            SpatialL2Branch::FixedL2,
        )
        .unwrap();
        assert_eq!(result.date_factor, repeated.date_factor);

        let mut b_map = Array2::zeros((5, 6));
        b_map
            .slice_mut(s![0..3, 0..3])
            .assign(&target.observation_phase_map());
        b_map
            .slice_mut(s![3..5, 3..6])
            .assign(&reference.observation_phase_map());
        let mut h_map = Array2::zeros((4, 5));
        h_map.slice_mut(s![0..2, 0..3]).assign(&target.h_map());
        h_map.slice_mut(s![2..4, 3..5]).assign(&reference.h_map());
        let e_map = array![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0]
        ];
        let direct_map = e_map.dot(&h_map.dot(&b_map));
        let direct = direct_map.dot(&covariance.dot(&direct_map.t()));
        assert!(result
            .propagation_map
            .iter()
            .zip(direct_map.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
        assert!(result
            .date_covariance
            .iter()
            .zip(direct.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
        assert!(result
            .date_covariance
            .row(0)
            .iter()
            .all(|value| *value == 0.0));
        assert!(result.date_factor.row(0).iter().all(|value| *value == 0.0));
        let reconstructed = result.date_factor.dot(&result.date_factor.t());
        assert!(reconstructed
            .iter()
            .zip(result.date_covariance.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
    }

    #[test]
    fn coincident_phase_sources_cancel_exactly_and_invalid_inputs_fail_closed() {
        let (target, _) = maps();
        let base = array![[1.0, 0.2, 0.1], [0.2, 1.2, 0.3], [0.1, 0.3, 0.9]];
        let joint = Array2::from_shape_fn((6, 6), |(row, column)| base[(row % 3, column % 3)]);
        let coincident = propagate_fixed_l2_difference_covariance(
            &target,
            &target,
            joint.view(),
            SpatialL2Branch::FixedL2,
        )
        .unwrap();
        assert_eq!(coincident.covariance_rank, 0);
        assert_eq!(coincident.date_factor.dim(), (3, 0));
        assert!(coincident.date_covariance.iter().all(|value| *value == 0.0));

        let error = propagate_fixed_l2_difference_covariance(
            &target,
            &target,
            joint.view(),
            SpatialL2Branch::L1,
        )
        .unwrap_err();
        assert_eq!(error.status, SpatialL2Status::UnsupportedL1);
        let non_psd = Array2::from_diag(&array![1.0, 1.0, 1.0, 1.0, 1.0, -1.0]);
        let error = propagate_fixed_l2_difference_covariance(
            &target,
            &target,
            non_psd.view(),
            SpatialL2Branch::FixedL2,
        )
        .unwrap_err();
        assert_eq!(error.status, SpatialL2Status::InvalidInput);
    }

    #[test]
    fn ill_conditioned_joint_phase_covariance_fails_closed() {
        let (target, reference) = maps();
        let covariance = Array2::from_diag(&array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-13]);
        let error = propagate_fixed_l2_difference_covariance(
            &target,
            &reference,
            covariance.view(),
            SpatialL2Branch::FixedL2,
        )
        .unwrap_err();
        assert_eq!(error.status, SpatialL2Status::IllConditioned);

        let ill_conditioned_date_covariance = Array2::from_diag(&array![0.0, 1.0, 1.0e-13]);
        let error = rank_revealing_psd_factor(ill_conditioned_date_covariance.view(), true)
            .err()
            .unwrap();
        assert_eq!(error.status, SpatialL2Status::IllConditioned);
    }
}
