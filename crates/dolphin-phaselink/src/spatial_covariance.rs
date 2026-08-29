//! Reference-specific phase-linking influence and spatial covariance.
//!
//! This module contracts one target and one reference looked pixel against the
//! same native source keys. It is deliberately a local, rectangular CPU/f64
//! kernel: sequential ancestry, L2 inversion, persistence, and calibration are
//! owned by the workflow layers that consume this factor.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use dolphin_core::{Cf64, HalfWindow, Strides};
use ndarray::{Array1, Array2, ArrayView2, ArrayView3};

use crate::covariance::{
    rect_pixel_source_coherence_jvp, replay_rect_pixel_covariance, CovarianceReplayError,
    NativeSourcePixel,
};
use crate::estimator::{phase_angle_jvp, EstimatorJvpError, FixedEstimatorBranch};
use crate::source_influence::{ProperComplexFactor, SourceModelError};

/// Stable identity for the local reference-specific influence kernel.
pub const SPATIAL_INFLUENCE_METHOD: &str = "reference_specific_influence_v1";

/// Identity for the effective-look scaling applied after source-factor binding.
pub const EFFECTIVE_LOOKS_MODEL: &str = "source_factor_declared_v1";

/// Stable disposition for a reference-specific local query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialInfluenceStatus {
    /// The fixed rectangular CPU/f64 branch is evaluable.
    Valid,
    /// A target or reference output is outside the realized output grid.
    InvalidReference,
    /// A source factor is missing or has the wrong complex dimension.
    InvalidSourceFactor,
    /// The local replay failed before estimator differentiation.
    ReplayFailure,
    /// The selected estimator or support branch is not differentiable.
    UnsupportedBranch,
    /// The source state or resulting factor is non-finite.
    NonFinite,
}

impl SpatialInfluenceStatus {
    /// Stable serialized status name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::InvalidReference => "invalid_reference",
            Self::InvalidSourceFactor => "invalid_source_factor",
            Self::ReplayFailure => "replay_failure",
            Self::UnsupportedBranch => "unsupported_branch",
            Self::NonFinite => "nonfinite",
        }
    }
}

/// Failure while evaluating a reference-specific influence query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialInfluenceError {
    /// One output is outside the realized output grid.
    InvalidReference,
    /// The native validity grid does not match the stack.
    NativeGridShapeMismatch,
    /// No valid native source keys remain in either support.
    EmptySupport,
    /// A source factor is absent or has the wrong complex dimension.
    SourceFactor,
    /// The local covariance replay failed.
    Covariance(CovarianceReplayError),
    /// The selected estimator branch could not be differentiated.
    Estimator(EstimatorJvpError),
    /// The contracted factor or covariance was non-finite.
    NonFiniteResult,
    /// Two manually supplied factors do not have the same shape.
    FactorShapeMismatch,
    /// A validated proper-complex source factor could not be bound.
    SourceModel(SourceModelError),
}

impl SpatialInfluenceError {
    /// Stable disposition corresponding to this error.
    #[must_use]
    pub const fn status(&self) -> SpatialInfluenceStatus {
        match self {
            Self::InvalidReference => SpatialInfluenceStatus::InvalidReference,
            Self::NativeGridShapeMismatch | Self::SourceFactor | Self::SourceModel(_) => {
                SpatialInfluenceStatus::InvalidSourceFactor
            }
            Self::EmptySupport | Self::FactorShapeMismatch | Self::Covariance(_) => {
                SpatialInfluenceStatus::ReplayFailure
            }
            Self::Estimator(_) => SpatialInfluenceStatus::UnsupportedBranch,
            Self::NonFiniteResult => SpatialInfluenceStatus::NonFinite,
        }
    }
}

impl Display for SpatialInfluenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidReference => f.write_str("reference-specific query output is invalid"),
            Self::NativeGridShapeMismatch => {
                f.write_str("native validity grid does not match stack")
            }
            Self::EmptySupport => f.write_str("reference-specific query has empty support"),
            Self::SourceFactor => {
                f.write_str("source factor is missing or has the wrong dimension")
            }
            Self::Covariance(error) => Display::fmt(error, f),
            Self::Estimator(error) => Display::fmt(error, f),
            Self::NonFiniteResult => f.write_str("reference-specific factor is non-finite"),
            Self::FactorShapeMismatch => f.write_str("target/reference factor shapes differ"),
            Self::SourceModel(error) => Display::fmt(error, f),
        }
    }
}

impl Error for SpatialInfluenceError {}

impl From<CovarianceReplayError> for SpatialInfluenceError {
    fn from(value: CovarianceReplayError) -> Self {
        Self::Covariance(value)
    }
}

impl From<EstimatorJvpError> for SpatialInfluenceError {
    fn from(value: EstimatorJvpError) -> Self {
        Self::Estimator(value)
    }
}

impl From<SourceModelError> for SpatialInfluenceError {
    fn from(value: SourceModelError) -> Self {
        Self::SourceModel(value)
    }
}

/// Joint target/reference source factor and its covariance blocks.
#[derive(Debug, Clone)]
pub struct SpatialInfluenceResult {
    /// Fixed source keys in native row-major order.
    pub source_pixels: Vec<NativeSourcePixel>,
    /// Target phase influence, `(nslc, canonical source columns)`.
    pub target_factor: Array2<f64>,
    /// Reference phase influence, `(nslc, canonical source columns)`.
    pub reference_factor: Array2<f64>,
    /// Target minus reference influence, `(nslc, canonical source columns)`.
    pub difference_factor: Array2<f64>,
    /// Target marginal covariance.
    pub target_covariance: Array2<f64>,
    /// Reference marginal covariance.
    pub reference_covariance: Array2<f64>,
    /// Target/reference cross covariance.
    pub target_reference_covariance: Array2<f64>,
    /// Covariance of target minus reference.
    pub difference_covariance: Array2<f64>,
    /// Exclusive end offsets for each source's canonical `2*nslc` columns.
    pub source_factor_offsets: Vec<usize>,
    /// Proper-complex source-factor receipt digests in `source_pixels` order.
    pub source_factor_receipt_digests: Vec<[u8; 32]>,
    /// Effective-look scale applied to each bound source factor.
    pub effective_looks: f64,
    /// Effective-look model identity.
    pub effective_looks_model: &'static str,
}

/// Contract two factors against their shared independent source keys.
pub fn contract_source_factors(
    target_factor: ArrayView2<f64>,
    reference_factor: ArrayView2<f64>,
) -> Result<SpatialInfluenceResult, SpatialInfluenceError> {
    if target_factor.dim() != reference_factor.dim() {
        return Err(SpatialInfluenceError::FactorShapeMismatch);
    }
    if target_factor.is_empty()
        || target_factor
            .iter()
            .chain(reference_factor.iter())
            .any(|value| !value.is_finite())
    {
        return Err(SpatialInfluenceError::NonFiniteResult);
    }
    let difference_factor = &target_factor - &reference_factor;
    let target_covariance = target_factor.dot(&target_factor.t());
    let reference_covariance = reference_factor.dot(&reference_factor.t());
    let target_reference_covariance = target_factor.dot(&reference_factor.t());
    let difference_covariance = difference_factor.dot(&difference_factor.t());
    if target_covariance
        .iter()
        .chain(reference_covariance.iter())
        .chain(target_reference_covariance.iter())
        .chain(difference_covariance.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SpatialInfluenceError::NonFiniteResult);
    }
    Ok(SpatialInfluenceResult {
        source_pixels: Vec::new(),
        target_factor: target_factor.to_owned(),
        reference_factor: reference_factor.to_owned(),
        difference_factor,
        target_covariance,
        reference_covariance,
        target_reference_covariance,
        difference_covariance,
        source_factor_offsets: Vec::new(),
        source_factor_receipt_digests: Vec::new(),
        effective_looks: 1.0,
        effective_looks_model: EFFECTIVE_LOOKS_MODEL,
    })
}

/// Evaluate one target/reference pair against common native source factors.
///
/// Each factor must have one component per acquisition. Its canonical `2*nslc`
/// real embedding is bound to the complete real/imaginary phase JVP basis, so
/// no arbitrary one-direction complex perturbation is used. Invalid native
/// pixels are excluded by `native_validity`; the returned factors are bounded
/// by the union of the two local Rect supports.
#[allow(clippy::too_many_arguments)]
pub fn reference_specific_influence_v1(
    stack: ArrayView3<Cf64>,
    target_output: (usize, usize),
    reference_output: (usize, usize),
    half_window: HalfWindow,
    strides: Strides,
    native_validity: ndarray::ArrayView2<bool>,
    source_factors: &BTreeMap<NativeSourcePixel, ProperComplexFactor>,
    branch: FixedEstimatorBranch,
    reference_idx: usize,
    branch_tolerance: f64,
    effective_looks: f64,
) -> Result<SpatialInfluenceResult, SpatialInfluenceError> {
    let (nslc, rows, columns) = stack.dim();
    if native_validity.dim() != (rows, columns) {
        return Err(SpatialInfluenceError::NativeGridShapeMismatch);
    }
    if !effective_looks.is_finite() || effective_looks <= 0.0 {
        return Err(SpatialInfluenceError::SourceFactor);
    }
    let target_replay =
        replay_rect_pixel_covariance(stack, target_output, half_window, strides, native_validity)
            .map_err(|error| match error {
            CovarianceReplayError::OutputOutOfBounds => SpatialInfluenceError::InvalidReference,
            other => SpatialInfluenceError::Covariance(other),
        })?;
    let reference_replay = replay_rect_pixel_covariance(
        stack,
        reference_output,
        half_window,
        strides,
        native_validity,
    )
    .map_err(|error| match error {
        CovarianceReplayError::OutputOutOfBounds => SpatialInfluenceError::InvalidReference,
        other => SpatialInfluenceError::Covariance(other),
    })?;
    if target_replay.source_pixels.is_empty() || reference_replay.source_pixels.is_empty() {
        return Err(SpatialInfluenceError::EmptySupport);
    }
    let source_pixels = target_replay
        .source_pixels
        .iter()
        .chain(reference_replay.source_pixels.iter())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let target_factor = factor_for_output(
        stack,
        &target_replay,
        &source_pixels,
        source_factors,
        branch,
        reference_idx,
        branch_tolerance,
        effective_looks,
    )?;
    let reference_factor = factor_for_output(
        stack,
        &reference_replay,
        &source_pixels,
        source_factors,
        branch,
        reference_idx,
        branch_tolerance,
        effective_looks,
    )?;
    let mut result = contract_source_factors(target_factor.view(), reference_factor.view())?;
    result.source_pixels = source_pixels;
    result.source_factor_offsets =
        source_factor_offsets(source_factors, &result.source_pixels, nslc)?;
    result.source_factor_receipt_digests = result
        .source_pixels
        .iter()
        .map(|source| {
            source_factors
                .get(source)
                .ok_or(SpatialInfluenceError::SourceFactor)
                .map(ProperComplexFactor::numeric_receipt_digest)
        })
        .collect::<Result<Vec<_>, _>>()?;
    result.effective_looks = effective_looks;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn factor_for_output(
    stack: ArrayView3<Cf64>,
    replay: &crate::covariance::RectPixelReplay,
    source_pixels: &[NativeSourcePixel],
    source_factors: &BTreeMap<NativeSourcePixel, ProperComplexFactor>,
    branch: FixedEstimatorBranch,
    reference_idx: usize,
    branch_tolerance: f64,
    effective_looks: f64,
) -> Result<Array2<f64>, SpatialInfluenceError> {
    let nslc = stack.dim().0;
    let offsets = source_factor_offsets(source_factors, source_pixels, nslc)?;
    let total_columns = *offsets.last().ok_or(SpatialInfluenceError::EmptySupport)?;
    let mut factor = Array2::zeros((nslc, total_columns));
    for (source_index, &source) in source_pixels.iter().enumerate() {
        if !replay.source_pixels.contains(&source) {
            continue;
        }
        let source_factor = source_factors
            .get(&source)
            .ok_or(SpatialInfluenceError::SourceFactor)?;
        let mut raw_jacobian = Array2::zeros((nslc, 2 * nslc));
        for component in 0..nslc {
            let real_direction =
                Array1::from_shape_fn(nslc, |date| Cf64::new(f64::from(date == component), 0.0));
            let real_delta = rect_pixel_source_coherence_jvp(
                stack,
                replay,
                source,
                real_direction.view(),
                branch_tolerance,
            )?;
            let real_phase = phase_angle_jvp(
                replay.coherence.view(),
                real_delta.view(),
                branch,
                reference_idx,
                branch_tolerance,
            )?;
            raw_jacobian.column_mut(component).assign(&real_phase);

            let imaginary_direction =
                Array1::from_shape_fn(nslc, |date| Cf64::new(0.0, f64::from(date == component)));
            let imaginary_delta = rect_pixel_source_coherence_jvp(
                stack,
                replay,
                source,
                imaginary_direction.view(),
                branch_tolerance,
            )?;
            let imaginary_phase = phase_angle_jvp(
                replay.coherence.view(),
                imaginary_delta.view(),
                branch,
                reference_idx,
                branch_tolerance,
            )?;
            raw_jacobian
                .column_mut(nslc + component)
                .assign(&imaginary_phase);
        }
        let bound = source_factor.bind_real_jacobian(raw_jacobian.view())?;
        let scale = effective_looks.sqrt().recip();
        let start = offsets[source_index];
        for row in 0..nslc {
            for column in 0..2 * nslc {
                factor[(row, start + column)] = bound.coefficient()[(row, column)] * scale;
            }
        }
    }
    Ok(factor)
}

fn source_factor_offsets(
    source_factors: &BTreeMap<NativeSourcePixel, ProperComplexFactor>,
    source_pixels: &[NativeSourcePixel],
    nslc: usize,
) -> Result<Vec<usize>, SpatialInfluenceError> {
    let mut offsets: Vec<usize> = Vec::with_capacity(source_pixels.len() + 1);
    offsets.push(0);
    for source in source_pixels {
        let factor = source_factors
            .get(source)
            .ok_or(SpatialInfluenceError::SourceFactor)?;
        if factor.lower().nrows() != nslc {
            return Err(SpatialInfluenceError::SourceFactor);
        }
        let next = offsets
            .last()
            .copied()
            .and_then(|offset| offset.checked_add(2 * nslc))
            .ok_or(SpatialInfluenceError::SourceFactor)?;
        offsets.push(next);
    }
    Ok(offsets)
}
