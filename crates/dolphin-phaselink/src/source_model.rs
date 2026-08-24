//! Empirical proper-complex primitive source factors.
//!
//! The primitive speckle process is zero mean, so the source covariance is
//! the uncentered second moment `Z Z^H / n` over fixed-valid native samples.

use std::error::Error;
use std::fmt::{Display, Formatter};

use dolphin_core::Cf64;
use ndarray::{Array2, ArrayView2, ArrayView3};
use sha2::{Digest, Sha256};

use crate::source_influence::{ProperComplexFactor, SourceId, SourceModelError};

/// Stable empirical proper-complex source-model method name.
pub const EMPIRICAL_PROPER_COMPLEX_METHOD: &str = "source_centered_empirical_proper_complex_v1";

/// Stable empirical proper-complex source-model version.
pub const EMPIRICAL_PROPER_COMPLEX_VERSION: u32 = 1;

const RELATIVE_DIAGONAL_FLOOR_RULE: &str = "mean_positive_covariance_diagonal_v1";
const CHOLESKY_DIAGONAL_IMAGINARY_RELATIVE_TOLERANCE: f64 = 64.0 * f64::EPSILON;

/// Fixed source-centered empirical covariance configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalProperComplexConfig {
    half_window_rows: usize,
    half_window_columns: usize,
    shrinkage_alpha: f64,
    relative_diagonal_floor: f64,
    model_identity: [u8; 32],
    config_digest: [u8; 32],
}

impl EmpiricalProperComplexConfig {
    /// Validate and construct an empirical source-model configuration.
    ///
    /// # Errors
    /// Returns an error for overflowing support, shrinkage outside `(0, 1]`,
    /// a relative diagonal floor outside `(0, 1]`, or a missing model identity.
    pub fn new(
        half_window_rows: usize,
        half_window_columns: usize,
        shrinkage_alpha: f64,
        relative_diagonal_floor: f64,
        model_identity: [u8; 32],
    ) -> Result<Self, EmpiricalSourceModelError> {
        let support_rows = half_window_rows
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(EmpiricalSourceModelError::InvalidSupport)?;
        let support_columns = half_window_columns
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(EmpiricalSourceModelError::InvalidSupport)?;
        if !(shrinkage_alpha.is_finite() && 0.0 < shrinkage_alpha && shrinkage_alpha <= 1.0) {
            return Err(EmpiricalSourceModelError::InvalidShrinkage);
        }
        if !(relative_diagonal_floor.is_finite()
            && 0.0 < relative_diagonal_floor
            && relative_diagonal_floor <= 1.0)
        {
            return Err(EmpiricalSourceModelError::InvalidRelativeDiagonalFloor);
        }
        if model_identity.iter().all(|byte| *byte == 0) {
            return Err(EmpiricalSourceModelError::MissingModelIdentity);
        }

        let mut digest = Sha256::new();
        digest.update(b"dolphinrust:empirical_proper_complex_config:v1");
        digest.update(EMPIRICAL_PROPER_COMPLEX_METHOD.as_bytes());
        digest.update(EMPIRICAL_PROPER_COMPLEX_VERSION.to_le_bytes());
        digest.update((support_rows as u64).to_le_bytes());
        digest.update((support_columns as u64).to_le_bytes());
        digest.update(shrinkage_alpha.to_bits().to_le_bytes());
        digest.update(RELATIVE_DIAGONAL_FLOOR_RULE.as_bytes());
        digest.update(relative_diagonal_floor.to_bits().to_le_bytes());
        digest.update(
            CHOLESKY_DIAGONAL_IMAGINARY_RELATIVE_TOLERANCE
                .to_bits()
                .to_le_bytes(),
        );
        digest.update(model_identity);

        Ok(Self {
            half_window_rows,
            half_window_columns,
            shrinkage_alpha,
            relative_diagonal_floor,
            model_identity,
            config_digest: digest.finalize().into(),
        })
    }

    /// Fixed native support shape in rows and columns.
    #[must_use]
    pub const fn support_shape(&self) -> (usize, usize) {
        (
            self.half_window_rows * 2 + 1,
            self.half_window_columns * 2 + 1,
        )
    }

    /// Declared shrinkage weight toward the empirical diagonal.
    #[must_use]
    pub const fn shrinkage_alpha(&self) -> f64 {
        self.shrinkage_alpha
    }

    /// Minimum component diagonal relative to the mean positive diagonal.
    #[must_use]
    pub const fn relative_diagonal_floor(&self) -> f64 {
        self.relative_diagonal_floor
    }

    /// Caller-supplied source-model implementation or parameter identity.
    #[must_use]
    pub const fn model_identity(&self) -> &[u8; 32] {
        &self.model_identity
    }

    /// Canonical digest of method, version, support, shrinkage, relative-floor rule/value,
    /// Cholesky diagonal tolerance, and model identity.
    #[must_use]
    pub const fn config_digest(&self) -> &[u8; 32] {
        &self.config_digest
    }
}

/// Canonical receipt for one empirical primitive source factor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmpiricalProperComplexReceipt {
    source: SourceId,
    window_origin: (usize, usize),
    window_shape: (usize, usize),
    sample_count: usize,
    config_digest: [u8; 32],
    content_digest: [u8; 32],
    digest: [u8; 32],
}

impl EmpiricalProperComplexReceipt {
    /// Stable method name.
    #[must_use]
    pub const fn method(&self) -> &'static str {
        EMPIRICAL_PROPER_COMPLEX_METHOD
    }

    /// Stable method version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        EMPIRICAL_PROPER_COMPLEX_VERSION
    }

    /// Primitive source identifier bound by this receipt.
    #[must_use]
    pub const fn source(&self) -> SourceId {
        self.source
    }

    /// Global native-grid origin of the inward-clamped support.
    #[must_use]
    pub const fn window_origin(&self) -> (usize, usize) {
        self.window_origin
    }

    /// Fixed native-grid support shape.
    #[must_use]
    pub const fn window_shape(&self) -> (usize, usize) {
        self.window_shape
    }

    /// Number of fixed-valid finite spatial samples.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Canonical configuration identity.
    #[must_use]
    pub const fn config_digest(&self) -> &[u8; 32] {
        &self.config_digest
    }

    /// Canonical source-window content and data identity.
    #[must_use]
    pub const fn content_digest(&self) -> &[u8; 32] {
        &self.content_digest
    }

    /// Canonical source, content, configuration, and component-order receipt.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

/// Empirical factor paired with the exact receipt used as its model hash.
#[derive(Debug, Clone)]
pub struct EmpiricalProperComplexEstimate {
    factor: ProperComplexFactor,
    receipt: EmpiricalProperComplexReceipt,
}

impl EmpiricalProperComplexEstimate {
    /// Validated proper-complex lower factor.
    #[must_use]
    pub const fn factor(&self) -> &ProperComplexFactor {
        &self.factor
    }

    /// Source-model receipt whose digest is the factor model hash.
    #[must_use]
    pub const fn receipt(&self) -> &EmpiricalProperComplexReceipt {
        &self.receipt
    }

    /// Consume the estimate into its validated factor and receipt.
    #[must_use]
    pub fn into_parts(self) -> (ProperComplexFactor, EmpiricalProperComplexReceipt) {
        (self.factor, self.receipt)
    }
}

/// Fail-closed empirical primitive source-model error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmpiricalSourceModelError {
    /// Fixed support dimensions overflowed.
    InvalidSupport,
    /// Shrinkage was non-finite or outside `(0, 1]`.
    InvalidShrinkage,
    /// The relative diagonal floor was non-finite or outside `(0, 1]`.
    InvalidRelativeDiagonalFloor,
    /// The model identity was the all-zero missing value.
    MissingModelIdentity,
    /// The data identity was the all-zero missing value.
    MissingDataIdentity,
    /// Stack, validity mask, or ordered component dimensions did not agree.
    ShapeMismatch,
    /// The requested global source pixel was outside the supplied native grid.
    SourceOutsideGrid,
    /// The grid could not supply the fixed window or contained no valid sample.
    MissingSupport,
    /// A fixed-valid sample contained NaN or infinity.
    NonFiniteSample,
    /// The covariance contained no positive finite diagonal.
    NoPositiveDiagonal,
    /// A covariance component was below the declared relative diagonal floor.
    DiagonalBelowRelativeFloor(usize),
    /// Deterministic complex Cholesky could not factor the shrunk covariance.
    CholeskyFailure,
    /// The generated factor violated the common proper-complex factor contract.
    Factor(SourceModelError),
}

impl Display for EmpiricalSourceModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSupport => write!(f, "empirical source support is invalid"),
            Self::InvalidShrinkage => write!(f, "empirical source shrinkage is invalid"),
            Self::InvalidRelativeDiagonalFloor => {
                write!(f, "empirical source relative diagonal floor is invalid")
            }
            Self::MissingModelIdentity => write!(f, "empirical source model identity is missing"),
            Self::MissingDataIdentity => write!(f, "empirical source data identity is missing"),
            Self::ShapeMismatch => write!(f, "empirical source stack dimensions do not agree"),
            Self::SourceOutsideGrid => write!(f, "empirical source pixel is outside the grid"),
            Self::MissingSupport => write!(f, "empirical source fixed support is unavailable"),
            Self::NonFiniteSample => write!(f, "empirical source sample is non-finite"),
            Self::NoPositiveDiagonal => {
                write!(f, "empirical source covariance has no positive diagonal")
            }
            Self::DiagonalBelowRelativeFloor(component) => write!(
                f,
                "empirical source component {component} is below the relative diagonal floor"
            ),
            Self::CholeskyFailure => {
                write!(f, "empirical source covariance is not positive definite")
            }
            Self::Factor(error) => Display::fmt(error, f),
        }
    }
}

impl Error for EmpiricalSourceModelError {}

impl From<SourceModelError> for EmpiricalSourceModelError {
    fn from(value: SourceModelError) -> Self {
        Self::Factor(value)
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedWindow {
    local_origin: (usize, usize),
    global_origin: (usize, usize),
    shape: (usize, usize),
}

struct SampleSummary {
    sample_count: usize,
    content_digest: [u8; 32],
}

fn resolve_window(
    grid_shape: (usize, usize),
    grid_origin: (usize, usize),
    source_pixel: (usize, usize),
    config: &EmpiricalProperComplexConfig,
) -> Result<ResolvedWindow, EmpiricalSourceModelError> {
    let (grid_rows, grid_columns) = grid_shape;
    let (support_rows, support_columns) = config.support_shape();
    if grid_rows < support_rows || grid_columns < support_columns {
        return Err(EmpiricalSourceModelError::MissingSupport);
    }
    let source_row = source_pixel
        .0
        .checked_sub(grid_origin.0)
        .ok_or(EmpiricalSourceModelError::SourceOutsideGrid)?;
    let source_column = source_pixel
        .1
        .checked_sub(grid_origin.1)
        .ok_or(EmpiricalSourceModelError::SourceOutsideGrid)?;
    if source_row >= grid_rows || source_column >= grid_columns {
        return Err(EmpiricalSourceModelError::SourceOutsideGrid);
    }
    let window_row = source_row
        .saturating_sub(config.half_window_rows)
        .min(grid_rows - support_rows);
    let window_column = source_column
        .saturating_sub(config.half_window_columns)
        .min(grid_columns - support_columns);
    let global_row = grid_origin
        .0
        .checked_add(window_row)
        .ok_or(EmpiricalSourceModelError::InvalidSupport)?;
    let global_column = grid_origin
        .1
        .checked_add(window_column)
        .ok_or(EmpiricalSourceModelError::InvalidSupport)?;
    Ok(ResolvedWindow {
        local_origin: (window_row, window_column),
        global_origin: (global_row, global_column),
        shape: (support_rows, support_columns),
    })
}

fn summarize_samples(
    component_ids: &[u64],
    values: ArrayView3<'_, Cf64>,
    valid: ArrayView2<'_, bool>,
    window: ResolvedWindow,
    data_identity: [u8; 32],
) -> Result<SampleSummary, EmpiricalSourceModelError> {
    let mut content = Sha256::new();
    content.update(b"dolphinrust:empirical_proper_complex_content:v1");
    content.update(data_identity);
    content.update((window.global_origin.0 as u64).to_le_bytes());
    content.update((window.global_origin.1 as u64).to_le_bytes());
    content.update((window.shape.0 as u64).to_le_bytes());
    content.update((window.shape.1 as u64).to_le_bytes());
    content.update((component_ids.len() as u64).to_le_bytes());
    for component in component_ids {
        content.update(component.to_le_bytes());
    }

    let mut sample_count = 0usize;
    for row in window.local_origin.0..window.local_origin.0 + window.shape.0 {
        for column in window.local_origin.1..window.local_origin.1 + window.shape.1 {
            let is_valid = valid[(row, column)];
            content.update([u8::from(is_valid)]);
            if !is_valid {
                continue;
            }
            sample_count += 1;
            for date in 0..component_ids.len() {
                let value = values[(date, row, column)];
                if !value.is_finite() {
                    return Err(EmpiricalSourceModelError::NonFiniteSample);
                }
                content.update(value.re.to_bits().to_le_bytes());
                content.update(value.im.to_bits().to_le_bytes());
            }
        }
    }
    if sample_count == 0 {
        return Err(EmpiricalSourceModelError::MissingSupport);
    }
    Ok(SampleSummary {
        sample_count,
        content_digest: content.finalize().into(),
    })
}

fn shrunk_second_moment(
    values: ArrayView3<'_, Cf64>,
    valid: ArrayView2<'_, bool>,
    window: ResolvedWindow,
    summary: &SampleSummary,
    config: &EmpiricalProperComplexConfig,
) -> Result<Array2<Cf64>, EmpiricalSourceModelError> {
    let date_count = values.dim().0;
    let mut covariance = Array2::from_elem((date_count, date_count), Cf64::new(0.0, 0.0));
    for row in window.local_origin.0..window.local_origin.0 + window.shape.0 {
        for column in window.local_origin.1..window.local_origin.1 + window.shape.1 {
            if !valid[(row, column)] {
                continue;
            }
            for left in 0..date_count {
                for right in 0..date_count {
                    covariance[(left, right)] +=
                        values[(left, row, column)] * values[(right, row, column)].conj();
                }
            }
        }
    }
    covariance.mapv_inplace(|value| value / summary.sample_count as f64);
    let mut positive_diagonal_sum = 0.0;
    let mut positive_diagonal_count = 0usize;
    for component in 0..date_count {
        let diagonal = covariance[(component, component)].re;
        if diagonal.is_finite() && diagonal > 0.0 {
            positive_diagonal_sum += diagonal;
            positive_diagonal_count += 1;
        }
    }
    if positive_diagonal_count == 0 || !positive_diagonal_sum.is_finite() {
        return Err(EmpiricalSourceModelError::NoPositiveDiagonal);
    }
    let relative_floor =
        config.relative_diagonal_floor * (positive_diagonal_sum / positive_diagonal_count as f64);
    for component in 0..date_count {
        let diagonal = covariance[(component, component)].re;
        if !diagonal.is_finite() || diagonal < relative_floor {
            return Err(EmpiricalSourceModelError::DiagonalBelowRelativeFloor(
                component,
            ));
        }
        covariance[(component, component)] = Cf64::new(diagonal, 0.0);
    }
    let off_diagonal_weight = 1.0 - config.shrinkage_alpha;
    for row in 0..date_count {
        for column in 0..date_count {
            if row != column {
                covariance[(row, column)] *= off_diagonal_weight;
            }
        }
    }
    Ok(covariance)
}

fn deterministic_complex_cholesky(
    covariance: &Array2<Cf64>,
) -> Result<Array2<Cf64>, EmpiricalSourceModelError> {
    let date_count = covariance.nrows();
    let mut lower = Array2::from_elem((date_count, date_count), Cf64::new(0.0, 0.0));
    for row in 0..date_count {
        for column in 0..=row {
            let mut residual = covariance[(row, column)];
            for inner in 0..column {
                residual -= lower[(row, inner)] * lower[(column, inner)].conj();
            }
            if row == column {
                let diagonal_scale = covariance[(row, row)].norm().max(residual.re.abs());
                let imaginary_limit =
                    CHOLESKY_DIAGONAL_IMAGINARY_RELATIVE_TOLERANCE * diagonal_scale;
                if !residual.re.is_finite()
                    || residual.re <= 0.0
                    || !residual.im.is_finite()
                    || residual.im.abs() > imaginary_limit
                {
                    return Err(EmpiricalSourceModelError::CholeskyFailure);
                }
                lower[(row, column)] = Cf64::new(residual.re.sqrt(), 0.0);
            } else {
                lower[(row, column)] = residual / lower[(column, column)].re;
            }
        }
    }
    if lower.iter().any(|value| !value.is_finite()) {
        return Err(EmpiricalSourceModelError::CholeskyFailure);
    }
    Ok(lower)
}

/// Estimate a source-centered empirical proper-complex primitive factor.
///
/// `values` is ordered `(date, native row, native column)`. `grid_origin`
/// maps its local upper-left pixel to the global native grid. The fixed support
/// is shifted inward at supplied-grid borders rather than truncated. The API
/// intentionally has no target, reference, or consuming tile identity.
///
/// # Errors
/// Returns a fail-closed error for missing identities or support, mismatched
/// dimensions, non-finite fixed-valid samples, insufficient relative diagonal,
/// or a factorization/contract failure.
#[allow(clippy::too_many_arguments)]
pub fn estimate_empirical_proper_complex_factor(
    source: SourceId,
    component_ids: &[u64],
    values: ArrayView3<'_, Cf64>,
    valid: ArrayView2<'_, bool>,
    grid_origin: (usize, usize),
    source_pixel: (usize, usize),
    data_identity: [u8; 32],
    config: &EmpiricalProperComplexConfig,
) -> Result<EmpiricalProperComplexEstimate, EmpiricalSourceModelError> {
    if data_identity.iter().all(|byte| *byte == 0) {
        return Err(EmpiricalSourceModelError::MissingDataIdentity);
    }
    let (date_count, grid_rows, grid_columns) = values.dim();
    if component_ids.len() != date_count
        || date_count == 0
        || valid.dim() != (grid_rows, grid_columns)
    {
        return Err(EmpiricalSourceModelError::ShapeMismatch);
    }
    let window = resolve_window((grid_rows, grid_columns), grid_origin, source_pixel, config)?;
    let summary = summarize_samples(component_ids, values, valid, window, data_identity)?;
    let covariance = shrunk_second_moment(values, valid, window, &summary, config)?;
    let lower = deterministic_complex_cholesky(&covariance)?;
    let mut receipt_digest = Sha256::new();
    receipt_digest.update(b"dolphinrust:empirical_proper_complex_receipt:v1");
    receipt_digest.update(source.get().to_le_bytes());
    receipt_digest.update(config.config_digest);
    receipt_digest.update(summary.content_digest);
    let digest = receipt_digest.finalize().into();
    let factor = ProperComplexFactor::new(source, component_ids.to_vec(), digest, lower)?;
    let receipt = EmpiricalProperComplexReceipt {
        source,
        window_origin: window.global_origin,
        window_shape: window.shape,
        sample_count: summary.sample_count,
        config_digest: config.config_digest,
        content_digest: summary.content_digest,
        digest,
    };
    Ok(EmpiricalProperComplexEstimate { factor, receipt })
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn cholesky_rejects_materially_complex_diagonal() {
        let covariance = array![[Cf64::new(1.0, 1.0e-8)]];

        assert_eq!(
            deterministic_complex_cholesky(&covariance).unwrap_err(),
            EmpiricalSourceModelError::CholeskyFailure
        );
    }
}
