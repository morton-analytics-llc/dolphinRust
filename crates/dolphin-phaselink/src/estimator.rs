//! EVD / EMI phase-linking estimators (port of `_core.process_coherence_matrices`).
//!
//! From a per-pixel coherence matrix `C` (Hermitian), recover the optimized
//! wrapped phase:
//! * **EVD** — dominant eigenvector of `C ⊙ |C|`.
//! * **EMI** (default) — least eigenvector of `Γ⁻¹ ⊙ C`, where `Γ = |C|`
//!   (regularized, thresholded, Cholesky-inverted). On a non-invertible `Γ`
//!   (Cholesky failure / non-finite inverse) fall back to EVD — part of the
//!   algorithm, kept.
//!
//! Both target matrices are Hermitian, so we use faer's direct selfadjoint
//! eigendecomposition (the crate's sanctioned optimization over dolphin's
//! power/inverse iteration; validated to tolerance against the oracle). The
//! phase is referenced to `reference_idx`: `θ ← θ · exp(-j∠θ[ref])`.

use dolphin_core::Cf64;
use faer::prelude::{c64, SpSolver};
use faer::{Mat, Side};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView4};
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Per-pixel phase-linking result.
#[derive(Debug, Clone)]
pub struct PixelEstimate {
    /// Referenced wrapped phase, length `nslc`.
    pub phase: Array1<Cf64>,
    /// The dominant (EVD) or least (EMI) eigenvalue.
    pub eigenvalue: f64,
    /// Distance from the selected eigenvalue to the nearest other mode.
    pub eigengap: f64,
    /// Estimator used: 0 = EVD, 1 = EMI.
    pub estimator: u8,
}

/// Estimator branch frozen for source-influence differentiation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FixedEstimatorBranch {
    /// Dominant mode of `C hadamard |C|`.
    Evd,
    /// Least mode of `Gamma^-1 hadamard C` with fixed EMI parameters.
    Emi {
        /// EMI magnitude regularization weight.
        beta: f64,
        /// Magnitudes below this cutoff are fixed to zero.
        zero_correlation_threshold: f64,
    },
}

/// Failure to differentiate a fixed estimator branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EstimatorJvpError {
    /// The coherence or direction was not a matching square matrix.
    MatrixShapeMismatch,
    /// The selected output reference was outside the estimator vector.
    ReferenceOutOfBounds,
    /// The coherence state or direction contained NaN or infinity.
    NonFiniteState,
    /// EMI would fall back to EVD at the supplied state.
    EmiFallback,
    /// A coherence magnitude was at a nondifferentiable zero branch.
    ZeroMagnitudeBranch,
    /// An EMI threshold decision was within the declared tolerance.
    ThresholdBoundary,
    /// The selected eigenvalue was tied within the declared tolerance.
    EigenvalueTie,
    /// The selected eigenvector's reference component vanished.
    VanishingReference,
    /// The resulting phase direction contained NaN or infinity.
    NonFiniteDerivative,
}

impl Display for EstimatorJvpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::MatrixShapeMismatch => "estimator JVP matrix shape mismatch",
            Self::ReferenceOutOfBounds => "estimator JVP reference is out of bounds",
            Self::NonFiniteState => "estimator JVP state is non-finite",
            Self::EmiFallback => "fixed EMI branch would fall back to EVD",
            Self::ZeroMagnitudeBranch => "estimator JVP has a zero-magnitude active entry",
            Self::ThresholdBoundary => "EMI threshold branch is unstable",
            Self::EigenvalueTie => "selected estimator eigenvalue is tied",
            Self::VanishingReference => "selected eigenvector reference component vanishes",
            Self::NonFiniteDerivative => "estimator phase derivative is non-finite",
        };
        f.write_str(message)
    }
}

impl Error for EstimatorJvpError {}

/// Stacked phase-linking output over an `(out_rows, out_cols)` grid.
pub struct StackEstimate {
    /// Referenced phase, shape `(nslc, out_rows, out_cols)`.
    pub cpx_phase: Array3<Cf64>,
    /// Eigenvalue per pixel, shape `(out_rows, out_cols)`.
    pub eigenvalues: Array2<f64>,
    /// Estimator per pixel, shape `(out_rows, out_cols)`.
    pub estimator: Array2<u8>,
}

/// Run the estimator over a `(out_rows, out_cols, nslc, nslc)` coherence stack.
#[must_use]
pub fn process_coherence_matrices(
    c_arrays: ArrayView4<Cf64>,
    use_evd: bool,
    beta: f64,
    zero_correlation_threshold: f64,
    reference_idx: usize,
) -> StackEstimate {
    let (out_rows, out_cols, nslc, _) = c_arrays.dim();
    let estimates: Vec<PixelEstimate> = (0..out_rows * out_cols)
        .into_par_iter()
        .map(|idx| {
            let c = c_arrays.slice(ndarray::s![idx / out_cols, idx % out_cols, .., ..]);
            process_coherence_matrix(c, use_evd, beta, zero_correlation_threshold, reference_idx)
        })
        .collect();
    pack(estimates, (out_rows, out_cols, nslc))
}

/// Estimate the linked phase for one pixel's coherence matrix.
#[must_use]
pub fn process_coherence_matrix(
    c: ArrayView2<Cf64>,
    use_evd: bool,
    beta: f64,
    zero_correlation_threshold: f64,
    reference_idx: usize,
) -> PixelEstimate {
    if use_evd {
        return reference(evd_eigenvector(c), 0, reference_idx);
    }
    // EMI is the common, successful case; compute the EVD eigendecomposition
    // only when EMI's `Γ` is singular and the EVD fallback is actually needed
    // (dolphin's NaN-triggered fallback). Eager EVD on every pixel was a wasted
    // second selfadjoint eigendecomposition — ~half the estimator's CPU.
    match emi_eigenvector(c, beta, zero_correlation_threshold) {
        Some(emi) => reference(emi, 1, reference_idx),
        None => reference(evd_eigenvector(c), 0, reference_idx),
    }
}

/// `(eigenvector, eigenvalue)` of the dominant mode of `C ⊙ |C|`.
fn evd_eigenvector(c: ArrayView2<Cf64>) -> (Array1<Cf64>, f64, f64) {
    let m = hadamard_abs(c);
    let (vals, vecs) = selfadjoint_eig(&m);
    let k = argmax(&vals);
    (column(&vecs, k), vals[k], selected_eigengap(&vals, k))
}

/// `(eigenvector, eigenvalue)` of the least mode of `Γ⁻¹ ⊙ C`, or `None` if
/// `Γ = |C|` cannot be inverted.
fn emi_eigenvector(
    c: ArrayView2<Cf64>,
    beta: f64,
    zero_correlation_threshold: f64,
) -> Option<(Array1<Cf64>, f64, f64)> {
    let n = c.nrows();
    let gamma = regularized_gamma(c, beta, zero_correlation_threshold);
    let gamma_inv = invert_spd(&gamma)?;
    let m = hadamard(&gamma_inv, c);
    let (vals, vecs) = selfadjoint_eig(&m);
    let k = argmin(&vals);
    let vec = normalize_norm(column(&vecs, k), (n as f64).sqrt());
    Some((vec, vals[k], selected_eigengap(&vals, k)))
}

/// Differentiate the referenced linked-phase angles for one fixed coherence direction.
///
/// The selected EVD/EMI branch is recomputed from `coherence`; no estimator
/// fallback or finite-difference branch switching is allowed.
///
/// # Errors
/// Returns an error for invalid shapes/state, an EMI fallback, a threshold or
/// zero-magnitude boundary, a tied selected mode, or a vanishing reference.
pub fn phase_angle_jvp(
    coherence: ArrayView2<Cf64>,
    delta_coherence: ArrayView2<Cf64>,
    branch: FixedEstimatorBranch,
    reference_idx: usize,
    branch_tolerance: f64,
) -> Result<Array1<f64>, EstimatorJvpError> {
    let n = coherence.nrows();
    if n == 0
        || coherence.ncols() != n
        || delta_coherence.dim() != coherence.dim()
        || !branch_tolerance.is_finite()
        || branch_tolerance < 0.0
    {
        return Err(EstimatorJvpError::MatrixShapeMismatch);
    }
    if reference_idx >= n {
        return Err(EstimatorJvpError::ReferenceOutOfBounds);
    }
    if coherence
        .iter()
        .chain(delta_coherence.iter())
        .any(|value| !value.is_finite())
    {
        return Err(EstimatorJvpError::NonFiniteState);
    }

    let (matrix, delta_matrix) = match branch {
        FixedEstimatorBranch::Evd => evd_matrix_jvp(coherence, delta_coherence, branch_tolerance)?,
        FixedEstimatorBranch::Emi {
            beta,
            zero_correlation_threshold,
        } => emi_matrix_jvp(
            coherence,
            delta_coherence,
            beta,
            zero_correlation_threshold,
            branch_tolerance,
        )?,
    };
    let (values, vectors) = selfadjoint_eig(&matrix);
    let selected = match branch {
        FixedEstimatorBranch::Evd => argmax(&values),
        FixedEstimatorBranch::Emi { .. } => argmin(&values),
    };
    if selected_eigengap(&values, selected) <= branch_tolerance {
        return Err(EstimatorJvpError::EigenvalueTie);
    }
    let vector = column(&vectors, selected);
    if vector[reference_idx].norm() <= branch_tolerance {
        return Err(EstimatorJvpError::VanishingReference);
    }

    let mut delta_vector = Array1::zeros(n);
    for other in 0..n {
        if other == selected {
            continue;
        }
        let basis = column(&vectors, other);
        let coefficient = quadratic_cross(basis.view(), delta_matrix.view(), vector.view())
            / (values[selected] - values[other]);
        delta_vector += &basis.mapv(|value| value * coefficient);
    }
    let raw = Array1::from_shape_fn(n, |i| {
        (vector[i].conj() * delta_vector[i]).im / vector[i].norm_sqr()
    });
    if raw.iter().any(|value| !value.is_finite()) {
        return Err(EstimatorJvpError::NonFiniteDerivative);
    }
    let reference = raw[reference_idx];
    Ok(raw - reference)
}

fn evd_matrix_jvp(
    coherence: ArrayView2<Cf64>,
    delta: ArrayView2<Cf64>,
    branch_tolerance: f64,
) -> Result<(Mat<c64>, Array2<Cf64>), EstimatorJvpError> {
    let n = coherence.nrows();
    let mut delta_matrix = Array2::zeros((n, n));
    for ((i, j), value) in coherence.indexed_iter() {
        let magnitude = value.norm();
        if magnitude <= branch_tolerance {
            return Err(EstimatorJvpError::ZeroMagnitudeBranch);
        }
        let delta_magnitude = (value.conj() * delta[(i, j)]).re / magnitude;
        delta_matrix[(i, j)] = delta[(i, j)] * magnitude + *value * delta_magnitude;
    }
    Ok((hadamard_abs(coherence), delta_matrix))
}

fn emi_matrix_jvp(
    coherence: ArrayView2<Cf64>,
    delta: ArrayView2<Cf64>,
    beta: f64,
    zero_cut: f64,
    branch_tolerance: f64,
) -> Result<(Mat<c64>, Array2<Cf64>), EstimatorJvpError> {
    let n = coherence.nrows();
    let mut delta_gamma = Array2::zeros((n, n));
    for ((i, j), value) in coherence.indexed_iter() {
        let magnitude = value.norm();
        let unthresholded = if beta > 0.0 {
            (1.0 - beta) * magnitude + beta * f64::from(i == j)
        } else {
            magnitude
        };
        if (unthresholded - zero_cut).abs() <= branch_tolerance {
            return Err(EstimatorJvpError::ThresholdBoundary);
        }
        if unthresholded < zero_cut {
            continue;
        }
        if magnitude <= branch_tolerance {
            return Err(EstimatorJvpError::ZeroMagnitudeBranch);
        }
        let scale = if beta > 0.0 { 1.0 - beta } else { 1.0 };
        delta_gamma[(i, j)] = scale * (value.conj() * delta[(i, j)]).re / magnitude;
    }
    let gamma = regularized_gamma(coherence, beta, zero_cut);
    let gamma_inverse = invert_spd(&gamma).ok_or(EstimatorJvpError::EmiFallback)?;
    let delta_gamma_inverse = Array2::from_shape_fn((n, n), |(i, j)| {
        let mut value = 0.0;
        for a in 0..n {
            for b in 0..n {
                value += gamma_inverse[(i, a)] * delta_gamma[(a, b)] * gamma_inverse[(b, j)];
            }
        }
        -value
    });
    let delta_matrix = Array2::from_shape_fn((n, n), |(i, j)| {
        delta_gamma_inverse[(i, j)] * coherence[(i, j)] + gamma_inverse[(i, j)] * delta[(i, j)]
    });
    Ok((hadamard(&gamma_inverse, coherence), delta_matrix))
}

fn quadratic_cross(
    left: ndarray::ArrayView1<Cf64>,
    matrix: ArrayView2<Cf64>,
    right: ndarray::ArrayView1<Cf64>,
) -> Cf64 {
    (0..matrix.nrows())
        .flat_map(|i| (0..matrix.ncols()).map(move |j| left[i].conj() * matrix[(i, j)] * right[j]))
        .sum()
}

/// `Γ = |C|`, regularized `(1-β)Γ + βI` and thresholded below the zero cutoff.
fn regularized_gamma(c: ArrayView2<Cf64>, beta: f64, zero_cut: f64) -> Mat<f64> {
    let n = c.nrows();
    Mat::from_fn(n, n, |i, j| {
        let mag = c[(i, j)].norm();
        let reg = (1.0 - beta) * mag + beta * f64::from(i == j);
        let val = if beta > 0.0 { reg } else { mag };
        snap_zero(val, zero_cut)
    })
}

/// Snap a correlation magnitude below `cut` to zero (dolphin's clipping).
fn snap_zero(val: f64, cut: f64) -> f64 {
    match val < cut {
        true => 0.0,
        false => val,
    }
}

/// Invert a real SPD matrix via Cholesky; `None` if not positive definite or
/// the inverse is non-finite (dolphin's NaN-triggered EVD fallback).
fn invert_spd(gamma: &Mat<f64>) -> Option<Mat<f64>> {
    let n = gamma.nrows();
    let chol = gamma.cholesky(Side::Lower).ok()?;
    let inv = chol.solve(Mat::<f64>::identity(n, n));
    let finite = (0..n).all(|i| (0..n).all(|j| inv[(i, j)].is_finite()));
    finite.then_some(inv)
}

/// Hadamard product `C ⊙ |C|` as a faer Hermitian matrix.
fn hadamard_abs(c: ArrayView2<Cf64>) -> Mat<c64> {
    let n = c.nrows();
    Mat::from_fn(n, n, |i, j| {
        let z = c[(i, j)] * c[(i, j)].norm();
        c64::new(z.re, z.im)
    })
}

/// Hadamard product of a real matrix with a complex one, as a faer matrix.
fn hadamard(real: &Mat<f64>, c: ArrayView2<Cf64>) -> Mat<c64> {
    Mat::from_fn(c.nrows(), c.ncols(), |i, j| {
        let z = c[(i, j)] * real[(i, j)];
        c64::new(z.re, z.im)
    })
}

/// Selfadjoint eigendecomposition: ascending real eigenvalues + eigenvectors.
fn selfadjoint_eig(m: &Mat<c64>) -> (Vec<f64>, Mat<c64>) {
    let eig = m.selfadjoint_eigendecomposition(Side::Lower);
    let s = eig.s().column_vector();
    let vals = (0..m.nrows()).map(|i| s.read(i).re).collect();
    (vals, eig.u().to_owned())
}

fn selected_eigengap(values: &[f64], selected: usize) -> f64 {
    values
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != selected)
        .map(|(_, value)| (values[selected] - value).abs())
        .fold(f64::INFINITY, f64::min)
}

/// Extract eigenvector column `k` as an ndarray vector.
fn column(vecs: &Mat<c64>, k: usize) -> Array1<Cf64> {
    Array1::from_shape_fn(vecs.nrows(), |i| vecs[(i, k)].to_num_complex())
}

/// Scale a vector to a target L2 norm.
fn normalize_norm(vec: Array1<Cf64>, target: f64) -> Array1<Cf64> {
    let norm: f64 = vec.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    match norm > 0.0 {
        true => vec.mapv(|z| z * (target / norm)),
        false => vec,
    }
}

/// Index of the maximum value.
fn argmax(vals: &[f64]) -> usize {
    extreme(vals, |a, b| a > b)
}

/// Index of the minimum value.
fn argmin(vals: &[f64]) -> usize {
    extreme(vals, |a, b| a < b)
}

/// Index of the value selected by `better` (strict comparison vs. running best).
fn extreme(vals: &[f64], better: impl Fn(f64, f64) -> bool) -> usize {
    vals.iter().enumerate().fold(
        0,
        |best, (i, &v)| if better(v, vals[best]) { i } else { best },
    )
}

/// Reference the eigenvector phase to `reference_idx` and package the result.
fn reference(
    (vec, eigenvalue, eigengap): (Array1<Cf64>, f64, f64),
    estimator: u8,
    reference_idx: usize,
) -> PixelEstimate {
    let shift = Cf64::from_polar(1.0, -vec[reference_idx].arg());
    PixelEstimate {
        phase: vec.mapv(|z| z * shift),
        eigenvalue,
        eigengap,
        estimator,
    }
}

/// Assemble per-pixel estimates into stacked output arrays.
fn pack(estimates: Vec<PixelEstimate>, shape: (usize, usize, usize)) -> StackEstimate {
    let (out_rows, out_cols, nslc) = shape;
    let mut cpx_phase = Array3::zeros((nslc, out_rows, out_cols));
    let mut eigenvalues = Array2::zeros((out_rows, out_cols));
    let mut estimator = Array2::zeros((out_rows, out_cols));
    for (idx, est) in estimates.into_iter().enumerate() {
        let (r, col) = (idx / out_cols, idx % out_cols);
        eigenvalues[(r, col)] = est.eigenvalue;
        estimator[(r, col)] = est.estimator;
        est.phase
            .iter()
            .enumerate()
            .for_each(|(t, &z)| cpx_phase[(t, r, col)] = z);
    }
    StackEstimate {
        cpx_phase,
        eigenvalues,
        estimator,
    }
}
