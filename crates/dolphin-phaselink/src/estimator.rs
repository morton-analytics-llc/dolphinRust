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

/// Per-pixel phase-linking result.
#[derive(Debug, Clone)]
pub struct PixelEstimate {
    /// Referenced wrapped phase, length `nslc`.
    pub phase: Array1<Cf64>,
    /// The dominant (EVD) or least (EMI) eigenvalue.
    pub eigenvalue: f64,
    /// Estimator used: 0 = EVD, 1 = EMI.
    pub estimator: u8,
}

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
    let evd = evd_eigenvector(c);
    if use_evd {
        return reference(evd, 0, reference_idx);
    }
    match emi_eigenvector(c, beta, zero_correlation_threshold) {
        Some(emi) => reference(emi, 1, reference_idx),
        None => reference(evd, 0, reference_idx),
    }
}

/// `(eigenvector, eigenvalue)` of the dominant mode of `C ⊙ |C|`.
fn evd_eigenvector(c: ArrayView2<Cf64>) -> (Array1<Cf64>, f64) {
    let m = hadamard_abs(c);
    let (vals, vecs) = selfadjoint_eig(&m);
    let k = argmax(&vals);
    (column(&vecs, k), vals[k])
}

/// `(eigenvector, eigenvalue)` of the least mode of `Γ⁻¹ ⊙ C`, or `None` if
/// `Γ = |C|` cannot be inverted.
fn emi_eigenvector(
    c: ArrayView2<Cf64>,
    beta: f64,
    zero_correlation_threshold: f64,
) -> Option<(Array1<Cf64>, f64)> {
    let n = c.nrows();
    let gamma = regularized_gamma(c, beta, zero_correlation_threshold);
    let gamma_inv = invert_spd(&gamma)?;
    let m = hadamard(&gamma_inv, c);
    let (vals, vecs) = selfadjoint_eig(&m);
    let k = argmin(&vals);
    let vec = normalize_norm(column(&vecs, k), (n as f64).sqrt());
    Some((vec, vals[k]))
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
    (vec, eigenvalue): (Array1<Cf64>, f64),
    estimator: u8,
    reference_idx: usize,
) -> PixelEstimate {
    let shift = Cf64::from_polar(1.0, -vec[reference_idx].arg());
    PixelEstimate {
        phase: vec.mapv(|z| z * shift),
        eigenvalue,
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
