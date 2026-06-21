//! CRLB (Cramér–Rao lower bound) per-pixel phase σ — port of dolphin `crlb.py`
//! and the CRLB branch of `_core.process_coherence_matrices`.
//!
//! From the Fisher information of the coherence model
//! `X = 2L · (Γ ⊙ Γ⁻¹ − I)`, where `Γ = |C|` (regularized like EMI), `L` is the
//! number of looks, and `⊙` is the Hadamard product. The per-date standard
//! deviation is `σ = sqrt(diag( (Θᵀ X Θ + ε·I)⁻¹ ))` with a 0 at the reference
//! date (`Θ` drops the reference column). On a non-positive-definite Fisher
//! matrix — a singular / fully-decorrelated `Γ` — the bound is `NaN`, mirroring
//! dolphin's v0.42 singular-matrix behaviour. CPU (`faer`, f64); a GPU CRLB path
//! is a later follow-up.

use dolphin_core::Cf64;
use faer::prelude::SpSolver;
use faer::{Mat, Side};
use ndarray::{s, Array1, Array3, ArrayView2, ArrayView4};
use rayon::prelude::*;

/// Jitter added to `Γ` before inversion (dolphin `gamma_jitter`).
const GAMMA_JITTER: f64 = 1e-6;
/// Jitter added to the Fisher information matrix before inversion (`fim_jitter`).
const FIM_JITTER: f64 = 1e-6;

/// Per-pixel CRLB σ (radians) over a coherence stack `(rows, cols, nslc, nslc)`.
///
/// Returns `(nslc, rows, cols)` band-major (date axis first, matching the linked
/// phase layout); `σ[reference_idx] = 0`, and a pixel with a singular `Γ` is
/// `NaN` on every non-reference date. `num_looks` is dolphin's conservative
/// `sqrt(half_y · half_x)`; `beta` / `zero_correlation_threshold` regularize `Γ`
/// exactly as in EMI.
#[must_use]
pub fn estimate_crlb(
    c_arrays: ArrayView4<Cf64>,
    beta: f64,
    zero_correlation_threshold: f64,
    reference_idx: usize,
    num_looks: f64,
) -> Array3<f64> {
    let (rows, cols, nslc, _) = c_arrays.dim();
    let sigmas: Vec<Array1<f64>> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            crlb_pixel(
                c_arrays.slice(s![idx / cols, idx % cols, .., ..]),
                beta,
                zero_correlation_threshold,
                reference_idx,
                num_looks,
            )
        })
        .collect();
    pack_bands(sigmas, (nslc, rows, cols))
}

/// CRLB σ vector (length `nslc`) for one pixel's coherence matrix.
pub(crate) fn crlb_pixel(
    c: ArrayView2<Cf64>,
    beta: f64,
    zero_correlation_threshold: f64,
    reference_idx: usize,
    num_looks: f64,
) -> Array1<f64> {
    let n = c.nrows();
    let gamma = regularized_abs_gamma(c, beta, zero_correlation_threshold);
    let Some(gamma_inv) = invert_pd(&gamma, GAMMA_JITTER) else {
        return nan_with_zero_ref(n, reference_idx);
    };
    let x = fisher_information(&gamma, &gamma_inv, num_looks);
    crlb_from_fisher(&x, reference_idx)
}

/// `Γ = |C|`, EMI-style regularized `(1−β)Γ + βI` and zero-thresholded.
fn regularized_abs_gamma(c: ArrayView2<Cf64>, beta: f64, zero_cut: f64) -> Mat<f64> {
    let n = c.nrows();
    Mat::from_fn(n, n, |i, j| {
        let mag = c[(i, j)].norm();
        let reg = if beta > 0.0 {
            (1.0 - beta) * mag + beta * f64::from(i == j)
        } else {
            mag
        };
        if reg < zero_cut {
            0.0
        } else {
            reg
        }
    })
}

/// `Γ⁻¹` via Cholesky of `Γ + jitter·I`; `None` if not positive definite or the
/// inverse is non-finite (dolphin's NaN-triggered degenerate case).
fn invert_pd(gamma: &Mat<f64>, jitter: f64) -> Option<Mat<f64>> {
    let n = gamma.nrows();
    let jittered = gamma + Mat::<f64>::identity(n, n) * jitter;
    let chol = jittered.cholesky(Side::Lower).ok()?;
    let inv = chol.solve(Mat::<f64>::identity(n, n));
    let finite = (0..n).all(|i| (0..n).all(|j| inv[(i, j)].is_finite()));
    finite.then_some(inv)
}

/// `X = 2L · (Γ ⊙ Γ⁻¹ − I)` (real, Hadamard product).
fn fisher_information(gamma: &Mat<f64>, gamma_inv: &Mat<f64>, num_looks: f64) -> Mat<f64> {
    let n = gamma.nrows();
    Mat::from_fn(n, n, |i, j| {
        2.0 * num_looks * (gamma[(i, j)] * gamma_inv[(i, j)] - f64::from(i == j))
    })
}

/// `σ = sqrt(diag( (Θᵀ X Θ + ε·I)⁻¹ ))`, with `σ[reference_idx] = 0`. `Θ` drops
/// the reference row/column; a non-PD Fisher matrix yields `NaN` everywhere off
/// the reference date.
fn crlb_from_fisher(x: &Mat<f64>, reference_idx: usize) -> Array1<f64> {
    let n = x.nrows();
    let idx: Vec<usize> = (0..n).filter(|&i| i != reference_idx).collect();
    let m = idx.len();
    let fim = Mat::from_fn(m, m, |a, b| {
        x[(idx[a], idx[b])] + FIM_JITTER * f64::from(a == b)
    });
    let Some(sigma) = invert_pd(&fim, 0.0) else {
        return nan_with_zero_ref(n, reference_idx);
    };
    let mut out = Array1::<f64>::zeros(n);
    for (a, &i) in idx.iter().enumerate() {
        out[i] = sigma[(a, a)].sqrt();
    }
    out[reference_idx] = 0.0;
    out
}

/// A σ vector that is `NaN` on every date except the reference (which is 0).
fn nan_with_zero_ref(n: usize, reference_idx: usize) -> Array1<f64> {
    let mut v = Array1::from_elem(n, f64::NAN);
    v[reference_idx] = 0.0;
    v
}

/// Pack per-pixel σ vectors into a `(nslc, rows, cols)` band-major array.
fn pack_bands(sigmas: Vec<Array1<f64>>, shape: (usize, usize, usize)) -> Array3<f64> {
    let (nslc, rows, cols) = shape;
    let mut out = Array3::zeros((nslc, rows, cols));
    for (idx, sig) in sigmas.into_iter().enumerate() {
        let (r, c) = (idx / cols, idx % cols);
        for t in 0..nslc {
            out[(t, r, c)] = sig[t];
        }
    }
    out
}
