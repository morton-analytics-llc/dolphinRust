//! Sliding-window sample-coherence estimation (port of `covariance.py`).
//!
//! For each (strided) output pixel, a `(2*half.y+1) x (2*half.x+1)` window is
//! read from the stack (clamped inward at borders, matching JAX
//! `dynamic_slice`), flattened to `(nslc, nsamples)`, and reduced to the
//! normalized coherence matrix `C_ij = Σ z_i z_j* / sqrt(Σ|z_i|² · Σ|z_j|²)`.
//! Parallelized over output pixels with `rayon` — the Rust analogue of dolphin's
//! `vmap(vmap(f))`. All math in `Cf64`.

use dolphin_core::{Cf64, HalfWindow, Strides};
use ndarray::{s, Array2, Array4, ArrayView2, ArrayView3};
use rayon::prelude::*;

/// Amplitude floor below which a coherence entry is set to 0 (dolphin uses 1e-6).
const AMP_FLOOR: f64 = 1e-6;

/// Estimate the per-pixel coherence matrix over a sliding window.
///
/// `stack` is `(nslc, rows, cols)`. Returns `(out_rows, out_cols, nslc, nslc)`
/// where the output grid is decimated by `strides`. A rectangular window is
/// used (SHP neighbor masking is wired in Phase 2).
///
/// # Errors
/// Returns `Err` if the window is larger than the stack in either dimension.
pub fn estimate_stack_covariance(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
) -> Result<Array4<Cf64>, &'static str> {
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    if win_h > rows || win_w > cols {
        return Err("covariance window larger than stack");
    }
    let (out_rows, out_cols) = strides.out_shape((rows, cols));

    let mats: Vec<Array2<Cf64>> = (0..out_rows * out_cols)
        .into_par_iter()
        .map(|idx| pixel_coh(stack, (idx / out_cols, idx % out_cols), half, strides))
        .collect();

    assemble(mats, (out_rows, out_cols, nslc))
}

/// Coherence matrix for a single output pixel `out = (out_r, out_c)`.
fn pixel_coh(
    stack: ArrayView3<Cf64>,
    out: (usize, usize),
    half: HalfWindow,
    strides: Strides,
) -> Array2<Cf64> {
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    let (r0, c0) = window_origin(out, half, strides, (rows, cols));
    let window = stack.slice(s![.., r0..r0 + win_h, c0..c0 + win_w]);
    coh_mat(window, nslc)
}

/// Top-left corner of the window for an output pixel, clamped inward at borders
/// so the window stays full-size (matches JAX `dynamic_slice` clamping).
fn window_origin(
    out: (usize, usize),
    half: HalfWindow,
    strides: Strides,
    shape: (usize, usize),
) -> (usize, usize) {
    let in_r = strides.y / 2 + out.0 * strides.y;
    let in_c = strides.x / 2 + out.1 * strides.x;
    let r0 = in_r.saturating_sub(half.y).min(shape.0 - (2 * half.y + 1));
    let c0 = in_c.saturating_sub(half.x).min(shape.1 - (2 * half.x + 1));
    (r0, c0)
}

/// Coherence matrix from a `(nslc, win_h, win_w)` window (port of `coh_mat_single`).
fn coh_mat(window: ArrayView3<Cf64>, nslc: usize) -> Array2<Cf64> {
    let samples = window
        .to_shape((nslc, window.len() / nslc))
        .expect("contiguous window reshape")
        .mapv(finite_or_zero);

    let conj_t = samples.t().mapv(|z| z.conj());
    let numer = samples.dot(&conj_t);
    normalize(numer.view())
}

/// Replace non-finite samples (NaN/Inf) with zero, matching dolphin's masking.
fn finite_or_zero(z: Cf64) -> Cf64 {
    match z.is_finite() {
        true => z,
        false => Cf64::new(0.0, 0.0),
    }
}

/// Normalize a cross-correlation matrix to a coherence matrix.
fn normalize(numer: ArrayView2<Cf64>) -> Array2<Cf64> {
    let n = numer.nrows();
    let amp: Vec<f64> = (0..n).map(|i| numer[(i, i)].re.max(0.0).sqrt()).collect();
    Array2::from_shape_fn((n, n), |(i, j)| {
        coherence_entry(numer[(i, j)], amp[i] * amp[j])
    })
}

/// One normalized coherence entry: `numer / denom`, or 0 when `denom` underflows.
fn coherence_entry(numer: Cf64, denom: f64) -> Cf64 {
    match denom > AMP_FLOOR {
        true => numer / denom,
        false => Cf64::new(0.0, 0.0),
    }
}

/// Stack per-pixel `(n, n)` matrices into an `(out_rows, out_cols, n, n)` array.
fn assemble(
    mats: Vec<Array2<Cf64>>,
    shape: (usize, usize, usize),
) -> Result<Array4<Cf64>, &'static str> {
    let (out_rows, out_cols, n) = shape;
    let flat: Vec<Cf64> = mats.into_iter().flat_map(IntoIterator::into_iter).collect();
    Array4::from_shape_vec((out_rows, out_cols, n, n), flat)
        .map_err(|_| "covariance assembly shape mismatch")
}
