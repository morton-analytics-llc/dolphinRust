//! Quality layers: temporal coherence (`metrics.py`) and compressed SLC
//! (`_compress.py`).
//!
//! Temporal coherence measures how well the linked phase reproduces the
//! observed interferometric phases: `|Σ_{i<j} e^{j(∠C_ij − (θ_i−θ_j))}| / N_pairs`
//! (equal weights, dolphin's default). The compressed SLC projects the stack
//! onto the linked phase: magnitude from the mean amplitude, phase from
//! `∠Σ_k z_k · conj(θ_k)`.
//!
//! CRLB uncertainty and sequential closure phase live in sibling modules
//! [`crate::crlb`] and [`crate::closure`] (validated against the v0.42.0 oracle).

use dolphin_core::Cf64;
use ndarray::{s, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use rayon::prelude::*;

/// Temporal coherence per pixel from the linked phase and coherence matrices.
///
/// `cpx_phase` is `(rows, cols, nslc)` (dolphin's pre-moveaxis layout);
/// `c_arrays` is `(rows, cols, nslc, nslc)`. Returns `(rows, cols)`.
#[must_use]
pub fn estimate_temp_coh(cpx_phase: ArrayView3<Cf64>, c_arrays: ArrayView4<Cf64>) -> Array2<f64> {
    let (rows, cols, _) = cpx_phase.dim();
    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let (r, c) = (idx / cols, idx % cols);
            temp_coh_single(
                cpx_phase.slice(s![r, c, ..]),
                c_arrays.slice(s![r, c, .., ..]),
            )
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("temp_coh shape")
}

/// Temporal coherence for one pixel (equal weights, upper triangle).
pub(crate) fn temp_coh_single(phase: ArrayView1<Cf64>, c: ArrayView2<Cf64>) -> f64 {
    let n = phase.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let sum: Cf64 = pairs.iter().map(|&(i, j)| pair_diff(phase, c, i, j)).sum();
    nan_to_num(sum.norm() / pairs.len() as f64)
}

/// Unit phasor for the residual between observed and reformed ifg phase at `(i, j)`.
fn pair_diff(phase: ArrayView1<Cf64>, c: ArrayView2<Cf64>, i: usize, j: usize) -> Cf64 {
    let reformed = (phase[i] * phase[j].conj()).arg();
    Cf64::from_polar(1.0, c[(i, j)].arg() - reformed)
}

/// Compressed SLC: project the stack onto the linked phase (port of `compress`).
///
/// `slc_stack` is `(nslc, rows, cols)`; `pl_cpx_phase` is `(nslc, out_rows,
/// out_cols)` (upsampled to full resolution). `first_real_slc_idx` excludes
/// leading compressed layers from the projection; `reference_idx` optionally
/// re-references the phase first. Returns the compressed SLC `(rows, cols)`.
#[must_use]
pub fn compress(
    slc_stack: ArrayView3<Cf64>,
    pl_cpx_phase: ArrayView3<Cf64>,
    first_real_slc_idx: usize,
    reference_idx: Option<usize>,
) -> Array2<Cf64> {
    let (_, rows, cols) = slc_stack.dim();
    let referenced = rereference(pl_cpx_phase, reference_idx);
    let upsampled = upsample_nearest(referenced.view(), (rows, cols));

    let values: Vec<Cf64> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let (r, c) = (idx / cols, idx % cols);
            compress_pixel(slc_stack, upsampled.view(), first_real_slc_idx, (r, c))
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("compressed shape")
}

/// Optionally re-reference the linked phase to `reference_idx`.
fn rereference(pl: ArrayView3<Cf64>, reference_idx: Option<usize>) -> Array3<Cf64> {
    let Some(ref_idx) = reference_idx else {
        return pl.to_owned();
    };
    let reference = pl.slice(s![ref_idx, .., ..]).to_owned();
    Array3::from_shape_fn(pl.dim(), |(t, r, c)| {
        pl[(t, r, c)] * reference[(r, c)].conj()
    })
}

/// One compressed-SLC pixel: mean magnitude × `exp(j ∠Σ z_k conj(θ_k))`.
fn compress_pixel(
    slc_stack: ArrayView3<Cf64>,
    upsampled: ArrayView3<Cf64>,
    first: usize,
    pixel: (usize, usize),
) -> Cf64 {
    let (nslc, r, c) = (slc_stack.dim().0, pixel.0, pixel.1);
    let acc: Cf64 = (first..nslc)
        .map(|t| finite_or_zero(slc_stack[(t, r, c)] * upsampled[(t, r, c)].conj()))
        .sum();
    let mag_sum: f64 = (first..nslc).map(|t| slc_stack[(t, r, c)].norm()).sum();
    let count = (nslc - first) as f64;
    let phase = acc.arg();
    let mean = if phase == 0.0 {
        f64::NAN
    } else {
        mag_sum / count
    };
    Cf64::from_polar(mean, phase)
}

/// Replace a non-finite complex value with zero (dolphin's `nansum` skip).
fn finite_or_zero(z: Cf64) -> Cf64 {
    match z.is_finite() {
        true => z,
        false => Cf64::new(0.0, 0.0),
    }
}

/// Nearest-neighbor upsample of `(nslc, in_rows, in_cols)` to `(out_rows, out_cols)`
/// by integer block repeat (port of `utils.upsample_nearest`).
fn upsample_nearest(arr: ArrayView3<Cf64>, output_shape: (usize, usize)) -> Array3<Cf64> {
    let (nslc, in_rows, in_cols) = arr.dim();
    let (out_rows, out_cols) = output_shape;
    if (in_rows, in_cols) == (out_rows, out_cols) {
        return arr.to_owned();
    }
    let row_looks = (out_rows / in_rows).max(1);
    let col_looks = (out_cols / in_cols).max(1);
    Array3::from_shape_fn((nslc, out_rows, out_cols), |(t, r, c)| {
        arr[(
            t,
            (r / row_looks).min(in_rows - 1),
            (c / col_looks).min(in_cols - 1),
        )]
    })
}

/// Replace NaN/±inf with 0.
fn nan_to_num(v: f64) -> f64 {
    match v.is_finite() {
        true => v,
        false => 0.0,
    }
}
