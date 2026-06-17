//! GLRT statistically-homogeneous-pixel test (port of `shp/_glrt.py`).
//!
//! Rayleigh amplitude model: scale `σ² = (var + mean²)/2`. For a center pixel
//! and each window neighbor, the test statistic is
//! `T = N(2 ln σ²_pooled − ln σ²_1 − ln σ²_2)` with `σ²_pooled = (σ²_1+σ²_2)/2`;
//! the neighbor is an SHP when `T < χ²(1, 1−α)`. Parallel over center pixels.

use dolphin_core::{HalfWindow, Strides};
use ndarray::{Array2, Array4, ArrayView2};
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::window::{clamped_window, neighbor_grid};

/// Estimate the SHP neighbor mask via the GLRT.
///
/// `mean`/`var` are the per-pixel temporal amplitude mean and variance over
/// `nslc` images. Returns `(out_rows, out_cols, 2*half.y+1, 2*half.x+1)`.
#[must_use]
pub fn estimate_neighbors_glrt(
    mean: ArrayView2<f64>,
    var: ArrayView2<f64>,
    half: HalfWindow,
    nslc: usize,
    strides: Strides,
    alpha: f64,
) -> Array4<bool> {
    let threshold = chi2_critical(alpha);
    let scale_sq = scale_squared(mean, var);
    let (rows, cols) = mean.dim();
    let (out_rows, out_cols) = strides.out_shape((rows, cols));
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);

    let slabs: Vec<Array2<bool>> = (0..out_rows * out_cols)
        .into_par_iter()
        .map(|idx| {
            let out = (idx / out_cols, idx % out_cols);
            glrt_pixel(scale_sq.view(), mean, out, half, strides, nslc, threshold)
        })
        .collect();

    crate::window::stack_slabs(slabs, (out_rows, out_cols, win_h, win_w))
}

/// Rayleigh scale parameter squared `(var + mean²)/2`.
fn scale_squared(mean: ArrayView2<f64>, var: ArrayView2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(mean.dim(), |ix| (var[ix] + mean[ix] * mean[ix]) / 2.0)
}

/// χ²(df=1) critical value at confidence `1−alpha`.
fn chi2_critical(alpha: f64) -> f64 {
    ChiSquared::new(1.0)
        .expect("df=1 is valid")
        .inverse_cdf(1.0 - alpha)
}

/// GLRT neighbor slab for one center (output) pixel.
fn glrt_pixel(
    scale_sq: ArrayView2<f64>,
    mean: ArrayView2<f64>,
    out: (usize, usize),
    half: HalfWindow,
    strides: Strides,
    nslc: usize,
    threshold: f64,
) -> Array2<bool> {
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    let mut slab = Array2::from_elem((win_h, win_w), false);
    let center = (
        strides.y / 2 + out.0 * strides.y,
        strides.x / 2 + out.1 * strides.x,
    );
    if mean[center] == 0.0 {
        return slab; // nodata center: no neighbors
    }
    let win = clamped_window(center, half, mean.dim());
    let scale_1 = scale_sq[center];
    neighbor_grid(win).for_each(|(r, c, r_off, c_off)| {
        slab[(r_off, c_off)] = is_shp(scale_1, scale_sq[(r, c)], nslc, threshold, (r, c) != center);
    });
    slab
}

/// Whether a neighbor passes the GLRT (false for the center pixel itself).
fn is_shp(scale_1: f64, scale_2: f64, nslc: usize, threshold: f64, is_neighbor: bool) -> bool {
    is_neighbor && threshold > test_stat(scale_1, scale_2, nslc)
}

/// GLRT test statistic `N(2 ln pooled − ln s1 − ln s2)`.
fn test_stat(scale_sq_1: f64, scale_sq_2: f64, nslc: usize) -> f64 {
    let pooled = (scale_sq_1 + scale_sq_2) / 2.0;
    nslc as f64 * (2.0 * pooled.ln() - scale_sq_1.ln() - scale_sq_2.ln())
}
