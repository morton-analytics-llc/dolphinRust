//! Two-sample Kolmogorov–Smirnov SHP test (port of `shp/_ks.py`).
//!
//! Per-pixel amplitude time series are sorted; a neighbor is an SHP when the
//! maximum empirical-CDF distance to the center series is below the Kolmogorov
//! critical distance for `nslc` samples at level `alpha`. Parallel over center
//! pixels (dolphin's numba `prange` loop). Border windows (clipped below full
//! size) yield no neighbors, matching dolphin.

use dolphin_core::{HalfWindow, Strides};
use ndarray::{s, Array2, Array3, Array4, ArrayView3, Axis};
use rayon::prelude::*;

use crate::window::{clamped_window, neighbor_grid, stack_slabs};

/// Estimate the SHP neighbor mask via the KS test.
///
/// `amp_stack` is `(nslc, rows, cols)`. When `is_sorted`, the per-pixel time
/// series are assumed already ascending. Returns
/// `(out_rows, out_cols, 2*half.y+1, 2*half.x+1)`.
#[must_use]
pub fn estimate_neighbors_ks(
    amp_stack: ArrayView3<f64>,
    half: HalfWindow,
    strides: Strides,
    alpha: f64,
    is_sorted: bool,
) -> Array4<bool> {
    let (nslc, rows, cols) = amp_stack.dim();
    let sorted = sort_pixels(amp_stack, is_sorted);
    let cutoff = ecdf_critical_distance(nslc, alpha);
    let (out_rows, out_cols) = strides.out_shape((rows, cols));
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);

    let slabs: Vec<Array2<bool>> = (0..out_rows * out_cols)
        .into_par_iter()
        .map(|idx| {
            let out = (idx / out_cols, idx % out_cols);
            ks_pixel(sorted.view(), out, half, strides, cutoff)
        })
        .collect();

    stack_slabs(slabs, (out_rows, out_cols, win_h, win_w))
}

/// Sort each pixel's amplitude time series ascending (no-op if already sorted).
fn sort_pixels(amp_stack: ArrayView3<f64>, is_sorted: bool) -> Array3<f64> {
    let mut sorted = amp_stack.to_owned();
    if is_sorted {
        return sorted;
    }
    sorted.lanes_mut(Axis(0)).into_iter().for_each(|mut lane| {
        let mut v = lane.to_vec();
        v.sort_by(f64::total_cmp);
        lane.iter_mut().zip(v).for_each(|(d, s)| *d = s);
    });
    sorted
}

/// KS neighbor slab for one center (output) pixel.
fn ks_pixel(
    sorted: ArrayView3<f64>,
    out: (usize, usize),
    half: HalfWindow,
    strides: Strides,
    cutoff: f64,
) -> Array2<bool> {
    let (_, rows, cols) = sorted.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    let mut slab = Array2::from_elem((win_h, win_w), false);
    let center = (
        strides.y / 2 + out.0 * strides.y,
        strides.x / 2 + out.1 * strides.x,
    );
    let win = clamped_window(center, half, (rows, cols));
    if win.r_end - win.r_start < win_h || win.c_end - win.c_start < win_w {
        return slab; // border window too small: no neighbors (dolphin behavior)
    }
    let x1 = sorted.slice(s![.., center.0, center.1]).to_vec();
    neighbor_grid(win).for_each(|(r, c, r_off, c_off)| {
        let neighbor = (r, c) != center;
        let x2 = sorted.slice(s![.., r, c]).to_vec();
        slab[(r_off, c_off)] = neighbor && max_cdf_dist(&x1, &x2) < cutoff;
    });
    slab
}

/// Maximum empirical-CDF distance between two equal-length sorted samples
/// (direct port of dolphin's `_get_max_cdf_dist` merge walk).
fn max_cdf_dist(x1: &[f64], x2: &[f64]) -> f64 {
    let n = x1.len();
    let step = 1.0 / n as f64;
    let mut w = Walk::default();
    while w.out < 2 * n {
        w.advance(x1, x2, n, step);
    }
    w.max_dist
}

/// Mutable state of the two-sample ECDF merge walk.
#[derive(Default)]
struct Walk {
    i1: usize,
    i2: usize,
    out: usize,
    cdf1: f64,
    cdf2: f64,
    max_dist: f64,
}

impl Walk {
    /// Consume the next value(s) per dolphin's if/elif/else (ties advance both).
    fn advance(&mut self, x1: &[f64], x2: &[f64], n: usize, step: f64) {
        if self.i1 == n {
            self.bump_right(step);
        } else if self.i2 == n || x1[self.i1] < x2[self.i2] {
            self.bump_left(step);
        } else if x1[self.i1] > x2[self.i2] {
            self.bump_right(step);
        } else {
            self.bump_left(step);
            self.bump_right(step);
            self.out += 1; // tie jumps two ahead
        }
        self.out += 1;
        self.max_dist = self.max_dist.max((self.cdf1 - self.cdf2).abs());
    }

    fn bump_left(&mut self, step: f64) {
        self.cdf1 += step;
        self.i1 += 1;
    }

    fn bump_right(&mut self, step: f64) {
        self.cdf2 += step;
        self.i2 += 1;
    }
}

/// Kolmogorov critical ECDF distance for `nslc` samples at level `alpha`
/// (port of dolphin's iterative `_get_ecdf_critical_distance`).
fn ecdf_critical_distance(nslc: usize, alpha: f64) -> f64 {
    let sqrt_n = (nslc as f64 / 2.0).sqrt();
    let mut cur = 0.01_f64;
    while cur <= 1.0 {
        let value = cur * (sqrt_n + 0.12 + 0.11 / sqrt_n);
        if kolmogorov_pvalue(value) <= alpha {
            return cur;
        }
        cur += 0.001;
    }
    0.1
}

/// Two-sided Kolmogorov p-value `2 Σ (-1)^(t-1) exp(-2 v² t²)`, clamped to [0, 1].
fn kolmogorov_pvalue(value: f64) -> f64 {
    let series: f64 = (1..=100)
        .map(|t| (-1.0_f64).powi(t - 1) * (-2.0 * value * value * (t * t) as f64).exp())
        .sum();
    (2.0 * series).clamp(0.0, 1.0)
}
