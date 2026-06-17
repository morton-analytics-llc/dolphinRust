//! SBAS L2 inversion and velocity (port of `timeseries.py`).
//!
//! The incidence matrix `A (n_ifgs × n_dates−1)` has -1 on the earlier date and
//! +1 on the later date of each ifg (the first date's column is dropped → it is
//! the zero-phase reference). The weighted least-squares solve `min ‖√W(Aφ−Δφ)‖`
//! is done per pixel; velocity is the slope of the displacement series.
//!
//! NOTE: dolphin defaults to L1; this is the L2 path (the documented temporary
//! divergence — L1/ADMM is Phase 6b). The solve uses normal equations +
//! Cholesky (full-rank `A`), equivalent to dolphin's `lstsq` to tolerance.

use std::collections::BTreeSet;

use faer::prelude::SpSolver;
use faer::{Mat, Side};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use rayon::prelude::*;

/// Build the incidence matrix from interferogram index pairs (port of
/// `get_incidence_matrix`, dropping the first date's column).
#[must_use]
pub fn get_incidence_matrix(pairs: &[(usize, usize)]) -> Array2<f64> {
    let sar_idxs: Vec<usize> = pairs
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let col_of: std::collections::HashMap<usize, usize> = sar_idxs
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, &d)| (d, c))
        .collect();

    let n_cols = sar_idxs.len() - 1;
    let mut a = Array2::zeros((pairs.len(), n_cols));
    for (row, &(early, later)) in pairs.iter().enumerate() {
        if let Some(&c) = col_of.get(&early) {
            a[(row, c)] = -1.0;
        }
        if let Some(&c) = col_of.get(&later) {
            a[(row, c)] = 1.0;
        }
    }
    a
}

/// Solve the SBAS stack `A φ = Δφ` per pixel (L2, optional per-pixel weights).
/// `dphi` is `(n_ifgs, rows, cols)`; returns `(n_dates−1, rows, cols)`.
#[must_use]
pub fn invert_stack(
    a: ArrayView2<f64>,
    dphi: ArrayView3<f64>,
    weights: Option<ArrayView3<f64>>,
) -> Array3<f64> {
    let (n_ifgs, rows, cols) = dphi.dim();
    let n_dates = a.ncols();
    let columns: Vec<Vec<f64>> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| solve_pixel(a, dphi, weights, (idx / cols, idx % cols), n_ifgs))
        .collect();

    Array3::from_shape_fn((n_dates, rows, cols), |(d, r, c)| columns[r * cols + c][d])
}

/// ADMM parameters for L1 (least-absolute-deviations) inversion. Defaults match
/// dolphin's `least_absolute_deviations` (`rho=0.4`, `alpha=1.0`, 20 iterations);
/// the network structure is regular enough that ADMM converges in few steps.
#[derive(Debug, Clone, Copy)]
pub struct L1Config {
    /// Augmented-Lagrangian penalty parameter ρ.
    pub rho: f64,
    /// Over-relaxation parameter α (typically 1.0–1.8).
    pub alpha: f64,
    /// Fixed ADMM iteration count.
    pub max_iter: usize,
}

impl Default for L1Config {
    fn default() -> Self {
        Self {
            rho: 0.4,
            alpha: 1.0,
            max_iter: 20,
        }
    }
}

/// Soft-thresholding (shrinkage) operator `max(0,a−κ) − max(0,−a−κ)`.
fn shrinkage(a: f64, kappa: f64) -> f64 {
    (a - kappa).max(0.0) - (-a - kappa).max(0.0)
}

/// Solve the SBAS stack in the **L1 norm** (`min ‖Aφ−Δφ‖₁`) per pixel via
/// ADMM/LAD — dolphin's default inversion, robust to unwrapping outliers.
/// `dphi` is `(n_ifgs, rows, cols)`; returns `(n_dates, rows, cols)`. Port of
/// dolphin `least_absolute_deviations` / `invert_stack_l1`.
#[must_use]
pub fn invert_stack_l1(a: ArrayView2<f64>, dphi: ArrayView3<f64>, cfg: L1Config) -> Array3<f64> {
    let (n_ifgs, rows, cols) = dphi.dim();
    let n = a.ncols();
    let ata = Mat::from_fn(n, n, |i, j| {
        (0..n_ifgs).map(|k| a[(k, i)] * a[(k, j)]).sum::<f64>()
    });
    let llt = ata
        .cholesky(Side::Lower)
        .expect("AtA not SPD (rank-deficient network)");
    let columns: Vec<Vec<f64>> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| lad_pixel(a, dphi, &llt, (idx / cols, idx % cols), cfg))
        .collect();
    Array3::from_shape_fn((n, rows, cols), |(d, r, c)| columns[r * cols + c][d])
}

/// Least-absolute-deviations ADMM solve for one pixel.
fn lad_pixel(
    a: ArrayView2<f64>,
    dphi: ArrayView3<f64>,
    llt: &faer::linalg::solvers::Cholesky<f64>,
    pixel: (usize, usize),
    cfg: L1Config,
) -> Vec<f64> {
    let n = a.ncols();
    let m = a.nrows();
    let b: Vec<f64> = (0..m).map(|k| dphi[(k, pixel.0, pixel.1)]).collect();
    let mut x = vec![0.0; n];
    let mut z = vec![0.0; m];
    let mut z_old = vec![0.0; m];
    let mut u = vec![0.0; m];
    let kappa = 1.0 / cfg.rho;

    for _ in 0..cfg.max_iter {
        let q = Mat::from_fn(n, 1, |i, _| {
            (0..m)
                .map(|k| a[(k, i)] * (b[k] + z[k] - u[k]))
                .sum::<f64>()
        });
        let xs = llt.solve(&q);
        (0..n).for_each(|i| x[i] = xs[(i, 0)]);

        let mut z_new = vec![0.0; m];
        for k in 0..m {
            let ax = (0..n).map(|i| a[(k, i)] * x[i]).sum::<f64>();
            let ax_hat = cfg.alpha * ax + (1.0 - cfg.alpha) * (z_old[k] + b[k]);
            z_new[k] = shrinkage(ax_hat - b[k] + u[k], kappa);
            u[k] += ax_hat - z_new[k] - b[k];
        }
        z_old = z;
        z = z_new;
    }
    x
}

/// Weighted least-squares solve for one pixel.
fn solve_pixel(
    a: ArrayView2<f64>,
    dphi: ArrayView3<f64>,
    weights: Option<ArrayView3<f64>>,
    pixel: (usize, usize),
    n_ifgs: usize,
) -> Vec<f64> {
    let n = a.ncols();
    let w = |k: usize| weights.map_or(1.0, |ws| ws[(k, pixel.0, pixel.1)]);
    let ata = Mat::from_fn(n, n, |i, j| {
        (0..n_ifgs)
            .map(|k| a[(k, i)] * w(k) * a[(k, j)])
            .sum::<f64>()
    });
    let atb = Mat::from_fn(n, 1, |i, _| {
        (0..n_ifgs)
            .map(|k| a[(k, i)] * w(k) * dphi[(k, pixel.0, pixel.1)])
            .sum::<f64>()
    });
    let x = ata
        .cholesky(Side::Lower)
        .expect("AtWA not SPD (rank-deficient network)")
        .solve(atb);
    (0..n).map(|i| x[(i, 0)]).collect()
}

/// Per-pixel linear velocity (slope × 365.25) of a displacement series.
/// `series` is `(n_time, rows, cols)`; `x` are the time positions (days).
#[must_use]
pub fn estimate_velocity(
    x: &[f64],
    series: ArrayView3<f64>,
    weights: Option<ArrayView3<f64>>,
) -> Array2<f64> {
    let (_, rows, cols) = series.dim();
    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| velocity_pixel(x, series, weights, (idx / cols, idx % cols)))
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("velocity shape")
}

/// Slope of a weighted degree-1 fit (numpy `polyfit` weighting), scaled to /year.
fn velocity_pixel(
    x: &[f64],
    series: ArrayView3<f64>,
    weights: Option<ArrayView3<f64>>,
    pixel: (usize, usize),
) -> f64 {
    let w = |t: usize| weights.map_or(1.0, |ws| ws[(t, pixel.0, pixel.1)]);
    let y = |t: usize| series[(t, pixel.0, pixel.1)];
    // Normal equations for min Σ w²(y - m x - c)² (numpy scales rows by w).
    let (mut sww, mut swx, mut swxx, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (t, &xt) in x.iter().enumerate() {
        let ww = w(t) * w(t);
        sww += ww;
        swx += ww * xt;
        swxx += ww * xt * xt;
        swy += ww * y(t);
        swxy += ww * xt * y(t);
    }
    let det = sww * swxx - swx * swx;
    let slope = (sww * swxy - swx * swy) / det;
    slope * 365.25
}
