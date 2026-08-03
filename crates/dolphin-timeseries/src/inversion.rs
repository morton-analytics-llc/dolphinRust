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

/// Full weighted-L2 solution for one pixel. The covariance is kept on demand
/// rather than materialized as an `n_dates^2 * area` workflow cube.
#[derive(Debug, Clone)]
pub struct PixelL2Solution {
    /// Fitted phase parameters.
    pub parameters: Vec<f64>,
    /// Full posterior parameter covariance.
    pub covariance: Array2<f64>,
    /// Residual root-mean-square in observation units.
    pub residual_rms: f64,
}

/// Bounded stack-level uncertainty products.
pub struct L2InversionOutput {
    /// Fitted phase stack.
    pub phase: Array3<f64>,
    /// Diagonal posterior parameter variance.
    pub posterior_variance: Array3<f64>,
    /// Residual root-mean-square in observation units.
    pub residual_rms: Array2<f64>,
}

/// Linear-rate estimate and its one-sigma standard error, both per year.
pub struct VelocityOutput {
    /// Linear velocity per year.
    pub velocity: Array2<f64>,
    /// One-sigma slope uncertainty per year.
    pub sigma: Array2<f64>,
    /// Regression residual root-mean-square in series units.
    pub residual_rms: Array2<f64>,
}

/// [`VelocityOutput`] plus an opt-in temporal-correlation (N_eff) correction.
///
/// InSAR displacement series carry temporally correlated (e.g. atmospheric) noise,
/// so `dof = n_valid_dates - 2` overstates the number of independent observations
/// and `sigma` alone understates the true slope uncertainty (Agram & Zebker 2015).
/// `sigma` is untouched here (regression-safe with [`estimate_velocity_with_uncertainty`]);
/// `sigma_temporal_corrected` and `inflation_factor` are reported alongside it rather
/// than silently replacing it, since a larger `sigma` can flip a downstream risk-tier
/// threshold and that change needs its own reviewed rollout.
pub struct VelocityOutputNeff {
    /// Linear velocity per year (identical to [`VelocityOutput::velocity`]).
    pub velocity: Array2<f64>,
    /// One-sigma slope uncertainty per year, uncorrected (identical to
    /// [`VelocityOutput::sigma`]).
    pub sigma: Array2<f64>,
    /// `sigma * inflation_factor` — the temporal-correlation-corrected one-sigma
    /// slope uncertainty per year.
    pub sigma_temporal_corrected: Array2<f64>,
    /// `sqrt(n_valid / n_eff) = sqrt((1 + rho) / (1 - rho))`, `rho` the estimated
    /// lag-1 residual autocorrelation clamped to `[0, 1)`. `1.0` where residuals are
    /// uncorrelated or too few to estimate `rho`.
    pub inflation_factor: Array2<f64>,
    /// Regression residual root-mean-square in series units (identical to
    /// [`VelocityOutput::residual_rms`]).
    pub residual_rms: Array2<f64>,
}

struct PixelUncertaintySummary {
    parameters: Vec<f64>,
    variance: Vec<f64>,
    residual_rms: f64,
}

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

/// Solve weighted L2 while retaining only covariance diagonals at stack scale.
#[must_use]
pub fn invert_stack_with_uncertainty(
    a: ArrayView2<f64>,
    dphi: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
) -> L2InversionOutput {
    let (n_ifgs, rows, cols) = dphi.dim();
    let n_dates = a.ncols();
    let columns: Vec<Option<PixelUncertaintySummary>> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            solve_pixel_with_covariance(a, dphi, Some(precisions), (idx / cols, idx % cols), n_ifgs)
                .map(|solution| {
                    let diagonal = (0..n_dates)
                        .map(|date| solution.covariance[(date, date)])
                        .collect();
                    PixelUncertaintySummary {
                        parameters: solution.parameters,
                        variance: diagonal,
                        residual_rms: solution.residual_rms,
                    }
                })
        })
        .collect();
    let phase = Array3::from_shape_fn((n_dates, rows, cols), |(d, r, c)| {
        columns[r * cols + c]
            .as_ref()
            .map_or(f64::NAN, |value| value.parameters[d])
    });
    let posterior_variance = Array3::from_shape_fn((n_dates, rows, cols), |(d, r, c)| {
        columns[r * cols + c]
            .as_ref()
            .map_or(f64::NAN, |value| value.variance[d])
    });
    let residual_rms = Array2::from_shape_fn((rows, cols), |(r, c)| {
        columns[r * cols + c]
            .as_ref()
            .map_or(f64::NAN, |value| value.residual_rms)
    });
    L2InversionOutput {
        phase,
        posterior_variance,
        residual_rms,
    }
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
    solve_pixel_with_covariance(a, dphi, weights, pixel, n_ifgs)
        .map_or_else(|| vec![f64::NAN; a.ncols()], |value| value.parameters)
}

/// Solve one L2 pixel and return its full covariance on demand.
#[must_use]
pub fn solve_pixel_with_covariance(
    a: ArrayView2<f64>,
    dphi: ArrayView3<f64>,
    weights: Option<ArrayView3<f64>>,
    pixel: (usize, usize),
    n_ifgs: usize,
) -> Option<PixelL2Solution> {
    let n = a.ncols();
    let precision = |k: usize| {
        let value = weights.map_or(1.0, |ws| ws[(k, pixel.0, pixel.1)]);
        if value.is_finite() && value > 0.0 {
            value
        } else {
            0.0
        }
    };
    let valid: Vec<_> = (0..n_ifgs)
        .filter(|&k| precision(k) > 0.0 && dphi[(k, pixel.0, pixel.1)].is_finite())
        .collect();
    if valid.len() < n {
        return None;
    }
    let ata = Mat::from_fn(n, n, |i, j| {
        valid
            .iter()
            .map(|&k| a[(k, i)] * precision(k) * a[(k, j)])
            .sum::<f64>()
    });
    let atb = Mat::from_fn(n, 1, |i, _| {
        valid
            .iter()
            .map(|&k| a[(k, i)] * precision(k) * dphi[(k, pixel.0, pixel.1)])
            .sum::<f64>()
    });
    let llt = ata.cholesky(Side::Lower).ok()?;
    let x = llt.solve(atb);
    let identity = Mat::from_fn(n, n, |i, j| f64::from(i == j));
    let inverse = llt.solve(identity);
    let residuals: Vec<_> = valid
        .iter()
        .map(|&k| {
            let predicted = (0..n).map(|i| a[(k, i)] * x[(i, 0)]).sum::<f64>();
            (k, dphi[(k, pixel.0, pixel.1)] - predicted)
        })
        .collect();
    let weighted_sse = residuals
        .iter()
        .map(|(k, residual)| precision(*k) * residual * residual)
        .sum::<f64>();
    let residual_sse = residuals
        .iter()
        .map(|(_, residual)| residual * residual)
        .sum::<f64>();
    let dof = valid.len().saturating_sub(n);
    let inflation = if dof == 0 {
        1.0
    } else {
        (weighted_sse / dof as f64).max(1.0)
    };
    Some(PixelL2Solution {
        parameters: (0..n).map(|i| x[(i, 0)]).collect(),
        covariance: Array2::from_shape_fn((n, n), |(i, j)| inverse[(i, j)] * inflation),
        residual_rms: (residual_sse / valid.len() as f64).sqrt(),
    })
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

/// Weighted linear velocity with residual and one-sigma slope uncertainty.
#[must_use]
pub fn estimate_velocity_with_uncertainty(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
) -> VelocityOutput {
    let (_, rows, cols) = series.dim();
    let values: Vec<(f64, f64, f64)> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| velocity_pixel_with_uncertainty(x, series, precisions, (idx / cols, idx % cols)))
        .collect();
    let layer = |index: usize| {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            let value = values[r * cols + c];
            [value.0, value.1, value.2][index]
        })
    };
    VelocityOutput {
        velocity: layer(0),
        sigma: layer(1),
        residual_rms: layer(2),
    }
}

/// Weighted linear velocity with uncertainty, plus an opt-in AR(1) temporal-
/// correlation (N_eff) correction reported alongside the uncorrected `sigma`
/// (see [`VelocityOutputNeff`]). `velocity`, `sigma`, and `residual_rms` are
/// identical to [`estimate_velocity_with_uncertainty`]'s output.
#[must_use]
pub fn estimate_velocity_with_uncertainty_neff(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
) -> VelocityOutputNeff {
    let (_, rows, cols) = series.dim();
    let values: Vec<(f64, f64, f64, f64)> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            velocity_pixel_with_temporal_correlation(
                x,
                series,
                precisions,
                (idx / cols, idx % cols),
            )
        })
        .collect();
    let layer = |index: usize| {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            let value = values[r * cols + c];
            [value.0, value.1, value.2, value.3][index]
        })
    };
    let sigma = layer(1);
    let inflation_factor = layer(2);
    let sigma_temporal_corrected = Array2::from_shape_fn((rows, cols), |(r, c)| {
        sigma[(r, c)] * inflation_factor[(r, c)]
    });
    VelocityOutputNeff {
        velocity: layer(0),
        sigma,
        sigma_temporal_corrected,
        inflation_factor,
        residual_rms: layer(3),
    }
}

/// Linear velocity using direct observation precisions without uncertainty outputs.
#[must_use]
pub fn estimate_velocity_with_precisions(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
) -> Array2<f64> {
    let (_, rows, cols) = series.dim();
    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            velocity_pixel_with_uncertainty(x, series, precisions, (idx / cols, idx % cols)).0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("velocity shape")
}

/// Shared weighted-linear fit for one pixel: velocity, uncorrected sigma,
/// residual RMS, and the in-time-order residuals (feeding the opt-in temporal-
/// correlation correction). All non-finite when the normal equations are singular.
struct PixelUncertaintyFit {
    velocity_per_year: f64,
    sigma_per_year: f64,
    residual_rms: f64,
    residuals: Vec<f64>,
}

fn velocity_pixel_uncertainty_fit(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
    pixel: (usize, usize),
) -> PixelUncertaintyFit {
    let valid = |t: usize| {
        let p = precisions[(t, pixel.0, pixel.1)];
        let y = series[(t, pixel.0, pixel.1)];
        p.is_finite() && p > 0.0 && y.is_finite()
    };
    let (mut sw, mut swx, mut swxx, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (t, &xt) in x.iter().enumerate().filter(|(t, _)| valid(*t)) {
        let p = precisions[(t, pixel.0, pixel.1)];
        let y = series[(t, pixel.0, pixel.1)];
        sw += p;
        swx += p * xt;
        swxx += p * xt * xt;
        swy += p * y;
        swxy += p * xt * y;
    }
    let det = sw * swxx - swx * swx;
    if !det.is_finite() || det <= 0.0 {
        return PixelUncertaintyFit {
            velocity_per_year: f64::NAN,
            sigma_per_year: f64::NAN,
            residual_rms: f64::NAN,
            residuals: Vec::new(),
        };
    }
    let slope = (sw * swxy - swx * swy) / det;
    let intercept = (swy - slope * swx) / sw;
    let indices: Vec<_> = (0..x.len()).filter(|&t| valid(t)).collect();
    let residuals: Vec<f64> = indices
        .iter()
        .map(|&t| series[(t, pixel.0, pixel.1)] - (intercept + slope * x[t]))
        .collect();
    let weighted_sse: f64 = indices
        .iter()
        .zip(&residuals)
        .map(|(&t, residual)| precisions[(t, pixel.0, pixel.1)] * residual * residual)
        .sum();
    let residual_sse: f64 = residuals.iter().map(|residual| residual * residual).sum();
    let dof = indices.len().saturating_sub(2);
    let variance_inflation = if dof == 0 {
        1.0
    } else {
        (weighted_sse / dof as f64).max(1.0)
    };
    PixelUncertaintyFit {
        velocity_per_year: slope * 365.25,
        sigma_per_year: (variance_inflation * sw / det).sqrt() * 365.25,
        residual_rms: (residual_sse / indices.len() as f64).sqrt(),
        residuals,
    }
}

fn velocity_pixel_with_uncertainty(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
    pixel: (usize, usize),
) -> (f64, f64, f64) {
    let fit = velocity_pixel_uncertainty_fit(x, series, precisions, pixel);
    (fit.velocity_per_year, fit.sigma_per_year, fit.residual_rms)
}

fn velocity_pixel_with_temporal_correlation(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
    pixel: (usize, usize),
) -> (f64, f64, f64, f64) {
    let fit = velocity_pixel_uncertainty_fit(x, series, precisions, pixel);
    let inflation_factor = if fit.sigma_per_year.is_finite() {
        temporal_correlation_inflation(&fit.residuals)
    } else {
        f64::NAN
    };
    (
        fit.velocity_per_year,
        fit.sigma_per_year,
        inflation_factor,
        fit.residual_rms,
    )
}

/// Lag-1 sample autocorrelation of `residuals`, clamped to `[0, 0.98]`. Negative
/// or undersampled estimates do not warrant inflating sigma below the uncorrected
/// value, and a clamp short of 1.0 keeps the inflation factor finite.
fn lag1_autocorrelation(residuals: &[f64]) -> f64 {
    if residuals.len() < 3 {
        return 0.0;
    }
    let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let centered: Vec<f64> = residuals.iter().map(|r| r - mean).collect();
    let denom: f64 = centered.iter().map(|c| c * c).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    let numer: f64 = centered.windows(2).map(|w| w[0] * w[1]).sum();
    (numer / denom).clamp(0.0, 0.98)
}

/// `sqrt((1 + rho) / (1 - rho))` — the AR(1) effective-sample-size inflation
/// factor for the slope standard error (Zhang et al. 1997; Agram & Zebker 2015),
/// `rho` the lag-1 residual autocorrelation. `1.0` at `rho == 0`.
fn temporal_correlation_inflation(residuals: &[f64]) -> f64 {
    let rho = lag1_autocorrelation(residuals);
    ((1.0 + rho) / (1.0 - rho)).sqrt()
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
