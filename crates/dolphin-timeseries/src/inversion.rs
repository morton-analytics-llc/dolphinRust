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

/// Full weighted-L2 solution for one pixel. The parameter-covariance
/// approximation assumes independent interferogram errors and is kept on demand
/// rather than materialized as an `n_dates^2 * area` workflow cube.
#[derive(Debug, Clone)]
pub struct PixelL2Solution {
    /// Fitted phase parameters.
    pub parameters: Vec<f64>,
    /// Parameter-covariance approximation under an independent-IFG error model.
    pub covariance: Array2<f64>,
    /// Residual root-mean-square in observation units.
    pub residual_rms: f64,
}

/// Bounded stack-level uncertainty products.
pub struct L2InversionOutput {
    /// Fitted phase stack.
    pub phase: Array3<f64>,
    /// Diagonal parameter-covariance approximation under an independent-IFG
    /// error model. The legacy field name is retained for API compatibility.
    pub posterior_variance: Array3<f64>,
    /// Residual root-mean-square in observation units.
    pub residual_rms: Array2<f64>,
}

/// Linear-rate estimate and its IID-conditional one-sigma standard error, both
/// per year. The standard error is conditional on the supplied relative
/// precisions and independent residuals.
pub struct VelocityOutput {
    /// Linear velocity per year.
    pub velocity: Array2<f64>,
    /// IID-conditional one-sigma slope standard error per year. `NaN` unless
    /// [`Self::uncertainty_status`] is [`VelocityUncertaintyStatus::IidConditional`].
    pub sigma: Array2<f64>,
    /// Regression residual root-mean-square in series units.
    pub residual_rms: Array2<f64>,
    /// Number of dates with finite observations and positive finite precision.
    pub valid_date_count: Array2<u32>,
    /// Per-pixel weighted design-matrix rank, at most two for intercept plus slope.
    pub rank: Array2<u32>,
    /// Residual degrees of freedom, `valid_date_count - rank`.
    pub regression_dof: Array2<u32>,
    /// Availability and interpretation of [`Self::sigma`].
    pub uncertainty_status: Array2<VelocityUncertaintyStatus>,
}

/// Interpretation of an IID-conditional velocity standard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VelocityUncertaintyStatus {
    /// The design is rank deficient, residual DOF is zero, or residual scale is
    /// zero/non-finite; no IID-conditional slope standard error is reported.
    Unavailable = 0,
    /// A full-rank fit with positive residual DOF and positive finite residual
    /// scale supports the reported IID-conditional slope standard error.
    IidConditional = 1,
}

/// Cadence classification used to gate lag-one residual diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VelocityCadenceStatus {
    /// Fewer than two valid dates are available to classify cadence.
    Unavailable = 0,
    /// Every acquisition is valid and consecutive day gaps are exactly equal.
    RegularContiguous = 1,
    /// Every acquisition is valid, but consecutive day gaps differ or are invalid.
    Irregular = 2,
    /// One or more acquisitions are absent from the otherwise ordered series.
    Missing = 3,
}

/// Linear velocity and IID-conditional fit evidence plus non-inferential temporal
/// correlation diagnostics. The diagnostic inflation and effective sample size
/// are not standard-error corrections.
pub struct VelocityDiagnosticsOutput {
    /// Linear velocity per year.
    pub velocity: Array2<f64>,
    /// IID-conditional one-sigma slope standard error per year.
    pub sigma: Array2<f64>,
    /// Regression residual root-mean-square in series units.
    pub residual_rms: Array2<f64>,
    /// Number of dates with finite observations and positive finite precision.
    pub valid_date_count: Array2<u32>,
    /// Per-pixel weighted design-matrix rank.
    pub rank: Array2<u32>,
    /// Residual degrees of freedom, `valid_date_count - rank`.
    pub regression_dof: Array2<u32>,
    /// Availability and interpretation of [`Self::sigma`].
    pub uncertainty_status: Array2<VelocityUncertaintyStatus>,
    /// Raw lag-one correlation of standardized residuals, including negative values.
    pub lag1_rho: Array2<f64>,
    /// Number of adjacent residual pairs used for [`Self::lag1_rho`].
    pub correlation_pair_count: Array2<u32>,
    /// Exact cadence classification for the valid date sequence.
    pub cadence_status: Array2<VelocityCadenceStatus>,
    /// Whether all requirements for the lag-one diagnostics were met.
    pub correlation_available: Array2<bool>,
    /// Diagnostic-only `sqrt(n / n_effective)`, with no deflation and effective
    /// sample size clamped to `[1, n]`. This does not rescale [`Self::sigma`].
    pub diagnostic_inflation_factor: Array2<f64>,
    /// Diagnostic-only AR(1) effective sample size clamped to `[1, n]`.
    pub diagnostic_effective_sample_size: Array2<f64>,
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

/// Solve weighted L2 while retaining only independent-IFG parameter-covariance
/// diagonals at stack scale.
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

/// Solve one L2 pixel and return its independent-IFG parameter-covariance
/// approximation on demand.
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

/// Weighted linear velocity with residual evidence and an IID-conditional slope
/// standard error. `precisions` are relative weights: multiplying all valid
/// values at a pixel by one positive constant leaves `sigma` unchanged. The
/// reported standard error is conditional on those relative precisions and
/// independent residuals; it is not calibrated by their absolute scale.
#[must_use]
pub fn estimate_velocity_with_uncertainty(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
) -> VelocityOutput {
    let (_, rows, cols) = series.dim();
    let values: Vec<PixelUncertaintyFit> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| velocity_pixel_uncertainty_fit(x, series, precisions, (idx / cols, idx % cols)))
        .collect();
    VelocityOutput {
        velocity: pixel_layer(&values, rows, cols, |fit| fit.velocity_per_year),
        sigma: pixel_layer(&values, rows, cols, |fit| fit.sigma_per_year),
        residual_rms: pixel_layer(&values, rows, cols, |fit| fit.residual_rms),
        valid_date_count: pixel_layer(&values, rows, cols, |fit| fit.valid_date_count),
        rank: pixel_layer(&values, rows, cols, |fit| fit.rank),
        regression_dof: pixel_layer(&values, rows, cols, |fit| fit.regression_dof),
        uncertainty_status: pixel_layer(&values, rows, cols, |fit| fit.uncertainty_status),
    }
}

/// Weighted linear velocity with IID-conditional uncertainty evidence and
/// diagnostic-only lag-one correlation summaries. `sigma` is conditional on
/// the supplied relative precisions and independent residuals; the correlation
/// diagnostics do not rescale it.
#[must_use]
pub fn estimate_velocity_with_diagnostics(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: ArrayView3<f64>,
) -> VelocityDiagnosticsOutput {
    let (_, rows, cols) = series.dim();
    let values: Vec<PixelVelocityDiagnostics> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let fit =
                velocity_pixel_uncertainty_fit(x, series, precisions, (idx / cols, idx % cols));
            let correlation = correlation_diagnostics(&fit);
            PixelVelocityDiagnostics { fit, correlation }
        })
        .collect();
    VelocityDiagnosticsOutput {
        velocity: pixel_layer(&values, rows, cols, |value| value.fit.velocity_per_year),
        sigma: pixel_layer(&values, rows, cols, |value| value.fit.sigma_per_year),
        residual_rms: pixel_layer(&values, rows, cols, |value| value.fit.residual_rms),
        valid_date_count: pixel_layer(&values, rows, cols, |value| value.fit.valid_date_count),
        rank: pixel_layer(&values, rows, cols, |value| value.fit.rank),
        regression_dof: pixel_layer(&values, rows, cols, |value| value.fit.regression_dof),
        uncertainty_status: pixel_layer(&values, rows, cols, |value| value.fit.uncertainty_status),
        lag1_rho: pixel_layer(&values, rows, cols, |value| value.correlation.lag1_rho),
        correlation_pair_count: pixel_layer(&values, rows, cols, |value| {
            value.correlation.pair_count
        }),
        cadence_status: pixel_layer(&values, rows, cols, |value| value.fit.cadence_status),
        correlation_available: pixel_layer(&values, rows, cols, |value| {
            value.correlation.available
        }),
        diagnostic_inflation_factor: pixel_layer(&values, rows, cols, |value| {
            value.correlation.inflation_factor
        }),
        diagnostic_effective_sample_size: pixel_layer(&values, rows, cols, |value| {
            value.correlation.effective_sample_size
        }),
    }
}

/// Linear velocity using relative observation precisions without uncertainty outputs.
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
            velocity_pixel_uncertainty_fit(x, series, precisions, (idx / cols, idx % cols))
                .velocity_per_year
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("velocity shape")
}

fn pixel_layer<T, U: Copy>(
    values: &[T],
    rows: usize,
    cols: usize,
    pick: impl Fn(&T) -> U,
) -> Array2<U> {
    Array2::from_shape_fn((rows, cols), |(row, col)| pick(&values[row * cols + col]))
}

/// Shared weighted-linear fit for one pixel. The velocity remains available for
/// a full-rank fit even when the residual evidence cannot support an IID SE.
struct PixelUncertaintyFit {
    velocity_per_year: f64,
    sigma_per_year: f64,
    residual_rms: f64,
    valid_date_count: u32,
    rank: u32,
    regression_dof: u32,
    uncertainty_status: VelocityUncertaintyStatus,
    cadence_status: VelocityCadenceStatus,
    standardized_residuals: Vec<f64>,
}

struct PixelCorrelationDiagnostics {
    lag1_rho: f64,
    pair_count: u32,
    available: bool,
    inflation_factor: f64,
    effective_sample_size: f64,
}

struct PixelVelocityDiagnostics {
    fit: PixelUncertaintyFit,
    correlation: PixelCorrelationDiagnostics,
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
        x[t].is_finite() && p.is_finite() && p > 0.0 && y.is_finite()
    };
    let indices: Vec<_> = (0..x.len()).filter(|&t| valid(t)).collect();
    let valid_date_count = u32::try_from(indices.len()).unwrap_or(u32::MAX);
    let cadence_status = velocity_cadence_status(x, &indices);
    let (mut sw, mut swx, mut swxx, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &t in &indices {
        let xt = x[t];
        let p = precisions[(t, pixel.0, pixel.1)];
        let y = series[(t, pixel.0, pixel.1)];
        sw += p;
        swx += p * xt;
        swxx += p * xt * xt;
        swy += p * y;
        swxy += p * xt * y;
    }
    let det = sw * swxx - swx * swx;
    let rank = match (sw.is_finite() && sw > 0.0, det.is_finite() && det > 0.0) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => 2,
    };
    let regression_dof = valid_date_count.saturating_sub(rank);
    if !det.is_finite() || det <= 0.0 {
        return PixelUncertaintyFit {
            velocity_per_year: f64::NAN,
            sigma_per_year: f64::NAN,
            residual_rms: f64::NAN,
            valid_date_count,
            rank,
            regression_dof,
            uncertainty_status: VelocityUncertaintyStatus::Unavailable,
            cadence_status,
            standardized_residuals: Vec::new(),
        };
    }
    let slope = (sw * swxy - swx * swy) / det;
    let intercept = (swy - slope * swx) / sw;
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
    let residual_rms = match residual_sse.is_finite() && !indices.is_empty() {
        true => (residual_sse / indices.len() as f64).sqrt(),
        false => f64::NAN,
    };
    let residual_scale = match regression_dof {
        0 => f64::NAN,
        dof => weighted_sse / f64::from(dof),
    };
    let slope_variance = residual_scale * sw / det;
    let uncertainty_available = rank == 2
        && regression_dof > 0
        && residual_scale.is_finite()
        && residual_scale > 0.0
        && slope_variance.is_finite()
        && slope_variance > 0.0;
    let sigma_per_year = match uncertainty_available {
        true => slope_variance.sqrt() * 365.25,
        false => f64::NAN,
    };
    let standardized_residuals = match uncertainty_available {
        true => indices
            .iter()
            .zip(&residuals)
            .map(|(&t, &residual)| {
                precisions[(t, pixel.0, pixel.1)].sqrt() * residual / residual_scale.sqrt()
            })
            .collect(),
        false => Vec::new(),
    };
    PixelUncertaintyFit {
        velocity_per_year: slope * 365.25,
        sigma_per_year,
        residual_rms,
        valid_date_count,
        rank,
        regression_dof,
        uncertainty_status: match uncertainty_available {
            true => VelocityUncertaintyStatus::IidConditional,
            false => VelocityUncertaintyStatus::Unavailable,
        },
        cadence_status,
        standardized_residuals,
    }
}

fn velocity_cadence_status(x: &[f64], indices: &[usize]) -> VelocityCadenceStatus {
    if indices.len() < 2 {
        return VelocityCadenceStatus::Unavailable;
    }
    if indices.len() != x.len() || indices.windows(2).any(|pair| pair[1] != pair[0] + 1) {
        return VelocityCadenceStatus::Missing;
    }
    let cadence = x[1] - x[0];
    if !cadence.is_finite()
        || cadence <= 0.0
        || x.windows(2).any(|pair| pair[1] - pair[0] != cadence)
    {
        return VelocityCadenceStatus::Irregular;
    }
    VelocityCadenceStatus::RegularContiguous
}

fn correlation_diagnostics(fit: &PixelUncertaintyFit) -> PixelCorrelationDiagnostics {
    let unavailable = || PixelCorrelationDiagnostics {
        lag1_rho: f64::NAN,
        pair_count: 0,
        available: false,
        inflation_factor: f64::NAN,
        effective_sample_size: f64::NAN,
    };
    let n = fit.standardized_residuals.len();
    if fit.uncertainty_status != VelocityUncertaintyStatus::IidConditional
        || fit.cadence_status != VelocityCadenceStatus::RegularContiguous
        || n < 4
        || n != fit.valid_date_count as usize
        || fit
            .standardized_residuals
            .iter()
            .any(|value| !value.is_finite())
    {
        return unavailable();
    }
    let mean = fit.standardized_residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = fit
        .standardized_residuals
        .iter()
        .map(|residual| residual - mean)
        .collect();
    let denominator = centered.iter().map(|value| value * value).sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return unavailable();
    }
    let numerator = centered
        .windows(2)
        .map(|pair| pair[0] * pair[1])
        .sum::<f64>();
    let rho = numerator / denominator;
    if !rho.is_finite() {
        return unavailable();
    }
    let n_f64 = n as f64;
    let applied_rho = rho.clamp(0.0, 1.0);
    let effective_sample_size =
        (n_f64 * (1.0 - applied_rho) / (1.0 + applied_rho)).clamp(1.0, n_f64);
    PixelCorrelationDiagnostics {
        lag1_rho: rho,
        pair_count: u32::try_from(n - 1).unwrap_or(u32::MAX),
        available: true,
        inflation_factor: (n_f64 / effective_sample_size).sqrt(),
        effective_sample_size,
    }
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
