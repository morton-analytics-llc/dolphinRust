//! Time-function decomposition of the displacement series (issue #22).
//!
//! [`inversion::estimate_velocity`](crate::inversion::estimate_velocity) and its
//! weighted siblings fit a bare linear trend, matching Python dolphin's own
//! `velocity.py`. On a series carrying a real seasonal cycle (groundwater,
//! thermal) or a step (co-seismic, anthropogenic, an instrument change), that
//! trend absorbs the seasonal amplitude and the step discontinuity into its slope
//! — the reported "velocity" is then partly a description of the seasonal phase
//! sampled by the acquisition times, not the long-term rate.
//!
//! This module fits those terms **jointly** with the rate in one weighted
//! least-squares solve, so the rate is the rate and the rest is reported
//! separately. It is a **forward divergence from the pinned dolphin oracle**
//! (dolphin does not do this), following the `correct_phase_bias` /
//! `correct_velocity_temporal_correlation` precedent: config-gated and off by
//! default, so the parity-critical path is untouched. With
//! [`VelocityModel::is_linear`] the caller stays on the existing degree-1
//! functions — there is no "linear through the general path" to drift.
//!
//! Basis, in column order: `[1, t, (sin ωt, cos ωt)?, H(t − t_k)…]` with
//! `ω = 2π/365.25 d⁻¹` and `H` the Heaviside step. Post-seismic
//! (exponential/logarithmic) relaxation is **not** here: it needs a relaxation
//! time constant, which is a fitted nonlinear parameter or another config knob,
//! and neither is justified before the seasonal/step terms have been used on
//! real data.

use faer::prelude::SpSolver;
use faer::{Mat, Side};
use ndarray::{Array2, ArrayView3};
use rayon::prelude::*;

use crate::inversion::temporal_correlation_inflation;

/// Days in a Julian year — the period of the seasonal basis and the rate scaling.
const DAYS_PER_YEAR: f64 = 365.25;

/// Optional basis terms fitted alongside the linear rate.
///
/// Two independent switches rather than one `linear | seasonal | step` enum: a
/// series can carry both, and a step needs its epoch supplied regardless.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VelocityModel {
    /// Fit an annual `sin`/`cos` pair (period 365.25 d) alongside the rate.
    pub seasonal: bool,
    /// Heaviside step epochs, in the same units and origin as `x` (decimal days
    /// from acquisition 0). One extra basis column each; the epoch is an input,
    /// never detected from the data.
    pub step_days: Vec<f64>,
}

impl VelocityModel {
    /// `true` when no optional term is configured — the caller should stay on the
    /// existing degree-1 estimators, which this path does not attempt to replace.
    #[must_use]
    pub fn is_linear(&self) -> bool {
        !self.seasonal && self.step_days.is_empty()
    }

    /// Number of fitted parameters: intercept + rate + optional terms.
    fn n_terms(&self) -> usize {
        2 + 2 * usize::from(self.seasonal) + self.step_days.len()
    }

    /// One row of the design matrix at time `t`.
    fn basis_row(&self, t: f64) -> Vec<f64> {
        let mut row = vec![1.0, t];
        if self.seasonal {
            let omega_t = std::f64::consts::TAU * t / DAYS_PER_YEAR;
            row.push(omega_t.sin());
            row.push(omega_t.cos());
        }
        row.extend(self.step_days.iter().map(|&step| f64::from(t >= step)));
        row
    }

    /// Column index of the first step term.
    fn step_offset(&self) -> usize {
        2 + 2 * usize::from(self.seasonal)
    }
}

/// Rate and its standard error, plus whatever optional terms were fitted.
///
/// `velocity`, `sigma`, and `residual_rms` carry the same meaning and units as
/// [`VelocityOutput`](crate::inversion::VelocityOutput) — a consumer reading only
/// those three sees the same product, now with the seasonal/step signal taken out
/// of the rate rather than left in it.
pub struct VelocityModelOutput {
    /// Linear rate per year, with the optional terms fitted out.
    pub velocity: Array2<f64>,
    /// One-sigma rate uncertainty per year.
    pub sigma: Array2<f64>,
    /// AR(1) effective-sample-size inflation of `sigma`, from this model's
    /// residuals (`1.0` when uncorrelated). The caller applies it or not, exactly
    /// as with [`VelocityOutputNeff`](crate::inversion::VelocityOutputNeff).
    pub inflation_factor: Array2<f64>,
    /// Regression residual root-mean-square in series units.
    pub residual_rms: Array2<f64>,
    /// `hypot(a, b)` of the annual `a·sin + b·cos` pair, series units — the
    /// peak-to-mean seasonal amplitude. `None` unless [`VelocityModel::seasonal`].
    pub seasonal_amplitude: Option<Array2<f64>>,
    /// Days after acquisition 0 at which the annual term peaks, in
    /// `[0, 365.25)`. `None` unless [`VelocityModel::seasonal`].
    pub seasonal_phase_days: Option<Array2<f64>>,
    /// Fitted magnitude of each configured step, series units, in `step_days`
    /// order. Empty when no step is configured.
    pub step_magnitude: Vec<Array2<f64>>,
}

/// Fitted parameters and diagnostics for one pixel; all non-finite when the
/// weighted normal equations are singular or the pixel has too few valid epochs.
struct PixelModelFit {
    parameters: Vec<f64>,
    sigma_per_year: f64,
    residual_rms: f64,
    inflation_factor: f64,
}

/// Joint weighted least-squares fit of the rate and the configured optional
/// terms, per pixel. `x` is decimal days (acquisition 0 = 0), `series` is
/// `(n_time, rows, cols)`, `precisions` the per-epoch observation precisions
/// (`1/σ²`) or `None` for an unweighted fit; an epoch is used where its precision
/// is finite and positive and the series value is finite.
///
/// # Panics
/// Panics if `model.is_linear()` — the linear case belongs on
/// [`estimate_velocity_with_uncertainty`](crate::inversion::estimate_velocity_with_uncertainty),
/// which is the parity-critical path this must not silently reimplement.
#[must_use]
pub fn estimate_velocity_with_model(
    x: &[f64],
    series: ArrayView3<f64>,
    precisions: Option<ArrayView3<f64>>,
    model: &VelocityModel,
) -> VelocityModelOutput {
    assert!(
        !model.is_linear(),
        "estimate_velocity_with_model is the optional-term path; a linear model must use \
         estimate_velocity_with_uncertainty so the parity-critical fit stays the only one"
    );
    let (_, rows, cols) = series.dim();
    let design: Vec<Vec<f64>> = x.iter().map(|&t| model.basis_row(t)).collect();
    let fits: Vec<PixelModelFit> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| pixel_model_fit(&design, series, precisions, (idx / cols, idx % cols), model))
        .collect();

    let layer = |value: &dyn Fn(&PixelModelFit) -> f64| {
        Array2::from_shape_fn((rows, cols), |(r, c)| value(&fits[r * cols + c]))
    };
    let parameter = |i: usize| layer(&move |fit: &PixelModelFit| fit.parameters[i]);
    let offset = model.step_offset();
    VelocityModelOutput {
        velocity: layer(&|fit| fit.parameters[1] * DAYS_PER_YEAR),
        sigma: layer(&|fit| fit.sigma_per_year),
        inflation_factor: layer(&|fit| fit.inflation_factor),
        residual_rms: layer(&|fit| fit.residual_rms),
        seasonal_amplitude: model
            .seasonal
            .then(|| layer(&|fit| fit.parameters[2].hypot(fit.parameters[3]))),
        seasonal_phase_days: model
            .seasonal
            .then(|| layer(&|fit| seasonal_peak_day(fit.parameters[2], fit.parameters[3]))),
        step_magnitude: (0..model.step_days.len())
            .map(|k| parameter(offset + k))
            .collect(),
    }
}

/// Day in `[0, 365.25)` at which `a·sin(ωt) + b·cos(ωt) = A·cos(ωt − φ)` peaks,
/// `φ = atan2(a, b)`. NaN in, NaN out.
fn seasonal_peak_day(a: f64, b: f64) -> f64 {
    let phase = a.atan2(b);
    let day = phase * DAYS_PER_YEAR / std::f64::consts::TAU;
    day.rem_euclid(DAYS_PER_YEAR)
}

fn pixel_model_fit(
    design: &[Vec<f64>],
    series: ArrayView3<f64>,
    precisions: Option<ArrayView3<f64>>,
    pixel: (usize, usize),
    model: &VelocityModel,
) -> PixelModelFit {
    let n = model.n_terms();
    let precision = |t: usize| {
        let p = precisions.map_or(1.0, |ps| ps[(t, pixel.0, pixel.1)]);
        let y = series[(t, pixel.0, pixel.1)];
        match p.is_finite() && p > 0.0 && y.is_finite() {
            true => p,
            false => 0.0,
        }
    };
    let valid: Vec<usize> = (0..design.len()).filter(|&t| precision(t) > 0.0).collect();
    // A fit with no residual degrees of freedom reproduces the data exactly and
    // reports a meaningless zero sigma; require at least one.
    if valid.len() <= n {
        return singular_fit(n);
    }
    let normal = Mat::from_fn(n, n, |i, j| {
        valid
            .iter()
            .map(|&t| design[t][i] * precision(t) * design[t][j])
            .sum::<f64>()
    });
    let rhs = Mat::from_fn(n, 1, |i, _| {
        valid
            .iter()
            .map(|&t| design[t][i] * precision(t) * series[(t, pixel.0, pixel.1)])
            .sum::<f64>()
    });
    let Ok(llt) = normal.cholesky(Side::Lower) else {
        return singular_fit(n);
    };
    let beta = llt.solve(rhs);
    let parameters: Vec<f64> = (0..n).map(|i| beta[(i, 0)]).collect();

    let residuals: Vec<f64> = valid
        .iter()
        .map(|&t| {
            let predicted: f64 = (0..n).map(|i| design[t][i] * parameters[i]).sum();
            series[(t, pixel.0, pixel.1)] - predicted
        })
        .collect();
    let weighted_sse: f64 = valid
        .iter()
        .zip(&residuals)
        .map(|(&t, r)| precision(t) * r * r)
        .sum();
    let residual_sse: f64 = residuals.iter().map(|r| r * r).sum();
    let inflation = (weighted_sse / (valid.len() - n) as f64).max(1.0);
    // Rate variance is the (1,1) entry of the inverted normal matrix; solving
    // against e1 costs one back-substitution instead of a full inverse.
    let unit_rate = Mat::from_fn(n, 1, |i, _| f64::from(i == 1));
    let rate_variance = llt.solve(unit_rate)[(1, 0)] * inflation;

    PixelModelFit {
        parameters,
        sigma_per_year: rate_variance.sqrt() * DAYS_PER_YEAR,
        residual_rms: (residual_sse / valid.len() as f64).sqrt(),
        inflation_factor: temporal_correlation_inflation(&residuals),
    }
}

fn singular_fit(n: usize) -> PixelModelFit {
    PixelModelFit {
        parameters: vec![f64::NAN; n],
        sigma_per_year: f64::NAN,
        residual_rms: f64::NAN,
        inflation_factor: f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inversion::estimate_velocity_with_uncertainty;
    use ndarray::{Array3, Axis};

    /// 4 years of 12-day Sentinel-1 sampling.
    fn sample_days() -> Vec<f64> {
        (0..122).map(|k| 12.0 * k as f64).collect()
    }

    /// A one-pixel `(n_time, 1, 1)` stack from a series, with unit precisions.
    fn one_pixel(values: &[f64]) -> (Array3<f64>, Array3<f64>) {
        let n = values.len();
        let series = Array3::from_shape_fn((n, 1, 1), |(t, _, _)| values[t]);
        (series, Array3::ones((n, 1, 1)))
    }

    /// Contract: a series built from a known rate + known annual sinusoid returns
    /// the rate, amplitude, and phase that built it — the linear-only fit does not.
    #[test]
    fn recovers_known_rate_amplitude_and_phase() {
        let days = sample_days();
        let (rate_per_year, amplitude, peak_day) = (-25.0, 8.0, 200.0);
        let omega = std::f64::consts::TAU / DAYS_PER_YEAR;
        let values: Vec<f64> = days
            .iter()
            .map(|&t| {
                rate_per_year * t / DAYS_PER_YEAR + amplitude * (omega * (t - peak_day)).cos()
            })
            .collect();
        let (series, precisions) = one_pixel(&values);

        let model = VelocityModel {
            seasonal: true,
            step_days: Vec::new(),
        };
        let out =
            estimate_velocity_with_model(&days, series.view(), Some(precisions.view()), &model);
        assert!(
            (out.velocity[(0, 0)] - rate_per_year).abs() < 1e-8,
            "rate {} != {rate_per_year}",
            out.velocity[(0, 0)]
        );
        let recovered_amplitude = out.seasonal_amplitude.as_ref().unwrap()[(0, 0)];
        assert!(
            (recovered_amplitude - amplitude).abs() < 1e-8,
            "amplitude {recovered_amplitude} != {amplitude}"
        );
        let recovered_peak = out.seasonal_phase_days.as_ref().unwrap()[(0, 0)];
        assert!(
            (recovered_peak - peak_day).abs() < 1e-6,
            "peak day {recovered_peak} != {peak_day}"
        );
        assert!(
            out.residual_rms[(0, 0)] < 1e-9,
            "exact fit leaves no residual"
        );
    }

    /// The bias this exists to remove. On a noiseless series carrying a true
    /// −25 mm/yr rate and an 8 mm annual cycle, the linear-only fit reports a rate
    /// that is wrong purely because the cycle is not sampled over whole years.
    /// How wrong depends on where the window cuts the cycle, so both a
    /// near-whole-year window and a half-cycle window are checked — the point is
    /// that the error is a function of the acquisition times, not of the ground.
    #[test]
    fn linear_only_fit_absorbs_the_seasonal_term() {
        let (rate_per_year, amplitude) = (-25.0, 8.0);
        let omega = std::f64::consts::TAU / DAYS_PER_YEAR;
        // (n_acquisitions at 12 d, minimum leaked rate error in mm/yr).
        // 30 x 12 d = 360 d, just short of a full cycle: ~2.4 mm/yr, ~9% of rate.
        // 16 x 12 d = 180 d, half a cycle: ~34 mm/yr, larger than the rate itself.
        for (count, min_error) in [(30, 2.0), (16, 30.0)] {
            let days: Vec<f64> = (0..count).map(|k| 12.0 * f64::from(k)).collect();
            let values: Vec<f64> = days
                .iter()
                .map(|&t| rate_per_year * t / DAYS_PER_YEAR + amplitude * (omega * t).cos())
                .collect();
            let (series, precisions) = one_pixel(&values);

            let linear =
                estimate_velocity_with_uncertainty(&days, series.view(), precisions.view());
            let model = VelocityModel {
                seasonal: true,
                step_days: Vec::new(),
            };
            let seasonal =
                estimate_velocity_with_model(&days, series.view(), Some(precisions.view()), &model);

            assert!(
                (seasonal.velocity[(0, 0)] - rate_per_year).abs() < 1e-8,
                "{count} epochs: seasonal model must recover the true rate, got {}",
                seasonal.velocity[(0, 0)]
            );
            let linear_error = (linear.velocity[(0, 0)] - rate_per_year).abs();
            assert!(
                linear_error > min_error,
                "{count} epochs: linear-only error {linear_error} mm/yr below the \
                 {min_error} mm/yr the fixture is built to show"
            );
        }
    }

    /// A step mid-series is recovered at its configured epoch, and the rate is
    /// the rate on either side of it rather than the rate plus the jump.
    #[test]
    fn recovers_known_step_without_biasing_the_rate() {
        let days = sample_days();
        let (rate_per_year, step_magnitude, step_day) = (10.0, -40.0, 600.0);
        let values: Vec<f64> = days
            .iter()
            .map(|&t| rate_per_year * t / DAYS_PER_YEAR + f64::from(t >= step_day) * step_magnitude)
            .collect();
        let (series, precisions) = one_pixel(&values);

        let model = VelocityModel {
            seasonal: false,
            step_days: vec![step_day],
        };
        let out =
            estimate_velocity_with_model(&days, series.view(), Some(precisions.view()), &model);
        assert!((out.velocity[(0, 0)] - rate_per_year).abs() < 1e-8);
        assert!((out.step_magnitude[0][(0, 0)] - step_magnitude).abs() < 1e-8);

        let linear = estimate_velocity_with_uncertainty(&days, series.view(), precisions.view());
        assert!(
            (linear.velocity[(0, 0)] - rate_per_year).abs() > 5.0,
            "linear-only fit should absorb the step into its slope"
        );
    }

    /// Seasonal and step together, on one series carrying both.
    #[test]
    fn recovers_seasonal_and_step_jointly() {
        let days = sample_days();
        let (rate_per_year, amplitude, peak_day) = (-25.0, 8.0, 200.0);
        let (step_magnitude, step_day) = (-40.0, 600.0);
        let omega = std::f64::consts::TAU / DAYS_PER_YEAR;
        let values: Vec<f64> = days
            .iter()
            .map(|&t| {
                rate_per_year * t / DAYS_PER_YEAR
                    + amplitude * (omega * (t - peak_day)).cos()
                    + f64::from(t >= step_day) * step_magnitude
            })
            .collect();
        let (series, precisions) = one_pixel(&values);

        let model = VelocityModel {
            seasonal: true,
            step_days: vec![step_day],
        };
        let out =
            estimate_velocity_with_model(&days, series.view(), Some(precisions.view()), &model);
        assert!((out.velocity[(0, 0)] - rate_per_year).abs() < 1e-8, "rate");
        assert!(
            (out.seasonal_amplitude.as_ref().unwrap()[(0, 0)] - amplitude).abs() < 1e-8,
            "amplitude"
        );
        assert!(
            (out.seasonal_phase_days.as_ref().unwrap()[(0, 0)] - peak_day).abs() < 1e-6,
            "phase"
        );
        assert!(
            (out.step_magnitude[0][(0, 0)] - step_magnitude).abs() < 1e-8,
            "step"
        );
    }

    /// Under-determined and singular pixels are NaN everywhere, never a
    /// confident-looking zero.
    #[test]
    fn undetermined_pixel_is_nan() {
        let days = vec![0.0, 12.0, 24.0];
        let (series, precisions) = one_pixel(&[1.0, 2.0, 3.0]);
        let model = VelocityModel {
            seasonal: true,
            step_days: Vec::new(),
        };
        let out =
            estimate_velocity_with_model(&days, series.view(), Some(precisions.view()), &model);
        assert!(out.velocity[(0, 0)].is_nan());
        assert!(out.sigma[(0, 0)].is_nan());
        assert!(out.seasonal_amplitude.as_ref().unwrap()[(0, 0)].is_nan());
    }

    /// Epochs with non-positive or non-finite precision are dropped, matching the
    /// linear estimators' validity rule.
    #[test]
    fn drops_invalid_epochs() {
        let days = sample_days();
        let rate_per_year = 7.5;
        let values: Vec<f64> = days.iter().map(|&t| rate_per_year * t / 365.25).collect();
        let (series, mut precisions) = one_pixel(&values);
        precisions[(3, 0, 0)] = 0.0;
        precisions[(7, 0, 0)] = f64::NAN;
        let mut series = series;
        series[(11, 0, 0)] = f64::NAN;
        precisions[(11, 0, 0)] = 1.0;

        let model = VelocityModel {
            seasonal: true,
            step_days: Vec::new(),
        };
        let out =
            estimate_velocity_with_model(&days, series.view(), Some(precisions.view()), &model);
        assert!((out.velocity[(0, 0)] - rate_per_year).abs() < 1e-8);
        assert!(out.seasonal_amplitude.as_ref().unwrap()[(0, 0)].abs() < 1e-8);
    }

    /// The model is fitted independently per pixel across the grid.
    #[test]
    fn fits_each_pixel_independently() {
        let days = sample_days();
        let rates = [[-25.0, 0.0], [12.5, 3.75]];
        let omega = std::f64::consts::TAU / DAYS_PER_YEAR;
        let series = Array3::from_shape_fn((days.len(), 2, 2), |(t, r, c)| {
            rates[r][c] * days[t] / DAYS_PER_YEAR + 5.0 * (omega * days[t]).cos()
        });
        let precisions = Array3::ones((days.len(), 2, 2));
        let model = VelocityModel {
            seasonal: true,
            step_days: Vec::new(),
        };
        let out =
            estimate_velocity_with_model(&days, series.view(), Some(precisions.view()), &model);
        for (r, row) in rates.iter().enumerate() {
            for (c, &expected) in row.iter().enumerate() {
                assert!(
                    (out.velocity[(r, c)] - expected).abs() < 1e-8,
                    "pixel ({r},{c})"
                );
            }
        }
        assert_eq!(out.velocity.len_of(Axis(0)), 2);
    }

    #[test]
    #[should_panic(expected = "optional-term path")]
    fn linear_model_is_rejected() {
        let days = sample_days();
        let (series, precisions) = one_pixel(&vec![0.0; days.len()]);
        let _ = estimate_velocity_with_model(
            &days,
            series.view(),
            Some(precisions.view()),
            &VelocityModel::default(),
        );
    }
}
