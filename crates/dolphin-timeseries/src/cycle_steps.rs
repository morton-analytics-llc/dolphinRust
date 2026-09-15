//! Per-date integer-cycle step detector on the inverted series (E4, 2026-09-15).
//!
//! # What it catches that the loop QC cannot
//!
//! `loop_closure` closes triangles across interferograms: an unwrap error in
//! one interferogram leaves a `2π` residual in every loop through it. On
//! calibration block 1 every loop closed to zero and every interferogram was
//! one connected component, yet three pixels of one GNSS window sat a whole
//! cycle above their neighbours from the ministack junction onward. A
//! closure-*consistent* integer inconsistency — the same wrong integer in every
//! interferogram touching a date, or an integer the inversion distributed
//! consistently — is invisible to any per-loop test. It is visible spatially:
//! the pixel's step into that date differs from its neighbours' by a whole
//! number of cycles while the neighbours agree with each other.
//!
//! # Rule
//!
//! For each date and pixel, the residual of the pixel's step (series band `k`
//! minus band `k − 1`, acquisition 0 as the implicit zero) against the median
//! step of the labelled, finite neighbours within `half_window` that share its
//! connected-component label. A residual within `tolerance_cycles` of a
//! non-zero whole cycle is a detected step of that many cycles. Pixels with no
//! label, a non-finite step, or fewer than [`MIN_CYCLE_STEP_NEIGHBOURS`]
//! qualifying neighbours are not judged (never flagged). Nothing is corrected
//! here: the output is counts, flags, and per-date fractions.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// How close to a whole cycle the neighbourhood residual must be to count as
/// an integer step. A quarter cycle keeps the 0.5–0.75 cycle steps E4 saw
/// across 6–36% of the frame out of the integer bin; they are reported through
/// [`CycleStepDateSummary::fraction_over_half_cycle`] instead.
pub const DEFAULT_CYCLE_STEP_TOLERANCE_CYCLES: f64 = 0.25;
/// Neighbourhood half-width: a 5×5 window, the same support the GNSS window
/// builder averages over.
pub const DEFAULT_CYCLE_STEP_HALF_WINDOW: usize = 2;
/// Fewest same-component finite neighbours a pixel needs before its residual
/// is judged; a median of fewer is one or two pixels' opinion.
pub const MIN_CYCLE_STEP_NEIGHBOURS: usize = 4;

/// Per-date integer-cycle steps detected inside connected components.
#[derive(Debug, Clone)]
pub struct CycleStepDetection {
    /// Detected step in whole cycles at each step index and pixel,
    /// `(n_dates - 1, rows, cols)`; step index `k` is acquisition `k` to
    /// `k + 1`. `0` where no step was detected or the pixel was not judged.
    pub steps: Array3<i8>,
    /// Number of step indices at which a step was detected, per pixel.
    pub step_count: Array2<u32>,
    /// Per step index, in order; `epoch` is the later acquisition.
    pub per_date: Vec<CycleStepDateSummary>,
}

/// Identifier-free frame statistics for one date's step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CycleStepDateSummary {
    /// Acquisition index the step lands on (`k + 1` for step index `k`).
    pub epoch: usize,
    /// Pixels whose residual was judged.
    pub evaluable_pixels: usize,
    /// Pixels flagged with a non-zero whole-cycle step at this epoch.
    pub flagged_pixels: usize,
    /// `flagged_pixels / evaluable_pixels`, `0` when nothing was judged.
    pub flagged_fraction: f64,
    /// Pixels whose detected steps up to and including this epoch sum to a
    /// non-zero offset: the pixels still sitting a whole cycle from their
    /// neighbours at this date.
    pub offset_pixels: usize,
    /// `offset_pixels / frame pixels`.
    pub offset_fraction: f64,
    /// Fraction of judged pixels whose residual exceeds half a cycle in
    /// magnitude, whole cycle or not — the ambiguous steps.
    pub fraction_over_half_cycle: f64,
}

impl CycleStepDetection {
    /// A detection with no step index judged: empty per-date statistics and a
    /// zero count at every pixel of a `rows x cols` frame.
    #[must_use]
    pub fn unjudged(rows: usize, cols: usize) -> Self {
        Self {
            steps: Array3::zeros((0, rows, cols)),
            step_count: Array2::zeros((rows, cols)),
            per_date: Vec::new(),
        }
    }
}

/// Run the detector. `series` is the `(n_dates - 1, rows, cols)` per-date
/// series in radians referenced to acquisition 0; `components` carries, per
/// step index, the connected-component label governing that step (`0` = none).
///
/// # Panics
/// If `components` and `series` differ in shape.
#[must_use]
pub fn detect_cycle_steps(
    series: ArrayView3<f64>,
    components: ArrayView3<u32>,
    half_window: usize,
    tolerance_cycles: f64,
) -> CycleStepDetection {
    assert_eq!(
        components.dim(),
        series.dim(),
        "component labels must match the series"
    );
    let (n_steps, rows, cols) = series.dim();
    let mut steps = Array3::<i8>::zeros((n_steps, rows, cols));
    let mut offset = Array2::<i32>::zeros((rows, cols));
    let mut per_date = Vec::with_capacity(n_steps);
    for k in 0..n_steps {
        let step = date_step_cycles(series, k);
        let labels = components.index_axis(Axis(0), k);
        let residuals = neighbourhood_residuals(step.view(), labels, half_window);
        let mut judged = 0;
        let mut flagged = 0;
        let mut over_half = 0;
        ndarray::Zip::from(steps.index_axis_mut(Axis(0), k))
            .and(&residuals)
            .and(&mut offset)
            .for_each(|detected, &residual, offset| {
                if !residual.is_finite() {
                    return;
                }
                judged += 1;
                over_half += usize::from(residual.abs() > 0.5);
                let whole = residual.round();
                if whole != 0.0 && (residual - whole).abs() <= tolerance_cycles {
                    flagged += 1;
                    *detected = whole.clamp(f64::from(i8::MIN), f64::from(i8::MAX)) as i8;
                    *offset += i32::from(*detected);
                }
            });
        let offset_pixels = offset.iter().filter(|&&value| value != 0).count();
        let fraction = |count: usize| match judged {
            0 => 0.0,
            _ => count as f64 / judged as f64,
        };
        per_date.push(CycleStepDateSummary {
            epoch: k + 1,
            evaluable_pixels: judged,
            flagged_pixels: flagged,
            flagged_fraction: fraction(flagged),
            offset_pixels,
            offset_fraction: match rows * cols {
                0 => 0.0,
                pixels => offset_pixels as f64 / pixels as f64,
            },
            fraction_over_half_cycle: fraction(over_half),
        });
    }
    let step_count = steps.fold_axis(Axis(0), 0_u32, |&count, &step| count + u32::from(step != 0));
    CycleStepDetection {
        steps,
        step_count,
        per_date,
    }
}

/// Step into acquisition `k + 1` in cycles.
fn date_step_cycles(series: ArrayView3<f64>, k: usize) -> Array2<f64> {
    let later = series.index_axis(Axis(0), k);
    let step = match k {
        0 => later.to_owned(),
        _ => &later - &series.index_axis(Axis(0), k - 1),
    };
    step / std::f64::consts::TAU
}

/// Each pixel's step minus the median step of its qualifying neighbours; NaN
/// where the pixel is not judged.
fn neighbourhood_residuals(
    step: ArrayView2<f64>,
    labels: ArrayView2<u32>,
    half_window: usize,
) -> Array2<f64> {
    let (rows, cols) = step.dim();
    let residuals: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map_iter(|row| {
            let mut neighbours = Vec::with_capacity((2 * half_window + 1).pow(2));
            (0..cols)
                .map(|col| pixel_residual(step, labels, (row, col), half_window, &mut neighbours))
                .collect::<Vec<_>>()
        })
        .collect();
    Array2::from_shape_vec((rows, cols), residuals).expect("one residual per pixel")
}

fn pixel_residual(
    step: ArrayView2<f64>,
    labels: ArrayView2<u32>,
    (row, col): (usize, usize),
    half_window: usize,
    neighbours: &mut Vec<f64>,
) -> f64 {
    let label = labels[(row, col)];
    let own = step[(row, col)];
    if label == 0 || !own.is_finite() {
        return f64::NAN;
    }
    let (rows, cols) = step.dim();
    neighbours.clear();
    for r in row.saturating_sub(half_window)..(row + half_window + 1).min(rows) {
        for c in col.saturating_sub(half_window)..(col + half_window + 1).min(cols) {
            let value = step[(r, c)];
            if (r, c) != (row, col) && labels[(r, c)] == label && value.is_finite() {
                neighbours.push(value);
            }
        }
    }
    if neighbours.len() < MIN_CYCLE_STEP_NEIGHBOURS {
        return f64::NAN;
    }
    own - median(neighbours)
}

/// Median of a non-empty slice (upper median for even lengths).
fn median(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    let (_, median, _) = values.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
    *median
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use std::f64::consts::TAU;

    /// A smooth series on a `rows x cols` frame: a per-date rate, a spatial
    /// gradient, and a mild seasonal term, in radians.
    fn smooth_series(n_dates: usize, rows: usize, cols: usize) -> Array3<f64> {
        Array3::from_shape_fn((n_dates - 1, rows, cols), |(k, row, col)| {
            let t = (k + 1) as f64;
            0.3 * t + 0.02 * (row as f64 + col as f64) * t + 0.4 * (t / 6.0).sin()
        })
    }

    fn one_component(series: &Array3<f64>) -> Array3<u32> {
        Array3::ones(series.dim())
    }

    fn detect(series: &Array3<f64>, components: &Array3<u32>) -> CycleStepDetection {
        detect_cycle_steps(
            series.view(),
            components.view(),
            DEFAULT_CYCLE_STEP_HALF_WINDOW,
            DEFAULT_CYCLE_STEP_TOLERANCE_CYCLES,
        )
    }

    /// Thirty acquisitions; three pixels gain a whole cycle at epoch 15 and
    /// keep it. The step is flagged at epoch 15 on exactly those pixels, the
    /// offset persists from epoch 15 on, and nothing else is flagged.
    #[test]
    fn whole_cycle_step_at_epoch_15_flags_those_three_pixels_only() {
        let (rows, cols) = (10, 12);
        let mut series = smooth_series(30, rows, cols);
        let stepped = [(2, 3), (7, 8), (7, 9)];
        for k in 14..29 {
            for &(row, col) in &stepped {
                series[(k, row, col)] += TAU;
            }
        }
        let components = one_component(&series);
        let detection = detect(&series, &components);
        assert_eq!(detection.steps.dim(), (29, rows, cols));
        for ((k, row, col), &step) in detection.steps.indexed_iter() {
            let expected = i8::from(k == 14 && stepped.contains(&(row, col)));
            assert_eq!(step, expected, "step index {k} pixel ({row},{col})");
        }
        for ((row, col), &count) in detection.step_count.indexed_iter() {
            assert_eq!(count, u32::from(stepped.contains(&(row, col))));
        }
        assert_eq!(detection.per_date.len(), 29);
        for summary in &detection.per_date {
            let at_junction = summary.epoch == 15;
            assert_eq!(summary.flagged_pixels, if at_junction { 3 } else { 0 });
            assert_eq!(
                summary.offset_pixels,
                if summary.epoch >= 15 { 3 } else { 0 }
            );
            assert_eq!(summary.evaluable_pixels, rows * cols);
            if at_junction {
                assert!((summary.flagged_fraction - 3.0 / 120.0).abs() < 1e-12);
                assert!((summary.fraction_over_half_cycle - 3.0 / 120.0).abs() < 1e-12);
            } else {
                assert_eq!(summary.fraction_over_half_cycle, 0.0);
            }
        }
    }

    #[test]
    fn smooth_deformation_flags_nothing() {
        let series = smooth_series(30, 9, 9);
        let components = one_component(&series);
        let detection = detect(&series, &components);
        assert!(detection.steps.iter().all(|&step| step == 0));
        assert!(detection.step_count.iter().all(|&count| count == 0));
        assert!(detection
            .per_date
            .iter()
            .all(|summary| summary.flagged_pixels == 0 && summary.fraction_over_half_cycle == 0.0));
    }

    /// A 0.6-cycle step is not a whole cycle: it is not flagged, but it does
    /// count toward the over-half-cycle fraction; a step down by a whole cycle
    /// is flagged as -1 and cancels the offset.
    #[test]
    fn fractional_steps_are_counted_not_flagged_and_a_return_step_cancels() {
        let (rows, cols) = (6, 6);
        let mut series = smooth_series(12, rows, cols);
        for k in 4..11 {
            series[(k, 1, 1)] += 0.6 * TAU;
        }
        for k in 5..8 {
            series[(k, 4, 4)] += TAU;
        }
        let components = one_component(&series);
        let detection = detect(&series, &components);
        assert_eq!(detection.steps[(4, 1, 1)], 0);
        assert_eq!(detection.step_count[(1, 1)], 0);
        assert!((detection.per_date[4].fraction_over_half_cycle - 1.0 / 36.0).abs() < 1e-12);
        assert_eq!(detection.steps[(5, 4, 4)], 1);
        assert_eq!(detection.steps[(8, 4, 4)], -1);
        assert_eq!(detection.step_count[(4, 4)], 2);
        assert_eq!(detection.per_date[5].offset_pixels, 1);
        assert_eq!(detection.per_date[7].offset_pixels, 1);
        assert_eq!(detection.per_date[8].offset_pixels, 0);
    }

    /// Neighbours in another component do not vote, an unlabelled pixel is
    /// not judged, and a pixel with too few labelled finite neighbours is not
    /// judged either.
    #[test]
    fn components_and_sparse_neighbourhoods_bound_the_vote() {
        let (rows, cols) = (6, 8);
        let mut series = smooth_series(5, rows, cols);
        let mut components = one_component(&series);
        // Columns 4.. are a second component a whole cycle away at step 2.
        for row in 0..rows {
            for col in 4..cols {
                components[(2, row, col)] = 2;
                series[(2, row, col)] += TAU;
            }
        }
        // Pixel (0,0) carries no label at step 2; pixel (5,7) is alone in its
        // component there.
        components[(2, 0, 0)] = 0;
        components[(2, 5, 7)] = 3;
        let detection = detect(&series, &components);
        assert!(detection.steps.iter().all(|&step| step == 0));
        assert_eq!(detection.per_date[2].evaluable_pixels, rows * cols - 2);
        let mut nan_series = series.clone();
        nan_series[(2, 3, 3)] = f64::NAN;
        let detection = detect(&nan_series, &components);
        assert_eq!(detection.steps[(2, 3, 3)], 0);
        assert_eq!(detection.per_date[2].evaluable_pixels, rows * cols - 3);
        assert_eq!(detection.step_count, Array2::<u32>::zeros((rows, cols)));
    }
}
