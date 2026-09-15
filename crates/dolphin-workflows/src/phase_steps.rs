//! Per-date phase-step diagnostics on the inverted series (E4, 2026-09-15).
//!
//! On calibration block 1 every interferogram unwrapped as one connected
//! component and every loop closed to zero, yet 3 of 25 pixels in one GNSS
//! window sat a whole cycle from the ministack junction onward and 8% of the
//! frame stepped by ~0.6 cycle at that epoch. Neither the wrapped closure layer
//! nor the unwrapped loop QC can see a closure-consistent integer inconsistency,
//! so this module looks at the one place it is visible: the per-date series
//! `solve_time_series` returns, before corrections and the spatial reference.
//!
//! The junction diagnostic reports each pixel's step across a ministack boundary
//! relative to the median step of its connected component. In a closure-exact
//! network the inverted step reproduces the unwrapped nearest-neighbour
//! interferogram, so the wrapped linked-phase step is exactly `wrap(step)`: the
//! fractional part of the reported step *is* the linked-phase step, and the
//! integer part is what the unwrapper added. A separate wrapped layer would
//! carry nothing more, so the summary splits the two instead.

use dolphin_timeseries::CycleStepDateSummary;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Serialize};

/// Frame-fraction thresholds on `|step|`, in cycles.
pub const JUNCTION_STEP_THRESHOLDS_CYCLES: [f64; 3] = [0.25, 0.5, 0.75];

/// One ministack junction's per-pixel step across the boundary.
#[derive(Debug, Clone)]
pub struct JunctionStep {
    /// Per-pixel step from the last date of one ministack to the first of the
    /// next, minus the median step of the pixel's connected component, in
    /// cycles. NaN where the step is non-finite or the component has no label.
    pub step_cycles: Array2<f64>,
    /// Frame statistics of [`Self::step_cycles`].
    pub summary: JunctionStepSummary,
}

/// Identifier-free frame statistics for one ministack junction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionStepSummary {
    /// Acquisition index of the first date of the later ministack.
    pub epoch: usize,
    /// Pixels with a finite, labelled step.
    pub evaluable_pixels: usize,
    /// Fraction of evaluable pixels with `|step| > 0.25` cycle.
    pub fraction_over_quarter_cycle: f64,
    /// Fraction of evaluable pixels with `|step| > 0.5` cycle.
    pub fraction_over_half_cycle: f64,
    /// Fraction of evaluable pixels with `|step| > 0.75` cycle.
    pub fraction_over_three_quarter_cycle: f64,
    /// Fraction of evaluable pixels whose step rounds to a non-zero whole
    /// number of cycles: the unwrapper's integer disagreement with the
    /// component.
    pub fraction_integer_cycle: f64,
    /// Fraction of evaluable pixels whose step, wrapped to `(-0.5, 0.5]`,
    /// exceeds a quarter cycle in magnitude: the linked phase itself disagrees
    /// with the component across the junction.
    pub fraction_linked_over_quarter_cycle: f64,
}

/// Ministack-junction provenance: the boundaries and their frame statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionProvenance {
    /// `phase_linking.ministack_size` the boundaries were planned with.
    pub ministack_size: usize,
    /// One entry per boundary, in acquisition order; empty for a single ministack.
    pub junctions: Vec<JunctionStepSummary>,
}

/// Per-date integer-cycle step provenance: the detector's rule parameters, the
/// per-date frame fractions, and how many analysis-frame pixels carry at least
/// one detected step (the per-pixel counts are `cycle_step_flag.tif`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CycleStepProvenance {
    /// Neighbourhood half-width in pixels.
    pub half_window: usize,
    /// How close to a whole cycle a residual must be to count.
    pub tolerance_cycles: f64,
    /// Fewest same-component finite neighbours before a pixel is judged.
    pub min_neighbours: usize,
    /// Analysis-frame pixels with at least one detected step.
    pub flagged_pixels: usize,
    /// One entry per step index, in acquisition order.
    pub per_date: Vec<CycleStepDateSummary>,
}

/// Acquisition indices at which a new ministack starts: `real_start` of every
/// planned ministack after the first.
#[must_use]
pub fn junction_epochs(plan: &[dolphin_stack::MiniStack]) -> Vec<usize> {
    plan.iter().skip(1).map(|block| block.real_start).collect()
}

/// The connected-component labels that govern each per-date step. Step `k`
/// (acquisition `k` to `k + 1`) takes the labels of the `(k, k + 1)`
/// interferogram when the network has it, else of the first interferogram
/// ending at `k + 1`, else a single component wherever the step is finite.
#[must_use]
pub fn step_components(
    pairs: &[(usize, usize)],
    connected_components: ArrayView3<u32>,
    series: ArrayView3<f64>,
) -> Array3<u32> {
    let (steps, rows, cols) = series.dim();
    let mut out = Array3::zeros((steps, rows, cols));
    for k in 0..steps {
        let later = k + 1;
        let band = pairs
            .iter()
            .position(|&pair| pair == (k, later))
            .or_else(|| pairs.iter().position(|&(_, second)| second == later));
        match band {
            Some(band) => out
                .index_axis_mut(Axis(0), k)
                .assign(&connected_components.index_axis(Axis(0), band)),
            None => {
                let step = date_step(series, k);
                out.index_axis_mut(Axis(0), k)
                    .assign(&step.mapv(|value| u32::from(value.is_finite())));
            }
        }
    }
    out
}

/// The per-pixel phase step into acquisition `k + 1`, in radians: series band
/// `k` minus band `k - 1`, with acquisition 0 as the implicit zero band.
#[must_use]
pub fn date_step(series: ArrayView3<f64>, k: usize) -> Array2<f64> {
    let later = series.index_axis(Axis(0), k);
    match k {
        0 => later.to_owned(),
        _ => &later - &series.index_axis(Axis(0), k - 1),
    }
}

/// Wrap a value in cycles to `(-0.5, 0.5]`.
#[must_use]
pub fn wrap_cycles(cycles: f64) -> f64 {
    cycles - (cycles - 0.5).ceil()
}

/// The junction step diagnostic for each boundary epoch. `series` is the
/// `(n_dates - 1, rows, cols)` per-date series in radians referenced to
/// acquisition 0; `components` are the per-step labels from [`step_components`].
#[must_use]
pub fn junction_steps(
    series: ArrayView3<f64>,
    components: ArrayView3<u32>,
    epochs: &[usize],
) -> Vec<JunctionStep> {
    epochs
        .iter()
        .filter(|&&epoch| epoch >= 1 && epoch <= series.dim().0)
        .map(|&epoch| junction_step(series, components, epoch))
        .collect()
}

fn junction_step(
    series: ArrayView3<f64>,
    components: ArrayView3<u32>,
    epoch: usize,
) -> JunctionStep {
    let k = epoch - 1;
    let step = date_step(series, k).mapv(|value| value / std::f64::consts::TAU);
    let labels = components.index_axis(Axis(0), k);
    let medians = component_medians(step.view(), labels);
    let step_cycles = Array2::from_shape_fn(step.dim(), |point| {
        let label = labels[point];
        match (label, medians.get(&label)) {
            (0, _) | (_, None) => f64::NAN,
            (_, Some(median)) => step[point] - median,
        }
    });
    let summary = summarize_junction(step_cycles.view(), epoch);
    JunctionStep {
        step_cycles,
        summary,
    }
}

/// Median of the finite values per non-zero label.
fn component_medians(
    values: ArrayView2<f64>,
    labels: ArrayView2<u32>,
) -> std::collections::HashMap<u32, f64> {
    let mut grouped: std::collections::HashMap<u32, Vec<f64>> = std::collections::HashMap::new();
    ndarray::Zip::from(values)
        .and(labels)
        .for_each(|&value, &label| {
            if label != 0 && value.is_finite() {
                grouped.entry(label).or_default().push(value);
            }
        });
    grouped
        .into_iter()
        .map(|(label, mut values)| (label, median(&mut values)))
        .collect()
}

/// Median of a non-empty slice (upper median for even lengths).
fn median(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    let (_, median, _) = values.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
    *median
}

fn summarize_junction(step_cycles: ArrayView2<f64>, epoch: usize) -> JunctionStepSummary {
    let finite: Vec<f64> = step_cycles
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let fraction = |keep: &dyn Fn(f64) -> bool| match finite.is_empty() {
        true => 0.0,
        false => finite.iter().filter(|&&value| keep(value)).count() as f64 / finite.len() as f64,
    };
    let [quarter, half, three_quarter] = JUNCTION_STEP_THRESHOLDS_CYCLES;
    JunctionStepSummary {
        epoch,
        evaluable_pixels: finite.len(),
        fraction_over_quarter_cycle: fraction(&|value| value.abs() > quarter),
        fraction_over_half_cycle: fraction(&|value| value.abs() > half),
        fraction_over_three_quarter_cycle: fraction(&|value| value.abs() > three_quarter),
        fraction_integer_cycle: fraction(&|value| value.round() != 0.0),
        fraction_linked_over_quarter_cycle: fraction(&|value| wrap_cycles(value).abs() > quarter),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dolphin_stack::MiniStackPlanner;
    use std::f64::consts::TAU;

    /// A smooth series of `n_dates` acquisitions on a `rows x cols` frame: a
    /// per-date rate plus a spatial gradient, in radians.
    fn smooth_series(n_dates: usize, rows: usize, cols: usize) -> Array3<f64> {
        Array3::from_shape_fn((n_dates - 1, rows, cols), |(k, row, col)| {
            0.3 * (k + 1) as f64 + 0.01 * (row as f64 + col as f64) * (k + 1) as f64
        })
    }

    fn one_component(series: &Array3<f64>) -> Array3<u32> {
        Array3::ones(series.dim())
    }

    #[test]
    fn junction_epochs_are_the_real_start_of_every_later_ministack() {
        let plan = MiniStackPlanner {
            num_slc: 30,
            max_num_compressed: 5,
            output_reference_idx: 0,
            compressed_slc_plan: Default::default(),
        }
        .plan(15)
        .unwrap();
        assert_eq!(junction_epochs(&plan), vec![15]);
        let plan = MiniStackPlanner {
            num_slc: 7,
            max_num_compressed: 5,
            output_reference_idx: 0,
            compressed_slc_plan: Default::default(),
        }
        .plan(3)
        .unwrap();
        assert_eq!(junction_epochs(&plan), vec![3, 6]);
        assert!(junction_epochs(&plan[..1]).is_empty());
    }

    /// Two ministacks of three dates; one pixel carries a whole extra cycle
    /// from the junction onward. The diagnostic names that pixel, the
    /// fraction is one pixel of the frame, and the extra cycle is an unwrap
    /// integer (its wrapped linked-phase step is zero), not a linked-phase step.
    #[test]
    fn injected_cycle_at_the_junction_is_reported_at_that_pixel_only() {
        let (rows, cols) = (4, 5);
        let mut series = smooth_series(6, rows, cols);
        for k in 2..5 {
            series[(k, 1, 3)] += TAU;
        }
        let components = one_component(&series);
        let junctions = junction_steps(series.view(), components.view(), &[3]);
        assert_eq!(junctions.len(), 1);
        let junction = &junctions[0];
        assert_eq!(junction.summary.epoch, 3);
        assert_eq!(junction.summary.evaluable_pixels, rows * cols);
        assert!((junction.step_cycles[(1, 3)] - 1.0).abs() < 0.02);
        for ((row, col), &value) in junction.step_cycles.indexed_iter() {
            if (row, col) != (1, 3) {
                assert!(value.abs() < 0.02, "pixel ({row},{col}) stepped {value}");
            }
        }
        let expected = 1.0 / (rows * cols) as f64;
        assert!((junction.summary.fraction_over_three_quarter_cycle - expected).abs() < 1e-12);
        assert!((junction.summary.fraction_over_half_cycle - expected).abs() < 1e-12);
        assert!((junction.summary.fraction_over_quarter_cycle - expected).abs() < 1e-12);
        assert!((junction.summary.fraction_integer_cycle - expected).abs() < 1e-12);
        assert_eq!(junction.summary.fraction_linked_over_quarter_cycle, 0.0);
    }

    #[test]
    fn smooth_series_reports_no_step_at_any_junction() {
        let series = smooth_series(9, 3, 3);
        let components = one_component(&series);
        for junction in junction_steps(series.view(), components.view(), &[3, 6]) {
            assert_eq!(junction.summary.fraction_over_quarter_cycle, 0.0);
            assert_eq!(junction.summary.fraction_integer_cycle, 0.0);
            assert_eq!(junction.summary.fraction_linked_over_quarter_cycle, 0.0);
        }
    }

    /// A half-cycle linked-phase disagreement is reported as such, and a
    /// pixel with no component label or a non-finite step is not judged.
    #[test]
    fn fractional_steps_and_unlabelled_pixels_are_kept_apart() {
        let mut series = smooth_series(4, 2, 4);
        series[(2, 0, 0)] += 0.4 * TAU;
        series[(2, 0, 1)] = f64::NAN;
        let mut components = one_component(&series);
        components[(2, 1, 3)] = 0;
        let junction = junction_steps(series.view(), components.view(), &[3]).remove(0);
        assert_eq!(junction.summary.evaluable_pixels, 6);
        assert!((junction.step_cycles[(0, 0)] - 0.4).abs() < 0.02);
        assert!(junction.step_cycles[(0, 1)].is_nan());
        assert!(junction.step_cycles[(1, 3)].is_nan());
        assert!((junction.summary.fraction_over_quarter_cycle - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(junction.summary.fraction_over_half_cycle, 0.0);
        assert_eq!(junction.summary.fraction_integer_cycle, 0.0);
        assert!((junction.summary.fraction_linked_over_quarter_cycle - 1.0 / 6.0).abs() < 1e-12);
    }

    /// Each component is judged against its own median: a whole-cycle datum
    /// difference between two components is not a step in either.
    #[test]
    fn components_carry_their_own_median() {
        let mut series = smooth_series(4, 2, 4);
        let mut components = one_component(&series);
        for col in 2..4 {
            for row in 0..2 {
                components[(2, row, col)] = 2;
                series[(2, row, col)] += TAU;
            }
        }
        let junction = junction_steps(series.view(), components.view(), &[3]).remove(0);
        assert_eq!(junction.summary.fraction_over_quarter_cycle, 0.0);
    }

    #[test]
    fn step_components_follow_the_nearest_neighbour_interferogram() {
        let mut series = smooth_series(4, 1, 2);
        series[(2, 0, 1)] = f64::NAN;
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let components = Array3::from_shape_fn((3, 1, 2), |(band, _, _)| band as u32 + 10);
        let per_step = step_components(&pairs, components.view(), series.view());
        assert_eq!(per_step[(0, 0, 0)], 10, "step 0 uses (0,1)");
        assert_eq!(per_step[(1, 0, 0)], 12, "step 1 uses (1,2), not (0,2)");
        assert_eq!(
            per_step[(2, 0, 0)],
            1,
            "no interferogram ends at 3: one component"
        );
        assert_eq!(per_step[(2, 0, 1)], 0, "a non-finite step has no component");
    }

    #[test]
    fn wrap_cycles_maps_to_the_half_open_unit_interval() {
        assert!((wrap_cycles(1.0)).abs() < 1e-12);
        assert!((wrap_cycles(0.6) + 0.4).abs() < 1e-12);
        assert!((wrap_cycles(-0.6) - 0.4).abs() < 1e-12);
        assert!((wrap_cycles(0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_cycles(-0.5) - 0.5).abs() < 1e-12);
    }
}
