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
//! relative to the median step of the same-component pixels around it, within
//! the #184 junction neighbourhood. In a closure-exact network the inverted
//! step reproduces the unwrapped nearest-neighbour interferogram, so the wrapped
//! linked-phase step is exactly `wrap(step)`: the fractional part of the
//! reported step *is* the linked-phase step, and the integer part is the
//! pixel's integer-cycle disagreement with its surroundings. A separate wrapped
//! layer would carry nothing more, so the summary splits the two instead.
//!
//! The reference is local, not the component median (issue #207). On Houston
//! T034-071054-IW3 the 12-day step at dates 45 and 75 spans 3.2-3.6 cycles
//! across a 20 km frame with no neighbour-to-neighbour jump, and OPERA DISP-S1,
//! unwrapped independently, carries the same field (r = 0.998, 0.08% of cells
//! a whole cycle apart). Against the component median that real field read as
//! half the frame stepping a whole cycle; against the neighbourhood it is the
//! 0.05% of cells every junction shows.

use dolphin_timeseries::{neighbourhood_residuals, CycleStepDateSummary};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Serialize};

/// Frame-fraction thresholds on `|step|`, in cycles.
pub const JUNCTION_STEP_THRESHOLDS_CYCLES: [f64; 3] = [0.25, 0.5, 0.75];

/// One ministack junction's per-pixel step across the boundary.
#[derive(Debug, Clone)]
pub struct JunctionStep {
    /// Per-pixel step from the last date of one ministack to the first of the
    /// next, minus the median step of the same-component pixels within the
    /// junction neighbourhood, in cycles. NaN where the step is non-finite, the
    /// pixel has no component label, or too few such neighbours.
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
    /// number of cycles: an integer-cycle disagreement with the surrounding
    /// same-component pixels.
    pub fraction_integer_cycle: f64,
    /// Fraction of evaluable pixels whose step, wrapped to `(-0.5, 0.5]`,
    /// exceeds a quarter cycle in magnitude: the linked phase itself disagrees
    /// with its surroundings across the junction.
    pub fraction_linked_over_quarter_cycle: f64,
    /// How the later ministack was referenced. `None` in records written
    /// before issue #192, which all used the oldest carried compressed SLC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<MinistackReference>,
}

/// The reference of one ministack in its combined stack (carried compressed
/// SLCs first, then real SLCs), as the planner resolved it. The compressed-SLC
/// plan alone does not say this: `always_first` meant the oldest carried
/// compressed SLC before issue #192 and the newest since.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinistackReference {
    /// Compressed SLCs carried into the ministack.
    pub num_compressed: usize,
    /// Position the phase-linking output was referenced to.
    pub output_reference_idx: usize,
    /// Position the ministack's own compressed SLC was referenced to.
    pub compressed_reference_idx: usize,
}

impl MinistackReference {
    /// The resolved reference of one planned ministack.
    ///
    /// # Errors
    /// Returns `Err` when a planned index is outside the combined ministack.
    pub fn of(block: &dolphin_stack::MiniStack) -> Result<Self, &'static str> {
        Ok(Self {
            num_compressed: block.num_compressed,
            output_reference_idx: block.resolved_output_reference_idx()?,
            compressed_reference_idx: block.resolved_compressed_reference_idx()?,
        })
    }
}

/// Ministack-junction provenance: the boundaries and their frame statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionProvenance {
    /// `phase_linking.ministack_size` the boundaries were planned with.
    pub ministack_size: usize,
    /// Half-width in pixels of the neighbourhood each junction step is measured
    /// against. `0` in records written before issue #207, which measured
    /// against the median of the whole connected component, and when there is
    /// no junction.
    #[serde(default)]
    pub half_window: usize,
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
    /// Acquisition indices judged by the junction rule (issue #184): the first
    /// date of each ministack after the first. Empty in records written before
    /// the rule existed.
    #[serde(default)]
    pub junction_epochs: Vec<usize>,
    /// Junction-rule neighbourhood half-width in pixels; at a junction a step
    /// more than half a cycle from that neighbourhood's median is flagged.
    /// `0` when no junction was judged.
    #[serde(default)]
    pub junction_half_window: usize,
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
/// acquisition 0; `components` are the per-step labels from [`step_components`];
/// `half_window` is the neighbourhood each step is measured against (the #184
/// junction half-window).
#[must_use]
pub fn junction_steps(
    series: ArrayView3<f64>,
    components: ArrayView3<u32>,
    epochs: &[usize],
    half_window: usize,
) -> Vec<JunctionStep> {
    epochs
        .iter()
        .filter(|&&epoch| epoch >= 1 && epoch <= series.dim().0)
        .map(|&epoch| junction_step(series, components, epoch, half_window))
        .collect()
}

fn junction_step(
    series: ArrayView3<f64>,
    components: ArrayView3<u32>,
    epoch: usize,
    half_window: usize,
) -> JunctionStep {
    let k = epoch - 1;
    let step = date_step(series, k).mapv(|value| value / std::f64::consts::TAU);
    let step_cycles =
        neighbourhood_residuals(step.view(), components.index_axis(Axis(0), k), half_window);
    let summary = summarize_junction(step_cycles.view(), epoch);
    JunctionStep {
        step_cycles,
        summary,
    }
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
        reference: None,
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
            output_reference_idx: None,
            compressed_slc_plan: Default::default(),
        }
        .plan(15)
        .unwrap();
        assert_eq!(junction_epochs(&plan), vec![15]);
        let plan = MiniStackPlanner {
            num_slc: 7,
            max_num_compressed: 5,
            output_reference_idx: None,
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
        let junctions = junction_steps(series.view(), components.view(), &[3], 2);
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
        for junction in junction_steps(series.view(), components.view(), &[3, 6], 2) {
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
        let junction = junction_steps(series.view(), components.view(), &[3], 3).remove(0);
        assert_eq!(junction.summary.evaluable_pixels, 6);
        assert!((junction.step_cycles[(0, 0)] - 0.4).abs() < 0.02);
        assert!(junction.step_cycles[(0, 1)].is_nan());
        assert!(junction.step_cycles[(1, 3)].is_nan());
        assert!((junction.summary.fraction_over_quarter_cycle - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(junction.summary.fraction_over_half_cycle, 0.0);
        assert_eq!(junction.summary.fraction_integer_cycle, 0.0);
        assert!((junction.summary.fraction_linked_over_quarter_cycle - 1.0 / 6.0).abs() < 1e-12);
    }

    /// Each pixel is judged against neighbours of its own component: a
    /// whole-cycle datum difference between two adjacent components is not a
    /// step in either.
    #[test]
    fn components_carry_their_own_median() {
        let mut series = smooth_series(4, 3, 6);
        let mut components = one_component(&series);
        for col in 3..6 {
            for row in 0..3 {
                components[(2, row, col)] = 2;
                series[(2, row, col)] += TAU;
            }
        }
        let junction = junction_steps(series.view(), components.view(), &[3], 3).remove(0);
        assert_eq!(junction.summary.evaluable_pixels, 18);
        assert_eq!(junction.summary.fraction_over_quarter_cycle, 0.0);
    }

    /// Issue #207: a spatially smooth step several cycles wide across the frame
    /// (Houston T034 dates 45 and 75: 3.2-3.6 cycles over 20 km, matched by
    /// OPERA DISP-S1) is no step at all against the surroundings, though most
    /// of the frame lies beyond half a cycle from the frame median. A whole
    /// cycle added at one pixel on top of it is still reported there alone.
    #[test]
    fn smooth_field_wider_than_a_cycle_is_not_an_integer_step() {
        let (rows, cols) = (6, 40);
        let ramp = |col: usize| 3.0 * col as f64 / (cols - 1) as f64;
        let mut series = smooth_series(4, rows, cols);
        for row in 0..rows {
            for col in 0..cols {
                series[(2, row, col)] += ramp(col) * TAU;
            }
        }
        let frame: Vec<f64> = (0..cols).map(ramp).collect();
        let frame_median = frame[cols / 2];
        let beyond_half_from_frame_median = frame
            .iter()
            .filter(|&&v| (v - frame_median).abs() > 0.5)
            .count() as f64
            / cols as f64;
        assert!(beyond_half_from_frame_median > 0.5);

        let components = one_component(&series);
        let junction = junction_steps(series.view(), components.view(), &[3], 2).remove(0);
        assert_eq!(junction.summary.evaluable_pixels, rows * cols);
        assert_eq!(junction.summary.fraction_over_quarter_cycle, 0.0);
        assert_eq!(junction.summary.fraction_integer_cycle, 0.0);

        series[(2, 3, 20)] += TAU;
        let junction = junction_steps(series.view(), components.view(), &[3], 2).remove(0);
        assert!((junction.step_cycles[(3, 20)] - 1.0).abs() < 0.1);
        let expected = 1.0 / (rows * cols) as f64;
        assert!((junction.summary.fraction_integer_cycle - expected).abs() < 1e-12);
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
