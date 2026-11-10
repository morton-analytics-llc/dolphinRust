//! Put overlapping bounded windows on one datum (issue #153).
//!
//! Each bounded run is referenced to its own pixel, so two windows cut from the
//! same burst stack differ by an unknown constant in every date of the series and
//! in velocity. [`estimate_window_datum`] estimates those constants from the
//! pixels the windows share.
//!
//! # Estimator
//!
//! For a window pair `(a, b)` and date `t`, every shared usable pixel `p` gives
//! `d_p = disp_b(p, t) − disp_a(p, t)`, an observation of `o_a(t) − o_b(t)`,
//! where adding `o_w(t)` to window `w` puts it on the common datum. The
//! adjustment minimises
//!
//! `Σ_pairs Σ_p w_p (o_a − o_b − d_p)²`,  `w_p = γ_a(p) · γ_b(p)`
//!
//! with `γ` each window's temporal coherence. Per pair this reduces exactly to
//! one weighted mean `m_ab = Σ w_p d_p / W_ab` with weight `W_ab = Σ w_p`, so the
//! solve is a weighted graph Laplacian over the windows with the anchor window's
//! offset fixed. The velocity offset uses the same weights. The tophu inter-tile
//! reconciliation in `dolphin-unwrap` estimates integer cycles over a
//! max-reliability spanning forest; a continuous offset uses every overlap at
//! once, so a least-squares adjustment fits better here.
//!
//! A pixel is usable in a pair when both windows mark it valid, it has a finite
//! displacement on every date and a finite velocity in both, both coherences are
//! positive, and in each window it lies in that window's reference unwrap
//! component (`unwrap_component`, issue #152) when the window carries one.

use std::collections::VecDeque;

use ndarray::{ArrayView2, ArrayView3};

use crate::displacement::DisplacementOutput;

/// One bounded window's products, as consumed by [`estimate_window_datum`].
#[derive(Debug, Clone, Copy)]
pub struct DatumWindow<'a> {
    /// Per-date cumulative displacement, `(n_dates - 1, rows, cols)`.
    pub displacement: ArrayView3<'a, f64>,
    /// Velocity, `(rows, cols)`.
    pub velocity: ArrayView2<'a, f64>,
    /// Temporal coherence, `(rows, cols)`; the adjustment weight.
    pub temporal_coherence: ArrayView2<'a, f64>,
    /// Published validity, `(rows, cols)`.
    pub validity_mask: ArrayView2<'a, bool>,
    /// Joint unwrap component per pixel (issue #152), when the run labelled one.
    pub unwrap_component: Option<ArrayView2<'a, u32>>,
    /// The window's spatial reference pixel.
    pub reference_point: Option<(usize, usize)>,
    /// Acquisition days from acquisition 0.
    pub acquisition_days: &'a [f64],
    /// GDAL geotransform of the window grid.
    pub geotransform: [f64; 6],
    /// EPSG of the window grid.
    pub epsg: Option<u32>,
}

impl<'a> From<&'a DisplacementOutput> for DatumWindow<'a> {
    fn from(output: &'a DisplacementOutput) -> Self {
        Self {
            displacement: output.displacement.view(),
            velocity: output.velocity.view(),
            temporal_coherence: output.temporal_coherence.view(),
            validity_mask: output.validity_mask.view(),
            unwrap_component: output.unwrap_component.as_ref().map(|labels| labels.view()),
            reference_point: output.reference_point,
            acquisition_days: &output.acquisition_days,
            geotransform: output.geotransform,
            epsg: output.epsg,
        }
    }
}

/// Which window fixes the datum, and at what values.
#[derive(Debug, Clone, PartialEq)]
pub enum DatumAnchor {
    /// This window keeps its own datum (all of its offsets are zero).
    Window(usize),
    /// This window's offsets are fixed to externally supplied values: one per
    /// displacement date, plus the velocity offset.
    Values {
        /// Index of the anchored window.
        window: usize,
        /// Offset added to each displacement date of the anchored window.
        displacement: Vec<f64>,
        /// Offset added to the anchored window's velocity.
        velocity: f64,
    },
}

/// Options for [`estimate_window_datum`].
#[derive(Debug, Clone, PartialEq)]
pub struct DatumOptions {
    /// The datum the other windows are tied to.
    pub anchor: DatumAnchor,
    /// Smallest usable shared-pixel count for a pair whose grids overlap.
    pub min_overlap_pixels: usize,
}

/// Residuals of one overlap after adjustment.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapResidual {
    /// First window of the pair.
    pub window_a: usize,
    /// Second window of the pair (`window_a < window_b`).
    pub window_b: usize,
    /// Usable shared pixels.
    pub pixel_count: usize,
    /// Per date: median of `(disp_a + o_a) − (disp_b + o_b)` over the pixels.
    pub displacement_median: Vec<f64>,
    /// Per date: median absolute deviation of the same residual.
    pub displacement_mad: Vec<f64>,
    /// Median velocity residual.
    pub velocity_median: f64,
    /// Median absolute deviation of the velocity residual.
    pub velocity_mad: f64,
}

/// Datum adjustment of a set of windows.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowDatum {
    /// `displacement_offsets[w][t]`: add to window `w`, displacement date `t`.
    pub displacement_offsets: Vec<Vec<f64>>,
    /// `velocity_offsets[w]`: add to window `w`'s velocity.
    pub velocity_offsets: Vec<f64>,
    /// One entry per overlapping pair.
    pub overlaps: Vec<OverlapResidual>,
}

/// Typed failures of [`estimate_window_datum`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DatumError {
    /// The windows do not share a date list, grid spacing, CRS or pixel lattice,
    /// or the anchor does not fit them.
    #[error("windows cannot share a datum: {0}")]
    Incompatible(String),
    /// A pair's grids overlap but too few shared pixels are usable.
    #[error(
        "windows {window_a} and {window_b} share {pixels} usable pixels; at least {minimum} are required"
    )]
    InsufficientOverlap {
        /// First window.
        window_a: usize,
        /// Second window.
        window_b: usize,
        /// Usable shared pixels.
        pixels: usize,
        /// Configured minimum.
        minimum: usize,
    },
    /// Some windows have no chain of overlaps to the anchor.
    #[error("windows {unreached:?} have no overlap path to anchor window {anchor}")]
    Disconnected {
        /// The anchor window.
        anchor: usize,
        /// Windows the anchor cannot reach.
        unreached: Vec<usize>,
    },
}

struct Pair {
    a: usize,
    b: usize,
    weight: f64,
    displacement_mean: Vec<f64>,
    velocity_mean: f64,
    displacement_differences: Vec<Vec<f64>>,
    velocity_differences: Vec<f64>,
}

/// Estimate per-window, per-date displacement offsets and velocity offsets that
/// put `windows` on the anchor's datum. See the module docs for the estimator.
///
/// # Errors
/// [`DatumError::Incompatible`] when the windows or anchor do not fit together,
/// [`DatumError::InsufficientOverlap`] when an overlapping pair has fewer usable
/// pixels than `options.min_overlap_pixels`, and [`DatumError::Disconnected`]
/// when a window has no chain of overlaps to the anchor.
pub fn estimate_window_datum(
    windows: &[DatumWindow<'_>],
    options: &DatumOptions,
) -> Result<WindowDatum, DatumError> {
    let n_dates = check_compatible(windows)?;
    let (anchor, anchor_displacement, anchor_velocity) = match &options.anchor {
        DatumAnchor::Window(window) => (*window, vec![0.0; n_dates], 0.0),
        DatumAnchor::Values {
            window,
            displacement,
            velocity,
        } => (*window, displacement.clone(), *velocity),
    };
    if anchor >= windows.len() {
        return Err(DatumError::Incompatible(format!(
            "anchor window {anchor} is out of range for {} windows",
            windows.len()
        )));
    }
    if anchor_displacement.len() != n_dates {
        return Err(DatumError::Incompatible(format!(
            "anchor supplies {} displacement offsets for {n_dates} dates",
            anchor_displacement.len()
        )));
    }
    let mut pairs = Vec::new();
    for a in 0..windows.len() {
        for b in a + 1..windows.len() {
            let Some(pair) = overlap(windows, a, b, n_dates)? else {
                continue;
            };
            if pair.velocity_differences.len() < options.min_overlap_pixels.max(1) {
                return Err(DatumError::InsufficientOverlap {
                    window_a: a,
                    window_b: b,
                    pixels: pair.velocity_differences.len(),
                    minimum: options.min_overlap_pixels.max(1),
                });
            }
            pairs.push(pair);
        }
    }
    check_connected(windows.len(), anchor, &pairs)?;

    let mut displacement_offsets = vec![vec![0.0; n_dates]; windows.len()];
    for date in 0..n_dates {
        let solved = solve_offsets(
            windows.len(),
            anchor,
            anchor_displacement[date],
            &pairs,
            |p| p.displacement_mean[date],
        );
        for (window, value) in solved.into_iter().enumerate() {
            displacement_offsets[window][date] = value;
        }
    }
    let velocity_offsets = solve_offsets(windows.len(), anchor, anchor_velocity, &pairs, |p| {
        p.velocity_mean
    });
    let overlaps = pairs
        .iter()
        .map(|pair| {
            let shift = |values: &[f64], a: f64, b: f64| -> Vec<f64> {
                values.iter().map(|d| (a - b) - d).collect()
            };
            let (displacement_median, displacement_mad) = (0..n_dates)
                .map(|date| {
                    median_mad(shift(
                        &pair.displacement_differences[date],
                        displacement_offsets[pair.a][date],
                        displacement_offsets[pair.b][date],
                    ))
                })
                .unzip();
            let (velocity_median, velocity_mad) = median_mad(shift(
                &pair.velocity_differences,
                velocity_offsets[pair.a],
                velocity_offsets[pair.b],
            ));
            OverlapResidual {
                window_a: pair.a,
                window_b: pair.b,
                pixel_count: pair.velocity_differences.len(),
                displacement_median,
                displacement_mad,
                velocity_median,
                velocity_mad,
            }
        })
        .collect();
    Ok(WindowDatum {
        displacement_offsets,
        velocity_offsets,
        overlaps,
    })
}

/// The shared displacement date count, after checking every window shares the
/// date list, CRS, pixel spacing and an integer pixel lattice.
fn check_compatible(windows: &[DatumWindow<'_>]) -> Result<usize, DatumError> {
    let first = windows
        .first()
        .ok_or_else(|| DatumError::Incompatible("no windows supplied".into()))?;
    let n_dates = first.displacement.dim().0;
    for (index, window) in windows.iter().enumerate() {
        let (dates, rows, cols) = window.displacement.dim();
        let shapes_match = [
            window.velocity.dim(),
            window.temporal_coherence.dim(),
            window.validity_mask.dim(),
        ]
        .iter()
        .chain(window.unwrap_component.as_ref().map(ArrayView2::dim).iter())
        .all(|&shape| shape == (rows, cols));
        let gt = window.geotransform;
        let origin_col = (gt[0] - first.geotransform[0]) / gt[1];
        let origin_row = (gt[3] - first.geotransform[3]) / gt[5];
        let reason = if dates != n_dates || window.acquisition_days != first.acquisition_days {
            Some("date lists differ")
        } else if !shapes_match {
            Some("layer shapes differ within the window")
        } else if window.epsg != first.epsg {
            Some("CRS differs")
        } else if gt[1] != first.geotransform[1]
            || gt[5] != first.geotransform[5]
            || gt[2] != 0.0
            || gt[4] != 0.0
        {
            Some("pixel spacing differs or the grid is rotated")
        } else if (origin_col - origin_col.round()).abs() > 1e-6
            || (origin_row - origin_row.round()).abs() > 1e-6
        {
            Some("grid origin is off the shared pixel lattice")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(DatumError::Incompatible(format!(
                "window {index}: {reason}"
            )));
        }
    }
    Ok(n_dates)
}

/// Usable shared pixels of windows `a` and `b`, or `None` when their grids do
/// not overlap.
fn overlap(
    windows: &[DatumWindow<'_>],
    a: usize,
    b: usize,
    n_dates: usize,
) -> Result<Option<Pair>, DatumError> {
    let origin = |window: &DatumWindow<'_>| {
        let gt = window.geotransform;
        let base = windows[0].geotransform;
        (
            ((gt[3] - base[3]) / gt[5]).round() as i64,
            ((gt[0] - base[0]) / gt[1]).round() as i64,
        )
    };
    let (wa, wb) = (&windows[a], &windows[b]);
    let ((ra, ca), (rb, cb)) = (origin(wa), origin(wb));
    let (_, rows_a, cols_a) = wa.displacement.dim();
    let (_, rows_b, cols_b) = wb.displacement.dim();
    let row_start = ra.max(rb);
    let row_stop = (ra + rows_a as i64).min(rb + rows_b as i64);
    let col_start = ca.max(cb);
    let col_stop = (ca + cols_a as i64).min(cb + cols_b as i64);
    if row_start >= row_stop || col_start >= col_stop {
        return Ok(None);
    }
    let reference_component = |window: &DatumWindow<'_>, index: usize| {
        window.unwrap_component.map(|labels| {
            window
                .reference_point
                .and_then(|point| labels.get(point).copied())
                .filter(|&label| label != 0)
                .ok_or_else(|| {
                    DatumError::Incompatible(format!(
                        "window {index} labels unwrap components but its reference has none"
                    ))
                })
        })
    };
    let component_a = reference_component(wa, a).transpose()?;
    let component_b = reference_component(wb, b).transpose()?;
    let usable = |window: &DatumWindow<'_>, component: Option<u32>, point: (usize, usize)| {
        window.validity_mask[point]
            && window.velocity[point].is_finite()
            && window.temporal_coherence[point] > 0.0
            && window
                .displacement
                .slice(ndarray::s![.., point.0, point.1])
                .iter()
                .all(|value| value.is_finite())
            && component
                .is_none_or(|label| window.unwrap_component.map(|l| l[point]) == Some(label))
    };
    let mut weight = 0.0;
    let mut displacement_sum = vec![0.0; n_dates];
    let mut velocity_sum = 0.0;
    let mut displacement_differences = vec![Vec::new(); n_dates];
    let mut velocity_differences = Vec::new();
    for row in row_start..row_stop {
        for col in col_start..col_stop {
            let pa = ((row - ra) as usize, (col - ca) as usize);
            let pb = ((row - rb) as usize, (col - cb) as usize);
            if !(usable(wa, component_a, pa) && usable(wb, component_b, pb)) {
                continue;
            }
            let w = wa.temporal_coherence[pa] * wb.temporal_coherence[pb];
            weight += w;
            for date in 0..n_dates {
                let d = wb.displacement[(date, pb.0, pb.1)] - wa.displacement[(date, pa.0, pa.1)];
                displacement_sum[date] += w * d;
                displacement_differences[date].push(d);
            }
            let d = wb.velocity[pb] - wa.velocity[pa];
            velocity_sum += w * d;
            velocity_differences.push(d);
        }
    }
    let safe = weight.max(f64::MIN_POSITIVE);
    Ok(Some(Pair {
        a,
        b,
        weight,
        displacement_mean: displacement_sum.iter().map(|sum| sum / safe).collect(),
        velocity_mean: velocity_sum / safe,
        displacement_differences,
        velocity_differences,
    }))
}

fn check_connected(n: usize, anchor: usize, pairs: &[Pair]) -> Result<(), DatumError> {
    let mut reached = vec![false; n];
    reached[anchor] = true;
    let mut queue = VecDeque::from([anchor]);
    while let Some(window) = queue.pop_front() {
        for pair in pairs {
            let next = match (pair.a == window, pair.b == window) {
                (true, _) => pair.b,
                (_, true) => pair.a,
                _ => continue,
            };
            if !reached[next] {
                reached[next] = true;
                queue.push_back(next);
            }
        }
    }
    let unreached: Vec<usize> = (0..n).filter(|&window| !reached[window]).collect();
    if unreached.is_empty() {
        Ok(())
    } else {
        Err(DatumError::Disconnected { anchor, unreached })
    }
}

/// Weighted least squares over the overlap graph: minimise
/// `Σ W_ab (o_a − o_b − m_ab)²` with `o_anchor` fixed, by Gaussian elimination
/// on the reduced Laplacian (positive definite once the graph is connected).
fn solve_offsets(
    n: usize,
    anchor: usize,
    anchor_value: f64,
    pairs: &[Pair],
    mean: impl Fn(&Pair) -> f64,
) -> Vec<f64> {
    let mut matrix = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];
    for pair in pairs {
        let (a, b, w, m) = (pair.a, pair.b, pair.weight, mean(pair));
        matrix[a][a] += w;
        matrix[b][b] += w;
        matrix[a][b] -= w;
        matrix[b][a] -= w;
        rhs[a] += w * m;
        rhs[b] -= w * m;
    }
    // Fix the anchor: replace its row with o_anchor = value.
    matrix[anchor] = vec![0.0; n];
    matrix[anchor][anchor] = 1.0;
    rhs[anchor] = anchor_value;
    for row in 0..n {
        if row != anchor {
            let coupling = matrix[row][anchor];
            matrix[row][anchor] = 0.0;
            rhs[row] -= coupling * anchor_value;
        }
    }
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|&x, &y| matrix[x][pivot].abs().total_cmp(&matrix[y][pivot].abs()))
            .expect("non-empty pivot range");
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let (head, tail) = matrix.split_at_mut(pivot + 1);
        let pivot_row = &head[pivot];
        for (offset, row) in tail.iter_mut().enumerate() {
            let factor = row[pivot] / pivot_row[pivot];
            if factor != 0.0 {
                for (value, &above) in row[pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                    *value -= factor * above;
                }
                rhs[pivot + 1 + offset] -= factor * rhs[pivot];
            }
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n)
            .map(|col| matrix[row][col] * solution[col])
            .sum();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    solution
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn median_mad(mut values: Vec<f64>) -> (f64, f64) {
    let center = median(&mut values);
    let mut deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    (center, median(&mut deviations))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array2, Array3};

    struct Owned {
        displacement: Array3<f64>,
        velocity: Array2<f64>,
        coherence: Array2<f64>,
        validity: Array2<bool>,
        components: Option<Array2<u32>>,
        reference: Option<(usize, usize)>,
        geotransform: [f64; 6],
    }

    const DAYS: [f64; 4] = [0.0, 12.0, 24.0, 36.0];

    impl Owned {
        fn view(&self) -> DatumWindow<'_> {
            DatumWindow {
                displacement: self.displacement.view(),
                velocity: self.velocity.view(),
                temporal_coherence: self.coherence.view(),
                validity_mask: self.validity.view(),
                unwrap_component: self.components.as_ref().map(Array2::view),
                reference_point: self.reference,
                acquisition_days: &DAYS,
                geotransform: self.geotransform,
                epsg: Some(32611),
            }
        }
    }

    /// A full-frame field: smooth in space, distinct per date.
    fn frame(rows: usize, cols: usize) -> (Array3<f64>, Array2<f64>) {
        let displacement = Array3::from_shape_fn((3, rows, cols), |(t, r, c)| {
            0.01 * (t + 1) as f64 * ((r as f64 * 0.3).sin() + (c as f64 * 0.17).cos())
        });
        let velocity =
            Array2::from_shape_fn((rows, cols), |(r, c)| 0.002 * r as f64 - 0.001 * c as f64);
        (displacement, velocity)
    }

    /// Cut a window at column `col_start`, width `width`, from the frame, then
    /// shift each date and the velocity by planted constants (a re-reference).
    fn cut(
        full: &(Array3<f64>, Array2<f64>),
        col_start: usize,
        width: usize,
        shift: &[f64; 3],
        velocity_shift: f64,
    ) -> Owned {
        let mut displacement = full
            .0
            .slice(s![.., .., col_start..col_start + width])
            .to_owned();
        for (date, offset) in shift.iter().enumerate() {
            displacement
                .index_axis_mut(ndarray::Axis(0), date)
                .mapv_inplace(|v| v + offset);
        }
        let velocity = full
            .1
            .slice(s![.., col_start..col_start + width])
            .mapv(|v| v + velocity_shift);
        let rows = full.1.nrows();
        Owned {
            displacement,
            velocity,
            coherence: Array2::from_elem((rows, width), 0.8),
            validity: Array2::from_elem((rows, width), true),
            components: None,
            reference: Some((0, 0)),
            geotransform: [
                500_000.0 + col_start as f64 * 30.0,
                30.0,
                0.0,
                4_000_000.0,
                0.0,
                -30.0,
            ],
        }
    }

    fn options(min_overlap_pixels: usize) -> DatumOptions {
        DatumOptions {
            anchor: DatumAnchor::Window(0),
            min_overlap_pixels,
        }
    }

    /// Contract: windows cut from one full-frame run and re-referenced by planted
    /// per-date constants come back to the full-frame series within 1e-6.
    #[test]
    fn re_referenced_windows_return_to_the_full_frame_series() {
        let full = frame(12, 40);
        let shifts = [
            [0.0, 0.0, 0.0],
            [0.031, -0.012, 0.007],
            [-0.044, 0.019, 0.052],
        ];
        let velocity_shifts = [0.0, 0.004, -0.009];
        let windows: Vec<Owned> = [(0, 16), (12, 16), (24, 16)]
            .iter()
            .zip(shifts.iter().zip(velocity_shifts))
            .map(|(&(start, width), (shift, v))| cut(&full, start, width, shift, v))
            .collect();
        let views: Vec<DatumWindow<'_>> = windows.iter().map(Owned::view).collect();
        let datum = estimate_window_datum(&views, &options(10)).unwrap();
        for (index, (window, start)) in windows.iter().zip([0, 12, 24]).enumerate() {
            for ((date, row, col), value) in window.displacement.indexed_iter() {
                let adjusted = value + datum.displacement_offsets[index][date];
                assert!((adjusted - full.0[(date, row, col + start)]).abs() < 1e-6);
            }
            for ((row, col), value) in window.velocity.indexed_iter() {
                let adjusted = value + datum.velocity_offsets[index];
                assert!((adjusted - full.1[(row, col + start)]).abs() < 1e-6);
            }
        }
        assert_eq!(datum.overlaps.len(), 2, "windows 0 and 2 do not overlap");
        for overlap in &datum.overlaps {
            assert!(overlap.displacement_median.iter().all(|m| m.abs() < 1e-9));
            assert!(overlap.velocity_mad < 1e-9);
        }
        assert_eq!(datum.overlaps[0].pixel_count, 12 * 4);
    }

    /// Deterministic standard-normal draws (64-bit LCG + Box–Muller).
    struct Normal(u64);
    impl Normal {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        }
        fn draw(&mut self) -> f64 {
            (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
        }
    }

    /// Contract: a chain of five windows with independent pixel noise σ recovers
    /// the planted offsets within four analytic standard errors. With uniform
    /// coherence each overlap of `n` pixels estimates its offset difference with
    /// variance `2σ²/n`, and window `k` sits `k` overlaps from the anchor, so its
    /// offset error has variance `2kσ²/n`.
    #[test]
    fn noisy_chain_recovers_planted_offsets_within_standard_error() {
        let full = frame(20, 84);
        let sigma = 0.002;
        let mut noise = Normal(0x5eed);
        let planted = [0.0, 0.013, -0.027, 0.041, -0.008];
        let windows: Vec<Owned> = (0..5)
            .map(|k| {
                let shift = [planted[k], 2.0 * planted[k], -planted[k]];
                let mut window = cut(&full, k * 16, 20, &shift, 0.1 * planted[k]);
                window
                    .displacement
                    .mapv_inplace(|v| v + sigma * noise.draw());
                window.velocity.mapv_inplace(|v| v + sigma * noise.draw());
                window
            })
            .collect();
        let views: Vec<DatumWindow<'_>> = windows.iter().map(Owned::view).collect();
        let datum = estimate_window_datum(&views, &options(10)).unwrap();
        let n = (20 * 4) as f64;
        for (k, (&plant, offsets)) in planted
            .iter()
            .zip(&datum.displacement_offsets)
            .enumerate()
            .skip(1)
        {
            let bound = 4.0 * (2.0 * k as f64 * sigma * sigma / n).sqrt();
            let truth = [-plant, -2.0 * plant, plant];
            for (date, (offset, expected)) in offsets.iter().zip(truth).enumerate() {
                let error = offset - expected;
                assert!(
                    error.abs() < bound,
                    "window {k} date {date}: {error} vs {bound}"
                );
            }
            let error = datum.velocity_offsets[k] + 0.1 * plant;
            assert!(
                error.abs() < bound,
                "window {k} velocity: {error} vs {bound}"
            );
        }
    }

    /// Contract: windows with no overlap path to the anchor are a typed error.
    #[test]
    fn disconnected_overlap_graph_is_a_typed_error() {
        let full = frame(8, 60);
        let windows = [
            cut(&full, 0, 12, &[0.0; 3], 0.0),
            cut(&full, 8, 12, &[0.0; 3], 0.0),
            cut(&full, 40, 12, &[0.0; 3], 0.0),
        ];
        let views: Vec<DatumWindow<'_>> = windows.iter().map(Owned::view).collect();
        assert_eq!(
            estimate_window_datum(&views, &options(4)),
            Err(DatumError::Disconnected {
                anchor: 0,
                unreached: vec![2]
            })
        );
    }

    /// Pixels outside a window's reference unwrap component never enter the
    /// estimate, and an overlap left with too few usable pixels is a typed error.
    #[test]
    fn off_component_pixels_are_excluded_and_thin_overlaps_rejected() {
        let full = frame(6, 30);
        let mut a = cut(&full, 0, 16, &[0.0; 3], 0.0);
        let mut b = cut(&full, 10, 16, &[0.05; 3], 0.0);
        // Window b's columns 0-2 (frame 10-12) sit in another component and carry
        // a whole-cycle-like jump that must not bias the estimate.
        let mut labels = Array2::from_elem((6, 16), 1_u32);
        labels.slice_mut(s![.., 0..3]).fill(2);
        b.displacement
            .slice_mut(s![.., .., 0..3])
            .mapv_inplace(|v| v + 0.028);
        b.reference = Some((0, 8));
        b.components = Some(labels);
        let views = [a.view(), b.view()];
        let datum = estimate_window_datum(&views, &options(12)).unwrap();
        assert_eq!(datum.overlaps[0].pixel_count, 6 * 3);
        assert!((datum.displacement_offsets[1][0] + 0.05).abs() < 1e-9);
        a.validity.slice_mut(s![.., 13..16]).fill(false);
        let views = [a.view(), b.view()];
        assert_eq!(
            estimate_window_datum(&views, &options(12)),
            Err(DatumError::InsufficientOverlap {
                window_a: 0,
                window_b: 1,
                pixels: 0,
                minimum: 12
            })
        );
    }

    /// An externally supplied anchor carries its values to every window.
    #[test]
    fn external_anchor_values_shift_every_window() {
        let full = frame(6, 30);
        let windows = [
            cut(&full, 0, 16, &[0.0; 3], 0.0),
            cut(&full, 10, 16, &[0.0; 3], 0.0),
        ];
        let views: Vec<DatumWindow<'_>> = windows.iter().map(Owned::view).collect();
        let datum = estimate_window_datum(
            &views,
            &DatumOptions {
                anchor: DatumAnchor::Values {
                    window: 1,
                    displacement: vec![0.1, 0.2, 0.3],
                    velocity: 0.01,
                },
                min_overlap_pixels: 4,
            },
        )
        .unwrap();
        for window in 0..2 {
            for (date, expected) in [0.1, 0.2, 0.3].into_iter().enumerate() {
                assert!((datum.displacement_offsets[window][date] - expected).abs() < 1e-12);
            }
            assert!((datum.velocity_offsets[window] - 0.01).abs() < 1e-12);
        }
    }
}
