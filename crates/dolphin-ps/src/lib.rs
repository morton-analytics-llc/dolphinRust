//! Persistent-scatterer selection — port of `dolphin/ps.py` and the PS-fill
//! step of `phase_link/_ps_filling.py`.
//!
//! Amplitude dispersion `D_A = std(|z|)/mean(|z|)` over the temporal stack gives
//! a uint8 PS mask (1 = PS, 0 = non-PS, 255 = nodata). At phase-linking output,
//! PS pixels bypass covariance: each looked output cell takes the phase of the
//! brightest PS in its window (referenced to `reference_idx`) and `temp_coh = 1`.
#![warn(missing_docs)]

use dolphin_core::{Cf64, Strides};
use ndarray::{s, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use rayon::prelude::*;

/// Amplitude-dispersion outputs for a stack.
pub struct PsResult {
    /// Mean amplitude per pixel (nodata = 0).
    pub amp_mean: Array2<f32>,
    /// Amplitude dispersion `std/mean` per pixel (nodata = 0).
    pub amp_dispersion: Array2<f32>,
    /// PS mask: 1 = PS, 0 = non-PS, 255 = nodata.
    pub ps: Array2<u8>,
}

/// Compute amplitude mean, dispersion, and the PS mask (port of `calc_ps_block`
/// with the `create_ps` nodata rule). `stack_mag` is `(nslc, rows, cols)` of
/// real amplitudes; pixels with fewer than `min_count` finite samples are nodata.
#[must_use]
pub fn calc_ps(stack_mag: ArrayView3<f64>, threshold: f64, min_count: usize) -> PsResult {
    let (_, rows, cols) = stack_mag.dim();
    let stats: Vec<(f32, f32, u8)> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let lane = stack_mag.slice(s![.., idx / cols, idx % cols]);
            pixel_stats(lane, threshold, min_count)
        })
        .collect();

    let mean = stats.iter().map(|s| s.0).collect();
    let disp = stats.iter().map(|s| s.1).collect();
    let ps = stats.iter().map(|s| s.2).collect();
    PsResult {
        amp_mean: Array2::from_shape_vec((rows, cols), mean).expect("mean shape"),
        amp_dispersion: Array2::from_shape_vec((rows, cols), disp).expect("disp shape"),
        ps: Array2::from_shape_vec((rows, cols), ps).expect("ps shape"),
    }
}

/// Per-pixel `(mean, dispersion, ps_flag)` over the temporal lane.
fn pixel_stats(lane: ArrayView1<f64>, threshold: f64, min_count: usize) -> (f32, f32, u8) {
    let finite: Vec<f64> = lane.iter().copied().filter(|v| v.is_finite()).collect();
    let (mean, disp) = mean_dispersion(&finite);
    let disp = if finite.len() < min_count {
        f64::NAN
    } else {
        disp
    };
    let mean = nan_to_num(mean);
    let disp = nan_to_num(disp);
    (mean as f32, disp as f32, ps_flag(disp, threshold))
}

/// Population mean and dispersion (`std/mean`, ddof=0) of finite samples.
fn mean_dispersion(finite: &[f64]) -> (f64, f64) {
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt() / mean)
}

/// Replace NaN/±inf with 0 (dolphin's `nan_to_num`).
fn nan_to_num(v: f64) -> f64 {
    match v.is_finite() {
        true => v,
        false => 0.0,
    }
}

/// PS flag from dispersion: 255 nodata (0 dispersion), 1 PS (below threshold), else 0.
fn ps_flag(disp: f64, threshold: f64) -> u8 {
    if disp == 0.0 {
        255
    } else if disp < threshold {
        1
    } else {
        0
    }
}

/// Fill PS pixels into a phase-linking estimate in place (port of `fill_ps_pixels`,
/// `use_max_ps=True`). `cpx_phase` is `(nslc, out_rows, out_cols)` and `temp_coh`
/// `(out_rows, out_cols)` on the decimated grid; `slc_stack`/`ps_mask` are full
/// resolution. Each output cell whose `strides` window contains a PS takes the
/// brightest PS's referenced phase and `temp_coh = 1`.
pub fn fill_ps_pixels(
    cpx_phase: &mut Array3<Cf64>,
    temp_coh: &mut Array2<f64>,
    slc_stack: ArrayView3<Cf64>,
    ps_mask: ArrayView2<bool>,
    strides: Strides,
    reference_idx: usize,
    avg_mag: Option<ArrayView2<f64>>,
) {
    let avg = match avg_mag {
        Some(v) => v.to_owned(),
        None => average_magnitude(slc_stack),
    };
    let mag = masked_magnitude(avg.view(), ps_mask);
    let (out_rows, out_cols) = (cpx_phase.dim().1, cpx_phase.dim().2);

    (0..out_rows * out_cols)
        .map(|idx| (idx / out_cols, idx % out_cols))
        .filter_map(|cell| brightest_in_window(mag.view(), cell, strides).map(|src| (cell, src)))
        .for_each(|(cell, src)| {
            fill_cell(cpx_phase, temp_coh, slc_stack, reference_idx, cell, src)
        });
}

/// Average magnitude across the stack, ignoring NaNs (`nanmean(|slc|, axis=0)`).
fn average_magnitude(slc_stack: ArrayView3<Cf64>) -> Array2<f64> {
    let (_, rows, cols) = slc_stack.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let lane = slc_stack.slice(s![.., r, c]);
        let finite: Vec<f64> = lane
            .iter()
            .map(|z| z.norm())
            .filter(|v| v.is_finite())
            .collect();
        mean_dispersion(&finite).0
    })
}

/// Average magnitude with non-PS pixels set to NaN (so they are ignored as maxima).
fn masked_magnitude(avg_mag: ArrayView2<f64>, ps_mask: ArrayView2<bool>) -> Array2<f64> {
    Array2::from_shape_fn(avg_mag.dim(), |ix| match ps_mask[ix] {
        true => avg_mag[ix],
        false => f64::NAN,
    })
}

/// Full-resolution index of the brightest PS in an output cell's window, if any.
fn brightest_in_window(
    mag: ArrayView2<f64>,
    cell: (usize, usize),
    strides: Strides,
) -> Option<(usize, usize)> {
    let block = mag.slice(s![
        cell.0 * strides.y..(cell.0 + 1) * strides.y,
        cell.1 * strides.x..(cell.1 + 1) * strides.x
    ]);
    block
        .indexed_iter()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|((r, c), _)| (cell.0 * strides.y + r, cell.1 * strides.x + c))
}

/// Write the brightest PS's referenced SLC phase into one output cell.
fn fill_cell(
    cpx_phase: &mut Array3<Cf64>,
    temp_coh: &mut Array2<f64>,
    slc_stack: ArrayView3<Cf64>,
    reference_idx: usize,
    cell: (usize, usize),
    src: (usize, usize),
) {
    let nslc = cpx_phase.dim().0;
    let ref_shift = Cf64::from_polar(1.0, -slc_stack[(reference_idx, src.0, src.1)].arg());
    (0..nslc).for_each(|t| {
        let amp = cpx_phase[(t, cell.0, cell.1)].norm();
        let phase = slc_stack[(t, src.0, src.1)].arg();
        cpx_phase[(t, cell.0, cell.1)] = Cf64::from_polar(amp, phase) * ref_shift;
    });
    temp_coh[cell] = 1.0;
}

/// Error from combining or summarizing [`AmplitudeStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PsError {
    /// Two statistics (or a statistic and a stack) cover different grids.
    ShapeMismatch,
    /// A pixel was observed more often than `u32` can count.
    CountOverflow,
}

impl std::fmt::Display for PsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch => f.write_str("amplitude statistics cover different grids"),
            Self::CountOverflow => f.write_str("amplitude sample count overflows u32"),
        }
    }
}

impl std::error::Error for PsError {}

/// Exact running amplitude statistics per pixel: finite-sample count, mean and
/// sum of squared deviations (`m2`), over `num_dates` acquisitions.
///
/// Built per batch of dates with [`Self::from_stack`] and combined with
/// [`Self::merge`] (Chan et al. pairwise update), so statistics over a long
/// series can be extended one ministack at a time without rereading old dates.
/// The result matches a one-shot [`calc_ps`] over all dates to rounding.
#[derive(Debug, Clone, PartialEq)]
pub struct AmplitudeStats {
    /// Acquisitions covered, finite or not.
    pub num_dates: usize,
    /// Finite amplitude samples per pixel.
    pub count: Array2<u32>,
    /// Mean amplitude of the finite samples (0 where `count` is 0).
    pub mean: Array2<f64>,
    /// Sum of squared deviations from `mean` (0 where `count` is 0).
    pub m2: Array2<f64>,
}

impl AmplitudeStats {
    /// Statistics of `|z|` over every date of `stack` `(nslc, rows, cols)`,
    /// ignoring non-finite samples.
    ///
    /// # Errors
    /// [`PsError::CountOverflow`] when `nslc` exceeds `u32`.
    pub fn from_stack(stack: ArrayView3<Cf64>) -> Result<Self, PsError> {
        let (nslc, rows, cols) = stack.dim();
        u32::try_from(nslc).map_err(|_| PsError::CountOverflow)?;
        let lanes: Vec<(u32, f64, f64)> = (0..rows * cols)
            .into_par_iter()
            .map(|idx| lane_stats(stack.slice(s![.., idx / cols, idx % cols])))
            .collect();
        let field = |pick: fn(&(u32, f64, f64)) -> f64| {
            Array2::from_shape_fn((rows, cols), |(r, c)| pick(&lanes[r * cols + c]))
        };
        Ok(Self {
            num_dates: nslc,
            count: Array2::from_shape_fn((rows, cols), |(r, c)| lanes[r * cols + c].0),
            mean: field(|lane| lane.1),
            m2: field(|lane| lane.2),
        })
    }

    /// Statistics over the dates of `self` followed by those of `later`.
    ///
    /// # Errors
    /// [`PsError::ShapeMismatch`] when the grids differ;
    /// [`PsError::CountOverflow`] when a pixel count exceeds `u32`.
    pub fn merge(&self, later: &Self) -> Result<Self, PsError> {
        if self.count.dim() != later.count.dim() {
            return Err(PsError::ShapeMismatch);
        }
        let mut merged = self.clone();
        merged.num_dates += later.num_dates;
        let mut overflow = false;
        ndarray::Zip::from(&mut merged.count)
            .and(&mut merged.mean)
            .and(&mut merged.m2)
            .and(&later.count)
            .and(&later.mean)
            .and(&later.m2)
            .for_each(|count, mean, m2, &count_b, &mean_b, &m2_b| {
                let Some(total) = count.checked_add(count_b) else {
                    overflow = true;
                    return;
                };
                (*mean, *m2) = chan_merge((*count, *mean, *m2), (count_b, mean_b, m2_b));
                *count = total;
            });
        match overflow {
            true => Err(PsError::CountOverflow),
            false => Ok(merged),
        }
    }

    /// Amplitude dispersion `std/mean` (population std), NaN where undefined.
    #[must_use]
    pub fn dispersion(&self) -> Array2<f64> {
        ndarray::Zip::from(&self.count)
            .and(&self.mean)
            .and(&self.m2)
            .map_collect(|&count, &mean, &m2| match count {
                0 => f64::NAN,
                n => (m2 / f64::from(n)).sqrt() / mean,
            })
    }

    /// PS selection from these statistics with [`calc_ps`]'s rules: a pixel with
    /// any non-finite date is nodata, as is one with zero dispersion.
    #[must_use]
    pub fn to_ps(&self, threshold: f64) -> PsResult {
        let dispersion = self.dispersion();
        let complete = self.count.mapv(|count| count as usize == self.num_dates);
        let disp =
            ndarray::Zip::from(&dispersion)
                .and(&complete)
                .map_collect(|&disp, &complete| match complete {
                    true => nan_to_num(disp),
                    false => 0.0,
                });
        PsResult {
            amp_mean: self.mean.mapv(|mean| nan_to_num(mean) as f32),
            ps: disp.mapv(|disp| ps_flag(disp, threshold)),
            amp_dispersion: disp.mapv(|disp| disp as f32),
        }
    }
}

/// `(count, mean, m2)` of the finite amplitudes in one temporal lane.
fn lane_stats(lane: ArrayView1<Cf64>) -> (u32, f64, f64) {
    let finite: Vec<f64> = lane
        .iter()
        .map(|z| z.norm())
        .filter(|v| v.is_finite())
        .collect();
    if finite.is_empty() {
        return (0, 0.0, 0.0);
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let m2 = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    // `from_stack` checked the date count fits u32, and finite.len() ≤ it.
    (finite.len() as u32, mean, m2)
}

/// Chan et al. pairwise combination of `(count, mean, m2)`; an empty side
/// passes the other through unchanged so seeding with nothing is exact.
fn chan_merge(a: (u32, f64, f64), b: (u32, f64, f64)) -> (f64, f64) {
    match (a.0, b.0) {
        (_, 0) => (a.1, a.2),
        (0, _) => (b.1, b.2),
        (na, nb) => {
            let (na, nb) = (f64::from(na), f64::from(nb));
            let total = na + nb;
            let delta = b.1 - a.1;
            let mean = a.1 + delta * nb / total;
            let m2 = a.2 + b.2 + delta * delta * na * nb / total;
            (mean, m2)
        }
    }
}
