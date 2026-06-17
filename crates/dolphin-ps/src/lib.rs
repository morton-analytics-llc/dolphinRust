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
