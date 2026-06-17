//! Phase filters: long-wavelength FFT Gaussian high-pass (`filtering.py`) and
//! the Goldstein adaptive filter (`goldstein.py`).

use std::f64::consts::PI;

use ndarray::{s, Array2, ArrayView2};
use num_complex::Complex;

use crate::fft::{fft2, fftfreq};

type C = Complex<f64>;

/// Filter out spatial wavelengths longer than `wavelength_cutoff` (meters).
///
/// FFT Gaussian high-pass: subtract the Gaussian-lowpassed field. This handles
/// the no-bad-pixel / no-nodata case (pixels equal to 0 are treated as nodata
/// and excluded, matching dolphin); GDAL-based gap filling for bad pixels is
/// deferred to the I/O phase.
///
/// # Errors
/// Returns `Err` if the derived filter `sigma` exceeds the image dimensions.
pub fn filter_long_wavelength(
    unwrapped: ArrayView2<f64>,
    wavelength_cutoff: f64,
    pixel_spacing: f64,
) -> Result<Array2<f64>, &'static str> {
    let (rows, cols) = unwrapped.dim();
    let sigma = filter_sigma(wavelength_cutoff, pixel_spacing);
    if sigma > rows as f64 || sigma > cols as f64 {
        return Err("wavelength_cutoff too large for image");
    }
    let filled = unwrapped.mapv(|v| if v.is_finite() { v } else { 0.0 });
    let lowpass = gaussian_lowpass(filled.view(), sigma);
    Ok(Array2::from_shape_fn((rows, cols), |ix| {
        let in_bounds = f64::from(filled[ix] != 0.0);
        filled[ix] - lowpass[ix] * in_bounds
    }))
}

/// Filter `sigma` (in pixels) giving a `wavelength_cutoff` half-power point.
fn filter_sigma(wavelength_cutoff: f64, pixel_spacing: f64) -> f64 {
    let sigma_f = 1.0 / wavelength_cutoff / (1.0 / 0.5f64).ln().sqrt();
    let sigma_x = 1.0 / PI / 2.0 / sigma_f;
    sigma_x / pixel_spacing
}

/// Gaussian lowpass via `Re(ifft2(fft2(x) · G))`, `G = exp(-2π²σ²(fu²+fv²))`.
fn gaussian_lowpass(filled: ArrayView2<f64>, sigma: f64) -> Array2<f64> {
    let (rows, cols) = filled.dim();
    let mut spectrum = filled.mapv(|v| C::new(v, 0.0));
    fft2(&mut spectrum, false);

    let (fr, fc) = (fftfreq(rows), fftfreq(cols));
    let k = -2.0 * PI * PI * sigma * sigma;
    spectrum.indexed_iter_mut().for_each(|((r, c), val)| {
        *val *= (k * (fr[r] * fr[r] + fc[c] * fc[c])).exp();
    });

    fft2(&mut spectrum, true);
    let norm = (rows * cols) as f64;
    spectrum.mapv(|z| z.re / norm)
}

/// Goldstein adaptive filter (port of `goldstein.py`).
///
/// Overlapping `psize`-square patches are power-spectrum weighted by `|F|^alpha`
/// in the frequency domain and blended with a tent window. `alpha` in `[0, 1]`.
#[must_use]
pub fn goldstein(phase: ArrayView2<C>, alpha: f64, psize: usize) -> Array2<C> {
    let (rows, cols) = phase.dim();
    let empty = phase.mapv(|z| z.is_nan() || z.arg() == 0.0);
    if empty.iter().all(|&e| e) {
        return phase.to_owned();
    }
    let weight = make_weight(psize);
    let mut out = Array2::<C>::zeros((rows, cols));
    for (i, j) in patch_starts(rows, cols, psize) {
        let patch = filter_patch(phase.slice(s![i..i + psize, j..j + psize]), &weight, alpha);
        accumulate(&mut out, &patch, (i, j));
    }
    zero_empty(&mut out, &empty);
    out
}

/// Top-left corners of overlapping patches stepping by `psize/2` (dolphin's
/// `range(0, dim - psize, psize//2)` — the trailing edge is left unprocessed).
fn patch_starts(rows: usize, cols: usize, psize: usize) -> Vec<(usize, usize)> {
    let step = psize / 2;
    let is: Vec<usize> = (0..rows.saturating_sub(psize)).step_by(step).collect();
    let js: Vec<usize> = (0..cols.saturating_sub(psize)).step_by(step).collect();
    is.iter()
        .flat_map(|&i| js.iter().map(move |&j| (i, j)))
        .collect()
}

/// One patch: `weight · ifft2(|fft2(d)|^alpha · fft2(d))`.
fn filter_patch(data: ArrayView2<C>, weight: &Array2<f64>, alpha: f64) -> Array2<C> {
    let psize = data.nrows();
    let mut f = data.to_owned();
    fft2(&mut f, false);
    f.mapv_inplace(|z| z * z.norm().powf(alpha));
    fft2(&mut f, true);
    let norm = (psize * psize) as f64;
    Array2::from_shape_fn((psize, psize), |ix| weight[ix] * f[ix] / norm)
}

/// Add a filtered patch into the output at corner `(i, j)`.
fn accumulate(out: &mut Array2<C>, patch: &Array2<C>, corner: (usize, usize)) {
    let (psize, _) = patch.dim();
    let mut region = out.slice_mut(s![corner.0..corner.0 + psize, corner.1..corner.1 + psize]);
    region += patch;
}

/// Zero out pixels flagged empty (NaN or exactly-zero phase).
fn zero_empty(out: &mut Array2<C>, empty: &Array2<bool>) {
    out.iter_mut()
        .zip(empty.iter())
        .filter(|(_, &e)| e)
        .for_each(|(z, _)| *z = C::new(0.0, 0.0));
}

/// Separable tent weight matrix `(psize, psize)` peaking at the patch center seams.
fn make_weight(psize: usize) -> Array2<f64> {
    let half = psize / 2;
    let w1d = |k: usize| 1.0 - ((k as f64) - (half as f64 - 1.0)).abs() / (half as f64 - 1.0);
    let mirror = |idx: usize| if idx < half { idx } else { psize - 1 - idx };
    Array2::from_shape_fn((psize, psize), |(r, c)| w1d(mirror(r)) * w1d(mirror(c)))
}
