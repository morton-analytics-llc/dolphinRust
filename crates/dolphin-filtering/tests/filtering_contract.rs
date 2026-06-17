//! Phase-7 (filtering) contract tests.
//!
//! Primary (analytic): the long-wavelength high-pass removes a pure ramp
//! (DC/low frequency) down to ~0. Secondary (oracle): long-wavelength and
//! Goldstein outputs match dolphin v0.35.0 to atol 1e-4. Oracle tests skip
//! without fixtures.

use std::path::{Path, PathBuf};

use dolphin_core::Cf64;
use dolphin_filtering::{filter_long_wavelength, goldstein};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex;

type Cf32 = Complex<f32>;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn max_abs_err_f64(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ------------------------------- analytic (primary) ---------------------------

#[test]
fn high_pass_removes_long_wavelength_ramp() {
    // A pure planar ramp is long-wavelength: the high-pass should leave ~0 in
    // the interior (edges wrap under the FFT, so check the center).
    let (rows, cols) = (64, 64);
    let ramp = Array2::from_shape_fn((rows, cols), |(r, c)| {
        10.0 + 0.01 * c as f64 + 0.005 * r as f64
    });
    let out = filter_long_wavelength(ramp.view(), 5_000.0, 30.0).unwrap();
    let center = out.slice(ndarray::s![24..40, 24..40]);
    let max_center = center.iter().map(|v| v.abs()).fold(0.0, f64::max);
    // Input center ~10.3; high-pass attenuates DC+ramp by >50x. Small residual
    // (~0.09) is Gibbs/edge wrap from the finite FFT.
    assert!(
        max_center < 0.2,
        "ramp not attenuated in interior: {max_center}"
    );
}

#[test]
fn rejects_cutoff_too_large() {
    let small = Array2::<f64>::zeros((16, 16));
    assert!(filter_long_wavelength(small.view(), 1_000_000.0, 30.0).is_err());
}

// ------------------------------- oracle (secondary) ---------------------------

#[test]
fn long_wavelength_matches_oracle() {
    let dir = fixtures();
    if !dir.join("filt_lw_output.npy").exists() {
        eprintln!("skipping long_wavelength oracle: no fixtures");
        return;
    }
    let input: Array2<f64> = ndarray_npy::read_npy(dir.join("filt_lw_input.npy")).unwrap();
    let oracle: Array2<f64> = ndarray_npy::read_npy(dir.join("filt_lw_output.npy")).unwrap();
    let out = filter_long_wavelength(input.view(), 5_000.0, 30.0).unwrap();
    assert!(
        max_abs_err_f64(out.view(), oracle.view()) < 1e-4,
        "long-wavelength differs"
    );
}

#[test]
fn goldstein_matches_oracle() {
    let dir = fixtures();
    if !dir.join("filt_gold_output.npy").exists() {
        eprintln!("skipping goldstein oracle: no fixtures");
        return;
    }
    let input: Array2<Cf32> = ndarray_npy::read_npy(dir.join("filt_gold_input.npy")).unwrap();
    let oracle: Array2<Cf32> = ndarray_npy::read_npy(dir.join("filt_gold_output.npy")).unwrap();
    let input = input.mapv(|z| Cf64::new(z.re as f64, z.im as f64));

    let out = goldstein(input.view(), 0.5, 16);
    let err = out
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - Cf64::new(b.re as f64, b.im as f64)).norm())
        .fold(0.0_f64, f64::max);
    assert!(err < 1e-4, "goldstein differs: {err}");
}
