//! Phase-4 (quality) contract tests: temporal coherence + compressed SLC.
//!
//! Primary (analytic): a phase that perfectly fits the coherence matrix gives
//! temp_coh = 1, an imperfect one gives < 1; a compressed SLC of a coherent
//! pixel returns the mean amplitude with zero residual phase. Secondary
//! (oracle): temp_coh (clean + noisy) and the compressed SLC match dolphin
//! v0.35.0. (CRLB / closure phase are absent in v0.35.0 — deferred.)

use std::path::{Path, PathBuf};

use dolphin_core::{Cf32, Cf64};
use dolphin_phaselink::{compress, estimate_temp_coh};
use ndarray::{Array2, Array3, Array4};

// ------------------------------- analytic (primary) ---------------------------

fn consistent_c(theta: &[f64]) -> Array4<Cf64> {
    let n = theta.len();
    Array4::from_shape_fn((1, 1, n, n), |(_, _, i, j)| {
        Cf64::from_polar(1.0, theta[i] - theta[j])
    })
}

#[test]
fn temp_coh_is_one_for_perfect_fit() {
    let theta = [0.0, 0.5, 1.0, 1.4, 2.1, 2.9];
    let c = consistent_c(&theta);
    let phase = Array3::from_shape_fn((1, 1, theta.len()), |(_, _, i)| {
        Cf64::from_polar(1.0, theta[i])
    });
    let coh = estimate_temp_coh(phase.view(), c.view());
    assert!(
        (coh[(0, 0)] - 1.0).abs() < 1e-12,
        "perfect fit -> 1, got {}",
        coh[(0, 0)]
    );
}

#[test]
fn temp_coh_below_one_for_imperfect_fit() {
    let theta = [0.0, 0.5, 1.0, 1.4, 2.1, 2.9];
    let c = consistent_c(&theta);
    let perturb = [0.0, 0.6, -0.4, 0.3, -0.5, 0.2];
    let phase = Array3::from_shape_fn((1, 1, theta.len()), |(_, _, i)| {
        Cf64::from_polar(1.0, theta[i] + perturb[i])
    });
    let coh = estimate_temp_coh(phase.view(), c.view())[(0, 0)];
    assert!(coh > 0.0 && coh < 1.0, "imperfect fit in (0,1), got {coh}");
}

#[test]
fn compress_returns_mean_amplitude_and_residual_phase() {
    // SLC phase = linked phase + constant offset 0.5: projected residual is 0.5,
    // magnitude is the mean amplitude. (A zero residual hits dolphin's phase==0
    // nodata sentinel, so use a nonzero offset.)
    let amp = [1.0, 2.0, 3.0, 4.0];
    let theta = [0.0, 0.3, 0.6, 0.9];
    let offset = 0.5;
    let slc = Array3::from_shape_fn((4, 1, 1), |(t, _, _)| {
        Cf64::from_polar(amp[t], theta[t] + offset)
    });
    let pl = Array3::from_shape_fn((4, 1, 1), |(t, _, _)| Cf64::from_polar(1.0, theta[t]));
    let out = compress(slc.view(), pl.view(), 0, None);
    assert!(
        (out[(0, 0)] - Cf64::from_polar(2.5, offset)).norm() < 1e-12,
        "got {}",
        out[(0, 0)]
    );
}

// ------------------------------- oracle (secondary) ---------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn to_c64<D: ndarray::Dimension>(a: ndarray::Array<Cf32, D>) -> ndarray::Array<Cf64, D> {
    a.mapv(|z| Cf64::new(z.re as f64, z.im as f64))
}

fn max_abs_err(a: &Array2<f64>, b: &Array2<f32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn check_temp_coh_oracle(phase_file: &str, coh_file: &str) {
    let dir = fixtures();
    if !dir.join(coh_file).exists() {
        eprintln!("skipping temp_coh oracle ({coh_file}): no fixtures");
        return;
    }
    let phase: Array3<Cf32> = ndarray_npy::read_npy(dir.join(phase_file)).unwrap();
    let c: Array4<Cf32> = ndarray_npy::read_npy(dir.join("cov_C.npy")).unwrap();
    let coh = estimate_temp_coh(to_c64(phase).view(), to_c64(c).view());
    let oracle: Array2<f32> = ndarray_npy::read_npy(dir.join(coh_file)).unwrap();
    assert!(
        max_abs_err(&coh, &oracle) < 1e-4,
        "temp_coh {coh_file} differs"
    );
}

#[test]
fn temp_coh_matches_oracle() {
    check_temp_coh_oracle("phase_emi.npy", "temp_coh_emi.npy");
}

#[test]
fn temp_coh_noisy_matches_oracle() {
    check_temp_coh_oracle("phase_noisy.npy", "temp_coh_noisy.npy");
}

#[test]
fn compress_matches_oracle() {
    let dir = fixtures();
    if !dir.join("compressed_slc.npy").exists() {
        eprintln!("skipping compress_matches_oracle: no fixtures");
        return;
    }
    let stack =
        to_c64(ndarray_npy::read_npy::<_, Array3<Cf32>>(dir.join("slc_stack.npy")).unwrap());
    let phase =
        to_c64(ndarray_npy::read_npy::<_, Array3<Cf32>>(dir.join("phase_emi.npy")).unwrap());
    // phase is (rows, cols, nslc); compress wants (nslc, rows, cols).
    let (rows, cols, nslc) = phase.dim();
    let pl = Array3::from_shape_fn((nslc, rows, cols), |(t, r, c)| phase[(r, c, t)]);

    let out = compress(stack.view(), pl.view(), 0, None);
    let oracle =
        to_c64(ndarray_npy::read_npy::<_, Array2<Cf32>>(dir.join("compressed_slc.npy")).unwrap());
    let err = out
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(err < 1e-4, "compressed SLC error {err}");
}
