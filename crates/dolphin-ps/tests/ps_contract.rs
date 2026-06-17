//! Phase-3 (PS) contract tests.
//!
//! Primary (analytic): amplitude dispersion and the threshold/nodata decision
//! on pixels with known stats; PS-fill on a known PS pixel referenced to SLC 0.
//! Secondary (oracle): amp mean/dispersion, PS mask, and a strided PS-fill must
//! match dolphin v0.35.0. Oracle tests skip when fixtures are absent.

use std::path::{Path, PathBuf};

use dolphin_core::{Cf32, Cf64, Strides};
use dolphin_ps::{calc_ps, fill_ps_pixels, PsResult};
use ndarray::{Array2, Array3};

// ------------------------------- analytic (primary) ---------------------------

#[test]
fn dispersion_threshold_and_nodata() {
    // Three pixels (1 row, 3 cols), nslc=4:
    //   col0: constant -> std 0 -> dispersion 0 -> nodata (255)
    //   col1: low dispersion -> PS (1)
    //   col2: high dispersion -> non-PS (0)
    let mag = Array3::from_shape_vec(
        (4, 1, 3),
        vec![
            5.0, 1.00, 1.0, //
            5.0, 1.02, 5.0, //
            5.0, 0.98, 1.0, //
            5.0, 1.00, 9.0,
        ],
    )
    .unwrap();
    let PsResult {
        amp_mean,
        amp_dispersion,
        ps,
    } = calc_ps(mag.view(), 0.25, 4);

    assert_eq!(ps[(0, 0)], 255, "constant pixel is nodata");
    assert_eq!(ps[(0, 1)], 1, "low-dispersion pixel is PS");
    assert_eq!(ps[(0, 2)], 0, "high-dispersion pixel is not PS");
    assert!((amp_mean[(0, 0)] - 5.0).abs() < 1e-6);
    assert_eq!(amp_dispersion[(0, 0)], 0.0);
    assert!(amp_dispersion[(0, 1)] < 0.25 && amp_dispersion[(0, 1)] > 0.0);
}

#[test]
fn ps_fill_takes_referenced_slc_phase() {
    // strides (1,1): each PS cell takes its own SLC phase, referenced to SLC 0.
    let nslc = 3;
    let phases = [0.4_f64, 1.1, -0.7];
    let slc = Array3::from_shape_fn((nslc, 1, 2), |(t, _, c)| {
        // col 0 is PS (amp 2.0), col 1 is not
        Cf64::from_polar(if c == 0 { 2.0 } else { 1.0 }, phases[t])
    });
    let ps_mask = Array2::from_shape_vec((1, 2), vec![true, false]).unwrap();

    let mut cpx = Array3::from_elem((nslc, 1, 2), Cf64::new(1.0, 0.0));
    let mut temp_coh = Array2::zeros((1, 2));
    fill_ps_pixels(
        &mut cpx,
        &mut temp_coh,
        slc.view(),
        ps_mask.view(),
        Strides { y: 1, x: 1 },
        0,
        None,
    );

    assert_eq!(temp_coh[(0, 0)], 1.0, "PS cell forced to temp_coh 1");
    assert_eq!(temp_coh[(0, 1)], 0.0, "non-PS cell untouched");
    for (t, &ph) in phases.iter().enumerate() {
        let got = cpx[(t, 0, 0)].arg();
        let want = ph - phases[0]; // referenced to SLC 0
        assert!((wrap(got - want)).abs() < 1e-9, "t {t}: {got} vs {want}");
        assert!(
            (cpx[(t, 0, 0)].norm() - 1.0).abs() < 1e-9,
            "magnitude preserved"
        );
    }
}

fn wrap(d: f64) -> f64 {
    let w = d.rem_euclid(std::f64::consts::TAU);
    if w > std::f64::consts::PI {
        w - std::f64::consts::TAU
    } else {
        w
    }
}

// ------------------------------- oracle (secondary) ---------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn slc_c64() -> Array3<Cf64> {
    let stack: Array3<Cf32> = ndarray_npy::read_npy(fixtures().join("slc_stack.npy")).unwrap();
    stack.mapv(|z| Cf64::new(z.re as f64, z.im as f64))
}

fn max_abs_err(a: &Array2<f32>, b: &Array2<f32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn calc_ps_matches_oracle() {
    let dir = fixtures();
    if !dir.join("ps_mask.npy").exists() {
        eprintln!("skipping calc_ps_matches_oracle: no fixtures");
        return;
    }
    let mag = slc_c64().mapv(|z| z.norm());
    let nslc = mag.dim().0;
    let r = calc_ps(mag.view(), 0.25, nslc);

    let mean_o: Array2<f32> = ndarray_npy::read_npy(dir.join("amp_mean.npy")).unwrap();
    let disp_o: Array2<f32> = ndarray_npy::read_npy(dir.join("amp_disp.npy")).unwrap();
    let ps_o: Array2<u8> = ndarray_npy::read_npy(dir.join("ps_mask.npy")).unwrap();

    assert!(max_abs_err(&r.amp_mean, &mean_o) < 1e-4, "amp_mean differs");
    assert!(
        max_abs_err(&r.amp_dispersion, &disp_o) < 1e-4,
        "amp_dispersion differs"
    );
    assert_eq!(r.ps, ps_o, "PS mask differs from oracle");
}

#[test]
fn ps_fill_matches_oracle() {
    let dir = fixtures();
    if !dir.join("ps_fill_cpx_phase.npy").exists() {
        eprintln!("skipping ps_fill_matches_oracle: no fixtures");
        return;
    }
    let slc = slc_c64();
    let ps_mask: Array2<bool> = ndarray_npy::read_npy(dir.join("ps_fill_mask.npy")).unwrap();
    let cpx_o: Array3<Cf32> = ndarray_npy::read_npy(dir.join("ps_fill_cpx_phase.npy")).unwrap();
    let coh_o: Array2<f32> = ndarray_npy::read_npy(dir.join("ps_fill_temp_coh.npy")).unwrap();

    // Same inputs dolphin used: unit-magnitude cpx_phase, zero temp_coh.
    let (nslc, out_rows, out_cols) = cpx_o.dim();
    let mut cpx = Array3::from_elem((nslc, out_rows, out_cols), Cf64::new(1.0, 0.0));
    let mut temp_coh = Array2::zeros((out_rows, out_cols));
    fill_ps_pixels(
        &mut cpx,
        &mut temp_coh,
        slc.view(),
        ps_mask.view(),
        Strides { y: 2, x: 2 },
        0,
        None,
    );

    let cpx_err = cpx
        .iter()
        .zip(cpx_o.iter())
        .map(|(a, b)| (a - Cf64::new(b.re as f64, b.im as f64)).norm())
        .fold(0.0_f64, f64::max);
    let coh_err = temp_coh
        .iter()
        .zip(coh_o.iter())
        .map(|(a, b)| (a - *b as f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(cpx_err < 1e-4, "PS-filled phase error {cpx_err}");
    assert!(coh_err < 1e-6, "PS-filled temp_coh error {coh_err}");
}
