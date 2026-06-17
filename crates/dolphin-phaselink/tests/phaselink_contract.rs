//! Phase-1 contract tests.
//!
//! Primary (analytic, no Python): coherence matrices with a known dominant /
//! least eigenvector — EVD and EMI must recover the construction phase ramp, and
//! a singular `Γ` must fall back to EVD. Secondary (oracle): match dolphin
//! v0.35.0 covariance and EVD/EMI phase to physical tolerances. Oracle tests
//! skip when fixtures are absent (run `oracle/gen_phaselink.py` to produce them).

use std::f64::consts::TAU;
use std::path::{Path, PathBuf};

use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_phaselink::{
    estimate_stack_covariance, process_coherence_matrices, process_coherence_matrix,
};
use ndarray::{Array1, Array2, Array3, Array4};

const N: usize = 8;

fn ramp() -> Vec<f64> {
    (0..N).map(|i| 0.5 * i as f64).collect()
}

/// Wrap a phase difference into (-π, π].
fn wrap(d: f64) -> f64 {
    let w = d.rem_euclid(TAU);
    if w > TAU / 2.0 {
        w - TAU
    } else {
        w
    }
}

/// Assert a referenced phase vector matches the ramp (referenced to index 0).
fn assert_recovers_ramp(phase: &Array1<Cf64>, theta: &[f64], tol: f64) {
    for (i, &th) in theta.iter().enumerate() {
        let got = phase[i].arg();
        let want = th - theta[0];
        assert!(
            wrap(got - want).abs() < tol,
            "idx {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn evd_recovers_rank_one_phase() {
    let theta = ramp();
    let c = Array2::from_shape_fn((N, N), |(i, j)| Cf64::from_polar(1.0, theta[i] - theta[j]));
    let est = process_coherence_matrix(c.view(), true, 0.0, 0.0, 0);
    assert_eq!(est.estimator, 0);
    assert_recovers_ramp(&est.phase, &theta, 1e-6);
}

#[test]
fn emi_recovers_kms_phase() {
    let theta = ramp();
    let gamma = 0.7_f64;
    let c = Array2::from_shape_fn((N, N), |(i, j)| {
        let mag = gamma.powi((i as i64 - j as i64).unsigned_abs() as i32);
        Cf64::from_polar(mag, theta[i] - theta[j])
    });
    let est = process_coherence_matrix(c.view(), false, 0.0, 0.0, 0);
    assert_eq!(est.estimator, 1, "well-conditioned Gamma should use EMI");
    assert_recovers_ramp(&est.phase, &theta, 1e-6);
}

#[test]
fn emi_falls_back_to_evd_on_singular_gamma() {
    // Rank-1 unit coherence => Gamma = all-ones (singular) => Cholesky fails.
    let theta = ramp();
    let c = Array2::from_shape_fn((N, N), |(i, j)| Cf64::from_polar(1.0, theta[i] - theta[j]));
    let est = process_coherence_matrix(c.view(), false, 0.0, 0.0, 0);
    assert_eq!(est.estimator, 0, "singular Gamma must fall back to EVD");
    assert_recovers_ramp(&est.phase, &theta, 1e-6);
}

// ----------------------------- oracle (secondary) -----------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn to_c64<D: ndarray::Dimension>(a: ndarray::Array<Cf32, D>) -> ndarray::Array<Cf64, D> {
    a.mapv(|z| Cf64::new(z.re as f64, z.im as f64))
}

/// Magnitude of the normalized inner product `|⟨a, b⟩|` (global-phase invariant).
fn cos_sim(a: &[Cf64], b: &[Cf64]) -> f64 {
    let inner: Cf64 = a.iter().zip(b).map(|(x, y)| x * y.conj()).sum();
    let na = a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let nb = b.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    inner.norm() / (na * nb)
}

#[test]
fn covariance_matches_oracle() {
    let dir = fixtures();
    if !dir.join("cov_C.npy").exists() {
        eprintln!("skipping covariance_matches_oracle: no fixtures at {dir:?}");
        return;
    }
    let stack: Array3<Cf32> = ndarray_npy::read_npy(dir.join("slc_stack.npy")).unwrap();
    let oracle: Array4<Cf32> = ndarray_npy::read_npy(dir.join("cov_C.npy")).unwrap();

    let c = estimate_stack_covariance(
        to_c64(stack).view(),
        HalfWindow { y: 2, x: 2 },
        Strides { y: 1, x: 1 },
        None,
    )
    .unwrap();

    let oracle = to_c64(oracle);
    let max_err = c
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-4, "max covariance error {max_err}");
}

fn assert_estimator_matches_oracle(use_evd: bool, phase_file: &str, expected_estimator: u8) {
    let dir = fixtures();
    if !dir.join(phase_file).exists() {
        eprintln!("skipping oracle estimator ({phase_file}): no fixtures");
        return;
    }
    let c: Array4<Cf32> = ndarray_npy::read_npy(dir.join("cov_C.npy")).unwrap();
    let oracle: Array3<Cf32> = ndarray_npy::read_npy(dir.join(phase_file)).unwrap();
    let oracle = to_c64(oracle); // shape (out_rows, out_cols, nslc)

    let out = process_coherence_matrices(to_c64(c).view(), use_evd, 0.0, 0.0, 0);
    let (nslc, rows, cols) = out.cpx_phase.dim();

    let mut min_sim = 1.0_f64;
    for r in 0..rows {
        for col in 0..cols {
            let rust: Vec<Cf64> = (0..nslc).map(|t| out.cpx_phase[(t, r, col)]).collect();
            let orc: Vec<Cf64> = (0..nslc).map(|t| oracle[(r, col, t)]).collect();
            min_sim = min_sim.min(cos_sim(&rust, &orc));
            assert_eq!(out.estimator[(r, col)], expected_estimator);
        }
    }
    assert!(min_sim > 1.0 - 1e-3, "min eigenvector cos-sim {min_sim}");
}

#[test]
fn evd_matches_oracle() {
    assert_estimator_matches_oracle(true, "phase_evd.npy", 0);
}

#[test]
fn emi_matches_oracle() {
    assert_estimator_matches_oracle(false, "phase_emi.npy", 1);
}

#[test]
fn oracle_estimator_flag_is_emi() {
    let dir = fixtures();
    if !dir.join("estimator_emi.npy").exists() {
        return;
    }
    let est: Array2<u8> = ndarray_npy::read_npy(dir.join("estimator_emi.npy")).unwrap();
    assert!(
        est.iter().all(|&e| e == 1),
        "oracle EMI should use EMI everywhere"
    );
}
