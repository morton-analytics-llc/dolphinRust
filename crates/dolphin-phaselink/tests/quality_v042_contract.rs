//! v1.2.0 quality-layer contracts: CRLB uncertainty + sequential closure phase.
//!
//! Primary (analytic): a perfectly consistent coherence matrix closes to 0; a
//! coherence matrix with one injected non-closing triplet does not. Secondary
//! (oracle): CRLB σ and closure phase match dolphin **v0.42.0** (the forward
//! oracle for the two layers v0.35.0 lacks), including the singular-Γ case (the
//! v0.42 fix → NaN past the reference date).

use std::path::{Path, PathBuf};

use dolphin_core::Cf64;
use dolphin_phaselink::{estimate_closure_phases, estimate_crlb};
use ndarray::{Array3, Array4};

// cov_C.npy was formed with HalfWindow(2, 2): num_looks = sqrt(2*2).
const NUM_LOOKS: f64 = 2.0;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

// ------------------------------- analytic (primary) ---------------------------

/// A coherence matrix built from a single per-date phase `C_ij = exp(j(φ_i−φ_j))`
/// is consistent, so every nearest-neighbour triplet closes to 0.
#[test]
fn closure_is_zero_for_consistent_coherence() {
    let phi = [0.0, 0.5, 1.1, 1.4, 2.3];
    let n = phi.len();
    let c = Array4::from_shape_fn((1, 1, n, n), |(_, _, i, j)| {
        Cf64::from_polar(1.0, phi[i] - phi[j])
    });
    let closure = estimate_closure_phases(c.view());
    assert_eq!(closure.dim(), (n - 2, 1, 1));
    let max = closure.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(
        max < 1e-12,
        "consistent coherence should close to 0, got {max}"
    );
}

/// Injecting a phase into one upper entry breaks exactly the triplets that use
/// it, so the closure is nonzero (and equals the injected non-closure).
#[test]
fn closure_detects_injected_non_closure() {
    let phi = [0.0, 0.5, 1.1, 1.4, 2.3];
    let n = phi.len();
    let bias = 0.3;
    let c = Array4::from_shape_fn((1, 1, n, n), |(_, _, i, j)| {
        let base = phi[i] - phi[j];
        // Bias the (0,2) long-baseline entry: closure[0] uses conj(C[0,2]).
        let extra = if (i, j) == (0, 2) { bias } else { 0.0 };
        Cf64::from_polar(1.0, base + extra)
    });
    let closure = estimate_closure_phases(c.view());
    assert!(
        (closure[(0, 0, 0)] + bias).abs() < 1e-12,
        "closure[0] should be -bias = {}, got {}",
        -bias,
        closure[(0, 0, 0)]
    );
    assert!(closure[(1, 0, 0)].abs() < 1e-12, "closure[1] unaffected");
}

/// CRLB σ is 0 on the reference date and strictly positive elsewhere for a
/// well-conditioned coherence matrix.
#[test]
fn crlb_zero_at_reference_positive_elsewhere() {
    let n = 5;
    let rho = 0.8_f64;
    // AR(1) coherence |C_ij| = rho^|i-j| — Tebaldini's well-conditioned example.
    let c = Array4::from_shape_fn((1, 1, n, n), |(_, _, i, j)| {
        Cf64::new(rho.powi((i as i32 - j as i32).abs()), 0.0)
    });
    let sigma = estimate_crlb(c.view(), 0.0, 0.0, 0, NUM_LOOKS);
    assert_eq!(sigma.dim(), (n, 1, 1));
    assert_eq!(sigma[(0, 0, 0)], 0.0, "reference date σ must be 0");
    for t in 1..n {
        let s = sigma[(t, 0, 0)];
        assert!(s.is_finite() && s > 0.0, "σ[{t}] should be > 0, got {s}");
    }
}

// ------------------------------- oracle (secondary) ---------------------------

fn read_f32_3d(name: &str) -> Option<Array3<f32>> {
    let path = fixtures().join(name);
    path.exists().then(|| ndarray_npy::read_npy(path).unwrap())
}

fn read_cov() -> Array4<Cf64> {
    let c: Array4<dolphin_core::Cf32> =
        ndarray_npy::read_npy(fixtures().join("cov_C.npy")).unwrap();
    c.mapv(|z| Cf64::new(z.re as f64, z.im as f64))
}

/// CRLB σ matches dolphin v0.42.0 within tolerance. Oracle layout is
/// `(rows, cols, nslc)`; the kernel is band-major `(nslc, rows, cols)`.
#[test]
fn crlb_matches_oracle_v042() {
    let Some(oracle) = read_f32_3d("crlb_sigma_v042.npy") else {
        eprintln!("skipping crlb_matches_oracle_v042: no v0.42 fixtures");
        return;
    };
    let c = read_cov();
    let sigma = estimate_crlb(c.view(), 0.0, 0.0, 0, NUM_LOOKS);
    let (rows, cols, nslc) = oracle.dim();
    let max_err = (0..rows)
        .flat_map(|r| (0..cols).flat_map(move |c| (0..nslc).map(move |t| (r, c, t))))
        .map(|(r, c, t)| (sigma[(t, r, c)] - oracle[(r, c, t)] as f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-4,
        "CRLB σ differs from oracle: max |Δ| = {max_err}"
    );
}

/// Closure phase matches dolphin v0.42.0 within tolerance (angle, wrap-safe via
/// the small magnitudes the fixture produces). Oracle `(rows, cols, nslc-2)`.
#[test]
fn closure_matches_oracle_v042() {
    let Some(oracle) = read_f32_3d("closure_phase_v042.npy") else {
        eprintln!("skipping closure_matches_oracle_v042: no v0.42 fixtures");
        return;
    };
    let c = read_cov();
    let closure = estimate_closure_phases(c.view());
    let (rows, cols, ntri) = oracle.dim();
    let max_err = (0..rows)
        .flat_map(|r| (0..cols).flat_map(move |c| (0..ntri).map(move |k| (r, c, k))))
        .map(|(r, c, k)| wrapped_diff(closure[(k, r, c)], oracle[(r, c, k)] as f64))
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-4,
        "closure differs from oracle: max |Δ| = {max_err}"
    );
}

/// The v0.42 singular-Γ fix: zero / identity blocks give NaN past the reference
/// date, a rank-1 (all-ones) block gives a finite small σ — exactly dolphin's.
#[test]
fn crlb_singular_matches_oracle_v042() {
    let Some(oracle) = read_f32_3d("crlb_singular_sigma_v042.npy") else {
        eprintln!("skipping crlb_singular_matches_oracle_v042: no v0.42 fixtures");
        return;
    };
    let c: Array4<dolphin_core::Cf32> =
        ndarray_npy::read_npy(fixtures().join("crlb_singular_C_v042.npy")).unwrap();
    let c = c.mapv(|z| Cf64::new(z.re as f64, z.im as f64));
    let sigma = estimate_crlb(c.view(), 0.0, 0.0, 0, NUM_LOOKS);
    let (rows, cols, nslc) = oracle.dim();
    for r in 0..rows {
        for col in 0..cols {
            for t in 0..nslc {
                let got = sigma[(t, r, col)];
                let want = oracle[(r, col, t)] as f64;
                if want.is_nan() {
                    assert!(
                        got.is_nan(),
                        "pixel ({r},{col}) date {t}: expected NaN, got {got}"
                    );
                } else {
                    assert!(
                        (got - want).abs() < 1e-4,
                        "pixel ({r},{col}) date {t}: expected {want}, got {got}"
                    );
                }
            }
        }
    }
}

/// Smallest signed angular distance between two phases.
fn wrapped_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(2.0 * std::f64::consts::PI);
    d.min(2.0 * std::f64::consts::PI - d).abs()
}
