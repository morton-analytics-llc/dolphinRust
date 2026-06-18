//! Phase-bias / non-closure correction (Michaelides et al., RSE 2022).
//!
//! The fading-signal bias adds a systematic phase `β_n` to each connection-1
//! (consecutive-date) interferogram, while the longer connection-2 interferogram
//! carries little. The nearest-neighbour closure phase of the coherence matrix
//! (see [`crate::estimate_closure_phases`]) is then
//! `Ξ_k = β_k + β_{k+1}`, i.e. `Ξ_k = B_{k+2} − B_k` for the cumulative bias
//! `B_n = Σ_{m<n} β_m` carried by the phase-linked time series.
//!
//! **Not in Python dolphin** — this leads the oracle, so there is no parity
//! target; it is validated by an analytic fixture (exact recovery of a constant
//! injected bias) and a measured reduction in non-closure on a long series.
//!
//! First-order model: a per-pixel **constant bias velocity** `β̄ = mean_k(Ξ_k)/2`,
//! cumulative `B_n = n·β̄`. This is exact when the bias is constant, removes the
//! systematic part of a noisy series without over-fitting (one parameter/pixel),
//! and is the linear-accumulation case of Michaelides' model. Time-varying bias
//! is a documented future refinement. CPU only.

use dolphin_core::Cf64;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};

/// Per-pixel constant bias velocity `β̄ = mean_k(Ξ_k)/2` from the nearest-neighbour
/// closure stack `(ntri, rows, cols)`. NaN closure bands are skipped per pixel;
/// an all-NaN (or empty) pixel yields `0` (no correction). Returns `(rows, cols)`.
#[must_use]
pub fn estimate_bias_velocity(closure: ArrayView3<f64>) -> Array2<f64> {
    let (_, rows, cols) = closure.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let col = closure.slice(ndarray::s![.., r, c]);
        half_mean(col)
    })
}

/// Mean of the finite entries, halved (`β̄ = mean(Ξ)/2`); `0` when none are finite.
fn half_mean(col: ndarray::ArrayView1<f64>) -> f64 {
    let (sum, n) = col
        .iter()
        .filter(|v| v.is_finite())
        .fold((0.0, 0_usize), |(s, k), v| (s + v, k + 1));
    match n {
        0 => 0.0,
        _ => 0.5 * sum / n as f64,
    }
}

/// Subtract the cumulative bias `B_n = n·β̄` from the phase-linked series in place:
/// band `n` of `linked` `(n_dates, rows, cols)` is multiplied by `exp(−j·n·β̄)`.
/// Band 0 (the reference) is unchanged. A constant injected bias is removed exactly.
pub fn correct_phase_bias(linked: &mut Array3<Cf64>, bias_velocity: ArrayView2<f64>) {
    let n_dates = linked.dim().0;
    for n in 1..n_dates {
        let mut band = linked.index_axis_mut(Axis(0), n);
        band.zip_mut_with(&bias_velocity, |z, &beta| {
            *z *= Cf64::from_polar(1.0, -(n as f64) * beta);
        });
    }
}

/// Closure remaining after removing the modeled bias: `Ξ'_k = Ξ_k − 2·β̄`
/// (the systematic `β_k + β_{k+1} ≈ 2β̄` contribution). The non-closure that the
/// constant-rate model does not explain — used to measure the reduction.
#[must_use]
pub fn residual_closure(closure: ArrayView3<f64>, bias_velocity: ArrayView2<f64>) -> Array3<f64> {
    let (ntri, rows, cols) = closure.dim();
    Array3::from_shape_fn((ntri, rows, cols), |(k, r, c)| {
        closure[(k, r, c)] - 2.0 * bias_velocity[(r, c)]
    })
}

/// Mean absolute closure over finite entries — the scalar non-closure metric
/// (lower is more self-consistent). `0` when the stack is empty/all-NaN.
#[must_use]
pub fn mean_abs_closure(closure: ArrayView3<f64>) -> f64 {
    let (sum, n) = closure
        .iter()
        .filter(|v| v.is_finite())
        .fold((0.0, 0_usize), |(s, k), v| (s + v.abs(), k + 1));
    match n {
        0 => 0.0,
        _ => sum / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Analytic contract: a constant per-connection bias `c` produces closures
    /// `Ξ = 2c`; the estimated velocity is exactly `c`, and correcting a series
    /// `θ_n = φ_n + n·c` recovers `φ_n` to machine precision.
    #[test]
    fn constant_bias_is_recovered_exactly() {
        let c = 0.3_f64;
        let (ntri, rows, cols) = (6, 2, 3);
        let closure = Array3::from_elem((ntri, rows, cols), 2.0 * c);
        let beta = estimate_bias_velocity(closure.view());
        for &b in &beta {
            assert!((b - c).abs() < 1e-12, "bias velocity {b} != {c}");
        }

        // True phases φ_n = 0.07·n; biased linked series θ_n = exp(j(φ_n + n·c)).
        let n_dates = 8;
        let phi = |n: usize| 0.07 * n as f64;
        let mut linked = Array3::from_shape_fn((n_dates, rows, cols), |(n, _, _)| {
            Cf64::from_polar(1.0, phi(n) + n as f64 * c)
        });
        correct_phase_bias(&mut linked, beta.view());
        for n in 0..n_dates {
            for r in 0..rows {
                for col in 0..cols {
                    let got = linked[(n, r, col)].arg();
                    let want = ((phi(n) + std::f64::consts::PI) % (2.0 * std::f64::consts::PI))
                        - std::f64::consts::PI;
                    assert!((got - want).abs() < 1e-9, "date {n}: {got} != {want}");
                }
            }
        }
    }

    /// Constant bias ⇒ residual closure is identically zero (the model fully
    /// explains a constant-rate bias).
    #[test]
    fn residual_is_zero_for_constant_bias() {
        let closure = Array3::from_elem((5, 2, 2), 0.5_f64);
        let beta = estimate_bias_velocity(closure.view());
        let resid = residual_closure(closure.view(), beta.view());
        assert!(mean_abs_closure(resid.view()) < 1e-12);
    }

    /// Measured reduction on a **long** series (98 triplets / 100 dates): closures
    /// are a constant systematic bias plus deterministic zero-mean noise; removing
    /// the modeled bias leaves the noise, cutting mean non-closure several-fold.
    /// Numbers recorded in VALIDATION.md.
    #[test]
    fn reduces_nonclosure_on_long_noisy_series() {
        let bias = 0.4_f64; // closure systematic = 2·bias = 0.8 rad
        let ntri = 98;
        // Deterministic zero-mean noise so the test is reproducible.
        let noise = |k: usize| 0.15 * ((k as f64 * 1.3).sin());
        let closure = Array3::from_shape_fn((ntri, 4, 4), |(k, _, _)| 2.0 * bias + noise(k));
        let beta = estimate_bias_velocity(closure.view());
        let before = mean_abs_closure(closure.view());
        let after = mean_abs_closure(residual_closure(closure.view(), beta.view()).view());
        eprintln!(
            "phase-bias non-closure (100-date series): before {before:.4} rad -> after {after:.4} rad ({:.1}x reduction)",
            before / after
        );
        assert!(
            after < before * 0.5,
            "expected a >2x reduction: {before} -> {after}"
        );
    }

    /// All-NaN closure ⇒ zero bias (no correction), series untouched.
    #[test]
    fn nan_closure_is_no_correction() {
        let closure = Array3::from_elem((3, 1, 1), f64::NAN);
        let beta = estimate_bias_velocity(closure.view());
        assert_eq!(beta[(0, 0)], 0.0);
        let mut linked = Array3::from_elem((4, 1, 1), Cf64::from_polar(1.0, 0.5));
        let before = linked.clone();
        correct_phase_bias(&mut linked, beta.view());
        assert_eq!(linked, before);
    }
}
