//! Contract: split-spectrum (main/side-band) ionosphere estimation.
//!
//! Synthetic two-band phase `φ(f) = φ_nd·f/f_A + φ_iono·f_A/f` with known
//! non-dispersive and dispersive fields. The estimator must return the
//! dispersive term at `f_A`, a `2π` side-band slip must shift it by the
//! documented constant, and a missing side band must be refused.

use std::f64::consts::TAU;

use dolphin_corrections::split_spectrum::{
    estimate_ionosphere, lowpass_ionosphere, side_band_cycle_offset, BandPhase, SplitSpectrumError,
};
use ndarray::Array2;

/// Illustrative L-band carriers: a wide main band and a narrow side band.
const MAIN_HZ: f64 = 1.2575e9;
const SIDE_HZ: f64 = 1.2935e9;
const ROWS: usize = 32;
const COLS: usize = 48;

/// Non-dispersive phase at `f_A`: a ramp up to ~30 rad (deformation + troposphere).
fn non_dispersive_at_main() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, COLS), |(r, c)| 0.4 * c as f64 + 0.3 * r as f64 - 2.0)
}

/// Dispersive (ionospheric) phase at `f_A`: a smooth screen of a few rad.
fn ionosphere_at_main() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, COLS), |(r, c)| {
        3.0 * (r as f64 / 9.0).sin() * (c as f64 / 13.0).cos() + 1.5
    })
}

/// Unwrapped phase observed at carrier `freq_hz`.
fn band_phase(freq_hz: f64) -> Array2<f64> {
    let non_dispersive = non_dispersive_at_main() * (freq_hz / MAIN_HZ);
    let dispersive = ionosphere_at_main() * (MAIN_HZ / freq_hz);
    non_dispersive + dispersive
}

fn band(phase: &Array2<f64>, freq_hz: f64) -> BandPhase<'_> {
    BandPhase {
        unwrapped_phase: phase.view(),
        center_frequency_hz: freq_hz,
    }
}

#[test]
fn recovers_dispersive_term_to_1e_9_rad() {
    let (phase_a, phase_b) = (band_phase(MAIN_HZ), band_phase(SIDE_HZ));
    let estimate =
        estimate_ionosphere(&band(&phase_a, MAIN_HZ), Some(&band(&phase_b, SIDE_HZ))).unwrap();
    let max_error = (&estimate - &ionosphere_at_main())
        .iter()
        .fold(0.0_f64, |acc, e| acc.max(e.abs()));
    assert!(max_error < 1e-9, "max |Δφ_iono| = {max_error:e} rad");
}

#[test]
fn side_band_cycle_slip_gives_documented_constant_offset() {
    let phase_a = band_phase(MAIN_HZ);
    let phase_b = band_phase(SIDE_HZ);
    let slipped_b = &phase_b + TAU;
    let main = band(&phase_a, MAIN_HZ);
    let clean = estimate_ionosphere(&main, Some(&band(&phase_b, SIDE_HZ))).unwrap();
    let slipped = estimate_ionosphere(&main, Some(&band(&slipped_b, SIDE_HZ))).unwrap();

    // Plan form with f₀ = f_A: f_A·f_B/(f₀(f_A² − f_B²)) · (2π·f_A).
    let documented =
        MAIN_HZ * SIDE_HZ / (MAIN_HZ * (MAIN_HZ.powi(2) - SIDE_HZ.powi(2))) * (TAU * MAIN_HZ);
    let helper = side_band_cycle_offset(MAIN_HZ, SIDE_HZ).unwrap();
    assert!(
        (helper - documented).abs() < 1e-9,
        "{helper} vs {documented}"
    );
    assert!((documented + 111.3).abs() < 0.1, "offset {documented} rad");
    (&slipped - &clean)
        .iter()
        .for_each(|offset| assert!((offset - documented).abs() < 1e-9, "{offset}"));
}

#[test]
fn missing_side_band_is_refused() {
    let phase_a = band_phase(MAIN_HZ);
    let result = estimate_ionosphere(&band(&phase_a, MAIN_HZ), None);
    assert!(matches!(result, Err(SplitSpectrumError::MissingSideBand)));
}

#[test]
fn lowpass_removes_short_wavelengths_and_keeps_the_mean() {
    // 100 m posting, 2 km cutoff: a Nyquist checkerboard is filtered out.
    let field = Array2::from_shape_fn((ROWS, COLS), |(r, c)| {
        0.7 + 0.3 * if (r + c) % 2 == 0 { 1.0 } else { -1.0 }
    });
    let smoothed = lowpass_ionosphere(field.view(), 2_000.0, 100.0).unwrap();
    smoothed
        .iter()
        .for_each(|v| assert!((v - 0.7).abs() < 1e-9, "{v}"));
}
