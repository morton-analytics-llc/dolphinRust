//! Contract: split-spectrum (main/side-band) ionosphere estimation.
//!
//! Synthetic two-band phase `φ(f) = φ_nd·f/f_A + φ_iono·f_A/f` with known
//! non-dispersive and dispersive fields. The estimator must return the
//! dispersive term at `f_A`, a `2π` side-band slip must shift it by the
//! documented constant, and a missing side band must be refused. Per-component
//! side-band cycle offsets (+1 and −2 planted in two components) must be found
//! and removed before estimation.

use std::f64::consts::TAU;

use dolphin_corrections::split_spectrum::{
    align_side_band_cycles, estimate_ionosphere, lowpass_ionosphere, side_band_cycle_offset,
    BandPhase, ComponentCycles, SplitSpectrumError,
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
fn planted_side_band_cycle_offsets_are_removed_per_component() {
    // Three components: left columns (0), middle (1, +1 cycle), right (2, −2 cycles).
    let labels = Array2::from_shape_fn((ROWS, COLS), |(_, c)| (c / 16) as u32);
    let planted = |label: u32| [0.0, 1.0, -2.0][label as usize];
    let phase_a = band_phase(MAIN_HZ);
    let phase_b = band_phase(SIDE_HZ);
    let slipped_b = ndarray::Zip::from(&phase_b)
        .and(&labels)
        .map_collect(|&phase, &label| phase + TAU * planted(label));
    let main = band(&phase_a, MAIN_HZ);

    let aligned = align_side_band_cycles(&main, &band(&slipped_b, SIDE_HZ), labels.view()).unwrap();
    let expected: Vec<ComponentCycles> = [(0, 0), (1, 1), (2, -2)]
        .map(|(component, k)| ComponentCycles {
            component,
            k,
            pixel_count: ROWS * 16,
        })
        .to_vec();
    assert_eq!(aligned.components, expected);

    let side = band(&aligned.unwrapped_phase, SIDE_HZ);
    let estimate = estimate_ionosphere(&main, Some(&side)).unwrap();
    let squared: f64 = (&estimate - &ionosphere_at_main()).mapv(|e| e * e).sum();
    let rms = (squared / estimate.len() as f64).sqrt();
    assert!(rms < 0.01, "ionosphere residual RMS {rms} rad");
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
    let one_component = Array2::from_elem((ROWS, COLS), 1_u32);
    let smoothed = lowpass_ionosphere(field.view(), one_component.view(), 2_000.0, 100.0).unwrap();
    // The field is not periodic-extended past the image edge, so the checkerboard only
    // cancels exactly away from it: check pixels at least 5σ (σ ≈ 2.65 px) inside.
    let margin = 14;
    smoothed
        .slice(ndarray::s![margin..ROWS - margin, margin..COLS - margin])
        .iter()
        .for_each(|v| assert!((v - 0.7).abs() < 1e-6, "{v}"));
}

/// Smooth screen at 100 m posting, offset from zero so a zero-fill bias would show.
fn smooth_screen(rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        3.0 + (TAU * (r as f64 + 0.5 * c as f64) / 2_000.0).sin()
    })
}

#[test]
fn masked_lowpass_recovers_a_smooth_screen_next_to_a_hole() {
    let (rows, cols) = (96, 96);
    let screen = smooth_screen(rows, cols);
    let hole = |r: usize, c: usize| (40..56).contains(&r) && (40..56).contains(&c);
    let with_hole = Array2::from_shape_fn((rows, cols), |(r, c)| match hole(r, c) {
        true => f64::NAN,
        false => screen[(r, c)],
    });
    let labels = Array2::from_elem((rows, cols), 1_u32);
    let smoothed = lowpass_ionosphere(with_hole.view(), labels.view(), 2_000.0, 100.0).unwrap();

    let near_hole = |r: usize, c: usize| (37..59).contains(&r) && (37..59).contains(&c);
    let mut worst_near_hole = 0.0_f64;
    for ((r, c), value) in smoothed.indexed_iter() {
        if hole(r, c) {
            assert!(value.is_nan(), "hole pixel ({r}, {c}) = {value}");
            continue;
        }
        let error = (value - screen[(r, c)]).abs();
        assert!(error < 0.01, "({r}, {c}): |error| {error} rad");
        if near_hole(r, c) {
            worst_near_hole = worst_near_hole.max(error);
        }
    }
    eprintln!("max |error| within 3 px of the hole: {worst_near_hole:.2e} rad");
}

#[test]
fn per_component_lowpass_does_not_bleed_offsets_across_components() {
    // Two components abutting at column 24, with constant offsets 25 rad apart.
    let labels = Array2::from_shape_fn((ROWS, COLS), |(_, c)| 1 + u32::from(c >= 24));
    let offset = |label: u32| if label == 1 { 5.0 } else { -20.0 };
    let field = labels.mapv(offset);
    let smoothed = lowpass_ionosphere(field.view(), labels.view(), 2_000.0, 100.0).unwrap();
    ndarray::Zip::from(&smoothed)
        .and(&labels)
        .for_each(|&value, &label| {
            assert!(
                (value - offset(label)).abs() < 1e-9,
                "{value} in component {label}"
            )
        });
}
