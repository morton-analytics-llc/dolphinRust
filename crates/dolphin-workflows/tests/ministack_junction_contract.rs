//! ENG-007: the compressed-SLC chain carries one phase datum across a
//! ministack junction.
//!
//! Ministack `k+1` is phase-linked against the compressed SLC of ministack
//! `k` and referenced to it (`output_reference_idx = 0` under
//! `CompressedSlcPlan::AlwaysFirst`), so its linked phases are relative to
//! the compressed SLC's phase, which is the projection of ministack `k` onto
//! its own linked phase, i.e. relative to ministack `k`'s reference date. A
//! per-date phase common to every pixel of the later ministack (an
//! atmospheric-like datum) must therefore come through the junction as a step
//! of exactly that datum plus the deformation increment, at every pixel.

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{run_sequential, SequentialConfig};
use ndarray::Array3;
use std::f64::consts::TAU;

const N_DATES: usize = 30;
const MINISTACK_SIZE: usize = 15;
const ROWS: usize = 16;
const COLS: usize = 16;
/// Constant phase added to every pixel of every date in the second ministack.
const DATUM_RAD: f64 = 1.1;

/// Deterministic uniform value in `[0, 1)` from an integer seed.
fn hash01(seed: u64) -> f64 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % 1_000_003) as f64 / 1_000_003.0
}

/// Smooth deformation rate in radians per epoch at a pixel. The gradient is
/// kept well below one look window's worth of phase so the window-averaged
/// estimate at a truncated frame edge stays inside the tolerance.
fn rate(row: usize, col: usize) -> f64 {
    0.05 + 0.0002 * (row + col) as f64
}

/// Deformation phase of acquisition `k` at a pixel.
fn deformation(k: usize, row: usize, col: usize) -> f64 {
    rate(row, col) * k as f64
}

/// The per-date datum: zero in the first ministack, `DATUM_RAD` in the second.
fn datum(k: usize) -> f64 {
    match k >= MINISTACK_SIZE {
        true => DATUM_RAD,
        false => 0.0,
    }
}

/// Distributed-scatterer stack: per-pixel amplitude and scatterer phase, the
/// smooth deformation, the second-ministack datum, and a small deterministic
/// noise so the coherence matrix is not singular.
fn synthetic_stack() -> Array3<Cf64> {
    Array3::from_shape_fn((N_DATES, ROWS, COLS), |(k, row, col)| {
        let pixel = (row * COLS + col) as u64;
        let amplitude = 0.5 + hash01(pixel * 7 + 1);
        let scatterer = TAU * hash01(pixel * 7 + 2);
        let phase = scatterer + deformation(k, row, col) + datum(k);
        let sample = k as u64 * 1_000 + pixel;
        let noise = Cf64::new(
            0.1 * (hash01(sample * 3 + 5) - 0.5),
            0.1 * (hash01(sample * 3 + 6) - 0.5),
        );
        Cf64::from_polar(amplitude, phase) + noise
    })
}

fn config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: MINISTACK_SIZE,
        max_num_compressed: 10,
        half_window: HalfWindow { y: 2, x: 2 },
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: false,
        shp_method: ShpMethod::Rect,
        shp_alpha: 0.001,
    }
}

/// Wrap radians to `(-π, π]`.
fn wrap(radians: f64) -> f64 {
    Cf64::from_polar(1.0, radians).arg()
}

#[test]
fn junction_step_is_the_deformation_increment_once_the_datum_is_removed() {
    let stack = synthetic_stack();
    let out = run_sequential(
        stack.view(),
        &config(),
        &ComputeEngine::new(ComputeBackend::Cpu),
    )
    .unwrap();
    assert_eq!(out.cpx_phase.dim(), (N_DATES, ROWS, COLS));

    let junction = MINISTACK_SIZE;
    let tolerance = 0.05;
    let mut worst_junction = 0.0_f64;
    let mut worst_series = 0.0_f64;
    for row in 0..ROWS {
        for col in 0..COLS {
            let linked = |k: usize| out.cpx_phase[(k, row, col)].arg();
            let step = wrap(linked(junction) - linked(junction - 1) - DATUM_RAD);
            worst_junction = worst_junction.max((step - rate(row, col)).abs());
            for k in 0..N_DATES {
                let expected = deformation(k, row, col) + datum(k);
                worst_series = worst_series.max(wrap(linked(k) - expected).abs());
            }
        }
    }
    assert!(
        worst_junction < tolerance,
        "junction step minus the datum differs from the deformation increment by up to \
         {worst_junction:.4} rad"
    );
    assert!(
        worst_series < tolerance,
        "linked series differs from deformation plus datum by up to {worst_series:.4} rad"
    );
}
