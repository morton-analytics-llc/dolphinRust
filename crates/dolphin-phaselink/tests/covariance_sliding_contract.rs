//! Sliding-sum covariance contract: the separable box-sum kernel (the default
//! for the unmasked rectangular-window path) must match the direct per-pixel
//! kernel to the crate's coherence tolerance (~1e-4), not bit-exactly — the
//! running-sum reorders FP accumulation and subtracts. The masked path is
//! unchanged and must stay bit-identical to direct.

use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::covariance::{estimate_stack_covariance, estimate_stack_covariance_direct};
use ndarray::{Array3, Array4};

/// Coherence-entry tolerance (crate contract: coherence ~1e-4).
const COH_TOL: f64 = 1e-4;

/// Deterministic pseudo-random complex stack — a cheap LCG, no rand dep.
fn synth_stack(nslc: usize, rows: usize, cols: usize) -> Array3<Cf64> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };
    Array3::from_shape_fn((nslc, rows, cols), |_| Cf64::new(next(), next()))
}

fn max_coh_diff(a: &Array4<Cf64>, b: &Array4<Cf64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0_f64, f64::max)
}

#[test]
fn sliding_matches_direct_within_tolerance_unmasked() {
    let stack = synth_stack(8, 40, 48);
    let cases = [
        (HalfWindow { y: 5, x: 11 }, Strides { y: 3, x: 6 }),
        (HalfWindow { y: 2, x: 2 }, Strides { y: 1, x: 1 }),
        (HalfWindow { y: 4, x: 3 }, Strides { y: 2, x: 3 }),
    ];
    for (half, strides) in cases {
        let sliding = estimate_stack_covariance(stack.view(), half, strides, None).unwrap();
        let direct = estimate_stack_covariance_direct(stack.view(), half, strides, None).unwrap();
        let diff = max_coh_diff(&sliding, &direct);
        assert!(
            diff < COH_TOL,
            "sliding vs direct coherence diff {diff:e} exceeds {COH_TOL:e} \
             for half={half:?} strides={strides:?}"
        );
    }
}

#[test]
fn diagonal_is_unit_coherence() {
    let stack = synth_stack(6, 32, 32);
    let half = HalfWindow { y: 3, x: 3 };
    let strides = Strides { y: 2, x: 2 };
    let cov = estimate_stack_covariance(stack.view(), half, strides, None).unwrap();
    let off_unit = cov
        .indexed_iter()
        .filter(|((_, _, i, j), _)| i == j)
        .map(|(_, z)| ((z.re - 1.0).powi(2) + z.im.powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(
        off_unit < COH_TOL,
        "diagonal not unit coherence: {off_unit:e}"
    );
}
