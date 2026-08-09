//! Issue #29: the SHP neighbor mask must reach the covariance kernel.
//!
//! `phase_linking.shp_method` was accepted in config but never applied — the
//! sequential workflow passed `neighbors: None` unconditionally, so covariance
//! always used the full rectangular window and `dolphin-shp` had no caller.
//!
//! That is a correctness gap. With `beta: 0.0` the estimator uses `Γ = |C|`
//! unregularized, and the entrywise modulus of a PSD matrix need not be PSD. A
//! window straddling two scattering populations can therefore produce an
//! indefinite `Γ`, which fails `crlb.rs`'s Cholesky and — since the CRLB path
//! has no EVD fallback — yields a NaN bound that the workflow's validity mask
//! propagates to every emitted layer.
//!
//! The fixture is exactly that: a 5x7 window whose left two columns are a
//! bright, temporally coherent population and whose remainder is dim and nearly
//! decorrelated. Unmasked, `Γ`'s smallest eigenvalue is about -0.30; under the
//! GLRT mask it is about +0.35.

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{run_sequential, SequentialConfig};
use ndarray::Array3;

const NSLC: usize = 12;
const HALF: HalfWindow = HalfWindow { y: 2, x: 3 };
const ROWS: usize = 2 * HALF.y + 1;
const COLS: usize = 2 * HALF.x + 1;
/// Columns left of this are the bright, coherent population.
const SPLIT: usize = 2;
const BRIGHT_GAIN: f64 = 16.0;
const BRIGHT_RHO: f64 = 0.90;
const DIM_RHO: f64 = 0.10;
/// Chosen so the unmasked `Γ` is indefinite and the GLRT-masked one is not.
const SEED: u64 = 366;

/// xorshift64*, so the fixture is reproducible without a PRNG dependency.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// One pixel's AR(1) complex series, written into `stack` at `(row, col)`.
fn write_series(stack: &mut Array3<Cf64>, rng: &mut Xorshift, row: usize, col: usize) {
    let bright = col < SPLIT;
    let gain = if bright { BRIGHT_GAIN } else { 1.0 };
    let rho = if bright { BRIGHT_RHO } else { DIM_RHO };
    let drive = (1.0 - rho * rho).max(0.0).sqrt();
    let mut prev = Cf64::new(rng.normal(), rng.normal());
    for date in 0..NSLC {
        let step = Cf64::new(rng.normal(), rng.normal());
        prev = prev * rho + step * drive;
        stack[(date, row, col)] = prev * gain;
    }
}

/// Two scattering populations side by side in one covariance window.
fn heterogeneous_stack() -> Array3<Cf64> {
    let mut rng = Xorshift(SEED);
    let mut stack = Array3::<Cf64>::zeros((NSLC, ROWS, COLS));
    for (row, col) in (0..ROWS).flat_map(|r| (0..COLS).map(move |c| (r, c))) {
        write_series(&mut stack, &mut rng, row, col);
    }
    stack
}

fn config(shp_method: ShpMethod) -> SequentialConfig {
    SequentialConfig {
        ministack_size: NSLC,
        max_num_compressed: 10,
        half_window: HALF,
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: true,
        compute_closure_phase: false,
        compute_average_coherence: false,
        shp_method,
        shp_alpha: 0.001,
    }
}

/// CRLB σ at the window's center pixel, per date.
fn center_crlb(shp_method: ShpMethod) -> Vec<f64> {
    let stack = heterogeneous_stack();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let out = run_sequential(stack.view(), &config(shp_method), &engine).unwrap();
    let crlb = out.crlb_sigma.expect("CRLB requested");
    (0..crlb.dim().0)
        .map(|date| crlb[(date, HALF.y, HALF.x)])
        .collect()
}

/// The rectangular window leaves `Γ = |C|` indefinite here, so the bound is NaN
/// on every non-reference date. Pinning today's behaviour is what makes the GLRT
/// case below a real difference rather than a no-op — and `rect` stays the
/// default, so every unmasked oracle contract keeps its meaning.
#[test]
fn rect_window_yields_a_singular_crlb_on_a_heterogeneous_window() {
    let sigma = center_crlb(ShpMethod::Rect);
    assert!(
        sigma.iter().skip(1).all(|value| value.is_nan()),
        "expected a singular CRLB under the rectangular window, got {sigma:?}"
    );
}

/// Selecting statistically homogeneous pixels drops the second population, `Γ`
/// becomes positive definite, and the bound is finite and positive.
#[test]
fn glrt_shp_mask_restores_a_finite_crlb() {
    let sigma = center_crlb(ShpMethod::Glrt);
    assert!(
        sigma
            .iter()
            .skip(1)
            .all(|value| value.is_finite() && *value > 0.0),
        "expected a finite CRLB under the GLRT mask, got {sigma:?}"
    );
}
