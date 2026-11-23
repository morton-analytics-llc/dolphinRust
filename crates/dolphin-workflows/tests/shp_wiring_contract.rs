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
use dolphin_phaselink::{ComputeEngine, FusedParams};
use dolphin_shp::estimate_neighbors_glrt;
use dolphin_workflows::{
    run_sequential, run_sequential_resumable, update_sequential, SequentialConfig,
};
use ndarray::{s, Array2, Array3, Array4, ArrayView3, Axis};

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

// ---- Issue #165: GLRT amplitude statistics come from the whole stack ----------

/// The dolphin v0.42.0 oracle scene of `oracle/gen_shp_fullstack_v042.py`: 30
/// dates in three ministacks of 10, half-window 2.
const ORACLE_MINISTACK: usize = 10;
const ORACLE_HALF: HalfWindow = HalfWindow { y: 2, x: 2 };

fn oracle_fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn oracle_stack() -> Array3<Cf64> {
    let stack: Array3<dolphin_core::Cf32> =
        ndarray_npy::read_npy(oracle_fixtures().join("shp_fullstack_slc_stack.npy")).unwrap();
    stack.mapv(|z| Cf64::new(f64::from(z.re), f64::from(z.im)))
}

fn oracle_config(shp_method: ShpMethod) -> SequentialConfig {
    SequentialConfig {
        ministack_size: ORACLE_MINISTACK,
        half_window: ORACLE_HALF,
        compute_crlb: false,
        ..config(shp_method)
    }
}

/// GLRT neighbours from the amplitude mean and variance of `dates`.
fn glrt_neighbors(dates: ArrayView3<Cf64>) -> Array4<bool> {
    let amplitude = dates.mapv(|z| z.norm());
    estimate_neighbors_glrt(
        amplitude.mean_axis(Axis(0)).unwrap().view(),
        amplitude.var_axis(Axis(0), 0.0).view(),
        ORACLE_HALF,
        dates.dim().0,
        Strides { y: 1, x: 1 },
        0.001,
    )
}

fn neighbor_counts(neighbors: &Array4<bool>) -> Array2<u16> {
    let (rows, cols, _, _) = neighbors.dim();
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        neighbors
            .slice(s![row, col, .., ..])
            .iter()
            .filter(|&&is_neighbor| is_neighbor)
            .count() as u16
    })
}

/// The pixels dolphin's workflow writes counts for: it blanks a half-window
/// border of the block, so only the interior is compared.
fn interior(counts: &Array2<u16>) -> ndarray::ArrayView2<'_, u16> {
    let (rows, cols) = counts.dim();
    counts.slice(s![
        ORACLE_HALF.y..rows - ORACLE_HALF.y,
        ORACLE_HALF.x..cols - ORACLE_HALF.x
    ])
}

/// Parity with dolphin v0.42.0's sequential workflow (`ps.create_ps` +
/// `run_wrapped_phase_sequential`, GLRT): it wrote the same `shp_counts` for
/// all three ministacks, and whole-stack statistics with `nslc = 30` reproduce
/// them exactly. Each ministack's own statistics do not.
#[test]
fn full_stack_glrt_statistics_reproduce_dolphins_shp_counts_for_every_ministack() {
    let stack = oracle_stack();
    let dolphin: Array3<u16> =
        ndarray_npy::read_npy(oracle_fixtures().join("shp_fullstack_counts.npy")).unwrap();
    assert_eq!(dolphin.dim().0, 3);
    let full_stack = neighbor_counts(&glrt_neighbors(stack.view()));
    for (index, ministack) in stack
        .axis_chunks_iter(Axis(0), ORACLE_MINISTACK)
        .enumerate()
    {
        let dolphin_counts = dolphin.index_axis(Axis(0), index).to_owned();
        assert_eq!(
            interior(&full_stack),
            interior(&dolphin_counts),
            "ministack {index}"
        );
        let per_ministack = neighbor_counts(&glrt_neighbors(ministack));
        let differing = interior(&per_ministack)
            .iter()
            .zip(interior(&dolphin_counts))
            .filter(|(ours, theirs)| ours != theirs)
            .count();
        assert!(
            differing > 30,
            "ministack {index}: per-ministack statistics differ from dolphin at only {differing} pixels"
        );
    }
}

/// First-ministack phases of a sequential run under `shp_method`.
fn first_ministack_phase(stack: &Array3<Cf64>, shp_method: ShpMethod) -> Array3<Cf64> {
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    run_sequential(stack.view(), &oracle_config(shp_method), &engine)
        .unwrap()
        .cpx_phase
        .slice(s![..ORACLE_MINISTACK, .., ..])
        .to_owned()
}

/// The first ministack carries no compressed SLC, so phase-linking it directly
/// with a given neighbour mask is the whole computation.
fn linked_with(stack: &Array3<Cf64>, neighbors: &Array4<bool>) -> Array3<Cf64> {
    let cfg = oracle_config(ShpMethod::Glrt);
    let params = FusedParams {
        use_evd: cfg.use_evd,
        beta: cfg.beta,
        zero_correlation_threshold: cfg.zero_correlation_threshold,
        reference_idx: 0,
        compute_crlb: false,
        crlb_reference_idx: 0,
        num_looks: (ORACLE_HALF.y as f64 * ORACLE_HALF.x as f64).sqrt(),
        compute_closure: false,
        compute_average_coherence: false,
        average_coherence_start_idx: 0,
    };
    ComputeEngine::new(ComputeBackend::Cpu)
        .link(
            stack.slice(s![..ORACLE_MINISTACK, .., ..]),
            ORACLE_HALF,
            Strides { y: 1, x: 1 },
            Some(neighbors.view()),
            params,
        )
        .unwrap()
        .cpx_phase
}

fn bit_identical(left: &Array3<Cf64>, right: &Array3<Cf64>) -> bool {
    left.iter()
        .zip(right)
        .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits())
}

/// The driver applies what the method says: under `glrt` the first
/// ministack is linked with neighbours chosen from all 30 dates, under
/// `glrt_ministack` from its own 10, and on this scene the two differ.
#[test]
fn the_sequential_driver_links_each_ministack_with_the_configured_statistics() {
    let stack = oracle_stack();
    let from_all_dates = glrt_neighbors(stack.view());
    let from_first_ministack = glrt_neighbors(stack.slice(s![..ORACLE_MINISTACK, .., ..]));
    assert_ne!(from_all_dates, from_first_ministack);

    let full_stack = first_ministack_phase(&stack, ShpMethod::Glrt);
    let ministack = first_ministack_phase(&stack, ShpMethod::GlrtMinistack);
    assert!(bit_identical(
        &full_stack,
        &linked_with(&stack, &from_all_dates)
    ));
    assert!(bit_identical(
        &ministack,
        &linked_with(&stack, &from_first_ministack)
    ));
    assert!(!bit_identical(&full_stack, &ministack));
}

/// `glrt_ministack` keeps the incremental guarantee: an update is
/// bit-identical to a full rerun. Under `glrt` it is not, because a
/// rerun reselects the sealed ministacks' neighbours with the later dates;
/// the update's own statistics still cover every date (see the unit tests in
/// `sequential.rs`).
#[test]
fn an_update_matches_a_rerun_under_ministack_statistics_only() {
    let stack = oracle_stack();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let split = 2 * ORACLE_MINISTACK;
    let updated = |shp_method| {
        let cfg = oracle_config(shp_method);
        let (_, state) =
            run_sequential_resumable(stack.slice(s![..split, .., ..]), &cfg, &engine).unwrap();
        let (output, _) =
            update_sequential(&state, stack.slice(s![split.., .., ..]), &cfg, &engine).unwrap();
        let rerun = run_sequential(stack.view(), &cfg, &engine).unwrap();
        (output.cpx_phase, rerun.cpx_phase)
    };
    let (update, rerun) = updated(ShpMethod::GlrtMinistack);
    assert!(bit_identical(&update, &rerun));
    let (update, rerun) = updated(ShpMethod::Glrt);
    assert!(!bit_identical(&update, &rerun));
}
