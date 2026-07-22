//! Phase 2 (v1.4.0) NRT incremental-update contract.
//!
//! Folding newly-arrived acquisitions into an existing sequential phase-linking
//! series via the carried compressed SLCs must reproduce a **full rerun** of the
//! extended stack. Because sequential phase-linking is feed-forward (a ministack
//! depends only on prior compressed SLCs + its own real SLCs), sealed (full)
//! ministacks are immutable, so the incremental result is **bit-identical** to a
//! full rerun — not merely within tolerance. We assert a tight numeric bound and
//! report the realized max difference.

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{
    run_sequential, run_sequential_resumable, update_sequential, SequentialConfig,
};
use ndarray::{s, Array2, Array3, ArrayView3};

/// Deterministic synthetic complex SLC stack: a per-date phase ramp plus a
/// pixel-dependent term, unit-ish magnitude. No physical meaning — only needs to
/// be deterministic and drive non-trivial phase-linking.
fn synth_stack(nslc: usize, rows: usize, cols: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((nslc, rows, cols), |(t, r, c)| {
        let phase = 0.30 * t as f64 + 0.05 * (r as f64) - 0.04 * (c as f64) + 0.01 * (t * r) as f64;
        let mag = 1.0 + 0.1 * ((r + c) as f64).sin();
        Cf64::from_polar(mag, phase)
    })
}

fn cfg() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 5,
        max_num_compressed: 10,
        half_window: HalfWindow { y: 1, x: 1 },
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: true,
        compute_closure_phase: true,
        compute_average_coherence: true,
    }
}

fn max_c(a: ArrayView3<Cf64>, b: ArrayView3<Cf64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "complex layer shape mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn max_compressed(a: &[Array2<Cf64>], b: &[Array2<Cf64>]) -> f64 {
    assert_eq!(a.len(), b.len(), "compressed count mismatch");
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).norm()))
        .fold(0.0, f64::max)
}

fn max_r(a: ndarray::ArrayView2<f64>, b: ndarray::ArrayView2<f64>) -> f64 {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn max_r3(a: ndarray::ArrayView3<f64>, b: ndarray::ArrayView3<f64>) -> f64 {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Assert two SequentialOutputs agree to f64 round-off across every layer.
fn assert_outputs_match(
    inc: &dolphin_workflows::SequentialOutput,
    full: &dolphin_workflows::SequentialOutput,
    tol: f64,
) {
    let dp = max_c(inc.cpx_phase.view(), full.cpx_phase.view());
    let dc = max_compressed(&inc.compressed_slcs, &full.compressed_slcs);
    let dt = max_r(
        inc.temporal_coherence.view(),
        full.temporal_coherence.view(),
    );
    let dphase_coh = match (&inc.phase_linking_coherence, &full.phase_linking_coherence) {
        (Some(a), Some(b)) => max_r(a.view(), b.view()),
        (None, None) => 0.0,
        _ => panic!("phase-linking-coherence presence mismatch"),
    };
    let dcrlb = match (&inc.crlb_sigma, &full.crlb_sigma) {
        (Some(a), Some(b)) => max_r3(a.view(), b.view()),
        (None, None) => 0.0,
        _ => panic!("crlb presence mismatch"),
    };
    let dclos = match (&inc.closure_phase, &full.closure_phase) {
        (Some(a), Some(b)) => max_r3(a.view(), b.view()),
        (None, None) => 0.0,
        _ => panic!("closure presence mismatch"),
    };
    eprintln!(
        "incremental vs full: dphase={dp:.2e} dcompressed={dc:.2e} dtcoh={dt:.2e} dphasecoh={dphase_coh:.2e} dcrlb={dcrlb:.2e} dclosure={dclos:.2e}"
    );
    assert!(dp < tol, "cpx_phase max|Δ| {dp}");
    assert!(dc < tol, "compressed max|Δ| {dc}");
    assert!(dt < tol, "temporal_coherence max|Δ| {dt}");
    assert!(
        dphase_coh < tol,
        "phase_linking_coherence max|Δ| {dphase_coh}"
    );
    assert!(dcrlb < tol, "crlb max|Δ| {dcrlb}");
    assert!(dclos < tol, "closure max|Δ| {dclos}");
}

const TOL: f64 = 1e-9;

/// Resumable run returns the same output as the plain run (no behaviour change).
#[test]
fn resumable_matches_plain_run() {
    let stack = synth_stack(9, 8, 8);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let plain = run_sequential(stack.view(), &cfg(), &engine).unwrap();
    let (resumable, _state) = run_sequential_resumable(stack.view(), &cfg(), &engine).unwrap();
    assert_outputs_match(&resumable, &plain, TOL);
}

/// Add 4 acquisitions at once to an existing 9-SLC series; the open trailing
/// ministack (4 real) seals at 5 and a new 3-real ministack opens. Result must
/// equal a full rerun of the 13-SLC stack.
#[test]
fn incremental_block_update_matches_full_rerun() {
    let full_stack = synth_stack(13, 8, 8);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let full = run_sequential(full_stack.view(), &cfg(), &engine).unwrap();

    let (_out9, state) =
        run_sequential_resumable(full_stack.slice(s![..9, .., ..]), &cfg(), &engine).unwrap();
    let new = full_stack.slice(s![9..13, .., ..]);
    let (inc, _state2) = update_sequential(&state, new, &cfg(), &engine).unwrap();

    assert_outputs_match(&inc, &full, TOL);
}

/// Stream acquisitions one at a time (the true NRT cadence): start from 9, fold
/// in #10, #11, #12, #13 individually. The final series must equal a full rerun.
#[test]
fn incremental_streaming_one_at_a_time_matches_full_rerun() {
    let full_stack = synth_stack(13, 8, 8);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let full = run_sequential(full_stack.view(), &cfg(), &engine).unwrap();

    let (_out, mut state) =
        run_sequential_resumable(full_stack.slice(s![..9, .., ..]), &cfg(), &engine).unwrap();
    let mut last = None;
    for t in 9..13 {
        let new = full_stack.slice(s![t..t + 1, .., ..]);
        let (out, next) = update_sequential(&state, new, &cfg(), &engine).unwrap();
        state = next;
        last = Some(out);
    }
    assert_outputs_match(&last.unwrap(), &full, TOL);
}

/// Edge: an existing series ending exactly on a ministack boundary (10 = 2×5)
/// has no open trailing ministack; folding in new SLCs still equals a full rerun.
#[test]
fn incremental_from_sealed_boundary_matches_full_rerun() {
    let full_stack = synth_stack(13, 8, 8);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let full = run_sequential(full_stack.view(), &cfg(), &engine).unwrap();

    let (_out10, state) =
        run_sequential_resumable(full_stack.slice(s![..10, .., ..]), &cfg(), &engine).unwrap();
    let new = full_stack.slice(s![10..13, .., ..]);
    let (inc, _s) = update_sequential(&state, new, &cfg(), &engine).unwrap();
    assert_outputs_match(&inc, &full, TOL);
}
