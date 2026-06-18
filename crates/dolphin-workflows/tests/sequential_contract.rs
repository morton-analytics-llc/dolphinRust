//! Phase-5 sequential-loop contract test (oracle).
//!
//! The end-to-end per-date linked phase and the carried compressed SLCs from
//! the Rust sequential estimator must match dolphin v0.35.0's primitive-based
//! sequential run (run_phase_linking + compress per ministack, MiniStackPlanner
//! carry-forward). Skips when fixtures are absent.

use std::path::{Path, PathBuf};

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend};
use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{run_sequential, SequentialConfig};
use ndarray::{Array2, Array3};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn to_c64<D: ndarray::Dimension>(a: ndarray::Array<Cf32, D>) -> ndarray::Array<Cf64, D> {
    a.mapv(|z| Cf64::new(z.re as f64, z.im as f64))
}

/// `|⟨a, b⟩|` normalized — global-phase-invariant eigenvector similarity.
fn cos_sim(a: &[Cf64], b: &[Cf64]) -> f64 {
    let inner: Cf64 = a.iter().zip(b).map(|(x, y)| x * y.conj()).sum();
    let na = a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let nb = b.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    inner.norm() / (na * nb)
}

#[test]
fn sequential_phase_history_matches_oracle() {
    let dir = fixtures();
    if !dir.join("sequential_phase.npy").exists() {
        eprintln!("skipping sequential oracle: no fixtures");
        return;
    }
    let stack = to_c64(
        ndarray_npy::read_npy::<_, Array3<Cf32>>(dir.join("sequential_slc_stack.npy")).unwrap(),
    );
    let phase_o =
        to_c64(ndarray_npy::read_npy::<_, Array3<Cf32>>(dir.join("sequential_phase.npy")).unwrap());
    let comp_o = to_c64(
        ndarray_npy::read_npy::<_, Array3<Cf32>>(dir.join("sequential_compressed.npy")).unwrap(),
    );

    let cfg = SequentialConfig {
        ministack_size: 5,
        max_num_compressed: 10,
        half_window: HalfWindow { y: 1, x: 1 },
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: false,
        compute_closure_phase: false,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let out = run_sequential(stack.view(), &cfg, &engine).unwrap();

    // Per-date phase: compare each pixel's time series (global-phase invariant).
    let (nslc, rows, cols) = out.cpx_phase.dim();
    assert_eq!((nslc, rows, cols), phase_o.dim(), "phase-history shape");
    let mut min_sim = 1.0_f64;
    for r in 0..rows {
        for c in 0..cols {
            let rust: Vec<Cf64> = (0..nslc).map(|t| out.cpx_phase[(t, r, c)]).collect();
            let orc: Vec<Cf64> = (0..nslc).map(|t| phase_o[(t, r, c)]).collect();
            min_sim = min_sim.min(cos_sim(&rust, &orc));
        }
    }
    assert!(min_sim > 1.0 - 1e-3, "phase-history cos-sim {min_sim}");

    // Compressed SLCs: magnitude + phase must match (amplitude is meaningful).
    assert_eq!(
        out.compressed_slcs.len(),
        comp_o.dim().0,
        "compressed count"
    );
    let max_err = compressed_error(&out.compressed_slcs, &comp_o);
    assert!(max_err < 1e-3, "compressed-SLC error {max_err}");
}

/// Stitched temporal coherence + concatenated CRLB/closure on a >=2-ministack
/// stack must match dolphin v0.42 composed per `sequential.rs` (gen_stitch_v042).
/// Closes the CRLB/closure many-ministack concatenation caveat: the same stack
/// produces all three layers and they hold against the oracle.
#[test]
fn stitching_and_quality_match_oracle_multiministack() {
    let dir = fixtures();
    if !dir.join("stitch_temp_coh_full.npy").exists() {
        eprintln!("skipping stitch oracle: no fixtures");
        return;
    }
    let stack = to_c64(
        ndarray_npy::read_npy::<_, Array3<Cf32>>(dir.join("sequential_slc_stack.npy")).unwrap(),
    );
    let tcoh_o: Array2<f32> = ndarray_npy::read_npy(dir.join("stitch_temp_coh_full.npy")).unwrap();
    let crlb_o: Array3<f32> = ndarray_npy::read_npy(dir.join("stitch_crlb.npy")).unwrap();
    let closure_o: Array3<f32> = ndarray_npy::read_npy(dir.join("stitch_closure.npy")).unwrap();

    let cfg = SequentialConfig {
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
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let out = run_sequential(stack.view(), &cfg, &engine).unwrap();

    // Two ministacks must have formed (else this isn't a stitching test).
    assert_eq!(out.compressed_slcs.len(), 2, "expected 2 ministacks");

    let tcoh_err = nan_aware_maxerr(out.temporal_coherence.view(), tcoh_o.view());
    assert!(
        tcoh_err < 1e-3,
        "stitched temporal coherence err {tcoh_err}"
    );

    let crlb = out.crlb_sigma.expect("crlb enabled");
    assert_eq!(crlb.dim(), crlb_o.dim(), "crlb shape");
    let crlb_err = nan_aware_maxerr3(crlb.view(), crlb_o.view());
    assert!(crlb_err < 1e-3, "concatenated CRLB err {crlb_err}");

    let closure = out.closure_phase.expect("closure enabled");
    assert_eq!(closure.dim(), closure_o.dim(), "closure shape");
    let closure_err = wrapped_maxerr3(closure.view(), closure_o.view());
    assert!(closure_err < 1e-3, "concatenated closure err {closure_err}");
}

/// Max abs error over pixels finite in both fields (mirrors `numpy.nanmean` —
/// NaN pixels are masked on both sides, not compared).
fn nan_aware_maxerr(a: ndarray::ArrayView2<f64>, b: ndarray::ArrayView2<f32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn nan_aware_maxerr3(a: ndarray::ArrayView3<f64>, b: ndarray::ArrayView3<f32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - *y as f64).abs())
        .fold(0.0, f64::max)
}

/// Phase-aware max error: wrap the difference to `[-π, π]` before taking abs.
fn wrapped_maxerr3(a: ndarray::ArrayView3<f64>, b: ndarray::ArrayView3<f32>) -> f64 {
    let pi = std::f64::consts::PI;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (x - *y as f64).rem_euclid(2.0 * pi);
            (if d > pi { d - 2.0 * pi } else { d }).abs()
        })
        .fold(0.0, f64::max)
}

fn compressed_error(rust: &[Array2<Cf64>], oracle: &Array3<Cf64>) -> f64 {
    rust.iter()
        .enumerate()
        .flat_map(|(k, slc)| {
            let layer = oracle.index_axis(ndarray::Axis(0), k);
            slc.iter()
                .zip(layer.into_iter().collect::<Vec<_>>())
                .map(|(a, b)| (a - b).norm())
                .collect::<Vec<_>>()
        })
        .fold(0.0_f64, f64::max)
}
