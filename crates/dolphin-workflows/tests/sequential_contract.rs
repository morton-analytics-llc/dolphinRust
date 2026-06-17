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
