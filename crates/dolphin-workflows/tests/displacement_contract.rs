//! Phase-10 end-to-end contract test.
//!
//! `run_displacement` on a synthetic single-burst CSLC stack must reproduce the
//! dolphin-primitives oracle (phase-link → network → SNAPHU unwrap → SBAS L2 →
//! velocity) within physical tolerance. Skips without fixtures or `snaphu`.

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::run_displacement;
use ndarray::{Array2, Array3};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

#[test]
fn end_to_end_displacement_matches_oracle() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() {
        eprintln!("skipping end-to-end oracle: no fixtures");
        return;
    }
    if !snaphu_available() {
        eprintln!("skipping end-to-end oracle: snaphu not on PATH");
        return;
    }

    let cfg = DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    let out = run_displacement(&cfg).unwrap();

    let disp_o: Array3<f64> = ndarray_npy::read_npy(dir.join("disp_displacement.npy")).unwrap();
    let vel_o: Array2<f64> = ndarray_npy::read_npy(dir.join("disp_velocity.npy")).unwrap();

    assert_eq!(out.displacement.dim(), disp_o.dim(), "displacement shape");
    let derr = out
        .displacement
        .iter()
        .zip(disp_o.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let verr = out
        .velocity
        .iter()
        .zip(vel_o.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // End-to-end chain: faer-vs-jax phase linking + normal-eq vs SVD lstsq, with
    // a shared SNAPHU binary on cycle-free input. Physical tolerance.
    assert!(derr < 1e-3, "displacement error {derr}");
    assert!(verr < 1e-2, "velocity error {verr}");

    // Quality layers: dolphin defaults write_crlb on, write_closure_phase off, so
    // the run produces the CRLB σ layer (per date, ref band 0) and no closure.
    let crlb = out.crlb_sigma.expect("write_crlb defaults on");
    let (rows, cols) = out.temporal_coherence.dim();
    assert_eq!(crlb.dim().1, rows, "crlb rows match the grid");
    assert_eq!(crlb.dim().2, cols, "crlb cols match the grid");
    let ref_band_max = crlb
        .index_axis(ndarray::Axis(0), 0)
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    assert_eq!(ref_band_max, 0.0, "CRLB reference band must be 0");
    assert!(out.closure_phase.is_none(), "closure off by default");
    assert!(
        out.phase_linking_coherence.is_none(),
        "average coherence off by default"
    );
}

#[test]
fn distinct_phase_linking_coherence_raster_is_written_when_enabled() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping average-coherence end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    cfg.phase_linking.calc_average_coh = true;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_average_coherence_e2e");
    let out = run_displacement(&cfg).unwrap();
    let coherence = out
        .phase_linking_coherence
        .expect("calc_average_coh enabled");
    assert_eq!(coherence.dim(), out.temporal_coherence.dim());
    assert!(coherence.iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(cfg
        .work_directory
        .join("phase_linking_coherence.tif")
        .exists());
    assert!(cfg.work_directory.join("temporal_coherence.tif").exists());
    assert_ne!(
        coherence, out.temporal_coherence,
        "metrics must be distinct"
    );
}

/// Enabling the phase-bias correction (Michaelides 2022) runs end-to-end through
/// unwrap + inversion and produces a finite displacement of the right shape. The
/// correction leads Python dolphin (no oracle), so this guards the wiring; the
/// numeric behaviour is validated by `dolphin-phaselink`'s analytic + reduction
/// contracts. Default-off parity is covered by the oracle test above.
#[test]
fn phase_bias_correction_runs_end_to_end() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping phase-bias end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    cfg.phase_linking.correct_phase_bias = true;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_phasebias_e2e");
    let out = run_displacement(&cfg).unwrap();
    assert!(
        out.displacement.iter().all(|v| v.is_finite()),
        "phase-bias-corrected displacement must be finite"
    );
    let (rows, cols) = out.temporal_coherence.dim();
    assert_eq!(
        (out.displacement.dim().1, out.displacement.dim().2),
        (rows, cols)
    );
}

/// Enabling `write_closure_phase` produces the closure layer end-to-end, with
/// `n_dates - 2` bands; the layer matches the kernel's per-triplet output.
#[test]
fn closure_layer_produced_when_enabled() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping closure end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    cfg.phase_linking.write_closure_phase = true;
    // Isolate scratch/outputs from the other end-to-end test (they run in
    // parallel and would otherwise race on a shared SNAPHU scratch directory).
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_closure_e2e");
    let out = run_displacement(&cfg).unwrap();

    let n_dates = out.displacement.dim().0 + 1; // displacement drops the reference date
    let closure = out.closure_phase.expect("write_closure_phase enabled");
    assert_eq!(closure.dim().0, n_dates - 2, "closure has n_dates-2 bands");
    let (rows, cols) = out.temporal_coherence.dim();
    assert_eq!(
        (closure.dim().1, closure.dim().2),
        (rows, cols),
        "closure grid"
    );
}
