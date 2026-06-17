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
}
