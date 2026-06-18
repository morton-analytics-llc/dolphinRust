//! Phase 2b (v1.4.0) end-to-end NRT front-door contract.
//!
//! An incremental displacement update — `run_displacement_resumable` on an
//! initial window then `update_displacement` folding in the later acquisitions —
//! must reproduce a full `run_displacement` of the extended stack. Phase-linking
//! is bit-identical (Phase 2) and the downstream is the shared deterministic
//! tail, so the whole displacement product matches. Skips without the disp
//! fixture or `snaphu`.

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::{run_displacement, run_displacement_resumable, update_displacement};
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

fn max3(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "layer shape mismatch");
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn max2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "layer shape mismatch");
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Incremental displacement (3 dates, then fold in the remaining 2) equals a full
/// run of all 5. `ministack_size = 2` so the open trailing ministack seals during
/// the update and a new ministack opens — exercising the carry across a boundary.
#[test]
fn incremental_displacement_matches_full_run() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !config.exists() || !snaphu_available() {
        eprintln!("skipping NRT displacement contract: no fixtures / snaphu");
        return;
    }
    let base = DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    assert!(base.cslc_file_list.len() >= 5, "fixture needs >=5 dates");

    // Full run of the extended stack.
    let mut full_cfg = base.clone();
    full_cfg.phase_linking.ministack_size = 2;
    full_cfg.work_directory = std::env::temp_dir().join("dolphinrust_nrt_full");
    let full = run_displacement(&full_cfg).unwrap();

    // Incremental: initial 3 dates, then fold in dates 4 and 5 together.
    let mut init_cfg = full_cfg.clone();
    init_cfg.work_directory = std::env::temp_dir().join("dolphinrust_nrt_inc");
    init_cfg.cslc_file_list = base.cslc_file_list[..3].to_vec();
    let (_out_init, state) = run_displacement_resumable(&init_cfg).unwrap();

    let mut ext_cfg = init_cfg.clone();
    ext_cfg.cslc_file_list = base.cslc_file_list.clone();
    let (inc, _state2) = update_displacement(&state, &ext_cfg).unwrap();

    let dd = max3(&inc.displacement, &full.displacement);
    let dv = max2(&inc.velocity, &full.velocity);
    let dt = max2(&inc.temporal_coherence, &full.temporal_coherence);
    let dc = match (&inc.crlb_sigma, &full.crlb_sigma) {
        (Some(a), Some(b)) => max3(a, b),
        _ => 0.0,
    };
    eprintln!("incremental vs full displacement: ddisp={dd:.2e} dvel={dv:.2e} dtcoh={dt:.2e} dcrlb={dc:.2e}");
    assert_eq!(inc.acquisition_days, full.acquisition_days, "dates match");
    assert_eq!(
        inc.reference_point, full.reference_point,
        "ref point matches"
    );
    // Phase-linking is bit-identical and the downstream is deterministic (same
    // SNAPHU input → same output), so the products match to f64 round-off.
    assert!(dd < 1e-6, "displacement max|Δ| {dd}");
    assert!(dv < 1e-6, "velocity max|Δ| {dv}");
    assert!(dt < 1e-6, "temporal coherence max|Δ| {dt}");
    assert!(dc < 1e-6, "crlb max|Δ| {dc}");
}

/// An update that extends no burst (same file list) is rejected, not silently a
/// no-op — guards the "every update extends every burst" contract.
#[test]
fn update_without_new_acquisitions_errors() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !config.exists() || !snaphu_available() {
        eprintln!("skipping NRT no-op guard: no fixtures / snaphu");
        return;
    }
    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    cfg.phase_linking.ministack_size = 2;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_nrt_noop");
    cfg.cslc_file_list = cfg.cslc_file_list[..3].to_vec();
    let (_out, state) = run_displacement_resumable(&cfg).unwrap();
    let err = match update_displacement(&state, &cfg) {
        Ok(_) => panic!("expected an error when no burst is extended"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("no new acquisitions"),
        "expected no-new-acquisitions error, got: {err}"
    );
}
