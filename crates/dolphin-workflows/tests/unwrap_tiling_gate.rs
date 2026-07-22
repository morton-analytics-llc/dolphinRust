//! Tier-1 #2 gate: opt-in SNAPHU auto-tiling.
//!
//! Auto-tiling changes SNAPHU numerics (tile boundaries + reconciliation), so it
//! is opt-in (`snaphu_options.auto_tile`, default off) and must clear the dolphin
//! oracle within physical tolerance before it can be recommended. On the shipped
//! oracle fixture (48x64) the conservative derivation yields a single tile, i.e.
//! auto-tile is a *no-op* — this test pins that: enabling the flag does not
//! regress end-to-end parity. (Tiling on large/noisy grids has no large oracle
//! fixture and is therefore held opt-in; see PLAYBOOK.) Skips without fixtures.

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::Cf64;
use dolphin_unwrap::{CostMode, InitMethod, UnwrapConfig};
use dolphin_workflows::{run_displacement, SnaphuBackend, UnwrapBackend};
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
fn auto_tile_opt_in_holds_oracle_parity() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping auto-tile gate: no fixtures / snaphu");
        return;
    }
    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    cfg.unwrap_options.snaphu_options.auto_tile = true;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_autotile_gate");
    std::fs::create_dir_all(&cfg.work_directory).unwrap();
    cfg.cslc_file_list = cfg
        .cslc_file_list
        .iter()
        .map(|source| {
            let target = cfg.work_directory.join(source.file_name().unwrap());
            std::fs::copy(source, &target).unwrap();
            let file = hdf5::File::open_rw(&target).unwrap();
            let group = file.group("data").unwrap();
            let shape = group.dataset("VV").unwrap().shape();
            let x = (0..shape[1])
                .map(|col| 500_015.0 + col as f64 * 30.0)
                .collect::<Vec<_>>();
            let y = (0..shape[0])
                .map(|row| 4_200_015.0 - row as f64 * 30.0)
                .collect::<Vec<_>>();
            group
                .new_dataset_builder()
                .with_data(&x)
                .create("x_coordinates")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&y)
                .create("y_coordinates")
                .unwrap();
            group
                .new_dataset::<i64>()
                .create("projection")
                .unwrap()
                .write_scalar(&32611_i64)
                .unwrap();
            target
        })
        .collect();
    let out = run_displacement(&cfg).unwrap();

    let disp_o: Array3<f64> = ndarray_npy::read_npy(dir.join("disp_displacement.npy")).unwrap();
    let vel_o: Array2<f64> = ndarray_npy::read_npy(dir.join("disp_velocity.npy")).unwrap();
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
    // Same physical tolerance as the default-path oracle test: auto-tile must not
    // regress parity on shipped-size scenes (here it derives to a single tile).
    assert!(derr < 1e-3, "auto-tile displacement error {derr}");
    assert!(verr < 1e-2, "auto-tile velocity error {verr}");
}

/// Measure the numeric deviation tiling introduces on a tiling-triggering grid:
/// single-tile `(1,1)` (the oracle-validated path) vs `(2,2)` tiled, on a smooth
/// ramp. This quantifies the seam/reconciliation error #2 adds. NOTE: a smooth
/// ramp is the *best case* for tiling; realistic noisy scenes deviate more and
/// have no large oracle fixture, which is why #2 ships opt-in/default-off.
#[test]
fn report_single_vs_tiled_deviation() {
    if !snaphu_available() {
        eprintln!("skipping tiling-deviation report: snaphu not on PATH");
        return;
    }
    let (rows, cols) = (1024, 768);
    let pl = Array3::from_shape_fn((2, rows, cols), |(t, r, c)| {
        Cf64::from_polar(1.0, t as f64 * (0.01 * r as f64 + 0.008 * c as f64))
    });
    let pairs = [(0_usize, 1_usize)];
    let corr = Array2::<f32>::from_elem((rows, cols), 1.0);

    let base = UnwrapConfig {
        cost: CostMode::Smooth,
        init: InitMethod::Mcf,
        ntiles: (1, 1),
        tile_overlap: (0, 0),
        nproc: 1,
        snaphu_path: "snaphu".to_string(),
    };
    let tiled = UnwrapConfig {
        ntiles: (2, 2),
        tile_overlap: (128, 128),
        nproc: 4,
        ..base.clone()
    };

    let run = |cfg: &UnwrapConfig, tag: &str| {
        let scratch = std::env::temp_dir().join(format!("dolphinrust_tiledev_{tag}"));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();
        SnaphuBackend(cfg.clone())
            .unwrap_network(pl.view(), &pairs, corr.view(), &scratch)
            .unwrap()
    };
    let single = run(&base, "single");
    let multi = run(&tiled, "tiled");
    let dev = single
        .iter()
        .zip(multi.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "TILING_DEVIATION grid={rows}x{cols} ntiles=(2,2) overlap=(128,128) max_abs_rad={dev:.6e}"
    );
    assert!(dev.is_finite(), "tiled deviation must be finite");
}
