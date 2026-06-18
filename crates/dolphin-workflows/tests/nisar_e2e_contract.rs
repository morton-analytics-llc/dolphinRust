//! End-to-end NISAR / L-band contract (DoD #4).
//!
//! A multi-acquisition synthesized NISAR-layout stack (complex-`f32` `{r,i}`
//! compound grids in the NISAR product group, dated granule names, a smooth LOS
//! ramp) is run through `run_displacement` with `input_type = nisar_gslc` and the
//! NISAR L-band wavelength. The pipeline must read the NISAR grid, derive the
//! custom geotransform/EPSG, and produce a displacement product (typed output +
//! COG files on disk) on the correct grid. Proves the L-band wiring without a
//! real granule. Skips without `snaphu` on PATH.
//!
//! This product is geometrically correct but **atmospherically uncorrected**:
//! ionospheric/tropospheric corrections are a separate later loop.

use std::path::Path;

use dolphin_core::config::{DisplacementWorkflow, InputType};
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_core::Cf32;
use dolphin_io::nisar_fixture::{write_nisar_fixture, FREQUENCY_A_GROUP};
use dolphin_workflows::run_displacement;
use ndarray::Array2;

const ROWS: usize = 24;
const COLS: usize = 40;
const N: usize = 5;
const DX: f64 = 20.0; // NISAR posting (m)
const ORIGIN_X: f64 = 300_000.0;
const ORIGIN_Y: f64 = 4_100_000.0;
const EPSG: u32 = 32610;
const NISAR_WAVELENGTH_M: f64 = 0.238_403_545;

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// Smooth column-wise ramp, growing with acquisition index — cycle-free so
/// unwrapping is exact.
fn sample(t: usize, col: usize) -> Cf32 {
    let phase = 0.3 * t as f64 * (col as f64 / COLS as f64);
    Cf32::from_polar(1.0, phase as f32)
}

fn write_stack(dir: &Path) -> Vec<std::path::PathBuf> {
    let x: Vec<f64> = (0..COLS)
        .map(|j| ORIGIN_X + DX / 2.0 + j as f64 * DX)
        .collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| ORIGIN_Y - DX / 2.0 - i as f64 * DX)
        .collect();
    let base = chrono::NaiveDate::from_ymd_opt(2024, 6, 1).unwrap();
    (0..N)
        .map(|t| {
            let stamp = (base + chrono::Duration::days(t as i64 * 12)).format("%Y%m%d");
            let path = dir.join(format!("NISAR_L2_PR_GSLC_001_A_{stamp}T120000_F.h5"));
            let grid = Array2::from_shape_fn((ROWS, COLS), |(_, j)| sample(t, j));
            write_nisar_fixture(&path, "HH", grid.view(), &x, &y, EPSG).unwrap();
            path
        })
        .collect()
}

#[test]
fn nisar_stack_runs_end_to_end() {
    if !snaphu_available() {
        eprintln!("skipping nisar e2e: snaphu not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("dolphin_nisar_e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let files = write_stack(&dir);

    let mut cfg = DisplacementWorkflow {
        cslc_file_list: files,
        work_directory: dir.clone(),
        ..Default::default()
    };
    cfg.input_options.input_type = InputType::NisarGslc;
    cfg.input_options.subdataset = Some(format!("{FREQUENCY_A_GROUP}/HH"));
    cfg.input_options.wavelength = Some(NISAR_WAVELENGTH_M);
    cfg.phase_linking.ministack_size = 15;
    cfg.phase_linking.half_window = HalfWindow { y: 2, x: 2 };
    cfg.output_options.strides = Strides { y: 1, x: 1 };
    cfg.interferogram_network.reference_idx = Some(0);

    let out = run_displacement(&cfg).unwrap();

    // Typed output on the NISAR grid, with the fixture's CRS + geotransform.
    assert_eq!(out.displacement.dim(), (N - 1, ROWS, COLS), "grid dims");
    assert_eq!(out.velocity_mm_yr.dim(), (ROWS, COLS));
    assert_eq!(out.epsg, Some(EPSG), "NISAR EPSG from projection.epsg_code");
    assert!((out.geotransform[0] - ORIGIN_X).abs() < 1e-6, "origin x");
    assert!((out.geotransform[1] - DX).abs() < 1e-6, "dx");
    assert!((out.geotransform[5] + DX).abs() < 1e-6, "dy");
    assert!(
        out.velocity_mm_yr.iter().all(|v| v.is_finite()),
        "vel finite"
    );

    // COGs written to disk.
    assert!(dir.join("velocity.tif").exists(), "velocity COG");
    assert!(dir.join("temporal_coherence.tif").exists(), "coherence COG");
    assert!(dir.join("displacement_00.tif").exists(), "displacement COG");
}
