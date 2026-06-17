//! Runnable end-to-end example: synthesize a small CSLC stack and run the
//! displacement pipeline in one command.
//!
//!     cargo run --release --example run_synthetic
//!
//! Requires the system dependencies the pipeline uses: GDAL, HDF5, and the
//! `snaphu` binary on `PATH` (see README "System requirements"). Writes a
//! synthetic 5-acquisition stack to a temp directory, runs `run_displacement`,
//! and prints a summary of the typed result + the COGs written.

use std::path::PathBuf;

use anyhow::Result;
use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_workflows::run_displacement;
use ndarray::Array2;
use num_complex::Complex;

const N: usize = 5;
const ROWS: usize = 32;
const COLS: usize = 48;
const DT_DAYS: i64 = 12;
const SENTINEL1_WAVELENGTH_M: f64 = 0.055_465_76;

fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("dolphin_run_synthetic");
    std::fs::create_dir_all(&dir)?;
    let files = write_stack(&dir)?;
    let cfg = build_config(&dir, files);

    let out = run_displacement(&cfg)?;

    let (nd, r, c) = out.displacement.dim();
    let vmax = out
        .velocity_mm_yr
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    println!("ran displacement on {N} synthetic acquisitions ({ROWS}x{COLS})");
    println!("  displacement cube: {nd} dates x {r}x{c}");
    println!("  velocity: max |rate| = {vmax:.3} mm/yr");
    println!(
        "  temporal coherence: mean = {:.4}",
        out.temporal_coherence.mean().unwrap_or(0.0)
    );
    println!(
        "  epsg = {:?}, geotransform = {:?}",
        out.epsg, out.geotransform
    );
    println!("  COGs written to {}", cfg.work_directory.display());
    Ok(())
}

/// Smooth range ramp growing linearly in time, plus light speckle — small enough
/// that unwrapping is cycle-free (mirrors `validation/gen_stack.py`).
fn synth_slc(t: usize) -> Array2<Complex<f32>> {
    Array2::from_shape_fn((ROWS, COLS), |(_, x)| {
        let phase = 0.3 * t as f64 * (x as f64 / COLS as f64);
        let speckle = 0.02 * ((t * 7 + x) % 5) as f64;
        Complex::from_polar(1.0, (phase + speckle) as f32)
    })
}

/// Write the N date-named CSLC HDF5 files at `/data/VV`; returns their paths.
fn write_stack(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let base = chrono::NaiveDate::from_ymd_opt(2022, 11, 19).unwrap();
    let mut files = Vec::new();
    for t in 0..N {
        let stamp = (base + chrono::Duration::days(t as i64 * DT_DAYS)).format("%Y%m%d");
        let path = dir.join(format!("cslc_{stamp}.h5"));
        let file = hdf5::File::create(&path)?;
        let group = file.create_group("data")?;
        group
            .new_dataset_builder()
            .with_data(&synth_slc(t))
            .create("VV")?;
        files.push(path);
    }
    Ok(files)
}

/// Minimal config: single-reference network, S1 wavelength so velocity is mm/yr.
fn build_config(dir: &std::path::Path, files: Vec<PathBuf>) -> DisplacementWorkflow {
    let mut cfg = DisplacementWorkflow {
        cslc_file_list: files,
        work_directory: dir.to_path_buf(),
        ..Default::default()
    };
    cfg.input_options.subdataset = Some("/data/VV".into());
    cfg.input_options.wavelength = Some(SENTINEL1_WAVELENGTH_M);
    cfg.phase_linking.ministack_size = 15;
    cfg.phase_linking.half_window = HalfWindow { y: 2, x: 2 };
    cfg.output_options.strides = Strides { y: 1, x: 1 };
    cfg.interferogram_network.reference_idx = Some(0);
    cfg
}
