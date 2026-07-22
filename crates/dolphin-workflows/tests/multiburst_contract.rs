//! Multi-burst stitching contract.
//!
//! Two synthetic CSLC bursts (IW1, IW2) tile a frame horizontally with a 4-pixel
//! overlap, each carrying an OPERA-style geotransform. `run_displacement` must
//! group them by burst, phase-link each, stitch onto the union frame grid, and
//! run the rest of the pipeline end to end — producing a frame-sized result with
//! the frame's CRS + geotransform. Skips without `snaphu`.

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_workflows::run_displacement;
use ndarray::Array2;
use num_complex::Complex;

const ROWS: usize = 24;
const BCOLS: usize = 32; // per-burst columns
const COL_OFF_IW2: usize = 28; // IW2 starts here in the frame (4-px overlap)
const FRAME_COLS: usize = COL_OFF_IW2 + BCOLS; // 60
const N: usize = 5;
const DX: f64 = 30.0;
const FRAME_ORIGIN_X: f64 = 1000.0;
const ORIGIN_Y: f64 = 2000.0;
const EPSG: i64 = 32611;

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// Smooth frame-wide ramp (continuous across the burst seam) so unwrapping is
/// cycle-free; `frame_col` is the pixel's column in the stitched frame.
fn sample(t: usize, frame_col: usize) -> Complex<f32> {
    let phase = 0.3 * t as f64 * (frame_col as f64 / FRAME_COLS as f64);
    Complex::from_polar(1.0, phase as f32)
}

fn write_burst(dir: &Path, iw: u8, col_off: usize) -> Vec<PathBuf> {
    let corner_x = FRAME_ORIGIN_X + col_off as f64 * DX;
    let x_centers: Vec<f64> = (0..BCOLS)
        .map(|j| corner_x + DX / 2.0 + j as f64 * DX)
        .collect();
    let y_centers: Vec<f64> = (0..ROWS)
        .map(|i| ORIGIN_Y - DX / 2.0 - i as f64 * DX)
        .collect();
    let base = chrono::NaiveDate::from_ymd_opt(2022, 11, 19).unwrap();

    (0..N)
        .map(|t| {
            let stamp = (base + chrono::Duration::days(t as i64 * 12)).format("%Y%m%d");
            let path = dir.join(format!("OPERA_T064-135518-IW{iw}_{stamp}.h5"));
            let grid = Array2::from_shape_fn((ROWS, BCOLS), |(_, j)| sample(t, col_off + j));
            let f = hdf5::File::create(&path).unwrap();
            let g = f.create_group("data").unwrap();
            g.new_dataset_builder()
                .with_data(&grid)
                .create("VV")
                .unwrap();
            g.new_dataset_builder()
                .with_data(&x_centers)
                .create("x_coordinates")
                .unwrap();
            g.new_dataset_builder()
                .with_data(&y_centers)
                .create("y_coordinates")
                .unwrap();
            g.new_dataset::<i64>()
                .create("projection")
                .unwrap()
                .write_scalar(&EPSG)
                .unwrap();
            path
        })
        .collect()
}

#[test]
fn two_bursts_stitch_into_a_frame() {
    if !snaphu_available() {
        eprintln!("skipping multi-burst: snaphu not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("dolphin_multiburst_contract");
    std::fs::create_dir_all(&dir).unwrap();

    let mut files = write_burst(&dir, 1, 0);
    files.extend(write_burst(&dir, 2, COL_OFF_IW2));

    let mut cfg = DisplacementWorkflow {
        cslc_file_list: files,
        work_directory: dir.clone(),
        ..Default::default()
    };
    cfg.input_options.subdataset = Some("/data/VV".into());
    cfg.input_options.wavelength = Some(0.055_465_76);
    cfg.phase_linking.ministack_size = 15;
    cfg.phase_linking.half_window = HalfWindow { y: 2, x: 2 };
    cfg.phase_linking.calc_average_coh = true;
    cfg.output_options.strides = Strides { y: 1, x: 1 };
    cfg.interferogram_network.reference_idx = Some(0);

    let out = run_displacement(&cfg).unwrap();

    // Stitched onto the 24x60 union frame, not a single 24x32 burst.
    assert_eq!(
        out.displacement.dim(),
        (N - 1, ROWS, FRAME_COLS),
        "frame dims"
    );
    assert_eq!(out.temporal_coherence.dim(), (ROWS, FRAME_COLS));
    assert_eq!(
        out.phase_linking_coherence.as_ref().unwrap().dim(),
        (ROWS, FRAME_COLS)
    );
    assert_eq!(out.velocity_mm_yr.dim(), (ROWS, FRAME_COLS));
    assert_eq!(out.epsg, Some(EPSG as u32), "frame CRS");
    assert!(
        (out.geotransform[0] - FRAME_ORIGIN_X).abs() < 1e-6,
        "frame origin x"
    );
    assert!((out.geotransform[1] - DX).abs() < 1e-6, "frame dx");
    assert!(
        out.velocity_mm_yr.iter().all(|v| v.is_finite()),
        "velocity finite"
    );

    // Bounded target crossing the real burst seam at 1x2. The analysis halo
    // retains both bursts and their overlap; only the returned/written arrays
    // are trimmed.
    let mut bounded_1x2 = cfg.clone();
    bounded_1x2.output_options.strides = Strides { y: 1, x: 2 };
    bounded_1x2.output_options.bounds = Some((1_600.0, 1_400.0, 2_380.0, 1_880.0));
    bounded_1x2.output_options.bounds_epsg = Some(EPSG as u32);
    bounded_1x2.timeseries_options.reference_point = None;
    bounded_1x2.work_directory = dir.join("bounded_1x2");
    let cropped_1x2 = run_displacement(&bounded_1x2).unwrap();
    assert_eq!(cropped_1x2.temporal_coherence.dim(), (16, 13));
    assert!(cropped_1x2.geometry_provenance.processing_bounds.is_some());

    // A stride-aligned 3x6 two-burst fixture keeps one output-column of seam
    // overlap (8 pixels), above the explicit four-pixel leveling gate.
    let stride_dir = std::env::temp_dir().join("dolphin_multiburst_bounds_3x6");
    std::fs::create_dir_all(&stride_dir).unwrap();
    let mut stride_files = write_burst(&stride_dir, 1, 0);
    stride_files.extend(write_burst(&stride_dir, 2, 24));
    let mut bounded_3x6 = bounded_1x2;
    bounded_3x6.cslc_file_list = stride_files;
    bounded_3x6.output_options.strides = Strides { y: 3, x: 6 };
    bounded_3x6.output_options.bounds = Some((1_540.0, 1_370.0, 2_440.0, 1_820.0));
    bounded_3x6.work_directory = stride_dir.join("bounded");
    let cropped_3x6 = run_displacement(&bounded_3x6).unwrap();
    assert_eq!(cropped_3x6.temporal_coherence.dim(), (5, 5));
    assert!(cropped_3x6
        .velocity_mm_yr
        .iter()
        .all(|value| value.is_finite()));
}
