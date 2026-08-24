//! Multi-burst stitching contract.
//!
//! Two synthetic CSLC bursts (IW1, IW2) tile a frame horizontally with a 4-pixel
//! overlap, each carrying an OPERA-style geotransform. `run_displacement` must
//! group them by burst, phase-link each, stitch onto the union frame grid, and
//! run the rest of the pipeline end to end — producing a frame-sized result with
//! the frame's CRS + geotransform through the default native unwrapper.

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_io::write_raster;
use dolphin_workflows::run_displacement;
use ndarray::{s, Array2};
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

fn write_mask(
    dir: &Path,
    iw: u8,
    col_off: usize,
    suffix: &str,
    values: &Array2<f32>,
    epsg: u32,
) -> PathBuf {
    let path = dir.join(format!("T064-135518-IW{iw}_{suffix}.tif"));
    write_raster(
        &path,
        values.view(),
        [
            FRAME_ORIGIN_X + col_off as f64 * DX,
            DX,
            0.0,
            ORIGIN_Y,
            0.0,
            -DX,
        ],
        Some(epsg),
        Some(255.0),
    )
    .unwrap();
    path
}

fn displacement_error(cfg: &DisplacementWorkflow) -> String {
    match run_displacement(cfg) {
        Err(error) => format!("{error:#}"),
        Ok(_) => panic!("expected displacement workflow to fail"),
    }
}

#[test]
fn equal_count_bursts_with_different_dates_fail_before_cslc_reads() {
    let dir = std::env::temp_dir().join(format!(
        "dolphin_multiburst_date_axis_contract_{}",
        std::process::id()
    ));
    let cfg = DisplacementWorkflow {
        cslc_file_list: vec![
            dir.join("OPERA_T064-135518-IW1_20221119.h5"),
            dir.join("OPERA_T064-135518-IW1_20221201.h5"),
            dir.join("OPERA_T064-135518-IW2_20221120.h5"),
            dir.join("OPERA_T064-135518-IW2_20221202.h5"),
        ],
        ..Default::default()
    };

    let detail = displacement_error(&cfg);
    assert!(
        detail.contains("bursts have different ordered acquisition dates"),
        "{detail}"
    );
    assert!(detail.contains("T064-135518-IW1"), "{detail}");
    assert!(detail.contains("T064-135518-IW2"), "{detail}");
}

#[test]
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn two_bursts_stitch_into_a_frame() {
    let dir = std::env::temp_dir().join(format!(
        "dolphin_multiburst_contract_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    let mut files = write_burst(&dir, 1, 0);
    files.extend(write_burst(&dir, 2, COL_OFF_IW2));

    // Both invalid pixels are outside the burst overlap, so list-order mapping
    // cannot be hidden by last-on-top stitching.
    let iw1_invalid = (8, 2);
    let iw2_invalid = (16, BCOLS - 2);
    let iw2_global = (iw2_invalid.0, COL_OFF_IW2 + iw2_invalid.1);
    let iw2_nonfinite = (4, BCOLS - 4);
    let iw2_nonfinite_global = (iw2_nonfinite.0, COL_OFF_IW2 + iw2_nonfinite.1);
    let mut iw1_values = Array2::from_elem((ROWS, BCOLS), 1.0_f32);
    iw1_values[iw1_invalid] = 0.0;
    iw1_values[(0, 0)] = -0.2;
    let mut iw2_values = Array2::from_elem((ROWS, BCOLS), 1.0_f32);
    iw2_values[iw2_invalid] = 0.0;
    iw2_values[iw2_nonfinite] = f32::INFINITY;
    let iw2_positive_fraction = (0, BCOLS - 1);
    let iw2_positive_fraction_global = (
        iw2_positive_fraction.0,
        COL_OFF_IW2 + iw2_positive_fraction.1,
    );
    iw2_values[iw2_positive_fraction] = 0.2;
    let iw1_mask = write_mask(&dir, 1, 0, "layover_shadow_mask", &iw1_values, EPSG as u32);
    let iw2_mask = write_mask(
        &dir,
        2,
        COL_OFF_IW2,
        "layover_shadow_mask",
        &iw2_values,
        EPSG as u32,
    );

    let mut cfg = DisplacementWorkflow {
        cslc_file_list: files,
        work_directory: dir.join("masked"),
        ..Default::default()
    };
    cfg.input_options.subdataset = Some("/data/VV".into());
    cfg.input_options.wavelength = Some(0.055_465_76);
    cfg.phase_linking.ministack_size = 15;
    cfg.phase_linking.half_window = HalfWindow { y: 2, x: 2 };
    cfg.phase_linking.calc_average_coh = true;
    cfg.output_options.strides = Strides { y: 1, x: 1 };
    cfg.interferogram_network.reference_idx = Some(0);
    cfg.layover_shadow_mask_files = vec![iw2_mask.clone(), iw1_mask.clone()];

    let all_valid = Array2::from_elem((ROWS, BCOLS), 1.0_f32);
    let wrong_crs_iw1 = write_mask(&dir, 1, 0, "wrong_crs_mask", &all_valid, EPSG as u32 + 1);
    let mut wrong_crs = cfg.clone();
    wrong_crs.layover_shadow_mask_files = vec![iw2_mask.clone(), wrong_crs_iw1.clone()];
    wrong_crs.work_directory = dir.join("wrong_crs");
    let detail = displacement_error(&wrong_crs);
    assert!(
        detail.contains(&wrong_crs_iw1.display().to_string()),
        "{detail}"
    );
    assert!(
        detail.contains("differs from target EPSG 32611"),
        "{detail}"
    );

    let all_invalid_iw1 = write_mask(
        &dir,
        1,
        0,
        "all_invalid_mask",
        &Array2::<f32>::zeros((ROWS, BCOLS)),
        EPSG as u32,
    );
    let mut fully_masked = cfg.clone();
    fully_masked.layover_shadow_mask_files = vec![iw2_mask.clone(), all_invalid_iw1.clone()];
    fully_masked.work_directory = dir.join("fully_masked");
    let detail = displacement_error(&fully_masked);
    assert!(
        detail.contains(&all_invalid_iw1.display().to_string()),
        "{detail}"
    );
    assert!(
        detail.contains("has no valid pixel in the processed burst window"),
        "{detail}"
    );
    assert!(
        !detail.contains("no tile with complete temporal support"),
        "{detail}"
    );

    let mut invalid_reference = cfg.clone();
    invalid_reference.timeseries_options.reference_point = Some(iw1_invalid);
    invalid_reference.work_directory = dir.join("invalid_reference");
    let detail = displacement_error(&invalid_reference);
    assert!(
        detail.contains(
            "timeseries_options.reference_point resolves to a layover/shadow-invalid pixel"
        ),
        "{detail}"
    );

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
    for point in [iw1_invalid, iw2_global, iw2_nonfinite_global] {
        assert!(!out.validity_mask[point]);
        assert!(out.temporal_coherence[point].is_nan());
        assert!(out.phase_linking_coherence.as_ref().unwrap()[point].is_nan());
        assert!(out.velocity[point].is_nan());
        assert!(out.velocity_mm_yr[point].is_nan());
        assert!(out
            .displacement
            .slice(s![.., point.0, point.1])
            .iter()
            .all(|value| value.is_nan()));
        assert!(out
            .crlb_sigma
            .as_ref()
            .unwrap()
            .slice(s![.., point.0, point.1])
            .iter()
            .all(|value| value.is_nan()));
        assert!(out
            .unwrap_connected_components
            .slice(s![.., point.0, point.1])
            .iter()
            .all(|value| *value == 0));
    }
    assert!(out.validity_mask[(0, 0)], "negative nonzero is valid");
    assert!(
        out.validity_mask[iw2_positive_fraction_global],
        "positive fractional nonzero is valid"
    );
    let reference = out.reference_point.expect("automatic reference");
    assert!(out.validity_mask[reference]);
    assert_eq!(
        out.validity_mask.iter().filter(|valid| !**valid).count(),
        3,
        "terrain invalidity must not spread through unwrapping"
    );
    for ((row, col), &valid) in out.validity_mask.indexed_iter() {
        assert_eq!(out.velocity_mm_yr[(row, col)].is_finite(), valid);
    }

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
    let stride_dir = std::env::temp_dir().join(format!(
        "dolphin_multiburst_bounds_3x6_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&stride_dir).unwrap();
    let mut stride_files = write_burst(&stride_dir, 1, 0);
    stride_files.extend(write_burst(&stride_dir, 2, 24));
    let mut bounded_3x6 = bounded_1x2;
    bounded_3x6.cslc_file_list = stride_files;
    bounded_3x6.layover_shadow_mask_files.clear();
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
