//! Contract for `dolphin pairs` coherence mode.
//!
//! - On the real cropped Mexico City OPERA CSLC stack, each pair COG equals `|Γ_ij|`
//!   from the full-stack `estimate_stack_covariance` to max |Δ| < 1e-6. Skips when
//!   the local real fixture is absent, like the other real-data gates.
//! - A 12-day series with a 24-day gap forms no pair across the gap.
//! - Every pair COG carries the six pair tags.
//! - Interferogram mode on a synthetic two-band NISAR pair: a planted ionosphere screen
//!   is removed from the main-band phase to < 0.01 rad RMS; `corrected − (unwrapped −
//!   ionosphere)` is exactly 0; all five layers, `los_enu.tif` and the T7 tags are written;
//!   a product without `frequencyB` is refused with a typed error; `pairs.json` records the
//!   side-band cycle alignment for every connected component in the written layer.
//! - In both modes a NaN input pixel is NaN in every output layer, and every COG has GDAL
//!   nodata NaN.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use dolphin_core::config::InputType;
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_core::{Cf32, Cf64};
use dolphin_io::nisar_fixture::{
    write_nisar_dual_band_fixture, write_nisar_fixture, FixtureBand, FREQUENCY_A_GROUP,
};
use dolphin_io::{read_cslc, read_raster, read_raster_header, write_raster};
use dolphin_phaselink::covariance::estimate_stack_covariance;
use dolphin_workflows::pairs::{PairsError, LOS_ENU_FILENAME};
use dolphin_workflows::{
    plan_pairs, run_pairs, PairAcquisition, PairPlan, PairsConfig, PairsIndex, PairsMode,
    PAIRS_INDEX_FILENAME,
};
use ndarray::{Array2, Array3, Axis};

const MEXICO: &str = "../../validation/real_data/cropped_mexico";
const SUBDATASET: &str = "/data/VV";

/// HDF5 is not thread-safe in this build; every test that touches it serializes here.
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn hdf5_guard() -> MutexGuard<'static, ()> {
    HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn twelve_day_plan() -> PairPlan {
    PairPlan {
        pair_class_days: vec![12.0],
        tolerance_days: 1.0,
        pairs: None,
    }
}

fn config(dir: &Path, acquisitions: Vec<PairAcquisition>) -> PairsConfig {
    PairsConfig {
        mode: PairsMode::Coherence,
        input_type: InputType::OperaCslc,
        subdataset: SUBDATASET.into(),
        work_directory: dir.to_path_buf(),
        acquisitions,
        pair_plan: twelve_day_plan(),
        half_window: HalfWindow { y: 2, x: 4 },
        strides: Strides { y: 2, x: 3 },
        bounds: None,
        nisar_geometry_group: None,
        nisar_ellipsoidal_dem_file: None,
        ionosphere_lowpass_wavelength_m: None,
    }
}

/// OPERA granule start time, e.g. `..._20230104T004053Z_...`.
fn opera_start_utc(stem: &str) -> DateTime<Utc> {
    let stamp = stem.split('_').nth(4).unwrap();
    let naive = chrono::NaiveDateTime::parse_from_str(stamp, "%Y%m%dT%H%M%SZ").unwrap();
    Utc.from_utc_datetime(&naive)
}

fn mexico_acquisitions(dir: &Path) -> Vec<PairAcquisition> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "h5"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let stem = path.file_stem().unwrap().to_str().unwrap().to_owned();
            PairAcquisition {
                acquisition_utc: opera_start_utc(&stem),
                acquisition_id: stem,
                path,
            }
        })
        .collect()
}

fn full_stack(acquisitions: &[PairAcquisition]) -> Array3<Cf64> {
    let layers: Vec<Array2<Cf64>> = acquisitions
        .iter()
        .map(|a| {
            read_cslc(&a.path, SUBDATASET)
                .unwrap()
                .mapv(|z| Cf64::new(f64::from(z.re), f64::from(z.im)))
        })
        .collect();
    let views: Vec<_> = layers.iter().map(Array2::view).collect();
    ndarray::stack(Axis(0), &views).unwrap()
}

fn find(acquisitions: &[PairAcquisition], id: &str) -> usize {
    acquisitions
        .iter()
        .position(|a| a.acquisition_id == id)
        .unwrap()
}

#[test]
fn real_pair_coherence_equals_normalized_covariance_off_diagonal() {
    let _hdf5 = hdf5_guard();
    let data = Path::new(MEXICO);
    if !data.exists() {
        eprintln!("real fixture absent; skipping pair coherence parity gate");
        return;
    }
    let dir = std::env::temp_dir().join("dolphin_pairs_mexico");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = config(&dir, mexico_acquisitions(data));
    let index = run_pairs(&cfg).unwrap();
    // 13 dates, a 24-day first interval, then 12-day spacing: 11 twelve-day pairs.
    assert_eq!(index.pairs.len(), 11);

    let stack = full_stack(&cfg.acquisitions);
    let covariance =
        estimate_stack_covariance(stack.view(), cfg.half_window, cfg.strides, None).unwrap();
    let mut max_delta = 0.0_f64;
    for record in &index.pairs {
        let i = find(&cfg.acquisitions, &record.reference_acquisition_id);
        let j = find(&cfg.acquisitions, &record.secondary_acquisition_id);
        let written = read_raster::<f32>(&dir.join(&record.files[0].file)).unwrap();
        let expected = covariance.slice(ndarray::s![.., .., i, j]).mapv(Cf64::norm);
        assert_eq!(written.data.dim(), expected.dim());
        let delta = written
            .data
            .iter()
            .zip(expected.iter())
            .map(|(&w, &e)| (f64::from(w) - e).abs())
            .fold(0.0, f64::max);
        max_delta = max_delta.max(delta);
    }
    eprintln!("pair coherence vs covariance off-diagonal: max |delta| = {max_delta:.3e}");
    assert!(max_delta < 1e-6, "max |delta| {max_delta} >= 1e-6");
}

fn synthetic_acquisitions(dir: &Path, offsets_days: &[i64]) -> Vec<PairAcquisition> {
    synthetic_acquisitions_with_nan(dir, offsets_days, None)
}

/// As [`synthetic_acquisitions`], with `nan` set to NaN in the second acquisition.
fn synthetic_acquisitions_with_nan(
    dir: &Path,
    offsets_days: &[i64],
    nan: Option<(usize, usize)>,
) -> Vec<PairAcquisition> {
    let start = NaiveDate::from_ymd_opt(2023, 6, 8)
        .unwrap()
        .and_hms_opt(13, 30, 0)
        .unwrap();
    offsets_days
        .iter()
        .enumerate()
        .map(|(t, &days)| {
            let utc = Utc.from_utc_datetime(&(start + Duration::days(days)));
            let id = format!("SYNTHETIC_CSLC_{}", utc.format("%Y%m%dT%H%M%SZ"));
            let path = dir.join(format!("{id}.h5"));
            write_cslc(&path, t, nan.filter(|_| t == 1));
            PairAcquisition {
                path,
                acquisition_id: id,
                acquisition_utc: utc,
            }
        })
        .collect()
}

/// Small OPERA-layout CSLC: `/data/VV` plus coordinates and projection.
fn write_cslc(path: &Path, seed: usize, nan: Option<(usize, usize)>) {
    let (rows, cols) = (16, 24);
    let mut grid = Array2::from_shape_fn((rows, cols), |(r, c)| {
        Cf32::from_polar(1.0, (0.1 * (seed * (r + 2 * c)) as f64) as f32)
    });
    if let Some(pixel) = nan {
        grid[pixel] = Cf32::new(f32::NAN, f32::NAN);
    }
    let file = hdf5::File::create(path).unwrap();
    let group = file.create_group("data").unwrap();
    group
        .new_dataset_builder()
        .with_data(&grid)
        .create("VV")
        .unwrap();
    let x: Vec<f64> = (0..cols).map(|c| 500_015.0 + c as f64 * 30.0).collect();
    let y: Vec<f64> = (0..rows).map(|r| 4_200_015.0 - r as f64 * 30.0).collect();
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
        .write_scalar(&32612_i64)
        .unwrap();
}

#[test]
fn twelve_day_series_with_gap_forms_no_pair_across_gap() {
    let _hdf5 = hdf5_guard();
    let dir = std::env::temp_dir().join("dolphin_pairs_gap");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // 0, 12, 24 | 24-day gap | 48, 60
    let acquisitions = synthetic_acquisitions(&dir, &[0, 12, 24, 48, 60]);
    let planned = plan_pairs(&acquisitions, &twelve_day_plan()).unwrap();
    let as_indices: Vec<(usize, usize)> =
        planned.iter().map(|p| (p.reference, p.secondary)).collect();
    assert_eq!(as_indices, vec![(0, 1), (1, 2), (3, 4)]);

    let index = run_pairs(&config(&dir.join("out"), acquisitions)).unwrap();
    let dates: Vec<(&str, &str)> = index
        .pairs
        .iter()
        .map(|p| (p.reference_date.as_str(), p.secondary_date.as_str()))
        .collect();
    assert!(!dates.contains(&("2023-06-08", "2023-07-26")));
    assert!(!dates.contains(&("2023-07-02", "2023-07-26")));
    assert!(!dir.join("out/coherence_20230702_20230726.tif").exists());
    assert_eq!(index.pairs.len(), 3);
}

#[test]
fn every_pair_cog_carries_the_six_pair_tags() {
    let _hdf5 = hdf5_guard();
    let dir = std::env::temp_dir().join("dolphin_pairs_tags");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let acquisitions = synthetic_acquisitions(&dir, &[0, 12]);
    let out = dir.join("out");
    run_pairs(&config(&out, acquisitions)).unwrap();
    let index: PairsIndex =
        serde_json::from_str(&std::fs::read_to_string(out.join(PAIRS_INDEX_FILENAME)).unwrap())
            .unwrap();
    let record = &index.pairs[0];
    let file = &record.files[0];
    assert_eq!(file.file, "coherence_20230608_20230620.tif");
    let header = read_raster_header(&out.join(&file.file)).unwrap();
    let tag = |key: &str| header.metadata.get(key).map(String::as_str);
    assert_eq!(
        tag("REFERENCE_ACQUISITION_ID"),
        Some("SYNTHETIC_CSLC_20230608T133000Z")
    );
    assert_eq!(
        tag("SECONDARY_ACQUISITION_ID"),
        Some("SYNTHETIC_CSLC_20230620T133000Z")
    );
    assert_eq!(tag("REFERENCE_DATE"), Some("2023-06-08"));
    assert_eq!(tag("SECONDARY_DATE"), Some("2023-06-20"));
    assert_eq!(tag("TEMPORAL_BASELINE_DAYS"), Some("12"));
    assert_eq!(tag("LAYER"), Some("coherence"));
    assert_eq!(file.sha256.len(), 64);
}

// ---------------------------------------------------------------------------
// Interferogram mode (NISAR two-band split-spectrum)
// ---------------------------------------------------------------------------

const MAIN_HZ: f64 = 1.2575e9;
const SIDE_HZ: f64 = 1.2935e9;
const SIDE_SPACING_M: f64 = 40.0;
const MAIN_SPACING_M: f64 = 20.0;
const ORIGIN_X: f64 = 500_000.0;
const ORIGIN_Y: f64 = 4_200_000.0;
const EPSG: u32 = 32612;
const RADAR_GRID: &str = "/science/LSAR/GSLC/metadata/radarGrid";
const LOS_EAST: f64 = 0.62;
const LOS_NORTH: f64 = -0.11;
const REFERENCE_ID: &str =
    "NISAR_L2_PR_GSLC_001_001_A_001_2000_SHSH_A_20251201T120000_20251201T120030_X05010_N_F_J_001";
const SECONDARY_ID: &str =
    "NISAR_L2_PR_GSLC_002_001_A_001_2000_SHSH_A_20251213T120000_20251213T120030_X05011_N_F_J_001";

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// Planted ionosphere in the main-band interferogram (radians), periodic on the
/// `side_n`-pixel frequencyB grid so the FFT low-pass passes it essentially intact.
fn planted_ionosphere(u: f64, v: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    1.5 * (tau * u).sin() * (tau * v).cos() + 0.5
}

/// Non-dispersive main-band phase (radians): a bowl spanning several cycles.
fn planted_non_dispersive(u: f64, v: f64) -> f64 {
    6.0 * std::f64::consts::PI * (u * u + v * v)
}

/// Interferometric phase of a band at `frequency_hz`: `(f/f_A)·d + (f_A/f)·I`.
fn band_phase(frequency_hz: f64, u: f64, v: f64) -> f64 {
    frequency_hz / MAIN_HZ * planted_non_dispersive(u, v)
        + MAIN_HZ / frequency_hz * planted_ionosphere(u, v)
}

/// Normalized grid coordinates of a pixel center at `spacing` on a `side_n` B grid.
fn normalized(row: usize, col: usize, spacing: f64, side_n: usize) -> (f64, f64) {
    let extent = side_n as f64 * SIDE_SPACING_M;
    (
        (col as f64 + 0.5) * spacing / extent,
        (row as f64 + 0.5) * spacing / extent,
    )
}

/// One band's complex grid: unit amplitude, the reference at zero phase and the
/// secondary at `−φ`, so `ref · conj(sec) = e^{iφ}`. `nan` pixels are NaN.
fn band_grid(
    secondary: bool,
    frequency_hz: f64,
    spacing: f64,
    side_n: usize,
    nan: &[(usize, usize)],
) -> Array2<Cf32> {
    let n = (side_n as f64 * SIDE_SPACING_M / spacing) as usize;
    let mut grid = Array2::from_shape_fn((n, n), |(row, col)| {
        let (u, v) = normalized(row, col, spacing, side_n);
        let phase = if secondary {
            -band_phase(frequency_hz, u, v)
        } else {
            0.0
        };
        Cf32::from_polar(1.0, phase as f32)
    });
    nan.iter()
        .for_each(|&pixel| grid[pixel] = Cf32::new(f32::NAN, f32::NAN));
    grid
}

fn centers(origin: f64, step: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| origin + (i as f64 + 0.5) * step).collect()
}

/// NaN pixels planted in the secondary, per band, as `(main, side)` pixel lists.
type NanPixels<'a> = (&'a [(usize, usize)], &'a [(usize, usize)]);

/// Write one two-band NISAR GSLC with a radarGrid LOS cube.
fn write_dual_band_acquisition(path: &Path, secondary: bool, side_n: usize, nan: NanPixels) {
    let main_n = side_n * 2;
    let main_cpx = band_grid(secondary, MAIN_HZ, MAIN_SPACING_M, side_n, nan.0);
    let side_cpx = band_grid(secondary, SIDE_HZ, SIDE_SPACING_M, side_n, nan.1);
    let (main_x, main_y) = (
        centers(ORIGIN_X, MAIN_SPACING_M, main_n),
        centers(ORIGIN_Y, -MAIN_SPACING_M, main_n),
    );
    let (side_x, side_y) = (
        centers(ORIGIN_X, SIDE_SPACING_M, side_n),
        centers(ORIGIN_Y, -SIDE_SPACING_M, side_n),
    );
    let main = FixtureBand {
        cpx: main_cpx.view(),
        x: &main_x,
        y: &main_y,
        center_frequency_hz: MAIN_HZ,
        range_bandwidth_hz: 40e6,
    };
    let side = FixtureBand {
        cpx: side_cpx.view(),
        x: &side_x,
        y: &side_y,
        center_frequency_hz: SIDE_HZ,
        range_bandwidth_hz: 5e6,
    };
    write_nisar_dual_band_fixture(path, "HH", &main, &side, EPSG).unwrap();
    write_radar_grid(path, side_n);
}

/// Coarse constant LOS cube (400 m posting, two heights) covering the B grid.
fn write_radar_grid(path: &Path, side_n: usize) {
    let step = 400.0;
    let n = (side_n as f64 * SIDE_SPACING_M / step) as usize + 2;
    let x: Vec<f64> = (0..n).map(|i| ORIGIN_X + i as f64 * step).collect();
    let y: Vec<f64> = (0..n).map(|i| ORIGIN_Y - i as f64 * step).collect();
    let file = hdf5::File::append(path).unwrap();
    let group = file.create_group(RADAR_GRID).unwrap();
    group
        .new_dataset_builder()
        .with_data(&x)
        .create("xCoordinates")
        .unwrap();
    group
        .new_dataset_builder()
        .with_data(&y)
        .create("yCoordinates")
        .unwrap();
    group
        .new_dataset_builder()
        .with_data(&[-500.0_f64, 5000.0])
        .create("heightAboveEllipsoid")
        .unwrap();
    let projection = group.new_dataset::<i64>().create("projection").unwrap();
    projection.write_scalar(&i64::from(EPSG)).unwrap();
    projection
        .new_attr::<i64>()
        .create("epsg_code")
        .unwrap()
        .write_scalar(&i64::from(EPSG))
        .unwrap();
    for (name, value) in [("losUnitVectorX", LOS_EAST), ("losUnitVectorY", LOS_NORTH)] {
        group
            .new_dataset_builder()
            .with_data(Array3::from_elem((2, n, n), value).view())
            .create(name)
            .unwrap();
    }
}

fn utc(stamp: &str) -> DateTime<Utc> {
    let naive = chrono::NaiveDateTime::parse_from_str(stamp, "%Y%m%dT%H%M%S").unwrap();
    Utc.from_utc_datetime(&naive)
}

/// Two-band reference/secondary pair (12 days) plus an ellipsoidal DEM on the B grid.
fn nisar_pair(dir: &Path, side_n: usize, nan: NanPixels) -> PairsConfig {
    std::fs::create_dir_all(dir).unwrap();
    let acquisitions: Vec<PairAcquisition> = [
        (REFERENCE_ID, "20251201T120000", false),
        (SECONDARY_ID, "20251213T120000", true),
    ]
    .into_iter()
    .map(|(id, stamp, secondary)| {
        let path = dir.join(format!("{id}.h5"));
        let planted = if secondary { nan } else { (&[][..], &[][..]) };
        write_dual_band_acquisition(&path, secondary, side_n, planted);
        PairAcquisition {
            path,
            acquisition_id: id.to_owned(),
            acquisition_utc: utc(stamp),
        }
    })
    .collect();
    let dem = dir.join("ellipsoidal_dem.tif");
    write_raster(
        &dem,
        Array2::from_elem((side_n, side_n), 1500.0_f64).view(),
        [
            ORIGIN_X,
            SIDE_SPACING_M,
            0.0,
            ORIGIN_Y,
            0.0,
            -SIDE_SPACING_M,
        ],
        Some(EPSG),
        None,
    )
    .unwrap();
    PairsConfig {
        mode: PairsMode::Interferogram,
        input_type: InputType::NisarGslc,
        subdataset: format!("{FREQUENCY_A_GROUP}/HH"),
        work_directory: dir.join("out"),
        acquisitions,
        pair_plan: twelve_day_plan(),
        half_window: HalfWindow { y: 2, x: 2 },
        strides: Strides { y: 2, x: 2 },
        bounds: None,
        nisar_geometry_group: Some(RADAR_GRID.into()),
        nisar_ellipsoidal_dem_file: Some(dem),
        ionosphere_lowpass_wavelength_m: Some(400.0),
    }
}

fn layer(out: &Path, name: &str) -> Array2<f32> {
    read_raster::<f32>(&out.join(format!("{name}_20251201_20251213.tif")))
        .unwrap()
        .data
}

const LAYERS: [&str; 5] = [
    "unwrapped",
    "ionosphere",
    "corrected",
    "coherence",
    "connected_components",
];

#[test]
fn planted_ionosphere_screen_is_removed_to_under_a_hundredth_radian() {
    let _hdf5 = hdf5_guard();
    if !snaphu_available() {
        eprintln!("snaphu absent; skipping split-spectrum ionosphere removal gate");
        return;
    }
    let side_n = 256;
    let dir = std::env::temp_dir().join("dolphin_pairs_ionosphere");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = nisar_pair(&dir, side_n, (&[], &[]));
    run_pairs(&cfg).unwrap();

    let (unwrapped, corrected) = (
        layer(&cfg.work_directory, "unwrapped"),
        layer(&cfg.work_directory, "corrected"),
    );
    assert_eq!(corrected.dim(), (side_n, side_n));
    let rms_against_non_dispersive = |phase: &Array2<f32>| {
        let sum: f64 = phase
            .indexed_iter()
            .map(|((row, col), &value)| {
                let (u, v) = normalized(row, col, SIDE_SPACING_M, side_n);
                (f64::from(value) - planted_non_dispersive(u, v)).powi(2)
            })
            .sum();
        (sum / phase.len() as f64).sqrt()
    };
    let before = rms_against_non_dispersive(&unwrapped);
    let after = rms_against_non_dispersive(&corrected);
    eprintln!("ionosphere residual RMS: before {before:.4} rad, after {after:.3e} rad");
    assert!(before > 0.5, "planted screen not present: {before}");
    assert!(after < 0.01, "residual ionosphere RMS {after} >= 0.01 rad");
}

#[test]
fn corrected_is_exactly_unwrapped_minus_ionosphere_and_carries_t7_tags() {
    let _hdf5 = hdf5_guard();
    if !snaphu_available() {
        eprintln!("snaphu absent; skipping interferogram layer contract");
        return;
    }
    let dir = std::env::temp_dir().join("dolphin_pairs_interferogram_layers");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = nisar_pair(&dir, 32, (&[], &[]));
    let index = run_pairs(&cfg).unwrap();
    let out = &cfg.work_directory;

    let (unwrapped, ionosphere, corrected) = (
        layer(out, "unwrapped"),
        layer(out, "ionosphere"),
        layer(out, "corrected"),
    );
    let max_delta = ndarray::Zip::from(&corrected)
        .and(&unwrapped)
        .and(&ionosphere)
        .fold(0.0_f32, |acc, &c, &u, &i| acc.max((c - (u - i)).abs()));
    assert_eq!(max_delta, 0.0);

    let record = &index.pairs[0];
    let written: Vec<&str> = record.files.iter().map(|f| f.layer.as_str()).collect();
    assert_eq!(written, LAYERS);
    let summary = record.interferogram.as_ref().unwrap();
    let coherence = layer(out, "coherence");
    let mut sorted: Vec<f32> = coherence.iter().copied().collect();
    sorted.sort_by(f32::total_cmp);
    let median = f64::from(sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0;
    let above = sorted.iter().filter(|&&c| c >= 0.2).count() as f64 / sorted.len() as f64;
    assert!((summary.median_coherence.unwrap() - median).abs() < 1e-12);
    assert!((summary.coherence_fraction_at_least_0_2.unwrap() - above).abs() < 1e-12);
    assert!(summary.ionosphere_std_rad.unwrap() > 0.1);

    let header = read_raster_header(&out.join(&record.files[2].file)).unwrap();
    let tag = |key: &str| header.metadata.get(key).cloned();
    assert_eq!(tag("LAYER").as_deref(), Some("corrected"));
    assert_eq!(tag("REFERENCE_CRID").as_deref(), Some("X05010"));
    assert_eq!(tag("SECONDARY_CRID").as_deref(), Some("X05011"));
    assert_eq!(tag("CENTER_FREQUENCY_A_HZ").as_deref(), Some("1257500000"));
    assert_eq!(tag("CENTER_FREQUENCY_B_HZ").as_deref(), Some("1293500000"));
    let wavelength: f64 = tag("WAVELENGTH_M").unwrap().parse().unwrap();
    assert!((wavelength - 299_792_458.0 / MAIN_HZ).abs() < 1e-12);
    assert_eq!(
        tag("IONOSPHERE_METHOD").as_deref(),
        Some("split_spectrum_main_side_band")
    );
    assert_eq!(
        tag("PHASE_SIGN").as_deref(),
        Some("reference_times_conj_secondary")
    );
    assert_eq!(tag("TEMPORAL_BASELINE_DAYS").as_deref(), Some("12"));

    let los = index.los_enu.as_ref().unwrap();
    assert_eq!(los.file, LOS_ENU_FILENAME);
    let dataset = gdal::Dataset::open(out.join(LOS_ENU_FILENAME)).unwrap();
    assert_eq!(dataset.raster_count(), 3);
    let band = |number: usize| -> f64 {
        let values = dataset
            .rasterband(number)
            .unwrap()
            .read_as::<f32>((0, 0), (32, 32), (32, 32), None)
            .unwrap();
        f64::from(values.data()[0])
    };
    let up = (1.0 - LOS_EAST * LOS_EAST - LOS_NORTH * LOS_NORTH).sqrt();
    assert!((band(1) - LOS_EAST).abs() < 1e-6);
    assert!((band(2) - LOS_NORTH).abs() < 1e-6);
    assert!((band(3) - up).abs() < 1e-6);
}

#[test]
fn product_without_frequency_b_is_refused_with_a_typed_error() {
    let _hdf5 = hdf5_guard();
    let dir = std::env::temp_dir().join("dolphin_pairs_missing_side_band");
    let _ = std::fs::remove_dir_all(&dir);
    let mut cfg = nisar_pair(&dir, 16, (&[], &[]));
    let main_x = centers(ORIGIN_X, MAIN_SPACING_M, 32);
    let main_y = centers(ORIGIN_Y, -MAIN_SPACING_M, 32);
    let single_band = band_grid(true, MAIN_HZ, MAIN_SPACING_M, 16, &[]);
    let path = dir.join("single_band.h5");
    write_nisar_fixture(&path, "HH", single_band.view(), &main_x, &main_y, EPSG).unwrap();
    cfg.acquisitions[1].path = path;

    let error = run_pairs(&cfg).unwrap_err();
    match error.downcast_ref::<PairsError>() {
        Some(PairsError::MissingFrequencyB {
            acquisition_id,
            subdataset,
        }) => {
            assert_eq!(acquisition_id, SECONDARY_ID);
            assert_eq!(subdataset, "/science/LSAR/GSLC/grids/frequencyB/HH");
        }
        None => panic!("expected PairsError::MissingFrequencyB, got {error:#}"),
    }
    assert!(!cfg.work_directory.exists());
}

/// NaN pixels in a written layer, as `(row, col)`.
fn nan_pixels(data: &Array2<f32>) -> Vec<(usize, usize)> {
    data.indexed_iter()
        .filter(|(_, value)| value.is_nan())
        .map(|(pixel, _)| pixel)
        .collect()
}

#[test]
fn input_nan_is_nan_in_every_output_layer_with_nan_nodata() {
    let _hdf5 = hdf5_guard();
    // Coherence mode: strides (2, 3), so input (5, 7) lands in output (2, 2).
    let dir = std::env::temp_dir().join("dolphin_pairs_nan_coherence");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let acquisitions = synthetic_acquisitions_with_nan(&dir, &[0, 12], Some((5, 7)));
    let out = dir.join("out");
    run_pairs(&config(&out, acquisitions)).unwrap();
    let path = out.join("coherence_20230608_20230620.tif");
    assert_eq!(
        nan_pixels(&read_raster::<f32>(&path).unwrap().data),
        [(2, 2)]
    );
    assert!(read_raster_header(&path).unwrap().nodata.unwrap().is_nan());

    if !snaphu_available() {
        eprintln!("snaphu absent; skipping interferogram-mode nodata check");
        return;
    }
    // Interferogram mode: main-band (10, 13) multilooks into B (5, 6); B (20, 3) is
    // NaN in the side band only.
    let dir = std::env::temp_dir().join("dolphin_pairs_nan_interferogram");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = nisar_pair(&dir, 32, (&[(10, 13)], &[(20, 3)]));
    let index = run_pairs(&cfg).unwrap();
    for file in &index.pairs[0].files {
        let path = cfg.work_directory.join(&file.file);
        let data = read_raster::<f32>(&path).unwrap().data;
        assert_eq!(nan_pixels(&data), [(5, 6), (20, 3)], "{}", file.layer);
        assert!(read_raster_header(&path).unwrap().nodata.unwrap().is_nan());
    }
    let los = read_raster_header(&cfg.work_directory.join(LOS_ENU_FILENAME)).unwrap();
    assert!(los.nodata.unwrap().is_nan());
}

#[test]
fn pairs_json_reports_side_band_cycle_alignment_per_connected_component() {
    let _hdf5 = hdf5_guard();
    if !snaphu_available() {
        eprintln!("snaphu absent; skipping side-band cycle alignment record check");
        return;
    }
    // A full-height nodata column splits the scene into two connected components.
    let side_n = 64;
    let side_stripe: Vec<(usize, usize)> = (0..side_n).map(|row| (row, side_n / 2)).collect();
    let main_stripe: Vec<(usize, usize)> = (0..2 * side_n)
        .flat_map(|row| [(row, side_n), (row, side_n + 1)])
        .collect();
    let dir = std::env::temp_dir().join("dolphin_pairs_cycle_alignment");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = nisar_pair(&dir, side_n, (&main_stripe, &side_stripe));
    run_pairs(&cfg).unwrap();

    let json = std::fs::read_to_string(cfg.work_directory.join(PAIRS_INDEX_FILENAME)).unwrap();
    let index: PairsIndex = serde_json::from_str(&json).unwrap();
    let alignment = &index.pairs[0]
        .interferogram
        .as_ref()
        .unwrap()
        .side_band_cycle_alignment;
    eprintln!("side-band cycle alignment: {alignment:?}");
    let components = layer(&cfg.work_directory, "connected_components");
    let mut expected = std::collections::BTreeMap::<u32, usize>::new();
    components
        .iter()
        .filter(|label| label.is_finite())
        .for_each(|&label| *expected.entry(label as u32).or_default() += 1);
    let reported: std::collections::BTreeMap<u32, usize> = alignment
        .iter()
        .map(|entry| (entry.component, entry.pixel_count))
        .collect();
    assert_eq!(reported, expected);
    assert!(
        expected.len() >= 2,
        "stripe did not split components: {expected:?}"
    );
}
