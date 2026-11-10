//! Contract for `dolphin pairs` coherence mode.
//!
//! - On the real cropped Mexico City OPERA CSLC stack, each pair COG equals `|Γ_ij|`
//!   from the full-stack `estimate_stack_covariance` to max |Δ| < 1e-6. Skips when
//!   the local real fixture is absent, like the other real-data gates.
//! - A 12-day series with a 24-day gap forms no pair across the gap.
//! - Every pair COG carries the six pair tags.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use dolphin_core::config::InputType;
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_core::{Cf32, Cf64};
use dolphin_io::{read_cslc, read_raster, read_raster_header};
use dolphin_phaselink::covariance::estimate_stack_covariance;
use dolphin_workflows::{
    plan_pairs, run_pairs, PairAcquisition, PairPlan, PairsConfig, PairsIndex, PairsMode,
    PAIRS_INDEX_FILENAME,
};
use ndarray::{Array2, Array3, Axis};

const MEXICO: &str = "../../validation/real_data/cropped_mexico";
const SUBDATASET: &str = "/data/VV";

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
            write_cslc(&path, t);
            PairAcquisition {
                path,
                acquisition_id: id,
                acquisition_utc: utc,
            }
        })
        .collect()
}

/// Small OPERA-layout CSLC: `/data/VV` plus coordinates and projection.
fn write_cslc(path: &Path, seed: usize) {
    let (rows, cols) = (16, 24);
    let grid = Array2::from_shape_fn((rows, cols), |(r, c)| {
        Cf32::from_polar(1.0, (0.1 * (seed * (r + 2 * c)) as f64) as f32)
    });
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
