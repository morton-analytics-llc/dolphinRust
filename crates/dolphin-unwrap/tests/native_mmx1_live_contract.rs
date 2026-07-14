//! Live MMX1 common-frame parity contract for the production native tile grid.
//!
//! The external fixture is captured by the GNSS validation harness, so fresh
//! checkouts without the live-data run skip this contract. When present, the
//! final 2023-01-04 -> 2023-06-09 interferogram must match the co-run SNAPHU
//! oracle to the shipped 0.5% per-connected-component integer-cycle gate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_unwrap::native::{unwrap_native, NativeConfig};
use ndarray::Array2;

const ROWS: usize = 352;
const COLS: usize = 2217;
const PRODUCTION_TILES: (usize, usize) = (5, 34);
const MAX_CYCLE_DISAGREE: f64 = 0.005;
const TWO_PI: f64 = std::f64::consts::TAU;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/runs/gps_mmx1/mmx1_icmx_common/work_snaphu/scratch")
}

fn read_f32(path: &Path) -> Array2<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    let values = raw
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    Array2::from_shape_vec((ROWS, COLS), values).unwrap()
}

fn read_u32(path: &Path) -> Array2<u32> {
    let raw = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    let values = raw
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    Array2::from_shape_vec((ROWS, COLS), values).unwrap()
}

fn read_ifg(path: &Path) -> Array2<Cf32> {
    let raw = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    let values = raw
        .chunks_exact(8)
        .map(|bytes| {
            let real = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            let imaginary = f32::from_le_bytes(bytes[4..].try_into().unwrap());
            Cf32::new(real, imaginary)
        })
        .collect();
    Array2::from_shape_vec((ROWS, COLS), values).unwrap()
}

fn dominant(cycles: &[i64]) -> i64 {
    let mut counts = HashMap::new();
    for &cycle in cycles {
        *counts.entry(cycle).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map_or(0, |(cycle, _)| cycle)
}

fn per_component_disagreement(
    candidate: &Array2<f32>,
    oracle: &Array2<f32>,
    components: &Array2<u32>,
) -> f64 {
    let mut cycles_by_component: HashMap<u32, Vec<i64>> = HashMap::new();
    for ((&component, &candidate_phase), &oracle_phase) in
        components.iter().zip(candidate.iter()).zip(oracle.iter())
    {
        if component == 0 {
            continue;
        }
        let cycle = ((candidate_phase as f64 - oracle_phase as f64) / TWO_PI).round() as i64;
        cycles_by_component
            .entry(component)
            .or_default()
            .push(cycle);
    }
    let (mut valid, mut disagree) = (0usize, 0usize);
    for cycles in cycles_by_component.values() {
        let offset = dominant(cycles);
        valid += cycles.len();
        disagree += cycles.iter().filter(|&&cycle| cycle != offset).count();
    }
    disagree as f64 / valid.max(1) as f64
}

#[test]
fn production_tiles_match_snaphu_on_mmx1_final_epoch() {
    let dir = fixture_dir();
    let pair = dir.join("pair_0011");
    if !pair.join("ifg.c8").exists() {
        eprintln!("skipping MMX1 live parity: run the gps_mmx1 common-frame harness first");
        return;
    }

    let ifg = read_ifg(&pair.join("ifg.c8"));
    let correlation = read_f32(&dir.join("corr.f4"));
    let oracle = read_f32(&pair.join("unw.f4"));
    let components = read_u32(&pair.join("conncomp.u4"));
    let config = NativeConfig {
        tile: Some(PRODUCTION_TILES),
        ..NativeConfig::default()
    };
    let native = unwrap_native(ifg.view(), correlation.view(), &config).unwrap();
    let disagreement = per_component_disagreement(&native.unwrapped, &oracle, &components);

    eprintln!(
        "MMX1 final epoch production tiles vs SNAPHU: {:.4}% cycle disagreement",
        disagreement * 100.0
    );
    assert!(
        disagreement <= MAX_CYCLE_DISAGREE,
        "MMX1 final epoch cycle disagreement {:.4}% exceeds {:.2}% gate",
        disagreement * 100.0,
        MAX_CYCLE_DISAGREE * 100.0
    );
}
