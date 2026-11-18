//! Live MMX1 common-frame parity contract for the whirlwind backend (#169).
//!
//! The external fixture is captured by the GNSS validation harness (real
//! Sentinel-1 data over the MMX1/ICMX stations), so fresh checkouts without the
//! live-data run skip this contract. When present, every one of the frame's
//! interferograms must match the co-run SNAPHU oracle to the shipped 0.5%
//! per-connected-component integer-cycle gate that native meets.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use dolphin_core::Cf32;
use dolphin_unwrap::whirlwind::{unwrap_whirlwind, WhirlwindConfig};
use ndarray::Array2;

const ROWS: usize = 352;
const COLS: usize = 2217;
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
    // Deterministic tie-break (mirrors `collect_seams`): HashMap iteration
    // order must not decide the dominant offset when counts tie.
    counts
        .into_iter()
        .max_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.0.abs().cmp(&a.0.abs()))
                .then_with(|| b.0.cmp(&a.0))
        })
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
#[ignore = "requires MMX1 common-frame fixture; run explicitly with --ignored"]
fn whirlwind_matches_snaphu_on_every_mmx1_interferogram() {
    let dir = fixture_dir();
    let correlation = read_f32(&dir.join("corr.f4"));
    let mut pairs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("unw.f4").is_file())
        .collect();
    pairs.sort();
    assert_eq!(
        pairs.len(),
        12,
        "MMX1 fixture should hold 12 interferograms"
    );
    let worst = pairs
        .iter()
        .map(|pair| {
            let ifg = read_ifg(&pair.join("ifg.c8"));
            let oracle = read_f32(&pair.join("unw.f4"));
            let components = read_u32(&pair.join("conncomp.u4"));
            let started = Instant::now();
            let result =
                unwrap_whirlwind(ifg.view(), correlation.view(), &WhirlwindConfig::default())
                    .unwrap();
            let disagreement = per_component_disagreement(&result.unwrapped, &oracle, &components);
            eprintln!(
                "MMX1 {} whirlwind vs SNAPHU: {:.4}% cycle disagreement ({:.2} s wall)",
                pair.file_name().unwrap().to_string_lossy(),
                disagreement * 100.0,
                started.elapsed().as_secs_f64()
            );
            disagreement
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst <= MAX_CYCLE_DISAGREE,
        "MMX1 worst cycle disagreement {:.4}% exceeds {:.2}% gate",
        worst * 100.0,
        MAX_CYCLE_DISAGREE * 100.0
    );
}
