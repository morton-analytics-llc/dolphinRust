//! Accuracy-vs-granularity check: native tiled unwrap at several tile counts
//! against the committed 1024^2 SNAPHU oracle, per connected component.
//!
//! Fine tiling slashes CPU-s (the network simplex is superlinear in residues per
//! tile), but only earns the default if parity holds as tiles shrink. This loads
//! `unwdense1024_*` and prints per-component disagreement for global + each tile
//! count so the throughput tiling can be chosen without trading accuracy.
//!
//! Run:  cargo run --release --example tile_parity -p dolphin-unwrap

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_unwrap::native::{unwrap_native, NativeConfig};
use ndarray::Array2;

const TWO_PI: f64 = std::f64::consts::TAU;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn mode(cycles: &[i64]) -> i64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &k in cycles {
        *counts.entry(k).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map_or(0, |(k, _)| k)
}

fn per_component_disagreement(cand: &Array2<f32>, oracle: &Array2<f32>, cc: &Array2<u32>) -> f64 {
    let mut by_comp: HashMap<u32, Vec<i64>> = HashMap::new();
    for ((&label, &c), &o) in cc.iter().zip(cand.iter()).zip(oracle.iter()) {
        if label == 0 {
            continue;
        }
        by_comp
            .entry(label)
            .or_default()
            .push(((c as f64 - o as f64) / TWO_PI).round() as i64);
    }
    let (mut valid, mut bad) = (0usize, 0usize);
    for ks in by_comp.values() {
        let off = mode(ks);
        valid += ks.len();
        bad += ks.iter().filter(|&&k| k != off).count();
    }
    bad as f64 / valid.max(1) as f64
}

fn main() {
    let dir = fixtures();
    let tag = std::env::var("TAG").unwrap_or_else(|_| "unwdense1024".to_string());
    let ifg: Array2<Cf32> = ndarray_npy::read_npy(dir.join(format!("{tag}_ifg.npy"))).unwrap();
    let corr: Array2<f32> = ndarray_npy::read_npy(dir.join(format!("{tag}_corr.npy"))).unwrap();
    let oracle: Array2<f32> = ndarray_npy::read_npy(dir.join(format!("{tag}_oracle.npy"))).unwrap();
    let cc: Array2<u32> = ndarray_npy::read_npy(dir.join(format!("{tag}_conncomp.npy"))).unwrap();
    let ncomp = cc.iter().copied().max().unwrap_or(0);
    println!("{tag} {:?} components={ncomp}", ifg.dim());

    for tiles in [None, Some(4usize), Some(8), Some(16), Some(32)] {
        let cfg = NativeConfig {
            tile: tiles.map(|t| (t, t)),
            ..NativeConfig::default()
        };
        let out = unwrap_native(ifg.view(), corr.view(), &cfg).unwrap();
        let d = per_component_disagreement(&out.unwrapped, &oracle, &cc);
        let lbl = tiles.map_or("global".to_string(), |t| format!("{t}x{t}"));
        println!("  {lbl:>7}  vs-SNAPHU per-component = {:.4}%", d * 100.0);
    }
}
