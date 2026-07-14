//! Native-vs-SNAPHU validation on a REAL OPERA CSLC-S1 burst.
//!
//! Reads a real wrapped interferogram + real empirical coherence derived from the
//! captured Mexico T005 burst stack (`oracle/prep_real_ifg.py`, epoch 0 vs 12,
//! 5x5 boxcar coherence — real temporal/geometric decorrelation, ~16% residues),
//! unwraps it with both the native solver (global and the production fine-tiled
//! default) and a freshly-run SNAPHU oracle subprocess, and reports:
//!   * per-connected-component cycle disagreement (native vs SNAPHU, grouped by
//!     the SNAPHU component labels — the same metric the seam contracts use), and
//!   * connected-component agreement (reliable-mask IoU vs SNAPHU).
//!
//! SNAPHU is the black-box oracle (subprocess only). Run:
//!   cargo run --release --example real_burst_validation -p dolphin-unwrap

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use dolphin_core::Cf32;
use dolphin_unwrap::native::{unwrap_native, NativeConfig};
use dolphin_unwrap::{snaphu, CostMode, InitMethod, UnwrapConfig, UnwrapResult};
use ndarray::Array2;

const TWO_PI: f64 = std::f64::consts::TAU;

fn data_dir() -> PathBuf {
    std::env::var("REAL_IFG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures/real_ifg")
        })
}

/// Infer the square side length from a flat f32 raster's byte length.
fn square_side(bytes: usize, elem: usize) -> usize {
    ((bytes / elem) as f64).sqrt().round() as usize
}

fn read_ifg(path: &Path) -> Array2<Cf32> {
    let raw = std::fs::read(path).expect("read ifg.c8");
    let n = square_side(raw.len(), 8);
    let vals: Vec<Cf32> = raw
        .chunks_exact(8)
        .map(|b| {
            let re = f32::from_le_bytes(b[0..4].try_into().unwrap());
            let im = f32::from_le_bytes(b[4..8].try_into().unwrap());
            Cf32::new(re, im)
        })
        .collect();
    Array2::from_shape_vec((n, n), vals).unwrap()
}

fn read_corr(path: &Path) -> Array2<f32> {
    let raw = std::fs::read(path).expect("read corr.f4");
    let n = square_side(raw.len(), 4);
    let vals: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Array2::from_shape_vec((n, n), vals).unwrap()
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

/// Per-connected-component cycle disagreement of `cand` vs `oracle`, grouped by
/// the SNAPHU labels `cc` (component 0 = masked, excluded). Pixels with coherence
/// below `gate` are skipped, isolating agreement on *trusted* pixels (`gate = 0`
/// counts every component pixel).
fn per_component_disagreement(
    cand: &Array2<f32>,
    oracle: &Array2<f32>,
    cc: &Array2<u32>,
    corr: &Array2<f32>,
    gate: f32,
) -> f64 {
    let mut by_comp: HashMap<u32, Vec<i64>> = HashMap::new();
    for (((&label, &c), &o), &g) in cc
        .iter()
        .zip(cand.iter())
        .zip(oracle.iter())
        .zip(corr.iter())
    {
        if label == 0 || g < gate {
            continue;
        }
        let k = ((c as f64 - o as f64) / TWO_PI).round() as i64;
        by_comp.entry(label).or_default().push(k);
    }
    let (mut valid, mut bad) = (0usize, 0usize);
    for ks in by_comp.values() {
        let off = mode(ks);
        valid += ks.len();
        bad += ks.iter().filter(|&&k| k != off).count();
    }
    bad as f64 / valid.max(1) as f64
}

/// Reliable-mask IoU: |{native cc>0} ∩ {snaphu cc>0}| / |union|.
fn mask_iou(a: &Array2<u32>, b: &Array2<u32>) -> f64 {
    let (mut inter, mut uni) = (0usize, 0usize);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (xa, yb) = (x > 0, y > 0);
        inter += usize::from(xa && yb);
        uni += usize::from(xa || yb);
    }
    inter as f64 / uni.max(1) as f64
}

/// SNAPHU oracle: fresh subprocess, smooth cost + MCF init, single tile.
fn snaphu_oracle(ifg: &Array2<Cf32>, corr: &Array2<f32>) -> UnwrapResult {
    let scratch = std::env::temp_dir().join("real_burst_snaphu");
    std::fs::create_dir_all(&scratch).unwrap();
    let cfg = UnwrapConfig {
        cost: CostMode::Smooth,
        init: InitMethod::Mcf,
        ..UnwrapConfig::default()
    };
    snaphu::unwrap(ifg.view(), corr.view(), &cfg, &scratch).expect("snaphu oracle")
}

/// Production auto-tiling: cores are at least 64 px (matches `dolphin-workflows`).
fn fine_tiles(n: usize) -> (usize, usize) {
    let t = (n / 64).max(1);
    (t, t)
}

fn run_native(
    label: &str,
    ifg: &Array2<Cf32>,
    corr: &Array2<f32>,
    cfg: &NativeConfig,
) -> UnwrapResult {
    let t0 = Instant::now();
    let out = unwrap_native(ifg.view(), corr.view(), cfg).expect("native unwrap");
    eprintln!("[native {label}] {:.2}s", t0.elapsed().as_secs_f64());
    out
}

fn report(name: &str, native: &UnwrapResult, oracle: &UnwrapResult, corr: &Array2<f32>) {
    let unw = &native.unwrapped;
    let (oun, occ) = (&oracle.unwrapped, &oracle.conncomp);
    let all = per_component_disagreement(unw, oun, occ, corr, 0.0);
    let trusted = per_component_disagreement(unw, oun, occ, corr, 0.5);
    let iou = mask_iou(&native.conncomp, &oracle.conncomp);
    println!(
        "  {name:<14} vs-SNAPHU per-comp: all-px={:.4}%  coh>=0.5={:.4}%   mask-IoU={:.4}",
        all * 100.0,
        trusted * 100.0,
        iou
    );
}

fn main() {
    let dir = data_dir();
    let ifg = read_ifg(&dir.join("ifg.c8"));
    let corr = read_corr(&dir.join("corr.f4"));
    let (rows, cols) = ifg.dim();
    let residues = corr.iter().filter(|&&c| c < 0.3).count();
    println!(
        "REAL burst {rows}x{cols} from {}\n  coh mean={:.3} frac<0.3={:.3}",
        dir.display(),
        corr.iter().sum::<f32>() / (rows * cols) as f32,
        residues as f64 / (rows * cols) as f64
    );

    let oracle = snaphu_oracle(&ifg, &corr);
    let snaphu_comps = oracle.conncomp.iter().copied().max().unwrap_or(0);
    println!("  SNAPHU: components={snaphu_comps}");

    let tiled = NativeConfig {
        tile: Some(fine_tiles(rows.min(cols))),
        ..NativeConfig::default()
    };
    let native_tiled = run_native("tiled", &ifg, &corr, &tiled);
    report("native-tiled", &native_tiled, &oracle, &corr);

    if std::env::var("SKIP_GLOBAL").is_err() {
        let native_global = run_native("global", &ifg, &corr, &NativeConfig::default());
        report("native-global", &native_global, &oracle, &corr);
    }
}
