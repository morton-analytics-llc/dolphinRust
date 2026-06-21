//! Measure native-vs-SNAPHU parity on a residue-DENSE scene (per-component).
//!
//! Loads a golden tag from oracle/fixtures (default `unwdense_ci`; override with
//! TAG=unwdense for the big 1024^2 scene). Reports, at real residue density:
//!   - per-SNAPHU-component cycle disagreement (each component gets its own
//!     integer offset — SNAPHU assigns these independently),
//!   - the old single-global-mode disagreement (for contrast),
//!   - conncomp partition agreement (native currently returns a trivial single
//!     component — this exposes exactly that),
//!   - max sub-cycle residual on agreeing pixels.
//!
//! SNAPHU is the oracle that produced the goldens; this never invokes it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

// Print-heavy measurement bench (residue-dense parity + perf/RSS sweep); the
// reporting is inherently linear and reads better as one flow.
#[allow(clippy::too_many_lines)]
fn main() {
    let tag = std::env::var("TAG").unwrap_or_else(|_| "unwdense_ci".into());
    let dir = fixtures();
    let ifg: Array2<Cf32> = ndarray_npy::read_npy(dir.join(format!("{tag}_ifg.npy"))).expect("ifg");
    let corr: Array2<f32> =
        ndarray_npy::read_npy(dir.join(format!("{tag}_corr.npy"))).expect("corr");
    let oracle: Array2<f32> =
        ndarray_npy::read_npy(dir.join(format!("{tag}_oracle.npy"))).expect("oracle");
    let cc: Array2<u32> =
        ndarray_npy::read_npy(dir.join(format!("{tag}_conncomp.npy"))).expect("cc");

    let (rows, cols) = ifg.dim();
    // TILE="NxM" enables the Chen-2002 tiled path (par_iter over tiles); unset = global MCF.
    let tile = std::env::var("TILE").ok().and_then(|s| {
        let (a, b) = s.split_once('x')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let cfg = NativeConfig {
        tile,
        ..NativeConfig::default()
    };
    let t0 = Instant::now();
    let out = unwrap_native(ifg.view(), corr.view(), &cfg).expect("native");
    let dt = t0.elapsed();
    println!(
        "=== tag={tag}  {rows}x{cols}  tile={:?}  threads={}  native {:?} ===",
        tile,
        rayon::current_num_threads(),
        dt
    );

    // Diff field native - oracle in cycles, per pixel.
    let n = &out.unwrapped;
    let kcyc: Array2<i64> = Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((n[(i, j)] as f64 - oracle[(i, j)] as f64) / TWO_PI).round() as i64
    });

    // --- Global single-mode disagreement over conncomp>0 (old metric) ---
    let valid: Vec<(usize, usize)> = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .filter(|&(i, j)| cc[(i, j)] > 0)
        .collect();
    let all_k: Vec<i64> = valid.iter().map(|&p| kcyc[p]).collect();
    let gmode = mode(&all_k);
    let gdis = all_k.iter().filter(|&&k| k != gmode).count();
    println!(
        "global-mode : valid={} disagree={} ({:.4}%)",
        all_k.len(),
        gdis,
        100.0 * gdis as f64 / all_k.len().max(1) as f64
    );

    // --- Per-component disagreement (each SNAPHU comp -> own offset) ---
    let mut by_comp: HashMap<u32, Vec<i64>> = HashMap::new();
    for &p in &valid {
        by_comp.entry(cc[p]).or_default().push(kcyc[p]);
    }
    let mut comp_ids: Vec<u32> = by_comp.keys().copied().collect();
    comp_ids.sort_unstable();
    let offsets: HashMap<u32, i64> = by_comp.iter().map(|(&c, ks)| (c, mode(ks))).collect();
    let mut total_valid = 0usize;
    let mut total_dis = 0usize;
    println!("per-component:");
    for c in &comp_ids {
        let ks = &by_comp[c];
        let m = offsets[c];
        let dis = ks.iter().filter(|&&k| k != m).count();
        total_valid += ks.len();
        total_dis += dis;
        println!(
            "  comp {:3}: n={:7} offset={:4} disagree={:6} ({:.4}%)",
            c,
            ks.len(),
            m,
            dis,
            100.0 * dis as f64 / ks.len().max(1) as f64
        );
    }
    println!(
        "per-component TOTAL: valid={} disagree={} ({:.4}%)",
        total_valid,
        total_dis,
        100.0 * total_dis as f64 / total_valid.max(1) as f64
    );

    // --- Sub-cycle residual on per-component-agreeing pixels ---
    let max_frac = valid
        .iter()
        .map(|&p| {
            let d = n[p] as f64 - oracle[p] as f64 - offsets[&cc[p]] as f64 * TWO_PI;
            (d - TWO_PI * (d / TWO_PI).round()).abs()
        })
        .fold(0.0f64, f64::max);
    println!("max sub-cycle residual (agreeing): {:.4} rad", max_frac);

    // --- Conncomp partition agreement ---
    let snaphu_masked = (rows * cols) - valid.len();
    let native_masked = out.conncomp.iter().filter(|&&v| v == 0).count();
    let snaphu_ncomp = comp_ids.len();
    let native_labels: std::collections::HashSet<u32> =
        out.conncomp.iter().copied().filter(|&v| v > 0).collect();
    println!(
        "conncomp: snaphu_ncomp={} native_ncomp={} | snaphu_masked={} native_masked={}",
        snaphu_ncomp,
        native_labels.len(),
        snaphu_masked,
        native_masked
    );
    // IoU of "masked" sets (where native should ideally also drop low-coh px).
    let masks = cc
        .iter()
        .zip(out.conncomp.iter())
        .map(|(&s, &n)| (s == 0, n == 0));
    let mask_inter = masks.clone().filter(|&(s, n)| s && n).count();
    let mask_union = masks.filter(|&(s, n)| s || n).count();
    println!(
        "mask IoU (native vs snaphu masked set): {:.4} ({}/{})",
        mask_inter as f64 / mask_union.max(1) as f64,
        mask_inter,
        mask_union
    );
}
