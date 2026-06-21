//! Native tiling contract (Chen 2002 inter-tile reconciliation).
//!
//! Unwrapping the multi-tile-sized fixture with a 2x2 tile grid must reproduce
//! both the whole-grid global solve and the SNAPHU oracle to integer-cycle
//! parity (equal up to a global constant) — the inter-tile offset reconciliation
//! seams the tiles back together losslessly. Skips when fixtures are absent.

use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_unwrap::native::{unwrap_native, NativeConfig};
use ndarray::Array2;

const TWO_PI: f64 = std::f64::consts::TAU;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

/// Fraction of pixels whose integer-cycle offset deviates from the dominant one.
fn cycle_disagreement(a: &Array2<f32>, b: &Array2<f32>) -> f64 {
    let cycles: Vec<i64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64 - *y as f64) / TWO_PI).round() as i64)
        .collect();
    let mut counts = std::collections::HashMap::new();
    for &k in &cycles {
        *counts.entry(k).or_insert(0usize) += 1;
    }
    let mode = counts
        .iter()
        .max_by_key(|&(_, c)| c)
        .map(|(&k, _)| k)
        .unwrap_or(0);
    cycles.iter().filter(|&&k| k != mode).count() as f64 / cycles.len().max(1) as f64
}

#[test]
fn tiled_matches_global_and_oracle() {
    let dir = fixtures();
    if !dir.join("unwsuite_multitile_oracle.npy").exists() {
        eprintln!("skipping native tiling: no golden suite");
        return;
    }
    let ifg: Array2<Cf32> = ndarray_npy::read_npy(dir.join("unwsuite_multitile_ifg.npy")).unwrap();
    let corr: Array2<f32> = ndarray_npy::read_npy(dir.join("unwsuite_multitile_corr.npy")).unwrap();
    let oracle: Array2<f32> =
        ndarray_npy::read_npy(dir.join("unwsuite_multitile_oracle.npy")).unwrap();

    let global = unwrap_native(ifg.view(), corr.view(), &NativeConfig::default()).unwrap();
    let tiled_cfg = NativeConfig {
        tile: Some((2, 2)),
        ..NativeConfig::default()
    };
    let tiled = unwrap_native(ifg.view(), corr.view(), &tiled_cfg).unwrap();

    let vs_global = cycle_disagreement(&tiled.unwrapped, &global.unwrapped);
    let vs_oracle = cycle_disagreement(&tiled.unwrapped, &oracle);
    eprintln!("tiled vs global={vs_global:.4}%  vs oracle={vs_oracle:.4}%");
    assert!(
        vs_global < 1e-9,
        "tiled must equal the global solve exactly"
    );
    assert!(vs_oracle < 0.005, "tiled must match SNAPHU to parity");
}
