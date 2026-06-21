//! Unwrap-network parallelization contract (Tier-1 #1 + #3).
//!
//! The interferogram network unwraps each ifg with an independent 2D SNAPHU
//! solve. Tier-1 parallelizes that loop (`.par_iter`) with per-pair scratch
//! isolation and hoists the shared correlation raster. Neither changes the math,
//! so the stacked output MUST be **bit-identical** to unwrapping every pair
//! independently in its own clean scratch (the serial golden) AND order-stable
//! regardless of completion order.
//!
//! This is the red→green guard for #1: with a shared scratch the concurrent
//! SNAPHU processes clobber each other's fixed-name scratch files
//! (`ifg.c8`/`unw.f4`/...) and the per-layer output diverges from its golden.
//! Skips without `snaphu`.

use std::path::Path;

use dolphin_core::{Cf32, Cf64};
use dolphin_unwrap::{unwrap, UnwrapConfig};
use dolphin_workflows::{SnaphuBackend, UnwrapBackend};
use ndarray::{Array2, Array3};

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// An `n_dates` linked-phase stack: date 0 is flat (single-reference base), date
/// `t` carries a smooth ramp scaled by `t`. Per-pixel gradient stays < π for the
/// deepest pair so every ifg is cleanly unwrappable and SNAPHU is deterministic.
fn ramp_stack(n_dates: usize, rows: usize, cols: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((n_dates, rows, cols), |(t, r, c)| {
        let phase = t as f64 * (0.05 * r as f64 + 0.03 * c as f64);
        Cf64::from_polar(1.0, phase)
    })
}

/// Single-reference network over `n_dates`: pairs `(0, j)` for `j = 1..n_dates`.
fn single_ref_pairs(n_dates: usize) -> Vec<(usize, usize)> {
    (1..n_dates).map(|j| (0, j)).collect()
}

/// Form the wrapped ifg `(i, j)` exactly as the backend does: `pl_i · conj(pl_j)`.
fn form_ifg(pl: &Array3<Cf64>, (i, j): (usize, usize)) -> Array2<Cf32> {
    let (_, rows, cols) = pl.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let z = pl[(i, r, c)] * pl[(j, r, c)].conj();
        Cf32::from_polar(1.0, z.arg() as f32)
    })
}

/// Golden: unwrap one pair independently in its OWN clean scratch dir, serially.
/// This is the per-pair serial reference the parallel network must reproduce.
fn golden_layer(
    pl: &Array3<Cf64>,
    pair: (usize, usize),
    corr: &Array2<f32>,
    dir: &Path,
) -> Array2<f64> {
    let ifg = form_ifg(pl, pair);
    std::fs::create_dir_all(dir).unwrap();
    unwrap(ifg.view(), corr.view(), &UnwrapConfig::default(), dir)
        .unwrap()
        .unwrapped
        .mapv(f64::from)
}

#[test]
fn parallel_network_is_bit_identical_to_serial_golden() {
    if !snaphu_available() {
        eprintln!("skipping unwrap-parallel contract: snaphu not on PATH");
        return;
    }
    let (n_dates, rows, cols) = (12, 40, 48);
    let pl = ramp_stack(n_dates, rows, cols);
    let pairs = single_ref_pairs(n_dates);
    let corr = Array2::<f32>::from_elem((rows, cols), 1.0);

    // Golden: each pair unwrapped independently in an isolated scratch, serially.
    let gdir = std::env::temp_dir().join("dolphinrust_unwrap_golden");
    let _ = std::fs::remove_dir_all(&gdir);
    let golden: Vec<Array2<f64>> = pairs
        .iter()
        .enumerate()
        .map(|(idx, &pair)| golden_layer(&pl, pair, &corr, &gdir.join(format!("g_{idx}"))))
        .collect();

    // Parallel network unwrap through the production backend + dispatch.
    let scratch = std::env::temp_dir().join("dolphinrust_unwrap_parallel");
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let backend = SnaphuBackend(UnwrapConfig::default());
    let out = backend
        .unwrap_network(pl.view(), &pairs, corr.view(), &scratch)
        .unwrap();

    assert_eq!(out.dim(), (pairs.len(), rows, cols), "stacked shape");

    // Bit-identical, layer by layer, in pairs order. Bitwise — not approximate.
    for (idx, g) in golden.iter().enumerate() {
        let layer = out.index_axis(ndarray::Axis(0), idx);
        let mismatches = layer
            .iter()
            .zip(g.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches, 0,
            "layer {idx} (pair {:?}) diverged from serial golden in {mismatches} px \
             — scratch isolation or ordering bug",
            pairs[idx]
        );
    }
}
