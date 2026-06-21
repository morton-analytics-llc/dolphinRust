//! Tiling + inter-tile reconciliation for large interferograms (Chen 2002).
//!
//! Chen & Zebker 2002 unwraps a large interferogram by partitioning it into
//! overlapping tiles, unwrapping each independently, then a secondary
//! optimization reconciles the integer-cycle offset between adjacent tiles. We
//! do the same in its practical form: each tile is unwrapped by the global MCF
//! ([`super::unwrap_grid`]) in parallel; tiles are then placed in raster order,
//! each shifted by the **modal** integer-cycle difference over its overlap with
//! the already-placed region so the seams are continuous. Because a residue-free
//! region unwraps uniquely up to a constant, this reproduces the global result
//! whenever no residue straddles a seam — verified against both the global solve
//! and SNAPHU in `tests/native_tiling_contract.rs`.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use super::CostMode;

const TAU: f64 = std::f64::consts::TAU;
/// Tile overlap (pixels) — the band used to reconcile inter-tile offsets.
const OVERLAP: usize = 8;

/// Unwrap `psi` by overlapping tiles, reconciling inter-tile offsets.
pub fn unwrap_tiled(
    psi: &Array2<f64>,
    corr: ArrayView2<f32>,
    cost: CostMode,
    tiles: (usize, usize),
) -> Array2<f64> {
    let (rows, cols) = psi.dim();
    let row_spans = spans(rows, tiles.0);
    let col_spans = spans(cols, tiles.1);

    // Raster-ordered tile windows, then unwrap each independently. `par_iter`
    // preserves order, so `solved` stays in raster order for the stitch.
    let combos: Vec<((usize, usize), (usize, usize))> = row_spans
        .iter()
        .flat_map(|&rs| col_spans.iter().map(move |&cs| (rs, cs)))
        .collect();
    let solved: Vec<Tile> = combos
        .par_iter()
        .map(|&(rs, cs)| solve_tile(psi, corr, cost, rs, cs))
        .collect();

    stitch(rows, cols, solved)
}

/// One unwrapped tile and the grid window it covers.
struct Tile {
    rs: (usize, usize),
    cs: (usize, usize),
    unwrapped: Array2<f64>,
}

/// Overlapping `[start, end)` spans covering `n` in `parts` tiles, each grown by
/// `OVERLAP` on the interior sides.
fn spans(n: usize, parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.max(1).min(n.max(1));
    let core = n.div_ceil(parts);
    (0..parts)
        .map(|p| {
            let start = (p * core).saturating_sub(OVERLAP);
            let end = (((p + 1) * core) + OVERLAP).min(n);
            (start, end.max(start + 1))
        })
        .filter(|&(s, _)| s < n)
        .collect()
}

/// Unwrap the tile window `(rs, cs)` of the grid with the global solver.
fn solve_tile(
    psi: &Array2<f64>,
    corr: ArrayView2<f32>,
    cost: CostMode,
    rs: (usize, usize),
    cs: (usize, usize),
) -> Tile {
    let sub_psi = psi.slice(ndarray::s![rs.0..rs.1, cs.0..cs.1]).to_owned();
    let sub_corr = corr.slice(ndarray::s![rs.0..rs.1, cs.0..cs.1]);
    Tile {
        rs,
        cs,
        unwrapped: super::unwrap_grid(&sub_psi, sub_corr, cost),
    }
}

/// Place tiles in raster order, each shifted by the modal integer-cycle offset
/// over its overlap with the already-filled region.
fn stitch(rows: usize, cols: usize, solved: Vec<Tile>) -> Array2<f64> {
    let mut out = Array2::zeros((rows, cols));
    let mut filled = Array2::from_elem((rows, cols), false);
    for tile in &solved {
        place(&mut out, &mut filled, tile);
    }
    out
}

/// Write one reconciled tile into the output, claiming only unfilled pixels.
fn place(out: &mut Array2<f64>, filled: &mut Array2<bool>, tile: &Tile) {
    let shift = TAU * modal_offset(out, filled, tile) as f64;
    for ((li, lj), &v) in tile.unwrapped.indexed_iter() {
        let (gi, gj) = (tile.rs.0 + li, tile.cs.0 + lj);
        if filled[(gi, gj)] {
            continue;
        }
        out[(gi, gj)] = v + shift;
        filled[(gi, gj)] = true;
    }
}

/// Modal integer-cycle difference `round((out - tile)/2pi)` over the tile's
/// already-filled overlap; `0` for the first (seed) tile.
fn modal_offset(out: &Array2<f64>, filled: &Array2<bool>, tile: &Tile) -> i64 {
    let mut counts = std::collections::HashMap::new();
    for ((li, lj), &v) in tile.unwrapped.indexed_iter() {
        let (gi, gj) = (tile.rs.0 + li, tile.cs.0 + lj);
        if !filled[(gi, gj)] {
            continue;
        }
        let k = ((out[(gi, gj)] - v) / TAU).round() as i64;
        *counts.entry(k).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(k, _)| k)
        .unwrap_or(0)
}
