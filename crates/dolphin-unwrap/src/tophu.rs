//! tophu-style multi-scale unwrapping, driven over the SNAPHU wrapper.
//!
//! Raw SNAPHU degrades on large, low-coherence scenes: its network-flow solver
//! can pick the wrong integer-cycle branch in vegetated/decorrelated regions far
//! from any reliable reference. tophu's remedy is a coarse→fine cascade:
//!
//! 1. **coarse** — multilook (downsample) the wrapped ifg + correlation by
//!    `downsample_factor` and unwrap that small grid once. Fewer pixels and
//!    higher per-pixel coherence make the coarse solution globally reliable,
//!    even if it lacks fine detail.
//! 2. **upsample** — nearest-neighbour the coarse unwrapped phase back to full
//!    resolution. This is the absolute-ambiguity reference every tile is anchored
//!    to.
//! 3. **tiled fine** — split the full-res grid into overlapping tiles and unwrap
//!    each independently (in parallel). Each tile is internally consistent but
//!    carries an arbitrary global 2π offset.
//! 4. **merge** — snap each tile to the coarse reference by the integer number of
//!    2π cycles that minimises its mean residual. Anchoring every tile to the one
//!    coarse solution makes adjacent tiles mutually consistent.
//!
//! This is heuristic orchestration over SNAPHU, not new unwrap math — the
//! reference is tophu's *algorithm* and *result quality*, not bit-parity (its own
//! tiling/merge is non-unique). dolphin reserves its `tophu.multiscale_unwrap`
//! driver for the ICU/PHASS per-tile solvers; dolphinRust drives the SNAPHU
//! wrapper per tile, which is the solver we ship.

use std::path::Path;

use dolphin_core::Cf32;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

use crate::snaphu::{unwrap, CostMode, InitMethod, Result, UnwrapConfig, UnwrapResult};

const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

/// Multi-scale (tophu) unwrap configuration. Mirrors dolphin's `TophuOptions`
/// (`ntiles`, `downsample_factor`, `init`, `cost`) plus the per-tile overlap and
/// SNAPHU binary path this driver needs.
#[derive(Debug, Clone)]
pub struct TophuConfig {
    /// Extra multilook factor `(row, col)` for the coarse pass.
    pub downsample_factor: (usize, usize),
    /// Full-res tile grid `(rows, cols)` for the fine pass.
    pub ntiles: (usize, usize),
    /// Per-tile overlap halo `(row, col)` in full-res pixels.
    pub tile_overlap: (usize, usize),
    /// SNAPHU statistical cost mode for every sub-unwrap.
    pub cost: CostMode,
    /// SNAPHU initialization method for every sub-unwrap.
    pub init: InitMethod,
    /// Path to (or name of) the SNAPHU executable.
    pub snaphu_path: String,
}

impl Default for TophuConfig {
    fn default() -> Self {
        Self {
            downsample_factor: (3, 3),
            ntiles: (2, 2),
            tile_overlap: (16, 16),
            cost: CostMode::Smooth,
            init: InitMethod::Mcf,
            snaphu_path: "snaphu".to_string(),
        }
    }
}

/// A tile's full-res core region `[r0, r1) × [c0, c1)` and its overlap-expanded
/// region `[er0, er1) × [ec0, ec1)`; the expanded region is what SNAPHU unwraps.
#[derive(Debug, Clone, Copy)]
struct TileRegion {
    core: (usize, usize, usize, usize),
    exp: (usize, usize, usize, usize),
}

/// An unwrapped tile: its region plus the expanded-region unwrapped phase.
struct UnwrappedTile {
    region: TileRegion,
    phase: Array2<f32>,
}

/// Unwrap a wrapped interferogram with the multi-scale tophu strategy.
///
/// # Errors
/// Returns `Err` if scratch I/O fails or any SNAPHU sub-unwrap exits non-zero.
pub fn unwrap_multiscale(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    cfg: &TophuConfig,
    scratch: &Path,
) -> Result<UnwrapResult> {
    let (rows, cols) = wrapped.dim();
    let coarse_up = coarse_reference(wrapped, correlation, cfg, scratch)?;
    let regions = tile_regions((rows, cols), cfg.ntiles, cfg.tile_overlap);
    let tiles = unwrap_tiles(wrapped, correlation, cfg, scratch, &regions)?;
    let unwrapped = merge_tiles(coarse_up.view(), (rows, cols), &tiles);
    // The merged solution is a single coarse-anchored surface; SNAPHU's per-tile
    // connected components are not re-stitched, so all pixels carry label 1.
    let conncomp = Array2::<u32>::ones((rows, cols));
    Ok(UnwrapResult {
        unwrapped,
        conncomp,
    })
}

/// Coarse pass: downsample, unwrap once, upsample back to full resolution as the
/// absolute-ambiguity reference.
fn coarse_reference(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    cfg: &TophuConfig,
    scratch: &Path,
) -> Result<Array2<f32>> {
    let factor = cfg.downsample_factor;
    let coarse_ifg = downsample_complex(wrapped, factor);
    let coarse_corr = downsample_real(correlation, factor);
    let dir = scratch.join("coarse");
    std::fs::create_dir_all(&dir)?;
    let coarse = unwrap(
        coarse_ifg.view(),
        coarse_corr.view(),
        &single_tile_cfg(cfg),
        &dir,
    )?;
    Ok(upsample_nearest(
        coarse.unwrapped.view(),
        wrapped.dim(),
        factor,
    ))
}

/// Fine pass: unwrap every overlap-expanded tile in parallel via SNAPHU.
fn unwrap_tiles(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    cfg: &TophuConfig,
    scratch: &Path,
    regions: &[TileRegion],
) -> Result<Vec<UnwrappedTile>> {
    let tile_cfg = single_tile_cfg(cfg);
    regions
        .par_iter()
        .enumerate()
        .map(|(idx, region)| {
            unwrap_one_tile(wrapped, correlation, &tile_cfg, scratch, idx, *region)
        })
        .collect()
}

/// Unwrap a single overlap-expanded tile.
fn unwrap_one_tile(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    tile_cfg: &UnwrapConfig,
    scratch: &Path,
    idx: usize,
    region: TileRegion,
) -> Result<UnwrappedTile> {
    let (er0, er1, ec0, ec1) = region.exp;
    let w = wrapped.slice(s![er0..er1, ec0..ec1]);
    let cor = correlation.slice(s![er0..er1, ec0..ec1]);
    let dir = scratch.join(format!("tile_{idx}"));
    std::fs::create_dir_all(&dir)?;
    let r = unwrap(w, cor, tile_cfg, &dir)?;
    Ok(UnwrappedTile {
        region,
        phase: r.unwrapped,
    })
}

/// SNAPHU config for one sub-unwrap (single internal tile, serial).
fn single_tile_cfg(cfg: &TophuConfig) -> UnwrapConfig {
    UnwrapConfig {
        cost: cfg.cost,
        init: cfg.init,
        ntiles: (1, 1),
        tile_overlap: (0, 0),
        nproc: 1,
        snaphu_path: cfg.snaphu_path.clone(),
    }
}

/// Even tile split of `shape` into `ntiles`, each grown by `overlap` (clamped).
fn tile_regions(
    shape: (usize, usize),
    ntiles: (usize, usize),
    overlap: (usize, usize),
) -> Vec<TileRegion> {
    let (rows, cols) = shape;
    let (tr, tc) = (ntiles.0.max(1), ntiles.1.max(1));
    (0..tr)
        .flat_map(|ti| (0..tc).map(move |tj| (ti, tj)))
        .map(|(ti, tj)| {
            let core = (
                ti * rows / tr,
                (ti + 1) * rows / tr,
                tj * cols / tc,
                (tj + 1) * cols / tc,
            );
            TileRegion {
                core,
                exp: expand(core, overlap, shape),
            }
        })
        .collect()
}

/// Grow a core region by the overlap halo, clamped to the grid.
fn expand(
    core: (usize, usize, usize, usize),
    overlap: (usize, usize),
    shape: (usize, usize),
) -> (usize, usize, usize, usize) {
    let (r0, r1, c0, c1) = core;
    let (ovr, ovc) = overlap;
    (
        r0.saturating_sub(ovr),
        (r1 + ovr).min(shape.0),
        c0.saturating_sub(ovc),
        (c1 + ovc).min(shape.1),
    )
}

/// Anchor each tile to the coarse reference by an integer 2π offset and paste its
/// core into the full-res output.
fn merge_tiles(
    coarse_up: ArrayView2<f32>,
    shape: (usize, usize),
    tiles: &[UnwrappedTile],
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(shape);
    for tile in tiles {
        let (er0, er1, ec0, ec1) = tile.region.exp;
        let reference = coarse_up.slice(s![er0..er1, ec0..ec1]);
        let offset = cycle_offset(tile.phase.view(), reference);
        let (r0, r1, c0, c1) = tile.region.core;
        let core = tile.phase.slice(s![r0 - er0..r1 - er0, c0 - ec0..c1 - ec0]);
        out.slice_mut(s![r0..r1, c0..c1]).assign(&(&core + offset));
    }
    out
}

/// Integer-cycle phase offset (multiple of 2π) that best aligns `tile` to
/// `reference` in the mean — the tophu inter-tile reconciliation step.
fn cycle_offset(tile: ArrayView2<f32>, reference: ArrayView2<f32>) -> f32 {
    let n = tile.len().max(1) as f32;
    let mean_diff: f32 = reference
        .iter()
        .zip(tile.iter())
        .map(|(r, t)| r - t)
        .sum::<f32>()
        / n;
    (mean_diff / TWO_PI).round() * TWO_PI
}

/// Block-average a complex array by `(dy, dx)` (multilooking) — the coarse-pass
/// downsample. Trailing partial blocks are dropped.
fn downsample_complex(a: ArrayView2<Cf32>, (dy, dx): (usize, usize)) -> Array2<Cf32> {
    let (nr, nc) = (a.nrows() / dy.max(1), a.ncols() / dx.max(1));
    let norm = (dy * dx) as f32;
    Array2::from_shape_fn((nr, nc), |(i, j)| {
        a.slice(s![i * dy..i * dy + dy, j * dx..j * dx + dx]).sum() / norm
    })
}

/// Block-average a real array by `(dy, dx)`.
fn downsample_real(a: ArrayView2<f32>, (dy, dx): (usize, usize)) -> Array2<f32> {
    let (nr, nc) = (a.nrows() / dy.max(1), a.ncols() / dx.max(1));
    let norm = (dy * dx) as f32;
    Array2::from_shape_fn((nr, nc), |(i, j)| {
        a.slice(s![i * dy..i * dy + dy, j * dx..j * dx + dx]).sum() / norm
    })
}

/// Nearest-neighbour upsample of `a` to `(rows, cols)` given the downsample factor.
fn upsample_nearest(
    a: ArrayView2<f32>,
    (rows, cols): (usize, usize),
    (dy, dx): (usize, usize),
) -> Array2<f32> {
    let (last_r, last_c) = (a.nrows().saturating_sub(1), a.ncols().saturating_sub(1));
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        a[((r / dy.max(1)).min(last_r), (c / dx.max(1)).min(last_c))]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Tile regions cover every pixel exactly once at their cores and stay within
    /// the grid when expanded.
    #[test]
    fn tile_regions_tile_the_grid() {
        let shape = (10, 14);
        let regions = tile_regions(shape, (2, 3), (2, 2));
        assert_eq!(regions.len(), 6);
        let mut covered = Array2::<u32>::zeros(shape);
        for reg in &regions {
            let (r0, r1, c0, c1) = reg.core;
            covered
                .slice_mut(s![r0..r1, c0..c1])
                .mapv_inplace(|v| v + 1);
            let (er0, er1, ec0, ec1) = reg.exp;
            assert!(er0 <= r0 && er1 >= r1 && ec0 <= c0 && ec1 >= c1);
            assert!(er1 <= shape.0 && ec1 <= shape.1);
        }
        assert!(covered.iter().all(|&v| v == 1), "every pixel covered once");
    }

    /// The merge resolves a planted inter-tile 2π jump: two tiles over a smooth
    /// ramp, one of which the "solver" landed a full cycle high, both reconcile to
    /// the coarse reference, leaving the shared boundary continuous.
    #[test]
    fn merge_resolves_planted_2pi_jump() {
        let shape = (4, 8);
        // Coarse reference: a smooth horizontal ramp, 1 rad/col.
        let coarse = Array2::from_shape_fn(shape, |(_, c)| c as f32);
        // Two column-halves, each unwrapped to the ramp but the right tile sits a
        // full +2π cycle off (a wrong global branch).
        let regions = tile_regions(shape, (1, 2), (0, 0));
        let tiles: Vec<UnwrappedTile> = regions
            .iter()
            .enumerate()
            .map(|(k, reg)| {
                let (er0, er1, ec0, ec1) = reg.exp;
                let bias = if k == 0 { 0.0 } else { TWO_PI };
                let phase =
                    Array2::from_shape_fn((er1 - er0, ec1 - ec0), |(_, c)| (ec0 + c) as f32 + bias);
                UnwrappedTile {
                    region: *reg,
                    phase,
                }
            })
            .collect();
        let merged = merge_tiles(coarse.view(), shape, &tiles);
        let err = merged
            .iter()
            .zip(coarse.iter())
            .map(|(m, c)| (m - c).abs())
            .fold(0.0_f32, f32::max);
        assert!(err < 1e-4, "planted 2π jump not reconciled: max err {err}");
    }

    /// `cycle_offset` snaps to the nearest whole cycle, ignoring sub-cycle noise.
    #[test]
    fn cycle_offset_snaps_to_nearest_cycle() {
        let tile = Array2::<f32>::zeros((3, 3));
        let reference = Array2::from_elem((3, 3), 2.0 * TWO_PI + 0.3);
        let off = cycle_offset(tile.view(), reference.view());
        assert!((off - 2.0 * TWO_PI).abs() < 1e-4);
    }

    /// Block-average downsample then nearest upsample round-trips a constant.
    #[test]
    fn downsample_upsample_preserve_constant() {
        let a = Array2::<f32>::from_elem((9, 9), 1.5);
        let down = downsample_real(a.view(), (3, 3));
        assert_eq!(down.dim(), (3, 3));
        let up = upsample_nearest(down.view(), (9, 9), (3, 3));
        assert!(up.iter().all(|&v| (v - 1.5).abs() < 1e-6));
    }
}
