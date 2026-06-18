//! tophu-style multi-scale unwrapping, driven over the SNAPHU wrapper.
//!
//! Raw SNAPHU degrades on large, low-coherence scenes: its network-flow solver
//! can pick the wrong integer-cycle branch in vegetated/decorrelated regions far
//! from any reliable reference. tophu's remedy is a coarse→fine cascade:
//!
//! 1. **coarse** — multilook (downsample) the wrapped ifg + correlation by
//!    `downsample_factor` and unwrap that small grid once. The multilook is
//!    **coherence-weighted** (each phasor is weighted by its correlation), so
//!    decorrelated pixels do not poison the block average; blocks whose resulting
//!    vector coherence falls below a trust floor are masked and filled from
//!    trusted neighbours rather than anchoring downstream work to garbage.
//! 2. **upsample** — nearest-neighbour the coarse unwrapped phase back to full
//!    resolution. This is the absolute-ambiguity reference used to fix the one
//!    global integer cycle of the merged surface.
//! 3. **tiled fine** — split the full-res grid into overlapping tiles and unwrap
//!    each independently (in parallel). Each tile is internally consistent but
//!    carries an arbitrary global 2π offset.
//! 4. **merge** — reconcile adjacent tiles by the integer number of 2π cycles
//!    estimated from their *overlap region* (robust median over coherent overlap
//!    pixels), then solve the consistent set of per-tile offsets across the
//!    tile-adjacency graph (a spanning forest over the overlap graph). Each
//!    connected component is then anchored to the coarse reference by a single
//!    global integer cycle. This replaces the old per-tile snap-to-coarse, which
//!    injected cross-tile cycle errors wherever the coarse field was noisy.
//!
//! This is heuristic orchestration over SNAPHU, not new unwrap math — the
//! reference is tophu's *algorithm* and *result quality*, not bit-parity (its own
//! tiling/merge is non-unique). dolphin reserves its `tophu.multiscale_unwrap`
//! driver for the ICU/PHASS per-tile solvers; dolphinRust drives the SNAPHU
//! wrapper per tile, which is the solver we ship.

use std::collections::VecDeque;
use std::path::Path;

use dolphin_core::Cf32;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

use crate::snaphu::{unwrap, CostMode, InitMethod, Result, UnwrapConfig, UnwrapResult};

const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
/// Coarse blocks whose resulting vector coherence is below this are untrusted and
/// filled from trusted neighbours instead of being used as an anchor.
const COARSE_TRUST_FLOOR: f32 = 0.4;
/// Overlap pixels below this correlation are excluded from inter-tile offset
/// estimation (decorrelated overlap pixels carry no reliable cycle information).
const OVERLAP_COH_FLOOR: f32 = 0.5;

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
    let unwrapped = merge_tiles(
        coarse_up.view(),
        (rows, cols),
        &tiles,
        cfg.ntiles,
        correlation,
    );
    // The merged solution is a single coarse-anchored surface; SNAPHU's per-tile
    // connected components are not re-stitched, so all pixels carry label 1.
    let conncomp = Array2::<u32>::ones((rows, cols));
    Ok(UnwrapResult {
        unwrapped,
        conncomp,
    })
}

/// Coarse pass: coherence-weighted downsample, unwrap once, mask+fill untrusted
/// blocks, upsample back to full resolution as the absolute-ambiguity reference.
fn coarse_reference(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    cfg: &TophuConfig,
    scratch: &Path,
) -> Result<Array2<f32>> {
    let factor = cfg.downsample_factor;
    let (coarse_ifg, coarse_coh) = downsample_weighted(wrapped, correlation, factor);
    let dir = scratch.join("coarse");
    std::fs::create_dir_all(&dir)?;
    let coarse = unwrap(
        coarse_ifg.view(),
        coarse_coh.view(),
        &single_tile_cfg(cfg),
        &dir,
    )?;
    let filled = fill_untrusted(coarse.unwrapped, coarse_coh.view(), COARSE_TRUST_FLOOR);
    Ok(upsample_nearest(filled.view(), wrapped.dim(), factor))
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

/// Reconcile tiles by overlap-region integer-cycle offsets, anchor each connected
/// component to the coarse reference, and paste cores into the full-res output.
fn merge_tiles(
    coarse_up: ArrayView2<f32>,
    shape: (usize, usize),
    tiles: &[UnwrappedTile],
    ntiles: (usize, usize),
    correlation: ArrayView2<f32>,
) -> Array2<f32> {
    let (rel, comp) = solve_tile_offsets(tiles, ntiles, correlation);
    let ncomp = comp.iter().copied().max().map_or(0, |m| m + 1);
    let anchor = component_anchors(coarse_up, tiles, &rel, &comp, ncomp);
    let mut acc = Array2::<f32>::zeros(shape);
    let mut wsum = Array2::<f32>::zeros(shape);
    for (k, tile) in tiles.iter().enumerate() {
        let offset = rel[k] + anchor[comp[k]];
        blend_tile(&mut acc, &mut wsum, tile, offset);
    }
    // Every pixel lies in exactly one tile core (weight 1 there), so wsum > 0.
    acc / wsum
}

/// Feather-blend one offset-aligned tile into the accumulators over its full
/// expanded region: the weight is 1 across the tile's core and ramps linearly to
/// ~0 at the halo fringe, so each tile contributes most where it is most reliable
/// and tile seams are smoothed instead of hard-pasted.
fn blend_tile(acc: &mut Array2<f32>, wsum: &mut Array2<f32>, tile: &UnwrappedTile, offset: f32) {
    let (er0, er1, ec0, ec1) = tile.region.exp;
    let core = tile.region.core;
    let exp = tile.region.exp;
    for r in er0..er1 {
        for c in ec0..ec1 {
            let w = feather_weight(r, c, core, exp);
            acc[(r, c)] += w * (tile.phase[(r - er0, c - ec0)] + offset);
            wsum[(r, c)] += w;
        }
    }
}

/// Feather weight at `(r, c)`: the product of per-axis ramps that are 1 within the
/// `core` span and decay linearly to ~0 across the halo out to the `exp` edge.
fn feather_weight(
    r: usize,
    c: usize,
    core: (usize, usize, usize, usize),
    exp: (usize, usize, usize, usize),
) -> f32 {
    let ramp = |x: usize, lo: usize, hi: usize, elo: usize, ehi: usize| -> f32 {
        if x < lo {
            (x - elo + 1) as f32 / (lo - elo + 1) as f32
        } else if x >= hi {
            (ehi - x) as f32 / (ehi - hi) as f32
        } else {
            1.0
        }
    };
    ramp(r, core.0, core.1, exp.0, exp.1) * ramp(c, core.2, core.3, exp.2, exp.3)
}

/// Minimum number of agreeing coherent overlap pixels for an inter-tile edge to be
/// trusted for offset propagation. Edges below this (e.g. overlaps that straddle a
/// decorrelation band) are dropped, leaving the tiles in separate components each
/// anchored independently to the coarse reference rather than propagating a guess.
const MIN_OVERLAP_AGREE: usize = 16;

/// A trusted inter-tile constraint: `O_i - O_j = cycles` with a reliability weight
/// (count of agreeing coherent overlap pixels) used to build the spanning tree.
struct TileEdge {
    i: usize,
    j: usize,
    cycles: f32,
    weight: usize,
}

/// Solve per-tile 2π offsets that make adjacent tiles mutually consistent. Edges
/// carry an overlap-derived integer-cycle constraint and a reliability weight; a
/// **maximum-reliability spanning forest** (Kruskal) propagates offsets only
/// through the most coherent overlaps, so a single decorrelated overlap cannot
/// shift a whole subtree. Returns `(relative_offset, component_id)` per tile.
fn solve_tile_offsets(
    tiles: &[UnwrappedTile],
    ntiles: (usize, usize),
    correlation: ArrayView2<f32>,
) -> (Vec<f32>, Vec<usize>) {
    let n = tiles.len();
    let mut edges: Vec<TileEdge> = grid_edges(ntiles)
        .into_iter()
        .filter_map(|(i, j)| {
            overlap_edge(&tiles[i], &tiles[j], correlation).map(|(cycles, weight)| TileEdge {
                i,
                j,
                cycles,
                weight,
            })
        })
        .filter(|e| e.weight >= MIN_OVERLAP_AGREE)
        .collect();
    edges.sort_by(|a, b| b.weight.cmp(&a.weight));

    let mut uf = UnionFind::new(n);
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for e in &edges {
        if !uf.union(e.i, e.j) {
            continue;
        }
        adj[e.i].push((e.j, e.cycles));
        adj[e.j].push((e.i, -e.cycles));
    }
    bfs_forest(&adj, n)
}

/// Union-find for Kruskal's maximum-reliability spanning forest over tile edges.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        self.parent[x] = root;
        root
    }

    /// Merge the sets of `a` and `b`; returns `false` if already joined (cycle).
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.parent[ra] = rb;
        true
    }
}

/// Right/down neighbour tile-index pairs of an `ntiles` grid (row-major indexing).
fn grid_edges((tr, tc): (usize, usize)) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for ti in 0..tr {
        for tj in 0..tc {
            let i = ti * tc + tj;
            if tj + 1 < tc {
                edges.push((i, i + 1));
            }
            if ti + 1 < tr {
                edges.push((i, i + tc));
            }
        }
    }
    edges
}

/// Assign each tile a relative 2π offset (root = 0) and a component id by BFS over
/// the constraint graph. Edge constraint `(v, c)` on node `u` means `O_u - O_v = c`.
fn bfs_forest(adj: &[Vec<(usize, f32)>], n: usize) -> (Vec<f32>, Vec<usize>) {
    let mut offset = vec![0.0_f32; n];
    let mut comp = vec![usize::MAX; n];
    let mut cid = 0;
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        bfs_component(adj, start, cid, &mut comp, &mut offset);
        cid += 1;
    }
    (offset, comp)
}

/// BFS one component from `start`, propagating relative offsets along constraints.
fn bfs_component(
    adj: &[Vec<(usize, f32)>],
    start: usize,
    cid: usize,
    comp: &mut [usize],
    offset: &mut [f32],
) {
    comp[start] = cid;
    offset[start] = 0.0;
    let mut queue = VecDeque::from([start]);
    while let Some(u) = queue.pop_front() {
        for &(v, c) in &adj[u] {
            if comp[v] != usize::MAX {
                continue;
            }
            comp[v] = cid;
            offset[v] = offset[u] - c;
            queue.push_back(v);
        }
    }
}

/// Inter-tile constraint between two adjacent tiles: the integer-cycle offset
/// `O_a - O_b` (a multiple of 2π) from the robust median of `phase_b - phase_a`
/// over their coherent overlap, plus a reliability weight = count of coherent
/// overlap pixels that agree with that cycle. `None` when the tiles do not overlap.
fn overlap_edge(
    a: &UnwrappedTile,
    b: &UnwrappedTile,
    correlation: ArrayView2<f32>,
) -> Option<(f32, usize)> {
    let diffs = overlap_diffs(a, b, correlation)?;
    let cycle = (median(diffs.clone()) / TWO_PI).round();
    let agree = diffs
        .iter()
        .filter(|&&d| (d / TWO_PI).round() == cycle)
        .count();
    Some((cycle * TWO_PI, agree))
}

/// `phase_b - phase_a` over the geometric intersection of the two expanded
/// regions, restricted to coherent pixels (falling back to all overlap pixels if
/// none clear the coherence floor). `None` when the regions do not overlap.
fn overlap_diffs(
    a: &UnwrappedTile,
    b: &UnwrappedTile,
    correlation: ArrayView2<f32>,
) -> Option<Vec<f32>> {
    let (r0, r1, c0, c1) = intersect(a.region.exp, b.region.exp);
    if r0 >= r1 || c0 >= c1 {
        return None;
    }
    let (ar0, _, ac0, _) = a.region.exp;
    let (br0, _, bc0, _) = b.region.exp;
    let mut coherent = Vec::new();
    let mut all = Vec::new();
    for r in r0..r1 {
        for c in c0..c1 {
            let d = b.phase[(r - br0, c - bc0)] - a.phase[(r - ar0, c - ac0)];
            all.push(d);
            if correlation[(r, c)] > OVERLAP_COH_FLOOR {
                coherent.push(d);
            }
        }
    }
    Some(if coherent.is_empty() { all } else { coherent })
}

/// Intersection rectangle `[r0, r1) × [c0, c1)` of two `(r0, r1, c0, c1)` regions.
fn intersect(
    a: (usize, usize, usize, usize),
    b: (usize, usize, usize, usize),
) -> (usize, usize, usize, usize) {
    (a.0.max(b.0), a.1.min(b.1), a.2.max(b.2), a.3.min(b.3))
}

/// Median of a non-empty slice of finite values.
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Per-component integer-cycle offset that aligns the relatively-reconciled tiles
/// to the coarse reference (the single global ambiguity per connected component).
fn component_anchors(
    coarse_up: ArrayView2<f32>,
    tiles: &[UnwrappedTile],
    rel: &[f32],
    comp: &[usize],
    ncomp: usize,
) -> Vec<f32> {
    let mut sum = vec![0.0_f32; ncomp];
    let mut count = vec![0_usize; ncomp];
    for (k, tile) in tiles.iter().enumerate() {
        let (er0, _, ec0, _) = tile.region.exp;
        let (r0, r1, c0, c1) = tile.region.core;
        let core = tile.phase.slice(s![r0 - er0..r1 - er0, c0 - ec0..c1 - ec0]);
        let reference = coarse_up.slice(s![r0..r1, c0..c1]);
        sum[comp[k]] += reference
            .iter()
            .zip(core.iter())
            .map(|(cu, p)| cu - (p + rel[k]))
            .sum::<f32>();
        count[comp[k]] += core.len();
    }
    sum.iter()
        .zip(count.iter())
        .map(|(&s, &n)| ((s / n.max(1) as f32) / TWO_PI).round() * TWO_PI)
        .collect()
}

/// Coherence-weighted block average of the complex ifg by `(dy, dx)`: each phasor
/// is weighted by its correlation, so decorrelated pixels do not poison the block.
/// Returns the coarse ifg and its resulting vector coherence `|Σ w z| / Σ w`.
fn downsample_weighted(
    ifg: ArrayView2<Cf32>,
    corr: ArrayView2<f32>,
    (dy, dx): (usize, usize),
) -> (Array2<Cf32>, Array2<f32>) {
    let (dy, dx) = (dy.max(1), dx.max(1));
    let (nr, nc) = (ifg.nrows() / dy, ifg.ncols() / dx);
    let mut coarse = Array2::<Cf32>::zeros((nr, nc));
    let mut coherence = Array2::<f32>::zeros((nr, nc));
    for i in 0..nr {
        for j in 0..nc {
            let block_ifg = ifg.slice(s![i * dy..i * dy + dy, j * dx..j * dx + dx]);
            let block_corr = corr.slice(s![i * dy..i * dy + dy, j * dx..j * dx + dx]);
            let (z, coh) = weighted_block(block_ifg, block_corr);
            coarse[(i, j)] = z;
            coherence[(i, j)] = coh;
        }
    }
    (coarse, coherence)
}

/// Coherence-weighted mean phasor of one block and its resulting vector coherence.
fn weighted_block(ifg: ArrayView2<Cf32>, corr: ArrayView2<f32>) -> (Cf32, f32) {
    let wsum: f32 = corr.iter().sum();
    let vsum: Cf32 = ifg
        .iter()
        .zip(corr.iter())
        .map(|(z, &w)| *z * w)
        .fold(Cf32::new(0.0, 0.0), |acc, z| acc + z);
    let denom = wsum.max(1e-12);
    (vsum / denom, vsum.norm() / denom)
}

/// Replace coarse pixels below `floor` coherence with the mean of their trusted
/// neighbours, propagating outward from the trusted region until none remain (or
/// no untrusted pixel has a trusted neighbour). Leaves an all-untrusted grid as-is.
fn fill_untrusted(mut phase: Array2<f32>, coherence: ArrayView2<f32>, floor: f32) -> Array2<f32> {
    let (nr, nc) = phase.dim();
    let mut trusted = Array2::from_shape_fn((nr, nc), |(r, c)| coherence[(r, c)] >= floor);
    if !trusted.iter().any(|&t| t) {
        return phase;
    }
    loop {
        let fills = pending_fills(&phase, &trusted);
        if fills.is_empty() {
            break;
        }
        for (r, c, v) in fills {
            phase[(r, c)] = v;
            trusted[(r, c)] = true;
        }
    }
    phase
}

/// Untrusted pixels that have at least one trusted 4-neighbour, with the fill value
/// (mean of those trusted neighbours).
fn pending_fills(phase: &Array2<f32>, trusted: &Array2<bool>) -> Vec<(usize, usize, f32)> {
    let (nr, nc) = phase.dim();
    let mut out = Vec::new();
    for r in 0..nr {
        for c in 0..nc {
            if trusted[(r, c)] {
                continue;
            }
            if let Some(v) = trusted_neighbour_mean(phase, trusted, (r, c), (nr, nc)) {
                out.push((r, c, v));
            }
        }
    }
    out
}

/// Mean of the trusted 4-neighbours of `(r, c)`, or `None` if it has none.
fn trusted_neighbour_mean(
    phase: &Array2<f32>,
    trusted: &Array2<bool>,
    (r, c): (usize, usize),
    (nr, nc): (usize, usize),
) -> Option<f32> {
    let neigh = [
        (r.wrapping_sub(1), c),
        (r + 1, c),
        (r, c.wrapping_sub(1)),
        (r, c + 1),
    ];
    let (mut sum, mut n) = (0.0_f32, 0_u32);
    for (rr, cc) in neigh {
        if rr < nr && cc < nc && trusted[(rr, cc)] {
            sum += phase[(rr, cc)];
            n += 1;
        }
    }
    (n > 0).then(|| sum / n as f32)
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

    /// Deterministic splitmix64 → phase in [0, 2π) for synthetic decorrelation.
    fn rand_phase(state: &mut u64) -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let u = ((z ^ (z >> 31)) >> 11) as f32 / (1u64 << 53) as f32;
        u * TWO_PI
    }

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
    /// ramp, one of which the "solver" landed a full cycle high. With zero overlap
    /// they form separate components, each anchored to the coarse reference,
    /// leaving the shared boundary continuous.
    #[test]
    fn merge_resolves_planted_2pi_jump() {
        let shape = (4, 8);
        // Coarse reference: a smooth horizontal ramp, 1 rad/col.
        let coarse = Array2::from_shape_fn(shape, |(_, c)| c as f32);
        let corr = Array2::<f32>::from_elem(shape, 1.0);
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
        let merged = merge_tiles(coarse.view(), shape, &tiles, (1, 2), corr.view());
        let err = merged
            .iter()
            .zip(coarse.iter())
            .map(|(m, c)| (m - c).abs())
            .fold(0.0_f32, f32::max);
        assert!(err < 1e-4, "planted 2π jump not reconciled: max err {err}");
    }

    /// Loop consistency across a 2×2 overlapping tile grid: each tile carries an
    /// arbitrary integer-cycle bias; the overlap graph solve reconciles all four to
    /// a single consistent surface (truth up to one global constant).
    #[test]
    fn merge_reconciles_2x2_grid_consistently() {
        let shape = (16, 16);
        let truth = Array2::from_shape_fn(shape, |(r, c)| 0.5 * r as f32 + 0.3 * c as f32);
        let corr = Array2::<f32>::from_elem(shape, 1.0);
        let regions = tile_regions(shape, (2, 2), (4, 4));
        let biases = [0.0, TWO_PI, -2.0 * TWO_PI, 3.0 * TWO_PI];
        let tiles: Vec<UnwrappedTile> = regions
            .iter()
            .enumerate()
            .map(|(k, reg)| {
                let (er0, er1, ec0, ec1) = reg.exp;
                let phase = Array2::from_shape_fn((er1 - er0, ec1 - ec0), |(r, c)| {
                    truth[(er0 + r, ec0 + c)] + biases[k]
                });
                UnwrappedTile {
                    region: *reg,
                    phase,
                }
            })
            .collect();
        let merged = merge_tiles(truth.view(), shape, &tiles, (2, 2), corr.view());
        let off = merged[(0, 0)] - truth[(0, 0)];
        let maxerr = merged
            .iter()
            .zip(truth.iter())
            .map(|(m, t)| ((m - off) - t).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            maxerr < 1e-3,
            "2×2 grid not reconciled consistently: {maxerr}"
        );
    }

    /// The coherence-weighted coarse multilook tracks truth better than an
    /// unweighted block mean when half of each block is decorrelated.
    #[test]
    fn coherence_weighted_coarse_tracks_truth_better() {
        let (rows, cols) = (12, 12);
        let factor = (4, 4);
        let truth = |r: usize, c: usize| 0.1 * (r as f32 + c as f32);
        let mut seed = 0xC0FF_EE00_u64;
        let mut corr = Array2::<f32>::zeros((rows, cols));
        let ifg = Array2::from_shape_fn((rows, cols), |(r, c)| {
            let decorr = (r * cols + c) % 2 == 0;
            let (w, phase) = match decorr {
                true => (0.1, rand_phase(&mut seed)),
                false => (0.9, truth(r, c)),
            };
            corr[(r, c)] = w;
            Cf32::from_polar(1.0, phase)
        });
        let (weighted, _) = downsample_weighted(ifg.view(), corr.view(), factor);
        let unweighted = unweighted_block_mean(ifg.view(), factor);

        let (mut werr, mut uerr) = (0.0_f32, 0.0_f32);
        for i in 0..weighted.nrows() {
            for j in 0..weighted.ncols() {
                let block_truth = truth(i * factor.0 + factor.0 / 2, j * factor.1 + factor.1 / 2);
                werr += wrap(weighted[(i, j)].arg() - block_truth).abs();
                uerr += wrap(unweighted[(i, j)].arg() - block_truth).abs();
            }
        }
        assert!(
            werr < uerr,
            "weighted coarse phase ({werr}) not closer to truth than unweighted ({uerr})"
        );
    }

    /// Unweighted complex block mean (the pre-fix coarse multilook), for comparison.
    fn unweighted_block_mean(a: ArrayView2<Cf32>, (dy, dx): (usize, usize)) -> Array2<Cf32> {
        let (nr, nc) = (a.nrows() / dy, a.ncols() / dx);
        let norm = (dy * dx) as f32;
        Array2::from_shape_fn((nr, nc), |(i, j)| {
            a.slice(s![i * dy..i * dy + dy, j * dx..j * dx + dx]).sum() / norm
        })
    }

    /// Wrap an angle to (-π, π].
    fn wrap(x: f32) -> f32 {
        let mut y = x % TWO_PI;
        if y > std::f32::consts::PI {
            y -= TWO_PI;
        }
        if y < -std::f32::consts::PI {
            y += TWO_PI;
        }
        y
    }

    /// `fill_untrusted` propagates trusted values into a masked low-coherence hole.
    #[test]
    fn fill_untrusted_fills_low_coherence_holes() {
        let phase =
            Array2::from_shape_fn((3, 3), |(r, c)| if (r, c) == (1, 1) { 99.0 } else { 1.0 });
        let coherence =
            Array2::from_shape_fn((3, 3), |(r, c)| if (r, c) == (1, 1) { 0.0 } else { 0.9 });
        let filled = fill_untrusted(phase, coherence.view(), COARSE_TRUST_FLOOR);
        assert!(
            (filled[(1, 1)] - 1.0).abs() < 1e-6,
            "hole not filled from trusted neighbours: {}",
            filled[(1, 1)]
        );
    }

    /// Nearest-neighbour upsample replicates each coarse cell across its block.
    #[test]
    fn upsample_nearest_replicates_blocks() {
        let a = Array2::from_shape_fn((2, 2), |(r, c)| (2 * r + c) as f32);
        let up = upsample_nearest(a.view(), (4, 4), (2, 2));
        assert_eq!(up.dim(), (4, 4));
        assert!((up[(0, 0)] - 0.0).abs() < 1e-6 && (up[(1, 1)] - 0.0).abs() < 1e-6);
        assert!((up[(3, 3)] - 3.0).abs() < 1e-6 && (up[(2, 2)] - 3.0).abs() < 1e-6);
    }
}
