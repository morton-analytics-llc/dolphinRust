//! Tiling + inter-tile reconciliation for large interferograms (Chen 2002).
//!
//! Chen & Zebker 2002 unwraps a large interferogram by partitioning it into
//! overlapping tiles, unwrapping each independently, then a secondary
//! optimization reconciles the integer-cycle offsets between adjacent tiles. The
//! reconciliation is *per reliable region*, not per tile: a single integer offset
//! per tile cannot fix a scene whose tiles each straddle several coherent
//! components, nor a coherent region that a seam bisects (the per-tile modal
//! stitch failed on both — > 30 % cycle disagreement on adversarial scenes).
//!
//! We solve each tile independently over an overlapped window, then:
//!   1. partition the grid into **regions** — maximal 4-connected sets of
//!      reliable (coherent) pixels owned by the *same* tile core, so a coherent
//!      component spanning several tiles becomes one region per tile;
//!   2. across every tile seam, each reliable adjacent pixel pair votes for the
//!      integer-cycle offset that makes the two tiles' values agree, weighted by
//!      edge coherence;
//!   3. assign one integer offset per region by propagating the consensus
//!      differences along a **maximum-reliability spanning forest** of the
//!      region-adjacency graph — optimal on the acyclic region backbone, and the
//!      strongest seams win on the rare cyclic inconsistency.
//!
//! Regions in disconnected coherent groups (separated by low-coherence moats)
//! seed independently, mirroring SNAPHU's independent per-component offsets.
//! Verified seam-robust across tile counts 2..=8 (odd and even) in
//! `examples/seam_sweep.rs` and the tiling contract.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use super::CostMode;

const TAU: f64 = std::f64::consts::TAU;
/// Tile overlap (pixels): each tile is solved over a window grown by this much on
/// its interior sides so residues near a core seam route past the seam, not into
/// it. Ownership (and the seams) stay on the non-overlapping core grid.
const OVERLAP: usize = 16;

/// Unwrap `psi` by overlapping tiles, reconciling inter-tile offsets per region.
/// `min_corr` is the coherence threshold below which a pixel is unreliable and
/// excluded from region reconciliation (it still receives its tile's offset).
pub fn unwrap_tiled(
    psi: &Array2<f64>,
    corr: ArrayView2<f32>,
    cost: CostMode,
    tiles: (usize, usize),
    min_corr: f32,
    regrow: usize,
) -> Array2<f64> {
    let (rows, cols) = psi.dim();
    let row_layout = tile_layout(rows, tiles.0);
    let col_layout = tile_layout(cols, tiles.1);

    let specs: Vec<TileSpec> = row_layout
        .iter()
        .flat_map(|&r| col_layout.iter().map(move |&c| TileSpec { row: r, col: c }))
        .collect();
    let solved: Vec<Tile> = specs
        .par_iter()
        .map(|spec| solve_tile(psi, corr, cost, *spec))
        .collect();

    // Region cores are the raw coherence threshold; we then *grow* each region's
    // label into the masked bridge (not merge masks). A thin sub-threshold
    // corridor — the kind SNAPHU's regrow keeps inside one component — is closed
    // from both sides until the two regions abut, so reconciliation finds a seam
    // and votes their true relative offset (from the per-tile field), instead of
    // leaving them disconnected with independent offsets. Merging the masks would
    // force the offset *equal*, which is wrong where the field has a branch cut
    // through the bridge.
    let reliable = corr.mapv(|c| c >= min_corr);
    let owner = Ownership::build(rows, cols, &solved);
    let mut regions = segment_regions(&reliable, &owner);
    grow_regions(&mut regions.label, &owner, regrow);
    let offsets = reconcile(psi, corr, &owner, &regions);
    compose(&owner, &regions, &offsets)
}

/// A tile's overlapped solve window and the non-overlapping core it owns.
#[derive(Clone, Copy)]
struct TileSpec {
    /// `(start, end)` of the solve window / owned core along one axis.
    row: Span,
    col: Span,
}

/// `(window, core)` spans along one axis.
#[derive(Clone, Copy)]
struct Span {
    window: (usize, usize),
    core: (usize, usize),
}

/// Per-axis tile spans: `parts` non-overlapping cores tiling `n`, each paired
/// with an overlapped solve window grown by `OVERLAP` on its interior sides.
fn tile_layout(n: usize, parts: usize) -> Vec<Span> {
    let parts = parts.max(1).min(n.max(1));
    let core = n.div_ceil(parts);
    (0..parts)
        .map(|p| (p * core, ((p + 1) * core).min(n)))
        .take_while(|&(c0, _)| c0 < n)
        .map(|(c0, c1)| Span {
            window: (c0.saturating_sub(OVERLAP), (c1 + OVERLAP).min(n)),
            core: (c0, c1),
        })
        .collect()
}

/// One unwrapped tile: the window it was solved over and that field.
struct Tile {
    win_r: (usize, usize),
    win_c: (usize, usize),
    core_r: (usize, usize),
    core_c: (usize, usize),
    unwrapped: Array2<f64>,
}

/// Unwrap one tile's overlapped window with the global solver.
fn solve_tile(psi: &Array2<f64>, corr: ArrayView2<f32>, cost: CostMode, spec: TileSpec) -> Tile {
    let (wr, wc) = (spec.row.window, spec.col.window);
    let sub_psi = psi.slice(ndarray::s![wr.0..wr.1, wc.0..wc.1]).to_owned();
    let sub_corr = corr.slice(ndarray::s![wr.0..wr.1, wc.0..wc.1]);
    Tile {
        win_r: wr,
        win_c: wc,
        core_r: spec.row.core,
        core_c: spec.col.core,
        unwrapped: super::unwrap_grid(&sub_psi, sub_corr, cost),
    }
}

/// Per-pixel owning-tile index and that tile's local unwrapped value. Cores
/// partition the grid, so every pixel has exactly one owner.
struct Ownership {
    owner: Array2<u32>,
    val: Array2<f64>,
    n_tiles: usize,
}

impl Ownership {
    fn build(rows: usize, cols: usize, tiles: &[Tile]) -> Self {
        let mut owner = Array2::<u32>::zeros((rows, cols));
        let mut val = Array2::<f64>::zeros((rows, cols));
        for (t, tile) in tiles.iter().enumerate() {
            claim_core(t as u32, tile, &mut owner, &mut val);
        }
        Self {
            owner,
            val,
            n_tiles: tiles.len(),
        }
    }
}

/// Stamp tile `t`'s core pixels with its index and local unwrapped value.
fn claim_core(t: u32, tile: &Tile, owner: &mut Array2<u32>, val: &mut Array2<f64>) {
    let cells = (tile.core_r.0..tile.core_r.1)
        .flat_map(|gi| (tile.core_c.0..tile.core_c.1).map(move |gj| (gi, gj)));
    for (gi, gj) in cells {
        owner[(gi, gj)] = t;
        val[(gi, gj)] = tile.unwrapped[(gi - tile.win_r.0, gj - tile.win_c.0)];
    }
}

/// Reliable-region labels (`0` = unreliable/masked): 4-connected coherent pixels
/// sharing the same owning tile. A coherent component spanning tiles splits into
/// one region per tile, each independently reconciled.
fn segment_regions(reliable: &Array2<bool>, own: &Ownership) -> Regions {
    let (rows, cols) = reliable.dim();
    let mut label = Array2::<u32>::zeros((rows, cols));
    let mut sizes = vec![0usize]; // index 0 = masked
    let mut tile_of = vec![u32::MAX]; // owning tile per region label
    let mut next = 1u32;
    for seed in (0..rows).flat_map(|i| (0..cols).map(move |j| (i, j))) {
        if !reliable[seed] || label[seed] != 0 {
            continue;
        }
        let size = flood_region(reliable, own, &mut label, seed, next);
        sizes.push(size);
        tile_of.push(own.owner[seed]);
        next += 1;
    }
    Regions {
        label,
        sizes,
        tile_of,
    }
}

/// Reliable-region partition and per-region bookkeeping.
struct Regions {
    label: Array2<u32>,
    sizes: Vec<usize>,
    tile_of: Vec<u32>,
}

/// Flood a region from `seed`: 4-connected reliable pixels with the same owner.
fn flood_region(
    reliable: &Array2<bool>,
    own: &Ownership,
    label: &mut Array2<u32>,
    seed: (usize, usize),
    id: u32,
) -> usize {
    let (rows, cols) = reliable.dim();
    let mine = own.owner[seed];
    let mut stack = vec![seed];
    label[seed] = id;
    let mut size = 0usize;
    while let Some((i, j)) = stack.pop() {
        size += 1;
        for (ni, nj) in neighbors(i, j, rows, cols) {
            let ok = reliable[(ni, nj)] && label[(ni, nj)] == 0 && own.owner[(ni, nj)] == mine;
            if ok {
                label[(ni, nj)] = id;
                stack.push((ni, nj));
            }
        }
    }
    size
}

/// Grow region labels into adjacent masked pixels by `r` 4-connected steps,
/// each masked pixel adopting a same-owner neighbour's region. Two regions
/// flanking a sub-threshold corridor ≤ `2r` px wide thus grow until they abut,
/// giving reconciliation a seam (and the correct relative offset) between them
/// instead of leaving them disconnected. Mirrors SNAPHU's component regrow, but
/// keeps the regions distinct so the offset is *voted*, not forced equal.
fn grow_regions(label: &mut Array2<u32>, own: &Ownership, r: usize) {
    for _ in 0..r {
        let prev = label.clone();
        let masked = prev
            .indexed_iter()
            .filter(|&(_, &l)| l == 0)
            .map(|(ix, _)| ix);
        for ix in masked {
            label[ix] = adopt_label(&prev, own, ix);
        }
    }
}

/// The label a masked pixel adopts when growing: the smallest same-owner
/// 4-neighbour region label, or `0` if it has no labelled same-owner neighbour
/// (so isolated masked pixels stay masked). The smallest-label tie-break keeps
/// growth deterministic and order-independent.
fn adopt_label(label: &Array2<u32>, own: &Ownership, (i, j): (usize, usize)) -> u32 {
    let (rows, cols) = label.dim();
    let mine = own.owner[(i, j)];
    neighbors(i, j, rows, cols)
        .filter(|&n| label[n] != 0 && own.owner[n] == mine)
        .map(|n| label[n])
        .min()
        .unwrap_or(0)
}

/// In-bounds 4-neighbours of `(i, j)`.
fn neighbors(i: usize, j: usize, rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize)> {
    let up = (i > 0).then(|| (i - 1, j));
    let down = (i + 1 < rows).then_some((i + 1, j));
    let left = (j > 0).then(|| (i, j - 1));
    let right = (j + 1 < cols).then_some((i, j + 1));
    [up, down, left, right].into_iter().flatten()
}

/// One reliable inter-region adjacency: the canonical region pair, the
/// consensus integer offset `o[lo] - o[hi]`, and its accumulated reliability.
struct Seam {
    lo: u32,
    hi: u32,
    diff: i64,
    weight: f64,
}

/// Assign one integer-cycle offset per region by propagating consensus seam
/// differences along a maximum-reliability spanning forest.
fn reconcile(psi: &Array2<f64>, corr: ArrayView2<f32>, own: &Ownership, reg: &Regions) -> Vec<i64> {
    let seams = collect_seams(psi, corr, own, &reg.label);
    spanning_offsets(reg.sizes.len(), &seams)
}

/// Per-region-pair vote tally: `votes[(lo, hi)][d]` = coherence weight backing
/// the offset difference `o[lo] - o[hi] == d`.
type Votes = std::collections::HashMap<(u32, u32), std::collections::HashMap<i64, f64>>;

/// Tally, per adjacent region pair, the coherence-weighted vote for the integer
/// offset that reconciles the two tiles' values across the seam, then reduce each
/// pair to its consensus difference and total reliability.
fn collect_seams(
    psi: &Array2<f64>,
    corr: ArrayView2<f32>,
    own: &Ownership,
    label: &Array2<u32>,
) -> Vec<Seam> {
    let (rows, cols) = psi.dim();
    let mut votes = Votes::new();
    // o[r] - o[s] making val[q]+2pi o[s] - (val[p]+2pi o[r]) == wrap(psi_q - psi_p).
    let mut tally = |p: (usize, usize), q: (usize, usize)| {
        let (r, s) = (label[p], label[q]);
        if s == 0 || s == r {
            return;
        }
        let grad = wrap_to_pi(psi[q] - psi[p]);
        let d_rs = (((own.val[q] - own.val[p]) - grad) / TAU).round() as i64;
        let w = corr[p].min(corr[q]) as f64;
        let (lo, hi, diff) = if r < s { (r, s, d_rs) } else { (s, r, -d_rs) };
        *votes
            .entry((lo, hi))
            .or_default()
            .entry(diff)
            .or_insert(0.0) += w;
    };
    let cells = (0..rows).flat_map(|i| (0..cols).map(move |j| (i, j)));
    for (i, j) in cells.filter(|&p| label[p] != 0) {
        for q in [
            (i + 1 < rows).then_some((i + 1, j)),
            (j + 1 < cols).then_some((i, j + 1)),
        ]
        .into_iter()
        .flatten()
        {
            tally((i, j), q);
        }
    }
    votes
        .into_iter()
        .map(|((lo, hi), tally)| {
            let (diff, _) = tally
                .iter()
                .max_by(|a, b| {
                    a.1.total_cmp(b.1)
                        .then_with(|| b.0.abs().cmp(&a.0.abs()))
                        .then_with(|| b.0.cmp(a.0))
                })
                .unwrap();
            Seam {
                lo,
                hi,
                diff: *diff,
                weight: tally.values().sum(),
            }
        })
        .collect()
}

/// Wrap a phase difference into `(-pi, pi]` (local copy of `native::wrap`).
fn wrap_to_pi(x: f64) -> f64 {
    x - TAU * (x / TAU).round()
}

/// Kruskal max-reliability spanning forest over regions, then propagate offsets
/// from each tree root (offset 0). Returns `offset[region]`, index 0 unused.
fn spanning_offsets(n_regions: usize, seams: &[Seam]) -> Vec<i64> {
    let mut order: Vec<usize> = (0..seams.len()).collect();
    // `(lo, hi)` is unique per seam (one `Seam` per `votes` key), so weight
    // then endpoints is already a total order.
    order.sort_by(|&a, &b| {
        seams[b]
            .weight
            .total_cmp(&seams[a].weight)
            .then_with(|| seams[a].lo.cmp(&seams[b].lo))
            .then_with(|| seams[a].hi.cmp(&seams[b].hi))
    });
    let mut uf = UnionFind::new(n_regions);
    let mut adj: Vec<Vec<(u32, i64)>> = vec![Vec::new(); n_regions];
    for &e in &order {
        let s = &seams[e];
        if uf.union(s.lo as usize, s.hi as usize) {
            // o[hi] = o[lo] - diff ;  o[lo] = o[hi] + diff.
            adj[s.lo as usize].push((s.hi, -s.diff));
            adj[s.hi as usize].push((s.lo, s.diff));
        }
    }
    propagate(&adj)
}

/// DFS each spanning-tree component from offset 0, accumulating edge deltas.
fn propagate(adj: &[Vec<(u32, i64)>]) -> Vec<i64> {
    let mut offset = vec![0i64; adj.len()];
    let mut seen = vec![false; adj.len()];
    for root in 1..adj.len() {
        if !seen[root] {
            dfs_offsets(root as u32, adj, &mut offset, &mut seen);
        }
    }
    offset
}

/// Iterative DFS from `root` (offset already set), assigning each tree child its
/// parent's offset plus the connecting edge delta.
fn dfs_offsets(root: u32, adj: &[Vec<(u32, i64)>], offset: &mut [i64], seen: &mut [bool]) {
    seen[root as usize] = true;
    let mut stack = vec![root];
    while let Some(u) = stack.pop() {
        for &(v, inc) in &adj[u as usize] {
            if seen[v as usize] {
                continue;
            }
            seen[v as usize] = true;
            offset[v as usize] = offset[u as usize] + inc;
            stack.push(v);
        }
    }
}

/// Union-find with union-by-size; `union` returns true if it merged two sets
/// (i.e. the edge joins the spanning forest).
struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a as u32), self.find(b as u32));
        if ra == rb {
            return false;
        }
        let (big, small) = if self.size[ra as usize] >= self.size[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = big;
        self.size[big as usize] += self.size[small as usize];
        true
    }
}

/// Compose the output field: reliable pixels take their region's offset, masked
/// pixels take their owning tile's representative offset (its largest region).
fn compose(own: &Ownership, reg: &Regions, offset: &[i64]) -> Array2<f64> {
    let tile_off = tile_offsets(own.n_tiles, reg, offset);
    let mut out = own.val.clone();
    for ((i, j), v) in out.indexed_iter_mut() {
        let r = reg.label[(i, j)] as usize;
        let k = if r != 0 {
            offset[r]
        } else {
            tile_off[own.owner[(i, j)] as usize]
        };
        *v += TAU * k as f64;
    }
    out
}

/// Each tile's representative offset: that of its largest reliable region (0 if
/// the tile has none), so masked pixels stay continuous with their tile.
fn tile_offsets(n_tiles: usize, reg: &Regions, offset: &[i64]) -> Vec<i64> {
    let mut best = vec![0usize; n_tiles]; // largest region size seen per tile
    let mut off = vec![0i64; n_tiles];
    let regions = reg.sizes.iter().zip(&reg.tile_of).zip(offset).skip(1);
    for ((&size, &tile), &o) in regions {
        let t = tile as usize;
        if size > best[t] {
            best[t] = size;
            off[t] = o;
        }
    }
    off
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_weight_seam_votes_prefer_smaller_cycle_offset() {
        // Two regions split by a vertical seam; row 0 votes diff=0, row 1
        // votes diff=1 (val jump of TAU), equal coherence weight — the tied
        // vote must resolve to the smaller |diff|, not HashMap order.
        let psi = Array2::<f64>::zeros((2, 2));
        let corr = Array2::<f32>::from_elem((2, 2), 1.0);
        let own = Ownership {
            owner: Array2::<u32>::zeros((2, 2)),
            val: Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, TAU]).unwrap(),
            n_tiles: 2,
        };
        let label = Array2::from_shape_vec((2, 2), vec![1u32, 2, 1, 2]).unwrap();

        let seams = collect_seams(&psi, corr.view(), &own, &label);

        assert_eq!(seams.len(), 1);
        assert_eq!((seams[0].lo, seams[0].hi), (1, 2));
        assert_eq!(
            seams[0].diff, 0,
            "tied vote {{0: 1.0, 1: 1.0}} must deterministically pick diff 0"
        );
        assert_eq!(seams[0].weight, 2.0);
    }

    #[test]
    fn equal_weight_seams_are_order_independent() {
        let seams = [
            Seam {
                lo: 1,
                hi: 2,
                diff: 0,
                weight: 1.0,
            },
            Seam {
                lo: 2,
                hi: 3,
                diff: 0,
                weight: 1.0,
            },
            Seam {
                lo: 1,
                hi: 3,
                diff: 1,
                weight: 1.0,
            },
        ];
        let reversed = [
            Seam {
                lo: 1,
                hi: 3,
                diff: 1,
                weight: 1.0,
            },
            Seam {
                lo: 2,
                hi: 3,
                diff: 0,
                weight: 1.0,
            },
            Seam {
                lo: 1,
                hi: 2,
                diff: 0,
                weight: 1.0,
            },
        ];

        assert_eq!(
            spanning_offsets(4, &seams),
            spanning_offsets(4, &reversed),
            "equal-reliability seam cycles must not depend on HashMap iteration order"
        );
    }
}
