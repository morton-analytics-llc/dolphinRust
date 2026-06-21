//! Minimum-cost-flow branch-cut router for residue-bearing fields.
//!
//! Independent implementation of the Costantini (1998) minimum-cost-flow
//! formulation: integer corrections `k` on the wrapped gradients make the
//! corrected field curl-free. Each primal edge's `k` is a flow between the two
//! dual faces it separates (or a face and the boundary "ground"); the residue at
//! each face is the net supply that flow must balance. The flow itself is solved
//! by a primal network simplex ([`super::simplex`]) — runtime decoupled from the
//! total flow, unlike the prior unit-augmenting shortest-paths solver. Derived
//! only from the published literature; the noncommercial CS2 is never read.

use ndarray::{Array2, ArrayView2};

use super::cost::edge_costs;
use super::simplex::MinCostFlow;
use super::CostMode;

const TAU: f64 = std::f64::consts::TAU;

/// Solve the branch-cut flow, returning integer corrections `(kx, ky)` (as
/// `f64`) aligned with the `ax`/`ay` gradient grids. Residue-free fields return
/// `None` — no correction arrays are allocated.
pub fn solve(
    ax: &Array2<f64>,
    ay: &Array2<f64>,
    corr: ArrayView2<f32>,
    mode: CostMode,
) -> Option<(Array2<f64>, Array2<f64>)> {
    let res = residues(ax, ay);
    if res.iter().all(|&r| r == 0) {
        return None;
    }
    let (wx, wy) = edge_costs(corr, mode);
    let (rows, cols) = (ax.dim().0, ay.dim().1); // pixel grid
    let mut grid = Grid::build(rows, cols, &res, &wx, &wy);
    let flows = grid
        .mcf
        .solve()
        .expect("balanced grid MCF is always feasible");
    Some(grid.recover(&flows, ax.dim(), ay.dim()))
}

/// Discrete curl of the wrapped gradients per `(rows-1, cols-1)` face, in
/// integer cycles — the residue (source/sink) the flow must cancel.
fn residues(ax: &Array2<f64>, ay: &Array2<f64>) -> Array2<i32> {
    let (rf, cf) = (ay.dim().0, ax.dim().1);
    Array2::from_shape_fn((rf, cf), |(i, j)| {
        let curl = ax[(i, j)] + ay[(i, j + 1)] - ax[(i + 1, j)] - ay[(i, j)];
        (curl / TAU).round() as i32
    })
}

/// Which gradient grid a correction edge writes back to, and its index.
#[derive(Clone, Copy)]
enum KVar {
    X(usize, usize),
    Y(usize, usize),
}

/// The Costantini dual graph wired onto a min-cost-flow instance: face nodes +
/// one ground node, a bidirectional dual arc per primal edge, and a handle
/// `(variable, forward-arc, backward-arc)` per edge so `k` is read back as the
/// net flow.
struct Grid {
    mcf: MinCostFlow,
    handles: Vec<(KVar, usize, usize)>,
}

impl Grid {
    /// Wire faces, supplies, and dual arcs from the residues and edge costs.
    fn build(rows: usize, cols: usize, res: &Array2<i32>, wx: &Array2<i64>, wy: &Array2<i64>) -> Self {
        let ground = (rows - 1) * (cols - 1);
        let cap = res.iter().map(|&r| r.unsigned_abs() as i64).sum::<i64>() + 1;
        let mut mcf = MinCostFlow::new(ground + 1);
        let mut ground_supply = 0i64;
        for ((i, j), &r) in res.indexed_iter() {
            ground_supply -= r as i64;
            mcf.set_supply(i * (cols - 1) + j, r as i64);
        }
        mcf.set_supply(ground, ground_supply);

        let mut grid = Self { mcf, handles: Vec::new() };
        grid.add_x_edges(rows, cols, wx, cap);
        grid.add_y_edges(rows, cols, wy, cap);
        grid
    }

    /// Horizontal dual arcs: positive `k` flows from the face above the edge to
    /// the face below (or ground on the boundary), so `curl(k) = -residue`.
    fn add_x_edges(&mut self, rows: usize, cols: usize, wx: &Array2<i64>, cap: i64) {
        let ground = (rows - 1) * (cols - 1);
        let face = |i: usize, j: usize| i * (cols - 1) + j;
        for ((i, j), &w) in wx.indexed_iter() {
            let upper = if i == 0 { ground } else { face(i - 1, j) };
            let lower = if i == rows - 1 { ground } else { face(i, j) };
            self.dual_arc(upper, lower, w, cap, KVar::X(i, j));
        }
    }

    /// Vertical dual arcs: positive `k` flows from the face right of the edge to
    /// the face left (or ground on the boundary).
    fn add_y_edges(&mut self, rows: usize, cols: usize, wy: &Array2<i64>, cap: i64) {
        let ground = (rows - 1) * (cols - 1);
        let face = |i: usize, j: usize| i * (cols - 1) + j;
        for ((i, j), &w) in wy.indexed_iter() {
            let left = if j == 0 { ground } else { face(i, j - 1) };
            let right = if j == cols - 1 { ground } else { face(i, j) };
            self.dual_arc(right, left, w, cap, KVar::Y(i, j));
        }
    }

    /// A bidirectional dual edge `u <-> v` of cost `w`; `k = flow(u->v) - flow(v->u)`.
    fn dual_arc(&mut self, u: usize, v: usize, w: i64, cap: i64, k: KVar) {
        let fwd = self.mcf.add_arc(u, v, w, cap);
        let bwd = self.mcf.add_arc(v, u, w, cap);
        self.handles.push((k, fwd, bwd));
    }

    /// Read the net integer flow on each correction edge into `kx`/`ky`.
    fn recover(
        &self,
        flows: &[i64],
        x_dim: (usize, usize),
        y_dim: (usize, usize),
    ) -> (Array2<f64>, Array2<f64>) {
        let mut kx = Array2::zeros(x_dim);
        let mut ky = Array2::zeros(y_dim);
        for &(k, fwd, bwd) in &self.handles {
            let net = (flows[fwd] - flows[bwd]) as f64;
            match k {
                KVar::X(i, j) => kx[(i, j)] = net,
                KVar::Y(i, j) => ky[(i, j)] = net,
            }
        }
        (kx, ky)
    }
}

#[cfg(test)]
mod tests;
