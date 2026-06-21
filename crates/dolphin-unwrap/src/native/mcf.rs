//! Minimum-cost-flow branch-cut router for residue-bearing fields.
//!
//! Independent implementation of the Costantini (1998) minimum-cost-flow
//! formulation: integer corrections `k` on the wrapped gradients make the
//! corrected field curl-free. Each primal edge's `k` is a flow between the two
//! dual faces it separates (or a face and the boundary "ground"); the residue
//! at each face is the net supply that flow must balance. We solve it with
//! successive shortest augmenting paths under Johnson potentials (Dijkstra on
//! non-negative reduced costs) — a textbook MCF, not the noncommercial CS2 solver.

use ndarray::{Array2, ArrayView2};

use super::cost::edge_costs;
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
    let mut kx = Array2::zeros(ax.dim());
    let mut ky = Array2::zeros(ay.dim());
    let (wx, wy) = edge_costs(corr, mode);
    let (rows, cols) = (ax.dim().0, ay.dim().1); // pixel grid
    let mut net = Network::new(rows, cols, &res);
    net.add_edges(rows, cols, &wx, &wy);
    net.solve();
    net.recover(&mut kx, &mut ky);
    Some((kx, ky))
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

/// A residual-graph arc.
struct Edge {
    to: usize,
    cap: i64,
    cost: i64,
}

/// MCF residual network over face nodes + a ground node + super source/sink.
/// `handles[k]` records the forward/backward directed-edge indices for the k-th
/// correction variable so the net flow (hence `k`) can be read back.
struct Network {
    edges: Vec<Edge>,
    adj: Vec<Vec<usize>>,
    handles: Vec<(KVar, usize, usize)>,
    source: usize,
    sink: usize,
    big: i64,
}

/// Which gradient grid a correction edge writes back to.
#[derive(Clone, Copy)]
enum KVar {
    X(usize, usize),
    Y(usize, usize),
}

impl Network {
    fn new(rows: usize, cols: usize, res: &Array2<i32>) -> Self {
        let ground = (rows - 1) * (cols - 1);
        let source = ground + 1;
        let sink = ground + 2;
        let mut net = Self {
            edges: Vec::new(),
            adj: vec![Vec::new(); ground + 3],
            handles: Vec::new(),
            source,
            sink,
            big: 0,
        };
        // Supplies: residue at each face, the negative sum at ground. Positive
        // supply flows from the super-source, negative into the super-sink.
        let mut total = 0i64;
        let mut ground_supply = 0i64;
        for ((i, j), &r) in res.indexed_iter() {
            ground_supply -= r as i64;
            total += net.feed(i * (cols - 1) + j, r as i64);
        }
        total += net.feed(ground, ground_supply);
        net.big = total.max(1);
        net
    }

    /// Wire one node's supply to the super-source (positive) or super-sink
    /// (negative); returns the positive supply routed (for the flow total).
    fn feed(&mut self, node: usize, supply: i64) -> i64 {
        match supply.cmp(&0) {
            std::cmp::Ordering::Greater => {
                self.link(self.source, node, supply, 0);
                supply
            }
            std::cmp::Ordering::Less => {
                self.link(node, self.sink, -supply, 0);
                0
            }
            std::cmp::Ordering::Equal => 0,
        }
    }

    /// Add a directed residual pair `u -> v` (cap, cost) + `v -> u` (0, -cost).
    fn link(&mut self, u: usize, v: usize, cap: i64, cost: i64) {
        let (a, b) = (self.edges.len(), self.edges.len() + 1);
        self.edges.push(Edge { to: v, cap, cost });
        self.edges.push(Edge {
            to: u,
            cap: 0,
            cost: -cost,
        });
        self.adj[u].push(a);
        self.adj[v].push(b);
    }

    /// Add the dual edges for every primal gradient edge. Orientation: a unit of
    /// `k` flows from the face on the residue-negative side to the positive side,
    /// so face conservation reproduces `curl(k) = -residue`.
    fn add_edges(&mut self, rows: usize, cols: usize, wx: &Array2<i64>, wy: &Array2<i64>) {
        let ground = (rows - 1) * (cols - 1);
        let face = |i: usize, j: usize| i * (cols - 1) + j;
        for ((i, j), &w) in wx.indexed_iter() {
            let upper = if i == 0 { ground } else { face(i - 1, j) };
            let lower = if i == rows - 1 { ground } else { face(i, j) };
            self.dual_edge(upper, lower, w, KVar::X(i, j));
        }
        for ((i, j), &w) in wy.indexed_iter() {
            let left = if j == 0 { ground } else { face(i, j - 1) };
            let right = if j == cols - 1 { ground } else { face(i, j) };
            self.dual_edge(right, left, w, KVar::Y(i, j));
        }
    }

    /// A bidirectional unit-cost dual edge `u -> v` (positive `k`) recorded for
    /// flow read-back.
    fn dual_edge(&mut self, u: usize, v: usize, cost: i64, k: KVar) {
        let fwd = self.edges.len();
        self.link(u, v, self.big, cost);
        let bwd = self.edges.len();
        self.link(v, u, self.big, cost);
        self.handles.push((k, fwd, bwd));
    }

    /// Read the net integer flow on each correction edge back into `kx`/`ky`.
    fn recover(&self, kx: &mut Array2<f64>, ky: &mut Array2<f64>) {
        for &(k, fwd, bwd) in &self.handles {
            let flow = (self.big - self.edges[fwd].cap) - (self.big - self.edges[bwd].cap);
            match k {
                KVar::X(i, j) => kx[(i, j)] = flow as f64,
                KVar::Y(i, j) => ky[(i, j)] = flow as f64,
            }
        }
    }

    /// Successive shortest augmenting paths under Johnson potentials.
    fn solve(&mut self) {
        let mut pot = vec![0i64; self.adj.len()];
        loop {
            let Some((dist, prev)) = self.dijkstra(&pot) else {
                break;
            };
            reweight(&mut pot, &dist);
            self.augment(&prev);
        }
    }

    /// One Dijkstra on reduced costs from the super-source; `None` when the sink
    /// is unreachable (all supply routed). Returns `(dist, prev-edge)`.
    fn dijkstra(&self, pot: &[i64]) -> Option<(Vec<i64>, Vec<usize>)> {
        let n = self.adj.len();
        let mut dist = vec![i64::MAX; n];
        let mut prev = vec![usize::MAX; n];
        let mut heap = std::collections::BinaryHeap::new();
        dist[self.source] = 0;
        heap.push((std::cmp::Reverse(0i64), self.source));
        while let Some((std::cmp::Reverse(d), u)) = heap.pop() {
            if d <= dist[u] {
                self.scan(u, d, pot, &mut dist, &mut prev, &mut heap);
            }
        }
        (dist[self.sink] < i64::MAX).then_some((dist, prev))
    }

    /// Relax every outgoing residual arc of `u`, pushing improved nodes.
    fn scan(
        &self,
        u: usize,
        d: i64,
        pot: &[i64],
        dist: &mut [i64],
        prev: &mut [usize],
        heap: &mut std::collections::BinaryHeap<(std::cmp::Reverse<i64>, usize)>,
    ) {
        for &e in &self.adj[u] {
            let Some((to, nd)) = self.reduced(e, d, pot[u], dist, pot) else {
                continue;
            };
            dist[to] = nd;
            prev[to] = e;
            heap.push((std::cmp::Reverse(nd), to));
        }
    }

    /// Reduced-cost relaxation of arc `e` from a node at distance `d`/potential
    /// `pu`; `Some((to, new_dist))` when it improves `to`, else `None`.
    fn reduced(
        &self,
        e: usize,
        d: i64,
        pu: i64,
        dist: &[i64],
        pot: &[i64],
    ) -> Option<(usize, i64)> {
        let edge = &self.edges[e];
        if edge.cap <= 0 || pot[edge.to] == i64::MAX {
            return None;
        }
        let nd = d + edge.cost + pu - pot[edge.to];
        (nd < dist[edge.to]).then_some((edge.to, nd))
    }

    /// Push one unit-bottleneck of flow along the recovered shortest path.
    fn augment(&mut self, prev: &[usize]) {
        let mut bottleneck = i64::MAX;
        let mut v = self.sink;
        while v != self.source {
            let e = prev[v];
            bottleneck = bottleneck.min(self.edges[e].cap);
            v = self.tail_of(e);
        }
        let mut v = self.sink;
        while v != self.source {
            let e = prev[v];
            self.edges[e].cap -= bottleneck;
            self.edges[e ^ 1].cap += bottleneck;
            v = self.tail_of(e);
        }
    }

    /// Tail node of directed edge `e` (its paired reverse points back at it).
    fn tail_of(&self, e: usize) -> usize {
        self.edges[e ^ 1].to
    }
}

/// Add each Dijkstra distance into the running Johnson potentials, leaving
/// unreached nodes (`dist == MAX`) unchanged.
fn reweight(pot: &mut [i64], dist: &[i64]) {
    for (p, &dv) in pot.iter_mut().zip(dist) {
        if dv < i64::MAX {
            *p = p.saturating_add(dv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// A residue dipole must be cancelled: after the MCF correction the curl of
    /// every face is zero (the +1/-1 pair is reconnected by a branch cut).
    #[test]
    fn residue_dipole_is_balanced() {
        let mut ax = Array2::<f64>::zeros((3, 2));
        let mut ay = Array2::<f64>::zeros((2, 3));
        ay[(0, 1)] = 0.9 * TAU; // one steep gradient -> a +1/-1 residue dipole
        let before = residues(&ax, &ay);
        assert_eq!(before.iter().filter(|&&r| r != 0).count(), 2);
        assert_eq!(before.iter().sum::<i32>(), 0);

        let corr = Array2::<f32>::from_elem((3, 3), 0.9);
        let (kx, ky) = solve(&ax, &ay, corr.view(), CostMode::Smooth).expect("residues present");
        ax.zip_mut_with(&kx, |a, &k| *a += TAU * k);
        ay.zip_mut_with(&ky, |a, &k| *a += TAU * k);

        assert!(
            residues(&ax, &ay).iter().all(|&r| r == 0),
            "corrected field must be residue-free"
        );
    }
}
