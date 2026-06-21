//! Network-simplex MCF contract tests.
//!
//! Correctness oracle: the prior successive-shortest-paths (SSP) solver, kept
//! here as `ssp_ref` only. It is slow but provably optimal, so on random small
//! grids the network-simplex `solve` must (a) cancel every residue and (b) reach
//! the *same optimal cost*. Both solvers are clean-room (Costantini 1998); the
//! SSP reference is the same algorithm the production path used before this phase.

use std::time::Instant;

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

/// Network simplex must match the SSP reference's optimal cost — and leave the
/// field residue-free — on a spread of random residue-dense grids.
#[test]
fn ns_matches_ssp_optimal_cost() {
    for seed in 0..40u64 {
        let (ax, ay, corr) = random_field(seed, 12, 14);
        let res = residues(&ax, &ay);
        if res.iter().all(|&r| r == 0) {
            continue;
        }
        let (wx, wy) = edge_costs(corr.view(), CostMode::Smooth);

        let (kx, ky) = solve(&ax, &ay, corr.view(), CostMode::Smooth).expect("residues present");
        let ns_cost = flow_cost(&kx, &ky, &wx, &wy);
        let ssp_cost = ssp_ref::optimal_cost(&res, &wx, &wy);
        assert_eq!(
            ns_cost, ssp_cost,
            "seed {seed}: network-simplex cost {ns_cost} != SSP optimum {ssp_cost}"
        );

        let mut cax = ax.clone();
        let mut cay = ay.clone();
        cax.zip_mut_with(&kx, |a, &k| *a += TAU * k);
        cay.zip_mut_with(&ky, |a, &k| *a += TAU * k);
        assert!(
            residues(&cax, &cay).iter().all(|&r| r == 0),
            "seed {seed}: corrected field must be residue-free"
        );
    }
}

/// Perf-regression guard: network simplex must finish at least 3x faster than
/// the unit-augmenting SSP reference on a residue-dense grid. The true margin is
/// ~10x+, so the 3x assertion is robust to CI timing noise while still tripping
/// if a change reintroduces flow-scaling (the gap that motivated this solver).
#[test]
fn network_simplex_outpaces_ssp_reference() {
    let (ax, ay, corr) = random_field(7, 56, 60);
    let res = residues(&ax, &ay);
    let n_res = res.iter().filter(|&&r| r != 0).count();
    assert!(n_res > 250, "need a residue-dense instance, got {n_res}");
    let (wx, wy) = edge_costs(corr.view(), CostMode::Smooth);

    let t = Instant::now();
    let ssp_cost = ssp_ref::optimal_cost(&res, &wx, &wy);
    let ssp = t.elapsed();

    let t = Instant::now();
    let (kx, ky) = solve(&ax, &ay, corr.view(), CostMode::Smooth).expect("residues present");
    let ns = t.elapsed();

    assert_eq!(flow_cost(&kx, &ky, &wx, &wy), ssp_cost, "must stay optimal");
    assert!(
        ns.saturating_mul(3) < ssp,
        "network simplex {ns:?} not >=3x faster than SSP {ssp:?} ({n_res} residues)"
    );
}

/// Total routing cost `sum w * |k|`, the objective both solvers minimize.
fn flow_cost(kx: &Array2<f64>, ky: &Array2<f64>, wx: &Array2<i64>, wy: &Array2<i64>) -> i64 {
    let cx: i64 = kx.iter().zip(wx).map(|(&k, &w)| (k.abs() as i64) * w).sum();
    let cy: i64 = ky.iter().zip(wy).map(|(&k, &w)| (k.abs() as i64) * w).sum();
    cx + cy
}

/// A noisy wrapped-gradient field + correlation seeded reproducibly. Steep
/// gradients sprinkle residues; correlation varies so edge costs are nonuniform.
fn random_field(seed: u64, rows: usize, cols: usize) -> (Array2<f64>, Array2<f64>, Array2<f32>) {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let ax = Array2::from_shape_fn((rows, cols - 1), |_| (next() - 0.5) * 2.4 * TAU);
    let ay = Array2::from_shape_fn((rows - 1, cols), |_| (next() - 0.5) * 2.4 * TAU);
    let corr = Array2::from_shape_fn((rows, cols), |_| (0.2 + 0.75 * next()) as f32);
    // Re-wrap the gradients into (-pi, pi] so they read as principal values.
    let wrap = |x: f64| x - TAU * (x / TAU).round();
    (ax.mapv(wrap), ay.mapv(wrap), corr)
}

/// Reference successive-shortest-paths MCF (the prior production solver), kept
/// only as a known-optimal oracle for the network-simplex cross-check.
mod ssp_ref {
    use ndarray::Array2;

    /// Optimal total cost `sum w*|k|` for the residue field, by SSP min-cost flow.
    pub fn optimal_cost(res: &Array2<i32>, wx: &Array2<i64>, wy: &Array2<i64>) -> i64 {
        let (rf, cf) = res.dim();
        let (rows, cols) = (rf + 1, cf + 1);
        let mut net = Network::new(rows, cols, res);
        net.add_edges(rows, cols, wx, wy);
        net.solve();
        net.total_cost()
    }

    struct Edge {
        to: usize,
        cap: i64,
        cost: i64,
        flow: i64,
    }

    struct Network {
        edges: Vec<Edge>,
        adj: Vec<Vec<usize>>,
        unit_cost: Vec<(usize, i64)>,
        source: usize,
        sink: usize,
        big: i64,
    }

    impl Network {
        fn new(rows: usize, cols: usize, res: &Array2<i32>) -> Self {
            let ground = (rows - 1) * (cols - 1);
            let (source, sink) = (ground + 1, ground + 2);
            let mut net = Self {
                edges: Vec::new(),
                adj: vec![Vec::new(); ground + 3],
                unit_cost: Vec::new(),
                source,
                sink,
                big: 0,
            };
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

        fn link(&mut self, u: usize, v: usize, cap: i64, cost: i64) {
            let (a, b) = (self.edges.len(), self.edges.len() + 1);
            self.edges.push(Edge {
                to: v,
                cap,
                cost,
                flow: 0,
            });
            self.edges.push(Edge {
                to: u,
                cap: 0,
                cost: -cost,
                flow: 0,
            });
            self.adj[u].push(a);
            self.adj[v].push(b);
        }

        fn add_edges(&mut self, rows: usize, cols: usize, wx: &Array2<i64>, wy: &Array2<i64>) {
            let ground = (rows - 1) * (cols - 1);
            let face = |i: usize, j: usize| i * (cols - 1) + j;
            for ((i, j), &w) in wx.indexed_iter() {
                let upper = if i == 0 { ground } else { face(i - 1, j) };
                let lower = if i == rows - 1 { ground } else { face(i, j) };
                self.dual_edge(upper, lower, w);
            }
            for ((i, j), &w) in wy.indexed_iter() {
                let left = if j == 0 { ground } else { face(i, j - 1) };
                let right = if j == cols - 1 { ground } else { face(i, j) };
                self.dual_edge(right, left, w);
            }
        }

        fn dual_edge(&mut self, u: usize, v: usize, cost: i64) {
            let fwd = self.edges.len();
            self.link(u, v, self.big, cost);
            self.link(v, u, self.big, cost);
            self.unit_cost.push((fwd, cost));
        }

        /// Total routed cost `sum cost*|net flow|` over the dual edges.
        fn total_cost(&self) -> i64 {
            self.unit_cost
                .iter()
                .map(|&(fwd, cost)| {
                    let net = self.edges[fwd].flow - self.edges[fwd + 2].flow;
                    net.abs() * cost
                })
                .sum()
        }

        fn solve(&mut self) {
            let mut pot = vec![0i64; self.adj.len()];
            while let Some((dist, prev)) = self.dijkstra(&pot) {
                for (p, &d) in pot.iter_mut().zip(&dist) {
                    if d < i64::MAX {
                        *p = p.saturating_add(d);
                    }
                }
                self.augment(&prev);
            }
        }

        fn dijkstra(&self, pot: &[i64]) -> Option<(Vec<i64>, Vec<usize>)> {
            let n = self.adj.len();
            let mut dist = vec![i64::MAX; n];
            let mut prev = vec![usize::MAX; n];
            let mut heap = std::collections::BinaryHeap::new();
            dist[self.source] = 0;
            heap.push((std::cmp::Reverse(0i64), self.source));
            while let Some((std::cmp::Reverse(d), u)) = heap.pop() {
                if d > dist[u] {
                    continue;
                }
                self.scan(u, d, pot, &mut dist, &mut prev, &mut heap);
            }
            (dist[self.sink] < i64::MAX).then_some((dist, prev))
        }

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
                let edge = &self.edges[e];
                if edge.cap - edge.flow <= 0 || pot[edge.to] == i64::MAX {
                    continue;
                }
                let nd = d + edge.cost + pot[u] - pot[edge.to];
                if nd < dist[edge.to] {
                    dist[edge.to] = nd;
                    prev[edge.to] = e;
                    heap.push((std::cmp::Reverse(nd), edge.to));
                }
            }
        }

        fn augment(&mut self, prev: &[usize]) {
            let mut bottleneck = i64::MAX;
            let mut v = self.sink;
            while v != self.source {
                let e = prev[v];
                bottleneck = bottleneck.min(self.edges[e].cap - self.edges[e].flow);
                v = self.edges[e ^ 1].to;
            }
            let mut v = self.sink;
            while v != self.source {
                let e = prev[v];
                self.edges[e].flow += bottleneck;
                self.edges[e ^ 1].flow -= bottleneck;
                v = self.edges[e ^ 1].to;
            }
        }
    }
}
