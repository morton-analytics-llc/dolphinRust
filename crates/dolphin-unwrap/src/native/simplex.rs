//! Primal network simplex for integer minimum-cost flow.
//!
//! Clean-room, papers-only (Cunningham 1973 strongly-feasible basis; Goldfarb
//! 1990 anti-stalling; Kovács 2012 arXiv:1207.6381 spanning-tree data structure).
//! No SNAPHU/CS2 source read. Replaces the unit-augmenting successive-shortest-
//! paths solver whose runtime scaled with total flow `F`; network simplex
//! decouples runtime from `F`, the win that closes the CPU gap on the Costantini
//! grid graph (see `docs/native_mcf_solver.md`).
//!
//! Generic over the graph: build with [`MinCostFlow::new`], [`add_arc`], and
//! [`set_supply`] (supplies must sum to zero), then [`solve`] returns the flow on
//! each user arc in insertion order. The solver adds an artificial root + one
//! artificial arc per node for a strongly-feasible initial basis; a feasible
//! instance drives every artificial flow to zero.
//!
//! [`add_arc`]: MinCostFlow::add_arc
//! [`set_supply`]: MinCostFlow::set_supply
//! [`solve`]: MinCostFlow::solve

const SENTINEL: u32 = u32::MAX;

/// A directed arc with integer cost and capacity.
struct Arc {
    from: u32,
    to: u32,
    cost: i64,
    cap: i64,
    flow: i64,
}

/// Min-cost-flow instance and its network-simplex solver state.
pub struct MinCostFlow {
    arcs: Vec<Arc>,
    /// User-arc count; arcs `[0, n_user)` are real, the rest artificial.
    n_user: usize,
    supply: Vec<i64>,
    /// Spanning-tree parent of each node (`SENTINEL` for the root).
    parent: Vec<u32>,
    /// Basic arc linking each node to its parent (`SENTINEL` for the root).
    pred_arc: Vec<u32>,
    /// Tree children adjacency.
    children: Vec<Vec<u32>>,
    /// Node potentials (`pi`); reduced cost of a tree arc is zero.
    pi: Vec<i64>,
    /// Per-pivot apex-finding stamps and the working stamp value.
    stamp: Vec<u32>,
    cur_stamp: u32,
    root: u32,
}

impl MinCostFlow {
    /// A flow problem over `n_nodes` nodes (root added internally).
    pub fn new(n_nodes: usize) -> Self {
        Self {
            arcs: Vec::new(),
            n_user: 0,
            supply: vec![0; n_nodes],
            parent: Vec::new(),
            pred_arc: Vec::new(),
            children: Vec::new(),
            pi: Vec::new(),
            stamp: Vec::new(),
            cur_stamp: 0,
            root: n_nodes as u32,
        }
    }

    /// Add a directed arc `from -> to` with the given per-unit `cost` and `cap`,
    /// returning its arc id (insertion order, used to read flow back).
    pub fn add_arc(&mut self, from: usize, to: usize, cost: i64, cap: i64) -> usize {
        let id = self.arcs.len();
        self.arcs.push(Arc {
            from: from as u32,
            to: to as u32,
            cost,
            cap,
            flow: 0,
        });
        id
    }

    /// Set node `node`'s supply (positive) or demand (negative).
    pub fn set_supply(&mut self, node: usize, b: i64) {
        self.supply[node] = b;
    }

    /// Solve to optimality; returns flow on each user arc in insertion order.
    /// Returns `None` if the instance is infeasible (an artificial arc keeps
    /// flow) — never expected for the balanced grid MCF.
    pub fn solve(&mut self) -> Option<Vec<i64>> {
        self.n_user = self.arcs.len();
        self.init_basis();
        self.run();
        self.feasible()
            .then(|| self.arcs[..self.n_user].iter().map(|a| a.flow).collect())
    }

    /// Strongly-feasible initial basis: a star of artificial arcs to the root.
    /// Zero-flow artificial arcs point away from the root (`r -> i`), so the
    /// basis is strongly feasible (Cunningham 1973).
    fn init_basis(&mut self) {
        let n = self.supply.len();
        let r = self.root as usize;
        let big = self.big_cost();
        self.parent = vec![SENTINEL; n + 1];
        self.pred_arc = vec![SENTINEL; n + 1];
        self.children = vec![Vec::new(); n + 1];
        self.pi = vec![0; n + 1];
        self.stamp = vec![0; n + 1];
        for i in 0..n {
            let b = self.supply[i];
            let arc = if b > 0 {
                self.pi[i] = big;
                self.push_arc(i, r, big, b)
            } else {
                self.pi[i] = -big;
                self.push_arc(r, i, big, -b)
            };
            self.parent[i] = self.root;
            self.pred_arc[i] = arc as u32;
            self.children[r].push(i as u32);
        }
    }

    /// An artificial-arc cost that exceeds any real path, so artificials leave.
    fn big_cost(&self) -> i64 {
        let max_cost = self.arcs.iter().map(|a| a.cost).max().unwrap_or(0);
        let n = self.supply.len() as i64;
        max_cost.saturating_mul(n).saturating_add(1)
    }

    /// Append an artificial arc with `flow` already routed; caps never bind.
    fn push_arc(&mut self, from: usize, to: usize, cost: i64, flow: i64) -> usize {
        let id = self.arcs.len();
        self.arcs.push(Arc {
            from: from as u32,
            to: to as u32,
            cost,
            cap: i64::MAX,
            flow,
        });
        id
    }

    /// Pivot until no nonbasic real arc has negative reduced cost. Block-search
    /// pricing scans the real arcs in `sqrt(m)`-sized blocks round-robin.
    fn run(&mut self) {
        let m = self.n_user;
        let block = ((m as f64).sqrt() as usize).max(1);
        let mut next = 0usize;
        let mut scanned = 0usize;
        while scanned < m {
            match self.best_in_block(next, block) {
                Some(enter) => {
                    self.pivot(enter);
                    scanned = 0;
                }
                None => scanned += block,
            }
            next = (next + block) % m.max(1);
        }
    }

    /// Most-negative-reduced-cost real arc in the block starting at `start`.
    fn best_in_block(&self, start: usize, block: usize) -> Option<usize> {
        let m = self.n_user;
        let mut best: Option<usize> = None;
        let mut best_rc = 0i64;
        for k in 0..block {
            let a = (start + k) % m;
            let rc = self.reduced_cost(a);
            if rc < best_rc {
                best_rc = rc;
                best = Some(a);
            }
        }
        best
    }

    /// Reduced cost `cost - pi[from] + pi[to]`; nonbasic real arcs sit at the
    /// lower bound (flow 0), so a negative value can be improved by entering.
    fn reduced_cost(&self, a: usize) -> i64 {
        let arc = &self.arcs[a];
        if arc.flow >= arc.cap {
            return 0; // saturated: cannot enter from the lower bound
        }
        arc.cost - self.pi[arc.from as usize] + self.pi[arc.to as usize]
    }

    /// Apply one pivot: entering arc `e` closes a cycle with the tree; augment
    /// by the cycle bottleneck, swap in `e` for the (strongly-feasible) leaving
    /// arc, and shift the moved subtree's potentials.
    fn pivot(&mut self, e: usize) {
        let (k, l) = (self.arcs[e].from, self.arcs[e].to);
        let apex = self.apex(k, l);
        let cyc = self.cycle_arcs(e, k, l, apex);
        let (delta, leave_pos) = self.bottleneck(&cyc);
        self.augment(&cyc, delta);
        self.restructure(e, &cyc, leave_pos);
    }

    /// Least-common-ancestor of `k` and `l` in the tree, via ancestor stamping.
    fn apex(&mut self, k: u32, l: u32) -> u32 {
        self.cur_stamp += 1;
        let s = self.cur_stamp;
        let mut a = k;
        while a != SENTINEL {
            self.stamp[a as usize] = s;
            a = self.parent[a as usize];
        }
        let mut b = l;
        while self.stamp[b as usize] != s {
            b = self.parent[b as usize];
        }
        b
    }

    /// The cycle arcs in cycle-orientation order: the `k`-side (root-ward,
    /// reversed to run apex->k), then entering arc, then the `l`-side (l->apex).
    /// Each entry is `(node, parent, increasing)` where `increasing` marks an
    /// arc whose flow rises with the augmentation.
    fn cycle_arcs(&self, e: usize, k: u32, l: u32, apex: u32) -> Vec<CycleArc> {
        let mut k_side = Vec::new();
        let mut n = k;
        while n != apex {
            // cycle runs apex->k here, i.e. parent->child: arc increases iff
            // it is oriented parent->child (from == parent).
            let arc = self.pred_arc[n as usize];
            let inc = self.arcs[arc as usize].from == self.parent[n as usize];
            k_side.push(CycleArc { arc, inc });
            n = self.parent[n as usize];
        }
        k_side.reverse();
        let mut out = k_side;
        out.push(CycleArc {
            arc: e as u32,
            inc: true,
        });
        let mut n = l;
        while n != apex {
            // cycle runs l->apex here, i.e. child->parent: arc increases iff
            // it is oriented child->parent (from == n).
            let arc = self.pred_arc[n as usize];
            let inc = self.arcs[arc as usize].from == n;
            out.push(CycleArc { arc, inc });
            n = self.parent[n as usize];
        }
        out
    }

    /// Bottleneck `delta` and the position of the leaving arc — the LAST
    /// decreasing arc that reaches flow zero (strongly-feasible tie-break).
    fn bottleneck(&self, cyc: &[CycleArc]) -> (i64, usize) {
        let mut delta = i64::MAX;
        let mut leave = 0usize;
        for (pos, c) in cyc.iter().enumerate() {
            let arc = &self.arcs[c.arc as usize];
            let residual = if c.inc { arc.cap - arc.flow } else { arc.flow };
            if residual <= delta {
                delta = residual;
                leave = pos;
            }
        }
        (delta, leave)
    }

    /// Push `delta` around the cycle: increasing arcs gain, decreasing arcs lose.
    fn augment(&mut self, cyc: &[CycleArc], delta: i64) {
        if delta == 0 {
            return;
        }
        for c in cyc {
            let arc = &mut self.arcs[c.arc as usize];
            if c.inc {
                arc.flow += delta;
            } else {
                arc.flow -= delta;
            }
        }
    }

    /// Swap the leaving tree arc for the entering arc `e` and re-root the
    /// detached subtree, shifting its potentials by the entering reduced cost.
    fn restructure(&mut self, e: usize, cyc: &[CycleArc], leave_pos: usize) {
        let leave = cyc[leave_pos].arc as usize;
        if leave == e {
            return; // entering arc immediately saturated: basis unchanged
        }
        let child = self.tree_child(leave);
        let subtree = self.collect_subtree(child);
        self.detach(child);
        let (k, l) = (self.arcs[e].from, self.arcs[e].to);
        let in_s = if self.in_set(&subtree, l) { l } else { k };
        let out = if in_s == l { k } else { l };
        self.reroot(in_s, child);
        self.attach(in_s, out, e);
        self.shift_potentials(&subtree, in_s, out, e);
    }

    /// The child-side endpoint of a tree arc (deeper node).
    fn tree_child(&self, arc: usize) -> u32 {
        let a = &self.arcs[arc];
        if self.pred_arc[a.from as usize] == arc as u32 {
            a.from
        } else {
            a.to
        }
    }

    /// Depth-first node list of the subtree rooted at `root` (current tree).
    fn collect_subtree(&self, root: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            out.push(v);
            stack.extend_from_slice(&self.children[v as usize]);
        }
        out
    }

    fn in_set(&self, nodes: &[u32], v: u32) -> bool {
        nodes.contains(&v)
    }

    /// Cut `child` from its parent (the leaving arc).
    fn detach(&mut self, child: u32) {
        let p = self.parent[child as usize];
        self.children[p as usize].retain(|&c| c != child);
        self.parent[child as usize] = SENTINEL;
        self.pred_arc[child as usize] = SENTINEL;
    }

    /// Reverse the parent chain from `x` up to `stop` so `x` becomes the
    /// subtree root (its old ancestors become its descendants).
    fn reroot(&mut self, x: u32, stop: u32) {
        if x == stop {
            return;
        }
        let mut prev = x;
        let mut cur = self.parent[x as usize];
        let mut prev_arc = self.pred_arc[x as usize];
        loop {
            let next = self.parent[cur as usize];
            let next_arc = self.pred_arc[cur as usize];
            self.children[cur as usize].retain(|&c| c != prev);
            self.parent[cur as usize] = prev;
            self.pred_arc[cur as usize] = prev_arc;
            self.children[prev as usize].push(cur);
            if cur == stop {
                break;
            }
            prev = cur;
            cur = next;
            prev_arc = next_arc;
        }
    }

    /// Hang re-rooted subtree (now rooted at `in_s`) under `out` via arc `e`.
    fn attach(&mut self, in_s: u32, out: u32, e: usize) {
        self.parent[in_s as usize] = out;
        self.pred_arc[in_s as usize] = e as u32;
        self.children[out as usize].push(in_s);
    }

    /// Shift every potential in the moved subtree by the constant that restores
    /// zero reduced cost on the entering arc.
    fn shift_potentials(&mut self, subtree: &[u32], in_s: u32, out: u32, e: usize) {
        let arc = &self.arcs[e];
        let new_pi = if arc.from == out {
            self.pi[out as usize] - arc.cost
        } else {
            self.pi[out as usize] + arc.cost
        };
        let delta = new_pi - self.pi[in_s as usize];
        for &v in subtree {
            self.pi[v as usize] += delta;
        }
    }

    /// True if no artificial arc carries flow (the instance is feasible).
    fn feasible(&self) -> bool {
        self.arcs[self.n_user..].iter().all(|a| a.flow == 0)
    }
}

/// One arc on a pivot cycle and whether its flow increases with augmentation.
struct CycleArc {
    arc: u32,
    inc: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-node transportation problem with a unique optimum: two sources
    /// (supply +2 each at 0,1), two sinks (-2 each at 2,3), arc costs chosen so
    /// the optimal routing is unambiguous. Verifies cost and conservation.
    #[test]
    fn transportation_optimum() {
        let mut mcf = MinCostFlow::new(4);
        mcf.set_supply(0, 2);
        mcf.set_supply(1, 2);
        mcf.set_supply(2, -2);
        mcf.set_supply(3, -2);
        let big = 1_000;
        let a02 = mcf.add_arc(0, 2, 1, big);
        let a03 = mcf.add_arc(0, 3, 5, big);
        let a12 = mcf.add_arc(1, 2, 5, big);
        let a13 = mcf.add_arc(1, 3, 1, big);
        let flow = mcf.solve().expect("feasible");
        assert_eq!(flow[a02], 2, "0->2 cheap, fully used");
        assert_eq!(flow[a13], 2, "1->3 cheap, fully used");
        assert_eq!(flow[a03], 0);
        assert_eq!(flow[a12], 0);
    }

    /// Routing must use a cheap two-hop path over a direct expensive arc.
    #[test]
    fn prefers_cheap_transshipment() {
        let mut mcf = MinCostFlow::new(3);
        mcf.set_supply(0, 1);
        mcf.set_supply(2, -1);
        let direct = mcf.add_arc(0, 2, 10, 5);
        let hop_a = mcf.add_arc(0, 1, 1, 5);
        let hop_b = mcf.add_arc(1, 2, 1, 5);
        let flow = mcf.solve().expect("feasible");
        assert_eq!(flow[direct], 0);
        assert_eq!(flow[hop_a], 1);
        assert_eq!(flow[hop_b], 1);
    }

    /// A capacity constraint must split flow onto the dearer path.
    #[test]
    fn respects_capacity() {
        let mut mcf = MinCostFlow::new(2);
        mcf.set_supply(0, 3);
        mcf.set_supply(1, -3);
        let cheap = mcf.add_arc(0, 1, 1, 2);
        let dear = mcf.add_arc(0, 1, 4, 10);
        let flow = mcf.solve().expect("feasible");
        assert_eq!(flow[cheap], 2, "cheap arc saturated at cap");
        assert_eq!(flow[dear], 1, "remainder on the dearer arc");
    }
}
