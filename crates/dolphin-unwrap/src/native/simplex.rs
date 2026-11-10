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
    /// Preorder successor of each node, circular through the root: a node's
    /// subtree is the `subtree_size` nodes starting at it.
    thread: Vec<u32>,
    /// Preorder predecessor (the inverse of `thread`).
    rev_thread: Vec<u32>,
    /// Node count of each node's subtree, itself included.
    subtree_size: Vec<u32>,
    /// Last node of each node's subtree in preorder.
    last_descendant: Vec<u32>,
    /// Node potentials (`pi`); reduced cost of a tree arc is zero.
    pi: Vec<i64>,
    /// The current pivot's cycle, kept to reuse its allocation.
    cycle: Vec<CycleArc>,
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
            thread: Vec::new(),
            rev_thread: Vec::new(),
            subtree_size: Vec::new(),
            last_descendant: Vec::new(),
            pi: Vec::new(),
            cycle: Vec::new(),
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
        // Preorder of the star: root, 0, 1, …, n-1, and back to the root.
        self.thread = (0..=n).map(|i| ((i + 1) % (n + 1)) as u32).collect();
        self.rev_thread = (0..=n).map(|i| ((i + n) % (n + 1)) as u32).collect();
        self.subtree_size = vec![1; n + 1];
        self.subtree_size[r] = (n + 1) as u32;
        self.last_descendant = (0..=n as u32).collect();
        self.last_descendant[r] = self.rev_thread[r];
        self.pi = vec![0; n + 1];
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
        let mut cyc = std::mem::take(&mut self.cycle);
        let entering_pos = self.fill_cycle(&mut cyc, e, k, l, apex);
        let (delta, leave_pos) = self.bottleneck(&cyc);
        self.augment(&cyc, delta);
        // The leaving arc cuts off the subtree holding the endpoint on its own
        // side of the cycle; if the entering arc itself leaves (it saturated
        // immediately), the basis is unchanged.
        match leave_pos.cmp(&entering_pos) {
            std::cmp::Ordering::Less => self.restructure(e, cyc[leave_pos].arc, k, l, apex),
            std::cmp::Ordering::Greater => self.restructure(e, cyc[leave_pos].arc, l, k, apex),
            std::cmp::Ordering::Equal => {}
        }
        self.cycle = cyc;
    }

    /// Least-common-ancestor of `k` and `l` in the tree. The endpoint with the
    /// smaller subtree cannot be an ancestor of the other, so it climbs.
    fn apex(&self, mut k: u32, mut l: u32) -> u32 {
        while k != l {
            if self.subtree_size[k as usize] < self.subtree_size[l as usize] {
                k = self.parent[k as usize];
            } else {
                l = self.parent[l as usize];
            }
        }
        k
    }

    /// Fill `cyc` with the cycle arcs in cycle-orientation order: the `k`-side
    /// (root-ward, reversed to run apex->k), then the entering arc, then the
    /// `l`-side (l->apex). Returns the entering arc's position.
    fn fill_cycle(&self, cyc: &mut Vec<CycleArc>, e: usize, k: u32, l: u32, apex: u32) -> usize {
        cyc.clear();
        let mut n = k;
        while n != apex {
            // cycle runs apex->k here, i.e. parent->child: arc increases iff
            // it is oriented parent->child (from == parent).
            let arc = self.pred_arc[n as usize];
            let inc = self.arcs[arc as usize].from == self.parent[n as usize];
            cyc.push(CycleArc { arc, inc });
            n = self.parent[n as usize];
        }
        cyc.reverse();
        let entering_pos = cyc.len();
        cyc.push(CycleArc {
            arc: e as u32,
            inc: true,
        });
        let mut n = l;
        while n != apex {
            // cycle runs l->apex here, i.e. child->parent: arc increases iff
            // it is oriented child->parent (from == n).
            let arc = self.pred_arc[n as usize];
            let inc = self.arcs[arc as usize].from == n;
            cyc.push(CycleArc { arc, inc });
            n = self.parent[n as usize];
        }
        entering_pos
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

    /// Swap the leaving tree arc for the entering arc `e`: cut the subtree
    /// below the leaving arc, re-root it at `inside` (the entering arc's
    /// endpoint in it), hang it under `outside`, and shift its potentials by
    /// the entering reduced cost.
    fn restructure(&mut self, e: usize, leave: u32, inside: u32, outside: u32, apex: u32) {
        let child = self.tree_child(leave as usize);
        self.cut(child, apex);
        self.reroot(inside, child);
        self.graft(inside, outside, e, apex);
        self.shift_potentials(inside, outside, e);
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

    /// Make `b` the preorder successor of `a`.
    fn link(&mut self, a: u32, b: u32) {
        self.thread[a as usize] = b;
        self.rev_thread[b as usize] = a;
    }

    /// Cut `child`'s subtree out of the tree (the leaving arc): close the
    /// thread over its segment and take its nodes off its ancestors. `apex`
    /// and the nodes above it keep their size, because the subtree is grafted
    /// back below `apex`.
    fn cut(&mut self, child: u32, apex: u32) {
        let size = self.subtree_size[child as usize];
        let last = self.last_descendant[child as usize];
        let before = self.rev_thread[child as usize];
        self.link(before, self.thread[last as usize]);
        let mut a = self.parent[child as usize];
        while a != apex {
            self.subtree_size[a as usize] -= size;
            a = self.parent[a as usize];
        }
        // An ancestor that ended on the cut segment now ends just before it.
        let mut a = self.parent[child as usize];
        while a != SENTINEL && self.last_descendant[a as usize] == last {
            self.last_descendant[a as usize] = before;
            a = self.parent[a as usize];
        }
        self.parent[child as usize] = SENTINEL;
        self.pred_arc[child as usize] = SENTINEL;
    }

    /// Re-root the cut subtree at `x`: reverse the parent chain from `x` up to
    /// its old root `stop`, and rewrite the thread to the new preorder.
    ///
    /// With the chain `x = u0, u1, …, um = stop`, the old preorder nests as
    /// `um A_m … u1 A_1 [u0's subtree] B_1 … B_m`, where `A_i` and `B_i` are the
    /// parts of `u_i`'s subtree before and after `u_{i-1}`'s. The new preorder is
    /// `[u0's subtree] u1 A_1 B_1 … um A_m B_m`: every `u_i` now hangs under
    /// `u_{i-1}` and keeps its other descendants.
    fn reroot(&mut self, x: u32, stop: u32) {
        if x == stop {
            return;
        }
        let total = self.subtree_size[stop as usize];
        let mut tail = self.last_descendant[x as usize];
        // Old end of the inner subtree and its old successor (the next `B`).
        let mut inner_last = tail;
        let mut after_inner = self.thread[tail as usize];
        let mut prev = x;
        let mut prev_old_rev = self.rev_thread[x as usize];
        let mut prev_old_size = self.subtree_size[x as usize];
        let mut carried_arc = self.pred_arc[x as usize];
        let mut cur = self.parent[x as usize];
        loop {
            let (next, next_arc) = (self.parent[cur as usize], self.pred_arc[cur as usize]);
            let cur_old_rev = self.rev_thread[cur as usize];
            let cur_old_size = self.subtree_size[cur as usize];
            let cur_old_last = self.last_descendant[cur as usize];
            let a_is_empty = self.thread[cur as usize] == prev;
            self.link(tail, cur);
            tail = if a_is_empty { cur } else { prev_old_rev };
            if cur_old_last != inner_last {
                let b_first = after_inner;
                after_inner = self.thread[cur_old_last as usize];
                inner_last = cur_old_last;
                self.link(tail, b_first);
                tail = cur_old_last;
            }
            self.parent[cur as usize] = prev;
            self.pred_arc[cur as usize] = carried_arc;
            self.subtree_size[cur as usize] = total - prev_old_size;
            if cur == stop {
                break;
            }
            prev = cur;
            prev_old_rev = cur_old_rev;
            prev_old_size = cur_old_size;
            carried_arc = next_arc;
            cur = next;
        }
        self.subtree_size[x as usize] = total;
        // Every node on the reversed chain now ends on the segment's last node.
        let mut v = stop;
        while v != x {
            self.last_descendant[v as usize] = tail;
            v = self.parent[v as usize];
        }
        self.last_descendant[x as usize] = tail;
    }

    /// Hang the re-rooted subtree (rooted at `inside`) under `outside` via arc
    /// `e`, as the first subtree after `outside` in preorder.
    fn graft(&mut self, inside: u32, outside: u32, e: usize, apex: u32) {
        let size = self.subtree_size[inside as usize];
        let last = self.last_descendant[inside as usize];
        self.parent[inside as usize] = outside;
        self.pred_arc[inside as usize] = e as u32;
        let after = self.thread[outside as usize];
        self.link(outside, inside);
        self.link(last, after);
        let mut a = outside;
        while a != apex {
            self.subtree_size[a as usize] += size;
            a = self.parent[a as usize];
        }
        // An ancestor that ended on `outside` now ends on the grafted segment.
        let mut a = outside;
        while a != SENTINEL && self.last_descendant[a as usize] == outside {
            self.last_descendant[a as usize] = last;
            a = self.parent[a as usize];
        }
    }

    /// Shift every potential in the moved subtree by the constant that restores
    /// zero reduced cost on the entering arc.
    fn shift_potentials(&mut self, inside: u32, outside: u32, e: usize) {
        let arc = &self.arcs[e];
        let new_pi = if arc.from == outside {
            self.pi[outside as usize] - arc.cost
        } else {
            self.pi[outside as usize] + arc.cost
        };
        let delta = new_pi - self.pi[inside as usize];
        let mut v = inside;
        for _ in 0..self.subtree_size[inside as usize] {
            self.pi[v as usize] += delta;
            v = self.thread[v as usize];
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

    const GOLDEN_FLOW_HASHES: [u64; 3] = [
        0xe1f9_9171_53e6_a431,
        0xbaa2_f996_6be3_f4f8,
        0xcca0_da53_e1e7_716c,
    ];

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

    /// SplitMix64, so the instances below need no rng crate.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A `side x side` grid of nodes plus a ground node, a pair of opposed arcs
    /// between 4-neighbours and from every border node to ground, costs in
    /// 1..=50, and unit supplies or demands on about a fifth of the nodes with
    /// the ground balancing them: the shape of the unwrapper's dual graph.
    fn grid_instance(side: usize, seed: u64) -> MinCostFlow {
        let mut state = seed;
        let ground = side * side;
        let mut mcf = MinCostFlow::new(ground + 1);
        let mut connect = |mcf: &mut MinCostFlow, a: usize, b: usize| {
            for (from, to) in [(a, b), (b, a)] {
                mcf.add_arc(from, to, 1 + (next(&mut state) % 50) as i64, 1_000_000);
            }
        };
        for node in 0..ground {
            let (row, col) = (node / side, node % side);
            if col + 1 < side {
                connect(&mut mcf, node, node + 1);
            }
            if row + 1 < side {
                connect(&mut mcf, node, node + side);
            }
            if row == 0 || col == 0 || row + 1 == side || col + 1 == side {
                connect(&mut mcf, node, ground);
            }
        }
        let mut total = 0;
        for node in 0..ground {
            let supply = match next(&mut state) % 10 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
            mcf.set_supply(node, supply);
            total += supply;
        }
        mcf.set_supply(ground, -total);
        mcf
    }

    fn flow_hash(flow: &[i64]) -> u64 {
        flow.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &f| {
            (h.rotate_left(5) ^ f as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        })
    }

    /// The pivot rule does not depend on how the spanning tree is stored, so
    /// the flows on these instances are pinned to the values the child-list
    /// tree produced before the thread representation replaced it (#161).
    #[test]
    fn flows_are_unchanged_by_the_tree_representation() {
        let hashes: Vec<u64> = [(12, 1), (24, 2), (40, 3)]
            .into_iter()
            .map(|(side, seed)| flow_hash(&grid_instance(side, seed).solve().unwrap()))
            .collect();
        assert_eq!(hashes, GOLDEN_FLOW_HASHES, "{hashes:#x?}");
    }

    /// Check the thread arrays against the tree the parent pointers define:
    /// the thread is one cycle through every node, `rev_thread` inverts it,
    /// each node's true subtree is exactly the `subtree_size` nodes starting
    /// at it and ending on `last_descendant`, and every tree arc has zero
    /// reduced cost.
    fn assert_tree_is_consistent(mcf: &MinCostFlow) {
        let n = mcf.parent.len();
        let mut order = Vec::with_capacity(n);
        let mut v = mcf.root;
        for _ in 0..n {
            order.push(v);
            assert_eq!(mcf.rev_thread[mcf.thread[v as usize] as usize], v);
            v = mcf.thread[v as usize];
        }
        assert_eq!(v, mcf.root, "thread must close after every node");
        let mut position = vec![usize::MAX; n];
        for (at, &node) in order.iter().enumerate() {
            assert_eq!(position[node as usize], usize::MAX, "node visited twice");
            position[node as usize] = at;
        }
        let mut true_size = vec![0u32; n];
        for node in 0..n as u32 {
            let mut a = node;
            while a != SENTINEL {
                true_size[a as usize] += 1;
                a = mcf.parent[a as usize];
            }
        }
        for node in 0..n {
            let size = mcf.subtree_size[node] as usize;
            assert_eq!(size, true_size[node] as usize, "subtree size of {node}");
            let end = position[node] + size;
            assert_eq!(mcf.last_descendant[node], order[end - 1], "last of {node}");
            let parent = mcf.parent[node];
            if parent == SENTINEL {
                continue;
            }
            let parent_start = position[parent as usize];
            let parent_end = parent_start + mcf.subtree_size[parent as usize] as usize;
            assert!(parent_start < position[node] && end <= parent_end);
            let arc = &mcf.arcs[mcf.pred_arc[node] as usize];
            assert_eq!(
                arc.cost - mcf.pi[arc.from as usize] + mcf.pi[arc.to as usize],
                0
            );
        }
    }

    /// #161: the thread, sizes and last descendants stay a valid preorder of
    /// the parent tree after every pivot, and the result is optimal by
    /// conservation plus complementary slackness.
    #[test]
    fn thread_stays_a_preorder_through_every_pivot_and_ends_optimal() {
        let mut mcf = grid_instance(12, 7);
        mcf.n_user = mcf.arcs.len();
        mcf.init_basis();
        assert_tree_is_consistent(&mcf);
        // `run`, with the check after each pivot.
        let m = mcf.n_user;
        let block = ((m as f64).sqrt() as usize).max(1);
        let (mut next, mut scanned, mut pivots) = (0, 0, 0);
        while scanned < m {
            match mcf.best_in_block(next, block) {
                Some(enter) => {
                    mcf.pivot(enter);
                    assert_tree_is_consistent(&mcf);
                    scanned = 0;
                    pivots += 1;
                }
                None => scanned += block,
            }
            next = (next + block) % m;
        }
        assert!(
            pivots > 100,
            "{pivots} pivots is too few to exercise re-rooting"
        );
        assert!(mcf.feasible());

        let mut net = mcf.supply.clone();
        for arc in &mcf.arcs[..m] {
            net[arc.from as usize] -= arc.flow;
            net[arc.to as usize] += arc.flow;
            let reduced = arc.cost - mcf.pi[arc.from as usize] + mcf.pi[arc.to as usize];
            assert!(reduced >= 0, "an arc could still enter");
            assert!(arc.flow == 0 || reduced == 0, "flow on a non-tight arc");
        }
        assert!(
            net.iter().all(|&balance| balance == 0),
            "flow not conserved"
        );
    }
}
