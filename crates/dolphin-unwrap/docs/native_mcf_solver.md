# Native MCF solver — design note (network simplex)

Clean-room, papers-only. SNAPHU/CS2 source never read; SNAPHU is a black-box oracle.

## Problem

Costantini (1998) min-cost-flow unwrapping. Nodes = the `(rows-1)×(cols-1)` dual
faces plus one **ground** node; supply at each face = its residue (discrete curl of
the wrapped gradients, integer cycles, mostly ±1), ground absorbs the negative sum
so the instance is balanced. Each primal gradient edge is a **bidirectional** dual
arc between the two faces it separates (or face↔ground on the boundary); the integer
correction `k` on that edge is the net arc flow. Per-unit arc cost = CRLB phase
precision `γ²/(1−γ²)` (Chen & Zebker 2001), quantized to `i64` (`×1e4`). Minimize
`Σ cost·|k|` so the corrected gradients are curl-free.

## Why network simplex

The prior solver was unit-augmenting successive-shortest-paths: one Dijkstra over
the ~10⁶-node grid per unit of flow, F ≈ R/2 ≈ 1.8×10⁴ units at 1024² → ~10¹² ops,
measured ~30× the CPU of SNAPHU's CS2. The runtime is tied to **total flow F**.

- **Capacity scaling** collapses to plain SSP here (unit arc capacities ⇒ one phase).
- **Blocking-flow SSP** buys only ~3–5× (Dinic bound `O(m·√m)`), still ~6–10× behind.
- **Network simplex (NS)** and **cost scaling (CS2)** decouple runtime from F. Per
  Kovács 2015 (EGRES-13-04, the definitive MCF benchmark) NS is the most efficient
  on **grid** families — 2–3× over CS2, 20–100× over SSP. NS also has the smaller
  implementation surface and a cleaner papers-only correctness trail.

**Chosen: hand-rolled primal network simplex.** Fallback: cost scaling (mirrors CS2
exactly, larger surface). No commercially-clean, production Rust crate exists
(`mcmf` = C++/LEMON FFI, unproven; `network-simplex-rust` = no LICENSE = all rights
reserved; `petgraph`/`rustworkx` have no NS). Hand-roll is the only IP-clean path.

## Algorithm

Primal NS on a spanning-tree basis with an artificial **root**:

- **Init:** root node `r`; one artificial arc per node carrying that node's supply to
  `r` (big-M cost). Artificial arcs form the initial spanning-tree star; all real
  arcs nonbasic at flow 0. Feasible and strongly-feasible by construction.
- **Pricing:** block **candidate-list** pivot (scan ~√m arcs for a negative reduced
  cost `c_ij − π_i + π_j`, refresh the block when exhausted) — O(√m) amortized/pivot.
- **Pivot:** the entering arc closes a cycle with the tree; augment the cycle by its
  bottleneck; the saturated tree arc leaves. Update potentials on the moved subtree
  and the thread/parent/depth arrays.
- **Anti-cycling (load-bearing):** **strongly-feasible basis** (Cunningham 1973) — on
  ties for the leaving arc pick the one nearest the root along the cycle. Without it
  degenerate pivots can loop forever. Termination at *an* optimum, not a unique one.

Tree as parallel arrays: `parent`, `parent_arc` (signed), `thread` (preorder),
`depth`, `pi` (potential). Reused across tiles. Real-arc caps set to a big-M ≥ F so
the upper bound never binds → nonbasic arcs are only ever at the lower bound (no
upper-bound state flips → smaller leaving-arc logic, less risk).

## Fidelity / risk

- **Strongly-feasible leaving-arc rule** — the single most load-bearing point; fully
  specified in Cunningham 1973 (2 pp), no code needed.
- **Integers throughout** — costs/potentials/flows all `i64`; worst potential
  `≤ n·C ≈ 10¹⁰ ≪ i64::MAX`. No float drift.
- **Non-unique optima** — NS may place equal-cost branch cuts differently than CS2.
  Expected, not a regression: the gate is per-component cycle parity ≤0.5%, not
  bit-exactness.
- **Verification** — keep the prior SSP solver as a `#[cfg(test)]` reference oracle;
  NS must (a) cancel all residues (curl-free) and (b) match SSP's *optimal cost* on
  random small grids. Then the committed residue-dense golden gates real density.

## Scope

`crates/dolphin-unwrap/src/native/mcf.rs` only. `residues()` / `edge_costs()` /
residue-free fast path unchanged; the solver core is swapped. No trait/pipeline change.
Conncomp segmentation (coherence mask + flood-fill) is a separate, solver-independent
gap addressed in `native.rs`.

Sources: Costantini 1998 (TGRS 36(3)); Chen & Zebker 2001 (JOSA A 18(2)); Cunningham
1973 (strongly-feasible basis); Goldfarb 1990 (anti-stalling, Networks 20); Kovács
2012 (arXiv:1207.6381, LEMON NS data structures); Kovács 2015 (EGRES-13-04, grid
benchmark); Goldberg 1997 (CS2, the oracle's solver — not read, cited for context).
