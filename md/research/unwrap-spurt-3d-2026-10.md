# spurt-style joint spatiotemporal (3D) unwrapping (issue #146)

**Date:** 2026-10-31
**Gate (from #146):** a design review weighing (a) whether loop-closure correction closes enough
of the 79.3% Houston rejection (#138) on its own, against (b) the cost of a clean-room joint
spatiotemporal solver and how it fits `UnwrapBackend`. The issue text cites "#139" for (a);
the loop-closure-correction issue is **#142** (`backlog-ready`). #139 is the Goldstein-filter port.
**Conclusion:** DEFER. Don't build a 3D backend. Reopen only on the conditions in §6.

## 1. What spurt actually implements

Read live at `isce-framework/spurt` `4be99de` (2026-04-24; repo pushed 2026-10-26),
BSD-3-Clause OR Apache-2.0.

- **It's EMCF, not Hooper & Zebker 2007.** `src/spurt/workflows/emcf/_solver.py` docstring:
  "Implements the Extended Minimum Cost Flow (EMCF) algorithm [Pepe & Lanari 2006]". #146
  attributes it to the Hooper & Zebker 3D line (JOSA A 24(9):2737, 2007,
  doi:10.1364/JOSAA.24.002737), but that's a different algorithm. EMCF (Pepe & Lanari, IEEE
  TGRS 44(9):2374–2383, 2006, doi:10.1109/TGRS.2006.873207) is two sequential 2D MCF stages,
  not one joint solve:
  1. **Temporal stage.** For every spatial arc of a Delaunay graph over selected points, take
     the wrapped spatial gradient in every ifg. Then solve an MCF over the *temporal* graph
     (dates as nodes, ifgs as arcs, triangles as faces) to make those gradients
     closure-consistent (`unwrap_gradients_in_time`, batched over `links_per_batch` arcs).
  2. **Spatial stage.** Integrate each ifg's now-unwrapped gradients with a spatial MCF over the
     same Delaunay graph (`unwrap_gradients_in_space`).
- **It works on sparse points, not the grid.** Points are chosen by a temporal-coherence
  threshold. dolphin's wrapper (`dolphin/unwrap/_unwrap_3d.py`, `isce-framework/dolphin`
  `6edd7e1`) passes a 0/1 mask from `temporal_coherence_threshold = 0.7` OR
  `similarity_threshold = 0.5` (`SpurtOptions`). Masked pixels are then filled by "ambiguity
  interpolation". Conncomps are synthesized from the dilated mask, not from the solve.
- **The temporal graph must be planar.** The spurt CLI uses `Hop3Graph` by default
  (`_cli.py:151-153`), a nearest-3-in-time triangulation. Arbitrary ifg networks
  (single-reference, `max_temporal_baseline`) aren't supported as the temporal graph.
- **Cost models are geometric.** `t_cost_type`/`s_cost_type` ∈ {constant, distance, centroid}
  (`_settings.py`), defaulting to `constant`. The solver is OR-Tools `SimpleMinCostFlow`
  (`mcf/_ortools.py`).

## 2. Published head-to-head (the only one found)

Staniewicz et al., arXiv:2609.36267 (2026-09-28), §III-F, Table III. On the Portuguese Bend
landslide, the same 150 dolphin-phase-linked Capella ifgs were unwrapped by five methods,
inverted with dolphin L1, and scored against 103 GPS stations:

| unwrapper | stations kept | r | slope | unwrap time |
|---|---:|---:|---:|---:|
| Whirlwind (2D) | 101 | 0.93 | 0.49 | 0.6 h |
| SNAPHU 2×2 (2D) | 103 | 0.89 | 0.45 | 8.0 h |
| PHASS (2D) | 103 | 0.81 | 0.32 | 1.7 h |
| Spurt EMCF (3D) | 37 | 0.77 | 0.23 | 2.0 h |
| ICU (2D) | 52 | 0.30 | 0.06 | 0.7 h |

The authors attribute spurt's result to insufficient spatial coverage of the sparse point set
for a fast gradient. The same paper (§III-E, South Florida, 72 nearest-3 ifgs) reports that a
**better 2D unwrapper reduced network-inversion misclosure** relative to SNAPHU. The largest
urban area inverts with zero residual for Whirlwind against ~30 cycles for SNAPHU, and
Everglades pixels with ≥1 cycle of misclosure drop from 82% to 43%. That is the symptom #146
wants a 3D solver for, reduced by 2D cost and bridging changes instead.

## 3. The seam #146 relies on is no longer network-level in production

`UnwrapBackend::unwrap_network` takes `(pl, pairs, correlation)` and its docs say a 3D backend
"can implement the same trait and unwrap the whole stack jointly without any pipeline change"
(`unwrap_backend.rs:1-9, 22-28`). Since #137 (`25ab07a`, today), the production path doesn't
call it that way. `unwrap_phase_correlated_network` (`unwrap_backend.rs:62-90`) loops over
pairs, computes a per-pair correlation outside the backend, and calls
`backend.unwrap_network(pl, &[pair], correlation, ...)` **one pair at a time**
(`displacement.rs:4878`). A 3D backend plugged in today would receive one-pair networks and
could do nothing joint. Any 3D backend therefore needs a seam change first: the trait takes a
per-pair correlation cube `(n_pairs, rows, cols)`, or a correlation provider, and production
calls it once with all pairs.

## 4. Measurement: is per-pair 2D unwrapping temporally inconsistent on synthetic data?

Scratch crate (not in repo) that path-links `dolphin-unwrap` and `whirlwind-core`. The setup is
the `unwrap_bench.rs` geometry: a 0.7-coherence background split into four regions by 0.02
moats, a 9-cycle cone plus a ramp, at 256². Each date has 8 looks sharing a common scatterer
(pair coherence γ), with per-date deformation fraction t/9. There are 10 dates, a nearest-3
network (24 ifgs, 22 triangles), per-ifg 8-look sample coherence, and 4 rayon threads.

Command: `RAYON_NUM_THREADS=4 N=256 DATES=10 LOOKS=8 unwrap-proto closure`. Machine: Apple M2
Pro (Mac14,10), 12 cores, macOS 26.6.2, host heavily contended (load ~50). Misclosure =
`round((u_ij + u_jk − u_ik)/2π) ≠ 0`, scored on truth-coherent pixels (γ ≥ 0.3):

| per-pair 2D unwrapper | single reference pixel per ifg: px with any misclosure | one reference per coherent region: px with any misclosure | triangle-pixels misclosed (per-region ref) |
|---|---:|---:|---:|
| native, production 64-px tiling | **49.92%** | 0.265% | 0.021% |
| native global (replica, CRLB cost) | 0.283% | 0.265% | 0.021% |
| native global (replica, Whirlwind cost) | 0.287% | 0.265% | 0.021% |
| Whirlwind end-to-end | 0.282% | 0.260% | 0.020% |

Read:

- **Inside a coherent region, per-pair 2D unwrapping is already closure-consistent** to 0.02%
  of triangle-pixels on this stack, for every solver. A joint solve has nothing to fix there.
- **The large misclosure is between regions.** Native's tiled path seeds moat-separated regions
  with independent integer offsets per ifg (`tile.rs` header: "disconnected groups seed
  independently, mirroring SNAPHU"). Those offsets don't close around triangles. This is a
  per-component integer-offset problem, the exact thing loop-closure correction (#142) solves:
  one integer per component per triangle. It needs no 3D unwrapper.
- **This synthetic run does not reproduce Houston.** #138's ±7-cycle spatially organized
  structure inside a *single* conncomp didn't appear. That points at component labelling
  (`conncomp::segment` at `min_corr = 0.15` with a scene minimum of 0.268: one component) and
  at in-ifg region offsets the labels can't see, not at temporal inconsistency of in-region
  unwraps. Whirlwind's reliability-based labelling is a candidate fix for that, and belongs
  with #138/#142. Its edges become boundaries where a ±1-cycle flip is cheap
  (arXiv:2609.36267 §II-D; `dolphin/unwrap/_whirlwind.py: conncomp_reliability`).

## 5. Cost of the alternative

A dense-grid EMCF in dolphinRust would mean: a temporal-graph builder restricted to planar
(Hop3-style) networks; one temporal MCF per spatial arc, ~2 per pixel (~2×10⁶ temporal solves
per 1024² frame, each over the ifg-triangle dual graph); then the existing spatial MCF per ifg;
plus the seam change in §3. That is a new solver family roughly the size of the native
unwrapper build. The only published comparison (§2) has it losing to a 2D unwrapper on coverage,
accuracy, and runtime.

## 6. Conclusion: DEFER

**Reason.** (1) spurt is sparse-point EMCF. Its only published head-to-head (arXiv:2609.36267
Table III) keeps 37/103 GPS stations at r = 0.77 against 101/103 at r = 0.93 for a 2D
unwrapper, at 3.3× the runtime. (2) On synthetic stacks, per-pair 2D unwrapping is
closure-consistent within coherent regions (0.02% of triangle-pixels). The large misclosure is
per-component integer offsets, which #142 targets directly. (3) #142 is already
`backlog-ready`, the cheaper path #146's own gate says to measure first. (4) The trait seam no
longer delivers a network to the backend (§3), so "drop in behind the existing seam" isn't true
today.

**Reopen when all of these hold:**

1. #142 has landed and been run on the Houston stack (#138 receipts,
   `eo md/reports/new-pipeline-2026-09-13/artifacts/kpcd-closure-2026-09-19/`), and the
   frame-rejection rate is still >20% (from 79.3%);
2. the remaining failures are shown to be **inside** properly labelled components, not
   between them. Run with a labelling that actually partitions the Houston scene, e.g.
   reliability-based conncomps, or a raised `conncomp_min_corr`;
3. a dense-grid EMCF prototype, or spurt itself run through dolphin on the same stack, beats the
   #142-corrected per-pair result on station agreement (KPCD/MRHK) without losing coverage.

When it reopens, the first code change is the seam (§3). Add a contract test that a backend
receives every pair of the network in a single call from `unwrap_phase_correlated_network`.

## Sources

- `github.com/isce-framework/spurt` @ `4be99de`: `src/spurt/workflows/emcf/_solver.py`,
  `_settings.py`, `_cli.py`, `src/spurt/graph/_hop3.py`, `src/spurt/mcf/_ortools.py`.
- `github.com/isce-framework/dolphin` @ `6edd7e1`: `src/dolphin/unwrap/_unwrap_3d.py`,
  `src/dolphin/workflows/config/_unwrap_options.py` (`SpurtOptions`),
  `src/dolphin/unwrap/_whirlwind.py`.
- Pepe & Lanari, "On the extension of the minimum cost flow algorithm for phase unwrapping of
  multitemporal differential SAR interferograms," IEEE TGRS 44(9):2374–2383, 2006,
  doi:10.1109/TGRS.2006.873207.
- Hooper & Zebker, "Phase unwrapping in three dimensions with application to InSAR time
  series," JOSA A 24(9):2737–2747, 2007, doi:10.1364/JOSAA.24.002737.
- Staniewicz et al., "A Probabilistic-Cost Algorithm for Large-Scale 2D Phase Unwrapping,"
  arXiv:2609.36267 (2026), §II-D, §III-E, §III-F, Table III.
- This repo: `crates/dolphin-workflows/src/unwrap_backend.rs`, `displacement.rs:4878`,
  `crates/dolphin-unwrap/src/native/{tile,conncomp}.rs`, issues #138, #142.
