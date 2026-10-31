# GPU-accelerated unwrap solve: feasibility spike (issue #147)

**Date:** 2026-10-31
**Gate (from #147):** read PyGMTSAR's actual unwrap algorithm, not its benchmark, and assess
whether the native network-simplex MCF, or a GPU-amenable alternative, could give a comparable
win without regressing `native_tiling_contract.rs` / `native_dense_parity.rs`.
**Conclusion:** REJECT the GPU solve. Profile the CPU solver first (see §5).

## 1. What the "PyGMTSAR GPU unwrapper" actually is

- **It isn't in PyGMTSAR.** The `AlexeyPechnikov/pygmtsar` HEAD (`65a9054`, 2026-10-02)
  unwraps through SNAPHU (`pygmtsar/pygmtsar/Stack_unwrap.py: unwrap_snaphu`) and a 1D temporal
  unwrapper. Its README announces a successor, InSAR.dev.
- **The GPU code is in InSAR.dev (`insardev` on PyPI, 2026.3.21.post3).** The function is
  `insardev/utils_unwrap2d.py: irls_unwrap_2d`. It runs iteratively reweighted least squares
  on the weighted L1 gradient objective `min Σ w|∇φ − wrap(∇ψ)|`, solving each weighted-L2 step
  with DCT-preconditioned conjugate gradient in PyTorch (CUDA / MPS / CPU). That is a continuous
  relaxation solved approximately, not min-cost flow.
- **The license rules out use.** `insardev` ships under "InSAR.dev Source-Available License
  1.0" (`dist-info/licenses/LICENSE`). It excludes any commercial, consulting, or
  government-funded use, so the code can't be used and isn't a reference for a port. I read the
  docstring only, to identify the algorithm family.
- **The published method is open.** IRLS for phase unwrapping is Dubois-Taine, Akiki &
  d'Aspremont, arXiv:2401.09961 (2024). Its PCG-based IRLS runs on GPU, benchmarked on an RTX8000
  against CPU SNAPHU. The paper says outright that it **drops 2π-congruence** (output not
  guaranteed to equal input mod 2π, §2, "we argue that this is an…"). It offers rounding as an
  optional post-step and recommends SNAPHU-generated weights. An MIT-licensed PyTorch
  implementation exists (`smuinsar/rapidphase`, PyPI 0.1.5).
- **The 15× figure is unconfirmed.** I couldn't find a source for "~15× on Apple Silicon,
  12,800², 5 min → 20 s" in either repo or on PyPI.

So the comparison #147 proposes isn't like-for-like. The GPU unwrapper solves a different,
non-congruent relaxation of the objective the native unwrapper solves exactly. Costantini 1998:
the min-cost-flow problem is the integer weighted-L1 problem on gradient corrections.

## 2. Where native unwrap CPU actually goes (measured)

Machine: Apple M2 Pro (Mac14,10), 12 cores, 32 GB, macOS 26.6.2. The host was heavily
contended during this session (load average ~50, other agents' builds and tests), so **every
number below is CPU time or sample share, not wall time**. Builds:
`HDF5_DIR=/opt/homebrew/opt/hdf5`, `CARGO_TARGET_DIR=<scratchpad>`, release, tree `25ab07a`.

**(a) Sampled profile of the shipped native path.** I ran
`EPOCHS=30 ROWS=1024 COLS=1024 DENSE=1 TILE=16 BACKEND=native RAYON_NUM_THREADS=4
target/release/examples/unwrap_bench` with `/usr/bin/sample <pid> 25 1`. The densest ifg has
27,750 residues (2.65%), and 16×16 tiles match the production 64-px auto-tiling. Of 70,877
non-idle samples:

| bucket | samples | share |
|---|---:|---:|
| `native::unwrap_grid` self (network simplex inlined) | 53,291 | 75.2% |
| allocator / memmove frames whose nearest dolphin caller is `unwrap_grid` | 15,532 | 21.9% |
| ifg formation + `atan2`/`sincos` | 1,001 | 1.4% |
| `tile::unwrap_tiled` / conncomp self | 434 | 0.6% |
| other | ~620 | <1% |

**The min-cost-flow solve is ~97% of native unwrap CPU.** Everything per-pixel and
embarrassingly parallel (gradients, residues, costs, integration, seam reconciliation, ifg
formation) totals ~3%.

**(b) The per-pair correlation added by #137 (today, `25ab07a`).** `unwrap_bench` calls
`unwrap_network` directly, so it skips `unwrap_phase_correlated_network` and its 31-tap
Gaussian correlation. I timed a verbatim copy of `interferogram_correlation`
(`unwrap_backend.rs:94-151`) against `unwrap_native` with 64-px tiling. Both ran on the same
simulated 8-look 1024² ifg (bench geometry), measured as process CPU-s over 3 reps,
`RAYON_NUM_THREADS=2`. Two runs gave 0.397 / 0.357 CPU-s of correlation against
6.761 / 6.145 CPU-s of unwrap, so **correlation is 5.5% of per-pair unwrap-stage CPU**.

**Amdahl bound.** Suppose a GPU made every data-parallel part of the unwrap stage free: the
correlation (5.5%) plus the ~3% outside the solver of the remaining 94.5%. That removes ~8% of
the stage, a ceiling of **~1.09×** on the unwrap stage. At unwrap = 76.1% of end-to-end
(PLAYBOOK, DISP-S1 comparison, issue #104), that is ~6% end-to-end. A meaningful GPU win
therefore requires putting the **solver** on the GPU.

## 3. Can the solver go on a GPU?

- **Network simplex, as shipped: no.** `simplex.rs: run()` (l.147) is one pivot loop. Each
  pivot rewrites the spanning tree (`restructure`, l.282) and the potentials of the moved subtree
  before the next pricing scan can be valid. Pivot *k+1* depends on pivot *k*'s tree, so there is
  no per-pixel SIMT decomposition. The parallelism that exists is already used on CPU: rayon
  over pairs (`unwrap_backend.rs`, `NativeUnwrapBackend`) and over tiles (`tile.rs:59`).
- **IRLS/PCG on GPU: possible, but it's a different backend, not acceleration.** It would need
  (i) a rounding step to restore congruence, (ii) weights (the IRLS paper uses SNAPHU's), and
  (iii) a fresh pass of the parity gates. Those gates measure per-component integer-cycle
  agreement ≤0.5% against SNAPHU (`native_tiling_contract.rs`, `native_dense_parity.rs`). A
  relaxation that drops congruence has no guarantee of passing them. It would also add a
  PyTorch/CUDA-class dependency to a crate that is deliberately pure compute (`dolphin-unwrap`
  builds under `--no-default-features --features no-gpu` for GroundPulse, per PLAYBOOK).

## 4. The cheaper win this spike found instead: the CPU solver

On the same 1024² ifg and harness, Whirlwind's CPU solver used **1.214 CPU-s** against native's
**6.145 CPU-s**, with identical per-region cycle error vs truth (0.0008% each). That is
`whirlwind_core::unwrap_linear`: parallel-augmenting successive shortest paths with Dial
buckets, capacity-1 arcs, untiled (`scottstanie/whirlwind-insar` `0ff1df6`, BSD-3-Clause OR
Apache-2.0; Staniewicz et al., arXiv:2609.36267). So Whirlwind spends **5.1× less CPU,
untiled, at equal accuracy** on this scene. One scene, one size, on a contended host. Treat it
as a signal to investigate, not a benchmark.

**Scaling probe.** One global solve, untiled, `RAYON_NUM_THREADS=1`, process CPU-s.
Uniform γ = 0.5, gentle ramp, 8-look simulation, so residues stay sparse
(`unwrap-proto scale`):

| grid | residues | native global CPU-s | Whirlwind CPU-s |
|---|---:|---:|---:|
| 128² | 107 | 0.111 | 0.010 |
| 192² | 270 | 0.527 | 0.023 |
| 256² | 481 | 1.510 | 0.046 |
| 384² | 1,093 | 6.545 | 0.075 |

Native grows ~n^1.86 in pixel count (59× CPU for 9× pixels). Whirlwind grows roughly linearly
(7.5×). Real data shows the same pattern: on MMX1 crops (352×554, Sentinel-1) with only 28–216
residues, a native global solve took 16–31 s wall. So native cost tracks **graph size, not
residue count**. The production 64-px auto-tiling (`displacement.rs: native_tiling`) is
largely a workaround for this: PLAYBOOK records the 8× CPU win of fine tiling as "per-tile
network simplex is superlinear".

Three observations from the code and profile point the same way:

1. **~22% of native unwrap CPU is allocator churn inside the simplex.** The tree is a
   `Vec<Vec<u32>>` child list with `retain`/`push` on every pivot (`simplex.rs:328,345,348,362`),
   and `collect_subtree` (l.311) allocates a fresh `Vec` per pivot.
2. **Each pivot does several passes linear in the moved subtree.** `collect_subtree` (l.311),
   `in_set` as a `Vec::contains` scan (l.321-322, called at l.293), and `shift_potentials`
   (l.375). The design note (`docs/native_mcf_solver.md` §Algorithm) specifies preorder
   `thread`/`depth` arrays (Kovács 2012, arXiv:1207.6381), which the code doesn't implement. The
   potential update over the subtree is inherent to network simplex. The allocation and the
   linear membership test aren't. Which of these drives the n^1.86 is unmeasured, and Arm A
   (§5) measures it.
3. The PLAYBOOK claim that network simplex beats SSP ~10× was measured against
   *unit-augmenting single-source* SSP (one Dijkstra per flow unit, PLAYBOOK "Native MCF
   solver"). It doesn't cover the multi-source parallel-augmenting SSP + Dial variant that
   Whirlwind ships.

## 5. Conclusion: REJECT (the GPU solve)

**Reason.** The solver is ~97% of native unwrap CPU, and the shipped network simplex has no
GPU mapping. The only GPU-native unwrappers found are IRLS relaxations: one non-commercially
licensed (`insardev`), one open (`rapidphase`, arXiv:2401.09961). Both drop 2π-congruence and
would be a new, unvalidated backend with a heavy dependency, not a speed-up of native. GPU work
on the data-parallel parts is capped at ~1.09× of the unwrap stage (§2). Meanwhile a CPU
solver change measured 5.1× less CPU on the same ifg (§4).

**What should happen instead (recommendation for the human design review, not filed):** a
CPU-solver measurement issue with two arms, each gated by the existing parity contracts
unchanged:

1. **Arm A.** Replace `children: Vec<Vec<u32>>` + `collect_subtree` + `in_set` with the
   thread/depth arrays the design note specifies: O(1) subtree membership by depth/thread, no
   per-pivot allocation. Acceptance: identical flows (hence bit-identical `unwrapped`) on
   `native_dense_parity` / `native_tiling_contract` / `native_mmx1_live_contract`, since the
   pivot sequence is unchanged. CPU-s at 1024² dense TILE=16 (the §2a command under
   `/usr/bin/time -l`) must drop by ≥15%. Re-run the §4 scaling probe and report the exponent.
2. **Arm B.** Benchmark a `WhirlwindBackend` behind `UnwrapBackend` (`whirlwind-core` as a path
   or git dependency; the license permits it). Acceptance: per-component disagreement ≤0.5% vs
   SNAPHU on the committed goldens and the MMX1 live contract, and ≥2× lower CPU-s than native
   on the §2a harness.

**What would reopen the GPU question:**

- after Arm A/B, the data-parallel share of unwrap CPU exceeds ~30% (re-run §2a and §2b); or
- a published **congruent** GPU MCF/SSP unwrapper appears under a commercially usable license,
  with SNAPHU parity reported per component.

## Sources

- PyGMTSAR, `github.com/AlexeyPechnikov/pygmtsar` @ `65a9054` (2026-10-02): `Stack_unwrap.py`,
  README (InSAR.dev announcement).
- InSAR.dev, `pypi.org/project/insardev` 2026.3.21.post3: `utils_unwrap2d.py: irls_unwrap_2d`
  docstring; `LICENSE` (InSARdev-SAL-1.0).
- Dubois-Taine, Akiki, d'Aspremont, "Iteratively Reweighted Least Squares for Phase
  Unwrapping," arXiv:2401.09961 (2024).
- `smuinsar/rapidphase` (MIT), PyPI 0.1.5.
- Staniewicz, Gunter, Mirzaee, Govorcin, Oliver-Cabrera, Fattahi, "A Probabilistic-Cost
  Algorithm for Large-Scale 2D Phase Unwrapping," arXiv:2609.36267 (2026-09-28);
  implementation `github.com/scottstanie/whirlwind-insar` @ `0ff1df6`.
- Costantini, "A novel phase unwrapping method based on network programming," IEEE TGRS 36(3),
  1998 (as cited in arXiv:2609.36267 ref. [14]).
- Measurement scratch code (not in repo): a crate under the session scratchpad that path-links
  `dolphin-unwrap` and `whirlwind-core`. The full commands are in §2 and §4.
