# NISAR azimuth-block splitting for full-frame GSLC (issue #105)

**Date:** 2026-10-31
**Question:** Should dolphinRust port dolphin's azimuth-block split (isce-framework/dolphin
PR #717 + #723) to parallelize full-frame NISAR GSLC processing?
**Answer:** No, not for throughput. The mechanism behind dolphin's speedup does not exist in
dolphinRust. The real full-frame limit is memory, and block splitting in dolphin's form would not
fix that either. See Conclusion.

## 1. What dolphin built (verified against the source)

- **PR #717** (merged 2026-06-01, `gh pr view 717 -R isce-framework/dolphin`) adds
  `workflows/_block_split.py`. When `input_options.azimuth_blocks > 1`, a frame is cut into N
  azimuth blocks, each block runs as a synthetic burst in the existing
  `ProcessPoolExecutor`, and outputs are cropped to the central rows and stitched. No
  phase-linking change.
- **Where the 3.88x comes from.** The PR's own benchmark (7 GSLCs, T042 F070, 8 blocks,
  `taskset -c 0-33`) lists "active threads FULL: 4" and "active threads BLK: 32". FULL took
  139.79 min and BLOCKED took 35.99 min. The PR attributes this to escaping the GIL and to
  Python control flow being "strictly serial" within one process. The gain is 8x more cores
  doing work, not a better algorithm.
- **Halo.** `_min_halo_rows` = `max(half_window_y, similarity_radius*stride_y,
  (11//2)*stride_y) + 5`. That is one window plus a safety margin. It ignores the compressed-SLC
  dependency chain. The module docstring also says each block still opens a full-frame
  VRTStack, so per-block RAM "scale[s] with the *full* frame".
- **PR #723** (merged 2026-06-09) stitches per-block compressed SLCs.

## 2. What dolphinRust already does

- **It already splits the frame into blocks.** `link_one_burst` → `phase_link_tiled_impl`
  (`crates/dolphin-workflows/src/displacement.rs:2795`, `:2964`) cuts every burst into
  `worker_settings.block_shape` tiles (default 256²) via `tiling::plan_tiles`. A NISAR frame is
  one "burst" on this path, so it is already tiled.
- **The halo is exact; dolphin's is not.** `plan_tiles` pads by `depth·(h+s)`, where depth is
  the number of ministacks (`displacement.rs:2982`, `tiling.rs:39-47,120-126`). That keeps the
  tiled run bit-identical to a whole-burst run (`tiled_phase_link_is_bit_identical_to_whole_burst`,
  `displacement.rs:7899`). dolphin's one-window halo would break that contract.
- **Parallelism is per pixel, not per process.** The tile loop (`displacement.rs:2999`) and the
  burst loop (`:421`) are serial. Each tile's covariance and estimation run on the global rayon
  pool. There is no GIL and no single-thread control flow to escape.

## 3. Measurements (this session)

Scratch crate at `…/scratchpad/research-seq/bench` (not in the repo). It has path dependencies
on this worktree's crates and a release profile with LTO off and codegen-units 16, built with
`HDF5_DIR=/opt/homebrew/opt/hdf5 CARGO_TARGET_DIR=…/research-seq/target cargo build --release`.
Host: 12 cores (8P+4E), 32 GB.

**Load caveat:** other agents' builds kept the load average at 22–45 for the whole session.
Absolute times are inflated. To limit the effect, configurations were interleaved and the
minimum of each is reported.

**(a) Intra-tile scaling.** `research-bench scale --rounds 5`. Calls `run_sequential` on one
production-shaped tile: 15 dates, 270×284 (256² plus the default (7,14) halo), EMI, rect window.

| threads | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| s/tile (min of 5) | 2.138 | 1.150 | 0.669 | 0.544 | 0.448 |
| speedup | 1.00 | 1.86 | 3.20 | 3.93 | 4.77 |

The flattening past 4 threads lines up with the background load (about 2 free cores per 12).
Under this load the 8- and 12-thread rows say nothing reliable about scaling.

**(b) Block-level concurrency vs serial tiles.** This is the direct test of the #717 idea.
`research-bench blocks --rows 1024 --cols 1024 --threads 12 --rounds 3`: a 15×1024×1024 frame,
16 tiles of 256², split into K contiguous azimuth groups. The groups run concurrently, each on
its own rayon pool of 12/K threads.

| K blocks × threads | 1×12 | 2×6 | 4×3 | 6×2 | 12×1 |
|---|---|---|---|---|---|
| wall (min of 3) | 6.40 s | 6.13 s | 6.50 s | 6.32 s | 6.76 s |
| vs serial tiles | 1.00x | 1.04x | 0.98x | 1.01x | 0.95x |

Block concurrency gains nothing beyond noise. The rayon pool already fills whatever cores are
free.

**(c) Peak memory vs frame size.** `NISAR_DIR=… /usr/bin/time -l research-bench nisar --n 10
--size S --block B`. Runs full `run_displacement` on a synthetic NISAR-layout stack from
`dolphin_io::nisar_fixture`: 10 dates, strides 2×2, defaults otherwise, SNAPHU unwrap.

| frame | block 256 RSS | block 1024 RSS | wall (block 256) |
|---|---|---|---|
| 512² | 192 MB | — | 1.67 s |
| 1024² | 594 MB | 1460 MB | 5.29 s |
| 2048² | 1411 MB | 1762 MB | 26.52 s |

At a fixed block size, peak RSS grows with frame area: about 340 B per input pixel at 2048²
with N=10. Tiling bounds the phase-linking kernel, but not the frame-sized accumulators:
`TiledOutput` holds `(N, out_rows, out_cols)` Cf64 phase and f64 CRLB (`displacement.rs:3657`),
and the stitch, network, unwrap and inversion that follow are all whole-frame in memory. No real
NISAR frame was run. One 7.2 GB granule is a single date (VALIDATION.md §NISAR), and the
≥15 GB a repeat pair needs is not on this disk. So no frame-scale RSS figure is claimed.

**Not measured: read/compute overlap.** `read_burst_tile` (`displacement.rs:3143`) reads each
date's window serially, and the read does not overlap compute. The fixture is contiguous
uncompressed HDF5, so timing it would say nothing about a chunked or compressed real granule.

## 4. Conclusion — REJECT (throughput), DEFER (memory)

**REJECT** porting the block-as-burst split for speed. dolphin's 3.88x comes from moving 4
active threads to 32 by escaping single-process Python. dolphinRust's tiles already run on all
free cores, and (b) measured 0.95–1.04x for block-level concurrency. If a split is ever built,
it must reuse `plan_tiles`'s `depth·(h+s)` halo, not `_min_halo_rows`. The latter is not exact
under the compressed-SLC chain.

**DEFER** the actual full-frame limit: frame-area memory in the accumulator and downstream
stages, per (c). Reopen when **either**:
1. a real NISAR repeat-pass frame is run and its peak RSS from `/usr/bin/time -l` passes the
   target host's limit (for example, the GroundPulse Fargate task size); **or**
2. that run's `pl_breakdown` log shows `read_s / compute_s > 0.25`. That reopens
   read-ahead/overlap in `phase_link_tiled_impl`, which is a separate item from block splitting.

The fix for case 1 would stream per-tile outputs to disk instead of `TiledOutput`, so the frame
never sits in RAM. It needs its own design. It is not dolphin's azimuth split.
