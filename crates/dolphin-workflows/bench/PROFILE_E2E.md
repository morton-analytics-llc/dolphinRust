# End-to-end DisplacementWorkflow profile (native-unwrap default)

Full `run_displacement` profiled at burst scale, **2048²**, at **12 and 30
epochs**, single-reference network, `half_window=5` (11×11), `strides=1×1` (full
res), `ministack_size=15`, native auto-tiled unwrap (the shipped default). macOS,
Apple-silicon, release build.

## How it was measured (macOS-correct)

- **Per-stage wall + CPU·s + RSS:** `examples/profile_e2e.rs` installs a tracing
  layer that snapshots `getrusage(RUSAGE_SELF)` on each library `timed(...)` stage
  event. CPU·s/stage = Δ(user+sys) between consecutive stage events (→ effective
  cores = CPU·s/wall, i.e. parallel efficiency). `rss_hwm` = `ru_maxrss`
  high-water (MiB) reached by each stage; `Δrss` = new high-water that stage added.
- **Authoritative max-RSS:** `/usr/bin/time -l` around the whole run.
- **Heap peak + allocation sites:** `dhat-rs` global allocator (`--features
  dhat-heap`), run at 1024² (sites are scale-invariant; absolute peak read from
  `time -l`/getrusage at 2048²).
- **CPU hotspots:** `samply record` (saved profile for interactive drill-down).

Reproduce:
```
cargo build --release --example profile_e2e -p dolphin-workflows
ROWS=2048 EPOCHS=12 /usr/bin/time -l target/release/examples/profile_e2e
ROWS=2048 EPOCHS=30 /usr/bin/time -l target/release/examples/profile_e2e
# heap sites:
cargo run --release --example profile_e2e -p dolphin-workflows --features dhat-heap   # ROWS=1024
# cpu:
samply record -- target/release/examples/profile_e2e
```

## Per-stage table

### 12 epochs, 2048²

| stage | wall_s | CPU·s | cores | rss_hwm (MiB) | Δrss (MiB) | %peak |
|---|--:|--:|--:|--:|--:|--:|
| phase_linking | **41.73** | **346.7** | 8.31 | 2909 | +2847 | 67% |
| stitch | 0.000 | 0.0 | — | 2909 | 0 | 67% |
| network | 0.000 | 0.0 | — | 2909 | 0 | 67% |
| unwrap | 1.18 | 7.3 | 6.23 | 4206 | +1297 | 97% |
| timeseries (SBAS L2) | 5.65 | 55.2 | 9.78 | 4206 | 0 | 97% |
| corrections (no-op¹) | ~0 | 0.1 | — | 4206 | 0 | 97% |
| velocity | 0.06 | 0.2 | 3.26 | 4206 | 0 | 97% |
| write (COG) | 3.06 | 3.1 | 1.01 | 4343 | +136 | 100% |
| **phase_linking split** | | | | | | |
| └ CSLC windowed read | 0.34 (wall) | | | | | |
| └ covariance+estimator | 41.10 (wall) | | | | | |

Totals: wall **51.9 s**, process CPU **412.7 s**, **max-RSS 4.24 GiB**
(`time -l`: 4 553 539 584 B).

### 30 epochs, 2048²

| stage | wall_s | CPU·s | cores | rss_hwm (MiB) | Δrss (MiB) | %peak |
|---|--:|--:|--:|--:|--:|--:|
| phase_linking | **89.89** | **740.0** | 8.23 | 5768 | +5706 | 74% |
| stitch | 0.000 | 0.0 | — | 5768 | 0 | 74% |
| network | 0.000 | 0.0 | — | 5768 | 0 | 74% |
| unwrap | 2.78 | 16.8 | 6.05 | 6544 | +776 | 84% |
| timeseries (SBAS L2) | **25.11** | **266.6** | 10.62 | 7573 | +1029 | 97% |
| corrections (no-op¹) | ~0 | 0.2 | — | 7586 | +13 | 97% |
| velocity | 0.30 | 0.7 | 2.33 | 7835 | +250 | 100% |
| write (COG) | 6.60 | 6.6 | 1.00 | 7835 | 0 | 100% |
| **phase_linking split** | | | | | | |
| └ CSLC windowed read | 0.91 (wall) | | | | | |
| └ covariance+estimator | 88.14 (wall) | | | | | |

Totals: wall **125.0 s**, process CPU **1031 s**, **max-RSS 7.65 GiB**
(`time -l`: 8 215 953 408 B).

¹ corrections runs with no correction files configured → ~0 wall; the tiny CPU
delta is getrusage quantization, so its `cores` ratio is meaningless and omitted.

## The new dominant bottleneck — phase-linking (covariance + estimator)

Confirmed with numbers, not assumed. With the native unwrapper as default, **unwrap
collapsed from the historical 76% of wall (SNAPHU subprocess) to ~2.2–2.3%**
(1.18 s / 2.78 s). What took its place is **phase-linking**:

| | 12 ep | 30 ep |
|---|--:|--:|
| phase_linking % of wall | **80.4%** | **71.9%** |
| phase_linking % of CPU·s | **84.0%** | **71.8%** |
| unwrap % of wall | 2.3% | 2.2% |

And within phase-linking the cost is **compute, not I/O**: the read/compute split
is 0.34 s vs 41.10 s (12 ep) and 0.91 s vs 88.14 s (30 ep) — the windowed CSLC
read is <1% of the stage. The covariance accumulation + per-pixel estimator
(faer self-adjoint eigendecomposition over the 11×11 window stack) is the hot
path. It scales ~2.1× for 2.5× epochs (41.7→89.9 s) — sub-linear in epochs (the
ministack caps the covariance rank at 15) but linear in pixels.

**Secondary cost: SBAS timeseries inversion.** 5.65 s → 25.11 s for 12→30 epochs
(**4.4× for 2.5× more epochs — superlinear in the number of ifgs**), 20% of wall
at 30 epochs and the best-parallelized stage (10.6 effective cores). This is the
next thing to watch as epoch counts grow; at 30 epochs it already costs 10× what
unwrap does.

## Top friction points

1. **Serial COG write** — `write` runs at **1.0 core** (3.1 s / 6.6 s), GDAL
   single-threaded and not overlapped with compute. At 30 ep it is 5% of wall doing
   nothing in parallel; the most obvious cheap win (thread per output band, or
   overlap with velocity/corrections which precede it).
2. **Retained Cf64 linked-phase cube + Cf32→Cf64 upcast.** `read_burst_tile`
   upcasts each tile Cf32→Cf64; the assembled `pl` history is held as Cf64
   (16 B/px) for the whole downstream tail: N×2048²×16 B = **805 MiB (12 ep) /
   2.01 GiB (30 ep)** resident floor that never drops before `write`. This Cf64
   doubling is the single largest retained allocation.
3. **Per-stage transient cubes stack on top of the retained PL cube** rather than
   overlapping — see the memory timeline below. unwrap's ifg stack (+1.3 GiB) and
   SBAS's displacement working set (+1.0 GiB) are allocated while the full PL cube
   is still resident, so peaks add instead of reusing.
4. **I/O is *not* a friction point here** — synthetic local HDF5; windowed read is
   <1 s. On real S3-staged CSLC this stage would grow and is the place to overlap
   read with compute, but it is not the current bottleneck.

## Memory high-water timeline

`ru_maxrss` is monotonic (high-water), so Δrss attributes *new* peak growth to the
stage that caused it.

```
12 ep:  start ~62 →  PL 2909 (+2847)  →  unwrap 4206 (+1297)  →  …  →  write 4343 (peak)
30 ep:  start ~62 →  PL 5768 (+5706)  →  unwrap 6544 (+776)  →  SBAS 7573 (+1029)  →  velocity 7835 (peak)
```

- **Phase-linking builds the bulk of the footprint** (67%/74% of the final peak is
  reached by the end of PL): the retained Cf64 `pl` cube + the transient per-tile
  N×N coherence cubes across rayon threads.
- The **global peak is reached late** (unwrap→SBAS→velocity) because those stages
  allocate transient cubes *on top of* the still-resident PL output, not because
  they each individually need that much.
- **Answer to "does the block-tiled PL + windowed readers still hold ~1.08 GB?":
  No — not at full-res 2048² burst scale.** The block tiling does its job on the
  *input* side (read working set is tile-bounded, <1 s, negligible RSS), but the
  *retained outputs* scale with N×area and are unbounded by tiling: peak **4.24 GiB
  (12 ep) / 7.65 GiB (30 ep)**. The 1.08 GB figure was a smaller/strided
  configuration; at `strides=1×1`, full 2048², the Cf64 linked-phase cube is the
  floor and unwrap/SBAS transients are the spikes on top.

## Tooling notes — dhat and samply (honest limits hit)

Both specialized tools were attempted; both hit real limits on this workload /
platform, so the quantitative per-stage numbers above come from `getrusage`
(exact at stage granularity) and `time -l` (authoritative max-RSS), not from these.

- **dhat-rs (heap sites): intractable for this pipeline.** The `dhat-heap` feature
  + global allocator is wired (`--features dhat-heap`), but dhat builds a backtrace
  on *every* allocation, and this pipeline (ndarray/faer temporaries) does a huge
  number of small allocations. It burned **>29 min of CPU at 512², 8 ep without
  producing a profile**, so it was killed. Per-stage peak *heap* is therefore read
  from the `ru_maxrss` high-water timeline above (for this array-dominated pipeline
  RSS closely tracks heap), and the allocation drivers are identified structurally:
  the retained Cf64 `pl` cube (N×2048²×16 B), the unwrap ifg stack (+1.3 GiB at
  12 ep), and the SBAS displacement working set (+1.0 GiB at 30 ep). To use dhat
  here it would need allocation sampling (record 1/N allocs), which dhat-rs lacks.
- **samply (CPU hotspots): captured but unsymbolicated on macOS.** `samply record
  --save-only` produced a valid profile (1024², 12 ep), but **100% of leaf frames
  resolve to raw addresses** — samply found no dSYM for the release example binary
  even with `CARGO_PROFILE_RELEASE_DEBUG=true`; macOS needs a `dsymutil` bundle
  alongside. The profile is usable for interactive drill-down once symbolicated,
  but yields no named hotspots as-is. Quantitative CPU attribution is taken from
  the exact per-stage `getrusage` CPU·s (which already answers the only question
  that matters here: **phase-linking is 84% / 72% of CPU at 12 / 30 ep; unwrap is
  ~2%**). Intra-`phase_linking` hotspots are structurally the faer self-adjoint
  eigendecomposition + covariance accumulation per 11×11 window (the estimator),
  confirmed by the read/compute split (compute is 99% of the stage's wall).

Net: the profile's measured claims rest on `getrusage` + `time -l`; the two
sampling tools are recorded as attempted with their platform limits, not faked.
