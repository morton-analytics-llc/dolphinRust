# dolphinRust vs published DISP-S1 production wall-clock

Issue #104. The reference is Staniewicz et al., "Near-Real-Time InSAR Phase
Estimation for Large-Scale Surface Displacement Monitoring", arXiv:2511.12051
(IEEE TGRS, Jan 2026) — the dolphin team's own numbers from real DISP-S1 North
America production, not a synthetic bench.

Measured **2026-09-03**, dolphinRust **1.6.0**. Raw numbers:
[`disp_s1.json`](disp_s1.json). Reproduce with the command at the bottom.

> **Read the scaling caveat before quoting any multiple.** The two runs are not
> the same experiment. Theirs is a 27-burst DISP-S1 frame on a 32-vCPU EC2
> instance; this is a 3.93 Mpix burst crop on a 12-core laptop. The stage
> *shares* below are directly comparable and carry no assumption. The
> *multiples* depend on a frame pixel count the paper does not publish, so they
> are derived here, banded, and should be read as "roughly half an order of
> magnitude", not as a measured speedup.

## What matches exactly

The processing geometry is the same on both sides, which is what makes any
comparison possible at all:

| Parameter | DISP-S1 production | This run |
|---|---|---|
| Input CSLC posting | 5 m (X) × 10 m (Y) | 5 m × 10 m |
| Strides (decimation) | 6 × 3 → 30 m output | 6 × 3 → 30 m output |
| Ministack size `M` | 15 dates | 15 dates |
| Half-window | 11 × 5 | 11 × 5 |

## Measured — dolphinRust

15 dates of OPERA CSLC-S1 `T005-008704-IW1` (2018-01-06 → 2018-05-06), a
936 × 4197 crop = **3.93 Mpix/date**, native unwrapper (default), Apple M2 Pro,
12 cores, 32 GiB, rayon on all cores. 3 reps into a wiped work dir each time;
warm = median of the last two.

**Warm total 20.0 s** (cold 20.5 s; reps 20.53 / 19.90 / 20.11 s).
**Peak RSS 0.70 GiB** (`/usr/bin/time -l`, whole process).

| stage | seconds | % of wall |
|---|---|---|
| **unwrap** | **15.225** | **76.1 %** |
| phase_linking | 2.691 | 13.5 % |
| write | 1.435 | 7.2 % |
| timeseries | 0.449 | 2.2 % |
| velocity | 0.033 | 0.2 % |
| corrections | 0.009 | 0.0 % |
| stitch / geometry_precheck / network / loop_closure | < 1e-4 each | ~0 % |

Stage times are the library's own `timed(...)` tracing events
(`crates/dolphin-workflows/src/displacement.rs:296`). The per-stage
`rss_kib`/`peak_rss_kib` breadcrumbs read 0 on this host, so whole-process
max-RSS is the only real memory number here and per-stage RSS is not reported.

## Published — DISP-S1 production

Quoted, not re-derived. Over 6,000 mini-stacks of 15 dates each, North America,
on EC2 instances with **32 vCPUs and 64 GiB**:

- "Median run times were **6.7 hours per mini-stack** in total, of which **∼80 %**
  is associated with phase unwrapping."
- "the wrapped phase estimation (phase linking, persistent scatterer selection)
  has a very narrow spread and consistently runs in **∼90 minutes** per
  mini-stack."

## The result that needs no scaling assumption

The bottleneck profile matches. Both systems are unwrap-dominated to within a
few points, which is the strongest apples-to-apples statement available here
because it is a ratio within each run:

| stage share | DISP-S1 | dolphinRust |
|---|---|---|
| unwrapping | ~80 % | **76.1 %** |
| phase-linking (+ PS selection) | ~22 % (90 min / 6.7 h) | **13.5 %** |

So the native in-process unwrapper has *not* moved unwrapping off the critical
path — it is still three-quarters of the run, exactly as it is for dolphin. What
did change is the phase-linking share: 13.5 % here vs 22.4 % there, consistent
with the phase-linking work being relatively cheaper in this implementation. That
is a shift in the mix, not by itself a throughput claim.

## The multiples, and why they are a band

The paper gives no frame pixel count, so it is derived. A frame is "nominally
27 bursts (9 bursts along-track for subswaths IW1, 2, and 3)". This crop spans
20.99 km × 9.36 km, against an IW1 subswath ~83 km in range and a burst ~20 km
in azimuth — about 12 % of one burst, so a burst is ~33 Mpix and a 27-burst
frame lands near **800 Mpix/date** once along-track burst overlap is allowed for.
Treat that as ±25 %.

Throughput in Mpix·dates per second, band from a 600–1000 Mpix frame:

| | dolphinRust (measured) | DISP-S1 (derived) | ratio |
|---|---|---|---|
| end-to-end | 2.95 | 0.37 – 0.62 | **~4.7 – 7.9×** (centre 5.9×) |
| phase-linking | 21.9 | 1.67 – 2.78 | **~7.9 – 13.1×** (centre 9.9×) |

**On fewer cores.** This ran on 12 M2 Pro cores against their 32 vCPUs. Per-core
the gap is wider — roughly 16× per vCPU end-to-end — but vCPUs are SMT threads,
not physical cores, and the instance type is unnamed, so per-core normalization
is softer than the aggregate and is offered only as a direction, not a figure to
quote.

## Honest summary

- **The stage profile reproduces**: unwrap 76 % here vs ~80 % published. The
  comparison is structurally sound, and unwrapping remains the thing to attack.
- **End-to-end is roughly 5–8× faster per pixel** on ~⅓ the cores, and
  **phase-linking roughly 8–13×**. Both rest on a derived frame size; neither
  should be quoted as a measured number without the band.
- **Memory is not the constraint at burst scale**: 0.70 GiB peak against their
  64 GiB instances.
- **What this does not show.** One burst crop, one scene, one host, no GPU path
  measured, and no multi-burst frame assembly — frame-scale stitching, I/O, and
  memory behaviour are exactly where a laptop burst run is least informative. A
  real head-to-head needs a full 27-burst frame on comparable hardware. The
  issue also cited a "GPU install 5–20× over CPU" claim; the paper's runtime
  section reports CPU-only EC2 runs and does not state that, so it is not
  carried into this comparison.

## Reproduce

```sh
cargo build --release -p dolphin-cli
oracle/.venv/bin/python bench/disp_s1_bench.py --dates 15 --reps 3
# writes bench/disp_s1.json; work dirs and logs under bench/runs/disp_s1/ (git-ignored)
```
