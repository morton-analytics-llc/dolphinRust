# AOI-bounded displacement benchmark

Measured 2026-07-22 on an Apple M2 Pro (`arm64`) with the release CPU/no-GPU
build. Both runs used two overlapping georeferenced synthetic bursts, 12 dates,
512×512 native pixels per burst, three four-date ministacks, a 5×5 half-window,
native unwrap, genuine average coherence enabled, and stride 1×2. Native posting
was 30 m azimuth × 15 m range, yielding the 30 m × 30 m output posting used by
the GroundPulse serving grid. The bounded run requested the central 25% target;
the versioned dependency halo expanded the analysis and per-burst native reads
before the final 128×112 trim.

| Run | Returned grid | Peak RSS | Wall time | Phase-link compute |
|---|---:|---:|---:|---:|
| Full frame | 512×448 | 638 MB | 1.45 s | 0.898 s |
| Bounded target | 128×112 | 112 MB | 0.21 s | 0.118 s |

The bounded run reduced measured process peak RSS by 82.4%, total wall time by
85.5%, and the phase-linking stage by 86.9%. This is a resource receipt, not a
scientific full-burst-identity
claim: finite-halo products are AOI-local processing versions. Contract tests
separately require exact target-interior phase-link quality parity at strides
1×2 and 3×6, while unwrap/reference are evaluated on the expanded analysis
domain.

Reproduce in separate processes so `getrusage` high-water marks are independent:

```sh
ROWS=512 COLS=512 EPOCHS=12 BURSTS=2 BURST_OVERLAP=128 \
MINISTACK=4 STRIDE_Y=1 STRIDE_X=2 BOUNDED=0 cargo run --release \
  --example profile_e2e -p dolphin-workflows \
  --no-default-features --features no-gpu

ROWS=512 COLS=512 EPOCHS=12 BURSTS=2 BURST_OVERLAP=128 \
MINISTACK=4 STRIDE_Y=1 STRIDE_X=2 BOUNDED=1 cargo run --release \
  --example profile_e2e -p dolphin-workflows \
  --no-default-features --features no-gpu
```
