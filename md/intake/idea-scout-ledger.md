# Idea Scout Ledger

Gate-tracked backlog for dolphinRust. `idea-scout` writes **Build after X** candidates here as
`## DEFERRED` entries with a re-entry gate; `backlog-pipeline` reads them, evaluates the gate, and
on a successful PR moves the entry to `## SHIPPED`. Items approved for immediate build live on
GitHub as issues labeled `backlog-ready` instead — not here.

This ledger is only for items that genuinely need gate-tracking (uncertain evidence, a blocking
prior phase, an upstream product). It is not a general TODO list — PLAYBOOK.md owns the phase
roadmap, and memory owns milestones/deferrals.

Entry format:

```
### D{n} — {short title}
- **Source**: {competitive / ecosystem / inbound issue #}
- **Issue**: #{n}  (enhancement-labeled, NOT yet backlog-ready)
- **Re-entry gate**: {the "X" — a verifiable-now check or an external/human condition}
- **Design sketch**: {one or two lines, or a md/design/ path}
- **Added**: {date} by {manual | scheduled scout run}
```

## DEFERRED

### D2 — External-TEC-free (split-spectrum) ionospheric correction
- **Source**: competitive + ecosystem (MintPy split-spectrum; Earth Planets and Space 2025 practical-recipe paper)
- **Issue**: #23 (closed 2026-08-10, gate checked and NOT met — closed as "deferred with
  the gate unchanged" rather than left open; this ledger entry is the live record)
- **Re-entry gate**: a real scene is found where IONEX/GNSS TEC coverage is missing or
  demonstrably insufficient, OR the real multi-date NISAR validation work (ROADMAP.md's
  "NISAR calibrated-data timing" external gate) surfaces an ionospheric residual large
  enough to justify a second correction path. Still unmet as of 2026-09-01 — see new
  evidence below; no NISAR validation run and no real-data IONEX shortfall reported since
  #23 was closed.
- **Design sketch**: analytic fixture with injected sub-band ionospheric phase ramp;
  recover known TEC/delay via split-spectrum estimator; additive fallback/cross-check to
  the existing IONEX path in `dolphin-corrections::ionosphere`, off by default.
- **Added**: 2026-08-01 by scheduled scout run
- **Reconciled**: 2026-08-31 scheduled scout run (ledger was stale — issue closure on
  2026-08-10 was never reflected here)
- **New evidence (2026-09-01 scheduled scout run, not sufficient to close the gate)**:
  NISAR L-band data is now flowing — calibrated observations from acquisitions on/after
  2026-06-17 are at the ASF DAAC, with provisional L-band products released 2026-07-20
  explicitly flagged by the mission as more uncertain at higher latitudes "where
  ionospheric variability" is higher. NISAR's built-in dual sub-bands are architecturally
  suited to the split-spectrum method, and a practical-recipe paper targeting NISAR
  specifically now exists (Earth Planets and Space, doi:10.1186/s40623-025-02206-9,
  2025-06). This is evidence the *precondition* for the gate (real NISAR data existing)
  is now met, but the gate itself requires an actual multi-date NISAR validation run
  surfacing a measured ionospheric residual, or a real scene with demonstrated IONEX/GNSS
  shortfall — neither has happened yet. No new issue opened; re-check at the next
  scheduled run once NISAR validation work (if any) produces a residual measurement.

### D7 — NISAR azimuth-block splitting for parallel full-frame processing
- **Source**: researcher (isce-framework/dolphin PR #717 + #723, merged 2026-06, unreleased on `main`)
- **Issue**: #105 (enhancement-labeled, NOT backlog-ready)
- **Re-entry gate**: a real full-frame NISAR GSLC processing case exists (validation
  harness or GroundPulse NISAR integration) where single-block wall-clock or memory is
  measured as an actual bottleneck.
- **Design sketch**: azimuth-block-splitting pass ahead of the existing burst-pool
  dispatch in `dolphin-workflows`, matching dolphin's halo computation, plus a
  compressed-SLC/output stitcher for block boundaries.
- **Added**: 2026-09-01 by scheduled scout run

### D8 — COFI-PL sequential covariance-fitting phase linking (third estimator)
- **Source**: researcher (El Hajjar et al., arXiv:2502.09248 / arXiv:2403.08646)
- **Issue**: #106
- **Re-entry gate**: dolphin adopts an equivalent estimator (parity target + oracle), OR
  an independent benchmark on a real dolphinRust-validated scene (e.g. MMX1/ICMX) shows
  EVD/EMI's bias/variance is measurably worse than the paper's reported COFI-PL numbers.
- **Design sketch**: new estimator variant in `dolphin-phaselink/src/estimator.rs`
  alongside `FixedEstimatorBranch::{Evd,Emi}`; analytic fixture first, then the paper's
  own real-data statistics as a secondary check (no dolphin oracle exists for this).
- **Added**: 2026-09-01 by scheduled scout run

### D9 — Incremental PS amplitude-dispersion updates + small-stack quality metrics
- **Source**: researcher (Staniewicz et al., arXiv:2511.12051 — dolphin's own lead-author team)
- **Issue**: #107
- **Re-entry gate**: dolphin ships either refinement in a release, giving a concrete
  algorithm and oracle to validate against.
- **Design sketch**: extend `MiniStackPlanner`/`update_sequential`'s carry-forward pattern
  to amplitude-dispersion running statistics rather than the full amplitude history.
- **Added**: 2026-09-01 by scheduled scout run

### D10 — Adaptive (data-driven) ministack partitioning
- **Source**: researcher (Wang et al. 2025, IJAEOG v139, S1569843225001992)
- **Issue**: #108
- **Re-entry gate**: a real production AOI with heterogeneous coherence regimes shows
  fixed `ministack_size` measurably underperforming an adaptive partition **and**
  PLAYBOOK §Elevated questions #2 (ministack/tile partitioning vs. the bit-identical
  tiled/whole contract) has been answered.
- **Design sketch**: alternative `MiniStackPlanner` path using Otsu's-algorithm
  coherence-matrix partitioning, config-gated, validated against the fixed-size planner's
  existing contract tests before enabling adaptivity.
- **Added**: 2026-09-01 by scheduled scout run

### D11 — Complex-elliptical-model SHP selection
- **Source**: researcher (Yao, Frery, Balz, arXiv:2509.12700)
- **Issue**: #109
- **Re-entry gate**: a production AOI with mixed urban/vegetated terrain shows GLRT/KS
  SHP selection producing PS/DS density anomalies or clustering issues this method would
  plausibly fix.
- **Design sketch**: third `shp_method` in `dolphin-shp` driven by the complex angular
  central Gaussian distribution, plus a non-Rayleigh phase estimator variant.
- **Added**: 2026-09-01 by scheduled scout run

### D12 — Sequential joint-scatterer (PS+DS unified) estimation
- **Source**: researcher (S-JSInSAR, Remote Sensing 18(2):329, 2026, doi:10.3390/rs18020329)
- **Issue**: #110 — high bar, scope-only, do not commit without the gate
- **Re-entry gate**: dolphin/MintPy/ISCE3/the OPERA DISP-S1 SAS adopts a joint PS+DS
  estimator (parity target), OR dolphinRust's separate PS/DS pipeline is shown on real
  data to have decorrelation bias or redundant compute a joint estimator would fix.
- **Design sketch**: none yet — needs a design review given the Phase 1↔Phase 3
  architectural boundary it crosses; do not scaffold ahead of the gate.
- **Added**: 2026-09-01 by scheduled scout run

### D13 — NSBAS-style temporal-smoothness gap-bridging inversion
- **Source**: competitive (LiCSBAS `LiCSBAS13_sb_inv.py`, NSBAS / Lopez-Quiroz et al. 2009)
- **Issue**: #111
- **Re-entry gate**: a real production interferogram network is shown (GNSS harness or
  GroundPulse run) to have a rank-deficient or gap-degraded inversion that denser network
  construction (`max_bandwidth`) cannot practically resolve.
- **Design sketch**: second-derivative temporal-smoothness regularization term added to
  the existing L2 design-matrix assembly in `dolphin-timeseries`, config-gated, off by
  default (dolphin has no NSBAS equivalent).
- **Added**: 2026-09-01 by scheduled scout run

## SHIPPED

### D6 — Expose orbit ephemeris class (POE/RESORB) for processing provenance
- **Source**: inbound (cross-repo signal, `../eo`#483 — Table 5 processing-provenance audit)
- **Issue**: #57
- **Gate result**: confirmed — `dolphin-io::cslc_metadata` already reads
  `/metadata/orbit/orbit_type` directly (added alongside the module's existing
  `/identification` and orbit-state-vector readers); no separate
  `processing_information/inputs/orbit_files` group was needed. Classification lives in
  `dolphin-workflows::provenance` (`read_cslc_orbit_type` → `POEORB`→`precise`,
  `RESORB`→`restituted`, case-insensitive, cross-granule consistency checked, unknown/mixed
  values kept explicit and non-fatal).
- **Design sketch**: as originally sketched — small `dolphin-io` reader plus a
  `dolphin-workflows::provenance` classification step, keeping the module's IO-only /
  interpretation-elsewhere split. Geometry-provenance artifact schema bumped to
  `dolphinrust-geometry-provenance/4` (prior `/2`/`/3` remain deserializable).
- **Added**: 2026-08-24 by scheduled scout run
- **Shipped**: 2026-08-24 by manual contract-first implementation (PR #63, `853cc5e`)
- **Reconciled**: 2026-08-31 scheduled scout run (ledger still listed this as DEFERRED
  after it had already shipped and closed)

### D3 — Automated loop-closure QC gate for unwrap-network errors
- **Source**: competitive (LiCSBAS)
- **Issue**: #24
- **Gate result**: design review confirmed the gap — wrapped-domain closure phase is
  mathematically blind to whole-cycle (2π) unwrap errors (`.arg()` discards them by
  construction), and the existing conncomp labels give correction granularity + a free
  prefilter but no cross-interferogram/network-loop signal, so new machinery was needed.
- **Design sketch**: `dolphin-timeseries::loop_closure` closes every network triangle on
  the unwrapped stack; residual `φ_ij + φ_jk − φ_ik` ≈ 0 for a good unwrap and ±2πn for an
  error. Masks failing pixels across every interferogram before the SBAS solve, gated by
  `timeseries_options.mask_unwrap_loop_errors` (off by default). Emits
  `loop_closure_bad_count.tif` / `loop_closure_worst_cycles.tif`. Noted scope finding: a
  single-reference network has no loops (no-op, warns not errors) — ties to issue #36's
  over-determined-network conclusion and dolphin v0.42's nearest-3 default (issue #25).
- **Added**: 2026-08-01 by scheduled scout run
- **Shipped**: 2026-08-09 by manual contract-first implementation (`dfdfeb1`)

### D1 — Degenerate all-non-finite input window silently yields temporal_coherence=1.0 / displacement=0.0
- **Source**: inbound (cross-repo signal, `../eo` `dolphin-safety-report.md` Finding #2)
- **Issue**: #8
- **Gate result**: pinned dolphin v0.35.0 raises `PhaseLinkRuntimeError` when any SLC date is
  all NaN (`oracle/check_all_nan_v035.py`), so this is a direct parity fix rather than a
  forward divergence.
- **Design sketch**: `dolphin-phaselink/src/covariance.rs::finite_or_zero` +
  `coherence_entry` (AMP_FLOOR underflow → 0+0j matrix) combined with
  `quality.rs::temp_coh_single`/`pair_diff` (phase-only, `arg(0+0j)==0.0`) reproduces the
  observed 1.0/0.0 exactly. The phase-link entry now rejects an all-non-finite acquisition
  before covariance estimation while preserving partially valid masking.
- **Added**: 2026-07-20 by scheduled scout run
- **Shipped**: 2026-07-21 by manual contract-first implementation

## OUT OF SCOPE

### D4 — Possible output-grid geometry ambiguity under asymmetric strides
- **Source**: inbound (cross-repo signal, `../eo`#277 — P1 production incident, unconfirmed hypothesis)
- **Issue**: #26 (closed 2026-08-09 — gate resolved away from dolphinRust)
- **Disposition**: `../eo`#277's own resolution (closed 2026-08-03) states the root cause
  as "fixed by rebuilding the AOI water mask from the authoritative CSLC output grid" — the
  eo-side branch of the gate, not dolphinRust's output-grid geometry math. The re-entry
  condition ("...and it points at dolphinRust's own output-grid/analysis-domain geometry as
  the actual locus") was not met, so no dolphinRust bug was built. Residual test coverage
  was added anyway: `crates/dolphin-core/tests/blocks_contract.rs` gained the exact
  `{y:1, x:2}` override from eo#277 and its transpose `{y:6, x:3}` (`dfdfeb1`) — both pass
  all three block-manager invariants, independently confirming the bug was never here.
- **Re-entry gate**: a future incident's reproduction points at dolphinRust's own
  output-grid/analysis-domain geometry math (not `../eo`'s water-mask/geotransform
  derivation) as the actual locus.
- **Added**: 2026-08-03 by scheduled scout run
- **Closed locally**: 2026-08-09 after eo-side root-cause resolution
- **Reconciled**: 2026-08-31 scheduled scout run (ledger was stale — resolution on
  2026-08-09 was never reflected here)

### D5 — ERA5-based tropospheric delay as a dolphinRust correction source
- **Source**: inbound (cross-repo signal, `../eo`#238, itself sourced from an ecosystem scan)
- **Issue**: #41 (close as out of scope in this repository)
- **Intake IDs**: EO-238-ARCH, DR-041
- **Disposition**: `../eo`#238 was closed by `../eo`#379 as superseded by open
  `../eo`#188. The successor constrains any future correction to eo's wrapper chain or an
  upstream pointer bump, never a local dolphinRust implementation. Do not scope an
  `era5_troposphere` module here.
- **Re-entry gate**: eo explicitly reverses that boundary and selects a per-pixel dolphinRust
  layer, and both source papers (PMC11819746 and ScienceDirect S0273117726001419) are read in
  full and show that the power-law+ERA5 estimate is reproducible from ERA5 alone without the
  proprietary Beijing GNSS-ZTD-gradient enhancement.
- **Added**: 2026-08-10 by scheduled scout run
- **Closed locally**: 2026-08-17 after eo ownership resolution
