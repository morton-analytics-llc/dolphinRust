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
- **Issue**: #23 (enhancement-labeled, NOT yet backlog-ready)
- **Re-entry gate**: a real scene is found where IONEX/GNSS TEC coverage is missing or
  demonstrably insufficient, OR the real multi-date NISAR validation work (ROADMAP.md's
  "NISAR calibrated-data timing" external gate) surfaces an ionospheric residual large
  enough to justify a second correction path.
- **Design sketch**: analytic fixture with injected sub-band ionospheric phase ramp;
  recover known TEC/delay via split-spectrum estimator; additive fallback/cross-check to
  the existing IONEX path in `dolphin-corrections::ionosphere`, off by default.
- **Added**: 2026-08-01 by scheduled scout run

### D4 — Possible output-grid geometry ambiguity under asymmetric strides
- **Source**: inbound (cross-repo signal, `../eo`#277 — P1 production incident, unconfirmed hypothesis)
- **Issue**: #26 (enhancement-labeled, NOT yet backlog-ready)
- **Re-entry gate**: `../eo`#277's own reproduction step (default `{y:3, x:6}` strides vs.
  the `{x:2, y:1}` override on the Montana harness AOI) runs and points at dolphinRust's
  own output-grid/analysis-domain geometry math — rather than `../eo`'s water-mask/
  geotransform derivation — as the actual locus of the bug.
- **Design sketch**: if gated in, a property/contract test asserting the reported
  output-grid geotransform/dimensions and "bounded analysis domain" coverage check agree
  pixel-for-pixel with the actual strided output raster for asymmetric strides (`y≠x`),
  extending the existing Phase 0 block-manager property tests.
- **Added**: 2026-08-03 by scheduled scout run

### D6 — Expose orbit ephemeris class (POE/RESORB) for processing provenance
- **Source**: inbound (cross-repo signal, `../eo`#483 — Table 5 processing-provenance audit)
- **Issue**: #57 (enhancement-labeled, NOT yet backlog-ready)
- **Re-entry gate**: confirm the exact HDF5 dataset/group path for orbit-file provenance
  (e.g. `/metadata/processing_information/inputs/orbit_files` or similar) against a real
  OPERA CSLC-S1 granule or the authoritative product spec — `dolphin-io::cslc_metadata`'s
  existing keys were granule-verified before shipping, and this field hasn't been.
- **Design sketch**: if confirmed, a small `dolphin-io` reader mirroring
  `read_cslc_burst_metadata`, plus a `dolphin-workflows::provenance` classification step
  (orbit filename pattern `*_AUX_POEORB_*` / `*_AUX_RESORB_*` → `Precise`/`Restituted`/
  `Unknown`), keeping this module's IO-only / interpretation-elsewhere split.
- **Added**: 2026-08-24 by scheduled scout run

## SHIPPED

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
