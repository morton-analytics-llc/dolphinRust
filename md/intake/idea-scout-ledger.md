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

### D3 — Automated loop-closure QC gate for unwrap-network errors
- **Source**: competitive (LiCSBAS)
- **Issue**: #24 (enhancement-labeled, NOT yet backlog-ready)
- **Re-entry gate**: a design review confirms this targets unwrap-network errors distinct
  from what the existing closure-phase/phase-bias quality layer already catches, and
  clarifies whether the existing per-ifg `conncomp_NN.tif` labels already provide enough
  signal to build this cheaply.
- **Design sketch**: network-level phase-closure-loop pass over the interferogram set,
  flagging/masking pixels with inconsistent loop sums before the SBAS solve — distinct
  from the per-pixel decorrelation-driven non-closure bias `dolphin-phaselink` already
  measures.
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

## SHIPPED

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
