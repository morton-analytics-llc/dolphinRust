# Session handoff — 2026-07-14

## Summary

Merged the scheduled covariance follow-up PR, executed the first full MMX1/ICMX independent
GNSS gate, fixed the native MCF final-epoch cycle slip that gate exposed, and landed a reviewed
follow-up. The corrected native and SNAPHU runs both pass the provisional GNSS bars. The
implementation head is synchronized with `origin/main` at `9fec550`; the local branch adds only
this EOD handoff commit. There are no open PRs or issues.

## Completed

- Reviewed and merged PR #6 as `3ba489e`; the measured box-sum win is documented and further
  cross-row accumulation remains deferred because it conflicts with tiled/whole bit identity.
- Downloaded and validated all 13 declared T005 CSLCs plus the matching STATIC product, then
  built the 384x384 MMX1 core and 352x2217 MMX1/ICMX common-frame fixtures.
- Ran native and SNAPHU on both fixtures and produced JSON/CSV/SVG ground-truth artifacts.
- Captured the initial common-frame result honestly: SNAPHU passed; native missed the TLS ceiling
  after a localized final-epoch 27.733 mm cycle slip.
- Added a red live parity contract and isolated two independent defects at the old 7x46 grid:
  randomized equal-weight seam ordering and an overly fine 48-pixel tile floor.
- Committed `18a1652 fix(unwrap): stabilize native seam reconciliation`:
  - deterministic seam vote and spanning-forest ordering;
  - 64-pixel production tile floor (5x34 on the common frame);
  - live MMX1 parity and permanent analytic/tile-floor contracts;
  - corrected validation and performance documentation.
- Landed review follow-up `9fec550`, which made the live test's dominant-offset vote
  deterministic, pinned the equal-weight vote tie-break analytically, removed unreachable sort
  clauses, and documented the isolated A/B attribution precisely.
- Closed issues #4 and #5. No GroundPulse submodule bump, release, publication, or deployment was
  performed.

## In progress

- No code is in progress and the working tree is clean.
- The 64-pixel floor's broad 1024x1024 throughput/concurrency benchmark has not been rerun. The
  older 48-pixel 10x-throughput claim is retained only as historical evidence and must not be
  reused as current.
- Atmospheric corrections remain off for the MMX1/ICMX comparison. Residuals include atmosphere
  and cannot be attributed solely to either unwrapper.
- The live CSLC, STATIC, GNSS, raster, and scoring artifacts are local and gitignored. The live
  Rust contract skips on fresh checkouts without that external fixture; permanent analytic tests
  still exercise determinism and the production tile floor.

## Verification

Corrected full live score (run after `18a1652` implementation, before review-only `9fec550`):

- Overall status: PASS; both native and SNAPHU pass provisional-1 bars.
- GNSS endpoint truth: -104.262 mm.
- Native estimate: -94.305 mm; residual +9.958 mm; correlation 0.9375; TLS 1.0357.
- SNAPHU has the same rounded metrics; native-minus-SNAPHU endpoint is 0.000033 mm.
- Final-epoch native/SNAPHU per-component disagreement: 0.1918% (gate <=0.5%).
- Runtime on the shared frame: native 61.3 s; SNAPHU 100.7 s.

Current `9fec550` HEAD verification completed and passing:

- `cargo check --workspace`.
- `cargo clippy --all-targets --workspace -- -D warnings`.
- `cargo test --workspace`, including GPU contracts, the live MMX1 contract, native tiling,
  oracle parity, end-to-end displacement, NRT, sign, geometry, and doc tests.
- `git diff --check` before the implementation commit.

Not run:

- Full GNSS acquisition/crop/score again after `9fec550`; that follow-up changed tests,
  deterministic test voting, documentation, and unreachable sort tie-breaks, not the selected
  5x34 production result. Its live contract and full workspace suite pass at current HEAD.
- Atmospheric-corrected MMX1/ICMX comparison.
- GroundPulse integration/submodule bump or live deployment.

## Open questions

1. Should the provisional GNSS thresholds become hard release gates, or remain report-only until
   more station pairs and atmospheric corrections are included?
2. Should the 64-pixel native floor be rebenchmarked now on the 1024x1024 concurrency harness, or
   only when a performance decision depends on updated numbers?
3. When should GroundPulse consume `9fec550` through its human-gated dolphinRust submodule bump?

## Next actions

1. If current performance claims matter, rerun the residue-dense 1024x1024 latency/concurrency
   benchmark at the 64-pixel floor and update PLAYBOOK/config commentary from measured evidence.
2. Decide whether to harden the GNSS bars and whether to acquire/apply 13-date atmospheric inputs
   before making unwrapper-specific residual claims.
3. If GroundPulse should consume the fix, bump its dolphinRust submodule in `../eo`, run the GP
   worker contracts, and keep deployment/release separately human-gated.
4. Run `$briefing` next session; the repo itself has no open PR or issue queue at this handoff.

## References

- Branch/upstream: implementation synchronized at `9fec550`; local `main` adds the EOD handoff
  commit and is one commit ahead of `origin/main`.
- Commits: `3ba489e` (merge PR #6), `18a1652` (native stabilization), `9fec550` (review follow-up).
- Issues: #4 and #5 closed; no open PRs.
- Validation narrative: `VALIDATION.md` MMX1/ICMX section.
- Solver changes: `crates/dolphin-unwrap/src/native/tile.rs`.
- Live contract: `crates/dolphin-unwrap/tests/native_mmx1_live_contract.rs`.
- Tile policy: `crates/dolphin-workflows/src/displacement.rs::native_tiling`.
- Local result: `validation/runs/gps_mmx1/mmx1_icmx_common/gps_ground_truth.json`.
