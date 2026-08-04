# Session handoff — 2026-07-25

## Summary

dolphinRust ends the session synchronized with `origin/main` at `bb79b4e`, with no open issues or
pull requests. The combined reliability and uncertainty work, CI repair, and tiled non-finite
coverage fix all merged through PRs #17–#19. GroundPulse then completed the fresh bounded
production acceptance gate and closed dolphinRust #10/#11 with correlated terminal-artifact
receipts. No code work remains in progress; only local handoff files are uncommitted.

## Completed

- Merged PR #17 (`6fb8014`, `feat: add reliability and uncertainty outputs`):
  - real temporal coherence and masks now feed every unwrap backend;
  - native/SNAPHU connected-component labels are retained and written pair-aligned;
  - CRLB-derived L2 SBAS and velocity weighting is enabled by default with a legacy opt-out;
  - bounded posterior variance, residual RMS, on-demand covariance, and velocity sigma outputs
    were added without retaining a full covariance cube;
  - the MMX1/ICMX harness now produces weighted/unweighted reliability JSON, CSV, and SVG.
- Merged PR #18 (`d0b71f9`, `ci: let pkg-config discover HDF5`): Ubuntu CI now uses distro HDF5
  discovery and a Mesa software Vulkan adapter. The 384-date hardware stress contract remains
  excluded only on that 128 MiB software-adapter binding limit; all other GPU contracts run.
- Merged PR #19 (`bb79b4e`, `fix: preserve nodata in tiled phase linking`): tiled phase linking
  preserves nodata instead of turning incomplete temporal tiles into an all-non-finite acquisition
  failure.
- GroundPulse production acceptance completed against the current dolphinRust fix and consumer
  corrections through GroundPulse `1f97f4c8`:
  - bounded T137 processing completed phase linking, nodata-aware two-burst stitching, unwrap,
    inversion, corrections, velocity, write, distinct phase-linking-coherence COG, COG, and
    PMTiles generation;
  - peak RSS was 2,483,532 KiB (about 2.37 GiB), below the 5.5 GiB acceptance ceiling;
  - provenance `/3` recorded 48 total tiles, 32 linked, 16 nodata, and 235,400 valid of 235,404
    output pixels;
  - no exit 139, SIGSEGV, OOM, lease loss, or abnormal container exit occurred;
  - the canonical correlated terminal-artifact verifier passed, and a repeat audited dispatch
    returned `enqueued: false` without duplicate computation.
- Closed dolphinRust issues #10–#16. GitHub currently reports zero open issues and zero open PRs.

## In progress

- No dolphinRust implementation is in progress.
- A later GroundPulse worker was still running after the accepted receipt was established. It is
  outside this repository/session and does not reopen or weaken the completed #10/#11 receipt.
- GroundPulse public map publication remains intentionally suppressed where the required water
  mask is absent. Acceptance did not bypass that serving policy.
- The real weighted MMX1/ICMX uncertainty coverage result remains honestly `not_evaluable`:
  ICMX lies in a singular-CRLB region with only 5/25 finite pixels in its primary 5x5 window.
  This is a real-data/scientific-evidence limitation, not a known implementation defect.

## Verification

Passing evidence:

- PR #17 local gates: `cargo fmt --all -- --check`, `cargo check --workspace`, strict workspace
  Clippy, `cargo test --workspace`, Python compile/unit tests, oracle/AOI/NRT/native/SNAPHU/GPU/
  multiburst contracts, four real MMX1/ICMX A/B runs, and a production-shaped RSS benchmark.
- PR #18 GitHub Actions run `30138907433`: formatting, workspace check, strict Clippy, Rust tests,
  and Python validation tests passed on Ubuntu/Rust 1.94.
- PR #19 GitHub Actions run `30166301446` passed.
- GroundPulse production receipt: exact run-linked summary, COG, phase-linking-coherence COG,
  PMTiles, provenance, bounded RSS, normal exit, and canonical verifier success.
- Post-session reconciliation: `main...origin/main = 0 0` at `bb79b4e`; no open issues/PRs.

Not established:

- Nominal uncertainty interval coverage against a valid weighted ICMX station sample remains
  `not_evaluable`; do not convert the current receipt into an empirical calibration claim.
- Public overlay availability is not proven for artifacts suppressed by the missing-water-mask
  policy.

## Open questions

1. Should the `not_evaluable` ICMX coverage boundary become a new research/data issue, or remain
   documented until a suitable station window or additional real stack is available?
2. Should the three completed feature/fix worktrees and their merged branches be removed in a
   later cleanup pass?
3. Should the local `.codex` handoff chain be committed so fresh clones can consume it, or remain
   local as before?

## Next actions

1. Start the next dolphinRust session with `$briefing`; there is currently no implementation
   backlog to select.
2. If empirical uncertainty calibration is prioritized, acquire or select a weighted-finite
   ICMX comparison window and regenerate the 68%, 90%, and 95% reliability artifacts without
   weakening the fixed 50% finite-pixel threshold.
3. Optionally remove the merged `dolphinRust-quality`, `dolphinRust-ci-hdf5`, and temporary
   nodata-fix worktrees after confirming no local-only artifacts are needed.
4. Keep GroundPulse water-mask/public-serving work and any later worker monitoring in the
   GroundPulse session rather than reopening completed dolphinRust acceptance.

## References

- Branch/upstream: `main` synchronized with `origin/main` at
  `bb79b4ee60ef6c33fe09bacc076cbdc8a164b62d`.
- PR #17: `6fb8014` — reliability and uncertainty outputs.
- PR #18: `d0b71f9` — HDF5/Vulkan CI configuration.
- PR #19: `bb79b4e` — tiled nodata/non-finite coverage fix.
- GroundPulse consumer acceptance head: `1f97f4c8`.
- CI runs: `30138907433` and `30166301446`.
- Production acceptance receipts: dolphinRust issues #10 and #11 final comments.
- Previous handoff: `.codex/handoffs/2026-07-21.md`.
