# Session handoff — 2026-08-29

> **Resolved 2026-09-03.** PR #97 merged as `bd0a825` on 2026-08-30 with green main CI; #94, #95, and #96 are closed. v1.6.0 was tagged and released 2026-08-31. The "awaiting merge" state described below is historical.

## Summary

The only three issues open at session start, #94, #95, and #96, now have contract-first
fixes committed and pushed on `codex/move-temporal-field-gates-to-eo` at
`128c0c48ca182b9ef942032c8aad0dbde8b8cdd3`. PR #97 is open, non-draft, `CLEAN`,
`MERGEABLE`, and green on that exact head. It contains `Closes #94`, `Closes #95`, and
`Closes #96`.

GitHub still reports all three issues open because PR #97 has not merged. The approved workflow
stops before merge, release, publication, GroundPulse pinning, or deployment. Ryan's next action is
the manual merge decision, followed by a live zero-open-issue and main-CI check.

The branch is 14 commits ahead and 0 behind `origin/main` at `e966ef0`. It was clean before this
EOD write. The PR diff spans 55 files with 23,881 insertions and 3,880 deletions and has no human
review or comments. This handoff is local and uncommitted.

## Completed

- Reconciled the live issue queue and captured every requirement for #94, #95, and #96 in
  `md/intake/open-issues-code-fixes-2026-08-29.md`, with matching design and implementation plan.
  The coverage audit has no deferred or dropped intake item.
- Fixed #94 while retaining the prepared estimator/JVP/cache and bounded parallel replay:
  restored per-attempt frozen DGP streams, the fixed-L2 `1e8` phase/date condition policy, the
  `ill_conditioned` and `nondifferentiable_node` statuses, and raw input identity
  `5411bda5c0d4ebde3d52afc504e07872e6d5104d5b15d3f3d4474358036163ff`.
- Fixed #95 by adding optional `ComparatorDiagnostics.source_status`, preserving the exact inner
  adjusted-variance failure under the public `WeakParameterIdentification` status, and omitting
  the field from successful comparator JSON.
- Fixed #96 with `coverage_bias_interval_score/7` and the
  `oracle_calibration`, `candidate_evaluation`, and `forensic_v5` modes. The scorer uses exact
  coverage/emission arithmetic, familywise bias calibration, per-comparator intersections,
  selected-only promotion, retained diagnostic tables, disjoint seed domains, and a
  hash-bound calibration receipt. Forensic v5 output cannot certify retroactively.
- Kept the frozen temporal v5 preregistration, generator, and no-go result immutable. Current
  estimator source is intentionally rejected against that frozen producer identity rather than
  silently re-frozen.
- Fixed two Linux CI-only contract instabilities without weakening production gates:
  interval-endpoint tolerance now scales to finite-difference interval width, and the public
  fallback fixture explicitly forces its inner curvature rejection.
- Added nine issue-closeout commits from `a99d6b7` through `128c0c4` and pushed them. Opened
  PR #97, `fix(validation): close covariance validation backlog`.

## In progress

- PR #97 is the only open pull request. It is green and mergeable at `128c0c4` but remains
  unmerged under the manual-merge boundary.
- Issues #94, #95, and #96 remain the only open issues. GitHub recognizes all three as closing
  references on PR #97.
- The issue-close goal is blocked only on Ryan's merge. No implementation, test, review-comment,
  branch-divergence, or CI failure remains.
- No post-merge `main` CI receipt exists. Do not describe the issues as closed or the fixes as
  merged until that exact state is refreshed.
- The v7 scorer is corrected validation infrastructure. It does not certify a new temporal
  estimator or change the frozen v5 no-go.
- The known parallel macOS GDAL/HDF5 native-library signal failure was not fixed. Serialized local
  execution and the full Linux CI suite passed.

## Verification

Passing evidence:

- GitHub Actions run `33272937234`, job `99154586169`, passed on exact head `128c0c4` in
  19m29s: formatting, workspace check, strict Clippy, Rust tests, and Python validation.
- The Linux no-GPU workflow group reported 191 passed and 2 ignored. Full Python validation
  reported 254 tests with 3 skipped.
- Local `cargo test --workspace -- --skip gpu_emi_deterministic_384 --test-threads=1` passed,
  including integration and doc tests.
- Local `cargo test -p dolphin-timeseries -- --test-threads=1` passed after the final fallback
  fixture change: 59 unit tests passed with 2 ignored, plus 32 temporal-covariance contracts,
  21 timeseries contracts, and the other package integration/doc tests.
- The exact single-threaded no-GPU workflow suite passed with 191 passed and 2 ignored.
- Spatial Python validation passed with 101 passed and 3 skipped. The selected temporal Python
  modules passed 63 tests.
- `cargo fmt --all -- --check`, `cargo check --workspace`, and
  `cargo clippy --all-targets -- -D warnings` passed after the final code/test changes.
- Frozen-identity tests passed for the temporal v5 producer, generator, and no-go hashes.
- Independent #94 and #96 reviews reported no unresolved findings.

Identity receipts:

- Spatial source-set SHA-256:
  `1daeeb72c360aaa8108552b87a48cca13ae1e6060e7eb6ce78be8f37d3409d17`.
- Spatial scorer generator SHA-256:
  `1ff5de040b5259887214b50d1865c42c26b9db58b2b7b006de280dc885491f76`.
- Frozen temporal v5 preregistration SHA-256:
  `bf8a0cc92d6f0f4e03bb3c0fea88ea411b897d20373376d021540c55dce77166`.
- Frozen temporal v5 generator SHA-256:
  `6684130b2b8f596bef67de70ed39f00b8cb65cb1023beb169307f660834f7d56`.
- Frozen temporal v5 no-go SHA-256:
  `0c885ac25f6680a18b1739e7c126c5821bc153c808c00e7b51c0b4e001ef483e`.
- Temporal scorer v7 policy SHA-256:
  `48fe684154a399ff8265b89b5e2c6a88f20d00e0794ab139ab58bdbe1828b73a`.

Not established:

- PR merge, issue closure, or post-merge `main` CI.
- A new candidate-evaluation receipt, new temporal-estimator certification, or real-data
  scientific validation.
- A source release newer than v1.5.0, crate publication, GroundPulse submodule bump, deployment,
  production terminal artifact, or downstream acceptance.
- A fix for the parallel macOS GDAL/HDF5 signal failure.
- Full local Python discovery because `asf_search`, `h5py`, and `rasterio` were unavailable
  locally. CI installed its validation dependencies and passed the complete 254-test suite.

## Open questions

1. Will Ryan manually merge PR #97 now that its exact head is green and mergeable?
2. After merge and main-CI verification, should the local handoff files be committed and should the
   topic branch be removed?

## Next actions

1. Manually review and merge PR #97 if approved.
2. Refresh `main`, verify #94, #95, and #96 close, and require green main CI on the exact merge SHA.
3. If the issue list is empty and main CI is green, record that separately from release,
   GroundPulse pin, deployment, and scientific-validation status.
4. Decide whether to commit the local handoff chain and clean the topic branch only after the merge
   receipt is preserved.
5. Do not release, publish crates, bump GroundPulse, deploy, or claim candidate certification
   without separate authorization and evidence.

## References

- Branch: `codex/move-temporal-field-gates-to-eo` at
  `128c0c48ca182b9ef942032c8aad0dbde8b8cdd3`.
- Base: `origin/main` at `e966ef0`; branch divergence is 0 behind, 14 ahead.
- PR #97: https://github.com/morton-analytics-llc/dolphinRust/pull/97
- PR CI: https://github.com/morton-analytics-llc/dolphinRust/actions/runs/33272937234
- Issues: https://github.com/morton-analytics-llc/dolphinRust/issues/94,
  https://github.com/morton-analytics-llc/dolphinRust/issues/95, and
  https://github.com/morton-analytics-llc/dolphinRust/issues/96.
- Intake: `md/intake/open-issues-code-fixes-2026-08-29.md`.
- Design: `md/design/open-issues-code-fixes-2026-08-29.md`.
- Plan: `md/plans/open-issues-code-fixes-2026-08-29.md`.
- #95 implementation: `crates/dolphin-timeseries/src/temporal_covariance_batch.rs`.
- #96 scorer: `validation/score_temporal_covariance_synthetic_v7.py`.
- Previous tracked handoff: `.codex/handoffs/2026-08-23.md`.
- Latest GitHub release remains v1.5.0; no release action was taken.
