# Plan: live dolphinRust issues under a 45-minute execution limit

**Status:** first execution slice only; implementation and merge not started.

**Intake:** `md/intake/open-issues-2026-08-24.md`.

**Master plan:** `md/plans/fixed-cube-eo-gnss-implementation-2026-08-24.md`. This document covers
only the #52 reconciliation/manual-merge gate. FC-01 is the next 45-minute coding slice; all GNSS
selection, projection, anchoring, persistence, serving, UI, and Fresno work belongs to EO.

**Prior plan:** `md/plans/open-issues-uncertainty-2026-08-23.md` remains the detailed scientific
contract for #52 -> #54 -> #53. This plan replaces its stale queue/PR status and selects only work
that can finish inside the stated time limit.

## Objective

Close DR-052-MERGE by updating PR #56 to current `main`, resolving its review-receipt wording,
running current-tree verification, obtaining Ryan's manual merge, and proving post-merge `main`
CI and #52 closure within 45 minutes.

Do not claim that this supplies calibrated uncertainty or unblocks `eo`. No fresh #54 or #53
scientific implementation can be coded, validated, independently reviewed, and merged in this
time box.

## Current state

- `origin/main`: `834f9a9ed8a3479afb4da670759af360db53b184`, green CI run `32726428182`.
- PR #56: open, non-draft, clean/mergeable, head `8eb413b3d111a7d6d983f81226c90251d4268c1f`,
  exact-head CI run `32683884440` green in 8m34s, no formal GitHub review.
- PR #56 is one commit behind `main`; the only base delta is
  `md/intake/idea-scout-ledger.md`.
- Root checkout: dirty and five commits behind. Preserve it. Use
  `/private/tmp/dolphinrust-issue52.N22LF4` or a new isolated worktree.
- `eo` pins `v1.5.0` / `8a63fe2`, twelve commits behind current dolphinRust `main`.

## Technical requirements

### DR-052-MERGE

1. Merge current `origin/main` into the PR branch without rebasing or force-pushing.
2. Clarify `md/design/sequential-global-covariance.md` so the statement that no reviewer approved
   kernel work refers only to the rejected `sequential_srif_v1` representation. It must not
   contradict the recorded approval of `sequential_source_dag_v1` and T52-06.
3. Preserve exact acquisition-0 gauge, bounded replay/preflight, fail-closed statuses, and the
   `uncalibrated` / `blocked_pending_issue_54_and_53` release boundary.
4. Require focused contracts and the complete GitHub `verify` job on the updated head.
5. Ryan performs the merge. Codex stops before merge under the automation policy.
6. Completion requires the merged commit on `main`, #52 closed, and the post-merge `main` CI green.

### Deferred issue requirements

- DR-054-SPATIAL stays in T54-01 through T54-07. A contract-only or metadata-only PR is not a fix.
- DR-053-TEMPORAL stays in T53-01 through T53-07. A larger sigma or scalar effective-N rescale is
  not calibrated inference.
- DR-057-ORBIT is independent provenance work. Its corrected source path is
  `/metadata/orbit/orbit_type`, not a guessed processing-input filename. Schedule it separately.

## Constraints and guardrails

- Touch only the PR #56 branch during T45. Do not use, clean, fast-forward, reset, or stash the
  root checkout or the other user-owned worktrees.
- No #54/#53 implementation, release/tag, crates.io publication, `eo` mutation, submodule pin,
  deployment, historical replay, or terminal-artifact claim.
- Do not treat source-DAG replay as calibrated covariance, inferential velocity weighting, total
  uncertainty, asset risk, or field validation.
- If updated CI is red, fix only a demonstrated #52 integration defect, rerun focused and full
  checks, and do not merge on a time deadline.
- A targeted acceptance audit is acceptable for the 45-minute decision because an independent
  review is already recorded. Do not describe it as a new exhaustive review of the 17,770-line
  diff.

## Test contract

| Contract | Proof |
|---|---|
| C45-01 current-base integration | The PR branch contains current `origin/main`, has no conflict, and CI tests the prospective combined tree rather than the stale head/base combination. |
| C45-02 analytic and fail-closed behavior | Existing phaselink, workflow, and IO covariance contracts remain green after reconciliation. |
| C45-03 release boundary | Method/status assertions still emit exact gauge zero, `uncalibrated`, and `blocked_pending_issue_54_and_53`; velocity and GroundPulse policies remain disconnected. |
| C45-04 merge completion | PR #56 is `MERGED`, issue #52 is `CLOSED`, the merge commit is reachable from `origin/main`, and its `main` CI is green. |

## 45-minute implementation plan

### T45-01 — reconcile and remove the review ambiguity (minutes 0-5)

- Work in the clean PR #56 worktree.
- Fetch `origin`; assert the branch head is the expected PR head.
- Merge `origin/main` into the branch without history rewriting.
- Apply the one-line review-receipt clarification in
  `md/design/sequential-global-covariance.md`.
- Confirm the diff is limited to the expected `main` intake update plus that clarification.

### T45-02 — focused verification and targeted audit (minutes 5-12)

Run:

```text
cargo test -p dolphin-phaselink --test global_covariance_contract --test source_influence_contract
cargo test -p dolphin-workflows --test global_covariance_contract
cargo test -p dolphin-io --test covariance_contract
cargo fmt --all -- --check
git diff --check
```

Audit the exact gauge, cap/preflight, stale-identity, unsupported-scope, and release-status
assertions. Stop on any mismatch; do not broaden the patch.

### T45-03 — push, CI, and manual merge gate (minutes 12-30)

- Commit only the branch reconciliation/clarification with the required co-author trailer.
- Push normally; do not force-push.
- Require the updated-head `verify` job to pass formatting, workspace check, strict Clippy,
  workspace Rust tests, and Python validation.
- Reconfirm the PR is non-draft, clean, mergeable, and still points at the audited head SHA.
- Ryan manually merges PR #56 using a head-SHA guard. If CI is not green by minute 30, stop rather
  than compressing review or validation.

### T45-04 — prove the merged state (minutes 30-45)

- Fetch `origin/main` and verify the merge commit is reachable.
- Verify PR #56 is merged and issue #52 is closed.
- Wait for the push-to-`main` CI run and require green status.
- Record only: #52 producer groundwork merged. Preserve #54, #53, release, `eo` pin, deployment,
  and terminal-artifact gates as open.

## Validation commands

```text
git fetch --prune origin
git status --short --branch
gh pr checks 56 --repo morton-analytics-llc/dolphinRust
gh pr view 56 --repo morton-analytics-llc/dolphinRust \
  --json state,mergeable,mergeStateStatus,headRefOid,statusCheckRollup
gh issue view 52 --repo morton-analytics-llc/dolphinRust --json state,url
gh run list --repo morton-analytics-llc/dolphinRust --branch main --limit 3
```

## Next standalone short coding slot: DR-057-ORBIT

Do not promise this merge in the same 45-minute window as DR-052-MERGE. After Ryan applies
`backlog-ready`, use a new isolated worktree from post-#56 `origin/main`.

### Test contract

1. The committed real-granule crop reads raw `POEORB` from
   `/metadata/orbit/orbit_type` and exports normalized `precise` with exact source provenance.
2. A synthetic `RESORB` fixture exports `restituted`.
3. Missing, unrecognized, or mixed-stack values fail closed for this field with an explicit
   reason; they do not erase existing orbit direction, heading, spacing, or time provenance.
4. Data-only and NISAR inputs remain explicit absence, never a default.
5. JSON round-trip and prior-schema deserialization remain compatible.

### T57 task manifest

1. **T57-01 red contracts:** update `crates/dolphin-io/src/cslc_metadata.rs` tests and
   `crates/dolphin-workflows/tests/geometry_provenance_contract.rs` first.
2. **T57-02 raw IO:** add a separate `read_cslc_orbit_type()` and export it from
   `crates/dolphin-io/src/lib.rs`. Do not make `read_cslc_orbit()` require the field.
3. **T57-03 normalization:** add optional `orbit_ephemeris_class` provenance in
   `crates/dolphin-workflows/src/provenance.rs`; all granules must agree on POEORB or RESORB.
4. **T57-04 schema/docs:** bump geometry provenance schema/method, update compatibility tests,
   `CHANGELOG.md`, and the idea-scout ledger with the satisfied data gate.
5. **T57-05 verify/PR:** run focused tests, workspace check, strict Clippy, full tests, open one PR,
   wait for CI, and stop for Ryan's manual merge.

`eo` currently accepts geometry-provenance schemas `/2` and `/3`; a `/4` producer merge must not
be released/pinned until EO-483-PROVENANCE adds the consumer and customer-facing disclosure.

## Open questions

None for DR-052-MERGE. DR-057-ORBIT needs Ryan's `backlog-ready` label before automation starts.

## Coding-agent prompt

```text
Complete DR-052-MERGE in dolphinRust within 45 minutes.

Use /private/tmp/dolphinrust-issue52.N22LF4, not the dirty root checkout. Refresh origin, verify
the expected PR #56 head, merge current origin/main without rebasing, and clarify the final line
of md/design/sequential-global-covariance.md so it refers to the rejected sequential_srif_v1
representation rather than contradicting the approved sequential_source_dag_v1 review receipt.

Touch no scientific implementation unless updated-base tests expose a demonstrated #52 defect.
Preserve the exact acquisition-0 gauge, bounded replay/preflight, all fail-closed statuses, and
the uncalibrated/blocked_pending_issue_54_and_53 boundary. Run the focused phaselink, workflow,
and IO covariance contracts, cargo fmt, and git diff --check; push normally and require the full
GitHub verify job on the updated head.

Stop for Ryan's manual merge. After he merges, verify PR #56 merged, issue #52 closed, the merge
commit is on origin/main, and post-merge main CI is green. Do not release, pin eo, deploy, replay,
or claim calibrated uncertainty or EO acceptance.
```

## Coverage audit

DR-052-MERGE is scheduled in T45-01 through T45-04. DR-054-SPATIAL and DR-053-TEMPORAL are
deferred to their existing named task manifests and gates. DR-057-ORBIT is deferred to T57-01
through T57-05 after human promotion. EO-487-SCIENCE and EO-483-PROVENANCE are deferred to named
`eo` work; EO-475-WIRING is out of scope here. No intake item is silently dropped.
