---
name: backlog-pipeline
description: Carry one backlog item — a GitHub issue labeled backlog-ready, or a gate-cleared md/intake/idea-scout-ledger.md entry — through dolphinRust's contract-first flow (design → red contract test → make it green → verify → pr), stopping before merge. Loop-compatible — designed to be the payload a weekly /loop or scheduled run invokes.
argument-hint: "[item-id | issue-number] | next"
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Backlog Pipeline

Carries exactly one backlog item from its source (GitHub issue or ledger entry) to an open PR.
Never merges the PR, tags a release, or bumps the `../eo` submodule pin — merging is the trigger
for the human-gated eo submodule bump / deploy (dolphinRust has no auto-deploy GHA; the host app
vendors it). So **"open PR" is the correct stop point**, not "ship it."

This skill does not create work — `idea-scout` sources and triages candidates upstream. This skill
only executes items already approved.

## Step 0 — Select the item

Two eligible sources — check both, GitHub issues first (pre-cleared by construction):

- **GitHub issues**: `gh issue list --label backlog-ready --state open --json number,title,body`.
  The `backlog-ready` label *is* the approval — a human (or `idea-scout` after a "Build now"
  triage) attached it deliberately, so these need no further gate check. If `$ARGUMENTS` names an
  issue number, use it directly regardless of label (explicit request overrides).
- **Ledger**: `md/intake/idea-scout-ledger.md`, entries under `## DEFERRED` headings, in file
  order. Unlike issues, ledger entries carry a stated re-entry gate that must be evaluated
  (Step 1). Use the ledger only for items that genuinely need gate-tracking (uncertain evidence,
  a prior phase, an upstream product), not to duplicate an already-labeled issue.
- If `$ARGUMENTS` names a ledger item ID (e.g. `D1`, `C2`), use it.
- If `$ARGUMENTS` is `next` or empty: oldest open `backlog-ready` issue first; if none, fall back
  to the ledger and evaluate gates (Step 1) until one clears.

## Step 1 — Gate check (ledger items only; GitHub issues skip to Step 2)

Every deferred ledger item states a re-entry gate — don't treat elapsed time as satisfying it.
Read its "Re-entry gate" line and classify:

- **Verifiable now** (e.g. "covariance bench shows ≥3× on 512²×16", "Phase 6 SBAS merged",
  "oracle regenerated at dolphin vX"): actually run the check — grep, read, `cargo bench`/run the
  named example, whatever the gate names. Record `VERIFIED ✓` or `NOT MET ✗` with evidence, same
  rigor as a contract test. Only a `VERIFIED ✓` item is eligible.
- **External / human-gated** (e.g. "needs founder walkthrough", "waits on NISAR DISP-NI product",
  "owner decision on GPU dependency"): cannot be picked up autonomously. Skip it and say so.

If scanning for `next` and nothing clears, stop and report "no eligible items" rather than force
one through. Do not lower a gate's bar or treat an unlabeled issue as eligible.

## Step 2 — Run the contract-first chain

dolphinRust's workflow is **dynamic and contract-first** (see root `CLAUDE.md` / `PLAYBOOK.md`) —
NOT the staged `/design→/implement→/code-the-plan` pipeline. A kernel is done only when its
contract test (analytic fixture and/or the dolphin oracle) is green. Run the item through:

1. **Design** — `/design` with the item's description (issue title+body, or ledger description +
   any "Design sketch" line) as topic. Output a `md/design/{slug}.md` that names the **contract**:
   the analytic fixture or dolphin-oracle case that will prove correctness, and the tolerance. If
   the ledger references an existing design doc, verify it still exists in `md/design/` before
   reuse. For a trivial item (single kernel, obvious contract) skip the doc and state the contract
   inline in the PR.
2. **Red contract test first** — write the failing test (fixture with the known analytic answer,
   and/or golden data from the pinned dolphin oracle). Confirm it's red before touching
   production code. This is the load-bearing discipline — do not implement first.
3. **Make it green** — implement the kernel to idiomatic-Rust conventions (Result-based errors,
   ≤3 nesting, small functions; the fmt hook + clippy gate apply). For a large multi-file item,
   this step is a good fit for a **dynamic Workflow** ("use a dynamic workflow to …") — escalate
   there rather than hand-threading many files.
4. **Verify green** — run and capture **real exit codes** (never pipe test runs through `tail` —
   that masks cargo's exit):
   ```
   cargo test --workspace > /tmp/bp_test.log 2>&1; echo "TEST_EXIT=$?"
   cargo clippy --all-targets -- -D warnings > /tmp/bp_clippy.log 2>&1; echo "CLIPPY_EXIT=$?"
   ```
   Both must be `0`, including the new contract test. If either fails, stop and report at this
   step — never claim green unverified.
5. **`/pr`** — only if Step 4 is fully green. For an issue-sourced item, include `Closes #{issue}`
   in the PR body so merging auto-closes it. Note the correctness evidence (which contract/oracle,
   tolerance) in the PR body.

Each step gates the next. If any step fails or surfaces a blocking gap it can't resolve, stop and
report at that step — don't push a half-verified item forward.

## Step 3 — Update the source

On a successful `/pr`:
- **GitHub issue**: `gh issue edit {n} --remove-label backlog-ready` and comment the PR link.
  Don't close it — `Closes #{issue}` handles that on merge. Removing the label stops re-pickup by
  the next scheduled run while the PR is open.
- **Ledger item**: in `md/intake/idea-scout-ledger.md`, move its entry out of `## DEFERRED` into a
  dated `## SHIPPED` section, replacing the re-entry-gate note with the PR number and a one-line
  summary.

## Step 4 — Stop

Report the PR URL and stop. Do not merge, do not bump the eo submodule pin, do not tag a release —
flag merge + eo-bump as the next human decision.

## Output

```
## Backlog Pipeline — {item-id}

### Gate check
{VERIFIED ✓ / NOT MET ✗ / external-gated, with evidence}

### Contract
{the analytic fixture / dolphin-oracle case + tolerance that proves it}

### Pipeline result
- Design: {md/design/ doc path, or "reused existing" / "inline" / "not reached"}
- Contract test: {path, red→green confirmed, or "not reached"}
- Implementation: {branch + files/crates touched, or "not reached"}
- Verify: TEST_EXIT={0/n}, CLIPPY_EXIT={0/n}  {n tests passed, or "not reached"}
- PR: {url, or "not opened — reason"}

### Source updated
{issue: label removed + comment, or ledger: moved to SHIPPED / "not reached"}

### Stopped before
Merge / eo submodule bump / release — maintainer's call.
```
