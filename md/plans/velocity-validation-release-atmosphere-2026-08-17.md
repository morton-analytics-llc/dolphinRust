# Implementation plan: velocity scorer, v1.5.0, and atmospheric follow-up

**Status:** approved for execution 2026-08-17.
**Intake:** `md/intake/velocity-validation-release-atmosphere-2026-08-17.md`.
**Order:** eo #419/#440 remains a separate operational incident; within this plan execute
T01 -> T02 -> T03 -> T04 -> T05 -> T06 -> T07.

## Objective

Restore a truthful local validation gate, make the GNSS scorer read the pipeline's velocity
model, release that contract as v1.5.0, pin GroundPulse to the tagged commit, bound the 2018
troposphere data decision without fetching the cohort, and close the stale ERA5 direction in
the repository that will not own it.

## Current state

- With the real repo `.env` present,
  `oracle/.venv/bin/python -m unittest discover -s validation/tests` runs 32 tests and fails
  only `test_token_required_without_leaking_value`: the injected empty environment falls
  through to the real file.
- `validation/gps_ground_truth.py` derives `insar_velocity_mm_yr` from a common-GNSS-epoch
  displacement polyfit. It does not use `velocity.tif` for the reported comparison; the only
  current read is the weighted/unweighted raster-delta diagnostic.
- Saved 2018 linear and seasonal work directories already contain the artifacts needed for
  rescoring. No CSLC or L4 fetch and no pipeline rerun is required for #44.
- `main` has 107 commits after annotated tag v1.4.0. `CHANGELOG.md` has 323 current
  Unreleased body lines, not 605. #42 records the breaking layer rename; #36 records changed
  uncertainty semantics; #46's output policy is not yet in the changelog.
- Workspace and internal crate versions remain `1.0.0`, which also feeds GroundPulse's
  reported Dolphin runtime identity. A tag alone would therefore be an incomplete release.
- eo `origin/main` pins dolphinRust commit `fd107b7`. EO-359 is already closed. EO-316 was
  closed as unnecessary as a separate tracker; EO-419 retains its fresh production-replay
  criterion, with EO-440 as the current restart dependency.
- The 52 matching OPERA L4 HRES objects total exactly 111,638,814,943 bytes in current CMR
  metadata. CMR advertises no server-side spatial subset. The existing Rust reader is
  horizontally windowed after local open; the 2018 frame maps to 6x5 source cells per height
  plane, but remote HDF5 chunk amplification is not measured.
- eo #238 is closed as superseded by eo #188. The successor says any future correction is in
  eo's wrapper chain or arrives through an upstream pointer bump, never as a local
  dolphinRust implementation.

## Technical requirements

### R1 — deterministic credential lookup (DR-047)

- Add a keyword-only env-file path to `resolve_token`, `require_token`, and
  `authenticated_session`, defaulted to `ROOT.parent / ".env"`.
- Preserve lookup order: caller-supplied environment mapping first, supplied env file second,
  netrc fallback only in `authenticated_session`.
- Tests must supply their own nonexistent or temporary env file and never depend on the
  developer's real `.env`.

### R2 — independent velocity estimators (DR-044-CONTRACT, DR-044-PAYLOAD)

- Keep `insar_velocity_mm_yr`, `gnss_velocity_mm_yr`, and their existing common-epoch
  `np.polyfit` definitions unchanged.
- Require finite direct station-pixel samples from each engine's `velocity.tif`; report their
  primary-minus-control difference as `insar_velocity_raster_mm_yr` and its GNSS residual as
  `difference_raster_mm_yr`.
- When both optional seasonal rasters exist, report each station's amplitude in millimeters
  and phase in days. Do not subtract or reconcile station amplitudes/phases.
- Validate optional rasters against the velocity raster's CRS, transform, width, and height.
- Keep schema `dolphinrust-gps-ground-truth/1` and every existing key. Add an `estimators` map
  naming the producer for each velocity scalar.
- Extend `validation/score_pairs.py` so cohort summaries retain both new raster fields.

### R3 — saved-run A/B (DR-044-AB)

- Rescore saved linear and seasonal native runs with `--no-run --score`.
- Accept only if the linear raster pair is about -252.337947 mm/yr with residual about
  -11.507462 mm/yr, the seasonal raster pair is about -246.568292 mm/yr with residual about
  -5.737807 mm/yr, the unchanged displacement polyfit is about -262.005007 mm/yr in both,
  and seasonal station amplitudes are present near 25 mm.
- Treat the raster/polyfit disagreement as two estimator results, not a value to reconcile.

### R4 — real v1.5.0 release (DR-REL-150)

- Move current Unreleased content under `[v1.5.0] — 2026-08-17`, add #46's GroundPulse
  output-policy change, preserve #42's breaking-change warning, and start a new empty
  Unreleased section.
- Set the workspace package version and every internal exact version requirement to `1.5.0`;
  regenerate `Cargo.lock` and verify every workspace package reports 1.5.0.
- Update `RELEASING.md` for the current eo submodule consumer, annotated tags, Python suite,
  workspace checks, exact-tag verification, and GitHub Release creation.
- Merge the release commit to `main`, require green Actions on that exact SHA, then create and
  push annotated tag `v1.5.0` and create a GitHub Release with sourced customer/internal notes.
- Do not publish crates to crates.io.

### R5 — tagged GroundPulse pin (EO-PIN-150, EO-316)

- Work from a fresh eo branch based on `origin/main`; do not reuse the user's stale feature
  branch.
- Update the submodule gitlink to the tag target and require
  `git -C vendor/dolphinRust describe --exact-match --tags HEAD` to print `v1.5.0`.
- Regenerate `crates/gp-dolphin/Cargo.lock`, run the worker identity/build checks, merge through
  green combined-tree CI, and monitor the post-merge workflow.
- Do not reopen eo #316 or claim the tag is production-verified. EO-419 owns the separate
  terminal replay and regression-residual provenance checks.

### R6 — one-number troposphere fetch decision (DR-TROPO-2018)

- Write the strategy only; do not download an L4 cohort or probe during this task.
- Record that server-side subsetting is unavailable and that on-ingest S3/HTTP range access is
  the only plausible reduction before local landing.
- Define `projected_total_transfer_bytes` as the sole go/no-go scalar. A later probe must meter
  actual bytes for both delay variables, required DEM-bracketing height levels, and the 6x5
  horizontal window from one named granule, then conservatively scale to 52 epochs and maximum
  object size.
- If range access fails or transfers most of the probe, set the scalar to the exact bulk
  fallback `111638814943` and reject the cohort fetch pending another source/format.
- Require the subset to preserve CF coordinates/metadata and pass the existing bounded-read,
  height interpolation, and warp contracts before the scalar can authorize a full fetch.

### R7 — ERA5 ownership disposition (EO-238-ARCH, DR-041)

- Update the stale D5 ledger entry to the live eo #188 boundary.
- Close dolphinRust #41 as out of scope here, linking eo #238/#379/#188. Do not add an ERA5
  module or download the paywalled paper.
- Reopen only if eo explicitly selects a per-pixel dolphinRust layer and both source papers
  satisfy the existing reproducibility gate.

## Constraints and guardrails

- Write and observe the #44 contract red before production code.
- Do not replace, rename, or tune the existing polyfit; #44 is additive.
- Direct station pixels are the raster estimator contract. Do not substitute the 5x5 sample
  window, which yields different numbers.
- No L4 cohort/probe download, crates.io publish, production enablement, deploy, or #419/#440
  work is in scope.
- CI, tag existence, an eo gitlink bump, deployment, and fresh terminal-artifact proof are
  separate claims.
- Preserve unrelated local work. Use explicit file staging and fresh main-derived branches.
- Commits include `Co-Authored-By: Claude <noreply@anthropic.com>`.

## Test contract

| ID | Contract | Location | Proof |
|---|---|---|---|
| C01 | Injected token wins and an isolated empty mapping fails even with a real repo `.env` | `validation/tests/test_gps_acquisition.py` | DR-047 cannot depend on host secrets; lookup order remains token first. |
| C02 | Linear and seasonal synthetic workdirs share polyfit inputs but yield different raster velocity estimates | `validation/tests/test_gps_ground_truth.py` | The scorer actually consumes `velocity.tif`. |
| C03 | Optional synthetic seasonal rasters report per-station mm amplitude/days phase and grid mismatch fails | same | Units, optionality, station identity, and alignment are explicit. |
| C04 | Pair summary preserves the raster estimate/residual | focused `score_pairs` test or extended ground-truth contract | Downstream cohort tables cannot remain blind to #44. |
| C05 | Saved linear/seasonal rescore matches the numeric acceptance receipt | local ignored run artifacts plus issue comment | The real 2018 conclusion is recomputed with the corrected scorer. |
| C06 | Release package metadata is uniformly 1.5.0 | `cargo metadata --no-deps` plus lockfile diff | The tag and reported runtime identity agree. |
| C07 | eo gitlink resolves exactly to v1.5.0 and worker build identities agree | submodule describe, lockfile, identity script | GroundPulse consumes the tagged release commit. |
| C08 | Troposphere strategy contains one named scalar, formula, fallback integer, and no fetch receipt | document review | Full L4 spending stays blocked until measured transfer is one number. |

## Task manifest

### T01 — fix #47 first

**Files:** `validation/fetch_real.py`, `validation/tests/test_gps_acquisition.py`.
Implement R1, run the exact 32-test suite with the real `.env` still present, and commit this
slice before touching #44.

### T02 — turn #44 red, then green

**Files:** `validation/tests/test_gps_ground_truth.py`, `validation/gps_ground_truth.py`,
`validation/score_pairs.py`, and only a focused existing test file if C04 cannot live in the
ground-truth contract.
Add C02/C03 first and capture the red failure. Implement the smallest raster-sampling helper,
wire it into each engine payload, preserve the polyfit, then turn focused/full Python tests
green.

### T03 — rescore the saved A/B and close #44/#47 through the PR

Run both saved 2018 score-only commands, inspect `gps_ground_truth.json`, and record the exact
polyfit, raster rates/residuals, amplitudes, phases, commands, and commit in the PR/issue
evidence. Do not rerun the pipeline or fetch data.

### T04 — prepare, merge, and publish v1.5.0

**Files:** `CHANGELOG.md`, `Cargo.toml`, `Cargo.lock`, `RELEASING.md`, plus this intake/plan.
Prepare the release metadata after T03, run all validation, open a ready PR, merge only after
required checks pass, monitor main Actions, tag the exact green merge SHA, and create the
GitHub Release.

### T05 — pin eo to the tag

**Files in eo:** `vendor/dolphinRust`, `crates/gp-dolphin/Cargo.lock` only unless an existing
identity check requires a narrow generated receipt. Start from fresh `origin/main`, validate,
open a ready PR, merge after green checks, and monitor post-merge Actions. Record that #316 is
already closed and that #419 retains the missing production-replay checks.

### T06 — land the no-fetch troposphere strategy

**Files:** new `md/research/gps-mmx1-2018-troposphere-fetch-strategy.md`.
Document the exact bulk size, no-server-subset finding, existing 6x5 logical window, probe
formula, validation contract, fallback, and the single integer gate. Open/merge as a docs-only
post-release PR and monitor Actions.

### T07 — retire the local ERA5 direction

**Files:** `md/intake/idea-scout-ledger.md`.
Replace the stale open-#238 gate with the eo #188 boundary, include the explicit reopen gate,
merge with T06, then post the evidence-backed closure comment and close dolphinRust #41.

## Validation

Run in this order:

```text
oracle/.venv/bin/python -m unittest discover -s validation/tests
oracle/.venv/bin/python -m compileall -q validation
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --no-deps --workspace
git diff --check
```

Release-specific checks:

```text
cargo metadata --no-deps --format-version 1
git tag -v v1.5.0
git ls-remote --tags origin refs/tags/v1.5.0 refs/tags/v1.5.0^{}
```

eo pin checks:

```text
git -C vendor/dolphinRust describe --exact-match --tags HEAD
cargo check --manifest-path crates/gp-dolphin/Cargo.toml
cargo test --manifest-path crates/gp-dolphin/Cargo.toml
scripts/check-dolphin-build-identities.sh
```

## Resolved decisions

- v1.5.0 is a minor release because the range contains features and breaking output semantics.
- The scorer exposes both disagreeing estimators and does not reconcile them.
- EO-359 receives no action because it is already closed.
- EO-316 receives no action because it is already closed as an unnecessary separate tracker;
  EO-419 retains the production-replay criterion, and no pin/deploy state implies that replay.
- #41 closes now because the live successor architecture excludes local dolphinRust code.
- EO-419/#440 is the portfolio predecessor for production verification, but not an item in this
  plan.

## Coding-agent execution contract

Execute T01-T07 in order. Stop only for a concrete technical impossibility; do not ask the user
to re-decide the choices fixed above. Preserve every claim boundary, write #44 red first, do no
L4 fetch, use fresh main-derived branches, merge only green PRs, and monitor post-merge GitHub
Actions before reporting completion.
