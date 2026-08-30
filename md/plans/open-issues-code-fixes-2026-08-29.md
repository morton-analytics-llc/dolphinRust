# Implementation plan: open-issue code fixes

**Intake:** `md/intake/open-issues-code-fixes-2026-08-29.md`

**Design:** `md/design/open-issues-code-fixes-2026-08-29.md`

**Base:** `7946b9a` on `codex/move-temporal-field-gates-to-eo`

## Objective

Fix #94, #95, and #96 with verified code while preserving the frozen temporal v5 record and
stopping before merge or release.

## Task manifest

| Task | Intake IDs | Files | Contract |
|---|---|---|---|
| T94-01 | GH-094-01, GH-094-02, GH-094-03 | Existing spatial Rust tests | Reproduce the three failures at `cef94f2` and the passing rollback at `7946b9a`. |
| T94-02 | GH-094-01, GH-094-02 | `crates/dolphin-workflows/src/spatial_covariance_validation.rs`; `crates/dolphin-timeseries/src/spatial_covariance.rs`; spatial validation Python/preregistration files | Restore the frozen DGP and `1e8` two-stage condition policy while retaining the bounded runner and prepared JVP work. |
| T94-03 | GH-094-01, GH-094-02, GH-094-03 | Spatial Rust and Python contracts | Run the exact no-GPU single-threaded workflow test plus focused Python validation. |
| T95-01 | GH-095-01, GH-095-02 | `crates/dolphin-timeseries/src/temporal_covariance_batch.rs` tests | Add a failing contract that forces the exact fallback and asserts the inner status is retained. |
| T95-02 | GH-095-01, GH-095-03 | `crates/dolphin-timeseries/src/temporal_covariance.rs`; `crates/dolphin-timeseries/src/temporal_covariance_batch.rs` | Add optional comparator source status and propagate the exact adjusted-variance error without changing the public fail-closed status. |
| T95-03 | GH-095-02, GH-095-03 | Temporal Rust contracts and frozen receipt | Run focused temporal tests and assert the committed v5 no-go summary hash is unchanged. |
| T96-01 | GH-096-01 through GH-096-05 | `validation/tests/test_temporal_covariance_scorer_v7.py` | Add red contracts for calibration count, exact boundaries, pairwise emissions, selected-only promotion, retained tables, forensic no-certification, and v5 hashes. |
| T96-02 | GH-096-02, GH-096-03, GH-096-04 | `validation/score_temporal_covariance_synthetic_v7.py` | Implement scorer `/7`, the three modes, exact arithmetic, pairwise reducers, calibration binding, and compact receipts. |
| T96-03 | GH-096-01, GH-096-04, GH-096-05 | `VALIDATION.md`; scorer policy fixture if required | Document the future calibration gate and the immutable v5 boundary. |
| T96-04 | GH-096-01 through GH-096-05 | Temporal Python validation | Run the deterministic 5,000-count oracle fixture and the full relevant Python suite. |

## Constraints

- Every analytic change starts with a failing contract.
- Preserve acquisition-0 gauge, frozen attempt identities, source hashes, status strings, and
  resource caps.
- Keep the v5 temporal preregistration, scorer implementation, no-go summary, and Rust product
  eligibility contract unchanged for #96.
- Keep expected spatial hashes and statuses frozen against the `3a63b3a` semantic drift.
- Add no stubs, TODOs, placeholder receipts, or synthetic production data.
- The known parallel GDAL/HDF5 crash remains explicitly unverified.

## Test contract

### #94

- `ill_conditioned_attempt_emits_a_fail_closed_receipt` returns `ill_conditioned`.
- `stochastic_attempts_are_emitted_with_stable_scope` returns
  `nondifferentiable_node` at attempt 22.
- `stride_two_attempt_binds_the_congruent_native_center_and_realized_support` retains
  `raw_input_sha256=5411bda5c0d4ebde3d52afc504e07872e6d5104d5b15d3f3d4474358036163ff`.
- Prepared-JVP and bounded-runner contracts remain green.

### #95

- A forced exact fallback emits comparator status `WeakParameterIdentification`.
- The same comparator includes the exact inner `source_status`.
- Successful comparator JSON omits `source_status`.
- `validation/results/temporal_covariance/no_go_summary.json` retains SHA-256
  `0c885ac25f6680a18b1739e7c126c5821bc153c808c00e7b51c0b4e001ef483e`.

### #96

- A 1,050-count bias family is ineligible for calibration.
- A deterministic 5,000-count unbiased oracle family can produce a passing calibration receipt.
- Exact lower coverage boundaries pass; one count below each boundary fails.
- Permitted asymmetric emissions preserve each valid pairwise comparison.
- A failing unselected method appears in diagnostics and cannot veto the selected method.
- Every cell/method and selected/baseline pair has a retained receipt row.
- Forensic mode can never certify v5.
- Frozen v5 preregistration, generator/scorer, and no-go hashes remain unchanged.

## Validation

```bash
cargo test -p dolphin-workflows --no-default-features --features no-gpu --lib -- --test-threads=1
cargo test -p dolphin-timeseries --lib
cargo test -p dolphin-timeseries --test temporal_covariance_contract
python3 -m unittest validation.tests.test_spatial_covariance_validation
python3 -m unittest validation.tests.test_temporal_covariance_scorer_v7
python3 -m unittest validation.tests.test_temporal_covariance
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --all-targets -- -D warnings
```

After the focused contracts pass, run the relevant broader Rust and Python suites. Record any
GDAL/HDF5 signal failure separately from deterministic assertion results.

## Completion

- Commit and push the exact verified branch.
- Open one unmerged PR with `Closes #94`, `Closes #95`, and `Closes #96` plus red-to-green
  evidence.
- Stop before merge, release, publication, GroundPulse pinning, or deployment.
