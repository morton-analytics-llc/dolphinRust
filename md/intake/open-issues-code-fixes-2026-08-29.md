# Intake: close every open dolphinRust issue with code fixes

**Snapshot:** `origin/main` at `e966ef0`; working branch
`codex/move-temporal-field-gates-to-eo` at `7946b9a` on 2026-08-29. Commit `7946b9a` is a
diagnostic full revert of `3a63b3a`: it restores the #94 frozen tests and also removes the bounded
replay work that the final fix will retain.

**Live queue:** #94, #95, and #96 are the only open issues. GitHub REST issue listing and
GitHub search agree; no open pull requests exist.

## Canonical requirements

| ID | Issue | Requirement | Disposition |
|---|---|---|---|
| GH-094-01 | #94 | Restore byte-identical frozen spatial-covariance replay after the bounded-replay change; do not re-freeze expected hashes without a documented scientific reason. | **Scheduled — T94-01 through T94-03.** |
| GH-094-02 | #94 | Restore the frozen `ill_conditioned` and `nondifferentiable_node` attempt statuses while retaining the replay resource bound. | **Scheduled — T94-01 through T94-03.** |
| GH-094-03 | #94 | Prove the three reported deterministic regressions red-to-green with the exact single-threaded no-GPU workflow command. | **Scheduled — T94-01 and T94-03.** |
| GH-095-01 | #95 | Preserve the adjusted-variance fallback's inner error while keeping the public `WeakParameterIdentification` fail-closed status. | **Scheduled — T95-01 and T95-02.** |
| GH-095-02 | #95 | Add a contract that forces the fallback and proves the underlying cause is present in machine-readable provenance or the error source chain. | **Scheduled — T95-01 through T95-03.** |
| GH-095-03 | #95 | Leave the frozen v5 temporal receipt byte-for-byte unchanged. | **Scheduled — T95-02 and T95-03.** |
| GH-096-01 | #96 | Keep the frozen v5 preregistration, receipt, and no-go verdict immutable; corrected code must use a successor scorer identity. | **Scheduled — T96-01 through T96-04.** |
| GH-096-02 | #96 | Retain per-cell, per-method summaries sufficient to identify every oracle and candidate gate failure. | **Scheduled — T96-01 and T96-02.** |
| GH-096-03 | #96 | Correct multiplicity handling, paired-comparator emission alignment, floating-point boundary handling, and the all-comparators promotion dependency. | **Scheduled — T96-01 through T96-03.** |
| GH-096-04 | #96 | Add a throwaway-seed oracle-calibration contract and prove a correctly specified oracle can satisfy the successor scorer before any future preregistration is frozen. | **Scheduled — T96-02 and T96-03.** |
| GH-096-05 | #96 | Verify the corrected scorer with deterministic fixtures and document that it cannot retroactively certify the v5-selected method. | **Scheduled — T96-03 and T96-04.** |

## Boundaries

- The known parallel GDAL/HDF5 SIGBUS/SIGSEGV is not part of #94's three deterministic replay
  regressions. It will not be silently claimed fixed by a single-threaded pass.
- No UI exists in this repository, so the backend/UI pairing rule does not apply.
- No merge, release, crate publication, GroundPulse pin, deploy, or external scientific claim is
  authorized by this intake.

## Coverage audit

Every intake ID is scheduled in the task manifest that will accompany the design. Nothing is
deferred or dropped.
