# Open uncertainty issues intake

**Source:** live GitHub open-issue queue, refreshed 2026-08-23 UTC. The queue contains exactly
issues #52, #53, and #54; none has comments, an assignee, or a milestone.

**Plan:** `md/plans/open-issues-uncertainty-2026-08-23.md`.

**Execution boundary:** planning only. No implementation, issue mutation, PR, merge, release,
publication, external-data acquisition, GroundPulse submodule bump, or deployment is authorized.

| ID | Issue | Canonical requirement | Disposition |
|---|---|---|---|
| DR-052-STATE | #52 | Define and propagate one global temporal-gauge covariance state across ministacks, including compressed-reference uncertainty and the cross-covariance needed for retained real dates. | **Scheduled: T52-01 through T52-04** |
| DR-052-MEMORY | #52 | Keep memory bounded; select and document a factor, banded, blockwise, or sufficient-statistic representation instead of a dense `n_dates x n_dates x area` cube. | **Scheduled: T52-01, T52-03, T52-06** |
| DR-052-GAUGE | #52 | Treat acquisition 0 as an exact gauge constraint or exclude it from stochastic fitting; never encode it as epsilon variance or extreme precision. | **Scheduled: T52-01 through T52-04** |
| DR-052-STATUS | #52 | Distinguish global propagated covariance from per-ministack marginal CRLB diagnostics in machine-readable output method/status. | **Scheduled: T52-04 and T52-05** |
| DR-052-ANALYTIC | #52 | Add an in-repo deterministic two-ministack analytic covariance contract that cannot skip for missing oracle data. | **Scheduled: T52-02** |
| DR-052-FAIL | #52 | Add fail-closed contracts for singular, nonfinite, invalid-reference, and `max_num_compressed = 0` cases. | **Scheduled: T52-02 through T52-04** |
| DR-052-BOUNDARY | #52 | Keep velocity inference independent of the covariance product until #54 propagation and #53 downstream coverage gates pass; kernel parity alone cannot support predictive or field-calibrated claims. | **Scheduled: T52-01, T52-05, T52-06, T54-07, and T53-07; production wiring is prohibited before the signed promotion manifest** |
| DR-054-QUANTITY | #54 | Define the target/reference spatial covariance required after phase linking and temporal inversion, including overlapping phase-link windows. | **Scheduled: T54-01 and T54-03** |
| DR-054-BOUNDED | #54 | Propagate `Var(target-reference)` without an unbounded dense pixel-pair covariance object. | **Scheduled: T54-01 and T54-03 through T54-06** |
| DR-054-ANALYTIC | #54 | Add independent, positive-correlation, negative-correlation, coincident, and invalid-reference analytic fixtures. | **Scheduled: T54-02** |
| DR-054-GEOMETRY | #54 | Preserve CRS, units, mask, exact temporal gauge, reference identity, and estimator provenance through whole-frame and bounded re-referencing. | **Scheduled: T54-02, T54-04, and T54-05** |
| DR-054-VALIDATION | #54 | Validate approximation error across phase-link window sizes and target/reference distances before exposing inferential status. | **Scheduled: T54-06 and T54-07** |
| DR-054-BOUNDARY | #54 | Keep current outputs labeled as an uncalibrated target/reference marginal approximation until independent review and downstream coverage pass; metadata or the current diagonal product cannot stand in for propagation, total uncertainty, or asset risk. | **Scheduled: T54-01, T54-05 through T54-07, T53-06, and T53-07** |
| DR-053-MODEL | #53 | Specify a temporal covariance model for irregular cadence and spatially referenced observations, including reference noise, missing dates, and heteroskedasticity. | **Scheduled: T53-01** |
| DR-053-ESTIMATION | #53 | Estimate covariance parameters with parameter uncertainty, using constrained REML/profile likelihood and a complete-fit bootstrap or another method proven against the same contract; keep scalar effective-N diagnostic-only. | **Scheduled: T53-02 through T53-04** |
| DR-053-PREREG | #53 | Preregister a seeded Monte Carlo grid and slope-bias plus 68/90/95 percent interval-coverage tolerances before running the experiment. | **Scheduled: T53-02 and T53-03** |
| DR-053-COMPARATORS | #53 | Compare OLS, oracle GLS, plug-in covariance fitting, and the selected corrected method; fail closed for boundary correlation, weak identification, ill-conditioning, too few dates, and unsupported cadence. | **Scheduled: T53-02 through T53-04** |
| DR-053-HELDOUT | #53 | Validate coverage on held-out GNSS station-pair data split by independent burst/site; Fresno cannot serve both method selection and validation. | **Scheduled: T53-05, with dataset sufficiency as a hard execution gate** |
| DR-053-PROVENANCE | #53 | Persist estimator version, valid-date count, rank/DOF, cadence status, raw and fitted correlation, covariance method, condition status, and calibration scope. | **Scheduled: T53-04 and T53-07** |
| DR-053-RELEASE | #53 | Emit corrected inferential slope standard error only after synthetic and held-out coverage gates pass; otherwise retain the conditional component and diagnostics without rescaling. Require independent scientific review of the coverage artifact before merge/release. | **Scheduled: T53-06 and T53-07** |

## Coverage audit

All acceptance gates and release boundaries in open issues #52 through #54 have an explicit
disposition. Each gate maps to a scheduled task in the plan. #54 composes #52's temporal factor
with spatial-reference covariance; #53 consumes the resulting referenced factor only after both
producer contracts pass. dolphinRust has no UI, so no paired UI task applies. Future GroundPulse
consumption requires a separate `eo` intake and plan after the engine and coverage gates pass.
