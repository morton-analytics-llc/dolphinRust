# Intake: fully realize every open dolphinRust GitHub issue

**Snapshot:** `origin/main` at `e38e88c8120fd395214157ec55ed448730467579` on 2026-08-24.

**Live queue:** [#53](https://github.com/morton-analytics-llc/dolphinRust/issues/53),
[#54](https://github.com/morton-analytics-llc/dolphinRust/issues/54), and
[#57](https://github.com/morton-analytics-llc/dolphinRust/issues/57). All three are open. No open
pull requests exist. This intake supersedes the time-boxed/deferred dispositions in
`md/intake/open-issues-2026-08-24.md` for these three issues.

**Scope decision (2026-08-28):** #53 closes in dolphinRust on frozen synthetic,
resource, and identity evidence. Held-out GNSS validation and independent
scientific review of temporal field evidence move to EO. A passing EO field
receipt permits a separate GroundPulse enablement review. Serving promotion,
deployment, publication, and external claims remain separate EO approvals.
#54's spatial-review contract is unchanged.

## Canonical requirements

| ID | Issue | Requirement | Disposition |
|---|---|---|---|
| GH-054-01 | #54 | Define and implement the target-minus-reference covariance quantity after phase linking and temporal inversion, including shared native-source influence and sequential ancestry. | **Scheduled — F54-01 through F54-04.** |
| GH-054-02 | #54 | Compute `C_pp + C_rr - C_pr - C_rp` for one selected reference without a dense pixel-pair covariance object or an independent-marginal fallback. | **Scheduled — F54-02 through F54-05.** |
| GH-054-03 | #54 | Cover independent, positive, negative, coincident, invalid-reference, masked, branch-unstable, and unsupported-scope cases with analytic and finite-difference contracts. | **Scheduled — F54-01 through F54-03.** |
| GH-054-04 | #54 | Preserve exact dates/gauge, CRS, affine transform, units, masks, reference identity, burst ownership, correction order, estimator identity, replay identity, and scope through whole, tiled, bounded, NRT, and multiburst paths. | **Scheduled — F54-04 through F54-06.** |
| GH-054-05 | #54 | Persist a byte-capped reference-specific covariance factor and machine-readable provenance with rank, conditioning, calibration, and failure statuses. | **Scheduled — F54-05 and F54-06.** |
| GH-054-06 | #54 | Run the preregistered approximation and resource matrix across actual window sizes, strides, support modes, target-reference distances, sequential depths, and source-process stress cells. | **Scheduled — F54-07.** |
| GH-054-07 | #54 | Replace the uncalibrated spatial approximation only for the validated scope, publish the immutable validation receipt, merge on green CI, and close #54. | **Scheduled — F54-08 and F54-09.** |
| GH-053-01 | #53 | Implement an explicit irregular-cadence temporal covariance model that consumes #54's difference factor and includes heteroskedasticity, missing dates, and reference noise. | **Scheduled — F53-01 through F53-03.** |
| GH-053-02 | #53 | Fit covariance parameters with constrained REML/profile likelihood and account for parameter uncertainty with the selected adjusted or complete-refit bootstrap method. | **Scheduled — F53-02 and F53-03.** |
| GH-053-03 | #53 | Preregister and run a seeded Monte Carlo matrix with immutable slope-bias and 68/90/95 percent coverage, proper-score, width, failure, and resource tolerances. | **Scheduled — F53-01, F53-04, and F53-05.** |
| GH-053-04 | #53 | Compare conditional OLS, oracle GLS, plug-in GLS, adjusted inference, and complete-refit bootstrap on the same origin-anchored design; fail closed on unsupported or ill-conditioned fits. | **Scheduled — F53-02 through F53-05.** |
| GH-053-05 | #53 | Validate the frozen method on an untouched, non-Fresno station-pair cohort disjoint by burst/orbit/footprint/site, using same-frame GNSS and InSAR slope differences. | **Deferred — destination: [EO #505](https://github.com/morton-analytics-llc/eo/issues/505); re-enters after dolphinRust #53 closes with matching synthetic, resource, and identity receipts.** EO owns acquisition, scoring, and the field disposition. |
| GH-053-06 | #53 | Persist estimator/version, date/rank/DOF/cadence diagnostics, fitted/raw correlation, #52/#54 identities, reference geometry, scope, bootstrap counts, calibration hashes, and per-pixel status. | **Scheduled — F53-03 and F53-07.** |
| GH-053-07 | #53 | Emit separately named temporal-candidate slope and standard-error products only for scope-matched successful synthetic, resource, and identity receipts; merge on green CI and close #53. The products retain a synthetic-only status and carry no field-calibration claim. | **Scheduled — F53-07 through F53-09.** GroundPulse enablement remains deferred to EO. |
| GH-053-08 | #53 release boundary | Obtain independent scientific review of the temporal field evidence before customer use or publication. | **Deferred — destination: [EO #505](https://github.com/morton-analytics-llc/eo/issues/505); re-enters only after the held-out GNSS gate has a complete, identity-matched result.** The dolphinRust merge and #53 closure contracts exclude this gate. |
| GH-057-01 | #57 | Read raw orbit ephemeris type independently from `/metadata/orbit/orbit_type` without making otherwise valid orbit metadata unreadable when the field is absent. | **Scheduled — F57-01 and F57-02.** |
| GH-057-02 | #57 | Normalize `POEORB` to `precise` and `RESORB` to `restituted`; unknown, missing, or inconsistent stacks remain explicit absence with a reason. | **Scheduled — F57-01 through F57-03.** |
| GH-057-03 | #57 | Add sourced per-run ephemeris-class provenance without changing orbit direction, heading, spacing, timing, or NISAR/data-only behavior. | **Scheduled — F57-03.** |
| GH-057-04 | #57 | Preserve prior geometry-provenance schema deserialization and add round-trip coverage for the new schema and source key. | **Scheduled — F57-04.** |
| GH-057-05 | #57 | Update docs/changelog, merge on green CI, and close #57. | **Scheduled — F57-05.** |

## Ownership boundary

- This intake covers dolphinRust only: analytic kernels, workflow integration, producer artifacts,
  frozen synthetic/resource/identity evidence, and GitHub issue closure.
- EO owns held-out GNSS acquisition and scoring, independent scientific review of the temporal
  field evidence, release pinning, GroundPulse enablement, serving, datum anchoring, station A/B
  behavior, Fresno backfill, API/UI work, deployment, and publication.
- EO must keep its #53 outer cohort non-Fresno and outcome-blinded until the dolphinRust estimator,
  scorer contract, attrition rules, and identities are frozen. EO records failure or
  non-evaluability as a no-go. #53 remains closed.
- #54's independent spatial review remains a dolphinRust #54 requirement.

## Coverage audit

Every intake ID has an explicit disposition. GH-053-05 and GH-053-08 are deferred to EO with named
re-entry gates; all other IDs remain scheduled in this plan. dolphinRust has no UI, so the
backend/UI pairing rule does not apply.
