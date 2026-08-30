# Implementation plan: fully realize every open dolphinRust GitHub issue

**Status:** ready for execution.

**Intake:** `md/intake/open-github-issues-full-2026-08-24.md`.

**Base snapshot:** `origin/main` at `e38e88c8120fd395214157ec55ed448730467579`.

**Open issues:** [#53](https://github.com/morton-analytics-llc/dolphinRust/issues/53),
[#54](https://github.com/morton-analytics-llc/dolphinRust/issues/54), and
[#57](https://github.com/morton-analytics-llc/dolphinRust/issues/57). This plan replaces the
time-boxed and deferred execution plans for these issues. It does not stop at contract-only PRs,
research harnesses, or unmerged branches: every workstream ends on green `main` with its issue
closed.

**Scope decision (2026-08-28):** #53 closes on its frozen synthetic, resource,
and identity gates. EO owns held-out GNSS validation, independent scientific
review of temporal field evidence, GroundPulse enablement, and publication.
The #54 independent spatial-review gate remains unchanged.

## Objective

Deliver the complete scientific and producer behavior described by all three open issues:

1. expose sourced POEORB/RESORB ephemeris provenance (#57);
2. compute and persist bounded, reference-specific spatial covariance from shared phase-link
   influence through the actual L2 displacement path (#54); and
3. fit and emit an irregular-cadence temporal slope candidate using #54's difference factor,
   including parameter uncertainty and frozen synthetic, resource, and identity evidence (#53).

The completed tree must emit a reproducible fixed-cube velocity product, sourced LOS geometry,
reference-specific displacement covariance, and separately named temporal-candidate
slope/standard-error products with machine-verifiable provenance. This repository ends at those
candidate artifacts and their receipts. EO owns held-out GNSS validation, independent temporal
scientific review, EO code, GroundPulse enablement, serving, Fresno anchoring, datum offsets,
deployment, publication, and submodule pinning.

## Current state

- #52's source-keyed sequential replay DAG, byte preflight, HDF5 operator, and source/config
  identity are merged.
- PR #58 merged a joint target/reference replay API that computes
  `C_pp + C_rr - C_pr - C_rp` from one supplied influence DAG with exact gauge zero and a byte cap.
  It does not yet derive production phase-link influence, compose the actual L2 map, or persist a
  #54 artifact.
- PR #61 merged a 36-cell bookkeeping loop. Window size and target/reference distance are labels;
  the test repeats one pixel-level fixture and measures no raster approximation error.
- PR #60 merged a 32-cell temporal harness with OLS/oracle-GLS point estimates and an explicit
  blocked status. It has no cadence-irregularity dimension, covariance-parameter fit, bootstrap,
  interval coverage, temporal-candidate product, or complete synthetic/resource receipt.
- Legacy `velocity_sigma.tif` remains an IID-conditional component and legacy displacement
  variance remains an independent-marginal approximation. Their non-inferential labels stay in
  place; the new synthetic-validated candidate products are separate artifacts.
- #57's data uncertainty is resolved. `oracle/fixtures/geomprov_ci_cslc.h5` contains scalar
  `/metadata/orbit/orbit_type = POEORB`; its HDF5 description defines POEORB as precise and RESORB
  as restituted. The current reader simply does not expose it.

## Design summary

### #57 orbit ephemeris provenance

Read `/metadata/orbit/orbit_type` independently from the existing orbit-vector bundle. Normalize
only `POEORB -> precise` and `RESORB -> restituted`. Add optional
`orbit_ephemeris_class` to geometry provenance schema `/4` with the exact source key and raw value.
Missing, unknown, or mixed-stack values make this field absent with a reason; they do not erase
valid direction, heading, spacing, timing, LOS, or decomposition geometry.

### #54 spatial-reference covariance

Use #52's immutable replay descriptor to reconstruct the fixed production branch for the selected
reference and a byte-capped target microbatch. For each phase-link pixel, derive shared native
source influence from the EMI/EVD estimating equation and implicit Jacobian. Compose that
influence through sequential compressed ancestry and the actual fixed-weight L2 inversion map.
For target `p` and reference `r`, persist the rank-revealing factor for:

```text
C_delta(p,r) = C_pp + C_rr - C_pr - C_rp
             = (H_p E_p R_p - H_r E_r R_r)
               (H_p E_p R_p - H_r E_r R_r)'
```

`R` is the replayed phase influence, `E` is the valid interferogram contrast map, and `H` is the
exact L2 solve used for that pixel. Acquisition 0 remains outside the stochastic state. Coincident
target/reference is exact zero. Production may cache one reference signature and one bounded
target microbatch; it may not allocate an all-pixel influence matrix, a dense pixel-pair object,
or a date-squared spatial cube.

Persist `referenced_displacement_covariance_factor.h5` plus
`referenced_displacement_covariance_provenance.json`. The artifact is reference-specific and binds
the date/gauge map, CRS/grid, units, mask, source burst, target/reference coordinates, estimator
branch, #52 operator identity, source/support/config hashes, rank/conditioning, approximation
receipt, resource receipt, and stable status codes.

### #53 temporal-covariance slope inference

The first corrected scope is a linear origin-anchored slope on a calibrated same-burst #54
difference factor. Seasonal and step models retain conditional output until separately validated.
For post-gauge observations:

```text
y_t - y_0 = beta * (t - t_0) + e_t
D[i,i] = sqrt(C54_delta[i,i] / geometric_mean_positive_diag(C54_delta))
V(theta) = C54_delta + sigma^2 D R(rho_12) D
R[i,j] = 1                                      when i = j
R[i,j] = rho_12^(abs(t_i - t_j) / 12 days)      otherwise
```

Missing dates subset `C54_delta`, `D`, and the design without imputation. Fit `sigma^2` and
`rho_12` with constrained REML/profile likelihood. Compare conditional OLS/WLS, oracle GLS,
plug-in GLS, Kenward-Roger-style adjusted scalar inference, profile inference, and complete-refit
parametric bootstrap on the same origin-anchored design. The selected candidate method must emit
one paired slope and scalar standard error whose symmetric 68/90/95 percent intervals pass the
frozen synthetic criteria; otherwise the temporal-candidate product is absent for that pixel with
a stable status.

Emit new products only:

- `velocity_temporal_gls.tif`;
- `velocity_sigma_corrected.tif`;
- temporal status/valid-date/rank/DOF/conditioning/correlation diagnostics; and
- `velocity_inference_provenance.json`.

Do not relabel or overwrite legacy `velocity.tif` or `velocity_sigma.tif`.
The new products remain synthetic-validated candidates until EO passes held-out GNSS and
independent temporal-review gates and separately authorizes GroundPulse use.

## Technical requirements

### R57 — ephemeris provenance

- Add `read_cslc_orbit_type(path) -> Result<String>` in `dolphin-io`; keep it independent from
  `read_cslc_orbit()`.
- Read exact fixed-width HDF5 scalar strings and preserve the raw value in field provenance.
- Require all readable CSLC granules in the stack to agree after case-normalized parsing.
- Emit `precise`, `restituted`, or explicit absence. Never guess from filenames when the sourced
  dataset is available.
- Bump geometry provenance to `dolphinrust-geometry-provenance/4`; `/2` and `/3` deserialize with
  the new field absent.

### R54 — production spatial covariance

- Add stable statuses for invalid reference, masked target, temporal-factor failure, replay
  mismatch, nonfinite/nondifferentiable influence, unstable support, unsupported estimator/model,
  multiburst ownership, condition failure, and calibration-scope mismatch.
- Derive EMI/EVD score/Hessian influence and effective-look scaling from the same realized Rect,
  GLRT, or KS support used by production. Central finite differences must validate every supported
  branch away from declared branch boundaries.
- Keep #52 Fisher transitions and #54 estimating-equation influence distinct. Neither may be
  rescaled to force equal marginals.
- Compose `F_influence` and `T_influence = w_s + w_x F_influence` through every sequential parent,
  including cap eviction and partial trailing blocks. Shared ancestry must carry covariance beyond
  immediate native-window overlap.
- Apply each pixel's actual valid-observation `E/H` map. L1, changed unwrap branches, phase-bias
  correction, unstable adaptive support, or unsupported correction order return a status; they do
  not fall back to independent marginals.
- Resolve the final whole/bounded reference before finalizing the contrast. Apply deterministic
  corrections to target and reference before subtraction. Apply `(wavelength / 4 pi)^2` once at
  the output boundary.
- Whole, tiled, bounded, and eligible NRT paths must agree. Same-pre-leveling-burst multiburst
  pairs are eligible only when the common seam rotation cancels in the tested contrast; mixed or
  ambiguous ownership remains explicitly unsupported.
- Stream the reference-specific factor and sidecar transactionally. Corrupt, stale, mismatched,
  partial, or scope-incompatible artifacts fail before corrected inference.
- The production scope becomes `calibrated_scope_match` only when its immutable #54 analytic,
  approximation, resource, and review receipt hashes match the code and inputs.

### R53 — synthetic-validated temporal candidate

- Require a calibrated same-scope #54 difference factor. Marginal CRLB, two-marginal addition,
  zero-cross metadata, or a bare #52 factor are rejected.
- Support strictly increasing dates with the frozen cadence predicate, at least 12 valid common
  dates, exact acquisition-0 gauge, positive post-gauge diagonal scale, and positive-definite total
  `V(theta)` under the frozen condition limit.
- Persist unclamped adjacent-residual correlation separately from fitted `rho_12`; record pair
  count and min/median/max elapsed gap. Fewer than three pairs means absent, not zero.
- The complete-refit bootstrap resimulates and refits both mean and covariance parameters for
  every replicate. Fit failures stay in the attempted denominator.
- The frozen synthetic matrix includes 12/24/48/96 valid dates; regular, alternating, jittered,
  and gapped cadence; none/MCAR/block missingness; variance ratios 1/4/16; reference contribution;
  #54 overlap/distance/sequential-depth/approximation strata; supported, boundary, weak-ID, and
  invalid cells; 1,050 seeds for each of 24 supported cells on both execution paths (50,400 frozen
  attempts); and end-to-end raw-look cells.
- Frozen per-cell criteria are standardized slope bias <= 0.05 empirical SD; absolute coverage
  error <= 0.03/0.02/0.015 at 68/90/95 percent; >= 99 percent supported-cell emission; improved
  proper interval score over conditional and plug-in baselines; and no width-only inflation pass.
- Close #53 only when the complete frozen synthetic matrix and resource limits pass and the exact
  preregistration, producer source set, binary, #52 replay, and reviewed #54 spatial-factor
  identities match.
- Emit temporal-candidate products only when those synthetic, resource, and identity receipts
  match. Unsupported pixels remain conditional-only with stable machine-readable reasons.
- Do not acquire or score held-out GNSS in this plan. EO owns the non-Fresno cohort, independent
  temporal scientific review, GroundPulse enablement, and publication after #53 closes.

## Constraints and guardrails

- Start every analytic change with a red contract or parity test. A harness that records labels
  without exercising the named dimensions does not count.
- Do not add stubs, TODOs, `unimplemented!()`, placeholder manifests, synthetic production data,
  or fallback inference.
- Preserve the exact acquisition-0 gauge, fixed-cube common-date contract, sourced LOS geometry,
  estimator identity, mask identity, and burst/reference lineage already on `main`.
- Never allocate a structure with two area-scaled axes. Run byte preflight before source reads or
  numeric replay.
- Rust executes every estimator and production operator. Python may generate fixtures, score
  receipts, and render deterministic plots; it may not substitute a second scientific
  implementation.
- Record failed validation attempts and fix the implementation or narrow an objectively
  unsupported pixel state. Do not loosen frozen tolerances, drop failed cells, top up successful
  seeds, or defer an acceptance item.
- Use separate clean worktrees from refreshed `origin/main`. Before parallel agents start, commit
  this intake/plan, require a clean tree, assign non-overlapping file manifests, and merge branches
  sequentially.
- #57 may run in parallel with early #54 kernel work because their write scopes do not overlap.
  #53 production integration starts only after the #54 public factor/schema is merged.
- The #54 reviewer must not author the code or validation artifacts under review. Its findings are
  fixed and rerun before merge; the #54 spatial-review contract is unchanged. Temporal field review
  is an EO gate after #53 closure.
- Merge each PR only after exact-head CI passes. After every merge, run combined-tree CI before
  starting the next dependent branch. Final completion requires all three issues closed on green
  `main`.
- Do not release, publish crates, pin EO, deploy, mutate serving state, or perform a Fresno datum
  anchor under this plan.

## Test contract

### #57 contracts

| ID | Scenario | Proof | Location |
|---|---|---|---|
| C57-01 | Real committed CSLC crop | Reads raw `POEORB` from `/metadata/orbit/orbit_type`. | `crates/dolphin-io/src/cslc_metadata.rs` |
| C57-02 | Synthetic RESORB, unknown, and missing field | Normalizes only supported values; missing/unknown does not break existing orbit reads. | IO unit tests and `geometry_provenance_contract.rs` |
| C57-03 | Mixed POEORB/RESORB stack | Emits only ephemeris-class absence with an explicit inconsistency reason. | `geometry_provenance_contract.rs` |
| C57-04 | Data-only and NISAR inputs | Remain explicit absence without a default or filename inference. | `geometry_provenance_contract.rs` |
| C57-05 | Schema compatibility | `/2` and `/3` deserialize; `/4` round-trips raw value, normalized class, and exact source key. | `geometry_provenance_contract.rs` |

### #54 contracts

| ID | Scenario | Proof | Location |
|---|---|---|---|
| C54-01 | Independent, positive, negative, coincident, invalid reference | Exact matrix algebra and stable status; coincident is exact zero. | phaselink/workflow spatial contracts |
| C54-02 | Interior, clamped border, masked, Rect/GLRT/KS, stride > 1 | Production support and source-key intersection are replayed exactly. | `dolphin-phaselink` spatial-influence contract |
| C54-03 | EMI/EVD finite differences, ties, wrap/branch changes | Supported score/Hessian/JVP agrees with finite differences; unstable cases abstain. | phaselink estimator/influence contracts |
| C54-04 | Gauge/reference congruence and joint factor | Joint target/reference covariance is PSD and transforms by congruence without forced marginal equality. | phaselink spatial contract |
| C54-05 | Two-ministack shared ancestry | `F_influence/T_influence` matches a tiny dense oracle and carries covariance past immediate overlap. | workflow sequential spatial contract |
| C54-06 | Actual fixed-weight L2 maps with different missing observations | Correct `E/H` maps, rank/nullity, pseudo-operations, units, and gauge. | timeseries spatial covariance contract |
| C54-07 | Whole/tiled/bounded/NRT final reference | All eligible paths return the same contrast/factor and preserve final-reference lineage. | workflow displacement/NRT contracts |
| C54-08 | Corrections and multiburst ownership | Correction order is target-minus-reference correct; same-burst seam cancels; mixed ownership abstains. | workflow correction/multiburst contracts |
| C54-09 | HDF5/JSON transaction and corruption | Round-trip preserves every identity/status; stale, corrupt, mismatched, or partial artifacts fail closed. | `dolphin-io` covariance contract |
| C54-10 | Frozen raster approximation matrix | Actual window/stride/support/distance/depth cells meet operator, variance, PSD, coverage, and emission criteria. | Rust batch plus Python scorer |
| C54-11 | 256x256 at 13/26/52 dates | Peak RSS < 24 GiB; area/date scaling rejects quadratic growth and two-area-axis allocation. | release benchmark |
| C54-12 | Output boundary | Legacy products stay uncalibrated; only the matching new factor reports calibrated spatial scope. | workflow output/provenance contracts |

### #53 contracts

| ID | Scenario | Proof | Location |
|---|---|---|---|
| C53-01 | Hand irregular/missing/heteroskedastic matrices | Correct `D`, continuous-time `R`, gauge removal, date subsetting, and #54 factor consumption. | `temporal_covariance_contract.rs` |
| C53-02 | Scalar effective-N counterexample | Demonstrates wrong slope variance/coverage and keeps scalar correction diagnostic-only. | compact analytic contract |
| C53-03 | Seeded oracle GLS and every comparator | Same origin-anchored data/design, separate method outputs, correct known slope. | timeseries contract |
| C53-04 | REML/profile and complete-refit bootstrap | Parameter uncertainty is included; boundary/weak-ID/bootstrap failures return stable status. | timeseries contract and Rust batch |
| C53-05 | Raw/fitted correlation diagnostics | Raw value is unclamped with pair/gap summary; absent is distinct from zero. | timeseries/workflow contracts |
| C53-06 | Invalid #54/scope/gauge/grid/unit/model inputs | No corrected sigma and one stable reason; no marginal fallback. | timeseries/workflow contracts |
| C53-07 | Frozen synthetic matrix | Every cell/method reports attempted/emitted/failed counts, coverage, bias, scores, widths, timing, RSS, and identities. | Rust batches plus Python scorer/schema test |
| C53-08 | End-to-end raw-look cells | The same realization produces #52/#54 and slope inference; fixed-factor success cannot hide production-path error. | workflow batch contract |
| C53-09 | Untouched station-pair cohort | **Deferred to EO.** EO scores identity-matched same-frame InSAR/GNSS slope differences after #53 closes. The dolphinRust closure proof excludes this field result. | EO intake/plan |
| C53-10 | Temporal-candidate output authorization | Only scope-matched successful synthetic/resource/identity receipts emit paired candidate slope/SE; legacy products remain unchanged. | workflow output contract |
| C53-11 | Provenance/status persistence | Rasters and sidecar preserve all estimator, date, reference, covariance, calibration, and receipt identities. | workflow provenance contract |
| C53-12 | 256x256 at 12/48/96 dates | Peak RSS < 24 GiB, <= 2x conditional-fit wall time, no whole-frame covariance, projected 3.9M pixels <= 60 minutes. | release benchmark |
| C53-13 | Full regression | Workspace Rust/Python tests remain green and pass/fail/not-evaluable remain distinct. | CI `verify` job |

## Implementation plan

### P00 — land the plan and establish execution worktrees

**Files:** this plan and its intake only.

1. Run the coverage audit below and commit the two planning files on the current documentation
   branch.
2. Merge that documentation PR, refresh `origin/main`, and verify a clean combined tree.
3. Create clean worktrees with explicit ownership:
   - `issue57-orbit`: `dolphin-io` metadata, workflow provenance, geometry-provenance tests/docs;
   - `issue54-kernel`: phaselink, sequential replay, timeseries spatial map, focused Rust contracts;
   - `issue54-validation`: validation JSON/Python/results/bench/docs only after the #54 API freezes.
4. Reject out-of-scope edits and merge each branch sequentially.

### F57-01 — write red orbit-class contracts

**Requirements:** GH-057-01 through GH-057-04.

**Files:** `crates/dolphin-io/src/cslc_metadata.rs`,
`crates/dolphin-workflows/tests/geometry_provenance_contract.rs`.

Add C57-01 through C57-05 first. Use the committed real crop for POEORB and synthetic fixtures for
RESORB, unknown, missing, and mixed stacks. Assert that missing ephemeris class leaves every
existing sourced field unchanged.

### F57-02 — add independent raw IO

**Files:** `crates/dolphin-io/src/cslc_metadata.rs`, `crates/dolphin-io/src/lib.rs`.

Implement `read_cslc_orbit_type()` against exact path `/metadata/orbit/orbit_type`. Keep the
existing orbit-vector reader unchanged and independently fallible. Turn C57-01/C57-02 green.

### F57-03 — normalize and emit sourced provenance

**Files:** `crates/dolphin-workflows/src/provenance.rs`, focused contracts.

Add `orbit_ephemeris_class`, raw value, exact source key, and per-field absence reason. Normalize
POEORB/RESORB, require stack agreement, and keep the field outside
`decomposition_geometry_complete`. Turn C57-03/C57-04 green.

### F57-04 — schema and compatibility

**Files:** `md/design/geometry-provenance.md`, schema tests, `CHANGELOG.md`.

Bump the producer to schema `/4`, add `serde(default)` compatibility for `/2` and `/3`, document
the new field and method version, and turn C57-05 green.

### F57-05 — merge and close #57

Run focused IO/provenance tests, formatting, workspace check, strict Clippy, and full workspace
tests. Open one PR with `Closes #57`, require exact-head CI, merge, refresh `origin/main`, and
require green push CI. Close #57 only if GitHub did not close it automatically.

### F54-01 — replace the bookkeeping harness with red production contracts

**Requirements:** GH-054-01 through GH-054-03.

**Files:** new phaselink/timeseries/workflow spatial covariance contracts, committed tiny JSON
fixtures, `md/design/spatial-reference-covariance.md`.

Freeze `reference_specific_influence_v1`, stable status registry, effective-look rule,
gauge/congruence semantics, correction/reference order, supported branches, resource formula, and
artifact schema. Add C54-01 through C54-08 red. Explicitly prove that PR #61's window/distance
labels do not satisfy C54-10 and replace that loop with production-path cells.

### F54-02 — implement local phase-link shared-source influence

**Files:** `crates/dolphin-phaselink/src/covariance.rs`, `estimator.rs`, `fused.rs`, `engine.rs`,
`source_influence.rs`, new `spatial_covariance.rs`, exports, focused contracts.

Implement the EMI/EVD score, Hessian/implicit Jacobian, effective-look source loading, source-key
intersection, joint target/reference factor, and stable branch/support failure statuses. Reuse the
production support iterator and #52 replay identity. Turn C54-01 through C54-04 green.

### F54-03 — compose influence through sequential ancestry

**Files:** `crates/dolphin-workflows/src/sequential_covariance.rs`, `sequential.rs`,
`crates/dolphin-phaselink/src/quality.rs`, sequential spatial contracts.

Compute and finite-difference `F_influence` and `T_influence`; propagate new-source innovations and
carried-parent influence without double counting. Implement a cached reference signature and
byte-capped target microbatch over the persisted #52 DAG. Cover partial blocks, cap eviction,
stride, tile halo, and dependency beyond immediate overlap. Turn C54-05 green.

### F54-04 — apply the actual L2 temporal map

**Files:** `crates/dolphin-timeseries/src/inversion.rs`, `reference.rs`, new spatial covariance
module and contracts; workflow integration boundary.

Expose each pixel's fixed valid-observation `E/H` map and contract the replayed phase influence
into displacement covariance. Add rank-revealing factor operations, diagonal/selected blocks,
pseudo-solve/whitening/log-pseudodeterminant where mathematically defined, exact gauge, and unit
conversion. Turn C54-06 green; make L1 and changed branches return explicit unsupported status.

### F54-05 — persist the reference-specific factor

**Files:** extend `crates/dolphin-io/src/covariance.rs`; new workflow spatial provenance/artifact
module; IO/workflow contracts.

Add a chunked reference-specific HDF5 schema and transactional JSON sidecar. Store only bounded
reference-specific blocks, rank/status, date map, source/replay IDs, grid/reference/burst/mask/unit
facts, approximation/resource hashes, and calibration scope. Validate hashes and byte caps before
numeric reads. Turn C54-09 green.

### F54-06 — wire whole, tiled, bounded, NRT, and multiburst outputs

**Files:** `crates/dolphin-workflows/src/displacement.rs`, `sequential.rs`, `tiling.rs`, bounded/NRT
and multiburst contracts, output policy/config disposition tests.

Select the final reference before contrast finalization, apply corrections before subtraction,
carry source-burst ownership, and stream the factor. Eligible NRT replay must reuse immutable
sealed parents; unsupported NRT layouts remain explicit. Prove whole/tiled/bounded equivalence and
same-burst seam cancellation. Keep legacy products labeled uncalibrated. Turn C54-07/C54-08/C54-12
green.

### F54-07 — run the real approximation and resource experiment

**Requirements:** GH-054-06.

**Files:** `validation/spatial_covariance_preregistration.json`, deterministic generator/scorer and
tests, release-mode Rust batch/benchmark, `validation/results/spatial_covariance/`, `VALIDATION.md`.

Freeze the exact cell list, seeds, tolerances, support/window/stride/distance/depth dimensions,
effective-look rule, hardware, and hashes. Run at least 5,000 attempted seeds per supported cell
through the production Rust operator. Record every cell, including failures and non-evaluable
states. Run the 256x256 x 13/26/52-date benchmark. Fix implementation defects and rerun the same
frozen matrix until C54-10/C54-11 pass; do not change thresholds or drop cells.

### F54-08 — independent review and immutable #54 receipt

**Requirements:** GH-054-07.

Assign a reviewer with no #54 implementation ownership. Review equations, source, analytic and
finite-difference evidence, full approximation matrix, failures, resource receipts, schema, and
scope. Resolve every finding in code/tests and rerun affected artifacts. Persist the review receipt
and method manifest with design/code/result/resource hashes. A schema test must prove that stale or
mismatched evidence cannot report `calibrated_scope_match`.

### F54-09 — merge and close #54

Merge the #54 kernel/integration, validation, and receipt PRs sequentially on green exact-head CI.
After the final merge, run combined-tree CI and verify the production factor reports calibrated
spatial scope only for the reviewed configuration. Close #54. Leave corrected temporal inference
disabled until #53 lands.

### F53-01 — freeze the full preregistration and write red contracts

**Requirements:** GH-053-01 through GH-053-04.

**Files:** `md/design/temporal-covariance-slope-inference.md`,
`validation/temporal_covariance_synthetic_engine_preregistration.json`, new Rust/Python analytic
contracts and fixtures.

Replace PR #60's reduced 32-cell grid with the complete model, cell matrix, seed hierarchy,
supported cadence, parameter bounds, status registry, bias/coverage/score/width criteria, resource
limits, and output schema. Add C53-01 through C53-06 red, including the scalar-effective-N
counterexample and direct #54 independent/positive/negative/coincident/invalid fixtures.

### F53-02 — implement estimators and comparators

**Files:** new `crates/dolphin-timeseries/src/temporal_covariance.rs`, exports,
`inversion.rs`, `velocity_model.rs`, focused contracts, release-mode JSONL batch target.

Implement covariance construction, constrained REML/profile fitting, origin-anchored GLS,
adjusted scalar inference, profile inference, complete-refit bootstrap, factorization/conditioning,
raw diagnostics, and stable statuses. Keep every comparator separate and label the legacy
intercept-plus-slope WLS non-comparable. Turn C53-01 through C53-06 green.

### F53-03 — add estimator and status provenance

**Requirements:** GH-053-06.

**Files:** temporal output structs, new `velocity_inference_provenance.rs`, schema tests.

Define per-pixel outputs and sidecar fields for estimator/version, valid dates, rank/DOF, cadence,
raw/fitted correlation, reference geometry, #52/#54 identities, window/overlap/distance stratum,
condition/scope, bootstrap counts, approximation bound, and receipt hashes. No writer is enabled
until F53-07.

### F53-04 — implement production-path simulation batches

**Files:** Rust timeseries and workflow release binaries/examples, Python generator/scorer and
tests, artifact schema tests.

Run every estimator in Rust. Support fixed-factor cells and end-to-end cells that regenerate raw
complex looks, #52 replay, #54 difference covariance, and slope inference from the same seed.
Write immutable JSONL/results with attempted/emitted/failed counts and resource facts. Turn the
batch/parity parts of C53-07/C53-08 green.

### F53-05 — run and pass the frozen synthetic matrix

**Requirements:** GH-053-03 and GH-053-04.

Generate outcomes once per code/result identity, score every frozen cell, and commit the complete
receipt under `validation/results/temporal_covariance/`. Preserve all failed attempts. Fix the Rust
implementation and rerun the unchanged preregistration until every supported cell passes bias,
coverage, emission, proper-score, width, and resource criteria. Turn C53-07/C53-08 green.

### F53-06 — hand the held-out contract to EO

**Requirements:** GH-053-05.

**Disposition:** deferred to [EO #505](https://github.com/morton-analytics-llc/eo/issues/505).
dolphinRust retains reusable field-validation tooling for EO but does not acquire, unblind, or
score the cohort. EO runs the identity-matched field gate after #53 closes.

EO re-enters after #53 closes with matching synthetic, resource, preregistration, producer, #52,
and reviewed #54 identities. EO freezes the non-Fresno cohort, outcome-blinding, attrition, scorer,
and GNSS provenance before unblinding, then records pass, fail, or non-evaluable. C53-09 remains a
downstream EO gate. F53-07 through F53-09 proceed independently of that result.

### F53-07 — wire temporal-candidate products

**Requirements:** GH-053-06 and GH-053-07.

**Files:** `crates/dolphin-core/src/config.rs`, config disposition contracts,
`crates/dolphin-workflows/src/displacement.rs`, temporal provenance/output modules, benchmarks,
docs/changelog.

Add the explicit uncertainty-method enum and verify matching synthetic, resource, #52/#54/#53
receipt identities before allocation. Assemble total `V(theta)`, emit `velocity_temporal_gls.tif`,
`velocity_sigma_corrected.tif`, diagnostics/status rasters, and the sidecar. Preserve legacy
products unchanged, label the new pair as a synthetic-validated candidate, and never fall back.
Run the 12/48/96-date benchmark and turn C53-10 through C53-12 green. This task does not enable
GroundPulse.

### F53-08 — bind the immutable #53 synthetic promotion manifest

Persist `temporal_covariance_promotion_manifest.json` schema
`dolphinrust-temporal-covariance-promotion/2`, binding the frozen preregistration, producer source
set and binary, complete synthetic result, resource receipt, #52 replay identity, reviewed #54
spatial-factor identity, output semantics, and every failed attempt. Prove missing, failed, stale,
or scope-mismatched evidence leaves output `conditional_only`. The manifest must identify the scope
as `synthetic_validated_scope_match`; field-calibrated and GroundPulse-enabled scopes are invalid.
Independent temporal scientific review is deferred to EO under GH-053-08.

### F53-09 — merge and close #53

Merge research, synthetic validation, product, and completion-receipt PRs sequentially on green
exact-head CI. Run combined-tree CI on the final merge. Execute one fixed-cube local smoke run and
verify the exact common dates, sourced LOS, mask/reference identity, reference-specific #54 factor,
temporal-candidate paired products, diagnostics, and matching manifests. Close #53 once the frozen
synthetic, resource, and identity gates pass. Record held-out GNSS, independent temporal review,
GroundPulse enablement, and publication as EO-owned downstream gates.

### P99 — final repository reconciliation

Refresh GitHub and assert:

- open issue count is zero;
- no open implementation PR remains;
- `origin/main` contains every merge;
- push CI for final `main` is green;
- the root worktree and user-owned worktrees were not reset or cleaned;
- the final synthetic/resource receipts match the merged code identity; and
- no held-out GNSS result, temporal independent review, release, publication, EO pin, GroundPulse
  enablement, deployment, serving mutation, or Fresno anchor was claimed.

## Validation commands

```text
cargo test -p dolphin-io cslc_metadata
cargo test -p dolphin-workflows --test geometry_provenance_contract

cargo test -p dolphin-phaselink --test spatial_reference_covariance_contract
cargo test -p dolphin-timeseries --test spatial_reference_covariance_contract
cargo test -p dolphin-workflows --test spatial_reference_covariance_contract
cargo test -p dolphin-workflows --test spatial_reference_covariance_validation
cargo run --release -p dolphin-workflows --example spatial_covariance_batch -- \
  --prereg validation/spatial_covariance_preregistration.json
cargo run --release -p dolphin-workflows --example spatial_covariance_bench

cargo test -p dolphin-timeseries --test temporal_covariance_contract
cargo run --release -p dolphin-timeseries --example temporal_covariance_batch -- \
  --contract validation/fixtures/temporal_covariance_batch.jsonl
cargo run --release -p dolphin-workflows --example temporal_covariance_e2e_batch -- \
  --contract validation/fixtures/temporal_covariance_e2e_batch.jsonl
cargo run --release -p dolphin-workflows --example temporal_inference_bench
oracle/.venv/bin/python -m unittest discover -s validation/tests
oracle/.venv/bin/python validation/temporal_covariance_simulation.py \
  --prereg validation/temporal_covariance_synthetic_engine_preregistration.json \
  --run-root validation/results/temporal_covariance/run \
  --resource-evidence-directory validation/results/temporal_covariance \
  --seeds 1050 \
  --output validation/results/temporal_covariance/coverage.json
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
git diff --check
gh issue list --state open --limit 100
gh pr list --state open --limit 100
```

Every validation driver must verify preregistration, code, input, factor, and result hashes before
scoring. It must return distinct process outcomes for pass, scientific failure, and non-evaluable
input.

## Merge sequence

```text
plan/intake
  -> #57 orbit provenance (independent; may overlap early #54 work)
  -> #54 analytic kernel
  -> #54 workflow/persistence
  -> #54 validation/review/promotion
  -> #54 closure
  -> #53 estimator/research
  -> #53 synthetic/resource validation
  -> #53 temporal-candidate product + identity receipt
  -> #53 closure
  -> final combined-tree reconciliation
  -> downstream EO held-out GNSS + independent temporal review + GroundPulse decision
```

## Open questions

None. The 2026-08-28 decision assigns held-out GNSS, independent temporal scientific review,
GroundPulse enablement, and publication to EO after #53 closes.

## Coding-agent prompt

```text
Execute md/plans/open-github-issues-full-implementation-2026-08-24.md from refreshed origin/main.

Complete and merge every task through #57, #54, and #53 closure. Start every scientific change
with the named red contract. Preserve the exact gauge, fixed-cube dates/masks/reference, sourced
LOS geometry, and legacy conditional-output labels. Do not add placeholders or independent-
marginal inference fallbacks. Run the full frozen #54 approximation/resource matrix and #53
synthetic/resource/identity validation; failed cells remain visible and trigger implementation
fixes. Tolerance changes and deferral are prohibited.

Use clean isolated worktrees and non-overlapping agent file ownership. Merge PRs sequentially only
after exact-head CI, then require combined-tree CI. Finish with zero open dolphinRust issues and no
open implementation PRs. Do not acquire or score held-out GNSS, perform temporal independent
scientific review, enable GroundPulse, publish, release, pin or modify EO, deploy, mutate serving
state, or perform a Fresno datum anchor.
```

## Coverage audit

| Intake IDs | Scheduled tasks |
|---|---|
| GH-057-01, GH-057-02, GH-057-03, GH-057-04, GH-057-05 | F57-01 through F57-05 |
| GH-054-01, GH-054-02, GH-054-03 | F54-01 through F54-04 |
| GH-054-04, GH-054-05 | F54-04 through F54-06 |
| GH-054-06 | F54-07 |
| GH-054-07 | F54-08 and F54-09 |
| GH-053-01, GH-053-02 | F53-01 through F53-03 |
| GH-053-03, GH-053-04 | F53-01, F53-02, F53-04, and F53-05 |
| GH-053-05 | Deferred to [EO #505](https://github.com/morton-analytics-llc/eo/issues/505); re-enters after F53-09 with matching synthetic/resource/identity receipts |
| GH-053-06 | F53-03 and F53-07 |
| GH-053-07 | F53-07 through F53-09 |
| GH-053-08 | Deferred to [EO #505](https://github.com/morton-analytics-llc/eo/issues/505); re-enters after the held-out GNSS result is complete and identity-matched |

Every intake ID has an explicit disposition. GH-053-05 and GH-053-08 are deferred to EO; all other
IDs are scheduled. #54's spatial review remains F54-08 and is unchanged.
