# Plan: land #54 and #53, then serve the GNSS demo locally

**Date:** 2026-08-24

**Status:** plan complete. Implementation, GitHub mutation, merge, release, external-data acquisition,
and database mutation have not started.

**Scope:** dolphinRust #52 -> #54 -> #53, the fixed-cube producer contract, EO GNSS station A/B,
asset-relative GNSS direction, and one local Fresno demo. Production deploy and production Fresno
backfill are excluded.

## End state

The work is complete when:

1. PR #56 / issue #52 is merged and post-merge `main` CI is green.
2. #54's bounded target/reference covariance implementation is independently reviewed, merged, and
   green on `main`.
3. #53's calibrated temporal velocity/uncertainty method passes its frozen simulation and untouched
   holdout gates, is independently reviewed, merged, and green on `main`.
4. dolphinRust emits a reproducible fixed-cube bundle containing the exact common epoch set,
   estimator, masks, sourced signed LOS geometry, burst/reference lineage, corrected velocity, and
   corrected uncertainty.
5. EO consumes that exact local bundle, projects full GNSS ENU into LOS, uses station A only for the
   datum offset, uses a distinct station B for the held-out result, and persists the evidence in a
   disposable local database.
6. EO serves the result locally at `http://localhost:3000` with the fixed-cube, station A/B,
   uncertainty, and station-to-asset direction visible in the UI.
7. The local receipt, SQL state, API payload, raster hashes, screenshots, and recording all identify
   the same run.

## Scientific and ownership rules

- dolphinRust owns the fixed-cube raster, masks, sourced LOS, burst/reference metadata, #54 spatial
  covariance, and #53 corrected temporal velocity/uncertainty.
- EO owns GNSS source/versioning, station selection, ENU-to-LOS projection, COG sampling, station-A
  datum tie, station-B held-out validation, persistence, local serving, UI, and asset-relative GNSS
  direction.
- Station A sets a datum. It never validates the field.
- Station B is selected without seeing residuals and is the only production-run GNSS field check.
- #53's untouched scientific calibration cohort is distinct from Fresno station B. Both must pass
  their own contracts; neither substitutes for the other.
- No nominal incidence angle is accepted. The fixed-cube science path requires sourced signed
  ground-to-sensor LOS geometry.
- No marginal RSS, scalar effective-N rescale, default sigma, or zero uncertainty can authorize a
  station-B result.
- Toward/away describes GNSS ground motion at the station relative to the nearest versioned asset.
  It does not state that the asset moved.

## Critical path

```text
#52 / PR #56
  -> fixed common epoch + burst/reference contract
      -> #54 target/reference covariance
          -> #53 calibrated temporal velocity and sigma
              -> local dolphinRust fixed-cube bundle
                  -> EO full-ENU station A/B receipt
                      -> local DB/API/UI serving
                          -> recorded Fresno demo

#57 -> EO #483 is independent and does not block this demo.
```

## Worktree and agent layout

Both repository roots are dirty. Preserve them.

1. Refresh remote refs and create clean worktrees from the intended base.
2. Give every agent an exact non-overlapping file manifest.
3. Use one dolphinRust integration worktree for sequential #52/#54/#53 merges.
4. Use one EO backend worktree and one EO frontend worktree; merge backend before frontend.
5. Reject empty diffs and out-of-scope edits. Merge worktrees sequentially.

Assignments:

- **Agent A — #54:** phaselink/source-DAG influence, timeseries composition, workflow/persistence,
  analytic/resource contracts. No #53 or EO files.
- **Agent B — #53:** preregistration, temporal estimator/comparator, simulation driver, corrected
  products, provenance. It begins red contracts against the frozen #54 interface and rebases only
  after #54 merges.
- **Agent C — EO:** fixed-cube consumer, GNSS parser/selection/projection, local persistence/API/UI.
  It develops against the versioned producer schema and does not alter dolphinRust.
- **Root:** #52 disposition, interface freeze, integration, independent-review reconciliation,
  sequential merges, local Fresno evidence run, serving, and demo capture.

## Phase 0 — land #52 and freeze the actual source-DAG interface

1. Reconcile PR #56 with current `origin/main`; preserve `sequential_source_dag_v1`.
2. Resolve the review-receipt wording without reopening the rejected `sequential_srif_v1` design.
3. Run focused global covariance/replay contracts and the full `verify` job on the updated head.
4. Ryan manually merges PR #56.
5. Require post-merge `main` CI green and issue #52 closed.
6. Update the #54 design/task manifest to the merged source-DAG API. Remove stale F/L/w assumptions
   that refer to the rejected representation.

Exit gate: #52 is merged, current `main` is green, and #54's source/replay interface is frozen.

## Phase 1 — finalize the fixed-cube producer contract

### FC-1 exact temporal axis

Add red contracts, then implementation, for:

- unique, strictly increasing absolute acquisition dates;
- identical ordered dates across every stitched burst, not only equal counts;
- deterministic canonical ordering under input permutation;
- acquisition-0 temporal gauge;
- every valid velocity pixel finite at every declared post-gauge common epoch.

The existing estimator remains `linear_post_gauge_unit_precision` until #53 adds a separate corrected
product. Do not silently relabel legacy `velocity.tif`.

### FC-2 masks and signed LOS

- Emit `velocity_validity_mask.tif` and make velocity nodata agree exactly with it.
- Emit masked `los_east.tif`, `los_north.tif`, and `los_up.tif`.
- Record `ENU_ground_to_sensor`, `positive=toward_sensor`, dimensionless units, and unit-vector
  tolerance.
- Require STATIC LOS identity, complete coverage, overlap agreement, and track/burst consistency.
- Reject the scalar 37-degree correction fallback and any nominal 38.5-degree angle for this bundle.

### FC-3 burst/reference receipt

Add `dolphinrust-fixed-cube/1`, exposed both as a typed `DisplacementOutput` field and an on-disk
receipt. It binds:

- absolute dates, decimal days, common-axis digest, temporal gauge;
- estimator/method, units, sign, weights, and finite-support policy;
- authoritative HDF5 burst IDs, deterministic burst order, source granules and hashes;
- EPSG, affine transform, strides, bounds, grid shape;
- final spatial-reference pixel/projected coordinate, selection method, source burst, bounded
  re-reference parent, and exact-zero assertion;
- velocity, validity mask, LOS component, config, and input artifact hashes.

Primary files:

- `crates/dolphin-workflows/src/dates.rs`
- `crates/dolphin-workflows/src/displacement.rs`
- `crates/dolphin-workflows/src/provenance.rs`
- new `crates/dolphin-workflows/src/velocity_provenance.rs`
- `crates/dolphin-workflows/tests/{multiburst_contract,displacement_contract,geometry_provenance_contract,nrt_displacement_contract}.rs`

Exit gate: fresh/NRT/whole/tiled/bounded runs with the same sealed cube reproduce the same receipt;
any date, mask, reference, burst, grid, estimator, or LOS change invalidates reuse.

## Phase 2 — implement, review, and merge #54

### T54-1 design and disclosure

- Re-freeze `reference_specific_influence_v1` against `sequential_source_dag_v1`.
- Preserve exact quantity `C_pp + C_rr - C_pr - C_rp`, signed shared-source influence, fixed realized
  support/branch, L2 inversion scope, correction/reference order, unit transform, and fail-closed
  unsupported states.
- Keep all legacy marginal products labeled
  `SPATIAL_COVARIANCE=target_reference_covariance_not_modeled` and inference-blocked.

### T54-2 red analytic and geometry contracts

- Independent, positive, negative, and coincident shared-source fixtures.
- Hand scalar/matrix covariance and PSD/rank/nullity checks.
- Finite-difference score/Hessian and gauge-congruence fixtures.
- Exact L2 temporal-map fixture with missing observations.
- Whole/tiled/NRT/bounded reference and correction-order fixtures.
- Corrupt provenance, scope mismatch, multiburst ambiguity, branch/support changes, and invalid
  reference failures.

### T54-3 source-DAG influence kernel

- Replay source-keyed coefficients only for one selected reference and a byte-capped target
  microbatch.
- Intersect signed source keys, contract target/reference influence, and discard target scratch.
- Never materialize an all-pixel source influence or date-squared spatial cube.
- Return exact zero for coincident target/reference without jitter.

### T54-4 compose through L2 and final reference

- Compose the phaselink influence through the exact fixed-weight L2 map.
- Resolve final whole/bounded reference before contrast finalization.
- Preserve source-burst ownership and reject unsupported stitched references.
- Apply deterministic corrections to target and reference before subtraction while labeling their
  uncertainty unmodeled.

### T54-5 persist corrected spatial covariance

Write `referenced_displacement_covariance_factor.h5` plus JSON provenance containing rank, date map,
reference/grid/burst/gauge/estimator/mask/correction identities, source-DAG parent, approximation
scope, hashes, status, and resource accounting.

### T54-6 validation and resources

- Run the frozen overlap/distance/support approximation grid.
- Record signed error; never normalize disagreement away.
- Prove coefficient/allocation formulas and the declared memory/wall-time bounds.
- Obtain independent scientific review of equations, approximation evidence, and unsupported scope.

### T54-7 merge

- Open one focused PR.
- Require focused contracts, workspace check, strict Clippy, full tests, exact-head CI, and independent
  approval.
- Ryan manually merges.
- Require green post-merge `main` CI and close #54 only after the merged artifacts/status match the
  reviewed contract.

Exit gate: #54 is merged and its corrected difference-covariance API is stable for #53.

## Phase 3 — implement, validate, and merge #53

### T53-1 freeze model and preregistration

- Consume #54's direct target/reference difference factor.
- Freeze irregular dates, heteroskedastic measurement covariance, target/reference cross covariance,
  residual temporal model, parameter bounds, optimization, bootstrap, small-sample inference,
  missingness, and fail-closed statuses.
- Freeze the simulation grid, seeds, attempted/emitted counts, coverage bands, interval scores,
  resource limits, and promotion thresholds before experiments.
- Freeze an untouched external cohort disjoint by burst/orbit/footprint/site/stations. Fresno,
  MMX1/ICMX, and prior five-burst evidence cannot enter this cohort.

### T53-2 red estimator/comparator contracts

- Hand GLS/REML/profile/bootstrap calculations on irregular heteroskedastic dates.
- Red fixture proving scalar effective-N and marginal RSS give the wrong slope variance.
- Parameter-boundary, rank, cadence, missing-date, bootstrap-failure, and scope mismatch failures.
- Exact #54 factor identity and unit/gauge/reference contracts.

### T53-3 corrected estimator kernel

- Implement the selected covariance-parameter estimator and small-sample interval method.
- Keep point estimate and corrected sigma paired under one method identity.
- Emit new products only: `velocity_temporal_gls.tif` and `velocity_sigma_corrected.tif`.
- Do not relabel or overwrite legacy `velocity.tif` or `velocity_sigma.tif`.

### T53-4 frozen synthetic experiment

- Run every preregistered fixed-factor and end-to-end raw-complex-look cell.
- Report conditional/unconditional 68/90/95 coverage, interval score, width, convergence, failures,
  #54 approximation stratum, resources, and comparators.
- Preserve failures and negative cells. Do not tune after results.

### T53-5 untouched field holdout

- Freeze acquisition/GNSS inputs, crop, station pairs, eligibility, scorer, binary SHA, and manifest
  before outcome access.
- Project per-epoch GNSS covariance into sourced LOS and fit GNSS slopes under the preregistered
  temporal error model.
- Score same-frame InSAR-minus-GNSS station-pair slope differences with combined independent-sensor
  covariance, cluster inference, interval score, and width.
- Keep this cohort distinct from EO's Fresno station A/B demonstration.

### T53-6 independent review and promotion manifest

Independent review must sign:

- model/estimator implementation;
- frozen simulation and holdout manifests;
- coverage and field results;
- supported cadence/reference/estimator scope;
- product/status/provenance schema;
- explicit negative/no-go result if any promotion threshold fails.

### T53-7 merge

- Open the promotion PR only after the signed gates pass. If a gate fails, land the containment,
  evidence, and blocked status without advertising calibrated inference; keep the issue open.
- Require focused contracts, simulation schema checks, workspace check, strict Clippy, full tests,
  exact-head CI, and independent approval.
- Ryan manually merges the scientifically signed PR.
- Require green post-merge `main` CI and close #53 only when the merged method is enabled strictly
  inside its signed scope.

Exit gate: #53 is merged with corrected velocity/sigma products and a scope-matched promotion
manifest, or the signed no-go is merged with inference still blocked. Local EO must reflect whichever
result actually occurred.

## Phase 4 — EO GNSS integration in a disposable local environment

### EO-1 consume exact Dolphin products

- Point EO's dolphinRust dependency at the combined local commit containing #52/#54/#53 and the
  fixed-cube receipt.
- Persist the receipt and hashes with the local run manifest.
- Write the served local COG from `velocity_temporal_gls.tif`; bind
  `velocity_sigma_corrected.tif`, validity mask, LOS components, reference, and date digest.
- Reject legacy or scope-mismatched products for GNSS validation.

### EO-2 parse and version full GNSS

Extend `crates/gp-ingest/src/gnss.rs` to retain:

- east/north/up rates;
- component covariance or available component sigmas;
- station coordinates and reference frame;
- solution start/end and temporal eligibility;
- catalog URL, SHA-256, ETag, Last-Modified, and row count.

Typed transport/HTTP/body/catalog failures remain operational failures, not `no reference`.

### EO-3 select station A/B without leakage

- Search the exact finite COG, not the AOI bbox.
- Apply solution-span, extrapolation, finite-mask, uncertainty, and geometry eligibility first.
- Select A nearest AOI center with stable station-ID tie-break.
- Select distinct B by the preregistered separation rule with stable tie-break.
- Freeze A/B membership before sampling outcome values.

### EO-4 project ENU and compute the independent result

For station `s`:

```text
g_los,s = u_s' g_s
Var(g_los,s) = u_s' C_gnss,s u_s
o_A = d_A - g_los,A
r_B = (d_B - o_A) - g_los,B
```

- Sample `u_s` from the manifest-bound signed LOS rasters.
- Use #54/#53 target/reference covariance for the Dolphin A/B term.
- Use GNSS cross-station/frame covariance when available; any bounded independence assumption must be
  recorded by method version.
- A writes the datum offset. Only B writes the held-out outcome.

### EO-5 local persistence

Add a forward-only local migration for an append-only tenant-scoped GNSS receipt containing catalog,
cube, COG, uncertainty, mask, LOS, stations, selection, samples, covariance, offset, residual,
threshold, outcome, method, generation, and hashes. Add RLS, immutable update/delete rejection,
idempotent receipt identity, and fenced promotion functions. Historical rows remain readable but do
not satisfy the new gate.

### EO-6 station-to-asset direction

- Resolve the nearest point on the selected versioned asset geometry.
- Project station horizontal EN motion and 2x2 covariance onto the station-to-asset axis.
- Report `toward`, `away`, or `indeterminate`; require the interval to exclude zero for a directional
  label.
- Store/display the geometry version and nearest-point method.
- Label it `GNSS ground motion at station relative to asset`.

### EO-7 API and UI

Expose one consistent response containing:

- fixed-cube/run identity, date range/count, estimator, mask, LOS, burst, reference;
- #54/#53 method and promotion status;
- station A datum offset and provenance;
- station B residual, interval, outcome, and provenance;
- station-to-asset direction and its uncertainty;
- reference frame and local serving status.

Update admin validation, displacement/assessment routes, reports/exports/chat, serving banner, Fresno
map, and asset detail together. Never call A validation or direction asset motion.

## Phase 5 — local serving and demo

### Local runtime

Use only the disposable local EO database/object store:

```text
docker compose up -d postgres redis minio
export DATABASE_URL=postgresql://groundpulse:groundpulse@localhost:5432/groundpulse
sqlx database create
sqlx migrate run --source migrations
cd frontend && pnpm install --frozen-lockfile && pnpm run build && cd ..
cargo run
```

Serve the built frontend and API together at `http://localhost:3000`. No production deployment or
production database write is part of this plan.

### Local Fresno run

1. Freeze and hash the exact real Fresno CSLC/STATIC/mask inputs.
2. Run the combined local dolphinRust producer once.
3. Validate fixed-cube, #54, and #53 product/receipt hashes before EO reads them.
4. Import only those immutable local artifacts into the disposable EO store.
5. Freeze and version the GNSS catalog; select A/B before residual access.
6. Run local validation and persist the receipt.
7. Materialize local COG, PMTiles, observations, API state, and UI state from the same run.
8. Preserve pass, fail, or not-evaluable exactly. Do not retune after B.

### Demo script

1. Open Fresno and select one versioned pipeline asset.
2. Show the fixed cube: exact dates, estimator, masks, sourced LOS, burst, and reference.
3. Show #54 spatial covariance and #53 corrected temporal sigma provenance.
4. Show station A: `A sets the LOS datum; it does not validate the field.`
5. Show station B: residual, interval, and actual held-out outcome.
6. Toggle GNSS station vectors and show toward/away/indeterminate relative to the asset.
7. Open provenance and show matching cube, COG, uncertainty, GNSS catalog, station-selection, and
   receipt hashes.
8. Show the local serving status. If #53 landed as a no-go or B is not evaluable/failed, keep asset
   risk withheld and show the reason.

## Verification and merge gates

### dolphinRust focused and full checks

```text
cargo fmt --all -- --check
cargo test -p dolphin-phaselink --test spatial_reference_covariance_contract
cargo test -p dolphin-timeseries --test spatial_reference_covariance_contract
cargo test -p dolphin-workflows --test spatial_reference_covariance_contract
cargo test -p dolphin-workflows --test multiburst_contract
cargo test -p dolphin-workflows --test geometry_provenance_contract
cargo test -p dolphin-timeseries --test temporal_covariance_contract
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
git diff --check
```

### EO checks

```text
cargo fmt --all -- --check
cargo test -p gp-ingest gnss
cargo test -p gp-tasks dolphin_gnss
cargo test -p gp-db --test correlated_dolphin_serving
cargo test -p gp-api dolphin_validation
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
pnpm --dir frontend test -- --run
pnpm --dir frontend run build
bash scripts/demo-validate.sh --base http://localhost:3000 --skip-s3
```

### Required evidence before each manual merge

- red contract captured before implementation;
- focused tests green;
- full checks green;
- exact-head GitHub CI green;
- independent scientific review signed for #54 and #53;
- PR head SHA unchanged after review;
- post-merge `main` CI green before the dependent branch is rebased.

No merge is performed by the planning step. Ryan retains the manual merge gate.

## Demo acceptance

The local demo is ready only when:

- #52, #54, and #53 have their actual merged/status outcomes reflected in the local build;
- all bursts share one exact ordered epoch set;
- velocity, corrected sigma, mask, and signed LOS are aligned and hash-bound;
- station A/B selection is frozen, distinct, deterministic, and outcome-blind;
- full ENU-to-LOS and covariance fixtures pass;
- the local DB, COG, PMTiles, API, and UI identify the same run and nonzero observations;
- the UI distinguishes datum anchor, held-out validation, and station-to-asset direction;
- pass/fail/not-evaluable is shown exactly; no failed gate is hidden;
- screenshots and recording contain no secrets or unrelated tenant data.

## Coverage

Scheduled here: #52 merge, fixed-cube axis/mask/LOS/reference contract, #54 implementation/review/
merge, #53 implementation/simulation/untouched holdout/review/merge, EO full-ENU parser, A/B
selection, corrected projection/uncertainty, local persistence, API/UI, asset-relative station
direction, real Fresno local run, and demo capture. #57/EO-483 remain independent. Production
release/pin/deploy and production Fresno backfill remain in the full follow-on plan.
