# Implementation plan: fixed-cube dolphinRust output and EO GNSS anchor

**Status:** full post-demo implementation plan. The same-day execution plan is
`md/plans/demo-today-fixed-cube-eo-gnss-2026-08-24.md`. Implementation, merge, release, deployment,
data fetch, and backfill have not started.

**Intake:** `md/intake/open-issues-2026-08-24.md`.

**Detailed uncertainty contract:** `md/plans/open-issues-uncertainty-2026-08-23.md` remains
authoritative for the #52, #54, and #53 equations, preregistration, resource gates, and independent
review requirements. This document is the canonical cross-repo execution order and ownership map.

## Objective

Finalize a reproducible dolphinRust fixed-cube product, then make EO perform a non-tautological GNSS
datum anchor and held-out field check against that exact product. The program spans multiple coding,
review, merge, release, deploy, and live-data gates.

The final accepted Fresno result must bind one common epoch set, estimator, masks, sourced signed
LOS geometry, burst/reference choice, immutable GNSS station-A/station-B evidence, corrected
uncertainty, persisted datum offset, serving state, and the same COG/PMTiles/database/API/UI lineage.

## Ownership boundary

### dolphinRust owns only producer facts

- A fixed-cube `velocity.tif` from one exact ordered common epoch set.
- Exact estimator, unit, sign, gauge, nodata, and mask identity.
- Sourced run-specific signed LOS geometry: masked east/north/up ground-to-sensor components or an
  equivalent sourced incidence/heading contract. The science bundle must never use a nominal scalar.
- Authoritative burst IDs, ordered source inputs, common dates, grid, reference pixel and projected
  coordinates, reference-selection mode, anchor burst, bounded re-reference, and reproducibility
  hashes.
- #52 global covariance groundwork, #54 spatial-reference covariance, #53 calibrated temporal
  inference, and #57 orbit ephemeris provenance.

dolphinRust does not fetch Fresno GNSS, choose CORS stations, apply a datum offset, mutate EO state,
promote serving, backfill Fresno, or interpret motion relative to an asset.

### EO owns GNSS and customer-visible state

- Catalog acquisition/versioning and station eligibility/selection.
- Full GNSS ENU plus covariance projection into the run's signed LOS.
- Point sampling the exact Dolphin COG and matching uncertainty product.
- Station A datum estimation and distinct station B held-out validation.
- Immutable offset/provenance persistence, generation fencing, serving gates, API/report/UI
  disclosure, asset-relative station direction, and eventual Fresno replay/backfill.

## Current defects to close

### dolphinRust

- Multi-burst stitching checks equal date counts but not identical ordered dates, then uses the first
  burst's dates for the frame (`crates/dolphin-workflows/src/displacement.rs`). Equal-count,
  different-date bursts can therefore produce a falsely comparable velocity cube.
- The existing `LinearPostGaugeUnitPrecision` estimator is stable and identified, but the output does
  not bind it to absolute dates, one common per-pixel epoch-support policy, or the final mask.
- The final validity mask is applied in memory but no mask raster is emitted and `velocity.tif` has
  no explicit nodata/mask identity.
- Signed ground-to-sensor LOS east/north/up exists in memory and is covered by source provenance, but
  no signed LOS rasters are emitted. Geometry is optional; the corrections path can use the scalar
  `incidence_angle_deg` default of 37 degrees. That fallback is forbidden for this science bundle.
- Absolute dates, normalized burst IDs, anchor burst, reference-selection method, projected reference
  coordinates, and bounded re-reference lineage are not bound into one artifact.

### EO

- `gp-ingest` currently parses one vertical MIDAS rate and selects the station nearest the AOI-bbox
  center. It does not persist full ENU rates/covariance or select against the exact finite COG.
- `validate_dolphin_gnss` currently converts Dolphin LOS to vertical with incidence only, compares one
  station at a fixed 1 mm/yr threshold, and uses the same station as the tie. That is a datum transform,
  not independent validation.
- Existing run-state columns preserve only scalar offset/station/time summaries. They do not bind the
  station solution, catalog bytes, exact COG/epoch manifest, signed look vector, samples, uncertainty,
  selection method, station-B residual, or outcome threshold.
- Existing held-out design correctly separates A and B, but its vertical-only projection and
  independence RSS are superseded by full ENU-to-LOS projection plus #54/#53 covariance.

## Dependency order

```text
PR #56 / #52 merged on green main
    -> FC-01..FC-06 fixed-cube contract
        -> dolphinRust release + EO consumer pin
            -> EO-01..EO-08 GNSS input/projection/anchor/receipt (fail closed without uncertainty)

#52 -> fixed axis/reference identity -> #54 spatial covariance -> #53 calibrated temporal inference
                                                    -> EO-07 station-B decision + EO-09 serving gate

#57 orbit provenance -> EO #483 consumer (independent of GNSS)

EO accepted anchor + deployed serving code -> Fresno backfill -> same-run terminal proof
```

EO GNSS ingestion/projection/persistence can be coded before #54/#53 finish, but it must emit
`not_evaluable: calibrated_uncertainty_unavailable` and cannot promote serving until the corrected
uncertainty artifact is present and scope-matched.

## Fixed-cube product contract

Add a versioned `dolphinrust-fixed-cube/1` receipt alongside the science rasters and expose the same
typed receipt on `DisplacementOutput` for EO's GroundPulse output policy.

Required fields:

- ordered absolute acquisition dates and decimal days; unique and strictly increasing;
- digest of the exact common epoch set; every burst must carry the identical ordered vector;
- temporal reference date/index and exact acquisition-0 gauge;
- velocity estimator `linear_post_gauge_unit_precision`, method version, weighting, required finite
  support policy, units `m/yr`, and sign convention;
- normalized authoritative burst IDs, deterministic burst order, ordered source granules and hashes,
  anchor burst, and any burst-leveling/re-reference parent identity;
- output grid dimensions, EPSG, affine transform, bounded processing window, output strides;
- configured/automatic spatial-reference method, final reference row/column, projected coordinate,
  source burst, and exact-zero-at-reference assertion;
- `velocity.tif`, `velocity_validity_mask.tif`, and `los_east/north/up.tif` keys plus byte hashes;
- input layover/shadow and other science-mask identities/hashes, final valid-pixel count, and final
  mask digest;
- signed LOS convention `ENU_ground_to_sensor`, `positive=toward_sensor`, unit-vector tolerance,
  STATIC source files/keys/hashes, coverage, and overlap agreement;
- build/version, normalized config identity, schema/method versions, and parent run identity.

`velocity.tif` is valid only where every declared post-gauge epoch is finite and every configured
science mask passes. The mask raster is the authoritative support; nodata and mask disagreement fail
the bundle. The 37-degree scalar correction fallback and any nominal 38.5-degree angle are never
valid geometry provenance.

## GNSS math and decision contract in EO

For station `s`, use the run-specific signed unit vector
`u_s = [u_e, u_n, u_u]` sampled from the fixed-cube LOS layers and the GNSS rate
`g_s = [g_e, g_n, g_u]`:

```text
g_los,s = u_s' g_s
Var(g_los,s) = u_s' C_gnss,s u_s
o_A = d_A - g_los,A
r_B = (d_B - o_A) - g_los,B
    = (d_B - d_A) - (g_los,B - g_los,A)
```

Signs and units must match the fixed-cube receipt before either scalar is computed. Station A is
selected without inspecting residuals and estimates only `o_A`. A distinct station B is selected by
the preregistered finite-extent/separation/temporal rule and alone supplies the field-check outcome.

The Dolphin term in `Var(r_B)` must come from #54's target/reference covariance propagated through
#53's calibrated slope estimator. Do not replace it with marginal RSS, a scalar effective-N rescale,
or zero. GNSS cross-station/frame covariance must be used when supplied; a diagonal/independence
assumption must be versioned, bounded, and independently reviewed before promotion. Missing or
scope-mismatched uncertainty is `not_evaluable`, never pass.

Keep three claims separate:

- station A: `datum_anchored`;
- station B: `held_out_gnss_point_check_passed|failed|not_evaluable`;
- asset direction: a separate descriptive EO feature, never an anchor or validation oracle.

## Test contract

| ID | Red-to-green contract |
|---|---|
| FC-C01 | Duplicate, non-increasing, or equal-count/different-date bursts fail before phase linking; input permutation produces the same canonical common-axis digest. |
| FC-C02 | A velocity pixel is valid only with finite displacement at every declared post-gauge epoch and every science mask passing. Velocity nodata, explicit mask, and mask digest agree exactly. |
| FC-C03 | The receipt identifies `linear_post_gauge_unit_precision`, acquisition-0 gauge, units, sign, dates, and mask; changing any input changes the receipt identity. |
| FC-C04 | Missing, mixed, misaligned, non-unit, sign-inconsistent, or incomplete STATIC LOS fails the science bundle. Scalar incidence can never satisfy the contract. |
| FC-C05 | Configured/automatic/bounded references persist exact row/column, projected coordinate, source burst, selection method, and zero-at-reference result. |
| FC-C06 | Fresh, NRT, whole, tiled, and bounded runs with the same sealed cube produce identical receipt fields; a changed date, mask, reference, burst, grid, or estimator fails reuse. |
| EO-C01 | Catalog parsing retains version/source receipt, station coordinates, ENU rates, component covariance/uncertainty, solution interval, and frame; malformed or incomplete data is ineligible or operational failure by type. |
| EO-C02 | Station A/B selection uses only preregistered geometry/quality/time rules on the exact finite COG; no residual enters selection; A and B are distinct and deterministically tie-broken. |
| EO-C03 | Hand ENU/look-vector fixtures match `u'g` and `u'Cu`; reversing LOS sign reverses the projected rate; vertical-only and nominal-angle implementations fail. |
| EO-C04 | Point samples resolve only the exact run/Cube/COG/uncertainty hashes. Nodata, mask, grid, epoch, reference, or run mismatch returns `not_evaluable` or a typed operational failure, never another artifact. |
| EO-C05 | Station A fixes `o_A`; changing station-B truth cannot change the offset. Station B residual uses the already-fixed offset and cannot pass when B is absent or reused as A. |
| EO-C06 | Analytic covariance fixtures prove the station-B interval uses target/reference cross covariance and calibrated temporal inference. Marginal RSS and scalar effective-N comparators remain failing controls. |
| EO-C07 | The append-only receipt round-trips every input/output identity and outcome. Retries with the same generation are idempotent; conflicting content fails closed. |
| EO-C08 | Datum-only, missing-uncertainty, insufficient-separation, stale-solution, and no-B states remain provisional/not evaluable. Only the current-generation held-out pass can enable the named serving transition. |
| EO-C09 | API, report, export, map, asset detail, alerts, and chat use one persisted reference/validation state and never call station A validation. |
| EO-C10 | A GNSS station-direction label is `toward`, `away`, or `indeterminate` from horizontal EN projection onto the station-to-nearest-versioned-asset axis; it says ground motion at the station, not asset motion. |
| EO-C11 | A Fresno replay proves the same fixed-cube manifest/COG through validation, DB, COG, PMTiles, serving, API, and authenticated UI, with nonzero observations and all hashes correlated. |

## Task manifest: dolphinRust

Each coding task is a separate branch/worktree session capped at 45 minutes. Stop with a focused
commit and green focused tests; run full workspace verification before each PR.

### M0 — close #52

- **T45-01..04:** execute `md/plans/open-issues-under-45-minutes-2026-08-24.md` through updated-head
  CI, Ryan's manual merge, and green post-merge `main` CI.

### M1 — fixed common temporal axis

- **FC-01, 45m:** write red date and multiburst contracts for duplicate/non-increasing dates and
  equal-count/different-date bursts; implement only the fail-closed ordered-axis guards.
- **FC-02, 45m:** bind the canonical absolute date vector/digest, acquisition-0 gauge, existing
  estimator ID, units, and sign into a typed fixed-cube receipt and `velocity.tif` metadata.
- **FC-03, 45m:** emit `velocity_validity_mask.tif`, declare nodata, require full common-epoch support,
  and prove mask propagation/identity across seams and bounded output.

### M2 — sourced LOS and stable reference lineage

- **FC-04, 45m:** make STATIC LOS mandatory only for the fixed-cube science bundle; emit masked signed
  LOS component rasters and red/green sign, norm, coverage, identity, and no-fallback contracts.
- **FC-05, 45m:** persist authoritative HDF5 burst IDs, ordered source inputs, temporal axis, grid,
  spatial reference coordinates/method/source burst, and bounded re-reference lineage.
- **FC-06, 45m:** prove fresh/NRT/whole/tiled/bounded identity, JSON and raster round-trip, corrupt
  receipt rejection, resource bounds, docs, and full verification; open one unmerged PR.

### M3 — uncertainty issues

- **DR-054:** execute existing T54-01..T54-07 after #52 and fixed axis/reference identity are on green
  `main`. Preserve the bounded reference-specific influence design and independent review gate.
- **DR-053:** execute existing T53-01..T53-07 after #54. EO supplies the untouched station-A/B field
  cohort and scorer inputs; dolphinRust supplies only the estimator and uncertainty producer.
- Split each T54/T53 coding step into a red-contract, minimal implementation, and focused-verification
  session when it cannot complete in 45 minutes. Research experiment and independent review time are
  not coding-session time and cannot be compressed into the cap.

### M4 — independent orbit provenance

- **T57-01..05:** implement #57 from `md/plans/open-issues-under-45-minutes-2026-08-24.md` after Ryan
  applies `backlog-ready`. EO #483 must accept the new schema before release/pin.

## Target change and touched EO surfaces

### Target change

Replace the current single-station vertical tie with a current-generation, full-ENU, exact-artifact
anchor/holdout pipeline while preserving raw observations and existing historical rows.

### Touched surfaces

- `crates/gp-ingest/src/gnss.rs`: catalog schema, versioned fetch receipt, full ENU/covariance parse.
- `crates/gp-tasks/src/tasks/dolphin_validate.rs`: finite-COG station selection, LOS projection,
  station-A offset, held-out B, uncertainty/outcome, and immutable receipt write.
- `crates/gp-dolphin/src/lib.rs`: consume/persist the fixed-cube receipt and bind the exact COG,
  masks, run manifest, burst leveling, and geometry products.
- `crates/gp-db/src/repositories/dolphin.rs`, `displacement_serving.rs`, pipeline-control functions,
  and DB integration tests: append/read receipt and fenced promotion.
- Forward-only migration: new append-only GNSS evidence receipt plus current-generation constraints;
  keep existing run-state summary columns for serving joins.
- `crates/gp-api/src/routes/admin.rs`, displacement/assessment routes, reports/exports/chat: expose one
  consistent evidence state and method/provenance.
- Frontend serving banner, run provenance, map/asset detail: datum-anchor vs held-out status, station
  IDs, evidence scope, and later station-to-asset direction.
- Scheduler/backfill/verifier: enqueue current-generation validation only after exact fixed-cube
  publication; Fresno production acceptance after deploy.

## Migration phases

### MIG-01 — additive schema, 45m

Create an append-only, tenant-scoped `dolphin_gnss_validation_receipts` table keyed to the validation
generation/result and Dolphin run. Store a versioned JSON evidence document plus generated/queryable
identity fields: catalog hash, cube-manifest hash, COG hash, uncertainty hash, selection/projection
method versions, station A/B IDs, outcome, and receipt hash. Add RLS, immutable update/delete trigger,
FKs, uniqueness on `(tenant_id, run_id, validation_generation, method_version)`, bounded lock/statement
timeouts, and `NOT VALID` constraints where existing rows would otherwise be scanned.

Do not widen direct runtime table-write privileges. The fenced pipeline-control function owns the
append and the mirrored run-state transition.

### MIG-02 — dual-write/current generation, 45m

Deploy code that writes the receipt and mirrors only the accepted scalar datum offset/station/time to
existing `dolphin_run_state` columns. Readers prefer a scope-matched current receipt; legacy rows stay
readable but cannot satisfy the new held-out gate. Conflicting retry content fails without mutation.

### MIG-03 — validate constraints and switch gates, 45m

After query-based validation, validate constraints and make current-generation serving predicates
require the receipt, exact fixed-cube identities, and held-out outcome. No destructive rollback or
historical rewrite; containment is disabling enqueue/promotion and continuing to serve the previous
accepted generation.

## Task manifest: EO

- **EO-01, 45m:** red catalog fixtures and full ENU/covariance/solution-interval parser; preserve
  typed transport/HTTP/body/catalog failures and immutable catalog receipt.
- **EO-02, 45m:** red deterministic station-selection contracts over the exact finite COG; implement
  eligibility, tie-breaks, A/B separation, and temporal gates without reading outcomes.
- **EO-03, 45m:** stage/sample the exact fixed-cube mask, velocity, LOS components, and uncertainty by
  manifest hash; reject mismatched grids, epochs, masks, references, and run identities.
- **EO-04, 45m:** red analytic ENU-to-LOS and covariance fixtures; implement the signed projection and
  provenance. Delete the validation path's vertical-only incidence conversion; retain no nominal-angle
  fallback.
- **EO-05, 45m:** implement station-A datum offset with generation fencing and datum-only outcome.
- **EO-06, 45m:** implement station-B residual independently of selection and station A; add
  `not_evaluable` reasons for no B, insufficient separation, temporal mismatch, nodata, and missing
  uncertainty.
- **EO-07, 45m:** consume #54/#53 covariance products, run analytic comparators, and enable the signed
  held-out decision only for a scope-matched calibrated method.
- **EO-08, 45m:** integrate MIG-01..03 repository/control functions, idempotent receipt persistence,
  admin API, and audit events.
- **EO-09, 45m:** update serving repositories and promotion checks; raw observation rows and COG stay
  immutable, offset application stays correlated to the exact run/reference frame.
- **EO-10, 45m:** update partner API, reports, exports, alerts, and chat to use the same persisted
  anchor/holdout state and fail closed on legacy/missing evidence.
- **EO-11, 45m:** paired frontend work: show `Datum anchored at A`, `Held-out GNSS point check at B`,
  `not evaluable`, method/version, and provenance links; never label station A as validation.
- **EO-12, 45m:** add a pure station-to-nearest-versioned-asset horizontal projection with covariance;
  classify toward/away only when the interval excludes zero, else indeterminate.
- **EO-13, 45m:** paired API/map/asset-detail display for the station vector and direction label. Copy
  must say `GNSS ground motion at station relative to asset`; it cannot imply the asset moved.
- **EO-14, 45m:** dry-run Fresno preflight without mutation: exact cube/geometry/masks, eligible A/B,
  solution intervals, uncertainty, auth, worker resources, and deploy state.
- **EO-15:** after explicit live approval, enqueue one normal publishable Fresno backfill; do not reuse
  or relabel a prior provisional cube.
- **EO-16:** run the same-run terminal verifier and authenticated UI walkthrough; require correlated
  COG, PMTiles, DB observations, GNSS receipt, serving state, API payload, frontend labels, nonzero
  observations, hashes, and deployment identity.

EO-15 compute and EO-16 wait/runtime may exceed 45 minutes; their Codex mutation/checkpoint actions
remain bounded, but external processing time is not represented as coding time.

## Backfill and validation

- No historical single-station tie is upgraded, relabeled, or used as station-B evidence.
- Backfill only the selected Fresno run after the producer release, EO pin, migrations, code deploy,
  and #54/#53 promotion gates are all verified.
- Preflight exact station eligibility from versioned inputs; do not acquire or inspect GNSS outcomes
  during station-selection implementation tests.
- One failed or not-evaluable Fresno result stays as evidence. Do not tune selection, thresholds,
  masks, or estimator after observing station B.
- The final verifier must start from the terminal run identity and prove every downstream artifact
  came from that same run. Health/deploy success alone is insufficient.

## Risks and rollback containment

- **Epoch mismatch:** fail before processing; never coerce by count or silently intersect after fit.
- **Geometry fallback:** fail the science bundle when sourced LOS is absent; do not use 37 or 38.5
  degrees.
- **Self-validation:** station A cannot update field-validation status.
- **Uncertainty gap:** anchor code may land behind `not_evaluable`; promotion stays disabled until
  #54/#53 scope-matched uncertainty exists.
- **Receipt conflict:** append-only identity collision fails; no overwrite/update path.
- **Runtime rollback:** stop enqueue/promotion, retain prior accepted serving generation, and deploy a
  forward fix. Do not drop columns/tables or rewrite evidence.
- **Asset-direction overclaim:** the direction feature describes station ground motion only and is
  independent of the datum/held-out oracle.

## Verification

### dolphinRust

```text
cargo fmt --all -- --check
cargo test -p dolphin-workflows --test multiburst_contract
cargo test -p dolphin-workflows --test displacement_contract
cargo test -p dolphin-workflows --test geometry_provenance_contract
cargo test -p dolphin-workflows --test nrt_displacement_contract
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

### EO

```text
cargo fmt --all -- --check
cargo test -p gp-ingest gnss
cargo test -p gp-tasks dolphin_gnss
cargo test -p gp-db --test correlated_dolphin_serving
cargo test -p gp-api dolphin_validation
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
npm --prefix frontend test -- --run
npm --prefix frontend run build
```

DB integration tests require the repository's normal PostgreSQL/SQLx environment. Live Fresno
verification additionally requires current AWS authentication and explicit authorization for the
backfill mutation.

## Coding-agent prompts

### Immediate 45-minute slice: FC-01

```text
Implement only FC-01 from md/plans/fixed-cube-eo-gnss-implementation-2026-08-24.md in a clean
dolphinRust worktree based on green post-#52 main.

Start red. Add contracts proving that duplicate or non-increasing dates and two bursts with the same
date count but different ordered dates fail before phase linking/stitching. Canonical input
permutation must resolve to the same ordered axis. Then add the smallest validation guards. Do not
add GNSS, output rasters, provenance schema, covariance, or unrelated refactors in this slice.

Run the focused date/multiburst/displacement tests, cargo fmt --check, cargo check --workspace, strict
Clippy, and git diff --check. Commit with the required co-author trailer, open an unmerged PR, wait
for CI, and stop for Ryan's manual merge.
```

### EO anchor sequence

```text
Implement one EO-XX task per session from
md/plans/fixed-cube-eo-gnss-implementation-2026-08-24.md. Work from the released/pinned
dolphinrust-fixed-cube/1 consumer contract. Preserve raw observations and prior evidence.

Every analytic change starts with its named red contract. Station selection cannot inspect outcomes;
station A estimates only the datum; station B alone evaluates the field. Use full ENU and the sourced
signed run-specific LOS. Require exact cube/COG/mask/reference/uncertainty identities and fail closed
on mismatch. Do not promote serving without current-generation #54/#53-calibrated uncertainty.

For persistence, use forward-only append-only tenant-scoped evidence and fenced pipeline-control
writes. Pair backend changes with API/report/UI disclosure. Run focused tests, workspace check,
strict Clippy, DB integration where configured, frontend tests/build when touched, then open an
unmerged PR and stop at the manual merge gate. Do not deploy or backfill without separate approval.
```

## Open questions

None for FC-01 or the ownership boundary. Live credentials, station availability, and production
backfill authorization are execution gates, not design questions.

## Coverage audit

| Intake ID | Scheduled tasks |
|---|---|
| DR-052-MERGE | T45-01..04 |
| DR-FC-AXIS | FC-01..02 |
| DR-FC-MASK | FC-03 |
| DR-FC-LOS | FC-04 |
| DR-FC-REF | FC-05..06 |
| DR-054-SPATIAL | existing T54-01..07 |
| DR-053-TEMPORAL | existing T53-01..07 plus EO-07 evidence consumer |
| DR-057-ORBIT | T57-01..05; EO-483 consumer |
| EO-GNSS-INPUT | EO-01..03 |
| EO-GNSS-PROJ | EO-04..05 |
| EO-GNSS-ANCHOR | EO-05..07 |
| EO-GNSS-PERSIST | MIG-01..03, EO-08 |
| EO-GNSS-SERVE | EO-09..11 |
| EO-GNSS-DIRECTION | EO-12..13 |
| EO-FRESNO-BACKFILL | EO-14..16 |
| EO-487-SCIENCE | DR-054, DR-053, EO-07..16 |
| EO-483-PROVENANCE | T57-04..05 plus EO schema consumer before pin |
| EO-475-WIRING | Out of scope: existing EO correction/config issues, not fixed-cube/GNSS work |

No intake item is silently dropped. dolphinRust has no UI. EO backend results have paired customer
disclosure tasks, and the Fresno backfill remains gated by explicit live authorization.
