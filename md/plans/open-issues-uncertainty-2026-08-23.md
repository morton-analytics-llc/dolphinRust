# Implementation plan: open uncertainty issues #52 through #54

**Status:** planning complete; implementation not started.

**Intake:** `md/intake/open-issues-uncertainty-2026-08-23.md`.

**Live queue:** exactly three open issues, [#52](https://github.com/morton-analytics-llc/dolphinRust/issues/52),
[#53](https://github.com/morton-analytics-llc/dolphinRust/issues/53), and
[#54](https://github.com/morton-analytics-llc/dolphinRust/issues/54). None has comments, an
assignee, or a milestone. Draft [PR #55](https://github.com/morton-analytics-llc/dolphinRust/pull/55)
implements the v1.6 conditional-uncertainty boundary. CI run 32663312067 is green; the PR remains
draft, mergeable, and unreviewed.
`main` is synchronized with `origin/main` at `be07c2141e0e53ef761991f1ffba29b928d8296b`;
main CI run 32602108406 is green.

**Execution boundary:** planning only. No implementation, issue mutation, PR, merge, release,
publication, GroundPulse submodule bump, external-data acquisition, or deployment is authorized.

## Outcome and order

Reconcile the shared containment first, then execute #52 -> #54 -> #53:

1. Treat draft PR #55 as an in-flight prerequisite. Verify its conditional-IID output,
   diagnostic-only scalar correlation, covariance-omission tags, and release boundary after its
   CI/review state settles; do not duplicate or silently overwrite its producer contract.
2. Resolve #52 with a bounded, acquisition-0-gauge temporal factor. Retain the existing
   per-ministack CRLB rasters as non-inferential diagnostics, and preserve the exact replay
   lineage/support #54 needs after the phase-link kernel discards local state.
3. Resolve #54 with a byte-capped, reference-specific target-minus-reference covariance pass over
   that replay path. The current independent-marginal product remains explicitly uncalibrated.
4. Resolve #53 by testing an irregular-cadence slope covariance model against frozen synthetic
   and untouched field contracts. Corrected temporal GLS remains unavailable unless #52, #54,
   synthetic coverage, field evidence, resources, and independent review all pass.

#52 and #54 each end with one reviewed, unmerged PR. #53 uses separate research and promotion
PRs. A failed #54 or #53 gate records the no-go and leaves PR #55's conditional output unchanged.
Each arrow is a human merge gate: stop after PR #55, #52, #54, and the #53 research PR; continue
only after its disposition is explicit and any merge is separately authorized, complete, and
verified on refreshed `main`. Do not stack dependent production PRs on unmerged heads.

## Current failure boundaries

### #52

- `crlb_pixel` reduces the local Fisher covariance to `sqrt(diag)` in
  `crates/dolphin-phaselink/src/crlb.rs`; `fused.rs` retains only that vector.
- `sequential.rs` references each ministack to its last carried compressed SLC, discards all
  compressed bands, and concatenates the real-date marginal sigmas. The old/new cross-covariance
  and compressed-reference uncertainty are then unrecoverable.
- `interferogram_precisions` assumes `sigma_i^2 + sigma_j^2`; `date_precisions` assumes diagonal
  date covariance. Acquisition 0's gauge zero is converted to precision `1e12`.
- On current `main`, `timeseries_options.use_coherence_weights` defaults to `true`, so the
  diagnostic changes SBAS and velocity point estimates and also feeds the opt-in uncertainty path.
  PR #55 removes it from the IID-conditional sigma path but retains point-estimation provenance.
- Existing sequential oracle tests can return early when ignored fixtures are absent. They are
  useful parity checks but cannot satisfy the required in-repo analytic contract.

### #53

- The current point fit uses actual acquisition days, but the temporal correction correlates
  adjacent retained residuals without elapsed time, clamps correlation to `[0, 0.98]`, and maps
  underspecified or degenerate series to zero correlation.
- With `correct_velocity_temporal_correlation = true`, `velocity_sigma.tif` is replaced by the
  scalar-inflated result. The raw correlation and factor are not persisted.
- Spatial referencing happens before the velocity fit, while its uncertainty still uses the
  target pixel's original marginal CRLB. Reference-pixel covariance is absent.
- Existing MMX1 and five-burst artifacts have exposed outcomes. They are development data, not a
  new outer holdout. The repository contains no Fresno recipe or receipt.
- The prior five-burst cohort produced only four evaluable sites. Five independent sites are also
  too few to support a nominal 68/90/95 percent coverage claim; field sample size must be powered
  and frozen before acquisition.

### #54

- Phase-link windows share native samples, but `fused.rs` keeps only each pixel's marginal CRLB
  and discards the coherence matrix, estimator branch, and any shared-sample influence.
- Sequential compression creates a dependency cone that grows with ministack depth. Immediate
  target/reference window overlap is therefore insufficient even though each block-local stencil
  is bounded.
- L2 inversion retains a full temporal covariance only inside one pixel solve; its stack output
  keeps the diagonal. No cross-pixel block survives phase linking, compression, inversion, burst
  stitching, or crop/reference changes.
- `reference_variance_to_point` currently emits `Var(p) + Var(r)` for different pixels and exact
  zero at `p = r`; it omits `Cov(p,r)` and cannot be used as total uncertainty.
- A bounded output can choose a new target-local reference and refit after the whole-frame series
  was already referenced. The covariance path currently trims the marginal cube and repeats the
  independent approximation rather than transforming the same final contrast.
- Current `main` does not yet contain the v1.6 `SPATIAL_COVARIANCE` disclosure named by #54. Draft
  PR #55 adds it; metadata still does not satisfy propagation or calibration.

## Shared release-boundary containment

PR #55 owns the initial producer-contract change. Before beginning T52-01, rebase on its reviewed
result or stop if it is closed without replacement. Its tests and diff must prove:

- `write_velocity_uncertainty = true` fits the finite post-gauge corrected and spatially
  referenced series with unit relative precision, and its point estimate and conditional sigma
  remain paired. Enabling it cannot reuse a stitched marginal CRLB as absolute precision.
- Configured deterministic corrections are applied per pixel before the final whole/bounded
  spatial reference, so the paired mean is `(z_p - a_p) - (z_r - a_r)` rather than
  `(z_p - z_r) - a_p`.
- `correct_velocity_temporal_correlation = true` is rejected before I/O. Raw temporal-correlation
  summaries remain diagnostics and never rescale `velocity_sigma.tif`.
- The current `use_coherence_weights` path remains point-estimation quality weighting only; no
  posterior, global covariance, total-uncertainty, or calibration claim follows from it.
- Existing CRLB and displacement-variance products identify their estimator, units, exact gauge,
  target/reference covariance omission, and `INFERENCE_READY=false` in machine-readable output.
- README, usage docs, changelog, public types, and exhaustive config-disposition contracts agree.

If PR #55 changes before merge, rerun this reconciliation against its final SHA. None of #52,
#54, or #53 may weaken those labels while its own promotion gates remain unsigned.

## Issue #52: global sequential covariance

### Selected representation

Use a block state-space, square-root-information representation. For ministack `b`, partition
the local gauge-reduced Fisher information over carried compressed phases `s_b` and new real
phases `x_b`:

```text
s_b = at most K carried compressed phases in acquisition-0 gauge
d_b = A_b [s_b; x_b]                         local contrast map
J_b = A_b' I_b A_b                           local Gaussian factor
Q_b = inverse(J_xx)
F_b = -Q_b J_xs
x_b = F_b s_b + L_b eta_b,  L_b L_b' = Q_b   conditional transition
c_b = w_b' [s_b; x_b]                        compression linearization
```

`K = max_num_compressed`; `x_b` contains the new real-date phase errors. Retain the full local
gauge-reduced Fisher factor. Combine its canonical block with the frontier prior by QR/Cholesky
square-root updates, never by explicitly inverting a whole history. Linearize the existing
compression projection to obtain `w_b`; the equations above are accepted only where `J_xx` is
positive definite and the compression Jacobian is finite.

Persist `F_b`, packed `L_b`, `w_b`, parent IDs, node/date mappings, and temporal status. The stored
transitions preserve cross-ministack temporal covariance implicitly and support selected blocks,
diagonals, and solves without materializing a dense date covariance cube. A per-pixel `L_b` basis
does not identify cross-pixel covariance and #54 may not multiply two such factors.

#52 also persists an exact replay descriptor for #54: the production support-iterator version,
bit-packed realized SHP/mask membership, global native/output grids, window clamp and stride,
estimator branch and solution identity, compression-parent lineage, normalized config, and ordered
source-manifest digest. It does not persist source-influence coefficients for every pixel. #54
replays those coefficients only for one reference and a byte-capped target microbatch, contracts
them immediately to reference-specific blocks, and discards them.

- The compression-to-next-window map follows the actual native-pixel nearest-neighbor upsampling,
  clamped covariance window, mask, and fixed SHP membership. Each block-local support `S_b` is
  bounded, but the composed dependency cone can grow with ministack depth. Do not claim a fixed
  global overlap radius. A `strides > 1` finite-difference contract validates the exact map.
- For block `b`, let `m_b` be new stochastic dates and `k_b <= K` carried states. The exact stored
  floating-point count per valid pixel is
  `m_b*k_b + m_b*(m_b+1)/2 + (m_b+k_b)` for `F_b`, packed `L_b`, and `w_b`, plus fixed-size maps and
  status. Bit-packed replay support adds `ceil(S_b/8)` bytes per pixel/block where required.
  At `m_b=13`, `k_b=10`, and the default `S_b=435`, that is 244 `f64` values plus 55 support
  bytes, or about 125.4 MiB for one 256x256 block before fixed metadata.
- Tile working memory is `O(microtile * (m_b*k_b + m_b^2 + S_b))`; choose `microtile` from a
  declared byte budget. No allocation multiplies `tile_area * S_b * d_b^2`.
- Stored size is linear in area and block count. Sealed parent edges remain sparse and are never
  expanded into full-history coefficients for every pixel.
- Exact gauge: acquisition 0 is removed from the stochastic vector and recorded in metadata.
- NRT: sealed block factors are immutable; only the open block and its descendants are
  recomputed.
- Scope assumptions: local Fisher likelihood blocks are conditionally independent given the
  carried compressed state, and compression uncertainty uses a first-order Jacobian at the linked
  solution. Each assumption is part of the method identifier and producer scope. T52-01 stops with
  a design no-go if the temporal factor or exact replay descriptor cannot meet the declared byte
  formula and resource bound.

Persist the product as a spatially chunked `phase_covariance_factor.h5` plus
`phase_covariance_provenance.json`. HDF5 groups are block-indexed so different trailing-block
sizes do not require padding to `n_dates`. Stream each completed tile/block through one
synchronous writer into a checked scratch artifact, then atomically finalize its manifest; never
retain the whole-frame factor in `TiledOutput` or `DisplacementOutput`. Those types carry only a
tile-scoped factor or finalized artifact reference. `DisplacementOutputPolicy::GroundPulse`
excludes this research product until a separate downstream plan authorizes it.

Method/status contract:

```text
method: sequential_srif_v1
gauge_date_index: 0
factor_status: valid | singular_local_fisher | nonfinite_local_fisher |
               nonfinite_jacobian | masked | no_contributor
replay_status: replayable | source_manifest_missing | source_manifest_mismatch |
               support_not_frozen | unsupported_backend
stitched_factor_status: unsupported_seam_covariance  (multiburst only)
downstream_inference_status: blocked_pending_issue_54_and_53
```

Invalid references, disconnected gauges, unsupported layouts, and multi-ministack
`max_num_compressed = 0` are run-level `Result` errors before allocation. Per-block/output-pixel
status is a `u16` dataset with stable codes for `valid`, `singular_local_fisher`,
`nonfinite_local_fisher`, `nonfinite_jacobian`, `masked`, and `no_contributor`.

Multiburst factors remain grouped per burst before overlap leveling. The seam rotation is estimated
from many overlap pixels, so its uncertainty induces cross-pixel covariance outside this method's
spatial-independence assumption. Do not copy one burst's factor onto the stitched phase. Record
`stitched_factor_status=unsupported_seam_covariance`. Modeling seam covariance is outside #52.
`factor_status=valid` means only that #52's per-burst algebra is evaluable and provides no
downstream calibration evidence. `replay_status=replayable` certifies only that #54 can reproduce
the realized source/support path. Spatial covariance remains blocked.

The sequential estimator follows Ansari, De Zan, and Bamler's
[Efficient Phase Estimation for Interferogram Stacks](https://elib.dlr.de/116285/). That paper
defines the estimator family. The covariance propagation here still requires its own contracts.

### #52 test contract

| ID | Contract | Primary location |
|---|---|---|
| C52-01 | An unconditional committed two-ministack fixture reconstructs `[[0,0,0,0],[0,4,2,2],[0,2,10,3],[0,2,3,17]]` from `Var(x1)=4`, `c1=.5*x0+.5*x1`, and second-block conditional covariance `[[9,2],[2,16]]`. A second rational fixture uses `K >= 2`, correlated carried states, a nontrivial `A_b`, and cap eviction. | new phaselink and workflow `global_covariance_contract.rs` targets plus committed JSON fixtures |
| C52-02 | The full local reduced covariance diagonal reproduces the current v0.42 marginal sigma within the existing oracle tolerance; staged, fused, CPU, and available GPU paths agree. | `crlb.rs`, `quality_v042_contract.rs`, `fused_contract.rs`, phaselink global contract |
| C52-03 | The analytic compression Jacobian agrees with a central finite-difference oracle away from wrap/branch boundaries, including `strides > 1` and a window touching multiple upstream looked cells. Nonfinite or nondifferentiable cases return an invalid status. | `quality.rs` and both global covariance contracts |
| C52-04 | Acquisition 0 is exactly absent from the stochastic factor; reconstructed row/column 0 is exact zero without epsilon variance or extreme precision. | global covariance contract |
| C52-05 | Singular or nonfinite local information invalidates the affected pixel/block; no partial factor or inferential eligibility is emitted. | global covariance contract |
| C52-06 | Negative/out-of-range references fail before indexing and `max_num_compressed = 0` fails before allocation whenever more than one ministack is required. | `dolphin-stack/tests/planner_contract.rs`, direct sequential and workflow config contracts |
| C52-07 | Single-ministack, trailing partial block, compressed-cap eviction, masked, bounded, and tiled paths retain the same factor/date mapping/status. A numeric multiburst seam fixture proves that pre-leveling factors remain separate and stitched output returns `unsupported_seam_covariance` after data-dependent rotation. | sequential, displacement, and multiburst contracts |
| C52-08 | Fresh-full and every incremental boundary produce identical sealed blocks, open-block recomputation, status, and reconstructed tiny covariance. | `nrt_incremental_contract.rs`, `nrt_displacement_contract.rs` |
| C52-09 | HDF5 round-trip preserves per-burst `F/L/w`, parent/date mappings, replay descriptor/support masks, schema/method version, gauge, stitched/replay status, and non-valid pixel status without a dense intermediate allocation. | new `dolphin-io` covariance contract |
| C52-10 | At fixed `K`, `M`, and microtile budget, measured coefficient count matches the exact `F/L/w` plus support-bit formula; each doubling from 13 to 26 to 52 dates keeps storage near 2x and rejects 4x quadratic growth. A 256x256 looked tile on the declared 32 GiB host stays below 24 GiB RSS and 2x the factor-disabled wall time; final bytes stay within 10 percent of payload plus declared HDF5 metadata. | new release-mode `sequential_covariance_bench` |
| C52-11 | Legacy marginal CRLB bands are machine-labeled non-inferential. If used as SBAS or point-estimator quality weights, that use is recorded in provenance; no downstream code treats them as absolute precision, posterior calibration, total uncertainty, or an eligibility decision. PR #55's IID-conditional output is invariant to `write_crlb`. | config/displacement/output contracts |
| C52-12 | Existing phase, compression, CRLB-marginal, mask, tile, burst, and NRT parity contracts remain green. | existing workspace suites |
| C52-13 | Interior, clamped-border, `strides > 1`, masked, and fixed-SHP windows replay exactly the native samples consumed by the phase-link kernel; whole, tiled, and bounded paths use identical global coordinates and bit-packed support. | new phaselink covariance-replay contract |
| C52-14 | The `F/L/w` factor reconstructs the same single-pixel temporal covariance under valid orthogonal rotations of each local `L` basis, including compression ancestry, cap eviction, a partial block, and the exact gauge. No cross-pixel query exists in #52. | covariance-replay and global covariance contracts |
| C52-15 | Source/config/support/estimator fingerprints match on deterministic replay. A stale source, changed SHP mask, branch mismatch, or corrupt descriptor changes `replay_status` and blocks #54 without changing the temporal factor result. | covariance-replay and provenance contracts |
| C52-16 | Fresh/NRT sealed replay descriptors are identical, open descendants alone change, and halo support needed to select a bounded final reference remains available. | NRT and bounded workflow contracts |
| C52-17 | HDF5 round-trip preserves `F/L/w`, parent edges, grid/support metadata, estimator branch, bit-packed masks, gauge, and factor/replay statuses. | `dolphin-io` covariance contract |
| C52-18 | Coefficient counts follow the exact blockwise formula; date and area doubling reject quadratic growth. Factor-only and factor-plus-replay receipts are reported separately, and no allocation contains `tile_area*S_b*d_b^2`. | release-mode covariance benchmark |
| C52-19 | A signed producer manifest binds the reviewed design, analytic/fail-closed/resource receipts, supported scope, factor/schema version, code identity, and unresolved findings. Missing, failed, stale, or mismatched evidence blocks #54 eligibility. | producer-manifest schema contract |

### #52 task manifest

#### T52-01: contain the current inference path and freeze the factor design

**Intake:** DR-052-STATE, DR-052-MEMORY, DR-052-GAUGE, DR-052-BOUNDARY.
**Files:** the reviewed result of PR #55, focused config/displacement/output contracts, new
`md/design/sequential-global-covariance.md`, `README.md`, and `docs/usage.md`.

Write a red regression proving the PR #55 boundary survives, then freeze the
contrast/canonical/conditional equations, exact support iterator and replay semantics, fixed-SHP
conditioning scope, local versus composed dependency cone, exact gauge, run errors versus
factor/replay statuses, deterministic replay requirements, streamed HDF5
schema, exact coefficient/RSS algebra, and versioning. Do not start the kernel until
scientific review approves the design. Record a no-go if the temporal factor or replay descriptor
cannot meet the byte formula and resource bound.

#### T52-02: turn the global analytic and fail-closed contracts red

**Intake:** DR-052-STATE, DR-052-ANALYTIC, DR-052-FAIL, DR-052-GAUGE.
**Files:** new `crates/dolphin-phaselink/tests/global_covariance_contract.rs`, new
`crates/dolphin-phaselink/tests/covariance_replay_contract.rs`, new
`crates/dolphin-workflows/tests/global_covariance_contract.rs`, their committed JSON fixtures,
`crates/dolphin-stack/src/lib.rs`, `crates/dolphin-stack/tests/planner_contract.rs`,
`crates/dolphin-workflows/src/sequential.rs`, and focused config contracts.

Add C52-01/C52-03 through C52-06 and C52-13 through C52-16 before production changes. Load each
fixture with `include_str!` or another unconditional compile-time path; missing input must fail the
build or test. Add checked signed-reference resolution and disconnected-gauge errors before any
cast, index, or allocation.
Freeze the `A/J/F/Q/L/w` equations and exact replay identities against the rational fixtures.

#### T52-03: retain local information and implement the bounded propagation kernel

**Intake:** DR-052-STATE, DR-052-MEMORY, DR-052-GAUGE, DR-052-FAIL.
**Files:** `crates/dolphin-phaselink/src/crlb.rs`, `covariance.rs`, `estimator.rs`, `fused.rs`,
`engine.rs`, `quality.rs`, `lib.rs`, new
`crates/dolphin-phaselink/src/sequential_covariance.rs`, and focused phase-link contracts.

Expose the full reduced local factor without changing the legacy marginal result. Share the exact
support iterator with the production coherence kernel and capture replay fingerprints, realized
support, estimator branch/solution identity, and compression Jacobian while local state exists.
Implement the frontier square-root update, packed block factor, reconstruction/solve API, and
stable factor/replay statuses. Turn C52-01 through C52-05 and C52-13 through C52-15 green; the full
coherence matrix remains per-pixel and is discarded after these consumers finish.

#### T52-04: carry the factor through sequential, NRT, tiles, and bursts

**Intake:** DR-052-STATE, DR-052-GAUGE, DR-052-STATUS, DR-052-FAIL.
**Files:** `crates/dolphin-workflows/src/sequential.rs`, `src/lib.rs`,
`src/displacement.rs`, and the sequential/NRT/multiburst/displacement contracts.

Add a separate tile-scoped `SequentialCovarianceFactor`; do not overload `crlb_sigma`. Extend
`SequentialState` with sealed blocks and the open frontier, preserve masked/invalid pixels, and
prove fresh/full/incremental identity in C52-07/C52-08. Stream completed blocks per burst instead
of adding a whole-frame factor to `TiledOutput` or `DisplacementOutput`. Preserve global grid
coordinates, sparse parent edges, bit-packed support, and halo replay metadata through tile/crop
offsets without expanding ancestry.
Turn C52-16 green. Preserve the numeric seam fixture and refuse a stitched factor after overlap
leveling. No new factor call is allowed from SBAS or velocity fitting.

#### T52-05: persist method, status, and factor data without enabling a consumer

**Intake:** DR-052-STATUS, DR-052-BOUNDARY.
**Files:** new `crates/dolphin-io/src/covariance.rs`, `crates/dolphin-io/src/lib.rs`,
`crates/dolphin-workflows/src/displacement.rs`, output/provenance contracts, and `CHANGELOG.md`.

Implement the scratch/final chunked HDF5 and JSON schema, round-trip C52-09, and legacy CRLB tags.
Include factor/schema/crate version, nullable build commit, gauge, date/block/stencil map,
`F/L/w`, parent edges, grid/support metadata, bit-packed masks, estimator branch/solution identity,
dimensions, coefficient count, factor/replay status registries, assumptions, normalized-config
digest, and an ordered source-manifest digest over burst/date/group/grid metadata plus content
hashes only where the caller already supplies them. Do not hash complete CSLC files during output.
Keep the factor out of GroundPulse output policy, turn C52-17 green, and set
`downstream_inference_status=blocked_pending_issue_54_and_53`.

#### T52-06: prove resources, sign the producer manifest, and open one unmerged PR

**Intake:** DR-052-MEMORY, DR-052-BOUNDARY.
**Files:** new release-mode benchmark/example, narrow validation receipts, immutable independent
review receipt, producer manifest/schema contract, changelog, design, intake, and this plan only
as evidence changes.

Run C52-10/C52-18 at 13, 26, and 52 dates with `M=13`, fixed `K`, the declared microtile budget,
and a 256x256 looked tile in a fresh process. Compare factor-disabled, temporal-factor-only, and
factor-plus-replay paths; record crate/build identity, dimensions, exact coefficient formula/count,
HDF5 bytes/throughput, wall time, and peak RSS. A projected full artifact must preflight final plus
scratch disk with 25 percent free margin. MMX1 2023 and 2018 may be shape/resource regressions
after the synthetic benchmark passes; they are
not calibration evidence. Give an independent scientific reviewer the design, source,
analytic/fail-closed evidence, all failures, and resource receipts. Only a review with no
unresolved finding may sign the #52 producer manifest and turn C52-19 green. The manifest
authorizes only the producer factor. #54 spatial calibration and #53 inference remain blocked.
Open one unmerged PR and stop before merge/release/pinning.

### #52 verification

```text
cargo test -p dolphin-stack --test planner_contract
cargo test -p dolphin-phaselink --test global_covariance_contract
cargo test -p dolphin-phaselink --test covariance_replay_contract
cargo test -p dolphin-phaselink --test quality_v042_contract
cargo test -p dolphin-phaselink --test fused_contract
cargo test -p dolphin-workflows --test global_covariance_contract
cargo test -p dolphin-workflows --test sequential_contract
cargo test -p dolphin-workflows --test nrt_incremental_contract
cargo test -p dolphin-workflows --test nrt_displacement_contract
cargo test -p dolphin-workflows --test multiburst_contract
cargo test -p dolphin-workflows --test displacement_contract
cargo test -p dolphin-io covariance
cargo test -p dolphin-io producer_manifest
cargo run --release -p dolphin-workflows --example sequential_covariance_bench
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --all-targets -- -D warnings
cargo test --workspace
git diff --check
```

## Issue #54: spatial-reference covariance

### Selected representation and scope

Use #52's per-burst temporal factor and exact replay descriptor in a reference-specific second
pass. Derive local cross covariance from a shared estimating-equation influence, then apply the
actual temporal-inversion and target-minus-reference maps. For native source `q` and pixel `p`:

```text
U_p(theta_p) = sum over q in S_p of u_pq(theta_p) = 0
A_p = -d U_p / d theta_p
G_pq = sqrt(n_p / ell_p) A_p^-1 (d u_pq / d xi_q)
                                                   # effective-look shared influence
Cov_local(p,r) = sum over q in S_p intersect S_r of G_pq G_rq'
delta theta_p = R_p xi                         # G composed through replayed influence Jacobians
delta z_p = H_p E_p R_p xi                    # fixed-branch L2 displacement influence
B = [I, -I]
C54_delta(p,r) = B Cov([z_p; z_r]) B'
                 = C_pp + C_rr - C_pr - C_rp
                 = (H_p E_p R_p - H_r E_r R_r)
                   (H_p E_p R_p - H_r E_r R_r)'
```

`E_p` builds the valid interferogram contrasts and `H_p` is the exact fixed-weight L2 map used by
the workflow, with acquisition 0 excluded. Target and reference maps are aligned to one ordered
common-date gauge before subtraction. The unwrap branch, SHP membership, coherence weights, and
selected reference are conditioned on their realized values; uncertainty in their selection is
not silently folded into this factor.

`xi_q` is one canonical whitening of the raw complex-look vector under the frozen local complex
Gaussian model, shared by every window that consumes `q`; `n_p` is active source count and `ell_p`
is the exact production CRLB effective-look rule, versioned as part of the method. The scalar
`sqrt(n_p / ell_p)` makes that heuristic an explicit source-loading assumption instead of
mislabeling hundreds of window samples as independent looks. Obtain the remaining influence from
the estimator score and Hessian/implicit Jacobian; do not left-normalize separate pixels to force
their marginals.
Under a valid gauge/reference reparameterization `T_p`, the implementation must satisfy
`G_new = T_p G_old` and `C_new = T_p C_old T_r'`. #54's sandwich marginal and cross block form one
joint factor; they must not be mixed with #52 marginals. Report their ratio and matrix discrepancy
against #52 as diagnostics, but do not force equality: #52 uses a legacy Fisher/effective-look
bound, while #54 is a separate calibrated influence method. A congruence, joint-PSD, eigen-gap,
branch-stability, or finite-difference failure records a no-go for that scope. Overlap fractions,
two unrelated Cholesky factors, and `C_pr = 0` are unsupported.

New-acquisition innovations are distinct from carried-parent state so sequential composition does
not count a source twice. Recompute the actual estimator/compression transition
`F_influence = d x_new / d s_carried` by implicit differentiation and finite difference; #52's
`F = -Q J_xs` is a temporal Fisher transition and is only a diagnostic comparator. The supported
compression Jacobian is partitioned as `c_new = w_s s_carried + w_x x_new`, so the carried-state
influence update is exactly `T_influence = w_s + w_x F_influence`. Reuse #52's checked `w_s/w_x`
only after its finite-difference replay contract passes; do not add another compression Jacobian.
The supported
model treats different native sources as independent looks. Spatial correlation beyond shared
support remains outside that model and is included as an explicit stress dimension.

Initial eligible scope is CPU EMI/EVD, fixed realized Rect/GLRT/KS support, no phase-bias
correction, fixed unwrap branch, and L2 inversion. Deterministic atmospheric/solid-earth
corrections must be applied to every pixel before the final spatial reference; their own
uncertainty remains explicitly unmodeled. L1, a changed unwrap branch, unstable adaptive support,
unsupported correction order, and unmodeled correction-uncertainty claims return explicit states.
These limits are part of the calibration hash; no fallback emits an inferential factor.

Expose matrix-free queries rather than pair rasters:

- `replay_reference(reference)` builds one source-keyed signature under a checked #52 manifest;
- `contract_target(target, reference_signature, temporal_map)` computes and discards one target
  influence while persisting only its reference-specific block;
- `difference_covariance(target, reference, common_dates)` returns a rank-revealing square-root,
  rank/nullity, diagonal, selected blocks, pseudo-solve, pseudo-whitening, and log-pseudodeterminant;
- coincident target/reference reuses the identical row and returns exact zero without jitter.

For one selected reference, cache its signature and stream a byte-capped target microbatch. Skip a
target only after an exact support-cone test proves disjoint source keys under the supported
independent-look model. A query may materialize one block-local target/reference covariance, never
a date-squared spatial cube, all-pixel influence matrix, or pixel-pair object. Persist only the
reference-specific block factors in a chunked `referenced_displacement_covariance_factor.h5` plus
JSON provenance. A different GNSS control reference requires a separate bounded replay; it cannot
reuse a factor for another reference. The artifact records the final reference in analysis,
output, and map coordinates; CRS/affine transform; radian and displacement unit transforms; masks;
date/gauge maps; burst ownership; source factor identity;
window/SHP/stride scope; approximation hash; and status.

Let `h_b` be the reference dependency-cone sources, `s_b` one target's local sources, `c_b` the
source innovation components, and `d_b <= K + M - 1`. Working memory is
`O(sum_b h_b*d_b*c_b + microbatch*max_b(s_b*d_b*c_b + d_b^2))`; the reference term is independent
of output area and `microbatch` is chosen from a byte cap. Stored reference-specific blocks are
`O(area*sum_b d_b^2)`. T54-01 records a no-go if the measured reference cone, output bytes, or
runtime cannot meet the frozen resource budget; it may not revert to all-pixel influence storage.

Whole-frame and bounded workflows select their final reference before finalizing the contrast
factor. A bounded target whose halo reference is replaced must query the new target-local
reference from the original per-burst factor, not re-reference an already differenced marginal.
The covariance remains in radian squared until the same output boundary applies
`(wavelength / 4 pi)^2`.

Multiburst scope remains reference-pair specific. Carry the stitched phase's source-burst
ownership under the existing finite-overwrite rule. A same-pre-leveling-burst pair is eligible only
after a numeric contract proves its common seam rotation cancels in the contrast. Mixed or
ambiguous ownership returns `unsupported_multiburst_reference`; no stitched global factor is
fabricated.

Method/status contract:

```text
method: reference_specific_influence_v1
spatial_covariance_status: valid | invalid_reference | masked_target |
                           temporal_factor_invalid |
                           replay_unavailable | replay_mismatch | influence_invalid |
                           nondifferentiable_estimator | unstable_adaptive_support |
                           unsupported_l1 | unsupported_phase_bias |
                           unsupported_correction_order |
                           unsupported_multiburst_reference | scope_mismatch
reference_relation: distinct | coincident
rank_status: full_rank | rank_deficient
condition_status: acceptable | ill_conditioned
calibration_status: uncalibrated | calibrated_scope_match | calibration_scope_mismatch
correction_uncertainty: not_modeled
inference_status: blocked_pending_issue_53
```

`valid` certifies only that the algebraic factor is evaluable. `calibrated_scope_match` additionally
requires #54's signed analytic, approximation, resource, and review manifest. Corrected slope
sigma and total uncertainty remain blocked. A calibrated rank-deficient measurement factor remains
consumable by #53 only when its total `V(theta)` passes #53's positive-definite gate.

### Draft approximation preregistration

Freeze the exact JSON grid, seeds, tolerances, hardware, and artifact schema before generating
outcomes. A performance-only dry run may narrow scope but cannot inspect covariance or coverage
results.

- Half-windows: `(1,1)`, `(3,6)`, and the default `(7,14)`; strides `(1,1)`, `(2,2)`, and `(4,4)`.
- Support: Rect plus frozen GLRT/KS masks; stable and deliberately unstable selection cases;
  interior, clamped border, tile edge, bounded halo, and masked samples.
- Pair geometry: coincident; shared support with positive and negative signed influence; 75, 50,
  and 25 percent key overlap; immediate disjoint support; and the first disjoint boundary after
  each sequential compression depth. Record both map distance and exact key intersection.
- Temporal structure: one block, two blocks, cap eviction, and four blocks; full and partial final
  blocks; regular and missing interferogram networks; EMI/EVD; well-separated and tied-eigenvalue
  stress cases.
- Source process: independent complex looks in the supported scope plus spatial-correlation stress
  cells beyond the shared-key model. An emitted stress cell must pass the same error gates or it
  blocks that scope; the implementation cannot abstain based on a latent generating parameter.
- Effective looks: freeze the exact current CRLB `ell_p` rule and source-loading normalization in
  the method hash. Report #54/#52 marginal ratios across support sizes and SHP masks. A discrepancy
  is evidence about two different methods, not permission to rescale #54 after outcomes.
- Design: a committed pairwise covering array plus every one-factor-at-a-time and worst-corner
  cell, with at least 5,000 attempted seeds per supported cell and no outcome-based top-up.
- Deterministic matrix fixtures must match within `1e-10` relative error, preserve symmetry and
  positive semidefiniteness to the same scaled tolerance, and keep the gauge/coincident contrast
  exactly zero.
- In stochastic cells, normalize cross-block error by the geometric mean marginal scale. The
  cellwise relative operator error and non-negligible contrast-variance error must each be at most
  0.10; nominal 95 percent Gaussian contrast coverage must be within 0.02; successful emission
  must be at least 99 percent. Report weak-zero variances separately instead of dividing by them.
- Report every cell, estimator status, attempted/emitted count, covariance errors, eigenvalue and
  contraction diagnostics, coverage, runtime, peak RSS, code SHA, factor hashes, and environment.
  Aggregate performance cannot hide a failed cell.
- The 256x256 looked-tile resource gate runs at 13, 26, and 52 dates on the declared 32 GiB host,
  stays below 24 GiB RSS, and records the reference-cone, microbatch, persisted-block, scratch, and
  final-byte formulas. Area/date doubling must reject quadratic growth; one-reference replay must
  allocate no two area-scaled axes and no `tile_area*S_b*d_b^2` buffer.

The 0.10 limits apply only to #54 research validation. Downstream intervals may not be rounded or
inflated to satisfy them. Independent review may tighten the preregistration before the first
outcome run; it may not loosen a frozen threshold.

### #54 test and evidence contract

| ID | Contract | Primary location |
|---|---|---|
| C54-01 | Independent, positive, negative, coincident, and invalid-reference scalar/matrix fixtures match `C_pp + C_rr - C_pr - C_rp`; equal marginals satisfy `Var_positive < Var_independent < Var_negative`. | new phaselink/workflow spatial-reference contracts |
| C54-02 | Interior, border-clamped, partial-SHP, disjoint-SHP, masked, and `strides > 1` hand windows reproduce shared-score cross covariance and reject unstable support. | phaselink spatial-influence contract |
| C54-03 | EMI/EVD score and Hessian match central finite differences away from branch boundaries; the effective-look influence transforms by congruence under every supported local-reference/gauge map and forms a joint PSD factor. #52 marginal discrepancies are reported, never normalized away. Ties, wrap changes, nonfinite scores, congruence/PSD failures, and support changes fail closed. | phaselink estimator/influence contracts |
| C54-04 | A two-ministack spatial fixture finite-differences `F_influence` and `T_influence = w_s + w_x F_influence`, proves covariance can extend beyond immediate-window overlap through compressed ancestry, rejects substitution of #52's Fisher `F`, and matches a tiny dense oracle without storing the dense object in production. | phaselink/workflow global spatial contract |
| C54-05 | The actual fixed-weight L2 map with different valid observations reproduces a hand SBAS network; gauge zero, temporal map, symmetry, rank/nullity, pseudo-operations, and units are exact. L1 and changed unwrap branches abstain. | timeseries spatial-reference covariance contract |
| C54-06 | Whole, tiled, NRT, and bounded paths produce the same final-reference contrast and preserve CRS, affine transform, units, mask, reference coordinates, date map, and provenance. Configured deterministic corrections are applied to target and reference before subtraction; their uncertainty remains labeled unmodeled. | displacement/bounded/NRT contracts |
| C54-07 | Source-burst ownership follows stitched phase overwrite rules. A proved same-burst seam rotation cancels; mixed or ambiguous burst pairs fail. | multiburst contract |
| C54-08 | Legacy displacement variance, CRLB, and conditional velocity products retain `SPATIAL_COVARIANCE=target_reference_covariance_not_modeled` and `INFERENCE_READY=false`; neither marginals nor a bare #52 factor can authorize corrected output. | output/config contracts |
| C54-09 | HDF5/JSON round-trip preserves reference-specific block factors, rank, parent/replay identity, final reference coordinates, grid, burst, gauge, estimator branch, masks, units, correction boundary, scope, hashes, and status. Corrupt or mismatched provenance fails closed. | `dolphin-io` and workflow contracts |
| C54-10 | The frozen approximation grid reports every cell and satisfies its cellwise gates; failures and non-evaluable states remain distinct. | Rust batch target, Python driver/scorer, versioned receipt |
| C54-11 | Coefficient counts and measured allocations satisfy the reference-cone/microbatch formula and tile gate; no allocation contains two area-scaled axes or all-pixel source influence. | release-mode spatial-covariance benchmark |

### #54 task manifest

#### T54-01: add disclosure regression and freeze the spatial design

**Intake:** DR-054-QUANTITY, DR-054-BOUNDED, DR-054-BOUNDARY.
**Files:** the reviewed PR #55 metadata path, focused output contracts, new
`md/design/spatial-reference-covariance.md`, `README.md`, `docs/usage.md`, and `CHANGELOG.md`.

Write C54-08 red against the settled PR #55 baseline. Freeze the target/reference quantity,
shared score/covariance/Hessian and effective-look model, #52/#54 marginal separation,
gauge-congruence rule, fixed/unsupported branches, rank-revealing operator semantics,
correction/reference order, multiburst ownership, reference-replay API, unit transform, status
registry, schema, resource formula, and no-go
conditions. T54-01 starts only after #52's PR disposition is explicit and any authorized merge is
green on refreshed `main`. Scientific review must approve this design before kernel changes.

#### T54-02: make the analytic and geometry contracts red

**Intake:** DR-054-ANALYTIC, DR-054-GEOMETRY.
**Files:** new `crates/dolphin-phaselink/tests/spatial_reference_covariance_contract.rs`, new
`crates/dolphin-timeseries/tests/spatial_reference_covariance_contract.rs`, new
`crates/dolphin-workflows/tests/spatial_reference_covariance_contract.rs`, and committed tiny
JSON fixtures.

Add C54-01 through C54-07 before production changes. Use compile-time fixture paths, compare tiny
dense matrices only inside tests, and freeze all numeric/status expectations. The two-ministack
fixture must cross an immediate support boundary so a local-overlap-only implementation stays red.

#### T54-03: implement and validate the local shared-source influence

**Intake:** DR-054-QUANTITY, DR-054-BOUNDED.
**Files:** `crates/dolphin-phaselink/src/covariance.rs`, `estimator.rs`, `fused.rs`, `engine.rs`,
the #52 `sequential_covariance.rs`, a new `spatial_covariance.rs`, and focused contracts.

Implement the shared-score influence by implicit differentiation under the frozen complex-look
and effective-look model, plus source-key intersection, target/reference contraction, and spatial
status registry. Reuse the production support iterator and #52 replay descriptor. Verify gauge
congruence, joint PSD, #52/#54 marginal separation, and central differences for EMI/EVD;
do not silently send GPU, phase-bias, unstable SHP, singular, or nonfinite cases through an
independent-pixel fallback.

#### T54-04: compose spatial influence through sequential state and L2 inversion

**Intake:** DR-054-BOUNDED, DR-054-GEOMETRY.
**Files:** the #52 sequential factor, `crates/dolphin-workflows/src/sequential.rs`, `tiling.rs`,
NRT contracts, `crates/dolphin-timeseries/src/inversion.rs`, and focused workflow contracts.

Compose a cached reference signature and byte-capped target microbatch through replayed
`F_influence/T_influence` parent edges without persisting all-pixel influence, then apply each
pixel's actual `E/H` maps. Treat #52 `F` as a comparator and reject it unless a contract proves
numeric equality for that estimator. Persist only reference-specific block factors. Preserve
exact common dates and acquisition-0 gauge. Turn C54-04/C54-05 green and expose bounded replay for
any valid same-burst pair, not only the
configured frame reference; #53's GNSS station pairs invoke a separate replay for their control.

#### T54-05: wire final reference, bounded output, bursts, and persistence

**Intake:** DR-054-BOUNDED, DR-054-GEOMETRY, DR-054-BOUNDARY.
**Files:** `crates/dolphin-timeseries/src/reference.rs`,
`crates/dolphin-workflows/src/displacement.rs`, a new spatial-covariance provenance module,
multiburst/bounded contracts, and the planned `crates/dolphin-io/src/covariance.rs`.

Resolve the final whole/bounded reference before contrast finalization, carry source-burst
ownership, prove the corrected-mean/reference order, apply the unit transform once, and stream the
factor/provenance artifact. Keep `reference_variance_to_point` only on the explicitly
non-inferential legacy path. Turn C54-06 through C54-09 green; never construct a two-marginal
fallback for the new method.

#### T54-06: freeze and run approximation/resource validation

**Intake:** DR-054-VALIDATION, DR-054-BOUNDED, DR-054-BOUNDARY.
**Files:** new `validation/spatial_covariance_preregistration.json`, a deterministic Python
generator/scorer and tests, a release-mode Rust batch target and benchmark, versioned results under
`validation/results/spatial_covariance/`, and `VALIDATION.md`.

Run a performance-only dry run, freeze the grid/hash/resources, then generate outcomes once. Rust
must execute the production influence/operator path; Python may generate and score fixtures only.
Turn C54-10/C54-11 green or record the failed cells and a no-go. No aggregate pass, metadata tag,
or larger variance can substitute for the frozen cellwise gates.

#### T54-07: obtain independent review and open one unmerged PR

**Intake:** DR-054-VALIDATION, DR-054-BOUNDARY, DR-052-BOUNDARY.
**Files:** immutable review receipt, signed #54 method manifest and schema test, `VALIDATION.md`,
design, intake, plan, and changelog only as evidence changes.

Give an independent scientific reviewer the design, source, analytic/finite-difference evidence,
full approximation artifact, all failures, and resource receipts. Only a review with no unresolved
finding may sign the method manifest. Open one reviewable, unmerged PR and stop. A signed #54
manifest still leaves `inference_status=blocked_pending_issue_53`; stop before corrected output,
merge, release, publication, GroundPulse pinning, or deployment.

### #54 verification

```text
cargo test -p dolphin-phaselink --test spatial_reference_covariance_contract
cargo test -p dolphin-timeseries --test spatial_reference_covariance_contract
cargo test -p dolphin-workflows --test spatial_reference_covariance_contract
cargo test -p dolphin-workflows --test multiburst_contract
cargo test -p dolphin-workflows --test nrt_displacement_contract
cargo test -p dolphin-io covariance
cargo run --release -p dolphin-workflows --example spatial_covariance_batch -- \
  --contract validation/fixtures/spatial_covariance_batch.jsonl
cargo run --release -p dolphin-workflows --example spatial_covariance_bench
oracle/.venv/bin/python -m unittest discover -s validation/tests -p 'test_spatial_covariance*.py'
oracle/.venv/bin/python validation/spatial_covariance_simulation.py \
  --prereg validation/spatial_covariance_preregistration.json \
  --rust-batch target/release/examples/spatial_covariance_batch \
  --output validation/results/spatial_covariance/coverage.json
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --all-targets -- -D warnings
cargo test --workspace
git diff --check
```

## Issue #53: temporal-covariance slope inference

### Initial statistical scope

The first calibrated scope is a linear Sentinel-1 slope only. Seasonal and step models return
`unsupported_model` and retain conditional output until their own grid is preregistered and
passed. Model the already spatially differenced series:

```text
y_(p-r,t) - y_(p-r,0) = beta * (t - t_0) + e_t
C54_delta(p,r) = difference_covariance(p, r, common_dates)
D[i,i] = sqrt(C54_delta[i,i] / geometric_mean_positive_diag(C54_delta))
V(theta) = C54_delta(p,r) + sigma^2 D R_rho D
R_rho[i,j] = I(i=j)                              when rho_12 = 0
R_rho[i,j] = rho_12^(abs(t_i-t_j)/12 days)       when 0 < rho_12 < 1
```

`C54_delta` is a calibrated, same-scope #54 factor for the already spatially differenced
measurement series; #53 never reconstructs it from two marginal products. It carries
target/reference measurement noise and their signed cross covariance. `D` is a fixed relative
standard-deviation shape derived from its positive post-gauge diagonal, while `sigma^2` models the
remaining differenced temporal process. Separate target and reference OU components are not
identifiable from one differenced series and are not fit. The simulation generates a block-PSD
target/reference residual process with frozen reference contribution and correlation, then scores
the induced difference; production estimates only its total `sigma^2` and `rho_12`.

Missing dates select factor rows/columns without imputation. Acquisition 0 fixes the temporal
gauge exactly, so the calibrated estimand is an origin-anchored slope with no intercept. The
legacy intercept-plus-slope conditional WLS remains a separately labeled migration comparator,
not one of the like-for-like selection methods. `p = r`, a missing gauge date, a nonpositive
post-gauge diagonal, or unavailable/invalid/uncalibrated #54 factor data fails closed. Reference
pixel, window/overlap/distance scope, factor identities, and approximation bound are provenance.

`C54_delta` may be positive semidefinite. #53 forms and factors the total `V(theta)`; it does not
call an ordinary inverse or log-determinant on the measurement factor alone. Corrected inference
requires total `V` to be positive definite and below the frozen condition-number limit. A fitted
`sigma^2` at its lower boundary, a singular total `V`, or failed factorization returns a stable
non-evaluable status rather than a pseudo-inferential interval.

Fit covariance parameters within preregistered bounds by constrained REML/profile likelihood.
Compare a Kenward-Roger-style adjusted scalar slope SE, joint/profile slope inference, and a
complete-refit parametric bootstrap; every bootstrap replicate resimulates data and refits both
mean and covariance parameters. A production candidate must define one scalar SE and its DOF so
the same symmetric 68/90/95 intervals pass. If only asymmetric profile/bootstrap intervals pass,
do not collapse them into a scalar sigma. The scalar `N_eff` method remains a comparator only.

### Draft preregistration to freeze before any experiment

The committed JSON enumerates every cell and attempted seed. A performance-only dry run on
fixtures excluded from the grid happens before the final hash; it may set the supported scope and
compute budget, but it cannot inspect coverage outcomes.

- Acquisition counts: 12, 24, 48, 96 retained valid dates after the frozen missingness mask.
- Correlation at a 12-day reference lag: an explicit iid branch at 0, then 0.3 and 0.6 across all
  counts; 0.85 is supported only at 48 and 96 dates. The 12/24-date 0.85 cells and 0.95 cells are
  weak-identification/boundary stress probes excluded from the supported-emission scope.
- Cadence: regular 12-day; alternating 6/18-day; independently jittered by up to 4 days; two
  36-day gaps. The draft production predicate requires strictly increasing dates, at least 12
  common dates, median gap 6-18 days, and maximum gap at most 36 days; the final predicate is
  frozen in JSON and does not depend on whether a diagnostic lag bin is populated.
- Missingness: none; 10 percent MCAR; 25 percent MCAR; one contiguous 20 percent block. The
  scheduled stack is lengthened so each cell retains its named valid-date count. Attempted masks
  are frozen and constrained to the observable cadence predicate before outcomes are generated.
- Heteroskedastic variance ratios: 1, 4, 16, represented by standard-deviation multipliers 1, 2,
  4 in valid block-PSD #54 joint factors. Their alternating and contiguous-block temporal
  arrangements are enumerated, not randomized after the seed is chosen.
- Spatial measurement covariance: consume frozen #54 independent, positive, negative,
  coincident, invalid, partial-overlap, sequential-depth, and approximation-envelope fixtures.
  Never synthesize a difference factor from two marginals.
- Reference residual contribution ratios: 0, 0.5, and 2, crossed with target/reference residual
  correlations only through block-positive-semidefinite joint matrices. A zero reference variance
  cannot be paired with nonzero cross covariance.
- Design: a committed pairwise covering array plus all one-factor-at-a-time and worst-corner
  cells; aggregate results cannot hide a failing cell.
- Factor-generation strata: compact conditional cells use frozen #54 factors; end-to-end outer
  cells regenerate raw complex looks, rerun the production #52/#54 path, and fit the slope from
  the same realized output/factor on every seed. These cells measure covariance-estimation error
  and its dependence on the slope outcome. Both strata must pass; a fixed-factor pass alone cannot
  promote the method.
- Outer Monte Carlo count: enough for the two-sided 95 percent binomial interval required by the
  coverage tolerance, with at least 5,000 attempted seeds per supported cell. There is no top-up;
  fit failures remain in the attempted denominator.
- Inner bootstrap count, seed hierarchy, exact cell count, concurrency, hardware class, maximum
  wall time, core-hours, RSS, and artifact bytes are frozen after the performance dry run. On the
  current 32 GiB host, any local run must stay below 24 GiB RSS. If the full nested design cannot
  fit the approved budget, reduce scope before hashing or retain a no-go.
- Candidate methods: OLS, oracle GLS, current conditional WLS, scalar `N_eff`, plug-in REML GLS,
  adjusted scalar inference, joint/profile inference, and complete-refit bootstrap. All methods
  except legacy intercept-plus-slope WLS use the origin-anchored estimand; legacy WLS is reported
  separately and cannot win method selection.
- Per-cell gates: absolute standardized slope bias at most 0.05 empirical SD; absolute coverage
  error at most 0.03, 0.02, and 0.015 at 68, 90, and 95 percent; at least 99 percent successful
  emission for supported cells. Report coverage both conditional on emission and with abstentions
  counted as misses. Deterministically observable invalid inputs and fitted boundary/weak-profile
  states must abstain 100 percent. A latent stress process has no abstention target because its
  generating correlation is not observable in production; every emitted stress interval is scored
  against the same coverage and proper-score gates, and a failure blocks promotion.
- The selected method's proper interval score and width are compared with the conditional and
  plug-in baselines at every nominal level so blanket interval inflation cannot pass.
- Failure codes and numeric thresholds for insufficient dates, cadence support, design rank and
  conditioning, covariance positive-definiteness, boundary correlation, profile curvature,
  bootstrap success, reference geometry, and calibration-scope match are immutable after the
  artifact hash is signed. Raw correlation is the unclamped correlation of adjacent residual
  pairs, accompanied by pair count and min/median/max elapsed gap. For fewer than three pairs,
  record correlation as absent. Do not record zero; the missing diagnostic does not invalidate an
  otherwise supported cadence.

If compute benchmarks make the nested bootstrap infeasible, reduce the supported scope or retain
the no-go. Do not reduce replicates, loosen tolerances, or substitute a plug-in method after seeing
coverage.

Stable non-evaluable codes include `insufficient_dates`, `unsupported_cadence`,
`dates_not_strictly_increasing`, `reference_pixel`, `gauge_missing`, `design_rank_deficient`,
`design_ill_conditioned`, `difference_covariance_unavailable`,
`difference_covariance_invalid`, `difference_covariance_uncalibrated`,
`spatial_scope_mismatch`, `factor_identity_mismatch`, `gauge_date_map_mismatch`,
`crs_grid_unit_mismatch`,
`covariance_nonfinite`, `total_covariance_not_positive_definite`,
`total_covariance_ill_conditioned`, `residual_variance_at_boundary`, `correlation_at_boundary`,
`weak_parameter_identification`, `bootstrap_insufficient_success`, and
`calibration_scope_mismatch`. A target/reference pair that does not use one common pre-leveling
burst returns `unsupported_multiburst_reference` until seam covariance is modeled.
`conditional_only` and `corrected_evaluable` are output states, not numeric-failure codes.

The continuous-time correlation form is grounded in Belcher, Hampton, and Tunnicliffe Wilson's
[irregularly sampled autoregressive model](https://academic.oup.com/jrsssb/article/56/1/141/7035888).
The adjusted fixed-effect comparator follows the covariance-parameter-uncertainty problem treated
by Kenward and Roger in
[Small sample inference for fixed effects from restricted maximum likelihood](https://repository.rothamsted.ac.uk/id/eprint/9275/).
Neither paper substitutes for the frozen coverage experiment.

### #53 test and evidence contract

| ID | Contract | Primary location |
|---|---|---|
| C53-01 | PR #55's legacy `true` flag rejection and paired conditional point/sigma contract remain intact. Marginal or `SPATIAL_COVARIANCE=target_reference_covariance_not_modeled` input cannot authorize corrected output. | config and displacement contracts |
| C53-02 | Exact iid/irregular-time construction, `D` scaling, missing-date subsetting, and gauge removal match hand matrices while consuming #54's independent/positive/negative/coincident/invalid difference fixtures directly. No two-marginal reconstruction is accepted; for equal marginals `SE_positive < SE_independent < SE_negative`; `p = r` is non-evaluable. | new `temporal_covariance_contract.rs` |
| C53-03 | Oracle GLS recovers a seeded known solution; plug-in, adjusted scalar, joint/profile, and complete-refit bootstrap methods use the identical origin-anchored data/design and return separate results. Legacy intercept-plus-slope WLS is labeled non-comparable. | timeseries contract and simulation tests |
| C53-04 | A hand covariance/design calculation using a direct #54 difference factor proves the scalar `N_eff` slope variance and implied Gaussian coverage are wrong for a fixed irregular, missing, heteroskedastic/reference-noise case. | compact analytic validation contract |
| C53-05 | Every temporal and #54 factor/scope/provenance fail-closed condition returns a stable status and no corrected sigma; raw adjacent-residual diagnostics remain unclamped and distinguish absent from zero with pair count and gap summary. | timeseries/workflow contracts |
| C53-06 | The Python driver generates frozen inputs but release-mode Rust runs every estimator. Cross-language hand fixtures agree; fixed-factor and end-to-end raw-complex-look strata report every #54 overlap/distance/approximation cell plus immutable attempted/emitted counts, conditional/unconditional coverage, proper scores, and resources. | Rust timeseries/workflow batch targets, versioned simulation artifact, and schema test |
| C53-07 | A calibrated same-burst rank-revealing factor returned by #54 is combined with the residual model, and #53 factors total `V` for solve/whitening/log-determinant. Marginal CRLB, two-marginal addition, zero-cross metadata, a bare #52 factor, pseudo-inversion of singular total `V`, and mixed/ambiguous bursts are rejected. | workflow integration contract |
| C53-08 | Corrected slope and scalar SE are a new paired product and exist only for a method whose symmetric 68/90/95 intervals passed; legacy `velocity.tif` and conditional `velocity_sigma.tif` are not relabeled or mixed. | output contract |
| C53-09 | Per-pixel rasters and the sidecar persist estimator version, valid dates, rank/DOF, cadence predicate, raw adjacent-residual correlation with pair-count/gap summary, fitted `rho_12`, final reference geometry, #52/#54 methods and source identities, overlap/window/distance stratum, approximation bound, conditioning, scope/review/calibration hashes, and bootstrap counts. | provenance/output contract |
| C53-10 | A frozen, untouched outer cohort is disjoint by burst/orbit/footprint/site/stations and obtains `difference_covariance(primary, control)` directly for each same-frame GNSS station pair, then scores slope differences with GNSS slope covariance, exact cluster inference, interval score, and width. Previously exposed Fresno/MMX1/five-burst data cannot enter the outer holdout. | GNSS cohort/schema/new scorer contracts |
| C53-11 | A signed promotion manifest contains matching successful #52/#54 analytic, approximation, scope, resource, and independent-review hashes plus #53 synthetic, field, resource, and review hashes before corrected output is configurable. Missing, failed, stale, or mismatched receipts leave `conditional_only`. | promotion-manifest contract |
| C53-12 | Full workspace and Python validation remain green; scientific artifacts report pass, fail, and not-evaluable as distinct states. | existing and new suites |
| C53-13 | The wired estimator, including #54 difference-factor retrieval/construction, processes 256x256 tiles at 12/48/96 dates below 24 GiB RSS without whole-frame covariance and takes at most 2x the conditional-fit wall time on the declared host. A recorded scaling formula must project 3.9 million pixels to at most 60 minutes. | release-mode temporal-inference benchmark and workflow receipt |

### #53 task manifest

#### T53-01: enforce the release boundary and freeze the model

**Intake:** DR-053-MODEL.
**Files:** config/workflow files from the shared containment, focused contracts, and new
`md/design/temporal-covariance-slope-inference.md`.

Write C53-01 as a regression against PR #55's settled boundary and preserve the scalar routine
only behind a research/test interface. Freeze the equation, mean design, immutable #54
`difference_covariance` interface, `D` construction, joint reference-residual generation,
supported initial model, parameter bounds, factor/scope failure states, and estimator outputs.
T53-01 starts only after #54's PR disposition is explicit and any authorized merge is green on
refreshed `main`. Independent scientific review must approve the design before T53-02 is frozen.

#### T53-02: draft the preregistration and write red analytic contracts

**Intake:** DR-053-ESTIMATION, DR-053-PREREG, DR-053-COMPARATORS.
**Files:** new `validation/temporal_covariance_preregistration.json`,
`validation/temporal_covariance_simulation.py`,
`validation/tests/test_temporal_covariance.py`,
`crates/dolphin-timeseries/tests/temporal_covariance_contract.rs`, and committed tiny fixtures.

Enumerate the proposed cells, attempted seeds, intervals, tolerances, failure thresholds, field
estimand, and artifact schema without generating outcomes. Add C53-02 through C53-05 red,
including the analytic scalar-`N_eff` variance/coverage counterexample. The compact analytic suite
runs in CI; the full Monte Carlo does not. The preregistration remains explicitly `draft` until
T53-03's performance-only dry run freezes the compute fields. Immutable #54-format analytic
fixtures support unit tests, but no fixture can stand in for a signed #54 method in a promotion
artifact.

#### T53-03: implement the estimator and comparator kernel

**Intake:** DR-053-ESTIMATION, DR-053-PREREG, DR-053-COMPARATORS.
**Files:** new `crates/dolphin-timeseries/src/temporal_covariance.rs`,
`crates/dolphin-timeseries/src/lib.rs`, `inversion.rs`, `velocity_model.rs`, a new release-mode
JSONL batch target, focused contracts, the Python driver, and only the seeded RNG dependency
required by the frozen bootstrap.

Implement covariance construction, constrained REML/profile fitting, GLS, parameter-uncertainty
inference, unclamped adjacent-residual diagnostics with gap summaries, complete-refit bootstrap,
and stable status codes. Reuse one origin-anchored mean design across the like-for-like
comparators. The Python harness may generate and score data, but every estimator call goes through
the release-mode Rust batch target; cross-language hand fixtures must agree.

Run only performance fixtures excluded from the grid, then freeze the exact cell list, attempted
outer seeds, inner bootstrap count, seed hierarchy, concurrency, hardware, wall/core-hour/RSS and
artifact-size limits. Set the final preregistration hash before any coverage outcome is generated.
Include a release-mode 256x256 estimator-only tile dry run at 12/48/96 dates. If the candidate
cannot meet the experimental or projected production budget, narrow the scope before hashing or
stop with a no-go.

#### T53-04: run the frozen synthetic experiment and open the research PR

**Intake:** DR-053-ESTIMATION, DR-053-COMPARATORS, DR-053-PROVENANCE.
**Files:** versioned `validation/results/temporal_covariance_coverage.json` plus CSV/plots only if
generated deterministically, a release-mode workflow end-to-end batch target, schema tests, and
`VALIDATION.md`.

Run the Rust timeseries batch for fixed-factor cells and the workflow batch for end-to-end cells
by preregistration hash against the signed #54 production operator and every frozen
overlap/distance/approximation stratum. Each end-to-end outer seed regenerates raw complex looks,
recomputes #52/#54, and fits the slope from the same realization. Retain every comparator and write
per-cell estimates, attempted/emitted/failed counts, conditional and unconditional coverage,
proper scores, widths, timing, peak RSS, code SHA, #52/#54 identities, and environment. Select a
research candidate only if every supported, deterministic-invalid, and latent out-of-scope gate
passes. A larger average sigma or aggregate pass cannot override one failed cell. Open one
reviewable, unmerged research PR containing T53-01 through T53-04 and stop. T53-05 starts only
after that PR's disposition is explicit and any merge is separately authorized.

#### T53-05: acquire and score a genuinely untouched outer cohort

**Intake:** DR-053-HELDOUT.
**Files:** a new immutable cohort manifest, extensions to `validation/gps_ground_truth.py`, a new
held-out slope-coverage scorer rather than the current 90-percent LOBO calibrator, their tests, and
small committed receipts under `validation/results/temporal_covariance/<cohort-hash>/`. Raw data
and large run directories remain ignored. External fetches remain separately authorized.

The synthetic grid evaluates nominal coverage only under its preregistered generating processes.
The field gate tests directional external validity; a small cohort cannot estimate nominal field
coverage precisely. Its draft primary rule is a one-sided exact cluster-binomial noninferiority
test at 68/90/95 percent:
unacceptable coverage is `nominal - 0.20`, acceptable planning power is calculated at
`nominal - 0.05`, family-wise alpha is 0.05 with Holm adjustment, and power is at least 0.80.
The paired interval score must improve on the conditional baseline and median interval width must
stay below twice that baseline at every level. Freeze the exact test, simultaneous adjustment,
power result, and required evaluable cluster count before acquisition. If the candidate pool or
approved transfer budget cannot meet it, record `not_evaluable` and stop promotion.

Use metadata-only discovery. Name a holdout custodian before freeze; the implementation team must
not receive GNSS outcome series until the Rust binary, scorer, manifest, surplus clusters, and
attrition rules are hashed. Clusters are disjoint across burst, orbit, overlapping footprint,
site, and station IDs. Freeze surplus sites up front, permit no outcome-informed replacement, and
invalidate the cohort after any outcome-informed estimator/scorer change.

First rehearse the exact binary/scorer/receipt path on exposed MMX1 and five-burst data as
development/resource evidence. Exclude those data and Fresno from the outer holdout. Lock burst,
station pair, dates, crop, pre-outcome evaluability rules, acquisition/GNSS hashes, binary SHA,
estimator version, and calibration hash before one-shot unblinding.

The field estimand is the difference between same-frame InSAR and GNSS station-pair slopes.
Obtain the InSAR factor with #54's arbitrary-pair
`difference_covariance(primary_station_pixel, control_station_pixel, common_dates)` operation;
never RSS-combine two raster sigmas.
Project per-epoch GNSS covariance to LOS, include the common epoch-zero reference covariance and
the preregistered GNSS temporal-error model, then fit the GNSS slope by GLS. Score whether zero is
in the combined independent-sensor slope-difference interval; if GNSS slope covariance cannot be
supported, the cluster is `not_evaluable` under the frozen attrition rule. Keep the direct
displacement-series estimator and the velocity-raster scorer as separate evidence.

#### T53-06: obtain independent review and sign the promotion manifest

**Intake:** DR-053-RELEASE, DR-054-BOUNDARY.
**Files:** immutable review receipt, promotion manifest/schema test, `VALIDATION.md`, design,
intake, and this plan only as execution evidence changes.

Give an independent scientific reviewer the design, preregistration, Rust/Python parity evidence,
source, matching signed #52/#54 manifests, synthetic artifact, exposed-data rehearsal, untouched
field artifact, all failures, and resource receipts. Only a review with no unresolved finding may
sign a promotion manifest that contains every artifact hash, supported scope,
estimator/schema version, and code identity. No production config or corrected raster writer
exists in this task. A failed gate or review records the no-go and ends #53 without a corrected
product.

#### T53-07: integrate the signed method and open the promotion PR

**Intake:** DR-053-PROVENANCE, DR-053-RELEASE, DR-052-BOUNDARY, DR-054-BOUNDARY.
**Files:** `crates/dolphin-core/src/config.rs`, its disposition contracts,
`crates/dolphin-workflows/src/displacement.rs`, new `velocity_inference_provenance.rs`,
output/promotion-manifest contracts, a release-mode `temporal_inference_bench`, `VALIDATION.md`,
and `CHANGELOG.md`.

Only after C53-11 is green, consume a calibrated #54 rank-revealing factor for a target and
reference within its signed spatial scope, assemble `V(theta)`, and use an ordinary
positive-definite factorization for solve/whitening/log-determinant.
Reject marginals, two-marginal addition, zero-cross metadata, a bare #52 factor, and
mixed/ambiguous bursts. Emit separately named, paired `velocity_temporal_gls.tif` and
`velocity_sigma_corrected.tif`. Only the signed promotion manifest authorizes inference.
`factor_status=valid` and `spatial_covariance_status=valid` without matching signed receipts do not.
Keep legacy point/sigma products conditional.

Write per-pixel valid-date count, rank, temporal-fit DOF, cadence status, raw adjacent-residual
correlation, pair count and gap summary, fitted `rho_12`, conditioning, reference geometry, and
status rasters. Put the method/schema version, supported cadence predicate, #52 factor/schema and
source identity, #54 method/status/source identities, overlap/window/distance stratum,
approximation bound, bootstrap counts, and #52/#54/#53 review/calibration/promotion hashes in
`velocity_inference_provenance.json`.

Add the explicit uncertainty-method enum only here. Before any output allocation, it verifies the
signed manifest and rejects missing, failed, stale, or scope-mismatched artifacts. It may never
fall back to marginal CRLB or scalar `N_eff`; seasonal/step runs remain conditional-only. Open the
promotion PR only after the wired 12/48/96-date tile benchmark turns C53-13 green; a research
kernel that misses the raster gate cannot be promoted. Stop before merge/release/GroundPulse
pinning.

### #53 verification

```text
cargo test -p dolphin-timeseries --test temporal_covariance_contract
cargo test -p dolphin-timeseries --test timeseries_contract
cargo run --release -p dolphin-timeseries --example temporal_covariance_batch -- \
  --contract validation/fixtures/temporal_covariance_batch.jsonl
cargo run --release -p dolphin-workflows --example temporal_covariance_e2e_batch -- \
  --contract validation/fixtures/temporal_covariance_e2e_batch.jsonl
cargo test -p dolphin-workflows temporal_correlation
cargo test -p dolphin-workflows --test displacement_contract
cargo run --release -p dolphin-workflows --example temporal_inference_bench
oracle/.venv/bin/python -m unittest discover -s validation/tests -p 'test_temporal_covariance*.py'
oracle/.venv/bin/python -m unittest discover -s validation/tests
oracle/.venv/bin/python validation/temporal_covariance_simulation.py \
  --prereg validation/temporal_covariance_preregistration.json \
  --rust-batch target/release/examples/temporal_covariance_batch \
  --rust-e2e-batch target/release/examples/temporal_covariance_e2e_batch \
  --output validation/results/temporal_covariance_coverage.json
oracle/.venv/bin/python validation/score_temporal_covariance_holdout.py \
  --cohort validation/temporal_covariance_holdout.json \
  --results-root validation/results/temporal_covariance
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --all-targets -- -D warnings
cargo test --workspace
git diff --check
```

The simulation and held-out commands must validate artifact hashes and return distinct pass,
fail, and not-evaluable exit states.

## Coverage audit

| Intake ID | Scheduled tasks |
|---|---|
| DR-052-STATE | T52-01 through T52-04 |
| DR-052-MEMORY | T52-01, T52-03, T52-06 |
| DR-052-GAUGE | T52-01 through T52-04 |
| DR-052-STATUS | T52-04, T52-05 |
| DR-052-ANALYTIC | T52-02 |
| DR-052-FAIL | T52-02 through T52-04 |
| DR-052-BOUNDARY | T52-01, T52-05, T52-06, T54-07, T53-07 |
| DR-054-QUANTITY | T54-01, T54-03 |
| DR-054-BOUNDED | T54-01, T54-03 through T54-06 |
| DR-054-ANALYTIC | T54-02 |
| DR-054-GEOMETRY | T54-02, T54-04, T54-05 |
| DR-054-VALIDATION | T54-06, T54-07 |
| DR-054-BOUNDARY | T54-01, T54-05 through T54-07, T53-06, T53-07 |
| DR-053-MODEL | T53-01 |
| DR-053-ESTIMATION | T53-02 through T53-04 |
| DR-053-PREREG | T53-02, T53-03 |
| DR-053-COMPARATORS | T53-02 through T53-04 |
| DR-053-HELDOUT | T53-05 |
| DR-053-PROVENANCE | T53-04, T53-07 |
| DR-053-RELEASE | T53-06, T53-07 |

All intake IDs have one scheduled disposition. dolphinRust has no UI, so no paired UI task
applies. Future GroundPulse consumption is out of this plan and requires a separate `eo` intake
only after engine, calibration, and review gates pass.

## Coding-agent execution contract

```text
Execute md/plans/open-issues-uncertainty-2026-08-23.md in PR #55 reconciliation, T52, T54, then
T53 gate order. Refresh issues, PRs, main CI, branch divergence, credentials, and disk before
starting. Preserve user-owned handoff files. Write each named analytic/fail-closed contract red
before its production slice. Never substitute marginal CRLB, two-marginal addition, an arbitrary
Cholesky product, or scalar N_eff for a failed or missing covariance method. Keep acquisition 0 as
an exact gauge; retain the #52 temporal factor and replay descriptor; and compute #54 influence in
a byte-capped, reference-specific pass without a date-squared spatial cube, all-pixel influence
matrix, or pixel-pair object. Freeze and hash #54/#53 preregistrations before outcomes; treat
existing real datasets as development-only. Run focused and full validation, open only the
reviewable PR authorized for the active gate, and stop before merge, release, publication,
external data acquisition, eo mutation, submodule pinning, deployment, or any scientific/serving
claim.
```
