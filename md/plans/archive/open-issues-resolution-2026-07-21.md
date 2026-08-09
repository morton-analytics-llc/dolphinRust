# Implementation plan: resolve the open dolphinRust issue queue

**Status:** implemented and verified 2026-07-21 from the live GitHub queue.
**Open issues:** #7, #8, and #9.
**Execution model:** contract-first, one compiling slice at a time; direct push to `main`
was explicitly requested after review. No release, publication, or GroundPulse submodule bump.

## Objective

Resolve the entire current issue queue without aliasing distinct scientific metrics:

1. implement a bounded phase-linking coherence metric from the coherence matrix and
   expose it separately from temporal coherence (#7 and #9);
2. determine the pinned Python dolphin behavior for all-non-finite inputs and make
   dolphinRust's behavior explicit and safe (#8);
3. leave every change supported by analytic contracts, oracle evidence, workflow/output
   tests, memory measurements, and the workspace verification gates.

## Design summary

### Current facts

- `temporal_coherence` is a 2D estimator-fit metric. It is written as
  `temporal_coherence.tif` and is currently also named as phase-linking coherence in
  `geometry_provenance.json`; that alias is false provenance.
- The fused phase-linker already owns each per-pixel coherence matrix, so a coherence-
  magnitude reduction belongs there and must not rematerialize the discarded
  `nslc^2 * area` cube.
- Pinned dolphin v0.35.0 computes
  `avg_coh_per_date = abs(C_arrays).mean(axis=3)`, but its public `avg_coh` output is
  `argmax(avg_coh_per_date, axis=2)`: a 2D reference-date index. Issue #7's current
  description of the public output as a per-date coherence cube is therefore inaccurate.
- GroundPulse needs one bounded 2D phase-linking-coherence raster and aggregate, not a
  reference-date index. It must continue treating the value as unknown until a genuine
  artifact exists.
- An all-non-finite window currently becomes a zero coherence matrix and can report
  `temporal_coherence=1.0` and zero displacement. The pinned oracle has not yet been run
  on that exact contract.

### Proposed metric contract for #7 and #9

Retain the oracle-equivalent intermediate, named unambiguously:

```text
average_coherence_per_date[i, r, c] = mean_j(abs(C[i, j, r, c]))
phase_linking_coherence[r, c] = nanmean_i(average_coherence_per_date[i, r, c])
```

The diagonal is included to match dolphin's internal calculation. Values are finite and
bounded in `[0, 1]` when the contributing coherence entries are valid. In sequential
processing, retain/reduce real acquisition-date bands only; carried compressed SLC bands
must not appear as GroundPulse observation dates. Weight the final scalar by contributing
real-date bands, not equally by ministack, so a short trailing ministack is not over-weighted.

The core `FusedEstimate` may expose the optional per-date array for parity and testing. The
workflow should reduce it promptly to one optional 2D `phase_linking_coherence` layer so it
does not retain another full-date cube across the whole burst.

## Technical requirements

### R1. Correct the issue contract before implementation

- Record the pinned v0.35.0 source behavior in #7 or its implementing PR.
- Do not expose the oracle's integer `argmax` as a coherence value.
- Use `average_coherence_per_date` for the floating-point intermediate and
  `phase_linking_coherence` for the 2D workflow product.
- Treat #7 and #9 as one vertical feature: #7 supplies the numerical kernel and oracle
  parity; #9 supplies sequential/tiled/GPU propagation, raster output, and provenance.

### R2. Add the optional core metric without restoring the covariance cube

- Add a small pure function in `dolphin-phaselink` that reduces one `N x N` coherence
  matrix to `N` per-date means.
- Add `compute_average_coherence` to `FusedParams`, defaulted off by callers unless the
  workflow option requests the product.
- Add `average_coherence_per_date: Option<Array3<f64>>` to `FusedEstimate`.
- Compute the vector inside `fused_from_coh` while the matrix is already resident and
  pack only the `N * area` optional result.
- Implement the same result in the GPU-resolved staged path from its materialized
  coherence matrix. CPU fused and staged/GPU-resolved outputs must agree within the
  existing covariance tolerance.
- Mirror dolphin YAML with `phase_linking.calc_average_coh`, default `false`, and preserve
  round-trip compatibility.

### R3. Propagate and reduce through sequential, tiled, multiburst, and NRT paths

- Thread the compute flag through `SequentialConfig` and every `FusedParams` constructor.
- Slice away carried compressed bands before accumulating real-date coherence.
- Maintain per-pixel finite sum and count across real dates; derive the 2D product from
  those accumulators. Do not average already-averaged ministack scalars.
- Persist sufficient sum/count state in `SequentialState` so incremental NRT output is
  bit-identical to a full rerun.
- Tile and multiburst stitching must copy/stitch the new 2D layer using the same output
  rectangles and georeferencing as temporal coherence.
- `DisplacementOutput.phase_linking_coherence` is `Option<Array2<f64>>`; temporal
  coherence remains unchanged and mandatory for compatibility.

### R4. Write an honest artifact and provenance contract

- When enabled and evaluable, write `phase_linking_coherence.tif` beside
  `temporal_coherence.tif`; never republish or rename the temporal raster.
- Change geometry provenance so `phase_linking_coherence` is optional/absent when the
  metric was not requested or could not be evaluated. Bump the provenance schema/method
  version if the serialized field type or meaning changes.
- Document the exact formula, inclusion of the diagonal, real-date-only sequential
  reduction, units (dimensionless), range, config flag, and distinction from estimator-fit
  temporal coherence.
- Preserve existing temporal-coherence filenames and API fields.
- GroundPulse parsing, persistence, deployment, and the dolphinRust submodule bump remain
  separate `eo` work.

### R5. Measure the added memory and runtime cost

- Measure flag-off versus flag-on peak RSS and wall time with identical input/config on
  the existing phase-linking benchmark at current reduced strides and at stride 1.
- Report array-theory overhead (`8 * n_real_dates * out_rows * out_cols` bytes only where
  the per-date core result is retained) separately from observed process RSS.
- Confirm the workflow reduction does not retain the per-date array after each
  tile/ministack and remains inside GroundPulse's 60 GiB worker envelope on the nominated
  representative AOI.
- Do not describe a synthetic or local memory result as deployed terminal-artifact proof.

### R6. Resolve all-non-finite behavior from oracle evidence

- Add a deterministic Python oracle generator that runs pinned dolphin v0.35.0 on an
  all-NaN SLC stack and records linked phase, temporal coherence, average-coherence
  intermediate/reference behavior, warnings, exceptions, and exit status.
- If dolphin returns nodata/NaN or fails loudly, write the red Rust parity contract and
  match it.
- If dolphin also returns confident-looking finite output, stop for explicit user approval
  of a forward divergence. The recommended divergence is: a window with zero finite input
  samples yields absent/NaN quality and displacement values; partially valid windows retain
  dolphin's existing finite-or-zero masking behavior.
- Define whole-run behavior independently: an entirely non-evaluable output should return
  an actionable error/receipt, while isolated nodata windows remain geospatial nodata.
- Do not fold the `eo` wrapper's crop-level rejection into this repo's issue.

## Constraints and guardrails

- Write each analytic or oracle contract red before its implementation.
- Read the target crate's `CLAUDE.md` before editing it.
- Preserve fused/staged numerical agreement, tiled/whole identity, NRT full/incremental
  identity, config round trips, and existing temporal-coherence compatibility.
- Do not materialize the `nslc^2 * area` coherence cube on the CPU fused path.
- Do not name an integer reference-date index, temporal coherence, or an unavailable layer
  as phase-linking coherence.
- Unknown/non-evaluable values remain absent or NaN, never fabricated as zero or one.
- Before starting #7, check for an existing `automation-pr`. Its `backlog-ready` label can
  trigger scheduled work based on the currently inaccurate issue description; review or
  supersede that work rather than creating a duplicate PR.
- No merge, tag, release, publication, GroundPulse mutation, or submodule bump is included.

## Test contract

| Contract | Location | Behavior proved |
|---|---|---|
| Closed-form per-date means | `crates/dolphin-phaselink/tests/quality_contract.rs` | A constructed Hermitian matrix produces the exact row means of `abs(C)`, including the diagonal, and values stay in `[0,1]`. |
| Distinct metrics | same | A fixture produces different phase-linking and temporal coherence values; neither can be aliased while tests pass. |
| Pinned-oracle intermediate | new/extended `oracle/gen_quality.py` fixture + quality contract | Rust per-date means match v0.35.0's internal `abs(C).mean(axis=3)` to about `1e-4`; the oracle's public argmax behavior is recorded separately. |
| Fused/staged parity | `crates/dolphin-phaselink/tests/fused_contract.rs` | Optional per-date output agrees across CPU fused and staged paths and is absent when disabled. |
| Config contract | `crates/dolphin-core/tests/config_contract.rs` | `calc_average_coh` defaults off, parses from dolphin YAML, and round-trips. |
| Real-date-only reduction | sequential unit/contract tests | Compressed carry bands are excluded and uneven ministacks use per-date sum/count weighting. |
| NRT identity | `crates/dolphin-workflows/tests/nrt_incremental_contract.rs` | Incremental and full reruns produce identical phase-linking coherence. |
| Tiled/whole identity | displacement tiling contract | The new layer is identical across tile boundaries and optional allocation does not change existing outputs. |
| Multiburst geometry | `crates/dolphin-workflows/tests/multiburst_contract.rs` | The new 2D layer stitches to the full frame with the expected shape/georeferencing. |
| Raster separation | displacement/NISAR output contracts | Both distinct TIFFs exist when enabled, temporal compatibility remains, and the phase-linking file is absent when disabled. |
| Provenance honesty | `crates/dolphin-workflows/tests/geometry_provenance_contract.rs` | Provenance points to the genuine new raster or explicitly records absence; it never points to `temporal_coherence.tif`. |
| Memory receipt | benchmark receipt/documentation | Flag-off/on RSS and wall deltas are measured at reduced stride and stride 1 with exact commands/config. |
| All-NaN oracle | new oracle generator/fixture | Pinned dolphin's actual result, warning/error behavior, and versions are reproducible. |
| All-NaN Rust behavior | covariance/quality plus workflow contract | Zero-finite windows follow the oracle or approved forward-divergence contract without reporting perfect confidence. |
| Partial-NaN regression | covariance contract | Partially valid windows preserve existing masking/parity and finite output where evaluable. |

## Implementation plan

### Phase 0 — reconcile issue definitions and reserve the work

1. Re-fetch the live issue/PR queue and ensure no `automation-pr` exists.
2. Post the pinned-source correction to #7 before coding; cross-link #9.
3. Confirm the proposed scalar formula and optional `calc_average_coh` default. If rejected,
   update this plan before writing contracts.

Exit: #7 and #9 share one unambiguous metric/output contract.

### Phase 1 — core red contracts, then the phase-linking kernel (#7)

1. Add the analytic and pinned-oracle fixtures; demonstrate red failure.
2. Implement the per-matrix reduction and optional fused/staged output.
3. Add config parsing/default/round-trip support.
4. Run focused phaselink/core tests, then keep the workspace compiling.

Exit: the numerical metric is proven, optional, and does not restore the covariance cube.

### Phase 2 — workflow, artifact, provenance, and memory (#9)

1. Add sequential real-date sum/count reduction and NRT state.
2. Thread the optional 2D layer through tile and multiburst assembly.
3. Add `DisplacementOutput`, raster writing, and versioned honest provenance.
4. Turn the workflow/output/provenance contracts green.
5. Run and document the reduced-stride memory/runtime A/B.

Exit: dolphinRust emits a genuine `phase_linking_coherence.tif`, distinct from temporal
coherence, and supplies a measured resource receipt. One reviewed PR may close both #7 and
#9 because they are one vertical contract; keep commits separable as core then workflow.

### Phase 3 — oracle decision and degenerate-input behavior (#8)

1. Generate and inspect the pinned all-NaN oracle receipt.
2. Select the parity branch or obtain user approval for the documented forward divergence.
3. Write the selected red analytic/workflow contracts.
4. Implement the smallest guard that distinguishes zero-finite from partially valid windows.
5. Document behavior, limitations, and any divergence from dolphin.

Exit: no all-non-finite input can silently masquerade as perfect confidence, and issue #8
has reproducible oracle evidence.

### Phase 4 — final verification and handoff

1. Run focused contracts after each phase.
2. Run `cargo fmt --all -- --check`.
3. Run `cargo check --workspace`.
4. Run `cargo clippy --all-targets --workspace -- -D warnings`.
5. Run `cargo test --workspace` plus oracle generators/contracts and `git diff --check`.
6. Review the diff for metric naming, output compatibility, memory claims, and scientific
   boundaries; open an unmerged PR and stop.

## Validation

Completion requires:

- all contracts above green;
- exact oracle version and fixture-generation commands recorded;
- flag-off/on RSS and wall-time results recorded with host/config details;
- no temporal-coherence alias remains in dolphinRust provenance;
- open issue references resolve through the PR(s), with #7/#9 scientifically reconciled;
- no claim that local fixtures, benchmarks, or output readiness prove GroundPulse deployment.

## Resolved decisions

1. **#7 correction:** implemented the bounded floating-point internal row means; the returned
   integer argmax is recorded as a reference-date selector and is not exposed as coherence.
2. **#8 parity:** pinned dolphin v0.35.0 raises `PhaseLinkRuntimeError`; Rust now rejects the
   same degenerate stack, so no forward-divergence approval was needed.
3. **Execution ownership:** no `automation-pr` existed; this manual implementation superseded
   unattended pickup and resolved #7 together with its dependent workflow issue #9.

## Coding agent prompt

```text
Execute md/plans/open-issues-resolution-2026-07-21.md in dolphinRust one phase at a time.
Start by refreshing the live issue/PR queue and correcting the pinned-v0.35 avg_coh contract
on issue #7. Write every analytic/oracle contract red before implementation. Preserve the
CPU fused memory architecture, temporal-coherence compatibility, tiled/whole identity, and
NRT full/incremental identity. Do not alias temporal coherence or dolphin's integer argmax as
phase-linking coherence. Stop for user approval if issue #8 requires a forward divergence.
Run the focused tests after each slice and the full workspace fmt/check/clippy/test gates before
opening an unmerged PR. Do not merge, release, publish, modify eo, or bump its submodule.
```
