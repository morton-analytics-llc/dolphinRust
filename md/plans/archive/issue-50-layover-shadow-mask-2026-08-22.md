# Implementation plan: issue #50 layover/shadow masks and config accountability

**Archived:** 2026-09-03. PR #51 merged as `be07c21`; issue #50 is closed. The status line below predates that merge.

**Status:** implementation complete in unmerged PR #51; local and GitHub CI verification green
2026-08-22.
**Intake:** `md/intake/issue-50-layover-shadow-mask-2026-08-22.md`.
**Live queue:** one open issue (#50), no open pull requests, `main` CI green at
`9edd192949552aa4ee2f7b4f549cf387868b081d` (run 32103681198).
**Execution boundary:** one contract-first, unmerged dolphinRust PR. No merge, release,
GroundPulse submodule bump, deployment, or `eo` mutation.

## Objective

Resolve #50 without changing the meaning of the later workflow `mask_file`:

1. make `layover_shadow_mask_files` affect per-burst phase linking with explicit mapping,
   alignment, polarity, tiling, and incremental-update behavior; and
2. make every public config field visibly consumed or rejected so a caller cannot mistake
   serialization compatibility for implemented behavior.

## Current state and selected contract

- `layover_shadow_mask_files` is declared, defaulted, parsed, and serialized in
  `crates/dolphin-core/src/config.rs`; no production code reads it.
- `mask_file` is read only in the downstream unwrap path when `zero_where_masked` is set.
  It cannot replace a layover/shadow mask that must affect covariance and phase linking.
- Pinned Python dolphin v0.35.0 (upstream commit
  `e567e554300f9bb2c6c4c49358d41876ce81e5a7`) treats 0 as invalid and nonzero as valid,
  combines raster nodata with mask invalidity, and supplies the mask before phase linking.
  Its multi-burst resolver silently takes the first matching mask. dolphinRust will fail on
  missing, duplicate, extra, or unmapped masks instead.
- Python's phase-link output is nodata only when every native pixel contributing to a
  stride cell is invalid. A partly valid stride cell remains evaluable.
- Setting masked Rust samples to NaN is necessary but insufficient: covariance currently
  converts non-finite samples to zero, and `TiledOutput::place` marks every computed output
  pixel valid. The looked validity mask must therefore be propagated explicitly.
- `dolphin_io::read_aligned_raster_window` already enforces CRS, posting, origin, and full
  coverage without resampling and folds GDAL nodata/mask-band invalidity to zero.
- GroundPulse currently passes OPERA STATIC products as geometry inputs but does not extract
  `/data/layover_shadow_mask` into per-burst rasters or populate this config field. Its water
  mask has opposite polarity and is downstream; it is not a substitute.

The selected #50 resolution is to wire the option. Removing or rejecting it would make the
engine truthful but would leave the identified infrastructure-monitoring need unmet.

## Technical requirements

### R1 — explicit per-burst resolution (DR-050-MASK)

- An empty list is the identity path and must preserve every existing output bit-for-bit.
- For OPERA inputs, extract the burst ID from every active CSLC group and mask filename.
  Require exactly one mask for each active burst and reject missing, duplicate, extra, or
  unparseable mappings before phase-linking I/O begins.
- For the single/non-OPERA group, require exactly one mask when the list is nonempty.
- Record resolved mask paths in the same deterministic burst order as `group_by_burst`.
- Accept single-band native-grid masks backed by the GDAL `GTiff` driver only. Direct OPERA
  STATIC HDF5 extraction belongs to the GroundPulse follow-up, not this engine PR.

### R2 — raster and polarity contract (DR-050-MASK)

- Read masks on the source CSLC grid with exact EPSG, geotransform, rotation, posting, and
  coverage checks. Do not reproject, resample, crop past coverage, or choose a fallback mask.
- Interpret stored zero, non-finite values, raster nodata, and GDAL-invalid pixels as invalid;
  every finite nonzero stored value is valid.
- Apply bounded source windows and tile windows to the resolved full-resolution mask using
  the same offsets as the CSLC reader.
- An all-valid mask must take the same numerical path as an empty mask after resolution and
  produce bit-identical output.

### R3 — pre-covariance masking and looked validity (DR-050-MASK)

- Compute SHP neighbors from the raw real acquisitions first. Do not let the terrain mask
  change the SHP amplitude statistics, and do not reproduce dolphin v0.35.0's mixed-block
  PS-selection omission.
- After SHP selection, invalidate the masked samples across real and carried inputs before
  covariance/phase linking. This excludes them from the covariance sums while preserving
  the existing rectangular and SHP-aware kernels.
- Derive one output-grid validity layer from the native mask: an output stride cell is
  invalid only when all native pixels in that cell are invalid. Keep edge-cell behavior
  consistent with `Strides::out_shape`.
- Set linked phase and every optional phase-link quality layer to NaN where looked validity
  is false. Keep `validity_mask` false through tile assembly, burst stitching, downstream
  products, and artifact writing.
- Skip covariance work for a wholly invalid tile without classifying any acquisition as
  globally missing. Acquisition completeness statistics must be computed from the raw CSLC
  tile before mask application.
- If a processed burst's resolved mask has no valid pixel in its bounded source window, return
  a path-specific error before covariance. Do not route this case through the
  `linked_tiles == 0` missing-input guard or emit an all-nodata single-burst product.

### R4 — batch, bounded, multi-burst, and reference behavior (DR-050-MASK)

- Use the same mask contract in whole-burst, tiled, bounded, and multi-burst runs. Preserve
  tiled/whole and bounded/full overlap identity.
- Preserve the existing finite-support overlap rule: a later finite burst value replaces an
  earlier one, while later invalid/nodata does not erase earlier valid support. Frame validity
  is true when any overlapping burst contributes valid support and never extends outside a
  contributing burst footprint.
- Reject a configured reference point that is invalid under the combined source/terrain
  validity mask. Automatic reference selection must continue to choose only finite pixels.
- Keep the existing downstream `mask_file` independent. If both masks are supplied, terrain
  invalidity enters at phase linking and the later mask may only remove additional pixels.
- The final validity fanout must null displacement, temporal coherence, phase-linking
  coherence, CRLB, closure, unwrapped phase, velocity, uncertainty, and correction layers
  wherever the terrain mask is invalid.

### R5 — resumable and incremental identity (DR-050-MASK)

- Add mask-aware sequential entry points while retaining the current unmasked wrappers for
  source compatibility and focused kernel tests.
- Store the resolved burst mask identity and source-grid contract in `BurstState`.
- `update_displacement` must require the same mask path/mapping, grid, polarity contract,
  and content identity as the state-producing run. Identity binds the primary raster and
  every effective backing file reported by GDAL, including any active mask, PAM, world-file,
  HFA auxiliary, or projection sidecar. Reject removal, replacement, remapping, or mutation
  before consuming new acquisitions.
- Carry the native/stride validity needed to mask sealed products and compressed SLCs so an
  incremental update remains bit-identical to a fresh full rerun.
- Bounded updates may retain the current full bounded recompute, but must resolve and verify
  the same mask before recomputation.

### R6 — exhaustive public-config accountability (DR-050-GUARD)

- Replace the narrowly named `validate_uncertainty_options` with one `validate_config`
  entry guard called by batch, resumable, and update paths before source layout or raster I/O.
- Perform a one-time, field-by-field audit of every public config struct. Give each full YAML
  path one disposition: `Consumed`, `Conditional`, or `CompatibilityOnly`.
- `Consumed` and `Conditional` entries require a named behavior contract and a production
  reader. `CompatibilityOnly` means the dolphin default may round-trip, but every non-default
  value must return an actionable config error before processing.
- The disposition registry must reference contract IDs from a checked catalog; the coverage
  test fails for a consumed/conditional entry whose contract ID is absent.
- Build the class guard from exhaustive Rust destructuring of every public config struct
  without `..`, plus a unique full-path disposition registry. Adding or renaming a field
  must fail compilation or the registry test until it is classified.
- Do not use token search as the guard. Repeated names such as `block_shape` and `alpha`,
  test-only reads, destructuring, and comments create false results.
- Treat the preliminary set of apparent no-reader fields only as audit candidates. Do not
  reject one until its runtime producers/consumers and existing YAML fixtures have been
  checked. In particular, real GroundPulse-shaped YAML may set worker fields away from their
  Rust defaults.
- Update Rustdoc for compatibility-only fields so accepted serialization is not described as
  implemented behavior.

## Constraints and guardrails

- Write the pinned-oracle and analytic behavior contracts red before production changes.
- Preserve the dynamic, sequential architecture; do not materialize a whole-burst mask cube
  or an `nslc² * area` covariance cube.
- Keep GDAL/HDF5 access synchronous and serialize HDF5-touching tests through the crate's
  existing lock.
- No resampling, implicit burst ordering, first-match behavior, silent mask drop, or
  confident finite output for an all-invalid cell.
- Do not change PS selection, SHP statistics, water-mask semantics, recommended-mask
  derivation, or downstream report text in this PR.
- Preserve existing defaults and unmasked numerical output bit-for-bit.
- Preserve unrelated local work. Stage only issue #50 files.
- Commits include `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Backlog automation may open one verified `automation-pr`; stop before merge, release,
  publication, or GroundPulse pinning.

## Test contract

| ID | Contract | Location | Proof |
|---|---|---|---|
| C01 | A fresh v0.35.0 fixture records 0/nonzero/nodata polarity, pre-phase-link application, and partially versus wholly invalid stride cells. | new `oracle/gen_layover_shadow_mask.py` plus committed fixtures | The scientific reference is reproducible and not inferred from config docs. |
| C02 | Empty and all-valid masks produce bit-identical linked and final outputs to the current unmasked path. | new `crates/dolphin-workflows/tests/layover_shadow_mask_contract.rs` | Enabling support does not perturb default runs. |
| C03 | On `ShpMethod::Rect`, a high-amplitude contaminant inside an invalid pixel cannot affect covariance/linking. On GLRT/KS, neighbor selection with a terrain mask matches neighbor selection from the same raw stack without that mask. | same plus focused sequential/SHP contracts | Covariance exclusion and raw-data SHP ordering are tested without conflating their effects. |
| C04 | Stored 0 and GDAL nodata are invalid, nonzero is valid, and CRS/posting/origin/coverage mismatch fails before compute. | `crates/dolphin-io` aligned-raster tests plus workflow contract | Polarity and no-resampling behavior are explicit. |
| C05 | A partly valid stride cell remains valid; a wholly invalid cell produces NaN layers and false validity, including edge cells. | layover/shadow contract | Look reduction matches the pinned behavior and cannot emit confident nodata. |
| C06 | Missing, duplicate, extra, and unparseable OPERA burst masks fail; correct masks follow their burst rather than list order. Single/non-OPERA input accepts exactly one. | layover/shadow and `multiburst_contract.rs` | Per-burst mapping is deterministic and fail-closed. |
| C07 | Whole/tiled and bounded/full-overlap outputs, validity, and tile-seam behavior agree. A wholly invalid tile is skipped without triggering the missing-acquisition guard; a wholly invalid processed burst fails with its mask path before covariance. | displacement tiling contract | Masking preserves spatial and acquisition-completeness invariants. |
| C08 | Batch, resumable, incremental, and fresh-full outputs agree with an unchanged mask; changed, removed, remapped, or mutated masks fail. | `nrt_incremental_contract.rs` and `nrt_displacement_contract.rs` | NRT cannot mix state from different terrain-validity contracts. |
| C09 | A configured invalid reference fails; automatic reference avoids invalid pixels; every final output layer is NaN/invalid at terrain-invalid cells. | layover/shadow and displacement contracts | Downstream layers cannot resurrect masked pixels. |
| C10 | Exhaustive destructuring and the full-path registry cover every public config field exactly once; every consumed/conditional entry names an ID in the checked behavior-contract catalog. Adding a synthetic field makes the contract fail until dispositioned. | new `crates/dolphin-core/tests/config_field_disposition_contract.rs` | New fields and unsupported claims of behavior are mechanically gated. |
| C11 | Every compatibility-only non-default tested by the audit returns a path-specific config error from all three workflow entries before I/O. | config/workflow validation contracts | Round-trip compatibility cannot become a silent no-op. |
| C12 | Existing config round trips, oracle contracts, default GroundPulse-shaped config, and the full workspace remain green. | existing suites | The guard does not relabel current working behavior by accident. |

## Task manifest

### T01 — pin the mask oracle and turn the behavior contract red

**Files:** new `oracle/gen_layover_shadow_mask.py`, its minimal committed fixtures, and new
`crates/dolphin-workflows/tests/layover_shadow_mask_contract.rs`.

Record exact v0.35.0 package/source identity and C01. Add the Rust contracts for empty,
all-valid, contaminant, polarity, and stride-cell validity. Capture the expected red failures
before production edits.

### T02 — resolve masks and enforce native-grid I/O

**Files:** `crates/dolphin-workflows/src/burst.rs`,
`crates/dolphin-workflows/src/displacement.rs`, focused `crates/dolphin-io` tests, and only a
small `dolphin-io` helper change if the existing aligned reader cannot expose the required
window/identity metadata.

Implement R1-R2 and C04/C06. Resolve the complete mapping before phase-link work, then read
only the bounded/tiled mask window that corresponds to each CSLC tile.

### T03 — apply the mask in sequential phase linking

**Files:** `crates/dolphin-workflows/src/sequential.rs`,
`crates/dolphin-workflows/src/displacement.rs`, and the T01 contract.

Add mask-aware wrappers, preserve raw-data SHP, invalidate covariance samples, reduce native
validity to the output stride, fail a wholly invalid burst explicitly, and make
`TiledOutput::place` copy explicit validity instead of filling `true`. Turn C02/C03/C05
green without changing the unmasked path.

### T04 — propagate validity through batch, bounds, bursts, and final outputs

**Files:** `crates/dolphin-workflows/src/displacement.rs`, existing multiburst/displacement
contracts, and only the output writers touched by the failing C09 fanout.

Implement R4, tile skipping, reference rejection, stitch behavior, and final-layer masking.
Turn C07/C09 green. Keep the later `mask_file` path independent.

### T05 — make resumable and incremental runs mask-stable

**Files:** `crates/dolphin-workflows/src/sequential.rs`,
`crates/dolphin-workflows/src/displacement.rs`, `nrt_incremental_contract.rs`, and
`nrt_displacement_contract.rs`.

Persist and validate mask identity, carry validity with sequential state, and turn C08 green
for unchanged and changed-mask cases.

### T06 — close the config-reader class, not only this field

**Files:** `crates/dolphin-core/src/config.rs`, new
`crates/dolphin-core/tests/config_field_disposition_contract.rs`, existing config contracts,
and the workflow validator/contracts in `crates/dolphin-workflows`.

Audit every public field, add the exhaustive full-path disposition registry, rename/extend the
entry validation guard, reject every proven compatibility-only non-default, and update its
Rustdoc. Turn C10/C11/C12 green. Do not mass-reject the preliminary candidates without
reader and fixture evidence.

### T07 — document, verify, and open one unmerged PR

**Files:** `CHANGELOG.md` plus narrow config/workflow documentation and this intake/plan if
execution evidence changes them.

Document mask polarity, mapping, alignment, stage ordering, NRT identity, and explicit
GroundPulse non-adoption. Run all gates, review the diff against #50, open one verified
`automation-pr`, and stop. File/link the named `eo` follow-up for GP-050-TRUTH only when the
engine PR/release re-entry gate is met; do not edit `eo` from the automated PR.

## Validation

Run focused gates as their tasks land, then the full set:

```text
oracle/.venv/bin/python oracle/gen_layover_shadow_mask.py
cargo test -p dolphin-io aligned_mask
cargo test -p dolphin-core --test config_contract
cargo test -p dolphin-core --test config_field_disposition_contract
cargo test -p dolphin-workflows --test layover_shadow_mask_contract
cargo test -p dolphin-workflows --test sequential_contract
cargo test -p dolphin-workflows --test shp_wiring_contract
cargo test -p dolphin-workflows --test multiburst_contract
cargo test -p dolphin-workflows --test nrt_incremental_contract
cargo test -p dolphin-workflows --test nrt_displacement_contract
cargo test -p dolphin-workflows --test displacement_contract
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
oracle/.venv/bin/python -m unittest discover -s validation/tests
oracle/.venv/bin/python -m compileall -q validation
git diff --check
```

Completion also requires the PR receipt to record the red/green contract order, exact oracle
identity, issue link, relevant commands, and the explicit statement that GroundPulse has not
yet populated or production-verified the new field.

## Execution receipt — 2026-08-22

| Task | Result |
|---|---|
| T01 | Complete in red-contract commit `21b6a73`. `cargo test -p dolphin-workflows --test layover_shadow_mask_contract` failed because `run_sequential_masked` and `SequentialOutput.validity_mask` did not exist. The pinned generator records dolphin 0.35.0 at upstream `e567e554300f9bb2c6c4c49358d41876ce81e5a7`. |
| T02-T04 | Complete. Per-burst resolution, exact single-band native-grid GTiff reads, pre-covariance masking, tiled/whole/bounded behavior, and final validity fanout are green. |
| T05 | Complete. Batch/resumable/update paths bind mapping, primary bytes, GDAL's effective dataset descriptor and backing files, full valid-pixel semantics, and sequential validity state. |
| T06 | Complete. The registry covers all 99 public config paths: 30 consumed, 42 conditional, and 27 compatibility-only. Non-default compatibility-only values and unsupported variants fail before workflow I/O. |
| T07 | Complete. PR #51 is open, labeled `automation-pr`, unmerged, and clean against `main`; CI run 32597447119 passed. |

Green evidence:

- `oracle/gen_layover_shadow_mask.py` reproduced native and stride-2 validity from the pinned
  Python oracle.
- Focused mask, aligned-I/O, config, displacement, multi-burst, sequential, SHP, resumable,
  and incremental contracts pass.
- `cargo fmt --all -- --check`, `cargo check --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` pass;
  the workflow library reports 109/109 unit tests.
- `python -m unittest discover -s validation/tests` passes 33/33, validation bytecode
  compilation passes, and `git diff --check` is clean.
- Independent config and implementation reviews returned no findings after the GDAL
  effective-dataset and single-band regressions were added.
- GitHub CI run 32597447119 passed formatting, workspace check, clippy, Rust tests, and
  Python validation on PR #51.

GroundPulse has not extracted, populated, or production-verified
`layover_shadow_mask_files`. Its checked-in real-dolphin YAML also sets
`worker_settings.threads_per_worker: 6`, which the new fail-before-I/O guard rejects until the
caller is normalized. GP-050-TRUTH remains deferred behind engine merge, release, and pin
selection; no `eo` files were changed.

## Resolved decisions

1. Wire `layover_shadow_mask_files`; do not remove or reject the requested capability.
2. Match pinned polarity and pre-covariance behavior, but fail closed instead of taking the
   first matching burst mask or tolerating incomplete mappings.
3. Keep PS/SHP selection based on raw real acquisitions and contract the terrain mask to phase
   linking; do not copy the pinned mixed-block PS omission.
4. Use native-grid rasters only in dolphinRust. OPERA STATIC extraction and GroundPulse
   configuration are GP-050-TRUTH after engine release selection.
5. Guard public config with exhaustive type-aware dispositions and runtime validation, not a
   lexical reader search.
6. One unmerged PR is the terminal state for this plan. Merge, release, pin, deploy, and fresh
   terminal-artifact proof remain separate human-gated work.

## Coding-agent execution contract

```text
Execute md/plans/issue-50-layover-shadow-mask-2026-08-22.md in T01-T07 order.
Refresh the live issue/PR/main-CI state before starting and stop if main is red or an
automation-pr already owns #50. Write C01-C12 red before their production slices, preserve
the unmasked bit-identity path, and require exact per-burst native-grid masks. Do not alias
mask_file, water masks, recommended masks, or STATIC geometry inputs. Audit every public
config field before rejecting any apparent no-reader candidate. Run focused tests after each
slice and the full fmt/check/clippy/test/Python gates before opening one unmerged PR. Do not
merge, release, publish, modify eo, bump its submodule, deploy, or claim production use.
```
