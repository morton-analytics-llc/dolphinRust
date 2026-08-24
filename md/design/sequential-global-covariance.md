# Sequential global covariance: T52-01 design gate

**Status:** `sequential_srif_v1` rejected at T52-01; replacement
`sequential_source_dag_v1` approved for contract-first implementation on 2026-08-23.

**Baseline:** `da9116c689d1b61c0ea8a9d145de6a57ffc28563`, the green combined tree after
PR #55.

## Required quantity

Issue #52 requires one acquisition-0-gauge covariance over all retained real dates that
propagates compressed-reference uncertainty across ministacks. The product must remain bounded
without a dense `n_dates x n_dates x area` allocation and must stay disconnected from velocity
inference until issues #54 and #53 pass.

The proposed block payload was:

```text
x_b = F_b s_b + L_b eta_b
c_b = w_b' [s_b; x_b]
```

with at most `K` carried scalars in `s_b` and per-pixel storage
`m*K + m*(m+1)/2 + m+K` floating-point values. That state does not match the
production estimator.

## Production dependency

For an upstream looked cell `g(q)` and native sample `q`, compression is

```text
S_q = sum over real t of z_t(q) exp(-i (phi_t(g(q)) - phi_r(g(q))))
gamma_q = arg(S_q)
```

Away from `S_q = 0`, its phase Jacobian is

```text
d gamma_q = d phi_r(g(q)) - sum over real t of
            Re(z_t(q) exp(-i phi'_t(g(q))) / S_q) d phi_t(g(q)).
```

The coefficients vary by native sample. `compress` emits this native-grid field after
nearest-neighbor upsampling the looked phase (`crates/dolphin-phaselink/src/quality.rs`). The next
ministack builds each coherence matrix from a clamped native window, optionally restricted by a
realized SHP mask (`crates/dolphin-phaselink/src/covariance.rs`). One downstream output cell can
therefore consume compressed samples derived from several upstream looked cells.

If `h` is the fixed EMI/EVD branch, the required transition is

```text
d theta_b(p) = sum over carried a and q in W_p of
               D[h(C_p)] / D[gamma_a(q)] * d gamma_a(q) + local innovation.
```

`F_b = -Q_b J_xs` describes a uniform perturbation of one carried temporal parameter in one local
Fisher model. It does not contain the score derivative with respect to each native carried sample.

## Non-identifiability counterexample

Take one carried band (`K=1`), two upstream looked errors `z0` and `z1`, and two new downstream
dates. A valid downstream linearization can be

```text
x = [z0, z1]' + eta.
```

Its parent-induced covariance can have rank two. The proposed state replaces the two spatial
parents with one scalar `s = a' [z0, z1]`, so `F Var(s) F'` has rank at most one. No choice of
per-pixel `F`, `L`, and `w` reproduces the production transition.

The problem remains in a scalar projection. For `s = (z0 + z1)/2` and
`Var(z0) = Var(z1) = 1`:

```text
Var(s) = 1/2  when Cov(z0,z1) = 0
Var(s) = 1    when Cov(z0,z1) = 1.
```

Both cases have identical per-pixel temporal factors, parent IDs, and support bits. They require
different propagated covariance. Overlapping production windows make the omitted cross-pixel
covariance nonzero in general.

A concrete stride example has four native columns, stride two, and half-window one. Nearest
upsampling maps columns `(0,1)` to upstream cell `c0` and `(2,3)` to `c1`. The downstream windows
are `(0,1,2)` and `(1,2,3)`. Under the linearized averages

```text
x0 = (2 c0 + c1) / 3
x1 = (c0 + 2 c1) / 3,
```

unit parent variances and correlation `rho` give

```text
Var(x0)    = 5/9 + 4 rho/9
Cov(x0,x1) = 4/9 + 5 rho/9.
```

The planned payload does not identify either quantity.

## Why replay metadata is insufficient

The proposed replay descriptor identifies the grid, support, source manifest, estimator branch,
and compression ancestry. It does not store:

- the numeric compression Jacobian for each native source sample;
- the EMI/EVD score or implicit Jacobian from each carried sample to the linked phase; or
- a shared source-innovation model or cross-pixel covariance.

Recomputing those values from the source data would perform the source-influence propagation
assigned to issue #54. A fingerprint and bit-packed support mask alone cannot reconstruct it.

## Resource consequence

An exact first-order transition needs a spatial operator. Its minimum local edge count depends on
the number of upstream looked cells intersecting the native window, and its stochastic covariance
also needs shared source IDs or sparse cross-pixel covariance. The count is not bounded by
`m*K + m*(m+1)/2 + m+K`; it includes window support or aggregated spatial-parent edges. The
selected 125.4 MiB-per-256x256-block payload estimate is therefore not an estimate of an exact
producer.

## Revision decision

Ryan selected the source-keyed revision. Numeric incidence matrices are still excluded: at the
default `m=13`, `k=10`, `S=435`, and a 256x256 output block, phase-only carried edges exceed 46
GiB and exact complex carried edges exceed 93 GiB before new-source influence or ancestry. The
replacement is an implicit computational DAG whose coefficients are regenerated during a
byte-capped query.

## Replacement method: `sequential_source_dag_v1`

### Frozen source model

Each primitive source has one consumer-independent identity:

```text
SourceKey = (source manifest, burst, logical block, ordered new-real dates,
             global native row, global native column, model version,
             raw source content digest)
```

For source `q` in block `b`, a proper-complex tangent factor supplied by the caller defines

```text
delta r_bq = L_bq xi_bq,  xi_bq ~ N(0, I).
```

The canonical real embedding of `L` is

```text
W_q = 1/sqrt(2) [[Re(L), -Im(L)],
                 [Im(L),  Re(L)]].
```

The lower factor has a positive real diagonal, fixed component order, source key, and model hash.
Every consumer of a shared source must use the same factor identity. Missing, non-positive-
definite, differently ordered, or mismatched factors fail closed. #52 supplies no identity or
target-window covariance fallback. The caller-supplied plug-in source model remains uncalibrated
and cannot authorize inference.

The primitive vector contains every raw complex acquisition consumed by the block, including
acquisition 0 in the first block. The exact gauge removes acquisition 0 only from the retained
phase/date coordinates; it does not remove or condition away that raw source component. Its real
and imaginary perturbations still affect coherence contrasts and compression.

Raw source samples and their `L_bq` factors are immutable external replay inputs. The artifact
binds their resolver, ordered component IDs, provider method/version, model receipt, raw-content
digest, and per-source numeric-factor receipt. Capture hashes each block/native pixel's ordered raw
samples and derives the source ID from that digest and the source locator. A Replayable writer also
requires the caller to bind the canonical digest of the exact `L_bq`; descriptor-only CLI capture
stores no factor receipt and remains `source_model_unavailable`. During replay, the resolver must
reproduce both receipts. The query resolves each source once within its reverse block and reuses
that exact sample/factor pair. A mismatch returns `source_identity_mismatch`; missing raw bytes
return `source_unavailable`, and an unavailable factor model returns `source_model_unavailable`.
Replay never substitutes stored linked phases, identity covariance, or a window-specific factor.
Persisting the raw inputs or source factors inside the artifact is an alternate schema whose bytes
must be counted separately.

Different source keys are independent conditional on carried history. This is the frozen #52
source model, not a field-calibration claim. #54 validates or rejects it for a target/reference
contrast.

### Graph

Every non-root node is linearized as

```text
y_v = sum over u in parents(v) of E_vu y_u + B_v xi_v.
```

The graph is acyclic under this strict order:

```text
all nodes in blocks < b
  < Source(b,q)
  < Phase(b,p)
  < Date(b,t,p)
  < Compressed(b,q).
```

Every parent precedes its child in that order, including direct
`Source(b,q) -> Compressed(b,q)` edges. The resulting global incidence matrix is strictly lower
triangular under the node order even though several edge types occur within one block. The
persistent graph contains four node families:

- `Source(b,q)`: the primitive new-real complex tangent at native source `q`;
- `Phase(b,p)`: the fixed-branch, local-gauge-reduced linked phase at looked output `p`;
- `Compressed(b,q)`: the full two-component complex compressed SLC perturbation at native `q`;
- `Date(b,t,p)`: one retained real date lifted into the acquisition-0 gauge.

`Phase` has implicit parents for each selected carried `Compressed` node and each current-block
`Source` in its realized native support. `Compressed` has parents in the nearest looked `Phase`
node and the same current-block `Source` roots used directly by compression. Direct and indirect
uses of a source meet at one root before covariance contraction.

The artifact persists operator descriptors and numeric replay state, never expanded edge
matrices:

- deterministic node/source IDs, ordered dates, block generation, and carry parents;
- global native/output grids, bounded/tile origins, clamp/stride/nearest-neighbor rules;
- bit-packed realized support and native validity;
- linked solution, estimator branch, reference transform, selected eigenvalue/eigengap, and
  status;
- compressed complex raster, projection accumulator, mean amplitude, and reference identity;
- normalized config, source manifest, kernel version, raw-content digests, and numeric-factor
  receipts.

### Local derivatives

For a production covariance numerator

```text
N_p = sum over q in W_p of M_pq v_q v_q^H,
C_ij = N_ij / sqrt(N_ii N_jj),
```

one source direction gives

```text
dN = dv_q v_q^H + v_q dv_q^H,
dC_ij = dN_ij / sqrt(N_ii N_jj)
         - C_ij/2 (dN_ii/N_ii + dN_jj/N_jj).
```

Differentiate the selected, unchanged estimator branch:

```text
EVD: M = C hadamard |C|
EMI: M = Gamma^-1 hadamard C
     d Gamma^-1 = -Gamma^-1 (d Gamma) Gamma^-1

d v_k = sum over j != k of v_j (v_j^H dM v_k) / (lambda_k-lambda_j)
d theta_i = Im(conj(v_i) d v_i) / |v_i|^2 - d theta_reference.
```

The production branches are part of the derivative contract. For EVD, away from `|C_ij|=0`:

```text
d(C_ij |C_ij|) = |C_ij| dC_ij
                  + C_ij Re(conj(C_ij) dC_ij) / |C_ij|.
```

For EMI, before thresholding:

```text
Gamma_ij = (1-beta) |C_ij| + beta 1[i=j]   when beta > 0
Gamma_ij = |C_ij|                           when beta = 0
d Gamma_ij = scale * Re(conj(C_ij) dC_ij) / |C_ij|,
scale = 1-beta or 1.
```

A safely thresholded-zero `Gamma` entry has derivative zero; a threshold crossing or an entry
within the declared branch tolerance is nondifferentiable. In covariance normalization, a
denominator safely below `AMP_FLOOR=1e-6` retains the production zero with derivative zero. A
denominator at the floor, a zero-magnitude active `C` entry, threshold crossing, branch change,
estimator fallback, tied selected eigenvalue, nonfinite state, or vanishing reference component
fails closed. Central raw-complex differences are the test oracle only.

Compression must carry its complete complex derivative because its magnitude changes the next
coherence normalization. For real dates `t`:

```text
S = sum z_t exp(-i phi'_t),  a = mean |z_t|,  c = a S/|S|
dS = sum exp(-i phi'_t) (dz_t - i z_t d phi'_t)
da = mean Re(conj(z_t) dz_t) / |z_t|
dc = (S/|S|) da + a [dS/|S| - S Re(conj(S)dS)/|S|^3].
```

Zero projection, zero included amplitude, nonfinite derivatives, or a changed compression branch
invalidates the node.

### Exact gauge and query

Version 1 supports only `CompressedSlcPlan::AlwaysFirst` with `output_reference_idx=0`. Acquisition
0 has no stochastic **date/phase coordinate**, but its raw complex samples remain in the first
block's `Source` roots. The first block removes only its phase coordinate from the estimator
output. Its compressed SLC carries that reference recursively. Each later block references the
first carried compressed channel, so its retained real-date phase components are already the
production recurrence in the same acquisition-0 gauge; `Date` is a deterministic selection from
that referenced `Phase` node, not an invented physical-date anchor. The compression and phase
operators carry the reference uncertainty through their actual shared sources. Reconstructing the
public matrix inserts a literal zero row and column. Other compressed-reference plans return
`unsupported_reference_plan` before graph allocation.

For selected same-pixel dates `S`, replay computes

```text
Z = B' (I-E)^-T S'
Cov = sum over primitive roots q of Z_q' Z_q.
```

The reverse traversal sums every path reaching one `SourceKey` before the root contraction. It
therefore retains cross-window and multi-path covariance under the frozen source model without
persisting or expanding ancestry. Each root contribution is discarded after contraction.

#52 exposes same-pixel temporal covariance, selected blocks, and matrix-free application only.
#54 must construct one joint target/reference/L2 selection on this same graph. It may not subtract
separately factored #52 marginals.

### Resource contract

Approximate persistent payload per block is

```text
16 * native_area                 compressed complex raster
+ 16 * native_area               complex projection accumulator
+ 8 * native_area                mean amplitude
+ 64 * native_area               raw-content and numeric-factor SHA-256 receipts
+ 8 * output_area * d_b          linked phase angles
+ output_area * ceil(S_b/8)      realized support
+ O(native_area + output_area)   IDs, status, branch, eigenvalue/gap, maps
```

At `native_area = output_area = 256^2`, `d_b=22`, and `S_b=435`, the listed terms are about
20.9 MiB before IDs, status, HDF5 metadata, and fixed registries. Full Fisher matrices and
eigensystems are not persisted. Replay recomputes them from verified external sources and checks
the stored branch, selected eigenvalue/eigengap, phase solution, and kernel digest before
differentiation.

Before loading source windows or allocating adjoints, a query enumerates its topology-only
dependency cone and computes a conservative byte bound from node dimensions, support, requested
dates, source rank, and microbatch. A cone above the configured cap returns
`dependency_cone_exceeds_budget`. Query memory is then bounded by the target microbatch, active
reverse frontier, one local source window, and the requested `D x D` covariance. The local bound
charges worst-case sparse B-tree nodes, exact support-vector capacities, every live Rect
replay/JVP matrix, and cache-line-padded Faer EVD, Cholesky, and solve scratch. Runtime and
dependency-cone growth are reported; the rejected
factor-disabled `2x` runtime gate does not apply to an intentional replay operator.

The byte cap covers query-owned heap after Faer runtime initialization. Faer's process-global
runtime state is retained across queries and is not included; this is not a cold-process total-heap
bound.

This payload excludes immutable external raw SLC bytes and caller-owned source-factor/model bytes.
Every resource receipt reports their resolved byte counts, read throughput, cache size, and
provider identity separately. If inputs must be copied into the artifact for durability, disk
preflight includes that copy plus scratch and 25 percent free margin.

The artifact schema must contain no dataset shaped like
`area * support * temporal_dimension * output_dimension` and no expanded ancestor coefficient
table.

### Supported producer scope

Initial support is CPU/f64, Rect support with a fixed native validity mask, a fixed successful EVD
or EMI branch, `CompressedSlcPlan::AlwaysFirst`, single-burst whole/tiled/bounded batch processing,
immutable strong source IDs, exact output reference 0, partial final blocks, and a caller-supplied
source-factor model.

GPU, EMI-to-EVD fallback, adaptive GLRT/KS support, phase-bias correction, weak source identity,
and stitched multiburst covariance return explicit unsupported status while leaving legacy phase
production unchanged. Per-burst graphs may be emitted, but the stitched status is
`unsupported_seam_covariance`.

The in-memory resumable API remains a small-array convenience path. Bounded production NRT needs
artifact-backed sealed groups, revisioned open blocks, and chunked last-`K` compressed rasters; it
is not claimed until its append/recovery contracts pass.

### #52/#54 boundary

`factor_status=valid` means the frozen branch and replay algebra are evaluable. It does not mean
the source model is calibrated. Every artifact records

```text
method = sequential_source_dag_v1
calibration_status = uncalibrated
downstream_inference_status = blocked_pending_issue_54_and_53
```

Legacy per-ministack CRLB stays a separate non-inferential diagnostic. No local CRLB marginal is
used to normalize or overwrite source-influence covariance.

## Gate disposition

- T52-01: original factor failed; replacement design approved for a new red contract.
- T52-02: analytic source-DAG and central-difference `J J^T` contracts pass.
- T52-03: canonical schema, replay identity, gauge, and fail-closed contracts pass.
- T52-04: full-batch whole/tiled/bounded capture and block-local capped replay are implemented;
  resumable and multiburst seam covariance remain unsupported.
- T52-05: release smoke emits storage, disk, cache, source-resolution, timing, and RSS receipts.
- T52-06: independent final review found no unresolved blockers or acceptance-gate findings.
- Inference wiring, release, and GroundPulse pin remain outside #52.
- PR #55's conditional-IID output and the legacy per-ministack CRLB diagnostic remain unchanged.

Two independent read-only derivations reached the same result: one by rank and covariance
non-identifiability, and one by differentiating the native compression and downstream covariance
window. No reviewer approved kernel work under the selected representation.
