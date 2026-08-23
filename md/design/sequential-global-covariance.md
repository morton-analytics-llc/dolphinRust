# Sequential global covariance: T52-01 design gate

**Status:** NO-GO for `sequential_srif_v1` as specified in the #52 plan.

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

## Allowed revisions

1. Narrow #52 to a spatially independent local-Fisher diagnostic. This requires removing the
   global propagation and GLS claims and does not resolve the current issue.
2. Move the source-keyed influence foundation forward from #54. Store a sparse spatial influence
   DAG with shared source-innovation IDs, numeric parent edges, and local innovation factors;
   revise the byte/RSS gates before implementation. Reference-specific contraction and field
   calibration can remain in #54.

The second revision is the only route that preserves #52's global-propagation requirement. It
changes the approved issue split and resource contract, so it needs an explicit plan revision.

## Gate disposition

- T52-01: **failed; design no-go recorded**.
- T52-02 through T52-06: **not started**.
- Production covariance code, HDF5 output, inference wiring, release, and GroundPulse pin: **not
  authorized by this failed gate**.
- PR #55's conditional-IID output and the legacy per-ministack CRLB diagnostic remain unchanged.

Two independent read-only derivations reached the same result: one by rank and covariance
non-identifiability, and one by differentiating the native compression and downstream covariance
window. No reviewer approved kernel work under the selected representation.
