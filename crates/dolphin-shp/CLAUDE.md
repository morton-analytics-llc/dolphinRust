# dolphin-shp — homogeneous-pixel selection (reference: `dolphin/shp/`)

Rust rebuild of the SHP algorithms. Produces the boolean `(rows, cols, win_h, win_w)`
neighbor mask consumed by dolphin-phaselink covariance estimation.

## Domain
- **GLRT (default):** Rayleigh amplitude model; `σ² = (var + mean²)/2`;
  `T = N(2 log σ_pooled − log σ_1 − log σ_2)`; threshold `χ²(1, 1−α)`, α=0.001 (statrs).
- **KS test:** non-parametric, sorted-amplitude ECDF max distance vs. critical value. In
  dolphin this is the numba `njit(parallel=True)` hot loop; here it is rayon over center
  pixels — a prime optimization target.

## Contracts
- Validate the mask against analytic distributions (known SHP / non-SHP pairs), with
  dolphin as a reference oracle. The decision boundary must agree; the vectorization is
  ours to optimize.
