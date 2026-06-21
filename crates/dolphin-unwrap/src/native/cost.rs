//! Statistical edge costs for the MCF (Chen & Zebker 2001 lineage).
//!
//! The cost of routing a branch cut across a primal edge is the phase
//! *precision* of that edge: the Cramer-Rao lower bound on interferometric
//! phase variance is `sigma^2 = (1 - gamma^2) / (2 N gamma^2)`, so the precision
//! (inverse variance, at fixed looks `N`) is proportional to `gamma^2/(1-gamma^2)`.
//! Cutting a high-coherence edge is therefore expensive and cutting a
//! decorrelated edge is cheap — the MCF pushes discontinuities into the noisy
//! regions exactly as the statistical-cost network does. Derived from the
//! published model; no SNAPHU cost tables are read.

use ndarray::{Array2, ArrayView2};

use super::CostMode;

/// Floor so every edge keeps a strictly positive integer cost (degenerate
/// zero-cost cuts would let the MCF wander freely through decorrelated ground).
const MIN_WEIGHT: f64 = 1.0;
/// Coherence ceiling — caps the `1/(1-gamma^2)` blow-up near `gamma = 1`.
const GAMMA_MAX: f64 = 0.99;
/// Integer scale: costs are integerized for the integer min-cost-flow.
const SCALE: f64 = 1.0e4;

/// Per-edge integer costs `(wx, wy)` aligned with the `ax`/`ay` gradient grids.
/// `wx[i,j]` weights the horizontal edge between pixels `(i,j)`-`(i,j+1)`;
/// `wy[i,j]` the vertical edge between `(i,j)`-`(i+1,j)`.
pub fn edge_costs(corr: ArrayView2<f32>, _mode: CostMode) -> (Array2<i64>, Array2<i64>) {
    let (rows, cols) = corr.dim();
    let wx = Array2::from_shape_fn((rows, cols - 1), |(i, j)| {
        precision(corr[(i, j)], corr[(i, j + 1)])
    });
    let wy = Array2::from_shape_fn((rows - 1, cols), |(i, j)| {
        precision(corr[(i, j)], corr[(i + 1, j)])
    });
    (wx, wy)
}

/// Integerized CRLB phase precision of the edge bounded by coherences `a`, `b`.
fn precision(a: f32, b: f32) -> i64 {
    let g = (0.5 * (a as f64 + b as f64)).clamp(0.0, GAMMA_MAX);
    let prec = g * g / (1.0 - g * g);
    (prec.max(MIN_WEIGHT) * SCALE).round() as i64
}
