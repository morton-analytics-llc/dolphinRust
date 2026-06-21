//! Clean-room native phase unwrapper (no SNAPHU, no CS2).
//!
//! Derived solely from the published literature — Costantini 1998 (minimum-cost-
//! flow formulation), Chen & Zebker 2001 (statistical network costs). The
//! wrapped phase's discrete gradients are integrated to the unwrapped field;
//! where residues make the wrapped gradients non-integrable, an independent
//! min-cost-flow over the dual grid graph (see [`mcf`]) routes branch cuts so
//! the corrected gradients are curl-free.
//!
//! This module shares no source with SNAPHU; SNAPHU is used only as a black-box
//! validation oracle (`oracle/gen_unwrap_suite.py`).

use dolphin_core::Cf32;
use ndarray::{Array2, ArrayView2};

use crate::snaphu::{CostMode, UnwrapError, UnwrapResult};

mod conncomp;
mod cost;
mod mcf;
mod simplex;
mod tile;

const TAU: f64 = std::f64::consts::TAU;

/// Native unwrapper configuration.
#[derive(Debug, Clone)]
pub struct NativeConfig {
    /// Statistical cost model used to weight branch-cut routing.
    pub cost: CostMode,
    /// Optional `(rows, cols)` tile grid for large interferograms; `None`
    /// unwraps the whole grid with one global MCF (the default).
    pub tile: Option<(usize, usize)>,
    /// Coherence below which a pixel is masked (connected-component 0).
    pub conncomp_min_corr: f32,
    /// Minimum component size as a fraction of the scene; smaller components are
    /// dropped to the masked label, mirroring SNAPHU's `minconncompfrac`.
    pub conncomp_min_frac: f64,
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            cost: CostMode::Smooth,
            tile: None,
            conncomp_min_corr: 0.15,
            conncomp_min_frac: 0.001,
        }
    }
}

/// Wrap a phase difference into `(-pi, pi]`.
fn wrap(x: f64) -> f64 {
    x - TAU * (x / TAU).round()
}

/// Unwrap a single wrapped interferogram in-process.
///
/// Returns the unwrapped phase (radians) and connected-component labels, the
/// same `(unwrapped, conncomp)` shape SNAPHU produces, with no subprocess or
/// scratch I/O.
///
/// # Errors
/// Returns `Err` if the inputs are smaller than 2x2 or shapes disagree.
pub fn unwrap_native(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    cfg: &NativeConfig,
) -> Result<UnwrapResult, UnwrapError> {
    let (rows, cols) = wrapped.dim();
    if rows < 2 || cols < 2 {
        return Err(UnwrapError::Shape(format!("grid too small: {rows}x{cols}")));
    }
    if correlation.dim() != (rows, cols) {
        return Err(UnwrapError::Shape(format!(
            "corr {:?} != ifg {:?}",
            correlation.dim(),
            (rows, cols)
        )));
    }

    let psi = wrapped.mapv(|z| z.arg() as f64);
    let unwrapped = match cfg.tile {
        Some(tiles) => tile::unwrap_tiled(&psi, correlation, cfg.cost, tiles),
        None => unwrap_grid(&psi, correlation, cfg.cost),
    };
    let conncomp = conncomp::segment(correlation, cfg.conncomp_min_corr, cfg.conncomp_min_frac);
    Ok(UnwrapResult {
        unwrapped: unwrapped.mapv(|v| v as f32),
        conncomp,
    })
}

/// Unwrap one whole grid: wrapped gradients, statistical-cost MCF branch-cut
/// correction (zero flow when residue-free), then integration. The shared core
/// of both the global and per-tile paths.
fn unwrap_grid(psi: &Array2<f64>, corr: ArrayView2<f32>, cost: CostMode) -> Array2<f64> {
    let (ax, ay) = wrapped_gradients(psi);
    // `None` for the residue-free fast path: no branch-cut corrections are
    // allocated at all, so high-coherence ifgs carry no flow-array overhead.
    let flow = mcf::solve(&ax, &ay, corr, cost);
    integrate(psi, &ax, &ay, flow.as_ref())
}

/// Wrapped row/column gradients: `ax[i,j] = W(psi[i,j+1]-psi[i,j])` of shape
/// `(rows, cols-1)`, `ay[i,j] = W(psi[i+1,j]-psi[i,j])` of shape `(rows-1, cols)`.
fn wrapped_gradients(psi: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (rows, cols) = psi.dim();
    let ax = Array2::from_shape_fn((rows, cols - 1), |(i, j)| {
        wrap(psi[(i, j + 1)] - psi[(i, j)])
    });
    let ay = Array2::from_shape_fn((rows - 1, cols), |(i, j)| {
        wrap(psi[(i + 1, j)] - psi[(i, j)])
    });
    (ax, ay)
}

/// Integrate the corrected gradients `a + 2pi*k` by a raster scan from `(0,0)`.
/// `flow` is `None` for residue-free fields (no correction, no allocation);
/// with curl-free corrected gradients the result is path-independent.
fn integrate(
    psi: &Array2<f64>,
    ax: &Array2<f64>,
    ay: &Array2<f64>,
    flow: Option<&(Array2<f64>, Array2<f64>)>,
) -> Array2<f64> {
    let (rows, cols) = psi.dim();
    let cx = |i: usize, j: usize| flow.map_or(0.0, |(kx, _)| TAU * kx[(i, j)]);
    let cy = |i: usize, j: usize| flow.map_or(0.0, |(_, ky)| TAU * ky[(i, j)]);
    let mut phi = Array2::zeros((rows, cols));
    phi[(0, 0)] = psi[(0, 0)];
    for j in 1..cols {
        phi[(0, j)] = phi[(0, j - 1)] + ax[(0, j - 1)] + cx(0, j - 1);
    }
    for i in 1..rows {
        phi[(i, 0)] = phi[(i - 1, 0)] + ay[(i - 1, 0)] + cy(i - 1, 0);
        for j in 1..cols {
            phi[(i, j)] = phi[(i, j - 1)] + ax[(i, j - 1)] + cx(i, j - 1);
        }
    }
    phi
}
