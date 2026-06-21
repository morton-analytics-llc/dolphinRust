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

mod cost;
mod mcf;

const TAU: f64 = std::f64::consts::TAU;

/// Native unwrapper configuration.
#[derive(Debug, Clone)]
pub struct NativeConfig {
    /// Statistical cost model used to weight branch-cut routing.
    pub cost: CostMode,
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            cost: CostMode::Smooth,
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
    let (ax, ay) = wrapped_gradients(&psi);
    // Statistical-cost MCF routes branch cuts so the corrected gradients are
    // curl-free; zero flow for residue-free fields.
    let (kx, ky) = mcf::solve(&ax, &ay, correlation, cfg.cost);
    let unwrapped = integrate(&psi, &ax, &ay, &kx, &ky);
    let conncomp = Array2::from_elem((rows, cols), 1u32);
    Ok(UnwrapResult {
        unwrapped: unwrapped.mapv(|v| v as f32),
        conncomp,
    })
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
/// With curl-free corrected gradients the result is path-independent.
fn integrate(
    psi: &Array2<f64>,
    ax: &Array2<f64>,
    ay: &Array2<f64>,
    kx: &Array2<f64>,
    ky: &Array2<f64>,
) -> Array2<f64> {
    let (rows, cols) = psi.dim();
    let mut phi = Array2::zeros((rows, cols));
    phi[(0, 0)] = psi[(0, 0)];
    for j in 1..cols {
        phi[(0, j)] = phi[(0, j - 1)] + ax[(0, j - 1)] + TAU * kx[(0, j - 1)];
    }
    for i in 1..rows {
        phi[(i, 0)] = phi[(i - 1, 0)] + ay[(i - 1, 0)] + TAU * ky[(i - 1, 0)];
        for j in 1..cols {
            phi[(i, j)] = phi[(i, j - 1)] + ax[(i, j - 1)] + TAU * kx[(i, j - 1)];
        }
    }
    phi
}
