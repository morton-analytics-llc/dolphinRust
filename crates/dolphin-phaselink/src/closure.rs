//! Sequential closure phase — port of dolphin `_closure_phase.py`.
//!
//! Per pixel, the wrapped non-closure of each nearest-neighbour triplet of the
//! coherence matrix: band `k` is
//! `∠( C[k,k+1] · C[k+1,k+2] · conj(C[k,k+2]) )`. A perfectly consistent
//! (phase-bias-free) pixel closes to 0; departures are the non-closure
//! diagnostic and the prerequisite signal for phase-bias correction.

use dolphin_core::Cf64;
use ndarray::{s, Array3, ArrayView2, ArrayView4};
use rayon::prelude::*;

/// Per-pixel nearest-neighbour closure phase (radians) over a coherence stack
/// `(rows, cols, nslc, nslc)`. Returns `(nslc-2, rows, cols)` band-major; an
/// empty band axis when `nslc < 3`.
#[must_use]
pub fn estimate_closure_phases(c_arrays: ArrayView4<Cf64>) -> Array3<f64> {
    let (rows, cols, nslc, _) = c_arrays.dim();
    let ntri = nslc.saturating_sub(2);
    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .flat_map_iter(|idx| {
            let c = c_arrays.slice(s![idx / cols, idx % cols, .., ..]);
            (0..ntri).map(move |k| triplet_closure(c, k))
        })
        .collect();
    // values are pixel-major (pixel, triplet); transpose to band-major.
    Array3::from_shape_fn((ntri, rows, cols), |(k, r, c)| {
        values[(r * cols + c) * ntri + k]
    })
}

/// Wrapped non-closure of the triplet `(k, k+1, k+2)`.
fn triplet_closure(c: ArrayView2<Cf64>, k: usize) -> f64 {
    (c[(k, k + 1)] * c[(k + 1, k + 2)] * c[(k, k + 2)].conj()).arg()
}
