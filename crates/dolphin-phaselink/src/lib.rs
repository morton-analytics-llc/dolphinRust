//! Phase linking — port of `dolphin/phase_link/`. The numerical core.
//!
//! Covariance estimation over sliding windows (`covariance.py`), EVD (power
//! iteration) and EMI (regularized inverse iteration via Cholesky) estimators
//! (`_core.py`, `_eigenvalues.py`), compressed-SLC generation (`_compress.py`),
//! temporal coherence (`metrics.py`), CRLB (`crlb.py`), and closure phase.
//!
//! Design: JAX `vmap(vmap(f))` over the (rows, cols) pixel grid maps to a
//! `rayon` parallel iterator where each closure solves one NxN complex matrix
//! via `faer`. This is the highest-value module to port first.
#![warn(missing_docs)]

pub mod closure;
pub mod covariance;
pub mod crlb;
pub mod engine;
pub mod estimator;
pub mod fused;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod phasebias;
pub mod quality;

pub use closure::estimate_closure_phases;
pub use covariance::estimate_stack_covariance;
pub use crlb::estimate_crlb;
pub use engine::{ComputeEngine, ResolvedBackend};
pub use estimator::{
    process_coherence_matrices, process_coherence_matrix, PixelEstimate, StackEstimate,
};
pub use fused::{link_fused, AverageCoherenceAggregate, FusedEstimate, FusedParams};
pub use phasebias::{
    correct_phase_bias, estimate_bias_velocity, mean_abs_closure, residual_closure,
};
pub use quality::{
    average_coherence_per_date, compress, estimate_average_coherence, estimate_temp_coh,
};
