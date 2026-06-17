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
