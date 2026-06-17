//! SBAS network inversion — port of `dolphin/timeseries.py`.
//!
//! Builds the `(n_ifgs, n_dates-1)` incidence matrix and solves `A φ = Δφ`:
//! weighted L2 least squares first (via `faer`), L1/ADMM deferred. Plus
//! correlation weighting and linear velocity estimation. Block-parallel.
