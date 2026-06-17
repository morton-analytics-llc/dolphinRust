//! Statistically homogeneous pixel (SHP) selection — port of `dolphin/shp/`.
//!
//! GLRT test under a Rayleigh amplitude model (`_glrt.py`, chi-squared
//! threshold via `statrs`) and the non-parametric KS test (`_ks.py`, the
//! numba `njit(parallel=True)` hot loop → `rayon`). Produces the boolean
//! neighbor array `(rows, cols, win_h, win_w)` consumed by covariance
//! estimation.
