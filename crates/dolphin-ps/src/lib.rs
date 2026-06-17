//! Persistent scatterer selection — port of `dolphin/ps.py`.
//!
//! Amplitude dispersion `D_A = std(|z|)/mean(|z|)` over the temporal stack,
//! thresholded (default 0.25) into a PS mask. Block-processed; PS pixels
//! bypass covariance estimation and take phase directly.
