//! Phase filters — port of `dolphin/filtering.py` and `goldstein.py`.
//!
//! Long-wavelength FFT Gaussian high-pass filter and the Goldstein adaptive
//! filter, both via `rustfft`. Used as optional pre-unwrap stages.

pub mod fft;
pub mod filters;

pub use filters::{filter_long_wavelength, goldstein};
