//! Shared types for the dolphin Rust port.
//!
//! Mirrors the cross-cutting primitives in the Python `dolphin` package:
//! complex SLC element types, the `HalfWindow`/`Strides` look geometry, the
//! `StridedBlockManager` tiling scheme (`dolphin/io/_blocks.py`), and the
//! pydantic workflow config models (`dolphin/workflows/config/`).
//!
//! Planned modules (see PLAYBOOK.md): `types`, `blocks`, `config`, `error`.

/// Complex SLC sample (single-precision), matching dolphin's `complex64`.
pub type Cf32 = num_complex::Complex<f32>;
/// Double-precision complex, used inside covariance/eigensolver kernels.
pub type Cf64 = num_complex::Complex<f64>;
