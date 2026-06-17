//! Cross-cutting primitive types: complex SLC samples and look geometry.

/// Complex SLC sample (single-precision), matching dolphin's `complex64`.
pub type Cf32 = num_complex::Complex<f32>;
/// Double-precision complex, used inside covariance/eigensolver kernels.
pub type Cf64 = num_complex::Complex<f64>;

/// Half-window size in the y (row) and x (column) directions.
///
/// The covariance/SHP neighborhood is `(2*y + 1) x (2*x + 1)`. Mirrors dolphin's
/// `HalfWindow` namedtuple; the default `(y=7, x=14)` matches the dolphin
/// `PhaseLinkingOptions.half_window` default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HalfWindow {
    pub y: usize,
    pub x: usize,
}

impl Default for HalfWindow {
    fn default() -> Self {
        Self { y: 7, x: 14 }
    }
}

/// Decimation/striding factor in the y (row) and x (column) directions.
///
/// Mirrors dolphin's `Strides` namedtuple. Default `(1, 1)` = no decimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Strides {
    pub y: usize,
    pub x: usize,
}

impl Default for Strides {
    fn default() -> Self {
        Self { y: 1, x: 1 }
    }
}

impl Strides {
    /// Output grid size after striding an input of `shape = (rows, cols)`.
    ///
    /// Integer division, matching dolphin's `compute_out_shape`: a trailing
    /// partial window is dropped rather than padded.
    #[must_use]
    pub fn out_shape(self, shape: (usize, usize)) -> (usize, usize) {
        (shape.0 / self.y, shape.1 / self.x)
    }
}
