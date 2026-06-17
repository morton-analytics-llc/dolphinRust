//! 2D FFT helpers over `Array2<Complex<f64>>` via `rustfft` (row then column
//! passes). Inverse transforms are unnormalized — callers divide by `rows*cols`.

use ndarray::Array2;
use num_complex::Complex;
use rustfft::FftPlanner;

type C = Complex<f64>;

/// In-place 2D FFT (forward or inverse). Inverse is unnormalized.
pub fn fft2(data: &mut Array2<C>, inverse: bool) {
    let (rows, cols) = data.dim();
    let mut planner = FftPlanner::new();
    let row_fft = plan(&mut planner, cols, inverse);
    data.rows_mut()
        .into_iter()
        .for_each(|mut row| transform(&mut row, row_fft.as_ref()));
    let col_fft = plan(&mut planner, rows, inverse);
    data.columns_mut()
        .into_iter()
        .for_each(|mut col| transform(&mut col, col_fft.as_ref()));
}

/// Plan a forward/inverse 1D FFT of length `n`.
fn plan(
    planner: &mut FftPlanner<f64>,
    n: usize,
    inverse: bool,
) -> std::sync::Arc<dyn rustfft::Fft<f64>> {
    match inverse {
        true => planner.plan_fft_inverse(n),
        false => planner.plan_fft_forward(n),
    }
}

/// Transform one (possibly strided) lane through a scratch buffer.
fn transform(lane: &mut ndarray::ArrayViewMut1<C>, fft: &dyn rustfft::Fft<f64>) {
    let mut buf: Vec<C> = lane.iter().copied().collect();
    fft.process(&mut buf);
    lane.iter_mut().zip(buf).for_each(|(slot, v)| *slot = v);
}

/// numpy `fftfreq(n)`: cycles per sample, with negative frequencies in the upper half.
#[must_use]
pub fn fftfreq(n: usize) -> Vec<f64> {
    let half = n.div_ceil(2);
    (0..n)
        .map(|k| if k < half { k as f64 } else { k as f64 - n as f64 } / n as f64)
        .collect()
}
