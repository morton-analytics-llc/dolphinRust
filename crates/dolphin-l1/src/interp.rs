//! Windowed-sinc interpolation of complex SAR samples.
//!
//! 8-tap sinc with a Hann window, tabulated at 1/2048 sample and normalised so
//! every fractional kernel sums to one. Sentinel-1 is oversampled by ≈1.33 in
//! range and ≈1.5 in deramped azimuth, so an 8-tap kernel keeps the passband
//! flat; ISCE3's `geocodeSlc` also uses an 8-tap sinc. The 2-D kernel is
//! separable: range first, then azimuth.

use ndarray::ArrayView2;
use num_complex::Complex32;

/// Kernel length (taps).
pub const SINC_TAPS: usize = 8;
/// Taps before the integer sample: the kernel spans `floor(x) − 3 ..= floor(x) + 4`.
pub const SINC_BEFORE: usize = SINC_TAPS / 2 - 1;
const OVERSAMPLE: usize = 2048;

/// Tabulated windowed-sinc kernel.
pub struct SincKernel {
    table: Vec<[f32; SINC_TAPS]>,
}

impl Default for SincKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl SincKernel {
    /// Build the table.
    pub fn new() -> Self {
        let half = SINC_TAPS as f64 / 2.0;
        let table = (0..=OVERSAMPLE)
            .map(|k| {
                let frac = k as f64 / OVERSAMPLE as f64;
                let raw: [f64; SINC_TAPS] = std::array::from_fn(|tap| {
                    let x = tap as f64 - SINC_BEFORE as f64 - frac;
                    let window = 0.5 * (1.0 + (std::f64::consts::PI * x / half).cos());
                    sinc(x) * window
                });
                let sum: f64 = raw.iter().sum();
                std::array::from_fn(|tap| (raw[tap] / sum) as f32)
            })
            .collect();
        Self { table }
    }

    fn weights(&self, frac: f64) -> &[f32; SINC_TAPS] {
        &self.table[(frac * OVERSAMPLE as f64).round() as usize]
    }

    /// Interpolate `data` at fractional `(row, col)`. Returns `None` when the
    /// kernel would reach outside `data`.
    pub fn interpolate(
        &self,
        data: ArrayView2<Complex32>,
        row: f64,
        col: f64,
    ) -> Option<Complex32> {
        let (rows, cols) = data.dim();
        let (r_int, c_int) = (row.floor(), col.floor());
        if r_int < SINC_BEFORE as f64 || c_int < SINC_BEFORE as f64 {
            return None;
        }
        let (r0, c0) = (r_int as usize - SINC_BEFORE, c_int as usize - SINC_BEFORE);
        if r0 + SINC_TAPS > rows || c0 + SINC_TAPS > cols {
            return None;
        }
        let wr = self.weights(row - r_int);
        let wc = self.weights(col - c_int);
        let mut sum = Complex32::new(0.0, 0.0);
        for (i, w_row) in wr.iter().enumerate() {
            let line = data.row(r0 + i);
            let along: Complex32 = (0..SINC_TAPS).map(|j| line[c0 + j] * wc[j]).sum();
            sum += along * *w_row;
        }
        Some(sum)
    }

    /// Interpolate an 8 × 8 chip whose tap `[SINC_BEFORE][SINC_BEFORE]` is
    /// the integer sample, at fractional offsets `frac_row`, `frac_col` in
    /// `[0, 1)` from it. Same weights as [`Self::interpolate`].
    pub fn interpolate_chip(
        &self,
        chip: &[[Complex32; SINC_TAPS]; SINC_TAPS],
        frac_row: f64,
        frac_col: f64,
    ) -> Complex32 {
        let wr = self.weights(frac_row);
        let wc = self.weights(frac_col);
        chip.iter()
            .zip(wr)
            .map(|(line, w_row)| {
                let along: Complex32 = line.iter().zip(wc).map(|(v, w)| *v * *w).sum();
                along * *w_row
            })
            .sum()
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        return 1.0;
    }
    let px = std::f64::consts::PI * x;
    px.sin() / px
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// A complex tone at 0.08 / 0.15 cycles/sample is reproduced at off-grid
    /// points to within 2 % amplitude (the 8-tap Hann taper that ISCE3's
    /// `Sinc2dInterpolator` also has costs 1.1 % at 0.15 cycles/sample) and
    /// 0.01 rad phase; integer positions return the sample itself.
    #[test]
    fn reproduces_bandlimited_tone() {
        let tone = |r: f64, c: f64| {
            let phase = 2.0 * std::f64::consts::PI * (0.08 * r + 0.15 * c);
            Complex32::new(phase.cos() as f32, phase.sin() as f32)
        };
        let data = Array2::from_shape_fn((32, 32), |(r, c)| tone(r as f64, c as f64));
        let kernel = SincKernel::new();
        for &(r, c) in &[(10.25, 12.6), (15.5, 15.5), (20.9, 9.1)] {
            let got = kernel.interpolate(data.view(), r, c).unwrap();
            let want = tone(r, c);
            assert!((got.norm() - 1.0).abs() < 0.02, "{}", got.norm());
            assert!((got * want.conj()).arg().abs() < 0.01);
        }
        let exact = kernel.interpolate(data.view(), 12.0, 13.0).unwrap();
        assert!((exact - data[[12, 13]]).norm() < 1e-5);
        assert!(kernel.interpolate(data.view(), 2.5, 10.0).is_none());
        assert!(kernel.interpolate(data.view(), 10.0, 28.5).is_none());
    }
}
