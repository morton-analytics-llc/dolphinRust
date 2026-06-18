//! Subtract a per-date atmospheric range delay from the LOS-phase displacement
//! time series.
//!
//! The displacement series `disp_rad` is `(n_dates-1, rows, cols)` in radians of
//! LOS phase, referenced to acquisition 0. The delay stack `delay_m` is
//! `(n_dates, rows, cols)` range delay in meters, one band per acquisition. The
//! correction applied to band `t` (acquisition `t+1`) is the delay **relative to
//! acquisition 0** — the series' own reference — converted to phase by
//! `φ = d · (-4π/λ)`, the inverse of the pipeline's `phase → displacement` factor
//! `-λ/4π`, so the corrected displacement is exactly `measured − relative_delay`.

use ndarray::{Array3, ArrayView3, Axis};

use crate::error::{CorrectionError, Result};

/// Subtract the per-date atmospheric range delay (referenced to acquisition 0)
/// from the displacement series, in place.
///
/// # Errors
/// [`CorrectionError::Shape`] if `delay_m` is not `(n_dates, rows, cols)` matching
/// the `(n_dates-1, rows, cols)` series; [`CorrectionError::MissingWavelength`] if
/// `wavelength_m` is not positive.
pub fn subtract_delay(
    disp_rad: &mut Array3<f64>,
    delay_m: ArrayView3<f64>,
    wavelength_m: f64,
) -> Result<()> {
    if wavelength_m <= 0.0 || wavelength_m.is_nan() {
        return Err(CorrectionError::MissingWavelength);
    }
    let (n_bands, rows, cols) = disp_rad.dim();
    let (n_dates, drows, dcols) = delay_m.dim();
    if n_dates != n_bands + 1 || drows != rows || dcols != cols {
        return Err(CorrectionError::Shape(format!(
            "delay {:?} incompatible with displacement {:?}",
            delay_m.dim(),
            disp_rad.dim()
        )));
    }
    let m_to_rad = -4.0 * std::f64::consts::PI / wavelength_m;
    let ref_delay = delay_m.index_axis(Axis(0), 0);
    for t in 0..n_bands {
        let rel = &delay_m.index_axis(Axis(0), t + 1) - &ref_delay;
        let mut band = disp_rad.index_axis_mut(Axis(0), t);
        band.zip_mut_with(&rel, |d, &r| *d -= r * m_to_rad);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    const LAMBDA: f64 = 0.238_403_545; // NISAR L-band

    /// Contract (DoD #1): a zero delay leaves the series bit-identical.
    #[test]
    fn zero_delay_is_identity() {
        let mut disp = Array3::from_shape_fn((2, 3, 4), |(t, r, c)| (t * 12 + r * 4 + c) as f64);
        let original = disp.clone();
        let delay = Array3::<f64>::zeros((3, 3, 4));
        subtract_delay(&mut disp, delay.view(), LAMBDA).unwrap();
        assert_eq!(disp, original);
    }

    /// Contract (DoD #4): the per-pixel subtraction is exact vs a known delay,
    /// referenced to acquisition 0.
    #[test]
    fn exact_subtraction() {
        let mut disp = Array3::<f64>::zeros((2, 1, 1));
        // delays: date0 = 1.0 m, date1 = 1.5 m, date2 = 3.0 m.
        let delay = Array3::from_shape_vec((3, 1, 1), vec![1.0, 1.5, 3.0]).unwrap();
        subtract_delay(&mut disp, delay.view(), LAMBDA).unwrap();
        let m_to_rad = -4.0 * std::f64::consts::PI / LAMBDA;
        // band0: -(1.5-1.0)*k ; band1: -(3.0-1.0)*k
        assert!((disp[(0, 0, 0)] - (-(0.5) * m_to_rad)).abs() < 1e-12);
        assert!((disp[(1, 0, 0)] - (-(2.0) * m_to_rad)).abs() < 1e-12);
    }

    /// A delay constant across all dates cancels (relative-to-date-0 is zero).
    #[test]
    fn constant_delay_cancels() {
        let mut disp = Array3::from_shape_fn((2, 2, 2), |(t, _, _)| 1.0 + t as f64);
        let original = disp.clone();
        let delay = Array3::from_elem((3, 2, 2), 2.7);
        subtract_delay(&mut disp, delay.view(), LAMBDA).unwrap();
        assert_eq!(disp, original);
    }

    /// Non-positive wavelength is rejected (can't convert meters to phase).
    #[test]
    fn rejects_missing_wavelength() {
        let mut disp = Array3::<f64>::zeros((1, 1, 1));
        let delay = Array3::<f64>::zeros((2, 1, 1));
        assert!(subtract_delay(&mut disp, delay.view(), 0.0).is_err());
    }
}
