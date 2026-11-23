//! Enhanced spectral diversity (ESD) on TOPS burst overlaps.
//!
//! Where bursts `k` and `k+1` overlap, the same ground is seen twice, through
//! the forward- and backward-squinted beams, at Doppler centroids that differ
//! by `Δf ≈ 4–5 kHz`. An azimuth misregistration `δt` of a secondary date
//! against the reference shifts the interferogram phase by `2π f δt`, so the
//! double difference of the two overlap interferograms,
//! `arg[(R_k S_k*) (R_{k+1} S_{k+1}*)*] = 2π Δf δt`, measures `δt` free of
//! topography and deformation (Prats-Iraola et al. 2012, TGRS 50(10)).
//!
//! Both bursts of both dates are geocoded onto one overlap grid. The double
//! difference is multilooked; looks with double-difference coherence above
//! `min_coherence` each give `δt = φ / (2π Δf)` and the estimate is their
//! median. The sign is the correction: add the returned shift to the
//! secondary's `azimuth_shift`.

use ndarray::s;
use num_complex::Complex32;

use crate::error::{L1Error, Result};
use crate::geocode::GeocodedBurst;

/// Multilook window (rows, cols) on the 20 m ESD grid: 80 m × 80 m.
pub const ESD_LOOKS: (usize, usize) = (4, 4);
/// Minimum double-difference coherence of a look.
pub const ESD_MIN_COHERENCE: f64 = 0.7;
/// Minimum number of accepted looks.
pub const ESD_MIN_LOOKS: usize = 50;

/// An ESD estimate for one secondary date.
#[derive(Debug, Clone, Copy)]
pub struct EsdEstimate {
    /// Azimuth correction (s) to add to the secondary's sampling time.
    pub azimuth_shift: f64,
    /// Accepted multilooked cells.
    pub looks: usize,
    /// Median double-difference coherence of the accepted cells.
    pub coherence: f64,
    /// Median overlap Doppler difference (Hz).
    pub doppler_difference: f64,
    /// ESD re-estimated after applying the shift (s); ≈ 0 when the estimate
    /// and its sign are right.
    pub residual_shift: f64,
}

/// The four geocoded overlap products of one reference/secondary pair.
pub struct OverlapPair<'a> {
    /// Reference date, burst k.
    pub reference_k: &'a GeocodedBurst,
    /// Reference date, burst k+1.
    pub reference_k1: &'a GeocodedBurst,
    /// Secondary date, burst k.
    pub secondary_k: &'a GeocodedBurst,
    /// Secondary date, burst k+1.
    pub secondary_k1: &'a GeocodedBurst,
}

impl<'a> OverlapPair<'a> {
    /// Estimate the secondary's azimuth misregistration.
    pub fn estimate(&self) -> Result<EsdEstimate> {
        let (rows, cols) = self.reference_k.data.dim();
        let (lr, lc) = ESD_LOOKS;
        let mut cells: Vec<(f64, f64, f64)> = Vec::new();
        for r in (0..rows.saturating_sub(lr - 1)).step_by(lr) {
            for c in (0..cols.saturating_sub(lc - 1)).step_by(lc) {
                if let Some(cell) = self.cell(r, c) {
                    cells.push(cell);
                }
            }
        }
        if cells.len() < ESD_MIN_LOOKS {
            return Err(L1Error::Geometry(format!(
                "ESD found {} coherent overlap looks; need {ESD_MIN_LOOKS}",
                cells.len()
            )));
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        Ok(EsdEstimate {
            looks: cells.len(),
            azimuth_shift: median(cells.iter().map(|c| c.0).collect()),
            coherence: median(cells.iter().map(|c| c.1).collect()),
            doppler_difference: median(cells.iter().map(|c| c.2).collect()),
            residual_shift: f64::NAN,
        })
    }

    /// One look at `(r, c)`: `(δt, coherence, Δf)` if every pixel is valid and
    /// the look is coherent.
    fn cell(&self, r: usize, c: usize) -> Option<(f64, f64, f64)> {
        let (lr, lc) = ESD_LOOKS;
        let window = s![r..r + lr, c..c + lc];
        let view = |b: &'a GeocodedBurst| b.data.slice(window);
        let (rk, rk1) = (view(self.reference_k), view(self.reference_k1));
        let (sk, sk1) = (view(self.secondary_k), view(self.secondary_k1));
        let mut sum = Complex32::new(0.0, 0.0);
        let mut magnitude = 0.0_f64;
        let mut df = 0.0_f64;
        let dk = self.reference_k.doppler.slice(window);
        let dk1 = self.reference_k1.doppler.slice(window);
        for (((a, b), (x, y)), (fa, fb)) in rk
            .iter()
            .zip(sk.iter())
            .zip(rk1.iter().zip(sk1.iter()))
            .zip(dk.iter().zip(dk1.iter()))
        {
            let z = (a * b.conj()) * (x * y.conj()).conj();
            if !z.re.is_finite() || !fa.is_finite() || !fb.is_finite() {
                return None;
            }
            sum += z;
            magnitude += f64::from(z.norm());
            df += f64::from(fa - fb);
        }
        let coherence = f64::from(sum.norm()) / magnitude;
        let df = df / (lr * lc) as f64;
        (coherence >= ESD_MIN_COHERENCE && df.abs() > 1.0).then(|| {
            (
                f64::from(sum.arg()) / (2.0 * std::f64::consts::PI * df),
                coherence,
                df,
            )
        })
    }
}
