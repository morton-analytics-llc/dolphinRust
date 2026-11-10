//! Pre-unwrap interpolation of low-quality wrapped phase.
//!
//! Reference: dolphin `src/dolphin/interpolation.py` (pinned v0.35.0; v0.42.0 is
//! identical bar the logger name), after Chen et al. 2015 and Wang et al. 2022.
//! A pixel whose weight is below `weight_cutoff` has its wrapped phase replaced by
//! a Gaussian-distance-weighted phasor sum over the first `num_neighbors` pixels
//! at or above the cutoff, visited in [`circle_offsets`] order (centre outward)
//! out to `max_radius`. Amplitude is kept; only the phase changes. Pixels that are
//! exactly zero are nodata and come back zero.

use dolphin_core::Cf32;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::similarity::circle_offsets;

/// dolphin's `interpolate` default for `num_neighbors`; the unwrap stage never
/// overrides it.
pub const INTERPOLATION_NUM_NEIGHBORS: usize = 20;

/// dolphin's `interpolate` default for the distance-weight exponent `alpha`; the
/// unwrap stage never overrides it. Unrelated to the Goldstein
/// `preprocess_options.alpha`.
pub const INTERPOLATION_ALPHA: f64 = 0.75;

/// Interpolate a wrapped interferogram at pixels whose weight is below
/// `weight_cutoff`. Port of dolphin's `interpolate` with `min_radius = 0`.
///
/// Weights are clipped to `[0, 1]` first, as dolphin does. A low-weight pixel with
/// no qualifying neighbour inside `max_radius` keeps its amplitude with zero phase
/// (dolphin's `angle(0) = 0`).
#[must_use]
pub fn interpolate(
    ifg: ArrayView2<Cf32>,
    weights: ArrayView2<f32>,
    weight_cutoff: f32,
    num_neighbors: usize,
    max_radius: usize,
    alpha: f64,
) -> Array2<Cf32> {
    assert_eq!(ifg.dim(), weights.dim(), "ifg and weights shapes differ");
    let weights = weights.mapv(|w| w.clamp(0.0, 1.0));
    let offsets = circle_offsets(max_radius);
    let (rows, cols) = ifg.dim();
    let search = Search {
        ifg: ifg.view(),
        weights: weights.view(),
        weight_cutoff,
        offsets: &offsets,
        num_neighbors,
        alpha,
    };
    let values: Vec<Cf32> = (0..rows)
        .into_par_iter()
        .flat_map_iter(|row| (0..cols).map(move |col| (row, col)))
        .map(|point| search.pixel(point))
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("row-major values match the ifg shape")
}

/// One interpolation pass's inputs, shared by every pixel.
struct Search<'a> {
    ifg: ArrayView2<'a, Cf32>,
    weights: ArrayView2<'a, f32>,
    weight_cutoff: f32,
    offsets: &'a [(i32, i32)],
    num_neighbors: usize,
    alpha: f64,
}

impl Search<'_> {
    /// dolphin's per-pixel branch: nodata stays zero, a qualifying pixel keeps its
    /// value, any other takes the neighbour phase at its own amplitude.
    fn pixel(&self, point: (usize, usize)) -> Cf32 {
        let z = self.ifg[point];
        if z == Cf32::new(0.0, 0.0) || self.weights[point] >= self.weight_cutoff {
            return z;
        }
        let value =
            Complex64::from_polar(f64::from(z.norm()), self.neighbour_phasor_sum(point).arg());
        Cf32::new(value.re as f32, value.im as f32)
    }

    /// Gaussian-distance-weighted sum of the unit phasors of the first
    /// `num_neighbors` qualifying neighbours, normalised by the farthest one's
    /// squared distance raised to `alpha` — dolphin's `_interp_loop` body.
    fn neighbour_phasor_sum(&self, (row, col): (usize, usize)) -> Complex64 {
        let (rows, cols) = self.ifg.dim();
        let neighbours: Vec<(f64, f64)> = self
            .offsets
            .iter()
            .filter_map(|&(dr, dc)| {
                let r = row.checked_add_signed(dr as isize).filter(|&r| r < rows)?;
                let c = col.checked_add_signed(dc as isize).filter(|&c| c < cols)?;
                (self.weights[(r, c)] >= self.weight_cutoff).then(|| {
                    let z = self.ifg[(r, c)];
                    (f64::from(dr * dr + dc * dc), f64::from(z.im.atan2(z.re)))
                })
            })
            .take(self.num_neighbors)
            .collect();
        let Some(&(farthest, _)) = neighbours.last() else {
            return Complex64::new(0.0, 0.0);
        };
        let norm = farthest.powf(self.alpha) / 2.0;
        neighbours
            .iter()
            .map(|&(distance2, phase)| Complex64::from_polar((-distance2 / norm).exp(), phase))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherent_pixels_and_nodata_pass_through_unchanged() {
        let mut ifg = Array2::from_shape_fn((5, 6), |(r, c)| {
            Cf32::from_polar(1.0 + r as f32, 0.1 * c as f32)
        });
        ifg[(2, 2)] = Cf32::new(0.0, 0.0);
        let weights = Array2::from_elem((5, 6), 0.9_f32);
        let out = interpolate(ifg.view(), weights.view(), 0.5, 20, 51, 0.75);
        assert_eq!(out, ifg);
    }
}
