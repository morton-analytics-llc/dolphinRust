//! Post-unwrap loop-closure QC over the interferogram network (issue #24).
//!
//! # Why this is not the closure-phase layer we already have
//!
//! `dolphin-phaselink::closure::estimate_closure_phases` computes
//! `∠(C[k,k+1]·C[k+1,k+2]·conj(C[k,k+2]))` on the **coherence matrix** — wrapped
//! phase, before unwrapping. Its output is bounded to `(−π, π]` by the `.arg()`,
//! and it measures decorrelation-driven systematic bias, which the
//! Michaelides et al. phase-bias correction then models out.
//!
//! An unwrapping error is an integer multiple of `2π` in one interferogram. That
//! is **exactly the quantity `.arg()` discards**: a clean 2π error wraps to zero,
//! so the existing closure layer cannot see it, however good it is at what it
//! does. This module closes loops on the **unwrapped** network, where the same
//! error shows up as a nonzero multiple of `2π` in the loop sum. The two layers
//! are not different views of one signal; one is blind to the other's target.
//!
//! # Scope: over-determined networks only
//!
//! A loop needs three interferograms among three dates. A **single-reference**
//! network has none — every pair shares date 0 — so this gate has nothing to
//! close and reports no loops. It becomes meaningful only with
//! `interferogram_network.max_bandwidth` / `max_temporal_baseline` set. That is
//! also the network shape used by the independent-IFG parameter-covariance
//! approximation, but redundant interferograms share acquisition errors and do
//! not provide independent empirical scale. dolphin v0.42 adopted this network
//! shape as its default (issue #25).
//!
//! # What connected components contribute
//!
//! The per-interferogram `conncomp_NN.tif` labels already shipped give the
//! *granularity* for a correction (an unwrap error is constant over a connected
//! component, so a fix is one integer `2π` shift per component, not per pixel)
//! and a free prefilter (label 0 is already-unreliable). They carry no cross-
//! interferogram information, so they cannot supply the *detection* — that needs
//! the loop residual here.

use ndarray::{Array2, Array3, ArrayView3};
use rayon::prelude::*;

/// Fraction of a `2π` cycle a loop may miss closure by before it is called bad.
/// A consistent unwrap closes a loop to exactly zero up to interpolation noise;
/// a single unwrap error contributes a full `2π`. Half a cycle is the natural
/// midpoint and is what puts a pixel on the wrong side of the nearest integer.
pub const DEFAULT_CLOSURE_TOLERANCE_CYCLES: f64 = 0.5;

/// A closed triangle in the interferogram network: the indices, into the
/// interferogram list, of the pairs `(i,j)`, `(j,k)` and `(i,k)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triplet {
    /// Index of the `(i, j)` interferogram.
    pub early: usize,
    /// Index of the `(j, k)` interferogram.
    pub late: usize,
    /// Index of the `(i, k)` interferogram that should equal their sum.
    pub span: usize,
}

/// Per-pixel loop-closure QC over the unwrapped network.
pub struct LoopClosureQc {
    /// Number of loops through each pixel that failed to close, `(rows, cols)`.
    /// `0` where every loop closed; NaN input counts as a non-failure, since a
    /// missing observation is not evidence of an unwrap error.
    pub bad_loop_count: Array2<f64>,
    /// Number of loops that were evaluable at each pixel (all three
    /// interferograms finite). `0` means the pixel is unjudged, **not** clean.
    pub evaluable_loop_count: Array2<f64>,
    /// Largest absolute loop residual at each pixel, in cycles. NaN where no
    /// loop was evaluable.
    pub worst_residual_cycles: Array2<f64>,
}

impl LoopClosureQc {
    /// Pixels to mask before the SBAS solve: at least one loop through them
    /// failed to close. A pixel with no evaluable loop is **not** masked — this
    /// gate only ever acts on positive evidence.
    #[must_use]
    pub fn failed_mask(&self) -> Array2<bool> {
        Array2::from_shape_fn(self.bad_loop_count.dim(), |index| {
            self.bad_loop_count[index] > 0.0
        })
    }
}

/// Every closed triangle in the network, as indices into `pairs`.
///
/// `pairs` are `(early, later)` date indices as produced by
/// [`build_network`](crate::network::build_network). A triangle is a set
/// `(i,j), (j,k), (i,k)` with `i < j < k` where all three are present.
#[must_use]
pub fn network_triplets(pairs: &[(usize, usize)]) -> Vec<Triplet> {
    let index_of = |pair: (usize, usize)| pairs.iter().position(|&p| p == pair);
    pairs
        .iter()
        .enumerate()
        .flat_map(|(early, &(i, j))| {
            pairs
                .iter()
                .enumerate()
                .filter(move |(_, &(a, _))| a == j)
                .filter_map(move |(late, &(_, k))| {
                    Some(Triplet {
                        early,
                        late,
                        span: index_of((i, k))?,
                    })
                })
        })
        .collect()
}

/// Close every network triangle on the **unwrapped** stack and count failures.
///
/// `unwrapped` is `(n_ifgs, rows, cols)` in radians, indexed to match `pairs`.
/// The residual of triangle `(i,j),(j,k),(i,k)` is
/// `φ_ij + φ_jk − φ_ik`, which a correctly unwrapped network closes to ~0; an
/// unwrap error of `n` cycles in any one member drives it to `±2πn`.
/// `tolerance_cycles` is the fraction of a cycle allowed before the loop is
/// called bad (see [`DEFAULT_CLOSURE_TOLERANCE_CYCLES`]).
#[must_use]
pub fn loop_closure_qc(
    unwrapped: ArrayView3<f64>,
    pairs: &[(usize, usize)],
    tolerance_cycles: f64,
) -> LoopClosureQc {
    let (_, rows, cols) = unwrapped.dim();
    let triplets = network_triplets(pairs);
    let tolerance = tolerance_cycles * std::f64::consts::TAU;

    let per_pixel: Vec<(f64, f64, f64)> = (0..rows * cols)
        .into_par_iter()
        .map(|index| {
            let (row, col) = (index / cols, index % cols);
            pixel_loop_stats(unwrapped, &triplets, (row, col), tolerance)
        })
        .collect();
    let layer = |pick: fn(&(f64, f64, f64)) -> f64| {
        Array2::from_shape_fn((rows, cols), |(r, c)| pick(&per_pixel[r * cols + c]))
    };
    LoopClosureQc {
        bad_loop_count: layer(|stats| stats.0),
        evaluable_loop_count: layer(|stats| stats.1),
        worst_residual_cycles: layer(|stats| stats.2),
    }
}

/// `(bad, evaluable, worst_residual_cycles)` for one pixel.
fn pixel_loop_stats(
    unwrapped: ArrayView3<f64>,
    triplets: &[Triplet],
    (row, col): (usize, usize),
    tolerance: f64,
) -> (f64, f64, f64) {
    let mut bad = 0.0;
    let mut evaluable = 0.0;
    let mut worst = f64::NAN;
    for triplet in triplets {
        let residual = unwrapped[(triplet.early, row, col)] + unwrapped[(triplet.late, row, col)]
            - unwrapped[(triplet.span, row, col)];
        if !residual.is_finite() {
            continue;
        }
        evaluable += 1.0;
        let cycles = residual.abs() / std::f64::consts::TAU;
        worst = match worst.is_nan() {
            true => cycles,
            false => worst.max(cycles),
        };
        bad += f64::from(residual.abs() > tolerance);
    }
    (bad, evaluable, worst)
}

/// Blank every interferogram at pixels where a loop failed to close, so a bad
/// unwrap becomes missing data rather than a confident wrong number in the SBAS
/// solve. Pixels with no evaluable loop are left alone.
pub fn mask_failed_loops(unwrapped: &mut Array3<f64>, qc: &LoopClosureQc) {
    let failed = qc.failed_mask();
    for mut band in unwrapped.outer_iter_mut() {
        ndarray::Zip::from(&mut band)
            .and(&failed)
            .for_each(|value, &bad| {
                if bad {
                    *value = f64::NAN;
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{build_network, NetworkConfig};

    /// A nearest-2 network on 4 dates and the true (unwrapped) phase of a linear
    /// ramp, so every loop closes exactly.
    fn consistent_network() -> (Vec<(usize, usize)>, Array3<f64>) {
        let pairs = build_network(
            4,
            &[0.0, 12.0, 24.0, 36.0],
            &NetworkConfig {
                max_bandwidth: Some(2),
                ..Default::default()
            },
        );
        // True per-date phase; every ifg is the difference, so loops close to 0.
        let phase = [0.0, 1.3, 2.9, 4.1];
        let unwrapped = Array3::from_shape_fn((pairs.len(), 3, 3), |(k, _, _)| {
            let (i, j) = pairs[k];
            phase[j] - phase[i]
        });
        (pairs, unwrapped)
    }

    /// A nearest-2 network on 4 dates has closed triangles; a single-reference
    /// one has none, which is the scope limit this module documents.
    #[test]
    fn triplets_need_an_over_determined_network() {
        let (pairs, _) = consistent_network();
        assert!(!network_triplets(&pairs).is_empty());

        let single = build_network(
            4,
            &[0.0, 12.0, 24.0, 36.0],
            &NetworkConfig {
                reference_idx: Some(0),
                ..Default::default()
            },
        );
        assert!(
            network_triplets(&single).is_empty(),
            "a single-reference network has no loops to close"
        );
    }

    /// A correctly unwrapped network flags nothing.
    #[test]
    fn consistent_network_flags_nothing() {
        let (pairs, unwrapped) = consistent_network();
        let qc = loop_closure_qc(unwrapped.view(), &pairs, DEFAULT_CLOSURE_TOLERANCE_CYCLES);
        assert!(qc.bad_loop_count.iter().all(|&count| count == 0.0));
        assert!(qc.evaluable_loop_count.iter().all(|&count| count > 0.0));
        assert!(qc.worst_residual_cycles.iter().all(|&r| r < 1e-12));
        assert!(!qc.failed_mask().iter().any(|&bad| bad));
    }

    /// The contract: a 2π unwrap error injected into one interferogram at one
    /// pixel is detected there, and only there.
    #[test]
    fn detects_a_single_cycle_unwrap_error() {
        let (pairs, mut unwrapped) = consistent_network();
        unwrapped[(1, 1, 1)] += std::f64::consts::TAU;

        let qc = loop_closure_qc(unwrapped.view(), &pairs, DEFAULT_CLOSURE_TOLERANCE_CYCLES);
        assert!(
            qc.bad_loop_count[(1, 1)] > 0.0,
            "the error pixel is flagged"
        );
        assert!(
            (qc.worst_residual_cycles[(1, 1)] - 1.0).abs() < 1e-12,
            "residual should be exactly one cycle, got {}",
            qc.worst_residual_cycles[(1, 1)]
        );
        let flagged: usize = qc.failed_mask().iter().filter(|&&bad| bad).count();
        assert_eq!(flagged, 1, "only the error pixel is flagged");
    }

    /// The design-review claim, as a test: the same 2π error is **invisible** to
    /// the wrapped closure phase the pipeline already computes, because wrapping
    /// maps a whole cycle to zero. This is why the two layers are not redundant.
    #[test]
    fn wrapped_closure_is_blind_to_a_whole_cycle_error() {
        let clean = 0.37_f64;
        let with_error = clean + std::f64::consts::TAU;
        // What a wrapped closure statistic sees, on either value.
        let wrap = |value: f64| {
            (value + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
        };
        assert!(
            (wrap(clean) - wrap(with_error)).abs() < 1e-12,
            "a wrapped statistic cannot distinguish a 2π unwrap error"
        );
        // What this module sees, on the same values.
        assert!((with_error - clean).abs() > std::f64::consts::PI);
    }

    /// A NaN in one interferogram makes a loop unevaluable, not failed — a
    /// missing observation is not evidence of an unwrap error.
    #[test]
    fn missing_data_is_unevaluable_not_failed() {
        let (pairs, mut unwrapped) = consistent_network();
        for k in 0..pairs.len() {
            unwrapped[(k, 0, 0)] = f64::NAN;
        }
        let qc = loop_closure_qc(unwrapped.view(), &pairs, DEFAULT_CLOSURE_TOLERANCE_CYCLES);
        assert_eq!(qc.evaluable_loop_count[(0, 0)], 0.0);
        assert_eq!(qc.bad_loop_count[(0, 0)], 0.0);
        assert!(qc.worst_residual_cycles[(0, 0)].is_nan());
        assert!(!qc.failed_mask()[(0, 0)], "unjudged is not masked");
    }

    /// Masking turns a bad unwrap into missing data across every interferogram,
    /// leaving good pixels untouched.
    #[test]
    fn masking_blanks_only_the_failed_pixels() {
        let (pairs, mut unwrapped) = consistent_network();
        unwrapped[(1, 2, 0)] += std::f64::consts::TAU;
        let qc = loop_closure_qc(unwrapped.view(), &pairs, DEFAULT_CLOSURE_TOLERANCE_CYCLES);
        mask_failed_loops(&mut unwrapped, &qc);

        assert!(unwrapped
            .slice(ndarray::s![.., 2, 0])
            .iter()
            .all(|v| v.is_nan()));
        assert!(unwrapped
            .slice(ndarray::s![.., 0, 0])
            .iter()
            .all(|v| v.is_finite()));
    }

    /// A sub-cycle residual (real noise, not an unwrap error) is not flagged.
    #[test]
    fn sub_cycle_noise_is_not_an_unwrap_error() {
        let (pairs, mut unwrapped) = consistent_network();
        unwrapped[(1, 1, 1)] += 0.4 * std::f64::consts::TAU;
        let qc = loop_closure_qc(unwrapped.view(), &pairs, DEFAULT_CLOSURE_TOLERANCE_CYCLES);
        assert_eq!(qc.bad_loop_count[(1, 1)], 0.0);
        assert!((qc.worst_residual_cycles[(1, 1)] - 0.4).abs() < 1e-12);
    }
}
