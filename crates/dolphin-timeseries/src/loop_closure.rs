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
use std::collections::HashMap;

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
    /// interferograms finite and labelled with a component that has a root).
    /// `0` means the pixel is unjudged, **not** clean.
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
/// `unwrapped` is `(n_ifgs, rows, cols)` in radians, indexed to match `pairs`;
/// `connected_components` are the unwrapper's per-interferogram labels on the
/// same grid (0 = no component). The residual of triangle `(i,j),(j,k),(i,k)`
/// is `φ_ij + φ_jk − φ_ik`, which a correctly unwrapped network closes to a
/// whole number of cycles (the members' independent unwrap roots). A root is
/// constant over a connected component, so that shared part is removed per
/// loop **and per component triple** `(c_ij, c_jk, c_ik)` before judging; an
/// unwrap error of `n` cycles in any one member at a pixel then drives it to
/// `±2πn`. A loop through a label-0 pixel has no root and is unevaluable.
/// `tolerance_cycles` is the fraction of a cycle allowed before the loop is
/// called bad (see [`DEFAULT_CLOSURE_TOLERANCE_CYCLES`]).
///
/// # Panics
/// If `connected_components` is not the shape of `unwrapped`.
#[must_use]
pub fn loop_closure_qc(
    unwrapped: ArrayView3<f64>,
    connected_components: ArrayView3<u32>,
    pairs: &[(usize, usize)],
    tolerance_cycles: f64,
) -> LoopClosureQc {
    assert_eq!(
        connected_components.dim(),
        unwrapped.dim(),
        "connected-component labels must match the unwrapped stack"
    );
    let (_, rows, cols) = unwrapped.dim();
    let triplets = network_triplets(pairs);
    let tolerance = tolerance_cycles * std::f64::consts::TAU;
    let roots: Vec<ComponentRoots> = triplets
        .par_iter()
        .map(|triplet| triangle_roots(unwrapped, connected_components, triplet))
        .collect();

    let per_pixel: Vec<(f64, f64, f64)> = (0..rows * cols)
        .into_par_iter()
        .map(|index| {
            let (row, col) = (index / cols, index % cols);
            pixel_loop_stats(
                unwrapped,
                connected_components,
                &triplets,
                &roots,
                (row, col),
                tolerance,
            )
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

/// The component labels of a loop's three members at one pixel.
type ComponentKey = [u32; 3];

/// One loop's integer-cycle root per component triple, in radians.
type ComponentRoots = HashMap<ComponentKey, f64>;

fn triangle_residual(
    unwrapped: ArrayView3<f64>,
    triplet: &Triplet,
    (row, col): (usize, usize),
) -> f64 {
    unwrapped[(triplet.early, row, col)] + unwrapped[(triplet.late, row, col)]
        - unwrapped[(triplet.span, row, col)]
}

/// The labels a loop's members carry at a pixel, or `None` when any member has
/// no component there: such a pixel belongs to no root and cannot be judged.
fn component_key(
    connected_components: ArrayView3<u32>,
    triplet: &Triplet,
    (row, col): (usize, usize),
) -> Option<ComponentKey> {
    let key = [
        connected_components[(triplet.early, row, col)],
        connected_components[(triplet.late, row, col)],
        connected_components[(triplet.span, row, col)],
    ];
    (!key.contains(&0)).then_some(key)
}

/// The integer-cycle part of a loop shared by every pixel of one component
/// triple. Each interferogram is unwrapped independently and carries an
/// arbitrary `2πn` root **per connected component**; the roots of a triangle's
/// three members do not cancel, so a consistent stack closes to a whole number
/// of cycles rather than to zero, and that number can differ between the
/// components an interferogram split into. It is the rounded median of the
/// finite residuals over the pixels sharing the triple — a property of the
/// loop and the components, not of any one pixel, so a reference pixel's own
/// unwrap error can never be charged to the rest of the frame. Triples with no
/// finite residual, and any pixel with a label-0 member, contribute no root.
fn triangle_roots(
    unwrapped: ArrayView3<f64>,
    connected_components: ArrayView3<u32>,
    triplet: &Triplet,
) -> ComponentRoots {
    let (_, rows, cols) = unwrapped.dim();
    let mut residuals: HashMap<ComponentKey, Vec<f64>> = HashMap::new();
    for point in (0..rows).flat_map(|row| (0..cols).map(move |col| (row, col))) {
        let Some(key) = component_key(connected_components, triplet, point) else {
            continue;
        };
        let residual = triangle_residual(unwrapped, triplet, point);
        if residual.is_finite() {
            residuals.entry(key).or_default().push(residual);
        }
    }
    residuals
        .into_iter()
        .map(|(key, mut values)| (key, rounded_median_cycles(&mut values)))
        .collect()
}

/// The median of `values` rounded to a whole number of cycles, in radians.
/// `values` must be non-empty.
fn rounded_median_cycles(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    let (_, median, _) = values.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
    (*median / std::f64::consts::TAU).round() * std::f64::consts::TAU
}

/// `(bad, evaluable, worst_residual_cycles)` for one pixel, after removing each
/// loop's integer-cycle root for the component triple the pixel belongs to.
fn pixel_loop_stats(
    unwrapped: ArrayView3<f64>,
    connected_components: ArrayView3<u32>,
    triplets: &[Triplet],
    roots: &[ComponentRoots],
    (row, col): (usize, usize),
    tolerance: f64,
) -> (f64, f64, f64) {
    let mut bad = 0.0;
    let mut evaluable = 0.0;
    let mut worst = f64::NAN;
    for (triplet, loop_roots) in triplets.iter().zip(roots) {
        let Some(root) = component_key(connected_components, triplet, (row, col))
            .and_then(|key| loop_roots.get(&key))
        else {
            continue;
        };
        let residual = triangle_residual(unwrapped, triplet, (row, col)) - root;
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

    /// Every interferogram unwrapped as one connected component.
    fn one_component(unwrapped: &Array3<f64>) -> Array3<u32> {
        Array3::from_elem(unwrapped.dim(), 1)
    }

    fn qc(unwrapped: &Array3<f64>, pairs: &[(usize, usize)]) -> LoopClosureQc {
        loop_closure_qc(
            unwrapped.view(),
            one_component(unwrapped).view(),
            pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        )
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
        let qc = qc(&unwrapped, &pairs);
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

        let qc = qc(&unwrapped, &pairs);
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
        let qc = qc(&unwrapped, &pairs);
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
        let qc = qc(&unwrapped, &pairs);
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

    /// Each interferogram is unwrapped independently, so it carries its own
    /// integer-cycle root. Those roots do not cancel around a triangle, and no
    /// pixel is a trustworthy reference for removing them: the common part of
    /// every loop must come from the loop itself, never from one pixel.
    #[test]
    fn per_interferogram_roots_flag_only_the_local_error() {
        let (pairs, mut unwrapped) = consistent_network();
        for (k, root) in [1.0, -2.0, 3.0, 0.0, 1.0].into_iter().enumerate() {
            unwrapped
                .index_axis_mut(ndarray::Axis(0), k)
                .mapv_inplace(|v| v + root * std::f64::consts::TAU);
        }
        unwrapped[(1, 2, 2)] += std::f64::consts::TAU;
        let qc = qc(&unwrapped, &pairs);
        let flagged: Vec<(usize, usize)> = qc
            .failed_mask()
            .indexed_iter()
            .filter(|(_, &bad)| bad)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            flagged,
            vec![(2, 2)],
            "roots are common to every pixel and must not be counted"
        );
        assert!((qc.worst_residual_cycles[(2, 2)] - 1.0).abs() < 1e-12);
        assert!(qc.worst_residual_cycles[(0, 0)] < 1e-12);
    }

    /// An unwrap root is constant over a connected component, not over the
    /// frame. When one interferogram unwraps as two components one cycle apart,
    /// a frame-wide median charges the smaller component with a whole-cycle
    /// residual and masks it wholesale; keyed by component the two roots are
    /// both removed and only a genuine local error is flagged.
    #[test]
    fn split_interferogram_flags_only_the_local_error() {
        let (pairs, mut unwrapped) = consistent_network();
        let mut components = one_component(&unwrapped);
        // Interferogram 1 splits: the bottom row is its own component, one cycle up.
        for col in 0..3 {
            unwrapped[(1, 2, col)] += std::f64::consts::TAU;
            components[(1, 2, col)] = 2;
        }
        let clean = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        );
        assert!(
            !clean.failed_mask().iter().any(|&bad| bad),
            "a component-wide cycle offset is a root, not an error: {:?}",
            clean.bad_loop_count
        );
        assert!(clean.evaluable_loop_count.iter().all(|&n| n > 0.0));

        unwrapped[(3, 0, 1)] += std::f64::consts::TAU;
        let flagged: Vec<(usize, usize)> = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        )
        .failed_mask()
        .indexed_iter()
        .filter(|(_, &bad)| bad)
        .map(|(index, _)| index)
        .collect();
        assert_eq!(flagged, vec![(0, 1)]);
    }

    /// Label 0 is the unwrapper's own "no component" verdict: a loop through such
    /// a pixel has no root to remove, so it is unevaluable rather than judged.
    #[test]
    fn unlabelled_pixels_are_unevaluable() {
        let (pairs, unwrapped) = consistent_network();
        let mut components = one_component(&unwrapped);
        components[(2, 1, 1)] = 0;
        let qc = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        );
        let full = qc.evaluable_loop_count[(0, 0)];
        assert!(qc.evaluable_loop_count[(1, 1)] < full);
        assert_eq!(qc.bad_loop_count[(1, 1)], 0.0);
        assert!(!qc.failed_mask()[(1, 1)]);
    }

    /// A sub-cycle residual (real noise, not an unwrap error) is not flagged.
    #[test]
    fn sub_cycle_noise_is_not_an_unwrap_error() {
        let (pairs, mut unwrapped) = consistent_network();
        unwrapped[(1, 1, 1)] += 0.4 * std::f64::consts::TAU;
        let qc = qc(&unwrapped, &pairs);
        assert_eq!(qc.bad_loop_count[(1, 1)], 0.0);
        assert!((qc.worst_residual_cycles[(1, 1)] - 0.4).abs() < 1e-12);
    }
}
