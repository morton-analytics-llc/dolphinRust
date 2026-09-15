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
//!
//! # Limit: the root is a majority vote
//!
//! Each loop's integer-cycle root is the rounded median of the residuals over
//! the pixels sharing a component triple, so a triple in which **more than half
//! the pixels carry the same `n`-cycle error takes that error as its root**:
//! the wrong majority passes and the correct minority is flagged. The gate
//! cannot tell a component-wide root from a component-wide error; nothing in
//! a closed loop can. [`MIN_ROOT_PIXELS`] bounds how small such a majority can
//! be (a 1- or 2-pixel triple would otherwise validate itself — two pixels
//! `[0, 2π]` have an upper median of `2π`, which flags the *correct* one) but
//! does not remove the limit: 9 wrong pixels of 16 still become the root.
//! Triples below the floor contribute no root; their pixels are unjudged, not
//! graded, and counted in [`LoopClosureQc::unjudged_small_triple_pixels`].

use ndarray::{Array2, Array3, ArrayView3};
use rayon::prelude::*;
use std::collections::HashMap;

/// Fraction of a `2π` cycle a loop may miss closure by before it is called bad.
/// A consistent unwrap closes a loop to exactly zero up to interpolation noise;
/// a single unwrap error contributes a full `2π`. Half a cycle is the natural
/// midpoint and is what puts a pixel on the wrong side of the nearest integer.
pub const DEFAULT_CLOSURE_TOLERANCE_CYCLES: f64 = 0.5;

/// Fewest finite residuals a component triple needs before its median is
/// trusted as the loop's root. Below this the triple contributes no root and
/// its pixels are unjudged (see the module header).
pub const MIN_ROOT_PIXELS: usize = 16;

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

/// The connected-component labels are not the shape of the unwrapped stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopClosureShapeError {
    /// Shape of the unwrapped stack, `(n_ifgs, rows, cols)`.
    pub unwrapped: (usize, usize, usize),
    /// Shape of the connected-component labels.
    pub connected_components: (usize, usize, usize),
}

impl std::fmt::Display for LoopClosureShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "connected-component labels {:?} must match the unwrapped stack {:?}",
            self.connected_components, self.unwrapped
        )
    }
}

impl std::error::Error for LoopClosureShapeError {}

/// Per-pixel loop-closure QC over the unwrapped network.
#[derive(Debug)]
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
    /// Pixels with no evaluable loop because at least one of their loops fell
    /// in a component triple below [`MIN_ROOT_PIXELS`]. Counted on the analysed
    /// grid, before any crop or publication mask.
    pub unjudged_small_triple_pixels: usize,
    /// Component triples, over every loop, that had fewer than
    /// [`MIN_ROOT_PIXELS`] finite residuals and so contributed no root.
    pub small_root_triples: usize,
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
/// # Errors
/// [`LoopClosureShapeError`] if `connected_components` is not the shape of
/// `unwrapped`.
pub fn loop_closure_qc(
    unwrapped: ArrayView3<f64>,
    connected_components: ArrayView3<u32>,
    pairs: &[(usize, usize)],
    tolerance_cycles: f64,
) -> Result<LoopClosureQc, LoopClosureShapeError> {
    if connected_components.dim() != unwrapped.dim() {
        return Err(LoopClosureShapeError {
            unwrapped: unwrapped.dim(),
            connected_components: connected_components.dim(),
        });
    }
    let (_, rows, cols) = unwrapped.dim();
    let triplets = network_triplets(pairs);
    let tolerance = tolerance_cycles * std::f64::consts::TAU;
    let roots: Vec<ComponentRoots> = triplets
        .par_iter()
        .map(|triplet| triangle_roots(unwrapped, connected_components, triplet))
        .collect();

    let small_root_triples = roots
        .iter()
        .map(|loop_roots| loop_roots.values().filter(|root| root.is_none()).count())
        .sum();
    let per_pixel: Vec<PixelLoopStats> = (0..rows * cols)
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
    let layer = |pick: fn(&PixelLoopStats) -> f64| {
        Array2::from_shape_fn((rows, cols), |(r, c)| pick(&per_pixel[r * cols + c]))
    };
    Ok(LoopClosureQc {
        bad_loop_count: layer(|stats| stats.bad),
        evaluable_loop_count: layer(|stats| stats.evaluable),
        worst_residual_cycles: layer(|stats| stats.worst_residual_cycles),
        unjudged_small_triple_pixels: per_pixel
            .iter()
            .filter(|stats| stats.evaluable == 0.0 && stats.withheld_small_triple)
            .count(),
        small_root_triples,
    })
}

/// The component labels of a loop's three members at one pixel.
type ComponentKey = [u32; 3];

/// One loop's integer-cycle root per component triple, in radians; `None` for
/// a triple with fewer than [`MIN_ROOT_PIXELS`] finite residuals.
type ComponentRoots = HashMap<ComponentKey, Option<f64>>;

/// One pixel's loop verdicts.
struct PixelLoopStats {
    bad: f64,
    evaluable: f64,
    worst_residual_cycles: f64,
    /// At least one loop through the pixel was withheld because its component
    /// triple fell below [`MIN_ROOT_PIXELS`].
    withheld_small_triple: bool,
}

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
/// unwrap error can never be charged to the rest of the frame. Triples with
/// fewer than [`MIN_ROOT_PIXELS`] finite residuals, and any pixel with a
/// label-0 member, contribute no root.
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
        .map(|(key, mut values)| {
            let root =
                (values.len() >= MIN_ROOT_PIXELS).then(|| rounded_median_cycles(&mut values));
            (key, root)
        })
        .collect()
}

/// The median of `values` rounded to a whole number of cycles, in radians.
/// `values` must be non-empty.
fn rounded_median_cycles(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    let (_, median, _) = values.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
    (*median / std::f64::consts::TAU).round() * std::f64::consts::TAU
}

/// One pixel's verdicts, after removing each loop's integer-cycle root for the
/// component triple the pixel belongs to.
fn pixel_loop_stats(
    unwrapped: ArrayView3<f64>,
    connected_components: ArrayView3<u32>,
    triplets: &[Triplet],
    roots: &[ComponentRoots],
    (row, col): (usize, usize),
    tolerance: f64,
) -> PixelLoopStats {
    let mut bad = 0.0;
    let mut evaluable = 0.0;
    let mut worst = f64::NAN;
    let mut withheld_small_triple = false;
    for (triplet, loop_roots) in triplets.iter().zip(roots) {
        let root = match component_key(connected_components, triplet, (row, col))
            .and_then(|key| loop_roots.get(&key))
        {
            Some(Some(root)) => *root,
            Some(None) => {
                withheld_small_triple = true;
                continue;
            }
            None => continue,
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
    PixelLoopStats {
        bad,
        evaluable,
        worst_residual_cycles: worst,
        withheld_small_triple,
    }
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

    /// Grid side of the test frame: 49 pixels, so one component clears
    /// [`MIN_ROOT_PIXELS`] and so does each half when a test splits it.
    const SIDE: usize = 7;

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
        let unwrapped = Array3::from_shape_fn((pairs.len(), SIDE, SIDE), |(k, _, _)| {
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
        .unwrap()
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
        // Interferogram 1 splits: the bottom three rows (21 pixels) are their own
        // component, one cycle up.
        for (row, col) in (4..SIDE).flat_map(|row| (0..SIDE).map(move |col| (row, col))) {
            unwrapped[(1, row, col)] += std::f64::consts::TAU;
            components[(1, row, col)] = 2;
        }
        let clean = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        )
        .unwrap();
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
        .unwrap()
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
        )
        .unwrap();
        let full = qc.evaluable_loop_count[(0, 0)];
        assert!(qc.evaluable_loop_count[(1, 1)] < full);
        assert_eq!(qc.bad_loop_count[(1, 1)], 0.0);
        assert!(!qc.failed_mask()[(1, 1)]);
    }

    /// Masked pixels are zero-filled before unwrapping and close every loop to
    /// exactly zero. When they are the majority they would pin a frame-wide
    /// median to that zero; carrying no label, they are outside every root and
    /// the valid minority keeps its own.
    #[test]
    fn masked_majority_does_not_pin_the_root() {
        let (pairs, mut unwrapped) = consistent_network();
        let mut components = one_component(&unwrapped);
        for (k, root) in [1.0, -2.0, 3.0, 0.0, 1.0].into_iter().enumerate() {
            unwrapped
                .index_axis_mut(ndarray::Axis(0), k)
                .mapv_inplace(|v| v + root * std::f64::consts::TAU);
        }
        // 28 of 49 pixels are masked: zero phase, no component.
        for (row, col) in (0..SIDE).flat_map(|row| (0..SIDE).map(move |col| (row, col))) {
            if row + col < SIDE {
                unwrapped.slice_mut(ndarray::s![.., row, col]).fill(0.0);
                components.slice_mut(ndarray::s![.., row, col]).fill(0);
            }
        }
        let qc = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        )
        .unwrap();
        assert!(
            !qc.failed_mask().iter().any(|&bad| bad),
            "{:?}",
            qc.bad_loop_count
        );
        assert_eq!(qc.evaluable_loop_count[(0, 0)], 0.0);
        assert!(qc.evaluable_loop_count[(6, 6)] > 0.0);
        assert!(qc.worst_residual_cycles[(6, 6)] < 1e-12);
    }

    /// A component triple with two pixels has no trustworthy median: `[0, 2π]`
    /// has an upper median of `2π`, which would make the error the root and
    /// flag the correct pixel. Below [`MIN_ROOT_PIXELS`] the triple contributes
    /// no root, both pixels are unjudged rather than graded, and the counts say
    /// so.
    #[test]
    fn a_two_pixel_triple_is_unjudged_not_graded() {
        let (pairs, mut unwrapped) = consistent_network();
        let mut components = one_component(&unwrapped);
        for ifg in 0..pairs.len() {
            components[(ifg, 0, 0)] = 2;
            components[(ifg, 0, 1)] = 2;
        }
        unwrapped[(1, 0, 1)] += std::f64::consts::TAU;
        let qc = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        )
        .unwrap();
        assert!(
            !qc.failed_mask().iter().any(|&bad| bad),
            "an unjudged pixel is never flagged: {:?}",
            qc.bad_loop_count
        );
        assert_eq!(qc.evaluable_loop_count[(0, 0)], 0.0);
        assert_eq!(qc.evaluable_loop_count[(0, 1)], 0.0);
        assert!(qc.worst_residual_cycles[(0, 1)].is_nan());
        assert!(qc.evaluable_loop_count[(3, 3)] > 0.0);
        assert_eq!(qc.unjudged_small_triple_pixels, 2);
        assert_eq!(qc.small_root_triples, network_triplets(&pairs).len());
    }

    /// The documented limit: at the floor, a 16-pixel triple in which 9 pixels
    /// share the same one-cycle error takes that error as its root, passes the
    /// nine and flags the seven correct pixels. The floor bounds how small a
    /// self-validating majority can be; it does not remove the limit.
    #[test]
    fn a_majority_error_at_the_floor_becomes_the_root() {
        let (pairs, mut unwrapped) = consistent_network();
        let mut components = one_component(&unwrapped);
        let triple: Vec<(usize, usize)> = (0..MIN_ROOT_PIXELS)
            .map(|index| (index / SIDE, index % SIDE))
            .collect();
        for ifg in 0..pairs.len() {
            for &(row, col) in &triple {
                components[(ifg, row, col)] = 2;
            }
        }
        let (wrong, correct) = triple.split_at(9);
        for &(row, col) in wrong {
            unwrapped[(0, row, col)] += std::f64::consts::TAU;
        }
        let qc = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        )
        .unwrap();
        let flagged: Vec<(usize, usize)> = qc
            .failed_mask()
            .indexed_iter()
            .filter(|(_, &bad)| bad)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(flagged, correct.to_vec());
        assert_eq!(qc.unjudged_small_triple_pixels, 0);
        assert_eq!(qc.small_root_triples, 0);
    }

    /// Labels of the wrong shape are a caller error reported as a value, not a
    /// panic in library code.
    #[test]
    fn mismatched_label_shape_is_an_error_not_a_panic() {
        let (pairs, unwrapped) = consistent_network();
        let components = Array3::<u32>::ones((pairs.len(), SIDE, SIDE - 1));
        let error = loop_closure_qc(
            unwrapped.view(),
            components.view(),
            &pairs,
            DEFAULT_CLOSURE_TOLERANCE_CYCLES,
        )
        .unwrap_err();
        assert_eq!(error.unwrapped, (pairs.len(), SIDE, SIDE));
        assert_eq!(error.connected_components, (pairs.len(), SIDE, SIDE - 1));
        assert!(error.to_string().contains("must match the unwrapped stack"));
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
