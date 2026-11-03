//! Integer-cycle correction of unwrap errors from loop closure (issue #142).
//!
//! # What it does
//!
//! [`loop_closure`](crate::loop_closure) closes every triangle of the
//! **unwrapped** network and masks a pixel whose loops miss closure by more than
//! half a cycle. This module runs first and repairs what it can: for each pixel
//! it solves for the integer number of `2π` cycles to add to each
//! interferogram so that every loop closes, choosing the solution that changes
//! the fewest cycles. That is the phase-closure correction of Yunjun et al.
//! (2019, Computers & Geosciences 133, 104331; MintPy's
//! `unwrap_error_phase_closure`): `U = argmin ‖U‖₁ subject to C U = −k`, with `C`
//! the triangle–interferogram incidence and `k` each loop's integer misclosure.
//! The L1 objective is the majority-rightness assumption: most interferograms
//! around a loop are right, so the smallest correction is the likeliest one.
//! The gate then runs on the corrected stack as QC.
//!
//! # How it is solved, exactly
//!
//! MintPy solves the relaxed problem and rounds. Here the solve is exact and
//! says when the answer is not unique. Every correction that closes all loops
//! is `U = a + B m`, where `a` is one particular solution and `B m` is the
//! gradient of an integer per-date potential `m` (a gradient closes every loop
//! by construction). `a` is built along a fan of triangles
//! `(i, k−1), (k−1, k), (i, k)`, which every bandwidth or temporal-baseline
//! network contains, so the triangles span the network's whole cycle space.
//! Minimising `‖a + B m‖₁` over integer `m` is a min-cost tension problem; its
//! LP dual is a min-cost circulation with unit capacities, solved here by
//! negative-cycle cancelling. The incidence `B` is totally unimodular, so the LP
//! optimum is integral, and the optimal potentials are read off the final
//! residual graph as shortest-path distances. They are unique (up to the
//! datum) exactly when every date's interval of optimal potentials collapses to
//! a point; anything else is a tie between equally small corrections, and the
//! pixel is **left uncorrected** for the gate to mask. The method never picks
//! among equally good answers.
//!
//! # What is left alone
//!
//! A pixel is corrected only when every loop through it is evaluable (all
//! three members finite, labelled, in a component triple with a root), every
//! misclosure is within [`MAX_CORRECTION_FRACTION_CYCLES`] of a whole number of
//! cycles, the integer misclosures admit an exact integer correction, that
//! correction is unique. A pixel failing any of these keeps its unwrapped values
//! and its status says which test it failed ([`PixelCorrection`]); the gate then
//! judges it exactly as it would without the correction.
//!
//! # Ministack junctions
//!
//! Interferograms are formed from the linked per-date phase, so the *wrapped*
//! network closes exactly whatever happens at a ministack junction: a per-date
//! offset, integer or not, is common to every interferogram touching that date
//! and cancels around every loop. A loop misclosure on this network is
//! therefore always a per-interferogram unwrap error. A junction can make such
//! errors more likely (the interferograms that span it are the ones the
//! unwrapper finds hardest), and [`LoopClosureCorrection::corrected_per_band`]
//! records which interferograms were corrected so that is visible; but a
//! junction offset itself is invisible to closure and is not corrected here.
//!
//! # Roots
//!
//! The integer misclosure `k` is measured against the same per-loop,
//! per-component-triple root the gate uses, so a component-wide unwrap offset
//! is a root, not an error, here as there, and the gate's majority-vote limit
//! on that root applies to the correction as well.

use crate::loop_closure::{
    component_key, network_triplets, triangle_residual, triangle_roots, ComponentRoots,
    LoopClosureShapeError, Triplet,
};
use ndarray::{Array2, Array3, ArrayView3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Largest distance, in cycles, a loop's misclosure may sit from a whole number
/// of cycles before the pixel is left uncorrected. Loops over linked-phase
/// interferograms close to whole cycles up to interpolation noise (#138
/// measured 5.7e-7 cycle); a misclosure far from an integer is not an unwrap
/// error, and rounding it would be a guess.
pub const MAX_CORRECTION_FRACTION_CYCLES: f64 = 0.25;

/// What the correction did at one pixel. The discriminant is the code written
/// to the `loop_closure_correction_status` raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelCorrection {
    /// Every loop already closed; nothing to correct.
    Closed = 0,
    /// A unique minimal integer correction was applied.
    Corrected = 1,
    /// More than one correction closes every loop with the same number of
    /// cycles (too few independent loops to tell them apart); left alone.
    Ambiguous = 2,
    /// At least one loop through the pixel is not evaluable (non-finite value,
    /// label 0, or a component triple with no root); left alone.
    Unevaluable = 3,
    /// At least one misclosure is not within
    /// [`MAX_CORRECTION_FRACTION_CYCLES`] of a whole cycle; left alone.
    NonInteger = 4,
    /// The integer misclosures are mutually inconsistent: no integer
    /// correction closes every loop; left alone.
    Inconsistent = 5,
}

/// Per-pixel provenance of the correction, on the analysed grid.
#[derive(Debug)]
pub struct LoopClosureCorrection {
    /// [`PixelCorrection`] code at each pixel, `(rows, cols)`.
    pub status: Array2<f64>,
    /// Number of interferograms whose integer cycles were changed at each
    /// pixel; 0 wherever the status is not [`PixelCorrection::Corrected`].
    pub corrected_interferogram_count: Array2<f64>,
    /// Pixels corrected in each interferogram, indexed like the network pairs.
    pub corrected_per_band: Vec<usize>,
}

impl LoopClosureCorrection {
    /// Pixels with the given status.
    #[must_use]
    pub fn count(&self, status: PixelCorrection) -> usize {
        let code = f64::from(status as u8);
        self.status.iter().filter(|&&value| value == code).count()
    }
}

/// The correction cannot run on this input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopCorrectionError {
    /// The connected-component labels are not the shape of the stack.
    Shape(LoopClosureShapeError),
    /// The network lacks the sequential pair `(date, date + 1)`.
    MissingSequentialPair {
        /// Earlier date index of the missing pair.
        date: usize,
    },
    /// The network has `pair` but not `(pair.0, pair.1 − 1)`, so its loops do
    /// not span the cycle space through a fan of triangles.
    MissingFanTriangle {
        /// The pair whose fan triangle is incomplete.
        pair: (usize, usize),
    },
}

impl std::fmt::Display for LoopCorrectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(error) => error.fmt(f),
            Self::MissingSequentialPair { date } => write!(
                f,
                "loop-closure correction needs every sequential pair; ({date}, {}) is missing",
                date + 1
            ),
            Self::MissingFanTriangle { pair: (i, k) } => write!(
                f,
                "loop-closure correction needs pair ({i}, {}) to close the triangle under ({i}, {k}); \
                 use a bandwidth or temporal-baseline network",
                k - 1
            ),
        }
    }
}

impl std::error::Error for LoopCorrectionError {}

impl From<LoopClosureShapeError> for LoopCorrectionError {
    fn from(error: LoopClosureShapeError) -> Self {
        Self::Shape(error)
    }
}

/// Check that `pairs` support the correction (see the module header), without
/// running it, so a run can fail before the expensive stages.
///
/// # Errors
/// [`LoopCorrectionError`] naming the pair the network lacks.
pub fn check_correction_network(pairs: &[(usize, usize)]) -> Result<(), LoopCorrectionError> {
    FanBasis::new(pairs, &network_triplets(pairs)).map(|_| ())
}

/// Correct integer-cycle unwrap errors in place, before the loop-closure gate.
///
/// `unwrapped` is `(n_ifgs, rows, cols)` in radians indexed like `pairs`, and
/// `connected_components` the unwrapper's labels on the same grid, exactly as
/// for [`loop_closure_qc`](crate::loop_closure_qc). Each corrected pixel has
/// `2π·U` added to its interferograms; every other pixel is untouched.
///
/// # Errors
/// [`LoopCorrectionError`] if the labels are the wrong shape or the network
/// does not support the correction.
pub fn correct_loop_closure_errors(
    unwrapped: &mut Array3<f64>,
    connected_components: ArrayView3<u32>,
    pairs: &[(usize, usize)],
) -> Result<LoopClosureCorrection, LoopCorrectionError> {
    if connected_components.dim() != unwrapped.dim() {
        return Err(LoopClosureShapeError {
            unwrapped: unwrapped.dim(),
            connected_components: connected_components.dim(),
        }
        .into());
    }
    let (n_ifgs, rows, cols) = unwrapped.dim();
    let triplets = network_triplets(pairs);
    let basis = FanBasis::new(pairs, &triplets)?;
    let view = unwrapped.view();
    let roots: Vec<ComponentRoots> = triplets
        .par_iter()
        .map(|triplet| triangle_roots(view, connected_components, triplet))
        .collect();
    let loops = Loops {
        unwrapped: view,
        connected_components,
        triplets: &triplets,
        roots: &roots,
    };
    let outcomes: Vec<(PixelCorrection, Vec<i64>)> = (0..rows * cols)
        .into_par_iter()
        .map(|index| correct_pixel(&loops, &basis, (index / cols, index % cols)))
        .collect();

    let mut corrected_per_band = vec![0; n_ifgs];
    let mut corrected_interferogram_count = Array2::zeros((rows, cols));
    let status = Array2::from_shape_fn((rows, cols), |(row, col)| {
        f64::from(outcomes[row * cols + col].0 as u8)
    });
    for (index, (_, cycles)) in outcomes.iter().enumerate() {
        let (row, col) = (index / cols, index % cols);
        for (band, &n) in cycles.iter().enumerate().filter(|&(_, &n)| n != 0) {
            unwrapped[(band, row, col)] += TAU * n as f64;
            corrected_per_band[band] += 1;
            corrected_interferogram_count[(row, col)] += 1.0;
        }
    }
    Ok(LoopClosureCorrection {
        status,
        corrected_interferogram_count,
        corrected_per_band,
    })
}

/// The loops of the network and their per-component roots on one stack.
struct Loops<'stack, 'labels, 'loops> {
    unwrapped: ArrayView3<'stack, f64>,
    connected_components: ArrayView3<'labels, u32>,
    triplets: &'loops [Triplet],
    roots: &'loops [ComponentRoots],
}

/// One pixel's status and, when corrected, the integer cycles to add to each
/// interferogram (empty otherwise).
fn correct_pixel(
    loops: &Loops<'_, '_, '_>,
    basis: &FanBasis,
    point: (usize, usize),
) -> (PixelCorrection, Vec<i64>) {
    let misclosure: Result<Vec<i64>, PixelCorrection> = loops
        .triplets
        .iter()
        .zip(loops.roots)
        .map(|(triplet, roots)| loop_misclosure(loops, triplet, roots, point))
        .collect();
    let misclosure = match misclosure {
        Ok(misclosure) => misclosure,
        Err(status) => return (status, Vec::new()),
    };
    if misclosure.iter().all(|&k| k == 0) {
        return (PixelCorrection::Closed, Vec::new());
    }
    let particular = basis.particular(&misclosure);
    if !closes_every_loop(loops.triplets, &particular, &misclosure) {
        return (PixelCorrection::Inconsistent, Vec::new());
    }
    match basis.minimal_correction(&particular) {
        Some(cycles) => (PixelCorrection::Corrected, cycles),
        None => (PixelCorrection::Ambiguous, Vec::new()),
    }
}

/// A loop's misclosure at a pixel in whole cycles, after removing the root of
/// the component triple the pixel belongs to.
fn loop_misclosure(
    loops: &Loops<'_, '_, '_>,
    triplet: &Triplet,
    roots: &ComponentRoots,
    point: (usize, usize),
) -> Result<i64, PixelCorrection> {
    let root = component_key(loops.connected_components, triplet, point)
        .and_then(|key| roots.get(&key).copied().flatten())
        .ok_or(PixelCorrection::Unevaluable)?;
    let cycles = (triangle_residual(loops.unwrapped, triplet, point) - root) / TAU;
    if !cycles.is_finite() {
        return Err(PixelCorrection::Unevaluable);
    }
    let whole = cycles.round();
    if (cycles - whole).abs() > MAX_CORRECTION_FRACTION_CYCLES {
        return Err(PixelCorrection::NonInteger);
    }
    Ok(whole as i64)
}

/// Whether adding `cycles` closes every loop: `U_ij + U_jk − U_ik = −k`.
fn closes_every_loop(triplets: &[Triplet], cycles: &[i64], misclosure: &[i64]) -> bool {
    triplets
        .iter()
        .zip(misclosure)
        .all(|(t, &k)| cycles[t.early] + cycles[t.late] - cycles[t.span] + k == 0)
}

/// The network's fan of triangles: each pair `(i, k)` with `k > i + 1` is
/// closed by `(i, k−1)`, `(k−1, k)` and their triangle.
struct FanBasis {
    pairs: Vec<(usize, usize)>,
    n_dates: usize,
    /// Pair indices by increasing span, each with its fan `(early, late,
    /// triangle)` indices, `None` for a sequential pair.
    order: Vec<(usize, Option<[usize; 3]>)>,
}

impl FanBasis {
    fn new(pairs: &[(usize, usize)], triplets: &[Triplet]) -> Result<Self, LoopCorrectionError> {
        let index: HashMap<(usize, usize), usize> =
            pairs.iter().enumerate().map(|(k, &p)| (p, k)).collect();
        let triangle: HashMap<(usize, usize, usize), usize> = triplets
            .iter()
            .enumerate()
            .map(|(t, tri)| ((tri.early, tri.late, tri.span), t))
            .collect();
        let n_dates = pairs.iter().map(|&(_, j)| j + 1).max().unwrap_or(0);
        if let Some(date) =
            (0..n_dates.saturating_sub(1)).find(|&d| !index.contains_key(&(d, d + 1)))
        {
            return Err(LoopCorrectionError::MissingSequentialPair { date });
        }
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by_key(|&k| pairs[k].1 - pairs[k].0);
        let order = order
            .into_iter()
            .map(|span| {
                let (i, k) = pairs[span];
                if k == i + 1 {
                    return Ok((span, None));
                }
                let early = *index
                    .get(&(i, k - 1))
                    .ok_or(LoopCorrectionError::MissingFanTriangle { pair: (i, k) })?;
                let late = index[&(k - 1, k)];
                Ok((span, Some([early, late, triangle[&(early, late, span)]])))
            })
            .collect::<Result<_, LoopCorrectionError>>()?;
        Ok(Self {
            pairs: pairs.to_vec(),
            n_dates,
            order,
        })
    }

    /// One integer correction satisfying every fan triangle: zero on the
    /// sequential pairs, `U_ik = U_i,k−1 + U_k−1,k + k_t` up the fan.
    fn particular(&self, misclosure: &[i64]) -> Vec<i64> {
        let mut cycles = vec![0; self.pairs.len()];
        for &(span, fan) in &self.order {
            if let Some([early, late, triangle]) = fan {
                cycles[span] = cycles[early] + cycles[late] + misclosure[triangle];
            }
        }
        cycles
    }

    /// The unique minimum-L1 correction `a + B m`, or `None` on a tie.
    fn minimal_correction(&self, particular: &[i64]) -> Option<Vec<i64>> {
        let potentials = unique_optimal_potentials(self.n_dates, &self.pairs, particular)?;
        Some(
            self.pairs
                .iter()
                .zip(particular)
                .map(|(&(i, k), &a)| a + potentials[k] - potentials[i])
                .collect(),
        )
    }
}

/// A residual arc of the unit-capacity circulation dual to
/// `min Σ |a_e + m_k − m_i|`: flow on pair `e` lies in `[−1, 1]`.
#[derive(Clone, Copy)]
struct Arc {
    from: usize,
    to: usize,
    cost: i64,
    pair: usize,
    forward: bool,
}

fn residual_arcs(pairs: &[(usize, usize)], particular: &[i64], flow: &[i64]) -> Vec<Arc> {
    pairs
        .iter()
        .zip(particular)
        .zip(flow)
        .enumerate()
        .flat_map(|(pair, ((&(i, k), &a), &f))| {
            let forward = (f < 1).then_some(Arc {
                from: i,
                to: k,
                cost: -a,
                pair,
                forward: true,
            });
            let backward = (f > -1).then_some(Arc {
                from: k,
                to: i,
                cost: a,
                pair,
                forward: false,
            });
            forward.into_iter().chain(backward)
        })
        .collect()
}

/// The optimal integer potentials `m` (with `m_0 = 0`) of
/// `min Σ_e |a_e + m_k − m_i|`, or `None` when the optimum is not unique.
fn unique_optimal_potentials(
    n_dates: usize,
    pairs: &[(usize, usize)],
    particular: &[i64],
) -> Option<Vec<i64>> {
    let mut flow = vec![0_i64; pairs.len()];
    loop {
        let arcs = residual_arcs(pairs, particular, &flow);
        let Some(cycle) = negative_cycle(n_dates, &arcs) else {
            break;
        };
        for arc in cycle.into_iter().map(|index| arcs[index]) {
            flow[arc.pair] += if arc.forward { 1 } else { -1 };
        }
    }
    let arcs = residual_arcs(pairs, particular, &flow);
    let upper = shortest_distances(n_dates, &arcs, false)?;
    let lower = shortest_distances(n_dates, &arcs, true)?;
    upper
        .iter()
        .zip(&lower)
        .all(|(up, down)| up + down == 0)
        .then_some(upper)
}

/// A negative-cost cycle of `arcs` as arc indices, by Bellman–Ford from a
/// virtual source joined to every node.
fn negative_cycle(n_nodes: usize, arcs: &[Arc]) -> Option<Vec<usize>> {
    let mut distance = vec![0_i64; n_nodes];
    let mut predecessor: Vec<Option<usize>> = vec![None; n_nodes];
    let mut relaxed = None;
    for _ in 0..n_nodes {
        relaxed = None;
        for (index, arc) in arcs.iter().enumerate() {
            if distance[arc.from] + arc.cost < distance[arc.to] {
                distance[arc.to] = distance[arc.from] + arc.cost;
                predecessor[arc.to] = Some(index);
                relaxed = Some(arc.to);
            }
        }
        relaxed?;
    }
    // Still relaxing after n passes: walk back n steps to land on the cycle.
    let mut node = relaxed?;
    for _ in 0..n_nodes {
        node = arcs[predecessor[node]?].from;
    }
    let start = node;
    let mut cycle = Vec::new();
    loop {
        let index = predecessor[node]?;
        cycle.push(index);
        node = arcs[index].from;
        if node == start {
            return Some(cycle);
        }
    }
}

/// Shortest distances from node 0 over `arcs` (`reversed`: to node 0), which
/// hold no negative cycle; `None` if some node is unreachable.
fn shortest_distances(n_nodes: usize, arcs: &[Arc], reversed: bool) -> Option<Vec<i64>> {
    let mut distance: Vec<Option<i64>> = vec![None; n_nodes];
    distance[0] = Some(0);
    for _ in 1..n_nodes {
        let mut changed = false;
        for arc in arcs {
            let (from, to) = if reversed {
                (arc.to, arc.from)
            } else {
                (arc.from, arc.to)
            };
            let Some(start) = distance[from] else {
                continue;
            };
            if distance[to].is_none_or(|current| start + arc.cost < current) {
                distance[to] = Some(start + arc.cost);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    distance.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_closure::{loop_closure_qc, DEFAULT_CLOSURE_TOLERANCE_CYCLES};
    use crate::network::{build_network, NetworkConfig};

    /// 7×7 = 49 pixels: one component clears `MIN_ROOT_PIXELS`.
    const SIDE: usize = 7;

    /// A bandwidth-`bandwidth` network on `n_dates` 12-day dates and the true
    /// unwrapped phase of a per-date ramp, so every loop closes exactly.
    fn clean_network(n_dates: usize, bandwidth: usize) -> (Vec<(usize, usize)>, Array3<f64>) {
        let days: Vec<f64> = (0..n_dates).map(|d| 12.0 * d as f64).collect();
        let pairs = build_network(
            n_dates,
            &days,
            &NetworkConfig {
                max_bandwidth: Some(bandwidth),
                ..Default::default()
            },
        );
        let phase = |date: usize, row: usize, col: usize| {
            0.7 * date as f64 + 0.05 * (row * SIDE + col) as f64
        };
        let unwrapped = Array3::from_shape_fn((pairs.len(), SIDE, SIDE), |(k, row, col)| {
            let (i, j) = pairs[k];
            phase(j, row, col) - phase(i, row, col)
        });
        (pairs, unwrapped)
    }

    fn labels(unwrapped: &Array3<f64>) -> Array3<u32> {
        Array3::from_elem(unwrapped.dim(), 1)
    }

    fn band(pairs: &[(usize, usize)], pair: (usize, usize)) -> usize {
        pairs.iter().position(|&p| p == pair).unwrap()
    }

    fn correct(pairs: &[(usize, usize)], unwrapped: &mut Array3<f64>) -> LoopClosureCorrection {
        let components = labels(unwrapped);
        correct_loop_closure_errors(unwrapped, components.view(), pairs).unwrap()
    }

    fn status(correction: &LoopClosureCorrection, point: (usize, usize)) -> f64 {
        correction.status[point]
    }

    /// The contract: ±1 and ±2 cycle errors planted in a minority of
    /// interferograms on a redundant network are recovered exactly — the
    /// corrected stack equals the clean one to rounding, and the gate passes.
    #[test]
    fn planted_cycle_errors_are_recovered_exactly() {
        for bandwidth in [2, 3] {
            let (pairs, clean) = clean_network(12, bandwidth);
            let mut unwrapped = clean.clone();
            let planted = [
                ((1, 1), (3, 4), 1.0),
                ((1, 1), (7, 9), -2.0),
                ((2, 5), (2, 4), 2.0),
                ((4, 3), (5, 6), -1.0),
                ((4, 3), (8, 10), 1.0),
            ];
            for &((row, col), pair, cycles) in &planted {
                unwrapped[(band(&pairs, pair), row, col)] += cycles * TAU;
            }
            let correction = correct(&pairs, &mut unwrapped);

            let worst = (&unwrapped - &clean)
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(worst < 1e-9, "bandwidth {bandwidth}: off by {worst} rad");
            for point in [(1, 1), (2, 5), (4, 3)] {
                assert_eq!(status(&correction, point), 1.0, "bandwidth {bandwidth}");
            }
            assert_eq!(correction.corrected_interferogram_count[(1, 1)], 2.0);
            assert_eq!(correction.corrected_interferogram_count[(2, 5)], 1.0);
            assert_eq!(correction.count(PixelCorrection::Corrected), 3);
            assert_eq!(correction.corrected_per_band[band(&pairs, (7, 9))], 1);
            let qc = loop_closure_qc(
                unwrapped.view(),
                labels(&unwrapped).view(),
                &pairs,
                DEFAULT_CLOSURE_TOLERANCE_CYCLES,
            )
            .unwrap();
            assert!(!qc.failed_mask().iter().any(|&bad| bad));
        }
    }

    /// A clean network is left bit-for-bit unchanged and every pixel reports
    /// `Closed`, including with per-interferogram roots (whole-cycle offsets
    /// common to every pixel), which are not errors.
    #[test]
    fn clean_network_is_unchanged() {
        let (pairs, mut unwrapped) = clean_network(10, 2);
        for k in 0..pairs.len() {
            let root = (k % 3) as f64 - 1.0;
            unwrapped
                .index_axis_mut(ndarray::Axis(0), k)
                .mapv_inplace(|v| v + root * TAU);
        }
        let before = unwrapped.clone();
        let correction = correct(&pairs, &mut unwrapped);
        assert_eq!(unwrapped, before);
        assert_eq!(correction.count(PixelCorrection::Closed), SIDE * SIDE);
        assert!(correction.corrected_per_band.iter().all(|&n| n == 0));
        assert!(correction
            .corrected_interferogram_count
            .iter()
            .all(|&n| n == 0.0));
    }

    /// Under-determined: on the first triangle of a bandwidth-2 chain, an
    /// error in `(0,1)` and one in `(0,2)` close the network with the same
    /// single cycle, and on a lone 3-date triangle all three members tie. The
    /// pixel is left alone and the gate still masks it.
    #[test]
    fn under_determined_pixel_is_left_masked_not_guessed() {
        for (n_dates, pair) in [(8, (0, 1)), (8, (0, 2)), (3, (1, 2))] {
            let (pairs, mut unwrapped) = clean_network(n_dates, 2);
            unwrapped[(band(&pairs, pair), 3, 3)] += TAU;
            let before = unwrapped.clone();
            let correction = correct(&pairs, &mut unwrapped);
            assert_eq!(
                status(&correction, (3, 3)),
                2.0,
                "{n_dates} dates, {pair:?}"
            );
            assert_eq!(unwrapped, before, "nothing is guessed");
            let qc = loop_closure_qc(
                unwrapped.view(),
                labels(&unwrapped).view(),
                &pairs,
                DEFAULT_CLOSURE_TOLERANCE_CYCLES,
            )
            .unwrap();
            assert!(qc.failed_mask()[(3, 3)], "the gate still masks it");
        }
    }

    /// A ministack junction is a per-date offset in the linked phase: common
    /// to every interferogram touching that date, so it cancels in every loop
    /// whatever its size. The correction sees nothing to correct and leaves
    /// it alone; closure cannot see a junction, only unwrap errors.
    #[test]
    fn junction_offset_closes_and_is_not_corrected() {
        let (pairs, mut unwrapped) = clean_network(10, 3);
        let junction = 5;
        for (k, &(i, j)) in pairs.iter().enumerate() {
            if i < junction && j >= junction {
                let mut layer = unwrapped.index_axis_mut(ndarray::Axis(0), k);
                layer.mapv_inplace(|v| v + 0.87 * TAU + TAU);
            }
        }
        let before = unwrapped.clone();
        let correction = correct(&pairs, &mut unwrapped);
        assert_eq!(unwrapped, before);
        assert_eq!(correction.count(PixelCorrection::Closed), SIDE * SIDE);
    }

    /// A misclosure far from a whole cycle is not an unwrap error, and a loop
    /// through an unlabelled member is not evaluable: both left alone.
    #[test]
    fn non_integer_and_unevaluable_are_left_alone() {
        let (pairs, mut unwrapped) = clean_network(6, 2);
        unwrapped[(band(&pairs, (2, 4)), 0, 0)] += 0.6 * TAU;
        unwrapped[(band(&pairs, (2, 4)), 0, 1)] += TAU;
        let mut components = labels(&unwrapped);
        components[(band(&pairs, (1, 2)), 0, 1)] = 0;
        let before = unwrapped.clone();
        let correction =
            correct_loop_closure_errors(&mut unwrapped, components.view(), &pairs).unwrap();
        assert_eq!(status(&correction, (0, 0)), 4.0);
        assert_eq!(status(&correction, (0, 1)), 3.0);
        assert_eq!(unwrapped, before);
    }

    /// Networks without a fan of triangles are refused, not silently skipped.
    #[test]
    fn unsupported_networks_are_errors() {
        assert_eq!(
            check_correction_network(&[(0, 1), (0, 2)]),
            Err(LoopCorrectionError::MissingSequentialPair { date: 1 })
        );
        assert_eq!(
            check_correction_network(&[(0, 1), (1, 2), (2, 3), (0, 3)]),
            Err(LoopCorrectionError::MissingFanTriangle { pair: (0, 3) })
        );
        let (pairs, _) = clean_network(9, 4);
        assert_eq!(check_correction_network(&pairs), Ok(()));
    }
}
