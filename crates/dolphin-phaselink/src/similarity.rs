//! Phase similarity — spatial neighbour-phase agreement per pixel.
//!
//! Reference: dolphin `src/dolphin/similarity.py` (Wang et al. 2022, "Accurate
//! Persistent Scatterer…", eq. 5 median / eq. 6 max). This is a *spatial* QA
//! signal, distinct from the temporal/coherence-based layers in [`crate::quality`]:
//! it asks whether a pixel's phase history agrees with its neighbours', so it
//! falls at discontinuities and on isolated scatterers.
//!
//! The metric between two unit-modulus phase vectors is the mean cosine of their
//! phase difference, `(1/n) Σ Re(x_i · conj(y_i))`; each pixel reports the median
//! (or max) of that quantity over its neighbourhood.

use dolphin_core::Cf64;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};
use rayon::prelude::*;

/// Which neighbourhood summary a pixel reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseSimilaritySummary {
    /// Wang et al. 2022 eq. 5 — robust to a minority of disagreeing neighbours.
    Median,
    /// Wang et al. 2022 eq. 6 — the best-agreeing neighbour.
    Max,
}

/// Relative `(row, col)` offsets of the neighbours compared at each pixel.
///
/// Port of dolphin's `get_circle_idxs`: the union of midpoint-drawn circles of
/// radius `1..max_radius`, which is a filled disk whose exact boundary pixels are
/// decided by that algorithm rather than by a radius test. The origin is excluded.
/// The set is reproduced exactly because it defines *which* pixels are compared —
/// a near-miss here shifts every similarity value.
#[must_use]
pub fn circle_offsets(max_radius: usize) -> Vec<(i32, i32)> {
    if max_radius < 2 {
        return Vec::new();
    }
    let mut visited = vec![false; max_radius * max_radius];
    visited[0] = true;
    let mut offsets = Vec::new();
    for radius in 1..max_radius as i64 {
        walk_circle(radius, max_radius, &mut visited, &mut offsets);
    }
    offsets
}

/// The eight-fold symmetric points of one midpoint-drawn circle of `radius`,
/// appended to `offsets` and recorded in `visited`.
fn walk_circle(radius: i64, side: usize, visited: &mut [bool], offsets: &mut Vec<(i32, i32)>) {
    let at = |x: i64, y: i64| (x as usize) * side + (y as usize);
    for (a, b) in [(radius, 0), (-radius, 0), (0, radius), (0, -radius)] {
        offsets.push((a as i32, b as i32));
    }
    visited[at(radius, 0)] = true;
    visited[at(0, radius)] = true;

    let (mut x, mut y, mut p, mut flag) = (radius, 0_i64, 1 - radius, 0_i64);
    while x > y {
        step(&mut x, &mut y, &mut p, &mut flag);
        if x < y {
            break;
        }
        // Close any gap the previous step opened between concentric circles.
        while !visited[at(x - 1, y)] {
            x -= 1;
            flag += 1;
        }
        visited[at(x, y)] = true;
        visited[at(y, x)] = true;
        push_octants(x, y, offsets);
        x += i64::from(flag > 0);
    }
}

/// One midpoint-circle step: advance `y` (moving `x` inward when the midpoint
/// falls outside the perimeter), or spend one pending hole-fill.
fn step(x: &mut i64, y: &mut i64, p: &mut i64, flag: &mut i64) {
    if *flag > 0 {
        *flag -= 1;
        return;
    }
    *y += 1;
    if *p <= 0 {
        *p += 2 * *y + 1;
        return;
    }
    *x -= 1;
    *p += 2 * *y - 2 * *x + 1;
}

/// The four (or eight, off the diagonal) symmetric images of `(x, y)`.
fn push_octants(x: i64, y: i64, offsets: &mut Vec<(i32, i32)>) {
    for (a, b) in [(x, y), (-x, -y), (x, -y), (-x, y)] {
        offsets.push((a as i32, b as i32));
    }
    if x == y {
        return;
    }
    for (a, b) in [(y, x), (-y, -x), (y, -x), (-y, x)] {
        offsets.push((a as i32, b as i32));
    }
}

/// Mean cosine of the phase difference between two unit-modulus phase vectors.
fn phase_similarity(a: ArrayView1<Cf64>, b: ArrayView1<Cf64>) -> f64 {
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x * y.conj()).re).sum();
    sum / a.len() as f64
}

/// Median of already-finite values, averaging the middle pair on an even count
/// (numpy's `nanmedian` convention, which the oracle fixtures encode).
fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

/// Per-pixel phase similarity over a `(n_ifg, rows, cols)` stack.
///
/// `mask` marks pixels to include (`true`); masked-out pixels are neither scored
/// nor used as neighbours. Pixels whose stack sums to exactly zero carry no phase
/// and are treated as invalid the same way. A pixel with no usable neighbour, and
/// every excluded pixel, reports `NaN`.
///
/// Neighbour coordinates are **clamped** to the raster edge rather than dropped,
/// matching dolphin, so border pixels compare against the edge repeatedly instead
/// of losing their neighbourhood.
#[must_use]
pub fn estimate_phase_similarity(
    ifg_stack: ArrayView3<Cf64>,
    search_radius: usize,
    summary: PhaseSimilaritySummary,
    mask: Option<ArrayView2<bool>>,
) -> Array2<f64> {
    let (n_ifg, rows, cols) = ifg_stack.dim();
    let offsets = circle_offsets(search_radius);

    // Unit phasors, so the comparison depends on phase alone. `arg` of a zero or
    // non-finite sample is not meaningful; those pixels are excluded below.
    let unit = ifg_stack.mapv(|z| {
        let angle = z.arg();
        if angle.is_finite() {
            Cf64::from_polar(1.0, angle)
        } else {
            Cf64::new(1.0, 0.0)
        }
    });

    let included = Array2::from_shape_fn((rows, cols), |(r, c)| {
        let carries_phase = finite_sum(ifg_stack, n_ifg, r, c) != Cf64::new(0.0, 0.0);
        carries_phase && mask.is_none_or(|m| m[(r, c)])
    });

    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            pixel_summary(
                unit.view(),
                &included,
                &offsets,
                summary,
                idx / cols,
                idx % cols,
            )
        })
        .collect();

    Array2::from_shape_vec((rows, cols), values).expect("phase similarity shape")
}

/// Complex sum down the stack at one pixel, non-finite samples read as zero —
/// dolphin's `np.nan_to_num(...).sum(axis=0)` invalidity test. Non-finite samples
/// are zeroed whole, the convention already used across this crate, rather than
/// per component; a sample with one finite part does not arise in CSLC data.
fn finite_sum(stack: ArrayView3<Cf64>, n_ifg: usize, r: usize, c: usize) -> Cf64 {
    (0..n_ifg)
        .map(|k| stack[[k, r, c]])
        .map(|z| match z.is_finite() {
            true => z,
            false => Cf64::new(0.0, 0.0),
        })
        .sum()
}

/// The neighbourhood summary at one pixel, or `NaN` where it is excluded or has
/// no usable neighbour.
fn pixel_summary(
    unit: ArrayView3<Cf64>,
    included: &Array2<bool>,
    offsets: &[(i32, i32)],
    summary: PhaseSimilaritySummary,
    r0: usize,
    c0: usize,
) -> f64 {
    if !included[(r0, c0)] {
        return f64::NAN;
    }
    let (_, rows, cols) = unit.dim();
    let x0 = unit.slice(ndarray::s![.., r0, c0]);
    let mut sims: Vec<f64> = offsets
        .iter()
        .filter_map(|&(dr, dc)| {
            let r = (r0 as i64 + i64::from(dr)).clamp(0, rows as i64 - 1) as usize;
            let c = (c0 as i64 + i64::from(dc)).clamp(0, cols as i64 - 1) as usize;
            let usable = (r, c) != (r0, c0) && included[(r, c)];
            usable.then(|| phase_similarity(x0, unit.slice(ndarray::s![.., r, c])))
        })
        .collect();
    match (sims.is_empty(), summary) {
        (true, _) => f64::NAN,
        (false, PhaseSimilaritySummary::Median) => median(&mut sims),
        (false, PhaseSimilaritySummary::Max) => {
            sims.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }
    }
}
