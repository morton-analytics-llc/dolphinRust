//! Automatic spatial reference-point selection.
//!
//! dolphin references the displacement time series to a single high-quality pixel
//! so every date is relative to stable ground. Two rules live here:
//!
//! - [`select_reference_point`] — the original dolphinRust rule: the largest
//!   4-connected region of pixels above a (low) threshold and the pixel nearest its
//!   quality-weighted centroid. Kept bit-for-bit as the
//!   `largest_component_centroid` method and as the fallback of the next rule.
//! - [`select_coherent_reference_point`] — the `coherent_support` method. Follows
//!   dolphin v0.42 `timeseries.select_reference_point`: candidates are pixels whose
//!   quality (temporal coherence) exceeds a *high* floor, grouped into 8-connected
//!   regions, and the largest region wins. Deliberate difference: dolphin then takes
//!   the pixel nearest the region's centre of mass; here the region's pixels are
//!   ranked by a stability score (the summed coherence of the pixel's 3×3
//!   neighbourhood inside the region, so a coherent interior pixel beats a coherent
//!   edge pixel), with dolphin's centre-of-mass distance and then `(row, col)` as
//!   deterministic tie-breaks.

use ndarray::{Array2, Array3, ArrayView2, Axis};

/// Select a spatial reference pixel `(row, col)` by the center of mass of the
/// highest-quality region of `quality` (e.g. temporal coherence).
///
/// Candidates are pixels with `quality > threshold`; the search is restricted to
/// the largest 4-connected component of those candidates, and the reference is
/// the candidate nearest that component's quality-weighted centroid. Returns
/// `None` if no pixel exceeds `threshold`.
#[must_use]
pub fn select_reference_point(quality: ArrayView2<f64>, threshold: f64) -> Option<(usize, usize)> {
    let component = largest_component(quality, threshold, neighbors4)?;
    let (mut sum_w, mut sum_r, mut sum_c) = (0.0, 0.0, 0.0);
    for &(r, c) in &component {
        let w = quality[(r, c)];
        sum_w += w;
        sum_r += w * r as f64;
        sum_c += w * c as f64;
    }
    let (centroid_r, centroid_c) = (sum_r / sum_w, sum_c / sum_w);
    component.into_iter().min_by(|&a, &b| {
        let da = dist2(a, centroid_r, centroid_c);
        let db = dist2(b, centroid_r, centroid_c);
        da.partial_cmp(&db)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    })
}

/// Provenance tag of [`select_coherent_reference_point`]; bump it whenever the
/// candidate pool, region rule, ranking, or tie-breaks change.
pub const COHERENT_SUPPORT_METHOD_VERSION: &str = "coherent-support/2";

/// A reference chosen by [`select_coherent_reference_point`], with the counts
/// behind the choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherentReference {
    /// The reference pixel `(row, col)`.
    pub point: (usize, usize),
    /// Candidate pool: finite pixels with quality above the floor.
    pub pool_pixels: usize,
    /// Pixels in the largest 8-connected region of the pool, which holds `point`.
    pub region_pixels: usize,
}

/// Select a spatial reference on coherent support.
///
/// Candidates are finite pixels with `quality > floor` (dolphin's strict
/// `> candidate_threshold`); callers mark ineligible pixels non-finite. The
/// largest 8-connected region of candidates is kept (equal sizes: the region
/// reached first in row-major order, as `scipy.ndimage.label` numbers them).
/// Within it the pixel with the highest summed quality over its 3×3
/// neighbourhood, counting only pixels of that region, wins; ties go to the
/// pixel nearest the region's unweighted centre of mass, then to the smallest
/// `(row, col)`. Returns `None` when no pixel exceeds `floor`.
#[must_use]
pub fn select_coherent_reference_point(
    quality: ArrayView2<f64>,
    floor: f64,
) -> Option<CoherentReference> {
    let region = largest_component(quality, floor, neighbors8)?;
    let pool_pixels = quality
        .iter()
        .filter(|value| value.is_finite() && **value > floor)
        .count();
    let (rows, cols) = quality.dim();
    let mut in_region = Array2::from_elem((rows, cols), false);
    for &point in &region {
        in_region[point] = true;
    }
    let neighbourhood_quality = |(r, c): (usize, usize)| -> f64 {
        let window_rows = r.saturating_sub(1)..(r + 2).min(rows);
        window_rows
            .flat_map(|nr| (c.saturating_sub(1)..(c + 2).min(cols)).map(move |nc| (nr, nc)))
            .filter(|&neighbour| in_region[neighbour])
            .map(|neighbour| quality[neighbour])
            .sum()
    };
    let n = region.len() as f64;
    let centroid_r = region.iter().map(|&(r, _)| r as f64).sum::<f64>() / n;
    let centroid_c = region.iter().map(|&(_, c)| c as f64).sum::<f64>() / n;
    let point = region.iter().copied().min_by(|&a, &b| {
        neighbourhood_quality(b)
            .total_cmp(&neighbourhood_quality(a))
            .then(dist2(a, centroid_r, centroid_c).total_cmp(&dist2(b, centroid_r, centroid_c)))
            .then(a.cmp(&b))
    })?;
    Some(CoherentReference {
        point,
        pool_pixels,
        region_pixels: region.len(),
    })
}

/// Reference every date band of `series` `(n_dates, rows, cols)` to `point` by
/// subtracting that pixel's value from the whole band — so the reference pixel
/// reads zero at every date and all displacements are relative to it.
pub fn reference_to_point(series: &mut Array3<f64>, point: (usize, usize)) {
    let (n_dates, _, _) = series.dim();
    for t in 0..n_dates {
        let v = series[(t, point.0, point.1)];
        if v.is_finite() {
            series.index_axis_mut(Axis(0), t).mapv_inplace(|x| x - v);
        }
    }
}

/// Squared Euclidean distance from pixel `(r, c)` to the centroid.
fn dist2((r, c): (usize, usize), centroid_r: f64, centroid_c: f64) -> f64 {
    (r as f64 - centroid_r).powi(2) + (c as f64 - centroid_c).powi(2)
}

/// In-bounds neighbours of `(r, c)` on a `rows × cols` grid.
type Neighbours = fn(usize, usize, usize, usize) -> Vec<(usize, usize)>;

/// In-bounds 4-connected neighbours of `(r, c)`.
fn neighbors4(r: usize, c: usize, rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(4);
    if r > 0 {
        v.push((r - 1, c));
    }
    if r + 1 < rows {
        v.push((r + 1, c));
    }
    if c > 0 {
        v.push((r, c - 1));
    }
    if c + 1 < cols {
        v.push((r, c + 1));
    }
    v
}

/// In-bounds 8-connected neighbours of `(r, c)` (dolphin labels candidate
/// regions with `structure=np.ones((3, 3))`).
fn neighbors8(r: usize, c: usize, rows: usize, cols: usize) -> Vec<(usize, usize)> {
    (r.saturating_sub(1)..(r + 2).min(rows))
        .flat_map(|nr| (c.saturating_sub(1)..(c + 2).min(cols)).map(move |nc| (nr, nc)))
        .filter(|&neighbour| neighbour != (r, c))
        .collect()
}

/// Flood-fill the candidate component containing `start`, marking `seen`.
fn flood(
    start: (usize, usize),
    quality: ArrayView2<f64>,
    threshold: f64,
    neighbours: Neighbours,
    seen: &mut Array2<bool>,
) -> Vec<(usize, usize)> {
    let (rows, cols) = quality.dim();
    let ok = |r: usize, c: usize| quality[(r, c)].is_finite() && quality[(r, c)] > threshold;
    let mut stack = vec![start];
    seen[start] = true;
    let mut component = Vec::new();
    while let Some((r, c)) = stack.pop() {
        component.push((r, c));
        let next: Vec<(usize, usize)> = neighbours(r, c, rows, cols)
            .into_iter()
            .filter(|&(nr, nc)| !seen[(nr, nc)] && ok(nr, nc))
            .collect();
        for p in next {
            seen[p] = true;
            stack.push(p);
        }
    }
    component
}

/// Largest component (under `neighbours`) of pixels with `quality > threshold`;
/// on equal sizes the component reached first in row-major order.
fn largest_component(
    quality: ArrayView2<f64>,
    threshold: f64,
    neighbours: Neighbours,
) -> Option<Vec<(usize, usize)>> {
    let (rows, cols) = quality.dim();
    let ok = |r: usize, c: usize| quality[(r, c)].is_finite() && quality[(r, c)] > threshold;
    let mut seen = Array2::from_elem((rows, cols), false);
    let mut best: Vec<(usize, usize)> = Vec::new();
    let starts: Vec<(usize, usize)> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .collect();
    for (r, c) in starts {
        if seen[(r, c)] || !ok(r, c) {
            continue;
        }
        let component = flood((r, c), quality, threshold, neighbours, &mut seen);
        if component.len() > best.len() {
            best = component;
        }
    }
    (!best.is_empty()).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A single uniform high-quality blob: the reference is the pixel nearest the
    /// geometric centroid of the blob.
    #[test]
    fn picks_centroid_of_uniform_blob() {
        let mut q = Array2::<f64>::zeros((5, 5));
        // 3x3 blob rows 1..=3, cols 1..=3 -> centroid (2, 2).
        for r in 1..=3 {
            for c in 1..=3 {
                q[(r, c)] = 0.9;
            }
        }
        assert_eq!(select_reference_point(q.view(), 0.5), Some((2, 2)));
    }

    /// Two disjoint blobs: the larger component wins, and the centroid lands in it
    /// (not in the empty gap between them).
    #[test]
    fn larger_component_wins() {
        let mut q = Array2::<f64>::zeros((3, 9));
        q[(1, 0)] = 0.8; // tiny component (1 px) on the left
        for c in 5..=8 {
            q[(1, c)] = 0.8; // larger component (4 px) on the right
        }
        let p = select_reference_point(q.view(), 0.5).unwrap();
        assert_eq!(p.0, 1);
        assert!(
            (5..=8).contains(&p.1),
            "ref {p:?} should be in the larger right blob"
        );
    }

    /// Quality weighting pulls the centroid toward the higher-coherence side.
    #[test]
    fn weighting_shifts_toward_high_quality() {
        // One row, 5 candidates; mass concentrated on the right. Unweighted the
        // centroid is col 2; weighting moves the discrete pick strictly past it.
        let q = array![[0.3, 0.3, 1.0, 1.0, 1.0]];
        let p = select_reference_point(q.view(), 0.2).unwrap();
        assert!(
            p.1 >= 3,
            "weighted ref col {} should exceed the unweighted center 2",
            p.1
        );
    }

    #[test]
    fn none_when_all_below_threshold() {
        let q = Array2::<f64>::from_elem((4, 4), 0.1);
        assert_eq!(select_reference_point(q.view(), 0.5), None);
    }

    /// Regression pin: the legacy rule's choices on a fragmented fixture, recorded
    /// from the code before `coherent_support` existed. The connectivity refactor
    /// must not move them.
    #[test]
    fn legacy_choice_is_unchanged() {
        let mut q = Array2::from_shape_fn((12, 15), |(r, c)| ((r * 7 + c * 13) % 17) as f64 / 17.0);
        q[(5, 6)] = f64::NAN;
        q[(0, 0)] = f64::NAN;
        assert_eq!(select_reference_point(q.view(), 0.2), Some((6, 7)));
        assert_eq!(select_reference_point(q.view(), 0.5), Some((1, 2)));
        assert_eq!(select_reference_point(q.view(), 0.8), Some((0, 5)));
    }

    /// Fresno's failure mode: a wide field of middling coherence holds a small
    /// coherent patch off-centre. The legacy rule lands mid-field; coherent
    /// support lands in the patch.
    #[test]
    fn coherent_support_leaves_the_middling_field() {
        let mut q = Array2::<f64>::from_elem((9, 9), 0.6);
        for r in 0..3 {
            for c in 6..9 {
                q[(r, c)] = 0.95;
            }
        }
        assert_eq!(select_reference_point(q.view(), 0.2), Some((4, 4)));
        let chosen = select_coherent_reference_point(q.view(), 0.85).unwrap();
        assert_eq!(chosen.point, (1, 7));
        assert_eq!((chosen.pool_pixels, chosen.region_pixels), (9, 9));
    }

    /// Regions are 8-connected (dolphin's `np.ones((3, 3))` structure): a
    /// diagonal chain is one region and outweighs a larger-valued 2-pixel blob.
    #[test]
    fn coherent_regions_are_eight_connected_and_largest_wins() {
        let mut q = Array2::<f64>::zeros((6, 6));
        for i in 0..4 {
            q[(i, i)] = 0.9;
        }
        q[(5, 4)] = 0.99;
        q[(5, 5)] = 0.99;
        let chosen = select_coherent_reference_point(q.view(), 0.85).unwrap();
        assert_eq!((chosen.pool_pixels, chosen.region_pixels), (6, 4));
        assert!(chosen.point.0 == chosen.point.1 && chosen.point.0 < 4);
    }

    /// A coherent interior pixel beats a slightly more coherent pixel on the
    /// region's edge, because the score sums the neighbourhood inside the region.
    #[test]
    fn coherent_support_prefers_interior_over_a_lone_edge_peak() {
        let mut q = Array2::<f64>::zeros((5, 5));
        for r in 1..=3 {
            for c in 1..=3 {
                q[(r, c)] = 0.9;
            }
        }
        q[(1, 3)] = 0.99;
        let chosen = select_coherent_reference_point(q.view(), 0.85).unwrap();
        assert_eq!(chosen.point, (2, 2));
    }

    /// Score ties go to the pixel nearest the unweighted centre of mass, and a
    /// remaining tie to the smaller `(row, col)`; equal-size regions resolve to
    /// the one reached first in row-major order.
    #[test]
    fn coherent_support_tie_breaks_are_deterministic() {
        // One row of four equal pixels (cols 1..=4): cols 2 and 3 tie on score
        // (three region pixels in the window) and on distance to the centre
        // (col 2.5), so the smaller (row, col) wins.
        let q = array![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.9, 0.9, 0.9, 0.0]
        ];
        let chosen = select_coherent_reference_point(q.view(), 0.85).unwrap();
        assert_eq!(chosen.point, (1, 2));
        // Two equal single-pixel regions: the first in row-major order wins.
        let mut q = Array2::<f64>::zeros((4, 4));
        q[(3, 0)] = 0.9;
        q[(0, 3)] = 0.9;
        let chosen = select_coherent_reference_point(q.view(), 0.85).unwrap();
        assert_eq!(chosen.point, (0, 3));
    }

    /// The floor is strict, as dolphin's `quality > candidate_threshold`, and
    /// non-finite pixels are never candidates.
    #[test]
    fn coherent_support_floor_is_strict_and_skips_nonfinite() {
        let q = array![[0.85, f64::NAN], [0.5, 0.85]];
        assert_eq!(select_coherent_reference_point(q.view(), 0.85), None);
        let q = array![[0.85, f64::NAN], [0.5, 0.86]];
        let chosen = select_coherent_reference_point(q.view(), 0.85).unwrap();
        assert_eq!(chosen.point, (1, 1));
    }

    /// Referencing zeroes the reference pixel at every date and shifts each band
    /// by that pixel's value.
    #[test]
    fn reference_to_point_zeroes_the_pixel() {
        let mut s = Array3::<f64>::from_shape_fn((3, 2, 2), |(t, r, c)| (t + r + c) as f64);
        reference_to_point(&mut s, (1, 1));
        for t in 0..3 {
            assert_eq!(s[(t, 1, 1)], 0.0, "ref pixel must be 0 at date {t}");
        }
        // Band shifted by a constant: relative structure preserved.
        assert_eq!(s[(0, 0, 0)], 0.0 - 2.0);
    }
}
