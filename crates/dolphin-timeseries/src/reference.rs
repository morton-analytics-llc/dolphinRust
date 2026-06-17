//! Automatic spatial reference-point selection (dolphin v0.36 center-of-mass).
//!
//! dolphin references the displacement time series to a single high-quality pixel
//! so every date is relative to stable ground. v0.35.0 picks the single
//! best-condition pixel (`argmin`); v0.36 added a **center-of-mass** method that
//! is spatially robust: take the largest connected region of high-quality pixels
//! and pick the one nearest its quality-weighted centroid — avoiding an isolated
//! outlier pixel as the reference. The pinned v0.35.0 oracle has no center-of-mass
//! method, so this is contract-tested against analytic fixtures.

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
    let component = largest_component(quality, threshold)?;
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

/// In-bounds 4-connected neighbours of `(r, c)`.
fn neighbors4(
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
) -> impl Iterator<Item = (usize, usize)> {
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
    v.into_iter()
}

/// Flood-fill the candidate component containing `start`, marking `seen`.
fn flood(
    start: (usize, usize),
    quality: ArrayView2<f64>,
    threshold: f64,
    seen: &mut Array2<bool>,
) -> Vec<(usize, usize)> {
    let (rows, cols) = quality.dim();
    let ok = |r: usize, c: usize| quality[(r, c)].is_finite() && quality[(r, c)] > threshold;
    let mut stack = vec![start];
    seen[start] = true;
    let mut component = Vec::new();
    while let Some((r, c)) = stack.pop() {
        component.push((r, c));
        let next: Vec<(usize, usize)> = neighbors4(r, c, rows, cols)
            .filter(|&(nr, nc)| !seen[(nr, nc)] && ok(nr, nc))
            .collect();
        for p in next {
            seen[p] = true;
            stack.push(p);
        }
    }
    component
}

/// Largest 4-connected component of pixels with `quality > threshold`.
fn largest_component(quality: ArrayView2<f64>, threshold: f64) -> Option<Vec<(usize, usize)>> {
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
        let component = flood((r, c), quality, threshold, &mut seen);
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
