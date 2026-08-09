//! Per-pixel line-of-sight geometry resolved onto the frame grid.
//!
//! Ingests the OPERA CSLC-S1-STATIC `los_east`/`los_north` unit-vector components
//! ([`dolphin_io::read_los_layers`]), reprojects each per-burst granule onto the
//! displacement frame grid, mosaics them (first covered burst wins), and derives
//! the up component. Where two granules' footprints overlap, first-covered-wins is
//! only defensible if they agree there, so the overlap is checked rather than
//! trusted (#39). The result drives the atmospheric zenith→slant projection
//! (`slant = 1/up = 1/cos(incidence)`) and the GPS-harness ENU→LOS projection
//! (`d_los = d_e·east + d_n·north + d_u·up`; ground→sensor, positive = toward sat).
//!
//! Nodata handling matches dolphin (`nodata = 0` for los_east/north): GDAL's warp
//! fills out-of-coverage frame pixels with exactly `0`, so a frame pixel is *valid*
//! iff `east != 0 || north != 0`. For Sentinel-1 (ellipsoidal incidence ≈ 30–46°) a
//! valid pixel always has a substantial `e` or `n`, so `(0, 0)` uniquely marks fill.
//! Partial coverage is a hard error — never a silent 0°/nadir pixel.

use dolphin_io::{GeoInfo, LosLayers};
use ndarray::{Array2, Zip};

use crate::error::{CorrectionError, Result};
use crate::troposphere::{warp_to_frame, DelayGrid};

/// Per-pixel LOS unit-vector components on the frame grid. `up` is derived as
/// `+sqrt(max(0, 1 - east² - north²))`; the incidence angle is [`Self::incidence_deg`].
#[derive(Debug, Clone)]
pub struct LosGeometry {
    /// East component of the ground→sensor LOS unit vector, `(rows, cols)`.
    pub east: Array2<f64>,
    /// North component of the ground→sensor LOS unit vector, `(rows, cols)`.
    pub north: Array2<f64>,
    /// Up component, `+sqrt(max(0, 1 - east² - north²))`, `(rows, cols)`.
    pub up: Array2<f64>,
}

impl LosGeometry {
    /// Per-pixel ellipsoidal incidence angle in degrees, `acos(up)·180/π` —
    /// character-identical to dolphin `atmosphere/ionosphere.py`. This is the angle
    /// the atmospheric zenith→slant `1/cos` mapping uses.
    #[must_use]
    pub fn incidence_deg(&self) -> Array2<f64> {
        self.up.mapv(|u| u.acos().to_degrees())
    }

    /// Spatial statistics of the ellipsoidal incidence angle over finite pixels
    /// (full frame coverage is already enforced by [`resolve_los_geometry`], so the
    /// finite filter is defensive, not load-bearing). `None` when no pixel is
    /// finite. Std is the population std (numpy `ddof=0`).
    #[must_use]
    pub fn incidence_stats(&self) -> Option<IncidenceStats> {
        let inc = self.incidence_deg();
        let finite: Vec<f64> = inc.iter().copied().filter(|d| d.is_finite()).collect();
        if finite.is_empty() {
            return None;
        }
        let count = finite.len() as f64;
        let mean_deg = finite.iter().sum::<f64>() / count;
        let sum_sq = finite.iter().map(|d| d * d).sum::<f64>();
        let variance = (sum_sq / count - mean_deg * mean_deg).max(0.0);
        Some(IncidenceStats {
            mean_deg,
            std_deg: variance.sqrt(),
            min_deg: finite.iter().copied().fold(f64::INFINITY, f64::min),
            max_deg: finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        })
    }
}

/// Spatial statistics of the per-pixel ellipsoidal incidence angle, degrees.
#[derive(Debug, Clone, Copy)]
pub struct IncidenceStats {
    /// Mean over finite pixels.
    pub mean_deg: f64,
    /// Population standard deviation (numpy `ddof=0`).
    pub std_deg: f64,
    /// Minimum.
    pub min_deg: f64,
    /// Maximum.
    pub max_deg: f64,
}

/// Maximum tolerated median LOS disagreement, in degrees, between granules where
/// their footprints overlap. Bursts on the same track image a shared ground point
/// at very nearly the same look geometry, so real agreement is far tighter; the
/// gate is loose enough to absorb warp resampling and tight enough to catch a
/// granule whose geometry does not belong to this frame.
const OVERLAP_AGREEMENT_GATE_DEG: f64 = 1.0;

/// Overlaps smaller than this are not gated — a sliver is dominated by the
/// bilinear seam ring and carries no usable statistic.
const MIN_OVERLAP_PIXELS: usize = 32;

/// Resolve per-pixel LOS geometry onto the frame grid from one-or-more per-burst
/// CSLC-S1-STATIC granules. Each granule is reprojected onto `(dst_gt, dst_epsg,
/// shape)` and mosaicked (first covered burst wins); coverage over the whole frame
/// is required, and granules that overlap must agree there.
///
/// # Errors
/// [`CorrectionError::GeometryCoverage`] if `layers` is empty or any frame pixel is
/// left uncovered; [`CorrectionError::GeometryOverlapMismatch`] if overlapping
/// granules carry materially different LOS;
/// [`CorrectionError::Gdal`]/[`CorrectionError::Shape`] on warp failure.
pub fn resolve_los_geometry(
    layers: &[LosLayers],
    dst_gt: [f64; 6],
    dst_epsg: u32,
    shape: (usize, usize),
) -> Result<LosGeometry> {
    if layers.is_empty() {
        return Err(CorrectionError::GeometryCoverage(
            "no geometry (CSLC-S1-STATIC) granules supplied".into(),
        ));
    }
    let mut east = Array2::<f64>::zeros(shape);
    let mut north = Array2::<f64>::zeros(shape);
    let mut covered = Array2::from_elem(shape, false);
    let mut disagreement_deg = Vec::new();
    for layer in layers {
        let e = warp_component(&layer.east, layer.geo, dst_gt, dst_epsg, shape)?;
        let n = warp_component(&layer.north, layer.geo, dst_gt, dst_epsg, shape)?;
        disagreement_deg.extend(overlap_disagreement_deg(&east, &north, &covered, &e, &n));
        fill_uncovered(&mut east, &mut north, &mut covered, &e, &n);
    }
    ensure_full_coverage(&covered)?;
    ensure_overlap_agreement(&mut disagreement_deg)?;
    let up = derive_up(&east, &north);
    Ok(LosGeometry { east, north, up })
}

/// Reproject one component grid onto the frame; GDAL fills out-of-coverage with 0.
fn warp_component(
    data: &Array2<f64>,
    geo: GeoInfo,
    dst_gt: [f64; 6],
    dst_epsg: u32,
    shape: (usize, usize),
) -> Result<Array2<f64>> {
    let src = DelayGrid {
        data: data.clone(),
        geotransform: geo.geotransform,
        epsg: Some(geo.epsg),
        srs_wkt: None,
    };
    warp_to_frame(&src, dst_gt, dst_epsg, shape)
}

/// Fill still-uncovered frame pixels from this burst where it carries valid LOS
/// (nodata is exactly `(0, 0)`); already-covered pixels keep the first burst's value.
fn fill_uncovered(
    east: &mut Array2<f64>,
    north: &mut Array2<f64>,
    covered: &mut Array2<bool>,
    e: &Array2<f64>,
    n: &Array2<f64>,
) {
    Zip::from(east)
        .and(north)
        .and(covered)
        .and(e)
        .and(n)
        .for_each(|eo, no, cov, &ev, &nv| {
            // Finite + non-(0,0): GDAL fills out-of-coverage with 0; a NaN (corrupt
            // granule) must NOT be accepted as valid — it would poison up/incidence
            // instead of tripping the coverage guard.
            if !*cov && ev.is_finite() && nv.is_finite() && (ev != 0.0 || nv != 0.0) {
                *eo = ev;
                *no = nv;
                *cov = true;
            }
        });
}

/// Angle between the already-mosaicked LOS and this granule's LOS, in degrees, at
/// every pixel where both are valid. Empty when the footprints do not overlap.
fn overlap_disagreement_deg(
    east: &Array2<f64>,
    north: &Array2<f64>,
    covered: &Array2<bool>,
    e: &Array2<f64>,
    n: &Array2<f64>,
) -> Vec<f64> {
    let mut diffs = Vec::new();
    Zip::from(east)
        .and(north)
        .and(covered)
        .and(e)
        .and(n)
        .for_each(|&eo, &no, &cov, &ev, &nv| {
            let valid = cov && ev.is_finite() && nv.is_finite() && (ev != 0.0 || nv != 0.0);
            if valid {
                diffs.push(los_angle_between_deg([eo, no], [ev, nv]));
            }
        });
    diffs
}

/// Angle in degrees between two ground→sensor LOS unit vectors given their
/// horizontal components (`up` is derived, so the pair is fully determined).
fn los_angle_between_deg(a: [f64; 2], b: [f64; 2]) -> f64 {
    let up = |[e, n]: [f64; 2]| (1.0 - e * e - n * n).max(0.0).sqrt();
    let dot = a[0] * b[0] + a[1] * b[1] + up(a) * up(b);
    dot.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Error if overlapping granules disagree. Gates on the **median** rather than the
/// max: both footprints' bilinear edge rings (GDAL blends against nodata `0`) fall
/// inside the overlap and would dominate a max, whereas a granule that does not
/// belong to this frame disagrees across the whole overlap.
fn ensure_overlap_agreement(diffs: &mut [f64]) -> Result<()> {
    if diffs.len() < MIN_OVERLAP_PIXELS {
        return Ok(());
    }
    diffs.sort_by(f64::total_cmp);
    let median = diffs[diffs.len() / 2];
    if median <= OVERLAP_AGREEMENT_GATE_DEG {
        return Ok(());
    }
    Err(CorrectionError::GeometryOverlapMismatch(format!(
        "supplied CSLC-S1-STATIC granules disagree where they overlap: median LOS difference \
         {median:.3}° over {} overlap pixels exceeds the {OVERLAP_AGREEMENT_GATE_DEG}° gate \
         (max {:.3}°) — the granules are not same-track neighbours of this frame",
        diffs.len(),
        diffs[diffs.len() - 1]
    )))
}

/// Error if any frame pixel is uncovered by every supplied granule.
fn ensure_full_coverage(covered: &Array2<bool>) -> Result<()> {
    let uncovered = covered.iter().filter(|&&c| !c).count();
    if uncovered == 0 {
        return Ok(());
    }
    let frac = 100.0 * uncovered as f64 / covered.len() as f64;
    Err(CorrectionError::GeometryCoverage(format!(
        "{uncovered} frame pixels ({frac:.1}%) fall outside the supplied CSLC-S1-STATIC \
         coverage; supply the per-burst STATIC granules covering the frame"
    )))
}

/// Up component of the unit LOS vector: `+sqrt(max(0, 1 - e² - n²))`.
fn derive_up(east: &Array2<f64>, north: &Array2<f64>) -> Array2<f64> {
    Zip::from(east)
        .and(north)
        .map_collect(|&e, &n| (1.0 - e * e - n * n).max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-burst `LosLayers` on the given grid, constant components (no HDF5
    /// needed — `LosLayers` is public, so the geometry math is proven in-memory).
    fn constant_layer(shape: (usize, usize), e: f64, n: f64, gt: [f64; 6], epsg: u32) -> LosLayers {
        LosLayers {
            east: Array2::from_elem(shape, e),
            north: Array2::from_elem(shape, n),
            geo: GeoInfo {
                epsg,
                geotransform: gt,
            },
        }
    }

    /// Bar #1: a constant incidence θ=34° (az=30°) grid resolves (same-CRS warp) to
    /// `up ≈ cos34°`, `incidence ≈ 34°`, unit-norm to 1e-9, e/n preserved.
    #[test]
    fn resolves_constant_incidence() {
        let inc = 34.0_f64.to_radians();
        let az = 30.0_f64.to_radians();
        let (e, n) = (-inc.sin() * az.sin(), -inc.sin() * az.cos());
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let layer = constant_layer((8, 8), e, n, gt, 32614);
        let los = resolve_los_geometry(&[layer], gt, 32614, (8, 8)).unwrap();

        let (r, c) = (4, 4);
        assert!((los.up[(r, c)] - inc.cos()).abs() < 1e-6, "up");
        assert!(
            (los.incidence_deg()[(r, c)] - 34.0).abs() < 1e-3,
            "incidence"
        );
        let norm = los.east[(r, c)].powi(2) + los.north[(r, c)].powi(2) + los.up[(r, c)].powi(2);
        assert!((norm - 1.0).abs() < 1e-9, "unit-norm");
        assert!((los.east[(r, c)] - e).abs() < 1e-6 && (los.north[(r, c)] - n).abs() < 1e-6);
    }

    /// Bar #4: a frame that extends beyond the STATIC footprint is a hard coverage
    /// error, never a silent 0°/nadir fill.
    #[test]
    fn partial_coverage_is_error() {
        let src_gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let layer = constant_layer((6, 6), -0.3, -0.45, src_gt, 32614);
        // Frame origin shifted far east of the 6×30 m source extent → mostly uncovered.
        let dst_gt = [600_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let err = resolve_los_geometry(&[layer], dst_gt, 32614, (6, 6)).unwrap_err();
        assert!(matches!(err, CorrectionError::GeometryCoverage(_)), "{err}");
    }

    /// Empty granule list is a coverage error (the deliverable's front door must not
    /// hand back an all-nadir geometry).
    #[test]
    fn empty_layers_is_error() {
        let gt = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let err = resolve_los_geometry(&[], gt, 32614, (3, 3)).unwrap_err();
        assert!(matches!(err, CorrectionError::GeometryCoverage(_)));
    }

    /// Regression: an interior nodata (0,0) hole with no other burst to fill it is a
    /// hard coverage error — the guard must reject it, not emit a 0°/nadir pixel.
    #[test]
    fn interior_nodata_hole_is_error() {
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let mut layer = constant_layer((6, 6), -0.30, -0.45, gt, 32614);
        layer.east[(3, 3)] = 0.0;
        layer.north[(3, 3)] = 0.0;
        let err = resolve_los_geometry(&[layer], gt, 32614, (6, 6)).unwrap_err();
        assert!(matches!(err, CorrectionError::GeometryCoverage(_)), "{err}");
    }

    /// Asc/desc is encoded in the signed LOS vector: flipping the azimuth by 180°
    /// flips the east/north signs (no separate heading read needed). Validates the
    /// design's assumption that the signed unit vector resolves track direction.
    #[test]
    fn los_sign_encodes_track_direction() {
        let inc = 40.0_f64.to_radians();
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let desc = constant_layer((5, 5), -inc.sin() * 0.5, -inc.sin() * 0.866, gt, 32614);
        let asc = constant_layer((5, 5), inc.sin() * 0.5, inc.sin() * 0.866, gt, 32614);
        let lo_d = resolve_los_geometry(&[desc], gt, 32614, (5, 5)).unwrap();
        let lo_a = resolve_los_geometry(&[asc], gt, 32614, (5, 5)).unwrap();
        // Opposite east/north signs, identical up (incidence is heading-independent).
        assert!(lo_d.east[(2, 2)] * lo_a.east[(2, 2)] < 0.0);
        assert!((lo_d.up[(2, 2)] - lo_a.up[(2, 2)]).abs() < 1e-9);
    }

    /// Bar #2: a STATIC grid on EPSG:4326 resolves onto a UTM 32610 frame via the
    /// GDAL warp path (constant field warps to constant) — the cross-CRS dispatch,
    /// mirroring the tropo 4326→UTM warp test.
    #[test]
    fn resolves_across_crs() {
        let inc = 36.0_f64.to_radians();
        let (e, n) = (-inc.sin() * 0.4, -inc.sin() * 0.916);
        // Source on a geographic grid covering the frame's footprint.
        let src_gt = [-124.0, 0.05, 0.0, 39.0, 0.0, -0.05];
        let layer = constant_layer((40, 40), e, n, src_gt, 4326);
        let dst_gt = [495_000.0, 2_000.0, 0.0, 4_211_000.0, 0.0, -2_000.0];
        let los = resolve_los_geometry(&[layer], dst_gt, 32610, (5, 5)).unwrap();
        assert!(
            (los.incidence_deg()[(2, 2)] - 36.0).abs() < 0.02,
            "cross-CRS incidence"
        );
    }

    /// Bar #6: two granules mosaic first-valid-wins — burst A's nodata hole is filled
    /// from burst B, and A's valid region is kept over B. B is the along-track
    /// neighbour, so its geometry agrees with A's in the overlap (0.3° apart).
    #[test]
    fn mosaics_first_valid_wins() {
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        // Burst A: valid everywhere except a nodata (0,0) hole in the last two cols.
        let mut a = constant_layer((8, 8), -0.30, -0.45, gt, 32614);
        a.east.slice_mut(ndarray::s![.., 6..]).fill(0.0);
        a.north.slice_mut(ndarray::s![.., 6..]).fill(0.0);
        // Burst B: the along-track neighbour — valid everywhere, near-identical LOS.
        let b = constant_layer((8, 8), -0.305, -0.452, gt, 32614);

        let los = resolve_los_geometry(&[a, b], gt, 32614, (8, 8)).unwrap();
        // Interior of A's valid region keeps A.
        assert!((los.east[(2, 1)] - (-0.30)).abs() < 1e-6, "A region");
        // The hole (col 7) is filled from B.
        assert!((los.east[(2, 7)] - (-0.305)).abs() < 1e-6, "B fills hole");
    }

    /// #39: relaxing the provenance rule to same-track means a neighbour burst's
    /// LOS is mosaicked in, so first-covered-wins must be *checked* in the overlap.
    /// Two granules whose look geometry differs by ~6° are rejected, not silently
    /// resolved to whichever one happened to be listed first.
    #[test]
    fn disagreeing_overlap_is_error() {
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let a = constant_layer((8, 8), -0.30, -0.45, gt, 32614);
        let b = constant_layer((8, 8), -0.20, -0.50, gt, 32614);
        let err = resolve_los_geometry(&[a, b], gt, 32614, (8, 8)).unwrap_err();
        assert!(
            matches!(err, CorrectionError::GeometryOverlapMismatch(_)),
            "{err}"
        );
    }

    /// The overlap gate must not fire on granules that agree: a full-frame overlap
    /// of two near-identical neighbours resolves normally.
    #[test]
    fn agreeing_overlap_resolves() {
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let a = constant_layer((8, 8), -0.30, -0.45, gt, 32614);
        let b = constant_layer((8, 8), -0.302, -0.451, gt, 32614);
        let los = resolve_los_geometry(&[a, b], gt, 32614, (8, 8)).unwrap();
        assert!((los.east[(4, 4)] - (-0.30)).abs() < 1e-6);
    }
}
