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
//!
//! Frame pixels no granule covers are **masked**: NaN in every component, so they
//! are nodata in every downstream product and never a silent 0°/nadir pixel. The
//! count and fraction are reported ([`LosGeometry::outside_static`]) for the
//! geometry provenance, and the run is refused only when the fraction exceeds
//! [`LosCoverageOptions::max_outside_static_fraction`] — a corridor frame that runs
//! a few hundred pixels past its burst's STATIC is an edge; a third of the frame
//! without geometry is a wrong or missing granule.

use dolphin_io::{GeoInfo, LosLayers};
use ndarray::{Array2, Zip};

use crate::error::{CorrectionError, Result};
use crate::troposphere::{warp_to_frame, DelayGrid};

/// Default [`LosCoverageOptions::max_outside_static_fraction`]. Set from the three
/// frames eo observed against a single per-burst STATIC granule: 0.3% and 0.6%
/// outside are corridor frames running an edge strip past the granule and are
/// masked; 31.4% outside is a third of the frame with no geometry — a wrong or
/// missing granule, not an edge — and is refused. 10% sits well clear of both.
pub const DEFAULT_MAX_OUTSIDE_STATIC_FRACTION: f64 = 0.10;

/// How much of the frame may fall outside every supplied CSLC-S1-STATIC granule
/// before [`resolve_los_geometry_with_options`] refuses the run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LosCoverageOptions {
    /// Refuse when `outside_pixels / frame_pixels` exceeds this (strictly above).
    /// Below it the outside pixels are masked to NaN and reported, not refused.
    pub max_outside_static_fraction: f64,
}

impl Default for LosCoverageOptions {
    fn default() -> Self {
        Self {
            max_outside_static_fraction: DEFAULT_MAX_OUTSIDE_STATIC_FRACTION,
        }
    }
}

/// Frame pixels outside every supplied STATIC granule (masked to NaN).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutsideStatic {
    /// Number of masked frame pixels.
    pub pixel_count: usize,
    /// `pixel_count / frame_pixels`; `0.0` for an empty frame.
    pub fraction: f64,
}

/// Per-pixel LOS unit-vector components on the frame grid. `up` is derived as
/// `+sqrt(max(0, 1 - east² - north²))`; the incidence angle is [`Self::incidence_deg`].
/// Pixels outside every supplied STATIC granule are NaN in all three components.
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
    /// Count and fraction of frame pixels masked as outside every supplied STATIC
    /// granule (NaN `up`). Feeds the `outside_static_pixel_count` /
    /// `outside_static_fraction` geometry-provenance fields.
    #[must_use]
    pub fn outside_static(&self) -> OutsideStatic {
        let pixel_count = self.up.iter().filter(|u| u.is_nan()).count();
        let fraction = match self.up.len() {
            0 => 0.0,
            total => pixel_count as f64 / total as f64,
        };
        OutsideStatic {
            pixel_count,
            fraction,
        }
    }

    /// Per-pixel ellipsoidal incidence angle in degrees, `acos(up)·180/π` —
    /// character-identical to dolphin `atmosphere/ionosphere.py`. This is the angle
    /// the atmospheric zenith→slant `1/cos` mapping uses.
    #[must_use]
    pub fn incidence_deg(&self) -> Array2<f64> {
        self.up.mapv(|u| u.acos().to_degrees())
    }

    /// Spatial statistics of the ellipsoidal incidence angle over finite pixels —
    /// pixels outside the STATIC coverage are NaN and excluded. `None` when no
    /// pixel is finite. Std is the population std (numpy `ddof=0`).
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

/// [`resolve_los_geometry_with_options`] under [`LosCoverageOptions::default`].
///
/// # Errors
/// As [`resolve_los_geometry_with_options`].
pub fn resolve_los_geometry(
    layers: &[LosLayers],
    dst_gt: [f64; 6],
    dst_epsg: u32,
    shape: (usize, usize),
) -> Result<LosGeometry> {
    resolve_los_geometry_with_options(
        layers,
        dst_gt,
        dst_epsg,
        shape,
        LosCoverageOptions::default(),
    )
}

/// Resolve per-pixel LOS geometry onto the frame grid from one-or-more per-burst
/// CSLC-S1-STATIC granules. Each granule is reprojected onto `(dst_gt, dst_epsg,
/// shape)` and mosaicked (first covered burst wins); granules that overlap must
/// agree there. Frame pixels no granule covers are masked to NaN and counted
/// ([`LosGeometry::outside_static`]); the run is refused only when their fraction
/// exceeds `options.max_outside_static_fraction`.
///
/// # Errors
/// [`CorrectionError::GeometryCoverage`] if `layers` is empty or the outside
/// fraction exceeds the gate; [`CorrectionError::GeometryOverlapMismatch`] if
/// overlapping granules carry materially different LOS;
/// [`CorrectionError::Gdal`]/[`CorrectionError::Shape`] on warp failure.
pub fn resolve_los_geometry_with_options(
    layers: &[LosLayers],
    dst_gt: [f64; 6],
    dst_epsg: u32,
    shape: (usize, usize),
    options: LosCoverageOptions,
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
    ensure_coverage_within_gate(&covered, options)?;
    ensure_overlap_agreement(&mut disagreement_deg)?;
    let mut up = derive_up(&east, &north);
    for component in [&mut east, &mut north, &mut up] {
        mask_uncovered(component, &covered);
    }
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

/// Error if the fraction of frame pixels uncovered by every supplied granule
/// exceeds the gate (strictly above); at or below it the pixels are masked instead.
fn ensure_coverage_within_gate(covered: &Array2<bool>, options: LosCoverageOptions) -> Result<()> {
    let uncovered = covered.iter().filter(|&&c| !c).count();
    if uncovered == 0 {
        return Ok(());
    }
    let fraction = uncovered as f64 / covered.len() as f64;
    if fraction <= options.max_outside_static_fraction {
        return Ok(());
    }
    Err(CorrectionError::GeometryCoverage(format!(
        "{uncovered} frame pixels ({:.1}%) fall outside the supplied CSLC-S1-STATIC \
         coverage, above the {:.1}% gate (max_outside_static_fraction); supply the \
         per-burst STATIC granules covering the frame",
        100.0 * fraction,
        100.0 * options.max_outside_static_fraction
    )))
}

/// NaN every pixel no granule covers, so it is nodata downstream rather than a
/// `(0, 0, 1)` nadir vector.
fn mask_uncovered(component: &mut Array2<f64>, covered: &Array2<bool>) {
    Zip::from(component).and(covered).for_each(|value, &cov| {
        if !cov {
            *value = f64::NAN;
        }
    });
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

    /// Bar #4: a frame that lies almost entirely beyond the STATIC footprint is a
    /// hard coverage error (far above the gate), never a silent 0°/nadir fill.
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

    /// Regression: an interior nodata (0,0) hole with no other burst to fill it is
    /// masked to NaN and counted (1 of 36, below the gate) — never a 0°/nadir pixel.
    #[test]
    fn interior_nodata_hole_is_masked() {
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let mut layer = constant_layer((6, 6), -0.30, -0.45, gt, 32614);
        layer.east[(3, 3)] = 0.0;
        layer.north[(3, 3)] = 0.0;
        let los = resolve_los_geometry(&[layer], gt, 32614, (6, 6)).unwrap();
        assert!(
            los.up[(3, 3)].is_nan() && los.east[(3, 3)].is_nan(),
            "hole masked"
        );
        assert!((los.up[(3, 4)] - (1.0_f64 - 0.09 - 0.2025).sqrt()).abs() < 1e-9);
        assert_eq!(los.outside_static().pixel_count, 1);
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

    /// A `rows × cols` frame whose single STATIC granule covers every row except
    /// the last `outside_rows` — the eo corridor-frame case, where the frame runs
    /// past the one granule staged for the burst. Outside count is `outside_rows·cols`.
    fn frame_past_granule(
        rows: usize,
        cols: usize,
        outside_rows: usize,
    ) -> (Vec<LosLayers>, [f64; 6]) {
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let layer = constant_layer((rows - outside_rows, cols), -0.30, -0.45, gt, 32614);
        (vec![layer], gt)
    }

    /// Resolve a 1000×100 frame with `outside_rows` rows past the granule under
    /// the default gate.
    fn resolve_past_granule(outside_rows: usize) -> Result<LosGeometry> {
        let (layers, gt) = frame_past_granule(1000, 100, outside_rows);
        resolve_los_geometry(&layers, gt, 32614, (1000, 100))
    }

    /// Masked pixels are NaN in every component (nodata downstream, never a
    /// 0°/nadir pixel), covered pixels keep their LOS, and the count/fraction
    /// report exactly the out-of-granule pixels.
    fn assert_masked(los: &LosGeometry, outside_rows: usize, fraction: f64) {
        let outside = los.outside_static();
        assert_eq!(outside.pixel_count, outside_rows * 100, "count");
        assert!(
            (outside.fraction - fraction).abs() < 1e-12,
            "fraction {}",
            outside.fraction
        );
        let first_outside = 1000 - outside_rows;
        for c in [0, 50, 99] {
            let inside = (first_outside - 1, c);
            assert!(
                (los.east[inside] - (-0.30)).abs() < 1e-6,
                "inside east {inside:?}"
            );
            assert!(los.up[inside].is_finite(), "inside up {inside:?}");
            let out = (first_outside, c);
            assert!(
                los.east[out].is_nan() && los.north[out].is_nan() && los.up[out].is_nan(),
                "{out:?} masked"
            );
            assert!(los.up[(999, c)].is_nan(), "last row masked");
        }
        let stats = los
            .incidence_stats()
            .expect("stats over the covered pixels");
        assert!(
            stats.mean_deg.is_finite() && stats.std_deg < 1e-9,
            "{stats:?}"
        );
    }

    /// eo held-out block t137_292324_iw1: 0.3% of the frame past the single STATIC
    /// granule (227 of ~75,000 real pixels). Masked, counted, not refused.
    #[test]
    fn edge_sliver_is_masked_not_refused() {
        let los = resolve_past_granule(3).expect("0.3% outside must resolve");
        assert_masked(&los, 3, 0.003);
    }

    /// eo calibration block t144_308004_iw1: 0.6% outside (1,315 pixels). Masked.
    #[test]
    fn small_edge_strip_is_masked_not_refused() {
        let los = resolve_past_granule(6).expect("0.6% outside must resolve");
        assert_masked(&los, 6, 0.006);
    }

    /// eo calibration block t137_292338_iw2: 31.4% outside (28,323 pixels) — a
    /// third of the frame without geometry is a wrong/missing-granule condition,
    /// refused under the default gate with the fraction in the message; a caller
    /// that raises the gate gets the masked geometry with the same count.
    #[test]
    fn third_of_frame_outside_is_refused_by_default() {
        let err = resolve_past_granule(314).expect_err("31.4% outside must refuse");
        assert!(matches!(err, CorrectionError::GeometryCoverage(_)), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("31400") && msg.contains("31.4%"), "{msg}");

        let (layers, gt) = frame_past_granule(1000, 100, 314);
        let options = LosCoverageOptions {
            max_outside_static_fraction: 0.5,
        };
        let los = resolve_los_geometry_with_options(&layers, gt, 32614, (1000, 100), options)
            .expect("raised gate admits the frame");
        assert_masked(&los, 314, 0.314);
    }

    /// The gate refuses strictly *above* the configured fraction: a frame exactly
    /// at the gate resolves, one just past it is refused. Default is 10%.
    #[test]
    fn refuses_only_above_the_configured_fraction() {
        assert!((LosCoverageOptions::default().max_outside_static_fraction - 0.10).abs() < 1e-12);
        let (layers, gt) = frame_past_granule(1000, 100, 3);
        let at_gate = LosCoverageOptions {
            max_outside_static_fraction: 0.003,
        };
        resolve_los_geometry_with_options(&layers, gt, 32614, (1000, 100), at_gate)
            .expect("fraction equal to the gate resolves");
        let below_gate = LosCoverageOptions {
            max_outside_static_fraction: 0.0029,
        };
        let err = resolve_los_geometry_with_options(&layers, gt, 32614, (1000, 100), below_gate)
            .expect_err("fraction above the gate refuses");
        assert!(matches!(err, CorrectionError::GeometryCoverage(_)), "{err}");
    }

    /// A fully covered frame reports zero outside pixels.
    #[test]
    fn full_coverage_reports_zero_outside() {
        let los = resolve_past_granule(0).unwrap();
        let outside = los.outside_static();
        assert_eq!(outside.pixel_count, 0);
        assert_eq!(outside.fraction, 0.0);
    }
}
