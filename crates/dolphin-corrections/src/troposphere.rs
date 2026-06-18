//! Tropospheric range delay from the OPERA L4 product (netCDF) or RAiDER.
//!
//! The tropospheric delay is **non-dispersive** (frequency-independent): the same
//! range delay in meters applies to L- and C-band alike. The primary source is the
//! free public OPERA L4 tropospheric product (a DISP-S1-aligned netCDF); RAiDER is
//! the fallback for scenes without an L4 product ([`crate::raider`]).
//!
//! netCDF is read through GDAL's `NETCDF:` driver (same GDAL the rest of the
//! pipeline links), then resampled onto the displacement frame grid.

use std::path::Path;

use gdal::raster::Buffer;
use gdal::Dataset;
use ndarray::{Array2, ArrayView2};

use crate::error::{CorrectionError, Result};

/// A georeferenced single-band delay grid in meters.
#[derive(Debug, Clone)]
pub struct DelayGrid {
    /// Range delay in meters, `(rows, cols)`.
    pub data: Array2<f64>,
    /// GDAL affine geotransform `[origin_x, dx, 0, origin_y, 0, dy]`.
    pub geotransform: [f64; 6],
    /// EPSG of the grid, if carried.
    pub epsg: Option<u32>,
    /// Full source CRS as WKT, if carried. Drives the 4326→UTM warp when the real
    /// OPERA L4 product reports no EPSG authority code but is georeferenced.
    pub srs_wkt: Option<String>,
}

/// Read a tropospheric range-delay band (meters) from an OPERA L4 netCDF variable.
///
/// `var` is the netCDF variable name (e.g. `troposphere`); GDAL is opened on the
/// `NETCDF:"path":var` connection string.
///
/// # Errors
/// Returns [`CorrectionError::Gdal`] if the variable or grid cannot be read.
pub fn read_l4_netcdf(path: &Path, var: &str) -> Result<DelayGrid> {
    let conn = format!("NETCDF:\"{}\":{var}", path.display());
    let ds = Dataset::open(Path::new(&conn))?;
    let (cols, rows) = ds.raster_size();
    let band = ds.rasterband(1)?;
    let buf: Buffer<f64> = band.read_as((0, 0), (cols, rows), (cols, rows), None)?;
    let (_, values) = buf.into_shape_and_vec();
    let data = Array2::from_shape_vec((rows, cols), values)
        .map_err(|e| CorrectionError::Shape(e.to_string()))?;
    let geotransform = ds.geo_transform()?;
    let (epsg, srs_wkt) = source_crs(ds.spatial_ref().ok(), geotransform);
    Ok(DelayGrid {
        data,
        geotransform,
        epsg,
        srs_wkt,
    })
}

/// Resolve the source CRS for a delay grid. Prefers the netCDF's embedded SRS;
/// when absent (the real OPERA L4 TROPO-ZENITH product exposes its per-variable
/// grids with **no CRS** through GDAL's NETCDF driver) it falls back to EPSG:4326
/// **only** if the geotransform spans geographic-degree ranges — which the global
/// plate-carrée L4 product does (origin ≈ −180°, ±90° latitude). A projected grid
/// with no SRS stays unset rather than being mislabeled geographic.
fn source_crs(
    srs: Option<gdal::spatial_ref::SpatialRef>,
    gt: [f64; 6],
) -> (Option<u32>, Option<String>) {
    if let Some(sr) = srs {
        let epsg = sr.auth_code().ok().map(|c| c as u32);
        return (epsg, sr.to_wkt().ok());
    }
    if !is_geographic_degrees(gt) {
        return (None, None);
    }
    let wgs84 = gdal::spatial_ref::SpatialRef::from_epsg(4326);
    (Some(4326), wgs84.ok().and_then(|sr| sr.to_wkt().ok()))
}

/// Whether a geotransform's pixel-centre extent lies within longitude/latitude
/// degree bounds (a cheap geographic-CRS heuristic for a CRS-less grid).
fn is_geographic_degrees(gt: [f64; 6]) -> bool {
    let [ox, dx, _, oy, _, dy] = gt;
    (-181.0..=181.0).contains(&ox)
        && (-91.0..=91.0).contains(&oy)
        && dx.abs() < 5.0
        && dy.abs() < 5.0
}

/// The two OPERA L4 TROPO-ZENITH variables whose sum is the total zenith
/// tropospheric delay (verified against a real `OPERA_L4_TROPO-ZENITH_V1` granule).
pub const L4_TOTAL_VARS: [&str; 2] = ["hydrostatic_delay", "wet_delay"];

/// Read the **total** zenith tropospheric delay (meters) from an OPERA L4 product
/// by summing its `hydrostatic_delay` and `wet_delay` variables. Use this for the
/// real product; [`read_l4_netcdf`] reads a single named variable (e.g. a
/// synthesized fixture or a product exposing a single field).
///
/// # Errors
/// Returns `Err` if either variable cannot be read or they differ in shape.
pub fn read_l4_total(path: &Path) -> Result<DelayGrid> {
    let hydro = read_l4_netcdf(path, L4_TOTAL_VARS[0])?;
    let wet = read_l4_netcdf(path, L4_TOTAL_VARS[1])?;
    if hydro.data.dim() != wet.data.dim() {
        return Err(CorrectionError::Shape(format!(
            "hydrostatic {:?} != wet {:?}",
            hydro.data.dim(),
            wet.data.dim()
        )));
    }
    Ok(DelayGrid {
        data: hydro.data + wet.data,
        geotransform: hydro.geotransform,
        epsg: hydro.epsg,
        srs_wkt: hydro.srs_wkt,
    })
}

/// Reproject (warp) a source delay grid onto a destination frame grid that may be
/// in a **different CRS** (e.g. a global EPSG:4326 OPERA L4 product onto a UTM
/// NISAR/DISP-S1 frame). Builds in-memory GDAL datasets carrying each grid's CRS
/// and geotransform and calls GDAL's bilinear reproject; out-of-coverage frame
/// pixels come back as the band no-data (0.0 here). Use this in place of
/// [`resample_bilinear`] whenever the source and frame CRS differ.
///
/// # Errors
/// [`CorrectionError::NoSourceCrs`] if the source carries neither a WKT nor an
/// EPSG to reproject from; [`CorrectionError::Gdal`] on a GDAL failure.
pub fn warp_to_frame(
    src: &DelayGrid,
    dst_gt: [f64; 6],
    dst_epsg: u32,
    dst_shape: (usize, usize),
) -> Result<Array2<f64>> {
    use gdal::raster::reproject;
    use gdal::raster::Buffer;
    use gdal::spatial_ref::{AxisMappingStrategy, SpatialRef};
    use gdal::DriverManager;

    let src_srs = source_srs(src)?;
    let (src_rows, src_cols) = src.data.dim();
    let mem = DriverManager::get_driver_by_name("MEM")?;

    let mut src_ds = mem.create_with_band_type::<f64, _>("", src_cols, src_rows, 1)?;
    src_ds.set_geo_transform(&src.geotransform)?;
    src_ds.set_spatial_ref(&src_srs)?;
    {
        let mut band = src_ds.rasterband(1)?;
        let mut buf = Buffer::new((src_cols, src_rows), src.data.iter().copied().collect());
        band.write((0, 0), (src_cols, src_rows), &mut buf)?;
    }

    let (dst_rows, dst_cols) = dst_shape;
    let mut dst_srs = SpatialRef::from_epsg(dst_epsg)?;
    dst_srs.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
    let mut dst_ds = mem.create_with_band_type::<f64, _>("", dst_cols, dst_rows, 1)?;
    dst_ds.set_geo_transform(&dst_gt)?;
    dst_ds.set_spatial_ref(&dst_srs)?;

    reproject(&src_ds, &dst_ds)?;

    let band = dst_ds.rasterband(1)?;
    let buf: Buffer<f64> =
        band.read_as((0, 0), (dst_cols, dst_rows), (dst_cols, dst_rows), None)?;
    let (_, values) = buf.into_shape_and_vec();
    Array2::from_shape_vec((dst_rows, dst_cols), values)
        .map_err(|e| CorrectionError::Shape(e.to_string()))
}

/// Reconstruct the source CRS for the warp: prefer the carried WKT (the real L4
/// product reports a CRS but no EPSG authority code), falling back to the EPSG.
fn source_srs(src: &DelayGrid) -> Result<gdal::spatial_ref::SpatialRef> {
    use gdal::spatial_ref::{AxisMappingStrategy, SpatialRef};
    let mut srs = match (&src.srs_wkt, src.epsg) {
        (Some(wkt), _) => SpatialRef::from_wkt(wkt)?,
        (None, Some(epsg)) => SpatialRef::from_epsg(epsg)?,
        (None, None) => return Err(CorrectionError::NoSourceCrs),
    };
    srs.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
    Ok(srs)
}

/// Resample a source delay grid onto a destination frame grid by bilinear
/// interpolation in geocoded coordinates. Both grids must share the CRS (the
/// caller validates EPSG); out-of-coverage destination pixels clamp to the source
/// edge. When the grids coincide this is the identity.
#[must_use]
pub fn resample_bilinear(
    src: ArrayView2<f64>,
    src_gt: [f64; 6],
    dst_gt: [f64; 6],
    dst_shape: (usize, usize),
) -> Array2<f64> {
    let (dst_rows, dst_cols) = dst_shape;
    let (src_rows, src_cols) = src.dim();
    Array2::from_shape_fn((dst_rows, dst_cols), |(r, c)| {
        let x = dst_gt[0] + (c as f64 + 0.5) * dst_gt[1];
        let y = dst_gt[3] + (r as f64 + 0.5) * dst_gt[5];
        let fc = (x - src_gt[0]) / src_gt[1] - 0.5;
        let fr = (y - src_gt[3]) / src_gt[5] - 0.5;
        sample(src, fr, fc, src_rows, src_cols)
    })
}

/// Bilinear sample at fractional `(row, col)` with edge clamping.
fn sample(src: ArrayView2<f64>, fr: f64, fc: f64, rows: usize, cols: usize) -> f64 {
    let r0 = (fr.floor() as isize).clamp(0, rows as isize - 1) as usize;
    let c0 = (fc.floor() as isize).clamp(0, cols as isize - 1) as usize;
    let r1 = (r0 + 1).min(rows - 1);
    let c1 = (c0 + 1).min(cols - 1);
    let dr = (fr - r0 as f64).clamp(0.0, 1.0);
    let dc = (fc - c0 as f64).clamp(0.0, 1.0);
    let top = src[(r0, c0)] * (1.0 - dc) + src[(r0, c1)] * dc;
    let bot = src[(r1, c0)] * (1.0 - dc) + src[(r1, c1)] * dc;
    top * (1.0 - dr) + bot * dr
}

/// Write a single-band netCDF fixture (an OPERA-L4-format delay field) via GDAL,
/// for the ingest contract test. The variable is GDAL's default `Band1`.
#[cfg(test)]
fn write_netcdf_fixture(
    path: &Path,
    data: ArrayView2<f64>,
    geotransform: [f64; 6],
    epsg: u32,
) -> Result<()> {
    use gdal::raster::Buffer;
    use gdal::spatial_ref::SpatialRef;
    use gdal::DriverManager;
    let (rows, cols) = data.dim();
    let mem = DriverManager::get_driver_by_name("MEM")?;
    let mut src = mem.create_with_band_type::<f64, _>("", cols, rows, 1)?;
    src.set_geo_transform(&geotransform)?;
    src.set_spatial_ref(&SpatialRef::from_epsg(epsg)?)?;
    {
        let mut band = src.rasterband(1)?;
        let mut buffer = Buffer::new((cols, rows), data.iter().copied().collect());
        band.write((0, 0), (cols, rows), &mut buffer)?;
    }
    let nc = DriverManager::get_driver_by_name("netCDF")?;
    src.create_copy(&nc, path, &Default::default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Contract (DoD #3): a synthesized OPERA-L4-format netCDF round-trips through
    /// the GDAL ingest with its known field and geotransform recovered.
    #[test]
    fn ingests_synthesized_l4_netcdf() {
        let path = std::env::temp_dir().join("dolphin_l4_contract.nc");
        let _ = std::fs::remove_file(&path);
        let field = array![[0.10_f64, 0.12, 0.14], [0.16, 0.18, 0.20]];
        let gt = [300_000.0, 20.0, 0.0, 4_100_000.0, 0.0, -20.0];
        write_netcdf_fixture(&path, field.view(), gt, 32610).unwrap();

        let grid = read_l4_netcdf(&path, "Band1").unwrap();
        assert_eq!(grid.data.dim(), (2, 3));
        assert!((grid.data[(0, 0)] - 0.10).abs() < 1e-9);
        assert!((grid.data[(1, 2)] - 0.20).abs() < 1e-9);
        assert_eq!(grid.epsg, Some(32610));
        // Resampling onto the same grid recovers the field exactly.
        let resampled = resample_bilinear(grid.data.view(), grid.geotransform, gt, (2, 3));
        assert!((resampled[(1, 1)] - field[(1, 1)]).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }

    /// Real-data gate: ingest a real `OPERA_L4_TROPO-ZENITH_V1` granule (path in
    /// `OPERA_L4_REAL`) and confirm the total zenith delay is physically plausible.
    /// Ignored unless the env var is set.
    #[test]
    fn real_opera_l4_total_is_physical() {
        let Ok(path) = std::env::var("OPERA_L4_REAL") else {
            return;
        };
        let grid = read_l4_total(std::path::Path::new(&path)).expect("read real L4 total");
        // Real-product facts (band 1 = first time step): a large global lat/lon
        // grid; the product CRS may carry no EPSG authority code.
        let (rows, cols) = grid.data.dim();
        eprintln!("L4 grid = {rows}x{cols}, epsg = {:?}", grid.epsg);
        assert!(rows > 1000 && cols > 2000, "global grid, got {rows}x{cols}");
        // Sample the densest valid region (mid-grid); corners are 9.97e36 no-data.
        // Total ZTD ≈ hydrostatic (~2.3 m) + wet (~0–0.4 m) at sea level.
        let ztd = grid.data[(rows / 2, cols / 2)];
        eprintln!("centre total ZTD = {ztd} m");
        assert!(
            (1.0..4.0).contains(&ztd),
            "centre total ZTD {ztd} m should be meters-scale"
        );
    }

    /// Contract (Phase 1): a synthesized **EPSG:4326** L4 delay field, linear in
    /// (lon, lat), warped onto a **UTM (32610)** frame recovers the analytic delay
    /// at known frame pixels — proving the 4326→UTM reprojecting resample, not the
    /// shared-CRS bilinear path. Tolerance is tight because bilinear interpolation
    /// of a field linear in (lon, lat) is exact in the source's uniform-degree
    /// index space.
    #[test]
    fn warps_4326_field_onto_utm_frame() {
        use gdal::spatial_ref::{AxisMappingStrategy, CoordTransform, SpatialRef};

        // Analytic 4326 delay field: delay(lon, lat) = a + b·(lon+123) + c·(lat−38).
        let (a, b, c) = (2.0_f64, 0.10_f64, 0.05_f64);
        let field = |lon: f64, lat: f64| a + b * (lon + 123.0) + c * (lat - 38.0);

        // Source 4326 grid: lon ∈ [−124, −122], lat ∈ [37, 39] at 0.1°, top-left origin.
        let (src_rows, src_cols) = (21_usize, 21_usize);
        let src_gt = [-124.0, 0.1, 0.0, 39.0, 0.0, -0.1];
        let src = Array2::from_shape_fn((src_rows, src_cols), |(r, col)| {
            let lon = src_gt[0] + (col as f64 + 0.5) * src_gt[1];
            let lat = src_gt[3] + (r as f64 + 0.5) * src_gt[5];
            field(lon, lat)
        });
        let grid = DelayGrid {
            data: src,
            geotransform: src_gt,
            epsg: Some(4326),
            srs_wkt: SpatialRef::from_epsg(4326).unwrap().to_wkt().ok(),
        };

        // UTM 10N frame interior to the source: 5×5 at 2 km around the zone centre.
        let (rows, cols) = (5_usize, 5_usize);
        let dst_gt = [495_000.0, 2_000.0, 0.0, 4_211_000.0, 0.0, -2_000.0];
        let warped = warp_to_frame(&grid, dst_gt, 32610, (rows, cols)).unwrap();

        // Expected: transform each frame pixel centre 32610→4326 and evaluate the field.
        let mut utm = SpatialRef::from_epsg(32610).unwrap();
        let mut wgs = SpatialRef::from_epsg(4326).unwrap();
        utm.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
        wgs.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
        let ct = CoordTransform::new(&utm, &wgs).unwrap();
        for (r, cc) in [(0_usize, 0_usize), (2, 2), (4, 4), (0, 4), (4, 0)] {
            let x = dst_gt[0] + (cc as f64 + 0.5) * dst_gt[1];
            let y = dst_gt[3] + (r as f64 + 0.5) * dst_gt[5];
            let (mut xs, mut ys, mut zs) = ([x], [y], []);
            ct.transform_coords(&mut xs, &mut ys, &mut zs).unwrap();
            let expected = field(xs[0], ys[0]);
            let got = warped[(r, cc)];
            assert!(
                (got - expected).abs() < 5e-3,
                "pixel ({r},{cc}): warped {got} vs analytic {expected}"
            );
        }
    }

    /// Resampling a grid onto its own geotransform is the identity.
    #[test]
    fn identity_resample() {
        let src = array![[1.0, 2.0], [3.0, 4.0]];
        let gt = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let out = resample_bilinear(src.view(), gt, gt, (2, 2));
        assert_eq!(out, src);
    }

    /// Bilinear interpolation onto a half-pixel-shifted grid averages neighbours.
    #[test]
    fn bilinear_midpoint() {
        let src = array![[0.0, 10.0], [0.0, 10.0]];
        let gt = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        // Destination pixel centered on the source column midpoint (x = 1.0).
        let dst_gt = [0.5, 1.0, 0.0, 0.0, 0.0, -1.0];
        let out = resample_bilinear(src.view(), gt, dst_gt, (1, 1));
        assert!((out[(0, 0)] - 5.0).abs() < 1e-9, "midpoint {}", out[(0, 0)]);
    }
}
