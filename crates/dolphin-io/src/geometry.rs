//! Per-pixel line-of-sight geometry from the OPERA CSLC-S1-STATIC product.
//!
//! The static-layer companion product carries the **ground→sensor** LOS unit-vector
//! components `/data/los_east`, `/data/los_north` (`float32`, dimensionless) on the
//! burst's projected grid, alongside the same coordinate/projection datasets the
//! CSLC grid uses (read via [`crate::geo::read_geotransform`]). There is no stored
//! `up`/`z` layer — the up component is derived downstream. Out-of-scene samples are
//! the product's nodata (`0`). Reference: OPERA CSLC-S1-STATIC Product Specification
//! §4.3 / §5.3.

use std::path::Path;

use ndarray::s;
use ndarray::Array2;

use crate::error::Result;
use crate::geo::{read_geotransform, GeoInfo};

/// Raw LOS unit-vector components from one CSLC-S1-STATIC granule, on the granule's
/// own projected grid (`geo`). Components are the East / North parts of the
/// **ground→sensor** unit vector; out-of-scene samples are the product's nodata (0).
#[derive(Debug, Clone)]
pub struct LosLayers {
    /// East component of the ground→sensor LOS unit vector, `(rows, cols)`.
    pub east: Array2<f64>,
    /// North component of the ground→sensor LOS unit vector, `(rows, cols)`.
    pub north: Array2<f64>,
    /// Georeferencing (EPSG + geotransform) of this granule's grid.
    pub geo: GeoInfo,
}

/// Read the `los_east` / `los_north` layers and georeferencing from a
/// CSLC-S1-STATIC HDF5 file. `group` is the parent group of the geometry layers
/// (`/data` for OPERA CSLC-S1-STATIC).
///
/// # Errors
/// Returns `Err` if the file, the `los_east`/`los_north` datasets, or the
/// coordinate/projection datasets are absent or unreadable.
pub fn read_los_layers(path: &Path, group: &str) -> Result<LosLayers> {
    let file = hdf5::File::open(path)?;
    let east = file
        .dataset(&format!("{group}/los_east"))?
        .read_2d::<f32>()?
        .mapv(f64::from);
    let north = file
        .dataset(&format!("{group}/los_north"))?
        .read_2d::<f32>()?
        .mapv(f64::from);
    let geo = read_geotransform(path, &format!("{group}/los_east"))?;
    Ok(LosLayers { east, north, geo })
}

/// Read only the native STATIC pixels intersecting an aligned target grid.
/// Returns `None` when this granule does not intersect the target. No resampling
/// is performed; CRS, posting, and pixel alignment must match exactly.
pub fn read_los_layers_for_grid(
    path: &Path,
    group: &str,
    target_geo: GeoInfo,
    target_shape: (usize, usize),
) -> Result<Option<LosLayers>> {
    const TOLERANCE: f64 = 1e-6;
    let source_geo = read_geotransform(path, &format!("{group}/los_east"))?;
    if source_geo.epsg != target_geo.epsg {
        return Err(crate::error::IoError::Geo(format!(
            "STATIC EPSG {} differs from target EPSG {}",
            source_geo.epsg, target_geo.epsg
        )));
    }
    let sg = source_geo.geotransform;
    let tg = target_geo.geotransform;
    let col_scale = tg[1] / sg[1];
    let row_scale = tg[5] / sg[5];
    if (col_scale - col_scale.round()).abs() > TOLERANCE
        || (row_scale - row_scale.round()).abs() > TOLERANCE
        || col_scale < 1.0
        || row_scale < 1.0
        || sg[2].abs() > TOLERANCE
        || sg[4].abs() > TOLERANCE
        || tg[2].abs() > TOLERANCE
        || tg[4].abs() > TOLERANCE
    {
        return Err(crate::error::IoError::Geo(
            "STATIC posting is not an integer-aligned native refinement of the target grid".into(),
        ));
    }
    let file = hdf5::File::open(path)?;
    let east_ds = file.dataset(&format!("{group}/los_east"))?;
    let north_ds = file.dataset(&format!("{group}/los_north"))?;
    let shape = east_ds.shape();
    if shape.len() != 2 || north_ds.shape() != shape {
        return Err(crate::error::IoError::Shape(
            "STATIC LOS components must share a two-dimensional shape".into(),
        ));
    }
    let source_row = ((sg[3] - tg[3]) / -sg[5]).round() as isize;
    let source_col = ((tg[0] - sg[0]) / sg[1]).round() as isize;
    if (((sg[3] - tg[3]) / -sg[5]) - source_row as f64).abs() > TOLERANCE
        || (((tg[0] - sg[0]) / sg[1]) - source_col as f64).abs() > TOLERANCE
    {
        return Err(crate::error::IoError::Geo(
            "STATIC origin has a subpixel target offset".into(),
        ));
    }
    let row_start = source_row.max(0) as usize;
    let col_start = source_col.max(0) as usize;
    let row_stop =
        (source_row + (target_shape.0 as f64 * row_scale).round() as isize).min(shape[0] as isize);
    let col_stop =
        (source_col + (target_shape.1 as f64 * col_scale).round() as isize).min(shape[1] as isize);
    if row_stop <= row_start as isize || col_stop <= col_start as isize {
        return Ok(None);
    }
    let row_stop = row_stop as usize;
    let col_stop = col_stop as usize;
    let east = east_ds
        .read_slice_2d::<f32, _>(s![row_start..row_stop, col_start..col_stop])?
        .mapv(f64::from);
    let north = north_ds
        .read_slice_2d::<f32, _>(s![row_start..row_stop, col_start..col_stop])?
        .mapv(f64::from);
    Ok(Some(LosLayers {
        east,
        north,
        geo: GeoInfo {
            epsg: source_geo.epsg,
            geotransform: [
                sg[0] + col_start as f64 * sg[1],
                sg[1],
                0.0,
                sg[3] + row_start as f64 * sg[5],
                0.0,
                sg[5],
            ],
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal CSLC-S1-STATIC-layout HDF5: `los_east`/`los_north` (f32)
    /// plus the coordinate/projection datasets `read_geotransform` needs.
    fn write_static_fixture(
        path: &Path,
        east: &Array2<f32>,
        north: &Array2<f32>,
        x: &[f64],
        y: &[f64],
        epsg: i64,
    ) {
        let _ = std::fs::remove_file(path);
        let f = hdf5::File::create(path).unwrap();
        let g = f.create_group("data").unwrap();
        g.new_dataset_builder()
            .with_data(east)
            .create("los_east")
            .unwrap();
        g.new_dataset_builder()
            .with_data(north)
            .create("los_north")
            .unwrap();
        g.new_dataset_builder()
            .with_data(x)
            .create("x_coordinates")
            .unwrap();
        g.new_dataset_builder()
            .with_data(y)
            .create("y_coordinates")
            .unwrap();
        g.new_dataset::<i64>()
            .create("projection")
            .unwrap()
            .write_scalar(&epsg)
            .unwrap();
    }

    /// Contract: `read_los_layers` returns the f32 LOS components (as f64) and the
    /// grid's EPSG + geotransform from the OPERA `/data` layout.
    #[test]
    fn reads_static_los_layers() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_static_los_contract.h5");
        let east = Array2::from_shape_fn((4, 5), |(r, c)| 0.1 * (r + c) as f32);
        let north = Array2::from_shape_fn((4, 5), |(r, c)| -0.2 * (r + 1) as f32 + c as f32 * 0.01);
        let x = [1000.0_f64, 1030.0, 1060.0, 1090.0, 1120.0];
        let y = [2000.0_f64, 1970.0, 1940.0, 1910.0];
        write_static_fixture(&path, &east, &north, &x, &y, 32614);

        let got = read_los_layers(&path, "/data").unwrap();
        assert_eq!(got.geo.epsg, 32614);
        assert_eq!(got.east.dim(), (4, 5));
        assert!((got.east[(1, 2)] - 0.3).abs() < 1e-6);
        assert!((got.north[(3, 4)] - (-0.2 * 4.0 + 0.04)).abs() < 1e-6);
        assert!((got.geo.geotransform[1] - 30.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bounded_static_read_uses_native_window_and_offset_georeference() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_static_window_contract.h5");
        let east = Array2::from_shape_fn((10, 12), |(r, c)| (r * 12 + c + 1) as f32 / 1000.0);
        let north = Array2::from_elem((10, 12), -0.2_f32);
        let x = (0..12)
            .map(|col| 1_015.0 + col as f64 * 30.0)
            .collect::<Vec<_>>();
        let y = (0..10)
            .map(|row| 1_985.0 - row as f64 * 30.0)
            .collect::<Vec<_>>();
        write_static_fixture(&path, &east, &north, &x, &y, 32611);
        let target = GeoInfo {
            epsg: 32611,
            geotransform: [1_060.0, 60.0, 0.0, 1_940.0, 0.0, -60.0],
        };
        let got = read_los_layers_for_grid(&path, "/data", target, (3, 4))
            .unwrap()
            .expect("intersecting native window");
        assert_eq!(got.east.dim(), (6, 8));
        assert_eq!(
            got.geo.geotransform,
            [1_060.0, 30.0, 0.0, 1_940.0, 0.0, -30.0]
        );
        assert!((got.east[(0, 0)] - east[(2, 2)] as f64).abs() < 1e-8);
        let _ = std::fs::remove_file(&path);
    }
}

/// Read bounded NISAR LOS geometry, interpolating a radarGrid cube at verified
/// ellipsoidal DEM heights.
///
/// # Errors
/// Rejects missing DEM, unsupported CRS/grid alignment, nodata, incomplete
/// coverage, and heights outside the source cube. Never extrapolates.
pub fn read_nisar_los_layers_for_grid(
    path: &Path,
    group: &str,
    target_geo: GeoInfo,
    target_shape: (usize, usize),
    ellipsoidal_dem: Option<&Path>,
) -> Result<Option<LosLayers>> {
    use crate::error::IoError;
    let geo = crate::nisar::read_nisar_geotransform(path, &format!("{group}/losUnitVectorX"))?;
    if geo.epsg != target_geo.epsg {
        return Err(IoError::Geo("NISAR LOS and target CRS differ".into()));
    }
    let file = hdf5::File::open(path)?;
    let x = file
        .dataset(&format!("{group}/xCoordinates"))?
        .read_raw::<f64>()?;
    let y = file
        .dataset(&format!("{group}/yCoordinates"))?
        .read_raw::<f64>()?;
    let east_ds = file.dataset(&format!("{group}/losUnitVectorX"))?;
    let north_ds = file.dataset(&format!("{group}/losUnitVectorY"))?;
    let shape = east_ds.shape();
    let heights = file
        .dataset(&format!("{group}/heightAboveEllipsoid"))?
        .read_raw::<f64>()?;
    if heights.len() < 2
        || heights.iter().any(|v| !v.is_finite())
        || heights.windows(2).any(|w| w[0] >= w[1])
        || shape != north_ds.shape()
        || shape != [heights.len(), y.len(), x.len()]
    {
        return Err(IoError::Shape("invalid NISAR LOS height dimensions".into()));
    }
    let dem = read_ellipsoidal_dem(
        ellipsoidal_dem.ok_or_else(|| {
            IoError::Geo("NISAR radarGrid requires a verified ellipsoidal DEM".into())
        })?,
        target_geo,
        target_shape,
    )?;
    let min_height = dem.iter().copied().fold(f64::INFINITY, f64::min);
    let max_height = dem.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min_height < heights[0] || max_height > heights[heights.len() - 1] {
        return Err(IoError::Geo(
            "ellipsoidal DEM height lies outside NISAR LOS cube".into(),
        ));
    }
    let h0 = heights
        .partition_point(|h| *h < min_height)
        .saturating_sub(1);
    let h1 = (heights.partition_point(|h| *h <= max_height) + 1).min(heights.len());
    let (rows, cols, offset) = nisar_xy_window(&x, &y, geo, target_geo, target_shape)?;
    let read = |ds: &hdf5::Dataset| -> Result<ndarray::Array3<f64>> {
        Ok(ds.read_slice::<f64, _, ndarray::Ix3>(s![h0..h1, rows.clone(), cols.clone()])?)
    };
    let east = read(&east_ds)?;
    let north = read(&north_ds)?;
    if east
        .iter()
        .zip(north.iter())
        .any(|(&e, &n)| !e.is_finite() || !n.is_finite() || e * e + n * n >= 1.0)
    {
        return Err(IoError::Geo(
            "NISAR LOS has invalid or non-upward unit vectors".into(),
        ));
    }
    let interpolate = |source: &ndarray::Array3<f64>| {
        Array2::from_shape_fn(target_shape, |(r, c)| {
            let sy = offset.0 + r as f64 * target_geo.geotransform[5] / geo.geotransform[5];
            let sx = offset.1 + c as f64 * target_geo.geotransform[1] / geo.geotransform[1];
            let (lo_y, lo_x) = (sy.floor() as usize, sx.floor() as usize);
            let (hi_y, hi_x) = (
                (lo_y + 1).min(source.dim().1 - 1),
                (lo_x + 1).min(source.dim().2 - 1),
            );
            let (wy, wx) = (sy - lo_y as f64, sx - lo_x as f64);
            let h = heights
                .partition_point(|v| *v <= dem[(r, c)])
                .saturating_sub(1);
            let hi_h = (h + 1).min(heights.len() - 1);
            let wh = if h == hi_h {
                0.0
            } else {
                (dem[(r, c)] - heights[h]) / (heights[hi_h] - heights[h])
            };
            let bilinear = |height: usize| {
                (1.0 - wy)
                    * ((1.0 - wx) * source[(height - h0, lo_y, lo_x)]
                        + wx * source[(height - h0, lo_y, hi_x)])
                    + wy * ((1.0 - wx) * source[(height - h0, hi_y, lo_x)]
                        + wx * source[(height - h0, hi_y, hi_x)])
            };
            (1.0 - wh) * bilinear(h) + wh * bilinear(hi_h)
        })
    };
    Ok(Some(LosLayers {
        east: interpolate(&east),
        north: interpolate(&north),
        geo: target_geo,
    }))
}

type NisarWindow = (std::ops::Range<usize>, std::ops::Range<usize>, (f64, f64));

fn nisar_xy_window(
    x: &[f64],
    y: &[f64],
    geo: GeoInfo,
    target_geo: GeoInfo,
    shape: (usize, usize),
) -> Result<NisarWindow> {
    use crate::error::IoError;
    let sg = geo.geotransform;
    let tg = target_geo.geotransform;
    if sg[1] <= 0.0
        || sg[5] >= 0.0
        || tg[1] <= 0.0
        || tg[5] >= 0.0
        || tg[2] != 0.0
        || tg[4] != 0.0
        || x.windows(2)
            .any(|w| !w[0].is_finite() || ((w[1] - w[0]) - sg[1]).abs() > 1e-6)
        || y.windows(2)
            .any(|w| !w[0].is_finite() || ((w[1] - w[0]) - sg[5]).abs() > 1e-6)
    {
        return Err(IoError::Geo(
            "NISAR LOS coordinates must form a regular north-up grid".into(),
        ));
    }
    let col0 = (tg[0] + 0.5 * tg[1] - x[0]) / sg[1];
    let row0 = (tg[3] + 0.5 * tg[5] - y[0]) / sg[5];
    let col1 = col0 + shape.1.saturating_sub(1) as f64 * tg[1] / sg[1];
    let row1 = row0 + shape.0.saturating_sub(1) as f64 * tg[5] / sg[5];
    if [col0, row0, col1, row1].iter().any(|v| !v.is_finite())
        || col0 < 0.0
        || row0 < 0.0
        || col1 > (x.len() - 1) as f64
        || row1 > (y.len() - 1) as f64
    {
        return Err(IoError::Geo(
            "NISAR LOS does not cover target pixel centers".into(),
        ));
    }
    let (c0, r0) = (col0.floor() as usize, row0.floor() as usize);
    let (c1, r1) = (
        (col1.ceil() as usize + 1).min(x.len()),
        (row1.ceil() as usize + 1).min(y.len()),
    );
    Ok((r0..r1, c0..c1, (row0 - r0 as f64, col0 - c0 as f64)))
}

fn read_ellipsoidal_dem(path: &Path, geo: GeoInfo, shape: (usize, usize)) -> Result<Array2<f64>> {
    use crate::error::IoError;
    let dataset = gdal::Dataset::open(path)?;
    let sg = dataset.geo_transform()?;
    let tg = geo.geotransform;
    if dataset.spatial_ref()?.auth_code()? != geo.epsg as i32
        || sg[2] != 0.0
        || sg[4] != 0.0
        || sg[1] <= 0.0
        || sg[5] >= 0.0
    {
        return Err(IoError::Geo(
            "ellipsoidal DEM must share the target CRS and north-up grid".into(),
        ));
    }
    let window = [
        (tg[0] - sg[0]) / sg[1],
        (tg[3] - sg[3]) / sg[5],
        shape.1 as f64 * tg[1] / sg[1],
        shape.0 as f64 * tg[5] / sg[5],
    ];
    if window
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0 || (*v - v.round()).abs() > 1e-6)
        || window[2] < 1.0
        || window[3] < 1.0
    {
        return Err(IoError::Geo(
            "ellipsoidal DEM target boundaries must align to source pixels".into(),
        ));
    }
    let offset = (window[0] as isize, window[1] as isize);
    let size = (window[2] as usize, window[3] as usize);
    if window[0] + window[2] > dataset.raster_size().0 as f64
        || window[1] + window[3] > dataset.raster_size().1 as f64
    {
        return Err(IoError::Geo("ellipsoidal DEM does not cover target".into()));
    }
    let band = dataset.rasterband(1)?;
    let nodata = band.no_data_value();
    // Check source support before interpolation; GDAL otherwise interpolates around nodata.
    let source = band.read_as::<f64>(offset, size, size, None)?;
    if source
        .data()
        .iter()
        .any(|v| !v.is_finite() || nodata == Some(*v))
    {
        return Err(IoError::Geo(
            "ellipsoidal DEM contains nodata in target".into(),
        ));
    }
    let values = band.read_as::<f64>(
        offset,
        size,
        (shape.1, shape.0),
        Some(gdal::raster::ResampleAlg::Bilinear),
    )?;
    Array2::from_shape_vec(shape, values.data().to_vec()).map_err(|e| IoError::Shape(e.to_string()))
}

#[cfg(test)]
mod nisar_tests {
    use super::*;
    #[test]
    fn nisar_cube_uses_ellipsoidal_height_and_rejects_missing_dem() {
        let _guard = crate::test_hdf5_lock::guard();
        let dir = std::env::temp_dir().join("dolphin_nisar_los_height_contract");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gslc.h5");
        let dem = dir.join("ellipsoidal.tif");
        let group = "/science/LSAR/GSLC/metadata/radarGrid";
        let geo = GeoInfo {
            epsg: 32611,
            geotransform: [500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0],
        };
        {
            let f = hdf5::File::create(&path).unwrap();
            let g = f.create_group(group).unwrap();
            g.new_dataset_builder()
                .with_data(&[500_015.0, 500_045.0, 500_075.0])
                .create("xCoordinates")
                .unwrap();
            g.new_dataset_builder()
                .with_data(&[4_199_985.0, 4_199_955.0, 4_199_925.0])
                .create("yCoordinates")
                .unwrap();
            g.new_dataset_builder()
                .with_data(&[0.0_f64, 1000.0])
                .create("heightAboveEllipsoid")
                .unwrap();
            g.new_dataset::<i64>()
                .create("projection")
                .unwrap()
                .write_scalar(&32611)
                .unwrap();
            let east = ndarray::Array3::from_shape_fn((2, 3, 3), |(h, r, c)| {
                0.3_f64 + h as f64 * 0.1 + r as f64 * 0.01 + c as f64 * 0.02
            });
            g.new_dataset_builder()
                .with_data(east.view())
                .create("losUnitVectorX")
                .unwrap();
            g.new_dataset_builder()
                .with_data(ndarray::Array3::from_elem((2, 3, 3), 0.1_f64).view())
                .create("losUnitVectorY")
                .unwrap();
        }
        crate::write_raster(
            &dem,
            Array2::from_elem((3, 3), 250.0_f64).view(),
            geo.geotransform,
            Some(32611),
            None,
        )
        .unwrap();
        let layers = read_nisar_los_layers_for_grid(&path, group, geo, (3, 3), Some(&dem))
            .unwrap()
            .unwrap();
        assert!((layers.east[(1, 1)] - 0.355).abs() < 1e-12);
        let shifted = GeoInfo {
            epsg: 32611,
            geotransform: [500_015.0, 30.0, 0.0, 4_199_985.0, 0.0, -30.0],
        };
        let shifted_dem = dir.join("shifted-dem.tif");
        crate::write_raster(
            &shifted_dem,
            Array2::from_elem((2, 2), 250.0_f64).view(),
            shifted.geotransform,
            Some(32611),
            None,
        )
        .unwrap();
        let interpolated =
            read_nisar_los_layers_for_grid(&path, group, shifted, (2, 2), Some(&shifted_dem))
                .unwrap()
                .unwrap();
        assert!((interpolated.east[(0, 0)] - 0.34).abs() < 1e-12);
        assert!(read_nisar_los_layers_for_grid(&path, group, geo, (3, 3), None).is_err());
        crate::write_raster(
            &dem,
            Array2::from_elem((3, 3), 1250.0_f64).view(),
            geo.geotransform,
            Some(32611),
            None,
        )
        .unwrap();
        assert!(read_nisar_los_layers_for_grid(&path, group, geo, (3, 3), Some(&dem)).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
