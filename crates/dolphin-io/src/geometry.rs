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
