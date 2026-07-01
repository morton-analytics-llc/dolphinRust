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
}
