//! GeoTIFF block read/write via GDAL (port of the GeoTIFF paths in `io/`).
//!
//! Real-typed rasters (f32 quality, u8 PS, etc.) round-trip through GDAL with
//! geotransform + CRS preserved. (GDAL complex types aren't exposed by the
//! `gdal` crate; complex SLCs are persisted via HDF5 or 2-band f32 — see
//! STATUS.md.)

use std::path::Path;

use gdal::raster::{Buffer, GdalType};
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DriverManager};
use ndarray::{Array2, ArrayView2};

use crate::error::{IoError, Result};

/// A raster with its georeferencing.
pub struct RasterData<T> {
    pub data: Array2<T>,
    pub geotransform: [f64; 6],
    pub epsg: Option<u32>,
}

/// Write a single-band GeoTIFF with the given geotransform, CRS, and nodata.
pub fn write_raster<T: GdalType + Copy>(
    path: &Path,
    data: ArrayView2<T>,
    geotransform: [f64; 6],
    epsg: Option<u32>,
    nodata: Option<f64>,
) -> Result<()> {
    let (rows, cols) = data.dim();
    let driver = DriverManager::get_driver_by_name("GTiff")?;
    let mut ds = driver.create_with_band_type::<T, _>(path, cols, rows, 1)?;
    ds.set_geo_transform(&geotransform)?;
    if let Some(code) = epsg {
        ds.set_spatial_ref(&SpatialRef::from_epsg(code)?)?;
    }
    let mut band = ds.rasterband(1)?;
    if let Some(nd) = nodata {
        band.set_no_data_value(Some(nd))?;
    }
    let mut buffer = Buffer::new((cols, rows), data.iter().copied().collect());
    band.write((0, 0), (cols, rows), &mut buffer)?;
    Ok(())
}

/// Read a single-band GeoTIFF into an `(rows, cols)` array plus georeferencing.
pub fn read_raster<T: Copy + GdalType>(path: &Path) -> Result<RasterData<T>> {
    let ds = Dataset::open(path)?;
    let (cols, rows) = ds.raster_size();
    let band = ds.rasterband(1)?;
    let buffer = band.read_as::<T>((0, 0), (cols, rows), (cols, rows), None)?;
    let ((width, height), values) = buffer.into_shape_and_vec();
    let data = Array2::from_shape_vec((height, width), values)
        .map_err(|e| IoError::Shape(e.to_string()))?;
    let geotransform = ds.geo_transform()?;
    let epsg = ds
        .spatial_ref()
        .ok()
        .and_then(|sr| sr.auth_code().ok())
        .map(|c| c as u32);
    Ok(RasterData {
        data,
        geotransform,
        epsg,
    })
}
