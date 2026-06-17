//! GeoTIFF block read/write via GDAL (port of the GeoTIFF paths in `io/`).
//!
//! Real-typed rasters (f32 quality, u8 PS, etc.) round-trip through GDAL with
//! geotransform + CRS preserved. Outputs are **Cloud-Optimized GeoTIFFs** (COG
//! driver: internally tiled, DEFLATE-compressed, with overviews) so eo can serve
//! them directly. (GDAL complex types aren't exposed by the `gdal` crate;
//! complex SLCs are persisted via HDF5 or 2-band f32 — see STATUS.md.)

use std::path::Path;

use gdal::raster::{Buffer, GdalType, RasterCreationOptions};
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DriverManager};
use ndarray::{Array2, ArrayView2};

use crate::error::{IoError, Result};

/// A raster with its georeferencing.
pub struct RasterData<T> {
    /// Pixel values, `(rows, cols)`.
    pub data: Array2<T>,
    /// GDAL affine geotransform `[origin_x, dx, 0, origin_y, 0, dy]`.
    pub geotransform: [f64; 6],
    /// EPSG code of the CRS, if the raster carried one.
    pub epsg: Option<u32>,
}

/// Write a single-band Cloud-Optimized GeoTIFF with the given geotransform, CRS,
/// and nodata. The data is staged in an in-memory dataset, then copied through
/// the GDAL **COG** driver (tiled, DEFLATE-compressed, overviews auto-built).
pub fn write_raster<T: GdalType + Copy>(
    path: &Path,
    data: ArrayView2<T>,
    geotransform: [f64; 6],
    epsg: Option<u32>,
    nodata: Option<f64>,
) -> Result<()> {
    let (rows, cols) = data.dim();
    let mem = DriverManager::get_driver_by_name("MEM")?;
    let mut src = mem.create_with_band_type::<T, _>("", cols, rows, 1)?;
    src.set_geo_transform(&geotransform)?;
    if let Some(code) = epsg {
        src.set_spatial_ref(&SpatialRef::from_epsg(code)?)?;
    }
    {
        let mut band = src.rasterband(1)?;
        if let Some(nd) = nodata {
            band.set_no_data_value(Some(nd))?;
        }
        let mut buffer = Buffer::new((cols, rows), data.iter().copied().collect());
        band.write((0, 0), (cols, rows), &mut buffer)?;
    }
    let cog = DriverManager::get_driver_by_name("COG")?;
    let options =
        RasterCreationOptions::from_iter(["COMPRESS=DEFLATE", "BLOCKSIZE=256", "OVERVIEWS=AUTO"]);
    src.create_copy(&cog, path, &options)?;
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
