//! GeoTIFF block read/write via GDAL (port of the GeoTIFF paths in `io/`).
//!
//! Real-typed rasters (f32 quality, u8 PS, etc.) round-trip through GDAL with
//! geotransform + CRS preserved. Outputs are **Cloud-Optimized GeoTIFFs** (COG
//! driver: internally tiled, DEFLATE-compressed, with overviews) so eo can serve
//! them directly. (GDAL complex types aren't exposed by the `gdal` crate;
//! complex SLCs are persisted via HDF5 or 2-band f32 — see STATUS.md.)

use std::path::Path;

use dolphin_core::BlockIndices;
use gdal::raster::{Buffer, GdalType, RasterCreationOptions};
use gdal::spatial_ref::{CoordTransform, SpatialRef};
use gdal::{Dataset, DriverManager};
use ndarray::{Array2, ArrayView2};

use crate::error::{IoError, Result};

/// Geographic (lon, lat) in degrees of a grid's centre pixel, transforming the
/// projected geotransform centre through `epsg` → EPSG:4326. Used to sample
/// coarse global atmospheric products (e.g. IONEX TEC maps) at the frame.
///
/// # Errors
/// Returns `Err` if the CRS or the coordinate transform cannot be built.
pub fn grid_centroid_lonlat(
    gt: [f64; 6],
    rows: usize,
    cols: usize,
    epsg: u32,
) -> Result<(f64, f64)> {
    let cx = gt[0] + (cols as f64 / 2.0) * gt[1] + (rows as f64 / 2.0) * gt[2];
    let cy = gt[3] + (cols as f64 / 2.0) * gt[4] + (rows as f64 / 2.0) * gt[5];
    let mut src = SpatialRef::from_epsg(epsg)?;
    let mut dst = SpatialRef::from_epsg(4326)?;
    src.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
    dst.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
    let ct = CoordTransform::new(&src, &dst)?;
    let mut x = [cx];
    let mut y = [cy];
    let mut z = [0.0];
    ct.transform_coords(&mut x, &mut y, &mut z)?;
    Ok((x[0], y[0]))
}

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

/// Read a rectangular `block` window of a single-band GeoTIFF into an
/// `(block.height(), block.width())` array via GDAL's windowed `read_as` — the
/// raster mirror of [`crate::read_cslc_window`], bit-identical to [`read_raster`]
/// sliced to the same rectangle.
pub fn read_raster_window<T: Copy + GdalType>(
    path: &Path,
    block: BlockIndices,
) -> Result<Array2<T>> {
    let ds = Dataset::open(path)?;
    let band = ds.rasterband(1)?;
    let (w, h) = (block.width(), block.height());
    let buffer = band.read_as::<T>(
        (block.col_start as isize, block.row_start as isize),
        (w, h),
        (w, h),
        None,
    )?;
    let ((width, height), values) = buffer.into_shape_and_vec();
    Array2::from_shape_vec((height, width), values).map_err(|e| IoError::Shape(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    /// Contract: a windowed GeoTIFF read equals the full read sliced to the same
    /// rectangle (the raster half of the block-tiled bit-identity invariant).
    #[test]
    fn raster_window_matches_full_read_sliced() {
        let path = std::env::temp_dir().join("dolphin_raster_window_contract.tif");
        let _ = std::fs::remove_file(&path);
        let full = Array2::from_shape_fn((30, 40), |(r, c)| (r * 40 + c) as f32);
        write_raster(
            &path,
            full.view(),
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            None,
            None,
        )
        .unwrap();

        let block = BlockIndices {
            row_start: 5,
            row_stop: 22,
            col_start: 9,
            col_stop: 33,
        };
        let win = read_raster_window::<f32>(&path, block).unwrap();
        let expected = full.slice(s![block.rows(), block.cols()]).to_owned();
        assert_eq!(win, expected);
        let _ = std::fs::remove_file(&path);
    }
}
