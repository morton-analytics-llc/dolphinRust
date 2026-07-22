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

/// Read a target grid from a larger raster with identical CRS, posting, and
/// pixel alignment. This is the bounded-workflow contract for binary masks: no
/// interpolation is permitted, and any partial coverage is an explicit error.
///
/// # Errors
/// Returns [`IoError::Geo`] for CRS/posting/alignment/coverage mismatches and
/// propagates GDAL read failures.
pub fn read_aligned_raster_window<T>(
    path: &Path,
    target_gt: [f64; 6],
    target_epsg: u32,
    target_shape: (usize, usize),
) -> Result<Array2<T>>
where
    T: Copy + Default + GdalType + Into<f64>,
{
    const TOLERANCE: f64 = 1e-6;
    let ds = Dataset::open(path)?;
    let source_gt = ds.geo_transform()?;
    let source_epsg = ds
        .spatial_ref()
        .ok()
        .and_then(|reference| reference.auth_code().ok())
        .map(|code| code as u32);
    if source_epsg != Some(target_epsg) {
        return Err(IoError::Geo(format!(
            "aligned raster EPSG {:?} differs from target EPSG {target_epsg}",
            source_epsg
        )));
    }
    if (source_gt[1] - target_gt[1]).abs() > TOLERANCE
        || (source_gt[5] - target_gt[5]).abs() > TOLERANCE
        || source_gt[2].abs() > TOLERANCE
        || source_gt[4].abs() > TOLERANCE
        || target_gt[2].abs() > TOLERANCE
        || target_gt[4].abs() > TOLERANCE
    {
        return Err(IoError::Geo(
            "aligned raster posting/rotation differs from target grid".into(),
        ));
    }
    let col = (target_gt[0] - source_gt[0]) / source_gt[1];
    let row = (source_gt[3] - target_gt[3]) / -source_gt[5];
    if (col - col.round()).abs() > TOLERANCE || (row - row.round()).abs() > TOLERANCE {
        return Err(IoError::Geo(
            "aligned raster origin has a subpixel target offset".into(),
        ));
    }
    if row < 0.0 || col < 0.0 {
        return Err(IoError::Geo(
            "aligned raster does not cover the target origin".into(),
        ));
    }
    let block = BlockIndices {
        row_start: row.round() as usize,
        row_stop: row.round() as usize + target_shape.0,
        col_start: col.round() as usize,
        col_stop: col.round() as usize + target_shape.1,
    };
    let (source_cols, source_rows) = ds.raster_size();
    if block.row_stop > source_rows || block.col_stop > source_cols {
        return Err(IoError::Geo(
            "aligned raster does not fully cover the target grid".into(),
        ));
    }
    let band = ds.rasterband(1)?;
    let buffer = band.read_as::<T>(
        (block.col_start as isize, block.row_start as isize),
        (block.width(), block.height()),
        (block.width(), block.height()),
        None,
    )?;
    let ((width, height), mut values) = buffer.into_shape_and_vec();

    // GDAL exposes one validity mask regardless of whether validity comes from
    // explicit .msk data, an alpha band, or a mask synthesized from nodata.
    // Apply it even when the stored mask value itself is non-zero (notably a
    // common byte-mask nodata value of 255). Keep the explicit nodata check as
    // a defensive contract for drivers whose mask flags are incomplete.
    let mask_band = band.open_mask_band()?;
    let mask = mask_band.read_as::<u8>(
        (block.col_start as isize, block.row_start as isize),
        (block.width(), block.height()),
        (block.width(), block.height()),
        None,
    )?;
    let nodata = band.no_data_value();
    for (value, valid) in values.iter_mut().zip(mask.data()) {
        let is_nodata = nodata.is_some_and(|sentinel| (*value).into() == sentinel);
        if *valid == 0 || is_nodata {
            *value = T::default();
        }
    }
    Array2::from_shape_vec((height, width), values)
        .map_err(|error| IoError::Shape(error.to_string()))
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

    #[test]
    fn aligned_mask_window_matches_target_grid_without_resampling() {
        let path = std::env::temp_dir().join("dolphin_aligned_mask_window.tif");
        let _ = std::fs::remove_file(&path);
        let full = Array2::from_shape_fn((12, 16), |(r, c)| ((r + c) % 2) as u8);
        let source_gt = [1_000.0, 30.0, 0.0, 2_000.0, 0.0, -30.0];
        write_raster(&path, full.view(), source_gt, Some(32611), Some(0.0)).unwrap();
        let target_gt = [1_120.0, 30.0, 0.0, 1_940.0, 0.0, -30.0];
        let target = read_aligned_raster_window::<u8>(&path, target_gt, 32611, (5, 7)).unwrap();
        assert_eq!(target, full.slice(s![2..7, 4..11]));
        let error = read_aligned_raster_window::<u8>(&path, target_gt, 32610, (5, 7)).unwrap_err();
        assert!(matches!(error, IoError::Geo(_)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn aligned_mask_rejects_partial_coverage_and_missing_crs() {
        let partial = std::env::temp_dir().join("dolphin_partial_mask_window.tif");
        let missing_crs = std::env::temp_dir().join("dolphin_missing_crs_mask.tif");
        let _ = std::fs::remove_file(&partial);
        let _ = std::fs::remove_file(&missing_crs);
        let data = Array2::from_elem((4, 4), 1_u8);
        let gt = [1_000.0, 30.0, 0.0, 2_000.0, 0.0, -30.0];
        write_raster(&partial, data.view(), gt, Some(32611), None).unwrap();
        write_raster(&missing_crs, data.view(), gt, None, None).unwrap();

        let partial_error =
            read_aligned_raster_window::<u8>(&partial, gt, 32611, (5, 4)).unwrap_err();
        assert!(matches!(partial_error, IoError::Geo(message) if message.contains("fully cover")));
        let crs_error =
            read_aligned_raster_window::<u8>(&missing_crs, gt, 32611, (4, 4)).unwrap_err();
        assert!(matches!(crs_error, IoError::Geo(message) if message.contains("EPSG None")));

        let _ = std::fs::remove_file(&partial);
        let _ = std::fs::remove_file(&missing_crs);
    }

    #[test]
    fn aligned_mask_zeroes_nodata_and_explicitly_masked_pixels() {
        let path = std::env::temp_dir().join("dolphin_mask_validity_window.tif");
        let _ = std::fs::remove_file(&path);
        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut dataset = driver
            .create_with_band_type::<u8, _>(&path, 4, 3, 1)
            .unwrap();
        let gt = [1_000.0, 30.0, 0.0, 2_000.0, 0.0, -30.0];
        dataset.set_geo_transform(&gt).unwrap();
        dataset
            .set_spatial_ref(&SpatialRef::from_epsg(32611).unwrap())
            .unwrap();
        {
            let mut band = dataset.rasterband(1).unwrap();
            band.set_no_data_value(Some(255.0)).unwrap();
            let mut values = Buffer::new((4, 3), vec![1, 1, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
            band.write((0, 0), (4, 3), &mut values).unwrap();
            band.create_mask_band(false).unwrap();
            let mut mask_band = band.open_mask_band().unwrap();
            let mut validity = Buffer::new(
                (4, 3),
                vec![255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255],
            );
            mask_band.write((0, 0), (4, 3), &mut validity).unwrap();
        }
        dataset.flush_cache().unwrap();
        drop(dataset);

        let result = read_aligned_raster_window::<u8>(&path, gt, 32611, (3, 4)).unwrap();
        assert_eq!(result[(0, 2)], 0, "255 nodata must be invalid");
        assert_eq!(result[(1, 1)], 0, "GDAL mask-band zero must be invalid");
        assert_eq!(result[(2, 3)], 1, "valid nonzero mask data must survive");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("tif.msk"));
    }
}
