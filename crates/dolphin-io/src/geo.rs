//! Geotransform + CRS extraction from OPERA CSLC HDF5.
//!
//! OPERA CSLC grids carry 1-D pixel-center coordinate arrays
//! (`<group>/x_coordinates`, `<group>/y_coordinates`) and an EPSG code
//! (`<group>/projection`) alongside the complex grid. The GDAL HDF5 driver
//! returns an identity geotransform for these, so the affine transform is
//! derived from the coordinate spacing directly (dolphin reads the same arrays
//! via `opera_utils`).

use std::path::Path;

use gdal::spatial_ref::{AxisMappingStrategy, CoordTransform, SpatialRef};

use crate::error::{IoError, Result};

/// Georeferencing for a CSLC grid: EPSG code + GDAL affine geotransform
/// `[origin_x, dx, 0, origin_y, 0, dy]`, referenced to the upper-left pixel
/// corner (the coordinate arrays are pixel centers).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoInfo {
    /// EPSG code of the (projected) CRS.
    pub epsg: u32,
    /// GDAL geotransform, upper-left-corner referenced.
    pub geotransform: [f64; 6],
}

/// Transform an axis-aligned `[left, bottom, right, top]` envelope between EPSG
/// CRSs. Edges are densified before taking the destination envelope so a
/// projected curve cannot be clipped by a four-corner approximation.
///
/// # Errors
/// Returns [`IoError::Geo`] for invalid/non-finite bounds and propagates GDAL CRS
/// construction or coordinate-transform failures.
pub fn transform_bounds(
    bounds: (f64, f64, f64, f64),
    source_epsg: u32,
    destination_epsg: u32,
) -> Result<(f64, f64, f64, f64)> {
    let (left, bottom, right, top) = bounds;
    if ![left, bottom, right, top].iter().all(|v| v.is_finite()) || left >= right || bottom >= top {
        return Err(IoError::Geo(
            "bounds must be finite with left < right and bottom < top".into(),
        ));
    }
    if source_epsg == destination_epsg {
        return Ok(bounds);
    }
    let mut source = SpatialRef::from_epsg(source_epsg)?;
    let mut destination = SpatialRef::from_epsg(destination_epsg)?;
    source.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
    destination.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
    let transform = CoordTransform::new(&source, &destination)?;
    // Use GDAL's bounds operation rather than manually transforming corners:
    // it handles curved edges and geographic wrap semantics. GDAL recommends
    // 21 densification points per edge.
    const DENSIFY_POINTS: i32 = 21;
    let transformed = transform.transform_bounds(&[left, bottom, right, top], DENSIFY_POINTS)?;
    if transformed.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(IoError::Geo(
            "bounds transform produced non-finite coordinates".into(),
        ));
    }
    Ok((
        transformed[0],
        transformed[1],
        transformed[2],
        transformed[3],
    ))
}

/// Read the geotransform + EPSG for a CSLC grid from its HDF5 metadata.
///
/// `subdataset` is the complex grid path (e.g. `/data/VV`); the coordinate and
/// projection datasets are read from its parent group.
///
/// # Errors
/// Returns `Err` if the HDF5 read fails or the coordinate/projection datasets
/// are absent or too short to define a pixel spacing.
pub fn read_geotransform(path: &Path, subdataset: &str) -> Result<GeoInfo> {
    let group = parent_group(subdataset);
    let file = hdf5::File::open(path)?;
    let x = file
        .dataset(&format!("{group}/x_coordinates"))?
        .read_raw::<f64>()?;
    let y = file
        .dataset(&format!("{group}/y_coordinates"))?
        .read_raw::<f64>()?;
    if x.len() < 2 || y.len() < 2 {
        return Err(IoError::Geo(
            "coordinate arrays too short to define a spacing".into(),
        ));
    }
    let dx = x[1] - x[0];
    let dy = y[1] - y[0];
    let epsg = read_epsg(&file, &group)?;
    Ok(GeoInfo {
        epsg,
        geotransform: [x[0] - dx / 2.0, dx, 0.0, y[0] - dy / 2.0, 0.0, dy],
    })
}

/// EPSG code from the `<group>/projection` dataset value.
fn read_epsg(file: &hdf5::File, group: &str) -> Result<u32> {
    let raw = file
        .dataset(&format!("{group}/projection"))?
        .read_raw::<i64>()?;
    let code = raw
        .first()
        .ok_or_else(|| IoError::Geo("empty projection dataset".into()))?;
    u32::try_from(*code).map_err(|_| IoError::Geo(format!("invalid EPSG {code}")))
}

/// Parent group of an HDF5 dataset path (`/data/VV` → `/data`).
fn parent_group(subdataset: &str) -> String {
    match subdataset.rsplit_once('/') {
        Some(("", _)) | None => "/".to_string(),
        Some((parent, _)) => parent.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_geotransform_from_opera_layout() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_geo_contract.h5");
        let _ = std::fs::remove_file(&path);
        {
            let f = hdf5::File::create(&path).unwrap();
            let g = f.create_group("data").unwrap();
            g.new_dataset_builder()
                .with_data(&[1000.0_f64, 1030.0, 1060.0]) // dx = 30, centers
                .create("x_coordinates")
                .unwrap();
            g.new_dataset_builder()
                .with_data(&[2000.0_f64, 1970.0, 1940.0]) // dy = -30
                .create("y_coordinates")
                .unwrap();
            g.new_dataset::<i64>()
                .create("projection")
                .unwrap()
                .write_scalar(&32611_i64)
                .unwrap();
        }
        let geo = read_geotransform(&path, "/data/VV").unwrap();
        assert_eq!(geo.epsg, 32611);
        assert!((geo.geotransform[0] - 985.0).abs() < 1e-9); // 1000 - 30/2
        assert!((geo.geotransform[1] - 30.0).abs() < 1e-9);
        assert!((geo.geotransform[3] - 2015.0).abs() < 1e-9); // 2000 + 30/2
        assert!((geo.geotransform[5] + 30.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bounds_transform_round_trip_contains_original() {
        let wgs84 = (-122.51, 37.70, -122.35, 37.84);
        let utm = transform_bounds(wgs84, 4326, 32610).unwrap();
        let round_trip = transform_bounds(utm, 32610, 4326).unwrap();
        assert!(round_trip.0 <= wgs84.0 && round_trip.1 <= wgs84.1);
        assert!(round_trip.2 >= wgs84.2 && round_trip.3 >= wgs84.3);
    }

    #[test]
    fn densified_bounds_cover_high_curvature_and_utm_zone_edge_cases() {
        for (bounds, projected_epsg) in [
            ((-48.0, 78.0, -12.0, 84.0), 3413),
            ((-120.2, 36.0, -116.8, 39.0), 32611),
        ] {
            let projected = transform_bounds(bounds, 4326, projected_epsg).unwrap();
            let round_trip = transform_bounds(projected, projected_epsg, 4326).unwrap();
            assert!(round_trip.0 <= bounds.0 && round_trip.1 <= bounds.1);
            assert!(round_trip.2 >= bounds.2 && round_trip.3 >= bounds.3);
        }
    }

    #[test]
    fn invalid_bounds_are_rejected_before_crs_work() {
        let error = transform_bounds((2.0, 0.0, 1.0, 3.0), 4326, 32610).unwrap_err();
        assert!(matches!(error, IoError::Geo(_)));
    }

    #[test]
    fn invalid_crs_is_reported() {
        let error = transform_bounds((-122.5, 37.7, -122.4, 37.8), 4326, 999_999).unwrap_err();
        assert!(matches!(error, IoError::Gdal(_)));
    }
}
