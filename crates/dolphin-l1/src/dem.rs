//! Copernicus GLO-30 DEM, referenced to the WGS84 ellipsoid.
//!
//! Tiles come from the public `copernicus-dem-30m` bucket as COGs read with
//! GDAL range requests (`/vsicurl/`); ocean tiles do not exist and read as sea
//! level. GLO-30 heights are EGM2008 orthometric, so the EGM2008 undulation
//! (PROJ CDN grid `us_nga_egm08_25.tif`, 2.5′) is added to give ellipsoidal
//! heights, the reference OPERA's DEM uses. A 1 m height error moves a pixel
//! ≈ 1.2 m in ground range at Sentinel-1 incidence, so the geoid is not
//! optional at the 0.1-pixel registration target.

use std::path::Path;

use gdal::Dataset;
use ndarray::Array2;

use crate::error::{product, L1Error, Result};

const GLO30_URL: &str = "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com";
const EGM2008_URL: &str = "/vsicurl/https://cdn.proj.org/us_nga_egm08_25.tif";

/// Geographic bounds in degrees.
#[derive(Debug, Clone, Copy)]
pub struct LonLatBounds {
    /// West edge.
    pub west: f64,
    /// South edge.
    pub south: f64,
    /// East edge.
    pub east: f64,
    /// North edge.
    pub north: f64,
}

impl LonLatBounds {
    /// Bounds of a set of `(lon, lat)` points grown by `margin` degrees.
    pub fn around(points: impl IntoIterator<Item = (f64, f64)>, margin: f64) -> Result<Self> {
        let mut b = Self {
            west: f64::INFINITY,
            south: f64::INFINITY,
            east: f64::NEG_INFINITY,
            north: f64::NEG_INFINITY,
        };
        for (lon, lat) in points {
            b.west = b.west.min(lon);
            b.east = b.east.max(lon);
            b.south = b.south.min(lat);
            b.north = b.north.max(lat);
        }
        if !b.west.is_finite() {
            return Err(L1Error::Product(
                "DEM bounds from an empty point set".into(),
            ));
        }
        Ok(Self {
            west: b.west - margin,
            south: b.south - margin,
            east: b.east + margin,
            north: b.north + margin,
        })
    }
}

/// A lon/lat height grid (ellipsoidal metres), node-registered.
#[derive(Debug, Clone)]
pub struct Dem {
    /// Longitude of node column 0 (deg).
    pub lon0: f64,
    /// Latitude of node row 0 (deg).
    pub lat0: f64,
    /// Longitude step (deg, > 0).
    pub dlon: f64,
    /// Latitude step (deg, < 0: rows run north to south).
    pub dlat: f64,
    /// Heights above the WGS84 ellipsoid (m).
    pub heights: Array2<f32>,
}

impl Dem {
    /// Mosaic the GLO-30 tiles covering `bounds` and convert to ellipsoidal
    /// heights.
    pub fn glo30(bounds: LonLatBounds) -> Result<Self> {
        let tiles: Vec<Dataset> = tile_names(bounds)
            .into_iter()
            .filter_map(|name| Dataset::open(format!("{GLO30_URL}/{name}/{name}.tif")).ok())
            .collect();
        let mut dem = match tiles.as_slice() {
            [] => Self::sea_level(bounds),
            _ => {
                let mosaic = gdal::programs::raster::build_vrt(None, &tiles, None)?;
                Self::read_window(&mosaic, bounds)?
            }
        };
        dem.add_geoid()?;
        Ok(dem)
    }

    /// Read `bounds` (plus a 2-node border for interpolation) from a geographic
    /// raster; areas outside it read as 0.
    fn read_window(dataset: &Dataset, bounds: LonLatBounds) -> Result<Self> {
        let gt = dataset.geo_transform()?;
        let (cols, rows) = dataset.raster_size();
        let col_of = |lon: f64| ((lon - gt[0]) / gt[1] - 0.5).floor() as isize;
        let row_of = |lat: f64| ((lat - gt[3]) / gt[5] - 0.5).floor() as isize;
        let c0 = (col_of(bounds.west) - 2).max(0) as usize;
        let c1 = ((col_of(bounds.east) + 3).max(0) as usize).min(cols);
        let r0 = (row_of(bounds.north) - 2).max(0) as usize;
        let r1 = ((row_of(bounds.south) + 3).max(0) as usize).min(rows);
        if c1 <= c0 || r1 <= r0 {
            return Err(L1Error::Product("DEM window is empty".into()));
        }
        let (w, h) = (c1 - c0, r1 - r0);
        let band = dataset.rasterband(1)?;
        let buffer = band.read_as::<f32>((c0 as isize, r0 as isize), (w, h), (w, h), None)?;
        let heights = Array2::from_shape_vec((h, w), buffer.data().to_vec()).map_err(product)?;
        Ok(Self {
            lon0: gt[0] + (c0 as f64 + 0.5) * gt[1],
            lat0: gt[3] + (r0 as f64 + 0.5) * gt[5],
            dlon: gt[1],
            dlat: gt[5],
            heights: heights.mapv(|v| if v.is_finite() && v > -1000.0 { v } else { 0.0 }),
        })
    }

    /// A zero orthometric-height grid at 1″ over `bounds` (all-ocean scenes).
    fn sea_level(bounds: LonLatBounds) -> Self {
        let step = 1.0 / 3600.0;
        let cols = ((bounds.east - bounds.west) / step).ceil() as usize + 5;
        let rows = ((bounds.north - bounds.south) / step).ceil() as usize + 5;
        Self {
            lon0: bounds.west - 2.0 * step,
            lat0: bounds.north + 2.0 * step,
            dlon: step,
            dlat: -step,
            heights: Array2::zeros((rows, cols)),
        }
    }

    fn add_geoid(&mut self) -> Result<()> {
        let (rows, cols) = self.heights.dim();
        let bounds = LonLatBounds {
            west: self.lon0,
            east: self.lon0 + cols as f64 * self.dlon,
            north: self.lat0,
            south: self.lat0 + rows as f64 * self.dlat,
        };
        let geoid = Self::read_window(&Dataset::open(EGM2008_URL)?, bounds)?;
        let (lon0, lat0, dlon, dlat) = (self.lon0, self.lat0, self.dlon, self.dlat);
        for ((row, col), height) in self.heights.indexed_iter_mut() {
            let lon = lon0 + col as f64 * dlon;
            let lat = lat0 + row as f64 * dlat;
            *height += geoid.bilinear(lon, lat)? as f32;
        }
        Ok(())
    }

    fn fractional(&self, lon: f64, lat: f64) -> (f64, f64) {
        ((lon - self.lon0) / self.dlon, (lat - self.lat0) / self.dlat)
    }

    fn bilinear(&self, lon: f64, lat: f64) -> Result<f64> {
        let (x, y) = self.fractional(lon, lat);
        let (i, j) = (y.floor(), x.floor());
        let (fy, fx) = (y - i, x - j);
        let node = |r: f64, c: f64| -> Result<f64> {
            self.heights
                .get((r as usize, c as usize))
                .filter(|_| r >= 0.0 && c >= 0.0)
                .map(|v| f64::from(*v))
                .ok_or_else(|| {
                    L1Error::Geometry(format!("({lon:.5}, {lat:.5}) is outside the grid"))
                })
        };
        Ok(
            (1.0 - fy) * ((1.0 - fx) * node(i, j)? + fx * node(i, j + 1.0)?)
                + fy * ((1.0 - fx) * node(i + 1.0, j)? + fx * node(i + 1.0, j + 1.0)?),
        )
    }

    /// Bicubic (Keys, a = −0.5) height at `(lon, lat)` in degrees.
    pub fn height(&self, lon: f64, lat: f64) -> Result<f64> {
        let (x, y) = self.fractional(lon, lat);
        let (i0, j0) = (y.floor() as isize - 1, x.floor() as isize - 1);
        let (rows, cols) = self.heights.dim();
        if i0 < 0 || j0 < 0 || i0 as usize + 4 > rows || j0 as usize + 4 > cols {
            return Err(L1Error::Geometry(format!(
                "({lon:.5}, {lat:.5}) is outside the DEM"
            )));
        }
        let wy = keys_weights(y - y.floor());
        let wx = keys_weights(x - x.floor());
        let window = self.heights.slice(ndarray::s![i0..i0 + 4, j0..j0 + 4]);
        Ok(window
            .outer_iter()
            .zip(wy)
            .map(|(row, w_row)| {
                w_row
                    * row
                        .iter()
                        .zip(wx)
                        .map(|(v, w)| f64::from(*v) * w)
                        .sum::<f64>()
            })
            .sum())
    }

    /// Write the grid as a GeoTIFF (EPSG:4326), for inspection.
    pub fn write_geotiff(&self, path: &Path) -> Result<()> {
        let (rows, cols) = self.heights.dim();
        let driver = gdal::DriverManager::get_driver_by_name("GTiff")?;
        let mut dataset = driver.create_with_band_type::<f32, _>(path, cols, rows, 1)?;
        dataset.set_geo_transform(&[
            self.lon0 - 0.5 * self.dlon,
            self.dlon,
            0.0,
            self.lat0 - 0.5 * self.dlat,
            0.0,
            self.dlat,
        ])?;
        dataset.set_spatial_ref(&gdal::spatial_ref::SpatialRef::from_epsg(4326)?)?;
        let mut buffer =
            gdal::raster::Buffer::new((cols, rows), self.heights.iter().copied().collect());
        dataset
            .rasterband(1)?
            .write((0, 0), (cols, rows), &mut buffer)?;
        Ok(())
    }
}

/// Keys cubic-convolution weights for the four nodes around fraction `t`.
fn keys_weights(t: f64) -> [f64; 4] {
    let a = -0.5;
    let near = |d: f64| ((a + 2.0) * d - (a + 3.0)) * d * d + 1.0;
    let far = |d: f64| ((a * d - 5.0 * a) * d + 8.0 * a) * d - 4.0 * a;
    [far(1.0 + t), near(t), near(1.0 - t), far(2.0 - t)]
}

/// GLO-30 tile names covering `bounds`.
fn tile_names(bounds: LonLatBounds) -> Vec<String> {
    let lats = (bounds.south.floor() as i32)..=(bounds.north.floor() as i32);
    let lons: Vec<i32> = ((bounds.west.floor() as i32)..=(bounds.east.floor() as i32)).collect();
    lats.flat_map(|lat| {
        lons.iter().map(move |&lon| {
            let ns = if lat >= 0 { 'N' } else { 'S' };
            let ew = if lon >= 0 { 'E' } else { 'W' };
            format!(
                "Copernicus_DSM_COG_10_{ns}{:02}_00_{ew}{:03}_00_DEM",
                lat.abs(),
                lon.abs()
            )
        })
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_names_follow_floor_convention() {
        let names = tile_names(LonLatBounds {
            west: -113.2,
            south: 39.1,
            east: -112.8,
            north: 39.6,
        });
        assert_eq!(
            names,
            vec![
                "Copernicus_DSM_COG_10_N39_00_W114_00_DEM",
                "Copernicus_DSM_COG_10_N39_00_W113_00_DEM"
            ]
        );
        let south = tile_names(LonLatBounds {
            west: 151.1,
            south: -33.9,
            east: 151.2,
            north: -33.8,
        });
        assert_eq!(south, vec!["Copernicus_DSM_COG_10_S34_00_E151_00_DEM"]);
    }

    /// Keys weights sum to one and reproduce a linear ramp exactly.
    #[test]
    fn bicubic_reproduces_planes() {
        let heights =
            Array2::from_shape_fn((8, 8), |(r, c)| (3.0 * r as f64 - 2.0 * c as f64) as f32);
        let dem = Dem {
            lon0: 10.0,
            lat0: 50.0,
            dlon: 0.01,
            dlat: -0.01,
            heights,
        };
        let (lon, lat) = (10.0337, 49.9629);
        let expected = 3.0 * (50.0 - lat) / 0.01 - 2.0 * (lon - 10.0) / 0.01;
        assert!((dem.height(lon, lat).unwrap() - expected).abs() < 1e-4);
        assert!((keys_weights(0.3).iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(dem.height(10.0, 50.0).is_err());
    }
}
