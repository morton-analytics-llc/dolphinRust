//! Output geographic grid: the OPERA burst grid (UTM, 5 m × 10 m posting).
//!
//! OPERA CSLC-S1 places each burst on a fixed grid from the burst database
//! bounding box (`burst_id_map`: EPSG, xmin, ymin, xmax, ymax): x from xmin
//! eastward at 5 m, y from ymax southward at 10 m, coordinates at pixel
//! centres. Using the same grid keeps the in-house CSLC stack interchangeable
//! with the OPERA backtest reference and with the phase-linking path.

use gdal::spatial_ref::{AxisMappingStrategy, CoordTransform, SpatialRef};

use crate::error::{product, L1Error, Result};

/// OPERA CSLC-S1 x posting (m).
pub const OPERA_X_POSTING: f64 = 5.0;
/// OPERA CSLC-S1 y posting (m, rows run south).
pub const OPERA_Y_POSTING: f64 = 10.0;

/// A north-up projected grid with pixel-centre coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoGrid {
    /// EPSG code of the projection.
    pub epsg: u32,
    /// x of the first column's pixel centre.
    pub x0: f64,
    /// y of the first row's pixel centre.
    pub y0: f64,
    /// Column spacing (> 0).
    pub dx: f64,
    /// Row spacing (< 0 for north-up).
    pub dy: f64,
    /// Columns.
    pub cols: usize,
    /// Rows.
    pub rows: usize,
}

impl GeoGrid {
    /// The OPERA grid for a burst-database bounding box.
    pub fn opera_burst(epsg: u32, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Result<Self> {
        let cols = ((xmax - xmin) / OPERA_X_POSTING).round();
        let rows = ((ymax - ymin) / OPERA_Y_POSTING).round();
        if cols < 1.0 || rows < 1.0 {
            return Err(L1Error::Product(format!(
                "burst bbox {xmin},{ymin},{xmax},{ymax} is empty"
            )));
        }
        Ok(Self {
            epsg,
            x0: xmin + 0.5 * OPERA_X_POSTING,
            y0: ymax - 0.5 * OPERA_Y_POSTING,
            dx: OPERA_X_POSTING,
            dy: -OPERA_Y_POSTING,
            cols: cols as usize,
            rows: rows as usize,
        })
    }

    /// A grid from explicit pixel-centre coordinate vectors (uniform spacing).
    pub fn from_coordinates(epsg: u32, x: &[f64], y: &[f64]) -> Result<Self> {
        let spacing = |v: &[f64]| -> Result<f64> {
            match v {
                [a, b, ..] => Ok(b - a),
                _ => Err(L1Error::Product("coordinate vector shorter than 2".into())),
            }
        };
        Ok(Self {
            epsg,
            x0: x[0],
            y0: y[0],
            dx: spacing(x)?,
            dy: spacing(y)?,
            cols: x.len(),
            rows: y.len(),
        })
    }

    /// The sub-grid `[row0, row0 + rows) × [col0, col0 + cols)`.
    pub fn window(&self, row0: usize, col0: usize, rows: usize, cols: usize) -> Result<Self> {
        if row0 + rows > self.rows || col0 + cols > self.cols {
            return Err(L1Error::Product(format!(
                "window {row0}+{rows} × {col0}+{cols} exceeds the {}×{} grid",
                self.rows, self.cols
            )));
        }
        Ok(Self {
            x0: self.x0 + col0 as f64 * self.dx,
            y0: self.y0 + row0 as f64 * self.dy,
            rows,
            cols,
            ..*self
        })
    }

    /// Pixel-centre x coordinates.
    pub fn x_coordinates(&self) -> Vec<f64> {
        (0..self.cols)
            .map(|c| self.x0 + c as f64 * self.dx)
            .collect()
    }

    /// Pixel-centre y coordinates.
    pub fn y_coordinates(&self) -> Vec<f64> {
        (0..self.rows)
            .map(|r| self.y0 + r as f64 * self.dy)
            .collect()
    }
}

/// Projected ↔ geographic (lon, lat degrees) transforms for one EPSG.
pub struct Projection {
    to_lonlat: CoordTransform,
}

impl Projection {
    /// Transforms for `epsg`.
    pub fn new(epsg: u32) -> Result<Self> {
        let mut projected = SpatialRef::from_epsg(epsg)?;
        projected.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
        let mut geographic = SpatialRef::from_epsg(4326)?;
        geographic.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
        Ok(Self {
            to_lonlat: CoordTransform::new(&projected, &geographic)?,
        })
    }

    /// Convert projected `(x, y)` in place to `(lon, lat)` degrees.
    pub fn to_lonlat(&self, x: &mut [f64], y: &mut [f64]) -> Result<()> {
        let mut z = vec![0.0; x.len()];
        self.to_lonlat
            .transform_coords(x, y, &mut z)
            .map_err(product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The OPERA T100-213486-IW2 crop starts at x = 341042.5 (xmin + 2.5) and
    /// y = 4365955 (ymax − 5) with 5 m × 10 m posting.
    #[test]
    fn opera_grid_uses_pixel_centres() {
        let grid =
            GeoGrid::opera_burst(32612, 341_040.0, 4_360_530.0, 370_630.0, 4_365_960.0).unwrap();
        assert_eq!((grid.x0, grid.y0), (341_042.5, 4_365_955.0));
        assert_eq!((grid.cols, grid.rows), (5918, 543));
        assert_eq!(grid.x_coordinates().last().copied(), Some(370_627.5));
        assert_eq!(grid.y_coordinates().last().copied(), Some(4_360_535.0));
        let sub = grid.window(10, 20, 5, 6).unwrap();
        assert_eq!((sub.x0, sub.y0), (341_142.5, 4_365_855.0));
        assert!(grid.window(540, 0, 5, 1).is_err());
    }
}
