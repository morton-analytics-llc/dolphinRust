//! Multi-burst frame stitching.
//!
//! An OPERA frame is tiled by several bursts (e.g. `T064-135518-IW1/2/3`). Each
//! burst is phase-linked independently, then the per-date linked phase and
//! quality layers are mosaicked onto one frame grid before unwrapping (so phase
//! is continuous across burst seams). dolphin does this with `gdal_merge`
//! (last-on-top in overlaps); because a frame's bursts share pixel posting and
//! CRS, an integer-offset paste onto the union grid is exact — no resampling.
//! Bursts with differing posting/CRS are rejected (reprojection is deferred).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use dolphin_io::GeoInfo;
use ndarray::{Array2, Array3};

/// Pixel posting must match this closely (in CRS units) to stitch without resampling.
const POSTING_TOL: f64 = 1e-6;

/// Group CSLC file indices by burst id parsed from each filename, preserving
/// input order within a group. Files with no recognizable burst id fall into a
/// single `"single"` group, so single-burst stacks take the identity path.
///
/// The burst id is the OPERA token of the form `T###-######-IW#` (e.g.
/// `T064-135518-IW2`); matched without a regex by scanning `_`-delimited tokens.
#[must_use]
pub fn group_by_burst(files: &[PathBuf]) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, f) in files.iter().enumerate() {
        let id = burst_id(f).unwrap_or_else(|| "single".to_string());
        groups.entry(id).or_default().push(i);
    }
    groups
}

/// Extract the `T###-######-IW#` burst id from a filename, if present.
fn burst_id(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    name.split('_')
        .find(|t| t.starts_with('T') && t.contains("-IW"))
        .map(str::to_string)
}

/// One burst's grid footprint on the output CRS.
#[derive(Debug, Clone, Copy)]
pub struct BurstGeo {
    /// Georeferencing (EPSG + geotransform) of the burst grid.
    pub geo: GeoInfo,
    /// Burst row count.
    pub rows: usize,
    /// Burst column count.
    pub cols: usize,
}

/// The stitched frame grid covering the union of all burst footprints.
#[derive(Debug, Clone, Copy)]
pub struct FrameGrid {
    /// Frame georeferencing (origin at the upper-left of the union).
    pub geo: GeoInfo,
    /// Frame row count.
    pub rows: usize,
    /// Frame column count.
    pub cols: usize,
}

/// Compute the union frame grid from the burst footprints.
///
/// # Errors
/// Returns `Err` if the bursts disagree on pixel posting or CRS (would need
/// resampling/reprojection, which is deferred).
pub fn frame_grid(bursts: &[BurstGeo]) -> Result<FrameGrid> {
    let first = bursts.first().ok_or_else(|| anyhow::anyhow!("no bursts"))?;
    let dx = first.geo.geotransform[1];
    let dy = first.geo.geotransform[5];
    let epsg = first.geo.epsg;
    for b in bursts {
        ensure!(
            (b.geo.geotransform[1] - dx).abs() < POSTING_TOL
                && (b.geo.geotransform[5] - dy).abs() < POSTING_TOL
                && b.geo.epsg == epsg,
            "bursts differ in posting/CRS; reprojection is not supported in v1"
        );
        let col_offset = (b.geo.geotransform[0] - first.geo.geotransform[0]) / dx;
        let row_offset = (first.geo.geotransform[3] - b.geo.geotransform[3]) / -dy;
        ensure!(
            (col_offset - col_offset.round()).abs() < POSTING_TOL
                && (row_offset - row_offset.round()).abs() < POSTING_TOL,
            "burst origins are not aligned to the common output pixel grid"
        );
    }
    let xmin = reduce(bursts, f64::min, |b| b.geo.geotransform[0]);
    let ymax = reduce(bursts, f64::max, |b| b.geo.geotransform[3]);
    let xmax = reduce(bursts, f64::max, |b| {
        b.geo.geotransform[0] + b.cols as f64 * dx
    });
    let ymin = reduce(bursts, f64::min, |b| {
        b.geo.geotransform[3] + b.rows as f64 * dy
    });
    Ok(FrameGrid {
        geo: GeoInfo {
            epsg,
            geotransform: [xmin, dx, 0.0, ymax, 0.0, dy],
        },
        rows: ((ymax - ymin) / -dy).round() as usize,
        cols: ((xmax - xmin) / dx).round() as usize,
    })
}

/// Upper-left `(row, col)` offset of a burst within the frame grid.
#[must_use]
pub fn burst_offset(frame: &FrameGrid, burst: &BurstGeo) -> (usize, usize) {
    let dx = frame.geo.geotransform[1];
    let dy = frame.geo.geotransform[5];
    let col = ((burst.geo.geotransform[0] - frame.geo.geotransform[0]) / dx).round() as usize;
    let row = ((frame.geo.geotransform[3] - burst.geo.geotransform[3]) / -dy).round() as usize;
    (row, col)
}

/// Paste a burst's 2-D layer onto `frame` at `(row_off, col_off)` (last-on-top).
pub fn paste2<T: Clone>(frame: &mut Array2<T>, burst: &Array2<T>, offset: (usize, usize)) {
    let (ro, co) = offset;
    let (br, bc) = burst.dim();
    frame
        .slice_mut(ndarray::s![ro..ro + br, co..co + bc])
        .assign(burst);
}

/// Paste a burst's 3-D cube (band, row, col) onto `frame` at `(row_off, col_off)`.
pub fn paste3<T: Clone>(frame: &mut Array3<T>, burst: &Array3<T>, offset: (usize, usize)) {
    let (ro, co) = offset;
    let (_, br, bc) = burst.dim();
    frame
        .slice_mut(ndarray::s![.., ro..ro + br, co..co + bc])
        .assign(burst);
}

/// min/max reduction over a burst-derived scalar.
fn reduce(bursts: &[BurstGeo], op: fn(f64, f64) -> f64, key: impl Fn(&BurstGeo) -> f64) -> f64 {
    bursts.iter().map(&key).fold(key(&bursts[0]), op)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo(ox: f64, oy: f64, rows: usize, cols: usize) -> BurstGeo {
        BurstGeo {
            geo: GeoInfo {
                epsg: 32611,
                geotransform: [ox, 30.0, 0.0, oy, 0.0, -30.0],
            },
            rows,
            cols,
        }
    }

    #[test]
    fn groups_by_opera_burst_id() {
        let files = vec![
            PathBuf::from("OPERA_L2_CSLC-S1_T064-135518-IW1_20221119T0_x.h5"),
            PathBuf::from("OPERA_L2_CSLC-S1_T064-135518-IW2_20221119T0_x.h5"),
            PathBuf::from("OPERA_L2_CSLC-S1_T064-135518-IW1_20221201T0_x.h5"),
        ];
        let g = group_by_burst(&files);
        assert_eq!(g.len(), 2);
        assert_eq!(g["T064-135518-IW1"], vec![0, 2]);
        assert_eq!(g["T064-135518-IW2"], vec![1]);
    }

    #[test]
    fn undated_names_collapse_to_single_group() {
        let files = vec![
            PathBuf::from("cslc_20221119.h5"),
            PathBuf::from("cslc_20221201.h5"),
        ];
        let g = group_by_burst(&files);
        assert_eq!(g.len(), 1);
        assert_eq!(g["single"], vec![0, 1]);
    }

    #[test]
    fn frame_grid_unions_two_adjacent_bursts() {
        // burst B sits to the right of and below A, overlapping by 5 px.
        let a = geo(1000.0, 2000.0, 20, 30); // x: 1000..1900, y: 1400..2000
        let b = geo(1000.0 + 25.0 * 30.0, 2000.0 - 15.0 * 30.0, 20, 30); // shifted (25,15) px
        let frame = frame_grid(&[a, b]).unwrap();
        // union spans cols 0..(25+30)=55, rows 0..(15+20)=35
        assert_eq!(frame.cols, 55);
        assert_eq!(frame.rows, 35);
        assert_eq!(burst_offset(&frame, &a), (0, 0));
        assert_eq!(burst_offset(&frame, &b), (15, 25));
    }

    #[test]
    fn rejects_mismatched_posting() {
        let a = geo(0.0, 0.0, 10, 10);
        let mut b = geo(300.0, 0.0, 10, 10);
        b.geo.geotransform[1] = 20.0; // different dx
        assert!(frame_grid(&[a, b]).is_err());
    }

    #[test]
    fn rejects_subpixel_misaligned_burst_origin() {
        let a = geo(0.0, 0.0, 10, 10);
        let b = geo(315.0, 0.0, 10, 10);
        let error = frame_grid(&[a, b]).unwrap_err();
        assert!(error.to_string().contains("origins are not aligned"));
    }

    #[test]
    fn paste_places_burst_block() {
        let mut frame = Array2::<f64>::zeros((4, 4));
        let burst = Array2::from_shape_fn((2, 2), |(i, j)| (i * 2 + j + 1) as f64);
        paste2(&mut frame, &burst, (1, 1));
        assert_eq!(frame[(1, 1)], 1.0);
        assert_eq!(frame[(2, 2)], 4.0);
        assert_eq!(frame[(0, 0)], 0.0);
    }
}
