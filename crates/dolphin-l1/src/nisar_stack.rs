//! NISAR stack: raw L1 RSLC in, geocoded NISAR-layout HDF5 out.
//!
//! For every date of one track/frame in the window: open the RSLC on ASF
//! (windowed range reads, [`crate::rslc`]), fetch the day's NISAR TEC
//! ancillary, geocode each requested band and polarisation onto the fixed
//! output grid ([`crate::nisar_geocode`]) and write the GSLC-layout file the
//! NISAR phase-linking path reads ([`crate::nisar_h5`]). The DEM is read once
//! per stack; the static layers (LOS, layover/shadow, and the ellipsoidal
//! DEM on the frequency-A grid that the NISAR LOS reader needs) are written
//! once, from the first date.
//!
//! **Grid.** JPL's NISAR GSLC grids sit on a lattice whose pixel edges are
//! multiples of 40 m in the frame's projection (track 5 frame 20: x from
//! 452160, y from 4228560, EPSG 32611), with the x posting set by the range
//! bandwidth (40 MHz: 5 m, 20 MHz: 10 m, 5 MHz: 40 m) and 5 m in y. The
//! requested bounds are snapped outward to that lattice, so the in-house
//! pixels coincide with JPL's and every date of a stack shares one grid.
//! Every date is geocoded independently onto it; the orbit and the timing
//! corrections coregister the dates.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};

use crate::dem::{Dem, LonLatBounds};
use crate::error::{L1Error, Result};
use crate::grid::{GeoGrid, Projection};
use crate::nisar_geocode::{geocode_band, los_cube, GeocodedBand};
use crate::nisar_h5::{write_nisar, NisarProvenance};
use crate::nisar_tec::{tec_corrections, TecCorrections};
use crate::rslc::{AsfClient, GranuleRecord, NisarFrame, Rslc};
use crate::stack::StackOutput;
use crate::static_layers::{static_layers, write_static, RadarGeometry};

/// Pixel-edge lattice of the NISAR GSLC grids (m).
pub const LATTICE: f64 = 40.0;
/// y posting of every NISAR GSLC band (m).
pub const Y_POSTING: f64 = 5.0;
/// DEM margin around the output grid (deg).
const DEM_MARGIN_DEG: f64 = 0.05;
/// TEC coverage required either side of the frame (s).
const TEC_MARGIN_S: i64 = 60;

/// What to process.
#[derive(Debug, Clone)]
pub struct NisarStackRequest {
    /// Track/frame.
    pub frame: NisarFrame,
    /// First acquisition time (inclusive).
    pub start: DateTime<Utc>,
    /// Last acquisition time (exclusive).
    pub end: DateTime<Utc>,
    /// Output CRS (the frame's UTM zone, as JPL's GSLC uses).
    pub epsg: u32,
    /// Output bounds `[xmin, ymin, xmax, ymax]` in `epsg`, snapped outward
    /// to [`LATTICE`].
    pub bounds: [f64; 4],
    /// Bands, e.g. `['A', 'B']`.
    pub frequencies: Vec<char>,
    /// Polarisations, e.g. `["HH"]`.
    pub polarizations: Vec<String>,
    /// Ancillary cache (TEC files).
    pub cache_dir: PathBuf,
    /// Where the HDF5 files go.
    pub output_dir: PathBuf,
    /// Apply the NISAR TEC ionospheric timing corrections.
    pub ionosphere: bool,
}

/// x posting (m) of a band with `range_bandwidth` Hz, as JPL's GSLC posts it.
///
/// # Errors
/// A bandwidth whose GSLC posting has not been checked against a JPL
/// product.
pub fn x_posting(range_bandwidth: f64) -> Result<f64> {
    match (range_bandwidth / 1e6).round() as i64 {
        40 => Ok(5.0),
        20 => Ok(10.0),
        5 => Ok(40.0),
        mhz => Err(L1Error::Product(format!(
            "no checked GSLC posting for a {mhz} MHz band"
        ))),
    }
}

/// The output grid of a band: `bounds` snapped outward to the lattice.
///
/// # Errors
/// Empty bounds or an unsupported bandwidth.
pub fn band_grid(epsg: u32, bounds: [f64; 4], range_bandwidth: f64) -> Result<GeoGrid> {
    let dx = x_posting(range_bandwidth)?;
    let [xmin, ymin, xmax, ymax] = bounds;
    let (x0, x1) = (
        (xmin / LATTICE).floor() * LATTICE,
        (xmax / LATTICE).ceil() * LATTICE,
    );
    let (y0, y1) = (
        (ymin / LATTICE).floor() * LATTICE,
        (ymax / LATTICE).ceil() * LATTICE,
    );
    if x1 <= x0 || y1 <= y0 {
        return Err(L1Error::Product(format!("empty bounds {bounds:?}")));
    }
    Ok(GeoGrid {
        epsg,
        x0: x0 + dx / 2.0,
        y0: y1 - Y_POSTING / 2.0,
        dx,
        dy: -Y_POSTING,
        cols: ((x1 - x0) / dx).round() as usize,
        rows: ((y1 - y0) / Y_POSTING).round() as usize,
    })
}

fn output_stem(frame: NisarFrame, rslc: &Rslc) -> String {
    format!(
        "CSLC-NISAR_{}_{}",
        frame.label(),
        rslc.sensing_start().format("%Y%m%dT%H%M%SZ")
    )
}

/// The output for `record`'s acquisition when one is already written.
fn existing(request: &NisarStackRequest, record: &GranuleRecord) -> Result<Option<PathBuf>> {
    let prefix = format!(
        "CSLC-NISAR_{}_{}",
        request.frame.label(),
        record.start.format("%Y%m%dT")
    );
    Ok(std::fs::read_dir(&request.output_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".h5"))
        }))
}

fn stack_dem(grid: &GeoGrid) -> Result<Dem> {
    let corners = [
        (0, 0),
        (0, grid.cols),
        (grid.rows, 0),
        (grid.rows, grid.cols),
    ];
    let mut x: Vec<f64> = corners
        .iter()
        .map(|&(_, c)| grid.x0 + c as f64 * grid.dx)
        .collect();
    let mut y: Vec<f64> = corners
        .iter()
        .map(|&(r, _)| grid.y0 + r as f64 * grid.dy)
        .collect();
    Projection::new(grid.epsg)?.to_lonlat(&mut x, &mut y)?;
    Dem::glo30(LonLatBounds::around(x.into_iter().zip(y), DEM_MARGIN_DEG)?)
}

/// The ellipsoidal heights of `dem` on `grid` as a GeoTIFF: the DEM the
/// NISAR LOS-cube reader (`nisar_ellipsoidal_dem_file`) requires on the
/// output grid.
fn write_grid_dem(path: &Path, dem: &Dem, grid: &GeoGrid) -> Result<()> {
    let projection = Projection::new(grid.epsg)?;
    let mut heights = Vec::with_capacity(grid.rows * grid.cols);
    for r in 0..grid.rows {
        let mut lon = grid.x_coordinates();
        let mut lat = vec![grid.y0 + r as f64 * grid.dy; grid.cols];
        projection.to_lonlat(&mut lon, &mut lat)?;
        heights.extend(
            lon.iter()
                .zip(&lat)
                .map(|(&lo, &la)| dem.height(lo, la).map_or(f32::NAN, |h| h as f32)),
        );
    }
    let driver = gdal::DriverManager::get_driver_by_name("GTiff")?;
    let mut dataset = driver.create_with_band_type::<f32, _>(path, grid.cols, grid.rows, 1)?;
    dataset.set_geo_transform(&[
        grid.x0 - grid.dx / 2.0,
        grid.dx,
        0.0,
        grid.y0 - grid.dy / 2.0,
        0.0,
        grid.dy,
    ])?;
    dataset.set_spatial_ref(&gdal::spatial_ref::SpatialRef::from_epsg(grid.epsg)?)?;
    let mut buffer = gdal::raster::Buffer::new((grid.cols, grid.rows), heights);
    dataset
        .rasterband(1)?
        .write((0, 0), (grid.cols, grid.rows), &mut buffer)?;
    Ok(())
}

/// The radar grid under a geocoded band, for the static layers.
fn window_geometry(rslc: &Rslc, geocoded: &GeocodedBand, frequency: char) -> Result<RadarGeometry> {
    let band = rslc.band(frequency)?;
    let [row0, col0, rows, cols] = geocoded.radar_window;
    Ok(RadarGeometry {
        sensing_start: rslc.datetime(rslc.line_time(row0 as f64)),
        azimuth_time_interval: rslc.line_spacing,
        starting_range: band.starting_range + col0 as f64 * band.range_spacing,
        range_pixel_spacing: band.range_spacing,
        lines: rows,
        samples: cols,
        valid_lines: (0, rows - 1),
        valid_samples: (0, cols - 1),
        look: rslc.look_side,
    })
}

/// Write the stack's static layers once: `STATIC-NISAR_<frame>.h5`
/// (CSLC-S1-STATIC layout: LOS east/north and layover/shadow on the
/// frequency-A grid) and `DEM-NISAR_<frame>.tif`.
fn write_static_once(
    request: &NisarStackRequest,
    rslc: &Rslc,
    dem: &Dem,
    first: (char, &GeocodedBand),
) -> Result<()> {
    let label = request.frame.label();
    let path = request.output_dir.join(format!("STATIC-NISAR_{label}.h5"));
    if path.is_file() {
        return Ok(());
    }
    let (frequency, geocoded) = first;
    let geometry = window_geometry(rslc, geocoded, frequency)?;
    let layers = static_layers(&geometry, &rslc.orbit, dem, &geocoded.grid)?;
    write_grid_dem(
        &request.output_dir.join(format!("DEM-NISAR_{label}.tif")),
        dem,
        &geocoded.grid,
    )?;
    write_static(&path, &layers, &label)?;
    tracing::info!(file = %path.display(), "wrote static layers");
    Ok(())
}

/// Process one date. Returns the written file.
fn process_date(
    client: &AsfClient,
    request: &NisarStackRequest,
    record: &GranuleRecord,
    dem: &mut Option<Dem>,
) -> Result<StackOutput> {
    let rslc = Rslc::open_remote(client, record)?;
    let tec_file = request
        .ionosphere
        .then(|| {
            let margin = Duration::seconds(TEC_MARGIN_S);
            let tec =
                client.search_tec(rslc.sensing_start() - margin, rslc.sensing_stop() + margin)?;
            client.download(&tec, &request.cache_dir.join("tec"))
        })
        .transpose()?;
    let mut bands: Vec<(char, GeocodedBand)> = Vec::with_capacity(request.frequencies.len());
    for &frequency in &request.frequencies {
        let band = rslc.band(frequency)?;
        let grid = band_grid(request.epsg, request.bounds, band.range_bandwidth)?;
        if dem.is_none() {
            *dem = Some(stack_dem(&grid)?);
        }
        let Some(dem) = dem.as_ref() else {
            unreachable!("DEM initialised above")
        };
        let tec: Option<TecCorrections> = tec_file
            .as_deref()
            .map(|p| tec_corrections(p, &rslc, band))
            .transpose()?;
        let geocoded = geocode_band(
            &rslc,
            band,
            &request.polarizations,
            dem,
            &grid,
            tec.as_ref(),
        )?;
        bands.push((frequency, geocoded));
    }
    let Some(dem) = dem.as_ref() else {
        return Err(L1Error::Product("no frequencies requested".into()));
    };
    let (first_frequency, first) = bands
        .first()
        .ok_or_else(|| L1Error::Product("no frequencies requested".into()))?;
    write_static_once(request, &rslc, dem, (*first_frequency, first))?;
    let cube = los_cube(&rslc, &first.grid)?;
    let stem = output_stem(request.frame, &rslc);
    let path = request.output_dir.join(format!("{stem}.h5"));
    let tec_name = tec_file
        .as_deref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned());
    // Written under a temporary name and renamed, so an interrupted write
    // never leaves a file that a later run would take as current.
    let part = path.with_extension("h5.part");
    write_nisar(
        &part,
        &rslc,
        &bands,
        &request.polarizations,
        &cube,
        &NisarProvenance {
            frame: request.frame,
            granule_id: &stem,
            tec_file: tec_name.as_deref(),
        },
    )?;
    std::fs::rename(&part, &path)?;
    tracing::info!(file = %path.display(), rslc = %rslc.granule_id, "wrote NISAR CSLC");
    Ok(StackOutput {
        sensing_start: rslc.sensing_start(),
        path,
        written: true,
    })
}

/// Process every date of `request`'s frame, oldest first. Dates already
/// written are left as they are.
///
/// # Errors
/// Catalogue, network, geometry or write failures.
pub fn process_nisar_stack(
    client: &AsfClient,
    request: &NisarStackRequest,
) -> Result<Vec<StackOutput>> {
    let records = client.search_rslc(request.frame, request.start, request.end)?;
    if records.is_empty() {
        return Err(L1Error::NotFound(format!(
            "no NISAR RSLC for {} in the window",
            request.frame.label()
        )));
    }
    std::fs::create_dir_all(&request.output_dir)?;
    let mut dem: Option<Dem> = None;
    let mut outputs = Vec::with_capacity(records.len());
    for record in &records {
        if let Some(path) = existing(request, record)? {
            outputs.push(StackOutput {
                sensing_start: record.start,
                path,
                written: false,
            });
            continue;
        }
        outputs.push(process_date(client, request, record, &mut dem)?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JPL's track-5 frame-20 GSLC: frequencyA x starts 452162.5 (5 m),
    /// frequencyB 452180 (40 m), y 4228557.5 (5 m), edges on 40 m.
    #[test]
    fn band_grids_match_the_gslc_lattice() {
        let bounds = [452_170.0, 4_228_500.0, 452_400.0, 4_228_555.0];
        let a = band_grid(32611, bounds, 40e6).unwrap();
        assert_eq!(
            (a.x0, a.y0, a.dx, a.dy),
            (452_162.5, 4_228_557.5, 5.0, -5.0)
        );
        assert_eq!((a.cols, a.rows), (48, 16));
        let b = band_grid(32611, bounds, 5e6).unwrap();
        assert_eq!((b.x0, b.dx, b.cols), (452_180.0, 40.0, 6));
        assert!(band_grid(32611, bounds, 77e6).is_err());
    }
}
