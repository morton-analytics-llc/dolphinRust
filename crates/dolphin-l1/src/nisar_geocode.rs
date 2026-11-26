//! Geocode one NISAR RSLC band onto a projected grid, as JPL's GSLC does
//! (ISCE3 0.25 `geocodeSlc`), from the raw RSLC.
//!
//! Per output pixel: grid (x, y) → lon/lat → DEM height → ECEF → zero-Doppler
//! `geo2rdr` on the RSLC orbit → timing corrections → fractional line/sample
//! → 8 × 8 sinc on the native-Doppler-demodulated chip → remodulate →
//! flatten by `exp(+j4πR/λ)` with R the geometric (uncorrected) range.
//!
//! - **Doppler.** NISAR RSLCs are focused on a zero-Doppler grid, but the
//!   azimuth spectrum sits at the native Doppler centroid (937–968 Hz on the
//!   track-5 frame, at a 1520 Hz line rate). The chip rows are demodulated by
//!   `exp(−j2π f_dc (i − i₀) Δt)` around the integer line, interpolated, and
//!   the fractional part `exp(+j2π f_dc ν Δt)` is restored, with `f_dc` the
//!   product's Doppler LUT at the pixel (ISCE3 `interpolate`). Without it a
//!   950 Hz spectrum would sit near the 8-tap kernel's band edge.
//! - **No TOPS, bistatic or troposphere terms.** The RSLC is stripmap; the
//!   focusing already removed the bistatic delay and applied the dry
//!   troposphere (`dry_troposphere_model: tsx` in its run configuration),
//!   which JPL's GSLC records as `dryTroposphericGeolocationCorrectionApplied`.
//! - **Timing corrections**, as in the GSLC: the ionosphere from the NISAR
//!   TEC ancillary ([`crate::nisar_tec`]) in azimuth and range, and the
//!   line-of-sight solid-earth tide (IERS 2010 degree 2) in range.
//! - **Mask** (the GSLC `mask` convention): the 1-based sub-swath of the
//!   interpolation centre when it and the four chip corners (±4 samples) are
//!   valid, 0 when not, 255 where nothing was geocoded. Pixels with mask 0 or
//!   255 are NaN, so the phase-linking path treats them as invalid.

use chrono::{DateTime, Utc};
use dolphin_corrections::solid_earth_tide::{tide_bodies, tide_displacement_enu_at, BodyPosition};
use ndarray::{Array2, Array3};
use num_complex::Complex32;
use rayon::prelude::*;

use crate::dem::Dem;
use crate::error::{L1Error, Result};
use crate::geometry::{dot, enu_basis, geo2rdr, llh_to_ecef, norm, sub};
use crate::grid::{GeoGrid, Projection};
use crate::interp::{SincKernel, SINC_BEFORE, SINC_TAPS};
use crate::nisar_tec::TecCorrections;
use crate::rslc::{Rslc, RslcBand};

/// ISCE3 `SINC_HALF`: the chip half-width used for the sub-swath check.
const CHIP_HALF: i64 = 4;
/// Mask value where nothing was geocoded.
pub const MASK_FILL: u8 = 255;
/// Output pixels per tile side. Each tile is located, windowed in the RSLC
/// and sampled on its own, so memory follows the tile, not the grid (#247).
const TILE: usize = 1024;

/// One band geocoded onto a grid.
pub struct GeocodedBand {
    /// The grid.
    pub grid: GeoGrid,
    /// One layer per requested polarisation.
    pub layers: Vec<Array2<Complex32>>,
    /// GSLC-convention mask.
    pub mask: Array2<u8>,
    /// RSLC window read, `[row0, col0, rows, cols]`.
    pub radar_window: [usize; 4],
}

/// Where one output pixel samples the RSLC.
#[derive(Clone, Copy)]
struct Located {
    /// Fractional line and sample of the corrected position.
    line: f64,
    sample: f64,
    /// Geometric (uncorrected) slant range, for the flattening.
    geometric_range: f64,
}

/// Inputs shared by every pixel.
struct Locator<'a> {
    rslc: &'a Rslc,
    band: &'a RslcBand,
    dem: &'a Dem,
    tec: Option<&'a TecCorrections>,
    tide: [(BodyPosition, f64); 2],
    t_mid: f64,
}

impl Locator<'_> {
    fn locate(&self, lon: f64, lat: f64) -> Option<Located> {
        let height = self.dem.height(lon, lat).ok().filter(|h| h.is_finite())?;
        let (la, lo) = (lat.to_radians(), lon.to_radians());
        let target = llh_to_ecef(la, lo, height);
        let radar = geo2rdr(target, &self.rslc.orbit, self.t_mid).ok()?;
        let look = sub(radar.platform.position, target);
        let tide_enu = tide_displacement_enu_at(&self.tide, lon, lat, height);
        let basis = enu_basis(la, lo);
        let tide_ecef: [f64; 3] =
            std::array::from_fn(|k| (0..3).map(|a| tide_enu[a] * basis[a][k]).sum());
        let tide_range = -dot(tide_ecef, look) / norm(look);
        let range = radar.slant_range;
        let (azimuth_shift, range_shift) = match self.tec {
            Some(tec) => {
                let shift = tec.azimuth.eval(radar.azimuth_time, range);
                let t = radar.azimuth_time + shift;
                (shift, tec.range.eval(t, range))
            }
            None => (0.0, 0.0),
        };
        let time = radar.azimuth_time + azimuth_shift;
        let at = Located {
            line: (time - self.rslc.first_line_time) / self.rslc.line_spacing,
            sample: (range + range_shift + tide_range - self.band.starting_range)
                / self.band.range_spacing,
            geometric_range: range,
        };
        let inside = (0.0..=(self.rslc.lines - 1) as f64).contains(&at.line)
            && (0.0..=(self.band.samples - 1) as f64).contains(&at.sample);
        inside.then_some(at)
    }

    /// GSLC mask value of a located pixel.
    fn mask(&self, at: &Located) -> u8 {
        let (line, sample) = (at.line.floor() as i64, at.sample.floor() as i64);
        let centre = self.band.subswath(line, sample);
        let corners_valid = [-CHIP_HALF, CHIP_HALF].iter().all(|&dl| {
            [-CHIP_HALF, CHIP_HALF]
                .iter()
                .all(|&ds| self.band.subswath(line + dl, sample + ds) != 0)
        });
        if corners_valid {
            centre
        } else {
            0
        }
    }
}

/// Fractional line and sample extent `[l0, l1, s0, s1]` of located pixels,
/// grown from `extent` (start from [`EMPTY_EXTENT`]).
fn grow_extent(mut extent: [f64; 4], located: &[Option<Located>]) -> [f64; 4] {
    for at in located.iter().flatten() {
        extent[0] = extent[0].min(at.line);
        extent[1] = extent[1].max(at.line);
        extent[2] = extent[2].min(at.sample);
        extent[3] = extent[3].max(at.sample);
    }
    extent
}

/// The extent of no pixels.
const EMPTY_EXTENT: [f64; 4] = [f64::MAX, f64::MIN, f64::MAX, f64::MIN];

/// The RSLC window `(row0, col0, rows, cols)` that every chip of pixels
/// within `extent` falls in, clipped to the swath.
fn radar_window(extent: [f64; 4], rslc: &Rslc, band: &RslcBand) -> Option<[usize; 4]> {
    let [l0, l1, s0, s1] = extent;
    let pad = SINC_TAPS as f64;
    let clip = |a: f64, b: f64, n: usize| -> Option<(usize, usize)> {
        let lo = (a.floor() - pad).max(0.0);
        let hi = (b.floor() + pad).min(n as f64 - 1.0);
        (hi > lo).then_some((lo as usize, (hi - lo) as usize + 1))
    };
    let (row0, rows) = clip(l0, l1, rslc.lines)?;
    let (col0, cols) = clip(s0, s1, band.samples)?;
    Some([row0, col0, rows, cols])
}

/// Demodulate, interpolate and remodulate one pixel from `data` (the RSLC
/// window starting at `row0`, `col0`).
fn sample_pixel(
    data: &Array2<Complex32>,
    (row0, col0): (usize, usize),
    at: &Located,
    doppler_step: f64,
    kernel: &SincKernel,
) -> Option<Complex32> {
    let row = at.line - row0 as f64;
    let col = at.sample - col0 as f64;
    let (r_int, c_int) = (row.floor(), col.floor());
    let (rows, cols) = data.dim();
    let half = SINC_BEFORE as f64;
    if r_int < half
        || c_int < half
        || r_int + (SINC_TAPS - SINC_BEFORE) as f64 > rows as f64
        || c_int + (SINC_TAPS - SINC_BEFORE) as f64 > cols as f64
    {
        return None;
    }
    let (r0, c0) = (r_int as usize - SINC_BEFORE, c_int as usize - SINC_BEFORE);
    let chip: [[Complex32; SINC_TAPS]; SINC_TAPS] = std::array::from_fn(|i| {
        let demod = Complex32::from_polar(1.0, (-doppler_step * (i as f64 - half)) as f32);
        std::array::from_fn(|j| data[[r0 + i, c0 + j]] * demod)
    });
    let (fr, fc) = (row - r_int, col - c_int);
    let value = kernel.interpolate_chip(&chip, fr, fc);
    Some(value * Complex32::from_polar(1.0, (doppler_step * fr) as f32))
}

fn wrap(phase: f64) -> f32 {
    (phase
        - std::f64::consts::TAU * ((phase + std::f64::consts::PI) / std::f64::consts::TAU).floor())
        as f32
}

/// Locate the pixels of `grid` in `rows` × `cols`, row-major.
fn locate_tile(
    locator: &Locator,
    projection: &Projection,
    grid: &GeoGrid,
    x: &[f64],
    (rows, cols): (std::ops::Range<usize>, std::ops::Range<usize>),
) -> Result<Vec<Option<Located>>> {
    let n = rows.len() * cols.len();
    let mut lon = Vec::with_capacity(n);
    let mut lat = Vec::with_capacity(n);
    for r in rows {
        lon.extend_from_slice(&x[cols.clone()]);
        lat.extend(std::iter::repeat_n(
            grid.y0 + r as f64 * grid.dy,
            cols.len(),
        ));
    }
    projection.to_lonlat(&mut lon, &mut lat)?;
    Ok(lon
        .par_iter()
        .zip(lat.par_iter())
        .map(|(&lo, &la)| locator.locate(lo, la))
        .collect())
}

/// Geocode `polarizations` of `band` onto `grid`. `tec` is the date's
/// ionospheric correction (`None` leaves it out of the sampling).
///
/// The grid is processed in [`TILE`] × [`TILE`] tiles: each tile's pixels
/// are located, its own RSLC window (the tile's chips plus the sinc pad) is
/// read, and its pixels are sampled from it. A pixel's value does not
/// depend on the window it is read from, so the result is the one a single
/// window under the whole grid gives.
///
/// # Errors
/// Projection, DEM or RSLC read failures, or a grid that misses the swath.
pub fn geocode_band(
    rslc: &Rslc,
    band: &RslcBand,
    polarizations: &[String],
    dem: &Dem,
    grid: &GeoGrid,
    tec: Option<&TecCorrections>,
) -> Result<GeocodedBand> {
    let mid: DateTime<Utc> = rslc.datetime(rslc.line_time(0.5 * rslc.lines as f64));
    let locator = Locator {
        rslc,
        band,
        dem,
        tec,
        tide: tide_bodies(mid.naive_utc()),
        t_mid: rslc.orbit.seconds(mid),
    };
    let projection = Projection::new(grid.epsg)?;
    let x = grid.x_coordinates();
    let kernel = SincKernel::new();
    let k = 4.0 * std::f64::consts::PI / band.wavelength();
    let nan = Complex32::new(f32::NAN, f32::NAN);
    let shape = (grid.rows, grid.cols);
    let mut mask = Array2::from_elem(shape, MASK_FILL);
    let mut layers = vec![Array2::from_elem(shape, nan); polarizations.len()];
    let mut extent = EMPTY_EXTENT;
    for r0 in (0..grid.rows).step_by(TILE) {
        for c0 in (0..grid.cols).step_by(TILE) {
            let (rows, cols) = (
                r0..(r0 + TILE).min(grid.rows),
                c0..(c0 + TILE).min(grid.cols),
            );
            let tile = ndarray::s![rows.clone(), cols.clone()];
            let tile_shape = (rows.len(), cols.len());
            let located = locate_tile(&locator, &projection, grid, &x, (rows, cols))?;
            let tile_extent = grow_extent(EMPTY_EXTENT, &located);
            extent = grow_extent(extent, &located);
            let Some([row0, col0, win_rows, win_cols]) = radar_window(tile_extent, rslc, band)
            else {
                continue;
            };
            let tile_mask: Vec<u8> = located
                .par_iter()
                .map(|at| at.as_ref().map_or(MASK_FILL, |a| locator.mask(a)))
                .collect();
            for (pol, layer) in polarizations.iter().zip(&mut layers) {
                let data =
                    rslc.read_window(band.frequency, pol, (row0, col0), (win_rows, win_cols))?;
                let values: Vec<Complex32> = located
                    .par_iter()
                    .zip(tile_mask.par_iter())
                    .map(|(at, &m)| {
                        let at = at.as_ref().filter(|_| m != 0 && m != MASK_FILL)?;
                        let time = rslc.line_time(at.line);
                        let range = band.starting_range + at.sample * band.range_spacing;
                        if !band.doppler.contains(time, range) {
                            return None;
                        }
                        let step = std::f64::consts::TAU
                            * band.doppler.eval(time, range)
                            * rslc.line_spacing;
                        let value = sample_pixel(&data, (row0, col0), at, step, &kernel)?;
                        Some(value * Complex32::from_polar(1.0, wrap(k * at.geometric_range)))
                    })
                    .map(|v| v.unwrap_or(nan))
                    .collect();
                layer.slice_mut(tile).assign(
                    &Array2::from_shape_vec(tile_shape, values).map_err(crate::error::product)?,
                );
            }
            mask.slice_mut(tile).assign(
                &Array2::from_shape_vec(tile_shape, tile_mask).map_err(crate::error::product)?,
            );
        }
    }
    let radar_window = radar_window(extent, rslc, band).ok_or_else(|| {
        L1Error::NotFound(format!(
            "grid does not overlap frequency{} of {}",
            band.frequency, rslc.granule_id
        ))
    })?;
    Ok(GeocodedBand {
        grid: *grid,
        radar_window,
        layers,
        mask,
    })
}

/// Cube node spacing (m).
pub const CUBE_POSTING: f64 = 500.0;
/// Cube heights (m above the ellipsoid), as in the RSLC geolocation grid.
pub const CUBE_HEIGHTS: std::ops::RangeInclusive<i32> = -500..=9000;
const CUBE_HEIGHT_STEP: usize = 500;

/// Ground-to-sensor LOS (east, north) on [`CUBE_POSTING`] nodes covering
/// `grid` (one node beyond each edge) at [`CUBE_HEIGHTS`], the layout of the
/// GSLC `radarGrid` cube the NISAR LOS readers interpolate with the DEM.
///
/// # Errors
/// Projection failures, or a node the orbit does not reach.
pub fn los_cube(rslc: &Rslc, grid: &GeoGrid) -> Result<crate::nisar_h5::LosCube> {
    let x_edges = [
        grid.x0 - grid.dx / 2.0,
        grid.x0 + (grid.cols as f64 - 0.5) * grid.dx,
    ];
    let y_edges = [
        grid.y0 - grid.dy / 2.0,
        grid.y0 + (grid.rows as f64 - 0.5) * grid.dy,
    ];
    let nodes = |lo: f64, hi: f64| -> Vec<f64> {
        let (a, b) = (
            (lo / CUBE_POSTING).floor() - 1.0,
            (hi / CUBE_POSTING).ceil() + 1.0,
        );
        (a as i64..=b as i64)
            .map(|k| k as f64 * CUBE_POSTING)
            .collect()
    };
    let x = nodes(x_edges[0].min(x_edges[1]), x_edges[0].max(x_edges[1]));
    let mut y = nodes(y_edges[0].min(y_edges[1]), y_edges[0].max(y_edges[1]));
    y.reverse();
    let heights: Vec<f64> = CUBE_HEIGHTS
        .step_by(CUBE_HEIGHT_STEP)
        .map(f64::from)
        .collect();
    let mut lon: Vec<f64> = y.iter().flat_map(|_| x.iter().copied()).collect();
    let mut lat: Vec<f64> = y
        .iter()
        .flat_map(|&v| std::iter::repeat_n(v, x.len()))
        .collect();
    Projection::new(grid.epsg)?.to_lonlat(&mut lon, &mut lat)?;
    let t_mid = rslc.line_time(0.5 * rslc.lines as f64);
    let shape = (heights.len(), y.len(), x.len());
    let los: Vec<(f32, f32)> = heights
        .iter()
        .flat_map(|&h| lon.iter().zip(&lat).map(move |(&lo, &la)| (h, lo, la)))
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&(h, lo, la)| {
            let (la, lo) = (la.to_radians(), lo.to_radians());
            let target = llh_to_ecef(la, lo, h);
            let radar = geo2rdr(target, &rslc.orbit, t_mid)?;
            let enu = crate::geometry::line_of_sight_enu(target, la, lo, radar.platform.position);
            Ok((enu[0] as f32, enu[1] as f32))
        })
        .collect::<Result<_>>()?;
    let pick = |k: usize| {
        Array3::from_shape_vec(
            shape,
            los.iter().map(|p| if k == 0 { p.0 } else { p.1 }).collect(),
        )
        .map_err(crate::error::product)
    };
    Ok(crate::nisar_h5::LosCube {
        epsg: grid.epsg,
        east: pick(0)?,
        north: pick(1)?,
        x,
        y,
        heights,
    })
}
