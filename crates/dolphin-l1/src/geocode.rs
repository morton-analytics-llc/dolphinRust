//! Geocode one Sentinel-1 burst onto a projected grid (COMPASS-equivalent).
//!
//! Every date is geocoded independently onto the same fixed burst grid, so the
//! geocoding *is* the coregistration: precise orbits and the DEM put each
//! date's samples on the same ground pixel, and the residual azimuth
//! misregistration that orbits leave is measured by ESD on burst overlaps
//! ([`crate::esd`]) and fed back as `azimuth_shift`. One interpolation per
//! date instead of resample-then-geocode.
//!
//! Per output pixel: grid (x, y) → lon/lat → DEM height → ECEF target →
//! zero-Doppler `geo2rdr` → timing corrections → fractional burst line/sample
//! → 8×8 sinc on the deramped burst → reramp → flatten: multiply by
//! `exp(+j4πR/λ)` with R the geometric (uncorrected) slant range, which
//! removes the SLC's `exp(−j4πR/λ)` propagation phase. That is ISCE3
//! `geocodeSlc` with `flatten = true`, `flattenWithCorrectedSRng = false`, as
//! COMPASS calls it for OPERA CSLC-S1.
//! Pixels whose kernel leaves the burst's valid window are NaN.
//!
//! The sampling range also carries the line-of-sight solid-earth tide
//! (IERS 2010 degree 2, at `sensing_start`), as COMPASS's range LUT does: the
//! sample is taken where the tidally displaced target is imaged, and because
//! flattening uses the geometric range the tide's phase stays in the CSLC for
//! the downstream correction, as in OPERA CSLC-S1. The ionospheric range
//! delay of the JPL GIM ([`crate::gim`]) is added the same way when a map is
//! given.

use chrono::{DateTime, Utc};
use dolphin_corrections::ionosphere::IonexMaps;
use dolphin_corrections::solid_earth_tide::{tide_bodies, tide_displacement_enu_at, BodyPosition};
use ndarray::{Array2, ArrayView2, Axis};
use num_complex::Complex32;
use rayon::prelude::*;

use crate::dem::Dem;
use crate::error::Result;
use crate::geometry::{dot, enu_basis, geo2rdr, llh_to_ecef, norm, sub};
use crate::grid::{GeoGrid, Projection};
use crate::interp::SincKernel;
use crate::orbit::Orbit;
use crate::safe::S1Burst;
use crate::tops::{TimingCorrections, TopsCarrier};

/// Grid rows transformed to lon/lat per chunk.
const ROW_CHUNK: usize = 128;

/// A geocoded burst on a [`GeoGrid`].
#[derive(Debug, Clone)]
pub struct GeocodedBurst {
    /// The grid.
    pub grid: GeoGrid,
    /// Complex samples, NaN outside the burst's valid window.
    pub data: Array2<Complex32>,
    /// Total TOPS Doppler (Hz) the pixel was imaged at, NaN where invalid.
    pub doppler: Array2<f32>,
    /// Reramp (azimuth carrier) and flattening phases (rad, wrapped), when
    /// requested: OPERA's `azimuth_carrier_phase` / `flattening_phase`.
    pub phases: Option<(Array2<f32>, Array2<f32>)>,
}

/// One geocoded pixel.
#[derive(Clone, Copy)]
struct Pixel {
    value: Complex32,
    doppler: f32,
    carrier: f32,
    flatten: f32,
}

fn wrap(phase: f64) -> f32 {
    (phase
        - std::f64::consts::TAU * ((phase + std::f64::consts::PI) / std::f64::consts::TAU).floor())
        as f32
}

/// The valid window of a burst with the TOPS carrier removed.
pub struct DerampedBurst {
    /// Deramped samples of the valid window.
    pub samples: Array2<Complex32>,
    /// Burst line of the window's first row.
    pub row0: usize,
    /// Burst sample of the window's first column.
    pub col0: usize,
}

impl DerampedBurst {
    /// Read `burst`'s valid window, multiply by `exp(−jφ)` (deramp) and by
    /// `exp(−j2π f_dc η)` (baseband).
    pub fn read(burst: &S1Burst, carrier: &TopsCarrier) -> Result<Self> {
        let (row0, col0) = (burst.first_valid_line, burst.first_valid_sample);
        let rows = burst.last_valid_line + 1 - row0;
        let cols = burst.last_valid_sample + 1 - col0;
        let mut samples = burst.read_window(row0, col0, rows, cols)?;
        let spacing = burst.range_pixel_spacing();
        let ranges: Vec<f64> = (0..cols)
            .map(|c| carrier.starting_range() + (col0 + c) as f64 * spacing)
            .collect();
        let terms: Vec<(f64, f64)> = ranges.iter().map(|&r| carrier.range_terms(r)).collect();
        samples
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(r, mut line)| {
                let line_index = (row0 + r) as f64;
                let eta = carrier.eta(line_index);
                for ((value, (kt, eta_ref)), &range) in line.iter_mut().zip(&terms).zip(&ranges) {
                    let d = eta - eta_ref;
                    let phase = -std::f64::consts::PI * kt * d * d
                        - carrier.baseband_phase(line_index, range);
                    *value *= Complex32::from_polar(1.0, wrap(phase));
                }
            });
        Ok(Self {
            samples,
            row0,
            col0,
        })
    }

    fn view(&self) -> ArrayView2<'_, Complex32> {
        self.samples.view()
    }
}

/// Inputs shared by every pixel of one burst.
struct PixelContext<'a> {
    burst: &'a S1Burst,
    orbit: &'a Orbit,
    dem: &'a Dem,
    corrections: &'a TimingCorrections,
    deramped: &'a DerampedBurst,
    kernel: &'a SincKernel,
    t_mid: f64,
    azimuth_shift: f64,
    tide: [(BodyPosition, f64); 2],
    ionosphere: Option<&'a IonexMaps>,
    mid_time: DateTime<Utc>,
}

impl PixelContext<'_> {
    /// Sample at `(lon, lat)` degrees, or `None` outside the burst or where
    /// the DEM has no valid height.
    fn sample(&self, lon: f64, lat: f64) -> Option<Pixel> {
        let height = self.dem.height(lon, lat).ok().filter(|h| h.is_finite())?;
        let (la, lo) = (lat.to_radians(), lon.to_radians());
        let target = llh_to_ecef(la, lo, height);
        let radar = geo2rdr(target, self.orbit, self.t_mid).ok()?;
        let look = sub(radar.platform.position, target);
        let incidence = (dot(look, enu_basis(la, lo)[2]) / norm(look)).acos();
        let tide_enu = tide_displacement_enu_at(&self.tide, lon, lat, height);
        let basis = enu_basis(la, lo);
        let tide_ecef: [f64; 3] =
            std::array::from_fn(|k| (0..3).map(|a| tide_enu[a] * basis[a][k]).sum());
        // Range change of the displaced target: −d·l̂, l̂ ground → platform.
        let tide_range = -dot(tide_ecef, look) / norm(look);
        let correction = self.corrections.at(
            radar.azimuth_time,
            radar.slant_range,
            target,
            &radar.platform,
            height,
            incidence,
        );
        let line = self
            .corrections
            .line(radar.azimuth_time + correction.azimuth_time + self.azimuth_shift);
        let ionosphere = match self.ionosphere {
            Some(maps) => crate::gim::slant_delay(
                maps,
                self.mid_time,
                lat,
                lon,
                incidence,
                self.burst.radar_frequency,
            )?,
            None => 0.0,
        };
        let range = radar.slant_range + correction.slant_range + tide_range + ionosphere;
        let carrier = self.corrections.carrier();
        let sample = (range - carrier.starting_range()) / self.burst.range_pixel_spacing();
        let row = line - self.deramped.row0 as f64;
        let col = sample - self.deramped.col0 as f64;
        let value = self.kernel.interpolate(self.deramped.view(), row, col)?;
        let carrier_phase = carrier.phase(line, range);
        let rebaseband = Complex32::from_polar(1.0, wrap(carrier.baseband_phase(line, range)));
        let flatten = 4.0 * std::f64::consts::PI * radar.slant_range / self.burst.wavelength();
        let (carrier_w, flatten_w) = (wrap(carrier_phase), wrap(flatten));
        Some(Pixel {
            value: value
                * rebaseband
                * Complex32::from_polar(1.0, carrier_w)
                * Complex32::from_polar(1.0, flatten_w),
            doppler: carrier.total_doppler(line, range) as f32,
            carrier: carrier_w,
            flatten: flatten_w,
        })
    }
}

/// Geocode `burst` (its `orbit` and the stack `dem`) onto `grid`.
/// `azimuth_shift` (s) is the ESD misregistration correction added to every
/// azimuth sampling time (0 for the reference date). `keep_phases` also
/// returns the carrier and flattening phase layers. `ionosphere` is the
/// date's GIM; `None` leaves the ionospheric delay out of the sampling.
pub fn geocode_burst(
    burst: &S1Burst,
    orbit: &Orbit,
    dem: &Dem,
    grid: &GeoGrid,
    azimuth_shift: f64,
    keep_phases: bool,
    ionosphere: Option<&IonexMaps>,
) -> Result<GeocodedBurst> {
    let corrections = TimingCorrections::new(burst, orbit)?;
    let deramped = DerampedBurst::read(burst, corrections.carrier())?;
    let kernel = SincKernel::new();
    let context = PixelContext {
        burst,
        orbit,
        dem,
        corrections: &corrections,
        deramped: &deramped,
        kernel: &kernel,
        t_mid: orbit.seconds(burst.sensing_mid()),
        azimuth_shift,
        tide: tide_bodies(burst.sensing_start.naive_utc()),
        ionosphere,
        mid_time: burst.sensing_mid(),
    };
    let projection = Projection::new(grid.epsg)?;
    let nan = Complex32::new(f32::NAN, f32::NAN);
    let mut data = Array2::from_elem((grid.rows, grid.cols), nan);
    let mut doppler = Array2::from_elem((grid.rows, grid.cols), f32::NAN);
    let phase_shape = if keep_phases {
        (grid.rows, grid.cols)
    } else {
        (0, 0)
    };
    let mut carrier = Array2::from_elem(phase_shape, f32::NAN);
    let mut flatten = Array2::from_elem(phase_shape, f32::NAN);
    let x = grid.x_coordinates();
    for chunk0 in (0..grid.rows).step_by(ROW_CHUNK) {
        let chunk_rows = ROW_CHUNK.min(grid.rows - chunk0);
        let (lon, lat) = chunk_lonlat(&projection, grid, &x, chunk0, chunk_rows)?;
        let pixels: Vec<Vec<Option<Pixel>>> = (0..chunk_rows)
            .into_par_iter()
            .map(|r| {
                let offset = r * grid.cols;
                (0..grid.cols)
                    .map(|c| context.sample(lon[offset + c], lat[offset + c]))
                    .collect()
            })
            .collect();
        for (r, row) in pixels.into_iter().enumerate() {
            for (c, pixel) in row.into_iter().enumerate() {
                let Some(p) = pixel else { continue };
                let at = (chunk0 + r, c);
                data[at] = p.value;
                doppler[at] = p.doppler;
                if keep_phases {
                    carrier[at] = p.carrier;
                    flatten[at] = p.flatten;
                }
            }
        }
    }
    Ok(GeocodedBurst {
        grid: *grid,
        data,
        doppler,
        phases: keep_phases.then_some((carrier, flatten)),
    })
}

fn chunk_lonlat(
    projection: &Projection,
    grid: &GeoGrid,
    x: &[f64],
    row0: usize,
    rows: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut lon = Vec::with_capacity(rows * grid.cols);
    let mut lat = Vec::with_capacity(rows * grid.cols);
    for r in row0..row0 + rows {
        lon.extend_from_slice(x);
        lat.extend(std::iter::repeat_n(grid.y0 + r as f64 * grid.dy, grid.cols));
    }
    projection.to_lonlat(&mut lon, &mut lat)?;
    Ok((lon, lat))
}
