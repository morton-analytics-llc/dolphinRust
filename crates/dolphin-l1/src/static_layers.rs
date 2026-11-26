//! Static layers on the burst grid, from our own geometry: ground-to-satellite
//! LOS east/north and the layover/shadow mask, in the OPERA CSLC-S1-STATIC
//! layout (`/data/los_east`, `/data/los_north`, `/data/layover_shadow_mask`,
//! `x_coordinates`, `y_coordinates`, `projection`) that the workflow's LOS,
//! correction and quality-mask readers take.
//!
//! - **LOS:** per pixel, the unit vector from the DEM target to the platform
//!   at the target's own zero-Doppler time.
//! - **Layover/shadow:** every [`PROFILE_STEP_LINES`] burst lines, a ground
//!   profile is traced across the swath in the zero-Doppler plane (10 m steps,
//!   DEM heights). Along it, a point is in *shadow* when its look angle is
//!   below the largest look angle of any nearer point (terrain nearer the
//!   radar hides it), and in *layover* when some nearer point has a greater
//!   slant range or some farther point a smaller one (two ground points share
//!   a range cell). Each output pixel takes the flags of the nearest profile
//!   at its own cross-track position. Values follow OPERA: 0 clear,
//!   1 shadow, 2 layover, 3 both, 127 outside the burst's valid window.

use ndarray::Array2;
use rayon::prelude::*;

use crate::cslc_h5::{put_str, time_string};
use crate::dem::Dem;
use crate::error::{L1Error, Result};
use crate::geometry::{
    cross, dot, ecef_to_llh, geo2rdr, line_of_sight_enu, llh_to_ecef, norm, rdr2geo, sub, LookSide,
};
use crate::grid::{GeoGrid, Projection};
use crate::orbit::{Orbit, Vec3};
use crate::safe::S1Burst;

/// Burst lines between layover/shadow profiles.
pub const PROFILE_STEP_LINES: usize = 5;
/// Ground step along a profile (m).
const PROFILE_STEP_M: f64 = 10.0;
/// Profile extension beyond the swath on each side (m).
const PROFILE_MARGIN_M: f64 = 3_000.0;
/// Grid rows converted to lon/lat per chunk.
const ROW_CHUNK: usize = 128;
/// Mask value outside the valid burst window.
pub const MASK_NODATA: u8 = 127;
const SHADOW: u8 = 1;
const LAYOVER: u8 = 2;

/// The radar grid the static layers are traced over: a Sentinel-1 burst or
/// the part of a NISAR RSLC under the output grid.
#[derive(Debug, Clone, Copy)]
pub struct RadarGeometry {
    /// Zero-Doppler time of line 0.
    pub sensing_start: chrono::DateTime<chrono::Utc>,
    /// Line spacing (s).
    pub azimuth_time_interval: f64,
    /// Slant range of sample 0 (m).
    pub starting_range: f64,
    /// Slant-range spacing (m).
    pub range_pixel_spacing: f64,
    /// Lines.
    pub lines: usize,
    /// Samples.
    pub samples: usize,
    /// Valid lines `[first, last]` (inclusive).
    pub valid_lines: (usize, usize),
    /// Valid samples `[first, last]` (inclusive).
    pub valid_samples: (usize, usize),
    /// Look side.
    pub look: LookSide,
}

impl RadarGeometry {
    /// The radar grid of a Sentinel-1 burst (right looking).
    pub fn from_s1(burst: &S1Burst) -> Self {
        Self {
            sensing_start: burst.sensing_start,
            azimuth_time_interval: burst.azimuth_time_interval,
            starting_range: burst.starting_range(),
            range_pixel_spacing: burst.range_pixel_spacing(),
            lines: burst.lines,
            samples: burst.samples,
            valid_lines: (burst.first_valid_line, burst.last_valid_line),
            valid_samples: (burst.first_valid_sample, burst.last_valid_sample),
            look: LookSide::Right,
        }
    }

    fn sensing_mid(&self) -> chrono::DateTime<chrono::Utc> {
        self.sensing_start
            + crate::orbit::duration_from_seconds(
                0.5 * self.lines as f64 * self.azimuth_time_interval,
            )
    }
}

/// The static layers of one burst on a grid.
pub struct StaticLayers {
    /// The grid.
    pub grid: GeoGrid,
    /// Ground-to-satellite LOS, east component (NaN outside the burst).
    pub los_east: Array2<f32>,
    /// Ground-to-satellite LOS, north component (NaN outside the burst).
    pub los_north: Array2<f32>,
    /// 0 clear, 1 shadow, 2 layover, 3 both, 127 outside the burst.
    pub layover_shadow_mask: Array2<u8>,
}

/// One cross-track ground profile at a fixed azimuth time.
struct Profile {
    line: f64,
    origin: Vec3,
    across: Vec3,
    s0: f64,
    flags: Vec<u8>,
}

impl Profile {
    fn flag_at(&self, target: Vec3) -> u8 {
        let s = dot(sub(target, self.origin), self.across);
        let index = ((s - self.s0) / PROFILE_STEP_M).round();
        if index < 0.0 {
            return 0;
        }
        self.flags.get(index as usize).copied().unwrap_or(0)
    }
}

/// Compute the static layers of the radar grid `burst` (on `orbit`, over
/// `dem`) on `grid`.
pub fn static_layers(
    burst: &RadarGeometry,
    orbit: &Orbit,
    dem: &Dem,
    grid: &GeoGrid,
) -> Result<StaticLayers> {
    let profiles = trace_profiles(burst, orbit, dem)?;
    let projection = Projection::new(grid.epsg)?;
    let t_mid = orbit.seconds(burst.sensing_mid());
    let t0 = orbit.seconds(burst.sensing_start);
    let (r0, dr) = (burst.starting_range, burst.range_pixel_spacing);
    let shape = (grid.rows, grid.cols);
    let mut los_east = Array2::from_elem(shape, f32::NAN);
    let mut los_north = Array2::from_elem(shape, f32::NAN);
    let mut layover_shadow_mask = Array2::from_elem(shape, MASK_NODATA);
    for chunk0 in (0..grid.rows).step_by(ROW_CHUNK) {
        let chunk_rows = ROW_CHUNK.min(grid.rows - chunk0);
        let mut lon: Vec<f64> = Vec::with_capacity(chunk_rows * grid.cols);
        let mut lat: Vec<f64> = Vec::with_capacity(chunk_rows * grid.cols);
        for r in chunk0..chunk0 + chunk_rows {
            lon.extend(grid.x_coordinates());
            lat.extend(std::iter::repeat_n(grid.y0 + r as f64 * grid.dy, grid.cols));
        }
        projection.to_lonlat(&mut lon, &mut lat)?;
        let chunk: Vec<(f32, f32, u8)> = (0..chunk_rows * grid.cols)
            .into_par_iter()
            .map(|k| {
                pixel(
                    burst,
                    orbit,
                    dem,
                    &profiles,
                    (lon[k], lat[k]),
                    (t_mid, t0, r0, dr),
                )
                .unwrap_or((f32::NAN, f32::NAN, MASK_NODATA))
            })
            .collect();
        let rows = ndarray::s![chunk0..chunk0 + chunk_rows, ..];
        for ((p, e), (n, m)) in chunk.iter().zip(los_east.slice_mut(rows)).zip(
            los_north
                .slice_mut(rows)
                .into_iter()
                .zip(layover_shadow_mask.slice_mut(rows)),
        ) {
            (*e, *n, *m) = *p;
        }
    }
    Ok(StaticLayers {
        grid: *grid,
        los_east,
        los_north,
        layover_shadow_mask,
    })
}

/// `(los_east, los_north, mask)` of one pixel, `None` outside the valid burst.
fn pixel(
    burst: &RadarGeometry,
    orbit: &Orbit,
    dem: &Dem,
    profiles: &[Profile],
    (lon, lat): (f64, f64),
    (t_mid, t0, r0, dr): (f64, f64, f64, f64),
) -> Option<(f32, f32, u8)> {
    let height = dem.height(lon, lat).ok().filter(|h| h.is_finite())?;
    let (la, lo) = (lat.to_radians(), lon.to_radians());
    let target = llh_to_ecef(la, lo, height);
    let radar = geo2rdr(target, orbit, t_mid).ok()?;
    let line = (radar.azimuth_time - t0) / burst.azimuth_time_interval;
    let sample = (radar.slant_range - r0) / dr;
    let inside = line >= burst.valid_lines.0 as f64
        && line <= burst.valid_lines.1 as f64
        && sample >= burst.valid_samples.0 as f64
        && sample <= burst.valid_samples.1 as f64;
    if !inside {
        return None;
    }
    let los = line_of_sight_enu(target, la, lo, radar.platform.position);
    let nearest = profiles
        .iter()
        .min_by(|a, b| (a.line - line).abs().total_cmp(&(b.line - line).abs()))?;
    Some((los[0] as f32, los[1] as f32, nearest.flag_at(target)))
}

fn trace_profiles(burst: &RadarGeometry, orbit: &Orbit, dem: &Dem) -> Result<Vec<Profile>> {
    let mut lines: Vec<usize> = (0..burst.lines).step_by(PROFILE_STEP_LINES).collect();
    lines.push(burst.lines - 1);
    lines
        .into_par_iter()
        .map(|line| trace_profile(burst, orbit, dem, line))
        .collect()
}

/// DEM height for the layover/shadow profile: 0 m where the DEM has a hole or no
/// coverage.
fn profile_height(dem: &Dem, lon: f64, lat: f64) -> f64 {
    dem.height(lon, lat)
        .ok()
        .filter(|h| h.is_finite())
        .unwrap_or(0.0)
}

/// A profile end: the DEM surface at `(t, range)`, or the 0 m surface where
/// the end sits on the stack-DEM edge. There the surface steps from the
/// terrain to the 0 m fill of [`profile_height`] and has no root (#248).
fn profile_end(
    t: f64,
    range: f64,
    orbit: &Orbit,
    look: LookSide,
    dem: &Dem,
) -> Result<crate::geometry::GroundPoint> {
    let height = |lon: f64, lat: f64| -> Result<f64> { Ok(profile_height(dem, lon, lat)) };
    match rdr2geo(t, range, orbit, look, height) {
        Err(L1Error::Geometry(_)) => rdr2geo(t, range, orbit, look, |_, _| Ok(0.0)),
        found => found,
    }
}

/// The layover/shadow flags along the ground profile of burst line `line`.
fn trace_profile(burst: &RadarGeometry, orbit: &Orbit, dem: &Dem, line: usize) -> Result<Profile> {
    let t = orbit.seconds(burst.sensing_start) + line as f64 * burst.azimuth_time_interval;
    let state = orbit.state(t)?;
    // A profile end outside the stack DEM is treated like a DEM hole (0 m): it only
    // places the profile line, and must not abort the burst's CSLC (#240).
    let height = |lon: f64, lat: f64| -> Result<f64> { Ok(profile_height(dem, lon, lat)) };
    let near = profile_end(t, burst.starting_range, orbit, burst.look, dem)?;
    let far_range = burst.starting_range + burst.samples as f64 * burst.range_pixel_spacing;
    let far = profile_end(t, far_range, orbit, burst.look, dem)?;
    let span = sub(far.ecef, near.ecef);
    let across = {
        let n = norm(span);
        [span[0] / n, span[1] / n, span[2] / n]
    };
    let s0 = -PROFILE_MARGIN_M;
    let steps = ((norm(span) + 2.0 * PROFILE_MARGIN_M) / PROFILE_STEP_M) as usize + 1;
    let down = {
        let p = state.position;
        let n = norm(p);
        [-p[0] / n, -p[1] / n, -p[2] / n]
    };
    let mut ranges = Vec::with_capacity(steps);
    let mut looks = Vec::with_capacity(steps);
    for k in 0..steps {
        let s = s0 + k as f64 * PROFILE_STEP_M;
        let guess: Vec3 = std::array::from_fn(|i| near.ecef[i] + s * across[i]);
        let (la, lo, _) = ecef_to_llh(guess);
        let h = height(lo.to_degrees(), la.to_degrees()).unwrap_or(0.0);
        let ground = llh_to_ecef(la, lo, h);
        let look = sub(ground, state.position);
        let range = norm(look);
        ranges.push(range);
        looks.push(norm(cross(look, down)).atan2(dot(look, down)));
    }
    Ok(Profile {
        line: line as f64,
        origin: near.ecef,
        across,
        s0,
        flags: flag_profile(&ranges, &looks),
    })
}

/// Shadow and layover flags along a profile ordered from near to far range.
fn flag_profile(ranges: &[f64], looks: &[f64]) -> Vec<u8> {
    let n = ranges.len();
    let mut flags = vec![0u8; n];
    let (mut max_look, mut max_range) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for k in 0..n {
        if looks[k] < max_look {
            flags[k] |= SHADOW;
        }
        if ranges[k] <= max_range {
            flags[k] |= LAYOVER;
        }
        max_look = max_look.max(looks[k]);
        max_range = max_range.max(ranges[k]);
    }
    let mut min_range = f64::INFINITY;
    for k in (0..n).rev() {
        if ranges[k] >= min_range {
            flags[k] |= LAYOVER;
        }
        min_range = min_range.min(ranges[k]);
    }
    flags
}

/// The acquisition a STATIC file was derived from: written to `/identification` so
/// readers can check pass, look side and timing against the CSLC stack (the same
/// fields the CSLC writer records).
#[derive(Debug, Clone, Copy)]
pub struct StaticIdentity<'a> {
    /// Burst id (S1) or frame label (NISAR).
    pub burst_id: &'a str,
    /// `Ascending` / `Descending`.
    pub pass_direction: &'a str,
    /// Radar look side.
    pub look: LookSide,
    /// Zero-Doppler time of the first line.
    pub zero_doppler_start: chrono::DateTime<chrono::Utc>,
    /// Zero-Doppler time of the last line.
    pub zero_doppler_end: chrono::DateTime<chrono::Utc>,
}

/// Write the static layers in the OPERA CSLC-S1-STATIC layout.
///
/// # Errors
/// HDF5 write failures.
pub fn write_static(
    path: &std::path::Path,
    layers: &StaticLayers,
    identity: &StaticIdentity<'_>,
) -> Result<()> {
    let file = hdf5::File::create(path)?;
    let data = file.create_group("data")?;
    let (rows, cols) = layers.los_east.dim();
    for (name, layer) in [
        ("los_east", &layers.los_east),
        ("los_north", &layers.los_north),
    ] {
        data.new_dataset::<f32>()
            .shape((rows, cols))
            .chunk((rows.min(128), cols.min(128)))
            .deflate(4)
            .create(name)?
            .write(layer)?;
    }
    data.new_dataset::<u8>()
        .shape((rows, cols))
        .chunk((rows.min(128), cols.min(128)))
        .deflate(4)
        .create("layover_shadow_mask")?
        .write(&layers.layover_shadow_mask)?;
    data.new_dataset_builder()
        .with_data(&ndarray::Array1::from(layers.grid.x_coordinates()))
        .create("x_coordinates")?;
    data.new_dataset_builder()
        .with_data(&ndarray::Array1::from(layers.grid.y_coordinates()))
        .create("y_coordinates")?;
    data.new_dataset::<i32>()
        .create("projection")?
        .write_scalar(&i32::try_from(layers.grid.epsg).map_err(crate::error::product)?)?;
    let ident = file.create_group("identification")?;
    let id = hdf5::types::FixedAscii::<64>::from_ascii(identity.burst_id.as_bytes())
        .map_err(crate::error::product)?;
    ident
        .new_dataset::<hdf5::types::FixedAscii<64>>()
        .create("burst_id")?
        .write_scalar(&id)?;
    put_str(&ident, "orbit_pass_direction", identity.pass_direction)?;
    put_str(
        &ident,
        "look_direction",
        match identity.look {
            LookSide::Right => "Right",
            LookSide::Left => "Left",
        },
    )?;
    put_str(
        &ident,
        "zero_doppler_start_time",
        &time_string(identity.zero_doppler_start),
    )?;
    put_str(
        &ident,
        "zero_doppler_end_time",
        &time_string(identity.zero_doppler_end),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// STATIC files carry the identification fields provenance readers need (#252).
    #[test]
    fn static_files_record_pass_look_and_timing() {
        let grid = GeoGrid {
            epsg: 32613,
            x0: 2.5,
            y0: -5.0,
            dx: 5.0,
            dy: -10.0,
            cols: 3,
            rows: 2,
        };
        let layers = StaticLayers {
            grid,
            los_east: Array2::from_elem((2, 3), 0.6),
            los_north: Array2::from_elem((2, 3), -0.1),
            layover_shadow_mask: Array2::zeros((2, 3)),
        };
        let dir = std::env::temp_dir().join(format!("static-id-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("STATIC-S1_T078-165677-IW1_20260705.h5");
        let start: chrono::DateTime<chrono::Utc> = "2026-07-05T00:55:57.337719Z".parse().unwrap();
        write_static(
            &path,
            &layers,
            &StaticIdentity {
                burst_id: "t078_165677_iw1",
                pass_direction: "Ascending",
                look: LookSide::Right,
                zero_doppler_start: start,
                zero_doppler_end: start + chrono::Duration::seconds(3),
            },
        )
        .unwrap();
        let file = hdf5::File::open(&path).unwrap();
        for (name, expected) in [
            ("orbit_pass_direction", "Ascending"),
            ("look_direction", "Right"),
        ] {
            let value = file
                .dataset(&format!("identification/{name}"))
                .unwrap()
                .read_scalar::<hdf5::types::FixedAscii<64>>()
                .unwrap();
            assert_eq!(value.as_str(), expected);
        }
        assert!(file
            .dataset("identification/zero_doppler_start_time")
            .is_ok());
        assert!(file.dataset("identification/zero_doppler_end_time").is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Outside the DEM and in a DEM hole the profile uses 0 m instead of failing (#240).
    #[test]
    fn profile_height_falls_back_to_zero_outside_the_dem() {
        let mut heights = ndarray::Array2::from_elem((12, 12), 100.0_f32);
        heights[[9, 9]] = f32::NAN;
        let dem = Dem::from_grid(10.0, 50.0, 0.01, -0.01, heights);
        assert!((profile_height(&dem, 10.02, 49.98) - 100.0).abs() < 1e-3);
        assert_eq!(profile_height(&dem, 10.085, 49.915), 0.0);
        assert_eq!(profile_height(&dem, -109.67, 48.19), 0.0);
    }

    /// A profile end on the stack-DEM edge, where the DEM surface has no
    /// root, lands on the 0 m surface instead of failing (#248).
    #[test]
    fn profile_end_on_the_dem_edge_falls_back_to_zero() {
        let (r, omega, incl) = (7.07e6, 1.06e-3, 98.2_f64.to_radians());
        let truth = |t: f64| {
            let (s, c) = (omega * t).sin_cos();
            let (si, ci) = incl.sin_cos();
            (
                [r * c, r * s * ci, r * s * si],
                [-r * omega * s, r * omega * c * ci, r * omega * c * si],
            )
        };
        let times: Vec<f64> = (0..40).map(|i| f64::from(i) * 10.0).collect();
        let orbit = Orbit {
            reference_epoch: chrono::Utc::now(),
            positions: times.iter().map(|&t| truth(t).0).collect(),
            velocities: times.iter().map(|&t| truth(t).1).collect(),
            times,
            orbit_type: "test".into(),
        };
        let (t, range) = (150.0, 9.3e5);
        let at = |h: f64| rdr2geo(t, range, &orbit, LookSide::Right, |_, _| Ok(h)).unwrap();
        let (high, zero) = (at(700.0), at(0.0));
        let deg = |p: &crate::geometry::GroundPoint| (p.lon.to_degrees(), p.lat.to_degrees());
        let ((lon_h, lat_h), (lon_z, lat_z)) = (deg(&high), deg(&zero));
        // A 700 m DEM that covers the 0 m solution (the nearer one) and ends
        // half-way to the 700 m one: inside, the surface is below the ray;
        // past the edge it reads 0 m and is above it.
        let (edge_lon, edge_lat) = ((lon_h + lon_z) / 2.0, (lat_h + lat_z) / 2.0);
        let step = 0.0001;
        let span = |a: f64, b: f64| (a.min(b) - 0.05, a.max(b) + 0.05);
        let (lon_lo, lon_hi) = span(lon_z, lon_z);
        let (lon_lo, lon_hi) = if lon_h > lon_z {
            (lon_lo, edge_lon)
        } else {
            (edge_lon, lon_hi)
        };
        let (lat_lo, lat_hi) = span(lat_z, edge_lat);
        let (cols, rows) = (
            ((lon_hi - lon_lo) / step) as usize,
            ((lat_hi - lat_lo) / step) as usize,
        );
        let dem = Dem::from_grid(
            lon_lo,
            lat_hi,
            step,
            -step,
            ndarray::Array2::from_elem((rows, cols), 700.0_f32),
        );
        let height = |lon: f64, lat: f64| -> Result<f64> { Ok(profile_height(&dem, lon, lat)) };
        assert!(rdr2geo(t, range, &orbit, LookSide::Right, height).is_err());
        let end = profile_end(t, range, &orbit, LookSide::Right, &dem).unwrap();
        assert!(end.height.abs() < 1e-4, "{}", end.height);
        // Away from the edge the DEM solve is used as before.
        let inside = profile_end(t, range - 2_000.0, &orbit, LookSide::Right, &dem).unwrap();
        assert!((inside.height - 700.0).abs() < 1e-4, "{}", inside.height);
    }

    /// A ridge facing the radar: the near-range back-slope behind the crest
    /// is shadowed, a slope steeper than the look angle lays over.
    #[test]
    fn flags_follow_look_angle_and_range_order() {
        let ranges = [100.0, 101.0, 100.5, 102.0, 103.0];
        let looks = [0.5, 0.6, 0.55, 0.65, 0.7];
        let flags = flag_profile(&ranges, &looks);
        assert_eq!(flags[0], 0);
        assert_eq!(flags[2] & SHADOW, SHADOW);
        assert_eq!(flags[2] & LAYOVER, LAYOVER);
        assert_eq!(flags[1] & LAYOVER, LAYOVER);
        assert_eq!(flags[4], 0);
    }
}
