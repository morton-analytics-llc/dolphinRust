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

use crate::dem::Dem;
use crate::error::Result;
use crate::geometry::{
    cross, dot, ecef_to_llh, geo2rdr, line_of_sight_enu, llh_to_ecef, norm, rdr2geo, sub,
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

/// Compute the static layers of `burst` (on `orbit`, over `dem`) on `grid`.
pub fn static_layers(
    burst: &S1Burst,
    orbit: &Orbit,
    dem: &Dem,
    grid: &GeoGrid,
) -> Result<StaticLayers> {
    let profiles = trace_profiles(burst, orbit, dem)?;
    let projection = Projection::new(grid.epsg)?;
    let t_mid = orbit.seconds(burst.sensing_mid());
    let t0 = orbit.seconds(burst.sensing_start);
    let (r0, dr) = (burst.starting_range(), burst.range_pixel_spacing());
    let mut rows: Vec<Vec<(f32, f32, u8)>> = Vec::with_capacity(grid.rows);
    for chunk0 in (0..grid.rows).step_by(ROW_CHUNK) {
        let chunk_rows = ROW_CHUNK.min(grid.rows - chunk0);
        let mut lon: Vec<f64> = Vec::with_capacity(chunk_rows * grid.cols);
        let mut lat: Vec<f64> = Vec::with_capacity(chunk_rows * grid.cols);
        for r in chunk0..chunk0 + chunk_rows {
            lon.extend(grid.x_coordinates());
            lat.extend(std::iter::repeat_n(grid.y0 + r as f64 * grid.dy, grid.cols));
        }
        projection.to_lonlat(&mut lon, &mut lat)?;
        let chunk: Vec<Vec<(f32, f32, u8)>> = (0..chunk_rows)
            .into_par_iter()
            .map(|r| {
                (0..grid.cols)
                    .map(|c| {
                        let k = r * grid.cols + c;
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
                    .collect()
            })
            .collect();
        rows.extend(chunk);
    }
    let shape = (grid.rows, grid.cols);
    let pick = |k: usize| -> Vec<f32> {
        rows.iter()
            .flat_map(|r| r.iter().map(move |p| if k == 0 { p.0 } else { p.1 }))
            .collect()
    };
    let mask: Vec<u8> = rows.iter().flat_map(|r| r.iter().map(|p| p.2)).collect();
    Ok(StaticLayers {
        grid: *grid,
        los_east: Array2::from_shape_vec(shape, pick(0)).map_err(crate::error::product)?,
        los_north: Array2::from_shape_vec(shape, pick(1)).map_err(crate::error::product)?,
        layover_shadow_mask: Array2::from_shape_vec(shape, mask).map_err(crate::error::product)?,
    })
}

/// `(los_east, los_north, mask)` of one pixel, `None` outside the valid burst.
fn pixel(
    burst: &S1Burst,
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
    let inside = line >= burst.first_valid_line as f64
        && line <= burst.last_valid_line as f64
        && sample >= burst.first_valid_sample as f64
        && sample <= burst.last_valid_sample as f64;
    if !inside {
        return None;
    }
    let los = line_of_sight_enu(target, la, lo, radar.platform.position);
    let nearest = profiles
        .iter()
        .min_by(|a, b| (a.line - line).abs().total_cmp(&(b.line - line).abs()))?;
    Some((los[0] as f32, los[1] as f32, nearest.flag_at(target)))
}

fn trace_profiles(burst: &S1Burst, orbit: &Orbit, dem: &Dem) -> Result<Vec<Profile>> {
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

/// The layover/shadow flags along the ground profile of burst line `line`.
fn trace_profile(burst: &S1Burst, orbit: &Orbit, dem: &Dem, line: usize) -> Result<Profile> {
    let t = orbit.seconds(burst.sensing_start) + line as f64 * burst.azimuth_time_interval;
    let state = orbit.state(t)?;
    // A profile end outside the stack DEM is treated like a DEM hole (0 m): it only
    // places the profile line, and must not abort the burst's CSLC (#240).
    let height = |lon: f64, lat: f64| -> Result<f64> { Ok(profile_height(dem, lon, lat)) };
    let near = rdr2geo(t, burst.starting_range(), orbit, height)?;
    let far_range = burst.starting_range() + burst.samples as f64 * burst.range_pixel_spacing();
    let far = rdr2geo(t, far_range, orbit, height)?;
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

/// Write `layers` in the CSLC-S1-STATIC layout.
pub fn write_static(path: &std::path::Path, layers: &StaticLayers, burst_id: &str) -> Result<()> {
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
    let id = hdf5::types::FixedAscii::<64>::from_ascii(burst_id.as_bytes())
        .map_err(crate::error::product)?;
    ident
        .new_dataset::<hdf5::types::FixedAscii<64>>()
        .create("burst_id")?
        .write_scalar(&id)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
