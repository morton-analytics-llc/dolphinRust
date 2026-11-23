//! Zero-Doppler range-Doppler geometry on the WGS84 ellipsoid.
//!
//! Sentinel-1 SLCs and NISAR RSLCs are focused to zero-Doppler geometry, so a
//! ground point `P` is imaged at the azimuth time `t` where the platform
//! velocity is perpendicular to the line of sight, `V(t)·(S(t) − P) = 0`, and
//! at slant range `|S(t) − P|`. `geo2rdr` solves that with Newton steps on the
//! Hermite-interpolated orbit (derivative `|V|² + A·(S − P)`).

use crate::error::{L1Error, Result};
use crate::orbit::{Orbit, PlatformState, Vec3};

/// WGS84 semi-major axis (m).
pub const WGS84_A: f64 = 6_378_137.0;
/// WGS84 first eccentricity squared.
pub const WGS84_E2: f64 = 6.694_379_990_141_316e-3;

/// Newton iterations allowed in `geo2rdr`.
const GEO2RDR_ITERATIONS: usize = 25;
/// `geo2rdr` convergence threshold on the azimuth-time step (s).
const GEO2RDR_THRESHOLD: f64 = 1e-9;

/// Geodetic latitude/longitude (radians) and ellipsoidal height (m) to ECEF.
pub fn llh_to_ecef(lat: f64, lon: f64, height: f64) -> Vec3 {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    [
        (n + height) * cos_lat * cos_lon,
        (n + height) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + height) * sin_lat,
    ]
}

/// ECEF to geodetic latitude/longitude (radians) and ellipsoidal height (m).
pub fn ecef_to_llh(p: Vec3) -> (f64, f64, f64) {
    let lon = p[1].atan2(p[0]);
    let rho = p[0].hypot(p[1]);
    let mut lat = p[2].atan2(rho * (1.0 - WGS84_E2));
    let mut height = 0.0;
    for _ in 0..6 {
        let sin_lat = lat.sin();
        let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
        height = rho / lat.cos() - n;
        lat = p[2].atan2(rho * (1.0 - WGS84_E2 * n / (n + height)));
    }
    (lat, lon, height)
}

pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// Radar coordinates of a ground point.
#[derive(Debug, Clone, Copy)]
pub struct RadarPoint {
    /// Zero-Doppler azimuth time, seconds since the orbit epoch.
    pub azimuth_time: f64,
    /// Slant range (m).
    pub slant_range: f64,
    /// Platform state at `azimuth_time`.
    pub platform: PlatformState,
}

/// Solve the zero-Doppler condition for `target` starting from `t_guess`
/// (seconds since the orbit epoch).
pub fn geo2rdr(target: Vec3, orbit: &Orbit, t_guess: f64) -> Result<RadarPoint> {
    let mut t = t_guess;
    for _ in 0..GEO2RDR_ITERATIONS {
        let state = orbit.state(t)?;
        let look = sub(state.position, target);
        let f = dot(state.velocity, look);
        let df = dot(state.velocity, state.velocity) + dot(state.acceleration, look);
        let step = f / df;
        t -= step;
        if step.abs() < GEO2RDR_THRESHOLD {
            let platform = orbit.state(t)?;
            return Ok(RadarPoint {
                azimuth_time: t,
                slant_range: norm(sub(platform.position, target)),
                platform,
            });
        }
    }
    Err(L1Error::Geometry(format!(
        "geo2rdr did not converge in {GEO2RDR_ITERATIONS} iterations"
    )))
}

/// Iterations allowed in `rdr2geo`.
const RDR2GEO_ITERATIONS: usize = 50;
/// `rdr2geo` convergence threshold on the height mismatch (m).
const RDR2GEO_THRESHOLD: f64 = 1e-5;

/// A ground point found by [`rdr2geo`].
#[derive(Debug, Clone, Copy)]
pub struct GroundPoint {
    /// Geodetic latitude (rad).
    pub lat: f64,
    /// Longitude (rad).
    pub lon: f64,
    /// Ellipsoidal height (m).
    pub height: f64,
    /// ECEF position (m).
    pub ecef: Vec3,
    /// Look angle from the platform's geocentric down direction (rad).
    pub look_angle: f64,
}

pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: Vec3) -> Vec3 {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// The right-looking zero-Doppler point at slant range `slant_range` from
/// the platform at `azimuth_time` (seconds since the orbit epoch) that lies on
/// the surface `height(lon_deg, lat_deg)`.
///
/// Every candidate `P + R (cos θ d̂ + sin θ r̂)`, with `d̂` the down direction
/// projected into the plane normal to the velocity and `r̂ = d̂ × v̂`, already
/// satisfies both range and zero Doppler, so the solve is one-dimensional in
/// the look angle θ: a secant iteration on `h_ellipsoid(θ) − h_dem(θ)`.
pub fn rdr2geo(
    azimuth_time: f64,
    slant_range: f64,
    orbit: &Orbit,
    height: impl Fn(f64, f64) -> Result<f64>,
) -> Result<GroundPoint> {
    let state = orbit.state(azimuth_time)?;
    let v = unit(state.velocity);
    let up = unit(state.position);
    let down = unit(sub([0.0; 3], sub(up, scale(v, dot(up, v)))));
    let right = cross(down, v);
    let point = |theta: f64| -> Result<(GroundPoint, f64)> {
        let (sin_t, cos_t) = theta.sin_cos();
        let ecef: Vec3 = std::array::from_fn(|k| {
            state.position[k] + slant_range * (cos_t * down[k] + sin_t * right[k])
        });
        let (lat, lon, h) = ecef_to_llh(ecef);
        let dem = height(lon.to_degrees(), lat.to_degrees())?;
        let found = GroundPoint {
            lat,
            lon,
            height: h,
            ecef,
            look_angle: theta,
        };
        Ok((found, h - dem))
    };
    // Start from the flat-ellipsoid look angle for a mid-swath guess.
    let (mut t0, mut t1) = (0.35_f64, 0.75_f64);
    let (mut f0, mut f1) = (point(t0)?.1, point(t1)?.1);
    for _ in 0..RDR2GEO_ITERATIONS {
        if (f1 - f0).abs() < f64::EPSILON {
            break;
        }
        let t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
        let (found, f2) = point(t2)?;
        if f2.abs() < RDR2GEO_THRESHOLD {
            return Ok(found);
        }
        (t0, f0, t1, f1) = (t1, f1, t2, f2);
    }
    Err(L1Error::Geometry(format!(
        "rdr2geo did not converge at t = {azimuth_time:.6} s, R = {slant_range:.3} m"
    )))
}

fn scale(a: Vec3, k: f64) -> Vec3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

/// East/north/up unit vectors at geodetic `(lat, lon)` (radians).
pub fn enu_basis(lat: f64, lon: f64) -> [Vec3; 3] {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    [
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ]
}

/// Ground-to-satellite unit vector in local east/north/up.
pub fn line_of_sight_enu(target: Vec3, lat: f64, lon: f64, platform: Vec3) -> Vec3 {
    let look = sub(platform, target);
    let range = norm(look);
    let unit = [look[0] / range, look[1] / range, look[2] / range];
    enu_basis(lat, lon).map(|axis| dot(axis, unit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llh_round_trip_is_sub_millimetre() {
        for &(lat, lon, h) in &[
            (39.4_f64, -112.5_f64, 1450.0),
            (-33.9, 151.2, -25.0),
            (78.2, 15.6, 30.0),
        ] {
            let p = llh_to_ecef(lat.to_radians(), lon.to_radians(), h);
            let (lat2, lon2, h2) = ecef_to_llh(p);
            assert!((lat2.to_degrees() - lat).abs() < 1e-10);
            assert!((lon2.to_degrees() - lon).abs() < 1e-10);
            assert!((h2 - h).abs() < 1e-4, "{h2} vs {h}");
        }
    }

    /// rdr2geo then geo2rdr on a circular orbit returns the radar
    /// coordinates it started from, and the point sits on the surface.
    #[test]
    fn rdr2geo_inverts_geo2rdr() {
        let r = 7.07e6;
        let omega = 1.06e-3;
        let incl = 98.2_f64.to_radians();
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
        let surface = |lon: f64, lat: f64| Ok(120.0 + 30.0 * (lon * 7.0).sin() * (lat * 5.0).cos());
        for &(t, range) in &[(150.0, 8.5e5), (200.0, 9.3e5), (250.0, 7.9e5)] {
            let ground = rdr2geo(t, range, &orbit, surface).unwrap();
            let expected = surface(ground.lon.to_degrees(), ground.lat.to_degrees()).unwrap();
            assert!((ground.height - expected).abs() < 1e-4);
            let back = geo2rdr(ground.ecef, &orbit, t + 1.0).unwrap();
            assert!(
                (back.azimuth_time - t).abs() < 1e-8,
                "{}",
                back.azimuth_time - t
            );
            assert!((back.slant_range - range).abs() < 1e-5);
        }
    }
}
