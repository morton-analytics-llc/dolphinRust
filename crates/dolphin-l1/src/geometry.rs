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
}
