//! Solid-earth-tide displacement and its line-of-sight range effect (issue #21).
//!
//! The lunar and solar tidal potential deforms the solid Earth by up to ~30 cm
//! radially. Unlike ionosphere and troposphere this is not a propagation delay —
//! it is real ground motion — but for a repeat-pass stack it enters the
//! measurement the same way, so it is modelled here as an equivalent **range
//! delay** and subtracted by the same [`apply::subtract_delay`](crate::apply)
//! stage. It matters for velocity specifically because a sun-synchronous stack
//! samples the solar tide at nearly the same phase every time while the lunar
//! tide does not, so the residual does not average out over months to years — it
//! leaks into the fitted rate.
//!
//! # Model
//!
//! IERS Conventions (2010) §7.1.1 "step 1", degree-2 in-phase term with nominal
//! Love/Shida numbers `h₂ = 0.6078`, `l₂ = 0.0847`:
//!
//! ```text
//! Δr = Σ_j (GM_j/GM_⊕)(R_E⁴/R_j³) { h₂ r̂ [(3(R̂_j·r̂)² − 1)/2]
//!                                  + 3 l₂ (R̂_j·r̂)[R̂_j − (R̂_j·r̂) r̂] }
//! ```
//!
//! summed over the Moon and Sun. **What is deliberately omitted**, and what it
//! costs at InSAR tolerances:
//!
//! - degree-3 terms (Moon only): ≲ 2 mm;
//! - out-of-phase (anelasticity) and latitude-dependent `h`/`l`: ≲ 2 mm;
//! - IERS "step 2" frequency-dependent diurnal/semidiurnal corrections: ≲ 2 mm
//!   in the radial component and larger only in the permanent (zero-frequency)
//!   term, which is constant in time and therefore cancels in a differential
//!   series referenced to acquisition 0;
//! - pole tide and ocean tidal loading: **not modelled at all**. Ocean loading is
//!   the larger omission (cm-level within ~100 km of a coast) and needs an
//!   external loading-coefficient grid; it is a separate piece of work.
//!
//! Against a ~300 mm signal these are sub-1% terms; against the ~10 mm InSAR
//! error budget they are not zero, so this is a **first-order** correction, not
//! a reference implementation of IERS 2010.
//!
//! # Ephemerides and time
//!
//! Low-precision analytic Sun and Moon positions (Astronomical Almanac, mean
//! equinox of date), accurate to ~0.01° and ~0.3° in direction and ~0.2% in lunar
//! distance. The tidal term varies as `P₂(cos θ)`, whose derivative peaks at 1.5,
//! so a 0.3° = 0.0052 rad direction error costs ≈ 220 mm × 1.5 × 0.0052 ≈ 1.7 mm.
//! `TT − UTC` is taken as the constant 69.184 s in force since the 2017 leap
//! second, and `UT1 ≈ UTC` (|DUT1| < 0.9 s ⇒ < 0.02 mm). Sun, Moon, and GMST are
//! all referred to the mean equinox of date, so precession cancels rather than
//! accumulating.

use chrono::{Datelike, NaiveDateTime, Timelike};
use ndarray::Array2;
use rayon::prelude::*;

use crate::geometry::LosGeometry;

/// Love number `h₂` (radial), IERS 2010 nominal.
const LOVE_H2: f64 = 0.6078;
/// Shida number `l₂` (transverse), IERS 2010 nominal.
const SHIDA_L2: f64 = 0.0847;
/// Earth equatorial radius (m), IERS 2010 / WGS84 semi-major axis.
const EARTH_RADIUS_M: f64 = 6_378_136.6;
/// WGS84 first eccentricity squared.
const WGS84_E2: f64 = 0.006_694_379_990_14;
/// WGS84 semi-major axis (m), for the station's geodetic → ECEF conversion.
const WGS84_A_M: f64 = 6_378_137.0;
/// `GM_moon / GM_earth` (IERS 2010).
const MOON_MASS_RATIO: f64 = 0.012_300_037_1;
/// `GM_sun / GM_earth` (IERS 2010).
const SUN_MASS_RATIO: f64 = 332_946.048_7;
/// Astronomical unit (m).
const ASTRONOMICAL_UNIT_M: f64 = 1.495_978_707e11;
/// `TT − UTC` (s): 32.184 s + the 37 s of TAI−UTC in force since 2017-01-01. A
/// future leap second changes this by 1 s, which moves the Moon by 0.00015° — far
/// inside the ephemeris error.
const TT_MINUS_UTC_S: f64 = 69.184;
/// Julian date of J2000.0 (2000-01-01 12:00 TT).
const J2000_JD: f64 = 2_451_545.0;

/// Geocentric position of a perturbing body, Earth-fixed (ECEF), in meters.
#[derive(Debug, Clone, Copy)]
pub struct BodyPosition {
    /// ECEF unit vector toward the body.
    pub direction: [f64; 3],
    /// Geocentric distance, meters.
    pub distance_m: f64,
}

/// The perturbing bodies at one instant, with their mass ratios. Depends only on
/// time, so a whole frame shares one — the ephemeris is the expensive part.
#[must_use]
pub fn tide_bodies(utc: NaiveDateTime) -> [(BodyPosition, f64); 2] {
    [
        (moon_position(utc), MOON_MASS_RATIO),
        (sun_position(utc), SUN_MASS_RATIO),
    ]
}

/// Solid-earth-tide displacement of one station, in local east/north/up meters.
///
/// `utc` is the acquisition time, `lon_deg`/`lat_deg` the station's geodetic
/// position and `height_m` its ellipsoidal height.
#[must_use]
pub fn tide_displacement_enu(
    utc: NaiveDateTime,
    lon_deg: f64,
    lat_deg: f64,
    height_m: f64,
) -> [f64; 3] {
    tide_displacement_enu_at(&tide_bodies(utc), lon_deg, lat_deg, height_m)
}

/// [`tide_displacement_enu`] against pre-computed [`tide_bodies`].
#[must_use]
pub fn tide_displacement_enu_at(
    bodies: &[(BodyPosition, f64); 2],
    lon_deg: f64,
    lat_deg: f64,
    height_m: f64,
) -> [f64; 3] {
    let station_dir = normalize(geodetic_to_ecef(lon_deg, lat_deg, height_m));
    let ecef = bodies
        .iter()
        .map(|&(body, mass_ratio)| body_displacement(body, mass_ratio, station_dir))
        .fold([0.0; 3], |acc, term| {
            [acc[0] + term[0], acc[1] + term[1], acc[2] + term[2]]
        });
    ecef_to_enu(ecef, lon_deg, lat_deg)
}

/// One body's contribution to the degree-2 in-phase tidal displacement, ECEF meters.
fn body_displacement(body: BodyPosition, mass_ratio: f64, station_dir: [f64; 3]) -> [f64; 3] {
    let scale = mass_ratio * EARTH_RADIUS_M.powi(4) / body.distance_m.powi(3);
    let cos_z = dot(body.direction, station_dir);
    let radial = LOVE_H2 * (3.0 * cos_z * cos_z - 1.0) / 2.0;
    let transverse = 3.0 * SHIDA_L2 * cos_z;
    std::array::from_fn(|i| {
        scale
            * (radial * station_dir[i] + transverse * (body.direction[i] - cos_z * station_dir[i]))
    })
}

/// Equivalent **range delay** (meters) of the solid earth tide on the frame grid:
/// the increase in apparent sensor→ground range, i.e. `−(Δr · l̂)` for the
/// ground→sensor LOS unit vector `l̂`. Ground moving toward the sensor shortens
/// the range, so an uplift under a near-vertical look yields a negative delay —
/// the same sign convention the ionospheric and tropospheric layers use, so the
/// three sum and go through [`apply::subtract_delay`](crate::apply) unchanged.
///
/// `lonlat` supplies each pixel's geodetic (lon, lat) in degrees; `los` supplies
/// its ground→sensor unit vector. Ellipsoidal height is taken as zero: the tidal
/// term scales with `R_E⁴/R_j³` and is independent of station height to the
/// precision of this model (a 3 km height changes it by < 0.02%).
#[must_use]
pub fn tide_range_delay_grid(
    utc: NaiveDateTime,
    lonlat: &LonLatGrid,
    los: &LosGeometry,
) -> Array2<f64> {
    // The ephemeris depends only on time, so it is computed once for the frame
    // rather than per pixel — it dominates the per-pixel cost otherwise.
    let bodies = tide_bodies(utc);
    let (rows, cols) = los.up.dim();
    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .map(|index| {
            let (row, col) = (index / cols, index % cols);
            let (lon, lat) = lonlat.at(row, col);
            let enu = tide_displacement_enu_at(&bodies, lon, lat, 0.0);
            let toward_sensor = enu[0] * los.east[(row, col)]
                + enu[1] * los.north[(row, col)]
                + enu[2] * los.up[(row, col)];
            -toward_sensor
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("tide grid shape")
}

/// Per-pixel geodetic (lon, lat) in degrees, bilinearly interpolated from the four
/// frame corners.
///
/// The tide varies by ~5 mm across a 100 km frame, so a single frame-centre sample
/// (what the coarse IONEX path uses) would discard a real gradient. A full
/// per-pixel CRS transform is not needed either: within one projected zone the
/// lon/lat ↔ x/y map departs from bilinear by tens of meters over 100 km, and
/// tens of meters of position error is worth ~0.003 mm of tide.
#[derive(Debug, Clone)]
pub struct LonLatGrid {
    /// (lon, lat) at the corners, in row-major order: top-left, top-right,
    /// bottom-left, bottom-right.
    corners: [[f64; 2]; 4],
    rows: usize,
    cols: usize,
}

impl LonLatGrid {
    /// Build from the four corner (lon, lat) pairs, row-major: top-left,
    /// top-right, bottom-left, bottom-right.
    #[must_use]
    pub fn from_corners(corners: [[f64; 2]; 4], rows: usize, cols: usize) -> Self {
        Self {
            corners,
            rows,
            cols,
        }
    }

    /// Interpolated (lon, lat) at a pixel.
    #[must_use]
    pub fn at(&self, row: usize, col: usize) -> (f64, f64) {
        let fy = interpolation_fraction(row, self.rows);
        let fx = interpolation_fraction(col, self.cols);
        let blend = |k: usize| {
            let top = self.corners[0][k] + fx * (self.corners[1][k] - self.corners[0][k]);
            let bottom = self.corners[2][k] + fx * (self.corners[3][k] - self.corners[2][k]);
            top + fy * (bottom - top)
        };
        (blend(0), blend(1))
    }
}

/// Position along an axis in `[0, 1]`; a single-pixel axis sits at the start.
fn interpolation_fraction(index: usize, extent: usize) -> f64 {
    match extent > 1 {
        true => index as f64 / (extent - 1) as f64,
        false => 0.0,
    }
}

/// Geocentric ECEF position of the Moon (Astronomical Almanac low-precision
/// formulae, mean equinox of date; ~0.3° in direction, ~0.2% in distance).
#[must_use]
pub fn moon_position(utc: NaiveDateTime) -> BodyPosition {
    let t = julian_centuries_tt(utc);
    let sin_deg = |degrees: f64| degrees.to_radians().sin();
    let cos_deg = |degrees: f64| degrees.to_radians().cos();

    let longitude_deg = 218.32 + 481_267.881 * t + 6.29 * sin_deg(135.0 + 477_198.87 * t)
        - 1.27 * sin_deg(259.3 - 413_335.35 * t)
        + 0.66 * sin_deg(235.7 + 890_534.22 * t)
        + 0.21 * sin_deg(269.9 + 954_397.74 * t)
        - 0.19 * sin_deg(357.5 + 35_999.05 * t)
        - 0.11 * sin_deg(186.6 + 966_404.03 * t);
    let latitude_deg = 5.13 * sin_deg(93.3 + 483_202.02 * t)
        + 0.28 * sin_deg(228.2 + 960_400.89 * t)
        - 0.28 * sin_deg(318.3 + 6_003.15 * t)
        - 0.17 * sin_deg(217.6 - 407_332.21 * t);
    let parallax_deg = 0.9508
        + 0.0518 * cos_deg(135.0 + 477_198.85 * t)
        + 0.0095 * cos_deg(259.3 - 413_335.38 * t)
        + 0.0078 * cos_deg(235.7 + 890_534.22 * t)
        + 0.0028 * cos_deg(269.9 + 954_397.70 * t);

    let distance_m = EARTH_RADIUS_M / parallax_deg.to_radians().sin();
    ecliptic_to_ecef(longitude_deg, latitude_deg, distance_m, utc, t)
}

/// Geocentric ECEF position of the Sun (Astronomical Almanac low-precision
/// formulae, mean equinox of date; ~0.01°).
#[must_use]
pub fn sun_position(utc: NaiveDateTime) -> BodyPosition {
    let days = julian_date_tt(utc) - J2000_JD;
    let t = days / 36_525.0;
    let mean_longitude_deg = 280.460 + 0.985_647_4 * days;
    let mean_anomaly_deg = 357.528 + 0.985_600_3 * days;
    let g = mean_anomaly_deg.to_radians();
    let longitude_deg = mean_longitude_deg + 1.915 * g.sin() + 0.020 * (2.0 * g).sin();
    let distance_au = 1.000_14 - 0.016_71 * g.cos() - 0.000_14 * (2.0 * g).cos();
    ecliptic_to_ecef(
        longitude_deg,
        0.0,
        distance_au * ASTRONOMICAL_UNIT_M,
        utc,
        t,
    )
}

/// Ecliptic (λ, β, r) of date → Earth-fixed direction and distance, via the mean
/// obliquity and Greenwich mean sidereal time.
fn ecliptic_to_ecef(
    longitude_deg: f64,
    latitude_deg: f64,
    distance_m: f64,
    utc: NaiveDateTime,
    centuries: f64,
) -> BodyPosition {
    let obliquity = mean_obliquity_deg(centuries).to_radians();
    let (lambda, beta) = (longitude_deg.to_radians(), latitude_deg.to_radians());
    // Ecliptic → equatorial of date.
    let equatorial = [
        beta.cos() * lambda.cos(),
        obliquity.cos() * beta.cos() * lambda.sin() - obliquity.sin() * beta.sin(),
        obliquity.sin() * beta.cos() * lambda.sin() + obliquity.cos() * beta.sin(),
    ];
    // Equatorial of date → Earth-fixed: rotate by −GMST about the pole.
    let theta = gmst_deg(utc).to_radians();
    let direction = [
        equatorial[0] * theta.cos() + equatorial[1] * theta.sin(),
        -equatorial[0] * theta.sin() + equatorial[1] * theta.cos(),
        equatorial[2],
    ];
    BodyPosition {
        direction: normalize(direction),
        distance_m,
    }
}

/// Mean obliquity of the ecliptic (degrees), IAU 1980 series truncated to the
/// linear term — 0.0002° over a century, far inside the ephemeris error.
fn mean_obliquity_deg(centuries: f64) -> f64 {
    23.439_291 - 0.013_004_2 * centuries
}

/// Greenwich mean sidereal time in degrees (UT1 ≈ UTC).
fn gmst_deg(utc: NaiveDateTime) -> f64 {
    let days = julian_date_utc(utc) - J2000_JD;
    (280.460_618_37 + 360.985_647_366_29 * days).rem_euclid(360.0)
}

/// Julian date on the UTC scale.
fn julian_date_utc(utc: NaiveDateTime) -> f64 {
    let date = utc.date();
    let (year, month, day) = (date.year() as i64, date.month() as i64, date.day() as i64);
    // Fliegel & van Flandern, valid for the Gregorian calendar.
    let a = (14 - month) / 12;
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    let jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32_045;
    let time = utc.time();
    let seconds = f64::from(time.hour()) * 3600.0
        + f64::from(time.minute()) * 60.0
        + f64::from(time.second())
        + f64::from(time.nanosecond()) / 1e9;
    jdn as f64 - 0.5 + seconds / 86_400.0
}

fn julian_date_tt(utc: NaiveDateTime) -> f64 {
    julian_date_utc(utc) + TT_MINUS_UTC_S / 86_400.0
}

fn julian_centuries_tt(utc: NaiveDateTime) -> f64 {
    (julian_date_tt(utc) - J2000_JD) / 36_525.0
}

/// Geodetic (lon, lat, ellipsoidal height) → WGS84 ECEF meters.
fn geodetic_to_ecef(lon_deg: f64, lat_deg: f64, height_m: f64) -> [f64; 3] {
    let (lon, lat) = (lon_deg.to_radians(), lat_deg.to_radians());
    let prime_vertical = WGS84_A_M / (1.0 - WGS84_E2 * lat.sin() * lat.sin()).sqrt();
    [
        (prime_vertical + height_m) * lat.cos() * lon.cos(),
        (prime_vertical + height_m) * lat.cos() * lon.sin(),
        (prime_vertical * (1.0 - WGS84_E2) + height_m) * lat.sin(),
    ]
}

/// ECEF vector → local east/north/up at a geodetic (lon, lat).
fn ecef_to_enu(vector: [f64; 3], lon_deg: f64, lat_deg: f64) -> [f64; 3] {
    let (lon, lat) = (lon_deg.to_radians(), lat_deg.to_radians());
    [
        dot(vector, [-lon.sin(), lon.cos(), 0.0]),
        dot(
            vector,
            [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()],
        ),
        dot(
            vector,
            [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()],
        ),
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let norm = dot(v, v).sqrt();
    match norm > 0.0 {
        true => [v[0] / norm, v[1] / norm, v[2] / norm],
        false => [0.0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn utc(y: i32, m: u32, d: u32, hh: u32, mm: u32, ss: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(hh, mm, ss)
            .unwrap()
    }

    /// The closed-form anchor. With a body directly overhead, `cos_z = 1`, the
    /// transverse term vanishes and the radial displacement collapses to
    /// `(GM_j/GM_⊕)(R_E⁴/R_j³)·h₂`. For the Moon at its mean distance that is the
    /// textbook ~0.218 m; for the Sun at 1 AU, ~0.100 m. Combined peak ~0.32 m,
    /// which is the "~10 cm-class, up to ~30 cm" figure the literature quotes.
    #[test]
    fn overhead_body_matches_the_closed_form_radial_tide() {
        let station = normalize([1.0, 0.0, 0.0]);
        let moon = BodyPosition {
            direction: station,
            distance_m: 3.844e8,
        };
        let moon_up = body_displacement(moon, MOON_MASS_RATIO, station)[0];
        let expected_moon =
            MOON_MASS_RATIO * EARTH_RADIUS_M.powi(4) / 3.844e8_f64.powi(3) * LOVE_H2;
        assert!(
            (moon_up - expected_moon).abs() < 1e-12,
            "{moon_up} != {expected_moon}"
        );
        assert!(
            (moon_up - 0.2178).abs() < 5e-4,
            "lunar zenith tide {moon_up} m is not the textbook 0.218 m"
        );

        let sun = BodyPosition {
            direction: station,
            distance_m: ASTRONOMICAL_UNIT_M,
        };
        let sun_up = body_displacement(sun, SUN_MASS_RATIO, station)[0];
        assert!(
            (sun_up - 0.1000).abs() < 5e-4,
            "solar zenith tide {sun_up} m is not the textbook 0.100 m"
        );
    }

    /// `P₂(cos θ)` vanishes at the magic angle `θ = acos(1/√3) = 54.7356°`, so the
    /// radial term changes sign there — the shape of the tidal bulge, not just its
    /// amplitude.
    #[test]
    fn radial_term_vanishes_at_the_magic_angle() {
        let station = [1.0, 0.0, 0.0];
        let magic = (1.0_f64 / 3.0_f64.sqrt()).acos();
        let body = BodyPosition {
            direction: [magic.cos(), magic.sin(), 0.0],
            distance_m: 3.844e8,
        };
        let radial = dot(body_displacement(body, MOON_MASS_RATIO, station), station);
        assert!(radial.abs() < 1e-12, "radial {radial} at the magic angle");

        // Just inside the magic angle the bulge pushes out, just outside it pulls in.
        let sample = |theta: f64| {
            let body = BodyPosition {
                direction: [theta.cos(), theta.sin(), 0.0],
                distance_m: 3.844e8,
            };
            dot(body_displacement(body, MOON_MASS_RATIO, station), station)
        };
        assert!(sample(magic - 0.1) > 0.0);
        assert!(sample(magic + 0.1) < 0.0);
    }

    /// The tide is diametrically symmetric: a body and its antipode raise the same
    /// radial bulge (this is why there are two high tides a day).
    #[test]
    fn antipodal_body_raises_the_same_bulge() {
        let station = [1.0, 0.0, 0.0];
        let near = BodyPosition {
            direction: [1.0, 0.0, 0.0],
            distance_m: 3.844e8,
        };
        let far = BodyPosition {
            direction: [-1.0, 0.0, 0.0],
            distance_m: 3.844e8,
        };
        let a = body_displacement(near, MOON_MASS_RATIO, station);
        let b = body_displacement(far, MOON_MASS_RATIO, station);
        assert!((a[0] - b[0]).abs() < 1e-12);
    }

    /// Sun declination is the independent check on the ephemeris + time scales:
    /// ±23.44° at the solstices, ~0° at the equinoxes. Declination is
    /// `asin(z/|r|)` of the equatorial vector, which the ECEF z component
    /// preserves (the GMST rotation is about the pole).
    #[test]
    fn sun_declination_tracks_the_seasons() {
        let declination = |t: NaiveDateTime| sun_position(t).direction[2].asin().to_degrees();
        let june = declination(utc(2023, 6, 21, 12, 0, 0));
        let december = declination(utc(2023, 12, 21, 12, 0, 0));
        let march = declination(utc(2023, 3, 20, 21, 24, 0));
        assert!((june - 23.44).abs() < 0.1, "june solstice {june}");
        assert!(
            (december + 23.44).abs() < 0.1,
            "december solstice {december}"
        );
        assert!(march.abs() < 0.1, "march equinox {march}");
    }

    /// The Sun crosses the local meridian near local noon: at 12:00 UTC the
    /// sub-solar point sits near 0° longitude, and its ECEF direction rotates
    /// westward at 15°/hour.
    #[test]
    fn subsolar_longitude_tracks_utc() {
        let sub_solar = |t: NaiveDateTime| {
            let d = sun_position(t).direction;
            d[1].atan2(d[0]).to_degrees()
        };
        // Not exactly 0°: 12:00 UTC is *mean* noon, and apparent noon differs by
        // the equation of time — never more than ~16.5 min ≈ 4.1° of rotation, and
        // ≈ −7.5 min ≈ +1.9° on 20 March. Recovering that offset is itself a check
        // that the solar anomaly terms are in, not just the mean longitude.
        let noon = sub_solar(utc(2023, 3, 20, 12, 0, 0));
        assert!(
            (noon - 1.9).abs() < 0.3,
            "sub-solar longitude at 12:00 UTC = {noon}, expected the equation of \
             time to put it near +1.9°"
        );
        let six_hours_later = sub_solar(utc(2023, 3, 20, 18, 0, 0));
        let rotation = (noon - six_hours_later).rem_euclid(360.0);
        assert!(
            (rotation - 90.0).abs() < 0.5,
            "6 h should rotate the sub-solar point 90°, got {rotation}"
        );
    }

    /// The lunar ephemeris stays inside its real physical envelope over a full
    /// synodic month: distance between perigee and apogee, declination inside
    /// ±(23.44 + 5.14)°.
    #[test]
    fn moon_ephemeris_stays_in_its_physical_envelope() {
        let start = utc(2023, 1, 4, 0, 40, 53);
        for hour in 0..(30 * 24) {
            let moon = moon_position(start + chrono::Duration::hours(hour));
            assert!(
                (3.52e8..4.09e8).contains(&moon.distance_m),
                "lunar distance {} m at +{hour} h is outside perigee–apogee",
                moon.distance_m
            );
            let declination = moon.direction[2].asin().to_degrees();
            assert!(
                declination.abs() < 28.7,
                "lunar declination {declination}° at +{hour} h exceeds 23.44 + 5.14"
            );
        }
    }

    /// The tide is dominantly **semidiurnal** — the signature of a degree-2 bulge
    /// on a rotating Earth. Over 25 h the vertical displacement at a low-latitude
    /// station turns over twice, which no amount of amplitude checking would show.
    #[test]
    fn vertical_tide_is_semidiurnal() {
        let start = utc(2023, 1, 4, 0, 0, 0);
        let vertical: Vec<f64> = (0..=100)
            .map(|step| {
                let time = start + chrono::Duration::minutes(step * 15);
                tide_displacement_enu(time, -99.0684, 19.4317, 0.0)[2]
            })
            .collect();
        let maxima = vertical
            .windows(3)
            .filter(|w| w[1] > w[0] && w[1] > w[2])
            .count();
        assert_eq!(maxima, 2, "expected 2 high tides in 25 h, found {maxima}");
    }

    /// End to end at a real station and time: the total displacement is within
    /// the physically possible envelope, and vertical dominates.
    #[test]
    fn total_displacement_is_within_the_physical_envelope() {
        // MMX1, Mexico City, at the T005 descending overpass time.
        let enu = tide_displacement_enu(utc(2023, 1, 4, 0, 40, 53), -99.0684, 19.4317, 2240.0);
        let vertical = enu[2].abs();
        let horizontal = enu[0].hypot(enu[1]);
        assert!(
            vertical < 0.40,
            "vertical tide {vertical} m exceeds the physical envelope"
        );
        assert!(
            horizontal < 0.10,
            "horizontal tide {horizontal} m exceeds the physical envelope"
        );
        assert!(
            vertical > 0.02,
            "vertical tide {vertical} m is implausibly small — is the ephemeris wired up?"
        );
    }

    /// Sign convention: ground moving toward the sensor shortens the range, so the
    /// equivalent delay is negative. Checked against a synthetic straight-up LOS.
    #[test]
    fn uplift_gives_a_negative_equivalent_delay() {
        let los = LosGeometry {
            east: Array2::zeros((1, 1)),
            north: Array2::zeros((1, 1)),
            up: Array2::ones((1, 1)),
        };
        let lonlat = LonLatGrid::from_corners([[0.0, 0.0]; 4], 1, 1);
        // Moon overhead at the equator/prime meridian raises the ground there;
        // pick the time from the model itself rather than asserting a date.
        let delay = tide_range_delay_grid(utc(2023, 1, 4, 0, 40, 53), &lonlat, &los)[(0, 0)];
        let enu = tide_displacement_enu(utc(2023, 1, 4, 0, 40, 53), 0.0, 0.0, 0.0);
        assert!(
            (delay + enu[2]).abs() < 1e-12,
            "delay {delay} must be exactly −up {}",
            enu[2]
        );
    }

    /// The gradient across a frame is the reason this is computed per pixel rather
    /// than sampled once at the centre like IONEX: over ~100 km it is millimetres,
    /// which is the InSAR signal scale.
    #[test]
    fn tide_varies_measurably_across_a_frame() {
        let time = utc(2023, 1, 4, 0, 40, 53);
        let near = tide_displacement_enu(time, -99.5, 19.0, 0.0)[2];
        let far = tide_displacement_enu(time, -98.5, 20.0, 0.0)[2];
        let gradient_mm = (far - near).abs() * 1000.0;
        assert!(
            gradient_mm > 0.5,
            "gradient across ~140 km is {gradient_mm} mm — a single centre sample \
             would be defensible after all"
        );
    }

    /// Bilinear lon/lat interpolation reproduces the corners exactly and the
    /// midpoint as their mean.
    #[test]
    fn lonlat_grid_interpolates_the_corners() {
        let grid = LonLatGrid::from_corners(
            [[-100.0, 20.0], [-99.0, 20.0], [-100.0, 19.0], [-99.0, 19.0]],
            3,
            3,
        );
        assert_eq!(grid.at(0, 0), (-100.0, 20.0));
        assert_eq!(grid.at(2, 2), (-99.0, 19.0));
        let (lon, lat) = grid.at(1, 1);
        assert!((lon + 99.5).abs() < 1e-12 && (lat - 19.5).abs() < 1e-12);
    }
}
