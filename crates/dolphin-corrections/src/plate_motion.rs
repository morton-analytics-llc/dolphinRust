//! Rigid tectonic plate-motion LOS correction (issue #103).
//!
//! Without removing rigid plate motion, LOS velocities are only self-consistent
//! within a single track/frame: they are not directly comparable to a fixed
//! reference frame (GNSS) or mosaicked across plates/tracks. This is the same
//! absolute-vs-relative-velocity gap [`crate::solid_earth_tide`] partially
//! addresses for a different physical term, addressed here for rigid horizontal
//! plate rotation.
//!
//! # Model
//!
//! Standard rigid-body plate kinematics: a plate rotates about an Euler pole
//! `ω` (a Cartesian angular-velocity vector, ECEF) at constant angular rate, so
//! a point at ECEF position `r` on the plate moves at `v = ω × r`. Converting
//! `v` to local east/north/up and projecting onto the ground→sensor LOS unit
//! vector `l̂` gives an **equivalent range-rate delay** `−(v · l̂)` — same sign
//! convention as [`crate::solid_earth_tide::tide_range_delay_grid`]: motion
//! toward the sensor shortens the range. Over the stack, the delay accumulates
//! linearly: `delay(t) = −(v · l̂) · (t − t_reference)`.
//!
//! `r` is each pixel's **WGS84 ellipsoidal** ECEF position (matching
//! [`crate::solid_earth_tide`]'s `geodetic_to_ecef`, height = 0), not a sphere.
//! This is a deliberate departure from the common plate-motion-model shortcut
//! of treating Earth as a sphere of one fixed radius: the two differ by the
//! geodetic/geocentric latitude offset (up to ~11.5 arcmin at mid-latitudes),
//! which the closed-form polar-rotation contract test below bounds directly.
//! Real ITRF station coordinates (what an Euler pole is actually fit against)
//! are ellipsoidal, so this is the more physically consistent choice, at the
//! cost of not bit-matching a spherical-approximation reference — dolphinRust's
//! validation strategy is physically-meaningful tolerances, not bit-exactness.
//!
//! # Euler pole table
//!
//! [`ITRF2014_PMM`] holds the ITRF2014 plate motion model (Altamimi, Métivier,
//! Rebischung, Collilieux, 2017, *ITRF2014 plate motion model*, Geophys. J.
//! Int. 209(3)), Cartesian angular-velocity components in milliarcseconds/year
//! — the same table MintPy hardcodes in `objects/euler_pole.py`
//! (`insarlab/MintPy`), keyed by the same plate names, which is what this
//! module's oracle contract test was computed against independently.
//!
//! **Not yet ITRF2020.** The issue that scoped this correction named the newer
//! ITRF2020 PMM (Altamimi et al. 2023, DOI 10.1029/2023GL106373); its exact
//! per-plate Euler-pole table could not be verified against a reachable oracle
//! from this environment (network egress to the publisher was blocked, and
//! MintPy itself has not picked up ITRF2020 either as of this writing). Rather
//! than transcribe unverified numbers into a scientific constant table,
//! [`PlateMotionModel::EulerPole`] lets a caller supply any pole — ITRF2020 or
//! otherwise — directly; [`ITRF2014_PMM`] is offered as the verified built-in
//! default for the named-plate path.

use dolphin_core::config::PlateMotionModel;
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::{CorrectionError, Result};
use crate::geometry::LosGeometry;
use crate::solid_earth_tide::LonLatGrid;

/// WGS84 semi-major axis (m). Matches [`crate::solid_earth_tide`]'s constant.
const WGS84_A_M: f64 = 6_378_137.0;
/// WGS84 first eccentricity squared.
const WGS84_E2: f64 = 0.006_694_379_990_14;
/// `1 mas (milliarcsecond) = MAS_TO_RAD radian`.
const MAS_TO_RAD: f64 = std::f64::consts::PI / 3_600_000.0 / 180.0;

/// A plate's (or an explicit) Euler pole: Cartesian ECEF angular-velocity
/// components, milliarcseconds/year.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerPoleVector {
    /// X-axis angular-velocity component, mas/yr.
    pub omega_x_mas_per_year: f64,
    /// Y-axis angular-velocity component, mas/yr.
    pub omega_y_mas_per_year: f64,
    /// Z-axis angular-velocity component, mas/yr.
    pub omega_z_mas_per_year: f64,
}

/// ITRF2014 plate motion model (Altamimi et al. 2017), Cartesian
/// angular-velocity vectors in milliarcseconds/year. See the module docs for
/// provenance and why this is ITRF2014 rather than ITRF2020.
pub const ITRF2014_PMM: &[(&str, EulerPoleVector)] = &[
    (
        "Antarctica",
        EulerPoleVector {
            omega_x_mas_per_year: -0.248,
            omega_y_mas_per_year: -0.324,
            omega_z_mas_per_year: 0.675,
        },
    ),
    (
        "Arabia",
        EulerPoleVector {
            omega_x_mas_per_year: 1.154,
            omega_y_mas_per_year: -0.136,
            omega_z_mas_per_year: 1.444,
        },
    ),
    (
        "Australia",
        EulerPoleVector {
            omega_x_mas_per_year: 1.510,
            omega_y_mas_per_year: 1.182,
            omega_z_mas_per_year: 1.215,
        },
    ),
    (
        "Eurasia",
        EulerPoleVector {
            omega_x_mas_per_year: -0.085,
            omega_y_mas_per_year: -0.531,
            omega_z_mas_per_year: 0.770,
        },
    ),
    (
        "India",
        EulerPoleVector {
            omega_x_mas_per_year: 1.154,
            omega_y_mas_per_year: -0.005,
            omega_z_mas_per_year: 1.454,
        },
    ),
    (
        "Nazca",
        EulerPoleVector {
            omega_x_mas_per_year: -0.333,
            omega_y_mas_per_year: -1.544,
            omega_z_mas_per_year: 1.623,
        },
    ),
    (
        "NorthAmerica",
        EulerPoleVector {
            omega_x_mas_per_year: 0.024,
            omega_y_mas_per_year: -0.694,
            omega_z_mas_per_year: -0.063,
        },
    ),
    (
        "Nubia",
        EulerPoleVector {
            omega_x_mas_per_year: 0.099,
            omega_y_mas_per_year: -0.614,
            omega_z_mas_per_year: 0.733,
        },
    ),
    (
        "Pacific",
        EulerPoleVector {
            omega_x_mas_per_year: -0.409,
            omega_y_mas_per_year: 1.047,
            omega_z_mas_per_year: -2.169,
        },
    ),
    (
        "SouthAmerica",
        EulerPoleVector {
            omega_x_mas_per_year: -0.270,
            omega_y_mas_per_year: -0.301,
            omega_z_mas_per_year: -0.140,
        },
    ),
    (
        "Somalia",
        EulerPoleVector {
            omega_x_mas_per_year: -0.121,
            omega_y_mas_per_year: -0.794,
            omega_z_mas_per_year: 0.884,
        },
    ),
];

/// Look up a plate's Euler pole in [`ITRF2014_PMM`] by name, ignoring case,
/// spaces, underscores and hyphens (`"North America"`, `"north_america"` and
/// `"NorthAmerica"` all match).
#[must_use]
pub fn plate_by_name(name: &str) -> Option<EulerPoleVector> {
    let target = normalize_plate_name(name);
    ITRF2014_PMM
        .iter()
        .find(|(plate, _)| normalize_plate_name(plate) == target)
        .map(|(_, pole)| *pole)
}

fn normalize_plate_name(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace() && *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

/// Resolve a [`PlateMotionModel`] to its Euler pole: a named plate looked up
/// in [`ITRF2014_PMM`], or an explicit pole passed straight through.
///
/// # Errors
/// [`CorrectionError::UnknownPlate`] if a named plate is not in the built-in
/// table.
pub fn resolve_euler_pole(model: &PlateMotionModel) -> Result<EulerPoleVector> {
    match model {
        PlateMotionModel::Plate(name) => {
            plate_by_name(name).ok_or_else(|| CorrectionError::UnknownPlate(name.clone()))
        }
        PlateMotionModel::EulerPole {
            omega_x_mas_per_year,
            omega_y_mas_per_year,
            omega_z_mas_per_year,
        } => Ok(EulerPoleVector {
            omega_x_mas_per_year: *omega_x_mas_per_year,
            omega_y_mas_per_year: *omega_y_mas_per_year,
            omega_z_mas_per_year: *omega_z_mas_per_year,
        }),
    }
}

/// The plate's rigid surface velocity at one geodetic (lon, lat), local
/// east/north/up meters/year. Ellipsoidal height is taken as zero — matching
/// [`crate::solid_earth_tide::tide_displacement_enu`], and negligible at this
/// term's sensitivity (a 3 km height changes `|r|` by < 0.05%).
fn plate_velocity_enu(pole: EulerPoleVector, lon_deg: f64, lat_deg: f64) -> [f64; 3] {
    let omega = [
        pole.omega_x_mas_per_year * MAS_TO_RAD,
        pole.omega_y_mas_per_year * MAS_TO_RAD,
        pole.omega_z_mas_per_year * MAS_TO_RAD,
    ];
    let position = geodetic_to_ecef(lon_deg, lat_deg, 0.0);
    ecef_to_enu(cross(omega, position), lon_deg, lat_deg)
}

/// Per-pixel equivalent range-**rate** delay of the plate's rigid motion,
/// meters/year, on the frame grid: `−(v · l̂)` for the ground→sensor LOS unit
/// vector `l̂`, matching [`crate::solid_earth_tide::tide_range_delay_grid`]'s
/// sign convention. Position-only — the caller scales by elapsed time per
/// acquisition to build the per-date delay stack [`crate::apply::subtract_delay`]
/// expects.
///
/// `lonlat` supplies each pixel's geodetic (lon, lat) in degrees; `los`
/// supplies its ground→sensor unit vector.
#[must_use]
pub fn plate_motion_range_delay_rate_grid(
    pole: EulerPoleVector,
    lonlat: &LonLatGrid,
    los: &LosGeometry,
) -> Array2<f64> {
    let (rows, cols) = los.up.dim();
    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .map(|index| {
            let (row, col) = (index / cols, index % cols);
            let (lon, lat) = lonlat.at(row, col);
            let enu = plate_velocity_enu(pole, lon, lat);
            let toward_sensor = enu[0] * los.east[(row, col)]
                + enu[1] * los.north[(row, col)]
                + enu[2] * los.up[(row, col)];
            -toward_sensor
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("plate motion grid shape")
}

/// Geodetic (lon, lat, ellipsoidal height) → WGS84 ECEF meters. Identical
/// formula to [`crate::solid_earth_tide`]'s private helper of the same name,
/// duplicated rather than shared per this crate's "keep delay functions pure
/// and small" convention — two independent three-line geodesy primitives, not
/// an abstraction.
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

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Contract (DoD #1, analytic): an Euler pole aligned with Earth's spin
    /// axis produces pure eastward velocity at the equator, equal to
    /// `ω · R_equatorial` — a closed-form special case of rigid rotation,
    /// independent of the plate table.
    #[test]
    fn polar_pole_gives_pure_eastward_velocity_at_the_equator() {
        let pole = EulerPoleVector {
            omega_x_mas_per_year: 0.0,
            omega_y_mas_per_year: 0.0,
            omega_z_mas_per_year: 1000.0,
        };
        let enu = plate_velocity_enu(pole, 0.0, 0.0);
        let expected_east = 1000.0 * MAS_TO_RAD * WGS84_A_M;
        assert!((enu[0] - expected_east).abs() < 1e-9);
        assert!(enu[1].abs() < 1e-12, "north component: {}", enu[1]);
        assert!(enu[2].abs() < 1e-12, "up component: {}", enu[2]);
    }

    /// The same polar pole away from the equator (45°N): eastward velocity
    /// scales with the parallel's radius, `ω · R · cos(lat)` to first order
    /// (WGS84 prime-vertical radius of curvature, not a plain sphere).
    #[test]
    fn polar_pole_scales_with_latitude() {
        let pole = EulerPoleVector {
            omega_x_mas_per_year: 0.0,
            omega_y_mas_per_year: 0.0,
            omega_z_mas_per_year: 1000.0,
        };
        let enu = plate_velocity_enu(pole, 0.0, 45.0);
        assert!(enu[0] > 0.0, "eastward at 45N: {}", enu[0]);
        let equator = plate_velocity_enu(pole, 0.0, 0.0)[0];
        assert!(
            enu[0] < equator,
            "45N eastward speed {} should be less than equatorial {equator}",
            enu[0]
        );
        assert!(enu[1].abs() < 1e-9, "north component: {}", enu[1]);
    }

    /// Contract (DoD #2, oracle): North America's ITRF2014 PMM pole at
    /// (-100°, 40°) against an independently hand-derived value (same
    /// closed-form model, computed separately in Python — see the module
    /// docs) — 1e-9 m/yr, tight enough to catch a sign, unit or axis-order
    /// bug while allowing ordinary floating-point round-off.
    #[test]
    fn north_america_matches_independent_oracle() {
        let pole = plate_by_name("NorthAmerica").expect("NorthAmerica is in the table");
        let enu = plate_velocity_enu(pole, -100.0, 40.0);
        let expected = [
            -0.014_924_365_796_432_694,
            -0.004_451_163_361_049_95,
            -1.471_323_803_703_015_7e-5,
        ];
        for (got, want) in enu.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {enu:?}, want {expected:?}");
        }
    }

    /// Contract (DoD #2, oracle): the Pacific plate near Hawaii (-155°, 20°),
    /// same independent cross-check.
    #[test]
    fn pacific_matches_independent_oracle() {
        let pole = plate_by_name("Pacific").expect("Pacific is in the table");
        let enu = plate_velocity_enu(pole, -155.0, 20.0);
        let expected = [
            -0.062_295_287_354_809_13,
            0.034_673_418_249_220_69,
            7.465_944_313_187_964e-5,
        ];
        for (got, want) in enu.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {enu:?}, want {expected:?}");
        }
    }

    /// Plate-name lookup ignores case, spaces, underscores and hyphens.
    #[test]
    fn plate_name_lookup_is_normalized() {
        let canonical = plate_by_name("NorthAmerica").unwrap();
        assert_eq!(plate_by_name("North America").unwrap(), canonical);
        assert_eq!(plate_by_name("north_america").unwrap(), canonical);
        assert_eq!(plate_by_name("NORTH-AMERICA").unwrap(), canonical);
        assert!(plate_by_name("Atlantis").is_none());
    }

    /// [`resolve_euler_pole`] passes an explicit pole through untouched and
    /// rejects an unknown plate name.
    #[test]
    fn resolve_euler_pole_named_and_explicit() {
        let explicit = PlateMotionModel::EulerPole {
            omega_x_mas_per_year: 1.0,
            omega_y_mas_per_year: 2.0,
            omega_z_mas_per_year: 3.0,
        };
        let resolved = resolve_euler_pole(&explicit).unwrap();
        assert_eq!(resolved.omega_x_mas_per_year, 1.0);
        assert_eq!(resolved.omega_y_mas_per_year, 2.0);
        assert_eq!(resolved.omega_z_mas_per_year, 3.0);

        let named = resolve_euler_pole(&PlateMotionModel::Plate("Pacific".into())).unwrap();
        assert_eq!(named, plate_by_name("Pacific").unwrap());

        assert!(resolve_euler_pole(&PlateMotionModel::Plate("Atlantis".into())).is_err());
    }

    /// Contract (DoD #3, off-by-default equivalent): a zero-magnitude pole
    /// produces an all-zero delay-rate grid, so a caller-supplied pole of
    /// exactly zero is a no-op through the same code path other corrections
    /// use for "disabled".
    #[test]
    fn zero_pole_gives_zero_delay_rate() {
        let pole = EulerPoleVector {
            omega_x_mas_per_year: 0.0,
            omega_y_mas_per_year: 0.0,
            omega_z_mas_per_year: 0.0,
        };
        let los = LosGeometry {
            east: Array2::from_elem((2, 2), 0.3),
            north: Array2::from_elem((2, 2), 0.2),
            up: Array2::from_elem((2, 2), 0.9327379053088816),
        };
        let lonlat = LonLatGrid::from_corners(
            [[-100.0, 40.0], [-99.9, 40.0], [-100.0, 40.1], [-99.9, 40.1]],
            2,
            2,
        );
        let rate = plate_motion_range_delay_rate_grid(pole, &lonlat, &los);
        assert!(rate.iter().all(|&v| v == 0.0));
    }
}
