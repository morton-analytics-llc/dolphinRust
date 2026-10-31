//! Per-pair perpendicular/parallel baseline (issue #101).
//!
//! A standalone SAR-geometry diagnostic — spatial-baseline decorrelation risk and
//! network-design QC — not wired into phase-linking/inversion (dolphin's own
//! `src/dolphin/baseline.py` is likewise a standalone utility, not a core-pipeline
//! dependency). Reuses the orbit ingest already built for geometry provenance
//! (`dolphin_io::CslcOrbit`, read via `read_cslc_orbit`).
//!
//! # Method
//!
//! Matches dolphin's along-track-shift correction (PR #681, ISCE2 issue #137
//! lineage): the raw ECEF vector between two satellites' positions at a shared
//! orbit-time epoch carries a spurious along-track offset, because the two
//! platforms are not in general at exactly the same along-track position at that
//! epoch — only close to it. Left uncorrected, that offset leaks into the
//! reported perpendicular baseline. The fix removes it before decomposing the
//! baseline into components that actually describe imaging geometry:
//!
//! ```text
//! L̂  = normalize(target − S_ref)      // look vector, reference satellite → target
//! V̂  = normalize(velocity_ref)         // reference platform's along-track direction
//! B   = S_sec − S_ref                  // raw ECEF baseline
//! B′  = B − (B·V̂) V̂                    // along-track component removed
//! B∥  = B′ · L̂                         // parallel (range/line-of-sight) baseline
//! B⊥  = B′ · n̂                         // perpendicular baseline, n̂ below
//! ```
//!
//! `n̂` is the unit vector orthogonal to `L̂`, in the plane spanned by `L̂` and the
//! target's local vertical (its ECEF position direction, i.e. radial from Earth's
//! center) — the plane perpendicular baseline is conventionally measured in. This
//! matches the ISCE2/dolphin sign convention: positive when the secondary orbit is
//! displaced away from Earth's center relative to the reference, in that plane.
//! `L̂` is by construction (the zero-Doppler imaging condition) orthogonal to `V̂`,
//! so `B∥` and `B⊥` carry none of the along-track offset once `B′` has removed it.

use std::path::PathBuf;

use dolphin_io::CslcOrbit;

/// WGS84 semi-major axis, meters.
const WGS84_A_M: f64 = 6_378_137.0;
/// WGS84 semi-minor axis, meters.
const WGS84_B_M: f64 = 6_356_752.314_245;

/// Perpendicular/parallel decomposition of one orbit pair's baseline at one
/// ground target, dolphin's along-track-corrected convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairBaseline {
    /// Signed perpendicular (cross-track) baseline, meters. Positive when the
    /// secondary orbit sits farther from Earth's center than the reference, in
    /// the look/vertical plane.
    pub perpendicular_m: f64,
    /// Parallel (range/line-of-sight) baseline, meters.
    pub parallel_m: f64,
}

/// Perpendicular/parallel baseline between two satellite ECEF states imaging a
/// common ground target.
///
/// `ref_position`/`ref_velocity` and `sec_position` are ECEF meters and
/// meters/second; `target` is the ECEF ground point the reference platform
/// images at zero Doppler (so that `target − ref_position` is orthogonal to
/// `ref_velocity` by construction).
#[must_use]
pub fn pair_baseline(
    target: [f64; 3],
    ref_position: [f64; 3],
    ref_velocity: [f64; 3],
    sec_position: [f64; 3],
) -> PairBaseline {
    let look = normalize(subtract(target, ref_position));
    let track = normalize(ref_velocity);
    let raw = subtract(sec_position, ref_position);
    let corrected = subtract(raw, scale(track, dot(raw, track)));

    let parallel_m = dot(corrected, look);
    let vertical_perp = normalize(subtract(
        normalize(target),
        scale(look, dot(normalize(target), look)),
    ));
    let perpendicular_m = dot(corrected, vertical_perp);

    PairBaseline {
        perpendicular_m,
        parallel_m,
    }
}

/// [`pair_baseline`] sourced directly from two CSLC orbits and a common ground
/// target, interpolating each orbit to `ref_time_s` / `sec_time_s` (seconds
/// since each orbit's own `reference_epoch` — typically each granule's
/// zero-Doppler mid-time, matching `dolphin_workflows::provenance`'s heading
/// derivation).
///
/// # Errors
/// `Err` when either orbit carries fewer than two state vectors, or when the
/// requested time falls outside an orbit's state-vector span — clamped
/// extrapolation would silently produce a plausible-but-wrong baseline.
pub fn orbit_pair_baseline(
    target_lonlat_deg: [f64; 2],
    ref_orbit: &CslcOrbit,
    ref_time_s: f64,
    sec_orbit: &CslcOrbit,
    sec_time_s: f64,
) -> Result<PairBaseline, String> {
    let ref_position = interpolate_span(ref_orbit, ref_time_s, "reference")?;
    let ref_velocity = interp3_clamped(&ref_orbit.time_s, &ref_orbit.velocity_mps, ref_time_s);
    let sec_position = interpolate_span(sec_orbit, sec_time_s, "secondary")?;
    let [lon_deg, lat_deg] = target_lonlat_deg;
    let target = geodetic_to_ecef(lon_deg, lat_deg, 0.0);
    Ok(pair_baseline(
        target,
        ref_position,
        ref_velocity,
        sec_position,
    ))
}

/// Per-date DEM-error sensitivity `B⊥(t) / (R sin θ)` (metres of LOS
/// displacement per metre of height error) at the first granule's burst centre,
/// one value per granule, zero at acquisition 0 (issue #143).
///
/// `B⊥(t)` is [`pair_baseline`]'s perpendicular baseline from granule 0 to
/// granule `t`, each platform at its own mid zero-Doppler time; `R` is the
/// slant range from granule 0's platform to the centre; `θ` is the incidence
/// there, the angle between the target-to-satellite vector and the ellipsoid
/// normal. Frame-centre values: `B⊥` and `R sin θ` vary by a few percent across a
/// burst, which scales the fitted height error, not the rate it is removed from.
///
/// # Errors
/// `Err` naming the granule when its orbit, identification or burst metadata
/// cannot be read, its times do not parse, or a time falls outside its orbit.
pub fn dem_error_sensitivity(granules: &[PathBuf]) -> Result<Vec<f64>, String> {
    let states = granules
        .iter()
        .map(|path| granule_state(path).map_err(|e| format!("{}: {e}", path.display())))
        .collect::<Result<Vec<_>, String>>()?;
    sensitivity_from_states(&states)
}

fn sensitivity_from_states(states: &[GranuleState]) -> Result<Vec<f64>, String> {
    let Some(reference) = states.first() else {
        return Err("DEM-error sensitivity needs at least one granule".into());
    };
    let [lon_deg, lat_deg] = reference.center_lonlat_deg;
    let target = geodetic_to_ecef(lon_deg, lat_deg, 0.0);
    let to_satellite = subtract(reference.position, target);
    let (lon, lat) = (lon_deg.to_radians(), lat_deg.to_radians());
    let normal = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    let slant_range = dot(to_satellite, to_satellite).sqrt();
    let cos_incidence = dot(normalize(to_satellite), normal);
    let denominator = slant_range * (1.0 - cos_incidence * cos_incidence).sqrt();
    Ok(states
        .iter()
        .map(|state| {
            pair_baseline(
                target,
                reference.position,
                reference.velocity,
                state.position,
            )
            .perpendicular_m
                / denominator
        })
        .collect())
}

/// One granule's platform state at its mid zero-Doppler time.
struct GranuleState {
    position: [f64; 3],
    velocity: [f64; 3],
    center_lonlat_deg: [f64; 2],
}

fn granule_state(path: &std::path::Path) -> Result<GranuleState, String> {
    const FORMAT: &str = "%Y-%m-%d %H:%M:%S%.f";
    let orbit = dolphin_io::read_cslc_orbit(path).map_err(|e| e.to_string())?;
    let ident = dolphin_io::read_cslc_identification(path).map_err(|e| e.to_string())?;
    let burst = dolphin_io::read_cslc_burst_metadata(path).map_err(|e| e.to_string())?;
    let parse = |raw: &str, what: &str| {
        chrono::NaiveDateTime::parse_from_str(raw.trim(), FORMAT)
            .map_err(|e| format!("unparseable {what} {raw:?}: {e}"))
    };
    let epoch = parse(&orbit.reference_epoch, "orbit reference_epoch")?;
    let start = parse(&ident.zero_doppler_start_time, "zero_doppler_start_time")?;
    let end = parse(&ident.zero_doppler_end_time, "zero_doppler_end_time")?;
    let mid = start + (end - start) / 2;
    let seconds = (mid - epoch)
        .num_microseconds()
        .ok_or("mid zero-Doppler time overflows")? as f64
        / 1e6;
    Ok(GranuleState {
        position: interpolate_span(&orbit, seconds, "granule")?,
        velocity: interp3_clamped(&orbit.time_s, &orbit.velocity_mps, seconds),
        center_lonlat_deg: burst.center_lonlat_deg,
    })
}

fn interpolate_span(orbit: &CslcOrbit, time_s: f64, which: &str) -> Result<[f64; 3], String> {
    if orbit.time_s.len() < 2 {
        return Err(format!("{which} orbit has fewer than 2 state vectors"));
    }
    let (first, last) = (orbit.time_s[0], orbit.time_s[orbit.time_s.len() - 1]);
    if !time_s.is_finite() || time_s < first || time_s > last {
        return Err(format!(
            "{which} orbit time {time_s:.1}s outside state-vector span [{first:.1}, {last:.1}]s"
        ));
    }
    Ok(interp3_clamped(&orbit.time_s, &orbit.position_m, time_s))
}

/// `numpy.interp`-style clamped linear interpolation of `[x, y, z]` samples.
fn interp3_clamped(t: &[f64], values: &[[f64; 3]], x: f64) -> [f64; 3] {
    let hi = t.partition_point(|&ti| ti < x).clamp(1, t.len() - 1);
    let lo = hi - 1;
    let span = t[hi] - t[lo];
    let frac = if span == 0.0 {
        0.0
    } else {
        ((x - t[lo]) / span).clamp(0.0, 1.0)
    };
    std::array::from_fn(|i| values[lo][i] + frac * (values[hi][i] - values[lo][i]))
}

/// Geodetic (lon, lat, ellipsoidal height) → WGS84 ECEF meters.
fn geodetic_to_ecef(lon_deg: f64, lat_deg: f64, height_m: f64) -> [f64; 3] {
    let (lon, lat) = (lon_deg.to_radians(), lat_deg.to_radians());
    let e2 = 1.0 - (WGS84_B_M / WGS84_A_M).powi(2);
    let prime_vertical = WGS84_A_M / (1.0 - e2 * lat.sin() * lat.sin()).sqrt();
    [
        (prime_vertical + height_m) * lat.cos() * lon.cos(),
        (prime_vertical + height_m) * lat.cos() * lon.sin(),
        (prime_vertical * (1.0 - e2) + height_m) * lat.sin(),
    ]
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    std::array::from_fn(|i| a[i] * s)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / dot(a, a).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #143: a satellite at 700 km altitude looking 30° off nadir at an
    /// equatorial target, and a secondary displaced 100 m along the
    /// perpendicular-baseline direction, give `B⊥ / (R sin θ)` with the
    /// closed-form `R` and `θ`; acquisition 0 is zero.
    #[test]
    fn dem_error_sensitivity_matches_closed_form_geometry() {
        let target = geodetic_to_ecef(0.0, 0.0, 0.0);
        let up = [1.0, 0.0, 0.0];
        let east = [0.0, 1.0, 0.0];
        let (height, look) = (700_000.0, 30.0_f64.to_radians());
        let slant = height / look.cos();
        let to_satellite = [look.cos(), look.sin(), 0.0];
        let reference = [
            target[0] + slant * to_satellite[0],
            target[1] + slant * to_satellite[1],
            0.0,
        ];
        // Perpendicular to the look in the look/vertical plane, away from Earth.
        let perpendicular = normalize(subtract(up, scale(to_satellite, dot(up, to_satellite))));
        let secondary = std::array::from_fn(|i| reference[i] + 100.0 * perpendicular[i]);
        let state = |position| GranuleState {
            position,
            velocity: [0.0, 0.0, 7_500.0],
            center_lonlat_deg: [0.0, 0.0],
        };
        let values = sensitivity_from_states(&[state(reference), state(secondary)]).unwrap();
        assert_eq!(values[0], 0.0);
        let expected = 100.0 / (slant * look.sin());
        assert!(
            (values[1].abs() - expected).abs() < 1e-9 * expected,
            "{} vs {expected}",
            values[1]
        );
        let _ = east;
    }

    /// A synthetic side-looking geometry: target on the `+X` axis (its own ECEF
    /// direction is `up = [1,0,0]`), 30° incidence look vector in the X/Y plane,
    /// and an along-track direction on the `Z` axis — orthogonal to the look
    /// vector by construction, matching the zero-Doppler condition.
    fn scene() -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        let target = [7_000_000.0, 0.0, 0.0];
        let look = normalize([30f64.to_radians().cos(), 30f64.to_radians().sin(), 0.0]);
        let track = [0.0, 0.0, 1.0];
        let range_m = 800_000.0;
        let ref_position = subtract(target, scale(look, range_m));
        let ref_velocity = scale(track, 7_000.0);
        (target, ref_position, ref_velocity, look)
    }

    /// Perpendicular baseline direction used to build fixtures, independently of
    /// the function under test's internal derivation.
    fn vertical_perp(target: [f64; 3], look: [f64; 3]) -> [f64; 3] {
        normalize(subtract(
            normalize(target),
            scale(look, dot(normalize(target), look)),
        ))
    }

    #[test]
    fn pure_along_track_offset_yields_zero_baseline() {
        let (target, ref_position, ref_velocity, _look) = scene();
        let track = normalize(ref_velocity);
        let sec_position = subtract(ref_position, scale(track, -450.0)); // +450 m along-track
        let result = pair_baseline(target, ref_position, ref_velocity, sec_position);
        assert!(result.perpendicular_m.abs() < 1e-6, "{result:?}");
        assert!(result.parallel_m.abs() < 1e-6, "{result:?}");
    }

    #[test]
    fn pure_perpendicular_offset_recovers_known_magnitude() {
        let (target, ref_position, ref_velocity, look) = scene();
        let n_perp = vertical_perp(target, look);
        let sec_position = subtract(ref_position, scale(n_perp, -200.0)); // +200 m perpendicular
        let result = pair_baseline(target, ref_position, ref_velocity, sec_position);
        assert!((result.perpendicular_m - 200.0).abs() < 1e-6, "{result:?}");
        assert!(result.parallel_m.abs() < 1e-6, "{result:?}");
    }

    #[test]
    fn pure_parallel_offset_recovers_known_magnitude() {
        let (target, ref_position, ref_velocity, look) = scene();
        let sec_position = subtract(ref_position, scale(look, -120.0)); // +120 m in range
        let result = pair_baseline(target, ref_position, ref_velocity, sec_position);
        assert!((result.parallel_m - 120.0).abs() < 1e-6, "{result:?}");
        assert!(result.perpendicular_m.abs() < 1e-6, "{result:?}");
    }

    /// The case PR #681 exists for: an along-track-contaminated offset must
    /// report the same perpendicular baseline as the clean case above, not a
    /// mixture of the two.
    #[test]
    fn along_track_contamination_is_removed_before_perpendicular_baseline() {
        let (target, ref_position, ref_velocity, look) = scene();
        let n_perp = vertical_perp(target, look);
        let track = normalize(ref_velocity);
        let contaminated = subtract(
            subtract(ref_position, scale(n_perp, -200.0)),
            scale(track, -75.0),
        );
        let result = pair_baseline(target, ref_position, ref_velocity, contaminated);
        assert!((result.perpendicular_m - 200.0).abs() < 1e-6, "{result:?}");
        assert!(result.parallel_m.abs() < 1e-6, "{result:?}");
    }

    #[test]
    fn orbit_pair_baseline_rejects_time_outside_span() {
        let orbit = CslcOrbit {
            time_s: vec![0.0, 10.0],
            position_m: vec![[7_000_000.0, 0.0, 0.0], [7_000_000.0, 70_000.0, 0.0]],
            velocity_mps: vec![[0.0, 7_000.0, 0.0], [0.0, 7_000.0, 0.0]],
            reference_epoch: "2026-01-01 00:00:00.000000".into(),
        };
        let err = orbit_pair_baseline([0.0, 0.0], &orbit, 20.0, &orbit, 5.0).unwrap_err();
        assert!(err.contains("outside state-vector span"), "{err}");
    }

    #[test]
    fn orbit_pair_baseline_agrees_with_pair_baseline_at_interpolated_states() {
        let orbit = CslcOrbit {
            time_s: vec![0.0, 10.0],
            position_m: vec![[7_000_000.0, 0.0, 0.0], [6_999_300.0, 70_000.0, 0.0]],
            velocity_mps: vec![[-70.0, 7_000.0, 0.0], [-70.0, 7_000.0, 0.0]],
            reference_epoch: "2026-01-01 00:00:00.000000".into(),
        };
        let sec = CslcOrbit {
            position_m: vec![[7_000_150.0, 100.0, 300.0], [6_999_450.0, 70_100.0, 300.0]],
            ..orbit.clone()
        };
        let got = orbit_pair_baseline([0.0, 89.9], &orbit, 5.0, &sec, 5.0).unwrap();

        let ref_position = interp3_clamped(&orbit.time_s, &orbit.position_m, 5.0);
        let ref_velocity = interp3_clamped(&orbit.time_s, &orbit.velocity_mps, 5.0);
        let sec_position = interp3_clamped(&sec.time_s, &sec.position_m, 5.0);
        let target = geodetic_to_ecef(0.0, 89.9, 0.0);
        let want = pair_baseline(target, ref_position, ref_velocity, sec_position);

        assert!((got.perpendicular_m - want.perpendicular_m).abs() < 1e-9);
        assert!((got.parallel_m - want.parallel_m).abs() < 1e-9);
    }
}
