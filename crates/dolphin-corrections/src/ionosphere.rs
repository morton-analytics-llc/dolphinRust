//! Ionospheric range delay from GNSS TEC (IONEX) maps.
//!
//! Scientific reference: dolphin `atmosphere/ionosphere.py` (Yunjun et al. 2022,
//! Chen & Zebker 2012). The dispersive ionospheric range delay scales as `1/f²`,
//! so it is the dominant atmospheric term at L-band: with the NISAR carrier
//! (`f ≈ 1.257 GHz`) the delay is `(f_C / f_L)² ≈ 18×` the Sentinel-1 C-band
//! (`f ≈ 5.405 GHz`) effect for the same TEC.

use ndarray::Array3;

use crate::error::{CorrectionError, Result};

/// Ionospheric constant `K` relating TEC to range delay, m·Hz²/(el/m²)·1e-16;
/// `K = 40.31` (dolphin `ionosphere.K`).
pub const K_IONO: f64 = 40.31;

/// Speed of light (m/s); `freq = SPEED_OF_LIGHT / wavelength`.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Closed-form slant range delay (meters) from zenith (vertical) TEC.
///
/// Mirrors dolphin `vtec_to_range_delay` (Yunjun et al. 2022, eq. 6–11):
/// the zenith TEC is mapped to the line-of-sight through the thin-shell
/// refraction angle, then converted to range delay via `delay = TEC·K/f²`.
/// `vtec` is in TECU (1 TECU = 1e16 el/m²), `inc_angle_deg` is the incidence
/// angle on the ionospheric shell, `freq_hz` is the radar carrier frequency.
///
/// At `inc_angle_deg = 0` (vertical) this reduces to the exact analytic relation
/// `delay = vtec·1e16·K / f²`, the always-provable contract anchor.
#[must_use]
pub fn vtec_to_range_delay(vtec: f64, inc_angle_deg: f64, freq_hz: f64) -> f64 {
    let inc_rad = inc_angle_deg.to_radians();
    // Group refractive index of the ionosphere (Bohm & Schuh 2013, eq. 26).
    let n_iono_group = 1.0 + K_IONO * vtec * 1e16 / freq_hz.powi(2);
    // Refracted angle on the shell (Yunjun et al. 2022, eq. 8).
    let ref_angle = (inc_rad.sin() / n_iono_group).asin();
    // Zenith → line-of-sight TEC (Chen & Zebker 2012, eq. 3).
    let tec_los = vtec / ref_angle.cos();
    // Range delay (Chen & Zebker 2012, eq. 1).
    tec_los * 1e16 * K_IONO / freq_hz.powi(2)
}

/// Parsed IONEX vertical-TEC maps: a `(n_epochs, n_lat, n_lon)` cube on a
/// regular lat/lon grid sampled through the day. Mirrors dolphin `read_ionex`.
pub struct IonexMaps {
    /// Epoch of each map in minutes-of-day, ascending.
    pub minutes: Vec<f64>,
    /// Latitudes (degrees); IONEX convention is descending.
    pub lats: Vec<f64>,
    /// Longitudes (degrees), ascending.
    pub lons: Vec<f64>,
    /// Vertical TEC in TECU, `(n_epochs, n_lat, n_lon)`.
    pub tec: Array3<f64>,
}

impl IonexMaps {
    /// Trilinearly interpolate vertical TEC (TECU) at a UTC time (seconds of day),
    /// latitude and longitude. Mirrors dolphin's `interpn` lookup; out-of-range
    /// coordinates clamp to the grid edge.
    #[must_use]
    pub fn value(&self, utc_sec: f64, lat: f64, lon: f64) -> f64 {
        let (i0, i1, ft) = bracket(&self.minutes, utc_sec / 60.0);
        let (j0, j1, fy) = bracket(&self.lats, lat);
        let (k0, k1, fx) = bracket(&self.lons, lon);
        let at = |i: usize, j: usize, k: usize| self.tec[(i, j, k)];
        let c00 = lerp(at(i0, j0, k0), at(i0, j0, k1), fx);
        let c01 = lerp(at(i0, j1, k0), at(i0, j1, k1), fx);
        let c10 = lerp(at(i1, j0, k0), at(i1, j0, k1), fx);
        let c11 = lerp(at(i1, j1, k0), at(i1, j1, k1), fx);
        let c0 = lerp(c00, c01, fy);
        let c1 = lerp(c10, c11, fy);
        lerp(c0, c1, ft)
    }
}

/// Linear interpolation `a·(1-t) + b·t`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Find the bracketing indices and fractional position of `x` in (possibly
/// descending) monotone `axis`; clamps to the ends.
fn bracket(axis: &[f64], x: f64) -> (usize, usize, f64) {
    let n = axis.len();
    if n == 1 {
        return (0, 0, 0.0);
    }
    let ascending = axis[1] > axis[0];
    let pos = |v: f64| if ascending { v } else { -v };
    let xp = pos(x);
    let upper = (1..n).find(|&i| pos(axis[i]) >= xp).unwrap_or(n - 1);
    let lower = upper - 1;
    let span = axis[upper] - axis[lower];
    let frac = if span == 0.0 {
        0.0
    } else {
        ((x - axis[lower]) / span).clamp(0.0, 1.0)
    };
    (lower, upper, frac)
}

/// Parse an IONEX-format TEC file into vertical-TEC maps.
///
/// Mirrors dolphin `read_ionex`: reads `DLAT`/`DLON`/`EXPONENT`/`# OF MAPS` from
/// the header, then each `START OF TEC MAP … END OF TEC MAP` block as one
/// `(n_lat, n_lon)` grid scaled by `10^EXPONENT`.
///
/// # Errors
/// Returns [`CorrectionError::Ionex`] if required header records are missing or a
/// map's value count does not match the declared grid.
pub fn read_ionex(content: &str) -> Result<IonexMaps> {
    let header = content
        .split("END OF HEADER")
        .next()
        .ok_or_else(|| CorrectionError::Ionex("no header".into()))?;
    let hdr = parse_header(header)?;
    let lats = axis(hdr.lat0, hdr.lat1, hdr.lat_step);
    let lons = axis(hdr.lon0, hdr.lon1, hdr.lon_step);
    let scale = 10f64.powf(hdr.exponent);
    let maps: Vec<Vec<f64>> = content
        .split("START OF TEC MAP")
        .skip(1)
        .map(|block| parse_map(block, lons.len(), scale))
        .collect::<Result<_>>()?;
    let n_epochs = maps.len();
    if n_epochs == 0 {
        return Err(CorrectionError::Ionex("no TEC maps".into()));
    }
    let step = if n_epochs > 1 {
        24.0 * 60.0 / (n_epochs as f64 - 1.0)
    } else {
        0.0
    };
    let minutes = (0..n_epochs).map(|i| i as f64 * step).collect();
    let flat: Vec<f64> = maps.into_iter().flatten().collect();
    let tec = Array3::from_shape_vec((n_epochs, lats.len(), lons.len()), flat)
        .map_err(|e| CorrectionError::Ionex(e.to_string()))?;
    Ok(IonexMaps {
        minutes,
        lats,
        lons,
        tec,
    })
}

/// IONEX header grid spec.
struct Header {
    lat0: f64,
    lat1: f64,
    lat_step: f64,
    lon0: f64,
    lon1: f64,
    lon_step: f64,
    exponent: f64,
}

/// Extract `DLAT`/`DLON`/`EXPONENT` from the header lines.
fn parse_header(header: &str) -> Result<Header> {
    let mut lat = None;
    let mut lon = None;
    let mut exponent = -1.0; // IONEX default
    for line in header.lines() {
        let t = line.trim_end();
        if t.ends_with("DLAT") {
            lat = Some(triple(line)?);
        } else if t.ends_with("DLON") {
            lon = Some(triple(line)?);
        } else if t.ends_with("EXPONENT") {
            exponent = first_f64(line)?;
        }
    }
    let (lat0, lat1, lat_step) = lat.ok_or_else(|| CorrectionError::Ionex("no DLAT".into()))?;
    let (lon0, lon1, lon_step) = lon.ok_or_else(|| CorrectionError::Ionex("no DLON".into()))?;
    Ok(Header {
        lat0,
        lat1,
        lat_step,
        lon0,
        lon1,
        lon_step,
        exponent,
    })
}

/// First three whitespace-separated floats of a line.
fn triple(line: &str) -> Result<(f64, f64, f64)> {
    let v: Vec<f64> = line
        .split_whitespace()
        .take(3)
        .filter_map(|s| s.parse().ok())
        .collect();
    match v[..] {
        [a, b, c] => Ok((a, b, c)),
        _ => Err(CorrectionError::Ionex(format!("bad grid line: {line}"))),
    }
}

/// First whitespace-separated float of a line.
fn first_f64(line: &str) -> Result<f64> {
    line.split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| CorrectionError::Ionex(format!("bad numeric line: {line}")))
}

/// Inclusive regular axis from `start` to `stop` with `step` (handles descending).
fn axis(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).round() as i64;
    (0..=n.unsigned_abs() as usize)
        .map(|i| start + i as f64 * step)
        .collect()
}

/// Parse one `START OF TEC MAP` block (the leading delimiter already stripped) as
/// `n_lat × n_lon` values, scaled by `scale`. The EPOCH line precedes the first
/// `LAT/LON1/LON2/DLON/H` record and is skipped.
fn parse_map(block: &str, n_lon: usize, scale: f64) -> Result<Vec<f64>> {
    let body = block.split("END OF TEC MAP").next().unwrap_or(block);
    let mut values = Vec::new();
    for chunk in body.split("LAT/LON1/LON2/DLON/H").skip(1) {
        let row: Vec<f64> = chunk
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .take(n_lon)
            .map(|v| v * scale)
            .collect();
        if row.len() != n_lon {
            return Err(CorrectionError::Ionex(format!(
                "TEC row has {} values, expected {n_lon}",
                row.len()
            )));
        }
        values.extend(row);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NISAR L-band carrier frequency (Hz): c / 0.2384 m ≈ 1.257 GHz.
    const NISAR_FREQ_HZ: f64 = SPEED_OF_LIGHT / 0.238_403_545;
    /// Sentinel-1 C-band carrier frequency (Hz): c / 0.05546576 m ≈ 5.405 GHz.
    const S1_FREQ_HZ: f64 = SPEED_OF_LIGHT / 0.055_465_76;

    /// Contract (DoD #2): the closed-form TEC→delay relation at vertical
    /// incidence is exactly `delay = vtec·1e16·K / f²`.
    #[test]
    fn closed_form_vertical_delay() {
        let vtec = 20.0; // TECU, a typical mid-latitude daytime value
        let got = vtec_to_range_delay(vtec, 0.0, NISAR_FREQ_HZ);
        let want = vtec * 1e16 * K_IONO / NISAR_FREQ_HZ.powi(2);
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        // Sanity: 20 TECU at L-band is a multi-meter delay.
        assert!(got > 1.0, "L-band delay should be meters-scale, got {got}");
    }

    /// Contract (DoD #2, the load-bearing decision): the delay is `1/f²`-scaled,
    /// so L-band dwarfs C-band by `(f_C / f_L)²` for the same TEC.
    #[test]
    fn l_band_dwarfs_c_band_by_freq_squared() {
        let vtec = 20.0;
        let l = vtec_to_range_delay(vtec, 0.0, NISAR_FREQ_HZ);
        let c = vtec_to_range_delay(vtec, 0.0, S1_FREQ_HZ);
        let ratio = l / c;
        let expected = (S1_FREQ_HZ / NISAR_FREQ_HZ).powi(2);
        assert!(
            (ratio - expected).abs() < 1e-9,
            "ratio {ratio}, expected {expected}"
        );
        assert!(ratio > 16.0, "L/C ratio should exceed 16×, got {ratio}");
    }

    /// Oblique incidence increases the slant delay vs vertical (longer LOS path).
    #[test]
    fn oblique_increases_delay() {
        let vtec = 30.0;
        let vertical = vtec_to_range_delay(vtec, 0.0, NISAR_FREQ_HZ);
        let oblique = vtec_to_range_delay(vtec, 37.0, NISAR_FREQ_HZ);
        assert!(
            oblique > vertical,
            "oblique {oblique} should exceed vertical {vertical}"
        );
    }

    /// Real-data gate: parse a real IGS final GIM IONEX file (path in `IONEX_REAL`,
    /// fetched from CDDIS) and confirm the recovered VTEC and the derived L-band
    /// range delay are physically plausible. Ignored unless the env var is set.
    #[test]
    fn real_ionex_parses_to_physical_delay() {
        let Ok(path) = std::env::var("IONEX_REAL") else {
            return;
        };
        let content = std::fs::read_to_string(&path).expect("read IONEX_REAL");
        let maps = read_ionex(&content).expect("parse real IONEX");
        // IGS final GIM: 13 epochs (2-hourly), 71 lats (87.5..-87.5), 73 lons.
        assert_eq!(maps.tec.dim(), (13, 71, 73));
        assert!((maps.lats[0] - 87.5).abs() < 1e-6);
        assert!((maps.lons[0] + 180.0).abs() < 1e-6);
        // Equatorial midday VTEC is the global daytime peak; sane range 0..150 TECU.
        let vtec = maps.value(12.0 * 3600.0, 0.0, 0.0);
        assert!(
            (0.0..150.0).contains(&vtec),
            "equatorial midday VTEC {vtec}"
        );
        assert!(vtec > 1.0, "daytime VTEC should be non-trivial, got {vtec}");
        // L-band range delay at that TEC is meters-scale (the reason iono is
        // mandatory at L-band).
        let delay = vtec_to_range_delay(vtec, 0.0, NISAR_FREQ_HZ);
        assert!(delay > 0.5, "L-band delay {delay} m should be meters-scale");
    }

    /// The IONEX parser recovers a known 2-epoch grid and interpolates it.
    #[test]
    fn parses_and_interpolates_ionex() {
        // 2 epochs, lat {2.5, 0.0, -2.5} (descending), lon {0, 5, 10}.
        let ionex = "\
     1.0                                                      EPOCH OF FIRST MAP
   -1                                                      EXPONENT
     2.5   -2.5   -2.5                                      DLAT
     0.0   10.0    5.0                                      DLON
     2                                                      # OF MAPS IN FILE
                                                            END OF HEADER
     1                                                      START OF TEC MAP
  2023     1     1     0     0     0                        EPOCH OF CURRENT MAP
     2.5    0.0   10.0    5.0  450.0                        LAT/LON1/LON2/DLON/H
   100    110    120
     0.0    0.0   10.0    5.0  450.0                        LAT/LON1/LON2/DLON/H
   200    210    220
    -2.5    0.0   10.0    5.0  450.0                        LAT/LON1/LON2/DLON/H
   300    310    320
     1                                                      END OF TEC MAP
     2                                                      START OF TEC MAP
  2023     1     1    12     0     0                        EPOCH OF CURRENT MAP
     2.5    0.0   10.0    5.0  450.0                        LAT/LON1/LON2/DLON/H
   140    150    160
     0.0    0.0   10.0    5.0  450.0                        LAT/LON1/LON2/DLON/H
   240    250    260
    -2.5    0.0   10.0    5.0  450.0                        LAT/LON1/LON2/DLON/H
   340    350    360
     2                                                      END OF TEC MAP
";
        let maps = read_ionex(ionex).unwrap();
        assert_eq!(maps.tec.dim(), (2, 3, 3));
        assert_eq!(maps.lats, vec![2.5, 0.0, -2.5]);
        assert_eq!(maps.lons, vec![0.0, 5.0, 10.0]);
        // 2 maps → epochs at 0 and 1440 min (IONEX `min_step = 1440/(n-1)`).
        // exponent -1 → ×0.1. Grid node (epoch0, lat=0, lon=5) = 210 × 0.1 = 21.0.
        let v = maps.value(0.0, 0.0, 5.0);
        assert!((v - 21.0).abs() < 1e-9, "node value {v}");
        // Second epoch (1440 min) at the same node = 250 × 0.1 = 25.0.
        let ep1 = maps.value(1440.0 * 60.0, 0.0, 5.0);
        assert!((ep1 - 25.0).abs() < 1e-9, "epoch1 node {ep1}");
        // Time-midpoint (720 min) → mean of 21.0 and 25.0.
        let half = maps.value(720.0 * 60.0, 0.0, 5.0);
        assert!((half - 23.0).abs() < 1e-9, "time-interp {half}");
    }
}
