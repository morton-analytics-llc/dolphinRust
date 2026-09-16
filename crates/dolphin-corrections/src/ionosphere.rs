//! Ionospheric range delay from GNSS TEC (IONEX) maps.
//!
//! Scientific reference: dolphin `atmosphere/ionosphere.py` (Yunjun et al. 2022,
//! Chen & Zebker 2012). The dispersive ionospheric range delay scales as `1/f²`,
//! so it is the dominant atmospheric term at L-band: with the NISAR carrier
//! (`f ≈ 1.257 GHz`) the delay is `(f_C / f_L)² ≈ 18×` the Sentinel-1 C-band
//! (`f ≈ 5.405 GHz`) effect for the same TEC.

use chrono::{DateTime, NaiveDate, Utc};
use ndarray::Array3;
use thiserror::Error;

/// Ionospheric constant `K` relating TEC to range delay, m·Hz²/(el/m²)·1e-16;
/// `K = 40.31` (dolphin `ionosphere.K`).
pub const K_IONO: f64 = 40.31;

/// Speed of light (m/s); `freq = SPEED_OF_LIGHT / wavelength`.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// IONEX "no value available" sentinel, compared against the raw integer cell
/// before the `EXPONENT` scaling is applied.
const MISSING_SENTINEL: f64 = 9999.0;

/// IONEX parsing and sampling failures, typed so a caller can tell a malformed
/// file, a map/header disagreement, an uncovered acquisition and a missing cell
/// apart.
#[derive(Debug, Error)]
pub enum IonexError {
    /// A required record is missing or failed to parse.
    #[error("ionex parse: {0}")]
    Malformed(String),
    /// Consecutive `EPOCH OF CURRENT MAP` records are not strictly increasing.
    #[error("ionex map {index} epoch {current} does not follow {previous}")]
    EpochOrder {
        /// Zero-based index of the offending map.
        index: usize,
        /// Epoch of the preceding map.
        previous: DateTime<Utc>,
        /// Epoch of the offending map.
        current: DateTime<Utc>,
    },
    /// The first or last map epoch disagrees with the header's `EPOCH OF FIRST
    /// MAP` / `EPOCH OF LAST MAP`.
    #[error("ionex {which} map epoch {map} disagrees with header {header}")]
    EpochMismatch {
        /// `"first"` or `"last"`.
        which: &'static str,
        /// The header's declared epoch.
        header: DateTime<Utc>,
        /// The map's `EPOCH OF CURRENT MAP`.
        map: DateTime<Utc>,
    },
    /// The acquisition lies outside `[first, last]` map epochs; TEC is never
    /// extrapolated in time.
    #[error("acquisition {acquisition} outside IONEX map span {first}..={last}")]
    TemporalCoverage {
        /// Requested acquisition time.
        acquisition: DateTime<Utc>,
        /// First map epoch.
        first: DateTime<Utc>,
        /// Last map epoch.
        last: DateTime<Utc>,
    },
    /// The requested coordinate lies outside the map grid.
    #[error("({lat}, {lon}) outside IONEX grid")]
    SpatialCoverage {
        /// Requested latitude (degrees).
        lat: f64,
        /// Requested longitude (degrees).
        lon: f64,
    },
    /// A cell with positive interpolation weight is the `9999` missing sentinel,
    /// so the sample is unavailable rather than a number.
    #[error("TEC missing (9999) at map epoch {epoch}, lat {lat}, lon {lon}")]
    MissingTec {
        /// Map epoch of the missing cell.
        epoch: DateTime<Utc>,
        /// Grid latitude of the missing cell.
        lat: f64,
        /// Grid longitude of the missing cell.
        lon: f64,
    },
}

/// Closed-form slant range delay (meters) from zenith (vertical) TEC.
///
/// Mirrors dolphin `vtec_to_range_delay` (Yunjun et al. 2022, eq. 6–11):
/// the zenith TEC is mapped to the line-of-sight through the thin-shell
/// refraction angle, then converted to range delay via `delay = TEC·K/f²`.
/// `vtec` is in TECU (1 TECU = 1e16 el/m²), `inc_angle_deg` is the incidence
/// angle at the ground, `freq_hz` is the radar carrier frequency.
///
/// At `inc_angle_deg = 0` (vertical) this reduces to the exact analytic relation
/// `delay = vtec·1e16·K / f²`, the always-provable contract anchor.
#[must_use]
pub fn vtec_to_range_delay(vtec: f64, inc_angle_deg: f64, freq_hz: f64) -> f64 {
    let inc_rad = ((6_371_000.0 / 6_821_000.0) * inc_angle_deg.to_radians().sin()).asin();
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
/// regular lat/lon grid at the file's own map epochs. Mirrors dolphin `read_ionex`.
pub struct IonexMaps {
    /// UTC `EPOCH OF CURRENT MAP` of each map, strictly ascending; the first and
    /// last equal the header's `EPOCH OF FIRST MAP` / `EPOCH OF LAST MAP`.
    pub epochs: Vec<DateTime<Utc>>,
    /// Latitudes (degrees); IONEX convention is descending.
    pub lats: Vec<f64>,
    /// Longitudes (degrees), ascending.
    pub lons: Vec<f64>,
    /// Vertical TEC in TECU, `(n_epochs, n_lat, n_lon)`; missing (`9999`) cells
    /// are `NaN`.
    pub tec: Array3<f64>,
}

impl IonexMaps {
    /// Trilinear (time, latitude, longitude) TEC at an acquisition UTC.
    ///
    /// # Errors
    /// [`IonexError::TemporalCoverage`] outside `[first, last]` map epochs (no
    /// extrapolation), [`IonexError::SpatialCoverage`] off the grid, and
    /// [`IonexError::MissingTec`] when any cell with positive weight is missing.
    pub fn value(&self, utc: DateTime<Utc>, lat: f64, lon: f64) -> Result<f64, IonexError> {
        let (Some(&first), Some(&last)) = (self.epochs.first(), self.epochs.last()) else {
            return Err(IonexError::Malformed("no TEC maps".into()));
        };
        if utc < first || utc > last {
            return Err(IonexError::TemporalCoverage {
                acquisition: utc,
                first,
                last,
            });
        }
        let covers = |axis: &[f64], x: f64| match (axis.first(), axis.last()) {
            (Some(&a), Some(&b)) => x.is_finite() && x >= a.min(b) && x <= a.max(b),
            _ => false,
        };
        if !covers(&self.lats, lat) || !covers(&self.lons, lon) {
            return Err(IonexError::SpatialCoverage { lat, lon });
        }
        let times: Vec<f64> = self.epochs.iter().map(|v| v.timestamp() as f64).collect();
        let timestamp = utc.timestamp() as f64 + f64::from(utc.timestamp_subsec_nanos()) / 1e9;
        let (i0, i1, ft) = bracket(&times, timestamp);
        let (j0, j1, fy) = bracket(&self.lats, lat);
        let (k0, k1, fx) = bracket(&self.lons, lon);
        let time_weights = [(i0, 1.0 - ft), (i1, ft)];
        let lat_weights = [(j0, 1.0 - fy), (j1, fy)];
        let lon_weights = [(k0, 1.0 - fx), (k1, fx)];
        let corners = time_weights.into_iter().flat_map(move |t| {
            lat_weights
                .into_iter()
                .flat_map(move |y| lon_weights.into_iter().map(move |x| (t, y, x)))
        });
        let mut value = 0.0;
        for ((i, wt), (j, wy), (k, wx)) in corners {
            let weight = wt * wy * wx;
            if weight == 0.0 {
                continue;
            }
            let tec = self.tec[(i, j, k)];
            if !tec.is_finite() {
                return Err(IonexError::MissingTec {
                    epoch: self.epochs[i],
                    lat: self.lats[j],
                    lon: self.lons[k],
                });
            }
            value += weight * tec;
        }
        Ok(value)
    }
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
/// Mirrors dolphin `read_ionex`: reads `DLAT`/`DLON`/`EXPONENT` and the first/last
/// map epochs from the header, then each `START OF TEC MAP … END OF TEC MAP`
/// block as one `(n_lat, n_lon)` grid scaled by `10^EXPONENT`, timed by its own
/// `EPOCH OF CURRENT MAP` record. `INTERVAL` is never used to place a map.
///
/// # Errors
/// [`IonexError::Malformed`] for missing records or a bad grid,
/// [`IonexError::EpochOrder`] when map epochs do not strictly increase, and
/// [`IonexError::EpochMismatch`] when the first/last map epoch disagrees with the
/// header.
pub fn read_ionex(content: &str) -> Result<IonexMaps, IonexError> {
    let (header, body) = content
        .split_once("END OF HEADER")
        .ok_or_else(|| IonexError::Malformed("no END OF HEADER".into()))?;
    let hdr = parse_header(header)?;
    let lats = axis(hdr.lat0, hdr.lat1, hdr.lat_step);
    let lons = axis(hdr.lon0, hdr.lon1, hdr.lon_step);
    let scale = 10f64.powf(hdr.exponent);
    let mut epochs = Vec::new();
    let mut flat = Vec::new();
    for block in body.split("START OF TEC MAP").skip(1) {
        // Drop the `END OF TEC MAP` record's own line so its map index is not a value.
        let map = block.split("END OF TEC MAP").next().unwrap_or(block);
        let map = map.rsplit_once('\n').map_or("", |(rows, _)| rows);
        epochs.push(map_epoch(map)?);
        flat.extend(parse_map(map, lons.len(), scale)?);
    }
    check_epochs(&epochs, hdr.first, hdr.last)?;
    let tec = Array3::from_shape_vec((epochs.len(), lats.len(), lons.len()), flat)
        .map_err(|e| IonexError::Malformed(e.to_string()))?;
    Ok(IonexMaps {
        epochs,
        lats,
        lons,
        tec,
    })
}

/// Map epochs must strictly increase and start/end exactly at the header's
/// declared first/last epochs.
fn check_epochs(
    epochs: &[DateTime<Utc>],
    first: DateTime<Utc>,
    last: DateTime<Utc>,
) -> Result<(), IonexError> {
    if let Some((index, pair)) = epochs
        .windows(2)
        .enumerate()
        .find(|(_, pair)| pair[0] >= pair[1])
    {
        return Err(IonexError::EpochOrder {
            index: index + 1,
            previous: pair[0],
            current: pair[1],
        });
    }
    let (Some(&map_first), Some(&map_last)) = (epochs.first(), epochs.last()) else {
        return Err(IonexError::Malformed("no TEC maps".into()));
    };
    if map_first != first {
        return Err(IonexError::EpochMismatch {
            which: "first",
            header: first,
            map: map_first,
        });
    }
    if map_last != last {
        return Err(IonexError::EpochMismatch {
            which: "last",
            header: last,
            map: map_last,
        });
    }
    Ok(())
}

/// The `EPOCH OF CURRENT MAP` record of one map block, as UTC.
fn map_epoch(map: &str) -> Result<DateTime<Utc>, IonexError> {
    map.lines()
        .find(|line| line.trim_end().ends_with("EPOCH OF CURRENT MAP"))
        .ok_or_else(|| IonexError::Malformed("missing EPOCH OF CURRENT MAP".into()))
        .and_then(parse_epoch)
}

/// `YYYY MM DD hh mm ss` epoch record (`EPOCH OF FIRST/LAST/CURRENT MAP`) as UTC.
fn parse_epoch(line: &str) -> Result<DateTime<Utc>, IonexError> {
    let invalid = || IonexError::Malformed(format!("invalid epoch record: {}", line.trim()));
    let fields = line
        .split_whitespace()
        .take(6)
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid())?;
    let [year, month, day, hour, minute, second] = fields[..] else {
        return Err(invalid());
    };
    NaiveDate::from_ymd_opt(year as i32, month, day)
        .and_then(|date| date.and_hms_opt(hour, minute, second))
        .map(|date| date.and_utc())
        .ok_or_else(invalid)
}

/// IONEX header grid spec and declared epoch span.
struct Header {
    lat0: f64,
    lat1: f64,
    lat_step: f64,
    lon0: f64,
    lon1: f64,
    lon_step: f64,
    exponent: f64,
    first: DateTime<Utc>,
    last: DateTime<Utc>,
}

/// Extract `DLAT`/`DLON`/`EXPONENT`/`EPOCH OF FIRST MAP`/`EPOCH OF LAST MAP`.
fn parse_header(header: &str) -> Result<Header, IonexError> {
    let mut lat = None;
    let mut lon = None;
    let mut first = None;
    let mut last = None;
    let mut exponent = -1.0; // IONEX default
    for line in header.lines() {
        let t = line.trim_end();
        match t {
            _ if t.ends_with("DLAT") => lat = Some(triple(line)?),
            _ if t.ends_with("DLON") => lon = Some(triple(line)?),
            _ if t.ends_with("EXPONENT") => exponent = first_f64(line)?,
            _ if t.ends_with("EPOCH OF FIRST MAP") => first = Some(parse_epoch(line)?),
            _ if t.ends_with("EPOCH OF LAST MAP") => last = Some(parse_epoch(line)?),
            _ => {}
        }
    }
    let missing = |record: &str| IonexError::Malformed(format!("no {record}"));
    let (lat0, lat1, lat_step) = lat.ok_or_else(|| missing("DLAT"))?;
    let (lon0, lon1, lon_step) = lon.ok_or_else(|| missing("DLON"))?;
    Ok(Header {
        lat0,
        lat1,
        lat_step,
        lon0,
        lon1,
        lon_step,
        exponent,
        first: first.ok_or_else(|| missing("EPOCH OF FIRST MAP"))?,
        last: last.ok_or_else(|| missing("EPOCH OF LAST MAP"))?,
    })
}

/// First three whitespace-separated floats of a line.
fn triple(line: &str) -> Result<(f64, f64, f64), IonexError> {
    let v: Vec<f64> = line
        .split_whitespace()
        .take(3)
        .filter_map(|s| s.parse().ok())
        .collect();
    match v[..] {
        [a, b, c] => Ok((a, b, c)),
        _ => Err(IonexError::Malformed(format!("bad grid line: {line}"))),
    }
}

/// First whitespace-separated float of a line.
fn first_f64(line: &str) -> Result<f64, IonexError> {
    line.split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| IonexError::Malformed(format!("bad numeric line: {line}")))
}

/// Inclusive regular axis from `start` to `stop` with `step` (handles descending).
fn axis(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).round() as i64;
    (0..=n.unsigned_abs() as usize)
        .map(|i| start + i as f64 * step)
        .collect()
}

/// Parse one map's rows (the `START OF TEC MAP` and `END OF TEC MAP` records
/// already stripped) as `n_lat × n_lon` values. A raw `9999` cell is missing
/// (`NaN`) and is never scaled; every other cell is scaled by `scale`. The EPOCH
/// line precedes the first `LAT/LON1/LON2/DLON/H` record and is skipped.
fn parse_map(map: &str, n_lon: usize, scale: f64) -> Result<Vec<f64>, IonexError> {
    let mut values = Vec::new();
    for chunk in map.split("LAT/LON1/LON2/DLON/H").skip(1) {
        let row: Vec<f64> = chunk
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .take(n_lon)
            .map(|raw| {
                if raw == MISSING_SENTINEL {
                    f64::NAN
                } else {
                    raw * scale
                }
            })
            .collect();
        if row.len() != n_lon {
            return Err(IonexError::Malformed(format!(
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
    #[ignore = "requires staged real-source input; run explicitly with --ignored"]
    fn real_ionex_parses_to_physical_delay() {
        let path = std::env::var("IONEX_REAL")
            .expect("IONEX_REAL must identify the staged real-source input");
        let content = std::fs::read_to_string(&path).expect("read IONEX_REAL");
        let maps = read_ionex(&content).expect("parse real IONEX");
        // IGS final GIM: 13 epochs (2-hourly), 71 lats (87.5..-87.5), 73 lons.
        assert_eq!(maps.tec.dim(), (13, 71, 73));
        assert!((maps.lats[0] - 87.5).abs() < 1e-6);
        assert!((maps.lons[0] + 180.0).abs() < 1e-6);
        // Equatorial midday VTEC is the global daytime peak; sane range 0..150 TECU.
        let vtec = maps
            .value(maps.epochs[0] + chrono::Duration::hours(12), 0.0, 0.0)
            .unwrap();
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

    #[test]
    fn ground_incidence_maps_to_450_km_shell() {
        // Independently evaluated shell angle at 40 degrees ground incidence:
        // asin(6371 / 6821 * sin(40 degrees)) = 36.89720077406529 degrees.
        let vtec = 20.0;
        for frequency in [S1_FREQ_HZ, NISAR_FREQ_HZ] {
            let zenith = vtec * 1e16 * K_IONO / frequency.powi(2);
            let expected = zenith
                / (36.89720077406529_f64.to_radians().sin() / (1.0 + zenith))
                    .asin()
                    .cos();
            assert!((vtec_to_range_delay(vtec, 40.0, frequency) - expected).abs() < 1e-10);
        }
    }

    /// The IONEX parser recovers a known 2-epoch grid and interpolates it.
    #[test]
    fn parses_and_interpolates_ionex() {
        // 2 epochs, lat {2.5, 0.0, -2.5} (descending), lon {0, 5, 10}.
        let ionex = "\
  2023     1     1     0     0     0                        EPOCH OF FIRST MAP
  2023     1     1    12     0     0                        EPOCH OF LAST MAP
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
        for invalid in [
            ionex.replace("1    12     0     0", "1     0     0     0"),
            ionex.replace("1    12     0     0", "1    25     0     0"),
        ] {
            assert!(read_ionex(&invalid).is_err());
        }
        let missing = read_ionex(&ionex.replace("100    110", "9999    110")).unwrap();
        assert!(missing.tec[(0, 0, 0)].is_nan());
        assert!(missing.value(missing.epochs[0], 2.5, 0.0).is_err());
        assert!(missing.value(missing.epochs[0], 2.5, 2.5).is_err());
        assert_eq!(missing.value(missing.epochs[0], 2.5, 5.0).unwrap(), 11.0);
        let midnight = read_ionex(
            &ionex
                .replace(
                    "1    12     0     0                        EPOCH OF LAST MAP",
                    "2     0     0     0                        EPOCH OF LAST MAP",
                )
                .replace(
                    "1    12     0     0                        EPOCH OF CURRENT MAP",
                    "2     0     0     0                        EPOCH OF CURRENT MAP",
                ),
        )
        .unwrap();
        assert_eq!((midnight.epochs[1] - midnight.epochs[0]).num_hours(), 24);
        let maps = read_ionex(ionex).unwrap();
        assert!(maps
            .value(maps.epochs[0] - chrono::Duration::seconds(1), 0.0, 5.0)
            .is_err());
        assert!(maps
            .value(maps.epochs[1] + chrono::Duration::seconds(1), 0.0, 5.0)
            .is_err());
        assert!(maps
            .value(maps.epochs[0] + chrono::Duration::days(1), 0.0, 5.0)
            .is_err());
        assert_eq!(maps.tec.dim(), (2, 3, 3));
        assert_eq!(maps.lats, vec![2.5, 0.0, -2.5]);
        assert_eq!(maps.lons, vec![0.0, 5.0, 10.0]);
        // Actual epochs are midnight and noon.
        // exponent -1 → ×0.1. Grid node (epoch0, lat=0, lon=5) = 210 × 0.1 = 21.0.
        let v = maps.value(maps.epochs[0], 0.0, 5.0).unwrap();
        assert!((v - 21.0).abs() < 1e-9, "node value {v}");
        // Second epoch (720 min) at the same node = 250 × 0.1 = 25.0.
        let ep1 = maps.value(maps.epochs[1], 0.0, 5.0).unwrap();
        assert!((ep1 - 25.0).abs() < 1e-9, "epoch1 node {ep1}");
        // Time-midpoint (360 min) → mean of 21.0 and 25.0.
        let half = maps
            .value(maps.epochs[0] + chrono::Duration::hours(6), 0.0, 5.0)
            .unwrap();
        assert!((half - 23.0).abs() < 1e-9, "time-interp {half}");
    }

    /// Two genuine IGS GIM maps (CDDIS `IGS0OPSFIN_20230010000_01D_02H_GIM.INX`,
    /// maps 1 and 13); see the COMMENT records for the lines kept.
    const REAL_EXCERPT: &str =
        include_str!("../tests/fixtures/IGS0OPSFIN_20230010000_01D_02H_GIM.maps01_13.INX");

    fn utc(day: u32, hour: u32) -> DateTime<Utc> {
        NaiveDate::from_ymd_opt(2023, 1, day)
            .unwrap()
            .and_hms_opt(hour, 0, 0)
            .unwrap()
            .and_utc()
    }

    /// Map 1, lat 87.5, lon -180.0 is the first TEC value (`100`); make it the sentinel.
    fn real_with_missing_first_cell() -> IonexMaps {
        let content = REAL_EXCERPT.replacen("  100  101  102  103", " 9999  101  102  103", 1);
        read_ionex(&content).unwrap()
    }

    /// Clause 1: map times are the `EPOCH OF CURRENT MAP` records, not
    /// `EPOCH OF FIRST MAP + n·INTERVAL` (the excerpt keeps INTERVAL 7200 while
    /// its two maps are 24 h apart).
    #[test]
    fn real_excerpt_map_times_come_from_epoch_records() {
        let maps = read_ionex(REAL_EXCERPT).unwrap();
        assert_eq!(maps.tec.dim(), (2, 71, 73));
        assert_eq!(maps.epochs, vec![utc(1, 0), utc(2, 0)]);
        assert_ne!(
            maps.epochs[1],
            maps.epochs[0] + chrono::Duration::seconds(7200)
        );
        // Map 13 node (lat 87.5, lon -180) is the raw 86 × 10^-1 (source line 5524).
        let node = maps.value(maps.epochs[1], 87.5, -180.0).unwrap();
        assert!((node - 8.6).abs() < 1e-9, "map 13 node {node}");
    }

    /// Clause 2: epochs strictly increase and bracket exactly the header's
    /// `EPOCH OF FIRST MAP` / `EPOCH OF LAST MAP`, else a typed error.
    #[test]
    fn real_excerpt_epochs_must_increase_and_match_header() {
        let last_line =
            "  2023     1     2     0     0     0                        EPOCH OF LAST MAP";
        let first_line =
            "  2023     1     1     0     0     0                        EPOCH OF FIRST MAP";
        let map1 =
            "  2023     1     1     0     0     0                        EPOCH OF CURRENT MAP";
        let map13 =
            "  2023     1     2     0     0     0                        EPOCH OF CURRENT MAP";
        assert!(REAL_EXCERPT.contains(last_line) && REAL_EXCERPT.contains(first_line));
        let last_moved = REAL_EXCERPT.replace(
            last_line,
            &last_line.replace("2     0     0     0", "1    22     0     0"),
        );
        assert!(matches!(
            read_ionex(&last_moved),
            Err(IonexError::EpochMismatch { which: "last", .. })
        ));
        let first_moved = REAL_EXCERPT.replace(
            first_line,
            &first_line.replace("2023     1     1", "2022    12    31"),
        );
        assert!(matches!(
            read_ionex(&first_moved),
            Err(IonexError::EpochMismatch { which: "first", .. })
        ));
        let swapped = REAL_EXCERPT
            .replace(map1, "SWAP")
            .replace(map13, map1)
            .replace("SWAP", map13);
        assert!(matches!(
            read_ionex(&swapped),
            Err(IonexError::EpochOrder { index: 1, .. })
        ));
        let repeated = REAL_EXCERPT.replace(map13, map1);
        assert!(matches!(
            read_ionex(&repeated),
            Err(IonexError::EpochOrder { index: 1, .. })
        ));
    }

    /// Clause 3: an acquisition outside `[first, last]` is a typed coverage
    /// error; the span ends themselves are covered.
    #[test]
    fn real_excerpt_acquisition_outside_span_is_coverage_error() {
        let maps = read_ionex(REAL_EXCERPT).unwrap();
        let (first, last) = (maps.epochs[0], maps.epochs[1]);
        for acquisition in [
            first - chrono::Duration::seconds(1),
            last + chrono::Duration::seconds(1),
            first + chrono::Duration::days(365),
        ] {
            assert!(matches!(
                maps.value(acquisition, 45.0, -110.0),
                Err(IonexError::TemporalCoverage { .. })
            ));
        }
        assert!(maps.value(first, 45.0, -110.0).is_ok());
        assert!(maps.value(last, 45.0, -110.0).is_ok());
        assert!(matches!(
            maps.value(first, 90.0, -110.0),
            Err(IonexError::SpatialCoverage { .. })
        ));
    }

    /// Clause 4: a raw `9999` cell is missing before the EXPONENT scaling is
    /// applied, so `999.9` TECU never enters the cube.
    #[test]
    fn real_excerpt_raw_9999_is_missing_not_tec() {
        let maps = real_with_missing_first_cell();
        assert!(maps.tec[(0, 0, 0)].is_nan());
        assert_eq!(maps.tec.iter().filter(|v| v.is_nan()).count(), 1);
        assert!(maps
            .tec
            .iter()
            .all(|v| v.is_nan() || (0.0..200.0).contains(v)));
        assert!((maps.tec[(0, 0, 1)] - 10.1).abs() < 1e-9);
    }

    /// Clause 5: any positive weight on a missing cell — in time, latitude or
    /// longitude — makes the sample unavailable; zero weight leaves it a number.
    #[test]
    fn real_excerpt_missing_cell_with_weight_is_unavailable() {
        let maps = real_with_missing_first_cell();
        let (t0, t1) = (maps.epochs[0], maps.epochs[1]);
        for (acquisition, lat, lon) in [
            (t0, 87.5, -180.0),
            (t0, 86.25, -180.0),
            (t0, 87.5, -177.5),
            (t0 + chrono::Duration::hours(12), 87.5, -180.0),
        ] {
            assert!(
                matches!(
                    maps.value(acquisition, lat, lon),
                    Err(IonexError::MissingTec {
                        lat: 87.5,
                        lon: -180.0,
                        ..
                    })
                ),
                "({acquisition}, {lat}, {lon}) should be unavailable"
            );
        }
        for (acquisition, lat, lon, want) in [
            (t0, 87.5, -175.0, 10.1),
            (t0, 85.0, -180.0, 10.1),
            (t1, 87.5, -180.0, 8.6),
        ] {
            let got = maps.value(acquisition, lat, lon).unwrap();
            assert!(
                (got - want).abs() < 1e-9,
                "({lat}, {lon}) got {got}, want {want}"
            );
        }
    }
}
