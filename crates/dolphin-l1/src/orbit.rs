//! Orbit state vectors (Sentinel-1 EOF files) and Hermite interpolation.
//!
//! POEORB/RESORB files list ECEF state vectors every 10 s. Between two
//! neighbours the position is a cubic Hermite polynomial in time built from
//! both positions and velocities; its first and second derivatives give the
//! velocity and acceleration. With 10 s spacing the cubic Hermite position
//! error is below a millimetre (fourth-derivative bound ≈ h⁴·a·ω²/384).

use std::path::Path;

use chrono::{DateTime, Duration, NaiveDateTime, Utc};

use crate::error::{product, L1Error, Result};

/// ECEF vector in metres (or m/s, m/s²).
pub type Vec3 = [f64; 3];

/// Orbit ephemeris on a time axis of seconds since `reference_epoch`.
#[derive(Debug, Clone)]
pub struct Orbit {
    /// Epoch the `times` are measured from.
    pub reference_epoch: DateTime<Utc>,
    /// State-vector times, seconds since `reference_epoch`, increasing.
    pub times: Vec<f64>,
    /// ECEF positions (m).
    pub positions: Vec<Vec3>,
    /// ECEF velocities (m/s).
    pub velocities: Vec<Vec3>,
    /// Orbit class from the file name: `POEORB`, `RESORB`, or `annotation`.
    pub orbit_type: String,
}

/// Interpolated platform state.
#[derive(Debug, Clone, Copy)]
pub struct PlatformState {
    /// ECEF position (m).
    pub position: Vec3,
    /// ECEF velocity (m/s).
    pub velocity: Vec3,
    /// ECEF acceleration (m/s²).
    pub acceleration: Vec3,
}

impl Orbit {
    /// Read the state vectors of an EOF file that fall in `[start - margin,
    /// stop + margin]`, timed from `reference_epoch`.
    pub fn from_eof(
        path: &Path,
        reference_epoch: DateTime<Utc>,
        start: DateTime<Utc>,
        stop: DateTime<Utc>,
        margin: Duration,
    ) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let doc = roxmltree::Document::parse(&text)?;
        let mut orbit = Self {
            reference_epoch,
            times: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            orbit_type: orbit_type_from_name(path),
        };
        for osv in doc.descendants().filter(|n| n.has_tag_name("OSV")) {
            let utc = child_text(osv, "UTC")?;
            let time = parse_utc(utc.trim_start_matches("UTC="))?;
            if time < start - margin || time > stop + margin {
                continue;
            }
            let value = |tag: &str| -> Result<f64> {
                child_text(osv, tag)?.parse::<f64>().map_err(product)
            };
            orbit.times.push(seconds_since(reference_epoch, time));
            orbit
                .positions
                .push([value("X")?, value("Y")?, value("Z")?]);
            orbit
                .velocities
                .push([value("VX")?, value("VY")?, value("VZ")?]);
        }
        if orbit.times.len() < 4 {
            return Err(L1Error::Product(format!(
                "{} has {} state vectors in {start} .. {stop}; need at least 4",
                path.display(),
                orbit.times.len()
            )));
        }
        Ok(orbit)
    }

    /// Position, velocity and acceleration at `t` seconds since the epoch.
    pub fn state(&self, t: f64) -> Result<PlatformState> {
        let last = self.times.len() - 1;
        if t < self.times[0] || t > self.times[last] {
            return Err(L1Error::Geometry(format!(
                "time {t:.3} s is outside the orbit span {:.3} .. {:.3} s",
                self.times[0], self.times[last]
            )));
        }
        let i = self.times.partition_point(|&x| x <= t).clamp(1, last) - 1;
        let h = self.times[i + 1] - self.times[i];
        let s = (t - self.times[i]) / h;
        let (p0, p1) = (self.positions[i], self.positions[i + 1]);
        let (v0, v1) = (self.velocities[i], self.velocities[i + 1]);
        let basis = |b: [f64; 4], scale: f64| -> Vec3 {
            std::array::from_fn(|k| {
                (b[0] * p0[k] + b[1] * h * v0[k] + b[2] * p1[k] + b[3] * h * v1[k]) / scale
            })
        };
        let (s2, s3) = (s * s, s * s * s);
        Ok(PlatformState {
            position: basis(
                [
                    2.0 * s3 - 3.0 * s2 + 1.0,
                    s3 - 2.0 * s2 + s,
                    -2.0 * s3 + 3.0 * s2,
                    s3 - s2,
                ],
                1.0,
            ),
            velocity: basis(
                [
                    6.0 * s2 - 6.0 * s,
                    3.0 * s2 - 4.0 * s + 1.0,
                    -6.0 * s2 + 6.0 * s,
                    3.0 * s2 - 2.0 * s,
                ],
                h,
            ),
            acceleration: basis(
                [
                    12.0 * s - 6.0,
                    6.0 * s - 4.0,
                    -12.0 * s + 6.0,
                    6.0 * s - 2.0,
                ],
                h * h,
            ),
        })
    }

    /// Seconds since this orbit's epoch for an absolute time.
    pub fn seconds(&self, time: DateTime<Utc>) -> f64 {
        seconds_since(self.reference_epoch, time)
    }

    /// Absolute time for seconds since this orbit's epoch.
    pub fn datetime(&self, t: f64) -> DateTime<Utc> {
        self.reference_epoch + duration_from_seconds(t)
    }
}

/// `time - epoch` in seconds, to the nanosecond.
pub fn seconds_since(epoch: DateTime<Utc>, time: DateTime<Utc>) -> f64 {
    let delta = time - epoch;
    let whole = delta.num_seconds();
    let nanos = (delta - Duration::seconds(whole))
        .num_nanoseconds()
        .unwrap_or(0);
    whole as f64 + nanos as f64 * 1e-9
}

/// A [`Duration`] from fractional seconds (nanosecond resolution).
pub fn duration_from_seconds(seconds: f64) -> Duration {
    Duration::nanoseconds((seconds * 1e9).round() as i64)
}

/// Parse `YYYY-MM-DDTHH:MM:SS[.ffffff]` (UTC, no zone suffix) or RFC 3339.
pub fn parse_utc(text: &str) -> Result<DateTime<Utc>> {
    let trimmed = text.trim().trim_end_matches('Z');
    NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%dT%H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f"))
        .map(|t| t.and_utc())
        .map_err(|e| L1Error::Product(format!("bad UTC time {text:?}: {e}")))
}

fn child_text<'a>(node: roxmltree::Node<'a, '_>, tag: &str) -> Result<&'a str> {
    node.children()
        .find(|c| c.has_tag_name(tag))
        .and_then(|c| c.text())
        .ok_or_else(|| L1Error::Product(format!("<{}> has no <{tag}>", node.tag_name().name())))
}

fn orbit_type_from_name(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    ["POEORB", "RESORB"]
        .into_iter()
        .find(|k| name.contains(k))
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A circular orbit sampled every 10 s: Hermite interpolation midway
    /// between vectors must reproduce position to < 1 mm, velocity to
    /// < 0.1 mm/s and acceleration to < 0.1 % of ω²r.
    #[test]
    fn hermite_reproduces_circular_orbit() {
        let r = 7.07e6;
        let omega = 1.06e-3;
        let truth = |t: f64| {
            let (s, c) = (omega * t).sin_cos();
            (
                [r * c, r * s, 0.0],
                [-r * omega * s, r * omega * c, 0.0],
                [-r * omega * omega * c, -r * omega * omega * s, 0.0],
            )
        };
        let times: Vec<f64> = (0..12).map(|i| f64::from(i) * 10.0).collect();
        let orbit = Orbit {
            reference_epoch: Utc::now(),
            positions: times.iter().map(|&t| truth(t).0).collect(),
            velocities: times.iter().map(|&t| truth(t).1).collect(),
            times,
            orbit_type: "test".into(),
        };
        let t = 55.0;
        let state = orbit.state(t).unwrap();
        let (p, v, a) = truth(t);
        let err = |x: Vec3, y: Vec3| (0..3).map(|k| (x[k] - y[k]).powi(2)).sum::<f64>().sqrt();
        assert!(err(state.position, p) < 1e-3, "{}", err(state.position, p));
        assert!(err(state.velocity, v) < 1e-4, "{}", err(state.velocity, v));
        assert!(err(state.acceleration, a) < 1e-3 * r * omega * omega);
        assert!(orbit.state(200.0).is_err());
    }

    #[test]
    fn parses_annotation_and_eof_times() {
        let a = parse_utc("2023-01-10T13:34:58.375289").unwrap();
        let b = parse_utc("2023-01-10 13:34:58.375289").unwrap();
        assert_eq!(a, b);
        let epoch = parse_utc("2023-01-10T13:34:00").unwrap();
        assert!((seconds_since(epoch, a) - 58.375289).abs() < 1e-9);
    }
}
