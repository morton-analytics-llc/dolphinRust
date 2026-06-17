//! Acquisition-date parsing from CSLC filenames.
//!
//! dolphin derives temporal baselines from the dates embedded in CSLC granule
//! names (`opera_utils.get_dates`); recovering them with the configured
//! `cslc_date_fmt` lets velocity carry a true physical rate (mm/yr) instead of
//! an assumed cadence. The scan is regex-free: it renders a reference date in
//! the format to learn the token width, then slides that window across the file
//! name and returns the first substring that parses.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use chrono::NaiveDate;

/// Width (chars) of a date rendered in `fmt`, used to size the scan window.
fn format_width(fmt: &str) -> usize {
    let probe = NaiveDate::from_ymd_opt(2000, 12, 25).expect("constant date is valid");
    probe.format(fmt).to_string().chars().count()
}

/// Parse the acquisition date embedded in one CSLC filename via `fmt`.
///
/// # Errors
/// Returns `Err` if the name is not UTF-8 or carries no substring matching `fmt`.
pub fn parse_date(path: &Path, fmt: &str) -> Result<NaiveDate> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("non-UTF8 CSLC filename: {}", path.display()))?;
    let width = format_width(fmt);
    let chars: Vec<char> = name.chars().collect();
    let last_start = chars.len().saturating_sub(width);
    (0..=last_start)
        .find_map(|i| {
            let window: String = chars[i..i + width].iter().collect();
            NaiveDate::parse_from_str(&window, fmt).ok()
        })
        .ok_or_else(|| anyhow!("no date matching '{fmt}' in {}", path.display()))
}

/// Decimal days of each acquisition relative to the first, parsed from the
/// CSLC filenames in input order. The first element is always `0.0`.
///
/// # Errors
/// Returns `Err` if the list is empty or any filename carries no parseable date.
pub fn decimal_days(paths: &[PathBuf], fmt: &str) -> Result<Vec<f64>> {
    let dates = paths
        .iter()
        .map(|p| parse_date(p, fmt))
        .collect::<Result<Vec<_>>>()?;
    let first = *dates
        .first()
        .ok_or_else(|| anyhow!("empty cslc_file_list"))?;
    Ok(dates
        .iter()
        .map(|d| (*d - first).num_days() as f64)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_name() {
        let d = parse_date(Path::new("/x/cslc_20221119.h5"), "%Y%m%d").unwrap();
        assert_eq!(d, NaiveDate::from_ymd_opt(2022, 11, 19).unwrap());
    }

    #[test]
    fn parses_opera_granule_name() {
        // First 8-digit run is the acquisition date; the later production date is ignored.
        let name =
            "OPERA_L2_CSLC-S1_T064-135518-IW2_20221119T232411Z_20221206T120000Z_S1A_VV_v1.0.h5";
        let d = parse_date(Path::new(name), "%Y%m%d").unwrap();
        assert_eq!(d, NaiveDate::from_ymd_opt(2022, 11, 19).unwrap());
    }

    #[test]
    fn parses_separated_format() {
        let d = parse_date(Path::new("burst_2022-11-19.tif"), "%Y-%m-%d").unwrap();
        assert_eq!(d, NaiveDate::from_ymd_opt(2022, 11, 19).unwrap());
    }

    #[test]
    fn decimal_days_real_cadence() {
        let files = [
            PathBuf::from("cslc_20221119.h5"),
            PathBuf::from("cslc_20221201.h5"),
            PathBuf::from("cslc_20221213.h5"),
        ];
        let days = decimal_days(&files, "%Y%m%d").unwrap();
        assert_eq!(days, vec![0.0, 12.0, 24.0]);
    }
}
