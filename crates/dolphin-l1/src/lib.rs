//! Level-1 front end: raw Sentinel-1 IW SLC bursts and NISAR L1 RSLC in,
//! geocoded CSLC out, on the OPERA burst grid that the phase-linking path reads.
//!
//! Production reads raw Level-1 data only. OPERA CSLC/DISP products are the
//! backtest reference (`validation/issue230_raw_front_end.py`), never an input.
#![warn(missing_docs)]

pub mod asf;
pub mod cdse;
pub mod error;
pub mod orbit;
pub mod safe;

pub use cdse::{BurstId, BurstQuery, BurstRecord, CdseClient, OrbitRecord, Swath};
pub use error::{L1Error, Result};

/// Load `KEY=VALUE` lines from an env file into the process environment.
///
/// Keys already set in the environment win. Blank lines and `#` comments are
/// skipped; one layer of matching single or double quotes is stripped. Values
/// are never logged.
pub fn load_env_file(path: &std::path::Path) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    for line in text.lines().map(str::trim) {
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=').filter(|_| !line.starts_with('#')) else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || std::env::var_os(key).is_some() {
            continue;
        }
        // Called from the single-threaded start of a CLI or test, before any
        // worker threads read the environment.
        std::env::set_var(key, unquote(value.trim()));
    }
    Ok(())
}

fn unquote(value: &str) -> &str {
    ['"', '\'']
        .iter()
        .find_map(|q| value.strip_prefix(*q).and_then(|v| v.strip_suffix(*q)))
        .unwrap_or(value)
}
