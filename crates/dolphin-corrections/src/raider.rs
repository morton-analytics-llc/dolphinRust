//! RAiDER tropospheric-delay fallback (subprocess), for scenes without an OPERA
//! L4 product.
//!
//! RAiDER is an **optional external dependency**, gated behind an availability
//! check exactly like SNAPHU — never stubbed. The primary tropospheric path is the
//! OPERA L4 product ([`crate::troposphere`]); this fallback only runs when RAiDER
//! is installed (`python -c "import RAiDER"` or `raider.py` on `PATH`). When it is
//! absent the caller receives [`CorrectionError::RaiderUnavailable`] and the
//! tropospheric correction is skipped (deferred), not faked.

use std::path::Path;
use std::process::Command;

use crate::error::{CorrectionError, Result};
use crate::troposphere::{read_l4_netcdf, DelayGrid};

/// Whether RAiDER can be invoked: the Python package imports, or `raider.py` is on
/// `PATH`.
#[must_use]
pub fn raider_available() -> bool {
    let py = Command::new("python")
        .args(["-c", "import RAiDER"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    py || which("raider.py")
}

/// `raider.py` reachable on `PATH`.
fn which(bin: &str) -> bool {
    Command::new("command")
        .args(["-v", bin])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run RAiDER from a config file and ingest the resulting delay netCDF.
///
/// `config` is a RAiDER YAML; `output` is the netCDF RAiDER writes (`var` the
/// delay variable to ingest). Returns the delay grid in meters.
///
/// # Errors
/// [`CorrectionError::RaiderUnavailable`] if RAiDER is not installed (checked
/// first, like SNAPHU); [`CorrectionError::Raider`] if the subprocess fails.
pub fn run_raider(config: &Path, output: &Path, var: &str) -> Result<DelayGrid> {
    if !raider_available() {
        return Err(CorrectionError::RaiderUnavailable);
    }
    let status = Command::new("raider.py")
        .arg("--file")
        .arg(config)
        .status()?;
    if !status.success() {
        return Err(CorrectionError::Raider(format!(
            "raider.py exited with {status}"
        )));
    }
    read_l4_netcdf(output, var)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The availability check returns a bool without panicking, and `run_raider`
    /// surfaces the deferred path (not a stub) when RAiDER is absent.
    #[test]
    fn unavailable_is_deferred_not_stubbed() {
        if raider_available() {
            return; // installed here: nothing to assert about the absent path
        }
        let err = run_raider(Path::new("/none.yaml"), Path::new("/none.nc"), "tropo").unwrap_err();
        assert!(matches!(err, CorrectionError::RaiderUnavailable));
    }
}
