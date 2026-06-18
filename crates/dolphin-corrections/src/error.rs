//! Error type for atmospheric corrections.

use thiserror::Error;

/// Errors raised while computing or applying atmospheric corrections.
#[derive(Debug, Error)]
pub enum CorrectionError {
    /// Underlying I/O failure (file read, subprocess).
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// GDAL failure reading a netCDF / GeoTIFF correction layer.
    #[error("gdal: {0}")]
    Gdal(#[from] gdal::errors::GdalError),
    /// dolphin-io failure (raster read/write).
    #[error(transparent)]
    DolphinIo(#[from] dolphin_io::IoError),
    /// Malformed IONEX TEC file.
    #[error("ionex parse: {0}")]
    Ionex(String),
    /// Array-shape mismatch between a correction layer and the frame grid.
    #[error("shape: {0}")]
    Shape(String),
    /// Corrections were requested but `input_options.wavelength` is unset, so a
    /// range delay in meters cannot be converted to LOS phase.
    #[error("atmospheric corrections require input_options.wavelength to be set")]
    MissingWavelength,
    /// A per-date correction file count did not match the acquisition count.
    #[error("expected {expected} correction files (one per date), got {got}")]
    FileCount {
        /// Number of acquisitions in the stack.
        expected: usize,
        /// Number of correction files supplied.
        got: usize,
    },
    /// RAiDER was requested but is not installed (no `python -c 'import RAiDER'`
    /// and no `raider.py` on PATH). Gated like SNAPHU — never stubbed.
    #[error("RAiDER is not installed; install it or supply troposphere_files (OPERA L4)")]
    RaiderUnavailable,
    /// RAiDER ran but failed.
    #[error("RAiDER subprocess failed: {0}")]
    Raider(String),
}

/// Result alias for the corrections crate.
pub type Result<T> = std::result::Result<T, CorrectionError>;
