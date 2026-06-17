//! I/O error type.

/// Errors from raster/HDF5 I/O.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// A GDAL operation failed.
    #[error("gdal: {0}")]
    Gdal(#[from] gdal::errors::GdalError),

    /// An HDF5 operation failed.
    #[error("hdf5: {0}")]
    Hdf5(#[from] hdf5::Error),

    /// A shape/assembly error.
    #[error("{0}")]
    Shape(String),
}

/// Convenience alias for fallible I/O operations.
pub type Result<T> = std::result::Result<T, IoError>;
