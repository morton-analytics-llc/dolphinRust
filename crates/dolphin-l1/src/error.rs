//! Front-end error type.

/// Errors from the Level-1 front end.
#[derive(Debug, thiserror::Error)]
pub enum L1Error {
    /// An HTTP request failed at the transport layer.
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    /// A data service answered with an unexpected status.
    #[error("{service} answered HTTP {status} for {url}: {body}")]
    Service {
        /// Which service (CDSE identity, catalogue, download).
        service: &'static str,
        /// HTTP status code.
        status: u16,
        /// Request URL.
        url: String,
        /// First bytes of the response body.
        body: String,
    },

    /// A download ended before its announced length.
    #[error("download of {url} stopped at {written} of {expected} bytes")]
    Incomplete {
        /// Request URL.
        url: String,
        /// `Content-Length`.
        expected: u64,
        /// Bytes received.
        written: u64,
    },

    /// A required credential is not in the environment.
    #[error("missing credential {0} (set it in the repo .env or the process environment)")]
    MissingCredential(&'static str),

    /// A search returned no product.
    #[error("not found: {0}")]
    NotFound(String),

    /// Local file I/O failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// A zip archive could not be read.
    #[error("zip: {0}")]
    Zip(#[from] zip::result::ZipError),

    /// JSON decoding failed.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// XML could not be parsed.
    #[error("xml: {0}")]
    Xml(#[from] roxmltree::Error),

    /// A GDAL operation failed.
    #[error("gdal: {0}")]
    Gdal(#[from] gdal::errors::GdalError),

    /// An HDF5 operation failed.
    #[error("hdf5: {0}")]
    Hdf5(#[from] hdf5::Error),

    /// Product content is missing a field or holds an invalid value.
    #[error("product: {0}")]
    Product(String),

    /// A geometry solve did not converge or left the valid domain.
    #[error("geometry: {0}")]
    Geometry(String),
}

/// Convenience alias for front-end results.
pub type Result<T> = std::result::Result<T, L1Error>;

/// Build an [`L1Error::Product`] from anything displayable.
pub(crate) fn product(message: impl std::fmt::Display) -> L1Error {
    L1Error::Product(message.to_string())
}
