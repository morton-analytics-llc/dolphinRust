//! Crate-wide error type.

/// Errors raised by `dolphin-core` primitives.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// A config file could not be parsed as the expected YAML schema.
    #[error("config parse error: {0}")]
    ConfigParse(#[from] serde_yaml::Error),

    /// A config value was out of its valid range.
    #[error("invalid config value: {0}")]
    InvalidConfig(String),
}

/// Convenience alias for fallible core operations.
pub type Result<T> = std::result::Result<T, CoreError>;
