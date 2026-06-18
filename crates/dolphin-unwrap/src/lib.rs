//! Unwrapping dispatch — port of `dolphin/unwrap/`.
//!
//! Thin orchestration over external unwrappers (SNAPHU, tophu, spurt,
//! whirlwind). dolphin contains no unwrapping math; this crate shells out to
//! the SNAPHU binary and manages tiling, nodata propagation, and connected
//! components. Not a reimplementation target.
#![warn(missing_docs)]

pub mod snaphu;
pub mod tophu;

pub use snaphu::{unwrap, CostMode, InitMethod, UnwrapConfig, UnwrapError, UnwrapResult};
pub use tophu::{unwrap_multiscale, TophuConfig};
