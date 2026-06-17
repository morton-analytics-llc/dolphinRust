//! Shared types for the dolphin Rust rebuild.
//!
//! Cross-cutting primitives every downstream crate depends on: complex SLC
//! element types and the `HalfWindow`/`Strides` look geometry ([`types`]), the
//! `StridedBlockManager` tiling scheme ([`blocks`], port of
//! `dolphin/io/_blocks.py`), the workflow config tree ([`config`], mirroring
//! dolphin's pydantic `DisplacementWorkflow`), and the crate error type
//! ([`error`]).
#![warn(missing_docs)]

pub mod blocks;
pub mod config;
pub mod error;
pub mod types;

pub use blocks::{iter_blocks, BlockIndices, StridedBlockManager, TileBlocks, Trim};
pub use error::{CoreError, Result};
pub use types::{Cf32, Cf64, HalfWindow, Strides};
