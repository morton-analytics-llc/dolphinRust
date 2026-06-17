//! Pipeline orchestration — port of `dolphin/workflows/`.
//!
//! The displacement pipeline (`displacement.py`) in execution order:
//! prepare/group inputs → per-burst wrapped_phase (mask → PS → SHP →
//! covariance → phase-link → compress → ifg network) → stitch bursts →
//! unwrap → timeseries inversion → velocity. Owns the YAML config models
//! (`config/`) and the burst-parallel executor.

pub mod displacement;
pub mod sequential;

pub use displacement::{run_displacement, DisplacementOutput};
pub use sequential::{run_sequential, SequentialConfig, SequentialOutput};
