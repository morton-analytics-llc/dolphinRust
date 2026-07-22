//! Pipeline orchestration — port of `dolphin/workflows/`.
//!
//! The displacement pipeline (`displacement.py`) in execution order:
//! prepare/group inputs → per-burst wrapped_phase (mask → PS → SHP →
//! covariance → phase-link → compress → ifg network) → stitch bursts →
//! unwrap → timeseries inversion → velocity. Owns the YAML config models
//! (`config/`) and the burst-parallel executor.
#![warn(missing_docs)]

pub mod burst;
pub mod corrections;
pub mod crop;
pub mod dates;
pub mod displacement;
pub mod provenance;
pub mod sequential;
pub mod tiling;
pub mod unwrap_backend;

pub use crop::{BoundsError, ProcessingBoundsProvenance};
pub use displacement::{
    run_displacement, run_displacement_resumable, update_displacement, DisplacementOutput,
    DisplacementState,
};
pub use provenance::{
    assemble_geometry_provenance, assemble_geometry_provenance_with_bounds,
    write_geometry_provenance, FieldProvenance, GeometryProvenance, GEOMETRY_PROVENANCE_FILENAME,
};
pub use sequential::{
    run_sequential, run_sequential_resumable, update_sequential, SequentialConfig,
    SequentialOutput, SequentialState,
};
pub use unwrap_backend::{NativeUnwrapBackend, SnaphuBackend, TophuBackend, UnwrapBackend};
