//! SBAS network inversion — port of `dolphin/timeseries.py`.
//!
//! Builds the `(n_ifgs, n_dates-1)` incidence matrix and solves `A φ = Δφ`:
//! weighted L2 least squares (via `faer`) and L1/ADMM (dolphin's default LAD).
//! Plus interferogram-network construction and linear velocity estimation.
//! Block-parallel.
#![warn(missing_docs)]

pub mod inversion;
pub mod loop_closure;
pub mod network;
pub mod reference;
pub mod temporal_validation;
pub mod velocity_model;

pub use inversion::{
    estimate_velocity, estimate_velocity_with_diagnostics, estimate_velocity_with_precisions,
    estimate_velocity_with_uncertainty, get_incidence_matrix, invert_stack, invert_stack_l1,
    invert_stack_with_uncertainty, solve_pixel_with_covariance, L1Config, L2InversionOutput,
    PixelL2Solution, VelocityCadenceStatus, VelocityDiagnosticsOutput, VelocityOutput,
    VelocityUncertaintyStatus,
};
pub use loop_closure::{
    loop_closure_qc, mask_failed_loops, network_triplets, LoopClosureQc, Triplet,
    DEFAULT_CLOSURE_TOLERANCE_CYCLES,
};
pub use network::{build_network, NetworkConfig};
pub use reference::{reference_to_point, select_reference_point};
pub use temporal_validation::{
    preregistered_grid, run_preregistered_validation, TemporalValidationReceipt, ValidationCell,
    ValidationCellResult, ValidationCellStatus, TEMPORAL_VALIDATION_SCHEMA,
};
pub use velocity_model::{estimate_velocity_with_model, VelocityModel, VelocityModelOutput};
