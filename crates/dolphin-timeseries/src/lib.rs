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
pub mod spatial_covariance;
pub mod temporal_covariance;
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
pub use spatial_covariance::{
    convert_covariance_units, date_contrast, solve_fixed_l2_spatial_covariance,
    solve_fixed_l2_spatial_covariance_from_factor, spatial_l2_branch_status, SpatialL2Branch,
    SpatialL2Covariance, SpatialL2Error, SpatialL2FactorCovariance, SpatialL2Status,
    FIXED_L2_SPATIAL_COVARIANCE_METHOD,
};
pub use temporal_covariance::{
    complete_refit_bootstrap_estimate, continuous_time_ar1_correlation, fit_temporal_covariance,
    raw_adjacent_correlation, relative_standard_deviation_shape, subset_origin_anchored_covariance,
    temporal_covariance_provenance, temporal_covariance_workspace_composition,
    temporal_parameter_boundary_status, total_difference_covariance,
    CompleteRefitBootstrapCadenceStatus, CompleteRefitBootstrapEstimate,
    CompleteRefitBootstrapEstimateStatus, RawCorrelationDiagnostics, Sha256Digest,
    TemporalCovarianceApproximation, TemporalCovarianceFit, TemporalCovarianceOptions,
    TemporalCovarianceProvenance, TemporalCovarianceProvenanceInputs,
    TemporalCovarianceWorkspaceComposition, TemporalInferenceStatus, TemporalReferenceProvenance,
    TemporalValidationScope, ValidationInterval, COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS,
    COMPLETE_REFIT_BOOTSTRAP_METHOD, COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
    COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES,
};
pub use velocity_model::{estimate_velocity_with_model, VelocityModel, VelocityModelOutput};
