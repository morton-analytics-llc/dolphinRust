//! Phase linking — port of `dolphin/phase_link/`. The numerical core.
//!
//! Covariance estimation over sliding windows (`covariance.py`), EVD (power
//! iteration) and EMI (regularized inverse iteration via Cholesky) estimators
//! (`_core.py`, `_eigenvalues.py`), compressed-SLC generation (`_compress.py`),
//! temporal coherence (`metrics.py`), CRLB (`crlb.py`), and closure phase.
//!
//! Design: JAX `vmap(vmap(f))` over the (rows, cols) pixel grid maps to a
//! `rayon` parallel iterator where each closure solves one NxN complex matrix
//! via `faer`. This is the highest-value module to port first.
#![warn(missing_docs)]

pub mod closure;
pub mod covariance;
pub mod crlb;
pub mod engine;
pub mod estimator;
pub mod fused;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod phasebias;
pub mod quality;
pub mod similarity;
pub mod source_influence;
pub mod source_model;
pub mod spatial_covariance;

pub use closure::estimate_closure_phases;
pub use covariance::{
    estimate_stack_covariance, normalize_numerator_jvp, rect_pixel_source_coherence_jvp,
    rect_source_values_coherence_jvp, replay_rect_pixel_covariance, replay_rect_source_values,
    CovarianceReplayError, NativeSourcePixel, RectPixelReplay, RectReplayDescriptor,
};
pub use crlb::estimate_crlb;
pub use engine::{ComputeEngine, ResolvedBackend};
pub use estimator::{
    phase_angle_jvp, phase_angle_jvp_workspace_bytes, process_coherence_matrices,
    process_coherence_matrix, EstimatorJvpError, FixedEstimatorBranch, PhaseAngleLinearization,
    PixelEstimate, StackEstimate,
};
pub use fused::{
    all_non_finite_acquisition_indices, link_fused, link_fused_with_source_replay,
    AverageCoherenceAggregate, FixedBranchStatus, FusedEstimate, FusedParams, PhaseReplayGrid,
    SourceReplayEstimate,
};
pub use phasebias::{
    correct_phase_bias, estimate_bias_velocity, mean_abs_closure, residual_closure,
};
pub use quality::{
    average_coherence_per_date, compress, compress_pixel_jvp, compress_with_replay,
    estimate_average_coherence, estimate_temp_coh, CompressionJvp, CompressionJvpError,
    CompressionReplayGrid, CompressionReplayStatus,
};
pub use similarity::{circle_offsets, estimate_phase_similarity, PhaseSimilaritySummary};
pub use source_influence::{
    InfluenceDag, InfluenceError, InfluenceNode, NodeId, ParentEdge, ProperComplexFactor,
    SourceDefinition, SourceEdge, SourceId, SourceModelError, TemporalCoordinate,
};
pub use source_model::{
    estimate_empirical_proper_complex_factor, EmpiricalProperComplexConfig,
    EmpiricalProperComplexEstimate, EmpiricalProperComplexReceipt, EmpiricalSourceModelError,
    EMPIRICAL_PROPER_COMPLEX_METHOD, EMPIRICAL_PROPER_COMPLEX_VERSION,
};
pub use spatial_covariance::{
    contract_source_factors, reference_specific_influence_v1, SpatialInfluenceError,
    SpatialInfluenceResult, SpatialInfluenceStatus, EFFECTIVE_LOOKS_MODEL,
    SPATIAL_INFLUENCE_METHOD,
};
