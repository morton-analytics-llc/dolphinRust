//! Workflow configuration tree, mirroring dolphin's pydantic
//! `DisplacementWorkflow`.
//!
//! Field names and defaults match dolphin so an existing dolphin displacement
//! YAML deserializes unchanged. Unknown fields are ignored (not denied), so the
//! deeply-nested unwrap solver options dolphin emits we don't model (spurt) pass
//! through harmlessly; `snaphu_options`/`tophu_options` are modeled and round-trip.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::types::{HalfWindow, Strides};

/// Runtime disposition of one modeled public workflow-config field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFieldDisposition {
    /// The workflow reads the field on every applicable run.
    Consumed {
        /// Behavior contract that proves the production reader.
        contract_id: &'static str,
    },
    /// The workflow reads the field only when its documented gate is enabled.
    Conditional {
        /// Behavior contract that proves the gated production reader.
        contract_id: &'static str,
        /// Config predicate under which the field affects runtime behavior.
        gate: &'static str,
    },
    /// The field is modeled only so dolphin YAML can deserialize and round-trip.
    CompatibilityOnly {
        /// Why a non-default value is unsupported.
        reason: &'static str,
    },
}

/// One full YAML path and its audited runtime disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigFieldDispositionEntry {
    /// Full path in serialized workflow YAML.
    pub path: &'static str,
    /// Audited runtime handling for the field.
    pub disposition: ConfigFieldDisposition,
}

/// Named behavior contract referenced by the config-field disposition registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigBehaviorContract {
    /// Stable contract identifier.
    pub id: &'static str,
    /// Production reader whose behavior the contract exercises.
    pub reader: &'static str,
    /// Test target that checks the behavior.
    pub evidence: &'static str,
}

/// Checked behavior-contract catalog for consumed and conditional config fields.
pub const CONFIG_BEHAVIOR_CONTRACTS: &[ConfigBehaviorContract] = &[
    ConfigBehaviorContract {
        id: "CFG-INPUT-READ",
        reader: "dolphin-workflows::displacement::{source_layouts, read_burst_tile, acquisition_days, finish_displacement, scale_outputs}",
        evidence: "dolphin-workflows::{displacement_contract, nisar_e2e_contract} and displacement nondefault-date-format mapping contract",
    },
    ConfigBehaviorContract {
        id: "CFG-BOUNDS-CROP",
        reader: "dolphin-workflows::{crop::plan_bounds, displacement::{phase_link_tiled, sequential_config, resolve_burst_geo}}",
        evidence: "dolphin-workflows::displacement_contract::bounded_target_trims_after_analysis_at_both_required_strides",
    },
    ConfigBehaviorContract {
        id: "CFG-PHASE-LINK",
        reader: "dolphin-workflows::displacement::{phase_link_tiled, sequential_config, finish_displacement, apply_phase_bias}",
        evidence: "dolphin-workflows::displacement workflow-to-sequential mapping and phase_bias_correction_runs_end_to_end contracts",
    },
    ConfigBehaviorContract {
        id: "CFG-SHP-WIRING",
        reader: "dolphin-workflows::{displacement::sequential_config, sequential::{shp_neighbors, link_and_compress}}",
        evidence: "dolphin-workflows::displacement workflow-to-sequential mapping contract and shp_wiring_contract",
    },
    ConfigBehaviorContract {
        id: "CFG-LAYOVER-SHADOW",
        reader: "dolphin-workflows::{burst::resolve_layover_shadow_masks, displacement::phase_link_tiled}",
        evidence: "dolphin-workflows::{layover_shadow_mask_contract, multiburst_contract, nrt_incremental_contract, nrt_displacement_contract} and burst/displacement unit contracts",
    },
    ConfigBehaviorContract {
        id: "CFG-IFG-NETWORK",
        reader: "dolphin-workflows::displacement::network",
        evidence: "dolphin-workflows::displacement::tests::workflow_network_options_map_to_network_builder plus dolphin-timeseries::timeseries_contract::networks_match_oracle",
    },
    ConfigBehaviorContract {
        id: "CFG-UNWRAP-BACKEND",
        reader: "dolphin-workflows::displacement::{unwrap_network, unwrap_backend, native_config, unwrap_config, tophu_config}",
        evidence: "dolphin-workflows::displacement workflow-to-backend mapping contracts plus unwrap_backend_contract and unwrap_parallel_contract",
    },
    ConfigBehaviorContract {
        id: "CFG-UNWRAP-MASK",
        reader: "dolphin-workflows::displacement::{unwrap_network, analysis_correlation, apply_phase_masks}",
        evidence: "dolphin-workflows::displacement::tests::{aligned_mask_crs_mismatch_fails_explicitly, mask_is_not_read_when_zero_where_masked_is_false, terrain_and_enabled_unwrap_masks_zero_linked_phase_before_interferograms}",
    },
    ConfigBehaviorContract {
        id: "CFG-TIMESERIES",
        reader: "dolphin-workflows::displacement::{finish_displacement, sequential_config, invert_time_series, fit_velocity, apply_loop_closure_qc}",
        evidence: "dolphin-workflows::{displacement_contract::l2_uncertainty_products_are_opt_in_and_unit_aligned, multiburst_contract} and displacement reference/correlation/loop-closure unit contracts",
    },
    ConfigBehaviorContract {
        id: "CFG-VELOCITY-MODEL",
        reader: "dolphin-workflows::displacement::{sequential_config, velocity_model, fit_velocity}",
        evidence: "dolphin-workflows::displacement velocity-model contracts",
    },
    ConfigBehaviorContract {
        id: "CFG-CORRECTIONS",
        reader: "dolphin-workflows::corrections::apply_corrections",
        evidence: "dolphin-workflows::corrections contracts and geometry_provenance_contract",
    },
    ConfigBehaviorContract {
        id: "CFG-OUTPUT-VALIDITY",
        reader: "dolphin-workflows::displacement::{emit_displacement, write_outputs}",
        evidence: "dolphin-workflows::displacement_contract::{groundpulse_output_policy_preserves_arrays_and_emits_only_coherence, distinct_phase_linking_coherence_raster_is_written_when_enabled}",
    },
    ConfigBehaviorContract {
        id: "CFG-COMPUTE-BACKEND",
        reader: "dolphin-workflows::displacement::{configured_compute_backend, run_displacement_with_output_policy, run_displacement_resumable, update_displacement}",
        evidence: "dolphin-workflows::displacement workflow-backend mapping contract plus dolphin-phaselink::engine_contract and dolphin-workflows::gpu_e2e_contract",
    },
];

macro_rules! consumed {
    ($path:literal, $contract_id:literal) => {
        ConfigFieldDispositionEntry {
            path: $path,
            disposition: ConfigFieldDisposition::Consumed {
                contract_id: $contract_id,
            },
        }
    };
}

macro_rules! conditional {
    ($path:literal, $contract_id:literal, $gate:literal) => {
        ConfigFieldDispositionEntry {
            path: $path,
            disposition: ConfigFieldDisposition::Conditional {
                contract_id: $contract_id,
                gate: $gate,
            },
        }
    };
}

macro_rules! compatibility_only {
    ($path:literal, $reason:literal) => {
        ConfigFieldDispositionEntry {
            path: $path,
            disposition: ConfigFieldDisposition::CompatibilityOnly { reason: $reason },
        }
    };
}

macro_rules! require_compatibility_default {
    ($config:ident, $defaults:ident, $($field:ident).+) => {
        ensure_compatibility_default(
            stringify!($($field).+),
            &$config.$($field).+,
            &$defaults.$($field).+,
        )?
    };
}

/// Audited disposition of every public field in the modeled config tree.
///
/// The companion contract exhaustively destructures every config struct without
/// `..`, checks these full YAML paths for exact coverage and uniqueness, and
/// verifies every behavior-contract ID against [`CONFIG_BEHAVIOR_CONTRACTS`].
pub const CONFIG_FIELD_DISPOSITIONS: &[ConfigFieldDispositionEntry] = &[
    consumed!("input_options", "CFG-INPUT-READ"),
    consumed!("cslc_file_list", "CFG-INPUT-READ"),
    consumed!("output_options", "CFG-BOUNDS-CROP"),
    compatibility_only!(
        "ps_options",
        "persistent-scatterer update inputs are not implemented"
    ),
    compatibility_only!(
        "amplitude_dispersion_files",
        "persistent-scatterer update inputs are not implemented"
    ),
    compatibility_only!(
        "amplitude_mean_files",
        "persistent-scatterer update inputs are not implemented"
    ),
    conditional!(
        "layover_shadow_mask_files",
        "CFG-LAYOVER-SHADOW",
        "layover_shadow_mask_files is nonempty"
    ),
    consumed!("phase_linking", "CFG-PHASE-LINK"),
    consumed!("interferogram_network", "CFG-IFG-NETWORK"),
    consumed!("unwrap_options", "CFG-UNWRAP-BACKEND"),
    consumed!("timeseries_options", "CFG-TIMESERIES"),
    conditional!(
        "correction_options",
        "CFG-CORRECTIONS",
        "correction_options.is_enabled() or geometry_files is nonempty"
    ),
    conditional!(
        "mask_file",
        "CFG-UNWRAP-MASK",
        "unwrap_options.zero_where_masked and mask_file is set"
    ),
    consumed!("work_directory", "CFG-OUTPUT-VALIDITY"),
    consumed!("worker_settings", "CFG-PHASE-LINK"),
    compatibility_only!("log_file", "the Rust CLI configures logging independently"),
    compatibility_only!(
        "ps_options.amp_dispersion_threshold",
        "persistent-scatterer selection and update inputs are not implemented"
    ),
    consumed!("phase_linking.ministack_size", "CFG-PHASE-LINK"),
    consumed!("phase_linking.max_num_compressed", "CFG-PHASE-LINK"),
    consumed!("phase_linking.output_reference_idx", "CFG-PHASE-LINK"),
    consumed!("phase_linking.half_window", "CFG-PHASE-LINK"),
    consumed!("phase_linking.use_evd", "CFG-PHASE-LINK"),
    consumed!("phase_linking.beta", "CFG-PHASE-LINK"),
    consumed!("phase_linking.zero_correlation_threshold", "CFG-PHASE-LINK"),
    consumed!("phase_linking.shp_method", "CFG-SHP-WIRING"),
    consumed!("phase_linking.shp_alpha", "CFG-SHP-WIRING"),
    compatibility_only!(
        "phase_linking.mask_input_ps",
        "the workflow has no persistent-scatterer label input"
    ),
    compatibility_only!(
        "phase_linking.baseline_lag",
        "sequential phase linking does not implement StBAS lag filtering"
    ),
    consumed!("phase_linking.compressed_slc_plan", "CFG-PHASE-LINK"),
    conditional!(
        "phase_linking.write_crlb",
        "CFG-PHASE-LINK",
        "phase_linking.write_crlb is true"
    ),
    conditional!(
        "phase_linking.write_closure_phase",
        "CFG-PHASE-LINK",
        "phase_linking.write_closure_phase is true"
    ),
    conditional!(
        "phase_linking.calc_average_coh",
        "CFG-PHASE-LINK",
        "phase_linking.calc_average_coh is true"
    ),
    conditional!(
        "phase_linking.correct_phase_bias",
        "CFG-PHASE-LINK",
        "phase_linking.correct_phase_bias is true"
    ),
    conditional!(
        "interferogram_network.reference_idx",
        "CFG-IFG-NETWORK",
        "interferogram_network.reference_idx is set"
    ),
    conditional!(
        "interferogram_network.max_bandwidth",
        "CFG-IFG-NETWORK",
        "interferogram_network.max_bandwidth is set"
    ),
    conditional!(
        "interferogram_network.max_temporal_baseline",
        "CFG-IFG-NETWORK",
        "interferogram_network.max_temporal_baseline is set"
    ),
    conditional!(
        "interferogram_network.indexes",
        "CFG-IFG-NETWORK",
        "interferogram_network.indexes is set"
    ),
    compatibility_only!(
        "timeseries_options.run_inversion",
        "the Rust workflow always runs timeseries inversion"
    ),
    consumed!("timeseries_options.method", "CFG-TIMESERIES"),
    conditional!(
        "timeseries_options.reference_point",
        "CFG-TIMESERIES",
        "timeseries_options.reference_point is set"
    ),
    compatibility_only!(
        "timeseries_options.run_velocity",
        "the Rust workflow always estimates velocity"
    ),
    compatibility_only!(
        "timeseries_options.apply_mask_to_timeseries",
        "the Rust workflow always propagates its validity mask"
    ),
    consumed!("timeseries_options.correlation_threshold", "CFG-TIMESERIES"),
    compatibility_only!(
        "timeseries_options.block_shape",
        "timeseries inversion is not configured through block scheduling"
    ),
    compatibility_only!(
        "timeseries_options.num_parallel_blocks",
        "timeseries inversion is not configured through block scheduling"
    ),
    conditional!(
        "timeseries_options.use_coherence_weights",
        "CFG-TIMESERIES",
        "timeseries_options.use_coherence_weights is true"
    ),
    conditional!(
        "timeseries_options.write_posterior_uncertainty",
        "CFG-TIMESERIES",
        "timeseries_options.method is l2 and write_posterior_uncertainty is true"
    ),
    conditional!(
        "timeseries_options.write_velocity_uncertainty",
        "CFG-VELOCITY-MODEL",
        "timeseries_options.write_velocity_uncertainty is true"
    ),
    conditional!(
        "timeseries_options.correct_velocity_temporal_correlation",
        "CFG-VELOCITY-MODEL",
        "write_velocity_uncertainty and correct_velocity_temporal_correlation are true"
    ),
    conditional!(
        "timeseries_options.velocity_seasonal",
        "CFG-VELOCITY-MODEL",
        "timeseries_options.velocity_seasonal is true"
    ),
    conditional!(
        "timeseries_options.velocity_step_dates",
        "CFG-VELOCITY-MODEL",
        "timeseries_options.velocity_step_dates is nonempty"
    ),
    conditional!(
        "timeseries_options.mask_unwrap_loop_errors",
        "CFG-TIMESERIES",
        "timeseries_options.mask_unwrap_loop_errors is true"
    ),
    conditional!(
        "unwrap_options.snaphu_options.ntiles",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is native or snaphu"
    ),
    conditional!(
        "unwrap_options.snaphu_options.tile_overlap",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is snaphu"
    ),
    conditional!(
        "unwrap_options.snaphu_options.n_parallel_tiles",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is snaphu"
    ),
    conditional!(
        "unwrap_options.snaphu_options.init_method",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is snaphu; native requires the pinned default"
    ),
    conditional!(
        "unwrap_options.snaphu_options.cost",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is native or snaphu"
    ),
    compatibility_only!(
        "unwrap_options.snaphu_options.single_tile_reoptimize",
        "the SNAPHU wrapper has no post-tile re-optimization pass"
    ),
    conditional!(
        "unwrap_options.snaphu_options.auto_tile",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is snaphu"
    ),
    conditional!(
        "unwrap_options.tophu_options.ntiles",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is tophu"
    ),
    conditional!(
        "unwrap_options.tophu_options.downsample_factor",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is tophu"
    ),
    conditional!(
        "unwrap_options.tophu_options.init_method",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is tophu"
    ),
    conditional!(
        "unwrap_options.tophu_options.cost",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is tophu"
    ),
    compatibility_only!(
        "unwrap_options.preprocess_options.alpha",
        "pre-unwrap Goldstein filtering is not implemented"
    ),
    compatibility_only!(
        "unwrap_options.preprocess_options.max_radius",
        "pre-unwrap interpolation is not implemented"
    ),
    compatibility_only!(
        "unwrap_options.preprocess_options.interpolation_cor_threshold",
        "pre-unwrap interpolation is not implemented"
    ),
    compatibility_only!(
        "unwrap_options.preprocess_options.interpolation_similarity_threshold",
        "pre-unwrap interpolation is not implemented"
    ),
    compatibility_only!(
        "unwrap_options.preprocess_options.zero_correlation_where_interpolating",
        "pre-unwrap interpolation is not implemented"
    ),
    compatibility_only!(
        "unwrap_options.run_unwrap",
        "the Rust workflow always unwraps its interferogram network"
    ),
    compatibility_only!(
        "unwrap_options.run_goldstein",
        "pre-unwrap Goldstein filtering is not implemented"
    ),
    compatibility_only!(
        "unwrap_options.run_interpolation",
        "pre-unwrap interpolation is not implemented"
    ),
    consumed!("unwrap_options.unwrap_method", "CFG-UNWRAP-BACKEND"),
    consumed!("unwrap_options.n_parallel_jobs", "CFG-UNWRAP-BACKEND"),
    conditional!(
        "unwrap_options.zero_where_masked",
        "CFG-UNWRAP-MASK",
        "unwrap_options.zero_where_masked is true and mask_file is set"
    ),
    compatibility_only!(
        "unwrap_options.preprocess_options",
        "pre-unwrap filtering and interpolation are not implemented"
    ),
    conditional!(
        "unwrap_options.snaphu_options",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is native or snaphu"
    ),
    conditional!(
        "unwrap_options.tophu_options",
        "CFG-UNWRAP-BACKEND",
        "unwrap_method is tophu"
    ),
    consumed!("input_options.input_type", "CFG-INPUT-READ"),
    consumed!("input_options.subdataset", "CFG-INPUT-READ"),
    consumed!("input_options.cslc_date_fmt", "CFG-INPUT-READ"),
    consumed!("input_options.wavelength", "CFG-INPUT-READ"),
    conditional!(
        "correction_options.ionosphere_files",
        "CFG-CORRECTIONS",
        "correction_options.ionosphere_files is nonempty"
    ),
    conditional!(
        "correction_options.troposphere_files",
        "CFG-CORRECTIONS",
        "correction_options.troposphere_files is nonempty"
    ),
    conditional!(
        "correction_options.geometry_files",
        "CFG-CORRECTIONS",
        "correction_options.geometry_files is nonempty"
    ),
    conditional!(
        "correction_options.dem_file",
        "CFG-CORRECTIONS",
        "troposphere_files is nonempty and dem_file is set"
    ),
    conditional!(
        "correction_options.incidence_angle_deg",
        "CFG-CORRECTIONS",
        "ionosphere_files or troposphere_files is nonempty and geometry_files is empty"
    ),
    conditional!(
        "correction_options.troposphere_variable",
        "CFG-CORRECTIONS",
        "correction_options.troposphere_files is nonempty"
    ),
    conditional!(
        "correction_options.solid_earth_tide",
        "CFG-CORRECTIONS",
        "correction_options.solid_earth_tide is true"
    ),
    consumed!("output_options.strides", "CFG-BOUNDS-CROP"),
    conditional!(
        "output_options.epsg",
        "CFG-BOUNDS-CROP",
        "output_options.bounds and output_options.epsg are set"
    ),
    conditional!(
        "output_options.bounds",
        "CFG-BOUNDS-CROP",
        "output_options.bounds is set"
    ),
    conditional!(
        "output_options.bounds_epsg",
        "CFG-BOUNDS-CROP",
        "output_options.bounds is set"
    ),
    compatibility_only!(
        "output_options.add_overviews",
        "the COG writer controls overview generation"
    ),
    compatibility_only!(
        "output_options.overview_levels",
        "the COG writer controls overview generation"
    ),
    compatibility_only!(
        "worker_settings.gpu_enabled",
        "worker_settings.compute_backend supersedes this dolphin field"
    ),
    consumed!("worker_settings.compute_backend", "CFG-COMPUTE-BACKEND"),
    compatibility_only!(
        "worker_settings.threads_per_worker",
        "the Rust workflow does not set a per-worker thread environment"
    ),
    compatibility_only!(
        "worker_settings.n_parallel_bursts",
        "the Rust workflow does not schedule bursts through this field"
    ),
    consumed!("worker_settings.block_shape", "CFG-PHASE-LINK"),
];

fn ensure_compatibility_default<T>(path: &str, actual: &T, default: &T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if actual == default {
        return Ok(());
    }
    Err(CoreError::InvalidConfig(format!(
        "{path} is modeled only for dolphin YAML compatibility and is not supported; expected the default {default:?}, got {actual:?}"
    )))
}

fn ensure_supported_choice(path: &str, actual: &str, allowed: &[&str]) -> Result<()> {
    if allowed.contains(&actual) {
        return Ok(());
    }
    Err(CoreError::InvalidConfig(format!(
        "{path} has unsupported value {actual:?}; expected one of {allowed:?}"
    )))
}

/// SHP-selection statistical test. dolphin `ShpMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShpMethod {
    /// Generalized likelihood ratio test.
    #[default]
    Glrt,
    /// Kolmogorov-Smirnov two-sample test.
    Ks,
    /// No SHP search; use the full rectangular window.
    Rect,
}

/// Compressed-SLC carry-forward plan. dolphin `CompressedSlcPlan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompressedSlcPlan {
    /// Always reference the first date of the first ministack.
    #[default]
    AlwaysFirst,
    /// Reference the first date of each ministack.
    FirstPerMinistack,
    /// Reference the last date of each ministack.
    LastPerMinistack,
}

/// Phase-unwrapping backend. dolphin `UnwrapMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UnwrapMethod {
    /// SNAPHU statistical-cost network-flow unwrapper. Selectable fallback; the
    /// default is [`UnwrapMethod::Native`], which matches SNAPHU per-component to
    /// <=0.5% but unwraps in-process with lower subprocess and scratch-I/O
    /// overhead.
    Snaphu,
    /// tophu multi-scale driver over the SNAPHU per-tile solver (coarse init →
    /// overlapping tiled SNAPHU → 2π-reconciled merge). dolphin reserves its
    /// `multiscale_unwrap` for the ICU/PHASS solvers; dolphinRust exposes it as a
    /// first-class method driving SNAPHU, the solver we ship. Configured by
    /// [`TophuOptions`].
    Tophu,
    /// ISCE ICU (residue-cut) unwrapper.
    Icu,
    /// ISCE PHASS unwrapper.
    Phass,
    /// spurt 3D temporal/spatial unwrapper.
    Spurt,
    /// Whirlwind unwrapper.
    Whirlwind,
    /// Clean-room in-process native unwrapper (Costantini MCF via network
    /// simplex, no SNAPHU subprocess) — **the default**. Auto-tiling keeps a
    /// 64-pixel core floor, which holds the <=0.5% SNAPHU-parity contract on the
    /// MMX1 live common frame while remaining faster there. Explicit
    /// `unwrap_options.snaphu_options.ntiles` overrides auto-tiling; per-frame
    /// thread count (`n_parallel_jobs`/the rayon pool) is the latency/throughput
    /// dial. Set `unwrap_method: snaphu` to fall back.
    #[default]
    Native,
}

/// Timeseries inversion norm. dolphin `TimeseriesOptions.method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimeseriesMethod {
    /// L1 (least-absolute-deviations) norm.
    #[default]
    L1,
    /// L2 (least-squares) norm.
    L2,
}

/// Persistent-scatterer selection. dolphin `PsOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PsOptions {
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// persistent-scatterer selection/update inputs are not implemented.
    pub amp_dispersion_threshold: f64,
}

impl Default for PsOptions {
    fn default() -> Self {
        Self {
            amp_dispersion_threshold: 0.25,
        }
    }
}

/// Phase-linking (covariance + EMI/EVD) options. dolphin `PhaseLinkingOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseLinkingOptions {
    /// Size of the ministack for the sequential estimator.
    pub ministack_size: usize,
    /// Maximum number of compressed images to use in the sequential estimator.
    pub max_num_compressed: usize,
    /// Index of the input SLC to reference for phase-linked interferograms after EVD/EMI.
    pub output_reference_idx: Option<usize>,
    /// Half-window size for multilooking during phase linking.
    pub half_window: HalfWindow,
    /// Use EVD on the coherence instead of the EMI algorithm.
    pub use_evd: bool,
    /// Beta regularization parameter for correlation-matrix inversion; 0 is none.
    pub beta: f64,
    /// Snap coherence-matrix correlation values below this threshold to 0.
    pub zero_correlation_threshold: f64,
    /// Statistical test used to find SHPs during phase linking.
    pub shp_method: ShpMethod,
    /// Significance level (false-alarm probability) for the SHP test.
    pub shp_alpha: f64,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// the workflow has no persistent-scatterer label input.
    pub mask_input_ps: bool,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// sequential phase linking does not implement StBAS lag filtering.
    pub baseline_lag: Option<i64>,
    /// Plan for which date each ministack's compressed SLC references.
    pub compressed_slc_plan: CompressedSlcPlan,
    /// Write the Cramer-Rao lower bound raster.
    pub write_crlb: bool,
    /// Write the closure-phase raster.
    pub write_closure_phase: bool,
    /// Calculate average coherence magnitude per SLC date (dolphin
    /// `calc_average_coh`) and emit the distinct phase-linking-coherence raster.
    pub calc_average_coh: bool,
    /// Apply the phase-bias / non-closure correction (Michaelides et al. 2022) to
    /// the linked-phase series before the interferogram network. **Off by default**
    /// (this leads Python dolphin, which has no such correction; enabling it changes
    /// the output). Forces closure-phase computation when on. Forward divergence.
    pub correct_phase_bias: bool,
}

impl Default for PhaseLinkingOptions {
    fn default() -> Self {
        Self {
            ministack_size: 15,
            max_num_compressed: 10,
            output_reference_idx: None,
            half_window: HalfWindow::default(),
            use_evd: false,
            beta: 0.0,
            zero_correlation_threshold: 0.0,
            shp_method: ShpMethod::default(),
            shp_alpha: 0.001,
            mask_input_ps: false,
            baseline_lag: None,
            compressed_slc_plan: CompressedSlcPlan::default(),
            write_crlb: true,
            write_closure_phase: false,
            calc_average_coh: false,
            correct_phase_bias: false,
        }
    }
}

/// Interferogram-network construction. dolphin `InterferogramNetwork`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InterferogramNetwork {
    /// Single-reference network: index of the reference image.
    pub reference_idx: Option<usize>,
    /// Max `n` to form the nearest-`n` interferograms by index.
    pub max_bandwidth: Option<usize>,
    /// Maximum temporal baseline of interferograms.
    pub max_temporal_baseline: Option<f64>,
    /// Manual-index network: list of (ref_idx, sec_idx) interferograms to form.
    pub indexes: Option<Vec<(usize, usize)>>,
}

/// Timeseries inversion + velocity. dolphin `TimeseriesOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TimeseriesOptions {
    /// Dolphin YAML compatibility only; must remain `true` because the Rust
    /// workflow always runs timeseries inversion.
    pub run_inversion: bool,
    /// Norm to use during timeseries inversion.
    pub method: TimeseriesMethod,
    /// Reference point (row, col); auto-selected if not provided.
    pub reference_point: Option<(usize, usize)>,
    /// Dolphin YAML compatibility only; must remain `true` because the Rust
    /// workflow always estimates velocity.
    pub run_velocity: bool,
    /// Dolphin YAML compatibility only; must remain `true` because the Rust
    /// workflow always propagates its validity mask.
    pub apply_mask_to_timeseries: bool,
    /// Pixels with correlation below this value are masked out.
    pub correlation_threshold: f64,
    /// Dolphin YAML compatibility only; non-default block scheduling is rejected.
    pub block_shape: (usize, usize),
    /// Dolphin YAML compatibility only; non-default block scheduling is rejected.
    pub num_parallel_blocks: usize,
    /// Use CRLB-derived observation precision for L2 SBAS and velocity fits.
    pub use_coherence_weights: bool,
    /// Emit L2 posterior displacement variance and residual RMS products.
    pub write_posterior_uncertainty: bool,
    /// Emit velocity one-sigma uncertainty.
    pub write_velocity_uncertainty: bool,
    /// Inflate the velocity one-sigma by the AR(1) effective-sample-size factor
    /// `sqrt((1+rho)/(1-rho))` (Zhang et al. 1997 / Agram & Zebker 2015), `rho`
    /// the lag-1 autocorrelation of the velocity-fit residuals. InSAR series
    /// carry temporally correlated noise, so the uncorrected sigma understates
    /// the slope uncertainty. **Forward divergence from dolphin, opt-in and off
    /// by default** — a larger sigma can flip a downstream risk-tier threshold,
    /// so enabling it is the reviewed rollout. Requires
    /// `write_velocity_uncertainty`.
    pub correct_velocity_temporal_correlation: bool,
    /// Fit an annual sinusoid (period 365.25 d) jointly with the linear rate, so
    /// a real seasonal cycle (groundwater, thermal) is reported as an amplitude
    /// and phase instead of leaking into the rate. **Forward divergence from
    /// dolphin, opt-in and off by default** — dolphin's `velocity.py` is
    /// linear-only, and with this false the velocity fit is the untouched
    /// degree-1 one. Emits `velocity_seasonal_amplitude.tif` and
    /// `velocity_seasonal_phase_days.tif`.
    pub velocity_seasonal: bool,
    /// Acquisition dates (`YYYY-MM-DD`) at which to fit a Heaviside step jointly
    /// with the linear rate — a co-seismic offset, an instrument change, a known
    /// anthropogenic event. Each date adds one basis column and emits
    /// `velocity_step_NN.tif` in list order. The epoch is an **input**, never
    /// detected from the data: a step whose timing is fitted is a different
    /// (nonlinear) estimator with its own failure modes. Empty by default.
    pub velocity_step_dates: Vec<String>,
    /// Close every triangle in the **unwrapped** interferogram network before the
    /// SBAS solve and blank pixels whose loops miss closure by more than half a
    /// cycle — a 2π unwrap error, which the wrapped closure-phase layer cannot
    /// see (`.arg()` maps a whole cycle to zero). Forward divergence from
    /// dolphin, **off by default**. A no-op on a single-reference network, which
    /// has no loops; needs `interferogram_network.max_bandwidth` or
    /// `max_temporal_baseline`. Emits `loop_closure_bad_count.tif` and
    /// `loop_closure_worst_cycles.tif`. See issue #24.
    pub mask_unwrap_loop_errors: bool,
}

impl Default for TimeseriesOptions {
    fn default() -> Self {
        Self {
            run_inversion: true,
            method: TimeseriesMethod::default(),
            reference_point: None,
            run_velocity: true,
            apply_mask_to_timeseries: true,
            correlation_threshold: 0.2,
            block_shape: (256, 256),
            num_parallel_blocks: 4,
            use_coherence_weights: true,
            write_posterior_uncertainty: false,
            write_velocity_uncertainty: false,
            correct_velocity_temporal_correlation: false,
            velocity_seasonal: false,
            velocity_step_dates: Vec::new(),
            mask_unwrap_loop_errors: false,
        }
    }
}

/// SNAPHU subprocess options. dolphin `SnaphuOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SnaphuOptions {
    /// Number of tiles (row, col) to split inputs into via SNAPHU's internal tiling.
    pub ntiles: (usize, usize),
    /// Tile overlap (in pixels) along the (row, col) directions.
    pub tile_overlap: (usize, usize),
    /// Number of tiles to unwrap in parallel for each interferogram.
    pub n_parallel_tiles: usize,
    /// SNAPHU initialization method (`mcf` or `mst`).
    pub init_method: String,
    /// SNAPHU statistical cost mode (`defo` or `smooth`).
    pub cost: String,
    /// Dolphin YAML compatibility only; must remain `false` because the SNAPHU
    /// wrapper has no post-tile re-optimization pass.
    pub single_tile_reoptimize: bool,
    /// **dolphinRust-only, opt-in.** When set, derive `ntiles`/`n_parallel_tiles`
    /// from the grid size and available cores instead of the explicit values
    /// above. Changes SNAPHU numerics (tile boundaries/reconciliation), so it is
    /// off by default and gated against the oracle; absent in dolphin YAML it
    /// deserializes to `false` and the config round-trips unchanged.
    pub auto_tile: bool,
}

impl Default for SnaphuOptions {
    fn default() -> Self {
        Self {
            ntiles: (1, 1),
            tile_overlap: (0, 0),
            n_parallel_tiles: 1,
            init_method: "mcf".into(),
            cost: "smooth".into(),
            single_tile_reoptimize: false,
            auto_tile: false,
        }
    }
}

/// tophu multi-scale unwrap options. dolphin `TophuOptions` (same field names,
/// so a real dolphin YAML's `tophu_options` block round-trips).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TophuOptions {
    /// Number of tiles (row, col) to split the full-res grid into for the fine pass.
    pub ntiles: (usize, usize),
    /// Extra multilook factor (row, col) for the coarse-pass downsample.
    pub downsample_factor: (usize, usize),
    /// SNAPHU initialization method (`mcf` or `mst`).
    pub init_method: String,
    /// SNAPHU statistical cost mode (`defo` or `smooth`).
    pub cost: String,
}

impl Default for TophuOptions {
    fn default() -> Self {
        Self {
            ntiles: (1, 1),
            downsample_factor: (1, 1),
            init_method: "mcf".into(),
            cost: "smooth".into(),
        }
    }
}

/// Pre-unwrap filtering/interpolation. dolphin `PreprocessOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreprocessOptions {
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// pre-unwrap Goldstein filtering is not implemented.
    pub alpha: f64,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// pre-unwrap interpolation is not implemented.
    pub max_radius: usize,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// pre-unwrap interpolation is not implemented.
    pub interpolation_cor_threshold: f64,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// pre-unwrap interpolation is not implemented.
    pub interpolation_similarity_threshold: f64,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// pre-unwrap interpolation is not implemented.
    pub zero_correlation_where_interpolating: bool,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            max_radius: 51,
            interpolation_cor_threshold: 0.3,
            interpolation_similarity_threshold: 0.3,
            zero_correlation_where_interpolating: false,
        }
    }
}

/// Unwrapping dispatch options. dolphin `UnwrapOptions` (solver-specific nested
/// blocks beyond SNAPHU are left to pass through as ignored fields).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UnwrapOptions {
    /// Dolphin YAML compatibility only; must remain `true` because the Rust
    /// workflow always unwraps the interferogram network.
    pub run_unwrap: bool,
    /// Dolphin YAML compatibility only; must remain `false` because pre-unwrap
    /// Goldstein filtering is not implemented.
    pub run_goldstein: bool,
    /// Dolphin YAML compatibility only; must remain `false` because pre-unwrap
    /// interpolation is not implemented.
    pub run_interpolation: bool,
    /// Phase-unwrapping backend to dispatch to.
    pub unwrap_method: UnwrapMethod,
    /// Number of interferograms to unwrap in parallel.
    pub n_parallel_jobs: i64,
    /// Set wrapped phase/correlation to 0 where the mask is 0 before unwrapping.
    pub zero_where_masked: bool,
    /// Dolphin YAML compatibility container for unsupported pre-unwrap
    /// filtering/interpolation options.
    pub preprocess_options: PreprocessOptions,
    /// SNAPHU subprocess options.
    pub snaphu_options: SnaphuOptions,
    /// tophu multi-scale options (used when `unwrap_method` is `tophu`).
    pub tophu_options: TophuOptions,
}

impl Default for UnwrapOptions {
    fn default() -> Self {
        Self {
            run_unwrap: true,
            run_goldstein: false,
            run_interpolation: false,
            unwrap_method: UnwrapMethod::default(),
            n_parallel_jobs: -1,
            zero_where_masked: false,
            preprocess_options: PreprocessOptions::default(),
            snaphu_options: SnaphuOptions::default(),
            tophu_options: TophuOptions::default(),
        }
    }
}

/// Input-product reader selection. **Forward divergence:** dolphin v0.35.0 has
/// no product-type field on `InputOptions` (it dispatches by workflow entry
/// point), so this field is dolphinRust-only. It deserializes to the OPERA
/// default when absent, so an existing dolphin YAML round-trips unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
    /// OPERA S1 CSLC: complex-f32 `(r, i)` HDF5 grids (the dolphin default).
    #[default]
    OperaCslc,
    /// NISAR L-band geocoded SLC: complex-`f32` `{r, i}` compound grids in the
    /// NISAR product group layout (camelCase coordinates + `epsg_code`
    /// attribute). Differs from OPERA only in the geocoding-grid metadata.
    NisarGslc,
}

/// Input granule discovery. dolphin `InputOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InputOptions {
    /// Input-product reader to use (OPERA CSLC vs NISAR GSLC). Forward
    /// divergence from dolphin v0.35.0 (see [`InputType`]).
    pub input_type: InputType,
    /// Subdataset to use from HDF5/NetCDF CSLC files. For NISAR this is the
    /// polarization grid path, e.g. `/science/LSAR/GSLC/grids/frequencyA/HH`.
    pub subdataset: Option<String>,
    /// Format of dates contained in CSLC filenames.
    pub cslc_date_fmt: String,
    /// Radar wavelength (meters); used to convert timeseries radians to meters.
    /// S1 C-band ≈ 0.0555; NISAR L-band ≈ 0.24.
    pub wavelength: Option<f64>,
}

impl Default for InputOptions {
    fn default() -> Self {
        Self {
            input_type: InputType::default(),
            subdataset: None,
            cslc_date_fmt: "%Y%m%d".into(),
            wavelength: None,
        }
    }
}

/// Auxiliary atmospheric-correction options. dolphin `CorrectionOptions`
/// (`ionosphere_files`, `geometry_files`, `dem_file`). Corrections are **opt-in**:
/// with every file list empty (the default) the displacement output is unchanged.
///
/// **Forward divergence:** dolphin derives the tropospheric delay from a DEM via
/// RAiDER and has no `troposphere_files` field; dolphinRust adds `troposphere_files`
/// for direct ingest of the public OPERA L4 tropospheric product (one netCDF per
/// date), with RAiDER as the fallback. `incidence_angle_deg` and
/// `troposphere_variable` are dolphinRust-only knobs for the delay projection and
/// the L4 netCDF variable name. dolphin's keys deserialize unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CorrectionOptions {
    /// GNSS-derived IONEX TEC maps for ionospheric correction (one per date).
    /// Source: <https://cddis.nasa.gov/archive/gnss/products/ionex/>. dolphin name.
    pub ionosphere_files: Vec<PathBuf>,
    /// OPERA L4 tropospheric netCDF products (one per date). dolphinRust forward
    /// divergence (dolphin uses `dem_file` + RAiDER instead).
    pub troposphere_files: Vec<PathBuf>,
    /// Line-of-sight geometry files resolved even in a geometry-only run and
    /// used by correction computations when enabled. The delay projection uses
    /// `incidence_angle_deg` when no geometry is resolved. dolphin name.
    pub geometry_files: Vec<PathBuf>,
    /// DEM file for tropospheric/topographic corrections (RAiDER path). dolphin name.
    pub dem_file: Option<PathBuf>,
    /// Incidence angle (degrees) used to project zenith delay to line-of-sight when
    /// no geometry file is supplied. dolphinRust-only; default 37° (NISAR nominal).
    pub incidence_angle_deg: f64,
    /// netCDF variable to read from the OPERA L4 product. dolphinRust-only.
    /// `"total"` (the default) sums the real product's `hydrostatic_delay` +
    /// `wet_delay` zenith fields; any other value reads that single variable.
    pub troposphere_variable: String,
    /// Subtract the lunisolar solid-earth tide (issue #21). Unlike the other
    /// corrections this needs **no external data file** — only the acquisition
    /// time from the granule name and per-pixel LOS geometry — so it is gated by
    /// a flag rather than by a file list. dolphinRust-only forward divergence,
    /// **off by default**. Requires `geometry_files`: the tide is a 3-D
    /// displacement vector and projecting it into line of sight needs the full
    /// LOS unit vector, which the scalar `incidence_angle_deg` cannot supply.
    pub solid_earth_tide: bool,
}

impl Default for CorrectionOptions {
    fn default() -> Self {
        Self {
            ionosphere_files: Vec::new(),
            troposphere_files: Vec::new(),
            geometry_files: Vec::new(),
            dem_file: None,
            incidence_angle_deg: 37.0,
            troposphere_variable: "total".into(),
            solid_earth_tide: false,
        }
    }
}

impl CorrectionOptions {
    /// Whether any correction is enabled (any correction file supplied, or the
    /// file-free solid-earth-tide flag set).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        !self.ionosphere_files.is_empty()
            || !self.troposphere_files.is_empty()
            || self.solid_earth_tide
    }
}

/// Output grid + raster options. dolphin `OutputOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputOptions {
    /// (x, y) strides (decimation factor) to apply while processing input.
    pub strides: Strides,
    /// Optional bounded-run assertion that the source frame has this EPSG.
    /// Output reprojection and missing-source-CRS fallback are not implemented.
    pub epsg: Option<u32>,
    /// Area of interest as [left, bottom, right, top] coordinates.
    pub bounds: Option<(f64, f64, f64, f64)>,
    /// EPSG code for the `bounds` coordinates.
    pub bounds_epsg: Option<u32>,
    /// Dolphin YAML compatibility only; must remain at its default because the
    /// COG writer owns overview generation.
    pub add_overviews: bool,
    /// Dolphin YAML compatibility only; must remain at its default because the
    /// COG writer owns overview generation.
    pub overview_levels: Vec<u32>,
}

impl Default for OutputOptions {
    fn default() -> Self {
        Self {
            strides: Strides::default(),
            epsg: None,
            bounds: None,
            bounds_epsg: Some(4326),
            add_overviews: true,
            overview_levels: vec![4, 8, 16, 32, 64],
        }
    }
}

/// Compute backend for phase linking (covariance + EVD/EMI). dolphin exposes a
/// bool `gpu_enabled`; we generalize to a tri-state. **The default is `Cpu`** (the
/// f64 correctness reference). `Gpu` and `Auto` are opt-in: on integrated GPUs the
/// CPU path is faster end-to-end — the GPU's win is on discrete hardware. See the
/// performance note in `bench/GPU.md` before selecting them. With no GPU adapter
/// (or a `no-gpu` build) every mode falls back to the CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ComputeBackend {
    /// Always the CPU (faer, f64) reference path. The default.
    #[default]
    Cpu,
    /// Size-based: GPU at/above the ~128² kernel crossover, CPU below; CPU if no
    /// GPU. Note the crossover is kernel-only — end-to-end on an integrated GPU the
    /// CPU is faster, so prefer explicit `Gpu` only on discrete hardware.
    Auto,
    /// GPU where supported; automatic CPU fallback if no adapter / unsupported.
    Gpu,
}

/// Parallelism / worker settings. dolphin `WorkerSettings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerSettings {
    /// Dolphin YAML compatibility only; must remain `false` because
    /// `compute_backend` is the operational backend selector.
    pub gpu_enabled: bool,
    /// Compute backend selection for phase linking (`auto` / `cpu` / `gpu`).
    pub compute_backend: ComputeBackend,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// the Rust workflow does not set a per-worker thread environment.
    pub threads_per_worker: usize,
    /// Dolphin YAML compatibility only; non-default values are rejected because
    /// the Rust workflow does not schedule bursts through this field.
    pub n_parallel_bursts: usize,
    /// Size (rows, columns) of data blocks to load at a time.
    pub block_shape: (usize, usize),
}

impl Default for WorkerSettings {
    fn default() -> Self {
        Self {
            gpu_enabled: false,
            compute_backend: ComputeBackend::default(),
            threads_per_worker: 1,
            n_parallel_bursts: 1,
            block_shape: (512, 512),
        }
    }
}

/// Top-level displacement workflow config. dolphin `DisplacementWorkflow`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplacementWorkflow {
    /// Options specifying the input datasets.
    pub input_options: InputOptions,
    /// List of input CSLC files.
    pub cslc_file_list: Vec<PathBuf>,
    /// Output size/format/compression options.
    pub output_options: OutputOptions,
    /// Dolphin YAML compatibility container for unsupported PS update options.
    pub ps_options: PsOptions,
    /// Dolphin YAML compatibility only; nonempty PS-update inputs are rejected.
    pub amplitude_dispersion_files: Vec<PathBuf>,
    /// Dolphin YAML compatibility only; nonempty PS-update inputs are rejected.
    pub amplitude_mean_files: Vec<PathBuf>,
    /// Single-band native-grid GTiff layover/shadow masks, one per active burst. Zero,
    /// non-finite, raster nodata, and GDAL-invalid pixels are invalid; every
    /// finite nonzero pixel is valid. Masks are resolved by OPERA burst ID and
    /// applied before phase linking.
    pub layover_shadow_mask_files: Vec<PathBuf>,
    /// Phase-linking (wrapped-phase estimation) options.
    pub phase_linking: PhaseLinkingOptions,
    /// Interferogram-network construction options.
    pub interferogram_network: InterferogramNetwork,
    /// Unwrapping dispatch options.
    pub unwrap_options: UnwrapOptions,
    /// Timeseries inversion and velocity options.
    pub timeseries_options: TimeseriesOptions,
    /// Auxiliary atmospheric (ionospheric/tropospheric) correction options.
    pub correction_options: CorrectionOptions,
    /// Mask file used to ignore low-correlation/bad data (0 = invalid, 1 = good).
    pub mask_file: Option<PathBuf>,
    /// Sub-directory for writing output files.
    pub work_directory: PathBuf,
    /// CPU/GPU and parallelism settings.
    pub worker_settings: WorkerSettings,
    /// Dolphin YAML compatibility only; the Rust CLI configures logging
    /// independently and rejects a nonempty value.
    pub log_file: Option<PathBuf>,
}

impl Default for DisplacementWorkflow {
    fn default() -> Self {
        Self {
            input_options: InputOptions::default(),
            cslc_file_list: Vec::new(),
            output_options: OutputOptions::default(),
            ps_options: PsOptions::default(),
            amplitude_dispersion_files: Vec::new(),
            amplitude_mean_files: Vec::new(),
            layover_shadow_mask_files: Vec::new(),
            phase_linking: PhaseLinkingOptions::default(),
            interferogram_network: InterferogramNetwork::default(),
            unwrap_options: UnwrapOptions::default(),
            timeseries_options: TimeseriesOptions::default(),
            correction_options: CorrectionOptions::default(),
            mask_file: None,
            work_directory: PathBuf::from("."),
            worker_settings: WorkerSettings::default(),
            log_file: None,
        }
    }
}

impl DisplacementWorkflow {
    /// Parse a workflow config from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(CoreError::from)
    }

    /// Serialize this workflow config to a YAML string.
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(CoreError::from)
    }

    /// Reject modeled dolphin options whose non-default behavior is not implemented.
    ///
    /// Parsing and serialization intentionally remain permissive so a dolphin YAML
    /// can round-trip. Workflow entry points call this guard before source or raster
    /// I/O so compatibility-only values cannot be mistaken for operational settings.
    ///
    /// # Errors
    /// Returns [`CoreError::InvalidConfig`] naming the first unsupported full YAML
    /// path, invalid backend choice, or config combination.
    #[allow(clippy::too_many_lines)]
    pub fn validate_supported_options(&self) -> Result<()> {
        let defaults = Self::default();
        if !matches!(
            self.unwrap_options.unwrap_method,
            UnwrapMethod::Native | UnwrapMethod::Snaphu | UnwrapMethod::Tophu
        ) {
            return Err(CoreError::InvalidConfig(format!(
                "unwrap_options.unwrap_method {:?} is modeled for dolphin YAML compatibility but dolphinRust supports only native, snaphu, and tophu",
                self.unwrap_options.unwrap_method
            )));
        }
        match self.unwrap_options.unwrap_method {
            UnwrapMethod::Native => {
                ensure_compatibility_default(
                    "unwrap_options.snaphu_options.init_method",
                    &self.unwrap_options.snaphu_options.init_method,
                    &defaults.unwrap_options.snaphu_options.init_method,
                )?;
                ensure_supported_choice(
                    "unwrap_options.snaphu_options.cost",
                    &self.unwrap_options.snaphu_options.cost,
                    &["smooth", "defo"],
                )?;
            }
            UnwrapMethod::Snaphu => {
                ensure_supported_choice(
                    "unwrap_options.snaphu_options.init_method",
                    &self.unwrap_options.snaphu_options.init_method,
                    &["mcf", "mst"],
                )?;
                ensure_supported_choice(
                    "unwrap_options.snaphu_options.cost",
                    &self.unwrap_options.snaphu_options.cost,
                    &["smooth", "defo"],
                )?;
            }
            UnwrapMethod::Tophu => {
                ensure_supported_choice(
                    "unwrap_options.tophu_options.init_method",
                    &self.unwrap_options.tophu_options.init_method,
                    &["mcf", "mst"],
                )?;
                ensure_supported_choice(
                    "unwrap_options.tophu_options.cost",
                    &self.unwrap_options.tophu_options.cost,
                    &["smooth", "defo"],
                )?;
            }
            UnwrapMethod::Icu
            | UnwrapMethod::Phass
            | UnwrapMethod::Spurt
            | UnwrapMethod::Whirlwind => unreachable!("unsupported methods returned above"),
        }
        if self.output_options.epsg.is_some() && self.output_options.bounds.is_none() {
            return Err(CoreError::InvalidConfig(
                "output_options.epsg is supported only as a source-CRS consistency check when output_options.bounds is set; unbounded runs require sourced CSLC georeferencing".into(),
            ));
        }
        if self.output_options.strides.y == 0 || self.output_options.strides.x == 0 {
            return Err(CoreError::InvalidConfig(
                "output_options.strides.y and output_options.strides.x must both be positive"
                    .into(),
            ));
        }
        if self.timeseries_options.write_posterior_uncertainty
            && self.timeseries_options.method != TimeseriesMethod::L2
        {
            return Err(CoreError::InvalidConfig(
                "timeseries_options.write_posterior_uncertainty requires timeseries_options.method: l2".into(),
            ));
        }
        require_compatibility_default!(self, defaults, output_options.add_overviews);
        require_compatibility_default!(self, defaults, output_options.overview_levels);
        require_compatibility_default!(self, defaults, ps_options.amp_dispersion_threshold);
        require_compatibility_default!(self, defaults, amplitude_dispersion_files);
        require_compatibility_default!(self, defaults, amplitude_mean_files);
        require_compatibility_default!(self, defaults, phase_linking.mask_input_ps);
        require_compatibility_default!(self, defaults, phase_linking.baseline_lag);
        require_compatibility_default!(self, defaults, unwrap_options.run_unwrap);
        require_compatibility_default!(self, defaults, unwrap_options.run_goldstein);
        require_compatibility_default!(self, defaults, unwrap_options.run_interpolation);
        require_compatibility_default!(self, defaults, unwrap_options.preprocess_options.alpha);
        require_compatibility_default!(
            self,
            defaults,
            unwrap_options.preprocess_options.max_radius
        );
        require_compatibility_default!(
            self,
            defaults,
            unwrap_options
                .preprocess_options
                .interpolation_cor_threshold
        );
        require_compatibility_default!(
            self,
            defaults,
            unwrap_options
                .preprocess_options
                .interpolation_similarity_threshold
        );
        require_compatibility_default!(
            self,
            defaults,
            unwrap_options
                .preprocess_options
                .zero_correlation_where_interpolating
        );
        require_compatibility_default!(
            self,
            defaults,
            unwrap_options.snaphu_options.single_tile_reoptimize
        );
        require_compatibility_default!(self, defaults, timeseries_options.run_inversion);
        require_compatibility_default!(self, defaults, timeseries_options.run_velocity);
        require_compatibility_default!(self, defaults, timeseries_options.apply_mask_to_timeseries);
        require_compatibility_default!(self, defaults, timeseries_options.block_shape);
        require_compatibility_default!(self, defaults, timeseries_options.num_parallel_blocks);
        require_compatibility_default!(self, defaults, worker_settings.gpu_enabled);
        require_compatibility_default!(self, defaults, worker_settings.threads_per_worker);
        require_compatibility_default!(self, defaults, worker_settings.n_parallel_bursts);
        require_compatibility_default!(self, defaults, log_file);
        Ok(())
    }
}
