//! Workflow configuration tree, mirroring dolphin's pydantic
//! `DisplacementWorkflow`.
//!
//! Field names and defaults match dolphin so an existing dolphin displacement
//! YAML deserializes unchanged. Unknown fields are ignored (not denied), so the
//! deeply-nested unwrap solver options dolphin emits (tophu/spurt/whirlwind —
//! documented Phase 9 gaps) pass through harmlessly without being modeled here.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::types::{HalfWindow, Strides};

/// SHP-selection statistical test. dolphin `ShpMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShpMethod {
    #[default]
    Glrt,
    Ks,
    Rect,
}

/// Compressed-SLC carry-forward plan. dolphin `CompressedSlcPlan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompressedSlcPlan {
    #[default]
    AlwaysFirst,
    FirstPerMinistack,
    LastPerMinistack,
}

/// Phase-unwrapping backend. dolphin `UnwrapMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UnwrapMethod {
    #[default]
    Snaphu,
    Icu,
    Phass,
    Spurt,
    Whirlwind,
}

/// Timeseries inversion norm. dolphin `TimeseriesOptions.method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimeseriesMethod {
    #[default]
    L1,
    L2,
}

/// Persistent-scatterer selection. dolphin `PsOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PsOptions {
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
    pub ministack_size: usize,
    pub max_num_compressed: usize,
    pub output_reference_idx: Option<usize>,
    pub half_window: HalfWindow,
    pub use_evd: bool,
    pub beta: f64,
    pub zero_correlation_threshold: f64,
    pub shp_method: ShpMethod,
    pub shp_alpha: f64,
    pub mask_input_ps: bool,
    pub baseline_lag: Option<i64>,
    pub compressed_slc_plan: CompressedSlcPlan,
    pub write_crlb: bool,
    pub write_closure_phase: bool,
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
        }
    }
}

/// Interferogram-network construction. dolphin `InterferogramNetwork`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InterferogramNetwork {
    pub reference_idx: Option<usize>,
    pub max_bandwidth: Option<usize>,
    pub max_temporal_baseline: Option<f64>,
    pub indexes: Option<Vec<(usize, usize)>>,
}

/// Timeseries inversion + velocity. dolphin `TimeseriesOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TimeseriesOptions {
    pub run_inversion: bool,
    pub method: TimeseriesMethod,
    pub reference_point: Option<(usize, usize)>,
    pub run_velocity: bool,
    pub apply_mask_to_timeseries: bool,
    pub correlation_threshold: f64,
    pub block_shape: (usize, usize),
    pub num_parallel_blocks: usize,
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
        }
    }
}

/// SNAPHU subprocess options. dolphin `SnaphuOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SnaphuOptions {
    pub ntiles: (usize, usize),
    pub tile_overlap: (usize, usize),
    pub n_parallel_tiles: usize,
    pub init_method: String,
    pub cost: String,
    pub single_tile_reoptimize: bool,
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
        }
    }
}

/// Pre-unwrap filtering/interpolation. dolphin `PreprocessOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreprocessOptions {
    pub alpha: f64,
    pub max_radius: usize,
    pub interpolation_cor_threshold: f64,
    pub interpolation_similarity_threshold: f64,
    pub zero_correlation_where_interpolating: bool,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            max_radius: 51,
            interpolation_cor_threshold: 0.25,
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
    pub run_unwrap: bool,
    pub run_goldstein: bool,
    pub run_interpolation: bool,
    pub unwrap_method: UnwrapMethod,
    pub n_parallel_jobs: i64,
    pub zero_where_masked: bool,
    pub preprocess_options: PreprocessOptions,
    pub snaphu_options: SnaphuOptions,
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
        }
    }
}

/// Input granule discovery. dolphin `InputOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InputOptions {
    pub subdataset: Option<String>,
    pub cslc_date_fmt: String,
    pub wavelength: Option<f64>,
}

impl Default for InputOptions {
    fn default() -> Self {
        Self {
            subdataset: None,
            cslc_date_fmt: "%Y%m%d".into(),
            wavelength: None,
        }
    }
}

/// Output grid + raster options. dolphin `OutputOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputOptions {
    pub strides: Strides,
    pub epsg: Option<u32>,
    pub bounds: Option<(f64, f64, f64, f64)>,
    pub bounds_epsg: Option<u32>,
    pub add_overviews: bool,
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

/// Parallelism / worker settings. dolphin `WorkerSettings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerSettings {
    pub gpu_enabled: bool,
    pub threads_per_worker: usize,
    pub n_parallel_bursts: usize,
    pub block_shape: (usize, usize),
}

impl Default for WorkerSettings {
    fn default() -> Self {
        Self {
            gpu_enabled: false,
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
    pub input_options: InputOptions,
    pub cslc_file_list: Vec<PathBuf>,
    pub output_options: OutputOptions,
    pub ps_options: PsOptions,
    pub amplitude_dispersion_files: Vec<PathBuf>,
    pub amplitude_mean_files: Vec<PathBuf>,
    pub layover_shadow_mask_files: Vec<PathBuf>,
    pub phase_linking: PhaseLinkingOptions,
    pub interferogram_network: InterferogramNetwork,
    pub unwrap_options: UnwrapOptions,
    pub timeseries_options: TimeseriesOptions,
    pub mask_file: Option<PathBuf>,
    pub work_directory: PathBuf,
    pub worker_settings: WorkerSettings,
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
}
