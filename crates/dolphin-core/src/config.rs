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
    /// SNAPHU statistical-cost network-flow unwrapper.
    #[default]
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
    /// Amplitude dispersion threshold to consider a pixel a PS.
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
    /// Set PS-labeled pixels to NaN during phase linking to avoid summing their phase.
    pub mask_input_ps: bool,
    /// StBAS lag: include only the nearest-N interferograms for phase linking.
    pub baseline_lag: Option<i64>,
    /// Plan for which date each ministack's compressed SLC references.
    pub compressed_slc_plan: CompressedSlcPlan,
    /// Write the Cramer-Rao lower bound raster.
    pub write_crlb: bool,
    /// Write the closure-phase raster.
    pub write_closure_phase: bool,
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
    /// Run the inversion step after unwrapping (if more than a single-reference network).
    pub run_inversion: bool,
    /// Norm to use during timeseries inversion.
    pub method: TimeseriesMethod,
    /// Reference point (row, col); auto-selected if not provided.
    pub reference_point: Option<(usize, usize)>,
    /// Run velocity estimation from the phase time series.
    pub run_velocity: bool,
    /// Apply the mask to the output timeseries rasters.
    pub apply_mask_to_timeseries: bool,
    /// Pixels with correlation below this value are masked out.
    pub correlation_threshold: f64,
    /// Size (rows, columns) of data blocks to load at a time.
    pub block_shape: (usize, usize),
    /// Number of parallel blocks to process at once.
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
    /// After multi-tile unwrapping, re-optimize the phase using a single tile.
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
    /// Adaptive-phase (Goldstein) filter exponent parameter.
    pub alpha: f64,
    /// Maximum radius (in pixels) to find scatterers during interpolation.
    pub max_radius: usize,
    /// Correlation threshold below which pixels are interpolated.
    pub interpolation_cor_threshold: f64,
    /// Similarity threshold below which pixels are interpolated.
    pub interpolation_similarity_threshold: f64,
    /// Zero out correlation at pixels that were interpolated.
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
    /// Run the unwrapping step after wrapped-phase estimation.
    pub run_unwrap: bool,
    /// Run Goldstein filtering on the wrapped interferogram.
    pub run_goldstein: bool,
    /// Run interpolation on the wrapped interferogram.
    pub run_interpolation: bool,
    /// Phase-unwrapping backend to dispatch to.
    pub unwrap_method: UnwrapMethod,
    /// Number of interferograms to unwrap in parallel.
    pub n_parallel_jobs: i64,
    /// Set wrapped phase/correlation to 0 where the mask is 0 before unwrapping.
    pub zero_where_masked: bool,
    /// Goldstein-filter / interpolation preprocessing options.
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
    /// Line-of-sight geometry files for the correction computations. dolphin name
    /// (carried for YAML round-trip; the delay projection uses
    /// `incidence_angle_deg` when no geometry is resolved).
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
        }
    }
}

impl CorrectionOptions {
    /// Whether any correction is enabled (any correction file supplied).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        !self.ionosphere_files.is_empty() || !self.troposphere_files.is_empty()
    }
}

/// Output grid + raster options. dolphin `OutputOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputOptions {
    /// (x, y) strides (decimation factor) to apply while processing input.
    pub strides: Strides,
    /// EPSG code of the output grid.
    pub epsg: Option<u32>,
    /// Area of interest as [left, bottom, right, top] coordinates.
    pub bounds: Option<(f64, f64, f64, f64)>,
    /// EPSG code for the `bounds` coordinates.
    pub bounds_epsg: Option<u32>,
    /// Add overviews to the output GeoTIFFs.
    pub add_overviews: bool,
    /// Overview levels to create (if `add_overviews`).
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
    /// Use the GPU for processing (if available). dolphin parity; superseded by
    /// `compute_backend` (kept so existing dolphin YAML deserializes unchanged).
    pub gpu_enabled: bool,
    /// Compute backend selection for phase linking (`auto` / `cpu` / `gpu`).
    pub compute_backend: ComputeBackend,
    /// Number of threads to use per worker.
    pub threads_per_worker: usize,
    /// Number of spatial bursts to run in parallel for wrapped-phase estimation.
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
    /// PS pixel-selection options.
    pub ps_options: PsOptions,
    /// Existing amplitude-dispersion files (1 per SLC region) for PS update.
    pub amplitude_dispersion_files: Vec<PathBuf>,
    /// Existing amplitude-mean files (1 per SLC region) for PS update.
    pub amplitude_mean_files: Vec<PathBuf>,
    /// Layover/shadow binary masks (0 = layover/shadow, 1 = good pixel).
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
    /// Path to the output log file (in addition to stderr).
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
}
