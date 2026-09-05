//! End-to-end displacement pipeline (port of `workflows/displacement.py`).
//!
//! Order: group inputs by burst → per-burst sequential phase linking → stitch
//! bursts onto the frame grid → ifg network → SNAPHU unwrap → SBAS inversion →
//! velocity → write COGs. Single-burst stacks take the stitch identity path.
//! Synchronous; the host app bridges to its runtime.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use dolphin_core::config::{DisplacementWorkflow, InputType, TimeseriesMethod, UnwrapMethod};
use dolphin_core::{BlockIndices, Cf32, Cf64};
use dolphin_io::{
    read_aligned_raster_window, read_cslc_shape, read_cslc_window, read_geotransform,
    read_nisar_geotransform, read_nisar_window, write_raster, write_raster_with_metadata, GeoInfo,
};
use dolphin_phaselink::{
    all_non_finite_acquisition_indices, correct_phase_bias, estimate_bias_velocity, ComputeEngine,
};
use dolphin_timeseries::{
    build_network, estimate_velocity, estimate_velocity_with_model,
    estimate_velocity_with_uncertainty, estimate_velocity_with_uncertainty_neff,
    get_incidence_matrix, invert_stack, invert_stack_l1, invert_stack_with_uncertainty,
    loop_closure_qc, mask_failed_loops, network_triplets, reference_to_point,
    select_reference_point, L1Config, LoopClosureQc, NetworkConfig, VelocityModel,
    DEFAULT_CLOSURE_TOLERANCE_CYCLES,
};
use dolphin_unwrap::native::NativeConfig;
use dolphin_unwrap::{CostMode, InitMethod, TophuConfig, UnwrapConfig};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis};

use crate::burst::{burst_offset, frame_grid, workflow_groups, BurstGeo, FrameGrid};
use crate::corrections::{apply_corrections, CorrectionLayers};
use crate::crop::{plan_bounds, BoundedPlan, BurstWindow};
use crate::dates::{acquisition_days, parse_date};
use crate::provenance::{
    BurstCoverageProvenance, GeometryProvenance, InputCoverageProvenance,
    INPUT_COVERAGE_POLICY_VERSION,
};
use crate::sequential::{
    run_sequential, run_sequential_resumable, update_sequential, SequentialConfig,
    SequentialOutput, SequentialState,
};
use crate::tiling::{plan_tiles, TilePlan};
use crate::unwrap_backend::{
    NativeUnwrapBackend, SnaphuBackend, TophuBackend, UnwrapBackend, UnwrapNetworkOutput,
};
use dolphin_corrections::LosGeometry;

/// Sentinel-1 C-band radar wavelength (m); used to express velocity in mm/yr
/// when the config carries no explicit `input_options.wavelength`.
const SENTINEL1_WAVELENGTH_M: f64 = 0.055_465_76;
const MIN_SEAM_SUPPORT: usize = 4;
const MIN_SEAM_COHERENCE: f64 = 0.5;

/// Typed failure from multi-burst phase-offset reconciliation.
#[derive(Debug, thiserror::Error)]
pub enum StitchError {
    /// A burst overlap exists geometrically but does not contain enough stable,
    /// coherent, finite samples to estimate a phase offset for one acquisition.
    #[error("burst {burst_index} acquisition {acquisition_index} has only {support} stable overlap samples; at least {required} are required")]
    InsufficientOffsetSupport {
        /// Zero-based burst index in stitch order.
        burst_index: usize,
        /// Zero-based acquisition index.
        acquisition_index: usize,
        /// Valid overlap sample count.
        support: usize,
        /// Required sample count.
        required: usize,
    },
}

/// Displacement pipeline outputs (in-memory mirror of the written rasters).
pub struct DisplacementOutput {
    /// Per-date cumulative displacement, `(n_dates-1, rows, cols)`, referenced
    /// to acquisition 0. Units are meters when `input_options.wavelength` is set,
    /// otherwise radians of wrapped LOS phase.
    pub displacement: Array3<f64>,
    /// Linear velocity per pixel in raster units/year (m/yr with wavelength,
    /// else rad/yr), `(rows, cols)`.
    pub velocity: Array2<f64>,
    /// Linear velocity per pixel in **mm/yr** (the rate eo stores for risk
    /// scoring). Derived from the LOS phase rate via `-λ/4π`, using the config
    /// wavelength or the Sentinel-1 default, `(rows, cols)`.
    pub velocity_mm_yr: Array2<f64>,
    /// One-sigma linear-rate uncertainty in the same units/year as `velocity`.
    pub velocity_sigma: Option<Array2<f64>>,
    /// L2 posterior displacement variance in the same units squared as `displacement`.
    pub displacement_variance: Option<Array3<f64>>,
    /// SBAS network-inversion misclosure RMS (residual of `A*phi = dphi` in the
    /// same units as `displacement`) — how well the interferogram network
    /// closed. `Some` only for `write_posterior_uncertainty` L2 runs. Distinct
    /// from `timeseries_residual_rms` (issue #40): a network can close perfectly
    /// while displacement still fits the temporal model badly, and vice versa.
    pub network_misclosure_rms: Option<Array2<f64>>,
    /// Temporal motion-model fit residual RMS in the same units as
    /// `displacement`: the per-pixel scatter of displacement around the fitted
    /// rate (+ seasonal/step terms, when configured). `None` only on the
    /// unweighted-linear fast path (no `use_coherence_weights` /
    /// `write_velocity_uncertainty` / time-function model configured).
    pub timeseries_residual_rms: Option<Array2<f64>>,
    /// Interferogram date-index pairs corresponding to unwrap output bands.
    pub interferogram_pairs: Vec<(usize, usize)>,
    /// Per-interferogram connected-component labels.
    pub unwrap_connected_components: Array3<u32>,
    /// Temporal coherence per pixel in `[0, 1]`, stitched across ministacks by
    /// NaN-aware mean (dolphin's `temporal_coherence_average` = `numpy.nanmean`);
    /// a phase-quality mask, `(rows, cols)`.
    pub temporal_coherence: Array2<f64>,
    /// Mean coherence-matrix magnitude across real acquisitions, distinct from
    /// estimator-fit temporal coherence. `None` unless `calc_average_coh` is on.
    pub phase_linking_coherence: Option<Array2<f64>>,
    /// Pixels with complete temporal input support after burst mosaicking and trim.
    pub validity_mask: Array2<bool>,
    /// Per-date CRLB phase-estimate σ (radians), `(n_dates, rows, cols)`, band 0 =
    /// reference (σ=0); a singular-Γ pixel is `NaN`. The physical uncertainty that
    /// feeds GroundPulse's `confidence_score`. `None` when `phase_linking.write_crlb`
    /// is off. Present by default (dolphin defaults `write_crlb = true`).
    pub crlb_sigma: Option<Array3<f64>>,
    /// Per-triplet nearest-neighbour closure phase (radians), band-major; the
    /// non-closure diagnostic. `None` unless `phase_linking.write_closure_phase`
    /// is on (dolphin defaults it off).
    pub closure_phase: Option<Array3<f64>>,
    /// Acquisition dates as decimal days from acquisition 0, length `n_dates`.
    pub acquisition_days: Vec<f64>,
    /// EPSG code of the output grid (`None` if neither the CSLC metadata nor the
    /// config supplied one).
    pub epsg: Option<u32>,
    /// GDAL affine geotransform `[origin_x, dx, 0, origin_y, 0, dy]` shared by all
    /// output rasters (read from the CSLC grid, else an identity placeholder).
    pub geotransform: [f64; 6],
    /// Spatial reference pixel `(row, col)` the series is referenced to: the
    /// configured `timeseries_options.reference_point`, else the auto-selected
    /// center-of-mass point, or `None` if no coherent pixel was found.
    pub reference_point: Option<(usize, usize)>,
    /// Per-date ionospheric range delay (meters), `(n_dates, rows, cols)`, that was
    /// subtracted from the series. `None` unless `correction_options.ionosphere_files`
    /// were supplied. The dominant L-band atmospheric term (`1/f²`-scaled).
    pub ionosphere_delay: Option<Array3<f64>>,
    /// Per-date tropospheric range delay (meters), `(n_dates, rows, cols)`, that was
    /// subtracted from the series. `None` unless `correction_options.troposphere_files`
    /// were supplied.
    pub troposphere_delay: Option<Array3<f64>>,
    /// Per-date solid-earth-tide equivalent range delay (meters), `(n_dates, rows,
    /// cols)`, that was subtracted from the series. `None` unless
    /// `correction_options.solid_earth_tide` was set. Not a propagation delay —
    /// real lunisolar ground motion, expressed as the range change it causes.
    pub solid_earth_tide_delay: Option<Array3<f64>>,
    /// Per-pixel LOS unit-vector geometry (east/north/up) on the output grid. `None`
    /// unless `correction_options.geometry_files` (CSLC-S1-STATIC) were supplied. The
    /// front door for the GPS ground-truth harness's ENU→LOS projection.
    pub los_geometry: Option<LosGeometry>,
    /// Geometry provenance for asc/desc decomposition gating (dolphinRust #1 /
    /// eo #120), mirrored on disk as `geometry_provenance.json`. Always present;
    /// unsourceable fields are explicitly absent inside it, never defaulted.
    pub geometry_provenance: GeometryProvenance,
}

/// Current and high-water resident memory from Linux procfs, in KiB. Zeros mean
/// the platform does not expose procfs; diagnostics remain portable and safe.
fn memory_kib() -> (u64, u64) {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .map_or((0, 0), |status| parse_memory_kib(&status))
}

fn parse_memory_kib(status: &str) -> (u64, u64) {
    let value = |key: &str| {
        status.lines().find_map(|line| {
            line.strip_prefix(key)?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
        })
    };
    (value("VmRSS:").unwrap_or(0), value("VmHWM:").unwrap_or(0))
}

/// Run `f`, emitting start/completion wall-clock and RSS breadcrumbs under
/// `stage` at INFO so native termination can be assigned to a stage.
fn timed<T>(stage: &str, f: impl FnOnce() -> T) -> T {
    let (rss_kib, peak_rss_kib) = memory_kib();
    tracing::info!(stage, event = "start", rss_kib, peak_rss_kib, "stage start");
    let t0 = Instant::now();
    let out = f();
    let (rss_kib, peak_rss_kib) = memory_kib();
    tracing::info!(
        stage,
        event = "complete",
        elapsed_s = t0.elapsed().as_secs_f64(),
        rss_kib,
        peak_rss_kib,
        "stage complete"
    );
    out
}

/// Run the displacement workflow from a parsed config.
///
/// # Errors
/// Returns `Err` on I/O, phase-linking, unwrapping, date-parsing, or config problems.
pub fn run_displacement(cfg: &DisplacementWorkflow) -> Result<DisplacementOutput> {
    run_displacement_with_output_policy(cfg, DisplacementOutputPolicy::Full)
}

/// Runtime-only serialization policy for a displacement run.
///
/// This is deliberately separate from scientific configuration: it changes only
/// which already-computed arrays are serialized into the work directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplacementOutputPolicy {
    /// Emit every compatibility, diagnostic, correction, and provenance file.
    Full,
    /// Emit only the phase-linking-coherence raster consumed by GroundPulse.
    GroundPulse,
}

/// Run the displacement workflow with an explicit runtime-only serialization policy.
///
/// # Errors
/// Returns `Err` on I/O, phase-linking, unwrapping, date-parsing, or config problems.
pub fn run_displacement_with_output_policy(
    cfg: &DisplacementWorkflow,
    output_policy: DisplacementOutputPolicy,
) -> Result<DisplacementOutput> {
    validate_uncertainty_options(cfg)?;
    let groups = workflow_groups(cfg)?;
    let layouts = source_layouts(cfg, &groups)?;
    let acquisitions = groups.values().map(Vec::len).max().unwrap_or(0);
    let crop = plan_bounds(cfg, &layouts, acquisitions)?;
    // One compute engine for the whole run: it acquires a single GPU context (if
    // selected and available) and is reused across every burst + ministack.
    let engine = ComputeEngine::new(cfg.worker_settings.compute_backend);
    let bursts = timed("phase_linking", || {
        groups
            .values()
            .enumerate()
            .filter_map(|(index, idxs)| {
                let window = crop
                    .as_ref()
                    .map_or(Some(None), |plan| plan.windows[index].map(Some))?;
                Some(link_one_burst(cfg, idxs, index, &engine, window))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    finish_displacement(cfg, bursts, crop.as_ref(), output_policy)
}

/// Shared downstream tail: stitch bursts → ifg network → SNAPHU unwrap → SBAS
/// inversion → atmospheric corrections → reference → velocity → write COGs.
/// Identical for a full run and an incremental update — both feed it the same
/// per-burst phase-linking products, so both produce the same output.
fn finish_displacement(
    cfg: &DisplacementWorkflow,
    bursts: Vec<BurstLink>,
    crop: Option<&BoundedPlan>,
    output_policy: DisplacementOutputPolicy,
) -> Result<DisplacementOutput> {
    let groups = workflow_groups(cfg)?;
    let days = bursts
        .first()
        .map(|b| b.days.clone())
        .context("cslc_file_list is empty")?;
    let stitched = timed("stitch", || stitch_bursts(bursts))?;
    let validity_mask = stitched.validity_mask;
    let burst_coverage = stitched.coverage;
    let mut pl = stitched.pl;
    if cfg.phase_linking.correct_phase_bias {
        apply_phase_bias(&mut pl, stitched.closure_phase.as_ref())?;
    }
    let temporal_coherence = stitched.temp_coh;
    let geo = stitched.geo;
    let epsg = (geo.epsg != 0).then_some(geo.epsg);
    let geotransform = geo.geotransform;
    anyhow::ensure!(
        days.len() == pl.dim().0,
        "parsed {} dates but phase-linking produced {} acquisitions",
        days.len(),
        pl.dim().0
    );
    // Cheap precondition, checked before the expensive stages: LOS geometry is
    // otherwise resolved during corrections, which runs after unwrapping and
    // inversion, so an uncovered frame surfaced ~90 minutes into a real 52-date
    // run having already paid for all of it.
    timed("geometry_precheck", || {
        crate::corrections::verify_geometry_coverage(
            &cfg.correction_options,
            epsg.unwrap_or(0),
            geotransform,
            temporal_coherence.dim(),
        )
    })?;
    let pairs = timed("network", || network(cfg, &days));
    anyhow::ensure!(!pairs.is_empty(), "interferogram_network produced no pairs");

    let unwrap = timed("unwrap", || {
        unwrap_network(
            cfg,
            pl.view(),
            &pairs,
            temporal_coherence.view(),
            geotransform,
            epsg,
        )
    })?;
    let (mut inversion, loop_closure) =
        solve_time_series(cfg, unwrap.unwrapped, &pairs, stitched.crlb_sigma.as_ref())?;
    // Spatially reference the series to a stable pixel (dolphin parity): the
    // configured point, else the center-of-mass of the high-coherence region.
    let configured_reference = configured_analysis_reference(
        cfg.timeseries_options.reference_point,
        crop,
        temporal_coherence.dim(),
    )?;
    let analysis_reference_point = configured_reference.or_else(|| {
        select_reference_point(
            temporal_coherence.view(),
            cfg.timeseries_options.correlation_threshold,
        )
    });
    let date_files = first_burst_files(cfg, &groups);
    let corrections = timed("corrections", || {
        correct_and_reference(
            cfg,
            &mut inversion.displacement,
            &date_files,
            GeoInfo {
                epsg: epsg.unwrap_or(0),
                geotransform,
            },
            analysis_reference_point,
        )
    })?;
    let (velocity_model, fit) = frame_velocity(
        cfg,
        inversion.displacement.view(),
        &days,
        stitched.crlb_sigma.as_ref(),
        &date_files,
    )?;
    let spatial = SpatialProducts {
        disp_rad: inversion.displacement,
        vel_rad: fit.velocity,
        velocity_model,
        velocity_terms: fit.terms,
        loop_closure,
        temporal_coherence,
        validity_mask,
        burst_coverage,
        phase_linking_coherence: stitched.phase_linking_coherence,
        crlb_sigma: stitched.crlb_sigma,
        closure_phase: stitched.closure_phase,
        corrections,
        posterior_variance_rad: inversion.posterior_variance,
        network_misclosure_rad: inversion.network_misclosure_rms,
        timeseries_residual_rad: fit.residual_rms,
        velocity_sigma_rad: fit.sigma,
        interferogram_pairs: pairs,
        unwrap_connected_components: unwrap.connected_components,
        geotransform,
        reference_point: analysis_reference_point,
    };
    emit_displacement(cfg, days, epsg, crop, spatial, output_policy)
}

struct InversionProducts {
    displacement: Array3<f64>,
    posterior_variance: Option<Array3<f64>>,
    /// L2 SBAS network-inversion misclosure RMS (residual of `A*phi = dphi`) —
    /// how well the interferogram network closed, not how well displacement
    /// fits a temporal motion model (issue #40; that quantity is
    /// [`VelocityFit::residual_rms`]). `Some` only for
    /// `write_posterior_uncertainty` L2 runs.
    network_misclosure_rms: Option<Array2<f64>>,
}

fn invert_time_series(
    cfg: &DisplacementWorkflow,
    incidence: ArrayView2<f64>,
    dphi: ArrayView3<f64>,
    crlb_sigma: Option<&Array3<f64>>,
    pairs: &[(usize, usize)],
) -> Result<InversionProducts> {
    anyhow::ensure!(
        !(cfg.timeseries_options.write_posterior_uncertainty
            && cfg.timeseries_options.method == TimeseriesMethod::L1),
        "posterior uncertainty is available only for L2 timeseries inversion"
    );
    let precision = if cfg.timeseries_options.method == TimeseriesMethod::L2
        && (cfg.timeseries_options.use_coherence_weights
            || cfg.timeseries_options.write_posterior_uncertainty)
    {
        Some(if cfg.timeseries_options.use_coherence_weights {
            let sigma = crlb_sigma
                .context("coherence weighting requires internally computed CRLB")?
                .view();
            interferogram_precisions(sigma, pairs, uncertainty_valid(sigma).view())
        } else {
            Array3::ones(dphi.dim())
        })
    } else {
        None
    };
    match cfg.timeseries_options.method {
        TimeseriesMethod::L1 => Ok(InversionProducts {
            displacement: invert_stack_l1(incidence, dphi, L1Config::default()),
            posterior_variance: None,
            network_misclosure_rms: None,
        }),
        TimeseriesMethod::L2 if cfg.timeseries_options.write_posterior_uncertainty => {
            let output = invert_stack_with_uncertainty(
                incidence,
                dphi,
                precision
                    .as_ref()
                    .context("posterior uncertainty requires L2 observation precision")?
                    .view(),
            );
            // Uniform weights recover the displacement at an unbounded pixel but
            // do not yield a posterior it can stand behind, so blank it there.
            let mut posterior_variance = output.posterior_variance;
            if let Some(sigma) = crlb_sigma {
                let valid = uncertainty_valid(sigma.view());
                for mut band in posterior_variance.outer_iter_mut() {
                    clear_unbounded_uncertainty_2d(&mut band, valid.view());
                }
            }
            Ok(InversionProducts {
                displacement: output.phase,
                posterior_variance: Some(posterior_variance),
                network_misclosure_rms: Some(output.residual_rms),
            })
        }
        TimeseriesMethod::L2 => Ok(InversionProducts {
            displacement: invert_stack(incidence, dphi, precision.as_ref().map(Array3::view)),
            posterior_variance: None,
            network_misclosure_rms: None,
        }),
    }
}

fn correct_and_reference(
    cfg: &DisplacementWorkflow,
    displacement: &mut Array3<f64>,
    date_files: &[PathBuf],
    geo: GeoInfo,
    reference: Option<(usize, usize)>,
) -> Result<CorrectionLayers> {
    let mut options = cfg.correction_options.clone();
    if !cfg.input_options.acquisition_metadata.is_empty() {
        options.acquisition_utc = date_files
            .iter()
            .map(|path| {
                cfg.input_options
                    .acquisition_metadata
                    .iter()
                    .find(|m| m.path == *path)
                    .map(|m| m.acquisition_utc)
                    .context("missing verified acquisition UTC")
            })
            .collect::<Result<Vec<_>>>()?;
    }
    let corrections = apply_corrections(
        &options,
        cfg.input_options.wavelength,
        displacement,
        date_files,
        geo.epsg,
        geo.geotransform,
    )?;
    if let Some(point) = reference {
        reference_to_point(displacement, point);
    }
    Ok(corrections)
}

/// Weighted velocity and its one-sigma, applying the opt-in AR(1)
/// temporal-correlation inflation when configured. The residual RMS is
/// identical between the two branches (the correction rescales sigma, not the
/// fit), so it is returned once alongside whichever sigma was requested.
fn velocity_and_sigma(
    correct_temporal_correlation: bool,
    days: &[f64],
    series: ArrayView3<f64>,
    precision: ArrayView3<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    if !correct_temporal_correlation {
        let output = estimate_velocity_with_uncertainty(days, series, precision);
        return (output.velocity, output.sigma, output.residual_rms);
    }
    let output = estimate_velocity_with_uncertainty_neff(days, series, precision);
    (
        output.velocity,
        output.sigma_temporal_corrected,
        output.residual_rms,
    )
}

/// Optional velocity time-function terms, in the same phase units (rad) as the
/// velocity fit — except `seasonal_phase_days`, which is days. Empty unless
/// `timeseries_options.velocity_seasonal` / `velocity_step_dates` are configured.
#[derive(Debug, Clone, Default)]
struct VelocityTerms {
    seasonal_amplitude_rad: Option<Array2<f64>>,
    seasonal_phase_days: Option<Array2<f64>>,
    step_magnitude_rad: Vec<Array2<f64>>,
}

/// Rate, one-sigma, and optional time-function terms in one place, so the
/// whole-frame and bounded/tiled paths cannot drift apart.
struct VelocityFit {
    velocity: Array2<f64>,
    sigma: Option<Array2<f64>>,
    /// Temporal motion-model fit residual RMS: the per-pixel scatter of
    /// displacement around the fitted rate (+ seasonal/step terms), in the same
    /// units as `displacement` (issue #40). Distinct from the SBAS
    /// network-inversion misclosure (`InversionProducts::network_misclosure_rms`)
    /// — a network can close perfectly while still carrying a phase history the
    /// model fits badly, and vice versa. `None` on the unweighted-linear fast
    /// path, which computes no fit statistics at all (matching `sigma`'s rule
    /// there).
    residual_rms: Option<Array2<f64>>,
    terms: VelocityTerms,
}

/// The configured time-function model, with step dates resolved to decimal days
/// from acquisition 0 — the same origin [`acquisition_days`] gives the `days` the fit
/// runs against. `date_files` is the first burst's files in date order.
///
/// # Errors
/// Returns `Err` if a step date is not `YYYY-MM-DD` or the acquisition-0 date is
/// unparseable. A step the user asked for and did not get is a wrong answer, not
/// a degraded one, so this fails the run rather than dropping the term.
fn velocity_model(cfg: &DisplacementWorkflow, date_files: &[PathBuf]) -> Result<VelocityModel> {
    let options = &cfg.timeseries_options;
    let model = VelocityModel {
        seasonal: options.velocity_seasonal,
        step_days: Vec::new(),
    };
    if model.is_linear() && options.velocity_step_dates.is_empty() {
        return Ok(model);
    }
    let first = date_files
        .first()
        .context("velocity time-function model requires at least one acquisition")?;
    let anchor = if cfg.input_options.acquisition_metadata.is_empty() {
        parse_date(first, &cfg.input_options.cslc_date_fmt)?
    } else {
        cfg.input_options
            .acquisition_metadata
            .iter()
            .find(|m| m.path == *first)
            .context("missing acquisition UTC for velocity model")?
            .acquisition_utc
            .date_naive()
    };
    let step_days = options
        .velocity_step_dates
        .iter()
        .map(|raw| {
            let date =
                chrono::NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d").with_context(|| {
                    format!("timeseries_options.velocity_step_dates: {raw:?} is not YYYY-MM-DD")
                })?;
            Ok((date - anchor).num_days() as f64)
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok(VelocityModel { step_days, ..model })
}

/// Every emitted layer scaled from LOS phase (rad) to displacement units, kept
/// together so `emit_displacement` reads as a sequence of steps.
struct ScaledOutputs {
    displacement: Array3<f64>,
    velocity: Array2<f64>,
    velocity_mm_yr: Array2<f64>,
    displacement_variance: Option<Array3<f64>>,
    network_misclosure_rms: Option<Array2<f64>>,
    timeseries_residual_rms: Option<Array2<f64>>,
    velocity_sigma: Option<Array2<f64>>,
    seasonal_amplitude: Option<Array2<f64>>,
    step_magnitude: Vec<Array2<f64>>,
}

fn scale_outputs(cfg: &DisplacementWorkflow, spatial: &SpatialProducts) -> ScaledOutputs {
    let phase_to_disp = cfg
        .input_options
        .wavelength
        .map_or(1.0, |w| -w / (4.0 * std::f64::consts::PI));
    let (seasonal_amplitude, step_magnitude) =
        scale_velocity_terms(&spatial.velocity_terms, phase_to_disp);
    ScaledOutputs {
        displacement: spatial.disp_rad.mapv(|phase| phase * phase_to_disp),
        velocity: spatial.vel_rad.mapv(|rate| rate * phase_to_disp),
        velocity_mm_yr: spatial
            .vel_rad
            .mapv(|rate| rate * mm_per_rad(cfg.input_options.wavelength)),
        // Variance is squared displacement; a sigma and an RMS are magnitudes.
        displacement_variance: spatial
            .posterior_variance_rad
            .as_ref()
            .map(|v| v.mapv(|value| value * phase_to_disp * phase_to_disp)),
        network_misclosure_rms: spatial
            .network_misclosure_rad
            .as_ref()
            .map(|v| v.mapv(|value| value * phase_to_disp.abs())),
        timeseries_residual_rms: spatial
            .timeseries_residual_rad
            .as_ref()
            .map(|v| v.mapv(|value| value * phase_to_disp.abs())),
        velocity_sigma: spatial
            .velocity_sigma_rad
            .as_ref()
            .map(|v| v.mapv(|value| value * phase_to_disp.abs())),
        seasonal_amplitude,
        step_magnitude,
    }
}

/// Seasonal amplitude and step magnitudes from phase (rad) to displacement units.
/// Both are series quantities, so they take the same factor as displacement
/// itself; amplitude is an unsigned magnitude, a step is signed. The seasonal peak
/// day needs no scaling — it is already days.
fn scale_velocity_terms(
    terms: &VelocityTerms,
    phase_to_disp: f64,
) -> (Option<Array2<f64>>, Vec<Array2<f64>>) {
    let amplitude = terms
        .seasonal_amplitude_rad
        .as_ref()
        .map(|values| values.mapv(|value| value * phase_to_disp.abs()));
    let steps = terms
        .step_magnitude_rad
        .iter()
        .map(|values| values.mapv(|value| value * phase_to_disp))
        .collect();
    (amplitude, steps)
}

/// Loop-closure QC on the unwrapped network, then the SBAS solve. The QC runs
/// first because a 2π unwrap error is a confident wrong number in the solve, and
/// it is invisible to the wrapped closure-phase layer (#24).
fn solve_time_series(
    cfg: &DisplacementWorkflow,
    mut dphi_rad: Array3<f64>,
    pairs: &[(usize, usize)],
    crlb_sigma: Option<&Array3<f64>>,
) -> Result<(InversionProducts, Option<LoopClosureQc>)> {
    let loop_closure = timed("loop_closure", || {
        apply_loop_closure_qc(cfg, &mut dphi_rad, pairs)
    });
    let incidence = get_incidence_matrix(pairs);
    let inversion = timed("timeseries", || {
        invert_time_series(cfg, incidence.view(), dphi_rad.view(), crlb_sigma, pairs)
    })?;
    Ok((inversion, loop_closure))
}

/// Run the opt-in post-unwrap loop-closure QC and blank the failing pixels.
/// Returns the QC layers for output, or `None` when the gate is off or the
/// network has no loops to close.
fn apply_loop_closure_qc(
    cfg: &DisplacementWorkflow,
    dphi_rad: &mut Array3<f64>,
    pairs: &[(usize, usize)],
) -> Option<LoopClosureQc> {
    if !cfg.timeseries_options.mask_unwrap_loop_errors {
        return None;
    }
    if network_triplets(pairs).is_empty() {
        tracing::warn!(
            stage = "loop_closure",
            "mask_unwrap_loop_errors is set but the interferogram network has no closed \
             triangles (single-reference?) — nothing to check; set \
             interferogram_network.max_bandwidth or max_temporal_baseline"
        );
        return None;
    }
    let qc = loop_closure_qc(dphi_rad.view(), pairs, DEFAULT_CLOSURE_TOLERANCE_CYCLES);
    let masked = qc.bad_loop_count.iter().filter(|&&n| n > 0.0).count();
    tracing::info!(
        stage = "loop_closure",
        masked_pixels = masked,
        pixels = qc.bad_loop_count.len(),
        "masked pixels whose unwrapped loops did not close"
    );
    mask_failed_loops(dphi_rad, &qc);
    Some(qc)
}

/// Resolve the configured time-function model and fit the whole-frame velocity.
fn frame_velocity(
    cfg: &DisplacementWorkflow,
    displacement: ArrayView3<f64>,
    days: &[f64],
    crlb_sigma: Option<&Array3<f64>>,
    date_files: &[PathBuf],
) -> Result<(VelocityModel, VelocityFit)> {
    let model = velocity_model(cfg, date_files)?;
    let fit = timed("velocity", || {
        fit_velocity(cfg, displacement, days, crlb_sigma, &model)
    })?;
    Ok((model, fit))
}

fn fit_velocity(
    cfg: &DisplacementWorkflow,
    displacement: ArrayView3<f64>,
    days: &[f64],
    crlb_sigma: Option<&Array3<f64>>,
    model: &VelocityModel,
) -> Result<VelocityFit> {
    let weighted = cfg.timeseries_options.use_coherence_weights
        || cfg.timeseries_options.write_velocity_uncertainty;
    if !weighted && model.is_linear() {
        // The cheapest path: no precision, no fit statistics at all. Computing a
        // residual here would mean fitting a second, otherwise-unneeded model per
        // pixel just to report it, so this path stays a rate-only estimate,
        // matching `sigma`'s existing `None` rule.
        return Ok(VelocityFit {
            velocity: velocity_of(displacement, days),
            sigma: None,
            residual_rms: None,
            terms: VelocityTerms::default(),
        });
    }
    let series = series_with_reference(displacement);
    if !weighted {
        return Ok(fit_velocity_with_model(
            cfg,
            days,
            series.view(),
            None,
            model,
        ));
    }
    let sigma = crlb_sigma
        .context("velocity weighting requires internally computed CRLB")?
        .view();
    let valid = uncertainty_valid(sigma);
    let precision = date_precisions(sigma, valid.view());
    if !model.is_linear() {
        let mut fit =
            fit_velocity_with_model(cfg, days, series.view(), Some(precision.view()), model);
        if let Some(sigma) = fit.sigma.as_mut() {
            clear_unbounded_uncertainty(sigma, valid.view());
        }
        return Ok(fit);
    }
    if !cfg.timeseries_options.write_velocity_uncertainty {
        // Same underlying per-pixel fit as `estimate_velocity_with_precisions`
        // (velocity is bit-identical); the uncertainty variant is used instead so
        // the residual it already computes is not thrown away.
        let output = estimate_velocity_with_uncertainty(days, series.view(), precision.view());
        return Ok(VelocityFit {
            velocity: output.velocity,
            sigma: None,
            residual_rms: Some(output.residual_rms),
            terms: VelocityTerms::default(),
        });
    }
    let (velocity, mut sigma, residual_rms) = velocity_and_sigma(
        cfg.timeseries_options.correct_velocity_temporal_correlation,
        days,
        series.view(),
        precision.view(),
    );
    clear_unbounded_uncertainty(&mut sigma, valid.view());
    Ok(VelocityFit {
        velocity,
        sigma: Some(sigma),
        residual_rms: Some(residual_rms),
        terms: VelocityTerms::default(),
    })
}

/// The joint seasonal/step fit, sharing the linear path's uncertainty switches:
/// sigma is emitted only when `write_velocity_uncertainty`, and the AR(1)
/// inflation applies to this model's own residuals when configured.
fn fit_velocity_with_model(
    cfg: &DisplacementWorkflow,
    days: &[f64],
    series: ArrayView3<f64>,
    precision: Option<ArrayView3<f64>>,
    model: &VelocityModel,
) -> VelocityFit {
    let output = estimate_velocity_with_model(days, series, precision, model);
    let sigma = cfg.timeseries_options.write_velocity_uncertainty.then(|| {
        match cfg.timeseries_options.correct_velocity_temporal_correlation {
            true => &output.sigma * &output.inflation_factor,
            false => output.sigma.clone(),
        }
    });
    VelocityFit {
        velocity: output.velocity,
        sigma,
        // Unlike sigma, the residual is not gated by write_velocity_uncertainty:
        // estimate_velocity_with_model always computes it as part of the fit, so
        // reporting it here costs nothing extra.
        residual_rms: Some(output.residual_rms),
        terms: VelocityTerms {
            seasonal_amplitude_rad: output.seasonal_amplitude,
            seasonal_phase_days: output.seasonal_phase_days,
            step_magnitude_rad: output.step_magnitude,
        },
    }
}

struct SpatialProducts {
    disp_rad: Array3<f64>,
    vel_rad: Array2<f64>,
    velocity_model: VelocityModel,
    velocity_terms: VelocityTerms,
    loop_closure: Option<LoopClosureQc>,
    temporal_coherence: Array2<f64>,
    validity_mask: Array2<bool>,
    burst_coverage: Vec<BurstCoverageProvenance>,
    phase_linking_coherence: Option<Array2<f64>>,
    crlb_sigma: Option<Array3<f64>>,
    closure_phase: Option<Array3<f64>>,
    corrections: CorrectionLayers,
    geotransform: [f64; 6],
    reference_point: Option<(usize, usize)>,
    posterior_variance_rad: Option<Array3<f64>>,
    /// SBAS network-inversion misclosure RMS (see
    /// `InversionProducts::network_misclosure_rms`) — how well the
    /// interferogram network closed.
    network_misclosure_rad: Option<Array2<f64>>,
    /// Temporal motion-model fit residual RMS (see `VelocityFit::residual_rms`)
    /// — how well displacement fits the configured velocity model. Distinct from
    /// `network_misclosure_rad` (issue #40).
    timeseries_residual_rad: Option<Array2<f64>>,
    velocity_sigma_rad: Option<Array2<f64>>,
    interferogram_pairs: Vec<(usize, usize)>,
    unwrap_connected_components: Array3<u32>,
}

#[allow(clippy::too_many_lines)]
fn emit_displacement(
    cfg: &DisplacementWorkflow,
    days: Vec<f64>,
    epsg: Option<u32>,
    crop: Option<&BoundedPlan>,
    mut spatial: SpatialProducts,
    output_policy: DisplacementOutputPolicy,
) -> Result<DisplacementOutput> {
    if let Some(plan) = crop {
        spatial.trim(plan.target_in_analysis, &days, cfg)?;
    }
    spatial.apply_validity_mask();
    if let (Some(variance), Some(point)) = (
        spatial.posterior_variance_rad.as_mut(),
        spatial.reference_point,
    ) {
        reference_variance_to_point(variance, point);
    }
    let scaled = scale_outputs(cfg, &spatial);
    let quality = QualityLayers {
        posterior_dof: spatial
            .interferogram_pairs
            .len()
            .saturating_sub(spatial.disp_rad.dim().0),
        phase_linking_coherence: spatial.phase_linking_coherence.as_ref(),
        crlb_sigma: cfg
            .phase_linking
            .write_crlb
            .then_some(spatial.crlb_sigma.as_ref())
            .flatten(),
        closure_phase: spatial.closure_phase.as_ref(),
        displacement_variance: scaled.displacement_variance.as_ref(),
        network_misclosure_rms: scaled.network_misclosure_rms.as_ref(),
        timeseries_residual_rms: scaled.timeseries_residual_rms.as_ref(),
        velocity_sigma: scaled.velocity_sigma.as_ref(),
        connected_components: &spatial.unwrap_connected_components,
        velocity_terms: VelocityTermLayers {
            seasonal_amplitude: scaled.seasonal_amplitude.as_ref(),
            seasonal_phase_days: spatial.velocity_terms.seasonal_phase_days.as_ref(),
            step_magnitude: &scaled.step_magnitude,
        },
        loop_closure: spatial.loop_closure.as_ref(),
    };
    let input_coverage = summarize_input_coverage(&spatial);
    let geometry_provenance = crate::provenance::assemble_geometry_provenance_with_coverage(
        cfg,
        spatial.corrections.los_geometry.as_ref(),
        crop.map(|plan| plan.provenance.clone()),
        Some(input_coverage),
    );
    timed("write", || -> Result<()> {
        match output_policy {
            DisplacementOutputPolicy::Full => {
                write_outputs(
                    cfg,
                    scaled.displacement.view(),
                    scaled.velocity.view(),
                    spatial.temporal_coherence.view(),
                    quality,
                    epsg,
                    spatial.geotransform,
                )?;
                write_correction_outputs(cfg, &spatial.corrections, epsg, spatial.geotransform)?;
                crate::provenance::write_geometry_provenance(
                    &cfg.work_directory,
                    &geometry_provenance,
                )
            }
            DisplacementOutputPolicy::GroundPulse => {
                std::fs::create_dir_all(&cfg.work_directory)?;
                if let Some(coherence) = quality.phase_linking_coherence {
                    write_raster(
                        &cfg.work_directory.join("phase_linking_coherence.tif"),
                        coherence.mapv(|value| value as f32).view(),
                        spatial.geotransform,
                        epsg,
                        None,
                    )?;
                }
                Ok(())
            }
        }
    })?;
    Ok(DisplacementOutput {
        displacement: scaled.displacement,
        velocity: scaled.velocity,
        velocity_mm_yr: scaled.velocity_mm_yr,
        velocity_sigma: scaled.velocity_sigma,
        displacement_variance: scaled.displacement_variance,
        network_misclosure_rms: scaled.network_misclosure_rms,
        timeseries_residual_rms: scaled.timeseries_residual_rms,
        interferogram_pairs: spatial.interferogram_pairs,
        unwrap_connected_components: spatial.unwrap_connected_components,
        temporal_coherence: spatial.temporal_coherence,
        phase_linking_coherence: spatial.phase_linking_coherence,
        validity_mask: spatial.validity_mask,
        crlb_sigma: cfg
            .phase_linking
            .write_crlb
            .then_some(spatial.crlb_sigma)
            .flatten(),
        closure_phase: spatial.closure_phase,
        acquisition_days: days,
        epsg,
        geotransform: spatial.geotransform,
        reference_point: spatial.reference_point,
        ionosphere_delay: spatial.corrections.ionosphere,
        troposphere_delay: spatial.corrections.troposphere,
        solid_earth_tide_delay: spatial.corrections.solid_earth_tide,
        los_geometry: spatial.corrections.los_geometry,
        geometry_provenance,
    })
}

fn summarize_input_coverage(spatial: &SpatialProducts) -> InputCoverageProvenance {
    let output_pixels = spatial.validity_mask.len();
    let valid_pixels = spatial.validity_mask.iter().filter(|&&valid| valid).count();
    let sum =
        |pick: fn(&BurstCoverageProvenance) -> usize| spatial.burst_coverage.iter().map(pick).sum();
    let total_tiles = sum(|burst| burst.total_tiles);
    let linked_tiles = sum(|burst| burst.linked_tiles);
    let nodata_tiles = sum(|burst| burst.nodata_tiles);
    let valid_fraction = if output_pixels == 0 {
        0.0
    } else {
        valid_pixels as f64 / output_pixels as f64
    };
    tracing::info!(
        stage = "input_coverage",
        total_tiles,
        linked_tiles,
        nodata_tiles,
        output_pixels,
        valid_pixels,
        valid_fraction,
        coverage_policy = INPUT_COVERAGE_POLICY_VERSION,
        "input coverage complete"
    );
    InputCoverageProvenance {
        policy_version: INPUT_COVERAGE_POLICY_VERSION.into(),
        total_tiles,
        linked_tiles,
        nodata_tiles,
        bursts: spatial.burst_coverage.clone(),
        output_pixels,
        valid_pixels,
        valid_fraction,
    }
}

impl SpatialProducts {
    fn apply_validity_mask(&mut self) {
        ndarray::Zip::from(&mut self.validity_mask)
            .and(&self.vel_rad)
            .for_each(|valid, &velocity| *valid &= velocity.is_finite());
        let mask = &self.validity_mask;
        mask3_f64(&mut self.disp_rad, mask);
        mask2_f64(&mut self.vel_rad, mask);
        mask2_f64(&mut self.temporal_coherence, mask);
        if let Some(layer) = self.phase_linking_coherence.as_mut() {
            mask2_f64(layer, mask);
        }
        if let Some(layer) = self.crlb_sigma.as_mut() {
            mask3_f64(layer, mask);
        }
        if let Some(layer) = self.closure_phase.as_mut() {
            mask3_f64(layer, mask);
        }
        if let Some(layer) = self.posterior_variance_rad.as_mut() {
            mask3_f64(layer, mask);
        }
        if let Some(layer) = self.network_misclosure_rad.as_mut() {
            mask2_f64(layer, mask);
        }
        if let Some(layer) = self.timeseries_residual_rad.as_mut() {
            mask2_f64(layer, mask);
        }
        if let Some(layer) = self.velocity_sigma_rad.as_mut() {
            mask2_f64(layer, mask);
        }
        if let Some(layer) = self.corrections.ionosphere.as_mut() {
            mask3_f64(layer, mask);
        }
        if let Some(layer) = self.corrections.troposphere.as_mut() {
            mask3_f64(layer, mask);
        }
        if let Some(layer) = self.corrections.solid_earth_tide.as_mut() {
            mask3_f64(layer, mask);
        }
        if let Some(geometry) = self.corrections.los_geometry.as_mut() {
            mask2_f64(&mut geometry.east, mask);
            mask2_f64(&mut geometry.north, mask);
            mask2_f64(&mut geometry.up, mask);
        }
        ndarray::Zip::from(self.unwrap_connected_components.axis_iter_mut(Axis(0))).for_each(
            |mut band| {
                ndarray::Zip::from(&mut band)
                    .and(mask)
                    .for_each(|value, &valid| {
                        if !valid {
                            *value = 0;
                        }
                    });
            },
        );
    }

    fn trim(
        &mut self,
        target: BlockIndices,
        days: &[f64],
        cfg: &DisplacementWorkflow,
    ) -> Result<()> {
        // A halo reference is scientifically valid for analysis but cannot be
        // represented by a target-local coordinate. Re-reference to a coherent
        // target pixel before trimming so the emitted reference is always real.
        if self.reference_point.is_none_or(|(row, col)| {
            row < target.row_start
                || row >= target.row_stop
                || col < target.col_start
                || col >= target.col_stop
        }) {
            let target_coherence = self.temporal_coherence.slice(s![
                target.row_start..target.row_stop,
                target.col_start..target.col_stop
            ]);
            let local = select_reference_point(
                target_coherence,
                cfg.timeseries_options.correlation_threshold,
            )
            .context(
                "bounded target has no pixel meeting the configured reference coherence threshold",
            )?;
            let global = (target.row_start + local.0, target.col_start + local.1);
            reference_to_point(&mut self.disp_rad, global);
            // Re-fit through the same front door as the whole-frame path, so the
            // configured time-function model cannot reach one and not the other.
            let fit = fit_velocity(
                cfg,
                self.disp_rad.view(),
                days,
                self.crlb_sigma.as_ref(),
                &self.velocity_model,
            )
            .context("bounded velocity re-fit after re-referencing")?;
            self.vel_rad = fit.velocity;
            self.velocity_sigma_rad = fit.sigma;
            // Re-referencing shifts every date's displacement, which shifts the
            // temporal-fit residual too; the network misclosure is unaffected (it
            // is computed upstream, from the inversion, before re-referencing).
            self.timeseries_residual_rad = fit.residual_rms;
            self.velocity_terms = fit.terms;
            self.reference_point = Some(global);
        }
        self.disp_rad = trim3(&self.disp_rad, target);
        self.vel_rad = trim2(&self.vel_rad, target);
        self.velocity_terms.seasonal_amplitude_rad = self
            .velocity_terms
            .seasonal_amplitude_rad
            .take()
            .map(|layer| trim2(&layer, target));
        self.velocity_terms.seasonal_phase_days = self
            .velocity_terms
            .seasonal_phase_days
            .take()
            .map(|layer| trim2(&layer, target));
        self.velocity_terms.step_magnitude_rad =
            std::mem::take(&mut self.velocity_terms.step_magnitude_rad)
                .iter()
                .map(|layer| trim2(layer, target))
                .collect();
        self.temporal_coherence = trim2(&self.temporal_coherence, target);
        self.validity_mask = trim2(&self.validity_mask, target);
        self.phase_linking_coherence = self
            .phase_linking_coherence
            .take()
            .map(|layer| trim2(&layer, target));
        self.crlb_sigma = self.crlb_sigma.take().map(|layer| trim3(&layer, target));
        self.closure_phase = self.closure_phase.take().map(|layer| trim3(&layer, target));
        self.posterior_variance_rad = self
            .posterior_variance_rad
            .take()
            .map(|layer| trim3(&layer, target));
        self.network_misclosure_rad = self
            .network_misclosure_rad
            .take()
            .map(|layer| trim2(&layer, target));
        self.timeseries_residual_rad = self
            .timeseries_residual_rad
            .take()
            .map(|layer| trim2(&layer, target));
        self.velocity_sigma_rad = self
            .velocity_sigma_rad
            .take()
            .map(|layer| trim2(&layer, target));
        self.unwrap_connected_components = trim3(&self.unwrap_connected_components, target);
        trim_corrections(&mut self.corrections, target);
        self.reference_point = trim_reference(self.reference_point, target);
        self.geotransform =
            offset_geotransform(self.geotransform, target.row_start, target.col_start);
        Ok(())
    }
}

/// One burst's phase-linking products, carried until stitched onto the frame.
struct BurstLink {
    /// Linked phase history `(n_dates, out_rows, out_cols)`.
    pl: Array3<Cf64>,
    /// Temporal coherence `(out_rows, out_cols)`.
    temp_coh: Array2<f64>,
    /// Distinct phase-linking coherence `(out_rows, out_cols)`, if enabled.
    phase_linking_coherence: Option<Array2<f64>>,
    /// Per-date CRLB σ `(n_dates, out_rows, out_cols)`, if enabled.
    crlb_sigma: Option<Array3<f64>>,
    /// Per-triplet closure phase (band-major), if enabled.
    closure_phase: Option<Array3<f64>>,
    validity_mask: Array2<bool>,
    coverage: BurstCoverageProvenance,
    /// Burst footprint on the output grid.
    geo: BurstGeo,
    /// Acquisition decimal-days for this burst's dates.
    days: Vec<f64>,
}

/// Phase-link a single burst from the CSLC files at `idxs` in `cfg.cslc_file_list`.
///
/// Block-tiled: the burst is read and phase-linked one tile at a time (see
/// [`crate::tiling`]) so peak memory is bounded by a tile (block + halo) and its
/// `N×N` coherence cube, never the whole stack. The result is bit-identical to a
/// whole-burst run.
fn link_one_burst(
    cfg: &DisplacementWorkflow,
    idxs: &[usize],
    burst_index: usize,
    engine: &ComputeEngine,
    bounded: Option<BurstWindow>,
) -> Result<BurstLink> {
    let files = burst_files(cfg, idxs);
    let days = acquisition_days(&files, &cfg.input_options)
        .context("parsing acquisition dates from CSLC filenames")?;
    let subdataset = cfg
        .input_options
        .subdataset
        .clone()
        .context("input_options.subdataset is required to read CSLC HDF5")?;
    let full_shape = read_cslc_shape(&files[0], &subdataset)?;
    let source = bounded.map_or(
        BlockIndices {
            row_start: 0,
            row_stop: full_shape.0,
            col_start: 0,
            col_stop: full_shape.1,
        },
        |window| window.source,
    );
    let tiled = phase_link_tiled(
        cfg,
        (source.height(), source.width()),
        files.len(),
        engine,
        |block| {
            read_burst_tile(
                cfg.input_options.input_type,
                &files,
                &subdataset,
                offset_block(block, source.row_start, source.col_start),
            )
        },
    )
    .with_context(|| format!("burst ordinal {burst_index} phase linking failed"))?;
    let mut link = burst_link(
        cfg,
        tiled.output,
        days,
        &files[0],
        (source.row_start, source.col_start),
    )?;
    link.validity_mask = tiled.validity_mask;
    link.coverage = BurstCoverageProvenance {
        burst_index,
        acquisition_count: files.len(),
        total_tiles: tiled.total_tiles,
        linked_tiles: tiled.linked_tiles,
        nodata_tiles: tiled.nodata_tiles,
    };
    Ok(link)
}

struct TiledPhaseLinkOutput {
    output: SequentialOutput,
    validity_mask: Array2<bool>,
    total_tiles: usize,
    linked_tiles: usize,
    nodata_tiles: usize,
}

struct TiledPhaseLinkStats {
    acquisition_has_finite: Vec<bool>,
    total_tiles: usize,
    linked_tiles: usize,
    nodata_tiles: usize,
    read_s: f64,
    compute_s: f64,
}

impl TiledPhaseLinkStats {
    fn finish(self, acc: TiledOutput) -> Result<TiledPhaseLinkOutput> {
        let globally_empty: Vec<usize> = self
            .acquisition_has_finite
            .iter()
            .enumerate()
            .filter_map(|(ordinal, &seen)| (!seen).then_some(ordinal))
            .collect();
        anyhow::ensure!(
            globally_empty.is_empty(),
            "burst has globally all-nonfinite acquisition ordinals {globally_empty:?}"
        );
        anyhow::ensure!(
            self.linked_tiles > 0,
            "burst has no tile with complete temporal support"
        );
        tracing::info!(
            stage = "pl_breakdown",
            read_s = self.read_s,
            compute_s = self.compute_s,
            total_tiles = self.total_tiles,
            linked_tiles = self.linked_tiles,
            nodata_tiles = self.nodata_tiles,
            coverage_policy = INPUT_COVERAGE_POLICY_VERSION,
            "stage complete"
        );
        let validity_mask = acc.validity_mask.clone();
        Ok(TiledPhaseLinkOutput {
            output: acc.into_output(),
            validity_mask,
            total_tiles: self.total_tiles,
            linked_tiles: self.linked_tiles,
            nodata_tiles: self.nodata_tiles,
        })
    }
}

/// Phase-link a burst tile-by-tile, assembling the per-tile sequential outputs
/// into the whole-burst [`SequentialOutput`]. `read_tile` fetches one tile's
/// input (block + halo) across all epochs as `Cf64`; tiling guarantees each
/// output pixel sees the same window it would in a whole-burst run, so the
/// assembled result is bit-identical.
fn phase_link_tiled(
    cfg: &DisplacementWorkflow,
    full_shape: (usize, usize),
    nslc: usize,
    engine: &ComputeEngine,
    read_tile: impl Fn(BlockIndices) -> Result<Array3<Cf64>>,
) -> Result<TiledPhaseLinkOutput> {
    let strides = cfg.output_options.strides;
    let half = cfg.phase_linking.half_window;
    let out_shape = strides.out_shape(full_shape);
    let (bh, bw) = cfg.worker_settings.block_shape;
    let out_block = ((bh / strides.y).max(1), (bw / strides.x).max(1));
    // A written pixel's data dependency cone spans `num_ministacks` half-windows
    // (each ministack's carried compressed SLC is itself window-based); the halo
    // must cover that or interior seams corrupt. div_ceil is an exact upper bound
    // on the planner's ministack count.
    let depth = nslc.div_ceil(cfg.phase_linking.ministack_size.max(1));
    let mut acc = TiledOutput::new(
        nslc,
        out_shape,
        cfg.phase_linking.write_crlb,
        cfg.phase_linking.calc_average_coh,
    );
    let plans = plan_tiles(full_shape, strides, half, depth, out_block);
    let tile_count = plans.len();
    let mut stats = TiledPhaseLinkStats {
        acquisition_has_finite: vec![false; nslc],
        total_tiles: tile_count,
        linked_tiles: 0,
        nodata_tiles: 0,
        read_s: 0.0,
        compute_s: 0.0,
    };
    for (tile_offset, plan) in plans.into_iter().enumerate() {
        let tile_index = tile_offset + 1;
        let (rss_kib, peak_rss_kib) = memory_kib();
        tracing::debug!(
            stage = "phase_linking_tile",
            event = "start",
            tile_index,
            tile_count,
            nslc,
            input_rows = plan.read.height(),
            input_cols = plan.read.width(),
            output_rows = plan.out.height(),
            output_cols = plan.out.width(),
            stride_y = strides.y,
            stride_x = strides.x,
            phase_linking_coherence = cfg.phase_linking.calc_average_coh,
            rss_kib,
            peak_rss_kib,
            "phase-linking tile start"
        );
        let t_read = Instant::now();
        let stack = read_tile(plan.read)?;
        stats.read_s += t_read.elapsed().as_secs_f64();
        let missing = all_non_finite_acquisition_indices(stack.view());
        let mut locally_finite = vec![true; nslc];
        for &ordinal in &missing {
            locally_finite[ordinal] = false;
        }
        for (seen, finite) in stats.acquisition_has_finite.iter_mut().zip(locally_finite) {
            *seen |= finite;
        }
        let (rss_kib, peak_rss_kib) = memory_kib();
        tracing::debug!(
            stage = "phase_linking_tile",
            event = "read_complete",
            tile_index,
            tile_count,
            rss_kib,
            peak_rss_kib,
            "phase-linking tile read complete"
        );
        if !missing.is_empty() {
            stats.nodata_tiles += 1;
            acc.place_nodata(&plan);
            continue;
        }
        let t_pl = Instant::now();
        let out = phase_link(cfg, stack.view(), engine)?;
        stats.compute_s += t_pl.elapsed().as_secs_f64();
        let (rss_kib, peak_rss_kib) = memory_kib();
        tracing::debug!(
            stage = "phase_linking_tile",
            event = "compute_complete",
            tile_index,
            tile_count,
            rss_kib,
            peak_rss_kib,
            "phase-linking tile compute complete"
        );
        acc.place(&plan, &out)?;
        stats.linked_tiles += 1;
        let (rss_kib, peak_rss_kib) = memory_kib();
        tracing::debug!(
            stage = "phase_linking_tile",
            event = "complete",
            tile_index,
            tile_count,
            rss_kib,
            peak_rss_kib,
            "phase-linking tile complete"
        );
    }
    // Sub-breakdown of the `phase_linking` stage: windowed CSLC read vs the
    // covariance+estimator compute, summed across tiles (wall, not exclusive CPU).
    stats.finish(acc)
}

/// Read one tile's input (`block`, including halo) across all `files` epochs as a
/// `(nslc, h, w)` `Cf64` stack. Each epoch is read as a `Cf32` window and upcast
/// in place — the global `Cf32→Cf64` doubling of the whole-burst load is gone;
/// only one tile (plus one transient `Cf32` window) is ever resident.
fn read_burst_tile(
    input_type: InputType,
    files: &[std::path::PathBuf],
    subdataset: &str,
    block: BlockIndices,
) -> Result<Array3<Cf64>> {
    let reader = match input_type {
        InputType::OperaCslc => read_cslc_window,
        InputType::NisarGslc => read_nisar_window,
    };
    let mut tile = Array3::<Cf64>::zeros((files.len(), block.height(), block.width()));
    for (k, path) in files.iter().enumerate() {
        let window = reader(path, subdataset, block)?;
        upcast_into(tile.index_axis_mut(Axis(0), k), window.view());
    }
    Ok(tile)
}

/// Upcast a `Cf32` window into a `Cf64` destination view (the only place the
/// stack is widened — per tile, not per whole burst).
fn upcast_into(dst: ArrayViewMut2<Cf64>, src: ndarray::ArrayView2<Cf32>) {
    ndarray::Zip::from(dst)
        .and(src)
        .for_each(|d, z| *d = Cf64::new(z.re as f64, z.im as f64));
}

fn offset_block(block: BlockIndices, row_offset: usize, col_offset: usize) -> BlockIndices {
    BlockIndices {
        row_start: block.row_start + row_offset,
        row_stop: block.row_stop + row_offset,
        col_start: block.col_start + col_offset,
        col_stop: block.col_stop + col_offset,
    }
}

fn trim2<T: Clone>(array: &Array2<T>, block: BlockIndices) -> Array2<T> {
    array.slice(s![block.rows(), block.cols()]).to_owned()
}

fn trim3<T: Clone>(array: &Array3<T>, block: BlockIndices) -> Array3<T> {
    array.slice(s![.., block.rows(), block.cols()]).to_owned()
}

fn trim_corrections(corrections: &mut CorrectionLayers, block: BlockIndices) {
    corrections.ionosphere = corrections
        .ionosphere
        .take()
        .map(|layer| trim3(&layer, block));
    corrections.troposphere = corrections
        .troposphere
        .take()
        .map(|layer| trim3(&layer, block));
    corrections.solid_earth_tide = corrections
        .solid_earth_tide
        .take()
        .map(|layer| trim3(&layer, block));
    corrections.los_geometry = corrections.los_geometry.take().map(|geometry| LosGeometry {
        east: trim2(&geometry.east, block),
        north: trim2(&geometry.north, block),
        up: trim2(&geometry.up, block),
    });
}

fn trim_reference(point: Option<(usize, usize)>, block: BlockIndices) -> Option<(usize, usize)> {
    let (row, col) = point?;
    (block.rows().contains(&row) && block.cols().contains(&col))
        .then_some((row - block.row_start, col - block.col_start))
}

fn configured_analysis_reference(
    point: Option<(usize, usize)>,
    crop: Option<&BoundedPlan>,
    analysis_shape: (usize, usize),
) -> Result<Option<(usize, usize)>> {
    let Some((row, col)) = point else {
        return Ok(None);
    };
    let Some(plan) = crop else {
        anyhow::ensure!(
            row < analysis_shape.0 && col < analysis_shape.1,
            "timeseries reference_point falls outside the output grid"
        );
        return Ok(Some((row, col)));
    };
    let [analysis_row, analysis_col] = plan.provenance.analysis_pixel_offset;
    anyhow::ensure!(
        row >= analysis_row && col >= analysis_col,
        "timeseries reference_point falls outside the bounded analysis domain"
    );
    let local = (row - analysis_row, col - analysis_col);
    anyhow::ensure!(
        local.0 < analysis_shape.0 && local.1 < analysis_shape.1,
        "timeseries reference_point falls outside the bounded analysis domain"
    );
    Ok(Some(local))
}

fn offset_geotransform(gt: [f64; 6], row: usize, col: usize) -> [f64; 6] {
    [
        gt[0] + col as f64 * gt[1] + row as f64 * gt[2],
        gt[1],
        gt[2],
        gt[3] + col as f64 * gt[4] + row as f64 * gt[5],
        gt[4],
        gt[5],
    ]
}

/// Accumulates per-tile sequential outputs into the whole-burst grid. The
/// per-tile compressed SLCs are not assembled (the batch path never consumes
/// them); the closure layer is allocated lazily once its band count is known.
struct TiledOutput {
    cpx: Array3<Cf64>,
    temp_coh: Array2<f64>,
    phase_linking_coherence: Option<Array2<f64>>,
    crlb: Option<Array3<f64>>,
    closure: Option<Array3<f64>>,
    validity_mask: Array2<bool>,
    out_shape: (usize, usize),
}

impl TiledOutput {
    fn new(
        nslc: usize,
        out_shape: (usize, usize),
        want_crlb: bool,
        want_average_coherence: bool,
    ) -> Self {
        let (or, oc) = out_shape;
        Self {
            cpx: Array3::from_elem((nslc, or, oc), Cf64::new(f64::NAN, f64::NAN)),
            temp_coh: Array2::from_elem((or, oc), f64::NAN),
            phase_linking_coherence: want_average_coherence
                .then(|| Array2::from_elem((or, oc), f64::NAN)),
            crlb: want_crlb.then(|| Array3::from_elem((nslc, or, oc), f64::NAN)),
            closure: None,
            validity_mask: Array2::from_elem((or, oc), false),
            out_shape,
        }
    }

    /// Copy the (halo-trimmed) tile output into its global output rectangle.
    fn place(&mut self, plan: &TilePlan, out: &SequentialOutput) -> Result<()> {
        let (h, w) = (plan.out.height(), plan.out.width());
        let g = (plan.out.row_start, plan.out.col_start);
        let l = (plan.local_row0, plan.local_col0);
        let (_, lor, loc) = out.cpx_phase.dim();
        anyhow::ensure!(
            l.0 + h <= lor && l.1 + w <= loc,
            "tile kernel output smaller than its written region"
        );
        assign_block3(&mut self.cpx, &out.cpx_phase, g, l, (h, w));
        self.temp_coh
            .slice_mut(s![g.0..g.0 + h, g.1..g.1 + w])
            .assign(&out.temporal_coherence.slice(s![l.0..l.0 + h, l.1..l.1 + w]));
        if let (Some(dst), Some(src)) = (
            self.phase_linking_coherence.as_mut(),
            out.phase_linking_coherence.as_ref(),
        ) {
            dst.slice_mut(s![g.0..g.0 + h, g.1..g.1 + w])
                .assign(&src.slice(s![l.0..l.0 + h, l.1..l.1 + w]));
        }
        if let (Some(dst), Some(src)) = (self.crlb.as_mut(), out.crlb_sigma.as_ref()) {
            assign_block3(dst, src, g, l, (h, w));
        }
        if let Some(src) = out.closure_phase.as_ref() {
            let (or, oc) = self.out_shape;
            let dst = self
                .closure
                .get_or_insert_with(|| Array3::from_elem((src.dim().0, or, oc), f64::NAN));
            assign_block3(dst, src, g, l, (h, w));
        }
        self.validity_mask
            .slice_mut(s![g.0..g.0 + h, g.1..g.1 + w])
            .fill(true);
        Ok(())
    }

    fn place_nodata(&mut self, plan: &TilePlan) {
        let g = (plan.out.row_start, plan.out.col_start);
        let (h, w) = (plan.out.height(), plan.out.width());
        self.validity_mask
            .slice_mut(s![g.0..g.0 + h, g.1..g.1 + w])
            .fill(false);
    }

    fn into_output(self) -> SequentialOutput {
        SequentialOutput {
            cpx_phase: self.cpx,
            compressed_slcs: Vec::new(),
            temporal_coherence: self.temp_coh,
            phase_linking_coherence: self.phase_linking_coherence,
            crlb_sigma: self.crlb,
            closure_phase: self.closure,
        }
    }
}

/// Assign a `(h, w)` block of a band-major `(bands, rows, cols)` array from the
/// `l`-offset region of `src` into the `g`-offset region of `dst`.
fn assign_block3<T: Clone>(
    dst: &mut Array3<T>,
    src: &Array3<T>,
    g: (usize, usize),
    l: (usize, usize),
    hw: (usize, usize),
) {
    let (h, w) = hw;
    dst.slice_mut(s![.., g.0..g.0 + h, g.1..g.1 + w])
        .assign(&src.slice(s![.., l.0..l.0 + h, l.1..l.1 + w]));
}

/// Build a [`BurstLink`] from a burst's sequential output, validating the
/// date/acquisition count and resolving its footprint on the output grid.
fn burst_link(
    cfg: &DisplacementWorkflow,
    out: SequentialOutput,
    days: Vec<f64>,
    first_file: &Path,
    source_offset: (usize, usize),
) -> Result<BurstLink> {
    let (_, rows, cols) = out.cpx_phase.dim();
    anyhow::ensure!(
        days.len() == out.cpx_phase.dim().0,
        "parsed {} dates but phase-linking produced {} acquisitions",
        days.len(),
        out.cpx_phase.dim().0
    );
    Ok(BurstLink {
        pl: out.cpx_phase,
        temp_coh: out.temporal_coherence,
        phase_linking_coherence: out.phase_linking_coherence,
        crlb_sigma: out.crlb_sigma,
        closure_phase: out.closure_phase,
        validity_mask: Array2::from_elem((rows, cols), true),
        coverage: BurstCoverageProvenance {
            burst_index: 0,
            acquisition_count: days.len(),
            total_tiles: 1,
            linked_tiles: 1,
            nodata_tiles: 0,
        },
        geo: resolve_burst_geo(cfg, first_file, rows, cols, source_offset)?,
        days,
    })
}

fn mask2_f64(values: &mut Array2<f64>, mask: &Array2<bool>) {
    ndarray::Zip::from(values)
        .and(mask)
        .for_each(|value, &valid| {
            if !valid {
                *value = f64::NAN;
            }
        });
}

fn mask3_f64(values: &mut Array3<f64>, mask: &Array2<bool>) {
    for mut band in values.axis_iter_mut(Axis(0)) {
        ndarray::Zip::from(&mut band)
            .and(mask)
            .for_each(|value, &valid| {
                if !valid {
                    *value = f64::NAN;
                }
            });
    }
}

/// Persisted state for an NRT incremental displacement update: per-burst
/// resumable phase-linking state and the files consumed so far. Obtain it from
/// [`run_displacement_resumable`] and thread it through [`update_displacement`].
///
/// Opaque; the same config (phase-linking parameters, strides, input type) must
/// be used across the resumed series.
pub struct DisplacementState {
    bursts: Vec<BurstState>,
}

/// One burst's resumable state.
struct BurstState {
    /// Burst id (the `group_by_burst` key).
    id: String,
    /// CSLC files consumed so far, in date order.
    files: Vec<PathBuf>,
    /// Footprint on the output grid (stable across updates).
    geo: BurstGeo,
    /// Full-resolution source read window (stable across updates).
    source_window: BlockIndices,
    /// Sequential phase-linking carry (sealed ministacks + open trailing SLCs).
    seq: SequentialState,
}

/// Phase-link a single burst, also returning its resumable [`SequentialState`].
fn link_one_burst_resumable(
    cfg: &DisplacementWorkflow,
    idxs: &[usize],
    engine: &ComputeEngine,
    bounded: Option<BurstWindow>,
    burst_index: usize,
) -> Result<(BurstLink, SequentialState, BlockIndices)> {
    let files = burst_files(cfg, idxs);
    let days = acquisition_days(&files, &cfg.input_options)
        .context("parsing acquisition dates from CSLC filenames")?;
    let subdataset = cfg
        .input_options
        .subdataset
        .as_deref()
        .context("input_options.subdataset is required to read CSLC HDF5")?;
    let full_shape = read_cslc_shape(&files[0], subdataset)?;
    let source = bounded.map_or(
        BlockIndices {
            row_start: 0,
            row_stop: full_shape.0,
            col_start: 0,
            col_stop: full_shape.1,
        },
        |window| window.source,
    );
    let stack = read_burst_tile(cfg.input_options.input_type, &files, subdataset, source)?;
    let (out, state) = run_sequential_resumable(stack.view(), &sequential_config(cfg), engine)
        .map_err(anyhow::Error::msg)?;
    let mut link = burst_link(
        cfg,
        out,
        days,
        &files[0],
        (source.row_start, source.col_start),
    )?;
    link.coverage.burst_index = burst_index;
    link.coverage.acquisition_count = files.len();
    Ok((link, state, source))
}

/// The CSLC files for a burst's indices into `cfg.cslc_file_list`.
fn burst_files(cfg: &DisplacementWorkflow, idxs: &[usize]) -> Vec<PathBuf> {
    idxs.iter()
        .map(|&i| cfg.cslc_file_list[i].clone())
        .collect()
}

fn source_layouts(
    cfg: &DisplacementWorkflow,
    groups: &std::collections::BTreeMap<String, Vec<usize>>,
) -> Result<Vec<BurstGeo>> {
    let subdataset = cfg
        .input_options
        .subdataset
        .as_deref()
        .context("input_options.subdataset is required to inspect CSLC grids")?;
    groups
        .values()
        .map(|indices| {
            let first = indices
                .first()
                .and_then(|&index| cfg.cslc_file_list.get(index))
                .context("burst has no CSLC files")?;
            let source_shape = read_cslc_shape(first, subdataset)?;
            let output_shape = cfg.output_options.strides.out_shape(source_shape);
            resolve_burst_geo(cfg, first, output_shape.0, output_shape.1, (0, 0))
        })
        .collect()
}

/// Run the displacement workflow and also return the [`DisplacementState`] needed
/// to fold in later acquisitions via [`update_displacement`]. The
/// [`DisplacementOutput`] is identical to [`run_displacement`]'s.
///
/// # Errors
/// Same as [`run_displacement`].
pub fn run_displacement_resumable(
    cfg: &DisplacementWorkflow,
) -> Result<(DisplacementOutput, DisplacementState)> {
    validate_uncertainty_options(cfg)?;
    let engine = ComputeEngine::new(cfg.worker_settings.compute_backend);
    let groups = workflow_groups(cfg)?;
    let layouts = source_layouts(cfg, &groups)?;
    let acquisitions = groups.values().map(Vec::len).max().unwrap_or(0);
    let crop = plan_bounds(cfg, &layouts, acquisitions)?;
    let mut bursts = Vec::with_capacity(groups.len());
    let mut states = Vec::with_capacity(groups.len());
    let linked = timed("phase_linking", || -> Result<Vec<_>> {
        groups
            .iter()
            .enumerate()
            .filter_map(|(index, (id, idxs))| {
                let window = crop
                    .as_ref()
                    .map_or(Some(None), |plan| plan.windows[index].map(Some))?;
                Some((|| {
                    let (link, seq, source) =
                        link_one_burst_resumable(cfg, idxs, &engine, window, index)?;
                    Ok((id.clone(), burst_files(cfg, idxs), link, seq, source))
                })())
            })
            .collect()
    })?;
    for (id, files, link, seq, source_window) in linked {
        states.push(BurstState {
            id,
            files,
            geo: link.geo,
            source_window,
            seq,
        });
        bursts.push(link);
    }
    let output = finish_displacement(cfg, bursts, crop.as_ref(), DisplacementOutputPolicy::Full)?;
    Ok((output, DisplacementState { bursts: states }))
}

/// Fold newly-arrived acquisitions into an existing displacement series. `cfg`
/// carries the **full extended** `cslc_file_list` (the prior files as a prefix
/// plus the new ones); `update_displacement` re-phase-links only each burst's
/// open trailing ministack + new ministacks (carrying the sealed compressed SLCs
/// in `state`), then recomputes the non-causal downstream. The result equals
/// [`run_displacement`] on the extended stack.
///
/// A streaming update must extend **every** burst by ≥1 acquisition (a new SAR
/// pass yields one CSLC per burst), and the prior files must be a date-ordered
/// prefix of the new list. `cfg` must match the run that produced `state`.
///
/// # Errors
/// Returns `Err` if a burst is missing/empty/not-a-prefix in the new list, or on
/// the usual I/O / phase-linking / unwrap / date-parsing failures.
pub fn update_displacement(
    state: &DisplacementState,
    cfg: &DisplacementWorkflow,
) -> Result<(DisplacementOutput, DisplacementState)> {
    validate_uncertainty_options(cfg)?;
    // The finite dependency cone grows when an update adds ministacks. Reusing a
    // prior bounded state could therefore omit newly-required halo pixels. A
    // bounded update deliberately recomputes the bounded analysis domain; it is
    // still memory-bounded and scientifically equivalent to a fresh AOI-local run.
    if cfg.output_options.bounds.is_some() {
        return run_displacement_resumable(cfg);
    }
    let engine = ComputeEngine::new(cfg.worker_settings.compute_backend);
    let groups = workflow_groups(cfg)?;
    let layouts = source_layouts(cfg, &groups)?;
    let acquisitions = groups.values().map(Vec::len).max().unwrap_or(0);
    let crop = plan_bounds(cfg, &layouts, acquisitions)?;
    let mut bursts = Vec::with_capacity(groups.len());
    let mut states = Vec::with_capacity(groups.len());
    let updated = timed("phase_linking", || -> Result<Vec<_>> {
        groups
            .iter()
            .enumerate()
            .filter_map(|(index, (id, idxs))| {
                let window = crop
                    .as_ref()
                    .map_or(Some(None), |plan| plan.windows[index].map(Some))?;
                Some(update_one_burst(
                    state, cfg, id, idxs, &engine, window, index,
                ))
            })
            .collect()
    })?;
    for (link, st) in updated {
        states.push(st);
        bursts.push(link);
    }
    let output = finish_displacement(cfg, bursts, crop.as_ref(), DisplacementOutputPolicy::Full)?;
    Ok((output, DisplacementState { bursts: states }))
}

/// Fold new acquisitions into one burst, returning its extended link + new state.
fn update_one_burst(
    state: &DisplacementState,
    cfg: &DisplacementWorkflow,
    id: &str,
    idxs: &[usize],
    engine: &ComputeEngine,
    bounded: Option<BurstWindow>,
    burst_index: usize,
) -> Result<(BurstLink, BurstState)> {
    let files = burst_files(cfg, idxs);
    let prev = state
        .bursts
        .iter()
        .find(|b| b.id == id)
        .with_context(|| format!("burst {id} is new; updates must not introduce bursts"))?;
    anyhow::ensure!(
        files.starts_with(&prev.files),
        "burst {id}: prior files must be a date-ordered prefix of the updated list"
    );
    let new_files = &files[prev.files.len()..];
    anyhow::ensure!(
        !new_files.is_empty(),
        "burst {id}: no new acquisitions; an update must extend every burst"
    );
    let planned_window = bounded.map_or(prev.source_window, |window| window.source);
    anyhow::ensure!(
        planned_window == prev.source_window,
        "bounded source window changed during an incremental update"
    );
    let subdataset = cfg
        .input_options
        .subdataset
        .as_deref()
        .context("input_options.subdataset is required to read CSLC HDF5")?;
    let new_stack = read_burst_tile(
        cfg.input_options.input_type,
        new_files,
        subdataset,
        prev.source_window,
    )?;
    let (out, seq) =
        update_sequential(&prev.seq, new_stack.view(), &sequential_config(cfg), engine)
            .map_err(anyhow::Error::msg)?;
    let days = acquisition_days(&files, &cfg.input_options)
        .context("parsing acquisition dates from CSLC filenames")?;
    let mut link = burst_link(
        cfg,
        out,
        days,
        &files[0],
        (prev.source_window.row_start, prev.source_window.col_start),
    )?;
    link.coverage.burst_index = burst_index;
    link.coverage.acquisition_count = files.len();
    let next = BurstState {
        id: id.to_string(),
        files,
        geo: prev.geo,
        source_window: prev.source_window,
        seq,
    };
    Ok((link, next))
}

fn validate_uncertainty_options(cfg: &DisplacementWorkflow) -> Result<()> {
    anyhow::ensure!(
        !cfg
            .timeseries_options
            .correct_velocity_temporal_correlation
            || cfg.timeseries_options.write_velocity_uncertainty,
        "timeseries_options.correct_velocity_temporal_correlation requires write_velocity_uncertainty"
    );
    // `preprocess_options` round-trips a dolphin YAML and `crop.rs` already
    // reserves `max_radius` of halo for it, but no interpolation stage exists —
    // so accepting the flag would widen every AOI read and change nothing. Reject
    // it rather than silently ignore it (same rule as the check above, #25).
    anyhow::ensure!(
        !cfg.unwrap_options.run_interpolation,
        "unwrap_options.run_interpolation is accepted for dolphin YAML round-trip but the \
         pre-unwrap interpolation stage is not implemented; leave it false (dolphin's own \
         default) rather than running with it silently ignored"
    );
    Ok(())
}

/// The frame-grid mosaic of the per-burst phase-linking products.
struct Stitched {
    /// Linked phase history `(n_dates, rows, cols)`.
    pl: Array3<Cf64>,
    /// Temporal coherence `(rows, cols)`.
    temp_coh: Array2<f64>,
    /// Distinct phase-linking coherence `(rows, cols)`, if enabled.
    phase_linking_coherence: Option<Array2<f64>>,
    /// Per-date CRLB σ `(n_dates, rows, cols)`, if enabled.
    crlb_sigma: Option<Array3<f64>>,
    /// Per-triplet closure phase (band-major), if enabled.
    closure_phase: Option<Array3<f64>>,
    validity_mask: Array2<bool>,
    coverage: Vec<BurstCoverageProvenance>,
    /// Frame grid georeferencing.
    geo: GeoInfo,
}

/// Mosaic the per-burst phase-linking products onto the frame grid. A single
/// burst is returned as-is (identity path).
fn stitch_bursts(mut bursts: Vec<BurstLink>) -> Result<Stitched> {
    anyhow::ensure!(!bursts.is_empty(), "no bursts to stitch");
    if bursts.len() == 1 {
        let b = bursts.remove(0);
        return Ok(Stitched {
            pl: b.pl,
            temp_coh: b.temp_coh,
            phase_linking_coherence: b.phase_linking_coherence,
            crlb_sigma: b.crlb_sigma,
            closure_phase: b.closure_phase,
            validity_mask: b.validity_mask,
            coverage: vec![b.coverage],
            geo: b.geo.geo,
        });
    }
    let geos: Vec<BurstGeo> = bursts.iter().map(|b| b.geo).collect();
    let frame = frame_grid(&geos)?;
    let nslc = bursts[0].pl.dim().0;
    let mut pl = Array3::<Cf64>::from_elem(
        (nslc, frame.rows, frame.cols),
        Cf64::new(f64::NAN, f64::NAN),
    );
    let mut temp_coh = Array2::<f64>::from_elem((frame.rows, frame.cols), f64::NAN);
    let mut covered = Array2::<bool>::from_elem((frame.rows, frame.cols), false);
    for (burst_index, b) in bursts.iter_mut().enumerate() {
        anyhow::ensure!(b.pl.dim().0 == nslc, "bursts have differing date counts");
        let off = burst_offset(&frame, &b.geo);
        if burst_index > 0 {
            level_burst_offsets(&pl, &temp_coh, &covered, b, off, burst_index)?;
        }
        paste3_finite_complex(&mut pl, &b.pl, off);
        paste2_finite(&mut temp_coh, &b.temp_coh, off);
        let (rows, cols) = b.temp_coh.dim();
        let mut target = covered.slice_mut(s![off.0..off.0 + rows, off.1..off.1 + cols]);
        ndarray::Zip::from(&mut target)
            .and(&b.validity_mask)
            .for_each(|dst, &src| *dst |= src);
    }
    let crlb_sigma = stitch_layer(&bursts, &frame, |b| b.crlb_sigma.as_ref());
    let closure_phase = stitch_layer(&bursts, &frame, |b| b.closure_phase.as_ref());
    let phase_linking_coherence =
        stitch_optional_2d(&bursts, &frame, |b| b.phase_linking_coherence.as_ref());
    Ok(Stitched {
        pl,
        temp_coh,
        phase_linking_coherence,
        crlb_sigma,
        closure_phase,
        validity_mask: covered,
        coverage: bursts.iter().map(|burst| burst.coverage.clone()).collect(),
        geo: frame.geo,
    })
}

fn paste2_finite(frame: &mut Array2<f64>, burst: &Array2<f64>, offset: (usize, usize)) {
    let (row, col) = offset;
    let (rows, cols) = burst.dim();
    let mut target = frame.slice_mut(s![row..row + rows, col..col + cols]);
    ndarray::Zip::from(&mut target)
        .and(burst)
        .for_each(|dst, &src| {
            if src.is_finite() {
                *dst = src;
            }
        });
}

fn paste3_finite_complex(frame: &mut Array3<Cf64>, burst: &Array3<Cf64>, offset: (usize, usize)) {
    let (row, col) = offset;
    let (_, rows, cols) = burst.dim();
    let mut target = frame.slice_mut(s![.., row..row + rows, col..col + cols]);
    ndarray::Zip::from(&mut target)
        .and(burst)
        .for_each(|dst, &src| {
            if src.re.is_finite() && src.im.is_finite() {
                *dst = src;
            }
        });
}

/// Rotate every acquisition of `burst` onto the phase datum already established
/// in `frame`. The circular mean uses only finite, nonzero samples whose temporal
/// coherence is stable on both sides of the seam.
fn level_burst_offsets(
    frame: &Array3<Cf64>,
    frame_coherence: &Array2<f64>,
    covered: &Array2<bool>,
    burst: &mut BurstLink,
    offset: (usize, usize),
    burst_index: usize,
) -> std::result::Result<(), StitchError> {
    let (_, rows, cols) = burst.pl.dim();
    for acquisition_index in 0..burst.pl.dim().0 {
        let mut sum = Cf64::new(0.0, 0.0);
        let mut support = 0;
        for row in 0..rows {
            for col in 0..cols {
                let global = (offset.0 + row, offset.1 + col);
                let existing = frame[(acquisition_index, global.0, global.1)];
                let candidate = burst.pl[(acquisition_index, row, col)];
                let stable = covered[global]
                    && frame_coherence[global].is_finite()
                    && frame_coherence[global] >= MIN_SEAM_COHERENCE
                    && burst.temp_coh[(row, col)].is_finite()
                    && burst.temp_coh[(row, col)] >= MIN_SEAM_COHERENCE;
                if stable
                    && existing.re.is_finite()
                    && existing.im.is_finite()
                    && candidate.re.is_finite()
                    && candidate.im.is_finite()
                    && existing.norm_sqr() > 0.0
                    && candidate.norm_sqr() > 0.0
                {
                    sum += (existing * candidate.conj()) / (existing.norm() * candidate.norm());
                    support += 1;
                }
            }
        }
        if support < MIN_SEAM_SUPPORT || sum.norm_sqr() == 0.0 {
            return Err(StitchError::InsufficientOffsetSupport {
                burst_index,
                acquisition_index,
                support,
                required: MIN_SEAM_SUPPORT,
            });
        }
        let rotation = Cf64::from_polar(1.0, sum.arg());
        burst
            .pl
            .index_axis_mut(Axis(0), acquisition_index)
            .mapv_inplace(|value| value * rotation);
    }
    Ok(())
}

/// Mosaic an optional per-burst 2D layer onto the frame grid.
fn stitch_optional_2d(
    bursts: &[BurstLink],
    frame: &FrameGrid,
    pick: impl Fn(&BurstLink) -> Option<&Array2<f64>>,
) -> Option<Array2<f64>> {
    pick(bursts.first()?)?;
    let mut out = Array2::<f64>::from_elem((frame.rows, frame.cols), f64::NAN);
    for burst in bursts {
        paste2_finite(&mut out, pick(burst)?, burst_offset(frame, &burst.geo));
    }
    Some(out)
}

/// Mosaic an optional per-burst band-major layer onto the frame grid; `None`
/// when the layer is disabled (no burst carries it).
fn stitch_layer(
    bursts: &[BurstLink],
    frame: &FrameGrid,
    pick: impl Fn(&BurstLink) -> Option<&Array3<f64>>,
) -> Option<Array3<f64>> {
    let bands = pick(bursts.first()?)?.dim().0;
    let mut out = Array3::<f64>::from_elem((bands, frame.rows, frame.cols), f64::NAN);
    for b in bursts {
        let layer = pick(b)?;
        let off = burst_offset(frame, &b.geo);
        let (_, rows, cols) = layer.dim();
        let mut target = out.slice_mut(s![.., off.0..off.0 + rows, off.1..off.1 + cols]);
        ndarray::Zip::from(&mut target)
            .and(layer)
            .for_each(|dst, &src| {
                if src.is_finite() {
                    *dst = src;
                }
            });
    }
    Some(out)
}

/// Burst footprint on the output grid: the CSLC geotransform (scaled by the
/// output strides for multilooking), else the config EPSG + identity placeholder.
fn resolve_burst_geo(
    cfg: &DisplacementWorkflow,
    path: &Path,
    rows: usize,
    cols: usize,
    source_offset: (usize, usize),
) -> Result<BurstGeo> {
    let geo_reader = match cfg.input_options.input_type {
        InputType::OperaCslc => read_geotransform,
        InputType::NisarGslc => read_nisar_geotransform,
    };
    let geo = cfg
        .input_options
        .subdataset
        .as_deref()
        .context("input_options.subdataset is required to source the burst georeference")
        .and_then(|sds| geo_reader(path, sds).context("reading required source georeference"))?;
    anyhow::ensure!(geo.epsg != 0, "source georeference has no valid EPSG");
    let (epsg, mut gt) = (geo.epsg, geo.geotransform);
    gt = offset_geotransform(gt, source_offset.0, source_offset.1);
    let (sx, sy) = (
        cfg.output_options.strides.x as f64,
        cfg.output_options.strides.y as f64,
    );
    Ok(BurstGeo {
        geo: GeoInfo {
            epsg,
            geotransform: [gt[0], gt[1] * sx, 0.0, gt[3], 0.0, gt[5] * sy],
        },
        rows,
        cols,
    })
}

/// Sequential phase linking over the stack; returns the linked phase history,
/// the averaged temporal coherence, and the optional CRLB / closure layers.
fn phase_link(
    cfg: &DisplacementWorkflow,
    stack: ArrayView3<Cf64>,
    engine: &ComputeEngine,
) -> Result<SequentialOutput> {
    run_sequential(stack, &sequential_config(cfg), engine).map_err(anyhow::Error::msg)
}

/// Subtract the phase-bias (non-closure) cumulative bias from the stitched linked
/// phase, estimated from the closure-phase layer (Michaelides et al. 2022). Opt-in
/// via `phase_linking.correct_phase_bias`; the closure layer is forced on with it.
fn apply_phase_bias(pl: &mut Array3<Cf64>, closure: Option<&Array3<f64>>) -> Result<()> {
    let closure = closure.context("phase-bias correction requires the closure-phase layer")?;
    let beta = estimate_bias_velocity(closure.view());
    correct_phase_bias(pl, beta.view());
    Ok(())
}

/// Map the workflow config onto the sequential-estimator config (shared by the
/// batch and incremental phase-linking paths).
fn sequential_config(cfg: &DisplacementWorkflow) -> SequentialConfig {
    SequentialConfig {
        ministack_size: cfg.phase_linking.ministack_size,
        max_num_compressed: cfg.phase_linking.max_num_compressed,
        half_window: cfg.phase_linking.half_window,
        strides: cfg.output_options.strides,
        use_evd: cfg.phase_linking.use_evd,
        beta: cfg.phase_linking.beta,
        zero_correlation_threshold: cfg.phase_linking.zero_correlation_threshold,
        output_reference_idx: cfg.phase_linking.output_reference_idx.unwrap_or(0),
        compressed_slc_plan: cfg.phase_linking.compressed_slc_plan,
        compute_crlb: cfg.phase_linking.write_crlb
            || cfg.timeseries_options.use_coherence_weights
            || cfg.timeseries_options.write_velocity_uncertainty,
        // The phase-bias correction consumes the closure layer, so force it on
        // when the correction is enabled even if the raster isn't written.
        compute_closure_phase: cfg.phase_linking.write_closure_phase
            || cfg.phase_linking.correct_phase_bias,
        compute_average_coherence: cfg.phase_linking.calc_average_coh,
        shp_method: cfg.phase_linking.shp_method,
        shp_alpha: cfg.phase_linking.shp_alpha,
    }
}

/// Build the interferogram index pairs from the config and real baselines.
fn network(cfg: &DisplacementWorkflow, days: &[f64]) -> Vec<(usize, usize)> {
    let configured = &cfg.interferogram_network;
    // An entirely unconfigured network means single-reference on date 0, not "no
    // interferograms" — parity with pinned dolphin v0.35.0's
    // `InterferogramNetwork._check_zero_parameters`. Without the fallback a bare
    // config produces zero pairs and the run fails where dolphin's would succeed.
    // v0.42.0 moved this fallback to nearest-3 (`max_bandwidth = 3`), an
    // output-changing default change tracked as issue #25 / PLAYBOOK §Elevated
    // questions; dolphinRust holds the pinned behavior until that is decided.
    let unconfigured = configured.reference_idx.is_none()
        && configured.max_bandwidth.is_none()
        && configured.max_temporal_baseline.is_none()
        && configured.indexes.is_none();
    let net = NetworkConfig {
        reference_idx: match unconfigured {
            true => Some(0),
            false => configured.reference_idx,
        },
        max_bandwidth: configured.max_bandwidth,
        max_temporal_baseline: configured.max_temporal_baseline,
        indexes: configured.indexes.clone(),
    };
    build_network(days.len(), days, &net)
}

/// Unwrap the interferogram network with the configured backend (dispatched
/// through the [`UnwrapBackend`] trait — a 3D spatiotemporal solver can drop in
/// as a new backend without changing this code).
fn unwrap_network(
    cfg: &DisplacementWorkflow,
    pl: ArrayView3<Cf64>,
    pairs: &[(usize, usize)],
    temporal_coherence: ArrayView2<f64>,
    geotransform: [f64; 6],
    epsg: Option<u32>,
) -> Result<UnwrapNetworkOutput> {
    let (_, rows, cols) = pl.dim();
    let scratch = cfg.work_directory.join("scratch");
    std::fs::create_dir_all(&scratch)?;
    let correlation =
        analysis_correlation(cfg, temporal_coherence, geotransform, epsg, (rows, cols))?;
    let masked_phase = (cfg.unwrap_options.zero_where_masked && cfg.mask_file.is_some())
        .then(|| apply_phase_mask(pl, correlation.view()));
    let backend = unwrap_backend(cfg, (rows, cols));
    // Bound network unwrap concurrency: N concurrent SNAPHU processes + N scratch
    // sets. Pinning the pool caps peak memory and keeps the block-tiled RSS win.
    let pool = unwrap_pool(cfg.unwrap_options.n_parallel_jobs)?;
    match masked_phase.as_ref() {
        Some(values) => pool
            .install(|| backend.unwrap_network(values.view(), pairs, correlation.view(), &scratch)),
        None => pool.install(|| backend.unwrap_network(pl, pairs, correlation.view(), &scratch)),
    }
}

fn apply_phase_mask(pl: ArrayView3<Cf64>, mask: ArrayView2<f32>) -> Array3<Cf64> {
    let mut values = pl.to_owned();
    for ((row, col), &validity) in mask.indexed_iter() {
        if validity == 0.0 {
            values.slice_mut(s![.., row, col]).fill(Cf64::new(0.0, 0.0));
        }
    }
    values
}

fn analysis_correlation(
    cfg: &DisplacementWorkflow,
    temporal_coherence: ArrayView2<f64>,
    geotransform: [f64; 6],
    epsg: Option<u32>,
    shape: (usize, usize),
) -> Result<Array2<f32>> {
    anyhow::ensure!(
        temporal_coherence.dim() == shape,
        "temporal coherence shape differs from unwrap grid"
    );
    let mut correlation = temporal_coherence.mapv(|value| {
        if value.is_finite() {
            value.clamp(0.0, 1.0) as f32
        } else {
            0.0
        }
    });
    if !cfg.unwrap_options.zero_where_masked {
        return Ok(correlation);
    }
    let Some(path) = cfg.mask_file.as_ref() else {
        return Ok(correlation);
    };
    let epsg = epsg.context("mask_file requires a sourced output EPSG")?;
    let mask = read_aligned_raster_window::<u8>(path, geotransform, epsg, shape)
        .context("reading configured aligned mask")?;
    for ((row, col), value) in mask.indexed_iter() {
        if *value == 0 {
            correlation[(row, col)] = 0.0;
        }
    }
    Ok(correlation)
}

/// Rayon pool sizing the ifg-network unwrap fan-out. `n_parallel_jobs` is
/// dolphin's knob: `<= 0` means all available cores, else clamp to the core count.
fn unwrap_pool(n_parallel_jobs: i64) -> Result<rayon::ThreadPool> {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let n = match n_parallel_jobs {
        j if j <= 0 => avail,
        j => (j as usize).min(avail),
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .context("building unwrap thread pool")
}

/// Build the unwrap backend from the config: tophu when selected, else SNAPHU.
/// `grid` is the unwrap grid `(rows, cols)`, used only for opt-in auto-tiling.
fn unwrap_backend(cfg: &DisplacementWorkflow, grid: (usize, usize)) -> Box<dyn UnwrapBackend> {
    match cfg.unwrap_options.unwrap_method {
        UnwrapMethod::Tophu => Box::new(TophuBackend(tophu_config(cfg))),
        UnwrapMethod::Native => Box::new(NativeUnwrapBackend(native_config(cfg, grid))),
        _ => Box::new(SnaphuBackend(unwrap_config(cfg, grid))),
    }
}

/// Map the config to the native unwrapper. Native auto-tiles *finely* by default
/// (`native_tiling`): unlike SNAPHU, its per-tile network simplex is superlinear
/// in residues-per-tile, so small tiles slash CPU·s (~8x at 1024^2) with no
/// accuracy loss (the per-region seam reconciliation holds). An explicit
/// `snaphu_options.ntiles` override still wins; conncomp masking uses the
/// `NativeConfig` defaults.
fn native_config(cfg: &DisplacementWorkflow, grid: (usize, usize)) -> NativeConfig {
    let snaphu = &cfg.unwrap_options.snaphu_options;
    let tile = match snaphu.ntiles {
        (1, 1) => native_tiling(grid),
        ntiles => Some(ntiles),
    };
    NativeConfig {
        cost: cost_mode(&snaphu.cost),
        tile,
        ..NativeConfig::default()
    }
}

/// Native auto-tiling: keep every core at least `TARGET_TILE` pixels per axis.
/// The former 48-pixel floor was the microbenchmark throughput optimum, but the
/// MMX1 common-frame live contract exposed unstable seam-graph branches at that
/// granularity (2.90-11.73% cycle disagreement). A 64-pixel floor holds the
/// shipped <=0.5% SNAPHU-parity bar while retaining fine-grained MCF solves.
/// Grids below `2 * TARGET_TILE` per axis stay untiled.
fn native_tiling((rows, cols): (usize, usize)) -> Option<(usize, usize)> {
    const TARGET_TILE: usize = 64;
    let per_axis = |n: usize| (n / TARGET_TILE).max(1);
    let tiles = (per_axis(rows), per_axis(cols));
    (tiles != (1, 1)).then_some(tiles)
}

/// Map the config's SNAPHU options to the unwrap wrapper config. When
/// `auto_tile` is set, `ntiles`/`nproc` are derived from the grid + cores
/// (opt-in; changes numerics), otherwise the explicit config values are used.
fn unwrap_config(cfg: &DisplacementWorkflow, grid: (usize, usize)) -> UnwrapConfig {
    let snaphu = &cfg.unwrap_options.snaphu_options;
    let (ntiles, nproc) = match snaphu.auto_tile {
        true => auto_tiling(grid),
        false => (snaphu.ntiles, snaphu.n_parallel_tiles),
    };
    UnwrapConfig {
        cost: cost_mode(&snaphu.cost),
        init: init_method(&snaphu.init_method),
        ntiles,
        tile_overlap: snaphu.tile_overlap,
        nproc,
        snaphu_path: "snaphu".to_string(),
    }
}

/// Conservative auto-tiling: split a large grid so each tile stays `>= MIN_TILE`
/// pixels per side, capping the tile count per axis at the core count, and run
/// the tiles in parallel (`nproc = ntiles_row * ntiles_col`). Grids smaller than
/// `2 * MIN_TILE` on an axis are left untiled, so small scenes are unchanged.
fn auto_tiling((rows, cols): (usize, usize)) -> ((usize, usize), usize) {
    const MIN_TILE: usize = 512;
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let per_axis = |n: usize| (n / MIN_TILE).clamp(1, avail);
    let ntiles = (per_axis(rows), per_axis(cols));
    (ntiles, ntiles.0 * ntiles.1)
}

/// Map the config's tophu options to the multi-scale driver config. dolphin's
/// `TophuOptions` carries no tile overlap; we add a fixed halo (clamped per tile)
/// so the fine pass has boundary context for the 2π-reconciled merge.
fn tophu_config(cfg: &DisplacementWorkflow) -> TophuConfig {
    let t = &cfg.unwrap_options.tophu_options;
    TophuConfig {
        downsample_factor: t.downsample_factor,
        ntiles: t.ntiles,
        tile_overlap: TophuConfig::default().tile_overlap,
        cost: cost_mode(&t.cost),
        init: init_method(&t.init_method),
        snaphu_path: "snaphu".to_string(),
    }
}

/// SNAPHU cost mode from the config string (`defo` → deformation, else smooth).
fn cost_mode(cost: &str) -> CostMode {
    match cost {
        "defo" => CostMode::Defo,
        _ => CostMode::Smooth,
    }
}

/// SNAPHU init method from the config string (`mst` → MST, else MCF).
fn init_method(init: &str) -> InitMethod {
    match init {
        "mst" => InitMethod::Mst,
        _ => InitMethod::Mcf,
    }
}

/// LOS-phase (rad) → displacement (mm) factor `-λ/4π · 1000`, falling back to
/// the Sentinel-1 wavelength when the config supplies none.
fn mm_per_rad(wavelength: Option<f64>) -> f64 {
    -wavelength.unwrap_or(SENTINEL1_WAVELENGTH_M) / (4.0 * std::f64::consts::PI) * 1000.0
}

/// Linear velocity (rad/yr) from the phase displacement series, fitting against
/// the real acquisition `days` (date 0 = 0 reference).
fn velocity_of(displacement: ArrayView3<f64>, days: &[f64]) -> Array2<f64> {
    let series = series_with_reference(displacement);
    estimate_velocity(days, series.view(), None)
}

fn series_with_reference(displacement: ArrayView3<f64>) -> Array3<f64> {
    let (nd, rows, cols) = displacement.dim();
    Array3::from_shape_fn((nd + 1, rows, cols), |(t, r, c)| match t {
        0 => 0.0,
        _ => displacement[(t - 1, r, c)],
    })
}

/// Propagate an independent-pixel approximation through spatial referencing.
/// The selected reference is identically zero; every other pixel receives the
/// reference pixel's temporal variance in quadrature.
fn reference_variance_to_point(variance: &mut Array3<f64>, point: (usize, usize)) {
    let reference: Vec<_> = variance
        .axis_iter(Axis(0))
        .map(|band| band[point])
        .collect();
    for ((date, row, col), value) in variance.indexed_iter_mut() {
        if (row, col) == point {
            *value = 0.0;
        } else {
            *value += reference[date];
        }
    }
}

/// Pixels whose CRLB is usable on every date, i.e. that have a bound at all.
///
/// A singular `Γ` yields a NaN bound — correct, and matched to dolphin v0.42 by
/// `quality_v042_contract`. But a missing bound is missing *information*, not
/// evidence the data is bad, so such a pixel weights uniformly rather than
/// collapsing to zero weight (issue #34).
fn uncertainty_valid(sigma: ArrayView3<f64>) -> Array2<bool> {
    let (dates, rows, cols) = sigma.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        (0..dates).all(|date| sigma[(date, r, c)].is_finite())
    })
}

/// Uniform weight for a pixel with no usable bound. For a single-reference
/// network the SBAS system is exactly determined, so weights cancel and this is
/// *identical* to the weighted solution; it is only a real estimator change for
/// an over-determined network (`max_bandwidth` set).
const UNIFORM_PRECISION: f64 = 1.0;

fn interferogram_precisions(
    sigma: ArrayView3<f64>,
    pairs: &[(usize, usize)],
    valid: ArrayView2<bool>,
) -> Array3<f64> {
    let (_, rows, cols) = sigma.dim();
    Array3::from_shape_fn((pairs.len(), rows, cols), |(k, r, c)| {
        let (i, j) = pairs[k];
        let variance = sigma[(i, r, c)].powi(2) + sigma[(j, r, c)].powi(2);
        match valid[(r, c)] && variance.is_finite() {
            true => 1.0 / variance.max(1e-12),
            false => UNIFORM_PRECISION,
        }
    })
}

fn date_precisions(sigma: ArrayView3<f64>, valid: ArrayView2<bool>) -> Array3<f64> {
    let (dates, rows, cols) = sigma.dim();
    Array3::from_shape_fn((dates, rows, cols), |(date, r, c)| {
        let variance = sigma[(date, r, c)] * sigma[(date, r, c)];
        match valid[(r, c)] && variance.is_finite() {
            true => 1.0 / variance.max(1e-12),
            false => UNIFORM_PRECISION,
        }
    })
}

/// Blank an uncertainty layer wherever the pixel has no usable bound. Uniform
/// weights recover the displacement; they do not manufacture an uncertainty, so
/// the derived σ must still read as absent.
fn clear_unbounded_uncertainty(layer: &mut Array2<f64>, valid: ArrayView2<bool>) {
    clear_unbounded_uncertainty_2d(&mut layer.view_mut(), valid);
}

/// [`clear_unbounded_uncertainty`] over a borrowed 2-D view (one band of a cube).
fn clear_unbounded_uncertainty_2d(
    layer: &mut ndarray::ArrayViewMut2<f64>,
    valid: ArrayView2<bool>,
) {
    ndarray::Zip::from(layer)
        .and(valid)
        .for_each(|value, &usable| {
            if !usable {
                *value = f64::NAN;
            }
        });
}

/// Write the velocity, temporal-coherence, per-date displacement, and (when
/// enabled) per-band CRLB σ + closure-phase rasters as GeoTIFFs, all sharing the
/// resolved geotransform + EPSG.
fn write_outputs(
    cfg: &DisplacementWorkflow,
    displacement: ArrayView3<f64>,
    velocity: ArrayView2<f64>,
    temporal_coherence: ArrayView2<f64>,
    quality: QualityLayers,
    epsg: Option<u32>,
    gt: [f64; 6],
) -> Result<()> {
    let dir = &cfg.work_directory;
    std::fs::create_dir_all(dir)?;
    let write_f32 = |name: &str, a: ArrayView2<f64>| {
        write_raster(&dir.join(name), a.mapv(|v| v as f32).view(), gt, epsg, None)
    };
    // Issue #36: which layer is *the* uncertainty is not a matter of taste, it
    // depends on whether the interferogram system has residual degrees of freedom.
    // Say so in the raster rather than leaving it to the reader.
    let (scale, scale_note) = uncertainty_scale(quality.posterior_dof);
    let dof_text = quality.posterior_dof.to_string();
    let posterior_tags = [
        ("UNCERTAINTY_SCALE", scale),
        ("POSTERIOR_DOF", dof_text.as_str()),
        ("DESCRIPTION", scale_note),
    ];
    write_f32("velocity.tif", velocity)?;
    if let Some(sigma) = quality.velocity_sigma {
        write_raster_with_metadata(
            &dir.join("velocity_sigma.tif"),
            sigma.mapv(|v| v as f32).view(),
            gt,
            epsg,
            None,
            &posterior_tags,
        )?;
    }
    if let Some(residual) = quality.timeseries_residual_rms {
        write_f32("timeseries_residual_rms.tif", residual.view())?;
    }
    if let Some(misclosure) = quality.network_misclosure_rms {
        write_f32("network_misclosure_rms.tif", misclosure.view())?;
    }
    write_f32("temporal_coherence.tif", temporal_coherence)?;
    if let Some(coherence) = quality.phase_linking_coherence {
        write_f32("phase_linking_coherence.tif", coherence.view())?;
    }
    write_bands(&write_f32, displacement, "displacement")?;
    if let Some(crlb) = quality.crlb_sigma {
        for band in 0..crlb.dim().0 {
            write_raster_with_metadata(
                &dir.join(format!("crlb_sigma_{band:02}.tif")),
                crlb.index_axis(Axis(0), band).mapv(|v| v as f32).view(),
                gt,
                epsg,
                None,
                &[
                    ("UNITTYPE", "rad"),
                    ("UNCERTAINTY_SCALE", "crlb_bound"),
                    ("DESCRIPTION", CRLB_BOUND_NOTE),
                ],
            )?;
        }
    }
    if let Some(closure) = quality.closure_phase {
        write_bands(&write_f32, closure.view(), "closure_phase")?;
    }
    if let Some(amplitude) = quality.velocity_terms.seasonal_amplitude {
        write_f32("velocity_seasonal_amplitude.tif", amplitude.view())?;
    }
    if let Some(phase) = quality.velocity_terms.seasonal_phase_days {
        write_raster_with_metadata(
            &dir.join("velocity_seasonal_phase_days.tif"),
            phase.mapv(|v| v as f32).view(),
            gt,
            epsg,
            None,
            &[("UNITTYPE", "days")],
        )?;
    }
    for (k, step) in quality.velocity_terms.step_magnitude.iter().enumerate() {
        write_f32(&format!("velocity_step_{k:02}.tif"), step.view())?;
    }
    if let Some(qc) = quality.loop_closure {
        write_f32("loop_closure_bad_count.tif", qc.bad_loop_count.view())?;
        write_f32(
            "loop_closure_worst_cycles.tif",
            qc.worst_residual_cycles.view(),
        )?;
    }
    if let Some(variance) = quality.displacement_variance {
        for band in 0..variance.dim().0 {
            write_raster_with_metadata(
                &dir.join(format!("displacement_variance_{band:02}.tif")),
                variance.index_axis(Axis(0), band).mapv(|v| v as f32).view(),
                gt,
                epsg,
                None,
                &posterior_tags,
            )?;
        }
    }
    for band in 0..quality.connected_components.dim().0 {
        write_raster(
            &dir.join(format!("conncomp_{band:02}.tif")),
            quality.connected_components.index_axis(Axis(0), band),
            gt,
            epsg,
            Some(0.0),
        )?;
    }
    Ok(())
}

/// The first burst's input files in date order (the dates the series is built on),
/// used to time-stamp frame corrections. For a multi-burst mosaic this retains
/// the reference burst timing approximation. Mirrors how `days` is taken from the first
/// burst; `groups` is a `BTreeMap`, so `.values().next()` is the first burst.
fn first_burst_files(
    cfg: &DisplacementWorkflow,
    groups: &std::collections::BTreeMap<String, Vec<usize>>,
) -> Vec<PathBuf> {
    groups
        .values()
        .next()
        .map(|idxs| {
            idxs.iter()
                .map(|&i| cfg.cslc_file_list[i].clone())
                .collect()
        })
        .unwrap_or_default()
}

/// Write the per-date correction-delay layers (meters) as `{kind}_NN.tif` COGs.
fn write_correction_outputs(
    cfg: &DisplacementWorkflow,
    corrections: &CorrectionLayers,
    epsg: Option<u32>,
    gt: [f64; 6],
) -> Result<()> {
    let dir = &cfg.work_directory;
    let write_f32 = |name: &str, a: ArrayView2<f64>| {
        write_raster(&dir.join(name), a.mapv(|v| v as f32).view(), gt, epsg, None)
    };
    if let Some(iono) = &corrections.ionosphere {
        write_bands(&write_f32, iono.view(), "ionosphere")?;
    }
    if let Some(tropo) = &corrections.troposphere {
        write_bands(&write_f32, tropo.view(), "troposphere")?;
    }
    if let Some(tide) = &corrections.solid_earth_tide {
        write_bands(&write_f32, tide.view(), "solid_earth_tide")?;
    }
    Ok(())
}

/// In-band note on `crlb_sigma_NN.tif`. The bound is correct and matches dolphin;
/// what it is not is a predictive sigma, and a consumer reading the band without
/// this would have no way to know (#36).
const CRLB_BOUND_NOTE: &str =
    "Cramer-Rao LOWER BOUND on the phase-linking sigma, not a predictive uncertainty. \
     It under-covers the true spread by construction. Do not publish it as the \
     uncertainty layer.";

/// In-band note on the posterior layers when the interferogram system is exactly
/// determined, so their scale is not empirical.
const POSTERIOR_BOUND_NOTE: &str =
    "dof=0: the interferogram system is exactly determined (single-reference network), so \
     the residual-based inflation is pinned to 1 and this sigma traces entirely to the \
     CRLB lower bound. It is NOT an empirically scaled uncertainty. Set \
     interferogram_network.max_bandwidth for a posterior scaled by the data.";

/// In-band note on the posterior layers when they do carry empirical scale.
const POSTERIOR_EMPIRICAL_NOTE: &str =
    "Posterior sigma scaled by the fit residuals (dof>0). This is the layer to carry as \
     the uncertainty product.";

/// Where an uncertainty layer's scale comes from, given the posterior degrees of
/// freedom `n_interferograms - n_unknowns`.
fn uncertainty_scale(posterior_dof: usize) -> (&'static str, &'static str) {
    match posterior_dof {
        0 => ("crlb_bound", POSTERIOR_BOUND_NOTE),
        _ => ("empirical", POSTERIOR_EMPIRICAL_NOTE),
    }
}

/// The optional per-pixel quality layers written alongside displacement.
struct QualityLayers<'a> {
    /// Residual degrees of freedom of the SBAS solve,
    /// `n_interferograms - (n_dates - 1)`. Zero on a single-reference network,
    /// which is what decides whether the posterior layers carry empirical scale.
    posterior_dof: usize,
    phase_linking_coherence: Option<&'a Array2<f64>>,
    crlb_sigma: Option<&'a Array3<f64>>,
    closure_phase: Option<&'a Array3<f64>>,
    displacement_variance: Option<&'a Array3<f64>>,
    network_misclosure_rms: Option<&'a Array2<f64>>,
    timeseries_residual_rms: Option<&'a Array2<f64>>,
    velocity_sigma: Option<&'a Array2<f64>>,
    connected_components: &'a Array3<u32>,
    /// Seasonal amplitude in displacement units, peak day, and per-step
    /// magnitudes — present only when the time-function model is configured.
    velocity_terms: VelocityTermLayers<'a>,
    /// Post-unwrap loop-closure QC, present only when the gate ran.
    loop_closure: Option<&'a LoopClosureQc>,
}

/// Emitted form of [`VelocityTerms`], already scaled to displacement units.
#[derive(Default)]
struct VelocityTermLayers<'a> {
    seasonal_amplitude: Option<&'a Array2<f64>>,
    seasonal_phase_days: Option<&'a Array2<f64>>,
    step_magnitude: &'a [Array2<f64>],
}

/// Write each band of a `(bands, rows, cols)` layer as `{prefix}_NN.tif`.
fn write_bands(
    write_f32: &impl Fn(&str, ArrayView2<f64>) -> dolphin_io::Result<()>,
    layer: ArrayView3<f64>,
    prefix: &str,
) -> Result<()> {
    for t in 0..layer.dim().0 {
        let band = layer.index_axis(ndarray::Axis(0), t);
        write_f32(&format!("{prefix}_{t:02}.tif"), band)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dolphin_core::config::ComputeBackend;
    use dolphin_core::{HalfWindow, Strides};

    /// A config whose velocity path is the unweighted degree-1 fit — the branch
    /// the bounded-trim reference-reselection tests exercise.
    fn unweighted_cfg(correlation_threshold: f64) -> DisplacementWorkflow {
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.correlation_threshold = correlation_threshold;
        cfg.timeseries_options.use_coherence_weights = false;
        cfg.timeseries_options.write_velocity_uncertainty = false;
        cfg.timeseries_options.correct_velocity_temporal_correlation = false;
        cfg
    }

    fn seam_burst(phase_offset: f64, coherence: f64) -> BurstLink {
        BurstLink {
            pl: Array3::from_shape_fn((2, 3, 3), |(date, _, _)| {
                Cf64::from_polar(1.0, date as f64 * 0.2 + phase_offset)
            }),
            temp_coh: Array2::from_elem((3, 3), coherence),
            phase_linking_coherence: None,
            crlb_sigma: None,
            closure_phase: None,
            validity_mask: Array2::from_elem((3, 3), true),
            coverage: BurstCoverageProvenance {
                burst_index: 0,
                acquisition_count: 2,
                total_tiles: 1,
                linked_tiles: 1,
                nodata_tiles: 0,
            },
            geo: BurstGeo {
                geo: GeoInfo {
                    epsg: 32611,
                    geotransform: [0.0, 30.0, 0.0, 90.0, 0.0, -30.0],
                },
                rows: 3,
                cols: 3,
            },
            days: vec![0.0, 12.0],
        }
    }

    #[test]
    fn multiburst_leveling_removes_injected_phase_offset() {
        let frame = Array3::from_shape_fn((2, 3, 3), |(date, _, _)| {
            Cf64::from_polar(1.0, date as f64 * 0.2)
        });
        let coherence = Array2::from_elem((3, 3), 0.9);
        let covered = Array2::from_elem((3, 3), true);
        let mut burst = seam_burst(0.7, 0.9);
        level_burst_offsets(&frame, &coherence, &covered, &mut burst, (0, 0), 1).unwrap();
        for (actual, expected) in burst.pl.iter().zip(frame.iter()) {
            assert!((*actual - *expected).norm() < 1e-12);
        }
    }

    #[test]
    fn multiburst_leveling_fails_typed_when_stable_support_is_insufficient() {
        let frame = Array3::from_elem((2, 3, 3), Cf64::new(1.0, 0.0));
        let coherence = Array2::from_elem((3, 3), 0.9);
        let covered = Array2::from_elem((3, 3), true);
        let mut burst = seam_burst(0.7, 0.1);
        let error =
            level_burst_offsets(&frame, &coherence, &covered, &mut burst, (0, 0), 1).unwrap_err();
        assert!(matches!(
            error,
            StitchError::InsufficientOffsetSupport { support: 0, .. }
        ));
    }

    #[test]
    fn multiburst_leveling_skips_nodata_but_uses_remaining_overlap() {
        let mut frame = Array3::from_shape_fn((2, 3, 3), |(date, _, _)| {
            Cf64::from_polar(1.0, date as f64 * 0.2)
        });
        frame[(0, 0, 0)] = Cf64::new(f64::NAN, f64::NAN);
        let coherence = Array2::from_elem((3, 3), 0.9);
        let covered = Array2::from_elem((3, 3), true);
        let mut burst = seam_burst(-0.4, 0.9);
        burst.pl[(1, 0, 1)] = Cf64::new(0.0, 0.0);
        level_burst_offsets(&frame, &coherence, &covered, &mut burst, (0, 0), 1).unwrap();
        assert!((burst.pl[(0, 1, 1)] - frame[(0, 1, 1)]).norm() < 1e-12);
        assert!((burst.pl[(1, 1, 1)] - frame[(1, 1, 1)]).norm() < 1e-12);
    }

    #[test]
    fn multiburst_stitch_does_not_overwrite_finite_overlap_with_nodata() {
        let first = seam_burst(0.0, 0.9);
        let expected = first.pl[(0, 0, 0)];
        let mut second = seam_burst(0.4, 0.9);
        second.pl[(0, 0, 0)] = Cf64::new(f64::NAN, f64::NAN);
        second.temp_coh[(0, 0)] = f64::NAN;
        second.validity_mask[(0, 0)] = false;
        second.coverage.burst_index = 1;
        let stitched = stitch_bursts(vec![first, second]).unwrap();
        assert_eq!(stitched.pl[(0, 0, 0)], expected);
        assert!(stitched.temp_coh[(0, 0)].is_finite());
        assert!(stitched.validity_mask[(0, 0)]);
    }
    #[test]
    fn proc_status_memory_parser_is_bounded_and_path_free() {
        let status = "Name:\tdolphin\nVmHWM:\t  65432 kB\nVmRSS:\t  54321 kB\n";
        assert_eq!(parse_memory_kib(status), (54_321, 65_432));
        assert_eq!(parse_memory_kib("Name:\tdolphin\n"), (0, 0));
    }

    #[test]
    fn bounded_trim_keeps_corrections_and_static_geometry_aligned() {
        let values = Array3::from_shape_fn((2, 6, 8), |(t, r, c)| (t * 100 + r * 10 + c) as f64);
        let geometry = LosGeometry {
            east: Array2::from_shape_fn((6, 8), |(r, c)| (r * 10 + c) as f64),
            north: Array2::from_shape_fn((6, 8), |(r, c)| -(r as f64 * 10.0 + c as f64)),
            up: Array2::from_elem((6, 8), 0.8),
        };
        let mut corrections = CorrectionLayers {
            ionosphere: Some(values.clone()),
            troposphere: Some(values),
            solid_earth_tide: None,
            los_geometry: Some(geometry),
        };
        let target = BlockIndices {
            row_start: 2,
            row_stop: 5,
            col_start: 3,
            col_stop: 7,
        };
        trim_corrections(&mut corrections, target);
        let ionosphere = corrections.ionosphere.unwrap();
        let los = corrections.los_geometry.unwrap();
        assert_eq!(ionosphere.dim(), (2, 3, 4));
        assert_eq!(los.east.dim(), (3, 4));
        assert_eq!(ionosphere[(1, 0, 0)], 123.0);
        assert_eq!(los.east[(0, 0)], 23.0);
        assert_eq!(los.north[(2, 3)], -46.0);
    }

    #[test]
    fn validity_mask_propagates_nodata_through_every_output_layer() {
        let mut validity_mask = Array2::from_elem((2, 2), true);
        validity_mask[(0, 0)] = false;
        let mut velocity = Array2::from_elem((2, 2), 1.0);
        velocity[(1, 1)] = f64::NAN;
        let mut products = SpatialProducts {
            disp_rad: Array3::from_elem((2, 2, 2), 1.0),
            vel_rad: velocity,
            velocity_model: VelocityModel::default(),
            velocity_terms: VelocityTerms::default(),
            loop_closure: None,
            temporal_coherence: Array2::from_elem((2, 2), 1.0),
            validity_mask,
            burst_coverage: Vec::new(),
            phase_linking_coherence: Some(Array2::from_elem((2, 2), 1.0)),
            crlb_sigma: Some(Array3::from_elem((2, 2, 2), 1.0)),
            closure_phase: Some(Array3::from_elem((1, 2, 2), 1.0)),
            corrections: CorrectionLayers {
                ionosphere: Some(Array3::from_elem((2, 2, 2), 1.0)),
                troposphere: Some(Array3::from_elem((2, 2, 2), 1.0)),
                solid_earth_tide: None,
                los_geometry: Some(LosGeometry {
                    east: Array2::from_elem((2, 2), 1.0),
                    north: Array2::from_elem((2, 2), 1.0),
                    up: Array2::from_elem((2, 2), 1.0),
                }),
            },
            geotransform: [0.0; 6],
            reference_point: None,
            posterior_variance_rad: Some(Array3::from_elem((2, 2, 2), 1.0)),
            network_misclosure_rad: Some(Array2::from_elem((2, 2), 1.0)),
            timeseries_residual_rad: Some(Array2::from_elem((2, 2), 1.0)),
            velocity_sigma_rad: Some(Array2::from_elem((2, 2), 1.0)),
            interferogram_pairs: Vec::new(),
            unwrap_connected_components: Array3::from_elem((2, 2, 2), 1),
        };

        products.apply_validity_mask();

        for (row, col) in [(0, 0), (1, 1)] {
            assert!(!products.validity_mask[(row, col)]);
            assert!(products.vel_rad[(row, col)].is_nan());
            assert!(products.temporal_coherence[(row, col)].is_nan());
            assert!(products.phase_linking_coherence.as_ref().unwrap()[(row, col)].is_nan());
            assert!(products.network_misclosure_rad.as_ref().unwrap()[(row, col)].is_nan());
            assert!(products.timeseries_residual_rad.as_ref().unwrap()[(row, col)].is_nan());
            assert!(products.velocity_sigma_rad.as_ref().unwrap()[(row, col)].is_nan());
            for band in 0..2 {
                assert!(products.disp_rad[(band, row, col)].is_nan());
                assert!(products.crlb_sigma.as_ref().unwrap()[(band, row, col)].is_nan());
                assert!(
                    products.posterior_variance_rad.as_ref().unwrap()[(band, row, col)].is_nan()
                );
                assert!(
                    products.corrections.ionosphere.as_ref().unwrap()[(band, row, col)].is_nan()
                );
                assert!(
                    products.corrections.troposphere.as_ref().unwrap()[(band, row, col)].is_nan()
                );
                assert_eq!(products.unwrap_connected_components[(band, row, col)], 0);
            }
            assert!(products.closure_phase.as_ref().unwrap()[(0, row, col)].is_nan());
            let geometry = products.corrections.los_geometry.as_ref().unwrap();
            assert!(geometry.east[(row, col)].is_nan());
            assert!(geometry.north[(row, col)].is_nan());
            assert!(geometry.up[(row, col)].is_nan());
        }
        assert!(products.validity_mask[(0, 1)]);
        assert_eq!(products.vel_rad[(0, 1)], 1.0);
    }

    #[test]
    fn bounded_trim_reselects_a_target_valid_reference_when_original_is_in_halo() {
        let mut products = SpatialProducts {
            disp_rad: Array3::from_shape_fn((2, 6, 8), |(date, row, col)| {
                date as f64 + row as f64 * 0.1 + col as f64 * 0.01
            }),
            vel_rad: Array2::zeros((6, 8)),
            velocity_model: VelocityModel::default(),
            velocity_terms: VelocityTerms::default(),
            loop_closure: None,
            temporal_coherence: Array2::from_elem((6, 8), 0.9),
            validity_mask: Array2::from_elem((6, 8), true),
            burst_coverage: Vec::new(),
            phase_linking_coherence: None,
            crlb_sigma: None,
            closure_phase: None,
            corrections: CorrectionLayers {
                ionosphere: None,
                troposphere: None,
                solid_earth_tide: None,
                los_geometry: None,
            },
            geotransform: [0.0, 30.0, 0.0, 180.0, 0.0, -30.0],
            reference_point: Some((0, 0)),
            posterior_variance_rad: None,
            network_misclosure_rad: None,
            timeseries_residual_rad: None,
            velocity_sigma_rad: None,
            interferogram_pairs: Vec::new(),
            unwrap_connected_components: Array3::zeros((0, 6, 8)),
        };
        let target = BlockIndices {
            row_start: 2,
            row_stop: 5,
            col_start: 3,
            col_stop: 7,
        };
        products
            .trim(target, &[0.0, 12.0, 24.0], &unweighted_cfg(0.5))
            .unwrap();
        let reference = products.reference_point.expect("target reference");
        assert!(reference.0 < 3 && reference.1 < 4);
        assert!(products
            .disp_rad
            .slice(s![.., reference.0, reference.1])
            .iter()
            .all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn bounded_trim_rejects_low_quality_target_when_reference_is_in_halo() {
        let mut products = SpatialProducts {
            disp_rad: Array3::zeros((2, 4, 4)),
            vel_rad: Array2::zeros((4, 4)),
            velocity_model: VelocityModel::default(),
            velocity_terms: VelocityTerms::default(),
            loop_closure: None,
            temporal_coherence: Array2::from_elem((4, 4), 0.1),
            validity_mask: Array2::from_elem((4, 4), true),
            burst_coverage: Vec::new(),
            phase_linking_coherence: None,
            crlb_sigma: None,
            closure_phase: None,
            corrections: CorrectionLayers {
                ionosphere: None,
                troposphere: None,
                solid_earth_tide: None,
                los_geometry: None,
            },
            geotransform: [0.0, 30.0, 0.0, 120.0, 0.0, -30.0],
            reference_point: Some((0, 0)),
            posterior_variance_rad: None,
            network_misclosure_rad: None,
            timeseries_residual_rad: None,
            velocity_sigma_rad: None,
            interferogram_pairs: Vec::new(),
            unwrap_connected_components: Array3::zeros((0, 4, 4)),
        };
        let error = products
            .trim(
                BlockIndices {
                    row_start: 1,
                    row_stop: 4,
                    col_start: 1,
                    col_stop: 4,
                },
                &[0.0, 12.0, 24.0],
                &unweighted_cfg(0.5),
            )
            .unwrap_err();
        assert!(error.to_string().contains("reference coherence threshold"));
    }

    #[test]
    fn configured_reference_translates_from_full_frame_to_analysis() {
        let plan = BoundedPlan {
            windows: Vec::new(),
            target_in_analysis: BlockIndices {
                row_start: 2,
                row_stop: 8,
                col_start: 3,
                col_stop: 9,
            },
            provenance: crate::crop::ProcessingBoundsProvenance {
                processing_method: crate::crop::AOI_PROCESSING_METHOD.into(),
                processing_method_version: crate::crop::AOI_PROCESSING_VERSION.into(),
                requested_target_bounds: [0.0; 4],
                requested_bounds_epsg: 32611,
                actual_output_bounds: [0.0; 4],
                actual_analysis_bounds: [0.0; 4],
                actual_read_bounds: [0.0; 4],
                output_epsg: 32611,
                target_pixel_offset: [12, 23],
                analysis_pixel_offset: [10, 20],
                analysis_halo_pixels: [2, 3],
                halo_policy_version: crate::crop::HALO_POLICY_VERSION.into(),
                native_reads: Vec::new(),
            },
        };
        assert_eq!(
            configured_analysis_reference(Some((14, 25)), Some(&plan), (10, 12)).unwrap(),
            Some((4, 5))
        );
        assert!(configured_analysis_reference(Some((9, 25)), Some(&plan), (10, 12)).is_err());
    }

    #[test]
    fn aligned_mask_crs_mismatch_fails_explicitly() {
        let path = std::env::temp_dir().join("dolphin_bounds_wrong_crs_mask.tif");
        let mask = Array2::from_elem((8, 8), 1_u8);
        write_raster(
            &path,
            mask.view(),
            [0.0, 30.0, 0.0, 240.0, 0.0, -30.0],
            Some(32610),
            Some(0.0),
        )
        .unwrap();
        let mut cfg = DisplacementWorkflow {
            mask_file: Some(path),
            ..Default::default()
        };
        cfg.unwrap_options.zero_where_masked = true;
        let error = analysis_correlation(
            &cfg,
            Array2::ones((8, 8)).view(),
            [0.0, 30.0, 0.0, 240.0, 0.0, -30.0],
            Some(32611),
            (8, 8),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("differs from target EPSG"));
    }

    #[test]
    fn mask_is_not_read_when_zero_where_masked_is_false() {
        let cfg = DisplacementWorkflow {
            mask_file: Some(std::env::temp_dir().join("does-not-exist.tif")),
            ..Default::default()
        };
        let correlation = analysis_correlation(
            &cfg,
            Array2::ones((8, 8)).view(),
            [0.0, 30.0, 0.0, 240.0, 0.0, -30.0],
            Some(32611),
            (8, 8),
        )
        .unwrap();
        assert!(correlation.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn unwrap_correlation_preserves_real_temporal_quality() {
        let cfg = DisplacementWorkflow::default();
        let temporal = ndarray::array![[0.1, 0.8], [f64::NAN, 1.2]];
        let correlation = analysis_correlation(
            &cfg,
            temporal.view(),
            [0.0, 30.0, 0.0, 60.0, 0.0, -30.0],
            Some(32611),
            (2, 2),
        )
        .unwrap();
        assert_eq!(correlation, ndarray::array![[0.1_f32, 0.8], [0.0, 1.0]]);
    }

    #[test]
    fn posterior_uncertainty_supports_unweighted_l2_and_rejects_l1() {
        let incidence = ndarray::array![[1.0], [1.0]];
        let dphi = Array3::from_shape_vec((2, 1, 1), vec![1.0, 2.0]).unwrap();
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.method = TimeseriesMethod::L2;
        cfg.timeseries_options.use_coherence_weights = false;
        cfg.timeseries_options.write_posterior_uncertainty = true;
        let output = invert_time_series(&cfg, incidence.view(), dphi.view(), None, &[(0, 1)])
            .expect("unweighted posterior");
        assert!(output.posterior_variance.is_some());
        cfg.timeseries_options.method = TimeseriesMethod::L1;
        let error = invert_time_series(&cfg, incidence.view(), dphi.view(), None, &[(0, 1)])
            .err()
            .expect("L1 must reject posterior output");
        assert!(error.to_string().contains("only for L2"));
    }

    /// Issue #40: the SBAS network-inversion misclosure and the temporal
    /// motion-model fit residual are different physical quantities and must not
    /// share one field. A redundant network whose interferograms are perfectly
    /// consistent (zero misclosure) can still carry a phase history a linear
    /// velocity model fits badly — proving the two residuals move independently.
    #[test]
    fn network_misclosure_and_temporal_fit_residual_are_decoupled() {
        // True per-date phase, referenced to date 0: a sharp late jump breaks the
        // linear velocity model even though every interferogram in this
        // over-determined network is exactly consistent with it.
        let days = [0.0, 12.0, 24.0, 36.0];
        let true_phi = [0.0_f64, 1.0, 1.2, 5.0];
        let pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)];
        let incidence = get_incidence_matrix(&pairs);
        let dphi = Array3::from_shape_fn((pairs.len(), 1, 1), |(k, _, _)| {
            let (a, b) = pairs[k];
            true_phi[b] - true_phi[a]
        });

        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.method = TimeseriesMethod::L2;
        cfg.timeseries_options.use_coherence_weights = false;
        cfg.timeseries_options.write_posterior_uncertainty = true;
        let inversion = invert_time_series(&cfg, incidence.view(), dphi.view(), None, &pairs)
            .expect("redundant, well-conditioned network");
        let misclosure = inversion
            .network_misclosure_rms
            .as_ref()
            .expect("posterior uncertainty computes the network misclosure")[(0, 0)];
        assert!(
            misclosure < 1e-9,
            "a perfectly consistent network must show ~0 misclosure, got {misclosure}"
        );

        // The same true phase history as a velocity-fit input: date 0 is the
        // dropped reference (implicit zero), dates 1..3 are the fitted bands.
        let series = Array3::from_shape_fn((3, 1, 1), |(d, _, _)| true_phi[d + 1]);
        cfg.timeseries_options.write_velocity_uncertainty = true;
        let crlb = Array3::from_elem((4, 1, 1), 1.0);
        let fit = fit_velocity(
            &cfg,
            series.view(),
            &days,
            Some(&crlb),
            &VelocityModel::default(),
        )
        .unwrap();
        let temporal_residual = fit.residual_rms.expect("weighted fit reports a residual")[(0, 0)];
        assert!(
            temporal_residual > 0.5,
            "a late jump must leave the linear model a large residual, got {temporal_residual}"
        );
    }

    /// The correction changes only the emitted uncertainty product, so enabling
    /// it without that product must fail rather than silently doing nothing.
    #[test]
    fn temporal_correlation_requires_velocity_uncertainty_output() {
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.correct_velocity_temporal_correlation = true;

        let error = validate_uncertainty_options(&cfg).unwrap_err();
        assert!(error
            .to_string()
            .contains("requires write_velocity_uncertainty"));

        cfg.timeseries_options.write_velocity_uncertainty = true;
        validate_uncertainty_options(&cfg).expect("valid uncertainty options");
    }

    /// Issue #33: velocity sigma understates the slope uncertainty because the
    /// residuals are temporally correlated. Opt-in AR(1) inflation, off by
    /// default so no downstream threshold moves unreviewed.
    #[test]
    fn temporal_correlation_inflates_velocity_sigma_only_when_enabled() {
        let days: Vec<f64> = (0..12).map(|t| f64::from(t) * 12.0).collect();
        // A drifting (positively autocorrelated) departure from the trend.
        let displacement = Array3::from_shape_fn((11, 1, 1), |(t, _, _)| {
            let time = days[t + 1];
            0.01 * time + 4.0 * (f64::from(t as i32) * 0.45).sin()
        });
        let crlb = Array3::from_elem((12, 1, 1), 0.5);
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.write_velocity_uncertainty = true;
        let linear = VelocityModel::default();

        let plain = fit_velocity(&cfg, displacement.view(), &days, Some(&crlb), &linear)
            .unwrap()
            .sigma;
        cfg.timeseries_options.correct_velocity_temporal_correlation = true;
        let corrected = fit_velocity(&cfg, displacement.view(), &days, Some(&crlb), &linear)
            .unwrap()
            .sigma;

        let plain = plain.expect("velocity sigma")[(0, 0)];
        let corrected = corrected.expect("velocity sigma")[(0, 0)];
        assert!(plain.is_finite() && corrected.is_finite());
        assert!(
            corrected > plain,
            "correlated residuals must inflate sigma: {corrected} !> {plain}"
        );
    }

    /// The correction is a no-op on uncorrelated residuals — it must not inflate
    /// sigma just because it is switched on.
    #[test]
    fn temporal_correlation_correction_is_a_no_op_without_correlation() {
        let days: Vec<f64> = (0..12).map(|t| f64::from(t) * 12.0).collect();
        // Exactly linear: zero residual, so no autocorrelation to find.
        let displacement = Array3::from_shape_fn((11, 1, 1), |(t, _, _)| 0.01 * days[t + 1]);
        let crlb = Array3::from_elem((12, 1, 1), 0.5);
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.write_velocity_uncertainty = true;
        let linear = VelocityModel::default();

        let plain = fit_velocity(&cfg, displacement.view(), &days, Some(&crlb), &linear)
            .unwrap()
            .sigma;
        cfg.timeseries_options.correct_velocity_temporal_correlation = true;
        let corrected = fit_velocity(&cfg, displacement.view(), &days, Some(&crlb), &linear)
            .unwrap()
            .sigma;
        let plain = plain.expect("velocity sigma")[(0, 0)];
        let corrected = corrected.expect("velocity sigma")[(0, 0)];
        assert!(
            (corrected - plain).abs() < 1e-12,
            "uncorrelated residuals must not inflate sigma: {corrected} vs {plain}"
        );
    }

    /// Issue #34: a NaN CRLB is a missing *bound*, not evidence the data is bad.
    /// dolphin v0.42 NaNs a singular block deliberately (matched by
    /// `quality_v042_contract`), so the bound is right — but mapping it to a zero
    /// weight makes the normal equations singular and destroys the displacement
    /// too. Such a pixel falls back to uniform weights instead.
    #[test]
    fn a_pixel_with_no_usable_bound_falls_back_to_uniform_weights() {
        // Two pixels, two dates; the second pixel's bound is missing on one date.
        let mut sigma = Array3::from_elem((2, 1, 2), 2.0);
        sigma[(1, 0, 1)] = f64::NAN;
        let valid = uncertainty_valid(sigma.view());
        assert_eq!(valid, ndarray::array![[true, false]]);

        let precision = date_precisions(sigma.view(), valid.view());
        assert!(
            precision
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "no weight may be zero or non-finite: {precision:?}"
        );
        // The bounded pixel keeps 1/sigma^2; the unbounded one goes uniform.
        assert!((precision[(0, 0, 0)] - 0.25).abs() < 1e-12);
        assert!((precision[(0, 0, 1)] - 1.0).abs() < 1e-12);
        assert!((precision[(1, 0, 1)] - 1.0).abs() < 1e-12);

        let pairs = [(0, 1)];
        let ifg = interferogram_precisions(sigma.view(), &pairs, valid.view());
        assert!(
            ifg.iter().all(|value| value.is_finite() && *value > 0.0),
            "no interferogram weight may be zero or non-finite: {ifg:?}"
        );
        assert!((ifg[(0, 0, 1)] - 1.0).abs() < 1e-12);
    }

    /// The bound must still be reported as absent — falling back to uniform
    /// weights recovers the displacement, it does not manufacture an uncertainty.
    #[test]
    fn an_unbounded_pixel_reports_no_uncertainty() {
        let mut sigma = Array3::from_elem((2, 1, 2), 2.0);
        sigma[(1, 0, 1)] = f64::NAN;
        let valid = uncertainty_valid(sigma.view());
        let mut velocity_sigma = ndarray::array![[0.5_f64, 0.5]];
        clear_unbounded_uncertainty(&mut velocity_sigma, valid.view());
        assert!(velocity_sigma[(0, 0)].is_finite());
        assert!(
            velocity_sigma[(0, 1)].is_nan(),
            "a pixel without a usable CRLB must not report a velocity sigma"
        );
    }

    #[test]
    fn spatial_reference_variance_includes_reference_pixel() {
        let mut variance = Array3::from_shape_vec((1, 1, 3), vec![1.0, 4.0, 9.0]).unwrap();
        reference_variance_to_point(&mut variance, (0, 1));
        assert_eq!(
            variance,
            Array3::from_shape_vec((1, 1, 3), vec![5.0, 0.0, 13.0]).unwrap()
        );
    }

    #[test]
    fn enabled_mask_zeros_linked_phase_before_wrapped_interferograms() {
        let phase = Array3::from_elem((2, 2, 2), Cf64::new(1.0, 1.0));
        let mask = ndarray::array![[1.0_f32, 0.0], [1.0, 1.0]];
        let masked = apply_phase_mask(phase.view(), mask.view());
        assert_eq!(masked[(0, 0, 1)], Cf64::new(0.0, 0.0));
        assert_eq!(masked[(1, 0, 1)], Cf64::new(0.0, 0.0));
        assert_eq!(masked[(1, 1, 1)], Cf64::new(1.0, 1.0));
    }

    #[test]
    fn native_tiling_keeps_mmx1_common_frame_above_stable_core_floor() {
        assert_eq!(
            native_tiling((352, 2217)),
            Some((5, 34)),
            "MMX1 live parity fails with the old 7x46 approximately 48px cores"
        );
    }

    /// A deterministic complex stack with spatial + temporal structure, so the
    /// coherence estimate is non-degenerate and tile boundaries actually matter.
    fn synth_stack(nslc: usize, rows: usize, cols: usize) -> Array3<Cf64> {
        Array3::from_shape_fn((nslc, rows, cols), |(t, r, c)| {
            let phase = 0.20 * t as f64 * (c as f64 / cols as f64)
                + 0.05 * r as f64
                + 0.30 * ((r * 7 + c * 3 + t) % 5) as f64;
            let amp = 1.0 + 0.1 * ((r + c + t) % 3) as f64;
            Cf64::from_polar(amp, phase)
        })
    }

    /// Config exercising both quality layers, with a small block so the burst
    /// tiles into several interior + edge tiles in both axes.
    fn tiled_cfg(
        strides: Strides,
        half: HalfWindow,
        block: (usize, usize),
    ) -> DisplacementWorkflow {
        let mut cfg = DisplacementWorkflow::default();
        cfg.phase_linking.ministack_size = 4;
        cfg.phase_linking.half_window = half;
        cfg.phase_linking.write_crlb = true;
        cfg.phase_linking.write_closure_phase = true;
        cfg.phase_linking.calc_average_coh = true;
        cfg.output_options.strides = strides;
        cfg.worker_settings.block_shape = block;
        cfg.worker_settings.compute_backend = ComputeBackend::Cpu;
        cfg
    }

    fn assert_c64_eq(a: ArrayView3<Cf64>, b: ArrayView3<Cf64>, what: &str) {
        assert_eq!(a.dim(), b.dim(), "{what}: shape");
        let (_, nr, nc) = a.dim();
        let mut diffs = 0;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x == y {
                continue;
            }
            let (band, r, c) = (i / (nr * nc), (i / nc) % nr, i % nc);
            if diffs < 12 {
                eprintln!("{what} @ band {band} ({r},{c}): {x} != {y}");
            }
            diffs += 1;
        }
        assert_eq!(diffs, 0, "{what}: {diffs} differing elements");
    }

    fn assert_f64_eq(a: ArrayView3<f64>, b: ArrayView3<f64>, what: &str) {
        assert_eq!(a.dim(), b.dim(), "{what}: shape");
        let bit = |v: f64| v.to_bits();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(x == y || bit(*x) == bit(*y), "{what}: {x} != {y}");
        }
    }

    /// Config exercising both quality layers. `block` is small so the burst
    /// tiles into many interior + edge seams in both axes.
    fn run_case(
        nslc: usize,
        dims: (usize, usize),
        strides: Strides,
        half: HalfWindow,
        block: (usize, usize),
    ) {
        let cfg = tiled_cfg(strides, half, block);
        let engine = ComputeEngine::new(ComputeBackend::Cpu);
        let stack = synth_stack(nslc, dims.0, dims.1);
        let whole = phase_link(&cfg, stack.view(), &engine).unwrap();
        let tiled = phase_link_tiled(&cfg, dims, nslc, &engine, |b| {
            Ok(stack.slice(s![.., b.rows(), b.cols()]).to_owned())
        })
        .unwrap();
        assert_eq!(tiled.nodata_tiles, 0);
        assert_eq!(tiled.linked_tiles, tiled.total_tiles);
        assert!(tiled.validity_mask.iter().all(|valid| *valid));
        let tiled = tiled.output;
        assert_c64_eq(tiled.cpx_phase.view(), whole.cpx_phase.view(), "cpx_phase");
        assert_f64_eq(
            tiled.temporal_coherence.view().insert_axis(Axis(0)),
            whole.temporal_coherence.view().insert_axis(Axis(0)),
            "temporal_coherence",
        );
        assert_f64_eq(
            tiled
                .phase_linking_coherence
                .as_ref()
                .unwrap()
                .view()
                .insert_axis(Axis(0)),
            whole
                .phase_linking_coherence
                .as_ref()
                .unwrap()
                .view()
                .insert_axis(Axis(0)),
            "phase_linking_coherence",
        );
        assert_f64_eq(
            tiled.crlb_sigma.as_ref().unwrap().view(),
            whole.crlb_sigma.as_ref().unwrap().view(),
            "crlb_sigma",
        );
        assert_f64_eq(
            tiled.closure_phase.as_ref().unwrap().view(),
            whole.closure_phase.as_ref().unwrap().view(),
            "closure_phase",
        );
    }

    #[test]
    fn locally_empty_edge_tile_becomes_nodata_without_aborting_burst() {
        use std::cell::Cell;

        let dims = (17, 19);
        let nslc = 5;
        let cfg = tiled_cfg(Strides { y: 1, x: 1 }, HalfWindow { y: 1, x: 1 }, (8, 8));
        let engine = ComputeEngine::new(ComputeBackend::Cpu);
        let stack = synth_stack(nslc, dims.0, dims.1);
        let calls = Cell::new(0_usize);
        let tiled = phase_link_tiled(&cfg, dims, nslc, &engine, |block| {
            let mut tile = stack.slice(s![.., block.rows(), block.cols()]).to_owned();
            if calls.replace(calls.get() + 1) == 0 {
                tile.index_axis_mut(Axis(0), 2)
                    .fill(Cf64::new(f64::NAN, f64::NAN));
            }
            Ok(tile)
        })
        .unwrap();
        assert_eq!(tiled.nodata_tiles, 1);
        assert!(tiled.linked_tiles > 0);
        assert!(tiled.validity_mask.iter().any(|valid| *valid));
        assert!(tiled.validity_mask.iter().any(|valid| !*valid));
        for ((_, row, col), value) in tiled.output.cpx_phase.indexed_iter() {
            if !tiled.validity_mask[(row, col)] {
                assert!(value.re.is_nan() && value.im.is_nan());
            }
        }
    }

    #[test]
    fn globally_empty_acquisition_reports_safe_ordinal() {
        let dims = (17, 19);
        let nslc = 5;
        let cfg = tiled_cfg(Strides { y: 1, x: 1 }, HalfWindow { y: 1, x: 1 }, (8, 8));
        let engine = ComputeEngine::new(ComputeBackend::Cpu);
        let stack = synth_stack(nslc, dims.0, dims.1);
        let error = phase_link_tiled(&cfg, dims, nslc, &engine, |block| {
            let mut tile = stack.slice(s![.., block.rows(), block.cols()]).to_owned();
            tile.index_axis_mut(Axis(0), 3)
                .fill(Cf64::new(f64::NAN, f64::NAN));
            Ok(tile)
        })
        .err()
        .expect("globally empty acquisition must fail");
        assert!(error.to_string().contains("ordinals [3]"));
    }

    #[test]
    fn no_tile_with_complete_temporal_support_fails_explicitly() {
        use std::cell::Cell;

        let dims = (17, 19);
        let nslc = 5;
        let cfg = tiled_cfg(Strides { y: 1, x: 1 }, HalfWindow { y: 1, x: 1 }, (8, 8));
        let engine = ComputeEngine::new(ComputeBackend::Cpu);
        let stack = synth_stack(nslc, dims.0, dims.1);
        let calls = Cell::new(0_usize);
        let error = phase_link_tiled(&cfg, dims, nslc, &engine, |block| {
            let mut tile = stack.slice(s![.., block.rows(), block.cols()]).to_owned();
            let ordinal = calls.replace(calls.get() + 1) % 2;
            tile.index_axis_mut(Axis(0), ordinal)
                .fill(Cf64::new(f64::NAN, f64::NAN));
            Ok(tile)
        })
        .err()
        .expect("no complete tile must fail");
        assert!(error
            .to_string()
            .contains("no tile with complete temporal support"));
    }

    /// Contract (the load-bearing one): block-tiled phase linking is BIT-IDENTICAL
    /// to a whole-burst run for every output layer, including the clamped raster
    /// border. The halo/trim math makes tiling a pure refactor; any drift is an
    /// indexing bug, not tolerance. Stressed across strides (border margin sizes),
    /// ministack depth (compressed-SLC dependency cone), and tiny blocks (many
    /// seams).
    #[test]
    fn tiled_phase_link_is_bit_identical_to_whole_burst() {
        // ministack_size is 4 (see tiled_cfg); nslc spans 1..=4 ministacks.
        run_case(
            6,
            (40, 50),
            Strides { y: 1, x: 1 },
            HalfWindow { y: 3, x: 4 },
            (16, 16),
        );
        run_case(
            8,
            (90, 110),
            Strides { y: 2, x: 2 },
            HalfWindow { y: 3, x: 5 },
            (12, 12),
        );
        run_case(
            10,
            (90, 110),
            Strides { y: 1, x: 1 },
            HalfWindow { y: 4, x: 6 },
            (20, 20),
        );
        run_case(
            14,
            (96, 96),
            Strides { y: 3, x: 3 },
            HalfWindow { y: 2, x: 4 },
            (18, 18),
        );
    }

    #[test]
    fn atmospheric_correction_preserves_reference_and_known_velocity() {
        for wavelength in [SENTINEL1_WAVELENGTH_M, 0.238403545] {
            let dir =
                std::env::temp_dir().join(format!("dolphin_reference_atmosphere_{wavelength}"));
            std::fs::create_dir_all(&dir).unwrap();
            let gt = [500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0];
            let mut cfg = DisplacementWorkflow::default();
            cfg.input_options.wavelength = Some(wavelength);
            cfg.correction_options.incidence_angle_deg = 0.0;
            cfg.correction_options.troposphere_variable = "Band1".into();
            let days = [0.0, 12.0, 24.0];
            for t in 0..3 {
                let path = dir.join(format!("delay_{t}.nc"));
                let raster = dir.join(format!("delay_{t}.tif"));
                let delay =
                    Array2::from_shape_fn((3, 3), |(_, c)| 0.01 * t as f64 * (1.0 + c as f64));
                write_raster(&raster, delay.view(), gt, Some(32611), None).unwrap();
                let src = gdal::Dataset::open(&raster).unwrap();
                src.create_copy(
                    &gdal::DriverManager::get_driver_by_name("netCDF").unwrap(),
                    &path,
                    &Default::default(),
                )
                .unwrap();
                cfg.correction_options.troposphere_files.push(path);
            }
            let scale = -4.0 * std::f64::consts::PI / wavelength;
            let mut disp = Array3::from_shape_fn((2, 3, 3), |(t, _, c)| {
                (0.008 * c as f64 * days[t + 1] / 365.25 + 0.01 * (t + 1) as f64 * (1.0 + c as f64))
                    * scale
            });
            correct_and_reference(
                &cfg,
                &mut disp,
                &[],
                GeoInfo {
                    epsg: 32611,
                    geotransform: gt,
                },
                Some((1, 0)),
            )
            .unwrap();
            assert!(disp
                .slice(ndarray::s![.., 1, 0])
                .iter()
                .all(|v| v.abs() < 1e-12));
            let fit = fit_velocity(
                &cfg,
                disp.view(),
                &days,
                Some(&Array3::from_elem((3, 3, 3), 1.0)),
                &VelocityModel::default(),
            )
            .unwrap();
            assert!((fit.velocity[(1, 2)] / scale - 0.016).abs() < 1e-8);
            assert!(fit.residual_rms.unwrap().iter().all(|r| r.abs() < 1e-8));
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    /// Contract: a noise-free phase series carrying a known LOS rate is recovered
    /// as exactly that rate in mm/yr, using the real temporal baselines — not the
    /// old hardcoded 12-day cadence. Exercises `velocity_of` + `mm_per_rad`, the
    /// two pieces the pipeline composes for `velocity_mm_yr`.
    #[test]
    fn recovers_injected_rate_in_mm_per_yr() {
        let wavelength = SENTINEL1_WAVELENGTH_M; // explicit S1 config
        let injected_mm_yr = -8.0; // subsidence, LOS
                                   // disp(t) [m] = rate * (days/365.25); phase = disp * (-4π/λ).
        let days = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0];
        let phase_per_m = -4.0 * std::f64::consts::PI / wavelength;
        let rate_m_yr = injected_mm_yr / 1000.0;
        // displacement-series bands are dates 1..n (date 0 is the implicit zero ref).
        let bands: Vec<f64> = days[1..]
            .iter()
            .map(|&d| rate_m_yr * (d / 365.25) * phase_per_m)
            .collect();
        let disp = Array3::from_shape_fn((bands.len(), 1, 1), |(t, _, _)| bands[t]);

        let vel_rad = velocity_of(disp.view(), &days);
        let got_mm_yr = vel_rad[(0, 0)] * mm_per_rad(Some(wavelength));
        assert!(
            (got_mm_yr - injected_mm_yr).abs() < 1e-6,
            "recovered {got_mm_yr} mm/yr, injected {injected_mm_yr}"
        );
    }

    /// NISAR L-band center wavelength (m): c / 1.2575 GHz ≈ 0.2384.
    const NISAR_WAVELENGTH_M: f64 = 0.238_403_545;

    /// Contract (DoD #3): the velocity→mm/yr scaling uses the configured NISAR
    /// L-band λ, not the S1 default. A known LOS rate is recovered only when
    /// `mm_per_rad` is fed the NISAR wavelength; feeding the S1 default mis-scales
    /// it by the λ ratio (≈4.3×).
    #[test]
    fn velocity_uses_nisar_wavelength() {
        let injected_mm_yr = -8.0; // subsidence, LOS
        let days = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0];
        let phase_per_m = -4.0 * std::f64::consts::PI / NISAR_WAVELENGTH_M;
        let rate_m_yr = injected_mm_yr / 1000.0;
        let bands: Vec<f64> = days[1..]
            .iter()
            .map(|&d| rate_m_yr * (d / 365.25) * phase_per_m)
            .collect();
        let disp = Array3::from_shape_fn((bands.len(), 1, 1), |(t, _, _)| bands[t]);
        let vel_rad = velocity_of(disp.view(), &days);

        let got_nisar = vel_rad[(0, 0)] * mm_per_rad(Some(NISAR_WAVELENGTH_M));
        assert!(
            (got_nisar - injected_mm_yr).abs() < 1e-6,
            "NISAR λ recovers {injected_mm_yr}, got {got_nisar}"
        );
        // mm_per_rad ∝ λ, so the S1 default mis-scales by λ_S1 / λ_NISAR ≈ 0.23×.
        let got_s1_default = vel_rad[(0, 0)] * mm_per_rad(None);
        let ratio = SENTINEL1_WAVELENGTH_M / NISAR_WAVELENGTH_M;
        assert!(
            (got_s1_default / got_nisar - ratio).abs() < 1e-6,
            "S1-default scaling differs from NISAR by the λ ratio"
        );
    }

    /// The old bug: assuming a 12-day cadence on a non-12-day stack mis-scales the
    /// rate by the cadence ratio. Real baselines must make the result cadence-free.
    #[test]
    fn rate_is_independent_of_cadence() {
        let phase_per_yr = 5.0; // arbitrary rad/yr
        let mk = |days: &[f64]| {
            let bands: Vec<f64> = days[1..]
                .iter()
                .map(|&d| phase_per_yr * d / 365.25)
                .collect();
            let disp = Array3::from_shape_fn((bands.len(), 1, 1), |(t, _, _)| bands[t]);
            velocity_of(disp.view(), days)[(0, 0)]
        };
        let v12 = mk(&[0.0, 12.0, 24.0, 36.0]);
        let v6 = mk(&[0.0, 6.0, 12.0, 18.0]);
        assert!((v12 - phase_per_yr).abs() < 1e-9);
        assert!((v6 - phase_per_yr).abs() < 1e-9);
    }

    /// Issue #36: the emitted rasters must say where their scale comes from. On a
    /// single-reference network `dof = 0`, the residual inflation is inert, and
    /// every uncertainty layer traces to the CRLB bound — so the bands are tagged
    /// `crlb_bound`, not left for a consumer to misread as a predictive sigma.
    #[test]
    fn uncertainty_layers_declare_their_scale_in_band() {
        let dir = std::env::temp_dir().join("dolphin_uncertainty_scale_tags");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = DisplacementWorkflow {
            work_directory: dir.clone(),
            ..Default::default()
        };
        let gt = [0.0, 30.0, 0.0, 120.0, 0.0, -30.0];
        let displacement = Array3::<f64>::zeros((2, 2, 2));
        let variance = Array3::from_elem((2, 2, 2), 4.0);
        let sigma = Array2::from_elem((2, 2), 2.0);
        let crlb = Array3::from_elem((2, 2, 2), 1.0);
        let conncomp = Array3::<u32>::zeros((0, 2, 2));

        let write = |dof: usize| {
            write_outputs(
                &cfg,
                displacement.view(),
                Array2::zeros((2, 2)).view(),
                Array2::from_elem((2, 2), 0.9).view(),
                QualityLayers {
                    posterior_dof: dof,
                    phase_linking_coherence: None,
                    crlb_sigma: Some(&crlb),
                    closure_phase: None,
                    displacement_variance: Some(&variance),
                    network_misclosure_rms: None,
                    timeseries_residual_rms: None,
                    velocity_sigma: Some(&sigma),
                    connected_components: &conncomp,
                    velocity_terms: VelocityTermLayers::default(),
                    loop_closure: None,
                },
                Some(32614),
                gt,
            )
            .unwrap();
        };
        let tag = |name: &str, key: &str| {
            use gdal::Metadata;
            gdal::Dataset::open(dir.join(name))
                .unwrap()
                .metadata_item(key, "")
                .unwrap_or_default()
        };

        write(0);
        assert_eq!(tag("velocity_sigma.tif", "UNCERTAINTY_SCALE"), "crlb_bound");
        assert_eq!(tag("velocity_sigma.tif", "POSTERIOR_DOF"), "0");
        assert_eq!(
            tag("displacement_variance_00.tif", "UNCERTAINTY_SCALE"),
            "crlb_bound"
        );
        assert!(tag("velocity_sigma.tif", "DESCRIPTION").contains("NOT an empirically scaled"));
        // The bound is always a bound, whatever the network.
        assert_eq!(tag("crlb_sigma_00.tif", "UNCERTAINTY_SCALE"), "crlb_bound");
        assert!(tag("crlb_sigma_00.tif", "DESCRIPTION").contains("LOWER BOUND"));
        assert_eq!(tag("crlb_sigma_00.tif", "UNITTYPE"), "rad");

        // An over-determined network is the case where the posterior earns the name.
        write(3);
        assert_eq!(tag("velocity_sigma.tif", "UNCERTAINTY_SCALE"), "empirical");
        assert_eq!(tag("velocity_sigma.tif", "POSTERIOR_DOF"), "3");
        assert_eq!(
            tag("displacement_variance_01.tif", "UNCERTAINTY_SCALE"),
            "empirical"
        );
        assert_eq!(
            tag("crlb_sigma_00.tif", "UNCERTAINTY_SCALE"),
            "crlb_bound",
            "the CRLB is a bound regardless of the network"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Issue #25, found by diffing dolphin v0.35.0's and v0.42.0's config
    /// defaults: an unconfigured `interferogram_network` produced **zero pairs**
    /// and failed the run, where dolphin falls back to a network. dolphinRust
    /// holds the pinned v0.35.0 behavior — single-reference on date 0.
    #[test]
    fn unconfigured_network_falls_back_to_single_reference() {
        let cfg = DisplacementWorkflow::default();
        assert_eq!(cfg.interferogram_network, Default::default());
        let pairs = network(&cfg, &[0.0, 12.0, 24.0, 36.0]);
        assert_eq!(pairs, vec![(0, 1), (0, 2), (0, 3)]);
    }

    /// The fallback must not fire when any option is set — a config asking for
    /// nearest-2 must not silently also get the single-reference pairs.
    #[test]
    fn configured_network_does_not_get_the_fallback() {
        let mut cfg = DisplacementWorkflow::default();
        cfg.interferogram_network.max_bandwidth = Some(2);
        let pairs = network(&cfg, &[0.0, 12.0, 24.0, 36.0]);
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    }

    /// Issue #25: `run_interpolation` round-trips from a dolphin YAML and already
    /// widens the AOI halo, but no interpolation stage exists — so it must be
    /// rejected, not silently ignored.
    #[test]
    fn run_interpolation_is_rejected_as_unimplemented() {
        let mut cfg = DisplacementWorkflow::default();
        assert!(
            validate_uncertainty_options(&cfg).is_ok(),
            "dolphin's own default (false) must pass"
        );
        cfg.unwrap_options.run_interpolation = true;
        let error = validate_uncertainty_options(&cfg).unwrap_err();
        assert!(error.to_string().contains("run_interpolation"), "{error}");
    }

    /// Issue #24: the gate is off by default, so the unwrapped stack reaches the
    /// solve untouched.
    #[test]
    fn loop_closure_gate_is_off_by_default() {
        let cfg = DisplacementWorkflow::default();
        let pairs = vec![(0, 1), (1, 2), (0, 2)];
        let mut dphi = Array3::from_shape_fn((3, 2, 2), |(k, _, _)| k as f64);
        let original = dphi.clone();
        assert!(apply_loop_closure_qc(&cfg, &mut dphi, &pairs).is_none());
        assert_eq!(dphi, original);
    }

    /// Enabled on a single-reference network it is a no-op with a warning, not an
    /// error: there are no loops to close, and that is a configuration mismatch
    /// rather than bad data.
    #[test]
    fn loop_closure_gate_is_a_no_op_without_loops() {
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.mask_unwrap_loop_errors = true;
        let pairs = vec![(0, 1), (0, 2), (0, 3)];
        let mut dphi = Array3::from_shape_fn((3, 2, 2), |(k, _, _)| k as f64);
        let original = dphi.clone();
        assert!(apply_loop_closure_qc(&cfg, &mut dphi, &pairs).is_none());
        assert_eq!(dphi, original);
    }

    /// Enabled on a network with loops, a 2π error is masked out of every
    /// interferogram at that pixel before the solve sees it.
    #[test]
    fn loop_closure_gate_masks_a_cycle_error() {
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.mask_unwrap_loop_errors = true;
        let pairs = vec![(0, 1), (0, 2), (1, 2)];
        let phase = [0.0, 1.3, 2.9];
        let mut dphi = Array3::from_shape_fn((pairs.len(), 2, 2), |(k, _, _)| {
            let (i, j) = pairs[k];
            phase[j] - phase[i]
        });
        dphi[(2, 1, 1)] += std::f64::consts::TAU;

        let qc = apply_loop_closure_qc(&cfg, &mut dphi, &pairs).expect("qc ran");
        assert!(qc.bad_loop_count[(1, 1)] > 0.0);
        assert!(dphi.slice(s![.., 1, 1]).iter().all(|v| v.is_nan()));
        assert!(dphi.slice(s![.., 0, 0]).iter().all(|v| v.is_finite()));
    }

    fn dated_files(dates: &[&str]) -> Vec<PathBuf> {
        dates
            .iter()
            .map(|d| PathBuf::from(format!("/x/cslc_{d}.h5")))
            .collect()
    }

    /// Issue #22, default path: with neither knob set the model is linear, so the
    /// velocity fit stays on the parity-critical degree-1 estimator and no
    /// time-function layer is produced.
    #[test]
    fn default_config_leaves_the_velocity_model_linear() {
        let cfg = DisplacementWorkflow::default();
        let files = dated_files(&["20230104", "20230116"]);
        let model = velocity_model(&cfg, &files).unwrap();
        assert!(model.is_linear());

        let days = vec![0.0, 12.0];
        let displacement = Array3::from_shape_fn((1, 1, 1), |_| 0.5);
        let crlb = Array3::from_elem((2, 1, 1), 0.5);
        let fit = fit_velocity(&cfg, displacement.view(), &days, Some(&crlb), &model).unwrap();
        assert!(fit.terms.seasonal_amplitude_rad.is_none());
        assert!(fit.terms.step_magnitude_rad.is_empty());
    }

    /// Step dates are resolved against acquisition 0, the same origin `days` uses.
    #[test]
    fn step_dates_resolve_to_days_from_acquisition_zero() {
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.velocity_step_dates = vec!["2023-01-04".into(), "2023-03-05".into()];
        let files = dated_files(&["20230104", "20230609"]);
        let model = velocity_model(&cfg, &files).unwrap();
        assert!(!model.is_linear());
        assert_eq!(model.step_days, vec![0.0, 60.0]);
    }

    /// A step the user asked for and did not get is a wrong answer, not a degraded
    /// one — a malformed date fails the run instead of dropping the term.
    #[test]
    fn malformed_step_date_fails_the_run() {
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.velocity_step_dates = vec!["04/01/2023".into()];
        let error = velocity_model(&cfg, &dated_files(&["20230104"])).unwrap_err();
        assert!(error.to_string().contains("velocity_step_dates"), "{error}");
    }

    /// The configured model reaches `fit_velocity`, separates the seasonal cycle
    /// from the rate, and emits its terms — where the linear fit on the same series
    /// reports a rate biased by the cycle.
    #[test]
    fn seasonal_model_separates_the_cycle_from_the_rate() {
        let days: Vec<f64> = (0..16).map(|t| f64::from(t) * 12.0).collect();
        let (rate_per_year, amplitude) = (5.0, 2.0);
        let omega = std::f64::consts::TAU / 365.25;
        // `fit_velocity` prepends the zero reference epoch, so the stack carries
        // dates 1..n and must evaluate to zero at day 0 — which this series does.
        let displacement = Array3::from_shape_fn((days.len() - 1, 1, 1), |(t, _, _)| {
            let time = days[t + 1];
            rate_per_year * time / 365.25 + amplitude * ((omega * time).cos() - 1.0)
        });
        let crlb = Array3::from_elem((days.len(), 1, 1), 0.5);
        let mut cfg = DisplacementWorkflow::default();
        cfg.timeseries_options.write_velocity_uncertainty = true;
        cfg.timeseries_options.velocity_seasonal = true;
        let model = velocity_model(&cfg, &dated_files(&["20230104"])).unwrap();

        let seasonal = fit_velocity(&cfg, displacement.view(), &days, Some(&crlb), &model).unwrap();
        assert!(
            (seasonal.velocity[(0, 0)] - rate_per_year).abs() < 1e-6,
            "rate {} != {rate_per_year}",
            seasonal.velocity[(0, 0)]
        );
        let recovered = seasonal.terms.seasonal_amplitude_rad.expect("amplitude")[(0, 0)];
        assert!(
            (recovered - amplitude).abs() < 1e-6,
            "amplitude {recovered}"
        );
        assert!(
            seasonal.sigma.is_some(),
            "sigma follows write_velocity_uncertainty"
        );

        cfg.timeseries_options.velocity_seasonal = false;
        let linear = fit_velocity(
            &cfg,
            displacement.view(),
            &days,
            Some(&crlb),
            &VelocityModel::default(),
        )
        .unwrap();
        assert!(
            (linear.velocity[(0, 0)] - rate_per_year).abs() > 1.0,
            "fixture must show the linear fit absorbing the cycle, got {}",
            linear.velocity[(0, 0)]
        );
    }

    /// The bounded/tiled path re-fits through the same front door, so a configured
    /// model reaches it too — a step is recovered after re-referencing, not lost.
    #[test]
    fn bounded_trim_refits_with_the_configured_model() {
        let days: Vec<f64> = (0..12).map(|t| f64::from(t) * 12.0).collect();
        let step_day = 60.0;
        let mut products = SpatialProducts {
            // Signal scales with row so re-referencing (which subtracts the
            // reference pixel's whole series) leaves a recoverable row-to-row
            // difference rather than a flat zero.
            disp_rad: Array3::from_shape_fn((days.len() - 1, 4, 4), |(t, row, _)| {
                let time = days[t + 1];
                let scale = row as f64 + 1.0;
                (0.01 * time + f64::from(time >= step_day) * 3.0) * scale
            }),
            vel_rad: Array2::zeros((4, 4)),
            velocity_model: VelocityModel {
                seasonal: false,
                step_days: vec![step_day],
            },
            loop_closure: None,
            velocity_terms: VelocityTerms::default(),
            temporal_coherence: Array2::from_elem((4, 4), 0.9),
            validity_mask: Array2::from_elem((4, 4), true),
            burst_coverage: Vec::new(),
            phase_linking_coherence: None,
            crlb_sigma: None,
            closure_phase: None,
            corrections: CorrectionLayers {
                ionosphere: None,
                troposphere: None,
                solid_earth_tide: None,
                los_geometry: None,
            },
            geotransform: [0.0, 30.0, 0.0, 120.0, 0.0, -30.0],
            // In the halo, so trim re-selects a reference and re-fits.
            reference_point: Some((0, 0)),
            posterior_variance_rad: None,
            network_misclosure_rad: None,
            timeseries_residual_rad: None,
            velocity_sigma_rad: None,
            interferogram_pairs: Vec::new(),
            unwrap_connected_components: Array3::zeros((0, 4, 4)),
        };
        let target = BlockIndices {
            row_start: 1,
            row_stop: 4,
            col_start: 1,
            col_stop: 4,
        };
        products.trim(target, &days, &unweighted_cfg(0.5)).unwrap();

        let step = products
            .velocity_terms
            .step_magnitude_rad
            .first()
            .expect("step layer");
        assert_eq!(step.dim(), (3, 3), "terms are trimmed with the rest");
        // Re-referencing removes a common offset, so the recoverable quantity is
        // the row-to-row difference: one row of scale is exactly one step of 3.0.
        assert!(
            (step[(1, 0)] - step[(0, 0)] - 3.0).abs() < 1e-9,
            "step magnitude gradient {} -> {}",
            step[(0, 0)],
            step[(1, 0)]
        );
        let rate_gradient = products.vel_rad[(1, 0)] - products.vel_rad[(0, 0)];
        assert!(
            (rate_gradient - 0.01 * 365.25).abs() < 1e-9,
            "rate gradient {rate_gradient} — the step must not leak into it"
        );
    }
}
