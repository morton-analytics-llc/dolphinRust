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
    read_nisar_geotransform, read_nisar_window, write_raster, GeoInfo,
};
use dolphin_phaselink::{correct_phase_bias, estimate_bias_velocity, ComputeEngine};
use dolphin_timeseries::{
    build_network, estimate_velocity, get_incidence_matrix, invert_stack, invert_stack_l1,
    reference_to_point, select_reference_point, L1Config, NetworkConfig,
};
use dolphin_unwrap::native::NativeConfig;
use dolphin_unwrap::{CostMode, InitMethod, TophuConfig, UnwrapConfig};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis};

use crate::burst::{burst_offset, frame_grid, group_by_burst, paste2, paste3, BurstGeo, FrameGrid};
use crate::corrections::{apply_corrections, CorrectionLayers};
use crate::crop::{plan_bounds, BoundedPlan, BurstWindow};
use crate::dates::decimal_days;
use crate::provenance::GeometryProvenance;
use crate::sequential::{
    run_sequential, run_sequential_resumable, update_sequential, SequentialConfig,
    SequentialOutput, SequentialState,
};
use crate::tiling::{plan_tiles, TilePlan};
use crate::unwrap_backend::{NativeUnwrapBackend, SnaphuBackend, TophuBackend, UnwrapBackend};
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
    /// Temporal coherence per pixel in `[0, 1]`, stitched across ministacks by
    /// NaN-aware mean (dolphin's `temporal_coherence_average` = `numpy.nanmean`);
    /// a phase-quality mask, `(rows, cols)`.
    pub temporal_coherence: Array2<f64>,
    /// Mean coherence-matrix magnitude across real acquisitions, distinct from
    /// estimator-fit temporal coherence. `None` unless `calc_average_coh` is on.
    pub phase_linking_coherence: Option<Array2<f64>>,
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
    let groups = group_by_burst(&cfg.cslc_file_list);
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
                Some(link_one_burst(cfg, idxs, &engine, window))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    finish_displacement(cfg, bursts, crop.as_ref())
}

/// Shared downstream tail: stitch bursts → ifg network → SNAPHU unwrap → SBAS
/// inversion → reference → atmospheric corrections → velocity → write COGs.
/// Identical for a full run and an incremental update — both feed it the same
/// per-burst phase-linking products, so both produce the same output.
fn finish_displacement(
    cfg: &DisplacementWorkflow,
    bursts: Vec<BurstLink>,
    crop: Option<&BoundedPlan>,
) -> Result<DisplacementOutput> {
    let groups = group_by_burst(&cfg.cslc_file_list);
    let days = bursts
        .first()
        .map(|b| b.days.clone())
        .context("cslc_file_list is empty")?;
    let stitched = timed("stitch", || stitch_bursts(bursts))?;
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
    let pairs = timed("network", || network(cfg, &days));
    anyhow::ensure!(!pairs.is_empty(), "interferogram_network produced no pairs");

    let dphi_rad = timed("unwrap", || {
        unwrap_network(cfg, pl.view(), &pairs, geotransform, epsg)
    })?;
    let incidence = get_incidence_matrix(&pairs);
    let mut disp_rad = timed("timeseries", || match cfg.timeseries_options.method {
        TimeseriesMethod::L1 => {
            invert_stack_l1(incidence.view(), dphi_rad.view(), L1Config::default())
        }
        TimeseriesMethod::L2 => invert_stack(incidence.view(), dphi_rad.view(), None),
    });
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
    if let Some(point) = analysis_reference_point {
        reference_to_point(&mut disp_rad, point);
    }
    // Atmospheric corrections subtract per-date delay from the inverted series,
    // before velocity (opt-in; no-op when no correction files are configured).
    let date_files = first_burst_files(cfg, &groups);
    let corrections = timed("corrections", || {
        apply_corrections(
            &cfg.correction_options,
            cfg.input_options.wavelength,
            &mut disp_rad,
            &date_files,
            epsg.unwrap_or(0),
            geotransform,
        )
    })?;
    let vel_rad = timed("velocity", || velocity_of(disp_rad.view(), &days));
    let spatial = SpatialProducts {
        disp_rad,
        vel_rad,
        temporal_coherence,
        phase_linking_coherence: stitched.phase_linking_coherence,
        crlb_sigma: stitched.crlb_sigma,
        closure_phase: stitched.closure_phase,
        corrections,
        geotransform,
        reference_point: analysis_reference_point,
    };
    emit_displacement(cfg, days, epsg, crop, spatial)
}

struct SpatialProducts {
    disp_rad: Array3<f64>,
    vel_rad: Array2<f64>,
    temporal_coherence: Array2<f64>,
    phase_linking_coherence: Option<Array2<f64>>,
    crlb_sigma: Option<Array3<f64>>,
    closure_phase: Option<Array3<f64>>,
    corrections: CorrectionLayers,
    geotransform: [f64; 6],
    reference_point: Option<(usize, usize)>,
}

fn emit_displacement(
    cfg: &DisplacementWorkflow,
    days: Vec<f64>,
    epsg: Option<u32>,
    crop: Option<&BoundedPlan>,
    mut spatial: SpatialProducts,
) -> Result<DisplacementOutput> {
    if let Some(plan) = crop {
        spatial.trim(
            plan.target_in_analysis,
            &days,
            cfg.timeseries_options.correlation_threshold,
        )?;
    }
    let phase_to_disp = cfg
        .input_options
        .wavelength
        .map_or(1.0, |w| -w / (4.0 * std::f64::consts::PI));
    let displacement = spatial.disp_rad.mapv(|phase| phase * phase_to_disp);
    let velocity = spatial.vel_rad.mapv(|rate| rate * phase_to_disp);
    let velocity_mm_yr = spatial
        .vel_rad
        .mapv(|rate| rate * mm_per_rad(cfg.input_options.wavelength));
    let quality = QualityLayers {
        phase_linking_coherence: spatial.phase_linking_coherence.as_ref(),
        crlb_sigma: spatial.crlb_sigma.as_ref(),
        closure_phase: spatial.closure_phase.as_ref(),
    };
    let geometry_provenance = crate::provenance::assemble_geometry_provenance_with_bounds(
        cfg,
        spatial.corrections.los_geometry.as_ref(),
        crop.map(|plan| plan.provenance.clone()),
    );
    timed("write", || -> Result<()> {
        write_outputs(
            cfg,
            displacement.view(),
            velocity.view(),
            spatial.temporal_coherence.view(),
            quality,
            epsg,
            spatial.geotransform,
        )?;
        write_correction_outputs(cfg, &spatial.corrections, epsg, spatial.geotransform)?;
        crate::provenance::write_geometry_provenance(&cfg.work_directory, &geometry_provenance)
    })?;
    Ok(DisplacementOutput {
        displacement,
        velocity,
        velocity_mm_yr,
        temporal_coherence: spatial.temporal_coherence,
        phase_linking_coherence: spatial.phase_linking_coherence,
        crlb_sigma: spatial.crlb_sigma,
        closure_phase: spatial.closure_phase,
        acquisition_days: days,
        epsg,
        geotransform: spatial.geotransform,
        reference_point: spatial.reference_point,
        ionosphere_delay: spatial.corrections.ionosphere,
        troposphere_delay: spatial.corrections.troposphere,
        los_geometry: spatial.corrections.los_geometry,
        geometry_provenance,
    })
}

impl SpatialProducts {
    fn trim(
        &mut self,
        target: BlockIndices,
        days: &[f64],
        correlation_threshold: f64,
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
            let local = select_reference_point(target_coherence, correlation_threshold).context(
                "bounded target has no pixel meeting the configured reference coherence threshold",
            )?;
            let global = (target.row_start + local.0, target.col_start + local.1);
            reference_to_point(&mut self.disp_rad, global);
            self.vel_rad = velocity_of(self.disp_rad.view(), days);
            self.reference_point = Some(global);
        }
        self.disp_rad = trim3(&self.disp_rad, target);
        self.vel_rad = trim2(&self.vel_rad, target);
        self.temporal_coherence = trim2(&self.temporal_coherence, target);
        self.phase_linking_coherence = self
            .phase_linking_coherence
            .take()
            .map(|layer| trim2(&layer, target));
        self.crlb_sigma = self.crlb_sigma.take().map(|layer| trim3(&layer, target));
        self.closure_phase = self.closure_phase.take().map(|layer| trim3(&layer, target));
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
    engine: &ComputeEngine,
    bounded: Option<BurstWindow>,
) -> Result<BurstLink> {
    let files = burst_files(cfg, idxs);
    let days = decimal_days(&files, &cfg.input_options.cslc_date_fmt)
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
    let out = phase_link_tiled(
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
    )?;
    burst_link(
        cfg,
        out,
        days,
        &files[0],
        (source.row_start, source.col_start),
    )
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
) -> Result<SequentialOutput> {
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
    let (mut read_s, mut compute_s) = (0.0_f64, 0.0_f64);
    let plans = plan_tiles(full_shape, strides, half, depth, out_block);
    let tile_count = plans.len();
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
        read_s += t_read.elapsed().as_secs_f64();
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
        let t_pl = Instant::now();
        let out = phase_link(cfg, stack.view(), engine)?;
        compute_s += t_pl.elapsed().as_secs_f64();
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
    tracing::info!(stage = "pl_breakdown", read_s, compute_s, "stage complete");
    Ok(acc.into_output())
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
            cpx: Array3::zeros((nslc, or, oc)),
            temp_coh: Array2::zeros((or, oc)),
            phase_linking_coherence: want_average_coherence.then(|| Array2::zeros((or, oc))),
            crlb: want_crlb.then(|| Array3::zeros((nslc, or, oc))),
            closure: None,
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
                .get_or_insert_with(|| Array3::zeros((src.dim().0, or, oc)));
            assign_block3(dst, src, g, l, (h, w));
        }
        Ok(())
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
        geo: resolve_burst_geo(cfg, first_file, rows, cols, source_offset)?,
        days,
    })
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
) -> Result<(BurstLink, SequentialState, BlockIndices)> {
    let files = burst_files(cfg, idxs);
    let days = decimal_days(&files, &cfg.input_options.cslc_date_fmt)
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
    let link = burst_link(
        cfg,
        out,
        days,
        &files[0],
        (source.row_start, source.col_start),
    )?;
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
    let engine = ComputeEngine::new(cfg.worker_settings.compute_backend);
    let groups = group_by_burst(&cfg.cslc_file_list);
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
                    let (link, seq, source) = link_one_burst_resumable(cfg, idxs, &engine, window)?;
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
    let output = finish_displacement(cfg, bursts, crop.as_ref())?;
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
    // The finite dependency cone grows when an update adds ministacks. Reusing a
    // prior bounded state could therefore omit newly-required halo pixels. A
    // bounded update deliberately recomputes the bounded analysis domain; it is
    // still memory-bounded and scientifically equivalent to a fresh AOI-local run.
    if cfg.output_options.bounds.is_some() {
        return run_displacement_resumable(cfg);
    }
    let engine = ComputeEngine::new(cfg.worker_settings.compute_backend);
    let groups = group_by_burst(&cfg.cslc_file_list);
    let layouts = source_layouts(cfg, &groups)?;
    let acquisitions = groups.values().map(Vec::len).max().unwrap_or(0);
    let crop = plan_bounds(cfg, &layouts, acquisitions)?;
    let scfg = sequential_config(cfg);
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
                    state, cfg, &scfg, id, idxs, &engine, window,
                ))
            })
            .collect()
    })?;
    for (link, st) in updated {
        states.push(st);
        bursts.push(link);
    }
    let output = finish_displacement(cfg, bursts, crop.as_ref())?;
    Ok((output, DisplacementState { bursts: states }))
}

/// Fold new acquisitions into one burst, returning its extended link + new state.
fn update_one_burst(
    state: &DisplacementState,
    cfg: &DisplacementWorkflow,
    scfg: &SequentialConfig,
    id: &str,
    idxs: &[usize],
    engine: &ComputeEngine,
    bounded: Option<BurstWindow>,
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
        update_sequential(&prev.seq, new_stack.view(), scfg, engine).map_err(anyhow::Error::msg)?;
    let days = decimal_days(&files, &cfg.input_options.cslc_date_fmt)
        .context("parsing acquisition dates from CSLC filenames")?;
    let link = burst_link(
        cfg,
        out,
        days,
        &files[0],
        (prev.source_window.row_start, prev.source_window.col_start),
    )?;
    let next = BurstState {
        id: id.to_string(),
        files,
        geo: prev.geo,
        source_window: prev.source_window,
        seq,
    };
    Ok((link, next))
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
            geo: b.geo.geo,
        });
    }
    let geos: Vec<BurstGeo> = bursts.iter().map(|b| b.geo).collect();
    let frame = frame_grid(&geos)?;
    let nslc = bursts[0].pl.dim().0;
    let mut pl = Array3::<Cf64>::zeros((nslc, frame.rows, frame.cols));
    let mut temp_coh = Array2::<f64>::zeros((frame.rows, frame.cols));
    let mut covered = Array2::<bool>::from_elem((frame.rows, frame.cols), false);
    for (burst_index, b) in bursts.iter_mut().enumerate() {
        anyhow::ensure!(b.pl.dim().0 == nslc, "bursts have differing date counts");
        let off = burst_offset(&frame, &b.geo);
        if burst_index > 0 {
            level_burst_offsets(&pl, &temp_coh, &covered, b, off, burst_index)?;
        }
        paste3(&mut pl, &b.pl, off);
        paste2(&mut temp_coh, &b.temp_coh, off);
        let (rows, cols) = b.temp_coh.dim();
        covered
            .slice_mut(s![off.0..off.0 + rows, off.1..off.1 + cols])
            .fill(true);
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
        geo: frame.geo,
    })
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
    let mut out = Array2::<f64>::zeros((frame.rows, frame.cols));
    for burst in bursts {
        paste2(&mut out, pick(burst)?, burst_offset(frame, &burst.geo));
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
    let mut out = Array3::<f64>::zeros((bands, frame.rows, frame.cols));
    for b in bursts {
        let layer = pick(b)?;
        paste3(&mut out, layer, burst_offset(frame, &b.geo));
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
        compute_crlb: cfg.phase_linking.write_crlb,
        // The phase-bias correction consumes the closure layer, so force it on
        // when the correction is enabled even if the raster isn't written.
        compute_closure_phase: cfg.phase_linking.write_closure_phase
            || cfg.phase_linking.correct_phase_bias,
        compute_average_coherence: cfg.phase_linking.calc_average_coh,
    }
}

/// Build the interferogram index pairs from the config and real baselines.
fn network(cfg: &DisplacementWorkflow, days: &[f64]) -> Vec<(usize, usize)> {
    let net = NetworkConfig {
        reference_idx: cfg.interferogram_network.reference_idx,
        max_bandwidth: cfg.interferogram_network.max_bandwidth,
        max_temporal_baseline: cfg.interferogram_network.max_temporal_baseline,
        indexes: cfg.interferogram_network.indexes.clone(),
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
    geotransform: [f64; 6],
    epsg: Option<u32>,
) -> Result<Array3<f64>> {
    let (_, rows, cols) = pl.dim();
    let scratch = cfg.work_directory.join("scratch");
    std::fs::create_dir_all(&scratch)?;
    let correlation = analysis_correlation(cfg, geotransform, epsg, (rows, cols))?;
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
    geotransform: [f64; 6],
    epsg: Option<u32>,
    shape: (usize, usize),
) -> Result<Array2<f32>> {
    if !cfg.unwrap_options.zero_where_masked {
        return Ok(Array2::from_elem(shape, 1.0));
    }
    let Some(path) = cfg.mask_file.as_ref() else {
        return Ok(Array2::from_elem(shape, 1.0));
    };
    let epsg = epsg.context("mask_file requires a sourced output EPSG")?;
    let mask = read_aligned_raster_window::<u8>(path, geotransform, epsg, shape)
        .context("reading configured aligned mask")?;
    Ok(mask.mapv(|value| if value == 0 { 0.0 } else { 1.0 }))
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
    let (nd, rows, cols) = displacement.dim();
    let series = Array3::from_shape_fn((nd + 1, rows, cols), |(t, r, c)| match t {
        0 => 0.0,
        _ => displacement[(t - 1, r, c)],
    });
    estimate_velocity(days, series.view(), None)
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
    write_f32("velocity.tif", velocity)?;
    write_f32("temporal_coherence.tif", temporal_coherence)?;
    if let Some(coherence) = quality.phase_linking_coherence {
        write_f32("phase_linking_coherence.tif", coherence.view())?;
    }
    write_bands(&write_f32, displacement, "displacement")?;
    if let Some(crlb) = quality.crlb_sigma {
        write_bands(&write_f32, crlb.view(), "crlb_sigma")?;
    }
    if let Some(closure) = quality.closure_phase {
        write_bands(&write_f32, closure.view(), "closure_phase")?;
    }
    Ok(())
}

/// The first burst's input files in date order (the dates the series is built on),
/// used to time-stamp the IONEX lookup. Mirrors how `days` is taken from the first
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
    Ok(())
}

/// The optional per-pixel quality layers written alongside displacement.
struct QualityLayers<'a> {
    phase_linking_coherence: Option<&'a Array2<f64>>,
    crlb_sigma: Option<&'a Array3<f64>>,
    closure_phase: Option<&'a Array3<f64>>,
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

    fn seam_burst(phase_offset: f64, coherence: f64) -> BurstLink {
        BurstLink {
            pl: Array3::from_shape_fn((2, 3, 3), |(date, _, _)| {
                Cf64::from_polar(1.0, date as f64 * 0.2 + phase_offset)
            }),
            temp_coh: Array2::from_elem((3, 3), coherence),
            phase_linking_coherence: None,
            crlb_sigma: None,
            closure_phase: None,
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
    fn bounded_trim_reselects_a_target_valid_reference_when_original_is_in_halo() {
        let mut products = SpatialProducts {
            disp_rad: Array3::from_shape_fn((2, 6, 8), |(date, row, col)| {
                date as f64 + row as f64 * 0.1 + col as f64 * 0.01
            }),
            vel_rad: Array2::zeros((6, 8)),
            temporal_coherence: Array2::from_elem((6, 8), 0.9),
            phase_linking_coherence: None,
            crlb_sigma: None,
            closure_phase: None,
            corrections: CorrectionLayers {
                ionosphere: None,
                troposphere: None,
                los_geometry: None,
            },
            geotransform: [0.0, 30.0, 0.0, 180.0, 0.0, -30.0],
            reference_point: Some((0, 0)),
        };
        let target = BlockIndices {
            row_start: 2,
            row_stop: 5,
            col_start: 3,
            col_stop: 7,
        };
        products.trim(target, &[0.0, 12.0, 24.0], 0.5).unwrap();
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
            temporal_coherence: Array2::from_elem((4, 4), 0.1),
            phase_linking_coherence: None,
            crlb_sigma: None,
            closure_phase: None,
            corrections: CorrectionLayers {
                ionosphere: None,
                troposphere: None,
                los_geometry: None,
            },
            geotransform: [0.0, 30.0, 0.0, 120.0, 0.0, -30.0],
            reference_point: Some((0, 0)),
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
                0.5,
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
            [0.0, 30.0, 0.0, 240.0, 0.0, -30.0],
            Some(32611),
            (8, 8),
        )
        .unwrap();
        assert!(correlation.iter().all(|&value| value == 1.0));
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
}
