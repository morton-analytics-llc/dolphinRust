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
    read_cslc_shape, read_cslc_stack, read_cslc_window, read_geotransform, read_nisar_geotransform,
    read_nisar_stack, read_nisar_window, write_raster, GeoInfo,
};
use dolphin_phaselink::{correct_phase_bias, estimate_bias_velocity, ComputeEngine};
use dolphin_timeseries::{
    build_network, estimate_velocity, get_incidence_matrix, invert_stack, invert_stack_l1,
    reference_to_point, select_reference_point, L1Config, NetworkConfig,
};
use dolphin_unwrap::{CostMode, InitMethod, TophuConfig, UnwrapConfig};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis};

use crate::burst::{burst_offset, frame_grid, group_by_burst, paste2, paste3, BurstGeo, FrameGrid};
use crate::corrections::{apply_corrections, CorrectionLayers};
use crate::dates::decimal_days;
use crate::sequential::{
    run_sequential, run_sequential_resumable, update_sequential, SequentialConfig,
    SequentialOutput, SequentialState,
};
use crate::tiling::{plan_tiles, TilePlan};
use crate::unwrap_backend::{SnaphuBackend, TophuBackend, UnwrapBackend};

/// Sentinel-1 C-band radar wavelength (m); used to express velocity in mm/yr
/// when the config carries no explicit `input_options.wavelength`.
const SENTINEL1_WAVELENGTH_M: f64 = 0.055_465_76;

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
}

/// Run `f`, emitting its wall-clock under `stage` at INFO (`stage` + `elapsed_s`
/// fields) so the benchmark and host app can read per-stage timing via `RUST_LOG`.
fn timed<T>(stage: &str, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let out = f();
    tracing::info!(
        stage,
        elapsed_s = t0.elapsed().as_secs_f64(),
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
    // One compute engine for the whole run: it acquires a single GPU context (if
    // selected and available) and is reused across every burst + ministack.
    let engine = ComputeEngine::new(cfg.worker_settings.compute_backend);
    let bursts = timed("phase_linking", || {
        groups
            .values()
            .map(|idxs| link_one_burst(cfg, idxs, &engine))
            .collect::<Result<Vec<_>>>()
    })?;
    finish_displacement(cfg, bursts)
}

/// Shared downstream tail: stitch bursts → ifg network → SNAPHU unwrap → SBAS
/// inversion → reference → atmospheric corrections → velocity → write COGs.
/// Identical for a full run and an incremental update — both feed it the same
/// per-burst phase-linking products, so both produce the same output.
fn finish_displacement(
    cfg: &DisplacementWorkflow,
    bursts: Vec<BurstLink>,
) -> Result<DisplacementOutput> {
    let groups = group_by_burst(&cfg.cslc_file_list);
    let days = bursts
        .first()
        .map(|b| b.days.clone())
        .context("cslc_file_list is empty")?;
    let stitched = stitch_bursts(bursts)?;
    let mut pl = stitched.pl;
    if cfg.phase_linking.correct_phase_bias {
        apply_phase_bias(&mut pl, stitched.closure_phase.as_ref())?;
    }
    let temporal_coherence = stitched.temp_coh;
    let crlb_sigma = stitched.crlb_sigma;
    let closure_phase = stitched.closure_phase;
    let geo = stitched.geo;
    let epsg = (geo.epsg != 0).then_some(geo.epsg);
    let geotransform = geo.geotransform;
    anyhow::ensure!(
        days.len() == pl.dim().0,
        "parsed {} dates but phase-linking produced {} acquisitions",
        days.len(),
        pl.dim().0
    );
    let pairs = network(cfg, &days);
    anyhow::ensure!(!pairs.is_empty(), "interferogram_network produced no pairs");

    let dphi_rad = timed("unwrap", || unwrap_network(cfg, pl.view(), &pairs))?;
    let incidence = get_incidence_matrix(&pairs);
    let mut disp_rad = timed("timeseries", || match cfg.timeseries_options.method {
        TimeseriesMethod::L1 => {
            invert_stack_l1(incidence.view(), dphi_rad.view(), L1Config::default())
        }
        TimeseriesMethod::L2 => invert_stack(incidence.view(), dphi_rad.view(), None),
    });
    // Spatially reference the series to a stable pixel (dolphin parity): the
    // configured point, else the center-of-mass of the high-coherence region.
    let reference_point = cfg.timeseries_options.reference_point.or_else(|| {
        select_reference_point(
            temporal_coherence.view(),
            cfg.timeseries_options.correlation_threshold,
        )
    });
    if let Some(point) = reference_point {
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

    let phase_to_disp = cfg
        .input_options
        .wavelength
        .map_or(1.0, |w| -w / (4.0 * std::f64::consts::PI));
    let mm = mm_per_rad(cfg.input_options.wavelength);

    let displacement = disp_rad.mapv(|p| p * phase_to_disp);
    let velocity = vel_rad.mapv(|v| v * phase_to_disp);
    let velocity_mm_yr = vel_rad.mapv(|v| v * mm);

    let quality = QualityLayers {
        crlb_sigma: crlb_sigma.as_ref(),
        closure_phase: closure_phase.as_ref(),
    };
    write_outputs(
        cfg,
        displacement.view(),
        velocity.view(),
        temporal_coherence.view(),
        quality,
        epsg,
        geotransform,
    )?;
    write_correction_outputs(cfg, &corrections, epsg, geotransform)?;
    Ok(DisplacementOutput {
        displacement,
        velocity,
        velocity_mm_yr,
        temporal_coherence,
        crlb_sigma,
        closure_phase,
        acquisition_days: days,
        epsg,
        geotransform,
        reference_point,
        ionosphere_delay: corrections.ionosphere,
        troposphere_delay: corrections.troposphere,
    })
}

/// One burst's phase-linking products, carried until stitched onto the frame.
struct BurstLink {
    /// Linked phase history `(n_dates, out_rows, out_cols)`.
    pl: Array3<Cf64>,
    /// Temporal coherence `(out_rows, out_cols)`.
    temp_coh: Array2<f64>,
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
    let out = phase_link_tiled(cfg, full_shape, files.len(), engine, |block| {
        read_burst_tile(cfg.input_options.input_type, &files, &subdataset, block)
    })?;
    burst_link(cfg, out, days, &files[0])
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
    let mut acc = TiledOutput::new(nslc, out_shape, cfg.phase_linking.write_crlb);
    for plan in plan_tiles(full_shape, strides, half, depth, out_block) {
        let stack = read_tile(plan.read)?;
        let out = phase_link(cfg, stack.view(), engine)?;
        acc.place(&plan, &out)?;
    }
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

/// Accumulates per-tile sequential outputs into the whole-burst grid. The
/// per-tile compressed SLCs are not assembled (the batch path never consumes
/// them); the closure layer is allocated lazily once its band count is known.
struct TiledOutput {
    cpx: Array3<Cf64>,
    temp_coh: Array2<f64>,
    crlb: Option<Array3<f64>>,
    closure: Option<Array3<f64>>,
    out_shape: (usize, usize),
}

impl TiledOutput {
    fn new(nslc: usize, out_shape: (usize, usize), want_crlb: bool) -> Self {
        let (or, oc) = out_shape;
        Self {
            cpx: Array3::zeros((nslc, or, oc)),
            temp_coh: Array2::zeros((or, oc)),
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
        crlb_sigma: out.crlb_sigma,
        closure_phase: out.closure_phase,
        geo: resolve_burst_geo(cfg, first_file, rows, cols),
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
    /// Sequential phase-linking carry (sealed ministacks + open trailing SLCs).
    seq: SequentialState,
}

/// Phase-link a single burst, also returning its resumable [`SequentialState`].
fn link_one_burst_resumable(
    cfg: &DisplacementWorkflow,
    idxs: &[usize],
    engine: &ComputeEngine,
) -> Result<(BurstLink, SequentialState)> {
    let files = burst_files(cfg, idxs);
    let days = decimal_days(&files, &cfg.input_options.cslc_date_fmt)
        .context("parsing acquisition dates from CSLC filenames")?;
    let stack = read_stack_files(cfg, &files)?;
    let (out, state) = run_sequential_resumable(stack.view(), &sequential_config(cfg), engine)
        .map_err(anyhow::Error::msg)?;
    let link = burst_link(cfg, out, days, &files[0])?;
    Ok((link, state))
}

/// The CSLC files for a burst's indices into `cfg.cslc_file_list`.
fn burst_files(cfg: &DisplacementWorkflow, idxs: &[usize]) -> Vec<PathBuf> {
    idxs.iter()
        .map(|&i| cfg.cslc_file_list[i].clone())
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
    let mut bursts = Vec::with_capacity(groups.len());
    let mut states = Vec::with_capacity(groups.len());
    let linked = timed("phase_linking", || -> Result<Vec<_>> {
        groups
            .iter()
            .map(|(id, idxs)| {
                let (link, seq) = link_one_burst_resumable(cfg, idxs, &engine)?;
                Ok((id.clone(), burst_files(cfg, idxs), link, seq))
            })
            .collect()
    })?;
    for (id, files, link, seq) in linked {
        states.push(BurstState {
            id,
            files,
            geo: link.geo,
            seq,
        });
        bursts.push(link);
    }
    let output = finish_displacement(cfg, bursts)?;
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
    let engine = ComputeEngine::new(cfg.worker_settings.compute_backend);
    let groups = group_by_burst(&cfg.cslc_file_list);
    let scfg = sequential_config(cfg);
    let mut bursts = Vec::with_capacity(groups.len());
    let mut states = Vec::with_capacity(groups.len());
    let updated = timed("phase_linking", || -> Result<Vec<_>> {
        groups
            .iter()
            .map(|(id, idxs)| update_one_burst(state, cfg, &scfg, id, idxs, &engine))
            .collect()
    })?;
    for (link, st) in updated {
        states.push(st);
        bursts.push(link);
    }
    let output = finish_displacement(cfg, bursts)?;
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
    let new_stack = read_stack_files(cfg, new_files)?;
    let (out, seq) =
        update_sequential(&prev.seq, new_stack.view(), scfg, engine).map_err(anyhow::Error::msg)?;
    let days = decimal_days(&files, &cfg.input_options.cslc_date_fmt)
        .context("parsing acquisition dates from CSLC filenames")?;
    let link = burst_link(cfg, out, days, &files[0])?;
    let next = BurstState {
        id: id.to_string(),
        files,
        geo: prev.geo,
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
    for b in &bursts {
        anyhow::ensure!(b.pl.dim().0 == nslc, "bursts have differing date counts");
        let off = burst_offset(&frame, &b.geo);
        paste3(&mut pl, &b.pl, off);
        paste2(&mut temp_coh, &b.temp_coh, off);
    }
    let crlb_sigma = stitch_layer(&bursts, &frame, |b| b.crlb_sigma.as_ref());
    let closure_phase = stitch_layer(&bursts, &frame, |b| b.closure_phase.as_ref());
    Ok(Stitched {
        pl,
        temp_coh,
        crlb_sigma,
        closure_phase,
        geo: frame.geo,
    })
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
) -> BurstGeo {
    let identity = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    let geo_reader = match cfg.input_options.input_type {
        InputType::OperaCslc => read_geotransform,
        InputType::NisarGslc => read_nisar_geotransform,
    };
    let read = cfg
        .input_options
        .subdataset
        .as_deref()
        .and_then(|sds| geo_reader(path, sds).ok());
    let (epsg, gt) = match read {
        Some(g) => (g.epsg, g.geotransform),
        None => (cfg.output_options.epsg.unwrap_or(0), identity),
    };
    let (sx, sy) = (
        cfg.output_options.strides.x as f64,
        cfg.output_options.strides.y as f64,
    );
    BurstGeo {
        geo: GeoInfo {
            epsg,
            geotransform: [gt[0], gt[1] * sx, 0.0, gt[3], 0.0, gt[5] * sy],
        },
        rows,
        cols,
    }
}

/// Read the CSLC files into a `(n, rows, cols)` `Cf64` stack.
fn read_stack_files(cfg: &DisplacementWorkflow, files: &[PathBuf]) -> Result<Array3<Cf64>> {
    let subdataset = cfg
        .input_options
        .subdataset
        .clone()
        .context("input_options.subdataset is required to read CSLC HDF5")?;
    let stack = match cfg.input_options.input_type {
        InputType::OperaCslc => {
            let pairs: Vec<(PathBuf, String)> = files
                .iter()
                .map(|p| (p.clone(), subdataset.clone()))
                .collect();
            read_cslc_stack(&pairs)?
        }
        InputType::NisarGslc => read_nisar_stack(files, &subdataset)?,
    };
    Ok(stack.mapv(|z| Cf64::new(z.re as f64, z.im as f64)))
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
) -> Result<Array3<f64>> {
    let (_, rows, cols) = pl.dim();
    let scratch = cfg.work_directory.join("scratch");
    std::fs::create_dir_all(&scratch)?;
    let correlation = Array2::<f32>::from_elem((rows, cols), 1.0);
    let backend = unwrap_backend(cfg, (rows, cols));
    // Bound network unwrap concurrency: N concurrent SNAPHU processes + N scratch
    // sets. Pinning the pool caps peak memory and keeps the block-tiled RSS win.
    let pool = unwrap_pool(cfg.unwrap_options.n_parallel_jobs)?;
    pool.install(|| backend.unwrap_network(pl, pairs, correlation.view(), &scratch))
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
        _ => Box::new(SnaphuBackend(unwrap_config(cfg, grid))),
    }
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
