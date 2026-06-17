//! End-to-end displacement pipeline (port of `workflows/displacement.py`).
//!
//! Order: group inputs by burst → per-burst sequential phase linking → stitch
//! bursts onto the frame grid → ifg network → SNAPHU unwrap → SBAS inversion →
//! velocity → write COGs. Single-burst stacks take the stitch identity path.
//! Synchronous; the host app bridges to its runtime.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use dolphin_core::config::{DisplacementWorkflow, TimeseriesMethod};
use dolphin_core::{Cf32, Cf64};
use dolphin_io::{read_cslc_stack, read_geotransform, write_raster, GeoInfo};
use dolphin_phaselink::ComputeEngine;
use dolphin_timeseries::{
    build_network, estimate_velocity, get_incidence_matrix, invert_stack, invert_stack_l1,
    reference_to_point, select_reference_point, L1Config, NetworkConfig,
};
use dolphin_unwrap::{unwrap, CostMode, InitMethod, UnwrapConfig};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::burst::{burst_offset, frame_grid, group_by_burst, paste2, paste3, BurstGeo, FrameGrid};
use crate::dates::decimal_days;
use crate::sequential::{run_sequential, SequentialConfig, SequentialOutput};

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
    /// Temporal coherence per pixel in `[0, 1]`, averaged across ministacks
    /// (dolphin's `temporal_coherence_average`); a phase-quality mask, `(rows, cols)`.
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
    let days = bursts
        .first()
        .map(|b| b.days.clone())
        .context("cslc_file_list is empty")?;
    let stitched = stitch_bursts(bursts)?;
    let pl = stitched.pl;
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
fn link_one_burst(
    cfg: &DisplacementWorkflow,
    idxs: &[usize],
    engine: &ComputeEngine,
) -> Result<BurstLink> {
    let files: Vec<PathBuf> = idxs
        .iter()
        .map(|&i| cfg.cslc_file_list[i].clone())
        .collect();
    let days = decimal_days(&files, &cfg.input_options.cslc_date_fmt)
        .context("parsing acquisition dates from CSLC filenames")?;
    let stack = read_stack_files(cfg, &files)?;
    let out = phase_link(cfg, stack.view(), engine)?;
    let pl = out.cpx_phase;
    anyhow::ensure!(
        days.len() == pl.dim().0,
        "parsed {} dates but phase-linking produced {} acquisitions",
        days.len(),
        pl.dim().0
    );
    let (_, rows, cols) = pl.dim();
    let geo = resolve_burst_geo(cfg, &files[0], rows, cols);
    Ok(BurstLink {
        pl,
        temp_coh: out.temporal_coherence,
        crlb_sigma: out.crlb_sigma,
        closure_phase: out.closure_phase,
        geo,
        days,
    })
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
    let read = cfg
        .input_options
        .subdataset
        .as_deref()
        .and_then(|sds| read_geotransform(path, sds).ok());
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
    let pairs: Vec<(PathBuf, String)> = files
        .iter()
        .map(|p| (p.clone(), subdataset.clone()))
        .collect();
    let stack = read_cslc_stack(&pairs)?;
    Ok(stack.mapv(|z| Cf64::new(z.re as f64, z.im as f64)))
}

/// Sequential phase linking over the stack; returns the linked phase history,
/// the averaged temporal coherence, and the optional CRLB / closure layers.
fn phase_link(
    cfg: &DisplacementWorkflow,
    stack: ArrayView3<Cf64>,
    engine: &ComputeEngine,
) -> Result<SequentialOutput> {
    let scfg = SequentialConfig {
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
        compute_closure_phase: cfg.phase_linking.write_closure_phase,
    };
    let out = run_sequential(stack, &scfg, engine).map_err(anyhow::Error::msg)?;
    Ok(out)
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

/// Form each ifg from the linked phase and unwrap it with SNAPHU.
fn unwrap_network(
    cfg: &DisplacementWorkflow,
    pl: ArrayView3<Cf64>,
    pairs: &[(usize, usize)],
) -> Result<Array3<f64>> {
    let (_, rows, cols) = pl.dim();
    let scratch = cfg.work_directory.join("scratch");
    std::fs::create_dir_all(&scratch)?;
    let ucfg = unwrap_config(cfg);
    let correlation = Array2::<f32>::from_elem((rows, cols), 1.0);

    let layers = pairs
        .iter()
        .map(|&pair| unwrap_pair(pl, pair, &correlation, &ucfg, &scratch))
        .collect::<Result<Vec<_>>>()?;
    let views: Vec<_> = layers.iter().map(Array2::view).collect();
    ndarray::stack(ndarray::Axis(0), &views).context("stacking unwrapped ifgs")
}

/// Unwrap one ifg `(i, j)` formed as `exp(j∠(pl_j · conj(pl_i)))`.
fn unwrap_pair(
    pl: ArrayView3<Cf64>,
    (i, j): (usize, usize),
    correlation: &Array2<f32>,
    ucfg: &UnwrapConfig,
    scratch: &Path,
) -> Result<Array2<f64>> {
    let (_, rows, cols) = pl.dim();
    let ifg = Array2::from_shape_fn((rows, cols), |(r, c)| {
        let z = pl[(j, r, c)] * pl[(i, r, c)].conj();
        Cf32::from_polar(1.0, z.arg() as f32)
    });
    let result = unwrap(ifg.view(), correlation.view(), ucfg, scratch)?;
    Ok(result.unwrapped.mapv(f64::from))
}

/// Map the config's SNAPHU options to the unwrap wrapper config.
fn unwrap_config(cfg: &DisplacementWorkflow) -> UnwrapConfig {
    let snaphu = &cfg.unwrap_options.snaphu_options;
    UnwrapConfig {
        cost: if snaphu.cost == "defo" {
            CostMode::Defo
        } else {
            CostMode::Smooth
        },
        init: if snaphu.init_method == "mst" {
            InitMethod::Mst
        } else {
            InitMethod::Mcf
        },
        ntiles: snaphu.ntiles,
        tile_overlap: snaphu.tile_overlap,
        nproc: snaphu.n_parallel_tiles,
        snaphu_path: "snaphu".to_string(),
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
