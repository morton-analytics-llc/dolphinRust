//! End-to-end displacement pipeline (port of `workflows/displacement.py`).
//!
//! Single-burst order: read CSLC stack → sequential phase linking → ifg network
//! → SNAPHU unwrap → SBAS L2 inversion → velocity → write GeoTIFFs. Multi-burst
//! stitching is a follow-up (see STATUS.md). Synchronous; the host app bridges
//! to its runtime.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::{Cf32, Cf64};
use dolphin_io::{read_cslc_stack, write_raster};
use dolphin_timeseries::{
    build_network, estimate_velocity, get_incidence_matrix, invert_stack, NetworkConfig,
};
use dolphin_unwrap::{unwrap, CostMode, InitMethod, UnwrapConfig};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::dates::decimal_days;
use crate::sequential::{run_sequential, SequentialConfig};

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
    /// Acquisition dates as decimal days from acquisition 0, length `n_dates`.
    pub acquisition_days: Vec<f64>,
}

/// Run the displacement workflow from a parsed config.
///
/// # Errors
/// Returns `Err` on I/O, phase-linking, unwrapping, date-parsing, or config problems.
pub fn run_displacement(cfg: &DisplacementWorkflow) -> Result<DisplacementOutput> {
    let stack = read_stack(cfg)?;
    let days = decimal_days(&cfg.cslc_file_list, &cfg.input_options.cslc_date_fmt)
        .context("parsing acquisition dates from CSLC filenames")?;
    let pl = phase_link(cfg, stack.view())?;
    anyhow::ensure!(
        days.len() == pl.dim().0,
        "parsed {} dates but phase-linking produced {} acquisitions",
        days.len(),
        pl.dim().0
    );
    let pairs = network(cfg, &days);
    anyhow::ensure!(!pairs.is_empty(), "interferogram_network produced no pairs");

    let dphi_rad = unwrap_network(cfg, pl.view(), &pairs)?;
    let incidence = get_incidence_matrix(&pairs);
    let disp_rad = invert_stack(incidence.view(), dphi_rad.view(), None);
    let vel_rad = velocity_of(disp_rad.view(), &days);

    let phase_to_disp = cfg
        .input_options
        .wavelength
        .map_or(1.0, |w| -w / (4.0 * std::f64::consts::PI));
    let mm = mm_per_rad(cfg.input_options.wavelength);

    let displacement = disp_rad.mapv(|p| p * phase_to_disp);
    let velocity = vel_rad.mapv(|v| v * phase_to_disp);
    let velocity_mm_yr = vel_rad.mapv(|v| v * mm);

    write_outputs(cfg, displacement.view(), velocity.view())?;
    Ok(DisplacementOutput {
        displacement,
        velocity,
        velocity_mm_yr,
        acquisition_days: days,
    })
}

/// Read the CSLC stack named in the config into `(n, rows, cols)` `Cf64`.
fn read_stack(cfg: &DisplacementWorkflow) -> Result<Array3<Cf64>> {
    let subdataset = cfg
        .input_options
        .subdataset
        .clone()
        .context("input_options.subdataset is required to read CSLC HDF5")?;
    let files: Vec<(PathBuf, String)> = cfg
        .cslc_file_list
        .iter()
        .map(|p| (p.clone(), subdataset.clone()))
        .collect();
    let stack = read_cslc_stack(&files)?;
    Ok(stack.mapv(|z| Cf64::new(z.re as f64, z.im as f64)))
}

/// Sequential phase linking over the stack.
fn phase_link(cfg: &DisplacementWorkflow, stack: ArrayView3<Cf64>) -> Result<Array3<Cf64>> {
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
    };
    let out = run_sequential(stack, &scfg).map_err(anyhow::Error::msg)?;
    Ok(out.cpx_phase)
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

/// Write the velocity and per-date displacement rasters as GeoTIFFs.
fn write_outputs(
    cfg: &DisplacementWorkflow,
    displacement: ArrayView3<f64>,
    velocity: ArrayView2<f64>,
) -> Result<()> {
    let gt = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    let dir = &cfg.work_directory;
    std::fs::create_dir_all(dir)?;
    write_raster(
        &dir.join("velocity.tif"),
        velocity.mapv(|v| v as f32).view(),
        gt,
        cfg.output_options.epsg,
        None,
    )?;
    for t in 0..displacement.dim().0 {
        let band = displacement
            .index_axis(ndarray::Axis(0), t)
            .mapv(|v| v as f32);
        write_raster(
            &dir.join(format!("displacement_{t:02}.tif")),
            band.view(),
            gt,
            cfg.output_options.epsg,
            None,
        )?;
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
