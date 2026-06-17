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

use crate::sequential::{run_sequential, SequentialConfig};

/// Even temporal sampling (days) assumed when the config carries no dates.
const DT_DAYS: f64 = 12.0;

/// Displacement pipeline outputs.
pub struct DisplacementOutput {
    /// Per-date displacement (dates 1..n, date 0 = 0 reference), `(n-1, rows, cols)`.
    pub displacement: Array3<f64>,
    /// Linear velocity per pixel (units/year), `(rows, cols)`.
    pub velocity: Array2<f64>,
}

/// Run the displacement workflow from a parsed config.
///
/// # Errors
/// Returns `Err` on I/O, phase-linking, unwrapping, or config problems.
pub fn run_displacement(cfg: &DisplacementWorkflow) -> Result<DisplacementOutput> {
    let stack = read_stack(cfg)?;
    let pl = phase_link(cfg, stack.view())?;
    let pairs = network(cfg, pl.dim().0);
    anyhow::ensure!(!pairs.is_empty(), "interferogram_network produced no pairs");

    let dphi = unwrap_network(cfg, pl.view(), &pairs)?;
    let incidence = get_incidence_matrix(&pairs);
    let displacement = invert_stack(incidence.view(), dphi.view(), None);
    let velocity = velocity_of(displacement.view());

    write_outputs(cfg, displacement.view(), velocity.view())?;
    Ok(DisplacementOutput {
        displacement,
        velocity,
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

/// Build the interferogram index pairs from the config.
fn network(cfg: &DisplacementWorkflow, n: usize) -> Vec<(usize, usize)> {
    let days: Vec<f64> = (0..n).map(|i| i as f64 * DT_DAYS).collect();
    let net = NetworkConfig {
        reference_idx: cfg.interferogram_network.reference_idx,
        max_bandwidth: cfg.interferogram_network.max_bandwidth,
        max_temporal_baseline: cfg.interferogram_network.max_temporal_baseline,
        indexes: cfg.interferogram_network.indexes.clone(),
    };
    build_network(n, &days, &net)
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

/// Linear velocity from the displacement series (date 0 = 0 reference).
fn velocity_of(displacement: ArrayView3<f64>) -> Array2<f64> {
    let (nd, rows, cols) = displacement.dim();
    let days: Vec<f64> = (0..=nd).map(|i| i as f64 * DT_DAYS).collect();
    let series = Array3::from_shape_fn((nd + 1, rows, cols), |(t, r, c)| match t {
        0 => 0.0,
        _ => displacement[(t - 1, r, c)],
    });
    estimate_velocity(&days, series.view(), None)
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
