//! Sequential (Ansari et al. 2017) phase-linking loop — port of
//! `workflows/sequential.py`.
//!
//! [`MiniStackPlanner`] partitions the stack; each ministack is phase-linked
//! over `[carried compressed SLCs] ++ [real SLCs]`, then compressed to one SLC
//! carried into the next ministack. The per-real-date linked phases are
//! concatenated into the end-to-end phase history.

use dolphin_core::config::CompressedSlcPlan;
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::{compress, estimate_stack_covariance, process_coherence_matrices};
use dolphin_stack::{MiniStack, MiniStackPlanner};
use ndarray::{concatenate, s, Array2, Array3, ArrayView3, Axis};

/// Configuration for a sequential phase-linking run.
#[derive(Debug, Clone, Copy)]
pub struct SequentialConfig {
    pub ministack_size: usize,
    pub max_num_compressed: usize,
    pub half_window: HalfWindow,
    pub strides: Strides,
    pub use_evd: bool,
    pub beta: f64,
    pub zero_correlation_threshold: f64,
    pub output_reference_idx: usize,
    pub compressed_slc_plan: CompressedSlcPlan,
}

/// Output of a sequential run.
pub struct SequentialOutput {
    /// Per-date linked phase (unit magnitude), `(nslc, out_rows, out_cols)`.
    pub cpx_phase: Array3<Cf64>,
    /// Compressed SLC produced by each ministack, `(rows, cols)` each.
    pub compressed_slcs: Vec<Array2<Cf64>>,
}

/// Run the sequential estimator over `slc_stack` `(nslc, rows, cols)`.
///
/// # Errors
/// Returns `Err` if planning fails or a covariance window exceeds the stack.
pub fn run_sequential(
    slc_stack: ArrayView3<Cf64>,
    cfg: &SequentialConfig,
) -> Result<SequentialOutput, &'static str> {
    let planner = MiniStackPlanner {
        num_slc: slc_stack.dim().0,
        max_num_compressed: cfg.max_num_compressed,
        output_reference_idx: cfg.output_reference_idx as isize,
        compressed_slc_plan: cfg.compressed_slc_plan,
    };
    let plans = planner.plan(cfg.ministack_size)?;

    let mut compressed_slcs: Vec<Array2<Cf64>> = Vec::new();
    let mut real_phases: Vec<Array3<Cf64>> = Vec::new();
    for ms in plans {
        let combined = assemble(&compressed_slcs, slc_stack, ms);
        let (cpx, compressed) = link_and_compress(combined.view(), ms, cfg)?;
        real_phases.push(cpx.slice(s![ms.num_compressed.., .., ..]).to_owned());
        compressed_slcs.push(compressed);
    }

    let views: Vec<ArrayView3<Cf64>> = real_phases.iter().map(Array3::view).collect();
    let cpx_phase = concatenate(Axis(0), &views).map_err(|_| "phase-history concat failed")?;
    Ok(SequentialOutput {
        cpx_phase,
        compressed_slcs,
    })
}

/// Stack the carried compressed SLCs ahead of this ministack's real SLCs.
fn assemble(
    compressed: &[Array2<Cf64>],
    slc_stack: ArrayView3<Cf64>,
    ms: MiniStack,
) -> Array3<Cf64> {
    let (_, rows, cols) = slc_stack.dim();
    let carried = &compressed[compressed.len() - ms.num_compressed..];
    Array3::from_shape_fn((ms.size(), rows, cols), |(k, r, c)| {
        match k < ms.num_compressed {
            true => carried[k][(r, c)],
            false => slc_stack[(ms.real_start + (k - ms.num_compressed), r, c)],
        }
    })
}

/// Phase-link a combined ministack and compress it to one SLC.
fn link_and_compress(
    combined: ArrayView3<Cf64>,
    ms: MiniStack,
    cfg: &SequentialConfig,
) -> Result<(Array3<Cf64>, Array2<Cf64>), &'static str> {
    let c = estimate_stack_covariance(combined, cfg.half_window, cfg.strides, None)?;
    let est = process_coherence_matrices(
        c.view(),
        cfg.use_evd,
        cfg.beta,
        cfg.zero_correlation_threshold,
        ms.output_reference_idx as usize,
    );
    let cpx = est.cpx_phase.mapv(unit_phasor);
    let compressed = compress(
        combined,
        cpx.view(),
        ms.num_compressed,
        Some(ms.compressed_reference_idx as usize),
    );
    Ok((cpx, compressed))
}

/// Unit-magnitude phasor `exp(j∠z)` (dolphin's `exp(1j*angle(cpx_phase))`).
fn unit_phasor(z: Cf64) -> Cf64 {
    Cf64::from_polar(1.0, z.arg())
}
