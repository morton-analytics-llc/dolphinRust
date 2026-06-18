//! Sequential (Ansari et al. 2017) phase-linking loop — port of
//! `workflows/sequential.py`.
//!
//! [`MiniStackPlanner`] partitions the stack; each ministack is phase-linked
//! over `[carried compressed SLCs] ++ [real SLCs]`, then compressed to one SLC
//! carried into the next ministack. The per-real-date linked phases are
//! concatenated into the end-to-end phase history.

use dolphin_core::config::CompressedSlcPlan;
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::{
    compress, estimate_closure_phases, estimate_crlb, estimate_temp_coh, ComputeEngine,
};
use dolphin_stack::{MiniStack, MiniStackPlanner};
use ndarray::{concatenate, s, Array2, Array3, ArrayView3, ArrayView4, Axis};

/// Configuration for a sequential phase-linking run.
#[derive(Debug, Clone, Copy)]
pub struct SequentialConfig {
    /// Number of real SLCs per ministack.
    pub ministack_size: usize,
    /// Maximum compressed SLCs carried into a ministack.
    pub max_num_compressed: usize,
    /// Covariance estimation half-window (rows, cols).
    pub half_window: HalfWindow,
    /// Output downsampling strides (rows, cols).
    pub strides: Strides,
    /// Use eigenvalue decomposition (EVD) instead of EMI phase linking.
    pub use_evd: bool,
    /// Coherence-matrix regularization weight (EMI `beta`).
    pub beta: f64,
    /// Coherence values at or below this are treated as zero.
    pub zero_correlation_threshold: f64,
    /// Index of the reference date for the output phase history.
    pub output_reference_idx: usize,
    /// Strategy for choosing each ministack's compressed-SLC reference.
    pub compressed_slc_plan: CompressedSlcPlan,
    /// Produce the per-date CRLB σ layer (dolphin `write_crlb`).
    pub compute_crlb: bool,
    /// Produce the per-triplet closure-phase layer (dolphin `write_closure_phase`).
    pub compute_closure_phase: bool,
}

/// Output of a sequential run.
pub struct SequentialOutput {
    /// Per-date linked phase (unit magnitude), `(nslc, out_rows, out_cols)`.
    pub cpx_phase: Array3<Cf64>,
    /// Compressed SLC produced by each ministack, `(rows, cols)` each.
    pub compressed_slcs: Vec<Array2<Cf64>>,
    /// Temporal coherence stitched across ministacks by NaN-aware mean
    /// (dolphin's `temporal_coherence_average` = `numpy.nanmean`), `(out_rows,
    /// out_cols)`. 1.0 = perfect phase consistency.
    pub temporal_coherence: Array2<f64>,
    /// Per-date CRLB σ (radians), `(nslc, out_rows, out_cols)` — real dates only,
    /// concatenated across ministacks. `None` when `compute_crlb` is off.
    pub crlb_sigma: Option<Array3<f64>>,
    /// Per-ministack nearest-neighbour closure phase (radians), band-major,
    /// concatenated across ministacks. `None` when `compute_closure_phase` is off.
    pub closure_phase: Option<Array3<f64>>,
}

/// Run the sequential estimator over `slc_stack` `(nslc, rows, cols)`.
///
/// # Errors
/// Returns `Err` if planning fails or a covariance window exceeds the stack.
pub fn run_sequential(
    slc_stack: ArrayView3<Cf64>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
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
    let mut temp_cohs: Vec<Array2<f64>> = Vec::new();
    let mut crlbs: Vec<Array3<f64>> = Vec::new();
    let mut closures: Vec<Array3<f64>> = Vec::new();
    for ms in plans {
        let combined = assemble(&compressed_slcs, slc_stack, ms);
        let r = link_and_compress(combined.view(), ms, cfg, engine)?;
        real_phases.push(r.cpx.slice(s![ms.num_compressed.., .., ..]).to_owned());
        compressed_slcs.push(r.compressed);
        temp_cohs.push(r.temp_coh);
        if let Some(s) = r.crlb_sigma {
            crlbs.push(s);
        }
        if let Some(s) = r.closure_phase {
            closures.push(s);
        }
    }

    let views: Vec<ArrayView3<Cf64>> = real_phases.iter().map(Array3::view).collect();
    let cpx_phase = concatenate(Axis(0), &views).map_err(|_| "phase-history concat failed")?;
    let temporal_coherence = stitch_temp_coh(&temp_cohs);
    Ok(SequentialOutput {
        cpx_phase,
        compressed_slcs,
        temporal_coherence,
        crlb_sigma: concat_bands(crlbs)?,
        closure_phase: concat_bands(closures)?,
    })
}

/// Concatenate per-ministack band-major layers along the date/triplet axis;
/// `None` when the layer was not produced.
fn concat_bands(layers: Vec<Array3<f64>>) -> Result<Option<Array3<f64>>, &'static str> {
    if layers.is_empty() {
        return Ok(None);
    }
    let views: Vec<ArrayView3<f64>> = layers.iter().map(Array3::view).collect();
    concatenate(Axis(0), &views)
        .map(Some)
        .map_err(|_| "quality-layer concat failed")
}

/// Per-ministack temporal-coherence stitch — dolphin's cross-ministack reduction
/// (`numpy.nanmean(A, axis=0)` in `_average_or_rename`): a per-pixel NaN-aware
/// mean of the per-ministack layers. A pixel that is masked/decorrelated (NaN) in
/// some ministacks averages only the finite ones; all-NaN stays NaN. Equals a
/// plain mean when every layer is finite (single-ministack and fully-coherent
/// many-ministack cases), so prior parity holds while a masked many-ministack
/// frame now matches dolphin instead of being diluted toward zero. This is the
/// reduction the per-band CRLB/closure layers are concatenated against, closing
/// their many-ministack caveat.
fn stitch_temp_coh(layers: &[Array2<f64>]) -> Array2<f64> {
    Array2::from_shape_fn(layers[0].dim(), |(r, c)| {
        nanmean(layers.iter().map(|l| l[(r, c)]))
    })
}

/// NaN-aware mean over an iterator: averages only the finite values; `NaN` when
/// none are finite (`numpy.nanmean` of an all-NaN slice).
fn nanmean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values
        .filter(|v| v.is_finite())
        .fold((0.0, 0_usize), |(s, n), v| (s + v, n + 1));
    match count {
        0 => f64::NAN,
        _ => sum / count as f64,
    }
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

/// One ministack's phase-linking products.
struct MinistackResult {
    /// Linked phase (unit magnitude), `(nslc, out_rows, out_cols)`.
    cpx: Array3<Cf64>,
    /// Compressed SLC, `(out_rows, out_cols)`.
    compressed: Array2<Cf64>,
    /// Temporal coherence, `(out_rows, out_cols)`.
    temp_coh: Array2<f64>,
    /// CRLB σ for this ministack's real dates, `(num_real, out_rows, out_cols)`.
    crlb_sigma: Option<Array3<f64>>,
    /// Closure phase for this ministack, `(num_combined-2, out_rows, out_cols)`.
    closure_phase: Option<Array3<f64>>,
}

/// Phase-link a combined ministack and compress it to one SLC, plus its
/// temporal coherence and (optionally) the CRLB / closure-phase quality layers.
fn link_and_compress(
    combined: ArrayView3<Cf64>,
    ms: MiniStack,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<MinistackResult, &'static str> {
    let c = engine.covariance(combined, cfg.half_window, cfg.strides, None)?;
    let est = engine.estimate(
        c.view(),
        cfg.use_evd,
        cfg.beta,
        cfg.zero_correlation_threshold,
        ms.output_reference_idx as usize,
    );
    let cpx = est.cpx_phase.mapv(unit_phasor);
    // estimate_temp_coh wants (rows, cols, nslc); cpx is (nslc, rows, cols).
    let temp_coh = estimate_temp_coh(cpx.view().permuted_axes([1, 2, 0]), c.view());
    let compressed = compress(
        combined,
        cpx.view(),
        ms.num_compressed,
        Some(ms.compressed_reference_idx as usize),
    );
    Ok(MinistackResult {
        cpx,
        compressed,
        temp_coh,
        crlb_sigma: cfg.compute_crlb.then(|| crlb_real_dates(c.view(), ms, cfg)),
        closure_phase: cfg
            .compute_closure_phase
            .then(|| estimate_closure_phases(c.view())),
    })
}

/// CRLB σ for the ministack's **real** dates only (drops carried compressed
/// layers), matching the phase-history concatenation. `num_looks` is dolphin's
/// conservative `sqrt(half_y · half_x)`; the CRLB reference is the last
/// compressed date (dolphin's `max(first_real_slc_idx − 1, 0)`).
fn crlb_real_dates(c: ArrayView4<Cf64>, ms: MiniStack, cfg: &SequentialConfig) -> Array3<f64> {
    let num_looks = (cfg.half_window.y as f64 * cfg.half_window.x as f64).sqrt();
    let reference_idx = ms.num_compressed.saturating_sub(1);
    let sigma = estimate_crlb(
        c,
        cfg.beta,
        cfg.zero_correlation_threshold,
        reference_idx,
        num_looks,
    );
    sigma.slice(s![ms.num_compressed.., .., ..]).to_owned()
}

/// Unit-magnitude phasor `exp(j∠z)` (dolphin's `exp(1j*angle(cpx_phase))`).
fn unit_phasor(z: Cf64) -> Cf64 {
    Cf64::from_polar(1.0, z.arg())
}

#[cfg(test)]
mod tests {
    use super::{stitch_temp_coh, Array2};

    /// On all-finite layers the stitch is a plain mean (prior parity preserved).
    #[test]
    fn stitch_is_plain_mean_when_finite() {
        let layers = [
            Array2::from_elem((1, 2), 0.8),
            Array2::from_elem((1, 2), 0.6),
        ];
        let out = stitch_temp_coh(&layers);
        assert!((out[(0, 0)] - 0.7).abs() < 1e-12);
    }

    /// A pixel masked (NaN) in one ministack averages only the finite ones —
    /// dolphin's `numpy.nanmean`, not a zero-diluted mean. The old `sum/len`
    /// would have poisoned the pixel to NaN (or, with zeros, halved it).
    #[test]
    fn stitch_skips_nan_per_pixel() {
        let mut a = Array2::from_elem((1, 1), 0.9);
        let b = Array2::from_elem((1, 1), f64::NAN);
        let out = stitch_temp_coh(&[a.clone(), b]);
        assert!((out[(0, 0)] - 0.9).abs() < 1e-12, "finite-only mean");

        // All-NaN stays NaN (nanmean of an all-NaN slice).
        a[(0, 0)] = f64::NAN;
        let allnan = stitch_temp_coh(&[a.clone(), a]);
        assert!(allnan[(0, 0)].is_nan(), "all-NaN pixel stays NaN");
    }
}
