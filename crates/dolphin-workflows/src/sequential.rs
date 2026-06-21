//! Sequential (Ansari et al. 2017) phase-linking loop — port of
//! `workflows/sequential.py`.
//!
//! [`MiniStackPlanner`] partitions the stack; each ministack is phase-linked
//! over `[carried compressed SLCs] ++ [real SLCs]`, then compressed to one SLC
//! carried into the next ministack. The per-real-date linked phases are
//! concatenated into the end-to-end phase history.
//!
//! ## NRT incremental updates
//! Because the loop is feed-forward — ministack `N` reads only the compressed
//! SLCs of ministacks `< N` and its own real SLCs, never future data — a
//! ministack that has filled to `ministack_size` ("sealed") never changes when
//! later acquisitions arrive. [`run_sequential_resumable`] returns a
//! [`SequentialState`] capturing the sealed ministacks' products plus the raw
//! SLCs of the still-open trailing ministack; [`update_sequential`] folds in new
//! acquisitions by re-phase-linking only the open ministack and any new ones,
//! producing a [`SequentialOutput`] **bit-identical** to a full rerun of the
//! extended stack (verified by `tests/nrt_incremental_contract.rs`). This is the
//! phase-linking-stage half of NRT; the non-causal downstream (ifg network →
//! unwrap → timeseries → velocity) recomputes from the updated phase history.

use dolphin_core::config::CompressedSlcPlan;
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::{compress, ComputeEngine, FusedParams};
use dolphin_stack::{MiniStack, MiniStackPlanner};
use ndarray::{concatenate, s, Array2, Array3, ArrayView3, Axis};

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

/// Persisted state for an NRT incremental update: the outputs of the **sealed**
/// (full) ministacks of a prior run, plus the raw real SLCs of the still-open
/// trailing ministack. Sequential phase-linking is feed-forward — a sealed
/// ministack never changes when later acquisitions arrive — so carrying this
/// state lets [`update_sequential`] fold in new SLCs without re-phase-linking the
/// sealed history, yielding a result **bit-identical** to a full rerun.
///
/// Opaque by design; obtain it from [`run_sequential_resumable`] and thread it
/// through [`update_sequential`]. The same [`SequentialConfig`] must be used
/// across the resumed sequence.
#[derive(Clone)]
pub struct SequentialState {
    /// Compressed SLC of each sealed ministack, in order (the carry-forward).
    sealed_compressed: Vec<Array2<Cf64>>,
    /// Per-real-date linked phase of each sealed ministack.
    sealed_phases: Vec<Array3<Cf64>>,
    /// Temporal coherence of each sealed ministack (kept per-ministack so the
    /// cross-ministack `nanmean` stitch stays exact under incremental updates).
    sealed_temp_coh: Vec<Array2<f64>>,
    /// Per-sealed-ministack CRLB σ layers (empty when CRLB is off).
    sealed_crlb: Vec<Array3<f64>>,
    /// Per-sealed-ministack closure-phase layers (empty when closure is off).
    sealed_closure: Vec<Array3<f64>>,
    /// Raw real SLCs of the open trailing ministack, `(n_open, rows, cols)`;
    /// `n_open = 0` when the prior run ended exactly on a ministack boundary.
    open_real_slcs: Array3<Cf64>,
}

/// Per-ministack products accumulated by [`drive`] over a (sub)sequence.
struct Drive {
    compressed: Vec<Array2<Cf64>>,
    phases: Vec<Array3<Cf64>>,
    temp_coh: Vec<Array2<f64>>,
    crlb: Vec<Array3<f64>>,
    closure: Vec<Array3<f64>>,
}

/// Phase-link + compress each planned ministack of `real_stack`, carrying the
/// compressed SLCs forward (seeded with `seed_compressed` from already-sealed
/// ministacks). Returns only the products of the ministacks it processed.
fn drive(
    plans: &[MiniStack],
    real_stack: ArrayView3<Cf64>,
    seed_compressed: &[Array2<Cf64>],
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<Drive, &'static str> {
    let mut carry: Vec<Array2<Cf64>> = seed_compressed.to_vec();
    let mut out = Drive {
        compressed: Vec::new(),
        phases: Vec::new(),
        temp_coh: Vec::new(),
        crlb: Vec::new(),
        closure: Vec::new(),
    };
    for &ms in plans {
        let combined = assemble(&carry, real_stack, ms);
        let r = link_and_compress(combined.view(), ms, cfg, engine)?;
        out.phases
            .push(r.cpx.slice(s![ms.num_compressed.., .., ..]).to_owned());
        carry.push(r.compressed.clone());
        out.compressed.push(r.compressed);
        out.temp_coh.push(r.temp_coh);
        if let Some(s) = r.crlb_sigma {
            out.crlb.push(s);
        }
        if let Some(s) = r.closure_phase {
            out.closure.push(s);
        }
    }
    Ok(out)
}

/// Assemble a [`SequentialOutput`] from the full per-ministack product lists
/// (sealed prefix already chained in by the caller).
fn build_output(
    phases: &[Array3<Cf64>],
    compressed: Vec<Array2<Cf64>>,
    temp_coh: &[Array2<f64>],
    crlb: Vec<Array3<f64>>,
    closure: Vec<Array3<f64>>,
) -> Result<SequentialOutput, &'static str> {
    let views: Vec<ArrayView3<Cf64>> = phases.iter().map(Array3::view).collect();
    let cpx_phase = concatenate(Axis(0), &views).map_err(|_| "phase-history concat failed")?;
    Ok(SequentialOutput {
        cpx_phase,
        compressed_slcs: compressed,
        temporal_coherence: stitch_temp_coh(temp_coh),
        crlb_sigma: concat_bands(crlb)?,
        closure_phase: concat_bands(closure)?,
    })
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
    Ok(run_sequential_resumable(slc_stack, cfg, engine)?.0)
}

/// Run the sequential estimator and also return the [`SequentialState`] needed to
/// fold in later acquisitions incrementally via [`update_sequential`]. The
/// [`SequentialOutput`] is identical to [`run_sequential`]'s.
///
/// # Errors
/// Returns `Err` if planning fails or a covariance window exceeds the stack.
pub fn run_sequential_resumable(
    slc_stack: ArrayView3<Cf64>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<(SequentialOutput, SequentialState), &'static str> {
    let planner = planner_for(slc_stack.dim().0, cfg);
    let plans = planner.plan(cfg.ministack_size)?;
    let d = drive(&plans, slc_stack, &[], cfg, engine)?;
    let output = build_output(
        &d.phases,
        d.compressed.clone(),
        &d.temp_coh,
        d.crlb.clone(),
        d.closure.clone(),
    )?;
    let state = seal_state(&plans, slc_stack, cfg.ministack_size, d);
    Ok((output, state))
}

/// Fold newly-arrived real SLCs into an existing sequential series. Only the open
/// trailing ministack and any new ministacks are phase-linked (carrying the
/// sealed compressed SLCs from `state`); the result is the same
/// [`SequentialOutput`] a full rerun of the extended stack would produce, and a
/// fresh [`SequentialState`] for the next update. `cfg` must match the run that
/// produced `state`.
///
/// # Errors
/// Returns `Err` if `new_slcs` is empty, its grid differs from `state`'s, or
/// planning / phase-linking fails.
pub fn update_sequential(
    state: &SequentialState,
    new_slcs: ArrayView3<Cf64>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<(SequentialOutput, SequentialState), &'static str> {
    let (n_open, rows, cols) = state.open_real_slcs.dim();
    let (n_new, nrows, ncols) = new_slcs.dim();
    if n_new == 0 {
        return Err("update_sequential: no new acquisitions");
    }
    if (nrows, ncols) != (rows, cols) {
        return Err("update_sequential: new SLC grid differs from the series");
    }
    // Tail = open trailing real SLCs ++ the new acquisitions, owned.
    let tail = Array3::from_shape_fn((n_open + n_new, rows, cols), |(k, r, c)| match k < n_open {
        true => state.open_real_slcs[(k, r, c)],
        false => new_slcs[(k - n_open, r, c)],
    });
    let num_sealed = state.sealed_compressed.len();
    let tail_plans =
        planner_for(tail.dim().0, cfg).plan_with_offset(cfg.ministack_size, num_sealed)?;
    let d = drive(
        &tail_plans,
        tail.view(),
        &state.sealed_compressed,
        cfg,
        engine,
    )?;

    let phases = chain(&state.sealed_phases, &d.phases);
    let temp_coh = chain(&state.sealed_temp_coh, &d.temp_coh);
    let compressed = chain(&state.sealed_compressed, &d.compressed);
    let crlb = chain(&state.sealed_crlb, &d.crlb);
    let closure = chain(&state.sealed_closure, &d.closure);
    let output = build_output(&phases, compressed, &temp_coh, crlb, closure)?;
    let next =
        seal_state(&tail_plans, tail.view(), cfg.ministack_size, d).with_sealed_prefix(state);
    Ok((output, next))
}

/// The [`MiniStackPlanner`] for a stack of `num_slc` real SLCs under `cfg`.
fn planner_for(num_slc: usize, cfg: &SequentialConfig) -> MiniStackPlanner {
    MiniStackPlanner {
        num_slc,
        max_num_compressed: cfg.max_num_compressed,
        output_reference_idx: cfg.output_reference_idx as isize,
        compressed_slc_plan: cfg.compressed_slc_plan,
    }
}

/// Partition `drive` products into the sealed (full) ministacks vs the open
/// trailing one, building the resumable state for *this* (sub)sequence. The only
/// possibly-open ministack is the last `plan`; its raw real SLCs are sliced from
/// `real_stack` so a later update can recompute it exactly.
fn seal_state(
    plans: &[MiniStack],
    real_stack: ArrayView3<Cf64>,
    ministack_size: usize,
    d: Drive,
) -> SequentialState {
    let (_, rows, cols) = real_stack.dim();
    let last = plans.last();
    let open = last.is_some_and(|ms| ms.num_real < ministack_size);
    let sealed = plans.len() - usize::from(open);
    let open_real_slcs = match (open, last) {
        (true, Some(ms)) => real_stack
            .slice(s![ms.real_start..ms.real_start + ms.num_real, .., ..])
            .to_owned(),
        _ => Array3::zeros((0, rows, cols)),
    };
    SequentialState {
        sealed_compressed: d.compressed[..sealed].to_vec(),
        sealed_phases: d.phases[..sealed].to_vec(),
        sealed_temp_coh: d.temp_coh[..sealed].to_vec(),
        sealed_crlb: take_prefix(&d.crlb, sealed),
        sealed_closure: take_prefix(&d.closure, sealed),
        open_real_slcs,
    }
}

impl SequentialState {
    /// Prepend a prior run's sealed products (the part `seal_state` didn't see in
    /// an incremental update) so the state describes the whole series.
    fn with_sealed_prefix(mut self, prev: &SequentialState) -> Self {
        self.sealed_compressed = chain(&prev.sealed_compressed, &self.sealed_compressed);
        self.sealed_phases = chain(&prev.sealed_phases, &self.sealed_phases);
        self.sealed_temp_coh = chain(&prev.sealed_temp_coh, &self.sealed_temp_coh);
        self.sealed_crlb = chain(&prev.sealed_crlb, &self.sealed_crlb);
        self.sealed_closure = chain(&prev.sealed_closure, &self.sealed_closure);
        self
    }
}

/// Quality layers are empty when the layer is disabled; otherwise take the first
/// `n` (the sealed ministacks).
fn take_prefix(v: &[Array3<f64>], n: usize) -> Vec<Array3<f64>> {
    match v.is_empty() {
        true => Vec::new(),
        false => v[..n].to_vec(),
    }
}

/// Concatenate two per-ministack product lists (sealed prefix ++ tail).
fn chain<T: Clone>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().chain(b).cloned().collect()
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
    let fused = engine.link(
        combined,
        cfg.half_window,
        cfg.strides,
        None,
        fused_params(ms, cfg),
    )?;
    let cpx = fused.cpx_phase;
    let compressed = compress(
        combined,
        cpx.view(),
        ms.num_compressed,
        Some(ms.compressed_reference_idx as usize),
    );
    // CRLB is produced for the full combined stack; keep only the real dates,
    // matching the phase-history concatenation (drops carried compressed layers).
    let crlb_sigma = fused
        .crlb_sigma
        .map(|s| s.slice(s![ms.num_compressed.., .., ..]).to_owned());
    Ok(MinistackResult {
        cpx,
        compressed,
        temp_coh: fused.temporal_coherence,
        crlb_sigma,
        closure_phase: fused.closure_phase,
    })
}

/// Build the fused-pass parameters. `num_looks` is dolphin's conservative
/// `sqrt(half_y · half_x)`; the CRLB reference is the last compressed date
/// (dolphin's `max(first_real_slc_idx − 1, 0)`), which may differ from the
/// output reference.
fn fused_params(ms: MiniStack, cfg: &SequentialConfig) -> FusedParams {
    FusedParams {
        use_evd: cfg.use_evd,
        beta: cfg.beta,
        zero_correlation_threshold: cfg.zero_correlation_threshold,
        reference_idx: ms.output_reference_idx as usize,
        compute_crlb: cfg.compute_crlb,
        crlb_reference_idx: ms.num_compressed.saturating_sub(1),
        num_looks: (cfg.half_window.y as f64 * cfg.half_window.x as f64).sqrt(),
        compute_closure: cfg.compute_closure_phase,
    }
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
