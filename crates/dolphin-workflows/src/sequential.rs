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

use dolphin_core::config::{CompressedSlcPlan, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_io::CovarianceOperatorBlock;
use dolphin_phaselink::{
    compress, compress_with_replay, AverageCoherenceAggregate, CompressionReplayGrid,
    ComputeEngine, FusedParams, PhaseReplayGrid, ResolvedBackend,
};
use dolphin_shp::{estimate_neighbors_glrt, estimate_neighbors_ks};
use dolphin_stack::{MiniStack, MiniStackPlanner};
use ndarray::{concatenate, s, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis};

pub use crate::sequential_covariance::SequentialCovarianceCaptureRequest;
use crate::sequential_covariance::{
    build_covariance_operator_block, ReplayBackend, ReplayExecutionScope, SequentialReplayError,
    SequentialReplayTopology,
};

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
    /// Produce the distinct phase-linking-coherence layer from dolphin's
    /// per-date average coherence magnitudes.
    pub compute_average_coherence: bool,
    /// Statistical test selecting the SHP neighbors covariance averages over.
    /// [`ShpMethod::Rect`] uses the full rectangular window.
    pub shp_method: ShpMethod,
    /// Significance level (false-alarm probability) for the SHP test.
    pub shp_alpha: f64,
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
    /// Mean coherence-matrix magnitude across real acquisition dates,
    /// `(out_rows, out_cols)`. `None` unless requested.
    pub phase_linking_coherence: Option<Array2<f64>>,
    /// Per-date CRLB σ (radians), `(nslc, out_rows, out_cols)` — real dates only,
    /// concatenated across ministacks. `None` when `compute_crlb` is off.
    pub crlb_sigma: Option<Array3<f64>>,
    /// Per-ministack nearest-neighbour closure phase (radians), band-major,
    /// concatenated across ministacks. `None` when `compute_closure_phase` is off.
    pub closure_phase: Option<Array3<f64>>,
    /// Output-grid validity after reducing an optional native layover/shadow
    /// mask. A stride cell is valid when any native pixel in it is valid.
    pub validity_mask: Array2<bool>,
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
    /// Native semantic validity used by the state-producing run. `None` means
    /// the unmasked API; `Some` retains even an all-valid configured mask so an
    /// update cannot mix masked and unmasked state.
    native_validity: Option<Array2<bool>>,
    /// Compressed SLC of each sealed ministack, in order (the carry-forward).
    sealed_compressed: Vec<Array2<Cf64>>,
    /// Per-real-date linked phase of each sealed ministack.
    sealed_phases: Vec<Array3<Cf64>>,
    /// Temporal coherence of each sealed ministack (kept per-ministack so the
    /// cross-ministack `nanmean` stitch stays exact under incremental updates).
    sealed_temp_coh: Vec<Array2<f64>>,
    /// Finite sum/count aggregates for sealed ministacks' real-date coherence.
    sealed_average_coherence: Vec<AverageCoherenceAggregate>,
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
    average_coherence: Vec<AverageCoherenceAggregate>,
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
    valid_mask: Option<ArrayView2<bool>>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<Drive, &'static str> {
    let mut carry: Vec<Array2<Cf64>> = seed_compressed.to_vec();
    let mut out = Drive {
        compressed: Vec::new(),
        phases: Vec::new(),
        temp_coh: Vec::new(),
        average_coherence: Vec::new(),
        crlb: Vec::new(),
        closure: Vec::new(),
    };
    for &ms in plans {
        let combined = assemble(&carry, real_stack, ms);
        let r = link_and_compress(combined.view(), ms, valid_mask, cfg, engine)?;
        out.phases
            .push(r.cpx.slice(s![ms.num_compressed.., .., ..]).to_owned());
        carry.push(r.compressed.clone());
        out.compressed.push(r.compressed);
        out.temp_coh.push(r.temp_coh);
        if let Some(average) = r.average_coherence {
            out.average_coherence.push(average);
        }
        if let Some(s) = r.crlb_sigma {
            out.crlb.push(s);
        }
        if let Some(s) = r.closure_phase {
            out.closure.push(s);
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn drive_with_covariance_capture<F>(
    plans: &[MiniStack],
    real_stack: ArrayView3<Cf64>,
    seed_compressed: &[Array2<Cf64>],
    valid_mask: Option<ArrayView2<bool>>,
    fixed_validity: ArrayView2<bool>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
    topology: &SequentialReplayTopology,
    request: &SequentialCovarianceCaptureRequest,
    emit: &mut F,
) -> Result<Drive, SequentialReplayError>
where
    F: FnMut(CovarianceOperatorBlock) -> Result<(), &'static str>,
{
    let mut carry: Vec<Array2<Cf64>> = seed_compressed.to_vec();
    let mut out = Drive {
        compressed: Vec::new(),
        phases: Vec::new(),
        temp_coh: Vec::new(),
        average_coherence: Vec::new(),
        crlb: Vec::new(),
        closure: Vec::new(),
    };
    for &ministack in plans {
        let combined = assemble(&carry, real_stack, ministack);
        let captured = link_and_compress_with_covariance_capture(
            combined.view(),
            ministack,
            valid_mask,
            fixed_validity,
            cfg,
            engine,
            request.branch_tolerance,
        )?;
        let block = build_covariance_operator_block(
            topology,
            request,
            ministack,
            combined.view(),
            captured.result.cpx.view(),
            &captured.phase,
            &captured.compression,
            cfg.use_evd,
        )?;
        emit(block).map_err(SequentialReplayError::Execution)?;

        let result = captured.result;
        out.phases.push(
            result
                .cpx
                .slice(s![ministack.num_compressed.., .., ..])
                .to_owned(),
        );
        carry.push(result.compressed.clone());
        out.compressed.push(result.compressed);
        out.temp_coh.push(result.temp_coh);
        if let Some(average) = result.average_coherence {
            out.average_coherence.push(average);
        }
        if let Some(sigma) = result.crlb_sigma {
            out.crlb.push(sigma);
        }
        if let Some(closure) = result.closure_phase {
            out.closure.push(closure);
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
    average_coherence: &[AverageCoherenceAggregate],
    crlb: Vec<Array3<f64>>,
    closure: Vec<Array3<f64>>,
    validity_mask: Array2<bool>,
) -> Result<SequentialOutput, &'static str> {
    let views: Vec<ArrayView3<Cf64>> = phases.iter().map(Array3::view).collect();
    let cpx_phase = concatenate(Axis(0), &views).map_err(|_| "phase-history concat failed")?;
    Ok(SequentialOutput {
        cpx_phase,
        compressed_slcs: compressed,
        temporal_coherence: stitch_temp_coh(temp_coh),
        phase_linking_coherence: finish_average_coherence(average_coherence),
        crlb_sigma: concat_bands(crlb)?,
        closure_phase: concat_bands(closure)?,
        validity_mask,
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

/// Run the sequential estimator with a native-grid layover/shadow validity
/// mask (`true` = valid). Invalid samples are excluded before covariance; an
/// output stride cell is invalid only when all native pixels in it are invalid.
///
/// # Errors
/// Returns `Err` if the mask grid differs from `slc_stack`, planning fails, or
/// a covariance window exceeds the stack.
pub fn run_sequential_masked(
    slc_stack: ArrayView3<Cf64>,
    valid_mask: ArrayView2<bool>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<SequentialOutput, &'static str> {
    Ok(run_sequential_resumable_masked(slc_stack, valid_mask, cfg, engine)?.0)
}

/// Run the fixed CPU/f64 Rect path and stream one replay operator block as
/// each ministack completes.
///
/// Legacy sequential entry points do not invoke this path. The callback sees
/// full local replay grids (including tile halo) with a distinct owned output
/// rectangle in each block.
///
/// # Errors
/// Returns a checked scope/identity/topology error, a production replay-capture
/// failure, or a streaming sink failure.
pub fn run_sequential_with_covariance_capture<F>(
    slc_stack: ArrayView3<Cf64>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
    request: &SequentialCovarianceCaptureRequest,
    emit: F,
) -> Result<SequentialOutput, SequentialReplayError>
where
    F: FnMut(CovarianceOperatorBlock) -> Result<(), &'static str>,
{
    run_sequential_with_covariance_capture_impl(slc_stack, None, cfg, engine, request, emit)
}

/// Run masked sequential phase linking with the same immutable validity mask
/// captured by the replay operator.
///
/// # Errors
/// Returns a checked scope/identity/topology error, a production replay-capture
/// failure, or a streaming sink failure.
pub fn run_sequential_masked_with_covariance_capture<F>(
    slc_stack: ArrayView3<Cf64>,
    valid_mask: ArrayView2<bool>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
    request: &SequentialCovarianceCaptureRequest,
    emit: F,
) -> Result<SequentialOutput, SequentialReplayError>
where
    F: FnMut(CovarianceOperatorBlock) -> Result<(), &'static str>,
{
    run_sequential_with_covariance_capture_impl(
        slc_stack,
        Some(valid_mask),
        cfg,
        engine,
        request,
        emit,
    )
}

fn run_sequential_with_covariance_capture_impl<F>(
    slc_stack: ArrayView3<Cf64>,
    native_validity: Option<ArrayView2<bool>>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
    request: &SequentialCovarianceCaptureRequest,
    mut emit: F,
) -> Result<SequentialOutput, SequentialReplayError>
where
    F: FnMut(CovarianceOperatorBlock) -> Result<(), &'static str>,
{
    let (_, rows, cols) = slc_stack.dim();
    let fixed_validity = match native_validity {
        Some(mask) if mask.dim() == (rows, cols) => mask.to_owned(),
        Some(_) => {
            return Err(SequentialReplayError::Invalid(
                "layover/shadow mask grid differs from the SLC stack",
            ))
        }
        None => Array2::from_elem((rows, cols), true),
    };
    let planner = planner_for(slc_stack.dim().0, cfg).map_err(SequentialReplayError::Execution)?;
    let plans = planner
        .plan(cfg.ministack_size)
        .map_err(SequentialReplayError::Execution)?;
    let output_shape = cfg.strides.out_shape((rows, cols));
    let output_area =
        output_shape
            .0
            .checked_mul(output_shape.1)
            .ok_or(SequentialReplayError::Invalid(
                "covariance capture output area overflows usize",
            ))?;
    let backend = match plans
        .iter()
        .any(|plan| engine.resolved(output_area, plan.size()) == ResolvedBackend::Gpu)
    {
        true => ReplayBackend::Gpu,
        false => ReplayBackend::CpuF64,
    };
    let scope = ReplayExecutionScope {
        enabled: true,
        backend,
        estimator_fallback: false,
        phase_bias_correction: false,
        strong_source_identity: true,
        stitched_burst_count: 1,
    };
    let namespace = request.namespace_for((rows, cols), output_shape, cfg.strides)?;
    let support_rows = cfg
        .half_window
        .y
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(SequentialReplayError::Invalid(
            "covariance capture support rows overflow usize",
        ))?;
    let support_cols = cfg
        .half_window
        .x
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(SequentialReplayError::Invalid(
            "covariance capture support columns overflow usize",
        ))?;
    let support_slots =
        support_rows
            .checked_mul(support_cols)
            .ok_or(SequentialReplayError::Invalid(
                "covariance capture support area overflows usize",
            ))?;
    let topology = SequentialReplayTopology::plan_identified(
        slc_stack.dim().0,
        (rows, cols),
        output_shape,
        support_slots,
        fixed_validity.view(),
        cfg,
        scope,
        namespace,
    )?;
    let valid_mask = native_validity
        .map(|mask| nontrivial_mask(mask, (rows, cols)))
        .transpose()
        .map_err(SequentialReplayError::Execution)?
        .flatten();
    let d = drive_with_covariance_capture(
        &plans,
        slc_stack,
        &[],
        valid_mask,
        fixed_validity.view(),
        cfg,
        engine,
        &topology,
        request,
        &mut emit,
    )?;
    build_output(
        &d.phases,
        d.compressed,
        &d.temp_coh,
        &d.average_coherence,
        d.crlb,
        d.closure,
        looked_validity(valid_mask, (rows, cols), cfg.strides),
    )
    .map_err(SequentialReplayError::Execution)
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
    run_sequential_resumable_impl(slc_stack, None, cfg, engine)
}

/// Run the masked sequential estimator and return resumable state.
///
/// # Errors
/// Returns `Err` if the mask grid differs from `slc_stack`, planning fails, or
/// a covariance window exceeds the stack.
pub fn run_sequential_resumable_masked(
    slc_stack: ArrayView3<Cf64>,
    valid_mask: ArrayView2<bool>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<(SequentialOutput, SequentialState), &'static str> {
    run_sequential_resumable_impl(slc_stack, Some(valid_mask), cfg, engine)
}

fn run_sequential_resumable_impl(
    slc_stack: ArrayView3<Cf64>,
    native_validity: Option<ArrayView2<bool>>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<(SequentialOutput, SequentialState), &'static str> {
    let valid_mask = match native_validity {
        Some(mask) => nontrivial_mask(mask, (slc_stack.dim().1, slc_stack.dim().2))?,
        None => None,
    };
    let planner = planner_for(slc_stack.dim().0, cfg)?;
    let plans = planner.plan(cfg.ministack_size)?;
    let d = drive(&plans, slc_stack, &[], valid_mask, cfg, engine)?;
    let output = build_output(
        &d.phases,
        d.compressed.clone(),
        &d.temp_coh,
        &d.average_coherence,
        d.crlb.clone(),
        d.closure.clone(),
        looked_validity(
            valid_mask,
            (slc_stack.dim().1, slc_stack.dim().2),
            cfg.strides,
        ),
    )?;
    let state = seal_state(&plans, slc_stack, cfg.ministack_size, native_validity, d);
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
    update_sequential_impl(state, new_slcs, None, cfg, engine)
}

/// Fold new acquisitions into masked resumable state using the same native
/// layover/shadow validity grid as the state-producing run.
///
/// # Errors
/// Returns `Err` if the mask or acquisition grid differs from the series,
/// `new_slcs` is empty, or planning/phase linking fails.
pub fn update_sequential_masked(
    state: &SequentialState,
    new_slcs: ArrayView3<Cf64>,
    valid_mask: ArrayView2<bool>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<(SequentialOutput, SequentialState), &'static str> {
    update_sequential_impl(state, new_slcs, Some(valid_mask), cfg, engine)
}

fn update_sequential_impl(
    state: &SequentialState,
    new_slcs: ArrayView3<Cf64>,
    native_validity: Option<ArrayView2<bool>>,
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
    let valid_mask = match (&state.native_validity, native_validity) {
        (None, None) => None,
        (Some(expected), Some(actual)) => {
            if actual.dim() != (rows, cols) {
                return Err("layover/shadow mask grid differs from the SLC stack");
            }
            if expected.view() != actual {
                return Err("update_sequential: layover/shadow validity differs from the series");
            }
            nontrivial_mask(actual, (rows, cols))?
        }
        _ => return Err("update_sequential: layover/shadow mask mode differs from the series"),
    };
    // Tail = open trailing real SLCs ++ the new acquisitions, owned.
    let tail = Array3::from_shape_fn((n_open + n_new, rows, cols), |(k, r, c)| match k < n_open {
        true => state.open_real_slcs[(k, r, c)],
        false => new_slcs[(k - n_open, r, c)],
    });
    let num_sealed = state.sealed_compressed.len();
    let tail_plans =
        planner_for(tail.dim().0, cfg)?.plan_with_offset(cfg.ministack_size, num_sealed)?;
    let d = drive(
        &tail_plans,
        tail.view(),
        &state.sealed_compressed,
        valid_mask,
        cfg,
        engine,
    )?;

    let phases = chain(&state.sealed_phases, &d.phases);
    let temp_coh = chain(&state.sealed_temp_coh, &d.temp_coh);
    let average_coherence = chain(&state.sealed_average_coherence, &d.average_coherence);
    let compressed = chain(&state.sealed_compressed, &d.compressed);
    let crlb = chain(&state.sealed_crlb, &d.crlb);
    let closure = chain(&state.sealed_closure, &d.closure);
    let output = build_output(
        &phases,
        compressed,
        &temp_coh,
        &average_coherence,
        crlb,
        closure,
        looked_validity(valid_mask, (rows, cols), cfg.strides),
    )?;
    let next = seal_state(
        &tail_plans,
        tail.view(),
        cfg.ministack_size,
        native_validity,
        d,
    )
    .with_sealed_prefix(state);
    Ok((output, next))
}

fn nontrivial_mask<'a>(
    valid_mask: ArrayView2<'a, bool>,
    expected_shape: (usize, usize),
) -> Result<Option<ArrayView2<'a, bool>>, &'static str> {
    if valid_mask.dim() != expected_shape {
        return Err("layover/shadow mask grid differs from the SLC stack");
    }
    Ok((!valid_mask.iter().all(|valid| *valid)).then_some(valid_mask))
}

fn looked_validity(
    valid_mask: Option<ArrayView2<bool>>,
    native_shape: (usize, usize),
    strides: Strides,
) -> Array2<bool> {
    let output_shape = strides.out_shape(native_shape);
    let Some(mask) = valid_mask else {
        return Array2::from_elem(output_shape, true);
    };
    Array2::from_shape_fn(output_shape, |(row, col)| {
        let rows = row * strides.y..(row + 1) * strides.y;
        let cols = col * strides.x..(col + 1) * strides.x;
        mask.slice(s![rows, cols]).iter().any(|valid| *valid)
    })
}

/// The [`MiniStackPlanner`] for a stack of `num_slc` real SLCs under `cfg`.
fn planner_for(num_slc: usize, cfg: &SequentialConfig) -> Result<MiniStackPlanner, &'static str> {
    Ok(MiniStackPlanner {
        num_slc,
        max_num_compressed: cfg.max_num_compressed,
        output_reference_idx: isize::try_from(cfg.output_reference_idx)
            .map_err(|_| "output reference index exceeds isize")?,
        compressed_slc_plan: cfg.compressed_slc_plan,
    })
}

/// Partition `drive` products into the sealed (full) ministacks vs the open
/// trailing one, building the resumable state for *this* (sub)sequence. The only
/// possibly-open ministack is the last `plan`; its raw real SLCs are sliced from
/// `real_stack` so a later update can recompute it exactly.
fn seal_state(
    plans: &[MiniStack],
    real_stack: ArrayView3<Cf64>,
    ministack_size: usize,
    native_validity: Option<ArrayView2<bool>>,
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
        native_validity: native_validity.map(|mask| mask.to_owned()),
        sealed_compressed: d.compressed[..sealed].to_vec(),
        sealed_phases: d.phases[..sealed].to_vec(),
        sealed_temp_coh: d.temp_coh[..sealed].to_vec(),
        sealed_average_coherence: take_prefix(&d.average_coherence, sealed),
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
        self.sealed_average_coherence = chain(
            &prev.sealed_average_coherence,
            &self.sealed_average_coherence,
        );
        self.sealed_crlb = chain(&prev.sealed_crlb, &self.sealed_crlb);
        self.sealed_closure = chain(&prev.sealed_closure, &self.sealed_closure);
        self
    }
}

/// Quality layers are empty when the layer is disabled; otherwise take the first
/// `n` (the sealed ministacks).
fn take_prefix<T: Clone>(v: &[T], n: usize) -> Vec<T> {
    match v.is_empty() {
        true => Vec::new(),
        false => v[..n].to_vec(),
    }
}

/// Collapse all real-date coherence aggregates to one 2D mean. `None` means the
/// optional metric was disabled, not a fabricated zero-valued layer.
fn finish_average_coherence(layers: &[AverageCoherenceAggregate]) -> Option<Array2<f64>> {
    let first = layers.first()?;
    Some(Array2::from_shape_fn(first.sum.dim(), |(r, c)| {
        let (sum, count) = layers.iter().fold((0.0, 0_u32), |(sum, count), layer| {
            (sum + layer.sum[(r, c)], count + layer.count[(r, c)])
        });
        match count {
            0 => f64::NAN,
            _ => sum / f64::from(count),
        }
    }))
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
    /// Real-date-only finite sum/count of average coherence.
    average_coherence: Option<AverageCoherenceAggregate>,
    /// CRLB σ for this ministack's real dates, `(num_real, out_rows, out_cols)`.
    crlb_sigma: Option<Array3<f64>>,
    /// Closure phase for this ministack, `(num_combined-2, out_rows, out_cols)`.
    closure_phase: Option<Array3<f64>>,
}

struct CapturedMinistackResult {
    result: MinistackResult,
    phase: PhaseReplayGrid,
    compression: CompressionReplayGrid,
}

#[allow(clippy::too_many_arguments)]
fn link_and_compress_with_covariance_capture(
    combined: ArrayView3<Cf64>,
    ms: MiniStack,
    valid_mask: Option<ArrayView2<bool>>,
    fixed_validity: ArrayView2<bool>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
    branch_tolerance: f64,
) -> Result<CapturedMinistackResult, SequentialReplayError> {
    let output_reference_idx = ms
        .resolved_output_reference_idx()
        .map_err(SequentialReplayError::Execution)?;
    let compressed_reference_idx = ms
        .resolved_compressed_reference_idx()
        .map_err(SequentialReplayError::Execution)?;
    let neighbors = shp_neighbors(combined.slice(s![ms.num_compressed.., .., ..]), cfg);
    let compute = |input: ArrayView3<Cf64>| {
        let mut replay = engine
            .link_with_source_replay(
                input,
                cfg.half_window,
                cfg.strides,
                neighbors.as_ref().map(Array4::view),
                fused_params(ms, cfg, output_reference_idx),
                fixed_validity,
                branch_tolerance,
            )
            .map_err(SequentialReplayError::Execution)?;
        let compression = compress_with_replay(
            input,
            replay.estimate.cpx_phase.view(),
            ms.num_compressed,
            Some(compressed_reference_idx),
            fixed_validity,
            branch_tolerance,
        )
        .map_err(|_| SequentialReplayError::Execution("compression replay capture failed"))?;
        if valid_mask.is_some() {
            let output_validity =
                looked_validity(Some(fixed_validity), fixed_validity.dim(), cfg.strides);
            replay
                .phase
                .apply_output_validity(output_validity.view())
                .map_err(SequentialReplayError::Execution)?;
        }
        let fused = replay.estimate;
        let mut result = MinistackResult {
            cpx: fused.cpx_phase,
            compressed: compression.compressed.clone(),
            temp_coh: fused.temporal_coherence,
            average_coherence: fused.average_coherence,
            crlb_sigma: fused
                .crlb_sigma
                .map(|sigma| sigma.slice(s![ms.num_compressed.., .., ..]).to_owned()),
            closure_phase: fused.closure_phase,
        };
        if let Some(mask) = valid_mask {
            let output_validity = looked_validity(Some(mask), mask.dim(), cfg.strides);
            apply_output_validity(&mut result, output_validity.view());
            mask_native_complex(&mut result.compressed, mask);
        }
        Ok(CapturedMinistackResult {
            result,
            phase: replay.phase,
            compression,
        })
    };
    match valid_mask {
        Some(mask) => compute(mask_stack(combined, mask).view()),
        None => compute(combined),
    }
}

/// Phase-link a combined ministack and compress it to one SLC, plus its
/// temporal coherence and (optionally) the CRLB / closure-phase quality layers.
fn link_and_compress(
    combined: ArrayView3<Cf64>,
    ms: MiniStack,
    valid_mask: Option<ArrayView2<bool>>,
    cfg: &SequentialConfig,
    engine: &ComputeEngine,
) -> Result<MinistackResult, &'static str> {
    let output_reference_idx = ms.resolved_output_reference_idx()?;
    let compressed_reference_idx = ms.resolved_compressed_reference_idx()?;
    // SHP neighbors are selected from the real acquisitions only; the carried
    // compressed SLCs are projections, not observations, so their amplitude
    // statistics would not describe the scatterer.
    let neighbors = shp_neighbors(combined.slice(s![ms.num_compressed.., .., ..]), cfg);
    let compute = |input: ArrayView3<Cf64>| {
        let fused = engine.link(
            input,
            cfg.half_window,
            cfg.strides,
            neighbors.as_ref().map(Array4::view),
            fused_params(ms, cfg, output_reference_idx),
        )?;
        let compressed = compress(
            input,
            fused.cpx_phase.view(),
            ms.num_compressed,
            Some(compressed_reference_idx),
        );
        Ok::<_, &'static str>((fused, compressed))
    };
    let (fused, compressed) = match valid_mask {
        Some(mask) => compute(mask_stack(combined, mask).view())?,
        None => compute(combined)?,
    };
    let mut result = MinistackResult {
        cpx: fused.cpx_phase,
        compressed,
        temp_coh: fused.temporal_coherence,
        average_coherence: fused.average_coherence,
        // CRLB is produced for the full combined stack; keep only the real
        // dates, matching the phase-history concatenation.
        crlb_sigma: fused
            .crlb_sigma
            .map(|sigma| sigma.slice(s![ms.num_compressed.., .., ..]).to_owned()),
        closure_phase: fused.closure_phase,
    };
    if let Some(mask) = valid_mask {
        let validity = looked_validity(Some(mask), mask.dim(), cfg.strides);
        apply_output_validity(&mut result, validity.view());
        mask_native_complex(&mut result.compressed, mask);
    }
    Ok(result)
}

fn mask_stack(stack: ArrayView3<Cf64>, valid_mask: ArrayView2<bool>) -> Array3<Cf64> {
    Array3::from_shape_fn(stack.dim(), |(date, row, col)| {
        match valid_mask[(row, col)] {
            true => stack[(date, row, col)],
            false => Cf64::new(f64::NAN, f64::NAN),
        }
    })
}

fn apply_output_validity(result: &mut MinistackResult, valid_mask: ArrayView2<bool>) {
    let invalid_complex = Cf64::new(f64::NAN, f64::NAN);
    for ((row, col), &valid) in valid_mask.indexed_iter() {
        if valid {
            continue;
        }
        result.cpx.slice_mut(s![.., row, col]).fill(invalid_complex);
        result.temp_coh[(row, col)] = f64::NAN;
        if let Some(average) = result.average_coherence.as_mut() {
            average.sum[(row, col)] = 0.0;
            average.count[(row, col)] = 0;
        }
        if let Some(sigma) = result.crlb_sigma.as_mut() {
            sigma.slice_mut(s![.., row, col]).fill(f64::NAN);
        }
        if let Some(closure) = result.closure_phase.as_mut() {
            closure.slice_mut(s![.., row, col]).fill(f64::NAN);
        }
    }
}

fn mask_native_complex(values: &mut Array2<Cf64>, valid_mask: ArrayView2<bool>) {
    ndarray::Zip::from(values)
        .and(valid_mask)
        .for_each(|value, &valid| {
            if !valid {
                *value = Cf64::new(f64::NAN, f64::NAN);
            }
        });
}

/// SHP neighbor mask for one ministack's real acquisitions, or `None` for the
/// full rectangular window ([`ShpMethod::Rect`], which keeps the unmasked
/// covariance kernel and so stays bit-identical to the pre-SHP output).
fn shp_neighbors(real: ArrayView3<Cf64>, cfg: &SequentialConfig) -> Option<Array4<bool>> {
    let nslc = real.dim().0;
    let amplitude = real.mapv(|z| z.norm());
    match cfg.shp_method {
        ShpMethod::Rect => None,
        ShpMethod::Glrt => {
            let mean = amplitude.mean_axis(Axis(0))?;
            let var = amplitude.var_axis(Axis(0), 0.0);
            Some(estimate_neighbors_glrt(
                mean.view(),
                var.view(),
                cfg.half_window,
                nslc,
                cfg.strides,
                cfg.shp_alpha,
            ))
        }
        ShpMethod::Ks => Some(estimate_neighbors_ks(
            amplitude.view(),
            cfg.half_window,
            cfg.strides,
            cfg.shp_alpha,
            false,
        )),
    }
}

/// Build the fused-pass parameters. `num_looks` is dolphin's conservative
/// `sqrt(half_y · half_x)`; the CRLB reference is the last compressed date
/// (dolphin's `max(first_real_slc_idx − 1, 0)`), which may differ from the
/// output reference.
fn fused_params(ms: MiniStack, cfg: &SequentialConfig, output_reference_idx: usize) -> FusedParams {
    FusedParams {
        use_evd: cfg.use_evd,
        beta: cfg.beta,
        zero_correlation_threshold: cfg.zero_correlation_threshold,
        reference_idx: output_reference_idx,
        compute_crlb: cfg.compute_crlb,
        crlb_reference_idx: ms.num_compressed.saturating_sub(1),
        num_looks: (cfg.half_window.y as f64 * cfg.half_window.x as f64).sqrt(),
        compute_closure: cfg.compute_closure_phase,
        compute_average_coherence: cfg.compute_average_coherence,
        average_coherence_start_idx: ms.num_compressed,
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
