//! Quality layers: temporal coherence (`metrics.py`) and compressed SLC
//! (`_compress.py`).
//!
//! Temporal coherence measures how well the linked phase reproduces the
//! observed interferometric phases: `|Σ_{i<j} e^{j(∠C_ij − (θ_i−θ_j))}| / N_pairs`
//! (equal weights, dolphin's default). The compressed SLC projects the stack
//! onto the linked phase: magnitude from the mean amplitude, phase from
//! `∠Σ_k z_k · conj(θ_k)`.
//!
//! CRLB uncertainty and sequential closure phase live in sibling modules
//! [`crate::crlb`] and [`crate::closure`] (validated against the v0.42.0 oracle).

use dolphin_core::Cf64;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Compression state retained for source-influence replay.
#[derive(Debug, Clone)]
pub struct CompressionReplayGrid {
    /// Production compressed SLC values on the native grid.
    pub compressed: Array2<Cf64>,
    /// Complex projection accumulator before phase normalization.
    pub projection: Array2<Cf64>,
    /// Mean included raw-SLC amplitude.
    pub mean_amplitude: Array2<f64>,
    /// Fixed compression-branch status per native pixel.
    pub status: Array2<CompressionReplayStatus>,
}

/// Fixed compression replay status at one native pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionReplayStatus {
    /// Projection and amplitude branches are differentiable.
    Valid,
    /// The fixed native validity mask excludes this pixel.
    Masked,
    /// An included raw sample or linked phase was non-finite.
    NonFiniteState,
    /// An included raw sample had zero amplitude within tolerance.
    ZeroIncludedAmplitude,
    /// The projection accumulator vanished within tolerance.
    ZeroProjection,
    /// The production zero-phase nodata sentinel was active or unstable.
    NodataBranch,
}

/// One full complex compression Jacobian-vector product.
#[derive(Debug, Clone, Copy)]
pub struct CompressionJvp {
    /// Baseline compressed complex value.
    pub value: Cf64,
    /// Complex compressed-value direction, including amplitude change.
    pub direction: Cf64,
    /// Baseline complex projection accumulator.
    pub projection: Cf64,
    /// Complex projection-accumulator direction.
    pub projection_direction: Cf64,
    /// Baseline mean amplitude.
    pub mean_amplitude: f64,
    /// Mean-amplitude direction.
    pub mean_amplitude_direction: f64,
}

/// Failure while capturing or differentiating compressed-SLC replay state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionJvpError {
    /// Input vectors had different lengths or were empty.
    ShapeMismatch,
    /// A raw sample, linked phase, or direction was NaN or infinite.
    NonFiniteState,
    /// An included raw sample had zero amplitude within branch tolerance.
    ZeroIncludedAmplitude,
    /// The projection accumulator vanished within branch tolerance.
    ZeroProjection,
    /// The production zero-phase nodata sentinel was active or unstable.
    NodataBranch,
    /// The resulting compression derivative was NaN or infinite.
    NonFiniteDerivative,
}

impl Display for CompressionJvpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::ShapeMismatch => "compression JVP shape mismatch",
            Self::NonFiniteState => "compression JVP state is non-finite",
            Self::ZeroIncludedAmplitude => "compression includes a zero-amplitude source",
            Self::ZeroProjection => "compression projection vanishes",
            Self::NodataBranch => "compression zero-phase nodata branch is active",
            Self::NonFiniteDerivative => "compression derivative is non-finite",
        };
        f.write_str(message)
    }
}

impl Error for CompressionJvpError {}

/// Average coherence magnitude for each SLC date in one square coherence matrix.
///
/// This preserves dolphin v0.35.0's bounded internal `abs(C).mean(axis=3)`
/// values, including the diagonal. Dolphin's public `avg_coh` applies an
/// `argmax` afterward and is a reference-date index, not a coherence value.
#[must_use]
pub fn average_coherence_per_date(c: ArrayView2<Cf64>) -> Array1<f64> {
    let n = c.nrows();
    debug_assert_eq!(n, c.ncols(), "coherence matrix must be square");
    Array1::from_iter((0..n).map(|i| c.row(i).iter().map(|z| z.norm()).sum::<f64>() / n as f64))
}

/// Per-date average coherence over `(rows, cols, nslc, nslc)` matrices.
///
/// Returns a band-major `(nslc, rows, cols)` array.
#[must_use]
pub fn estimate_average_coherence(c_arrays: ArrayView4<Cf64>) -> Array3<f64> {
    let (rows, cols, nslc, _) = c_arrays.dim();
    Array3::from_shape_fn((nslc, rows, cols), |(date, r, col)| {
        average_coherence_per_date(c_arrays.slice(s![r, col, .., ..]))[date]
    })
}

/// Temporal coherence per pixel from the linked phase and coherence matrices.
///
/// `cpx_phase` is `(rows, cols, nslc)` (dolphin's pre-moveaxis layout);
/// `c_arrays` is `(rows, cols, nslc, nslc)`. Returns `(rows, cols)`.
#[must_use]
pub fn estimate_temp_coh(cpx_phase: ArrayView3<Cf64>, c_arrays: ArrayView4<Cf64>) -> Array2<f64> {
    let (rows, cols, _) = cpx_phase.dim();
    let values: Vec<f64> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let (r, c) = (idx / cols, idx % cols);
            temp_coh_single(
                cpx_phase.slice(s![r, c, ..]),
                c_arrays.slice(s![r, c, .., ..]),
            )
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("temp_coh shape")
}

/// Temporal coherence for one pixel (equal weights, upper triangle).
pub(crate) fn temp_coh_single(phase: ArrayView1<Cf64>, c: ArrayView2<Cf64>) -> f64 {
    let n = phase.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let sum: Cf64 = pairs.iter().map(|&(i, j)| pair_diff(phase, c, i, j)).sum();
    nan_to_num(sum.norm() / pairs.len() as f64)
}

/// Unit phasor for the residual between observed and reformed ifg phase at `(i, j)`.
fn pair_diff(phase: ArrayView1<Cf64>, c: ArrayView2<Cf64>, i: usize, j: usize) -> Cf64 {
    let reformed = (phase[i] * phase[j].conj()).arg();
    Cf64::from_polar(1.0, c[(i, j)].arg() - reformed)
}

/// Compressed SLC: project the stack onto the linked phase (port of `compress`).
///
/// `slc_stack` is `(nslc, rows, cols)`; `pl_cpx_phase` is `(nslc, out_rows,
/// out_cols)` (upsampled to full resolution). `first_real_slc_idx` excludes
/// leading compressed layers from the projection; `reference_idx` optionally
/// re-references the phase first. Returns the compressed SLC `(rows, cols)`.
#[must_use]
pub fn compress(
    slc_stack: ArrayView3<Cf64>,
    pl_cpx_phase: ArrayView3<Cf64>,
    first_real_slc_idx: usize,
    reference_idx: Option<usize>,
) -> Array2<Cf64> {
    let (_, rows, cols) = slc_stack.dim();
    let referenced = rereference(pl_cpx_phase, reference_idx);
    let upsampled = upsample_nearest(referenced.view(), (rows, cols));

    let values: Vec<Cf64> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let (r, c) = (idx / cols, idx % cols);
            compress_pixel(slc_stack, upsampled.view(), first_real_slc_idx, (r, c))
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values).expect("compressed shape")
}

/// Compress a stack while retaining the exact projection and amplitude replay state.
///
/// This follows the same arithmetic and native-grid mapping as [`compress`].
/// A masked or nondifferentiable pixel receives an explicit per-pixel status;
/// it does not discard valid replay state elsewhere in the grid.
///
/// # Errors
/// Returns an error for invalid array shapes, date/reference bounds, or branch
/// tolerance. Numeric pixel failures are reported in the returned status grid.
pub fn compress_with_replay(
    slc_stack: ArrayView3<Cf64>,
    pl_cpx_phase: ArrayView3<Cf64>,
    first_real_slc_idx: usize,
    reference_idx: Option<usize>,
    native_validity: ArrayView2<bool>,
    branch_tolerance: f64,
) -> Result<CompressionReplayGrid, CompressionJvpError> {
    let (nslc, rows, cols) = slc_stack.dim();
    if nslc == 0
        || first_real_slc_idx >= nslc
        || pl_cpx_phase.dim().0 != nslc
        || reference_idx.is_some_and(|index| index >= nslc)
        || native_validity.dim() != (rows, cols)
        || !branch_tolerance.is_finite()
        || branch_tolerance < 0.0
    {
        return Err(CompressionJvpError::ShapeMismatch);
    }
    let referenced = rereference(pl_cpx_phase, reference_idx);
    let upsampled = upsample_nearest(referenced.view(), (rows, cols));
    let states: Vec<(CompressionState, CompressionReplayStatus)> = (0..rows * cols)
        .into_par_iter()
        .map(|index| {
            let pixel = (index / cols, index % cols);
            let state = compression_state(slc_stack, upsampled.view(), first_real_slc_idx, pixel);
            let status = match native_validity[pixel] {
                false => CompressionReplayStatus::Masked,
                true => compression_replay_status(
                    slc_stack,
                    upsampled.view(),
                    first_real_slc_idx,
                    pixel,
                    branch_tolerance,
                    state,
                ),
            };
            (state, status)
        })
        .collect();
    let compressed = Array2::from_shape_vec(
        (rows, cols),
        states
            .iter()
            .map(|(state, status)| match status {
                CompressionReplayStatus::Masked => Cf64::new(f64::NAN, f64::NAN),
                _ => state.value,
            })
            .collect(),
    )
    .map_err(|_| CompressionJvpError::ShapeMismatch)?;
    let projection = Array2::from_shape_vec(
        (rows, cols),
        states.iter().map(|(state, _)| state.projection).collect(),
    )
    .map_err(|_| CompressionJvpError::ShapeMismatch)?;
    let mean_amplitude = Array2::from_shape_vec(
        (rows, cols),
        states
            .iter()
            .map(|(state, _)| state.mean_amplitude)
            .collect(),
    )
    .map_err(|_| CompressionJvpError::ShapeMismatch)?;
    let status = Array2::from_shape_vec(
        (rows, cols),
        states.iter().map(|(_, status)| *status).collect(),
    )
    .map_err(|_| CompressionJvpError::ShapeMismatch)?;
    Ok(CompressionReplayGrid {
        compressed,
        projection,
        mean_amplitude,
        status,
    })
}

/// Differentiate one compressed complex pixel in raw-sample and phase directions.
///
/// `linked_phase_direction` contains angular directions `d phi`; the linked
/// phase values themselves are the complex phasors used by production.
///
/// # Errors
/// Returns an error for mismatched/non-finite inputs, a zero included
/// amplitude/projection, the zero-phase nodata branch, or a non-finite result.
pub fn compress_pixel_jvp(
    samples: ArrayView1<Cf64>,
    linked_phase: ArrayView1<Cf64>,
    sample_direction: ArrayView1<Cf64>,
    linked_phase_direction: ArrayView1<f64>,
    branch_tolerance: f64,
) -> Result<CompressionJvp, CompressionJvpError> {
    let n = samples.len();
    if n == 0
        || linked_phase.len() != n
        || sample_direction.len() != n
        || linked_phase_direction.len() != n
        || !branch_tolerance.is_finite()
        || branch_tolerance < 0.0
    {
        return Err(CompressionJvpError::ShapeMismatch);
    }
    if samples
        .iter()
        .chain(linked_phase.iter())
        .chain(sample_direction.iter())
        .any(|value| !value.is_finite())
        || linked_phase_direction
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(CompressionJvpError::NonFiniteState);
    }
    if samples.iter().any(|value| value.norm() <= branch_tolerance) {
        return Err(CompressionJvpError::ZeroIncludedAmplitude);
    }

    let projection: Cf64 = samples
        .iter()
        .zip(linked_phase.iter())
        .map(|(&sample, &phase)| sample * phase.conj())
        .sum();
    let projection_direction: Cf64 = samples
        .iter()
        .zip(linked_phase.iter())
        .zip(sample_direction.iter().zip(linked_phase_direction.iter()))
        .map(|((&sample, &phase), (&delta_sample, &delta_phase))| {
            phase.conj() * (delta_sample - Cf64::i() * sample * delta_phase)
        })
        .sum();
    let projection_norm = projection.norm();
    if projection_norm <= branch_tolerance {
        return Err(CompressionJvpError::ZeroProjection);
    }
    let projection_phase = projection.arg();
    if projection_phase.abs() <= branch_tolerance {
        return Err(CompressionJvpError::NodataBranch);
    }
    let mean_amplitude = samples.iter().map(|value| value.norm()).sum::<f64>() / n as f64;
    let mean_amplitude_direction = samples
        .iter()
        .zip(sample_direction.iter())
        .map(|(&sample, &delta)| (sample.conj() * delta).re / sample.norm())
        .sum::<f64>()
        / n as f64;
    let unit_projection = projection / projection_norm;
    let direction = unit_projection * mean_amplitude_direction
        + mean_amplitude
            * (projection_direction / projection_norm
                - projection * (projection.conj() * projection_direction).re
                    / projection_norm.powi(3));
    let value = unit_projection * mean_amplitude;
    if !projection_direction.is_finite()
        || !mean_amplitude_direction.is_finite()
        || !direction.is_finite()
        || !value.is_finite()
    {
        return Err(CompressionJvpError::NonFiniteDerivative);
    }
    Ok(CompressionJvp {
        value,
        direction,
        projection,
        projection_direction,
        mean_amplitude,
        mean_amplitude_direction,
    })
}

/// Optionally re-reference the linked phase to `reference_idx`.
fn rereference(pl: ArrayView3<Cf64>, reference_idx: Option<usize>) -> Array3<Cf64> {
    let Some(ref_idx) = reference_idx else {
        return pl.to_owned();
    };
    let reference = pl.slice(s![ref_idx, .., ..]).to_owned();
    Array3::from_shape_fn(pl.dim(), |(t, r, c)| {
        pl[(t, r, c)] * reference[(r, c)].conj()
    })
}

/// One compressed-SLC pixel: mean magnitude × `exp(j ∠Σ z_k conj(θ_k))`.
fn compress_pixel(
    slc_stack: ArrayView3<Cf64>,
    upsampled: ArrayView3<Cf64>,
    first: usize,
    pixel: (usize, usize),
) -> Cf64 {
    compression_state(slc_stack, upsampled, first, pixel).value
}

#[derive(Debug, Clone, Copy)]
struct CompressionState {
    value: Cf64,
    projection: Cf64,
    mean_amplitude: f64,
}

fn compression_state(
    slc_stack: ArrayView3<Cf64>,
    upsampled: ArrayView3<Cf64>,
    first: usize,
    pixel: (usize, usize),
) -> CompressionState {
    let (nslc, r, c) = (slc_stack.dim().0, pixel.0, pixel.1);
    let projection: Cf64 = (first..nslc)
        .map(|t| finite_or_zero(slc_stack[(t, r, c)] * upsampled[(t, r, c)].conj()))
        .sum();
    let mag_sum: f64 = (first..nslc).map(|t| slc_stack[(t, r, c)].norm()).sum();
    let count = (nslc - first) as f64;
    let phase = projection.arg();
    let mean = if phase == 0.0 {
        f64::NAN
    } else {
        mag_sum / count
    };
    CompressionState {
        value: Cf64::from_polar(mean, phase),
        projection,
        mean_amplitude: mag_sum / count,
    }
}

fn compression_replay_status(
    slc_stack: ArrayView3<Cf64>,
    upsampled: ArrayView3<Cf64>,
    first: usize,
    pixel: (usize, usize),
    branch_tolerance: f64,
    state: CompressionState,
) -> CompressionReplayStatus {
    let (nslc, row, column) = (slc_stack.dim().0, pixel.0, pixel.1);
    for date in first..nslc {
        let sample = slc_stack[(date, row, column)];
        let phase = upsampled[(date, row, column)];
        if !sample.is_finite() || !phase.is_finite() {
            return CompressionReplayStatus::NonFiniteState;
        }
        if sample.norm() <= branch_tolerance {
            return CompressionReplayStatus::ZeroIncludedAmplitude;
        }
    }
    if state.projection.norm() <= branch_tolerance {
        return CompressionReplayStatus::ZeroProjection;
    }
    if state.projection.arg().abs() <= branch_tolerance {
        return CompressionReplayStatus::NodataBranch;
    }
    if !state.value.is_finite() || !state.mean_amplitude.is_finite() {
        return CompressionReplayStatus::NonFiniteState;
    }
    CompressionReplayStatus::Valid
}

/// Replace a non-finite complex value with zero (dolphin's `nansum` skip).
fn finite_or_zero(z: Cf64) -> Cf64 {
    match z.is_finite() {
        true => z,
        false => Cf64::new(0.0, 0.0),
    }
}

/// Nearest-neighbor upsample of `(nslc, in_rows, in_cols)` to `(out_rows, out_cols)`
/// by integer block repeat (port of `utils.upsample_nearest`).
fn upsample_nearest(arr: ArrayView3<Cf64>, output_shape: (usize, usize)) -> Array3<Cf64> {
    let (nslc, in_rows, in_cols) = arr.dim();
    let (out_rows, out_cols) = output_shape;
    if (in_rows, in_cols) == (out_rows, out_cols) {
        return arr.to_owned();
    }
    let row_looks = (out_rows / in_rows).max(1);
    let col_looks = (out_cols / in_cols).max(1);
    Array3::from_shape_fn((nslc, out_rows, out_cols), |(t, r, c)| {
        arr[(
            t,
            (r / row_looks).min(in_rows - 1),
            (c / col_looks).min(in_cols - 1),
        )]
    })
}

/// Replace NaN/±inf with 0.
fn nan_to_num(v: f64) -> f64 {
    match v.is_finite() {
        true => v,
        false => 0.0,
    }
}
