//! Sliding-window sample-coherence estimation (port of `covariance.py`).
//!
//! For each (strided) output pixel, a `(2*half.y+1) x (2*half.x+1)` window is
//! read from the stack (clamped inward at borders, matching JAX
//! `dynamic_slice`), flattened to `(nslc, nsamples)`, and reduced to the
//! normalized coherence matrix `C_ij = Σ z_i z_j* / sqrt(Σ|z_i|² · Σ|z_j|²)`.
//! Parallelized over output pixels with `rayon` — the Rust analogue of dolphin's
//! `vmap(vmap(f))`. All math in `Cf64`.

use dolphin_core::{Cf64, HalfWindow, Strides};
use ndarray::{s, Array1, Array2, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Amplitude floor below which a coherence entry is set to 0 (dolphin uses 1e-6).
const AMP_FLOOR: f64 = 1e-6;

/// One raw native-grid source location in a rectangular covariance window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NativeSourcePixel {
    /// Zero-based native row.
    pub row: usize,
    /// Zero-based native column.
    pub column: usize,
}

impl NativeSourcePixel {
    /// Construct a native source location.
    #[must_use]
    pub const fn new(row: usize, column: usize) -> Self {
        Self { row, column }
    }
}

/// Recomputable topology for the fixed rectangular covariance kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RectReplayDescriptor {
    /// Native source-grid shape.
    pub native_shape: (usize, usize),
    /// Strided output-grid shape.
    pub output_shape: (usize, usize),
    /// Rectangular covariance half-window.
    pub half_window: HalfWindow,
    /// Output strides.
    pub strides: Strides,
}

impl RectReplayDescriptor {
    /// Validate and construct a rectangular replay descriptor.
    ///
    /// # Errors
    /// Returns an error for zero strides or a window larger than the native grid.
    pub fn new(
        native_shape: (usize, usize),
        half_window: HalfWindow,
        strides: Strides,
    ) -> Result<Self, CovarianceReplayError> {
        if strides.y == 0 || strides.x == 0 {
            return Err(CovarianceReplayError::ZeroStride);
        }
        let window = (2 * half_window.y + 1, 2 * half_window.x + 1);
        if window.0 > native_shape.0 || window.1 > native_shape.1 {
            return Err(CovarianceReplayError::WindowLargerThanStack);
        }
        let output_shape = strides.out_shape(native_shape);
        if output_shape.0 == 0 || output_shape.1 == 0 {
            return Err(CovarianceReplayError::EmptyOutputGrid);
        }
        Ok(Self {
            native_shape,
            output_shape,
            half_window,
            strides,
        })
    }

    /// Enumerate one output pixel's row-major realized native support.
    ///
    /// This is topology-only: it does not load or inspect source samples.
    ///
    /// # Errors
    /// Returns an error for an invalid output coordinate or validity shape.
    pub fn source_pixels(
        self,
        output: (usize, usize),
        native_validity: ArrayView2<bool>,
    ) -> Result<Vec<NativeSourcePixel>, CovarianceReplayError> {
        if native_validity.dim() != self.native_shape {
            return Err(CovarianceReplayError::ValidityShapeMismatch);
        }
        if output.0 >= self.output_shape.0 || output.1 >= self.output_shape.1 {
            return Err(CovarianceReplayError::OutputOutOfBounds);
        }
        let row_start = window_origin_row(
            output.0,
            self.half_window,
            self.strides,
            self.native_shape.0,
        );
        let column_start = window_origin_col(
            output.1,
            self.half_window,
            self.strides,
            self.native_shape.1,
        );
        let window = (2 * self.half_window.y + 1, 2 * self.half_window.x + 1);
        Ok((row_start..row_start + window.0)
            .flat_map(|row| {
                (column_start..column_start + window.1)
                    .filter(move |&column| native_validity[(row, column)])
                    .map(move |column| NativeSourcePixel::new(row, column))
            })
            .collect())
    }

    /// Map a native compression pixel to its nearest repeated output pixel.
    ///
    /// This matches the integer repeat and edge clamp used by compression's
    /// linked-phase upsampling.
    ///
    /// # Errors
    /// Returns an error when the native source coordinate is out of bounds.
    pub fn nearest_output(
        self,
        source: NativeSourcePixel,
    ) -> Result<(usize, usize), CovarianceReplayError> {
        if source.row >= self.native_shape.0 || source.column >= self.native_shape.1 {
            return Err(CovarianceReplayError::NativeSourceOutOfBounds);
        }
        let row_looks = (self.native_shape.0 / self.output_shape.0).max(1);
        let column_looks = (self.native_shape.1 / self.output_shape.1).max(1);
        Ok((
            (source.row / row_looks).min(self.output_shape.0 - 1),
            (source.column / column_looks).min(self.output_shape.1 - 1),
        ))
    }
}

/// Exact numerator/coherence replay for one rectangular output pixel.
#[derive(Debug, Clone)]
pub struct RectPixelReplay {
    /// Descriptor used to regenerate the window and source IDs.
    pub descriptor: RectReplayDescriptor,
    /// Output-grid location being replayed.
    pub output: (usize, usize),
    /// Row-major native sources retained by the fixed validity mask.
    pub source_pixels: Vec<NativeSourcePixel>,
    /// Production-order Hermitian covariance numerator.
    pub numerator: Array2<Cf64>,
    /// Numerator normalized with the production amplitude-floor branch.
    pub coherence: Array2<Cf64>,
}

/// Failure while replaying or differentiating a rectangular covariance pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovarianceReplayError {
    /// A stride was zero.
    ZeroStride,
    /// The covariance window was larger than the native grid.
    WindowLargerThanStack,
    /// The strides yielded an empty output grid.
    EmptyOutputGrid,
    /// The requested output pixel was outside the strided grid.
    OutputOutOfBounds,
    /// The fixed native-validity mask did not match the native grid.
    ValidityShapeMismatch,
    /// A retained raw source sample was NaN or infinite.
    NonFiniteSource,
    /// A covariance numerator or direction contained NaN or infinity.
    NonFiniteState,
    /// A source direction had the wrong acquisition count.
    DirectionLengthMismatch,
    /// The loaded source-value matrix did not match the ordered support.
    SourceValueShapeMismatch,
    /// A source direction was NaN or infinite.
    NonFiniteDirection,
    /// The requested source was not in the realized fixed support.
    SourceOutsideSupport,
    /// A native source coordinate was outside the descriptor grid.
    NativeSourceOutOfBounds,
    /// The loaded support was not strictly row-major or repeated a source.
    SourceOrderMismatch,
    /// The loaded support contained a source outside the requested window.
    SourceOutsideWindow,
    /// A coherence denominator was at the amplitude-floor branch boundary.
    AmplitudeFloorBoundary,
    /// A replay matrix shape did not match its numerator.
    MatrixShapeMismatch,
    /// A replay derivative contained NaN or infinity.
    NonFiniteDerivative,
    /// The declared branch tolerance was negative or non-finite.
    InvalidBranchTolerance,
}

impl Display for CovarianceReplayError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::ZeroStride => "rectangular replay stride is zero",
            Self::WindowLargerThanStack => "covariance window larger than stack",
            Self::EmptyOutputGrid => "rectangular replay output grid is empty",
            Self::OutputOutOfBounds => "rectangular replay output is out of bounds",
            Self::ValidityShapeMismatch => "native validity shape mismatch",
            Self::NonFiniteSource => "retained native source is non-finite",
            Self::NonFiniteState => "coherence normalization state is non-finite",
            Self::DirectionLengthMismatch => "source direction length mismatch",
            Self::SourceValueShapeMismatch => "source-value matrix shape mismatch",
            Self::NonFiniteDirection => "source direction is non-finite",
            Self::SourceOutsideSupport => "source is outside realized rectangular support",
            Self::NativeSourceOutOfBounds => "native source is out of bounds",
            Self::SourceOrderMismatch => "source support is not strictly row-major",
            Self::SourceOutsideWindow => "source lies outside replay window",
            Self::AmplitudeFloorBoundary => "coherence amplitude-floor branch is unstable",
            Self::MatrixShapeMismatch => "coherence replay matrix shape mismatch",
            Self::NonFiniteDerivative => "coherence derivative is non-finite",
            Self::InvalidBranchTolerance => "coherence branch tolerance is invalid",
        };
        f.write_str(message)
    }
}

impl Error for CovarianceReplayError {}

/// Replay one Rect output pixel with the exact production accumulation order.
///
/// `native_validity` is fixed for the complete source replay. Invalid native
/// pixels are excluded from the source list and contribute zeros to every
/// numerator entry, matching the workflow's masked-stack path.
///
/// # Errors
/// Returns an error for invalid geometry, shape mismatch, an out-of-range
/// output, or a non-finite sample at a retained native source.
pub fn replay_rect_pixel_covariance(
    stack: ArrayView3<Cf64>,
    output: (usize, usize),
    half_window: HalfWindow,
    strides: Strides,
    native_validity: ArrayView2<bool>,
) -> Result<RectPixelReplay, CovarianceReplayError> {
    let (_, rows, columns) = stack.dim();
    let descriptor = RectReplayDescriptor::new((rows, columns), half_window, strides)?;
    if native_validity.dim() != (rows, columns) {
        return Err(CovarianceReplayError::ValidityShapeMismatch);
    }
    if output.0 >= descriptor.output_shape.0 || output.1 >= descriptor.output_shape.1 {
        return Err(CovarianceReplayError::OutputOutOfBounds);
    }
    let source_pixels = descriptor.source_pixels(output, native_validity)?;
    if source_pixels.iter().any(|source| {
        (0..stack.dim().0).any(|date| !stack[(date, source.row, source.column)].is_finite())
    }) {
        return Err(CovarianceReplayError::NonFiniteSource);
    }
    let numerator = sliding_row_numerators_with_validity(
        stack,
        output.0,
        half_window,
        strides,
        Some(native_validity),
    )[output.1]
        .clone();
    let coherence = normalize(numerator.view());
    Ok(RectPixelReplay {
        descriptor,
        output,
        source_pixels,
        numerator,
        coherence,
    })
}

/// Replay one Rect numerator from a byte-bounded ordered source-value matrix.
///
/// `source_values` has shape `(nslc, source_pixels.len())`. The source list is
/// strictly row-major and may omit fixed-invalid native pixels. Accumulation is
/// nevertheless performed in the production sliding kernel's column-of-
/// vertical-sums order, so the numerator is bit-identical to a full-frame Rect
/// pass over the same finite values.
///
/// # Errors
/// Returns an error for invalid output/support geometry, a shape mismatch, or
/// non-finite retained source values.
pub fn replay_rect_source_values(
    descriptor: RectReplayDescriptor,
    output: (usize, usize),
    source_pixels: &[NativeSourcePixel],
    source_values: ArrayView2<Cf64>,
) -> Result<RectPixelReplay, CovarianceReplayError> {
    if output.0 >= descriptor.output_shape.0 || output.1 >= descriptor.output_shape.1 {
        return Err(CovarianceReplayError::OutputOutOfBounds);
    }
    if source_values.nrows() == 0 || source_values.ncols() != source_pixels.len() {
        return Err(CovarianceReplayError::SourceValueShapeMismatch);
    }
    if source_values.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceReplayError::NonFiniteSource);
    }
    if source_pixels.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CovarianceReplayError::SourceOrderMismatch);
    }
    let row_start = window_origin_row(
        output.0,
        descriptor.half_window,
        descriptor.strides,
        descriptor.native_shape.0,
    );
    let column_start = window_origin_col(
        output.1,
        descriptor.half_window,
        descriptor.strides,
        descriptor.native_shape.1,
    );
    let window = (
        2 * descriptor.half_window.y + 1,
        2 * descriptor.half_window.x + 1,
    );
    if source_pixels.iter().any(|source| {
        source.row < row_start
            || source.row >= row_start + window.0
            || source.column < column_start
            || source.column >= column_start + window.1
    }) {
        return Err(CovarianceReplayError::SourceOutsideWindow);
    }
    let nslc = source_values.nrows();
    let mut numerator = Array2::zeros((nslc, nslc));
    for i in 0..nslc {
        for j in i..nslc {
            let mut value = Cf64::new(0.0, 0.0);
            for column in column_start..column_start + window.1 {
                let mut vertical = Cf64::new(0.0, 0.0);
                for row in row_start..row_start + window.0 {
                    let source = NativeSourcePixel::new(row, column);
                    if let Ok(index) = source_pixels.binary_search(&source) {
                        vertical += source_values[(i, index)] * source_values[(j, index)].conj();
                    }
                }
                value += vertical;
            }
            numerator[(i, j)] = value;
            numerator[(j, i)] = value.conj();
        }
    }
    let coherence = normalize(numerator.view());
    Ok(RectPixelReplay {
        descriptor,
        output,
        source_pixels: source_pixels.to_vec(),
        numerator,
        coherence,
    })
}

/// Differentiate one replayed Rect coherence matrix with respect to one native source direction.
///
/// # Errors
/// Returns an error for a mismatched/non-finite direction, a source outside
/// the replay support, or an amplitude-floor branch boundary.
pub fn rect_pixel_source_coherence_jvp(
    stack: ArrayView3<Cf64>,
    replay: &RectPixelReplay,
    source: NativeSourcePixel,
    direction: ArrayView1<Cf64>,
    branch_tolerance: f64,
) -> Result<Array2<Cf64>, CovarianceReplayError> {
    let nslc = stack.dim().0;
    if direction.len() != nslc {
        return Err(CovarianceReplayError::DirectionLengthMismatch);
    }
    if direction.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceReplayError::NonFiniteDirection);
    }
    if !replay.source_pixels.contains(&source) {
        return Err(CovarianceReplayError::SourceOutsideSupport);
    }
    if source.row >= stack.dim().1 || source.column >= stack.dim().2 {
        return Err(CovarianceReplayError::SourceOutsideSupport);
    }
    let source_values =
        Array2::from_shape_fn((nslc, replay.source_pixels.len()), |(date, index)| {
            let pixel = replay.source_pixels[index];
            stack[(date, pixel.row, pixel.column)]
        });
    rect_source_values_coherence_jvp(
        source_values.view(),
        replay,
        source,
        direction,
        branch_tolerance,
    )
}

/// Differentiate a byte-bounded Rect replay with respect to one loaded source direction.
///
/// # Errors
/// Returns an error for source/value/direction mismatches, non-finite state, or
/// an amplitude-floor branch boundary.
pub fn rect_source_values_coherence_jvp(
    source_values: ArrayView2<Cf64>,
    replay: &RectPixelReplay,
    source: NativeSourcePixel,
    direction: ArrayView1<Cf64>,
    branch_tolerance: f64,
) -> Result<Array2<Cf64>, CovarianceReplayError> {
    let nslc = source_values.nrows();
    if source_values.ncols() != replay.source_pixels.len() || nslc == 0 {
        return Err(CovarianceReplayError::SourceValueShapeMismatch);
    }
    if source_values.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceReplayError::NonFiniteSource);
    }
    if direction.len() != nslc {
        return Err(CovarianceReplayError::DirectionLengthMismatch);
    }
    if direction.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceReplayError::NonFiniteDirection);
    }
    let source_index = replay
        .source_pixels
        .iter()
        .position(|candidate| *candidate == source)
        .ok_or(CovarianceReplayError::SourceOutsideSupport)?;
    let mut delta_numerator = Array2::zeros((nslc, nslc));
    for i in 0..nslc {
        let zi = source_values[(i, source_index)];
        for j in i..nslc {
            let zj = source_values[(j, source_index)];
            let value = direction[i] * zj.conj() + zi * direction[j].conj();
            delta_numerator[(i, j)] = value;
            delta_numerator[(j, i)] = value.conj();
        }
    }
    normalize_numerator_jvp(
        replay.numerator.view(),
        delta_numerator.view(),
        branch_tolerance,
    )
}

/// Differentiate production covariance normalization in one numerator direction.
///
/// Entries safely below the amplitude floor retain a zero derivative. An entry
/// at the branch boundary fails closed.
///
/// # Errors
/// Returns an error for mismatched shapes, a branch-boundary denominator, or a
/// non-finite derivative.
pub fn normalize_numerator_jvp(
    numerator: ArrayView2<Cf64>,
    delta_numerator: ArrayView2<Cf64>,
    branch_tolerance: f64,
) -> Result<Array2<Cf64>, CovarianceReplayError> {
    if numerator.dim() != delta_numerator.dim() || numerator.nrows() != numerator.ncols() {
        return Err(CovarianceReplayError::MatrixShapeMismatch);
    }
    if !branch_tolerance.is_finite() || branch_tolerance < 0.0 {
        return Err(CovarianceReplayError::InvalidBranchTolerance);
    }
    if numerator
        .iter()
        .chain(delta_numerator.iter())
        .any(|value| !value.is_finite())
    {
        return Err(CovarianceReplayError::NonFiniteState);
    }
    validate_numerator_branch(numerator, branch_tolerance)?;
    let n = numerator.nrows();
    let coherence = normalize(numerator);
    let mut derivative = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let ni = numerator[(i, i)].re;
            let nj = numerator[(j, j)].re;
            let denominator = ni.max(0.0).sqrt() * nj.max(0.0).sqrt();
            if denominator < AMP_FLOOR {
                continue;
            }
            if ni <= 0.0 || nj <= 0.0 {
                return Err(CovarianceReplayError::AmplitudeFloorBoundary);
            }
            derivative[(i, j)] = delta_numerator[(i, j)] / denominator
                - coherence[(i, j)]
                    * 0.5
                    * (delta_numerator[(i, i)].re / ni + delta_numerator[(j, j)].re / nj);
        }
    }
    if derivative.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceReplayError::NonFiniteDerivative);
    }
    Ok(derivative)
}

pub(crate) fn validate_numerator_branch(
    numerator: ArrayView2<Cf64>,
    branch_tolerance: f64,
) -> Result<(), CovarianceReplayError> {
    if numerator.nrows() != numerator.ncols() {
        return Err(CovarianceReplayError::MatrixShapeMismatch);
    }
    if !branch_tolerance.is_finite() || branch_tolerance < 0.0 {
        return Err(CovarianceReplayError::InvalidBranchTolerance);
    }
    if numerator.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceReplayError::NonFiniteState);
    }
    let n = numerator.nrows();
    for i in 0..n {
        for j in 0..n {
            let denominator =
                numerator[(i, i)].re.max(0.0).sqrt() * numerator[(j, j)].re.max(0.0).sqrt();
            if (denominator - AMP_FLOOR).abs() <= branch_tolerance {
                return Err(CovarianceReplayError::AmplitudeFloorBoundary);
            }
        }
    }
    Ok(())
}

/// Estimate the per-pixel coherence matrix over a sliding window.
///
/// `stack` is `(nslc, rows, cols)`. Returns `(out_rows, out_cols, nslc, nslc)`
/// where the output grid is decimated by `strides`. When `neighbors` is given
/// (the SHP `(out_rows, out_cols, win_h, win_w)` mask from `dolphin-shp`), the
/// masked direct per-pixel kernel is used; otherwise the rectangular window is
/// evaluated with the row-separable box-sum kernel.
///
/// # Errors
/// Returns `Err` if a stride is zero or the window is larger than the stack.
pub fn estimate_stack_covariance(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
) -> Result<Array4<Cf64>, &'static str> {
    if strides.y == 0 || strides.x == 0 {
        return Err("covariance stride is zero");
    }
    match neighbors.is_some() {
        true => estimate_stack_covariance_direct(stack, half, strides, neighbors),
        false => estimate_stack_covariance_sliding(stack, half, strides),
    }
}

/// Direct per-pixel covariance: each output pixel reads its full window and sums
/// the Hermitian cross-products independently. Retained as the SHP-masked path
/// implementation and as the sliding kernel's tolerance oracle.
///
/// Same signature and result layout as [`estimate_stack_covariance`].
///
/// # Errors
/// Returns `Err` if a stride is zero or the window is larger than the stack.
pub fn estimate_stack_covariance_direct(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
) -> Result<Array4<Cf64>, &'static str> {
    if strides.y == 0 || strides.x == 0 {
        return Err("covariance stride is zero");
    }
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    if win_h > rows || win_w > cols {
        return Err("covariance window larger than stack");
    }
    let (out_rows, out_cols) = strides.out_shape((rows, cols));

    let mats: Vec<Array2<Cf64>> = (0..out_rows * out_cols)
        .into_par_iter()
        .map(|idx| {
            pixel_coh(
                stack,
                (idx / out_cols, idx % out_cols),
                half,
                strides,
                neighbors,
            )
        })
        .collect();

    assemble(mats, (out_rows, out_cols, nslc))
}

/// Row-separable box-sum covariance for the unmasked rectangular window.
///
/// Parallel over output rows; each row task holds only per-row buffers
/// (`vsum`/`hpref`, `npairs·cols` each), never an `nslc²·area` cube. Coherence
/// entries match the direct kernel to ~1e-4 (running sums reorder + subtract FP).
fn estimate_stack_covariance_sliding(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
) -> Result<Array4<Cf64>, &'static str> {
    if strides.y == 0 || strides.x == 0 {
        return Err("covariance stride is zero");
    }
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    if win_h > rows || win_w > cols {
        return Err("covariance window larger than stack");
    }
    let (out_rows, out_cols) = strides.out_shape((rows, cols));

    let rows_of_mats: Vec<Vec<Array2<Cf64>>> = (0..out_rows)
        .into_par_iter()
        .map(|orow| {
            sliding_row_numerators(stack, orow, half, strides)
                .into_iter()
                .map(|numer| normalize(numer.view()))
                .collect()
        })
        .collect();

    let mats: Vec<Array2<Cf64>> = rows_of_mats.into_iter().flatten().collect();
    assemble(mats, (out_rows, out_cols, nslc))
}

/// Per-output-col Hermitian **numerator** matrices for a single output row.
///
/// Shared by the staged covariance path and the fused unmasked path so both go
/// through the identical accumulation order (⇒ fused==staged stays bit-identical).
/// Returns `out_cols` matrices of shape `(nslc, nslc)`; the caller normalizes.
pub(crate) fn sliding_row_numerators(
    stack: ArrayView3<Cf64>,
    orow: usize,
    half: HalfWindow,
    strides: Strides,
) -> Vec<Array2<Cf64>> {
    sliding_row_numerators_with_validity(stack, orow, half, strides, None)
}

pub(crate) fn sliding_row_numerators_with_validity(
    stack: ArrayView3<Cf64>,
    orow: usize,
    half: HalfWindow,
    strides: Strides,
    native_validity: Option<ArrayView2<bool>>,
) -> Vec<Array2<Cf64>> {
    if strides.y == 0 || strides.x == 0 {
        return Vec::new();
    }
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    let (_, out_cols) = strides.out_shape((rows, cols));
    let r0 = window_origin_row(orow, half, strides, rows);

    let pairs = hermitian_pairs(nslc);
    let vsum = vertical_pair_sums(stack, r0, win_h, &pairs, cols, native_validity);

    (0..out_cols)
        .map(|ocol| {
            let c0 = window_origin_col(ocol, half, strides, cols);
            expand_hermitian(&vsum, &pairs, nslc, c0, win_w)
        })
        .collect()
}

/// Upper-triangle (`i ≤ j`) pair list, packed once per row task.
fn hermitian_pairs(nslc: usize) -> Vec<(usize, usize)> {
    (0..nslc)
        .flat_map(|i| (i..nslc).map(move |j| (i, j)))
        .collect()
}

/// `vsum[p][c] = Σ_{r=r0..r0+win_h} finite(z_i[r][c])·conj(finite(z_j[r][c]))`
/// for every input column `c` and Hermitian pair `p`.
fn vertical_pair_sums(
    stack: ArrayView3<Cf64>,
    r0: usize,
    win_h: usize,
    pairs: &[(usize, usize)],
    cols: usize,
    native_validity: Option<ArrayView2<bool>>,
) -> Vec<Vec<Cf64>> {
    pairs
        .iter()
        .map(|&(i, j)| pair_vertical_sum(stack, i, j, r0, win_h, cols, native_validity))
        .collect()
}

/// Vertical sum of one Hermitian pair `(i, j)` over the window rows, per column.
fn pair_vertical_sum(
    stack: ArrayView3<Cf64>,
    i: usize,
    j: usize,
    r0: usize,
    win_h: usize,
    cols: usize,
    native_validity: Option<ArrayView2<bool>>,
) -> Vec<Cf64> {
    let mut col_sum = vec![Cf64::new(0.0, 0.0); cols];
    for r in r0..r0 + win_h {
        let zi = stack.slice(s![i, r, ..]);
        let zj = stack.slice(s![j, r, ..]);
        match native_validity {
            Some(validity) => accumulate_valid_row(&mut col_sum, zi, zj, validity.row(r)),
            None => accumulate_row(&mut col_sum, zi, zj),
        }
    }
    col_sum
}

fn accumulate_valid_row(
    col_sum: &mut [Cf64],
    zi: ArrayView1<Cf64>,
    zj: ArrayView1<Cf64>,
    validity: ArrayView1<bool>,
) {
    col_sum
        .iter_mut()
        .zip(zi.iter().zip(zj.iter()).zip(validity))
        .for_each(|(acc, ((&a, &b), &valid))| {
            if valid {
                *acc += finite_or_zero(a) * finite_or_zero(b).conj();
            }
        });
}

/// Add one stack row's per-column cross-products into the running vertical sum.
fn accumulate_row(col_sum: &mut [Cf64], zi: ArrayView1<Cf64>, zj: ArrayView1<Cf64>) {
    col_sum
        .iter_mut()
        .zip(zi.iter().zip(zj.iter()))
        .for_each(|(acc, (&a, &b))| {
            *acc += finite_or_zero(a) * finite_or_zero(b).conj();
        });
}

/// Expand one output col's Hermitian numerators into the full `(nslc, nslc)`
/// matrix. Each pair's numerator is the windowed sum of its shared vertical sums
/// over the window's own columns `c0..c0+win_w`, in fixed left-to-right order —
/// so the value depends only on the window's samples, not on the block width
/// (⇒ tiled==whole and fused==staged stay bit-identical). `numer[j][i]=conj`.
fn expand_hermitian(
    vsum: &[Vec<Cf64>],
    pairs: &[(usize, usize)],
    nslc: usize,
    c0: usize,
    win_w: usize,
) -> Array2<Cf64> {
    let mut numer = Array2::<Cf64>::zeros((nslc, nslc));
    for (p, &(i, j)) in pairs.iter().enumerate() {
        let val = window_sum(&vsum[p][c0..c0 + win_w]);
        numer[(i, j)] = val;
        numer[(j, i)] = val.conj();
    }
    numer
}

/// Sum a window's vertical partial sums in fixed left-to-right order.
fn window_sum(cols: &[Cf64]) -> Cf64 {
    cols.iter().fold(Cf64::new(0.0, 0.0), |acc, &v| acc + v)
}

/// Coherence matrix for a single output pixel `out = (out_r, out_c)`.
pub(crate) fn pixel_coh(
    stack: ArrayView3<Cf64>,
    out: (usize, usize),
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
) -> Array2<Cf64> {
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    let r0 = window_origin_row(out.0, half, strides, rows);
    let c0 = window_origin_col(out.1, half, strides, cols);
    let window = stack.slice(s![.., r0..r0 + win_h, c0..c0 + win_w]);
    let mask = neighbors.map(|nbr| nbr.slice_move(s![out.0, out.1, .., ..]));
    coh_mat(window, nslc, mask)
}

/// Top row of the window for output row `out_r`, clamped inward at the top/bottom
/// borders so the window stays full-size (matches JAX `dynamic_slice` clamping).
/// The single source of clamp truth for the row axis (direct + sliding paths).
fn window_origin_row(out_r: usize, half: HalfWindow, strides: Strides, rows: usize) -> usize {
    let in_r = strides.y / 2 + out_r * strides.y;
    in_r.saturating_sub(half.y).min(rows - (2 * half.y + 1))
}

/// Left column of the window for output col `out_c`, clamped inward at the
/// left/right borders. The single source of clamp truth for the column axis.
fn window_origin_col(out_c: usize, half: HalfWindow, strides: Strides, cols: usize) -> usize {
    let in_c = strides.x / 2 + out_c * strides.x;
    in_c.saturating_sub(half.x).min(cols - (2 * half.x + 1))
}

/// Coherence matrix from a `(nslc, win_h, win_w)` window (port of `coh_mat_single`).
/// `mask` is the per-pixel SHP neighbor flags `(win_h, win_w)`, if any.
fn coh_mat(window: ArrayView3<Cf64>, nslc: usize, mask: Option<ArrayView2<bool>>) -> Array2<Cf64> {
    let nsamps = window.len() / nslc;
    let mut masked = window
        .to_shape((nslc, nsamps))
        .expect("contiguous window reshape")
        .mapv(finite_or_zero);
    if let Some(flags) = mask {
        let flags = flags.to_shape(nsamps).expect("mask reshape").to_owned();
        zero_unflagged_columns(&mut masked, &flags);
    }

    normalize(hermitian_product(&masked, nslc).view())
}

/// Cross-correlation `numer[i][j] = Σ_s z_i[s] · conj(z_j[s])` from the masked
/// `(nslc, nsamps)` sample matrix. The result is Hermitian, so only the upper
/// triangle is summed and the lower mirrored — half the work of a full matmul,
/// and a tight contiguous-row loop instead of ndarray's generic complex `dot`
/// (which has no SIMD/BLAS path for `Complex<f64>`) plus its conjugate-transpose
/// allocation.
fn hermitian_product(masked: &Array2<Cf64>, nslc: usize) -> Array2<Cf64> {
    let mut numer = Array2::<Cf64>::zeros((nslc, nslc));
    for i in 0..nslc {
        let zi = masked.row(i);
        for j in i..nslc {
            let dot = row_conj_dot(zi, masked.row(j));
            numer[(i, j)] = dot;
            numer[(j, i)] = dot.conj();
        }
    }
    numer
}

/// `Σ_s a[s] · conj(b[s])` over two contiguous sample rows.
fn row_conj_dot(a: ArrayView1<Cf64>, b: ArrayView1<Cf64>) -> Cf64 {
    a.iter().zip(b).map(|(x, y)| x * y.conj()).sum()
}

/// Replace non-finite samples (NaN/Inf) with zero, matching dolphin's masking.
fn finite_or_zero(z: Cf64) -> Cf64 {
    match z.is_finite() {
        true => z,
        false => Cf64::new(0.0, 0.0),
    }
}

/// Zero every sample column not flagged as an SHP neighbor.
fn zero_unflagged_columns(masked: &mut Array2<Cf64>, flags: &Array1<bool>) {
    flags
        .iter()
        .enumerate()
        .filter(|(_, &keep)| !keep)
        .for_each(|(k, _)| masked.column_mut(k).fill(Cf64::new(0.0, 0.0)));
}

/// Normalize a Hermitian numerator matrix to a coherence matrix. Shared with the
/// fused unmasked path so it applies the identical `AMP_FLOOR` semantics.
pub(crate) fn normalize_numerator(numer: ArrayView2<Cf64>) -> Array2<Cf64> {
    normalize(numer)
}

/// Normalize a cross-correlation matrix to a coherence matrix.
fn normalize(numer: ArrayView2<Cf64>) -> Array2<Cf64> {
    let n = numer.nrows();
    let amp: Vec<f64> = (0..n).map(|i| numer[(i, i)].re.max(0.0).sqrt()).collect();
    Array2::from_shape_fn((n, n), |(i, j)| {
        coherence_entry(numer[(i, j)], amp[i] * amp[j])
    })
}

/// One normalized coherence entry: `numer / denom`, or 0 when `denom` underflows.
fn coherence_entry(numer: Cf64, denom: f64) -> Cf64 {
    match denom > AMP_FLOOR {
        true => numer / denom,
        false => Cf64::new(0.0, 0.0),
    }
}

/// Stack per-pixel `(n, n)` matrices into an `(out_rows, out_cols, n, n)` array.
fn assemble(
    mats: Vec<Array2<Cf64>>,
    shape: (usize, usize, usize),
) -> Result<Array4<Cf64>, &'static str> {
    let (out_rows, out_cols, n) = shape;
    let flat: Vec<Cf64> = mats.into_iter().flat_map(IntoIterator::into_iter).collect();
    Array4::from_shape_vec((out_rows, out_cols, n, n), flat)
        .map_err(|_| "covariance assembly shape mismatch")
}
