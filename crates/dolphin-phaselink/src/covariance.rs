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

/// Amplitude floor below which a coherence entry is set to 0 (dolphin uses 1e-6).
const AMP_FLOOR: f64 = 1e-6;

/// Estimate the per-pixel coherence matrix over a sliding window.
///
/// `stack` is `(nslc, rows, cols)`. Returns `(out_rows, out_cols, nslc, nslc)`
/// where the output grid is decimated by `strides`. When `neighbors` is given
/// (the SHP `(out_rows, out_cols, win_h, win_w)` mask from `dolphin-shp`), the
/// masked direct per-pixel kernel is used; otherwise the rectangular window is
/// evaluated with the row-separable box-sum kernel.
///
/// # Errors
/// Returns `Err` if the window is larger than the stack in either dimension.
pub fn estimate_stack_covariance(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
) -> Result<Array4<Cf64>, &'static str> {
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
/// Returns `Err` if the window is larger than the stack in either dimension.
pub fn estimate_stack_covariance_direct(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
) -> Result<Array4<Cf64>, &'static str> {
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
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    let (_, out_cols) = strides.out_shape((rows, cols));
    let r0 = window_origin_row(orow, half, strides, rows);

    let pairs = hermitian_pairs(nslc);
    let vsum = vertical_pair_sums(stack, r0, win_h, &pairs, cols);

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
) -> Vec<Vec<Cf64>> {
    pairs
        .iter()
        .map(|&(i, j)| pair_vertical_sum(stack, i, j, r0, win_h, cols))
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
) -> Vec<Cf64> {
    let mut col_sum = vec![Cf64::new(0.0, 0.0); cols];
    for r in r0..r0 + win_h {
        let zi = stack.slice(s![i, r, ..]);
        let zj = stack.slice(s![j, r, ..]);
        accumulate_row(&mut col_sum, zi, zj);
    }
    col_sum
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
