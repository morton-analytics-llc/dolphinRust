//! Fused covariance → estimator → quality pass (Lever 1: break the memory floor).
//!
//! The separate-stage path materializes the whole
//! `(out_rows, out_cols, nslc, nslc)` `Cf64` coherence cube in
//! [`crate::estimate_stack_covariance`], then re-reads it from
//! [`crate::process_coherence_matrices`], [`crate::estimate_temp_coh`],
//! [`crate::estimate_crlb`] and [`crate::estimate_closure_phases`]. That cube
//! (`nslc²·area·16 B`) is the phase-linking memory floor.
//!
//! This pass computes each output pixel's coherence matrix once, runs every
//! consumer against it, and retains only the per-date phase + scalar quality +
//! per-date CRLB + per-triplet closure — **discarding the `N×N` matrix before
//! moving to the next pixel**. The math is identical to the separate stages
//! (same kernels, same pixel/`idx` ordering), so the result is bit-identical;
//! only the cube is never held. Parallelized over output pixels with `rayon`,
//! exactly as the stages it replaces.

use dolphin_core::{Cf64, HalfWindow, Strides};
use ndarray::{Array1, Array2, Array3, ArrayView3, ArrayView4};
use rayon::prelude::*;

use crate::closure::triplet_closure;
use crate::covariance::{normalize_numerator, pixel_coh, sliding_row_numerators};
use crate::crlb::crlb_pixel;
use crate::estimator::process_coherence_matrix;
use crate::quality::temp_coh_single;

/// Estimator + quality parameters for a fused pass (mirrors the separate-stage
/// call arguments). Grouped to keep the per-pixel signature small.
#[derive(Debug, Clone, Copy)]
pub struct FusedParams {
    /// Use EVD instead of EMI for the linked phase.
    pub use_evd: bool,
    /// EMI regularization weight `β`.
    pub beta: f64,
    /// Coherence values at or below this are treated as zero.
    pub zero_correlation_threshold: f64,
    /// Reference date for the output linked phase.
    pub reference_idx: usize,
    /// Produce the per-date CRLB σ layer.
    pub compute_crlb: bool,
    /// CRLB reference date (dolphin's last carried compressed date — may differ
    /// from `reference_idx`).
    pub crlb_reference_idx: usize,
    /// CRLB look count `L` (`sqrt(half_y·half_x)`).
    pub num_looks: f64,
    /// Produce the per-triplet closure-phase layer.
    pub compute_closure: bool,
    /// Produce the per-date average coherence-magnitude layer.
    pub compute_average_coherence: bool,
    /// First real acquisition in the combined stack. Leading compressed SLC
    /// pseudo-dates are excluded from the aggregate coherence metric.
    pub average_coherence_start_idx: usize,
}

/// Finite sum/count of real-date average coherence at every output pixel.
///
/// The workflow combines these bounded 2-D aggregates across ministacks and
/// divides once at the end. This avoids retaining a date-major quality cube.
#[derive(Clone)]
pub struct AverageCoherenceAggregate {
    /// Sum of finite per-real-date coherence magnitudes.
    pub sum: Array2<f64>,
    /// Number of finite real-date values contributing to each sum.
    pub count: Array2<u32>,
}

/// Fused phase-linking output over the strided `(out_rows, out_cols)` grid.
/// Identical fields to what the separate stages produce, minus the retained
/// coherence cube.
pub struct FusedEstimate {
    /// Linked phase (unit magnitude), `(nslc, out_rows, out_cols)`.
    pub cpx_phase: Array3<Cf64>,
    /// Temporal coherence, `(out_rows, out_cols)`.
    pub temporal_coherence: Array2<f64>,
    /// Per-date CRLB σ, `(nslc, out_rows, out_cols)`; `None` unless requested.
    pub crlb_sigma: Option<Array3<f64>>,
    /// Per-triplet closure phase (band-major), `(nslc-2, out_rows, out_cols)`;
    /// `None` unless requested.
    pub closure_phase: Option<Array3<f64>>,
    /// Bounded real-date average-coherence sum/count; `None` unless requested.
    pub average_coherence: Option<AverageCoherenceAggregate>,
}

/// One output pixel's fused products — the only thing retained per pixel.
struct PixelFused {
    phase: Array1<Cf64>,
    temp_coh: f64,
    crlb: Option<Array1<f64>>,
    closure: Option<Vec<f64>>,
    average_coherence: Option<(f64, u32)>,
}

/// Run the fused covariance + estimator + quality pass over `stack`
/// `(nslc, rows, cols)`. `neighbors` is the optional SHP mask
/// `(out_rows, out_cols, win_h, win_w)`.
///
/// # Errors
/// Returns `Err` if the covariance window is larger than the stack.
pub fn link_fused(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
    params: FusedParams,
) -> Result<FusedEstimate, &'static str> {
    let (nslc, rows, cols) = stack.dim();
    if has_all_non_finite_acquisition(stack) {
        return Err("slc stack contains an all non-finite acquisition");
    }
    if params.compute_average_coherence && params.average_coherence_start_idx > nslc {
        return Err("average coherence start exceeds stack depth");
    }
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    if win_h > rows || win_w > cols {
        return Err("covariance window larger than stack");
    }
    let (out_rows, out_cols) = strides.out_shape((rows, cols));
    let pixels = match neighbors {
        Some(_) => fused_pixels_masked(
            stack,
            half,
            strides,
            neighbors,
            params,
            (out_rows, out_cols),
        ),
        None => fused_pixels_sliding(stack, half, strides, params, (out_rows, out_cols)),
    };
    Ok(pack(pixels, (out_rows, out_cols, nslc), &params))
}

/// Whether any acquisition contains no finite complex sample. Pinned dolphin
/// v0.35.0 rejects the same degenerate stack before covariance estimation.
pub(crate) fn has_all_non_finite_acquisition(stack: ArrayView3<Cf64>) -> bool {
    (0..stack.dim().0).any(|date| {
        stack
            .index_axis(ndarray::Axis(0), date)
            .iter()
            .all(|z| !z.re.is_finite() || !z.im.is_finite())
    })
}

/// SHP-masked path: flat per-output-pixel, each pixel builds its own coherence
/// matrix via the direct `pixel_coh` kernel. Unchanged from the previous fused pass.
fn fused_pixels_masked(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
    params: FusedParams,
    out_shape: (usize, usize),
) -> Vec<PixelFused> {
    let (out_rows, out_cols) = out_shape;
    (0..out_rows * out_cols)
        .into_par_iter()
        .map(|idx| {
            let coh = pixel_coh(
                stack,
                (idx / out_cols, idx % out_cols),
                half,
                strides,
                neighbors,
            );
            fused_from_coh(coh, params)
        })
        .collect()
}

/// Unmasked path: parallel over output rows using the shared row-separable
/// sliding numerators, then normalize and run the shared consumer per pixel.
/// Same idx ordering (`idx = r*out_cols + c`) as the staged/masked paths.
fn fused_pixels_sliding(
    stack: ArrayView3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    params: FusedParams,
    out_shape: (usize, usize),
) -> Vec<PixelFused> {
    let (out_rows, _) = out_shape;
    let rows_of_pixels: Vec<Vec<PixelFused>> = (0..out_rows)
        .into_par_iter()
        .map(|orow| {
            sliding_row_numerators(stack, orow, half, strides)
                .into_iter()
                .map(|numer| fused_from_coh(normalize_numerator(numer.view()), params))
                .collect()
        })
        .collect();
    rows_of_pixels.into_iter().flatten().collect()
}

/// Run every fused consumer against a precomputed coherence matrix, retaining
/// only the per-pixel products. Shared by the masked and unmasked paths.
fn fused_from_coh(c: Array2<Cf64>, p: FusedParams) -> PixelFused {
    let est = process_coherence_matrix(
        c.view(),
        p.use_evd,
        p.beta,
        p.zero_correlation_threshold,
        p.reference_idx,
    );
    let phase = est.phase.mapv(unit_phasor);
    let temp_coh = temp_coh_single(phase.view(), c.view());
    let crlb = p.compute_crlb.then(|| {
        crlb_pixel(
            c.view(),
            p.beta,
            p.zero_correlation_threshold,
            p.crlb_reference_idx,
            p.num_looks,
        )
    });
    let ntri = c.nrows().saturating_sub(2);
    let closure = p
        .compute_closure
        .then(|| (0..ntri).map(|k| triplet_closure(c.view(), k)).collect());
    let average_coherence = p
        .compute_average_coherence
        .then(|| average_coherence_sum_count(c.view(), p.average_coherence_start_idx));
    PixelFused {
        phase,
        temp_coh,
        crlb,
        closure,
        average_coherence,
    }
}

/// Unit-magnitude phasor `exp(j∠z)` (dolphin's `exp(1j*angle(cpx_phase))`).
fn unit_phasor(z: Cf64) -> Cf64 {
    Cf64::from_polar(1.0, z.arg())
}

/// Pack per-pixel fused products into the stacked output arrays, idx-ordered the
/// same way the separate stages assemble theirs (`r = idx/out_cols`).
fn pack(
    pixels: Vec<PixelFused>,
    shape: (usize, usize, usize),
    params: &FusedParams,
) -> FusedEstimate {
    let (out_rows, out_cols, nslc) = shape;
    let ntri = nslc.saturating_sub(2);
    let mut cpx_phase = Array3::zeros((nslc, out_rows, out_cols));
    let mut temporal_coherence = Array2::zeros((out_rows, out_cols));
    let mut crlb = params
        .compute_crlb
        .then(|| Array3::<f64>::zeros((nslc, out_rows, out_cols)));
    let mut closure = params
        .compute_closure
        .then(|| Array3::<f64>::zeros((ntri, out_rows, out_cols)));
    let mut average_coherence =
        params
            .compute_average_coherence
            .then(|| AverageCoherenceAggregate {
                sum: Array2::zeros((out_rows, out_cols)),
                count: Array2::zeros((out_rows, out_cols)),
            });

    for (idx, px) in pixels.into_iter().enumerate() {
        let (r, c) = (idx / out_cols, idx % out_cols);
        temporal_coherence[(r, c)] = px.temp_coh;
        px.phase
            .iter()
            .enumerate()
            .for_each(|(t, &z)| cpx_phase[(t, r, c)] = z);
        write_band(crlb.as_mut(), px.crlb, (r, c));
        write_closure(closure.as_mut(), px.closure, (r, c));
        if let (Some(dst), Some((sum, count))) = (average_coherence.as_mut(), px.average_coherence)
        {
            dst.sum[(r, c)] = sum;
            dst.count[(r, c)] = count;
        }
    }
    FusedEstimate {
        cpx_phase,
        temporal_coherence,
        crlb_sigma: crlb,
        closure_phase: closure,
        average_coherence,
    }
}

/// Reduce the selected matrix rows directly to a finite sum/count. Each row's
/// value is dolphin's `abs(C).mean(axis=3)`, including the diagonal.
pub(crate) fn average_coherence_sum_count(
    c: ndarray::ArrayView2<Cf64>,
    start_idx: usize,
) -> (f64, u32) {
    let n = c.nrows();
    debug_assert_eq!(n, c.ncols(), "coherence matrix must be square");
    (start_idx..n)
        .map(|date| c.row(date).iter().map(|z| z.norm()).sum::<f64>() / n as f64)
        .filter(|value| value.is_finite())
        .fold((0.0, 0_u32), |(sum, count), value| (sum + value, count + 1))
}

/// Write a per-date σ vector into the band-major CRLB array at `(r, c)`.
fn write_band(dst: Option<&mut Array3<f64>>, src: Option<Array1<f64>>, rc: (usize, usize)) {
    let (Some(dst), Some(src)) = (dst, src) else {
        return;
    };
    src.iter()
        .enumerate()
        .for_each(|(t, &v)| dst[(t, rc.0, rc.1)] = v);
}

/// Write a per-triplet closure vector into the band-major closure array at `(r, c)`.
fn write_closure(dst: Option<&mut Array3<f64>>, src: Option<Vec<f64>>, rc: (usize, usize)) {
    let (Some(dst), Some(src)) = (dst, src) else {
        return;
    };
    src.iter()
        .enumerate()
        .for_each(|(k, &v)| dst[(k, rc.0, rc.1)] = v);
}

#[cfg(test)]
mod tests {
    use super::average_coherence_sum_count;
    use dolphin_core::Cf64;
    use ndarray::Array2;

    #[test]
    fn average_coherence_aggregate_skips_nan_dates() {
        let mut coherence = Array2::from_elem((3, 3), Cf64::new(1.0, 0.0));
        coherence[(1, 0)] = Cf64::new(f64::NAN, 0.0);
        let (sum, count) = average_coherence_sum_count(coherence.view(), 0);
        assert_eq!(sum, 2.0);
        assert_eq!(count, 2);

        let (real_sum, real_count) = average_coherence_sum_count(coherence.view(), 1);
        assert_eq!(real_sum, 1.0);
        assert_eq!(real_count, 1);
    }
}
