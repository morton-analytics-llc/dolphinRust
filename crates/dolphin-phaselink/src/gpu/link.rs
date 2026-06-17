//! GPU phase-linking entry — host side for `evd.wgsl` / `emi.wgsl`.
//!
//! Mirrors the CPU [`crate::process_coherence_matrices`] (the `β = 0`,
//! `zero_correlation_threshold = 0` path): from a `(out_rows, out_cols, nslc,
//! nslc)` coherence stack, recover the referenced wrapped phase with EVD
//! (dominant eigenvector of `C ⊙ |C|`) or EMI (least eigenvector of `Γ⁻¹ ⊙ C`,
//! EVD fallback on a non-PD `Γ`). Single precision.

use dolphin_core::Cf32;
use ndarray::{Array2, Array3, ArrayView4};

use super::context::{GpuContext, GpuError};
use super::covariance::MAX_NSLC;
use super::dispatch::{dispatch_compute, input_buffer, output_buffer, readback, uniform_buffer};

/// Default power-iteration count — ample for the eigenvalue gaps of DS coherence
/// matrices; the target eigenvector converges well before this.
pub const DEFAULT_LINK_ITERS: u32 = 100;

/// Stacked GPU phase-linking output, mirroring [`crate::StackEstimate`] (f32).
pub struct GpuStackEstimate {
    /// Referenced wrapped phase, shape `(nslc, out_rows, out_cols)`.
    pub cpx_phase: Array3<Cf32>,
    /// Eigenvalue per pixel, shape `(out_rows, out_cols)`.
    pub eigenvalues: Array2<f32>,
    /// Estimator per pixel: 0 = EVD, 1 = EMI. Shape `(out_rows, out_cols)`.
    pub estimator: Array2<u8>,
}

/// Raw GPU readback: per-pixel phase pairs, eigenvalues, estimator flags.
type RawLink = (Vec<[f32; 2]>, Vec<f32>, Vec<u32>);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nslc: u32,
    n_pix: u32,
    ref_idx: u32,
    iters: u32,
}

/// Phase-link a coherence stack on the GPU. `use_evd` selects EVD; otherwise EMI
/// (with per-pixel EVD fallback). Compare to the CPU path within f32 tolerance.
///
/// # Errors
/// Returns [`GpuError`] if `nslc > MAX_NSLC` or a GPU dispatch/readback fails.
pub fn process_coherence_matrices_gpu(
    ctx: &GpuContext,
    c_arrays: ArrayView4<Cf32>,
    use_evd: bool,
    reference_idx: usize,
    iters: u32,
) -> Result<GpuStackEstimate, GpuError> {
    let (out_rows, out_cols, nslc, _) = c_arrays.dim();
    if nslc > MAX_NSLC {
        return Err(GpuError::DeviceRequest(format!(
            "nslc {nslc} exceeds GPU MAX_NSLC {MAX_NSLC}"
        )));
    }
    let n_pix = out_rows * out_cols;
    let params = Params {
        nslc: nslc as u32,
        n_pix: n_pix as u32,
        ref_idx: reference_idx as u32,
        iters,
    };
    let flat: Vec<[f32; 2]> = c_arrays
        .as_standard_layout()
        .iter()
        .map(|z| [z.re, z.im])
        .collect();

    let (phase, eig, est) = run(ctx, &flat, &params, use_evd, (n_pix, nslc))?;
    Ok(pack(phase, eig, est, (out_rows, out_cols, nslc)))
}

/// Dispatch the chosen kernel and read back phase, eigenvalue, estimator.
fn run(
    ctx: &GpuContext,
    cmat: &[[f32; 2]],
    params: &Params,
    use_evd: bool,
    dims: (usize, usize),
) -> Result<RawLink, GpuError> {
    let (n_pix, nslc) = dims;
    let c_buf = input_buffer(ctx, "cmat", cmat);
    let param_buf = uniform_buffer(ctx, "link-params", params);
    let phase_buf = output_buffer(
        ctx,
        "link-phase",
        (n_pix * nslc * std::mem::size_of::<[f32; 2]>()) as u64,
    );
    let eig_buf = output_buffer(ctx, "link-eig", (n_pix * std::mem::size_of::<f32>()) as u64);
    let est_buf = output_buffer(ctx, "link-est", (n_pix * std::mem::size_of::<u32>()) as u64);

    let groups = (n_pix as u32).div_ceil(if use_evd { 64 } else { 32 });
    dispatch_link(
        ctx,
        use_evd,
        &[&c_buf, &param_buf, &phase_buf, &eig_buf, &est_buf],
        groups,
    );

    let phase = readback::<[f32; 2]>(ctx, &phase_buf, n_pix * nslc)?;
    let eig = readback::<f32>(ctx, &eig_buf, n_pix)?;
    let est = match use_evd {
        true => vec![0_u32; n_pix],
        false => readback::<u32>(ctx, &est_buf, n_pix)?,
    };
    Ok((phase, eig, est))
}

/// EVD binds the first 4 buffers; EMI also binds the estimator output.
fn dispatch_link(ctx: &GpuContext, use_evd: bool, bufs: &[&wgpu::Buffer], groups: u32) {
    match use_evd {
        true => dispatch_compute(ctx, include_str!("evd.wgsl"), "evd", &bufs[..4], groups),
        false => dispatch_compute(ctx, include_str!("emi.wgsl"), "emi", bufs, groups),
    }
}

/// Repack flat GPU output into stacked arrays.
fn pack(
    phase: Vec<[f32; 2]>,
    eig: Vec<f32>,
    est: Vec<u32>,
    shape: (usize, usize, usize),
) -> GpuStackEstimate {
    let (out_rows, out_cols, nslc) = shape;
    let mut cpx_phase = Array3::zeros((nslc, out_rows, out_cols));
    for (pix, slot) in phase.chunks_exact(nslc).enumerate() {
        let (r, c) = (pix / out_cols, pix % out_cols);
        for (i, z) in slot.iter().enumerate() {
            cpx_phase[(i, r, c)] = Cf32::new(z[0], z[1]);
        }
    }
    let eigenvalues =
        Array2::from_shape_vec((out_rows, out_cols), eig).expect("eigenvalue grid matches pixels");
    let estimator =
        Array2::from_shape_vec((out_rows, out_cols), est.iter().map(|&e| e as u8).collect())
            .expect("estimator grid matches pixels");
    GpuStackEstimate {
        cpx_phase,
        eigenvalues,
        estimator,
    }
}
