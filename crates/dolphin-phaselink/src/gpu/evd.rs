//! GPU EVD phase linking — host side for `evd.wgsl`.

use dolphin_core::Cf32;
use ndarray::{Array2, Array3, ArrayView4};

use super::context::{GpuContext, GpuError};
use super::covariance::MAX_NSLC;
use super::dispatch::{dispatch_compute, input_buffer, output_buffer, readback, uniform_buffer};

/// Default power-iteration count — ample for the eigenvalue gaps of DS coherence
/// matrices; the dominant eigenvector converges well before this.
pub const DEFAULT_EVD_ITERS: u32 = 100;

/// Stacked GPU phase-linking output, mirroring [`crate::StackEstimate`] (f32).
pub struct GpuStackEstimate {
    /// Referenced wrapped phase, shape `(nslc, out_rows, out_cols)`.
    pub cpx_phase: Array3<Cf32>,
    /// Eigenvalue per pixel, shape `(out_rows, out_cols)`.
    pub eigenvalues: Array2<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nslc: u32,
    n_pix: u32,
    ref_idx: u32,
    iters: u32,
}

/// EVD-link a `(out_rows, out_cols, nslc, nslc)` coherence stack on the GPU:
/// dominant eigenvector of `C ⊙ |C|` by power iteration, phase referenced to
/// `reference_idx`. Single precision — compare to the CPU EVD path within f32
/// tolerance (eigenvector overlap, not bit-exactness).
///
/// # Errors
/// Returns [`GpuError`] if `nslc > MAX_NSLC` or a GPU dispatch/readback fails.
pub fn evd_link_gpu(
    ctx: &GpuContext,
    c_arrays: ArrayView4<Cf32>,
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
    let (phase, eig) = run(ctx, &flat, &params, n_pix, nslc)?;
    Ok(pack(phase, eig, (out_rows, out_cols, nslc)))
}

/// Dispatch the EVD kernel and read back phase + eigenvalue buffers.
fn run(
    ctx: &GpuContext,
    cmat: &[[f32; 2]],
    params: &Params,
    n_pix: usize,
    nslc: usize,
) -> Result<(Vec<[f32; 2]>, Vec<f32>), GpuError> {
    let c_buf = input_buffer(ctx, "cmat", cmat);
    let param_buf = uniform_buffer(ctx, "evd-params", params);
    let phase_buf = output_buffer(
        ctx,
        "evd-phase",
        (n_pix * nslc * std::mem::size_of::<[f32; 2]>()) as u64,
    );
    let eig_buf = output_buffer(ctx, "evd-eig", (n_pix * std::mem::size_of::<f32>()) as u64);
    dispatch_compute(
        ctx,
        include_str!("evd.wgsl"),
        "evd",
        &[&c_buf, &param_buf, &phase_buf, &eig_buf],
        (n_pix as u32).div_ceil(64),
    );
    let phase = readback::<[f32; 2]>(ctx, &phase_buf, n_pix * nslc)?;
    let eig = readback::<f32>(ctx, &eig_buf, n_pix)?;
    Ok((phase, eig))
}

/// Repack flat GPU output into `(nslc, out_rows, out_cols)` phase + eigenvalues.
fn pack(phase: Vec<[f32; 2]>, eig: Vec<f32>, shape: (usize, usize, usize)) -> GpuStackEstimate {
    let (out_rows, out_cols, nslc) = shape;
    let mut cpx_phase = Array3::zeros((nslc, out_rows, out_cols));
    for r in 0..out_rows {
        for c in 0..out_cols {
            let pbase = (r * out_cols + c) * nslc;
            for (i, slot) in phase[pbase..pbase + nslc].iter().enumerate() {
                cpx_phase[(i, r, c)] = Cf32::new(slot[0], slot[1]);
            }
        }
    }
    let eigenvalues = Array2::from_shape_vec((out_rows, out_cols), eig)
        .expect("eigenvalue grid matches pixel count");
    GpuStackEstimate {
        cpx_phase,
        eigenvalues,
    }
}
