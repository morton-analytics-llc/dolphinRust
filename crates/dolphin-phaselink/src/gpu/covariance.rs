//! GPU sliding-window coherence — host side for `covariance.wgsl`.

use dolphin_core::{Cf32, HalfWindow, Strides};
use ndarray::{Array4, ArrayView3, ArrayView4};

use super::context::{GpuContext, GpuError};
use super::dispatch::{dispatch_compute, input_buffer, output_buffer, readback, uniform_buffer};

/// Largest ministack the f32 kernel supports (matches `MAX_NSLC` in the WGSL).
pub const MAX_NSLC: usize = 32;

/// Compute parameters mirrored into the shader's `Params` uniform.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nslc: u32,
    rows: u32,
    cols: u32,
    half_y: u32,
    half_x: u32,
    stride_y: u32,
    stride_x: u32,
    out_rows: u32,
    out_cols: u32,
    win_h: u32,
    win_w: u32,
    has_mask: u32,
}

/// Estimate the per-pixel coherence matrix on the GPU (f32).
///
/// Mirrors [`crate::estimate_stack_covariance`]: `stack` is `(nslc, rows, cols)`,
/// the output is `(out_rows, out_cols, nslc, nslc)` decimated by `strides`. When
/// `neighbors` is given (the SHP `(out_rows, out_cols, win_h, win_w)` mask), only
/// flagged window samples contribute, matching the CPU path. Single precision —
/// compare to the CPU path within an f32 tolerance, not bit-exactly.
///
/// # Errors
/// Returns [`GpuError`] if the window exceeds the stack, `nslc > MAX_NSLC`, or a
/// GPU dispatch/readback fails.
pub fn estimate_stack_covariance_gpu(
    ctx: &GpuContext,
    stack: ArrayView3<Cf32>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<ArrayView4<bool>>,
) -> Result<Array4<Cf32>, GpuError> {
    let (nslc, rows, cols) = stack.dim();
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    if win_h > rows || win_w > cols {
        return Err(GpuError::DeviceRequest(
            "covariance window larger than stack".into(),
        ));
    }
    if nslc > MAX_NSLC {
        return Err(GpuError::DeviceRequest(format!(
            "nslc {nslc} exceeds GPU MAX_NSLC {MAX_NSLC}"
        )));
    }
    let (out_rows, out_cols) = strides.out_shape((rows, cols));
    let mask = mask_buffer(neighbors);
    let params = build_params(
        (nslc, rows, cols),
        half,
        strides,
        (out_rows, out_cols),
        &mask,
    );

    let flat = flatten(stack);
    let out = run(
        ctx,
        &flat,
        &mask,
        &params,
        out_rows * out_cols * nslc * nslc,
    )?;
    pack(out, (out_rows, out_cols, nslc))
}

/// Flatten the SHP neighbor mask to `u32` (1 = keep), or a 1-element dummy when
/// no mask is given (the binding must always exist).
fn mask_buffer(neighbors: Option<ArrayView4<bool>>) -> Vec<u32> {
    match neighbors {
        Some(n) => n.iter().map(|&keep| u32::from(keep)).collect(),
        None => vec![0_u32],
    }
}

fn build_params(
    dims: (usize, usize, usize),
    half: HalfWindow,
    strides: Strides,
    out: (usize, usize),
    mask: &[u32],
) -> Params {
    let (nslc, rows, cols) = dims;
    Params {
        nslc: nslc as u32,
        rows: rows as u32,
        cols: cols as u32,
        half_y: half.y as u32,
        half_x: half.x as u32,
        stride_y: strides.y as u32,
        stride_x: strides.x as u32,
        out_rows: out.0 as u32,
        out_cols: out.1 as u32,
        win_h: (2 * half.y + 1) as u32,
        win_w: (2 * half.x + 1) as u32,
        has_mask: u32::from(mask.len() > 1),
    }
}

/// Flatten `(nslc, rows, cols)` complex stack to `[re, im]` pairs, row-major.
fn flatten(stack: ArrayView3<Cf32>) -> Vec<[f32; 2]> {
    let std = stack.as_standard_layout();
    std.iter().map(|z| [z.re, z.im]).collect()
}

/// Build the pipeline, dispatch one thread per output pixel, read back.
fn run(
    ctx: &GpuContext,
    stack: &[[f32; 2]],
    mask: &[u32],
    params: &Params,
    out_len: usize,
) -> Result<Vec<[f32; 2]>, GpuError> {
    let stack_buf = input_buffer(ctx, "stack", stack);
    let param_buf = uniform_buffer(ctx, "params", params);
    let mask_buf = input_buffer(ctx, "neighbors", mask);
    let out_buf = output_buffer(
        ctx,
        "cov-out",
        (out_len * std::mem::size_of::<[f32; 2]>()) as u64,
    );
    let n_pix = params.out_rows * params.out_cols;
    dispatch_compute(
        ctx,
        include_str!("covariance.wgsl"),
        "covariance",
        &[&stack_buf, &param_buf, &out_buf, &mask_buf],
        n_pix.div_ceil(64),
    );
    readback(ctx, &out_buf, out_len)
}

/// Reshape the flat `[re, im]` output to `(out_rows, out_cols, nslc, nslc)`.
fn pack(flat: Vec<[f32; 2]>, shape: (usize, usize, usize)) -> Result<Array4<Cf32>, GpuError> {
    let (out_rows, out_cols, n) = shape;
    let data: Vec<Cf32> = flat.into_iter().map(|p| Cf32::new(p[0], p[1])).collect();
    Array4::from_shape_vec((out_rows, out_cols, n, n), data)
        .map_err(|_| GpuError::Readback("covariance assembly shape mismatch".into()))
}
