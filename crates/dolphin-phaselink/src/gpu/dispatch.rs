//! Shared wgpu plumbing: input upload, storage buffers, blocking readback.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use super::context::{GpuContext, GpuError};

/// Upload a slice as a read-only `STORAGE` buffer.
pub(crate) fn input_buffer<T: Pod>(ctx: &GpuContext, label: &str, data: &[T]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
}

/// Upload a `Pod` struct as a `UNIFORM` buffer.
pub(crate) fn uniform_buffer<T: Pod>(ctx: &GpuContext, label: &str, value: &T) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// A `read_write` `STORAGE` output buffer of `byte_len` bytes, copyable to host.
pub(crate) fn output_buffer(ctx: &GpuContext, label: &str, byte_len: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Build a compute pipeline from `source` (entry `main`), bind `buffers` in
/// order at bindings `0..n`, and dispatch `workgroups` groups of 64 threads.
pub(crate) fn dispatch_compute(
    ctx: &GpuContext,
    source: &str,
    label: &str,
    buffers: &[&wgpu::Buffer],
    workgroups: u32,
) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
}

/// Copy `src` to a staging buffer, block until mapped, and read back `count`
/// elements of `T`.
///
/// # Errors
/// [`GpuError::Readback`] if the device poll or buffer mapping fails.
pub(crate) fn readback<T: Pod>(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, GpuError> {
    let byte_len = (count * std::mem::size_of::<T>()) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuError::Readback(e.to_string()))?;
    rx.recv()
        .map_err(|e| GpuError::Readback(e.to_string()))?
        .map_err(|e| GpuError::Readback(e.to_string()))?;

    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}
