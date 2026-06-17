//! GPU phase-linking path (R4 spike), behind the `gpu` cargo feature.
//!
//! A `wgpu` port of the covariance + EVD/EMI kernels that runs on the Apple
//! Metal GPU on this machine (and ports unchanged to a Vulkan/DX12 backend).
//! **Single precision (`f32`)** — Apple GPUs are f32-only and dolphin's own GPU
//! path is f32 too, so this path is *not* bit-identical to the CPU (`f64`) one;
//! the CPU path stays the default and the correctness reference. Complex values
//! cross the host/device boundary as `vec2<f32>` (re, im).
//!
//! One GPU thread per output pixel: each thread builds its `nslc × nslc`
//! coherence matrix and runs the in-shader eigensolver (power iteration for EVD,
//! inverse iteration for EMI), mirroring `covariance.rs`/`estimator.rs`.

mod context;
mod covariance;
mod dispatch;
mod link;

pub use context::{enumerate_adapters, AdapterReport, GpuContext, GpuError};
pub use covariance::{estimate_stack_covariance_gpu, MAX_NSLC};
pub use link::{process_coherence_matrices_gpu, GpuStackEstimate, DEFAULT_LINK_ITERS};
