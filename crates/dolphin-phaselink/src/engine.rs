//! Runtime compute-backend selection (CPU faer / GPU wgpu) with CPU fallback.
//!
//! [`ComputeEngine`] is the single entry the pipeline uses for covariance +
//! phase-linking. It resolves the configured [`ComputeBackend`] once at
//! construction (acquiring a GPU context if appropriate) and routes each call:
//! `Auto` runs on the GPU at/above the ~128² crossover and on the CPU below;
//! `Gpu` forces the GPU where supported; `Cpu` is always the f64 reference. With
//! no GPU adapter, an unsupported `nslc`, or a `no-gpu` build, every path falls
//! back to the CPU — never a panic. The CPU (faer, f64) result is the correctness
//! reference; GPU output is upcast to f64 so downstream stages see one type.

use dolphin_core::config::ComputeBackend;
use dolphin_core::{Cf64, HalfWindow, Strides};
use ndarray::{Array4, ArrayView3, ArrayView4};

use crate::{estimate_stack_covariance, process_coherence_matrices, StackEstimate};

/// `Auto` uses the GPU at/above this output-pixel count (≈128²); below it, the
/// CPU wins (the spike's measured crossover — dispatch + readback dominate).
#[cfg(feature = "gpu")]
const GPU_CROSSOVER_PIXELS: usize = 128 * 128;

/// Backend actually used for a call, after resolution + fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedBackend {
    /// CPU faer (f64) reference path.
    Cpu,
    /// GPU wgpu (f32 + hybrid CPU fallback for ill-conditioned EMI pixels).
    Gpu,
}

/// A constructed compute engine holding the (optional) GPU context.
pub struct ComputeEngine {
    #[cfg(feature = "gpu")]
    requested: ComputeBackend,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuContext>,
}

impl ComputeEngine {
    /// Build an engine for `requested`, acquiring a GPU context when the mode and
    /// build allow it. A missing adapter logs a warning and falls back to the CPU.
    #[must_use]
    pub fn new(requested: ComputeBackend) -> Self {
        Self::build(requested, true)
    }

    /// Build an engine that behaves as if the host had **no GPU adapter** — the
    /// fallback path exercised by the no-adapter contract test.
    #[doc(hidden)]
    #[must_use]
    pub fn new_simulating_no_adapter(requested: ComputeBackend) -> Self {
        Self::build(requested, false)
    }

    #[cfg(feature = "gpu")]
    fn build(requested: ComputeBackend, try_gpu: bool) -> Self {
        let want = matches!(requested, ComputeBackend::Gpu | ComputeBackend::Auto);
        let gpu = (want && try_gpu).then(acquire_gpu).flatten();
        if want && gpu.is_none() {
            tracing::warn!(
                ?requested,
                "no GPU adapter available — falling back to the CPU backend"
            );
        }
        Self { requested, gpu }
    }

    #[cfg(not(feature = "gpu"))]
    fn build(requested: ComputeBackend, _try_gpu: bool) -> Self {
        if matches!(requested, ComputeBackend::Gpu) {
            tracing::warn!("built without GPU support (no-gpu) — using the CPU backend");
        }
        Self {}
    }

    /// Which backend a call over `n_pix` output pixels at `nslc` will use.
    #[must_use]
    pub fn resolved(&self, n_pix: usize, nslc: usize) -> ResolvedBackend {
        match self.gpu_ready(n_pix, nslc) {
            true => ResolvedBackend::Gpu,
            false => ResolvedBackend::Cpu,
        }
    }

    /// Sliding-window coherence over `stack`, on the resolved backend.
    ///
    /// # Errors
    /// Returns `Err` if the covariance window exceeds the stack.
    pub fn covariance(
        &self,
        stack: ArrayView3<Cf64>,
        half: HalfWindow,
        strides: Strides,
        neighbors: Option<ArrayView4<bool>>,
    ) -> Result<Array4<Cf64>, &'static str> {
        #[cfg(feature = "gpu")]
        {
            let (nslc, rows, cols) = stack.dim();
            let (out_rows, out_cols) = strides.out_shape((rows, cols));
            let gpu = self
                .gpu_ready(out_rows * out_cols, nslc)
                .then(|| self.covariance_gpu(stack, half, strides, neighbors))
                .flatten();
            if let Some(c) = gpu {
                return Ok(c);
            }
        }
        estimate_stack_covariance(stack, half, strides, neighbors)
    }

    /// EVD/EMI phase linking over a `(out_rows, out_cols, nslc, nslc)` coherence
    /// stack, on the resolved backend (GPU uses the hybrid CPU fallback).
    #[must_use]
    pub fn estimate(
        &self,
        c_arrays: ArrayView4<Cf64>,
        use_evd: bool,
        beta: f64,
        zero_correlation_threshold: f64,
        reference_idx: usize,
    ) -> StackEstimate {
        #[cfg(feature = "gpu")]
        {
            let (out_rows, out_cols, nslc, _) = c_arrays.dim();
            let gpu = self
                .gpu_ready(out_rows * out_cols, nslc)
                .then(|| {
                    self.estimate_gpu(
                        c_arrays,
                        use_evd,
                        beta,
                        zero_correlation_threshold,
                        reference_idx,
                    )
                })
                .flatten();
            if let Some(est) = gpu {
                return est;
            }
        }
        process_coherence_matrices(
            c_arrays,
            use_evd,
            beta,
            zero_correlation_threshold,
            reference_idx,
        )
    }

    /// Whether a GPU context exists and the resolved mode + problem size pick it.
    fn gpu_ready(&self, n_pix: usize, nslc: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            let Some(_) = self.gpu.as_ref() else {
                return false;
            };
            if nslc > crate::gpu::MAX_NSLC {
                return false;
            }
            match self.requested {
                ComputeBackend::Cpu => false,
                ComputeBackend::Gpu => true,
                ComputeBackend::Auto => n_pix >= GPU_CROSSOVER_PIXELS,
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (n_pix, nslc);
            false
        }
    }
}

#[cfg(feature = "gpu")]
fn acquire_gpu() -> Option<crate::gpu::GpuContext> {
    match crate::gpu::GpuContext::new() {
        Ok(ctx) => {
            tracing::info!(adapter = %ctx.adapter, "GPU compute backend active");
            Some(ctx)
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU adapter init failed");
            None
        }
    }
}

#[cfg(feature = "gpu")]
impl ComputeEngine {
    /// GPU covariance (f32), upcast to f64; `None` on dispatch error (→ CPU).
    fn covariance_gpu(
        &self,
        stack: ArrayView3<Cf64>,
        half: HalfWindow,
        strides: Strides,
        neighbors: Option<ArrayView4<bool>>,
    ) -> Option<Array4<Cf64>> {
        use dolphin_core::Cf32;
        let ctx = self.gpu.as_ref()?;
        let stack32 = stack.mapv(|z| Cf32::new(z.re as f32, z.im as f32));
        match crate::gpu::estimate_stack_covariance_gpu(
            ctx,
            stack32.view(),
            half,
            strides,
            neighbors,
        ) {
            Ok(c) => Some(c.mapv(|z| Cf64::new(z.re.into(), z.im.into()))),
            Err(e) => {
                tracing::warn!(error = %e, "GPU covariance failed — CPU fallback");
                None
            }
        }
    }

    /// GPU hybrid phase linking (f64 out); `None` on dispatch error (→ CPU).
    fn estimate_gpu(
        &self,
        c_arrays: ArrayView4<Cf64>,
        use_evd: bool,
        beta: f64,
        zero_correlation_threshold: f64,
        reference_idx: usize,
    ) -> Option<StackEstimate> {
        let ctx = self.gpu.as_ref()?;
        match crate::gpu::process_coherence_matrices_gpu_hybrid(
            ctx,
            c_arrays,
            use_evd,
            beta,
            zero_correlation_threshold,
            reference_idx,
            crate::gpu::DEFAULT_LINK_ITERS,
        ) {
            Ok(est) => Some(est),
            Err(e) => {
                tracing::warn!(error = %e, "GPU phase linking failed — CPU fallback");
                None
            }
        }
    }
}
