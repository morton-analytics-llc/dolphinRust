//! wgpu device/queue acquisition and adapter enumeration (Step 1: the gate).

use std::fmt;

/// A reachable GPU adapter, summarized for logging / the adapter-check gate.
#[derive(Debug, Clone)]
pub struct AdapterReport {
    /// Adapter (GPU) name, e.g. "Apple M-series".
    pub name: String,
    /// Backend: `Metal`, `Vulkan`, `Dx12`, …
    pub backend: String,
    /// Device kind: discrete / integrated / virtual / CPU.
    pub device_type: String,
}

impl fmt::Display for AdapterReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{} / {}]", self.name, self.backend, self.device_type)
    }
}

/// Failure modes for GPU setup — kept narrow; the host always has a CPU path.
#[derive(Debug)]
pub enum GpuError {
    /// No GPU adapter initialized (the Step-1 stop condition).
    NoAdapter,
    /// Adapter found but `request_device` failed.
    DeviceRequest(String),
    /// A compute dispatch failed to map its results back (timeout / validation).
    Readback(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no GPU adapter initialized"),
            Self::DeviceRequest(e) => write!(f, "GPU device request failed: {e}"),
            Self::Readback(e) => write!(f, "GPU readback failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// A live GPU device + queue, the handle every kernel dispatches through.
pub struct GpuContext {
    /// The logical device (owns pipelines, buffers, bind groups).
    pub device: wgpu::Device,
    /// The command queue (submits work, polls for completion).
    pub queue: wgpu::Queue,
    /// The adapter this context bound to (for reporting).
    pub adapter: AdapterReport,
}

/// Enumerate every reachable adapter across all backends.
#[must_use]
pub fn enumerate_adapters() -> Vec<AdapterReport> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .iter()
        .map(|a| describe(&a.get_info()))
        .collect()
}

fn describe(info: &wgpu::AdapterInfo) -> AdapterReport {
    AdapterReport {
        name: info.name.clone(),
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
    }
}

impl GpuContext {
    /// Acquire a high-performance GPU device, preferring a real GPU over CPU.
    ///
    /// # Errors
    /// [`GpuError::NoAdapter`] if no adapter initializes (the Step-1 stop), or
    /// [`GpuError::DeviceRequest`] if the device cannot be created.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;
        let report = describe(&adapter.get_info());
        // Use the adapter's real limits — Metal allows storage buffers far past
        // the conservative 128 MiB default (a 384² coherence stack is ~200 MiB).
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("dolphin-phaselink-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;
        Ok(Self {
            device,
            queue,
            adapter: report,
        })
    }
}
