//! GPU phase-linking contract tests (feature `gpu`).
//!
//! Step 1 — the adapter gate: a GPU adapter must initialize on this machine, and
//! on macOS it must be the Metal backend. Later steps add covariance/EVD parity
//! tests against the CPU (f64) reference to f32 tolerance.
#![cfg(feature = "gpu")]

use std::path::{Path, PathBuf};

use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_phaselink::estimate_stack_covariance;
use dolphin_phaselink::gpu::{enumerate_adapters, estimate_stack_covariance_gpu, GpuContext};
use ndarray::Array3;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

#[test]
fn metal_adapter_initializes() {
    let adapters = enumerate_adapters();
    assert!(
        !adapters.is_empty(),
        "no GPU adapter initialized — GPU path cannot run on this machine"
    );
    for a in &adapters {
        eprintln!("adapter: {a}");
    }

    let ctx = GpuContext::new().expect("acquire a GPU device");
    eprintln!("bound: {}", ctx.adapter);

    if cfg!(target_os = "macos") {
        assert_eq!(
            ctx.adapter.backend, "Metal",
            "expected the Metal backend on macOS, got {}",
            ctx.adapter.backend
        );
    }
}

#[test]
fn gpu_covariance_matches_cpu() {
    let dir = fixtures();
    if !dir.join("slc_stack.npy").exists() {
        eprintln!("skipping gpu_covariance_matches_cpu: no fixtures at {dir:?}");
        return;
    }
    let stack: Array3<Cf32> = ndarray_npy::read_npy(dir.join("slc_stack.npy")).unwrap();
    let (half, strides) = (HalfWindow { y: 2, x: 2 }, Strides { y: 1, x: 1 });

    let cpu = estimate_stack_covariance(
        stack.mapv(|z| Cf64::new(z.re.into(), z.im.into())).view(),
        half,
        strides,
        None,
    )
    .unwrap();

    let ctx = GpuContext::new().expect("GPU device");
    let gpu = estimate_stack_covariance_gpu(&ctx, stack.view(), half, strides).unwrap();
    assert_eq!(cpu.dim(), gpu.dim());

    let max_err = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - Cf64::new(b.re.into(), b.im.into())).norm())
        .fold(0.0_f64, f64::max);
    eprintln!("gpu-vs-cpu covariance max |Δ| = {max_err:.3e}");
    assert!(
        max_err < 1e-4,
        "GPU covariance f32 delta {max_err} exceeds 1e-4"
    );
}
