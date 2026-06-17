//! Runtime backend-selection contracts for [`ComputeEngine`] (item 4).
//!
//! These run in both the default (gpu) and `no-gpu` builds. The no-adapter
//! fallback is exercised by `ComputeEngine::new_simulating_no_adapter`, which
//! takes the same path as a host where `GpuContext::new()` finds no adapter.

use dolphin_core::config::ComputeBackend;
use dolphin_core::Cf64;
use dolphin_phaselink::{process_coherence_matrices, ComputeEngine, ResolvedBackend};
use ndarray::Array4;

/// A small Hermitian rank-1 coherence stack `(rows, cols, nslc, nslc)`.
fn coherence_stack(rows: usize, cols: usize, nslc: usize) -> Array4<Cf64> {
    Array4::from_shape_fn((rows, cols, nslc, nslc), |(r, c, i, j)| {
        let theta = 0.3 * (i as f64 - j as f64) * (1.0 + 0.01 * (r + c) as f64);
        Cf64::from_polar(1.0_f64, theta)
    })
}

/// Phase cubes equal to f64 bit-precision (the no-adapter path *is* the CPU path).
fn phase_eq(a: &ndarray::Array3<Cf64>, b: &ndarray::Array3<Cf64>) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x.re == y.re && x.im == y.im)
}

/// DoD: no GPU adapter ⇒ automatic CPU fallback, never a panic. With a simulated
/// missing adapter, `Gpu` mode resolves to CPU and returns the exact CPU result.
#[test]
fn no_adapter_falls_back_to_cpu_without_panic() {
    let c = coherence_stack(12, 10, 6);
    let (rows, cols, nslc, _) = c.dim();

    let engine = ComputeEngine::new_simulating_no_adapter(ComputeBackend::Gpu);
    assert_eq!(
        engine.resolved(rows * cols, nslc),
        ResolvedBackend::Cpu,
        "with no adapter, Gpu must resolve to Cpu"
    );

    let cpu = process_coherence_matrices(c.view(), false, 0.0, 0.0, 0);
    let got = engine.estimate(c.view(), false, 0.0, 0.0, 0);
    assert!(
        phase_eq(&cpu.cpx_phase, &got.cpx_phase),
        "fallback estimate must equal the CPU reference exactly"
    );
}

/// `Cpu` mode always resolves to CPU regardless of size or GPU presence.
#[test]
fn cpu_mode_always_resolves_cpu() {
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    assert_eq!(engine.resolved(512 * 512, 13), ResolvedBackend::Cpu);
    assert_eq!(engine.resolved(8 * 8, 13), ResolvedBackend::Cpu);
}

/// `Auto` never uses the GPU below the ~128² crossover, on any build.
#[test]
fn auto_uses_cpu_below_crossover() {
    let engine = ComputeEngine::new(ComputeBackend::Auto);
    assert_eq!(engine.resolved(64 * 64, 13), ResolvedBackend::Cpu);
}

/// With a real adapter (default build on a GPU host), `Auto` crosses over to the
/// GPU at scale and `Gpu` is honored; oversized `nslc` always falls back to CPU.
#[cfg(feature = "gpu")]
#[test]
fn gpu_present_resolution() {
    let engine = ComputeEngine::new(ComputeBackend::Gpu);
    // Skip the assertions if this host genuinely has no adapter (e.g. headless CI).
    if engine.resolved(200 * 200, 13) == ResolvedBackend::Cpu {
        eprintln!("skipping gpu_present_resolution: no GPU adapter on this host");
        return;
    }
    let auto = ComputeEngine::new(ComputeBackend::Auto);
    assert_eq!(
        auto.resolved(200 * 200, 13),
        ResolvedBackend::Gpu,
        "auto large → gpu"
    );
    assert_eq!(
        auto.resolved(64 * 64, 13),
        ResolvedBackend::Cpu,
        "auto small → cpu"
    );
    // nslc beyond the GPU kernel cap falls back to CPU even in Gpu mode.
    assert_eq!(
        engine.resolved(200 * 200, 64),
        ResolvedBackend::Cpu,
        "nslc>cap → cpu"
    );
}
