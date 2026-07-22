//! Item 5: the GPU backend, driven through the real `run_sequential` pipeline
//! (covariance → EVD/EMI hybrid → compress → temporal coherence), agrees with the
//! CPU backend on the same config — "one config, CPU or GPU, same result".
//!
//! The authoritative all-pixel accuracy check is on the **real** Mexico stack
//! (`bench/GPU.md` / VALIDATION.md, item 6). This fast fixture-free check uses a
//! realistic synthetic DS stack and asserts the two backends agree to **sub-mm at
//! p99** end-to-end; a tiny fraction of near-degenerate pixels (ambiguous EMI
//! optima, masked downstream by coherence) may differ — reported, not hidden.
//!
//! Runs only in the default (gpu) build and only when a GPU adapter is present.
#![cfg(feature = "gpu")]

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::{ComputeEngine, ResolvedBackend};
use dolphin_workflows::{run_sequential, SequentialConfig};
use ndarray::Array3;

/// Sentinel-1 C-band displacement scale, mm per rad.
const MM_PER_RAD: f64 = 55.465_76 / (4.0 * std::f64::consts::PI);

/// A realistic distributed-scatterer stack `(nslc, n, n)`: each sample mixes a
/// shared temporal signal `√γ·e^{jθ[slc]}` with independent per-sample circular
/// decorrelation `√(1−γ)·e^{jψ}`. The windowed coherence is ≈ γ = 0.8, so
/// `Γ = |C|` is well-conditioned (min eigenvalue ≈ 1−γ) on the bulk of pixels —
/// EMI is well posed, unlike a degenerate rank-1 (unit-phasor) stack.
fn synthetic_stack(nslc: usize, n: usize) -> Array3<Cf64> {
    let gamma = 0.8_f64;
    let (cs, cn) = (gamma.sqrt(), (1.0 - gamma).sqrt());
    Array3::from_shape_fn((nslc, n, n), |(slc, r, c)| {
        let theta = 0.7 * slc as f64; // shared temporal phase history
        let k = (slc as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((r as u32).wrapping_mul(40_503))
            .wrapping_add((c as u32).wrapping_mul(2_246_822_519));
        let psi = f64::from(k) / f64::from(u32::MAX) * std::f64::consts::TAU;
        Cf64::from_polar(cs, theta) + Cf64::from_polar(cn, psi)
    })
}

fn config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 5,
        max_num_compressed: 2,
        half_window: HalfWindow { y: 2, x: 2 },
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: true,
    }
}

/// Wrap into (-π, π].
fn wrap(d: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let w = d.rem_euclid(t);
    if w > t / 2.0 {
        w - t
    } else {
        w
    }
}

#[test]
fn gpu_backend_matches_cpu_end_to_end() {
    let (nslc, n) = (8usize, 130usize); // 130² = 16900 ≥ the ~128² crossover
    let stack = synthetic_stack(nslc, n);
    let cfg = config();

    let gpu_engine = ComputeEngine::new(ComputeBackend::Gpu);
    if gpu_engine.resolved(n * n, nslc) != ResolvedBackend::Gpu {
        eprintln!("skipping gpu_backend_matches_cpu_end_to_end: no GPU adapter");
        return;
    }
    let cpu = run_sequential(stack.view(), &cfg, &ComputeEngine::new(ComputeBackend::Cpu)).unwrap();
    let gpu = run_sequential(stack.view(), &cfg, &gpu_engine).unwrap();
    assert_eq!(cpu.cpx_phase.dim(), gpu.cpx_phase.dim());
    let cpu_phase_coherence = cpu
        .phase_linking_coherence
        .as_ref()
        .expect("coherence enabled on CPU");
    let gpu_phase_coherence = gpu
        .phase_linking_coherence
        .as_ref()
        .expect("coherence enabled on GPU");
    let coherence_max_error = cpu_phase_coherence
        .iter()
        .zip(gpu_phase_coherence)
        .filter(|(cpu, gpu)| cpu.is_finite() && gpu.is_finite())
        .map(|(cpu, gpu)| (cpu - gpu).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        coherence_max_error < 1e-3,
        "phase-linking coherence backend error {coherence_max_error}"
    );

    // Per-pixel max wrapped phase delta across dates (unit-magnitude phasors).
    let (nd, rows, cols) = cpu.cpx_phase.dim();
    let mut per_pixel: Vec<f64> = (0..rows * cols)
        .map(|pix| {
            let (r, c) = (pix / cols, pix % cols);
            (0..nd)
                .map(|t| {
                    wrap(cpu.cpx_phase[(t, r, c)].arg() - gpu.cpx_phase[(t, r, c)].arg()).abs()
                })
                .fold(0.0_f64, f64::max)
        })
        .collect();
    per_pixel.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pick = |q: f64| per_pixel[((per_pixel.len() - 1) as f64 * q) as usize];
    let (median, p99, max) = (pick(0.5), pick(0.99), *per_pixel.last().unwrap());
    let submm = 1.0 / MM_PER_RAD; // rad for 1 mm
    let over = per_pixel.iter().filter(|&&d| d > submm).count();
    eprintln!(
        "end-to-end GPU vs CPU [{}px]: median {:.2e} rad, p99 {:.2e} rad ({:.3} mm), max {:.3} rad ({:.3} mm); {over} px (>{:.2}%) over 1 mm",
        per_pixel.len(),
        median,
        p99,
        p99 * MM_PER_RAD,
        max,
        max * MM_PER_RAD,
        100.0 * over as f64 / per_pixel.len() as f64,
    );

    assert!(median < 5e-3, "end-to-end median {median} rad not ~exact");
    assert!(
        p99 < submm,
        "end-to-end p99 {} mm exceeds sub-mm",
        p99 * MM_PER_RAD
    );
    // Near-degenerate ambiguous pixels (masked downstream) may differ; they must
    // stay a tiny minority.
    assert!(
        over < per_pixel.len() / 100,
        "{over} pixels over 1 mm exceeds 1% — backends materially disagree"
    );
}
