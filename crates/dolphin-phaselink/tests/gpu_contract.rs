//! GPU phase-linking contract tests (feature `gpu`).
//!
//! Step 1 — the adapter gate: a GPU adapter must initialize on this machine, and
//! on macOS it must be the Metal backend. Later steps add covariance/EVD parity
//! tests against the CPU (f64) reference to f32 tolerance.
#![cfg(feature = "gpu")]

use std::path::{Path, PathBuf};

use std::f64::consts::TAU;

use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_phaselink::gpu::{
    enumerate_adapters, estimate_stack_covariance_gpu, process_coherence_matrices_gpu,
    process_coherence_matrices_gpu_hybrid, unreliable_count, GpuContext, DEFAULT_LINK_ITERS,
};
use dolphin_phaselink::{estimate_stack_covariance, process_coherence_matrices, StackEstimate};
use ndarray::{Array3, Array4};

/// Sentinel-1 C-band displacement scale: |Δd| = λ/(4π)·|Δφ|, mm per rad.
const MM_PER_RAD: f64 = 55.465_76 / (4.0 * std::f64::consts::PI);

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

/// `|⟨a, b⟩|` normalized — global-phase-invariant eigenvector overlap.
fn cos_sim(a: &[Cf32], b: &[Cf32]) -> f64 {
    let inner: Cf64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            Cf64::new(x.re.into(), x.im.into()) * Cf64::new(y.re.into(), y.im.into()).conj()
        })
        .sum();
    let na = a.iter().map(|z| z.norm_sqr() as f64).sum::<f64>().sqrt();
    let nb = b.iter().map(|z| z.norm_sqr() as f64).sum::<f64>().sqrt();
    inner.norm() / (na * nb)
}

/// Wrap a phase difference into (-π, π].
fn wrap(d: f64) -> f64 {
    let w = d.rem_euclid(TAU);
    if w > TAU / 2.0 {
        w - TAU
    } else {
        w
    }
}

fn to_c64(a: &Array4<Cf32>) -> Array4<Cf64> {
    a.mapv(|z| Cf64::new(z.re.into(), z.im.into()))
}

/// Compare GPU vs CPU phase linking (EVD if `use_evd`, else EMI) over a
/// coherence stack: min eigenvector overlap and max referenced-phase Δ (rad).
fn compare_link(ctx: &GpuContext, c32: &Array4<Cf32>, use_evd: bool) -> (f64, f64) {
    let cpu = process_coherence_matrices(to_c64(c32).view(), use_evd, 0.0, 0.0, 0);
    let gpu =
        process_coherence_matrices_gpu(ctx, c32.view(), use_evd, 0, DEFAULT_LINK_ITERS).unwrap();
    let (nslc, rows, cols) = cpu.cpx_phase.dim();

    let mut min_sim = 1.0_f64;
    let mut max_dphi = 0.0_f64;
    for pix in 0..rows * cols {
        let (r, c) = (pix / cols, pix % cols);
        let cv: Vec<Cf32> = (0..nslc)
            .map(|t| {
                let z = cpu.cpx_phase[(t, r, c)];
                Cf32::new(z.re as f32, z.im as f32)
            })
            .collect();
        let gv: Vec<Cf32> = (0..nslc).map(|t| gpu.cpx_phase[(t, r, c)]).collect();
        min_sim = min_sim.min(cos_sim(&cv, &gv));
        let dphi = (0..nslc)
            .map(|t| wrap(f64::from(gv[t].arg()) - f64::from(cv[t].arg())).abs())
            .fold(0.0_f64, f64::max);
        max_dphi = max_dphi.max(dphi);
    }
    (min_sim, max_dphi)
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

#[test]
fn gpu_evd_recovers_analytic_rank_one() {
    const N: usize = 8;
    let theta: Vec<f32> = (0..N).map(|i| 0.5 * i as f32).collect();
    let mut c = Array4::<Cf32>::zeros((1, 1, N, N));
    for i in 0..N {
        for j in 0..N {
            c[(0, 0, i, j)] = Cf32::from_polar(1.0, theta[i] - theta[j]);
        }
    }
    let ctx = GpuContext::new().expect("GPU device");
    let (min_sim, max_dphi) = compare_link(&ctx, &c, true);
    eprintln!("analytic EVD: overlap={min_sim:.6} max Δφ={max_dphi:.3e} rad");
    assert!(min_sim > 0.999, "eigenvector overlap {min_sim} <= 0.999");
    assert!(
        max_dphi < 1e-3,
        "referenced phase delta {max_dphi} rad exceeds 1e-3"
    );
}

#[test]
fn gpu_evd_matches_cpu_on_oracle_ds() {
    let dir = fixtures();
    if !dir.join("cov_C.npy").exists() {
        eprintln!("skipping gpu_evd_matches_cpu_on_oracle_ds: no fixtures");
        return;
    }
    let c: Array4<Cf32> = ndarray_npy::read_npy(dir.join("cov_C.npy")).unwrap();
    let ctx = GpuContext::new().expect("GPU device");
    let (min_sim, max_dphi) = compare_link(&ctx, &c, true);
    eprintln!("oracle DS EVD: overlap={min_sim:.6} max Δφ={max_dphi:.3e} rad");
    assert!(min_sim > 0.999, "eigenvector overlap {min_sim} <= 0.999");
    assert!(
        max_dphi < 5e-3,
        "referenced phase delta {max_dphi} rad exceeds 5e-3"
    );
}

#[test]
fn gpu_emi_matches_cpu_on_oracle_ds() {
    let dir = fixtures();
    if !dir.join("cov_C.npy").exists() {
        eprintln!("skipping gpu_emi_matches_cpu_on_oracle_ds: no fixtures");
        return;
    }
    let c: Array4<Cf32> = ndarray_npy::read_npy(dir.join("cov_C.npy")).unwrap();
    let ctx = GpuContext::new().expect("GPU device");
    let (min_sim, max_dphi) = compare_link(&ctx, &c, false);
    eprintln!("oracle DS EMI: overlap={min_sim:.6} max Δφ={max_dphi:.3e} rad");
    assert!(
        min_sim > 0.999,
        "EMI eigenvector overlap {min_sim} <= 0.999"
    );
    assert!(
        max_dphi < 5e-3,
        "EMI referenced phase delta {max_dphi} rad exceeds 5e-3"
    );
}

/// Max referenced-phase Δ (rad) over ALL pixels between two f64 stack estimates.
fn max_dphi_all(a: &StackEstimate, b: &StackEstimate) -> f64 {
    let (nslc, rows, cols) = a.cpx_phase.dim();
    (0..rows * cols)
        .map(|pix| {
            let (r, c) = (pix / cols, pix % cols);
            (0..nslc)
                .map(|t| wrap(a.cpx_phase[(t, r, c)].arg() - b.cpx_phase[(t, r, c)].arg()).abs())
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max)
}

/// Item 1 (gating): the GPU EMI hybrid must match CPU EMI to **sub-mm on every
/// pixel** of the real Mexico stack — the spike's π-rad (13.9 mm) tail removed by
/// recomputing the flagged near-degenerate minority on the f64 CPU path.
#[test]
fn gpu_emi_hybrid_no_pi_tail_on_real_stack() {
    let dir = fixtures();
    if !dir.join("real_cov_C.npy").exists() {
        eprintln!(
            "skipping gpu_emi_hybrid_no_pi_tail_on_real_stack: run oracle/gen_phaselink_real.py"
        );
        return;
    }
    let c32: Array4<Cf32> = ndarray_npy::read_npy(dir.join("real_cov_C.npy")).unwrap();
    let c64 = to_c64(&c32);
    let ctx = GpuContext::new().expect("GPU device");

    let cpu = process_coherence_matrices(c64.view(), false, 0.0, 0.0, 0);
    let hybrid = process_coherence_matrices_gpu_hybrid(
        &ctx,
        c64.view(),
        false,
        0.0,
        0.0,
        0,
        DEFAULT_LINK_ITERS,
    )
    .unwrap();

    // Diagnostic: how many pixels the kernel flagged for CPU recompute.
    let raw =
        process_coherence_matrices_gpu(&ctx, c32.view(), false, 0, DEFAULT_LINK_ITERS).unwrap();
    let n_pix = raw.reliable.len();
    let n_flag = unreliable_count(&raw);

    let max_dphi = max_dphi_all(&cpu, &hybrid);
    let max_mm = max_dphi * MM_PER_RAD;
    eprintln!(
        "hybrid EMI vs CPU [ALL {n_pix} px]: max Δφ={max_dphi:.3e} rad ({max_mm:.4} mm); \
         CPU-recomputed {n_flag} px ({:.1}%)",
        100.0 * n_flag as f64 / n_pix as f64
    );
    assert!(
        max_mm < 1.0,
        "max Δφ {max_mm:.4} mm over all pixels is not sub-mm — π-rad tail not removed"
    );
}

#[test]
fn gpu_emi_falls_back_to_evd_on_singular_gamma() {
    // Rank-1 unit coherence ⇒ Γ = all-ones (singular) ⇒ Cholesky fails ⇒ EVD.
    const N: usize = 8;
    let theta: Vec<f32> = (0..N).map(|i| 0.5 * i as f32).collect();
    let mut c = Array4::<Cf32>::zeros((1, 1, N, N));
    for i in 0..N {
        for j in 0..N {
            c[(0, 0, i, j)] = Cf32::from_polar(1.0, theta[i] - theta[j]);
        }
    }
    let ctx = GpuContext::new().expect("GPU device");
    let gpu = process_coherence_matrices_gpu(&ctx, c.view(), false, 0, DEFAULT_LINK_ITERS).unwrap();
    assert_eq!(gpu.estimator[(0, 0)], 0, "singular Γ must fall back to EVD");
    let (min_sim, max_dphi) = compare_link(&ctx, &c, true);
    eprintln!("EMI→EVD fallback: overlap={min_sim:.6} max Δφ={max_dphi:.3e} rad");
    assert!(min_sim > 0.999);
}
