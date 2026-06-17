//! GPU phase-linking accuracy + speed harness (first-class backend).
//!
//! Accuracy: loads the real Mexico stack's dolphin v0.35.0 coherence + EMI phase
//! (`oracle/fixtures/real_cov_C.npy`, `real_phase_emi.npy`) and reports the **GPU
//! hybrid** EMI (f32 kernel + f64 CPU recompute of flagged pixels) vs CPU(f64) vs
//! oracle referenced-phase deltas across *all* pixels — showing the raw-f32 π-rad
//! tail and that the hybrid removes it.
//!
//! Speed: times the **end-to-end** path (covariance, phase-linking, host↔device
//! transfer, and the hybrid recompute) for GPU vs CPU, on the real 384² stack and
//! a synthetic size sweep, to locate the crossover where the GPU starts to win.
//!
//! Run: `cargo run -p dolphin-phaselink --release --example gpu_bench`

use std::path::{Path, PathBuf};
use std::time::Instant;

use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_phaselink::gpu::{
    estimate_stack_covariance_gpu, process_coherence_matrices_gpu,
    process_coherence_matrices_gpu_hybrid, unreliable_count, GpuContext, DEFAULT_LINK_ITERS,
};
use dolphin_phaselink::{estimate_stack_covariance, process_coherence_matrices, StackEstimate};
use ndarray::{Array3, Array4};

/// Sentinel-1 C-band: |displacement| = λ/(4π)·|Δφ|.
const MM_PER_RAD: f64 = 55.465_76 / (4.0 * std::f64::consts::PI);
const REPS: usize = 5;
/// Real Mexico config: dolphin `HalfWindow(y=5, x=11)`, strides (1, 1).
const REAL_HALF: HalfWindow = HalfWindow { y: 5, x: 11 };
const STRIDES: Strides = Strides { y: 1, x: 1 };

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn to_c64_4(a: &Array4<Cf32>) -> Array4<Cf64> {
    a.mapv(|z| Cf64::new(z.re.into(), z.im.into()))
}

fn to_c64_3(a: &Array3<Cf32>) -> Array3<Cf64> {
    a.mapv(|z| Cf64::new(z.re.into(), z.im.into()))
}

fn to_c32_3(a: &Array3<Cf64>) -> Array3<Cf32> {
    a.mapv(|z| Cf32::new(z.re as f32, z.im as f32))
}

fn wrap(d: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let w = d.rem_euclid(t);
    if w > t / 2.0 {
        w - t
    } else {
        w
    }
}

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
    match na * nb > 0.0 {
        true => inner.norm() / (na * nb),
        false if na + nb == 0.0 => 1.0,
        false => 0.0,
    }
}

/// Distribution of referenced-phase deltas (rad) between two phase cubes.
struct PhaseDelta {
    median_rad: f64,
    p99_rad: f64,
    max_rad: f64,
    overlap_min: f64,
}

fn phase_delta(a: &Array3<Cf32>, b: &Array3<Cf32>, keep: impl Fn(usize) -> bool) -> PhaseDelta {
    let (nslc, rows, cols) = a.dim();
    let mut deltas = Vec::new();
    let mut overlap_min = 1.0_f64;
    for pix in (0..rows * cols).filter(|&p| keep(p)) {
        let (r, c) = (pix / cols, pix % cols);
        let av: Vec<Cf32> = (0..nslc).map(|t| a[(t, r, c)]).collect();
        let bv: Vec<Cf32> = (0..nslc).map(|t| b[(t, r, c)]).collect();
        overlap_min = overlap_min.min(cos_sim(&av, &bv));
        deltas
            .extend((0..nslc).map(|t| wrap(f64::from(av[t].arg()) - f64::from(bv[t].arg())).abs()));
    }
    deltas.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let pick = |q: f64| deltas[((deltas.len() - 1) as f64 * q) as usize];
    PhaseDelta {
        median_rad: pick(0.5),
        p99_rad: pick(0.99),
        max_rad: *deltas.last().unwrap(),
        overlap_min,
    }
}

fn report(label: &str, d: &PhaseDelta) {
    println!(
        "  {label:<28} overlap≥{:.4}  median {:.2e} rad ({:.2e} mm)  p99 {:.2e} rad ({:.3} mm)  max {:.2e} rad ({:.3} mm)",
        d.overlap_min,
        d.median_rad,
        d.median_rad * MM_PER_RAD,
        d.p99_rad,
        d.p99_rad * MM_PER_RAD,
        d.max_rad,
        d.max_rad * MM_PER_RAD,
    );
}

fn cpu_phase(est: &StackEstimate) -> Array3<Cf32> {
    est.cpx_phase.mapv(|z| Cf32::new(z.re as f32, z.im as f32))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median wall-clock (seconds) of `f` over `REPS` warm runs (one warm-up first).
fn time_warm<T>(mut f: impl FnMut() -> T) -> f64 {
    let _ = f();
    let mut ts = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        let _ = f();
        ts.push(t0.elapsed().as_secs_f64());
    }
    median(ts)
}

fn accuracy(ctx: &GpuContext) {
    let dir = fixtures();
    if !dir.join("real_cov_C.npy").exists() {
        println!("ACCURACY: skipped (run oracle/gen_phaselink_real.py first)");
        return;
    }
    let c: Array4<Cf32> = ndarray_npy::read_npy(dir.join("real_cov_C.npy")).unwrap();
    let oracle: Array3<Cf32> = ndarray_npy::read_npy(dir.join("real_phase_emi.npy")).unwrap();
    let oracle = oracle
        .permuted_axes([2, 0, 1])
        .as_standard_layout()
        .to_owned();
    let c64 = to_c64_4(&c);

    let cpu_emi = cpu_phase(&process_coherence_matrices(c64.view(), false, 0.0, 0.0, 0));
    let cpu_evd = cpu_phase(&process_coherence_matrices(c64.view(), true, 0.0, 0.0, 0));
    let gpu_evd =
        process_coherence_matrices_gpu(ctx, c.view(), true, 0.0, 0.0, 0, DEFAULT_LINK_ITERS)
            .unwrap();
    let gpu_raw =
        process_coherence_matrices_gpu(ctx, c.view(), false, 0.0, 0.0, 0, DEFAULT_LINK_ITERS)
            .unwrap();
    let gpu_hybrid = cpu_phase(
        &process_coherence_matrices_gpu_hybrid(
            ctx,
            c64.view(),
            false,
            0.0,
            0.0,
            0,
            DEFAULT_LINK_ITERS,
        )
        .unwrap(),
    );
    let n_pix = gpu_raw.reliable.len();
    let n_flag = unreliable_count(&gpu_raw);

    let all = |_: usize| true;
    println!(
        "ACCURACY — real Mexico stack (13 acqs, 384², half_window 11×5), ALL {n_pix} px; \
         hybrid recomputed {n_flag} px ({:.1}%) on the f64 CPU:",
        100.0 * n_flag as f64 / n_pix as f64
    );
    report(
        "EVD GPU(f32) vs CPU [all]",
        &phase_delta(&cpu_evd, &gpu_evd.cpx_phase, all),
    );
    report(
        "EMI raw GPU(f32) vs CPU [all]",
        &phase_delta(&cpu_emi, &gpu_raw.cpx_phase, all),
    );
    report(
        "EMI hybrid vs CPU [all]",
        &phase_delta(&cpu_emi, &gpu_hybrid, all),
    );
    report(
        "EMI hybrid vs oracle [all]",
        &phase_delta(&oracle, &gpu_hybrid, all),
    );
    report(
        "EMI CPU vs oracle [all]",
        &phase_delta(&oracle, &cpu_emi, all),
    );
}

/// End-to-end CPU phase linking: f64 covariance + EMI.
fn e2e_cpu(stack: &Array3<Cf64>) -> StackEstimate {
    let c = estimate_stack_covariance(stack.view(), REAL_HALF, STRIDES, None).unwrap();
    process_coherence_matrices(c.view(), false, 0.0, 0.0, 0)
}

/// End-to-end GPU phase linking through the engine path: f32 covariance →
/// upcast → hybrid (f32 kernel + f64 CPU recompute of flagged pixels).
fn e2e_gpu(ctx: &GpuContext, stack32: &Array3<Cf32>) -> StackEstimate {
    let c32 = estimate_stack_covariance_gpu(ctx, stack32.view(), REAL_HALF, STRIDES, None).unwrap();
    let c64 = to_c64_4(&c32);
    process_coherence_matrices_gpu_hybrid(ctx, c64.view(), false, 0.0, 0.0, 0, DEFAULT_LINK_ITERS)
        .unwrap()
}

/// A realistic distributed-scatterer SLC stack `(nslc, s, s)` at coherence ≈ 0.8.
fn synth_slc(s: usize, nslc: usize) -> Array3<Cf64> {
    let gamma = 0.8_f64;
    let (cs, cn) = (gamma.sqrt(), (1.0 - gamma).sqrt());
    Array3::from_shape_fn((nslc, s, s), |(slc, r, c)| {
        let theta = 0.7 * slc as f64;
        let k = (slc as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((r as u32).wrapping_mul(40_503))
            .wrapping_add((c as u32).wrapping_mul(2_246_822_519));
        let psi = f64::from(k) / f64::from(u32::MAX) * std::f64::consts::TAU;
        Cf64::from_polar(cs, theta) + Cf64::from_polar(cn, psi)
    })
}

fn speed(ctx: &GpuContext) {
    println!("\nSPEED — END-TO-END (covariance + phase-link + readback + hybrid recompute), median of {REPS} warm reps:");

    if let Ok(real) =
        ndarray_npy::read_npy::<_, Array3<Cf32>>(fixtures().join("real_slc_stack.npy"))
    {
        let real64 = to_c64_3(&real);
        let cpu_s = time_warm(|| e2e_cpu(&real64));
        let gpu_s = time_warm(|| e2e_gpu(ctx, &real));
        println!(
            "  real 384² nslc=13:  CPU {cpu_s:.4}s  GPU {gpu_s:.4}s  speedup {:.2}×",
            cpu_s / gpu_s
        );
    }

    println!("\n  synthetic sweep (nslc=13):");
    println!(
        "  {:>7}  {:>10}  {:>10}  {:>8}",
        "size", "CPU (s)", "GPU (s)", "speedup"
    );
    let mut crossover = None;
    for &s in &[64_usize, 128, 192, 256, 384, 512] {
        let stack = synth_slc(s, 13);
        let stack32 = to_c32_3(&stack);
        let cpu_s = time_warm(|| e2e_cpu(&stack));
        let gpu_s = time_warm(|| e2e_gpu(ctx, &stack32));
        let speedup = cpu_s / gpu_s;
        if speedup >= 1.0 && crossover.is_none() {
            crossover = Some(s);
        }
        println!("  {s:>5}²  {cpu_s:>10.4}  {gpu_s:>10.4}  {speedup:>7.2}×");
    }
    match crossover {
        Some(s) => println!("  crossover: GPU wins end-to-end at ≥ {s}² pixels"),
        None => println!("  crossover: CPU wins end-to-end at every tested size"),
    }
}

fn main() {
    let ctx = GpuContext::new().expect("GPU device");
    println!("GPU: {}\n", ctx.adapter);
    accuracy(&ctx);
    speed(&ctx);
}
