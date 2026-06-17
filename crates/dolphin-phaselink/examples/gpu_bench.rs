//! GPU phase-linking accuracy + speed harness for the R4 spike.
//!
//! Accuracy: loads the real Mexico stack's dolphin v0.35.0 coherence + EMI phase
//! (`oracle/fixtures/real_cov_C.npy`, `real_phase_emi.npy`; produced by
//! `oracle/gen_phaselink_real.py`) and reports GPU(f32) vs CPU(f64) vs oracle
//! referenced-phase deltas in radians and millimeters.
//!
//! Speed: times CPU vs GPU phase-linking on the real 384² stack and on a
//! synthetic size sweep to locate the crossover where the GPU starts to win.
//!
//! Run: `cargo run -p dolphin-phaselink --features gpu --release --example gpu_bench`

use std::path::{Path, PathBuf};
use std::time::Instant;

use dolphin_core::{Cf32, Cf64};
use dolphin_phaselink::gpu::{process_coherence_matrices_gpu, GpuContext, DEFAULT_LINK_ITERS};
use dolphin_phaselink::{process_coherence_matrices, StackEstimate};
use ndarray::{Array3, Array4};

/// Sentinel-1 C-band: |displacement| = λ/(4π)·|Δφ|.
const MM_PER_RAD: f64 = 55.465_76 / (4.0 * std::f64::consts::PI);
const REPS: usize = 5;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn to_c64(a: &Array4<Cf32>) -> Array4<Cf64> {
    a.mapv(|z| Cf64::new(z.re.into(), z.im.into()))
}

/// Distribution of referenced-phase deltas (rad) between two phase cubes,
/// weighted to the pixels where `ref` has signal (|coherence-driven| phase).
struct PhaseDelta {
    median_rad: f64,
    p99_rad: f64,
    max_rad: f64,
    overlap_min: f64,
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
        // Both zero ⇒ trivially aligned; exactly one zero ⇒ genuine mismatch.
        true => inner.norm() / (na * nb),
        false if na + nb == 0.0 => 1.0,
        false => 0.0,
    }
}

/// Mean off-diagonal `|C_ij|` per pixel — a temporal-coherence proxy used to
/// restrict the accuracy check to pixels where displacement is meaningful.
fn coherence_proxy(c: &Array4<Cf32>) -> ndarray::Array2<f64> {
    let (rows, cols, n, _) = c.dim();
    ndarray::Array2::from_shape_fn((rows, cols), |(r, cc)| {
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sum += f64::from(c[(r, cc, i, j)].norm());
                }
            }
        }
        sum / (n * (n - 1)) as f64
    })
}

/// Compare two `(nslc, rows, cols)` phase cubes over the pixels passing `keep`.
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
        "  {label:<26} overlap≥{:.4}  median {:.2e} rad ({:.2e} mm)  p99 {:.2e} rad ({:.3} mm)  max {:.2e} rad ({:.3} mm)",
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
    // oracle npy is (rows, cols, nslc); transpose to (nslc, rows, cols).
    let oracle = oracle
        .permuted_axes([2, 0, 1])
        .as_standard_layout()
        .to_owned();

    let cpu_emi = cpu_phase(&process_coherence_matrices(
        to_c64(&c).view(),
        false,
        0.0,
        0.0,
        0,
    ));
    let cpu_evd = cpu_phase(&process_coherence_matrices(
        to_c64(&c).view(),
        true,
        0.0,
        0.0,
        0,
    ));
    let gpu_emi =
        process_coherence_matrices_gpu(ctx, c.view(), false, 0.0, 0.0, 0, DEFAULT_LINK_ITERS)
            .unwrap()
            .cpx_phase;
    let gpu_evd =
        process_coherence_matrices_gpu(ctx, c.view(), true, 0.0, 0.0, 0, DEFAULT_LINK_ITERS)
            .unwrap()
            .cpx_phase;

    // Coherent mask: γ̄ > 0.6 — where displacement is physically meaningful and
    // the least eigenvector (EMI) is well defined.
    let gamma = coherence_proxy(&c);
    let coh: Vec<bool> = gamma.iter().map(|&g| g > 0.6).collect();
    let n_coh = coh.iter().filter(|&&b| b).count();
    let all = |_: usize| true;
    let keep = |p: usize| coh[p];
    println!(
        "ACCURACY — real Mexico stack (13 acqs, 384², half_window 11×5); coherent (γ̄>0.6) = {}/{} px ({:.0}%):",
        n_coh,
        coh.len(),
        100.0 * n_coh as f64 / coh.len() as f64
    );
    report(
        "EVD GPU vs CPU [all px]",
        &phase_delta(&gpu_evd, &cpu_evd, all),
    );
    report(
        "EMI GPU vs CPU [all px]",
        &phase_delta(&gpu_emi, &cpu_emi, all),
    );
    report(
        "EMI GPU vs CPU [coherent]",
        &phase_delta(&gpu_emi, &cpu_emi, keep),
    );
    report(
        "EMI GPU vs oracle [coherent]",
        &phase_delta(&gpu_emi, &oracle, keep),
    );
    report(
        "EMI CPU vs oracle [coherent]",
        &phase_delta(&cpu_emi, &oracle, keep),
    );
}

/// Synthetic KMS coherence stack `(s, s, nslc, nslc)` — PD, exercises EMI.
fn synth_stack(s: usize, nslc: usize) -> Array4<Cf32> {
    let gamma = 0.85_f32;
    Array4::from_shape_fn((s, s, nslc, nslc), |(r, c, i, j)| {
        let theta = 0.1 * (r + c) as f32;
        let mag = gamma.powi((i as i32 - j as i32).abs());
        Cf32::from_polar(mag, theta * (i as f32 - j as f32))
    })
}

fn speed(ctx: &GpuContext) {
    println!("\nSPEED — phase-linking wall-clock, median of {REPS} warm reps (nslc=13):");
    println!(
        "  {:>7}  {:>10}  {:>10}  {:>8}",
        "size", "CPU (s)", "GPU (s)", "speedup"
    );
    let mut crossover = None;
    for &s in &[64_usize, 128, 192, 256, 384, 512] {
        let c = synth_stack(s, 13);
        let cpu_s = time_warm(|| process_coherence_matrices(to_c64(&c).view(), false, 0.0, 0.0, 0));
        let gpu_s = time_warm(|| {
            process_coherence_matrices_gpu(ctx, c.view(), false, 0.0, 0.0, 0, DEFAULT_LINK_ITERS)
                .unwrap()
        });
        let speedup = cpu_s / gpu_s;
        if speedup >= 1.0 && crossover.is_none() {
            crossover = Some(s);
        }
        println!("  {s:>5}²  {cpu_s:>10.4}  {gpu_s:>10.4}  {speedup:>7.2}×");
    }
    match crossover {
        Some(s) => println!("  crossover: GPU wins at ≥ {s}² pixels"),
        None => println!("  crossover: CPU wins at every tested size"),
    }
}

fn main() {
    let ctx = GpuContext::new().expect("GPU device");
    println!("GPU: {}\n", ctx.adapter);
    accuracy(&ctx);
    speed(&ctx);
}
