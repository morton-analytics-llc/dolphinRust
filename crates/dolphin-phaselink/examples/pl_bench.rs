//! Intra-phase-linking microbench: clean, repeatable, in-memory (no I/O, no
//! SNAPHU). Times the separate-stage path (covariance → estimator → temp_coh →
//! CRLB) sub-stage by sub-stage, then the fused [`link_fused`] pass, reporting
//! wall + getrusage CPU·s + max-RSS high-water for each. Answers three questions:
//! (1) what drives PL CPU vs the cube, (2) the fused-vs-staged CPU/RSS delta at a
//! representative shipping tile, and (3) the box-sum covariance win vs the
//! pre-optimization direct kernel (`covariance (direct, pre-box-sum)` vs
//! `covariance`, both at the same window/strides).
//!
//!   ROWS=512 NSLC=16 ITERS=3 cargo run --release --example pl_bench \
//!       -p dolphin-phaselink --no-default-features --features no-gpu

use std::time::Instant;

use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::covariance::estimate_stack_covariance_direct;
use dolphin_phaselink::{
    estimate_crlb, estimate_stack_covariance, estimate_temp_coh, link_fused,
    process_coherence_matrices, FusedParams,
};
use ndarray::Array3;

fn rusage() -> (f64, i64) {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        let cpu = ru.ru_utime.tv_sec as f64
            + ru.ru_utime.tv_usec as f64 / 1e6
            + ru.ru_stime.tv_sec as f64
            + ru.ru_stime.tv_usec as f64 / 1e6;
        (cpu, ru.ru_maxrss)
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    env_usize(key, 0) == 1
}

fn synth(nslc: usize, rows: usize, cols: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((nslc, rows, cols), |(t, r, c)| {
        let phase = 0.3 * t as f64 * ((c as f64 + 1.0) / cols as f64)
            + 0.1 * r as f64
            + 0.05 * ((t * 7 + r * 3 + c) % 11) as f64;
        Cf64::from_polar(1.0, phase)
    })
}

/// Run `f`, returning (wall_s, cpu_s, rss_hwm_MB) for this call.
fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let (c0, _) = rusage();
    let t0 = Instant::now();
    let out = f();
    let wall = t0.elapsed().as_secs_f64();
    let (c1, rss) = rusage();
    println!(
        "  {label:<26} wall={:>7.3}s cpu={:>8.3}s cores={:>5.2} rss_hwm={:>7.1}MB",
        wall,
        c1 - c0,
        (c1 - c0) / wall.max(1e-9),
        rss as f64 / 1.048_576e6
    );
    out
}

fn main() {
    let rows = env_usize("ROWS", 512);
    let cols = rows;
    let nslc = env_usize("NSLC", 16);
    let iters = env_usize("ITERS", 3);
    let half = HalfWindow { y: 5, x: 5 };
    let strides = Strides {
        y: env_usize("STRIDE_Y", 1),
        x: env_usize("STRIDE_X", 1),
    };
    let compute_average_coherence = env_flag("AVERAGE_COH");
    let fused_only = env_flag("FUSED_ONLY");
    let beta = 0.1;
    let zct = 0.0;
    let p = FusedParams {
        use_evd: false,
        beta,
        zero_correlation_threshold: zct,
        reference_idx: 0,
        compute_crlb: true,
        crlb_reference_idx: 0,
        num_looks: (half.y as f64 * half.x as f64).sqrt(),
        compute_closure: false,
        compute_average_coherence,
    };
    let stack = synth(nslc, rows, cols);
    println!(
        "PL microbench: {nslc}×{rows}² tile, half=5 (11×11), strides={}x{}, EMI+CRLB, average_coh={}, {iters} iters\n",
        strides.y,
        strides.x,
        compute_average_coherence
    );

    for it in 0..iters {
        println!("iter {it}:");
        if !fused_only {
            println!(" staged:");
            timed("covariance (direct, pre-box-sum)", || {
                estimate_stack_covariance_direct(stack.view(), half, strides, None).unwrap()
            });
            let c = timed("covariance", || {
                estimate_stack_covariance(stack.view(), half, strides, None).unwrap()
            });
            let est = timed("estimator", || {
                process_coherence_matrices(c.view(), p.use_evd, beta, zct, p.reference_idx)
            });
            let cpx = est.cpx_phase.mapv(|z| Cf64::from_polar(1.0, z.arg()));
            timed("temp_coh", || {
                estimate_temp_coh(cpx.view().permuted_axes([1, 2, 0]), c.view())
            });
            timed("crlb", || {
                estimate_crlb(c.view(), beta, zct, p.crlb_reference_idx, p.num_looks)
            });
        }
        println!(" fused:");
        timed("link_fused (all)", || {
            link_fused(stack.view(), half, strides, None, p).unwrap()
        });
        println!();
    }
}
