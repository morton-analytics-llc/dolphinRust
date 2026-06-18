//! tophu-vs-raw-SNAPHU benchmark on a large, low-coherence scene (honest).
//!
//! `#[ignore]` — this is a measurement, not a pass/fail contract. Run with:
//!   cargo test -p dolphin-unwrap --test tophu_bench -- --ignored --nocapture
//! and transcribe the printed numbers into `bench/UNWRAP.md`. The scene (a
//! subsidence bowl under vegetation-style coherence loss) and its parameters are
//! fixed up front and NOT tuned to favour either method — whatever the numbers
//! say is what gets recorded. Skips when `snaphu` is absent.

use dolphin_core::Cf32;
use dolphin_unwrap::{unwrap, unwrap_multiscale, TophuConfig, UnwrapConfig};
use ndarray::{Array2, ArrayView2};

const ROWS: usize = 512;
const COLS: usize = 512;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// Deterministic splitmix64 → U(0,1), so the scene is identical every run.
fn u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal via Box–Muller.
fn gauss(state: &mut u64) -> f64 {
    let (u1, u2) = (u01(state).max(1e-12), u01(state));
    (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
}

/// The fixed scene: truth phase, its wrapped+noised ifg, and the coherence map.
struct Scene {
    truth: Array2<f64>,
    ifg: Array2<Cf32>,
    corr: Array2<f32>,
}

/// Scene parameters — fixed per named case, NOT swept to favour either method.
struct SceneParams {
    name: &'static str,
    sigma: f64,
    amp: f64,
    /// Add a near-zero-coherence decorrelation ring at the bowl's steep flank
    /// (real fast-subsidence centres decorrelate, isolating the coherent peak).
    decorr_ring: bool,
}

/// Subsidence bowl (Gaussian) + linear ramp under a coherence map with low-γ
/// "vegetation" blobs (and optionally a central decorrelation ring); phase noise
/// scales with the per-pixel CRLB.
fn build_scene(p: &SceneParams) -> Scene {
    let (cr, cc) = (ROWS as f64 * 0.45, COLS as f64 * 0.55);
    let truth = Array2::from_shape_fn((ROWS, COLS), |(r, c)| {
        let (dr, dc) = (r as f64 - cr, c as f64 - cc);
        let bowl = -p.amp * (-(dr * dr + dc * dc) / (2.0 * p.sigma * p.sigma)).exp();
        bowl + 0.04 * r as f64 + 0.03 * c as f64
    });
    let corr = coherence_map(p, cr, cc);
    let looks = 4.0;
    let mut state = 0xD1CE_5EED_u64;
    let ifg = Array2::from_shape_fn((ROWS, COLS), |(r, c)| {
        let g = corr[(r, c)].max(0.05) as f64;
        let sd = ((1.0 - g * g) / (2.0 * looks * g * g)).sqrt();
        let phase = truth[(r, c)] + sd * gauss(&mut state);
        Cf32::from_polar(1.0, phase as f32)
    });
    Scene { truth, ifg, corr }
}

/// Base coherence 0.82 with five low-γ (0.25) vegetation patches; optional
/// near-zero-γ ring at radius ~1.4σ from the bowl centre.
fn coherence_map(p: &SceneParams, cr: f64, cc: f64) -> Array2<f32> {
    let blobs = [
        (120.0, 150.0, 70.0),
        (300.0, 360.0, 90.0),
        (400.0, 120.0, 60.0),
        (200.0, 260.0, 55.0),
        (60.0, 420.0, 50.0),
    ];
    let ring_r = 1.4 * p.sigma;
    Array2::from_shape_fn((ROWS, COLS), |(r, c)| {
        let in_blob = blobs.iter().any(|&(br, bc, rad)| {
            let (dr, dc) = (r as f64 - br, c as f64 - bc);
            (dr * dr + dc * dc).sqrt() < rad
        });
        let (dr, dc) = (r as f64 - cr, c as f64 - cc);
        let on_ring = p.decorr_ring && ((dr * dr + dc * dc).sqrt() - ring_r).abs() < 6.0;
        match in_blob || on_ring {
            true => 0.08,
            false => 0.82,
        }
    })
}

/// Count adjacent valid-pixel pairs whose unwrapped difference exceeds π — a
/// correctly unwrapped continuous field should have essentially none.
fn discontinuities(unw: ArrayView2<f64>) -> usize {
    let pi = std::f64::consts::PI;
    let mut n = 0;
    for r in 0..ROWS {
        for c in 0..COLS {
            if c + 1 < COLS && (unw[(r, c)] - unw[(r, c + 1)]).abs() > pi {
                n += 1;
            }
            if r + 1 < ROWS && (unw[(r, c)] - unw[(r + 1, c)]).abs() > pi {
                n += 1;
            }
        }
    }
    n
}

/// RMS error and gross-cycle-error fraction vs truth, after removing the best
/// global integer-free constant offset (unwrapping is up to an additive const).
fn error_vs_truth(unw: ArrayView2<f64>, truth: ArrayView2<f64>) -> (f64, f64) {
    let n = (ROWS * COLS) as f64;
    let mean_diff: f64 = unw
        .iter()
        .zip(truth.iter())
        .map(|(u, t)| u - t)
        .sum::<f64>()
        / n;
    let mut sse = 0.0;
    let mut gross = 0usize;
    unw.iter().zip(truth.iter()).for_each(|(u, t)| {
        let e = (u - t) - mean_diff;
        sse += e * e;
        if e.abs() > std::f64::consts::PI {
            gross += 1;
        }
    });
    ((sse / n).sqrt(), gross as f64 / n)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
#[ignore = "measurement; run with --ignored --nocapture"]
fn bench_tophu_vs_snaphu_low_coherence() {
    if !snaphu_available() {
        eprintln!("skipping tophu bench: snaphu not on PATH");
        return;
    }
    println!("=== tophu-vs-SNAPHU low-coherence bench ({ROWS}x{COLS}) ===");
    measure(&SceneParams {
        name: "gentle-bowl",
        sigma: 80.0,
        amp: 60.0,
        decorr_ring: false,
    });
    measure(&SceneParams {
        name: "steep-bowl+decorr-ring",
        sigma: 45.0,
        amp: 90.0,
        decorr_ring: true,
    });
}

/// Unwrap `scene` with both backends and print the honest metric line for each.
fn measure(p: &SceneParams) {
    let scene = build_scene(p);
    let low_frac = scene.corr.iter().filter(|&&g| g < 0.5).count() as f64 / (ROWS * COLS) as f64;

    let raw = unwrap(
        scene.ifg.view(),
        scene.corr.view(),
        &UnwrapConfig::default(),
        &scratch("dolphinrust_bench_raw"),
    )
    .unwrap()
    .unwrapped
    .mapv(f64::from);

    let tcfg = TophuConfig {
        downsample_factor: (3, 3),
        ntiles: (4, 4),
        tile_overlap: (32, 32),
        ..TophuConfig::default()
    };
    let multi = unwrap_multiscale(
        scene.ifg.view(),
        scene.corr.view(),
        &tcfg,
        &scratch("dolphinrust_bench_tophu"),
    )
    .unwrap()
    .unwrapped
    .mapv(f64::from);

    let (raw_rms, raw_gross) = error_vs_truth(raw.view(), scene.truth.view());
    let (mlt_rms, mlt_gross) = error_vs_truth(multi.view(), scene.truth.view());
    println!(
        "[{}] sigma={} amp={} ring={}  low-gamma-frac={:.3}",
        p.name, p.sigma, p.amp, p.decorr_ring, low_frac
    );
    println!(
        "  raw SNAPHU : discont={:6}  rms={:.4} rad  gross-cycle-err-frac={:.4}",
        discontinuities(raw.view()),
        raw_rms,
        raw_gross
    );
    println!(
        "  tophu      : discont={:6}  rms={:.4} rad  gross-cycle-err-frac={:.4}",
        discontinuities(multi.view()),
        mlt_rms,
        mlt_gross
    );
}
