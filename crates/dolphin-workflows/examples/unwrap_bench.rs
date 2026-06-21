//! Benchmark for the ifg-network unwrap: native in-process MCF vs SNAPHU.
//!
//!     EPOCHS=12 ROWS=1024 COLS=1024 DENSE=1 TILE=4 BACKEND=native \
//!         RAYON_NUM_THREADS=8 cargo run --release --example unwrap_bench
//!
//! Builds an `EPOCHS`-date linked-phase stack + single-reference network
//! (`EPOCHS-1` ifgs) and times one `unwrap_network`. `DENSE=1` (the realistic
//! mode) gives every date coherence-driven decorrelation noise so each ifg
//! carries thousands of residues and the native MCF runs its real solver; the
//! default smooth ramp is residue-free (native's fast path) and only exercises
//! Tier-1. `TILE=n` tiles the native solve `n x n`. The backend uses the global
//! rayon pool: `RAYON_NUM_THREADS` is the per-frame thread budget.
//!
//! Wall is printed here; wrap the process in `/usr/bin/time -l` to capture total
//! CPU seconds (self + reaped SNAPHU children) and max RSS for the throughput
//! and CPU-per-frame comparison (see `oracle/bench_unwrap_throughput.sh`).

use std::time::Instant;

use anyhow::Result;
use dolphin_core::Cf64;
use dolphin_unwrap::native::NativeConfig;
use dolphin_unwrap::UnwrapConfig;
use dolphin_workflows::{NativeUnwrapBackend, SnaphuBackend, UnwrapBackend};
use ndarray::{Array2, Array3};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

const TAU: f64 = std::f64::consts::TAU;

/// A deterministic unit-variance pseudo-random value from integer coordinates —
/// a hash-based RNG so the bench needs no rng crate and is reproducible.
fn noise(seed: u64, t: usize, r: usize, c: usize) -> f64 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((t as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add((r as u64).wrapping_mul(0xA0761_D6478_BD642F))
        .wrapping_add((c as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB));
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    // map to (-pi, pi] via a uniform in [0,1)
    ((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * TAU
}

/// Coherence field: high-coherence background with two near-zero moats so the
/// dense scene splits into components (mirrors `gen_unwrap_dense.py`).
fn coherence(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let near_v = (c as f64 - cols as f64 * 0.5).abs() < cols as f64 * 0.02;
        let near_h = (r as f64 - rows as f64 * 0.5).abs() < rows as f64 * 0.02;
        if near_v || near_h {
            0.02
        } else {
            0.7
        }
    })
}

/// Deformation phase at date fraction `f`: a ramp plus a steep subsidence cone.
fn deformation(f: f64, rows: usize, cols: usize, r: usize, c: usize) -> f64 {
    let ramp = 0.05 * c as f64 + 0.03 * r as f64;
    let (cy, cx) = (rows as f64 * 0.55, cols as f64 * 0.45);
    let rr = ((c as f64 - cx).powi(2) + (r as f64 - cy).powi(2)).sqrt();
    let cone = 9.0 * TAU * (-rr / (rows as f64 * 0.16)).exp();
    f * (ramp + cone)
}

/// Build the linked-phase stack. Dense: `pl[0]=1`, `pl[t]=exp(-i(def+noise))`
/// with CRLB noise from the coherence field, so `ifg(0,t)` is residue-dense.
fn build_stack(
    epochs: usize,
    rows: usize,
    cols: usize,
    dense: bool,
    corr: &Array2<f32>,
) -> Array3<Cf64> {
    Array3::from_shape_fn((epochs, rows, cols), |(t, r, c)| {
        if !dense {
            return Cf64::from_polar(1.0, t as f64 * (0.05 * r as f64 + 0.03 * c as f64));
        }
        if t == 0 {
            return Cf64::from_polar(1.0, 0.0);
        }
        let f = t as f64 / (epochs - 1).max(1) as f64;
        let g = corr[(r, c)].clamp(0.05, 0.999) as f64;
        let sigma = ((1.0 - g * g) / (2.0 * g * g)).sqrt() / (6.0_f64).sqrt();
        let phase = deformation(f, rows, cols, r, c) + sigma * noise(1, t, r, c);
        Cf64::from_polar(1.0, -phase)
    })
}

/// Count residues (nonzero discrete curl of wrapped gradients) in `ifg(0, j)` —
/// the realism check for the dense scene.
fn residues(pl: &Array3<Cf64>, (i, j): (usize, usize)) -> usize {
    let (_, rows, cols) = pl.dim();
    let phase = Array2::from_shape_fn((rows, cols), |(r, c)| {
        (pl[(i, r, c)] * pl[(j, r, c)].conj()).arg()
    });
    let wrap = |x: f64| x - TAU * (x / TAU).round();
    (0..rows - 1)
        .flat_map(|r| (0..cols - 1).map(move |c| (r, c)))
        .filter(|&(r, c)| {
            let curl = wrap(phase[(r, c + 1)] - phase[(r, c)])
                + wrap(phase[(r + 1, c + 1)] - phase[(r, c + 1)])
                - wrap(phase[(r + 1, c + 1)] - phase[(r + 1, c)])
                - wrap(phase[(r + 1, c)] - phase[(r, c)]);
            (curl / TAU).round() != 0.0
        })
        .count()
}

fn main() -> Result<()> {
    let epochs = env_usize("EPOCHS", 12);
    let rows = env_usize("ROWS", 512);
    let cols = env_usize("COLS", 512);
    let dense = env_usize("DENSE", 0) == 1;
    let tile = env_usize("TILE", 0);
    let threads = rayon::current_num_threads();

    let corr = if dense {
        coherence(rows, cols)
    } else {
        Array2::from_elem((rows, cols), 1.0)
    };
    let pl = build_stack(epochs, rows, cols, dense, &corr);
    let pairs: Vec<(usize, usize)> = (1..epochs).map(|j| (0, j)).collect();
    if dense && env_usize("RESIDUES", 0) == 1 {
        let n = residues(&pl, *pairs.last().unwrap());
        eprintln!(
            "densest-ifg residues={n} ({:.2}%)",
            100.0 * n as f64 / (rows * cols) as f64
        );
    }
    let scratch = std::env::temp_dir().join(format!(
        "dolphinrust_unwrap_bench_{epochs}_{threads}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch)?;

    let name = std::env::var("BACKEND").unwrap_or_else(|_| "snaphu".to_string());
    let native_cfg = NativeConfig {
        tile: (tile > 0).then_some((tile, tile)),
        ..NativeConfig::default()
    };
    let backend: Box<dyn UnwrapBackend> = match name.as_str() {
        "native" => Box::new(NativeUnwrapBackend(native_cfg)),
        _ => Box::new(SnaphuBackend(UnwrapConfig::default())),
    };
    let t0 = Instant::now();
    let out = backend.unwrap_network(pl.view(), &pairs, corr.view(), &scratch)?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let _ = std::fs::remove_dir_all(&scratch);

    let checksum = out.iter().copied().fold(0.0_f64, |a, b| a + b);
    println!(
        "backend={name} dense={} tile={tile} epochs={epochs} ifgs={} grid={rows}x{cols} threads={threads} wall_ms={wall_ms:.1} checksum={checksum:.3}",
        dense as u8,
        pairs.len()
    );
    Ok(())
}
