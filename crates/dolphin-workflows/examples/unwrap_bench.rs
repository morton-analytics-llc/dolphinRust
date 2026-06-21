//! Micro-benchmark for the ifg-network unwrap (Tier-1 #1/#3 perf validation).
//!
//!     EPOCHS=12 ROWS=512 COLS=512 RAYON_NUM_THREADS=1 \
//!         cargo run --release --example unwrap_bench
//!
//! Builds an `EPOCHS`-date smooth-ramp linked-phase stack + single-reference
//! network (EPOCHS-1 ifgs) and times one `unwrap_network`. The backend uses the
//! global rayon pool, so `RAYON_NUM_THREADS=1` is the serial baseline and 2/4/8
//! the parallel sweep. `BACKEND=snaphu` (default, needs `snaphu` on PATH) times
//! the subprocess path; `BACKEND=native` times the in-process clean-room MCF.

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

fn main() -> Result<()> {
    let epochs = env_usize("EPOCHS", 12);
    let rows = env_usize("ROWS", 512);
    let cols = env_usize("COLS", 512);
    let threads = rayon::current_num_threads();

    // Smooth ramp per date; deepest pair gradient < π so every ifg unwraps cleanly.
    let pl = Array3::from_shape_fn((epochs, rows, cols), |(t, r, c)| {
        Cf64::from_polar(1.0, t as f64 * (0.05 * r as f64 + 0.03 * c as f64))
    });
    let pairs: Vec<(usize, usize)> = (1..epochs).map(|j| (0, j)).collect();
    let corr = Array2::<f32>::from_elem((rows, cols), 1.0);
    let scratch = std::env::temp_dir().join(format!("dolphinrust_unwrap_bench_{epochs}_{threads}"));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch)?;

    let name = std::env::var("BACKEND").unwrap_or_else(|_| "snaphu".to_string());
    let backend: Box<dyn UnwrapBackend> = match name.as_str() {
        "native" => Box::new(NativeUnwrapBackend(NativeConfig::default())),
        _ => Box::new(SnaphuBackend(UnwrapConfig::default())),
    };
    let t0 = Instant::now();
    let out = backend.unwrap_network(pl.view(), &pairs, corr.view(), &scratch)?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

    let checksum = out.iter().copied().fold(0.0_f64, |a, b| a + b);
    println!(
        "backend={name} epochs={epochs} ifgs={} grid={rows}x{cols} threads={threads} wall_ms={wall_ms:.1} checksum={checksum:.3}",
        pairs.len()
    );
    Ok(())
}
