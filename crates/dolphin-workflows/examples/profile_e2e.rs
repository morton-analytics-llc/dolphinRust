//! End-to-end profiling harness for the full `DisplacementWorkflow` at burst
//! scale (native-unwrap default). Synthesizes a square CSLC stack, runs
//! `run_displacement`, and captures per-stage wall-clock, exclusive CPU·seconds
//! (getrusage delta → parallel efficiency), and the max-RSS high-water timeline.
//!
//!   ROWS=2048 EPOCHS=12 cargo run --release --example profile_e2e -p dolphin-workflows
//!   /usr/bin/time -l <the same command>      # authoritative whole-run max-RSS
//!
//! The per-stage boundaries are the library's `timed(...)` tracing events; this
//! harness installs a tracing layer that snapshots getrusage on each event. CPU·s
//! per stage is the delta of cumulative (user+sys) CPU between consecutive stage
//! events; RSS is the macOS `ru_maxrss` high-water (bytes) reached by each stage.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_workflows::run_displacement;
use ndarray::Array2;
use num_complex::Complex;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const DT_DAYS: i64 = 12;
const SENTINEL1_WAVELENGTH_M: f64 = 0.055_465_76;

/// Cumulative (user+sys) CPU seconds and `ru_maxrss` (bytes on macOS) right now.
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

/// One captured stage event: the `timed` wall plus the getrusage snapshot taken
/// when the event fired, and (for `pl_breakdown`) the read/compute split.
#[derive(Clone)]
struct Sample {
    stage: String,
    wall_s: f64,
    read_s: f64,
    compute_s: f64,
    cpu_s: f64,
    maxrss_b: i64,
}

#[derive(Default)]
struct Fields {
    stage: Option<String>,
    wall_s: f64,
    read_s: f64,
    compute_s: f64,
}

impl Visit for Fields {
    fn record_f64(&mut self, field: &Field, value: f64) {
        match field.name() {
            "elapsed_s" => self.wall_s = value,
            "read_s" => self.read_s = value,
            "compute_s" => self.compute_s = value,
            _ => {}
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "stage" {
            self.stage = Some(value.to_string());
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "stage" && self.stage.is_none() {
            self.stage = Some(format!("{value:?}").trim_matches('"').to_string());
        }
    }
}

/// Tracing layer that records a getrusage snapshot on every `stage complete`
/// event **and prints the per-stage line immediately**, so a stage's numbers are
/// observable even if a *later* stage hangs (e.g. the x86_64-via-Rosetta SNAPHU
/// unwrap stalls on synthetic full-res data). `prev` carries the previous event's
/// (cpu_s, maxrss_b) so the live line shows the same Δ the final table does.
struct ProfLayer {
    samples: Arc<Mutex<Vec<Sample>>>,
    prev: Arc<Mutex<(f64, i64)>>,
}

impl<S: tracing::Subscriber> Layer<S> for ProfLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut f = Fields::default();
        event.record(&mut f);
        let Some(stage) = f.stage else {
            return;
        };
        let (cpu_s, maxrss_b) = rusage();
        let mb = |b: i64| b as f64 / 1.048_576e6;
        if stage == "pl_breakdown" {
            eprintln!(
                "  [live] pl split: read={:.3}s compute={:.3}s",
                f.read_s, f.compute_s
            );
        } else {
            let (pc, pr) = *self.prev.lock().unwrap();
            let cpu = cpu_s - pc;
            let cores = if f.wall_s > 1e-6 { cpu / f.wall_s } else { 0.0 };
            eprintln!(
                "  [live] {:<14} wall={:>8.3}s cpu={:>8.3}s cores={:>5.2} rss_hwm={:>8.1}MB drss={:>+8.1}MB",
                stage, f.wall_s, cpu, cores, mb(maxrss_b), mb(maxrss_b - pr)
            );
            *self.prev.lock().unwrap() = (cpu_s, maxrss_b);
        }
        self.samples.lock().unwrap().push(Sample {
            stage,
            wall_s: f.wall_s,
            read_s: f.read_s,
            compute_s: f.compute_s,
            cpu_s,
            maxrss_b,
        });
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Smooth range ramp growing in time + light speckle (cycle-free unwrap).
fn synth_slc(t: usize, rows: usize, cols: usize) -> Array2<Complex<f32>> {
    Array2::from_shape_fn((rows, cols), |(_, x)| {
        let phase = 0.3 * t as f64 * (x as f64 / cols as f64);
        let speckle = 0.02 * ((t * 7 + x) % 5) as f64;
        Complex::from_polar(1.0, (phase + speckle) as f32)
    })
}

fn write_stack(dir: &std::path::Path, n: usize, rows: usize, cols: usize) -> Result<Vec<PathBuf>> {
    let base = chrono::NaiveDate::from_ymd_opt(2022, 11, 19).unwrap();
    let mut files = Vec::new();
    for t in 0..n {
        let stamp = (base + chrono::Duration::days(t as i64 * DT_DAYS)).format("%Y%m%d");
        let path = dir.join(format!("cslc_{stamp}.h5"));
        if !path.exists() {
            let file = hdf5::File::create(&path)?;
            let group = file.create_group("data")?;
            group
                .new_dataset_builder()
                .with_data(&synth_slc(t, rows, cols))
                .create("VV")?;
        }
        files.push(path);
    }
    Ok(files)
}

fn build_config(dir: &std::path::Path, files: Vec<PathBuf>) -> DisplacementWorkflow {
    let mut cfg = DisplacementWorkflow {
        cslc_file_list: files,
        work_directory: dir.to_path_buf(),
        ..Default::default()
    };
    cfg.input_options.subdataset = Some("/data/VV".into());
    cfg.input_options.wavelength = Some(SENTINEL1_WAVELENGTH_M);
    cfg.phase_linking.ministack_size = 15;
    cfg.phase_linking.half_window = HalfWindow { y: 5, x: 5 };
    cfg.output_options.strides = Strides { y: 1, x: 1 };
    cfg.interferogram_network.reference_idx = Some(0);
    cfg
}

/// Print the per-stage table + memory high-water timeline from the samples.
fn report(samples: &[Sample], cpu0: f64, rss0: i64, total_wall: f64) {
    let peak = samples
        .iter()
        .map(|s| s.maxrss_b)
        .max()
        .unwrap_or(rss0)
        .max(rss0);
    let mb = |b: i64| b as f64 / 1.048_576e6;
    println!(
        "\n{:<14} {:>9} {:>9} {:>7} {:>11} {:>9} {:>7}",
        "stage", "wall_s", "cpu_s", "cores", "rss_hwm_MB", "drss_MB", "%peak"
    );
    let (mut prev_cpu, mut prev_rss) = (cpu0, rss0);
    let mut sum_wall = 0.0;
    for s in samples.iter().filter(|s| s.stage != "pl_breakdown") {
        let cpu = s.cpu_s - prev_cpu;
        let drss = s.maxrss_b - prev_rss;
        let cores = if s.wall_s > 1e-6 { cpu / s.wall_s } else { 0.0 };
        sum_wall += s.wall_s;
        println!(
            "{:<14} {:>9.3} {:>9.3} {:>7.2} {:>11.1} {:>9.1} {:>6.1}%",
            s.stage,
            s.wall_s,
            cpu,
            cores,
            mb(s.maxrss_b),
            mb(drss),
            100.0 * s.maxrss_b as f64 / peak as f64
        );
        prev_cpu = s.cpu_s;
        prev_rss = s.maxrss_b;
    }
    if let Some(b) = samples.iter().find(|s| s.stage == "pl_breakdown") {
        println!(
            "  └─ phase_linking split (wall): read={:.3}s  covariance+estimator={:.3}s",
            b.read_s, b.compute_s
        );
    }
    let (cpu_total, _) = rusage();
    println!(
        "\ntotal: wall={:.2}s  measured-stage-wall={:.2}s  process-cpu={:.1}s  peak-rss(getrusage)={:.0} MB",
        total_wall, sum_wall, cpu_total - cpu0, mb(peak)
    );
}

fn main() -> Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _dhat = dhat::Profiler::new_heap();
    let samples = Arc::new(Mutex::new(Vec::<Sample>::new()));
    tracing_subscriber::registry()
        .with(ProfLayer {
            samples: samples.clone(),
            prev: Arc::new(Mutex::new(rusage())),
        })
        .init();

    let rows = env_usize("ROWS", 2048);
    let epochs = env_usize("EPOCHS", 12);
    let dir = std::env::temp_dir().join(format!("dolphin_profile_{rows}x{rows}_n{epochs}"));
    std::fs::create_dir_all(&dir)?;
    eprintln!(
        "synthesizing {epochs}×{rows}² CSLC stack in {}",
        dir.display()
    );
    let files = write_stack(&dir, epochs, rows, rows)?;
    let cfg = build_config(&dir, files);

    let (cpu0, rss0) = rusage();
    let t0 = std::time::Instant::now();
    let out = run_displacement(&cfg)?;
    let total_wall = t0.elapsed().as_secs_f64();

    let (nd, r, c) = out.displacement.dim();
    eprintln!(
        "ran {epochs}×{rows}²: displacement {nd}d × {r}×{c}, temp_coh mean={:.4}",
        out.temporal_coherence.mean().unwrap_or(0.0)
    );
    report(&samples.lock().unwrap(), cpu0, rss0, total_wall);
    Ok(())
}
