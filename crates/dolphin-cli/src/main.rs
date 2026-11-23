//! `dolphin` CLI — entry point mirroring the Python `dolphin run` command.
#![warn(missing_docs)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::{
    run_displacement, run_displacement_resumable, run_pairs, update_displacement, PairsConfig,
};

#[derive(Parser)]
#[command(
    name = "dolphin",
    version,
    about = "Optimized Rust rebuild of the dolphin InSAR displacement pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the displacement workflow from a YAML config.
    Run {
        /// Path to the displacement workflow config (YAML).
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Stream (NRT) mode: process an initial window, then fold each remaining
    /// acquisition into the series incrementally via the carried compressed SLC,
    /// re-phase-linking only the new ministack work. Outputs are rewritten after
    /// every fold. Demonstrates the incremental front door in one process.
    Stream {
        /// Path to the displacement workflow config (YAML); its `cslc_file_list`
        /// is the full date-ordered series to stream through.
        #[arg(short, long)]
        config: PathBuf,
        /// Acquisitions to process in the initial batch before streaming the rest.
        #[arg(short, long, default_value_t = 2)]
        initial: usize,
    },
    /// Write dated pairwise products for a planned set of acquisition pairs: pair
    /// coherence (`mode: coherence`), or the NISAR split-spectrum ionosphere-corrected
    /// unwrapped interferogram plus `los_enu.tif` (`mode: interferogram`), and a
    /// `pairs.json` run index.
    Pairs {
        /// Path to the pairs config (YAML).
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Build geocoded CSLC HDF5 files for one Sentinel-1 burst from raw IW SLC
    /// bursts on the Copernicus Data Space (CDSE), on the OPERA burst grid.
    /// Re-running with `--start` at the last processed date picks up new
    /// acquisitions and keeps the existing ones.
    Cslc {
        /// Burst id, e.g. `T100-213486-IW2`.
        #[arg(long)]
        burst_id: String,
        /// Polarisation.
        #[arg(long, default_value = "VV")]
        polarization: String,
        /// First acquisition date (YYYY-MM-DD, inclusive).
        #[arg(long)]
        start: chrono::NaiveDate,
        /// Last acquisition date (YYYY-MM-DD, inclusive).
        #[arg(long)]
        end: chrono::NaiveDate,
        /// OPERA burst database (`opera-burst-bbox-only.sqlite3`, burst_db
        /// v0.10.0): the burst's fixed UTM grid, 5 m x 10 m posting.
        #[arg(long)]
        burst_db: PathBuf,
        /// Process only the window `ROW0,COL0,ROWS,COLS` of the burst grid.
        #[arg(long, value_delimiter = ',')]
        window: Option<Vec<usize>>,
        /// Raw-data cache directory (bursts, orbits).
        #[arg(long)]
        cache: PathBuf,
        /// Output directory for the CSLC HDF5 files.
        #[arg(long)]
        out: PathBuf,
        /// `KEY=VALUE` env file holding `CDSE_USERNAME` / `CDSE_PASSWORD`.
        #[arg(long)]
        env_file: Option<PathBuf>,
        /// Keep downloaded raw bursts in the cache after use.
        #[arg(long)]
        keep_raw: bool,
        /// Also write the azimuth-carrier and flattening phase layers.
        #[arg(long)]
        phase_layers: bool,
        /// Leave the JPL GIM ionospheric delay out of the sampling range
        /// (otherwise it needs EARTHDATA_TOKEN).
        #[arg(long)]
        no_ionosphere: bool,
    },
    /// Build a geocoded NISAR stack from raw NISAR L1 RSLC on ASF (no GSLC),
    /// in the NISAR GSLC layout that `input_type: l1_nisar` reads.
    Nisar {
        /// Track (relative orbit).
        #[arg(long)]
        track: u32,
        /// Pass direction: `A` or `D`.
        #[arg(long)]
        direction: char,
        /// Frame number.
        #[arg(long)]
        frame: u32,
        /// First acquisition date (YYYY-MM-DD, inclusive).
        #[arg(long)]
        start: chrono::NaiveDate,
        /// Last acquisition date (YYYY-MM-DD, inclusive).
        #[arg(long)]
        end: chrono::NaiveDate,
        /// Output CRS: the frame's UTM zone, as JPL's GSLC uses.
        #[arg(long)]
        epsg: u32,
        /// Output bounds `XMIN,YMIN,XMAX,YMAX` in `--epsg` (snapped outward
        /// to the 40 m GSLC lattice).
        #[arg(long, value_delimiter = ',')]
        bounds: Vec<f64>,
        /// Frequency bands.
        #[arg(long, value_delimiter = ',', default_value = "A,B")]
        frequencies: Vec<char>,
        /// Polarisations.
        #[arg(long, value_delimiter = ',', default_value = "HH")]
        polarizations: Vec<String>,
        /// Ancillary cache directory (TEC files).
        #[arg(long)]
        cache: PathBuf,
        /// Output directory.
        #[arg(long)]
        out: PathBuf,
        /// `KEY=VALUE` env file holding the Earthdata token or login.
        #[arg(long)]
        env_file: Option<PathBuf>,
        /// Leave the NISAR TEC ionospheric timing corrections out.
        #[arg(long)]
        no_ionosphere: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Run { config } => run(&config),
        Command::Stream { config, initial } => stream(&config, initial),
        Command::Pairs { config } => pairs(&config),
        Command::Cslc {
            burst_id,
            polarization,
            start,
            end,
            burst_db,
            window,
            cache,
            out,
            env_file,
            keep_raw,
            phase_layers,
            no_ionosphere,
        } => {
            if let Some(path) = env_file {
                dolphin_l1::load_env_file(&path)?;
            }
            let burst_id = dolphin_l1::BurstId::parse(&burst_id)?;
            let mut grid = dolphin_l1::grid::GeoGrid::from_burst_db(&burst_db, &burst_id)?;
            if let Some(window) = window {
                let [row0, col0, rows, cols] = window[..] else {
                    anyhow::bail!("--window takes ROW0,COL0,ROWS,COLS");
                };
                grid = grid.window(row0, col0, rows, cols)?;
            }
            let day = |d: chrono::NaiveDate| d.and_hms_opt(0, 0, 0).map(|t| t.and_utc());
            let request = dolphin_l1::stack::StackRequest {
                burst_id,
                polarization,
                start: day(start).context("start date")?,
                end: day(end + chrono::Duration::days(1)).context("end date")?,
                grid,
                cache_dir: cache,
                output_dir: out,
                keep_raw,
                phase_layers,
                ionosphere: !no_ionosphere,
            };
            cslc(&request)
        }
        Command::Nisar {
            track,
            direction,
            frame,
            start,
            end,
            epsg,
            bounds,
            frequencies,
            polarizations,
            cache,
            out,
            env_file,
            no_ionosphere,
        } => {
            if let Some(path) = env_file {
                dolphin_l1::load_env_file(&path)?;
            }
            let [xmin, ymin, xmax, ymax] = bounds[..] else {
                anyhow::bail!("--bounds takes XMIN,YMIN,XMAX,YMAX");
            };
            let day = |d: chrono::NaiveDate| d.and_hms_opt(0, 0, 0).map(|t| t.and_utc());
            let request = dolphin_l1::nisar_stack::NisarStackRequest {
                frame: dolphin_l1::rslc::NisarFrame {
                    track,
                    direction: direction.to_ascii_uppercase(),
                    frame,
                },
                start: day(start).context("start date")?,
                end: day(end + chrono::Duration::days(1)).context("end date")?,
                epsg,
                bounds: [xmin, ymin, xmax, ymax],
                frequencies: frequencies.iter().map(char::to_ascii_uppercase).collect(),
                polarizations,
                cache_dir: cache,
                output_dir: out,
                ionosphere: !no_ionosphere,
            };
            let client = dolphin_l1::rslc::AsfClient::from_env()?;
            for output in dolphin_l1::nisar_stack::process_nisar_stack(&client, &request)? {
                let state = if output.written { "wrote" } else { "current" };
                println!(
                    "{}  {state}  {}",
                    output.sensing_start,
                    output.path.display()
                );
            }
            Ok(())
        }
    }
}

/// Run the Sentinel-1 raw front end and print one line per date.
fn cslc(request: &dolphin_l1::stack::StackRequest) -> Result<()> {
    let mut client = dolphin_l1::CdseClient::from_env()?;
    let outputs = dolphin_l1::stack::process_stack(&mut client, request)?;
    for output in &outputs {
        let state = if output.written { "wrote" } else { "current" };
        println!(
            "{}  {state}  {}",
            output.sensing_start,
            output.path.display()
        );
    }
    Ok(())
}

/// Parse a displacement config from YAML.
fn parse_config(config: &PathBuf) -> Result<DisplacementWorkflow> {
    let yaml = std::fs::read_to_string(config)
        .with_context(|| format!("reading config {}", config.display()))?;
    DisplacementWorkflow::from_yaml(&yaml).context("parsing displacement config")
}

/// Parse the YAML config and run the displacement workflow.
fn run(config: &PathBuf) -> Result<()> {
    let cfg = parse_config(config)?;
    let out = run_displacement(&cfg)?;
    let (dates, rows, cols) = out.displacement.dim();
    println!(
        "displacement: {dates} dates x {rows}x{cols}; outputs written to {}",
        cfg.work_directory.display()
    );
    Ok(())
}

/// Parse the pairs config and write one product per planned pair.
fn pairs(config: &PathBuf) -> Result<()> {
    let yaml = std::fs::read_to_string(config)
        .with_context(|| format!("reading config {}", config.display()))?;
    let cfg = PairsConfig::from_yaml(&yaml)?;
    let index = run_pairs(&cfg)?;
    println!(
        "{} pairs; outputs written to {}",
        index.pairs.len(),
        cfg.work_directory.display()
    );
    Ok(())
}

/// Stream the series: process the first `initial` acquisitions, then fold each
/// later one in incrementally, rewriting outputs after every fold.
fn stream(config: &PathBuf, initial: usize) -> Result<()> {
    let cfg = parse_config(config)?;
    let all = cfg.cslc_file_list.clone();
    anyhow::ensure!(
        (2..=all.len()).contains(&initial),
        "--initial must be between 2 and the number of acquisitions ({})",
        all.len()
    );

    let mut window = cfg.clone();
    window.cslc_file_list = all[..initial].to_vec();
    let (out, mut state) = run_displacement_resumable(&window)?;
    println!(
        "initial: {initial} acquisitions -> {} displacement bands",
        out.displacement.dim().0
    );

    for k in initial..all.len() {
        window.cslc_file_list = all[..=k].to_vec();
        let (out, next) = update_displacement(&state, &window)?;
        state = next;
        let name = all[k].file_name().and_then(|s| s.to_str()).unwrap_or("?");
        println!(
            "folded acquisition {} ({name}) -> {} displacement bands",
            k + 1,
            out.displacement.dim().0
        );
    }
    println!(
        "streaming complete; outputs written to {}",
        cfg.work_directory.display()
    );
    Ok(())
}
