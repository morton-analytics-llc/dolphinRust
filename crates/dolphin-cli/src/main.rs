//! `dolphin` CLI — entry point mirroring the Python `dolphin run` command.
#![warn(missing_docs)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::{run_displacement, run_displacement_resumable, update_displacement};

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
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Run { config } => run(&config),
        Command::Stream { config, initial } => stream(&config, initial),
    }
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
