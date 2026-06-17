//! `dolphin` CLI — entry point mirroring the Python `dolphin run` command.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dolphin_core::config::DisplacementWorkflow;
use dolphin_workflows::run_displacement;

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
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Run { config } => run(&config),
    }
}

/// Parse the YAML config and run the displacement workflow.
fn run(config: &PathBuf) -> Result<()> {
    let yaml = std::fs::read_to_string(config)
        .with_context(|| format!("reading config {}", config.display()))?;
    let cfg = DisplacementWorkflow::from_yaml(&yaml).context("parsing displacement config")?;
    let out = run_displacement(&cfg)?;
    let (dates, rows, cols) = out.displacement.dim();
    println!(
        "displacement: {dates} dates x {rows}x{cols}; outputs written to {}",
        cfg.work_directory.display()
    );
    Ok(())
}
