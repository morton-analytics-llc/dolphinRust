//! `dolphin` CLI — entry point mirroring the Python `dolphin run` command.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

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
        Command::Run { config } => {
            let _yaml = std::fs::read_to_string(&config)
                .with_context(|| format!("reading config {}", config.display()))?;
            // Pipeline stages are not yet wired; see PLAYBOOK.md for the port plan.
            bail!("displacement pipeline is not yet implemented (scaffold only) — see PLAYBOOK.md");
        }
    }
}
