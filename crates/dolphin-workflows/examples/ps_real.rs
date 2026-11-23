//! #232 real-data check: PS selection on a real OPERA CSLC stack.
//!
//! ```text
//! cargo run --release -p dolphin-workflows --example ps_real -- <cslc_dir> [max_cols]
//! ```
//!
//! Reports, on the stack in `<cslc_dir>` (every `*.h5`, date order, `/data/VV`):
//! - exact Chan merge over 15-date ministacks vs one-shot `calc_ps`: max |ΔD_A|
//!   and PS flips;
//! - NRT resume (first 20 dates, then the rest) vs a full run with PS on: max
//!   |Δ| on linked phase, temporal coherence, PS, D_A and mean;
//! - how many cells PS fills and their temporal coherence.

use std::path::PathBuf;

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{BlockIndices, Cf64, HalfWindow, Strides};
use dolphin_io::{read_cslc_shape, read_cslc_window};
use dolphin_phaselink::ComputeEngine;
use dolphin_ps::{calc_ps, AmplitudeStats};
use dolphin_workflows::{
    run_sequential_resumable_with_ps, update_sequential_with_ps, PsSelection, SequentialConfig,
};
use ndarray::{s, Array3, Axis};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: ps_real <cslc_dir> [max_cols]"));
    let max_cols: Option<usize> = args.next().map(|v| v.parse()).transpose()?;
    let stack = read_stack(&dir, max_cols)?;
    merge_vs_batch(&stack)?;
    nrt_vs_full(&stack)
}

/// Every `*.h5` CSLC in `dir`, in file-name (date) order, as one stack.
fn read_stack(dir: &std::path::Path, max_cols: Option<usize>) -> anyhow::Result<Array3<Cf64>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "h5"))
        .collect();
    files.sort();
    let (rows, cols) = read_cslc_shape(&files[0], "/data/VV")?;
    let cols = max_cols.map_or(cols, |limit| limit.min(cols));
    let block = BlockIndices {
        row_start: 0,
        row_stop: rows,
        col_start: 0,
        col_stop: cols,
    };
    let mut stack = Array3::<Cf64>::zeros((files.len(), rows, cols));
    for (date, path) in files.iter().enumerate() {
        let window = read_cslc_window(path, "/data/VV", block)?;
        stack
            .index_axis_mut(Axis(0), date)
            .assign(&window.mapv(|z| Cf64::new(f64::from(z.re), f64::from(z.im))));
    }
    println!(
        "stack: {} dates x {rows} x {cols} from {}",
        files.len(),
        dir.display()
    );
    Ok(stack)
}

/// Exact Chan merge over 15-date ministacks vs one-shot `calc_ps`.
fn merge_vs_batch(stack: &Array3<Cf64>) -> anyhow::Result<()> {
    let nslc = stack.dim().0;
    let merged = (0..nslc)
        .step_by(15)
        .map(|start| {
            AmplitudeStats::from_stack(stack.slice(s![start..(start + 15).min(nslc), .., ..]))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .try_fold(None::<AmplitudeStats>, |acc, next| match acc {
            None => Ok::<_, dolphin_ps::PsError>(Some(next)),
            Some(prior) => prior.merge(&next).map(Some),
        })?
        .expect("at least one date")
        .to_ps(0.25);
    let batch = calc_ps(stack.mapv(|z| z.norm()).view(), 0.25, nslc);
    let max_dispersion = merged
        .amp_dispersion
        .iter()
        .zip(batch.amp_dispersion.iter())
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    let flips = merged
        .ps
        .iter()
        .zip(batch.ps.iter())
        .filter(|(a, b)| a != b)
        .count();
    let ps_count = batch.ps.iter().filter(|&&flag| flag == 1).count();
    println!("merge vs batch: max|dD_A| = {max_dispersion:e}, PS flips = {flips}, PS pixels = {ps_count}");

    Ok(())
}

/// NRT resume (first 20 dates, then the rest) vs a full run with PS on, and
/// what PS did on the published grid.
fn nrt_vs_full(stack: &Array3<Cf64>) -> anyhow::Result<()> {
    let nslc = stack.dim().0;
    let cfg = SequentialConfig {
        ministack_size: 15,
        max_num_compressed: 10,
        half_window: HalfWindow::default(),
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::default(),
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: false,
        shp_method: ShpMethod::Rect,
        shp_alpha: 0.001,
    };
    let ps = Some(PsSelection {
        amp_dispersion_threshold: 0.25,
        mask_input_ps: false,
    });
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let split = 20.min(nslc - 1);
    let (full, _) = run_sequential_resumable_with_ps(stack.view(), None, &cfg, ps, &engine)
        .map_err(anyhow::Error::msg)?;
    let (_, state) =
        run_sequential_resumable_with_ps(stack.slice(s![..split, .., ..]), None, &cfg, ps, &engine)
            .map_err(anyhow::Error::msg)?;
    let (inc, _) = update_sequential_with_ps(
        &state,
        stack.slice(s![split.., .., ..]),
        None,
        &cfg,
        ps,
        &engine,
    )
    .map_err(anyhow::Error::msg)?;
    let phase = inc
        .cpx_phase
        .iter()
        .zip(full.cpx_phase.iter())
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() {
                (a - b).norm()
            } else {
                0.0
            }
        })
        .fold(0.0, f64::max);
    let coherence = inc
        .temporal_coherence
        .iter()
        .zip(full.temporal_coherence.iter())
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() {
                (a - b).abs()
            } else {
                0.0
            }
        })
        .fold(0.0, f64::max);
    let bit_identical_phase = inc
        .cpx_phase
        .iter()
        .zip(full.cpx_phase.iter())
        .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits());
    let (inc_ps, full_ps) = (inc.ps.expect("ps"), full.ps.expect("ps"));
    println!(
        "NRT (split at {split}) vs full: max|dphase| = {phase:e}, max|dcoh| = {coherence:e}, phase bit-identical = {bit_identical_phase}, PS equal = {}, D_A equal = {}, mean equal = {}",
        inc_ps.ps == full_ps.ps,
        inc_ps.amp_dispersion == full_ps.amp_dispersion,
        inc_ps.amp_mean == full_ps.amp_mean,
    );

    // 3. What PS did on the published grid.
    let filled: Vec<f64> = full_ps
        .ps
        .indexed_iter()
        .filter(|(_, &flag)| flag == 1)
        .map(|(cell, _)| full.temporal_coherence[cell])
        .collect();
    let unit = filled.iter().filter(|&&c| c == 1.0).count();
    println!(
        "final PS pixels = {}, of which temporal coherence == 1: {unit}",
        filled.len()
    );
    Ok(())
}
