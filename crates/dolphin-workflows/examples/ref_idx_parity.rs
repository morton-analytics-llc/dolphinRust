//! Issue #192 simulation parity: run the production sequential driver on a
//! stack written by `oracle/ref_idx_parity.py` and write its per-date linked
//! phase back, so the script can score dolphinRust and dolphin on one stack.
//!
//! ```text
//! ref_idx_parity <stack.npy> <phase_out.npy> <newest|oldest>
//! ```
//!
//! `newest` is `CompressedSlcPlan::AlwaysFirst`, `oldest` is
//! `AlwaysFirstLegacy`. Everything else is the experiment's fixed setup: 15
//! dates per ministack, five compressed SLCs carried, half-window 5, the
//! rectangular window, EMI.

use anyhow::{bail, Context, Result};
use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{run_sequential, SequentialConfig};
use ndarray::Array3;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [stack_path, out_path, rule] = args.as_slice() else {
        bail!("usage: ref_idx_parity <stack.npy> <phase_out.npy> <newest|oldest>");
    };
    let compressed_slc_plan = match rule.as_str() {
        "newest" => CompressedSlcPlan::AlwaysFirst,
        "oldest" => CompressedSlcPlan::AlwaysFirstLegacy,
        other => bail!("unknown reference rule {other:?}; expected newest or oldest"),
    };
    let stack: Array3<Cf32> =
        ndarray_npy::read_npy(stack_path).with_context(|| format!("reading {stack_path}"))?;
    let stack = stack.mapv(|z| Cf64::new(f64::from(z.re), f64::from(z.im)));
    let cfg = SequentialConfig {
        ministack_size: 15,
        max_num_compressed: 5,
        half_window: HalfWindow { y: 5, x: 5 },
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan,
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: false,
        shp_method: ShpMethod::Rect,
        shp_alpha: 0.001,
    };
    let out = run_sequential(stack.view(), &cfg, &ComputeEngine::new(ComputeBackend::Cpu))
        .map_err(anyhow::Error::msg)?;
    let phase = out.cpx_phase.mapv(|z| Cf32::new(z.re as f32, z.im as f32));
    ndarray_npy::write_npy(out_path, &phase).with_context(|| format!("writing {out_path}"))
}
