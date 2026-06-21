//! Native-backend oracle-parity contract (clean-room unwrapper vs SNAPHU).
//!
//! The native unwrapper must reproduce SNAPHU's **integer-cycle field** — equal
//! up to a single global constant — on every fixture class in the golden suite
//! (`oracle/gen_unwrap_suite.py`): smooth, steep/near-aliasing, fault-step
//! discontinuity, low-coherence/noisy, and a multi-tile-sized grid.
//!
//! Parity is measured as per-pixel cycle disagreement, NOT RMS: for each valid
//! pixel `k = round((native - oracle) / 2pi)`; a correct unwrap makes `k` a
//! single constant across the grid. We report the fraction of valid pixels whose
//! `k` deviates from the dominant offset, and the max fractional (sub-cycle)
//! residual. Valid pixels are those SNAPHU placed in a connected component
//! (`conncomp > 0`) — the low-coherence band SNAPHU masks is judged separately.
//!
//! Skips when fixtures are absent. SNAPHU is the oracle that produced the
//! goldens; this test never invokes it.

use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_unwrap::native::{unwrap_native, NativeConfig};
use ndarray::Array2;

const TWO_PI: f64 = std::f64::consts::TAU;
/// Max fraction of valid pixels allowed to disagree in integer cycles.
const MAX_CYCLE_DISAGREE: f64 = 0.005;
/// Max sub-cycle (fractional) residual in radians on agreeing pixels.
const MAX_FRAC_RESIDUAL: f64 = 0.20;

const CLASSES: &[&str] = &["smooth", "steep", "discont", "lowcoh", "multitile"];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

struct Parity {
    disagree_frac: f64,
    max_frac_residual: f64,
    n_valid: usize,
}

/// Cycle-field parity of `native` vs `oracle` over `valid` pixels.
fn parity(native: &Array2<f32>, oracle: &Array2<f32>, valid: &Array2<bool>) -> Parity {
    let cycles: Vec<i64> = native
        .iter()
        .zip(oracle.iter())
        .zip(valid.iter())
        .filter(|(_, &v)| v)
        .map(|((n, o), _)| ((*n as f64 - *o as f64) / TWO_PI).round() as i64)
        .collect();
    let n_valid = cycles.len();
    let mode = dominant(&cycles);
    let disagree = cycles.iter().filter(|&&k| k != mode).count();
    let max_frac_residual = native
        .iter()
        .zip(oracle.iter())
        .zip(valid.iter())
        .filter(|(_, &v)| v)
        .map(|((n, o), _)| {
            let d = *n as f64 - *o as f64 - mode as f64 * TWO_PI;
            (d - TWO_PI * (d / TWO_PI).round()).abs()
        })
        .fold(0.0, f64::max);
    Parity {
        disagree_frac: disagree as f64 / n_valid.max(1) as f64,
        max_frac_residual,
        n_valid,
    }
}

/// Most common integer offset (the global constant).
fn dominant(cycles: &[i64]) -> i64 {
    let mut counts = std::collections::HashMap::new();
    for &k in cycles {
        *counts.entry(k).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(k, _)| k)
        .unwrap_or(0)
}

#[test]
fn native_matches_snaphu_on_every_class() {
    let dir = fixtures();
    if !dir.join("unwsuite_smooth_oracle.npy").exists() {
        eprintln!(
            "skipping native oracle parity: no golden suite (run oracle/gen_unwrap_suite.py)"
        );
        return;
    }
    let mut failures = Vec::new();
    for &class in CLASSES {
        let ifg: Array2<Cf32> =
            ndarray_npy::read_npy(dir.join(format!("unwsuite_{class}_ifg.npy"))).unwrap();
        let corr: Array2<f32> =
            ndarray_npy::read_npy(dir.join(format!("unwsuite_{class}_corr.npy"))).unwrap();
        let oracle: Array2<f32> =
            ndarray_npy::read_npy(dir.join(format!("unwsuite_{class}_oracle.npy"))).unwrap();
        let cc: Array2<u32> =
            ndarray_npy::read_npy(dir.join(format!("unwsuite_{class}_conncomp.npy"))).unwrap();
        let valid = cc.mapv(|c| c > 0);

        let out = unwrap_native(ifg.view(), corr.view(), &NativeConfig::default())
            .unwrap_or_else(|e| panic!("native unwrap failed on {class}: {e}"));

        let p = parity(&out.unwrapped, &oracle, &valid);
        eprintln!(
            "{class:10} valid={:6} cycle-disagree={:.4}% frac-resid={:.4} rad",
            p.n_valid,
            p.disagree_frac * 100.0,
            p.max_frac_residual,
        );
        if p.disagree_frac > MAX_CYCLE_DISAGREE || p.max_frac_residual > MAX_FRAC_RESIDUAL {
            failures.push(class);
        }
    }
    assert!(
        failures.is_empty(),
        "native unwrap missed SNAPHU parity on: {failures:?}"
    );
}
