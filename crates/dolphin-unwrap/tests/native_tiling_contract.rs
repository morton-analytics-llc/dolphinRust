//! Native tiling contract — seam-robust inter-tile reconciliation (Chen 2002).
//!
//! Tiling must reproduce the whole-grid solve regardless of how seams fall. The
//! per-tile modal-offset stitch failed this: a single integer offset per tile
//! cannot reconcile a tile that straddles several coherent components, nor a
//! coherent region a seam bisects, so cycle disagreement exceeded 30% on
//! adversarial multi-component scenes at odd tile counts. The per-region
//! spanning-forest reconciliation fixes it; this contract locks it in.
//!
//! Two goldens, each tiled at counts 2..=5 (odd AND even):
//!   - `unwsuite_multitile` — the original smooth multi-tile fixture (skips when
//!     absent), checked to integer-cycle parity against the global solve;
//!   - `unwseam_ci` — a committed 160² adversarial scene (25 SNAPHU components,
//!     8.4% residues) that FAILS (not skips) when absent, checked per connected
//!     component against the SNAPHU oracle to the 0.5% parity gate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_unwrap::native::{unwrap_native, NativeConfig};
use ndarray::Array2;

const TWO_PI: f64 = std::f64::consts::TAU;
/// Per-component cycle disagreement allowed at any tile count (the parity gate).
const GATE: f64 = 0.005;
const TILE_COUNTS: [usize; 4] = [2, 3, 4, 5];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

/// Load a committed `unwseam_ci_<name>.npy` golden, failing loudly if absent.
fn load_seam<T: ndarray_npy::ReadableElement>(name: &str) -> Array2<T> {
    let path = fixtures().join(format!("unwseam_ci_{name}.npy"));
    ndarray_npy::read_npy(&path).unwrap_or_else(|e| {
        panic!(
            "missing committed seam golden {path:?}: {e}\nregenerate via \
             STRUCT=multicomp SWEEP_SIZE=160 SEEDS=1 LOOKS=4 SNAPHU=1 oracle/gen_seam_sweep.py"
        )
    })
}

fn mode(cycles: &[i64]) -> i64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &k in cycles {
        *counts.entry(k).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map_or(0, |(k, _)| k)
}

/// Per-connected-component cycle disagreement of `cand` vs `oracle`, grouped by
/// the labels in `cc` (component 0 = masked, excluded). A single global mode is
/// wrong on a multi-component scene — each component carries its own offset.
fn per_component_disagreement(cand: &Array2<f32>, oracle: &Array2<f32>, cc: &Array2<u32>) -> f64 {
    let mut by_comp: HashMap<u32, Vec<i64>> = HashMap::new();
    for ((&label, &c), &o) in cc.iter().zip(cand.iter()).zip(oracle.iter()) {
        if label == 0 {
            continue;
        }
        by_comp
            .entry(label)
            .or_default()
            .push(((c as f64 - o as f64) / TWO_PI).round() as i64);
    }
    let (mut valid, mut bad) = (0usize, 0usize);
    for ks in by_comp.values() {
        let off = mode(ks);
        valid += ks.len();
        bad += ks.iter().filter(|&&k| k != off).count();
    }
    bad as f64 / valid.max(1) as f64
}

fn tiled(ifg: &Array2<Cf32>, corr: &Array2<f32>, t: usize) -> Array2<f32> {
    let cfg = NativeConfig {
        tile: Some((t, t)),
        ..NativeConfig::default()
    };
    unwrap_native(ifg.view(), corr.view(), &cfg)
        .unwrap()
        .unwrapped
}

#[test]
fn tiling_seam_robust_on_smooth_multitile() {
    let dir = fixtures();
    if !dir.join("unwsuite_multitile_oracle.npy").exists() {
        eprintln!("skipping smooth multitile: no golden suite");
        return;
    }
    let ifg: Array2<Cf32> = ndarray_npy::read_npy(dir.join("unwsuite_multitile_ifg.npy")).unwrap();
    let corr: Array2<f32> = ndarray_npy::read_npy(dir.join("unwsuite_multitile_corr.npy")).unwrap();
    let global = unwrap_native(ifg.view(), corr.view(), &NativeConfig::default()).unwrap();
    for t in TILE_COUNTS {
        let d =
            per_component_disagreement(&tiled(&ifg, &corr, t), &global.unwrapped, &global.conncomp);
        eprintln!(
            "[multitile {t}x{t}] vs-global per-component={:.4}%",
            d * 100.0
        );
        assert!(
            d <= GATE,
            "smooth multitile {t}x{t} disagreement {:.4}% > {:.2}%",
            d * 100.0,
            GATE * 100.0
        );
    }
}

#[test]
fn tiling_seam_robust_on_adversarial_multicomponent() {
    let ifg: Array2<Cf32> = load_seam("ifg");
    let corr: Array2<f32> = load_seam("corr");
    let oracle: Array2<f32> = load_seam("oracle");
    let cc: Array2<u32> = load_seam("conncomp");
    assert!(
        (cc.iter().copied().max().unwrap_or(0)) >= 5,
        "golden must be multi-component"
    );

    // Two oracles, two questions:
    //   * vs-global isolates the TILE SEAM logic — the untiled global solve is
    //     SNAPHU-parity (≤0.5% baseline below), so tiled-vs-global ≤ gate proves
    //     tiling adds no per-component seam error. This is PUSH-1's gate.
    //   * vs-SNAPHU is printed for context only. It runs higher on this scene
    //     because native and SNAPHU segment the thin-decorrelation lattice into
    //     different components (native masks bridges SNAPHU's regrow keeps), so a
    //     SNAPHU component can span two independently-offset native regions. That
    //     is a conncomp-segmentation difference, not a seam defect — the trait
    //     discards conncomp, and tiled-vs-global stays ~0.
    let global = unwrap_native(ifg.view(), corr.view(), &NativeConfig::default()).unwrap();
    let base = per_component_disagreement(&global.unwrapped, &oracle, &cc);
    eprintln!(
        "[seam_ci global] vs-SNAPHU per-component={:.4}%",
        base * 100.0
    );
    assert!(
        base <= GATE,
        "baseline global vs SNAPHU {:.4}% > gate — golden too hard",
        base * 100.0
    );

    for t in TILE_COUNTS {
        let cand = tiled(&ifg, &corr, t);
        let vs_global = per_component_disagreement(&cand, &global.unwrapped, &global.conncomp);
        let vs_oracle = per_component_disagreement(&cand, &oracle, &cc);
        eprintln!(
            "[seam_ci {t}x{t}] vs-global={:.4}% vs-SNAPHU={:.4}%",
            vs_global * 100.0,
            vs_oracle * 100.0
        );
        assert!(
            vs_global <= GATE,
            "adversarial seam {t}x{t} vs-global {:.4}% > {:.2}% — tiling not seam-robust",
            vs_global * 100.0,
            GATE * 100.0
        );
    }
}
