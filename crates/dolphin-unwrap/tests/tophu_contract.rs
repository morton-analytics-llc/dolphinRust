//! tophu multi-scale unwrap contract tests (SNAPHU-gated).
//!
//! tophu is heuristic orchestration over SNAPHU, so the contract is comparative,
//! not bit-parity: on a clean analytic ramp the multi-scale path must recover the
//! true phase to within the same envelope as a raw single-pass SNAPHU unwrap.
//! Skips cleanly when the `snaphu` binary is absent.

use dolphin_core::Cf32;
use dolphin_unwrap::{unwrap, unwrap_multiscale, TophuConfig, UnwrapConfig};
use ndarray::{Array2, ArrayView2};

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// Max-abs error after referencing both fields to pixel (0,0) — unwrapping is
/// only determined up to a global additive constant.
fn referenced_maxerr(got: ArrayView2<f32>, truth: ArrayView2<f32>) -> f32 {
    let g0 = got[(0, 0)];
    let t0 = truth[(0, 0)];
    got.iter()
        .zip(truth.iter())
        .map(|(g, t)| ((g - g0) - (t - t0)).abs())
        .fold(0.0_f32, f32::max)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Full multi-scale path recovers an analytic ramp to the raw-SNAPHU envelope.
#[test]
fn tophu_recovers_analytic_ramp_within_snaphu_envelope() {
    if !snaphu_available() {
        eprintln!("skipping tophu ramp: snaphu not on PATH");
        return;
    }
    let (rows, cols) = (60, 60);
    let slope = 0.3_f32; // < π/pixel → unambiguous, but wraps over the scene
    let truth = Array2::from_shape_fn((rows, cols), |(r, c)| slope * (r as f32 + c as f32));
    let ifg = truth.mapv(|p| Cf32::from_polar(1.0, p));
    let corr = Array2::<f32>::from_elem((rows, cols), 1.0);

    let raw = unwrap(
        ifg.view(),
        corr.view(),
        &UnwrapConfig::default(),
        &scratch("dolphinrust_tophu_raw"),
    )
    .unwrap();
    let raw_err = referenced_maxerr(raw.unwrapped.view(), truth.view());

    let cfg = TophuConfig {
        downsample_factor: (3, 3),
        ntiles: (2, 2),
        tile_overlap: (8, 8),
        ..TophuConfig::default()
    };
    let multi = unwrap_multiscale(
        ifg.view(),
        corr.view(),
        &cfg,
        &scratch("dolphinrust_tophu_multi"),
    )
    .unwrap();
    let multi_err = referenced_maxerr(multi.unwrapped.view(), truth.view());

    assert!(
        multi_err <= raw_err + 0.5,
        "tophu ramp err {multi_err} exceeds raw SNAPHU envelope {raw_err} + 0.5"
    );
    assert!(
        multi_err < 1.0,
        "tophu absolute ramp err too large: {multi_err}"
    );
}

/// Coarse pass alone (one full-res tile, no inter-tile merge) round-trips a ramp:
/// downsample → coarse SNAPHU → upsample → single-tile anchor recovers the slope.
#[test]
fn tophu_coarse_pass_round_trips_ramp() {
    if !snaphu_available() {
        eprintln!("skipping tophu coarse: snaphu not on PATH");
        return;
    }
    let (rows, cols) = (48, 48);
    let slope = 0.2_f32;
    let truth = Array2::from_shape_fn((rows, cols), |(r, c)| slope * (r as f32 + c as f32));
    let ifg = truth.mapv(|p| Cf32::from_polar(1.0, p));
    let corr = Array2::<f32>::from_elem((rows, cols), 1.0);

    let cfg = TophuConfig {
        downsample_factor: (4, 4),
        ntiles: (1, 1),
        tile_overlap: (0, 0),
        ..TophuConfig::default()
    };
    let out = unwrap_multiscale(
        ifg.view(),
        corr.view(),
        &cfg,
        &scratch("dolphinrust_tophu_coarse"),
    )
    .unwrap();
    let err = referenced_maxerr(out.unwrapped.view(), truth.view());
    assert!(
        err < 1.0,
        "coarse-pass ramp round-trip err too large: {err}"
    );
}
