//! Phase 5 (v1.4.0) unwrap-dispatch trait contract.
//!
//! The unwrap backend is behind the [`UnwrapBackend`] trait; both shipped
//! backends (SNAPHU, tophu) implement it and unwrap an interferogram network from
//! the linked phase + date pairs. This locks the seam a future 3D spatiotemporal
//! solver plugs into. Skips without `snaphu`. (Output-unchanged is covered by the
//! end-to-end oracle contract, which still passes through this dispatch.)

use dolphin_core::Cf64;
use dolphin_unwrap::native::NativeConfig;
use dolphin_unwrap::{TophuConfig, UnwrapConfig};
use dolphin_workflows::{NativeUnwrapBackend, SnaphuBackend, TophuBackend, UnwrapBackend};
use ndarray::{Array2, Array3};

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// A 2-date linked phase with a smooth ramp on date 1 → one unwrappable ifg.
fn ramp_pl(rows: usize, cols: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((2, rows, cols), |(t, r, c)| {
        let phase = if t == 0 {
            0.0
        } else {
            0.20 * r as f64 + 0.15 * c as f64
        };
        Cf64::from_polar(1.0, phase)
    })
}

#[test]
fn both_backends_unwrap_through_the_trait() {
    if !snaphu_available() {
        eprintln!("skipping unwrap-backend trait contract: snaphu not on PATH");
        return;
    }
    let (rows, cols) = (24, 24);
    let pl = ramp_pl(rows, cols);
    let pairs = [(0_usize, 1_usize)];
    let corr = Array2::<f32>::from_elem((rows, cols), 1.0);
    let scratch = std::env::temp_dir().join("dolphinrust_unwrap_trait");
    std::fs::create_dir_all(&scratch).unwrap();

    // Dispatch each backend purely through the trait object.
    let backends: Vec<Box<dyn UnwrapBackend>> = vec![
        Box::new(SnaphuBackend(UnwrapConfig::default())),
        Box::new(TophuBackend(TophuConfig::default())),
    ];
    for backend in &backends {
        let out = backend
            .unwrap_network(pl.view(), &pairs, corr.view(), &scratch)
            .unwrap();
        assert_eq!(
            out.unwrapped.dim(),
            (1, rows, cols),
            "one unwrapped ifg of the grid"
        );
        assert!(
            out.unwrapped.iter().all(|v: &f64| v.is_finite()),
            "finite unwrapped phase"
        );
    }
}

/// The native backend unwraps the network in-process (no SNAPHU) and, on a
/// residue-free ramp, matches the SNAPHU backend to integer-cycle parity (equal
/// up to a global constant). The native path needs no `snaphu`; the parity leg
/// is checked only when the oracle binary is present.
#[test]
fn native_backend_matches_snaphu_network() {
    let (rows, cols) = (24, 24);
    let pl = ramp_pl(rows, cols);
    let pairs = [(0_usize, 1_usize)];
    let corr = Array2::<f32>::from_elem((rows, cols), 1.0);
    let scratch = std::env::temp_dir().join("dolphinrust_unwrap_native");
    std::fs::create_dir_all(&scratch).unwrap();

    let native = NativeUnwrapBackend(NativeConfig::default())
        .unwrap_network(pl.view(), &pairs, corr.view(), &scratch)
        .unwrap();
    assert_eq!(native.unwrapped.dim(), (1, rows, cols));
    assert!(native.unwrapped.iter().all(|v: &f64| v.is_finite()));

    if !snaphu_available() {
        return;
    }
    let snaphu = SnaphuBackend(UnwrapConfig::default())
        .unwrap_network(pl.view(), &pairs, corr.view(), &scratch)
        .unwrap();
    let tau = std::f64::consts::TAU;
    let cycles: Vec<i64> = native
        .unwrapped
        .iter()
        .zip(snaphu.unwrapped.iter())
        .map(|(n, s)| ((n - s) / tau).round() as i64)
        .collect();
    let mode = cycles[0];
    let disagree = cycles.iter().filter(|&&k| k != mode).count();
    assert_eq!(
        disagree, 0,
        "native network must match SNAPHU up to a constant"
    );
}

#[test]
fn native_backend_uses_nonuniform_coherence_for_segmentation() {
    let (rows, cols) = (24, 24);
    let pl = ramp_pl(rows, cols);
    let pairs = [(0_usize, 1_usize)];
    let uniform = Array2::<f32>::from_elem((rows, cols), 1.0);
    let mut split = uniform.clone();
    split.slice_mut(ndarray::s![.., 11..13]).fill(0.0);
    let backend = NativeUnwrapBackend(NativeConfig::default());
    let scratch = std::env::temp_dir().join("dolphinrust_unwrap_coherence");
    let connected = backend
        .unwrap_network(pl.view(), &pairs, uniform.view(), &scratch)
        .unwrap();
    let segmented = backend
        .unwrap_network(pl.view(), &pairs, split.view(), &scratch)
        .unwrap();
    assert_ne!(
        connected.connected_components, segmented.connected_components,
        "a zero-coherence moat must alter component segmentation"
    );
    assert!(segmented
        .connected_components
        .slice(ndarray::s![0, .., 11..13])
        .iter()
        .all(|label| *label == 0));
}
