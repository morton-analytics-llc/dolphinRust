//! #232: persistent-scatterer selection and fill in the sequential pipeline.
//!
//! - a stable pixel (D_A ≈ 0.05) is PS and gets temporal coherence 1; a
//!   speckle-like one (D_A ≈ 0.40) is not;
//! - an NRT update equals a full rerun exactly on phase, temporal coherence,
//!   PS, D_A and mean (PS selection is causal per ministack);
//! - `ps: None` leaves the output unchanged;
//! - covariance capture refuses PS until #233 settles PS cells there.

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_io::CovarianceOperatorGrid;
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{
    run_sequential, run_sequential_resumable, run_sequential_with_covariance_capture,
    update_sequential, PsSelection, SequentialConfig, SequentialCovarianceCaptureRequest,
};
use ndarray::{s, Array3};

const STABLE: (usize, usize) = (3, 3);
const BRIGHT_SPECKLE: (usize, usize) = (6, 6);

/// Uniform in [-1, 1) from a fixed integer hash.
fn noise(seed: usize) -> f64 {
    let x = (seed as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((x >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
}

/// Decorrelating background with two planted scatterers: `STABLE` (amplitude
/// 10 ± 5%·√3, so D_A ≈ 0.05) and `BRIGHT_SPECKLE` (D_A ≈ 0.40), both with
/// their own phase history.
fn stack(nslc: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((nslc, 10, 10), |(t, r, c)| {
        let seed = t * 1009 + r * 101 + c;
        let phase = 3.0 * noise(seed * 7 + 3);
        let (amp, phase) = match (r, c) {
            STABLE => (10.0 * (1.0 + 0.0866 * noise(seed)), 0.4 * t as f64),
            BRIGHT_SPECKLE => (10.0 * (1.0 + 0.69 * noise(seed)), -0.3 * t as f64),
            _ => (1.0 + 0.6 * noise(seed), phase),
        };
        Cf64::from_polar(amp, phase)
    })
}

fn cfg(ps: Option<PsSelection>) -> SequentialConfig {
    SequentialConfig {
        ministack_size: 5,
        max_num_compressed: 10,
        half_window: HalfWindow { y: 1, x: 1 },
        strides: Strides { y: 1, x: 1 },
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: false,
        shp_method: ShpMethod::Rect,
        shp_alpha: 0.001,
        ps,
    }
}

const PS: PsSelection = PsSelection {
    amp_dispersion_threshold: 0.25,
    mask_input_ps: false,
};

fn cpu() -> ComputeEngine {
    ComputeEngine::new(ComputeBackend::Cpu)
}

#[test]
fn a_stable_pixel_is_ps_with_unit_coherence_and_speckle_is_not() {
    for mask_input_ps in [false, true] {
        let selection = PsSelection {
            mask_input_ps,
            ..PS
        };
        let out = run_sequential(stack(20).view(), &cfg(Some(selection)), &cpu()).unwrap();
        let ps = out.ps.as_ref().unwrap();
        assert_eq!(ps.ps[STABLE], 1);
        assert_eq!(ps.ps[BRIGHT_SPECKLE], 0);
        assert!((f64::from(ps.amp_dispersion[STABLE]) - 0.05).abs() < 0.02);
        assert!((f64::from(ps.amp_dispersion[BRIGHT_SPECKLE]) - 0.40).abs() < 0.08);
        assert_eq!(out.temporal_coherence[STABLE], 1.0);
        assert!(out.temporal_coherence[BRIGHT_SPECKLE] < 1.0);
        // The PS cell carries its own phase history, referenced to date 0.
        for t in 0..20 {
            let want = Cf64::from_polar(1.0, 0.4 * t as f64);
            assert!((out.cpx_phase[(t, STABLE.0, STABLE.1)] - want).norm() < 1e-9);
        }
    }
}

#[test]
fn nrt_update_equals_a_full_rerun_with_ps() {
    let data = stack(23);
    let config = cfg(Some(PS));
    let full = run_sequential(data.view(), &config, &cpu()).unwrap();
    // Split inside a ministack so the update reruns an open one.
    let (_, state) =
        run_sequential_resumable(data.slice(s![..12, .., ..]), &config, &cpu()).unwrap();
    let (inc, _) =
        update_sequential(&state, data.slice(s![12.., .., ..]), &config, &cpu()).unwrap();
    assert_eq!(inc.cpx_phase, full.cpx_phase);
    assert_eq!(inc.temporal_coherence, full.temporal_coherence);
    let (a, b) = (inc.ps.unwrap(), full.ps.unwrap());
    assert_eq!(a.ps, b.ps);
    assert_eq!(a.amp_dispersion, b.amp_dispersion);
    assert_eq!(a.amp_mean, b.amp_mean);
}

#[test]
fn an_update_refuses_different_ps_options() {
    let data = stack(12);
    let (_, state) =
        run_sequential_resumable(data.slice(s![..7, .., ..]), &cfg(Some(PS)), &cpu()).unwrap();
    let error = update_sequential(&state, data.slice(s![7.., .., ..]), &cfg(None), &cpu())
        .err()
        .unwrap();
    assert!(error.contains("ps_options"), "{error}");
}

#[test]
fn ps_off_publishes_no_ps_and_ps_changes_only_ps_cells() {
    let data = stack(20);
    let off = run_sequential(data.view(), &cfg(None), &cpu()).unwrap();
    let on = run_sequential(data.view(), &cfg(Some(PS)), &cpu()).unwrap();
    assert!(off.ps.is_none());
    assert_ne!(
        off.temporal_coherence[STABLE],
        on.temporal_coherence[STABLE]
    );
}

#[test]
fn covariance_capture_refuses_ps() {
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 10,
        cols: 10,
        stride_y: 1,
        stride_x: 1,
    };
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "ps-contract".into(),
        source_manifest_digest: [1; 32],
        source_model_version_digest: [2; 32],
        native_grid: grid,
        output_grid: grid,
        owned_output_grid: grid,
        branch_tolerance: 1e-9,
    };
    let result = run_sequential_with_covariance_capture(
        stack(10).view(),
        &cfg(Some(PS)),
        &cpu(),
        &request,
        |_| Ok(()),
    );
    let message = format!("{:?}", result.err().unwrap());
    assert!(message.contains("ps_options"), "{message}");
}
