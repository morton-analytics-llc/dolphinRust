//! Pre-unwrap interpolation contracts (`unwrap_options.run_interpolation`).
//!
//! Primary (analytic): hand-built rasters whose interpolated phase is known.
//! Secondary (oracle): pinned dolphin v0.35.0 `dolphin.interpolation.interpolate`
//! on a committed fixture (`oracle/gen_interpolation.py`), for both the
//! continuous-weight call and the unwrap stage's boolean-mask call.

use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_phaselink::{interpolate, INTERPOLATION_ALPHA, INTERPOLATION_NUM_NEIGHBORS};
use ndarray::Array2;

/// dolphin stores complex64 and takes the phase in float32; f32 trig is the only
/// expected difference.
const TOL: f32 = 2e-5;

fn fixture<T: ndarray_npy::ReadableElement>(name: &str) -> Array2<T> {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../oracle/fixtures")
        .join(name);
    ndarray_npy::read_npy(path).unwrap()
}

fn assert_close(got: &Array2<Cf32>, want: &Array2<Cf32>) {
    assert_eq!(got.dim(), want.dim());
    for ((point, g), w) in got.indexed_iter().zip(want.iter()) {
        assert!(
            (g - w).norm() < TOL,
            "pixel {point:?}: got {g}, dolphin {w}"
        );
    }
}

fn interpolate_defaults(ifg: &Array2<Cf32>, weights: &Array2<f32>, cutoff: f32) -> Array2<Cf32> {
    interpolate(
        ifg.view(),
        weights.view(),
        cutoff,
        INTERPOLATION_NUM_NEIGHBORS,
        51,
        INTERPOLATION_ALPHA,
    )
}

// ------------------------------- analytic (primary) ---------------------------

/// A decorrelated pixel inside a uniform-phase field takes that phase and keeps
/// its own amplitude.
#[test]
fn low_weight_pixel_takes_uniform_neighbour_phase() {
    let mut ifg = Array2::from_elem((9, 9), Cf32::from_polar(1.0, 0.8));
    ifg[(4, 4)] = Cf32::from_polar(3.0, -2.5);
    let mut weights = Array2::from_elem((9, 9), 1.0_f32);
    weights[(4, 4)] = 0.0;
    let out = interpolate_defaults(&ifg, &weights, 0.5);
    assert!((out[(4, 4)].norm() - 3.0).abs() < 1e-6);
    assert!((out[(4, 4)].arg() - 0.8).abs() < 1e-6);
    let mut unchanged = out.clone();
    unchanged[(4, 4)] = ifg[(4, 4)];
    assert_eq!(unchanged, ifg);
}

/// With two qualifying neighbours at squared distances 1 and 4, dolphin's weights
/// are `exp(-d2 / (4^alpha / 2))`: the near phase dominates by a known ratio.
#[test]
fn neighbour_weights_follow_gaussian_distance_with_alpha() {
    let mut ifg = Array2::from_elem((1, 5), Cf32::from_polar(1.0, 0.0));
    ifg[(0, 1)] = Cf32::from_polar(1.0, 0.0);
    ifg[(0, 4)] = Cf32::from_polar(1.0, std::f32::consts::FRAC_PI_2);
    let weights = Array2::from_shape_vec((1, 5), vec![0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
    // Centre (0, 2): neighbour (0, 1) at d2 = 1, phase 0; (0, 4) at d2 = 4, phase pi/2.
    let out = interpolate_defaults(&ifg, &weights, 0.5);
    let norm = 4.0_f64.powf(INTERPOLATION_ALPHA) / 2.0;
    let (near, far) = ((-1.0 / norm).exp(), (-4.0 / norm).exp());
    let expected = far.atan2(near) as f32;
    assert!(
        (out[(0, 2)].arg() - expected).abs() < 1e-6,
        "{}",
        out[(0, 2)]
    );
}

/// No qualifying neighbour within `max_radius`: amplitude kept, phase zero.
#[test]
fn isolated_pixel_falls_back_to_zero_phase() {
    let mut ifg = Array2::from_elem((7, 7), Cf32::from_polar(1.0, 1.0));
    ifg[(3, 3)] = Cf32::from_polar(2.0, -1.0);
    let mut weights = Array2::from_elem((7, 7), 0.0_f32);
    weights[(0, 0)] = 1.0;
    let out = interpolate(ifg.view(), weights.view(), 0.5, 20, 3, INTERPOLATION_ALPHA);
    assert!(
        (out[(3, 3)] - Cf32::new(2.0, 0.0)).norm() < 1e-6,
        "{}",
        out[(3, 3)]
    );
}

/// Only the first `num_neighbors` qualifying pixels, centre outward, contribute.
#[test]
fn neighbour_count_caps_the_search() {
    let mut ifg = Array2::from_elem((1, 9), Cf32::from_polar(1.0, -1.0));
    ifg[(0, 5)] = Cf32::from_polar(1.0, 0.4);
    let mut weights = Array2::from_elem((1, 9), 1.0_f32);
    weights[(0, 4)] = 0.0;
    // dolphin's ring order visits (0, +1) before (0, -1), so with a cap of one
    // only (0, 5) contributes.
    let out = interpolate(ifg.view(), weights.view(), 0.5, 1, 51, INTERPOLATION_ALPHA);
    assert!((out[(0, 4)].arg() - 0.4).abs() < 1e-6);
}

// ------------------------------- oracle (secondary) ---------------------------

#[test]
fn continuous_weights_match_dolphin_v035() {
    let ifg = fixture::<Cf32>("interpolation_ifg.npy");
    let weights = fixture::<f32>("interpolation_weights.npy");
    let out = interpolate_defaults(&ifg, &weights, 0.5);
    assert_close(&out, &fixture("interpolation_cont_out.npy"));
}

#[test]
fn unwrap_stage_mask_call_matches_dolphin_v035() {
    let ifg = fixture::<Cf32>("interpolation_ifg.npy");
    let mask = fixture::<f32>("interpolation_weights.npy").mapv(|w| f32::from(u8::from(w >= 0.3)));
    let out = interpolate(
        ifg.view(),
        mask.view(),
        0.3,
        INTERPOLATION_NUM_NEIGHBORS,
        4,
        INTERPOLATION_ALPHA,
    );
    assert_close(&out, &fixture("interpolation_mask_out.npy"));
}
