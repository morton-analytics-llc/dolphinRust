//! Phase-7 (filtering) contract tests.
//!
//! Primary (analytic): the long-wavelength high-pass removes a pure ramp
//! (DC/low frequency) down to ~0. Secondary (oracle): long-wavelength and
//! Goldstein outputs match dolphin v0.35.0 to atol 1e-4. Oracle tests skip
//! without fixtures.

use std::path::{Path, PathBuf};

use dolphin_core::Cf64;
use dolphin_filtering::{filter_long_wavelength, goldstein};
use ndarray::{Array2, ArrayView2};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn max_abs_err_f64(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ------------------------------- analytic (primary) ---------------------------

#[test]
fn high_pass_removes_long_wavelength_ramp() {
    // A pure planar ramp is long-wavelength: the high-pass should leave ~0 in
    // the interior (edges wrap under the FFT, so check the center).
    let (rows, cols) = (64, 64);
    let ramp = Array2::from_shape_fn((rows, cols), |(r, c)| {
        10.0 + 0.01 * c as f64 + 0.005 * r as f64
    });
    let out = filter_long_wavelength(ramp.view(), 5_000.0, 30.0).unwrap();
    let center = out.slice(ndarray::s![24..40, 24..40]);
    let max_center = center.iter().map(|v| v.abs()).fold(0.0, f64::max);
    // Input center ~10.3; high-pass attenuates DC+ramp by >50x. Small residual
    // (~0.09) is Gibbs/edge wrap from the finite FFT.
    assert!(
        max_center < 0.2,
        "ramp not attenuated in interior: {max_center}"
    );
}

#[test]
fn rejects_cutoff_too_large() {
    let small = Array2::<f64>::zeros((16, 16));
    assert!(filter_long_wavelength(small.view(), 1_000_000.0, 30.0).is_err());
}

// ------------------------------- oracle (secondary) ---------------------------

#[test]
fn long_wavelength_matches_oracle() {
    let dir = fixtures();
    if !dir.join("filt_lw_output.npy").exists() {
        eprintln!("skipping long_wavelength oracle: no fixtures");
        return;
    }
    let input: Array2<f64> = ndarray_npy::read_npy(dir.join("filt_lw_input.npy")).unwrap();
    let oracle: Array2<f64> = ndarray_npy::read_npy(dir.join("filt_lw_output.npy")).unwrap();
    let out = filter_long_wavelength(input.view(), 5_000.0, 30.0).unwrap();
    assert!(
        max_abs_err_f64(out.view(), oracle.view()) < 1e-4,
        "long-wavelength differs"
    );
}

/// dolphin PR #680 Goldstein (reflection padding, full trailing-edge coverage,
/// weight-normalised blending, zero-magnitude-only empty mask) on a divisible
/// grid, a grid the patch step does not divide, and a grid with valid
/// zero-phase pixels plus zero and NaN pixels. Fixture:
/// `oracle/gen_goldstein_pr680.py`.
#[test]
fn goldstein_matches_upstream_pr680() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/goldstein_pr680.json");
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let values = |case: &serde_json::Value, key: &str| -> Vec<f64> {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect()
    };
    for case in fixture["cases"].as_array().unwrap() {
        let shape = (
            case["shape"][0].as_u64().unwrap() as usize,
            case["shape"][1].as_u64().unwrap() as usize,
        );
        let complex = |re: Vec<f64>, im: Vec<f64>| {
            Array2::from_shape_vec(
                shape,
                re.into_iter()
                    .zip(im)
                    .map(|(r, i)| Cf64::new(r, i))
                    .collect(),
            )
            .unwrap()
        };
        let input = complex(values(case, "input_re"), values(case, "input_im"));
        let expected = complex(values(case, "expected_re"), values(case, "expected_im"));
        let out = goldstein(
            input.view(),
            case["alpha"].as_f64().unwrap(),
            case["psize"].as_u64().unwrap() as usize,
        );
        // Upstream lets a NaN input spread NaN through every patch that holds
        // it; the port must reproduce exactly that pattern, and match elsewhere.
        for ((point, actual), wanted) in out.indexed_iter().zip(expected.iter()) {
            match wanted.is_nan() {
                true => assert!(actual.is_nan(), "{}: {point:?} should be NaN", case["name"]),
                false => assert!(
                    (actual - wanted).norm() < 1e-4,
                    "{}: {point:?} {actual} != {wanted}",
                    case["name"]
                ),
            }
        }
        if case["name"] == "small_patch_zero_phase_and_empty" {
            let zero = Cf64::new(0.0, 0.0);
            assert!(out
                .slice(ndarray::s![3..9, 4..12])
                .iter()
                .all(|&z| z != zero));
            assert!(out
                .slice(ndarray::s![14..17, 20..25])
                .iter()
                .all(|&z| z == zero));
            assert_eq!(out[(0, 0)], zero);
        }
    }
}
