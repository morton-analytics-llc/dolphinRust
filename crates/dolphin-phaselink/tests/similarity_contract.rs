//! Phase-similarity quality-layer contracts (issue #100).
//!
//! Primary (analytic): a phase field with a planted discontinuity shows a sharp
//! similarity drop across the seam and high agreement inside each smooth half —
//! the behaviour the metric exists to detect, provable without an oracle.
//!
//! Secondary (oracle): the neighbour offsets and both summaries match dolphin
//! **v0.42.0** (`dolphin.similarity`, Wang et al. 2022 eq. 5 median / eq. 6 max),
//! the same forward oracle already used for CRLB and closure phase — v0.35.0
//! predates this layer. Fixtures are committed, so this runs in CI rather than
//! skipping.

use std::path::{Path, PathBuf};

use dolphin_core::Cf64;
use dolphin_phaselink::{circle_offsets, estimate_phase_similarity, PhaseSimilaritySummary};
use ndarray::{Array2, Array3};

const SEARCH_RADIUS: usize = 5;
/// dolphin computes and stores this layer in float32; f64-vs-f32 accumulation
/// and the even-count median average are the only expected differences.
const TOL: f64 = 1e-5;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn load_stack() -> Array3<Cf64> {
    let stack: Array3<num_complex::Complex<f32>> =
        ndarray_npy::read_npy(fixtures().join("similarity_stack.npy")).unwrap();
    stack.mapv(|z| Cf64::new(f64::from(z.re), f64::from(z.im)))
}

/// Largest absolute difference over pixels finite in **both** arrays, plus a
/// check that the NaN patterns agree — an implementation that silently emitted
/// a number where dolphin emits NaN would otherwise pass on the finite subset.
fn max_abs_diff(got: &Array2<f64>, want: &Array2<f32>) -> f64 {
    assert_eq!(got.dim(), want.dim(), "shape mismatch");
    let mut worst = 0.0_f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let w = f64::from(*w);
        assert_eq!(
            g.is_finite(),
            w.is_finite(),
            "NaN pattern differs: got {g}, want {w}"
        );
        if g.is_finite() {
            worst = worst.max((g - w).abs());
        }
    }
    worst
}

// ------------------------------- analytic (primary) ---------------------------

/// Two smooth half-planes separated by a pi phase jump: neighbour agreement is
/// high well inside either half and drops at the seam, where every pixel's
/// neighbourhood straddles the discontinuity.
#[test]
fn similarity_drops_at_planted_discontinuity() {
    let (n_ifg, rows, cols) = (4, 20, 24);
    let seam = cols / 2;
    let stack = Array3::from_shape_fn((n_ifg, rows, cols), |(k, r, c)| {
        let ramp = 0.15 * (k + 1) as f64 * (0.5 * r as f64 + 0.25 * c as f64);
        let step = if c >= seam { std::f64::consts::PI } else { 0.0 };
        Cf64::from_polar(1.0, ramp + step)
    });

    let sim = estimate_phase_similarity(
        stack.view(),
        SEARCH_RADIUS,
        PhaseSimilaritySummary::Median,
        None,
    );

    let interior: f64 = sim
        .slice(ndarray::s![.., 2..seam - SEARCH_RADIUS])
        .mean()
        .unwrap();
    let at_seam: f64 = sim.slice(ndarray::s![.., seam - 1..=seam]).mean().unwrap();
    assert!(
        at_seam < interior - 0.2,
        "seam similarity {at_seam} should fall well below interior {interior}"
    );
}

/// A stack whose pixels all share one phase history is perfectly self-similar:
/// every neighbour comparison is `mean(cos 0) == 1`.
#[test]
fn similarity_is_one_for_a_spatially_constant_field() {
    let stack = Array3::from_shape_fn((3, 12, 12), |(k, _, _)| {
        Cf64::from_polar(1.0, 0.4 * k as f64)
    });
    let sim = estimate_phase_similarity(
        stack.view(),
        SEARCH_RADIUS,
        PhaseSimilaritySummary::Median,
        None,
    );
    let worst = sim.iter().fold(0.0_f64, |m, &v| m.max((v - 1.0).abs()));
    assert!(
        worst < 1e-12,
        "constant field should be similarity 1, off by {worst}"
    );
}

/// An all-zero pixel carries no phase; dolphin marks it invalid and emits NaN
/// rather than folding a meaningless `arg(0) == 0` into its neighbours.
#[test]
fn zero_amplitude_pixels_are_invalid() {
    let mut stack =
        Array3::from_shape_fn((3, 9, 9), |(k, _, _)| Cf64::from_polar(1.0, 0.3 * k as f64));
    for k in 0..3 {
        stack[[k, 4, 4]] = Cf64::new(0.0, 0.0);
    }
    let sim = estimate_phase_similarity(
        stack.view(),
        SEARCH_RADIUS,
        PhaseSimilaritySummary::Median,
        None,
    );
    assert!(sim[[4, 4]].is_nan(), "all-zero pixel should be NaN");
    assert!(sim[[0, 0]].is_finite(), "valid pixels should stay finite");
}

// ------------------------------- oracle (secondary) ---------------------------

/// The midpoint-circle neighbour enumeration decides *which* pixels are
/// compared, so it is pinned to dolphin's set exactly — a drift here would move
/// every similarity value for a reason no numeric tolerance could explain.
#[test]
fn circle_offsets_match_oracle() {
    for radius in [3_usize, 5, 8] {
        let path = fixtures().join(format!("similarity_circle_idxs_r{radius}.npy"));
        let oracle: Array2<i32> = ndarray_npy::read_npy(&path).unwrap();

        let mut want: Vec<(i32, i32)> = oracle.rows().into_iter().map(|r| (r[0], r[1])).collect();
        let mut got = circle_offsets(radius);
        want.sort_unstable();
        got.sort_unstable();

        assert_eq!(
            got,
            want,
            "circle offsets differ at radius {radius}: {} vs oracle {}",
            got.len(),
            want.len()
        );
    }
}

#[test]
fn median_similarity_matches_oracle() {
    let want: Array2<f32> =
        ndarray_npy::read_npy(fixtures().join("similarity_median.npy")).unwrap();
    let got = estimate_phase_similarity(
        load_stack().view(),
        SEARCH_RADIUS,
        PhaseSimilaritySummary::Median,
        None,
    );
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < TOL,
        "median similarity differs from oracle by {worst}"
    );
}

#[test]
fn max_similarity_matches_oracle() {
    let want: Array2<f32> = ndarray_npy::read_npy(fixtures().join("similarity_max.npy")).unwrap();
    let got = estimate_phase_similarity(
        load_stack().view(),
        SEARCH_RADIUS,
        PhaseSimilaritySummary::Max,
        None,
    );
    let worst = max_abs_diff(&got, &want);
    assert!(worst < TOL, "max similarity differs from oracle by {worst}");
}

/// Masked-out pixels are neither scored nor used as neighbours.
#[test]
fn masked_similarity_matches_oracle() {
    let mask: Array2<bool> = ndarray_npy::read_npy(fixtures().join("similarity_mask.npy")).unwrap();
    let want: Array2<f32> =
        ndarray_npy::read_npy(fixtures().join("similarity_median_masked.npy")).unwrap();
    let got = estimate_phase_similarity(
        load_stack().view(),
        SEARCH_RADIUS,
        PhaseSimilaritySummary::Median,
        Some(mask.view()),
    );
    let worst = max_abs_diff(&got, &want);
    assert!(
        worst < TOL,
        "masked median similarity differs from oracle by {worst}"
    );
}
