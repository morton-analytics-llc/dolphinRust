//! Phase-6 (timeseries / SBAS L2) contract tests.
//!
//! Primary (analytic): a single-reference incidence matrix has the expected
//! ±1 structure; inverting a noise-free network recovers the true displacement;
//! velocity equals the known slope. Secondary (oracle): network pairs,
//! incidence matrix, L2-inverted weighted displacement, and velocity match
//! dolphin v0.35.0. Oracle tests skip without fixtures.

use std::path::{Path, PathBuf};

use dolphin_timeseries::{
    build_network, estimate_velocity, estimate_velocity_with_precisions,
    estimate_velocity_with_uncertainty, estimate_velocity_with_uncertainty_neff,
    get_incidence_matrix, invert_stack, invert_stack_l1, invert_stack_with_uncertainty,
    solve_pixel_with_covariance, L1Config, NetworkConfig,
};
use ndarray::{Array2, Array3};

// ------------------------------- analytic (primary) ---------------------------

#[test]
fn single_reference_incidence_structure() {
    let pairs = build_network(
        4,
        &[0.0, 12.0, 24.0, 36.0],
        &NetworkConfig {
            reference_idx: Some(0),
            ..Default::default()
        },
    );
    assert_eq!(pairs, vec![(0, 1), (0, 2), (0, 3)]);
    let a = get_incidence_matrix(&pairs); // drops date-0 column -> 3 columns
    assert_eq!(a.dim(), (3, 3));
    // Each ifg (0, j): -1 on date 0 (dropped) so only +1 on column j-1.
    assert_eq!(a.row(0).to_vec(), vec![1.0, 0.0, 0.0]);
    assert_eq!(a.row(2).to_vec(), vec![0.0, 0.0, 1.0]);
}

#[test]
fn inversion_recovers_true_displacement() {
    // Bandwidth-2 network, noise-free: invert must recover the true series.
    let pairs = build_network(
        5,
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &NetworkConfig {
            max_bandwidth: Some(2),
            ..Default::default()
        },
    );
    let a = get_incidence_matrix(&pairs);
    let truth = [0.0, 1.5, -0.7, 2.2, 0.4]; // date 0 = 0 reference
    let mut dphi = Array3::zeros((pairs.len(), 1, 1));
    for (k, &(i, j)) in pairs.iter().enumerate() {
        dphi[(k, 0, 0)] = truth[j] - truth[i];
    }
    let phase = invert_stack(a.view(), dphi.view(), None);
    for (d, &t) in truth.iter().enumerate().skip(1) {
        assert!(
            (phase[(d - 1, 0, 0)] - t).abs() < 1e-9,
            "date {d}: {} vs {t}",
            phase[(d - 1, 0, 0)]
        );
    }
}

#[test]
fn velocity_is_slope_per_year() {
    // y = 2*x (days); velocity = slope * 365.25.
    let x = [0.0, 10.0, 20.0, 30.0];
    let series = Array3::from_shape_fn((4, 1, 1), |(t, _, _)| 2.0 * x[t]);
    let vel = estimate_velocity(&x, series.view(), None);
    assert!(
        (vel[(0, 0)] - 2.0 * 365.25).abs() < 1e-6,
        "got {}",
        vel[(0, 0)]
    );
}

#[test]
fn weighted_l2_returns_bounded_posterior_variance() {
    let a = ndarray::array![[1.0], [1.0], [1.0]];
    let dphi = Array3::from_shape_vec((3, 1, 1), vec![1.0, 2.0, 9.0]).unwrap();
    let precision = Array3::from_shape_vec((3, 1, 1), vec![4.0, 4.0, 0.01]).unwrap();
    let weighted = invert_stack_with_uncertainty(a.view(), dphi.view(), precision.view());
    let unweighted = invert_stack(a.view(), dphi.view(), None);
    assert!((weighted.phase[(0, 0, 0)] - 12.09 / 8.01).abs() < 1e-12);
    assert!(weighted.phase[(0, 0, 0)] < unweighted[(0, 0, 0)]);
    assert!(weighted.posterior_variance[(0, 0, 0)].is_finite());
    assert!(weighted.residual_rms[(0, 0)].is_finite());
}

#[test]
fn pixel_covariance_matches_diagonal_normal_equation() {
    let a = ndarray::array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let dphi = Array3::from_shape_vec((3, 1, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let precision = Array3::from_elem((3, 1, 1), 1.0);
    let out = solve_pixel_with_covariance(a.view(), dphi.view(), Some(precision.view()), (0, 0), 3)
        .unwrap();
    assert!((out.parameters[0] - 1.0).abs() < 1e-12);
    assert!((out.parameters[1] - 2.0).abs() < 1e-12);
    assert!((out.covariance[(0, 0)] - 2.0 / 3.0).abs() < 1e-12);
    assert!((out.covariance[(0, 1)] + 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn rank_deficient_weighted_pixel_returns_none() {
    let a = ndarray::array![[1.0, 0.0], [1.0, 0.0]];
    let dphi = Array3::zeros((2, 1, 1));
    assert!(solve_pixel_with_covariance(a.view(), dphi.view(), None, (0, 0), 2).is_none());
}

#[test]
fn zero_precision_excludes_nonfinite_observation() {
    let a = ndarray::array![[1.0], [1.0]];
    let dphi = Array3::from_shape_vec((2, 1, 1), vec![2.0, f64::NAN]).unwrap();
    let precision = Array3::from_shape_vec((2, 1, 1), vec![1.0, 0.0]).unwrap();
    let out = solve_pixel_with_covariance(a.view(), dphi.view(), Some(precision.view()), (0, 0), 2)
        .expect("one valid observation determines one parameter");
    assert_eq!(out.parameters, vec![2.0]);
    assert_eq!(out.residual_rms, 0.0);
}

#[test]
fn velocity_uncertainty_matches_closed_form_line() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let series = Array3::from_shape_vec((4, 1, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let precision = Array3::from_elem((4, 1, 1), 1.0);
    let out = estimate_velocity_with_uncertainty(&x, series.view(), precision.view());
    assert!((out.velocity[(0, 0)] - 365.25).abs() < 1e-9);
    assert!(out.sigma[(0, 0)].is_finite() && out.sigma[(0, 0)] > 0.0);
    assert!(out.residual_rms[(0, 0)] < 1e-12);
}

#[test]
fn date_precision_changes_velocity_fit() {
    let x = [0.0, 1.0, 2.0];
    let series = Array3::from_shape_vec((3, 1, 1), vec![0.0, 1.0, 20.0]).unwrap();
    let precision = Array3::from_shape_vec((3, 1, 1), vec![1.0, 1.0, 1e-6]).unwrap();
    let weighted = estimate_velocity_with_precisions(&x, series.view(), precision.view());
    let unweighted = estimate_velocity(&x, series.view(), None);
    assert!(weighted[(0, 0)] < unweighted[(0, 0)] / 2.0);
}

// ------------------- temporal-correlation (N_eff) uncertainty correction -------

/// Deterministic xorshift64 sequence mapped to ~unit variance — reproducible,
/// no RNG dependency. Not a statistical-quality PRNG; good enough to build an
/// AR(1) fixture with a known target correlation.
fn deterministic_white_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64; // uniform [0, 1)
            (u - 0.5) * 2.0 * 3.0_f64.sqrt() // uniform(-1,1) scaled to unit variance
        })
        .collect()
}

/// AR(1) sequence `e[t] = rho*e[t-1] + sqrt(1-rho^2)*w[t]` from unit-variance white
/// noise `w`; theoretical lag-1 autocorrelation of `e` is `rho`.
fn ar1_series(n: usize, rho: f64, seed: u64) -> Vec<f64> {
    let w = deterministic_white_noise(n, seed);
    let scale = (1.0 - rho * rho).sqrt();
    let mut e = vec![0.0; n];
    e[0] = w[0];
    for t in 1..n {
        e[t] = rho * e[t - 1] + scale * w[t];
    }
    e
}

#[test]
fn temporal_correlation_inflation_matches_known_ar1_factor() {
    const N: usize = 400;
    let x: Vec<f64> = (0..N).map(|t| t as f64 * 6.0).collect();
    let rho_true = 0.6;
    let noise = ar1_series(N, rho_true, 0xC0FFEE);
    let series = Array3::from_shape_fn((N, 1, 1), |(t, _, _)| 3.0 + 0.01 * x[t] + noise[t]);
    let precision = Array3::from_elem((N, 1, 1), 1.0);

    let out = estimate_velocity_with_uncertainty_neff(&x, series.view(), precision.view());
    let baseline = estimate_velocity_with_uncertainty(&x, series.view(), precision.view());

    // Existing fields are untouched (regression-safety): the opt-in path never
    // silently changes today's velocity/sigma.
    assert!((out.velocity[(0, 0)] - baseline.velocity[(0, 0)]).abs() < 1e-9);
    assert!((out.sigma[(0, 0)] - baseline.sigma[(0, 0)]).abs() < 1e-9);

    let closed_form_factor = ((1.0 + rho_true) / (1.0 - rho_true)).sqrt(); // 2.0 at rho=0.6
    let factor = out.inflation_factor[(0, 0)];
    assert!(
        (factor - closed_form_factor).abs() < 0.3,
        "inflation factor {factor} far from closed-form {closed_form_factor} (rho={rho_true})"
    );
    assert!(
        factor > 1.2,
        "expected clear inflation for strong AR(1) correlation, got {factor}"
    );

    // sigma_temporal_corrected is definitionally sigma * inflation_factor.
    let expected = out.sigma[(0, 0)] * factor;
    assert!((out.sigma_temporal_corrected[(0, 0)] - expected).abs() < 1e-9);
    assert!(out.sigma_temporal_corrected[(0, 0)] > out.sigma[(0, 0)]);
}

#[test]
fn temporal_correlation_correction_is_noop_at_zero_correlation() {
    const N: usize = 400;
    let x: Vec<f64> = (0..N).map(|t| t as f64 * 6.0).collect();
    let noise = deterministic_white_noise(N, 0xC0FFEE); // rho == 0 by construction
    let series = Array3::from_shape_fn((N, 1, 1), |(t, _, _)| 3.0 + 0.01 * x[t] + noise[t]);
    let precision = Array3::from_elem((N, 1, 1), 1.0);

    let out = estimate_velocity_with_uncertainty_neff(&x, series.view(), precision.view());
    let factor = out.inflation_factor[(0, 0)];
    assert!(
        (factor - 1.0).abs() < 0.15,
        "expected ~no inflation for uncorrelated residuals, got {factor}"
    );
    assert!(
        (out.sigma_temporal_corrected[(0, 0)] - out.sigma[(0, 0)]).abs() < 0.15 * out.sigma[(0, 0)],
        "corrected sigma should stay close to uncorrected sigma at rho~0"
    );
}

// ------------------------------- oracle (secondary) ---------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

const N_DATES: usize = 6;
const DT: f64 = 12.0;

fn days() -> Vec<f64> {
    (0..N_DATES).map(|i| i as f64 * DT).collect()
}

fn check_net(name: &str, cfg: &NetworkConfig) {
    let path = fixtures().join(format!("net_{name}.npy"));
    if !path.exists() {
        eprintln!("skipping net oracle ({name}): no fixtures");
        return;
    }
    let oracle: Array2<i64> = ndarray_npy::read_npy(&path).unwrap();
    let pairs = build_network(N_DATES, &days(), cfg);
    let want: Vec<(usize, usize)> = (0..oracle.nrows())
        .map(|r| (oracle[(r, 0)] as usize, oracle[(r, 1)] as usize))
        .collect();
    assert_eq!(pairs, want, "network {name}");
}

#[test]
fn networks_match_oracle() {
    check_net(
        "single_ref",
        &NetworkConfig {
            reference_idx: Some(0),
            ..Default::default()
        },
    );
    check_net(
        "bandwidth2",
        &NetworkConfig {
            max_bandwidth: Some(2),
            ..Default::default()
        },
    );
    check_net(
        "temporal30",
        &NetworkConfig {
            max_temporal_baseline: Some(30.0),
            ..Default::default()
        },
    );
    check_net(
        "indexes",
        &NetworkConfig {
            indexes: Some(vec![(0, 1), (0, 3), (2, 5)]),
            ..Default::default()
        },
    );
}

#[test]
fn l2_inversion_and_velocity_match_oracle() {
    let dir = fixtures();
    if !dir.join("ts_phase.npy").exists() {
        eprintln!("skipping l2 oracle: no fixtures");
        return;
    }
    let a: Array2<i64> = ndarray_npy::read_npy(dir.join("ts_incidence.npy")).unwrap();
    let a = a.mapv(|v| v as f64);
    let dphi: Array3<f64> = ndarray_npy::read_npy(dir.join("ts_dphi.npy")).unwrap();
    let weights: Array3<f64> = ndarray_npy::read_npy(dir.join("ts_weights.npy")).unwrap();
    let phase_o: Array3<f64> = ndarray_npy::read_npy(dir.join("ts_phase.npy")).unwrap();
    let vel_o: Array2<f64> = ndarray_npy::read_npy(dir.join("ts_velocity.npy")).unwrap();

    // Incidence matrix from our own network must match the oracle's.
    let pairs = build_network(
        N_DATES,
        &days(),
        &NetworkConfig {
            max_bandwidth: Some(2),
            ..Default::default()
        },
    );
    assert_eq!(get_incidence_matrix(&pairs), a, "incidence matrix");

    let phase = invert_stack(a.view(), dphi.view(), Some(weights.view()));
    let perr = phase
        .iter()
        .zip(phase_o.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    // Normal-equations vs dolphin's SVD lstsq diverge ~1e-6; physical tolerance.
    assert!(perr < 1e-4, "L2 displacement error {perr}");

    let (n, rows, cols) = phase.dim();
    let series = Array3::from_shape_fn((n + 1, rows, cols), |(t, r, c)| match t {
        0 => 0.0,
        _ => phase[(t - 1, r, c)],
    });
    let vel = estimate_velocity(&days(), series.view(), None);
    let verr = vel
        .iter()
        .zip(vel_o.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(verr < 1e-4, "velocity error {verr}");
}

#[test]
fn l1_inversion_matches_oracle() {
    let dir = fixtures();
    if !dir.join("ts_phase_l1.npy").exists() {
        eprintln!("skipping l1 oracle: no fixtures");
        return;
    }
    let a: Array2<i64> = ndarray_npy::read_npy(dir.join("ts_incidence.npy")).unwrap();
    let a = a.mapv(|v| v as f64);
    let dphi: Array3<f64> = ndarray_npy::read_npy(dir.join("ts_dphi.npy")).unwrap();
    let phase_o: Array3<f64> = ndarray_npy::read_npy(dir.join("ts_phase_l1.npy")).unwrap();

    // dolphin's default L1/ADMM on the redundant bandwidth-2 network: identical
    // fixed 20-iteration ADMM (rho=0.4, alpha=1.0). dolphin runs it in jax (float32
    // by default); Rust accumulates in f64, so the floor is the float32-vs-f64
    // difference (~1.5e-6), tighter than the L2 path's SVD-vs-normal-eq 1e-4.
    let phase = invert_stack_l1(a.view(), dphi.view(), L1Config::default());
    let perr = phase
        .iter()
        .zip(phase_o.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(perr < 1e-5, "L1 displacement error {perr}");
}
