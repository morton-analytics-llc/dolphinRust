//! Phase-6 (timeseries / SBAS L2) contract tests.
//!
//! Primary (analytic): a single-reference incidence matrix has the expected
//! ±1 structure; inverting a noise-free network recovers the true displacement;
//! velocity equals the known slope. Secondary (oracle): network pairs,
//! incidence matrix, L2-inverted weighted displacement, and velocity match
//! dolphin v0.35.0. Oracle tests skip without fixtures.

use std::path::{Path, PathBuf};

use dolphin_timeseries::{
    build_network, estimate_velocity, estimate_velocity_with_diagnostics,
    estimate_velocity_with_precisions, estimate_velocity_with_uncertainty, get_incidence_matrix,
    invert_stack, invert_stack_l1, invert_stack_with_uncertainty, solve_pixel_with_covariance,
    L1Config, NetworkConfig, VelocityCadenceStatus, VelocityUncertaintyStatus,
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
fn weighted_l2_returns_finite_independent_ifg_covariance_diagonal() {
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
fn primary_docs_reject_calibrated_posterior_and_global_crlb_claims() {
    let primary_docs = [
        ("README.md", include_str!("../../../README.md")),
        ("inversion.rs", include_str!("../src/inversion.rs")),
        ("loop_closure.rs", include_str!("../src/loop_closure.rs")),
        (
            "crates/dolphin-timeseries/CLAUDE.md",
            include_str!("../CLAUDE.md"),
        ),
    ];
    let rejected = [
        "per-pixel, per-date physical uncertainty",
        "Full posterior parameter covariance",
        "Diagonal posterior parameter variance",
        "posterior uncertainty carries empirical scale",
        "empirically scaled posterior",
    ];
    for (name, contents) in primary_docs {
        for claim in rejected {
            assert!(
                !contents.contains(claim),
                "{name} reintroduced rejected uncertainty claim {claim:?}"
            );
        }
    }
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
fn velocity_iid_conditional_se_matches_closed_form() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let series = Array3::from_shape_vec((4, 1, 1), vec![0.0, 1.0, 2.0, 4.0]).unwrap();
    let precision = Array3::from_elem((4, 1, 1), 1.0);
    let out = estimate_velocity_with_uncertainty(&x, series.view(), precision.view());
    let expected_sigma = (0.3_f64 / 2.0 / 5.0).sqrt() * 365.25;
    assert!((out.velocity[(0, 0)] - 1.3 * 365.25).abs() < 1e-9);
    assert!((out.sigma[(0, 0)] - expected_sigma).abs() < 1e-9);
    assert!((out.residual_rms[(0, 0)] - (0.3_f64 / 4.0).sqrt()).abs() < 1e-12);
    assert_eq!(out.valid_date_count[(0, 0)], 4);
    assert_eq!(out.rank[(0, 0)], 2);
    assert_eq!(out.regression_dof[(0, 0)], 2);
    assert_eq!(
        out.uncertainty_status[(0, 0)],
        VelocityUncertaintyStatus::IidConditional
    );
}

#[test]
fn iid_conditional_se_is_invariant_to_common_precision_scale() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let series = Array3::from_shape_vec((5, 1, 1), vec![0.2, 0.9, 2.3, 2.8, 4.5]).unwrap();
    let relative = Array3::from_shape_vec((5, 1, 1), vec![1.0, 2.0, 0.5, 3.0, 1.5]).unwrap();
    let scaled = relative.mapv(|weight| weight * 1_000.0);
    let first = estimate_velocity_with_uncertainty(&x, series.view(), relative.view());
    let second = estimate_velocity_with_uncertainty(&x, series.view(), scaled.view());
    assert!((first.velocity[(0, 0)] - second.velocity[(0, 0)]).abs() < 1e-10);
    assert!((first.sigma[(0, 0)] - second.sigma[(0, 0)]).abs() < 1e-10);
}

#[test]
fn iid_conditional_status_requires_full_rank_and_positive_dof() {
    let two_dates = [0.0, 1.0];
    let two_values = Array3::from_shape_vec((2, 1, 1), vec![1.0, 3.0]).unwrap();
    let two_precisions = Array3::from_elem((2, 1, 1), 1.0);
    let zero_dof =
        estimate_velocity_with_uncertainty(&two_dates, two_values.view(), two_precisions.view());
    assert_eq!(zero_dof.rank[(0, 0)], 2);
    assert_eq!(zero_dof.regression_dof[(0, 0)], 0);
    assert_eq!(
        zero_dof.uncertainty_status[(0, 0)],
        VelocityUncertaintyStatus::Unavailable
    );
    assert!(zero_dof.sigma[(0, 0)].is_nan());
    assert!((zero_dof.velocity[(0, 0)] - 2.0 * 365.25).abs() < 1e-9);

    let repeated_dates = [2.0, 2.0, 2.0];
    let repeated_values = Array3::from_shape_vec((3, 1, 1), vec![1.0, 2.0, 4.0]).unwrap();
    let repeated_precisions = Array3::from_elem((3, 1, 1), 1.0);
    let deficient = estimate_velocity_with_uncertainty(
        &repeated_dates,
        repeated_values.view(),
        repeated_precisions.view(),
    );
    assert_eq!(deficient.rank[(0, 0)], 1);
    assert_eq!(deficient.regression_dof[(0, 0)], 2);
    assert_eq!(
        deficient.uncertainty_status[(0, 0)],
        VelocityUncertaintyStatus::Unavailable
    );
    assert!(deficient.velocity[(0, 0)].is_nan());
    assert!(deficient.sigma[(0, 0)].is_nan());
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

// ---------------- temporal-correlation diagnostics (non-inferential) ----------

/// Deterministic xorshift64 sequence mapped to ~unit variance — reproducible,
/// no RNG dependency. Not a statistical-quality PRNG; sufficient for a stable
/// noisy-fit fixture.
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

#[test]
fn exact_linear_fit_reports_point_estimate_but_no_iid_se() {
    let x: Vec<f64> = (0..6).map(|t| t as f64).collect();
    let series = Array3::from_shape_fn((6, 1, 1), |(t, _, _)| 3.0 + 2.0 * x[t]);
    let precision = Array3::from_elem((6, 1, 1), 1.0);
    let out = estimate_velocity_with_diagnostics(&x, series.view(), precision.view());
    assert!((out.velocity[(0, 0)] - 2.0 * 365.25).abs() < 1e-9);
    assert!(out.sigma[(0, 0)].is_nan());
    assert_eq!(out.valid_date_count[(0, 0)], 6);
    assert_eq!(out.rank[(0, 0)], 2);
    assert_eq!(out.regression_dof[(0, 0)], 4);
    assert_eq!(
        out.uncertainty_status[(0, 0)],
        VelocityUncertaintyStatus::Unavailable
    );
    assert!(!out.correlation_available[(0, 0)]);
    assert!(out.lag1_rho[(0, 0)].is_nan());
    assert!(out.diagnostic_inflation_factor[(0, 0)].is_nan());
    assert!(out.diagnostic_effective_sample_size[(0, 0)].is_nan());
}

#[test]
fn irregular_and_missing_cadence_disable_correlation_diagnostics() {
    let irregular_x = [0.0, 6.0, 13.0, 19.0, 25.0];
    let series = Array3::from_shape_vec((5, 1, 1), vec![0.0, 1.0, 0.2, 1.5, 0.7]).unwrap();
    let precision = Array3::from_elem((5, 1, 1), 1.0);
    let irregular =
        estimate_velocity_with_diagnostics(&irregular_x, series.view(), precision.view());
    assert_eq!(
        irregular.cadence_status[(0, 0)],
        VelocityCadenceStatus::Irregular
    );
    assert!(!irregular.correlation_available[(0, 0)]);
    assert_eq!(irregular.correlation_pair_count[(0, 0)], 0);

    let regular_x = [0.0, 6.0, 12.0, 18.0, 24.0];
    let mut missing_precision = precision;
    missing_precision[(2, 0, 0)] = 0.0;
    let missing =
        estimate_velocity_with_diagnostics(&regular_x, series.view(), missing_precision.view());
    assert_eq!(
        missing.cadence_status[(0, 0)],
        VelocityCadenceStatus::Missing
    );
    assert_eq!(missing.valid_date_count[(0, 0)], 4);
    assert!(!missing.correlation_available[(0, 0)]);
    assert_eq!(missing.correlation_pair_count[(0, 0)], 0);
}

#[test]
fn fewer_than_four_dates_disable_correlation_diagnostics() {
    let x = [0.0, 6.0, 12.0];
    let series = Array3::from_shape_vec((3, 1, 1), vec![0.0, 1.0, 0.2]).unwrap();
    let precision = Array3::from_elem((3, 1, 1), 1.0);
    let out = estimate_velocity_with_diagnostics(&x, series.view(), precision.view());
    assert_eq!(
        out.uncertainty_status[(0, 0)],
        VelocityUncertaintyStatus::IidConditional
    );
    assert_eq!(
        out.cadence_status[(0, 0)],
        VelocityCadenceStatus::RegularContiguous
    );
    assert!(!out.correlation_available[(0, 0)]);
    assert_eq!(out.correlation_pair_count[(0, 0)], 0);
    assert!(out.lag1_rho[(0, 0)].is_nan());
}

#[test]
fn negative_raw_rho_is_retained_without_diagnostic_deflation() {
    let x: Vec<f64> = (0..8).map(|t| t as f64).collect();
    let series = Array3::from_shape_fn((8, 1, 1), |(t, _, _)| {
        0.1 * x[t] + if t % 2 == 0 { 1.0 } else { -1.0 }
    });
    let precision = Array3::from_elem((8, 1, 1), 1.0);
    let out = estimate_velocity_with_diagnostics(&x, series.view(), precision.view());
    assert_eq!(
        out.cadence_status[(0, 0)],
        VelocityCadenceStatus::RegularContiguous
    );
    assert!(out.correlation_available[(0, 0)]);
    assert_eq!(out.correlation_pair_count[(0, 0)], 7);
    assert!(out.lag1_rho[(0, 0)] < 0.0);
    assert_eq!(out.diagnostic_inflation_factor[(0, 0)], 1.0);
    assert_eq!(out.diagnostic_effective_sample_size[(0, 0)], 8.0);
}

#[test]
fn diagnostic_effective_sample_size_is_clamped_to_one_and_n() {
    const N: usize = 120;
    let x: Vec<f64> = (0..N).map(|t| t as f64).collect();
    let series = Array3::from_shape_fn((N, 1, 1), |(t, _, _)| {
        (3.0 * std::f64::consts::TAU * t as f64 / N as f64).sin()
    });
    let precision = Array3::from_elem((N, 1, 1), 1.0);
    let out = estimate_velocity_with_diagnostics(&x, series.view(), precision.view());
    assert!(
        out.lag1_rho[(0, 0)] > 0.98,
        "fixture rho={}",
        out.lag1_rho[(0, 0)]
    );
    assert_eq!(out.diagnostic_effective_sample_size[(0, 0)], 1.0);
    assert_eq!(out.diagnostic_inflation_factor[(0, 0)], (N as f64).sqrt());
}

#[test]
fn diagnostic_and_plain_weighted_fits_preserve_the_same_point_estimate() {
    const N: usize = 12;
    let x: Vec<f64> = (0..N).map(|t| t as f64 * 6.0).collect();
    let noise = deterministic_white_noise(N, 0xC0FFEE);
    let series = Array3::from_shape_fn((N, 1, 1), |(t, _, _)| 3.0 + 0.01 * x[t] + noise[t]);
    let precision = Array3::from_shape_fn((N, 1, 1), |(t, _, _)| 0.5 + t as f64 / N as f64);
    let direct = estimate_velocity_with_precisions(&x, series.view(), precision.view());
    let iid = estimate_velocity_with_uncertainty(&x, series.view(), precision.view());
    let diagnostics = estimate_velocity_with_diagnostics(&x, series.view(), precision.view());
    assert_eq!(iid.velocity, direct);
    assert_eq!(diagnostics.velocity, direct);
    assert_eq!(diagnostics.sigma, iid.sigma);
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
