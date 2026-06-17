//! Phase-6 (timeseries / SBAS L2) contract tests.
//!
//! Primary (analytic): a single-reference incidence matrix has the expected
//! ±1 structure; inverting a noise-free network recovers the true displacement;
//! velocity equals the known slope. Secondary (oracle): network pairs,
//! incidence matrix, L2-inverted weighted displacement, and velocity match
//! dolphin v0.35.0. Oracle tests skip without fixtures.

use std::path::{Path, PathBuf};

use dolphin_timeseries::{
    build_network, estimate_velocity, get_incidence_matrix, invert_stack, NetworkConfig,
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
