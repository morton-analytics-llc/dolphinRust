//! Phase-2 (SHP) contract tests.
//!
//! Primary (analytic): a two-region amplitude field — same-region neighbors are
//! SHPs, cross-region neighbors are not — for both GLRT and KS, plus the KS
//! ECDF-distance docstring cases. Secondary (oracle): GLRT/KS masks must match
//! dolphin v0.35.0 exactly, and SHP-weighted covariance must match dolphin's
//! within tolerance. Oracle tests skip when fixtures are absent.

use std::path::{Path, PathBuf};

use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_shp::{estimate_neighbors_glrt, estimate_neighbors_ks};
use ndarray::{Array2, Array3, Array4, Axis};

const HALF1: HalfWindow = HalfWindow { y: 1, x: 1 };
const S1: Strides = Strides { y: 1, x: 1 };

// ------------------------------- analytic (primary) ---------------------------

#[test]
fn glrt_separates_two_regions() {
    // mean is 1.0 everywhere except column 3 (scale very different).
    let mean = Array2::from_shape_fn((5, 5), |(_, c)| if c == 3 { 10.0 } else { 1.0 });
    let var = Array2::from_elem((5, 5), 0.01);
    let nbr = estimate_neighbors_glrt(mean.view(), var.view(), HALF1, 10, S1, 0.001);

    // Center pixel (2,2): window cols {1,2,3} -> offsets {0,1,2}.
    let slab = nbr.slice(ndarray::s![2, 2, .., ..]);
    assert!(!slab[(1, 1)], "center is never its own neighbor");
    assert!(slab[(0, 0)], "same-scale neighbor (col 1) is an SHP");
    assert!(
        !slab[(0, 2)],
        "different-scale neighbor (col 3) is not an SHP"
    );
}

#[test]
fn ks_separates_two_regions() {
    // Column 3 series is shifted into a disjoint range; the rest share a series.
    let nslc = 20;
    let amp = Array3::from_shape_fn((nslc, 5, 5), |(t, _, c)| {
        t as f64 + if c == 3 { 100.0 } else { 0.0 }
    });
    let nbr = estimate_neighbors_ks(amp.view(), HALF1, S1, 0.001, false);

    let slab = nbr.slice(ndarray::s![2, 2, .., ..]);
    assert!(!slab[(1, 1)], "center is never its own neighbor");
    assert!(
        slab[(0, 0)],
        "identical-distribution neighbor (col 1) is an SHP"
    );
    assert!(
        !slab[(0, 2)],
        "disjoint-distribution neighbor (col 3) is not an SHP"
    );
}

// ------------------------------- oracle (secondary) ---------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn amp_stack() -> Array3<f64> {
    let stack: Array3<Cf32> = ndarray_npy::read_npy(fixtures().join("slc_stack.npy")).unwrap();
    stack.mapv(|z| z.norm() as f64)
}

fn count_mismatch(a: &Array4<bool>, b: &Array4<bool>) -> usize {
    assert_eq!(a.dim(), b.dim(), "neighbor-mask shapes differ");
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
fn glrt_matches_oracle() {
    let dir = fixtures();
    if !dir.join("glrt_neighbors.npy").exists() {
        eprintln!("skipping glrt_matches_oracle: no fixtures");
        return;
    }
    let amp = amp_stack();
    let nslc = amp.len_of(Axis(0));
    let mean = amp.mean_axis(Axis(0)).unwrap();
    let var = amp.var_axis(Axis(0), 0.0);
    let rust = estimate_neighbors_glrt(
        mean.view(),
        var.view(),
        HalfWindow { y: 2, x: 2 },
        nslc,
        S1,
        0.001,
    );
    let oracle: Array4<bool> = ndarray_npy::read_npy(dir.join("glrt_neighbors.npy")).unwrap();
    assert_eq!(
        count_mismatch(&rust, &oracle),
        0,
        "GLRT mask differs from oracle"
    );
}

#[test]
fn ks_matches_oracle() {
    let dir = fixtures();
    if !dir.join("ks_neighbors.npy").exists() {
        eprintln!("skipping ks_matches_oracle: no fixtures");
        return;
    }
    let amp = amp_stack();
    let rust = estimate_neighbors_ks(amp.view(), HalfWindow { y: 2, x: 2 }, S1, 0.001, false);
    let oracle: Array4<bool> = ndarray_npy::read_npy(dir.join("ks_neighbors.npy")).unwrap();
    assert_eq!(
        count_mismatch(&rust, &oracle),
        0,
        "KS mask differs from oracle"
    );
}

#[test]
fn shp_weighted_covariance_matches_oracle() {
    let dir = fixtures();
    if !dir.join("cov_C_shp.npy").exists() {
        eprintln!("skipping shp_weighted_covariance_matches_oracle: no fixtures");
        return;
    }
    let amp = amp_stack();
    let nslc = amp.len_of(Axis(0));
    let mean = amp.mean_axis(Axis(0)).unwrap();
    let var = amp.var_axis(Axis(0), 0.0);
    let neighbors = estimate_neighbors_glrt(
        mean.view(),
        var.view(),
        HalfWindow { y: 2, x: 2 },
        nslc,
        S1,
        0.001,
    );

    let stack: Array3<Cf32> = ndarray_npy::read_npy(dir.join("slc_stack.npy")).unwrap();
    let stack = stack.mapv(|z| Cf64::new(z.re as f64, z.im as f64));
    let c = dolphin_phaselink::estimate_stack_covariance(
        stack.view(),
        HalfWindow { y: 2, x: 2 },
        S1,
        Some(neighbors.view()),
    )
    .unwrap();

    let oracle: Array4<Cf32> = ndarray_npy::read_npy(dir.join("cov_C_shp.npy")).unwrap();
    let oracle = oracle.mapv(|z| Cf64::new(z.re as f64, z.im as f64));
    let max_err = c
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-4, "SHP-weighted covariance error {max_err}");
}
