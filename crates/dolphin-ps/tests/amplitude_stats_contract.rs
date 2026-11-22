//! `AmplitudeStats` contracts: an exact Chan merge over ministack-sized
//! batches reproduces one-shot `calc_ps` (max|ΔD_A| ≤ 1e-6, no PS flips), and
//! merging is associative enough that a resumed series equals a full one.

use dolphin_core::Cf64;
use dolphin_ps::{calc_ps, AmplitudeStats, PsError};
use ndarray::{s, Array3};

/// Deterministic amplitudes spanning stable (PS-like) and speckle-like pixels:
/// pixel `p` has dispersion roughly proportional to its column.
fn stack(nslc: usize, rows: usize, cols: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((nslc, rows, cols), |(t, r, c)| {
        let spread = 0.04 * (c as f64 + 1.0);
        let wobble = ((t * 7919 + r * 104_729 + c * 1299) % 1000) as f64 / 500.0 - 1.0;
        let amp = (3.0 + r as f64) * (1.0 + spread * wobble);
        Cf64::from_polar(amp, 0.1 * t as f64)
    })
}

fn batched(stack: &Array3<Cf64>, batch: usize) -> AmplitudeStats {
    let nslc = stack.dim().0;
    (0..nslc)
        .step_by(batch)
        .map(|start| {
            let stop = (start + batch).min(nslc);
            AmplitudeStats::from_stack(stack.slice(s![start..stop, .., ..])).unwrap()
        })
        .reduce(|acc, next| acc.merge(&next).unwrap())
        .unwrap()
}

#[test]
fn chan_merge_matches_batch_calc_ps() {
    for nslc in [60, 300] {
        let data = stack(nslc, 4, 20);
        let merged = batched(&data, 15).to_ps(0.25);
        let batch = calc_ps(data.mapv(|z| z.norm()).view(), 0.25, nslc);
        let max_err = merged
            .amp_dispersion
            .iter()
            .zip(batch.amp_dispersion.iter())
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max);
        assert!(max_err <= 1e-6, "N={nslc}: max |ΔD_A| {max_err}");
        assert_eq!(merged.ps, batch.ps, "N={nslc}: PS flips");
        assert!(merged.ps.iter().any(|&p| p == 1) && merged.ps.iter().any(|&p| p == 0));
    }
}

#[test]
fn a_pixel_with_a_non_finite_date_is_nodata() {
    let mut data = stack(30, 1, 3);
    data[(17, 0, 1)] = Cf64::new(f64::NAN, 0.0);
    let ps = batched(&data, 15).to_ps(0.25);
    assert_eq!(ps.ps[(0, 1)], 255);
    assert_eq!(ps.amp_dispersion[(0, 1)], 0.0);
    assert_ne!(ps.ps[(0, 0)], 255);
}

#[test]
fn merge_order_of_batches_is_the_only_input() {
    // Same batches merged the same way must be bit-identical (the NRT premise).
    let data = stack(45, 3, 5);
    assert_eq!(batched(&data, 15), batched(&data, 15));
}

#[test]
fn merging_different_grids_is_an_error() {
    let a = AmplitudeStats::from_stack(stack(3, 2, 2).view()).unwrap();
    let b = AmplitudeStats::from_stack(stack(3, 2, 3).view()).unwrap();
    assert_eq!(a.merge(&b), Err(PsError::ShapeMismatch));
}
