//! Lever-1 fusion contract: [`link_fused`] must produce **bit-identical** output
//! to the separate-stage path (covariance → estimator → temp_coh → CRLB →
//! closure), proving the fused pass only avoids retaining the `N×N` cube and
//! changes no math. Run over a multi-pixel strided grid with an SHP mask, EMI
//! and EVD, so every kernel and the `idx→(r,c)` packing is exercised.

use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::{
    estimate_average_coherence, estimate_closure_phases, estimate_crlb, estimate_stack_covariance,
    estimate_temp_coh, link_fused, process_coherence_matrices, FusedParams,
};
use ndarray::{s, Array3, Array4};

/// Smooth ramp + per-(slc,row,col) speckle so coherence matrices are non-trivial
/// and pixels differ (exercises the per-pixel packing).
fn synth_stack(nslc: usize, rows: usize, cols: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((nslc, rows, cols), |(t, r, c)| {
        let phase = 0.4 * t as f64 * ((c as f64 + 1.0) / cols as f64)
            + 0.15 * r as f64
            + 0.05 * ((t * 5 + r * 3 + c) % 7) as f64;
        Cf64::from_polar(1.0, phase)
    })
}

fn unit_phasor(z: Cf64) -> Cf64 {
    Cf64::from_polar(1.0, z.arg())
}

/// Separate-stage reference output for one parameter set.
fn staged(
    stack: &Array3<Cf64>,
    half: HalfWindow,
    strides: Strides,
    neighbors: Option<&Array4<bool>>,
    p: FusedParams,
) -> (Array3<Cf64>, ndarray::Array2<f64>, Array3<f64>, Array3<f64>) {
    let c = estimate_stack_covariance(stack.view(), half, strides, neighbors.map(Array4::view))
        .unwrap();
    let est = process_coherence_matrices(
        c.view(),
        p.use_evd,
        p.beta,
        p.zero_correlation_threshold,
        p.reference_idx,
    );
    let cpx = est.cpx_phase.mapv(unit_phasor);
    let temp_coh = estimate_temp_coh(cpx.view().permuted_axes([1, 2, 0]), c.view());
    let crlb = estimate_crlb(
        c.view(),
        p.beta,
        p.zero_correlation_threshold,
        p.crlb_reference_idx,
        p.num_looks,
    );
    let closure = estimate_closure_phases(c.view());
    (cpx, temp_coh, crlb, closure)
}

fn assert_bit_identical(use_evd: bool, with_mask: bool) {
    let (nslc, rows, cols) = (7, 10, 12);
    let stack = synth_stack(nslc, rows, cols);
    let half = HalfWindow { y: 2, x: 2 };
    let strides = Strides { y: 2, x: 2 };
    let (out_rows, out_cols) = strides.out_shape((rows, cols));
    let (win_h, win_w) = (2 * half.y + 1, 2 * half.x + 1);
    // Deterministic SHP mask: drop a few samples so masking paths run.
    let neighbors = with_mask.then(|| {
        Array4::from_shape_fn((out_rows, out_cols, win_h, win_w), |(or, oc, wy, wx)| {
            (or + oc + wy * win_w + wx) % 5 != 0
        })
    });
    let p = FusedParams {
        use_evd,
        beta: 0.1,
        zero_correlation_threshold: 0.0,
        reference_idx: 2,
        compute_crlb: true,
        crlb_reference_idx: 1,
        num_looks: (half.y as f64 * half.x as f64).sqrt(),
        compute_closure: true,
        compute_average_coherence: true,
        average_coherence_start_idx: 2,
    };

    let (cpx_ref, tc_ref, crlb_ref, clo_ref) = staged(&stack, half, strides, neighbors.as_ref(), p);
    let fused = link_fused(
        stack.view(),
        half,
        strides,
        neighbors.as_ref().map(Array4::view),
        p,
    )
    .unwrap();

    assert_eq!(fused.cpx_phase, cpx_ref, "cpx_phase not bit-identical");
    assert_eq!(
        fused.temporal_coherence, tc_ref,
        "temp_coh not bit-identical"
    );
    let aggregate = fused
        .average_coherence
        .expect("average coherence requested");
    let cube = estimate_stack_covariance(
        stack.view(),
        half,
        strides,
        neighbors.as_ref().map(Array4::view),
    )
    .unwrap();
    let per_date = estimate_average_coherence(cube.view());
    for r in 0..out_rows {
        for c in 0..out_cols {
            let expected: Vec<_> = per_date
                .slice(s![p.average_coherence_start_idx.., r, c])
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            assert_eq!(aggregate.count[(r, c)], expected.len() as u32);
            assert_eq!(aggregate.sum[(r, c)], expected.iter().sum::<f64>());
        }
    }
    let crlb = fused.crlb_sigma.expect("crlb requested");
    nan_aware_eq(&crlb, &crlb_ref, "crlb");
    assert_eq!(
        fused.closure_phase.expect("closure requested"),
        clo_ref,
        "closure not bit-identical"
    );
}

#[test]
fn all_non_finite_slc_is_rejected_like_dolphin_v035() {
    let mut stack = synth_stack(4, 5, 5);
    stack
        .index_axis_mut(ndarray::Axis(0), 2)
        .fill(Cf64::new(f64::NAN, f64::NAN));
    let p = FusedParams {
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        reference_idx: 0,
        compute_crlb: false,
        crlb_reference_idx: 0,
        num_looks: 1.0,
        compute_closure: false,
        compute_average_coherence: false,
        average_coherence_start_idx: 0,
    };
    let result = link_fused(
        stack.view(),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        None,
        p,
    );
    let err = match result {
        Ok(_) => panic!("an all-non-finite acquisition must fail"),
        Err(err) => err,
    };
    assert!(err.contains("all non-finite"), "unexpected error: {err}");
}

#[test]
fn disabled_average_coherence_allocates_no_aggregate() {
    let stack = synth_stack(4, 7, 9);
    let params = FusedParams {
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        reference_idx: 0,
        compute_crlb: false,
        crlb_reference_idx: 0,
        num_looks: 1.0,
        compute_closure: false,
        compute_average_coherence: false,
        average_coherence_start_idx: 4,
    };
    let output = link_fused(
        stack.view(),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 2 },
        None,
        params,
    )
    .unwrap();
    assert!(output.average_coherence.is_none());
}

#[test]
fn average_coherence_flag_does_not_change_primary_outputs() {
    let stack = synth_stack(7, 10, 12);
    let mut params = FusedParams {
        use_evd: false,
        beta: 0.1,
        zero_correlation_threshold: 0.0,
        reference_idx: 0,
        compute_crlb: true,
        crlb_reference_idx: 0,
        num_looks: 2.0,
        compute_closure: true,
        compute_average_coherence: false,
        average_coherence_start_idx: 0,
    };
    let run = |params| {
        link_fused(
            stack.view(),
            HalfWindow { y: 2, x: 2 },
            Strides { y: 1, x: 2 },
            None,
            params,
        )
        .unwrap()
    };
    let disabled = run(params);
    params.compute_average_coherence = true;
    let enabled = run(params);
    assert_eq!(enabled.cpx_phase, disabled.cpx_phase);
    assert_eq!(enabled.temporal_coherence, disabled.temporal_coherence);
    nan_aware_eq(
        enabled.crlb_sigma.as_ref().unwrap(),
        disabled.crlb_sigma.as_ref().unwrap(),
        "crlb flag parity",
    );
    assert_eq!(enabled.closure_phase, disabled.closure_phase);
    assert!(disabled.average_coherence.is_none());
    assert!(enabled.average_coherence.is_some());
}

#[test]
fn enabled_average_coherence_rejects_out_of_range_real_date_start() {
    let stack = synth_stack(4, 7, 9);
    let params = FusedParams {
        use_evd: false,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        reference_idx: 0,
        compute_crlb: false,
        crlb_reference_idx: 0,
        num_looks: 1.0,
        compute_closure: false,
        compute_average_coherence: true,
        average_coherence_start_idx: 5,
    };
    let error = link_fused(
        stack.view(),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 2 },
        None,
        params,
    )
    .err()
    .expect("invalid start must fail before allocation");
    assert!(error.contains("average coherence start"));
}

/// CRLB can be NaN on singular pixels; compare NaN==NaN and finite bit-identical.
fn nan_aware_eq(a: &Array3<f64>, b: &Array3<f64>, what: &str) {
    assert_eq!(a.dim(), b.dim(), "{what} shape differs");
    for (x, y) in a.iter().zip(b.iter()) {
        let same = (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
        assert!(same, "{what} differs: {x} vs {y}");
    }
}

#[test]
fn fused_equals_staged_emi() {
    assert_bit_identical(false, false);
}

#[test]
fn fused_equals_staged_evd() {
    assert_bit_identical(true, false);
}

#[test]
fn fused_equals_staged_emi_masked() {
    assert_bit_identical(false, true);
}
