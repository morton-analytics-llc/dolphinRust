//! Issue #50: layover/shadow masks must affect phase linking, not only config.
//!
//! dolphin v0.35.0 treats zero (and raster nodata) as invalid, any nonzero
//! value as valid, removes invalid native samples before covariance, and marks
//! a strided output invalid only when its whole stride cell is invalid.

use std::path::{Path, PathBuf};

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_phaselink::{ComputeEngine, FusedParams};
use dolphin_shp::{estimate_neighbors_glrt, estimate_neighbors_ks};
use dolphin_workflows::{run_sequential, run_sequential_masked, SequentialConfig};
use ndarray::{Array2, Array3, Array4, Axis};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn config(nslc: usize, stride: usize) -> SequentialConfig {
    SequentialConfig {
        ministack_size: nslc,
        max_num_compressed: 10,
        half_window: HalfWindow { y: 1, x: 1 },
        strides: Strides {
            y: stride,
            x: stride,
        },
        use_evd: true,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: true,
        shp_method: ShpMethod::Rect,
        shp_alpha: 0.001,
    }
}

fn fixture_stack() -> Array3<Cf64> {
    let stack: Array3<Cf32> =
        ndarray_npy::read_npy(fixtures().join("layover_shadow_mask_stack.npy")).unwrap();
    stack.mapv(|z| Cf64::new(f64::from(z.re), f64::from(z.im)))
}

fn fixture_mask() -> Array2<bool> {
    let values: Array2<u8> =
        ndarray_npy::read_npy(fixtures().join("layover_shadow_mask_validity.npy")).unwrap();
    values.mapv(|value| value != 0)
}

fn assert_bit_identical(
    actual: &dolphin_workflows::SequentialOutput,
    expected: &dolphin_workflows::SequentialOutput,
) {
    assert_eq!(actual.cpx_phase, expected.cpx_phase);
    assert_eq!(actual.compressed_slcs, expected.compressed_slcs);
    assert_eq!(actual.temporal_coherence, expected.temporal_coherence);
    assert_eq!(
        actual.phase_linking_coherence,
        expected.phase_linking_coherence
    );
    assert_eq!(actual.crlb_sigma, expected.crlb_sigma);
    assert_eq!(actual.closure_phase, expected.closure_phase);
    assert_eq!(actual.validity_mask, expected.validity_mask);
}

#[test]
fn all_valid_mask_is_bit_identical_to_unmasked_run() {
    let stack = fixture_stack();
    let mask = Array2::from_elem((stack.dim().1, stack.dim().2), true);
    let cfg = config(stack.dim().0, 1);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);

    let unmasked = run_sequential(stack.view(), &cfg, &engine).unwrap();
    let masked = run_sequential_masked(stack.view(), mask.view(), &cfg, &engine).unwrap();

    assert_bit_identical(&masked, &unmasked);
    assert!(masked.validity_mask.iter().all(|valid| *valid));
}

fn contaminated_stack(center_phase: impl Fn(usize) -> f64) -> Array3<Cf64> {
    const NSLC: usize = 8;
    let mut stack = Array3::from_shape_fn((NSLC, 5, 5), |(date, row, col)| {
        let phase = 0.22 * date as f64 + 0.04 * row as f64 - 0.03 * col as f64
            + 0.012 * (date * row) as f64
            + 0.008 * (date * col) as f64;
        let magnitude = 1.0 + 0.03 * (row + col) as f64;
        Cf64::from_polar(magnitude, phase)
    });
    for date in 0..NSLC {
        stack[(date, 2, 2)] = Cf64::from_polar(1.0e6, center_phase(date));
    }
    stack
}

fn phase_difference_at(
    left: &dolphin_workflows::SequentialOutput,
    right: &dolphin_workflows::SequentialOutput,
    pixel: (usize, usize),
) -> f64 {
    (0..left.cpx_phase.dim().0)
        .map(|date| {
            (left.cpx_phase[(date, pixel.0, pixel.1)] - right.cpx_phase[(date, pixel.0, pixel.1)])
                .norm()
        })
        .fold(0.0, f64::max)
}

fn selected_neighbors(stack: &Array3<Cf64>, cfg: &SequentialConfig) -> Array4<bool> {
    let amplitude = stack.mapv(|value| value.norm());
    match cfg.shp_method {
        ShpMethod::Glrt => estimate_neighbors_glrt(
            amplitude.mean_axis(Axis(0)).unwrap().view(),
            amplitude.var_axis(Axis(0), 0.0).view(),
            cfg.half_window,
            stack.dim().0,
            cfg.strides,
            cfg.shp_alpha,
        ),
        ShpMethod::Ks => estimate_neighbors_ks(
            amplitude.view(),
            cfg.half_window,
            cfg.strides,
            cfg.shp_alpha,
            false,
        ),
        ShpMethod::Rect => panic!("adaptive-neighbor contract requires GLRT or KS"),
    }
}

fn masked_stack(stack: &Array3<Cf64>, mask: &Array2<bool>) -> Array3<Cf64> {
    Array3::from_shape_fn(stack.dim(), |(date, row, col)| match mask[(row, col)] {
        true => stack[(date, row, col)],
        false => Cf64::new(f64::NAN, f64::NAN),
    })
}

fn fused_params(cfg: &SequentialConfig) -> FusedParams {
    FusedParams {
        use_evd: cfg.use_evd,
        beta: cfg.beta,
        zero_correlation_threshold: cfg.zero_correlation_threshold,
        reference_idx: cfg.output_reference_idx,
        compute_crlb: cfg.compute_crlb,
        crlb_reference_idx: 0,
        num_looks: (cfg.half_window.y as f64 * cfg.half_window.x as f64).sqrt(),
        compute_closure: cfg.compute_closure_phase,
        compute_average_coherence: cfg.compute_average_coherence,
        average_coherence_start_idx: 0,
    }
}

#[test]
fn adaptive_neighbors_are_selected_from_raw_samples_before_masking_covariance() {
    let stack = contaminated_stack(|date| 0.91 * (date * date) as f64);
    let mut mask = Array2::from_elem((5, 5), true);
    mask[(1, 1)] = false;
    let covariance_stack = masked_stack(&stack, &mask);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);

    for method in [ShpMethod::Glrt, ShpMethod::Ks] {
        let mut cfg = config(stack.dim().0, 1);
        cfg.shp_method = method;
        let raw_neighbors = selected_neighbors(&stack, &cfg);
        let post_mask_neighbors = selected_neighbors(&covariance_stack, &cfg);
        assert_ne!(
            raw_neighbors, post_mask_neighbors,
            "fixture does not distinguish raw and post-mask {method:?} selection"
        );

        let expected = engine
            .link(
                covariance_stack.view(),
                cfg.half_window,
                cfg.strides,
                Some(raw_neighbors.view()),
                fused_params(&cfg),
            )
            .unwrap();
        let post_mask_alternative = engine
            .link(
                covariance_stack.view(),
                cfg.half_window,
                cfg.strides,
                Some(post_mask_neighbors.view()),
                fused_params(&cfg),
            )
            .unwrap();
        assert!(
            expected
                .cpx_phase
                .iter()
                .zip(post_mask_alternative.cpx_phase.iter())
                .any(|(raw, masked)| (*raw - *masked).norm() > 1e-12),
            "fixture does not make {method:?} selection order observable"
        );

        let actual = run_sequential_masked(stack.view(), mask.view(), &cfg, &engine).unwrap();
        for ((date, row, col), value) in actual.cpx_phase.indexed_iter() {
            if mask[(row, col)] {
                assert_eq!(
                    *value,
                    expected.cpx_phase[(date, row, col)],
                    "{method:?} selected neighbors after applying the terrain mask"
                );
            }
        }
    }
}

#[test]
fn rect_covariance_excludes_a_masked_high_amplitude_contaminant() {
    let first = contaminated_stack(|date| 0.91 * (date * date) as f64);
    let second = contaminated_stack(|date| -1.17 * (date * date) as f64 + 0.4);
    let mut mask = Array2::from_elem((5, 5), true);
    mask[(2, 2)] = false;
    let cfg = config(first.dim().0, 1);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let neighbor = (2, 1);

    // Negative control: the contaminant changes a neighboring rectangular
    // covariance window when it is not masked.
    let unmasked_first = run_sequential(first.view(), &cfg, &engine).unwrap();
    let unmasked_second = run_sequential(second.view(), &cfg, &engine).unwrap();
    assert!(
        phase_difference_at(&unmasked_first, &unmasked_second, neighbor) > 1e-3,
        "contaminant fixture did not perturb the unmasked covariance"
    );

    let masked_first = run_sequential_masked(first.view(), mask.view(), &cfg, &engine).unwrap();
    let masked_second = run_sequential_masked(second.view(), mask.view(), &cfg, &engine).unwrap();
    assert!(masked_first.validity_mask[neighbor]);
    assert_eq!(
        phase_difference_at(&masked_first, &masked_second, neighbor),
        0.0,
        "masked native sample still influenced a neighboring covariance window"
    );
    assert_eq!(
        masked_first.temporal_coherence[neighbor],
        masked_second.temporal_coherence[neighbor]
    );
}

fn assert_oracle_validity(stride: usize) {
    let dir = fixtures();
    let stack = fixture_stack();
    let valid_mask = fixture_mask();
    let oracle_phase: Array3<Cf32> =
        ndarray_npy::read_npy(dir.join(format!("layover_shadow_mask_phase_stride{stride}.npy")))
            .unwrap();
    let oracle_quality: Array2<f32> = ndarray_npy::read_npy(dir.join(format!(
        "layover_shadow_mask_temporal_coherence_stride{stride}.npy"
    )))
    .unwrap();
    let cfg = config(stack.dim().0, stride);
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let output = run_sequential_masked(stack.view(), valid_mask.view(), &cfg, &engine).unwrap();
    let average_coherence = output
        .phase_linking_coherence
        .as_ref()
        .expect("average coherence requested");

    assert_eq!(output.cpx_phase.dim(), oracle_phase.dim());
    assert_eq!(output.temporal_coherence.dim(), oracle_quality.dim());
    assert_eq!(output.validity_mask.dim(), oracle_quality.dim());
    for ((row, col), oracle_value) in oracle_quality.indexed_iter() {
        let oracle_valid = oracle_value.is_finite();
        assert_eq!(
            output.validity_mask[(row, col)],
            oracle_valid,
            "stride {stride} validity differs from dolphin at ({row}, {col})"
        );
        assert_eq!(
            (0..oracle_phase.dim().0).all(|date| oracle_phase[(date, row, col)].re.is_finite()),
            oracle_valid,
            "oracle phase and quality masks disagree"
        );
        for date in 0..output.cpx_phase.dim().0 {
            let phase = output.cpx_phase[(date, row, col)];
            assert_eq!(
                phase.re.is_finite() && phase.im.is_finite(),
                oracle_valid,
                "stride {stride} phase validity differs at date {date}, ({row}, {col})"
            );
        }
        assert_eq!(
            output.temporal_coherence[(row, col)].is_finite(),
            oracle_valid,
            "stride {stride} temporal-coherence validity differs at ({row}, {col})"
        );
        assert_eq!(
            average_coherence[(row, col)].is_finite(),
            oracle_valid,
            "stride {stride} average-coherence validity differs at ({row}, {col})"
        );
    }
}

#[test]
fn stride_one_uses_zero_invalid_nonzero_valid_oracle_polarity() {
    assert_oracle_validity(1);
}

#[test]
fn strided_output_is_invalid_only_when_the_whole_cell_is_invalid() {
    assert_oracle_validity(2);
}
