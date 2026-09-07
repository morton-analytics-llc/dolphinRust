//! Phase-10 end-to-end contract test.
//!
//! `run_displacement` on a synthetic single-burst CSLC stack must reproduce the
//! dolphin-primitives oracle (phase-link → network → SNAPHU unwrap → SBAS L2 →
//! velocity) within physical tolerance. Skips without fixtures or `snaphu`.

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::Strides;
use dolphin_io::write_raster;
use dolphin_workflows::{
    run_displacement, run_displacement_with_output_policy, DisplacementOutputPolicy,
    VelocityEstimator,
};
use gdal::{Dataset, Metadata};
use ndarray::{Array2, Array3};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// Copy the historical numeric oracle into a georeferenced fixture stack. The
/// oracle predates the fail-closed source-georeference contract, so tests add
/// realistic 30 m UTM coordinate/projection datasets without changing samples.
fn georeferenced_config(label: &str) -> DisplacementWorkflow {
    let source = fixtures().join("disp/config.yaml");
    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(source).unwrap()).unwrap();
    let dir = std::env::temp_dir().join(format!("dolphin_georef_oracle_{label}"));
    std::fs::create_dir_all(&dir).unwrap();
    cfg.work_directory = dir.clone();
    cfg.cslc_file_list = cfg
        .cslc_file_list
        .iter()
        .map(|path| {
            let target = dir.join(path.file_name().unwrap());
            std::fs::copy(path, &target).unwrap();
            let file = hdf5::File::open_rw(&target).unwrap();
            let group = file.group("data").unwrap();
            let shape = group.dataset("VV").unwrap().shape();
            let x = (0..shape[1])
                .map(|col| 500_015.0 + col as f64 * 30.0)
                .collect::<Vec<_>>();
            let y = (0..shape[0])
                .map(|row| 4_200_015.0 - row as f64 * 30.0)
                .collect::<Vec<_>>();
            group
                .new_dataset_builder()
                .with_data(&x)
                .create("x_coordinates")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&y)
                .create("y_coordinates")
                .unwrap();
            group
                .new_dataset::<i64>()
                .create("projection")
                .unwrap()
                .write_scalar(&32611_i64)
                .unwrap();
            target
        })
        .collect();
    cfg
}

#[test]
fn end_to_end_displacement_matches_oracle() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() {
        eprintln!("skipping end-to-end oracle: no fixtures");
        return;
    }
    if !snaphu_available() {
        eprintln!("skipping end-to-end oracle: snaphu not on PATH");
        return;
    }

    let mut cfg = georeferenced_config("oracle");
    cfg.timeseries_options.use_coherence_weights = false;
    let out = run_displacement(&cfg).unwrap();
    assert_eq!(
        out.velocity_estimator,
        VelocityEstimator::LinearFullSeriesUnitPrecision
    );

    let disp_o: Array3<f64> = ndarray_npy::read_npy(dir.join("disp_displacement.npy")).unwrap();
    let vel_o: Array2<f64> = ndarray_npy::read_npy(dir.join("disp_velocity.npy")).unwrap();

    assert_eq!(out.displacement.dim(), disp_o.dim(), "displacement shape");
    let derr = out
        .displacement
        .iter()
        .zip(disp_o.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let verr = out
        .velocity
        .iter()
        .zip(vel_o.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // End-to-end chain: faer-vs-jax phase linking + normal-eq vs SVD lstsq, with
    // a shared SNAPHU binary on cycle-free input. Physical tolerance.
    assert!(derr < 1e-3, "displacement error {derr}");
    assert!(verr < 1e-2, "velocity error {verr}");

    // Quality layers: dolphin defaults write_crlb on, write_closure_phase off, so
    // the run produces the CRLB σ layer (per date, ref band 0) and no closure.
    let crlb = out.crlb_sigma.expect("write_crlb defaults on");
    let (rows, cols) = out.temporal_coherence.dim();
    assert_eq!(crlb.dim().1, rows, "crlb rows match the grid");
    assert_eq!(crlb.dim().2, cols, "crlb cols match the grid");
    let ref_band_max = crlb
        .index_axis(ndarray::Axis(0), 0)
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    assert_eq!(ref_band_max, 0.0, "CRLB reference band must be 0");
    assert!(out.closure_phase.is_none(), "closure off by default");
    assert!(
        out.phase_linking_coherence.is_none(),
        "average coherence off by default"
    );
}

#[test]
fn phase_similarity_raster_is_written_when_enabled() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping phase-similarity end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg = georeferenced_config("phase_similarity");
    cfg.unwrap_options.unwrap_method = dolphin_core::config::UnwrapMethod::Snaphu;
    cfg.phase_linking.write_phase_similarity = true;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_phase_similarity_e2e");
    let out = run_displacement(&cfg).unwrap();

    let similarity = out
        .phase_similarity
        .expect("write_phase_similarity enabled");
    assert_eq!(similarity.dim(), out.temporal_coherence.dim());
    // The metric is a mean cosine, so it is bounded on [-1, 1]; excluded pixels
    // are NaN rather than an in-range sentinel that would read as real agreement.
    assert!(similarity
        .iter()
        .all(|v| v.is_nan() || (-1.0..=1.0).contains(v)));
    assert!(similarity.iter().any(|v| v.is_finite()), "all-NaN raster");
    assert!(cfg.work_directory.join("phase_similarity.tif").exists());
    assert_ne!(
        similarity, out.temporal_coherence,
        "spatial similarity and temporal coherence must be distinct metrics"
    );
}

/// The layer is opt-in: nothing is computed or written unless it is enabled.
#[test]
fn phase_similarity_is_absent_by_default() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping phase-similarity default: no fixtures / snaphu");
        return;
    }
    let mut cfg = georeferenced_config("phase_similarity_off");
    cfg.unwrap_options.unwrap_method = dolphin_core::config::UnwrapMethod::Snaphu;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_phase_similarity_off_e2e");
    assert!(!cfg.phase_linking.write_phase_similarity);
    let out = run_displacement(&cfg).unwrap();
    assert!(out.phase_similarity.is_none());
    assert!(!cfg.work_directory.join("phase_similarity.tif").exists());
}

#[test]
fn distinct_phase_linking_coherence_raster_is_written_when_enabled() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping average-coherence end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg = georeferenced_config("average_coherence");
    cfg.unwrap_options.unwrap_method = dolphin_core::config::UnwrapMethod::Snaphu;
    cfg.phase_linking.calc_average_coh = true;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_average_coherence_e2e");
    let out = run_displacement(&cfg).unwrap();
    let coherence = out
        .phase_linking_coherence
        .expect("calc_average_coh enabled");
    assert_eq!(coherence.dim(), out.temporal_coherence.dim());
    assert!(coherence.iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(cfg
        .work_directory
        .join("phase_linking_coherence.tif")
        .exists());
    assert!(cfg.work_directory.join("temporal_coherence.tif").exists());
    assert!(cfg.work_directory.join("conncomp_00.tif").exists());
    assert_ne!(
        coherence, out.temporal_coherence,
        "metrics must be distinct"
    );
}

#[test]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
fn groundpulse_output_policy_preserves_arrays_and_emits_only_coherence() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping GroundPulse output-policy contract: no fixtures / snaphu");
        return;
    }

    let output_root = std::env::temp_dir().join(format!(
        "dolphinrust_output_policy_contract_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&output_root).unwrap();
    let mut full = georeferenced_config("output_policy");
    full.unwrap_options.unwrap_method = dolphin_core::config::UnwrapMethod::Snaphu;
    full.phase_linking.calc_average_coh = true;
    full.work_directory = output_root.join("full");
    let mut groundpulse = full.clone();
    groundpulse.work_directory = output_root.join("groundpulse");

    let full_output = run_displacement(&full).unwrap();
    let groundpulse_output =
        run_displacement_with_output_policy(&groundpulse, DisplacementOutputPolicy::GroundPulse)
            .unwrap();

    assert_eq!(full_output.displacement, groundpulse_output.displacement);
    assert_eq!(full_output.velocity, groundpulse_output.velocity);
    assert_eq!(
        full_output.velocity_estimator,
        groundpulse_output.velocity_estimator
    );
    assert_eq!(
        full_output.velocity_mm_yr,
        groundpulse_output.velocity_mm_yr
    );
    assert_eq!(
        full_output.velocity_sigma,
        groundpulse_output.velocity_sigma
    );
    assert_eq!(
        full_output.velocity_diagnostics,
        groundpulse_output.velocity_diagnostics
    );
    assert_eq!(
        full_output.displacement_variance,
        groundpulse_output.displacement_variance
    );
    assert_eq!(
        full_output.network_misclosure_rms,
        groundpulse_output.network_misclosure_rms
    );
    assert_eq!(
        full_output.timeseries_residual_rms,
        groundpulse_output.timeseries_residual_rms
    );
    assert_eq!(
        full_output.interferogram_pairs,
        groundpulse_output.interferogram_pairs
    );
    assert_eq!(
        full_output.unwrap_connected_components,
        groundpulse_output.unwrap_connected_components
    );
    assert_eq!(
        full_output.temporal_coherence,
        groundpulse_output.temporal_coherence
    );
    assert_eq!(
        full_output.phase_linking_coherence,
        groundpulse_output.phase_linking_coherence
    );
    assert_eq!(full_output.validity_mask, groundpulse_output.validity_mask);
    assert_eq!(full_output.crlb_sigma, groundpulse_output.crlb_sigma);
    assert_eq!(full_output.closure_phase, groundpulse_output.closure_phase);
    assert_eq!(
        full_output.acquisition_days,
        groundpulse_output.acquisition_days
    );
    assert_eq!(full_output.epsg, groundpulse_output.epsg);
    assert_eq!(full_output.geotransform, groundpulse_output.geotransform);
    assert_eq!(
        full_output.reference_point,
        groundpulse_output.reference_point
    );
    assert_eq!(
        full_output.ionosphere_delay,
        groundpulse_output.ionosphere_delay
    );
    assert_eq!(
        full_output.troposphere_delay,
        groundpulse_output.troposphere_delay
    );
    assert_eq!(
        full_output.solid_earth_tide_delay,
        groundpulse_output.solid_earth_tide_delay
    );
    match (&full_output.los_geometry, &groundpulse_output.los_geometry) {
        (None, None) => {}
        (Some(full), Some(groundpulse)) => {
            assert_eq!(full.east, groundpulse.east);
            assert_eq!(full.north, groundpulse.north);
            assert_eq!(full.up, groundpulse.up);
        }
        _ => panic!("LOS geometry presence changed with output policy"),
    }
    assert_eq!(
        serde_json::to_value(&full_output.geometry_provenance).unwrap(),
        serde_json::to_value(&groundpulse_output.geometry_provenance).unwrap()
    );

    let full_coherence = full.work_directory.join("phase_linking_coherence.tif");
    let groundpulse_coherence = groundpulse
        .work_directory
        .join("phase_linking_coherence.tif");
    assert_eq!(
        std::fs::read(full_coherence).unwrap(),
        std::fs::read(groundpulse_coherence).unwrap()
    );
    assert!(full.work_directory.join("velocity.tif").exists());
    assert!(!groundpulse.work_directory.join("velocity.tif").exists());
    assert!(full
        .work_directory
        .join("geometry_provenance.json")
        .exists());
    assert!(!groundpulse
        .work_directory
        .join("geometry_provenance.json")
        .exists());
    let groundpulse_artifacts = std::fs::read_dir(&groundpulse.work_directory)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| {
            matches!(
                entry.path().extension().and_then(|value| value.to_str()),
                Some("tif" | "json")
            )
        })
        .count();
    assert_eq!(groundpulse_artifacts, 1);
}

#[test]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
fn l2_diagnostic_products_are_opt_in_and_unit_aligned() {
    let dir = fixtures();
    if !dir.join("disp_displacement.npy").exists() || !snaphu_available() {
        eprintln!("skipping uncertainty end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg = georeferenced_config("uncertainty");
    cfg.timeseries_options.method = dolphin_core::config::TimeseriesMethod::L2;
    cfg.timeseries_options.write_posterior_uncertainty = true;
    cfg.timeseries_options.write_velocity_uncertainty = true;
    let out = run_displacement(&cfg).unwrap();
    assert_eq!(
        out.velocity_estimator,
        VelocityEstimator::LinearPostGaugeUnitPrecision
    );
    let variance = out
        .displacement_variance
        .expect("network parameter-covariance diagonal approximation");
    let sigma = out
        .velocity_sigma
        .as_ref()
        .expect("IID-conditional velocity component");
    let diagnostics = out
        .velocity_diagnostics
        .as_ref()
        .expect("velocity diagnostics");
    assert_eq!(variance.dim(), out.displacement.dim());
    assert_eq!(sigma.dim(), out.velocity.dim());
    assert!(variance.iter().any(|value| value.is_finite()));
    assert_eq!(diagnostics.valid_date_count.dim(), out.velocity.dim());
    assert_eq!(diagnostics.regression_dof.dim(), out.velocity.dim());
    assert!(cfg
        .work_directory
        .join("displacement_variance_00.tif")
        .exists());
    // Issue #40: the temporal motion-model fit residual and the SBAS
    // network-inversion misclosure are distinct rasters now, not one field
    // wearing two meanings.
    assert!(cfg
        .work_directory
        .join("timeseries_residual_rms.tif")
        .exists());
    assert!(cfg
        .work_directory
        .join("network_misclosure_rms.tif")
        .exists());
    assert!(out.timeseries_residual_rms.is_some());
    assert!(out.network_misclosure_rms.is_some());
    assert!(cfg.work_directory.join("velocity_sigma.tif").exists());
    for name in [
        "velocity_valid_date_count.tif",
        "velocity_regression_rank.tif",
        "velocity_regression_dof.tif",
        "velocity_uncertainty_status.tif",
        "velocity_lag1_rho.tif",
        "velocity_correlation_pair_count.tif",
        "velocity_cadence_status.tif",
        "velocity_correlation_available.tif",
        "velocity_diagnostic_inflation_factor.tif",
        "velocity_diagnostic_effective_sample_size.tif",
    ] {
        assert!(cfg.work_directory.join(name).exists(), "missing {name}");
    }
    assert!(cfg.work_directory.join("conncomp_00.tif").exists());
    let crlb = Dataset::open(cfg.work_directory.join("crlb_sigma_00.tif")).unwrap();
    assert_eq!(crlb.metadata_item("UNITTYPE", "").as_deref(), Some("rad"));
    let velocity_sigma = Dataset::open(cfg.work_directory.join("velocity_sigma.tif")).unwrap();
    let velocity_unit = if cfg.input_options.wavelength.is_some() {
        "m/yr"
    } else {
        "rad/yr"
    };
    assert_eq!(
        velocity_sigma.metadata_item("UNITTYPE", "").as_deref(),
        Some(velocity_unit)
    );
    assert_eq!(
        velocity_sigma
            .metadata_item("UNCERTAINTY_COMPONENT", "")
            .as_deref(),
        Some("independent_residual_conditional")
    );
    assert_eq!(
        velocity_sigma
            .metadata_item("TEMPORAL_COVARIANCE", "")
            .as_deref(),
        Some("not_modeled")
    );
    assert_eq!(
        velocity_sigma
            .metadata_item("CALIBRATION_STATUS", "")
            .as_deref(),
        Some("uncalibrated_component")
    );
    let lag1 = Dataset::open(cfg.work_directory.join("velocity_lag1_rho.tif")).unwrap();
    assert_eq!(
        lag1.metadata_item("EVIDENCE_ROLE", "").as_deref(),
        Some("diagnostic_only")
    );
    let status = Dataset::open(cfg.work_directory.join("velocity_uncertainty_status.tif")).unwrap();
    assert!(status.rasterband(1).unwrap().no_data_value().is_none());
    let network_variance =
        Dataset::open(cfg.work_directory.join("displacement_variance_00.tif")).unwrap();
    let variance_unit = if cfg.input_options.wavelength.is_some() {
        "m^2"
    } else {
        "rad^2"
    };
    assert_eq!(
        network_variance.metadata_item("UNITTYPE", "").as_deref(),
        Some(variance_unit)
    );
    assert_eq!(
        network_variance
            .metadata_item("UNCERTAINTY_SCOPE", "")
            .as_deref(),
        Some("independent_ifg_parameter_covariance_diagonal_approximation")
    );
    assert_eq!(
        network_variance
            .metadata_item("IFG_ERROR_ASSUMPTION", "")
            .as_deref(),
        Some("independent")
    );
    assert_eq!(
        network_variance
            .metadata_item("CALIBRATION_STATUS", "")
            .as_deref(),
        Some("not_calibrated")
    );
    assert_eq!(
        network_variance
            .metadata_item("SPATIAL_COVARIANCE", "")
            .as_deref(),
        Some("target_reference_covariance_not_modeled")
    );
}

/// Enabling the phase-bias correction (Michaelides 2022) runs end-to-end through
/// unwrap + inversion and produces a finite displacement of the right shape. The
/// correction leads Python dolphin (no oracle), so this guards the wiring; the
/// numeric behaviour is validated by `dolphin-phaselink`'s analytic + reduction
/// contracts. Default-off parity is covered by the oracle test above.
#[test]
fn phase_bias_correction_runs_end_to_end() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping phase-bias end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg = georeferenced_config("phase_bias");
    cfg.phase_linking.correct_phase_bias = true;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_phasebias_e2e");
    let out = run_displacement(&cfg).unwrap();
    assert!(
        out.displacement.iter().all(|v| v.is_finite()),
        "phase-bias-corrected displacement must be finite"
    );
    let (rows, cols) = out.temporal_coherence.dim();
    assert_eq!(
        (out.displacement.dim().1, out.displacement.dim().2),
        (rows, cols)
    );
}

/// Enabling `write_closure_phase` produces the closure layer end-to-end, with
/// `n_dates - 2` bands; the layer matches the kernel's per-triplet output.
#[test]
fn closure_layer_produced_when_enabled() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !dir.join("disp_displacement.npy").exists() || !config.exists() || !snaphu_available() {
        eprintln!("skipping closure end-to-end: no fixtures / snaphu");
        return;
    }
    let mut cfg = georeferenced_config("closure");
    cfg.phase_linking.write_closure_phase = true;
    // Isolate scratch/outputs from the other end-to-end test (they run in
    // parallel and would otherwise race on a shared SNAPHU scratch directory).
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_closure_e2e");
    let out = run_displacement(&cfg).unwrap();

    let n_dates = out.displacement.dim().0 + 1; // displacement drops the reference date
    let closure = out.closure_phase.expect("write_closure_phase enabled");
    assert_eq!(closure.dim().0, n_dates - 2, "closure has n_dates-2 bands");
    let (rows, cols) = out.temporal_coherence.dim();
    assert_eq!(
        (closure.dim().1, closure.dim().2),
        (rows, cols),
        "closure grid"
    );
}

fn assert_bounded_case(strides: Strides, target: (usize, usize, usize, usize), label: &str) {
    let mut full = georeferenced_config(&format!("bounded_{label}"));
    full.output_options.strides = strides;
    full.phase_linking.calc_average_coh = true;
    full.timeseries_options.reference_point =
        Some(((target.0 + target.1) / 2, (target.2 + target.3) / 2));
    full.work_directory = std::env::temp_dir().join(format!("dolphinrust_bounds_full_{label}"));
    let full_output = run_displacement(&full).unwrap();
    let gt = full_output.geotransform;
    let (row_start, row_stop, col_start, col_stop) = target;
    let mut bounded = full.clone();
    bounded.output_options.epsg = Some(32611);
    bounded.output_options.bounds_epsg = Some(32611);
    bounded.output_options.bounds = Some((
        gt[0] + col_start as f64 * gt[1],
        gt[3] + row_stop as f64 * gt[5],
        gt[0] + col_stop as f64 * gt[1],
        gt[3] + row_start as f64 * gt[5],
    ));
    bounded.work_directory =
        std::env::temp_dir().join(format!("dolphinrust_bounds_target_{label}"));
    if label == "1x2" {
        let mask_path = std::env::temp_dir().join("dolphinrust_bounds_aligned_mask.tif");
        let mask = Array2::from_elem(full_output.temporal_coherence.dim(), 1_u8);
        write_raster(&mask_path, mask.view(), gt, Some(32611), Some(0.0)).unwrap();
        bounded.mask_file = Some(mask_path);
    }
    let cropped = run_displacement(&bounded).unwrap();

    assert_eq!(
        cropped.temporal_coherence.dim(),
        (row_stop - row_start, col_stop - col_start)
    );
    assert_eq!(cropped.geotransform[0], gt[0] + col_start as f64 * gt[1]);
    assert_eq!(cropped.geotransform[3], gt[3] + row_start as f64 * gt[5]);
    let expected = full_output
        .phase_linking_coherence
        .as_ref()
        .unwrap()
        .slice(ndarray::s![row_start..row_stop, col_start..col_stop]);
    let actual = cropped.phase_linking_coherence.as_ref().unwrap();
    assert_eq!(actual.view(), expected, "phase-link halo parity at {label}");
    let expected_displacement =
        full_output
            .displacement
            .slice(ndarray::s![.., row_start..row_stop, col_start..col_stop]);
    let displacement_error = cropped
        .displacement
        .iter()
        .zip(expected_displacement.iter())
        .filter(|(actual, expected)| actual.is_finite() && expected.is_finite())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        displacement_error < 1e-3,
        "AOI-local displacement interior error {displacement_error} at {label}"
    );
    let provenance = cropped
        .geometry_provenance
        .processing_bounds
        .expect("bounded provenance");
    assert_eq!(provenance.output_epsg, 32611);
    assert_eq!(provenance.target_pixel_offset, [row_start, col_start]);
    if let Some((row, col)) = cropped.reference_point {
        assert!(row < cropped.temporal_coherence.dim().0);
        assert!(col < cropped.temporal_coherence.dim().1);
    }
}

#[test]
fn bounded_target_trims_after_analysis_at_both_required_strides() {
    if !fixtures().join("disp/config.yaml").exists() || !snaphu_available() {
        eprintln!("skipping bounded displacement contract: no fixtures / snaphu");
        return;
    }
    assert_bounded_case(Strides { y: 1, x: 2 }, (8, 30, 6, 24), "1x2");
    assert_bounded_case(Strides { y: 3, x: 6 }, (2, 13, 1, 9), "3x6");
}
