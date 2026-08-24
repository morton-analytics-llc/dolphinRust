//! Contract for exhaustive, fail-loud public workflow configuration.

use std::collections::BTreeSet;
use std::path::PathBuf;

use dolphin_core::config::{
    ConfigFieldDisposition, CorrectionOptions, DisplacementWorkflow, EmpiricalSourceFactorOptions,
    InputOptions, InterferogramNetwork, OutputOptions, PhaseLinkingOptions, PreprocessOptions,
    PsOptions, SnaphuOptions, TimeseriesOptions, TophuOptions, UnwrapMethod, UnwrapOptions,
    WorkerSettings, CONFIG_BEHAVIOR_CONTRACTS, CONFIG_FIELD_DISPOSITIONS,
};
use dolphin_core::CoreError;

macro_rules! audit_fields {
    ($value:expr, $kind:ident, $paths:ident, $prefix:literal, [$($field:ident),+ $(,)?]) => {
        let $kind { $($field),+ } = $value;
        $($paths.push(concat!($prefix, stringify!($field)));)+
    };
}

const EXPECTED_CONFIG_PATHS: &[&str] = &[
    "input_options",
    "cslc_file_list",
    "output_options",
    "ps_options",
    "amplitude_dispersion_files",
    "amplitude_mean_files",
    "layover_shadow_mask_files",
    "phase_linking",
    "interferogram_network",
    "unwrap_options",
    "timeseries_options",
    "correction_options",
    "mask_file",
    "work_directory",
    "worker_settings",
    "log_file",
    "ps_options.amp_dispersion_threshold",
    "phase_linking.ministack_size",
    "phase_linking.max_num_compressed",
    "phase_linking.output_reference_idx",
    "phase_linking.half_window",
    "phase_linking.use_evd",
    "phase_linking.beta",
    "phase_linking.zero_correlation_threshold",
    "phase_linking.shp_method",
    "phase_linking.shp_alpha",
    "phase_linking.mask_input_ps",
    "phase_linking.baseline_lag",
    "phase_linking.compressed_slc_plan",
    "phase_linking.empirical_source_factor",
    "phase_linking.empirical_source_factor.half_window",
    "phase_linking.empirical_source_factor.shrinkage_alpha",
    "phase_linking.empirical_source_factor.relative_diagonal_floor",
    "phase_linking.write_covariance_operator",
    "phase_linking.write_crlb",
    "phase_linking.write_closure_phase",
    "phase_linking.calc_average_coh",
    "phase_linking.correct_phase_bias",
    "interferogram_network.reference_idx",
    "interferogram_network.max_bandwidth",
    "interferogram_network.max_temporal_baseline",
    "interferogram_network.indexes",
    "timeseries_options.run_inversion",
    "timeseries_options.method",
    "timeseries_options.reference_point",
    "timeseries_options.run_velocity",
    "timeseries_options.apply_mask_to_timeseries",
    "timeseries_options.correlation_threshold",
    "timeseries_options.block_shape",
    "timeseries_options.num_parallel_blocks",
    "timeseries_options.use_coherence_weights",
    "timeseries_options.write_posterior_uncertainty",
    "timeseries_options.write_velocity_uncertainty",
    "timeseries_options.correct_velocity_temporal_correlation",
    "timeseries_options.velocity_seasonal",
    "timeseries_options.velocity_step_dates",
    "timeseries_options.mask_unwrap_loop_errors",
    "unwrap_options.snaphu_options.ntiles",
    "unwrap_options.snaphu_options.tile_overlap",
    "unwrap_options.snaphu_options.n_parallel_tiles",
    "unwrap_options.snaphu_options.init_method",
    "unwrap_options.snaphu_options.cost",
    "unwrap_options.snaphu_options.single_tile_reoptimize",
    "unwrap_options.snaphu_options.auto_tile",
    "unwrap_options.tophu_options.ntiles",
    "unwrap_options.tophu_options.downsample_factor",
    "unwrap_options.tophu_options.init_method",
    "unwrap_options.tophu_options.cost",
    "unwrap_options.preprocess_options.alpha",
    "unwrap_options.preprocess_options.max_radius",
    "unwrap_options.preprocess_options.interpolation_cor_threshold",
    "unwrap_options.preprocess_options.interpolation_similarity_threshold",
    "unwrap_options.preprocess_options.zero_correlation_where_interpolating",
    "unwrap_options.run_unwrap",
    "unwrap_options.run_goldstein",
    "unwrap_options.run_interpolation",
    "unwrap_options.unwrap_method",
    "unwrap_options.n_parallel_jobs",
    "unwrap_options.zero_where_masked",
    "unwrap_options.preprocess_options",
    "unwrap_options.snaphu_options",
    "unwrap_options.tophu_options",
    "input_options.input_type",
    "input_options.subdataset",
    "input_options.cslc_date_fmt",
    "input_options.wavelength",
    "correction_options.ionosphere_files",
    "correction_options.troposphere_files",
    "correction_options.geometry_files",
    "correction_options.dem_file",
    "correction_options.incidence_angle_deg",
    "correction_options.troposphere_variable",
    "correction_options.solid_earth_tide",
    "output_options.strides",
    "output_options.epsg",
    "output_options.bounds",
    "output_options.bounds_epsg",
    "output_options.add_overviews",
    "output_options.overview_levels",
    "worker_settings.gpu_enabled",
    "worker_settings.compute_backend",
    "worker_settings.threads_per_worker",
    "worker_settings.n_parallel_bursts",
    "worker_settings.block_shape",
];

/// Each macro invocation is both an exhaustive no-`..` destructure and the
/// source of its full registry paths. Adding a field first fails compilation;
/// adding it here then adds an expected path, so the registry still fails until
/// the field receives a disposition.
fn audited_config_paths() -> Vec<&'static str> {
    let mut paths = Vec::new();
    audit_fields!(
        DisplacementWorkflow::default(),
        DisplacementWorkflow,
        paths,
        "",
        [
            input_options,
            cslc_file_list,
            output_options,
            ps_options,
            amplitude_dispersion_files,
            amplitude_mean_files,
            layover_shadow_mask_files,
            phase_linking,
            interferogram_network,
            unwrap_options,
            timeseries_options,
            correction_options,
            mask_file,
            work_directory,
            worker_settings,
            log_file,
        ]
    );

    audit_input(input_options, &mut paths);
    audit_output(output_options, &mut paths);
    audit_ps(ps_options, &mut paths);
    audit_phase_linking(phase_linking, &mut paths);
    audit_network(interferogram_network, &mut paths);
    audit_timeseries(timeseries_options, &mut paths);
    audit_unwrap(unwrap_options, &mut paths);
    audit_corrections(correction_options, &mut paths);
    audit_worker(worker_settings, &mut paths);
    let _ = (
        cslc_file_list,
        amplitude_dispersion_files,
        amplitude_mean_files,
        layover_shadow_mask_files,
        mask_file,
        work_directory,
        log_file,
    );
    paths
}

fn audit_input(value: InputOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        InputOptions,
        paths,
        "input_options.",
        [input_type, subdataset, cslc_date_fmt, wavelength,]
    );
    let _ = (input_type, subdataset, cslc_date_fmt, wavelength);
}

fn audit_output(value: OutputOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        OutputOptions,
        paths,
        "output_options.",
        [
            strides,
            epsg,
            bounds,
            bounds_epsg,
            add_overviews,
            overview_levels,
        ]
    );
    let _ = (
        strides,
        epsg,
        bounds,
        bounds_epsg,
        add_overviews,
        overview_levels,
    );
}

fn audit_ps(value: PsOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        PsOptions,
        paths,
        "ps_options.",
        [amp_dispersion_threshold,]
    );
    let _ = amp_dispersion_threshold;
}

fn audit_phase_linking(value: PhaseLinkingOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        PhaseLinkingOptions,
        paths,
        "phase_linking.",
        [
            ministack_size,
            max_num_compressed,
            output_reference_idx,
            half_window,
            use_evd,
            beta,
            zero_correlation_threshold,
            shp_method,
            shp_alpha,
            mask_input_ps,
            baseline_lag,
            compressed_slc_plan,
            empirical_source_factor,
            write_covariance_operator,
            write_crlb,
            write_closure_phase,
            calc_average_coh,
            correct_phase_bias,
        ]
    );
    let _ = (
        ministack_size,
        max_num_compressed,
        output_reference_idx,
        half_window,
        use_evd,
        beta,
        zero_correlation_threshold,
        shp_method,
        shp_alpha,
        mask_input_ps,
        baseline_lag,
        compressed_slc_plan,
        write_covariance_operator,
        write_crlb,
        write_closure_phase,
        calc_average_coh,
        correct_phase_bias,
    );
    audit_fields!(
        empirical_source_factor,
        EmpiricalSourceFactorOptions,
        paths,
        "phase_linking.empirical_source_factor.",
        [half_window, shrinkage_alpha, relative_diagonal_floor,]
    );
    let _ = (half_window, shrinkage_alpha, relative_diagonal_floor);
}

fn audit_network(value: InterferogramNetwork, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        InterferogramNetwork,
        paths,
        "interferogram_network.",
        [reference_idx, max_bandwidth, max_temporal_baseline, indexes,]
    );
    let _ = (reference_idx, max_bandwidth, max_temporal_baseline, indexes);
}

fn audit_timeseries(value: TimeseriesOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        TimeseriesOptions,
        paths,
        "timeseries_options.",
        [
            run_inversion,
            method,
            reference_point,
            run_velocity,
            apply_mask_to_timeseries,
            correlation_threshold,
            block_shape,
            num_parallel_blocks,
            use_coherence_weights,
            write_posterior_uncertainty,
            write_velocity_uncertainty,
            correct_velocity_temporal_correlation,
            velocity_seasonal,
            velocity_step_dates,
            mask_unwrap_loop_errors,
        ]
    );
    let _ = (
        run_inversion,
        method,
        reference_point,
        run_velocity,
        apply_mask_to_timeseries,
        correlation_threshold,
        block_shape,
        num_parallel_blocks,
        use_coherence_weights,
        write_posterior_uncertainty,
        write_velocity_uncertainty,
        correct_velocity_temporal_correlation,
        velocity_seasonal,
        velocity_step_dates,
        mask_unwrap_loop_errors,
    );
}

fn audit_unwrap(value: UnwrapOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        UnwrapOptions,
        paths,
        "unwrap_options.",
        [
            run_unwrap,
            run_goldstein,
            run_interpolation,
            unwrap_method,
            n_parallel_jobs,
            zero_where_masked,
            preprocess_options,
            snaphu_options,
            tophu_options,
        ]
    );
    audit_preprocess(preprocess_options, paths);
    audit_snaphu(snaphu_options, paths);
    audit_tophu(tophu_options, paths);
    let _ = (
        run_unwrap,
        run_goldstein,
        run_interpolation,
        unwrap_method,
        n_parallel_jobs,
        zero_where_masked,
    );
}

fn audit_preprocess(value: PreprocessOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        PreprocessOptions,
        paths,
        "unwrap_options.preprocess_options.",
        [
            alpha,
            max_radius,
            interpolation_cor_threshold,
            interpolation_similarity_threshold,
            zero_correlation_where_interpolating,
        ]
    );
    let _ = (
        alpha,
        max_radius,
        interpolation_cor_threshold,
        interpolation_similarity_threshold,
        zero_correlation_where_interpolating,
    );
}

fn audit_snaphu(value: SnaphuOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        SnaphuOptions,
        paths,
        "unwrap_options.snaphu_options.",
        [
            ntiles,
            tile_overlap,
            n_parallel_tiles,
            init_method,
            cost,
            single_tile_reoptimize,
            auto_tile,
        ]
    );
    let _ = (
        ntiles,
        tile_overlap,
        n_parallel_tiles,
        init_method,
        cost,
        single_tile_reoptimize,
        auto_tile,
    );
}

fn audit_tophu(value: TophuOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        TophuOptions,
        paths,
        "unwrap_options.tophu_options.",
        [ntiles, downsample_factor, init_method, cost,]
    );
    let _ = (ntiles, downsample_factor, init_method, cost);
}

fn audit_corrections(value: CorrectionOptions, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        CorrectionOptions,
        paths,
        "correction_options.",
        [
            ionosphere_files,
            troposphere_files,
            geometry_files,
            dem_file,
            incidence_angle_deg,
            troposphere_variable,
            solid_earth_tide,
        ]
    );
    let _ = (
        ionosphere_files,
        troposphere_files,
        geometry_files,
        dem_file,
        incidence_angle_deg,
        troposphere_variable,
        solid_earth_tide,
    );
}

fn audit_worker(value: WorkerSettings, paths: &mut Vec<&'static str>) {
    audit_fields!(
        value,
        WorkerSettings,
        paths,
        "worker_settings.",
        [
            gpu_enabled,
            compute_backend,
            threads_per_worker,
            n_parallel_bursts,
            block_shape,
        ]
    );
    let _ = (
        gpu_enabled,
        compute_backend,
        threads_per_worker,
        n_parallel_bursts,
        block_shape,
    );
}

type MutateConfig = fn(&mut DisplacementWorkflow);

fn compatibility_only_cases() -> Vec<(&'static str, MutateConfig)> {
    vec![
        ("output_options.add_overviews", |c| {
            c.output_options.add_overviews = false
        }),
        ("output_options.overview_levels", |c| {
            c.output_options.overview_levels = vec![2]
        }),
        ("ps_options.amp_dispersion_threshold", |c| {
            c.ps_options.amp_dispersion_threshold = 0.5
        }),
        ("amplitude_dispersion_files", |c| {
            c.amplitude_dispersion_files
                .push(PathBuf::from("amp-disp.tif"))
        }),
        ("amplitude_mean_files", |c| {
            c.amplitude_mean_files.push(PathBuf::from("amp-mean.tif"))
        }),
        ("phase_linking.mask_input_ps", |c| {
            c.phase_linking.mask_input_ps = true
        }),
        ("phase_linking.baseline_lag", |c| {
            c.phase_linking.baseline_lag = Some(2)
        }),
        ("unwrap_options.run_unwrap", |c| {
            c.unwrap_options.run_unwrap = false
        }),
        ("unwrap_options.run_goldstein", |c| {
            c.unwrap_options.run_goldstein = true
        }),
        ("unwrap_options.run_interpolation", |c| {
            c.unwrap_options.run_interpolation = true
        }),
        ("unwrap_options.preprocess_options.alpha", |c| {
            c.unwrap_options.preprocess_options.alpha = 0.75
        }),
        ("unwrap_options.preprocess_options.max_radius", |c| {
            c.unwrap_options.preprocess_options.max_radius = 25
        }),
        (
            "unwrap_options.preprocess_options.interpolation_cor_threshold",
            |c| {
                c.unwrap_options
                    .preprocess_options
                    .interpolation_cor_threshold = 0.4
            },
        ),
        (
            "unwrap_options.preprocess_options.interpolation_similarity_threshold",
            |c| {
                c.unwrap_options
                    .preprocess_options
                    .interpolation_similarity_threshold = 0.4
            },
        ),
        (
            "unwrap_options.preprocess_options.zero_correlation_where_interpolating",
            |c| {
                c.unwrap_options
                    .preprocess_options
                    .zero_correlation_where_interpolating = true
            },
        ),
        (
            "unwrap_options.snaphu_options.single_tile_reoptimize",
            |c| c.unwrap_options.snaphu_options.single_tile_reoptimize = true,
        ),
        ("timeseries_options.run_inversion", |c| {
            c.timeseries_options.run_inversion = false
        }),
        ("timeseries_options.run_velocity", |c| {
            c.timeseries_options.run_velocity = false
        }),
        ("timeseries_options.apply_mask_to_timeseries", |c| {
            c.timeseries_options.apply_mask_to_timeseries = false
        }),
        ("timeseries_options.block_shape", |c| {
            c.timeseries_options.block_shape = (128, 128)
        }),
        ("timeseries_options.num_parallel_blocks", |c| {
            c.timeseries_options.num_parallel_blocks = 2
        }),
        (
            "timeseries_options.correct_velocity_temporal_correlation",
            |c| c.timeseries_options.correct_velocity_temporal_correlation = true,
        ),
        ("worker_settings.gpu_enabled", |c| {
            c.worker_settings.gpu_enabled = true
        }),
        ("worker_settings.threads_per_worker", |c| {
            c.worker_settings.threads_per_worker = 2
        }),
        ("worker_settings.n_parallel_bursts", |c| {
            c.worker_settings.n_parallel_bursts = 2
        }),
        ("log_file", |c| {
            c.log_file = Some(PathBuf::from("dolphin.log"))
        }),
    ]
}

#[test]
fn registry_covers_every_public_config_field_once() {
    let audited_paths = audited_config_paths();
    let expected: BTreeSet<_> = audited_paths.iter().copied().collect();
    let documented: BTreeSet<_> = EXPECTED_CONFIG_PATHS.iter().copied().collect();
    let actual: BTreeSet<_> = CONFIG_FIELD_DISPOSITIONS
        .iter()
        .map(|entry| entry.path)
        .collect();
    assert_eq!(
        actual.len(),
        CONFIG_FIELD_DISPOSITIONS.len(),
        "duplicate config path"
    );
    assert_eq!(
        documented.len(),
        EXPECTED_CONFIG_PATHS.len(),
        "duplicate documented path"
    );
    assert_eq!(
        expected.len(),
        audited_paths.len(),
        "duplicate audited path"
    );
    assert_eq!(
        documented, expected,
        "manual path receipt drifted from exhaustive config destructures"
    );
    assert_eq!(actual, expected);
}

#[test]
fn consumed_and_conditional_fields_name_checked_behavior_contracts() {
    for contract in CONFIG_BEHAVIOR_CONTRACTS {
        assert!(
            !contract.reader.trim().is_empty(),
            "{} has no production reader",
            contract.id
        );
        assert!(
            !contract.evidence.trim().is_empty(),
            "{} has no test evidence",
            contract.id
        );
    }
    let contract_ids: BTreeSet<_> = CONFIG_BEHAVIOR_CONTRACTS
        .iter()
        .map(|contract| contract.id)
        .collect();
    assert_eq!(
        contract_ids.len(),
        CONFIG_BEHAVIOR_CONTRACTS.len(),
        "duplicate contract ID"
    );

    let mut referenced = BTreeSet::new();
    for entry in CONFIG_FIELD_DISPOSITIONS {
        match entry.disposition {
            ConfigFieldDisposition::Consumed { contract_id } => {
                assert!(
                    contract_ids.contains(contract_id),
                    "{} names unknown contract {contract_id}",
                    entry.path
                );
                referenced.insert(contract_id);
            }
            ConfigFieldDisposition::Conditional { contract_id, gate } => {
                assert!(
                    contract_ids.contains(contract_id),
                    "{} names unknown contract {contract_id}",
                    entry.path
                );
                assert!(
                    !gate.trim().is_empty(),
                    "{} has no runtime gate",
                    entry.path
                );
                referenced.insert(contract_id);
            }
            ConfigFieldDisposition::CompatibilityOnly { reason } => {
                assert!(
                    !reason.trim().is_empty(),
                    "{} has no compatibility reason",
                    entry.path
                );
            }
        }
    }
    assert_eq!(
        referenced, contract_ids,
        "catalog contains an unchecked contract ID"
    );
}

#[test]
fn compatibility_only_non_defaults_fail_with_full_yaml_path() {
    let cases = compatibility_only_cases();
    let tested_paths: BTreeSet<_> = cases.iter().map(|(path, _)| *path).collect();

    for entry in CONFIG_FIELD_DISPOSITIONS {
        if matches!(
            entry.disposition,
            ConfigFieldDisposition::CompatibilityOnly { reason: _ }
        ) {
            assert!(
                tested_paths.contains(entry.path)
                    || tested_paths
                        .iter()
                        .any(|path| path.starts_with(&format!("{}.", entry.path))),
                "{} has no non-default validation case",
                entry.path
            );
        }
    }

    for (path, mutate) in cases {
        let mut config = DisplacementWorkflow::default();
        mutate(&mut config);
        match config.validate_supported_options() {
            Err(CoreError::InvalidConfig(message)) => {
                assert!(message.contains(path), "{path}: {message}");
            }
            other => panic!("{path} was not rejected: {other:?}"),
        }
    }
}

#[test]
fn unimplemented_unwrap_backends_fail_instead_of_falling_through_to_snaphu() {
    for method in [
        UnwrapMethod::Icu,
        UnwrapMethod::Phass,
        UnwrapMethod::Spurt,
        UnwrapMethod::Whirlwind,
    ] {
        let mut config = DisplacementWorkflow::default();
        config.unwrap_options.unwrap_method = method;
        let error = config.validate_supported_options().unwrap_err().to_string();
        assert!(error.contains("unwrap_options.unwrap_method"), "{error}");
        assert!(
            error.contains("supports only native, snaphu, and tophu"),
            "{error}"
        );
    }
}

#[test]
fn invalid_unwrap_backend_option_strings_fail_instead_of_collapsing_to_defaults() {
    let mut config = DisplacementWorkflow::default();
    config.unwrap_options.snaphu_options.init_method = "typo".into();
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("unwrap_options.snaphu_options.init_method"),
        "{error}"
    );

    let mut config = DisplacementWorkflow::default();
    config.unwrap_options.snaphu_options.cost = "typo".into();
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("unwrap_options.snaphu_options.cost"),
        "{error}"
    );

    let mut config = DisplacementWorkflow::default();
    config.unwrap_options.unwrap_method = UnwrapMethod::Tophu;
    config.unwrap_options.tophu_options.init_method = "typo".into();
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("unwrap_options.tophu_options.init_method"),
        "{error}"
    );

    let mut config = DisplacementWorkflow::default();
    config.unwrap_options.unwrap_method = UnwrapMethod::Tophu;
    config.unwrap_options.tophu_options.cost = "typo".into();
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("unwrap_options.tophu_options.cost"),
        "{error}"
    );
}

#[test]
fn native_rejects_the_inert_snaphu_init_method_but_snaphu_consumes_it() {
    let mut config = DisplacementWorkflow::default();
    config.unwrap_options.snaphu_options.init_method = "mst".into();
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("unwrap_options.snaphu_options.init_method"),
        "{error}"
    );

    config.unwrap_options.unwrap_method = UnwrapMethod::Snaphu;
    config.validate_supported_options().unwrap();
}

#[test]
fn unbounded_output_epsg_fails_instead_of_claiming_an_unimplemented_fallback() {
    let mut config = DisplacementWorkflow::default();
    config.output_options.epsg = Some(32611);
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(error.contains("output_options.epsg"), "{error}");

    config.output_options.bounds = Some((0.0, 0.0, 1.0, 1.0));
    config.output_options.bounds_epsg = Some(32611);
    config.validate_supported_options().unwrap();
}

#[test]
fn posterior_uncertainty_rejects_l1_before_workflow_io() {
    let mut config = DisplacementWorkflow::default();
    config.timeseries_options.write_posterior_uncertainty = true;
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("timeseries_options.write_posterior_uncertainty"),
        "{error}"
    );

    config.timeseries_options.method = dolphin_core::config::TimeseriesMethod::L2;
    config.validate_supported_options().unwrap();
}

#[test]
fn zero_output_stride_fails_before_workflow_io() {
    for strides in [
        dolphin_core::Strides { y: 0, x: 1 },
        dolphin_core::Strides { y: 1, x: 0 },
    ] {
        let mut config = DisplacementWorkflow::default();
        config.output_options.strides = strides;
        let error = config.validate_supported_options().unwrap_err().to_string();
        assert!(error.contains("output_options.strides"), "{error}");
        assert!(error.contains("must both be positive"), "{error}");
    }
}

#[test]
fn covariance_operator_is_opt_in_and_round_trips_without_changing_legacy_defaults() {
    let defaults = DisplacementWorkflow::default();
    assert!(!defaults.phase_linking.write_covariance_operator);
    assert_eq!(
        defaults.phase_linking.empirical_source_factor,
        EmpiricalSourceFactorOptions::default()
    );

    let mut enabled = defaults;
    enabled.phase_linking.write_covariance_operator = true;
    enabled
        .phase_linking
        .empirical_source_factor
        .shrinkage_alpha = 0.25;
    let reparsed = DisplacementWorkflow::from_yaml(&enabled.to_yaml().unwrap()).unwrap();
    assert!(reparsed.phase_linking.write_covariance_operator);
    assert_eq!(
        reparsed
            .phase_linking
            .empirical_source_factor
            .shrinkage_alpha,
        0.25
    );
    reparsed.validate_supported_options().unwrap();
}

#[test]
fn invalid_empirical_source_factor_fails_before_covariance_source_io() {
    let mut config = DisplacementWorkflow::default();
    config.phase_linking.write_covariance_operator = true;
    config.phase_linking.empirical_source_factor.half_window.y = usize::MAX;
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("empirical_source_factor.half_window"),
        "{error}"
    );

    config.phase_linking.empirical_source_factor.half_window.y = 1;
    config.phase_linking.empirical_source_factor.shrinkage_alpha = 0.0;
    let error = config.validate_supported_options().unwrap_err().to_string();
    assert!(
        error.contains("empirical_source_factor.shrinkage_alpha"),
        "{error}"
    );
}
