use dolphin_core::config::{
    DisplacementWorkflow, TemporalUncertaintyMethod, TemporalUncertaintyOptions,
};

#[test]
fn corrected_temporal_uncertainty_is_disabled_and_fail_closed_by_default() {
    let cfg = DisplacementWorkflow::default();
    assert_eq!(
        cfg.timeseries_options.temporal_uncertainty,
        TemporalUncertaintyOptions::default()
    );
    assert_eq!(
        cfg.timeseries_options.temporal_uncertainty.method,
        TemporalUncertaintyMethod::Disabled
    );

    let mut enabled = cfg;
    enabled.timeseries_options.temporal_uncertainty.method =
        TemporalUncertaintyMethod::RemlCovarianceParameterAdjustedScalar;
    let error = enabled
        .validate_supported_options()
        .unwrap_err()
        .to_string();
    assert!(error.contains("evidence_directory"), "{error}");
}
