//! Contract tests for the workflow config tree.
//!
//! Defaults must match dolphin's pydantic `DisplacementWorkflow`, enum strings
//! must match dolphin's YAML values, a real dolphin YAML (including option
//! groups we don't model) must deserialize, and our own emit must round-trip.

use dolphin_core::config::{
    CompressedSlcPlan, DisplacementWorkflow, ShpMethod, TimeseriesMethod, UnwrapMethod,
};

#[test]
fn defaults_match_dolphin() {
    let c = DisplacementWorkflow::default();
    assert_eq!(c.phase_linking.ministack_size, 15);
    assert_eq!(c.phase_linking.max_num_compressed, 10);
    assert_eq!(c.phase_linking.half_window.y, 7);
    assert_eq!(c.phase_linking.half_window.x, 14);
    assert_eq!(c.phase_linking.shp_method, ShpMethod::Glrt);
    assert_eq!(c.phase_linking.shp_alpha, 0.001);
    assert_eq!(
        c.phase_linking.compressed_slc_plan,
        CompressedSlcPlan::AlwaysFirst
    );
    assert!(c.phase_linking.write_crlb);
    assert_eq!(c.ps_options.amp_dispersion_threshold, 0.25);
    assert_eq!(c.timeseries_options.method, TimeseriesMethod::L1);
    assert_eq!(c.timeseries_options.correlation_threshold, 0.2);
    assert_eq!(c.timeseries_options.block_shape, (256, 256));
    assert_eq!(c.output_options.strides.y, 1);
    assert_eq!(c.output_options.strides.x, 1);
    assert_eq!(c.output_options.overview_levels, vec![4, 8, 16, 32, 64]);
    assert_eq!(c.unwrap_options.unwrap_method, UnwrapMethod::Snaphu);
    assert_eq!(c.unwrap_options.snaphu_options.init_method, "mcf");
    assert_eq!(c.unwrap_options.snaphu_options.cost, "smooth");
    assert_eq!(c.worker_settings.block_shape, (512, 512));
}

#[test]
fn enum_yaml_values_match_dolphin() {
    assert_eq!(
        serde_yaml::to_string(&ShpMethod::Glrt).unwrap().trim(),
        "glrt"
    );
    assert_eq!(serde_yaml::to_string(&ShpMethod::Ks).unwrap().trim(), "ks");
    assert_eq!(
        serde_yaml::to_string(&CompressedSlcPlan::AlwaysFirst)
            .unwrap()
            .trim(),
        "always_first"
    );
    assert_eq!(
        serde_yaml::to_string(&UnwrapMethod::Snaphu).unwrap().trim(),
        "snaphu"
    );
    assert_eq!(
        serde_yaml::to_string(&TimeseriesMethod::L1).unwrap().trim(),
        "L1"
    );
}

#[test]
fn default_config_round_trips() {
    let original = DisplacementWorkflow::default();
    let yaml = original.to_yaml().unwrap();
    let parsed = DisplacementWorkflow::from_yaml(&yaml).unwrap();
    assert_eq!(original, parsed);
}

/// A dolphin-style YAML: partial fields set, plus an unwrap solver block
/// (`tophu_options`) we deliberately don't model. Unspecified fields must take
/// dolphin defaults; the unmodeled block must be ignored, not rejected.
#[test]
fn real_dolphin_yaml_deserializes_with_defaults() {
    let yaml = r#"
cslc_file_list:
  - /data/t087_burst01_20200101.h5
  - /data/t087_burst01_20200113.h5
phase_linking:
  ministack_size: 20
  half_window:
    y: 11
    x: 5
output_options:
  strides:
    y: 6
    x: 3
unwrap_options:
  unwrap_method: spurt
  tophu_options:
    ntiles: [2, 2]
    downsample_factor: [3, 3]
work_directory: /work/run1
"#;
    let c = DisplacementWorkflow::from_yaml(yaml).unwrap();

    // Explicitly-set fields.
    assert_eq!(c.cslc_file_list.len(), 2);
    assert_eq!(c.phase_linking.ministack_size, 20);
    assert_eq!(c.phase_linking.half_window.y, 11);
    assert_eq!(c.phase_linking.half_window.x, 5);
    assert_eq!(c.output_options.strides.y, 6);
    assert_eq!(c.output_options.strides.x, 3);
    assert_eq!(c.unwrap_options.unwrap_method, UnwrapMethod::Spurt);

    // Unspecified fields fall back to dolphin defaults.
    assert_eq!(c.phase_linking.max_num_compressed, 10);
    assert_eq!(c.ps_options.amp_dispersion_threshold, 0.25);
    assert_eq!(c.timeseries_options.method, TimeseriesMethod::L1);
    assert_eq!(c.unwrap_options.snaphu_options.init_method, "mcf");
}
