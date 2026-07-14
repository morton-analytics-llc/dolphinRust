//! Contract tests for the workflow config tree.
//!
//! Defaults must match dolphin's pydantic `DisplacementWorkflow`, enum strings
//! must match dolphin's YAML values, a real dolphin YAML (including option
//! groups we don't model) must deserialize, and our own emit must round-trip.

use dolphin_core::config::{
    CompressedSlcPlan, DisplacementWorkflow, InputType, ShpMethod, TimeseriesMethod, UnwrapMethod,
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
    assert!(!c.phase_linking.write_closure_phase);
    assert_eq!(c.ps_options.amp_dispersion_threshold, 0.25);
    assert_eq!(c.timeseries_options.method, TimeseriesMethod::L1);
    assert_eq!(c.timeseries_options.correlation_threshold, 0.2);
    assert_eq!(c.timeseries_options.block_shape, (256, 256));
    assert_eq!(c.output_options.strides.y, 1);
    assert_eq!(c.output_options.strides.x, 1);
    assert_eq!(c.output_options.overview_levels, vec![4, 8, 16, 32, 64]);
    // Deliberate divergence from dolphin's `snaphu` default: dolphinRust defaults
    // to the in-process clean-room Native unwrapper (SNAPHU-parity <=0.5%,
    // IP-clean). `unwrap_method: snaphu` restores the dolphin backend.
    assert_eq!(c.unwrap_options.unwrap_method, UnwrapMethod::Native);
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
        serde_yaml::to_string(&UnwrapMethod::Tophu).unwrap().trim(),
        "tophu"
    );
    assert_eq!(
        serde_yaml::to_string(&UnwrapMethod::Native).unwrap().trim(),
        "native"
    );
    assert_eq!(
        serde_yaml::from_str::<UnwrapMethod>("native").unwrap(),
        UnwrapMethod::Native
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

/// Forward divergence: `input_options.input_type` is dolphinRust-only. A legacy
/// dolphin YAML (no `input_type`) must default to OPERA CSLC; a NISAR config must
/// parse the `nisar_gslc` value and round-trip. The enum string is `snake_case`.
#[test]
fn nisar_input_type_round_trips_and_defaults_to_opera() {
    assert_eq!(
        DisplacementWorkflow::default().input_options.input_type,
        InputType::OperaCslc
    );
    assert_eq!(
        serde_yaml::to_string(&InputType::NisarGslc).unwrap().trim(),
        "nisar_gslc"
    );

    // Legacy dolphin YAML with no input_type → OPERA default (round-trips).
    let legacy = DisplacementWorkflow::from_yaml("cslc_file_list: [/data/a.h5]").unwrap();
    assert_eq!(legacy.input_options.input_type, InputType::OperaCslc);

    // A NISAR config parses and round-trips with the L-band wavelength + grid path.
    let yaml = r#"
input_options:
  input_type: nisar_gslc
  subdataset: /science/LSAR/GSLC/grids/frequencyA/HH
  wavelength: 0.238403545
cslc_file_list: [/data/nisar_20240601.h5]
"#;
    let c = DisplacementWorkflow::from_yaml(yaml).unwrap();
    assert_eq!(c.input_options.input_type, InputType::NisarGslc);
    assert_eq!(
        c.input_options.subdataset.as_deref(),
        Some("/science/LSAR/GSLC/grids/frequencyA/HH")
    );
    assert_eq!(c.input_options.wavelength, Some(0.238_403_545));
    let reparsed = DisplacementWorkflow::from_yaml(&c.to_yaml().unwrap()).unwrap();
    assert_eq!(reparsed, c, "NISAR config round-trips");
}

/// A dolphin-style YAML: partial fields set, plus the `tophu_options` solver
/// block. Modeled fields (now including `tophu_options`) must parse; unspecified
/// fields must take dolphin defaults; the YAML must round-trip.
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
  write_crlb: false
  write_closure_phase: true
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
    assert!(
        !c.phase_linking.write_crlb,
        "explicit write_crlb: false parsed"
    );
    assert!(
        c.phase_linking.write_closure_phase,
        "write_closure_phase: true parsed"
    );
    assert_eq!(c.unwrap_options.unwrap_method, UnwrapMethod::Spurt);
    assert_eq!(c.unwrap_options.tophu_options.ntiles, (2, 2));
    assert_eq!(c.unwrap_options.tophu_options.downsample_factor, (3, 3));

    // The quality flags survive a serialize → parse round-trip.
    let reparsed = DisplacementWorkflow::from_yaml(&c.to_yaml().unwrap()).unwrap();
    assert_eq!(
        reparsed, c,
        "real dolphin YAML round-trips with quality flags"
    );

    // Unspecified fields fall back to dolphin defaults.
    assert_eq!(c.phase_linking.max_num_compressed, 10);
    assert_eq!(c.ps_options.amp_dispersion_threshold, 0.25);
    assert_eq!(c.timeseries_options.method, TimeseriesMethod::L1);
    assert_eq!(c.unwrap_options.snaphu_options.init_method, "mcf");
}

/// A dolphin `correction_options` block (dolphin's `ionosphere_files` /
/// `geometry_files` / `dem_file` names) deserializes into [`CorrectionOptions`],
/// and corrections are off by default. The dolphinRust-only `troposphere_files`
/// forward divergence parses alongside.
#[test]
fn dolphin_correction_options_round_trips() {
    // Default: corrections disabled (no files).
    let bare = DisplacementWorkflow::default();
    assert!(!bare.correction_options.is_enabled(), "off by default");
    assert_eq!(bare.correction_options.incidence_angle_deg, 37.0);

    let yaml = r#"
work_directory: /work/run1
correction_options:
  ionosphere_files:
    - /aux/jplg0010.23i
    - /aux/jplg0130.23i
  geometry_files:
    - /aux/los_east.tif
  dem_file: /aux/dem.tif
  troposphere_files:
    - /aux/l4_20230101.nc
    - /aux/l4_20230113.nc
"#;
    let c = DisplacementWorkflow::from_yaml(yaml).unwrap();
    assert_eq!(c.correction_options.ionosphere_files.len(), 2);
    assert_eq!(c.correction_options.geometry_files.len(), 1);
    assert_eq!(
        c.correction_options.dem_file.as_deref(),
        Some(std::path::Path::new("/aux/dem.tif"))
    );
    assert_eq!(c.correction_options.troposphere_files.len(), 2);
    assert!(c.correction_options.is_enabled(), "files → enabled");
    // Round-trips through serialize → parse.
    let reparsed = DisplacementWorkflow::from_yaml(&c.to_yaml().unwrap()).unwrap();
    assert_eq!(reparsed, c, "correction_options round-trips");
}
