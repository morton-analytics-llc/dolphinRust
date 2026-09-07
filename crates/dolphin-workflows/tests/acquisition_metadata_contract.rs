use chrono::{TimeZone, Utc};
use dolphin_core::config::{AcquisitionMetadata, DisplacementWorkflow, InputType};
use dolphin_workflows::burst::workflow_groups;
use dolphin_workflows::dates::acquisition_days;

#[test]
fn explicit_identity_and_utc_override_opaque_filenames_and_reject_mixed_nisar_frames() {
    let mut cfg = DisplacementWorkflow {
        cslc_file_list: vec!["opaque-a.h5".into(), "opaque-b.h5".into()],
        ..Default::default()
    };
    cfg.input_options.input_type = InputType::NisarGslc;
    cfg.input_options.acquisition_metadata = cfg
        .cslc_file_list
        .iter()
        .enumerate()
        .map(|(i, path)| AcquisitionMetadata {
            path: path.clone(),
            acquisition_utc: Utc
                .with_ymd_and_hms(2026, 1, 1 + 12 * i as u32, 6 + i as u32, 0, 0)
                .unwrap(),
            spatial_group: "track-1/frame-2/A/HH".into(),
            grid_id: "epsg32611/30m/grid-2".into(),
        })
        .collect();
    assert_eq!(workflow_groups(&cfg).unwrap().len(), 1);
    let days = acquisition_days(&cfg.cslc_file_list, &cfg.input_options).unwrap();
    assert!((days[1] - (12.0 + 1.0 / 24.0)).abs() < 1e-12);
    cfg.input_options.acquisition_metadata[1].spatial_group = "track-1/frame-3/A/HH".into();
    assert!(workflow_groups(&cfg)
        .unwrap_err()
        .to_string()
        .contains("one spatial group"));
    cfg.input_options.acquisition_metadata[1].spatial_group = "track-1/frame-2/A/HH".into();
    cfg.input_options.acquisition_metadata[1].grid_id = "another-grid".into();
    assert!(workflow_groups(&cfg)
        .unwrap_err()
        .to_string()
        .contains("grid"));
    cfg.input_options.acquisition_metadata.pop();
    assert!(workflow_groups(&cfg).is_err());
}

#[test]
fn sentinel_bursts_share_pass_dates_without_discarding_their_utc() {
    let mut cfg = DisplacementWorkflow::default();
    for group in ["burst-b", "burst-a"] {
        for day in [1, 13] {
            let path = std::path::PathBuf::from(format!("opaque-{group}-{day}.h5"));
            cfg.input_options
                .acquisition_metadata
                .push(AcquisitionMetadata {
                    path: path.clone(),
                    acquisition_utc: Utc
                        .with_ymd_and_hms(2026, 1, day, 6, 0, u32::from(group == "burst-b"))
                        .unwrap(),
                    spatial_group: group.into(),
                    grid_id: group.into(),
                });
            cfg.cslc_file_list.push(path);
        }
    }
    let groups = workflow_groups(&cfg).unwrap();
    assert_eq!(groups.keys().next().unwrap(), "burst-a");
    cfg.correction_options.solid_earth_tide = true;
    assert!(workflow_groups(&cfg)
        .unwrap_err()
        .to_string()
        .contains("identical UTC"));
    cfg.correction_options.solid_earth_tide = false;
    cfg.correction_options.ionosphere_files = vec!["ionex.dat".into()];
    assert!(workflow_groups(&cfg)
        .unwrap_err()
        .to_string()
        .contains("identical UTC"));
    cfg.correction_options.ionosphere_files.clear();
    cfg.input_options.acquisition_metadata[3].acquisition_utc =
        cfg.input_options.acquisition_metadata[2].acquisition_utc;
    assert!(workflow_groups(&cfg).is_err());
}
