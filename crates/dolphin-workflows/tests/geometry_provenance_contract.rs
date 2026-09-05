//! Geometry-provenance contract tests (dolphinRust #1 / eo #120).
//!
//! Oracle constants come from `validation/make_geomprov_fixture.py` (independent
//! Python derivation on the real OPERA_L2_CSLC-S1_T144-308011-IW2 granules), which
//! also writes the committed fixtures `oracle/fixtures/geomprov_ci_*.h5`:
//!   heading (orbit ENU)        = 189.981317°
//!   heading (LOS-implied)      = 190.0904°
//!   native_range_spacing_m     = 2.329562114715323 (pure read)
//!   native_azimuth_spacing_m   = 14.063791 m
//!   incidence mean, 64×64 crop = 39.329207°

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use dolphin_core::config::{DisplacementWorkflow, InputType};
use dolphin_corrections::geometry::resolve_los_geometry;
use dolphin_corrections::LosGeometry;
use dolphin_workflows::provenance::{
    assemble_geometry_provenance, assemble_geometry_provenance_with_coverage,
    BurstCoverageProvenance, FieldProvenance, GeometryProvenance, InputCoverageProvenance,
    GEOMETRY_PROVENANCE_FILENAME, INPUT_COVERAGE_POLICY_VERSION,
};
use ndarray::Array2;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

/// Serialize HDF5 access across this binary's parallel tests (`hdf5-metno` links
/// a non-thread-safe HDF5; dolphin-io's guard is `pub(crate)`, so cross-crate test
/// binaries carry their own).
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn hdf5_guard() -> MutexGuard<'static, ()> {
    HDF5_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

fn cfg_with_inputs(files: &[PathBuf]) -> DisplacementWorkflow {
    DisplacementWorkflow {
        cslc_file_list: files.to_vec(),
        ..Default::default()
    }
}

fn orbit_type_fixture(value: Option<&str>, suffix: &str) -> PathBuf {
    let source = fixtures().join("geomprov_ci_cslc.h5");
    let target = std::env::temp_dir().join(format!("dolphin_geomprov_orbit_type_{suffix}.h5"));
    let _ = std::fs::remove_file(&target);
    std::fs::copy(source, &target).unwrap();
    let file = hdf5::File::open_rw(&target).unwrap();
    let orbit = file.group("metadata/orbit").unwrap();
    match value {
        Some(value) => orbit
            .dataset("orbit_type")
            .unwrap()
            .write_scalar(&hdf5::types::FixedAscii::<64>::from_ascii(value).unwrap())
            .unwrap(),
        None => orbit.unlink("orbit_type").unwrap(),
    }
    target
}

/// Config whose `geometry_files` mirror the run invariant that a resolved
/// [`LosGeometry`] always came from supplied CSLC-S1-STATIC granules.
fn cfg_with_static(static_file: &Path) -> DisplacementWorkflow {
    let mut cfg = cfg_with_inputs(&[fixtures().join("geomprov_ci_cslc.h5")]);
    cfg.correction_options.geometry_files = vec![static_file.to_path_buf()];
    cfg
}

fn sourced(prov: &GeometryProvenance, field: &str) -> bool {
    matches!(
        prov.geometry_provenance.fields.get(field),
        Some(FieldProvenance::Sourced { .. })
    )
}

fn absent_reason<'a>(prov: &'a GeometryProvenance, field: &str) -> &'a str {
    match prov.geometry_provenance.fields.get(field) {
        Some(FieldProvenance::Absent { reason }) => reason,
        other => panic!("{field}: expected absent, got {other:?}"),
    }
}

/// Contract 1: the real output-metadata sample maps to the exported fields, with
/// exact source keys recorded.
#[test]
fn real_metadata_sample_maps_to_exported_fields() {
    let _hdf5 = hdf5_guard();
    let cfg = cfg_with_inputs(&[fixtures().join("geomprov_ci_cslc.h5")]);
    let prov = assemble_geometry_provenance(&cfg, None);

    assert_eq!(prov.orbit_direction.as_deref(), Some("descending"));
    let Some(FieldProvenance::Sourced {
        source_keys,
        raw_value,
        ..
    }) = prov.geometry_provenance.fields.get("orbit_direction")
    else {
        panic!("orbit_direction not sourced");
    };
    assert!(
        source_keys.contains(&"/identification/orbit_pass_direction".to_string()),
        "orbit_direction source key: {source_keys:?}"
    );
    assert_eq!(raw_value.as_deref(), Some("Descending"));

    let heading = prov.heading_deg.expect("heading sourced");
    assert!(
        (heading - 189.981_317).abs() < 0.1,
        "heading {heading} vs oracle 189.981317"
    );
    assert_eq!(prov.native_range_spacing_m, Some(2.329_562_114_715_323));
    assert_eq!(prov.orbit_ephemeris_class.as_deref(), Some("precise"));
    let Some(FieldProvenance::Sourced {
        source_keys,
        raw_value,
        ..
    }) = prov.geometry_provenance.fields.get("orbit_ephemeris_class")
    else {
        panic!("orbit_ephemeris_class not sourced");
    };
    assert_eq!(raw_value.as_deref(), Some("POEORB"));
    assert_eq!(source_keys, &["/metadata/orbit/orbit_type"]);
    let decoded: GeometryProvenance =
        serde_json::from_str(&serde_json::to_string(&prov).unwrap()).unwrap();
    assert_eq!(decoded.schema, "dolphinrust-geometry-provenance/4");
    assert_eq!(decoded.orbit_ephemeris_class.as_deref(), Some("precise"));
    assert!(matches!(
        decoded.geometry_provenance.fields.get("orbit_ephemeris_class"),
        Some(FieldProvenance::Sourced { source_keys, .. })
            if source_keys == &["/metadata/orbit/orbit_type"]
    ));
    let az = prov
        .native_azimuth_spacing_m
        .expect("azimuth spacing sourced");
    assert!(
        (az - 14.063_791).abs() < 0.02,
        "azimuth spacing {az} vs oracle 14.063791"
    );
    assert!(sourced(&prov, "heading_deg"));
    assert!(sourced(&prov, "native_range_spacing_m"));
    assert!(sourced(&prov, "native_azimuth_spacing_m"));

    // No STATIC geometry supplied: incidence is explicitly absent, so the
    // decomposition-safety bit must be off even with everything else sourced.
    assert_eq!(prov.incidence_angle_deg, None);
    absent_reason(&prov, "incidence_angle_deg");
    assert!(!prov.decomposition_geometry_complete);
}

#[test]
fn phase_linking_coherence_provenance_is_optional_and_never_aliases_temporal() {
    let mut cfg = DisplacementWorkflow::default();
    let absent = assemble_geometry_provenance(&cfg, None);
    assert_eq!(absent.phase_linking_coherence, None);

    cfg.phase_linking.calc_average_coh = true;
    let present = assemble_geometry_provenance(&cfg, None);
    assert_eq!(
        present.phase_linking_coherence.as_deref(),
        Some("phase_linking_coherence.tif")
    );
    assert_ne!(
        present.phase_linking_coherence.as_deref(),
        Some("temporal_coherence.tif")
    );
}

#[test]
fn coverage_v3_round_trips_without_identifiers() {
    let coverage = InputCoverageProvenance {
        policy_version: INPUT_COVERAGE_POLICY_VERSION.into(),
        total_tiles: 3,
        linked_tiles: 2,
        nodata_tiles: 1,
        bursts: vec![BurstCoverageProvenance {
            burst_index: 0,
            acquisition_count: 4,
            total_tiles: 3,
            linked_tiles: 2,
            nodata_tiles: 1,
        }],
        output_pixels: 48,
        valid_pixels: 32,
        valid_fraction: 2.0 / 3.0,
    };
    let provenance = assemble_geometry_provenance_with_coverage(
        &DisplacementWorkflow::default(),
        None,
        None,
        Some(coverage.clone()),
    );
    let json = serde_json::to_string(&provenance).unwrap();
    let decoded: GeometryProvenance = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.schema, "dolphinrust-geometry-provenance/4");
    assert_eq!(decoded.input_coverage, Some(coverage));
    for forbidden in [
        "source_path",
        "object_key",
        "aoi_geometry",
        "acquisition_id",
    ] {
        assert!(!json.contains(forbidden), "coverage leaked {forbidden}");
    }
}

#[test]
fn provenance_v2_without_coverage_still_deserializes() {
    let current = assemble_geometry_provenance(&DisplacementWorkflow::default(), None);
    let mut value = serde_json::to_value(current).unwrap();
    let object = value.as_object_mut().unwrap();
    object.insert(
        "schema".into(),
        serde_json::Value::String("dolphinrust-geometry-provenance/2".into()),
    );
    object.insert(
        "method_version".into(),
        serde_json::Value::String("2.0.0".into()),
    );
    object.remove("input_coverage");

    let decoded: GeometryProvenance = serde_json::from_value(value).unwrap();
    assert_eq!(decoded.schema, "dolphinrust-geometry-provenance/2");
    assert_eq!(decoded.input_coverage, None);
    assert_eq!(decoded.orbit_ephemeris_class, None);
}

#[test]
fn orbit_type_normalizes_resorb_without_changing_geometry() {
    let _hdf5 = hdf5_guard();
    let path = orbit_type_fixture(Some("RESORB"), "resorb");
    let prov = assemble_geometry_provenance(&cfg_with_inputs(std::slice::from_ref(&path)), None);

    assert_eq!(prov.orbit_ephemeris_class.as_deref(), Some("restituted"));
    assert_eq!(prov.orbit_direction.as_deref(), Some("descending"));
    assert!(prov.heading_deg.is_some());
    let _ = std::fs::remove_file(path);
}

#[test]
fn missing_or_unknown_orbit_type_is_absent_only_for_new_field() {
    let _hdf5 = hdf5_guard();
    for (value, suffix, expected_reason) in [
        (None, "missing", "orbit_type"),
        (Some("UNKNOWN"), "unknown", "unrecognized orbit_type"),
    ] {
        let path = orbit_type_fixture(value, suffix);
        let prov =
            assemble_geometry_provenance(&cfg_with_inputs(std::slice::from_ref(&path)), None);
        assert_eq!(prov.orbit_ephemeris_class, None);
        assert!(absent_reason(&prov, "orbit_ephemeris_class").contains(expected_reason));
        assert_eq!(prov.orbit_direction.as_deref(), Some("descending"));
        assert!(prov.heading_deg.is_some());
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn mixed_orbit_types_are_absent_without_erasing_geometry() {
    let _hdf5 = hdf5_guard();
    let precise = orbit_type_fixture(Some("POEORB"), "mixed_precise");
    let restituted = orbit_type_fixture(Some("RESORB"), "mixed_restituted");
    let prov = assemble_geometry_provenance(
        &cfg_with_inputs(&[precise.clone(), restituted.clone()]),
        None,
    );

    assert_eq!(prov.orbit_ephemeris_class, None);
    assert!(absent_reason(&prov, "orbit_ephemeris_class").contains("inconsistent"));
    assert_eq!(prov.orbit_direction.as_deref(), Some("descending"));
    assert!(prov.heading_deg.is_some());
    let _ = std::fs::remove_file(precise);
    let _ = std::fs::remove_file(restituted);
}

#[test]
fn nisar_orbit_type_is_explicitly_absent() {
    let mut cfg = DisplacementWorkflow::default();
    cfg.input_options.input_type = InputType::NisarGslc;
    let prov = assemble_geometry_provenance(&cfg, None);

    assert_eq!(prov.orbit_ephemeris_class, None);
    assert!(absent_reason(&prov, "orbit_ephemeris_class").contains("NISAR"));
}

/// Contract 2a: a /data-only granule (cropped, no metadata groups) yields explicit
/// absence for every geometry field — never a default.
#[test]
fn data_only_granule_is_explicitly_absent() {
    let _hdf5 = hdf5_guard();
    let cfg = cfg_with_inputs(&[fixtures().join("geomprov_ci_data_only.h5")]);
    let prov = assemble_geometry_provenance(&cfg, None);

    assert_eq!(prov.orbit_direction, None);
    assert_eq!(prov.orbit_ephemeris_class, None);
    assert_eq!(prov.incidence_angle_deg, None);
    assert_eq!(prov.heading_deg, None);
    assert_eq!(prov.native_range_spacing_m, None);
    assert_eq!(prov.native_azimuth_spacing_m, None);
    assert_eq!(prov.acquisition_time_of_day_utc_s, None);
    assert!(!prov.decomposition_geometry_complete);
    for field in [
        "orbit_direction",
        "orbit_ephemeris_class",
        "incidence_angle_deg",
        "heading_deg",
        "native_range_spacing_m",
        "native_azimuth_spacing_m",
        "acquisition_time_of_day_utc_s",
    ] {
        assert!(
            !absent_reason(&prov, field).is_empty(),
            "{field} carries a reason"
        );
    }
}

/// Contract 2b (adversarial): the atmospheric-projection scalar knob
/// (`correction_options.incidence_angle_deg`) can never leak into provenance.
#[test]
fn incidence_knob_never_leaks_into_provenance() {
    let _hdf5 = hdf5_guard();
    let mut cfg = cfg_with_inputs(&[fixtures().join("geomprov_ci_cslc.h5")]);
    cfg.correction_options.incidence_angle_deg = 55.0; // distinctive non-default
    let prov = assemble_geometry_provenance(&cfg, None);

    assert_eq!(
        prov.incidence_angle_deg, None,
        "knob leaked into provenance"
    );
    absent_reason(&prov, "incidence_angle_deg");
}

/// Contract 2c: a mixed stack (one full-metadata granule + one data-only granule)
/// is absent, not sourced-from-the-readable-subset.
#[test]
fn mixed_stack_is_absent_not_partially_sourced() {
    let _hdf5 = hdf5_guard();
    let cfg = cfg_with_inputs(&[
        fixtures().join("geomprov_ci_cslc.h5"),
        fixtures().join("geomprov_ci_data_only.h5"),
    ]);
    let prov = assemble_geometry_provenance(&cfg, None);

    assert_eq!(prov.orbit_direction, None);
    assert_eq!(prov.heading_deg, None);
    assert_eq!(prov.native_range_spacing_m, None);
    assert!(!prov.decomposition_geometry_complete);
}

/// Contract 3a: constant synthetic LOS → exact incidence stats, zero spread, and a
/// fully-sourced provenance flips the decomposition-safety bit on.
#[test]
fn constant_los_yields_exact_incidence_and_complete_geometry() {
    let _hdf5 = hdf5_guard();
    let theta = 34.0_f64.to_radians();
    let (az_sin, az_cos) = 30.0_f64.to_radians().sin_cos();
    let los = LosGeometry {
        east: Array2::from_elem((8, 8), theta.sin() * az_sin),
        north: Array2::from_elem((8, 8), theta.sin() * az_cos),
        up: Array2::from_elem((8, 8), theta.cos()),
    };
    let cfg = cfg_with_static(&fixtures().join("geomprov_ci_static.h5"));
    let prov = assemble_geometry_provenance(&cfg, Some(&los));

    let inc = prov.incidence_angle_deg.expect("incidence sourced");
    assert!((inc - 34.0).abs() < 1e-9, "incidence {inc}");
    assert!(prov.incidence_angle_spread_deg.expect("spread") < 1e-9);
    assert!(sourced(&prov, "incidence_angle_deg"));
    assert!(prov.decomposition_geometry_complete);
}

/// Contract 3b: the real STATIC crop reproduces the Python oracle mean incidence
/// on the identical 64×64 window.
#[test]
fn real_static_crop_matches_incidence_oracle() {
    let _hdf5 = hdf5_guard();
    let layers =
        dolphin_io::read_los_layers(&fixtures().join("geomprov_ci_static.h5"), "/data").unwrap();
    let shape = layers.east.dim();
    let (gt, epsg) = (layers.geo.geotransform, layers.geo.epsg);
    let los = resolve_los_geometry(&[layers], gt, epsg, shape).unwrap();

    let cfg = cfg_with_static(&fixtures().join("geomprov_ci_static.h5"));
    let prov = assemble_geometry_provenance(&cfg, Some(&los));

    let inc = prov.incidence_angle_deg.expect("incidence sourced");
    assert!(
        (inc - 39.329_207).abs() < 0.05,
        "incidence {inc} vs oracle 39.329207 (identical crop)"
    );
    assert!(prov.incidence_angle_min_deg.unwrap() <= inc);
    assert!(prov.incidence_angle_max_deg.unwrap() >= inc);
    assert!(prov.decomposition_geometry_complete);
}

/// Contract 3c: incidence spread beyond the 3° gate (multi-subswath-like mix)
/// keeps the raw stats but turns the decomposition-safety bit off.
#[test]
fn wide_incidence_spread_gates_decomposition() {
    let _hdf5 = hdf5_guard();
    // Two halves at 31° and 45° — spread far beyond any single subswath.
    let mut up = Array2::from_elem((8, 8), 31.0_f64.to_radians().cos());
    up.slice_mut(ndarray::s![4.., ..])
        .fill(45.0_f64.to_radians().cos());
    let east = up.mapv(|u: f64| (1.0 - u * u).sqrt());
    let los = LosGeometry {
        east,
        north: Array2::zeros((8, 8)),
        up,
    };
    let cfg = cfg_with_static(&fixtures().join("geomprov_ci_static.h5"));
    let prov = assemble_geometry_provenance(&cfg, Some(&los));

    assert!(prov.incidence_angle_deg.is_some(), "stats stay populated");
    assert!(prov.incidence_angle_spread_deg.unwrap() > 3.0);
    assert!(
        !prov.decomposition_geometry_complete,
        "spread gate must disable decomposition safety"
    );
}

/// Contract 5: heading from orbit velocity agrees with the heading implied by the
/// real STATIC LOS signs + right-looking geometry to 2° (measured 0.12°; the gate
/// catches a ground-track-convention error at 3.2° and any ±90°/sign mistake).
#[test]
fn heading_cross_derivation_agrees() {
    let _hdf5 = hdf5_guard();
    let cfg = cfg_with_inputs(&[fixtures().join("geomprov_ci_cslc.h5")]);
    let prov = assemble_geometry_provenance(&cfg, None);
    let heading = prov.heading_deg.expect("heading sourced");

    let layers =
        dolphin_io::read_los_layers(&fixtures().join("geomprov_ci_static.h5"), "/data").unwrap();
    let (e, n) = (layers.east.mean().unwrap(), layers.north.mean().unwrap());
    let los_heading = (e.atan2(n).to_degrees() + 90.0).rem_euclid(360.0);

    let delta = (heading - los_heading + 540.0).rem_euclid(360.0) - 180.0;
    assert!(
        delta.abs() < 2.0,
        "orbit heading {heading} vs LOS-implied {los_heading}"
    );
}

/// Contract 4: the run emits `geometry_provenance.json` next to the rasters; it
/// parses, mirrors `DisplacementOutput.geometry_provenance`, and names the
/// phase-linking coherence artifact. Data-only inputs → all-absent, explicitly.
#[test]
fn run_writes_geometry_provenance_artifact() {
    let _hdf5 = hdf5_guard();
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !config.exists() {
        eprintln!("skipping artifact e2e: no disp fixtures");
        return;
    }
    let snaphu = std::process::Command::new("snaphu").arg("--help").output();
    if snaphu.is_err() {
        eprintln!("skipping artifact e2e: snaphu not on PATH");
        return;
    }

    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(&config).unwrap()).unwrap();
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_geomprov_e2e");
    std::fs::create_dir_all(&cfg.work_directory).unwrap();
    cfg.cslc_file_list = cfg
        .cslc_file_list
        .iter()
        .map(|source| {
            let target = cfg.work_directory.join(source.file_name().unwrap());
            std::fs::copy(source, &target).unwrap();
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
    let out = dolphin_workflows::run_displacement(&cfg).unwrap();

    let artifact = cfg.work_directory.join(GEOMETRY_PROVENANCE_FILENAME);
    let parsed: GeometryProvenance =
        serde_json::from_str(&std::fs::read_to_string(&artifact).unwrap()).unwrap();

    assert_eq!(parsed.phase_linking_coherence, None);
    assert_eq!(parsed.orbit_direction, None, "disp fixtures are data-only");
    assert_eq!(parsed.incidence_angle_deg, None);
    assert_eq!(parsed.heading_deg, None);
    assert_eq!(parsed.native_range_spacing_m, None);
    assert_eq!(parsed.native_azimuth_spacing_m, None);
    assert_eq!(parsed.acquisition_time_of_day_utc_s, None);
    assert!(!parsed.decomposition_geometry_complete);
    assert_eq!(
        parsed.orbit_direction,
        out.geometry_provenance.orbit_direction
    );
    assert_eq!(
        parsed.decomposition_geometry_complete,
        out.geometry_provenance.decomposition_geometry_complete
    );
}

/// A STATIC granule from the wrong pass must not source incidence: `up` is
/// sign-insensitive, so a wrong-pass LOS yields perfectly plausible incidence —
/// the identity cross-check is the only guard.
#[test]
fn wrong_pass_static_is_absent_not_sourced() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join("geomprov_wrong_pass_static.h5");
    let _ = std::fs::remove_file(&path);
    // The CSLC stack is Descending.
    write_static_identification(&path, "Ascending", "t144_308011_iw2");

    let cfg = cfg_with_static(&path);
    let prov = assemble_geometry_provenance(&cfg, Some(&plausible_los()));

    assert_eq!(prov.incidence_angle_deg, None, "wrong-pass STATIC sourced");
    assert!(
        absent_reason(&prov, "incidence_angle_deg").contains("orbit_pass_direction"),
        "reason names the mismatch"
    );
    assert!(!prov.decomposition_geometry_complete);
    let _ = std::fs::remove_file(&path);
}

/// #39: the along-track neighbour STATIC is the granule the LOS mosaic needs when
/// the frame extends past the processed burst's valid LOS. Its burst is by
/// definition not in the CSLC stack, so requiring burst identity silently dropped
/// LOS to the scalar-incidence fallback. Same track + pass must be accepted.
#[test]
fn same_track_neighbour_burst_static_is_sourced() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join("geomprov_neighbour_burst_static.h5");
    let _ = std::fs::remove_file(&path);
    // Stack burst is t144_308011_iw2; this is the along-track neighbour.
    write_static_identification(&path, "Descending", "t144_308012_iw2");

    let cfg = cfg_with_static(&path);
    let prov = assemble_geometry_provenance(&cfg, Some(&plausible_los()));

    assert!(
        prov.incidence_angle_deg.is_some(),
        "same-track neighbour STATIC not sourced: {}",
        absent_reason(&prov, "incidence_angle_deg")
    );
    assert!(sourced(&prov, "incidence_angle_deg"));
    let _ = std::fs::remove_file(&path);
}

/// The relaxation is track-scoped, not a removal: a STATIC from a different track
/// that happens to share the pass direction is still rejected. The GNSS-fixture
/// frame intersects T041/T078/T143, whose LOS would be wrong to mosaic.
#[test]
fn wrong_track_static_is_absent_not_sourced() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join("geomprov_wrong_track_static.h5");
    let _ = std::fs::remove_file(&path);
    write_static_identification(&path, "Descending", "t041_308011_iw2");

    let cfg = cfg_with_static(&path);
    let prov = assemble_geometry_provenance(&cfg, Some(&plausible_los()));

    assert_eq!(prov.incidence_angle_deg, None, "wrong-track STATIC sourced");
    assert!(
        absent_reason(&prov, "incidence_angle_deg").contains("t041"),
        "reason names the offending track"
    );
    assert!(!prov.decomposition_geometry_complete);
    let _ = std::fs::remove_file(&path);
}

/// A uniform LOS with plausible S1 incidence — the identity cross-check, not the
/// LOS values, is what these tests exercise.
fn plausible_los() -> LosGeometry {
    LosGeometry {
        east: Array2::from_elem((8, 8), 0.62),
        north: Array2::from_elem((8, 8), -0.11),
        up: Array2::from_elem((8, 8), (1.0_f64 - 0.62 * 0.62 - 0.11 * 0.11).sqrt()),
    }
}

/// Minimal STATIC-shaped file carrying only `/identification` (the group the
/// consistency check reads).
fn write_static_identification(path: &Path, orbit_pass: &str, burst_id: &str) {
    let f = hdf5::File::create(path).unwrap();
    let g = f.create_group("identification").unwrap();
    for (key, value) in [
        ("orbit_pass_direction", orbit_pass),
        ("look_direction", "Right"),
        ("burst_id", burst_id),
        ("zero_doppler_start_time", "2014-04-03 00:00:00.000000"),
        ("zero_doppler_end_time", "2014-04-03 00:00:03.000000"),
    ] {
        let ascii = hdf5::types::FixedAscii::<64>::from_ascii(value).unwrap();
        g.new_dataset::<hdf5::types::FixedAscii<64>>()
            .create(key)
            .unwrap()
            .write_scalar(&ascii)
            .unwrap();
    }
}
