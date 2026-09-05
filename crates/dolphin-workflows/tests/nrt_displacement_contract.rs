//! Phase 2b (v1.4.0) end-to-end NRT front-door contract.
//!
//! An incremental displacement update — `run_displacement_resumable` on an
//! initial window then `update_displacement` folding in the later acquisitions —
//! must reproduce a full `run_displacement` of the extended stack. Phase-linking
//! is bit-identical (Phase 2) and the downstream is the shared deterministic
//! tail, so the whole displacement product matches. Skips without the disp
//! fixture or `snaphu`.

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_io::{read_cslc_shape, read_geotransform, write_raster};
use dolphin_workflows::{run_displacement, run_displacement_resumable, update_displacement};
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

fn georeferenced_config(label: &str) -> DisplacementWorkflow {
    let source = fixtures().join("disp/config.yaml");
    let mut cfg =
        DisplacementWorkflow::from_yaml(&std::fs::read_to_string(source).unwrap()).unwrap();
    let dir = std::env::temp_dir().join(format!("dolphin_nrt_georef_{label}"));
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

fn configure_layover_shadow_mask(
    cfg: &mut DisplacementWorkflow,
    label: &str,
) -> (PathBuf, Array2<u8>) {
    let first_cslc = &cfg.cslc_file_list[0];
    let subdataset = cfg.input_options.subdataset.as_deref().unwrap();
    let shape = read_cslc_shape(first_cslc, subdataset).unwrap();
    let geo = read_geotransform(first_cslc, subdataset).unwrap();
    let mut values = Array2::from_elem(shape, 1_u8);
    values
        .slice_mut(ndarray::s![shape.0 - 3.., shape.1 - 3..])
        .fill(0);
    let path = cfg
        .work_directory
        .join(format!("layover_shadow_mask_{label}.tif"));
    std::fs::create_dir_all(&cfg.work_directory).unwrap();
    if path.exists() {
        std::fs::remove_file(&path).unwrap();
    }
    write_raster(
        &path,
        values.view(),
        geo.geotransform,
        Some(geo.epsg),
        Some(0.0),
    )
    .unwrap();
    cfg.layover_shadow_mask_files = vec![path.clone()];
    (path, values)
}

fn max3(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "layer shape mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| match (x.is_finite(), y.is_finite()) {
            (true, true) => (x - y).abs(),
            (false, false) => 0.0,
            _ => panic!("layer finite/non-finite pattern mismatch: {x} vs {y}"),
        })
        .fold(0.0, f64::max)
}

fn max2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "layer shape mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| match (x.is_finite(), y.is_finite()) {
            (true, true) => (x - y).abs(),
            (false, false) => 0.0,
            _ => panic!("layer finite/non-finite pattern mismatch: {x} vs {y}"),
        })
        .fold(0.0, f64::max)
}

/// Incremental displacement (3 dates, then fold in the remaining 2) equals a full
/// run of all 5. `ministack_size = 2` so the open trailing ministack seals during
/// the update and a new ministack opens — exercising the carry across a boundary.
#[test]
fn incremental_displacement_matches_full_run() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !config.exists() || !snaphu_available() {
        eprintln!("skipping NRT displacement contract: no fixtures / snaphu");
        return;
    }
    let mut base = georeferenced_config("incremental");
    assert!(base.cslc_file_list.len() >= 5, "fixture needs >=5 dates");
    let (mask_path, mask_values) = configure_layover_shadow_mask(&mut base, "incremental");

    // Full run of the extended stack.
    let mut full_cfg = base.clone();
    full_cfg.phase_linking.ministack_size = 2;
    full_cfg.work_directory = std::env::temp_dir().join("dolphinrust_nrt_full");
    let full = run_displacement(&full_cfg).unwrap();

    // Incremental: initial 3 dates, then fold in dates 4 and 5 together.
    let mut init_cfg = full_cfg.clone();
    init_cfg.work_directory = std::env::temp_dir().join("dolphinrust_nrt_inc");
    init_cfg.cslc_file_list = base.cslc_file_list[..3].to_vec();
    let (_out_init, state) = run_displacement_resumable(&init_cfg).unwrap();

    let mut ext_cfg = init_cfg.clone();
    ext_cfg.cslc_file_list = base.cslc_file_list.clone();
    let (inc, _state2) = update_displacement(&state, &ext_cfg).unwrap();

    let dd = max3(&inc.displacement, &full.displacement);
    let dv = max2(&inc.velocity, &full.velocity);
    let dt = max2(&inc.temporal_coherence, &full.temporal_coherence);
    let dc = match (&inc.crlb_sigma, &full.crlb_sigma) {
        (Some(a), Some(b)) => max3(a, b),
        _ => 0.0,
    };
    eprintln!("incremental vs full displacement: ddisp={dd:.2e} dvel={dv:.2e} dtcoh={dt:.2e} dcrlb={dc:.2e}");
    assert_eq!(inc.acquisition_days, full.acquisition_days, "dates match");
    assert_eq!(
        inc.interferogram_pairs, full.interferogram_pairs,
        "pair order"
    );
    assert_eq!(
        inc.unwrap_connected_components, full.unwrap_connected_components,
        "unwrap component labels"
    );
    assert_eq!(
        inc.reference_point, full.reference_point,
        "ref point matches"
    );
    assert_eq!(inc.validity_mask, full.validity_mask, "validity mask");
    assert!(
        full.validity_mask.iter().any(|valid| !*valid),
        "configured layover/shadow mask must invalidate output pixels"
    );
    // Phase-linking is bit-identical and the downstream is deterministic (same
    // SNAPHU input → same output), so the products match to f64 round-off.
    assert!(dd < 1e-6, "displacement max|Δ| {dd}");
    assert!(dv < 1e-6, "velocity max|Δ| {dv}");
    assert!(dt < 1e-6, "temporal coherence max|Δ| {dt}");
    assert!(dc < 1e-6, "crlb max|Δ| {dc}");

    // The persisted NRT state binds the native mask's path and valid-pixel
    // content. Rewrite the same path, then append a nonexistent acquisition:
    // the mask mutation must fail before the new CSLC is read.
    let mut changed_mask = mask_values;
    let last = (changed_mask.dim().0 - 1, changed_mask.dim().1 - 1);
    changed_mask[last] = 1;
    std::fs::remove_file(&mask_path).unwrap();
    let geo = read_geotransform(
        &init_cfg.cslc_file_list[0],
        init_cfg.input_options.subdataset.as_deref().unwrap(),
    )
    .unwrap();
    write_raster(
        &mask_path,
        changed_mask.view(),
        geo.geotransform,
        Some(geo.epsg),
        Some(0.0),
    )
    .unwrap();
    let mut changed_cfg = init_cfg.clone();
    changed_cfg
        .cslc_file_list
        .push(changed_cfg.work_directory.join("cslc_20230118.h5"));
    assert!(!changed_cfg.cslc_file_list.last().unwrap().exists());
    let err = match update_displacement(&state, &changed_cfg) {
        Ok(_) => panic!("expected a changed layover/shadow mask to be rejected"),
        Err(error) => error,
    };
    assert!(
        err.to_string().contains("valid-pixel content changed"),
        "expected mask-content error before reading the missing CSLC, got: {err}"
    );
}

/// An update that extends no burst (same file list) is rejected, not silently a
/// no-op — guards the "every update extends every burst" contract.
#[test]
fn update_without_new_acquisitions_errors() {
    let dir = fixtures();
    let config = dir.join("disp/config.yaml");
    if !config.exists() || !snaphu_available() {
        eprintln!("skipping NRT no-op guard: no fixtures / snaphu");
        return;
    }
    let mut cfg = georeferenced_config("noop");
    cfg.phase_linking.ministack_size = 2;
    cfg.work_directory = std::env::temp_dir().join("dolphinrust_nrt_noop");
    configure_layover_shadow_mask(&mut cfg, "noop");
    cfg.cslc_file_list = cfg.cslc_file_list[..3].to_vec();
    let (_out, state) = run_displacement_resumable(&cfg).unwrap();
    let err = match update_displacement(&state, &cfg) {
        Ok(_) => panic!("expected an error when no burst is extended"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("no new acquisitions"),
        "expected no-new-acquisitions error, got: {err}"
    );
}
