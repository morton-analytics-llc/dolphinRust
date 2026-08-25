use std::path::{Path, PathBuf};
use std::sync::Mutex;

use dolphin_core::config::{EmpiricalSourceFactorOptions, InputType};
use dolphin_core::{Cf32, HalfWindow};
use dolphin_io::CovarianceOperatorGrid;
use dolphin_workflows::{
    sequential_source_model_identity_digest, CslcCovarianceManifest, CSLC_COVARIANCE_SOURCE_MODEL,
    CSLC_COVARIANCE_SOURCE_MODEL_VERSION, CSLC_COVARIANCE_SOURCE_PROVIDER,
    CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
};
use ndarray::Array2;

static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn write_member(path: &Path, date: usize, changed: bool) {
    let _ = std::fs::remove_file(path);
    let values = Array2::from_shape_fn((3, 4), |(row, col)| {
        let mutation = if changed && (row, col) == (1, 2) {
            5.0
        } else {
            0.0
        };
        Cf32::new(
            1.0 + date as f32 * 0.2 + row as f32 * 0.01 + mutation,
            0.5 + col as f32 * 0.03,
        )
    });
    let file = hdf5::File::create(path).unwrap();
    file.new_dataset_builder()
        .with_data(&values)
        .create("data")
        .unwrap();
}

fn paths(root: &Path, count: usize) -> Vec<PathBuf> {
    (0..count)
        .map(|date| root.join(format!("source_{date}.h5")))
        .collect()
}

#[test]
fn verified_prefix_extension_preserves_generation_receipt_and_binds_resolver() {
    let _hdf5 = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = std::env::temp_dir().join(format!("dolphin-cslc-nrt-prefix-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let all_paths = paths(&root, 3);
    for (date, path) in all_paths.iter().enumerate() {
        write_member(path, date, false);
    }
    let parent =
        CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &all_paths[..2]).unwrap();
    let sealed_receipt = parent
        .generation_member_manifest_digest(&[0, 1], "burst", 0)
        .unwrap();
    let extended = parent.verify_prefix_and_extend(&all_paths).unwrap();
    assert_ne!(extended.digest(), parent.digest());
    assert_eq!(
        extended
            .generation_member_manifest_digest(&[0, 1], "burst", 0)
            .unwrap(),
        sealed_receipt
    );

    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let resolver = extended
        .resolver_for_generation(
            &[0, 1],
            &[0, 1],
            "burst",
            0,
            (0, 0),
            (3, 4),
            CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: 3,
                cols: 4,
                stride_y: 1,
                stride_x: 1,
            },
            &EmpiricalSourceFactorOptions {
                half_window: HalfWindow { y: 1, x: 1 },
                shrinkage_alpha: 0.2,
                relative_diagonal_floor: 1e-8,
            },
            model_version,
            None,
        )
        .unwrap();
    assert_eq!(resolver.generation_manifest_digest(), sealed_receipt);
    assert_eq!(resolver.full_revision_manifest_digest(), extended.digest());
    let tail_receipt = extended
        .generation_member_manifest_digest(&[2], "burst", 1)
        .unwrap();
    let tail = extended
        .resolver_for_generation(
            &[0, 1, 2],
            &[2],
            "burst",
            1,
            (0, 0),
            (3, 4),
            CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: 3,
                cols: 4,
                stride_y: 1,
                stride_x: 1,
            },
            &EmpiricalSourceFactorOptions {
                half_window: HalfWindow { y: 1, x: 1 },
                shrinkage_alpha: 0.2,
                relative_diagonal_floor: 1e-8,
            },
            model_version,
            None,
        )
        .unwrap();
    assert_eq!(tail.generation_manifest_digest(), tail_receipt);
    assert_eq!(tail.full_revision_manifest_digest(), extended.digest());
    for path in all_paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

#[test]
fn prefix_drift_fails_before_missing_extension_is_read() {
    let _hdf5 = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = std::env::temp_dir().join(format!("dolphin-cslc-nrt-drift-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let prior_paths = paths(&root, 2);
    for (date, path) in prior_paths.iter().enumerate() {
        write_member(path, date, false);
    }
    let parent =
        CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &prior_paths).unwrap();
    let missing_extension = root.join("missing_extension.h5");
    let mut extended_paths = prior_paths.clone();
    extended_paths.push(missing_extension);
    write_member(&prior_paths[0], 0, true);
    let error = parent
        .verify_prefix_and_extend(&extended_paths)
        .unwrap_err()
        .to_string();
    assert!(error.contains("CSLC member"), "{error}");
    assert!(!error.contains("missing_extension"), "{error}");

    let reordered = vec![prior_paths[1].clone(), prior_paths[0].clone()];
    let error = parent
        .verify_prefix_and_extend(&reordered)
        .unwrap_err()
        .to_string();
    assert!(error.contains("ordered prefix"), "{error}");
    let error = parent
        .verify_prefix_and_extend(&prior_paths[..1])
        .unwrap_err()
        .to_string();
    assert!(error.contains("ordered prefix"), "{error}");
    let substituted = vec![root.join("substituted.h5"), prior_paths[1].clone()];
    let error = parent
        .verify_prefix_and_extend(&substituted)
        .unwrap_err()
        .to_string();
    assert!(error.contains("ordered prefix"), "{error}");
    for path in prior_paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}
