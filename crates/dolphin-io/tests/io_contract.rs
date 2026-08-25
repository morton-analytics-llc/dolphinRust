//! Phase-8 (I/O) contract tests.
//!
//! Primary (round-trip): a GeoTIFF written by Rust reads back with identical
//! pixels, geotransform, and EPSG. Secondary (oracle): Rust reads a GDAL-written
//! GeoTIFF and an h5py-written OPERA-style CSLC HDF5 matching the known arrays.
//! Oracle tests skip without fixtures.

use std::path::{Path, PathBuf};

use dolphin_core::{BlockIndices, Cf32};
use dolphin_io::{
    read_cslc, read_cslc_stack, read_raster, read_raster_header, write_raster, BoundedCogWriter,
    RasterData,
};
use gdal::Metadata;
use ndarray::{s, Array2, Array3};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

const GT: [f64; 6] = [500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0];

// ------------------------------- round-trip (primary) -------------------------

#[test]
fn geotiff_f32_round_trips() {
    let dir = std::env::temp_dir().join("dolphinrust_io_rt.tif");
    let data = Array2::from_shape_fn((8, 10), |(r, c)| (r * 10 + c) as f32 * 0.25 - 1.0);
    write_raster(&dir, data.view(), GT, Some(32611), Some(-9999.0)).unwrap();

    let RasterData {
        data: back,
        geotransform,
        epsg,
    } = read_raster::<f32>(&dir).unwrap();
    assert_eq!(back, data, "pixels round-trip");
    assert_eq!(geotransform, GT, "geotransform round-trips");
    assert_eq!(epsg, Some(32611), "EPSG round-trips");
    let _ = std::fs::remove_file(&dir);
}

#[test]
fn geotiff_u8_round_trips() {
    let dir = std::env::temp_dir().join("dolphinrust_io_ps.tif");
    let data = Array2::from_shape_fn((6, 7), |(r, c)| ((r + c) % 3) as u8);
    write_raster(&dir, data.view(), GT, Some(4326), None).unwrap();
    let back = read_raster::<u8>(&dir).unwrap().data;
    assert_eq!(back, data, "uint8 PS mask round-trips");
    let _ = std::fs::remove_file(&dir);
}

#[test]
fn bounded_cog_is_published_only_after_all_window_writes() {
    let root = std::env::temp_dir().join(format!("dolphinrust_bounded_cog_{}", std::process::id()));
    let scratch = root.with_extension("scratch.tif");
    let destination = root.with_extension("tif");
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&destination);
    let expected = Array2::from_shape_fn((5, 7), |(row, col)| (row * 7 + col) as f32);
    let metadata = [("units", "millimeters/year"), ("estimator", "gls-v1")];

    let mut writer = BoundedCogWriter::<f32>::create(
        &scratch,
        (5, 7),
        GT,
        Some(32611),
        Some(-9999.0),
        &metadata,
    )
    .unwrap();
    writer
        .write_window(
            BlockIndices {
                row_start: 0,
                row_stop: 2,
                col_start: 0,
                col_stop: 7,
            },
            expected.slice(s![0..2, ..]),
        )
        .unwrap();
    assert!(scratch.exists());
    assert!(!destination.exists());
    writer
        .write_window(
            BlockIndices {
                row_start: 2,
                row_stop: 5,
                col_start: 0,
                col_stop: 7,
            },
            expected.slice(s![2..5, ..]),
        )
        .unwrap();
    writer.finalize(&destination).unwrap();

    assert_eq!(read_raster::<f32>(&destination).unwrap().data, expected);
    let header = read_raster_header(&destination).unwrap();
    assert_eq!(header.shape, (5, 7));
    assert_eq!(header.geotransform, GT);
    assert_eq!(header.epsg, Some(32611));
    assert_eq!(header.nodata, Some(-9999.0));
    assert_eq!(
        header.metadata.get("units").map(String::as_str),
        Some("millimeters/year")
    );
    assert_eq!(
        header.metadata.get("estimator").map(String::as_str),
        Some("gls-v1")
    );
    let dataset = gdal::Dataset::open(&destination).unwrap();
    assert_eq!(dataset.driver().short_name(), "GTiff");
    assert_eq!(
        dataset
            .metadata_item("LAYOUT", "IMAGE_STRUCTURE")
            .as_deref(),
        Some("COG")
    );
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&destination);
}

#[test]
fn dropping_incomplete_bounded_writer_never_creates_destination() {
    let root =
        std::env::temp_dir().join(format!("dolphinrust_incomplete_cog_{}", std::process::id()));
    let scratch = root.with_extension("scratch.tif");
    let destination = root.with_extension("tif");
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&destination);
    {
        let mut writer =
            BoundedCogWriter::<u8>::create(&scratch, (8, 8), GT, Some(32611), Some(255.0), &[])
                .unwrap();
        let values = Array2::from_elem((2, 2), 1_u8);
        writer
            .write_window(
                BlockIndices {
                    row_start: 0,
                    row_stop: 2,
                    col_start: 0,
                    col_stop: 2,
                },
                values.view(),
            )
            .unwrap();
    }
    assert!(scratch.exists());
    assert!(!destination.exists());
    let _ = std::fs::remove_file(&scratch);
}

#[test]
fn incomplete_bounded_writer_cannot_finalize() {
    let root = std::env::temp_dir().join(format!("dolphinrust_gapped_cog_{}", std::process::id()));
    let scratch = root.with_extension("scratch.tif");
    let destination = root.with_extension("tif");
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&destination);
    let mut writer =
        BoundedCogWriter::<u8>::create(&scratch, (4, 4), GT, Some(32611), Some(255.0), &[])
            .unwrap();
    writer
        .write_window(
            BlockIndices {
                row_start: 0,
                row_stop: 3,
                col_start: 0,
                col_stop: 4,
            },
            Array2::from_elem((3, 4), 1_u8).view(),
        )
        .unwrap();
    assert!(writer.finalize(&destination).is_err());
    assert!(!destination.exists());
    assert!(scratch.exists());
    let _ = std::fs::remove_file(&scratch);
}

#[test]
fn bounded_writer_rejects_overlapping_and_duplicate_windows() {
    let scratch = std::env::temp_dir().join(format!(
        "dolphinrust_overlapping_window_{}.tif",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&scratch);
    let mut writer =
        BoundedCogWriter::<f32>::create(&scratch, (6, 6), GT, Some(32611), None, &[]).unwrap();
    let first = BlockIndices {
        row_start: 0,
        row_stop: 3,
        col_start: 0,
        col_stop: 3,
    };
    writer
        .write_window(first, Array2::zeros((3, 3)).view())
        .unwrap();
    assert!(writer
        .write_window(first, Array2::ones((3, 3)).view())
        .is_err());
    assert!(writer
        .write_window(
            BlockIndices {
                row_start: 2,
                row_stop: 5,
                col_start: 2,
                col_stop: 5,
            },
            Array2::ones((3, 3)).view(),
        )
        .is_err());
    writer
        .write_window(
            BlockIndices {
                row_start: 3,
                row_stop: 6,
                col_start: 0,
                col_stop: 3,
            },
            Array2::ones((3, 3)).view(),
        )
        .unwrap();
    let _ = std::fs::remove_file(&scratch);
}

#[test]
fn bounded_writer_rejects_window_shape_and_bounds_mismatches() {
    let scratch =
        std::env::temp_dir().join(format!("dolphinrust_bad_window_{}.tif", std::process::id()));
    let _ = std::fs::remove_file(&scratch);
    let mut writer =
        BoundedCogWriter::<f32>::create(&scratch, (4, 4), GT, Some(32611), None, &[]).unwrap();
    let values = Array2::zeros((2, 2));
    let wrong_shape = writer.write_window(
        BlockIndices {
            row_start: 0,
            row_stop: 3,
            col_start: 0,
            col_stop: 2,
        },
        values.view(),
    );
    assert!(wrong_shape.is_err());
    let out_of_bounds = writer.write_window(
        BlockIndices {
            row_start: 3,
            row_stop: 5,
            col_start: 0,
            col_stop: 2,
        },
        values.view(),
    );
    assert!(out_of_bounds.is_err());
    let _ = std::fs::remove_file(&scratch);
}

// ------------------------------- oracle (secondary) ---------------------------

#[test]
fn reads_gdal_written_geotiff() {
    let dir = fixtures();
    if !dir.join("io_ref.tif").exists() {
        eprintln!("skipping reads_gdal_written_geotiff: no fixtures");
        return;
    }
    let RasterData {
        data,
        geotransform,
        epsg,
    } = read_raster::<f32>(&dir.join("io_ref.tif")).unwrap();
    let expected: Array2<f32> = ndarray_npy::read_npy(dir.join("io_ref_tif.npy")).unwrap();
    assert_eq!(data, expected, "GDAL-written pixels");
    assert_eq!(geotransform, GT, "GDAL geotransform");
    assert_eq!(epsg, Some(32611), "GDAL EPSG");
}

#[test]
fn reads_h5py_written_cslc() {
    let dir = fixtures();
    if !dir.join("io_cslc.h5").exists() {
        eprintln!("skipping reads_h5py_written_cslc: no fixtures");
        return;
    }
    let data = read_cslc(&dir.join("io_cslc.h5"), "/data/VV").unwrap();
    let expected: Array2<Cf32> = ndarray_npy::read_npy(dir.join("io_cslc.npy")).unwrap();
    assert_eq!(data, expected, "h5py-written complex CSLC");
}

#[test]
fn reads_cslc_stack() {
    let dir = fixtures();
    if !dir.join("io_cslc.h5").exists() {
        eprintln!("skipping reads_cslc_stack: no fixtures");
        return;
    }
    // Two layers from the same file -> (2, rows, cols).
    let files = vec![
        (dir.join("io_cslc.h5"), "/data/VV".to_string()),
        (dir.join("io_cslc.h5"), "/data/VV".to_string()),
    ];
    let stack: Array3<Cf32> = read_cslc_stack(&files).unwrap();
    assert_eq!(stack.dim().0, 2, "stack depth");
    assert_eq!(
        stack.index_axis(ndarray::Axis(0), 0),
        stack.index_axis(ndarray::Axis(0), 1)
    );
}
