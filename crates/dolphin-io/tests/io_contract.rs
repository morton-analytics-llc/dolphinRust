//! Phase-8 (I/O) contract tests.
//!
//! Primary (round-trip): a GeoTIFF written by Rust reads back with identical
//! pixels, geotransform, and EPSG. Secondary (oracle): Rust reads a GDAL-written
//! GeoTIFF and an h5py-written OPERA-style CSLC HDF5 matching the known arrays.
//! Oracle tests skip without fixtures.

use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_io::{read_cslc, read_cslc_stack, read_raster, write_raster, RasterData};
use ndarray::{Array2, Array3};

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
