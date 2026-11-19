//! NISAR / L-band geocoded-SLC reading from HDF5.
//!
//! **Verified against a real NISAR GSLC granule** (`NISAR_L2_GSLC_BETA_V1`, see
//! `VALIDATION.md`). NISAR products differ from OPERA S1 CSLC in exactly one way
//! that GDAL's HDF5 driver does not handle — the geocoding-grid metadata — so
//! that is the only NISAR-specific reader here:
//!
//! - **Complex samples are an `{r: f32, i: f32}` compound** (complex64), the same
//!   h5py-compatible `(r, i)` layout `read_cslc` already reads as [`Cf32`]. The
//!   prompt's "complex-int16" assumption did **not** hold on real data; real NISAR
//!   GSLC is float32, so [`read_nisar_rslc`] reads `Cf32` directly.
//! - **The geocoding grid lives in a NISAR product group** (e.g.
//!   `/science/LSAR/GSLC/grids/frequencyA/`) with camelCase coordinate datasets
//!   (`xCoordinates` F64, `yCoordinates` F64) and the EPSG carried as an
//!   `epsg_code` attribute (I64) on the `projection` dataset — not as the scalar
//!   dataset value OPERA uses. GDAL returns an identity geotransform for this, so
//!   the affine transform is derived from the coordinate spacing directly.
//!
//! - **Each frequency band is its own grid.** A dual-band product carries the
//!   main band under `frequencyA` and the side band under `frequencyB`, each with
//!   its own coordinates (the 5 MHz side band is coarser), so every reader here
//!   takes the band's own subdataset path. [`read_nisar_band_metadata`] reads the
//!   band's `centerFrequency` and `rangeBandwidth` (Hz) from the same group — the
//!   inputs to split-spectrum ionosphere estimation.
//!
//! Atmospheric (ionospheric/tropospheric) corrections are **out of scope** here:
//! this path yields a geometrically-correct but atmospherically-uncorrected
//! L-band product. Ionosphere is ~16× the C-band effect and is mandatory for a
//! *usable* L-band displacement product; it lands in a later loop.

use std::path::{Path, PathBuf};

use dolphin_core::{BlockIndices, Cf32};
use ndarray::{s, Array2, Array3, Axis};

use crate::error::{IoError, Result};
use crate::geo::GeoInfo;

/// Read a 2D NISAR complex grid at `dataset` as [`Cf32`] from the `{r: f32, i:
/// f32}` compound.
///
/// # Errors
/// Returns `Err` if the HDF5 read fails or the dataset is not the expected
/// complex-float32 compound type.
pub fn read_nisar_rslc(path: &Path, dataset: &str) -> Result<Array2<Cf32>> {
    let file = hdf5::File::open(path)?;
    Ok(file.dataset(dataset)?.read_2d::<Cf32>()?)
}

/// Read a rectangular `block` window of a NISAR complex grid via HDF5 hyperslab
/// selection — the NISAR mirror of [`crate::read_cslc_window`], bit-identical to
/// [`read_nisar_rslc`] sliced to the same rectangle.
///
/// # Errors
/// Returns `Err` if the HDF5 read fails or the dataset is not the expected
/// complex-float32 compound type.
pub fn read_nisar_window(path: &Path, dataset: &str, block: BlockIndices) -> Result<Array2<Cf32>> {
    let file = hdf5::File::open(path)?;
    Ok(file
        .dataset(dataset)?
        .read_slice_2d::<Cf32, _>(s![block.rows(), block.cols()])?)
}

/// Read a date-ordered set of NISAR files into an `(n_slc, rows, cols)` stack,
/// all from the same `dataset` (polarization grid) path. Mirrors
/// [`crate::read_cslc_stack`] for the OPERA path.
///
/// # Errors
/// Returns `Err` on any read failure or if the grids differ in shape.
pub fn read_nisar_stack(files: &[PathBuf], dataset: &str) -> Result<Array3<Cf32>> {
    let layers = files
        .iter()
        .map(|path| read_nisar_rslc(path, dataset))
        .collect::<Result<Vec<_>>>()?;
    let views: Vec<_> = layers.iter().map(Array2::view).collect();
    ndarray::stack(Axis(0), &views).map_err(|e| IoError::Shape(e.to_string()))
}

/// Read the geotransform + EPSG for a NISAR geocoded grid from its HDF5 metadata.
///
/// `subdataset` is the complex grid path (e.g.
/// `/science/LSAR/GSLC/grids/frequencyA/HH`); the `xCoordinates`/`yCoordinates`
/// arrays and the `projection` dataset are read from its parent group. The EPSG
/// is read from the `epsg_code` attribute of the `projection` dataset (NISAR
/// convention), falling back to the scalar dataset value when the attribute is
/// absent.
///
/// # Errors
/// Returns `Err` if the HDF5 read fails or the coordinate/projection datasets are
/// absent or too short to define a pixel spacing.
pub fn read_nisar_geotransform(path: &Path, subdataset: &str) -> Result<GeoInfo> {
    let group = parent_group(subdataset);
    let file = hdf5::File::open(path)?;
    let x = file
        .dataset(&format!("{group}/xCoordinates"))?
        .read_raw::<f64>()?;
    let y = file
        .dataset(&format!("{group}/yCoordinates"))?
        .read_raw::<f64>()?;
    if x.len() < 2 || y.len() < 2 {
        return Err(IoError::Geo(
            "coordinate arrays too short to define a spacing".into(),
        ));
    }
    let dx = x[1] - x[0];
    let dy = y[1] - y[0];
    let epsg = read_epsg(&file, &group)?;
    Ok(GeoInfo {
        epsg,
        geotransform: [x[0] - dx / 2.0, dx, 0.0, y[0] - dy / 2.0, 0.0, dy],
    })
}

/// Radar metadata for one NISAR frequency band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NisarBandMetadata {
    /// `centerFrequency`: band center (carrier) frequency in Hz.
    pub center_frequency_hz: f64,
    /// `rangeBandwidth`: processed range bandwidth in Hz.
    pub range_bandwidth_hz: f64,
}

/// Read `centerFrequency` and `rangeBandwidth` for the band holding
/// `subdataset` (e.g. `/science/LSAR/GSLC/grids/frequencyB/HH` → the
/// `frequencyB` group's scalars).
///
/// # Errors
/// Returns `Err` if the file has no such band group or either scalar is absent.
pub fn read_nisar_band_metadata(path: &Path, subdataset: &str) -> Result<NisarBandMetadata> {
    let group = parent_group(subdataset);
    let file = hdf5::File::open(path)?;
    let scalar = |name: &str| -> Result<f64> {
        Ok(file
            .dataset(&format!("{group}/{name}"))?
            .read_scalar::<f64>()?)
    };
    Ok(NisarBandMetadata {
        center_frequency_hz: scalar("centerFrequency")?,
        range_bandwidth_hz: scalar("rangeBandwidth")?,
    })
}

/// EPSG from the `projection` dataset's `epsg_code` attribute (NISAR), else from
/// the dataset's scalar value.
fn read_epsg(file: &hdf5::File, group: &str) -> Result<u32> {
    let proj = file.dataset(&format!("{group}/projection"))?;
    let code = match proj.attr("epsg_code") {
        Ok(attr) => attr.read_scalar::<i64>()?,
        Err(_) => proj.read_scalar::<i64>()?,
    };
    u32::try_from(code).map_err(|_| IoError::Geo(format!("invalid EPSG {code}")))
}

/// Parent group of an HDF5 dataset path (`/grids/frequencyA/HH` →
/// `/grids/frequencyA`).
fn parent_group(subdataset: &str) -> String {
    match subdataset.rsplit_once('/') {
        Some(("", _)) | None => "/".to_string(),
        Some((parent, _)) => parent.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nisar_fixture::{
        write_nisar_dual_band_fixture, write_nisar_fixture, FixtureBand, FREQUENCY_A_GROUP,
        FREQUENCY_B_GROUP,
    };

    /// Contract: the reader recovers the known pixel values, grid shape, and
    /// geotransform/EPSG from a synthesized NISAR-layout fixture — the de-risk
    /// that `hdf5-metno` reads the `{r,i}` f32 compound and that the custom
    /// geotransform reader replaces GDAL's identity transform.
    #[test]
    fn reads_synthesized_nisar_fixture() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_nisar_contract.h5");
        let _ = std::fs::remove_file(&path);
        // 2x3 grid with distinct, signed samples.
        let cpx = Array2::from_shape_fn((2, 3), |(r, c)| {
            let n = (r * 3 + c) as f32;
            Cf32::new(n - 2.0, 10.0 + n)
        });
        let x = [300_000.0_f64, 300_020.0, 300_040.0]; // dx = 20, centers
        let y = [4_100_000.0_f64, 4_099_980.0]; // dy = -20
        write_nisar_fixture(&path, "HH", cpx.view(), &x, &y, 32610).unwrap();

        let dataset = format!("{FREQUENCY_A_GROUP}/HH");
        let grid = read_nisar_rslc(&path, &dataset).unwrap();
        assert_eq!(grid.dim(), (2, 3));
        assert_eq!(grid[(0, 0)], Cf32::new(-2.0, 10.0));
        assert_eq!(grid[(1, 2)], Cf32::new(3.0, 15.0));

        let geo = read_nisar_geotransform(&path, &dataset).unwrap();
        assert_eq!(geo.epsg, 32610);
        assert!((geo.geotransform[0] - 299_990.0).abs() < 1e-6); // 300000 - 20/2
        assert!((geo.geotransform[1] - 20.0).abs() < 1e-9);
        assert!((geo.geotransform[3] - 4_100_010.0).abs() < 1e-6); // 4100000 + 20/2
        assert!((geo.geotransform[5] + 20.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }

    /// Contract: the windowed NISAR read equals the full read sliced to the same
    /// rectangle (the NISAR half of the block-tiled bit-identity invariant).
    #[test]
    fn nisar_window_matches_full_read_sliced() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_nisar_window_contract.h5");
        let _ = std::fs::remove_file(&path);
        let full = Array2::from_shape_fn((12, 16), |(r, c)| Cf32::new(r as f32, c as f32 - 4.0));
        let x: Vec<f64> = (0..16).map(|i| 300_000.0 + 20.0 * i as f64).collect();
        let y: Vec<f64> = (0..12).map(|i| 4_100_000.0 - 20.0 * i as f64).collect();
        write_nisar_fixture(&path, "HH", full.view(), &x, &y, 32610).unwrap();

        let dataset = format!("{FREQUENCY_A_GROUP}/HH");
        let block = BlockIndices {
            row_start: 3,
            row_stop: 10,
            col_start: 5,
            col_stop: 14,
        };
        let win = read_nisar_window(&path, &dataset, block).unwrap();
        let expected = full.slice(s![block.rows(), block.cols()]).to_owned();
        assert_eq!(win, expected);
        let _ = std::fs::remove_file(&path);
    }

    /// Contract: a dual-band fixture yields the `frequencyB` grid on its own
    /// coarser geotransform, plus `centerFrequency`/`rangeBandwidth` per band,
    /// while `frequencyA` reads exactly as before.
    #[test]
    fn reads_nisar_frequency_b_grid_and_band_metadata() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_nisar_dual_band_contract.h5");
        let _ = std::fs::remove_file(&path);
        let main_cpx = Array2::from_shape_fn((4, 6), |(r, c)| Cf32::new(r as f32, c as f32));
        let side_cpx =
            Array2::from_shape_fn((2, 3), |(r, c)| Cf32::new(-(r as f32), 7.0 + c as f32));
        let main_x: Vec<f64> = (0..6).map(|i| 300_000.0 + 10.0 * i as f64).collect();
        let main_y: Vec<f64> = (0..4).map(|i| 4_100_000.0 - 10.0 * i as f64).collect();
        let side_x = [300_005.0_f64, 300_025.0, 300_045.0];
        let side_y = [4_099_995.0_f64, 4_099_975.0];
        let main = FixtureBand {
            cpx: main_cpx.view(),
            x: &main_x,
            y: &main_y,
            center_frequency_hz: 1.2575e9,
            range_bandwidth_hz: 40e6,
        };
        let side = FixtureBand {
            cpx: side_cpx.view(),
            x: &side_x,
            y: &side_y,
            center_frequency_hz: 1.2935e9,
            range_bandwidth_hz: 5e6,
        };
        write_nisar_dual_band_fixture(&path, "HH", &main, &side, 32610).unwrap();

        let main_ds = format!("{FREQUENCY_A_GROUP}/HH");
        let side_ds = format!("{FREQUENCY_B_GROUP}/HH");
        assert_eq!(read_nisar_rslc(&path, &main_ds).unwrap(), main_cpx);
        assert_eq!(read_nisar_rslc(&path, &side_ds).unwrap(), side_cpx);

        let side_geo = read_nisar_geotransform(&path, &side_ds).unwrap();
        assert_eq!(side_geo.epsg, 32610);
        assert!((side_geo.geotransform[0] - 299_995.0).abs() < 1e-6);
        assert!((side_geo.geotransform[1] - 20.0).abs() < 1e-9);
        assert!((side_geo.geotransform[5] + 20.0).abs() < 1e-9);

        let main_meta = read_nisar_band_metadata(&path, &main_ds).unwrap();
        let side_meta = read_nisar_band_metadata(&path, &side_ds).unwrap();
        assert_eq!(main_meta.center_frequency_hz, 1.2575e9);
        assert_eq!(main_meta.range_bandwidth_hz, 40e6);
        assert_eq!(side_meta.center_frequency_hz, 1.2935e9);
        assert_eq!(side_meta.range_bandwidth_hz, 5e6);
        let _ = std::fs::remove_file(&path);
    }

    /// Contract: a single-band product has no `frequencyB`, so its band metadata
    /// read fails rather than returning a default.
    #[test]
    fn nisar_band_metadata_absent_without_frequency_b() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_nisar_single_band_contract.h5");
        let _ = std::fs::remove_file(&path);
        let cpx = Array2::from_elem((2, 2), Cf32::new(1.0, 0.0));
        write_nisar_fixture(&path, "HH", cpx.view(), &[0.0, 10.0], &[0.0, -10.0], 32610).unwrap();
        let side_ds = format!("{FREQUENCY_B_GROUP}/HH");
        assert!(read_nisar_band_metadata(&path, &side_ds).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
