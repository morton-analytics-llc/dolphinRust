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
//! Atmospheric (ionospheric/tropospheric) corrections are **out of scope** here:
//! this path yields a geometrically-correct but atmospherically-uncorrected
//! L-band product. Ionosphere is ~16× the C-band effect and is mandatory for a
//! *usable* L-band displacement product; it lands in a later loop.

use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use ndarray::{Array2, Array3, Axis};

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
    use crate::nisar_fixture::{write_nisar_fixture, FREQUENCY_A_GROUP};

    /// Contract: the reader recovers the known pixel values, grid shape, and
    /// geotransform/EPSG from a synthesized NISAR-layout fixture — the de-risk
    /// that `hdf5-metno` reads the `{r,i}` f32 compound and that the custom
    /// geotransform reader replaces GDAL's identity transform.
    #[test]
    fn reads_synthesized_nisar_fixture() {
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
}
