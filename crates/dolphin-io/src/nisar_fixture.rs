//! Synthesized NISAR-layout HDF5 fixtures (feature `nisar-fixture`).
//!
//! Writes a minimal but structurally faithful NISAR geocoded-SLC product
//! (matching a real `NISAR_L2_GSLC_BETA_V1` granule's layout): a complex-`f32`
//! `{r, i}` compound grid under a `frequencyA` group with camelCase
//! `xCoordinates`/`yCoordinates` F64 arrays and a `projection` dataset carrying
//! the EPSG as an `epsg_code` I64 attribute. Used by the `dolphin-io` reader
//! contract test and the `dolphin-workflows` end-to-end NISAR stack test, so the
//! wiring is provable without a real granule.

use std::path::Path;

use dolphin_core::Cf32;
use ndarray::ArrayView2;

use crate::error::Result;

/// The NISAR group holding the geocoded grid in these fixtures.
pub const FREQUENCY_A_GROUP: &str = "/science/LSAR/GSLC/grids/frequencyA";

/// The NISAR side-band (`frequencyB`) group, written only by
/// [`write_nisar_dual_band_fixture`].
pub const FREQUENCY_B_GROUP: &str = "/science/LSAR/GSLC/grids/frequencyB";

/// One frequency band of a dual-band fixture: its complex grid, pixel-center
/// coordinates, and the `centerFrequency` / `rangeBandwidth` scalars (Hz).
#[derive(Debug, Clone, Copy)]
pub struct FixtureBand<'a> {
    /// Complex-`f32` grid written at `<group>/<pol>`.
    pub cpx: ArrayView2<'a, Cf32>,
    /// Pixel-center x coordinates.
    pub x: &'a [f64],
    /// Pixel-center y coordinates.
    pub y: &'a [f64],
    /// `centerFrequency` (Hz).
    pub center_frequency_hz: f64,
    /// `rangeBandwidth` (Hz).
    pub range_bandwidth_hz: f64,
}

/// Write a NISAR-layout fixture to `path`: the complex-`f32` `{r, i}` compound
/// grid `cpx` at `<FREQUENCY_A_GROUP>/<pol>`, with pixel-center `x`/`y`
/// coordinate arrays and `projection.epsg_code = epsg`.
///
/// # Errors
/// Returns `Err` on any HDF5 write failure.
pub fn write_nisar_fixture(
    path: &Path,
    pol: &str,
    cpx: ArrayView2<Cf32>,
    x: &[f64],
    y: &[f64],
    epsg: u32,
) -> Result<()> {
    let file = hdf5::File::create(path)?;
    let grids = file.create_group(FREQUENCY_A_GROUP)?;
    grids.new_dataset_builder().with_data(&cpx).create(pol)?;
    grids
        .new_dataset_builder()
        .with_data(x)
        .create("xCoordinates")?;
    grids
        .new_dataset_builder()
        .with_data(y)
        .create("yCoordinates")?;
    let proj = grids.new_dataset::<i64>().create("projection")?;
    proj.write_scalar(&i64::from(epsg))?;
    proj.new_attr::<i64>()
        .create("epsg_code")?
        .write_scalar(&i64::from(epsg))?;
    Ok(())
}

/// Write a dual-band NISAR-layout fixture: `main` under [`FREQUENCY_A_GROUP`]
/// and `side` under [`FREQUENCY_B_GROUP`], each with its own grid, coordinates,
/// `projection.epsg_code = epsg`, and `centerFrequency`/`rangeBandwidth`
/// scalars. [`write_nisar_fixture`] is unchanged and still writes `frequencyA`
/// only, without band metadata.
///
/// # Errors
/// Returns `Err` on any HDF5 write failure.
pub fn write_nisar_dual_band_fixture(
    path: &Path,
    pol: &str,
    main: &FixtureBand,
    side: &FixtureBand,
    epsg: u32,
) -> Result<()> {
    let file = hdf5::File::create(path)?;
    write_band(&file, FREQUENCY_A_GROUP, pol, main, epsg)?;
    write_band(&file, FREQUENCY_B_GROUP, pol, side, epsg)
}

/// Write one band's grid, coordinates, projection and frequency scalars.
fn write_band(
    file: &hdf5::File,
    group_path: &str,
    pol: &str,
    band: &FixtureBand,
    epsg: u32,
) -> Result<()> {
    let group = file.create_group(group_path)?;
    group
        .new_dataset_builder()
        .with_data(&band.cpx)
        .create(pol)?;
    group
        .new_dataset_builder()
        .with_data(band.x)
        .create("xCoordinates")?;
    group
        .new_dataset_builder()
        .with_data(band.y)
        .create("yCoordinates")?;
    let proj = group.new_dataset::<i64>().create("projection")?;
    proj.write_scalar(&i64::from(epsg))?;
    proj.new_attr::<i64>()
        .create("epsg_code")?
        .write_scalar(&i64::from(epsg))?;
    group
        .new_dataset::<f64>()
        .create("centerFrequency")?
        .write_scalar(&band.center_frequency_hz)?;
    group
        .new_dataset::<f64>()
        .create("rangeBandwidth")?
        .write_scalar(&band.range_bandwidth_hz)?;
    Ok(())
}
