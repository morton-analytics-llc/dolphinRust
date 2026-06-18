//! Synthesized NISAR-layout HDF5 fixtures (feature `nisar-fixture`).
//!
//! Writes a minimal but structurally faithful NISAR geocoded-SLC product: a
//! `{r: i16, i: i16}` compound grid under a `frequencyA` group with camelCase
//! `xCoordinates`/`yCoordinates` arrays and a `projection` dataset carrying the
//! EPSG as an `epsg_code` attribute. Used by the `dolphin-io` reader contract
//! test and the `dolphin-workflows` end-to-end NISAR stack test, so the wiring is
//! provable without a real granule.

use std::path::Path;

use ndarray::ArrayView2;

use crate::error::Result;
use crate::nisar::ComplexI16;

/// The NISAR group holding the geocoded grid in these fixtures.
pub const FREQUENCY_A_GROUP: &str = "/science/LSAR/GSLC/grids/frequencyA";

/// Write a NISAR-layout fixture to `path`: the `i16`-compound complex grid
/// `cpx` at `<FREQUENCY_A_GROUP>/<pol>`, with pixel-center `x`/`y` coordinate
/// arrays and `projection.epsg_code = epsg`.
///
/// # Errors
/// Returns `Err` on any HDF5 write failure.
pub fn write_nisar_fixture(
    path: &Path,
    pol: &str,
    cpx: ArrayView2<ComplexI16>,
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
