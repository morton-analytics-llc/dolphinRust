//! Write a geocoded NISAR date in the NISAR GSLC HDF5 layout.
//!
//! The phase-linking path reads NISAR grids at
//! `/science/LSAR/GSLC/grids/frequencyX/<POL>` with `xCoordinates`,
//! `yCoordinates`, `projection` (`epsg_code` attribute), `centerFrequency`,
//! `rangeBandwidth` and `mask` beside them, and the LOS cube under
//! `/science/LSAR/GSLC/metadata/radarGrid` (`losUnitVectorX/Y` on
//! `heightAboveEllipsoid` × y × x). Those are written with JPL's names, so
//! an in-house file drops into the existing NISAR readers (`input_type:
//! l1_nisar`). `identification/processingCenter` names this front end, which
//! is how the workflow tells it from a JPL GSLC.

use std::path::Path;

use hdf5::types::FixedAscii;
use ndarray::{Array2, Array3};

use crate::cslc_h5::{put_f64, put_str, put_vec, time_string, DEM_SOURCE};
use crate::error::{product, Result};
use crate::nisar_geocode::GeocodedBand;
use crate::rslc::{NisarFrame, Rslc};

/// `identification/processingCenter` of in-house files.
pub const PROCESSING_CENTER: &str = "Morton Analytics dolphinRust";

/// Ground-to-sensor LOS on a coarse grid at several heights.
pub struct LosCube {
    /// EPSG of the cube grid.
    pub epsg: u32,
    /// Node x coordinates (increasing).
    pub x: Vec<f64>,
    /// Node y coordinates (decreasing).
    pub y: Vec<f64>,
    /// Ellipsoidal heights (increasing).
    pub heights: Vec<f64>,
    /// East component `[height, y, x]`.
    pub east: Array3<f32>,
    /// North component `[height, y, x]`.
    pub north: Array3<f32>,
}

/// What went into one file.
pub struct NisarProvenance<'a> {
    /// Track/frame.
    pub frame: NisarFrame,
    /// Output granule id (file stem).
    pub granule_id: &'a str,
    /// TEC ancillary used, when the ionosphere was applied.
    pub tec_file: Option<&'a str>,
}

fn put_projection(group: &hdf5::Group, epsg: u32) -> Result<()> {
    let dataset = group.new_dataset::<i64>().create("projection")?;
    dataset.write_scalar(&i64::from(epsg))?;
    dataset
        .new_attr::<i64>()
        .create("epsg_code")?
        .write_scalar(&i64::from(epsg))?;
    Ok(())
}

fn put_strings(group: &hdf5::Group, name: &str, values: &[String]) -> Result<()> {
    let fixed = values
        .iter()
        .map(|v| FixedAscii::<256>::from_ascii(v.as_bytes()).map_err(product))
        .collect::<Result<Vec<_>>>()?;
    group
        .new_dataset_builder()
        .with_data(&ndarray::Array1::from(fixed))
        .create(name)?;
    Ok(())
}

fn put_with_units(group: &hdf5::Group, name: &str, values: &[f64], units: &str) -> Result<()> {
    put_vec(group, name, values)?;
    let text = FixedAscii::<64>::from_ascii(units.as_bytes()).map_err(product)?;
    group
        .dataset(name)?
        .new_attr::<FixedAscii<64>>()
        .create("units")?
        .write_scalar(&text)?;
    Ok(())
}

fn rows(values: &[[f64; 3]]) -> Result<Array2<f64>> {
    Array2::from_shape_vec(
        (values.len(), 3),
        values.iter().flat_map(|v| v.iter().copied()).collect(),
    )
    .map_err(product)
}

fn write_band(
    root: &hdf5::Group,
    geocoded: &GeocodedBand,
    rslc: &Rslc,
    frequency: char,
    polarizations: &[String],
) -> Result<()> {
    let band = rslc.band(frequency)?;
    let group = root.create_group(&format!("frequency{frequency}"))?;
    let (rows_n, cols_n) = geocoded.mask.dim();
    let chunk = (rows_n.min(512), cols_n.min(512));
    for (pol, layer) in polarizations.iter().zip(&geocoded.layers) {
        group
            .new_dataset::<num_complex::Complex32>()
            .shape((rows_n, cols_n))
            .chunk(chunk)
            .deflate(4)
            .create(pol.as_str())?
            .write(layer)?;
    }
    group
        .new_dataset::<u8>()
        .shape((rows_n, cols_n))
        .chunk(chunk)
        .deflate(4)
        .create("mask")?
        .write(&geocoded.mask)?;
    put_vec(&group, "xCoordinates", &geocoded.grid.x_coordinates())?;
    put_vec(&group, "yCoordinates", &geocoded.grid.y_coordinates())?;
    put_f64(&group, "xCoordinateSpacing", geocoded.grid.dx)?;
    put_f64(&group, "yCoordinateSpacing", geocoded.grid.dy)?;
    put_projection(&group, geocoded.grid.epsg)?;
    put_f64(&group, "centerFrequency", band.center_frequency)?;
    put_f64(&group, "rangeBandwidth", band.range_bandwidth)?;
    put_f64(&group, "slantRangeSpacing", band.range_spacing)?;
    put_f64(&group, "zeroDopplerTimeSpacing", rslc.line_spacing)?;
    put_strings(&group, "listOfPolarizations", polarizations)
}

fn write_identification(
    lsar: &hdf5::Group,
    rslc: &Rslc,
    provenance: &NisarProvenance,
    frequencies: &[String],
) -> Result<()> {
    let ident = lsar.create_group("identification")?;
    put_str(&ident, "granuleId", provenance.granule_id)?;
    put_str(&ident, "productType", "GSLC")?;
    put_str(&ident, "processingCenter", PROCESSING_CENTER)?;
    put_str(&ident, "missionId", "NISAR")?;
    put_str(&ident, "isGeocoded", "True")?;
    put_str(&ident, "lookDirection", &format!("{:?}", rslc.look_side))?;
    put_str(&ident, "orbitPassDirection", &rslc.pass_direction)?;
    put_str(
        &ident,
        "zeroDopplerStartTime",
        &time_string(rslc.sensing_start()),
    )?;
    put_str(
        &ident,
        "zeroDopplerEndTime",
        &time_string(rslc.sensing_stop()),
    )?;
    for (name, value) in [
        ("absoluteOrbitNumber", rslc.absolute_orbit),
        ("trackNumber", provenance.frame.track),
        ("frameNumber", provenance.frame.frame),
    ] {
        ident
            .new_dataset::<u32>()
            .create(name)?
            .write_scalar(&value)?;
    }
    put_strings(&ident, "listOfFrequencies", frequencies)
}

fn write_orbit_attitude(metadata: &hdf5::Group, rslc: &Rslc) -> Result<()> {
    let units = format!("seconds since {}", rslc.epoch.format("%Y-%m-%dT%H:%M:%S"));
    let orbit = metadata.create_group("orbit")?;
    put_with_units(&orbit, "time", &rslc.orbit.times, &units)?;
    orbit
        .new_dataset_builder()
        .with_data(&rows(&rslc.orbit.positions)?)
        .create("position")?;
    orbit
        .new_dataset_builder()
        .with_data(&rows(&rslc.orbit.velocities)?)
        .create("velocity")?;
    put_str(&orbit, "orbitType", &rslc.orbit.orbit_type)?;
    let attitude = metadata.create_group("attitude")?;
    put_with_units(&attitude, "time", &rslc.attitude.times, &units)?;
    let quaternions = Array2::from_shape_vec(
        (rslc.attitude.quaternions.len(), 4),
        rslc.attitude
            .quaternions
            .iter()
            .flat_map(|q| q.iter().copied())
            .collect(),
    )
    .map_err(product)?;
    attitude
        .new_dataset_builder()
        .with_data(&quaternions)
        .create("quaternions")?;
    Ok(())
}

fn write_cube(metadata: &hdf5::Group, cube: &LosCube) -> Result<()> {
    let group = metadata.create_group("radarGrid")?;
    for (name, layer) in [
        ("losUnitVectorX", &cube.east),
        ("losUnitVectorY", &cube.north),
    ] {
        group.new_dataset_builder().with_data(layer).create(name)?;
    }
    put_vec(&group, "xCoordinates", &cube.x)?;
    put_vec(&group, "yCoordinates", &cube.y)?;
    put_vec(&group, "heightAboveEllipsoid", &cube.heights)?;
    put_projection(&group, cube.epsg)
}

fn write_processing(
    metadata: &hdf5::Group,
    rslc: &Rslc,
    provenance: &NisarProvenance,
) -> Result<()> {
    let processing = metadata.create_group("processingInformation")?;
    let inputs = processing.create_group("inputs")?;
    put_strings(
        &inputs,
        "l1SlcGranules",
        std::slice::from_ref(&rslc.granule_id),
    )?;
    put_str(&inputs, "demSource", DEM_SOURCE)?;
    if let Some(tec) = provenance.tec_file {
        put_strings(&inputs, "tecFiles", &[tec.to_string()])?;
    }
    let parameters = processing.create_group("parameters")?;
    let flag = |b: bool| if b { "True" } else { "False" };
    let ionosphere = provenance.tec_file.is_some();
    for (name, applied) in [
        ("rangeIonosphericGeolocationCorrectionApplied", ionosphere),
        ("azimuthIonosphericGeolocationCorrectionApplied", ionosphere),
        ("solidEarthTidesGeolocationCorrectionApplied", true),
        ("dryTroposphericGeolocationCorrectionApplied", true),
        ("wetTroposphericGeolocationCorrectionApplied", false),
        ("ellipsoidalFlatteningApplied", true),
        ("topographicFlatteningApplied", true),
    ] {
        put_str(&parameters, name, flag(applied))?;
    }
    Ok(())
}

/// Write one date: `bands` pairs each geocoded band with its frequency.
///
/// # Errors
/// HDF5 failures.
pub fn write_nisar(
    path: &Path,
    rslc: &Rslc,
    bands: &[(char, GeocodedBand)],
    polarizations: &[String],
    cube: &LosCube,
    provenance: &NisarProvenance,
) -> Result<()> {
    let file = hdf5::File::create(path)?;
    let frequencies: Vec<String> = bands.iter().map(|(f, _)| f.to_string()).collect();
    let lsar = file.create_group("science")?.create_group("LSAR")?;
    write_identification(&lsar, rslc, provenance, &frequencies)?;
    let product = lsar.create_group("GSLC")?;
    let grids = product.create_group("grids")?;
    for (frequency, geocoded) in bands {
        write_band(&grids, geocoded, rslc, *frequency, polarizations)?;
    }
    let metadata = product.create_group("metadata")?;
    write_orbit_attitude(&metadata, rslc)?;
    write_cube(&metadata, cube)?;
    write_processing(&metadata, rslc, provenance)
}
