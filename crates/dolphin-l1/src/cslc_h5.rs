//! Write a geocoded burst in the OPERA CSLC-S1 HDF5 layout.
//!
//! The phase-linking path reads `/data/<POL>` with `x_coordinates`,
//! `y_coordinates` and `projection` beside it (`input_type: opera_cslc`);
//! provenance and baselines read `/identification`, `/metadata/orbit` and
//! `input_burst_metadata`. Those groups are written with the same names and
//! string formats, so the in-house CSLC drops into the existing reader
//! unchanged. `/metadata/processing_information/parameters` records which
//! corrections this front end applied.

use std::path::Path;

use chrono::{DateTime, Utc};
use hdf5::types::FixedAscii;
use ndarray::Array1;

use crate::error::{product, Result};
use crate::geocode::GeocodedBurst;
use crate::orbit::Orbit;
use crate::safe::S1Burst;

/// Product provenance written into `/identification` and `inputs`.
pub struct CslcProvenance<'a> {
    /// OPERA-form burst id, `t100_213486_iw2`.
    pub burst_id: &'a str,
    /// Relative orbit (track).
    pub track: u32,
    /// Parent SAFE name.
    pub l1_slc: &'a str,
    /// Orbit file name.
    pub orbit_file: &'a str,
    /// GIM (IONEX) file of the ionospheric delay, when applied.
    pub gim_file: Option<&'a str>,
    /// Burst centre `[lon, lat]` (deg).
    pub center_lonlat: [f64; 2],
}

/// The DEM description written to `inputs/dem_source`.
pub const DEM_SOURCE: &str = "Copernicus GLO-30 + EGM2008 (WGS84 ellipsoid)";

pub(crate) fn time_string(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%d %H:%M:%S%.6f").to_string()
}

/// Fixed-length ASCII scalar; 64 bytes (what `dolphin-io` reads) when it
/// fits, 256 otherwise.
pub(crate) fn put_str(group: &hdf5::Group, name: &str, value: &str) -> Result<()> {
    if value.len() <= 64 {
        let fixed = FixedAscii::<64>::from_ascii(value.as_bytes()).map_err(product)?;
        group
            .new_dataset::<FixedAscii<64>>()
            .create(name)?
            .write_scalar(&fixed)?;
    } else {
        let fixed = FixedAscii::<256>::from_ascii(value.as_bytes()).map_err(product)?;
        group
            .new_dataset::<FixedAscii<256>>()
            .create(name)?
            .write_scalar(&fixed)?;
    }
    Ok(())
}

pub(crate) fn put_f64(group: &hdf5::Group, name: &str, value: f64) -> Result<()> {
    group
        .new_dataset::<f64>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

fn put_bool(group: &hdf5::Group, name: &str, value: bool) -> Result<()> {
    group
        .new_dataset::<bool>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

pub(crate) fn put_vec(group: &hdf5::Group, name: &str, values: &[f64]) -> Result<()> {
    group
        .new_dataset_builder()
        .with_data(&Array1::from(values.to_vec()))
        .create(name)?;
    Ok(())
}

/// Write `geocoded` (from `burst` on `orbit`) to `path`.
pub fn write_cslc(
    path: &Path,
    geocoded: &GeocodedBurst,
    burst: &S1Burst,
    orbit: &Orbit,
    provenance: &CslcProvenance,
) -> Result<()> {
    let file = hdf5::File::create(path)?;
    let data = file.create_group("data")?;
    let (rows, cols) = geocoded.data.dim();
    data.new_dataset::<num_complex::Complex32>()
        .shape((rows, cols))
        .chunk((rows.min(128), cols.min(128)))
        .deflate(4)
        .create(burst.polarization.as_str())?
        .write(&geocoded.data)?;
    if let Some((carrier, flatten)) = &geocoded.phases {
        for (name, layer) in [
            ("azimuth_carrier_phase", carrier),
            ("flattening_phase", flatten),
        ] {
            data.new_dataset::<f32>()
                .shape((rows, cols))
                .chunk((rows.min(128), cols.min(128)))
                .deflate(4)
                .create(name)?
                .write(layer)?;
        }
    }
    put_vec(&data, "x_coordinates", &geocoded.grid.x_coordinates())?;
    put_vec(&data, "y_coordinates", &geocoded.grid.y_coordinates())?;
    data.new_dataset::<i32>()
        .create("projection")?
        .write_scalar(&i32::try_from(geocoded.grid.epsg).map_err(product)?)?;
    write_identification(&file, burst, provenance)?;
    write_orbit(&file, orbit, burst)?;
    write_processing(&file, burst, provenance)
}

fn write_identification(
    file: &hdf5::File,
    burst: &S1Burst,
    provenance: &CslcProvenance,
) -> Result<()> {
    let ident = file.create_group("identification")?;
    put_str(&ident, "burst_id", provenance.burst_id)?;
    put_str(&ident, "orbit_pass_direction", &burst.pass_direction)?;
    put_str(&ident, "look_direction", "Right")?;
    put_str(
        &ident,
        "zero_doppler_start_time",
        &time_string(burst.sensing_start),
    )?;
    put_str(
        &ident,
        "zero_doppler_end_time",
        &time_string(burst.sensing_stop()),
    )?;
    put_str(&ident, "mission_id", &burst.mission)?;
    put_str(&ident, "product_type", "CSLC_S1")?;
    put_str(&ident, "is_geocoded", "True")?;
    put_str(&ident, "processing_center", "Morton Analytics dolphinRust")?;
    ident
        .new_dataset::<i64>()
        .create("absolute_orbit_number")?
        .write_scalar(&i64::from(burst.absolute_orbit))?;
    ident
        .new_dataset::<i64>()
        .create("track_number")?
        .write_scalar(&i64::from(provenance.track))?;
    Ok(())
}

fn write_orbit(file: &hdf5::File, orbit: &Orbit, burst: &S1Burst) -> Result<()> {
    let group = file.create_group("metadata")?.create_group("orbit")?;
    put_vec(&group, "time", &orbit.times)?;
    for (axis, k) in [("x", 0), ("y", 1), ("z", 2)] {
        let position: Vec<f64> = orbit.positions.iter().map(|p| p[k]).collect();
        let velocity: Vec<f64> = orbit.velocities.iter().map(|v| v[k]).collect();
        put_vec(&group, &format!("position_{axis}"), &position)?;
        put_vec(&group, &format!("velocity_{axis}"), &velocity)?;
    }
    put_str(
        &group,
        "reference_epoch",
        &time_string(orbit.reference_epoch),
    )?;
    put_str(&group, "orbit_type", &orbit.orbit_type)?;
    put_str(&group, "orbit_direction", &burst.pass_direction)
}

fn write_processing(file: &hdf5::File, burst: &S1Burst, provenance: &CslcProvenance) -> Result<()> {
    let processing = file
        .group("metadata")?
        .create_group("processing_information")?;
    let meta = processing.create_group("input_burst_metadata")?;
    put_f64(&meta, "range_pixel_spacing", burst.range_pixel_spacing())?;
    put_f64(&meta, "azimuth_time_interval", burst.azimuth_time_interval)?;
    put_f64(&meta, "starting_range", burst.starting_range())?;
    put_f64(&meta, "wavelength", burst.wavelength())?;
    put_f64(&meta, "radar_center_frequency", burst.radar_frequency)?;
    put_f64(&meta, "prf_raw_data", burst.prf_raw)?;
    put_f64(&meta, "range_bandwidth", burst.range_bandwidth)?;
    put_f64(&meta, "iw2_mid_range", burst.iw2_mid_range)?;
    put_vec(&meta, "center", &provenance.center_lonlat)?;
    put_str(&meta, "sensing_start", &time_string(burst.sensing_start))?;
    put_str(&meta, "polarization", &burst.polarization)?;
    put_str(&meta, "platform_id", &burst.mission)?;
    put_str(&meta, "ipf_version", &burst.ipf_version)?;
    let inputs = processing.create_group("inputs")?;
    put_str(&inputs, "dem_source", DEM_SOURCE)?;
    put_str(&inputs, "l1_slc_files", provenance.l1_slc)?;
    put_str(&inputs, "orbit_files", provenance.orbit_file)?;
    if let Some(gim) = provenance.gim_file {
        put_str(&inputs, "tec_file", gim)?;
    }
    let parameters = processing.create_group("parameters")?;
    for (name, applied) in [
        ("bistatic_delay_applied", true),
        ("azimuth_fm_rate_applied", true),
        ("geometry_doppler_applied", true),
        ("static_troposphere_applied", true),
        ("ellipsoidal_flattening_applied", true),
        ("topographic_flattening_applied", true),
        ("ionosphere_tec_applied", provenance.gim_file.is_some()),
        ("los_solid_earth_tides_applied", true),
        ("azimuth_solid_earth_tides_applied", false),
        ("wet_troposphere_weather_model_applied", false),
        ("dry_troposphere_weather_model_applied", false),
    ] {
        put_bool(&parameters, name, applied)?;
    }
    Ok(())
}
