//! Geometry-provenance metadata from OPERA CSLC-S1 HDF5.
//!
//! Raw reads of the `/identification`, `/metadata/orbit`, and
//! `/metadata/processing_information/input_burst_metadata` groups that source the
//! geometry-provenance artifact (dolphinRust #1). IO only — interpretation
//! (normalization, heading geodesy, consistency gates) lives in
//! `dolphin-workflows::provenance`. Keys verified against a real
//! `OPERA_L2_CSLC-S1_T144-308011-IW2` v1.1 granule.

use std::path::Path;

use hdf5::types::FixedAscii;

use crate::error::Result;

/// `/identification` scalars used for provenance.
#[derive(Debug, Clone)]
pub struct CslcIdentification {
    /// Raw `orbit_pass_direction` string (e.g. `"Descending"`).
    pub orbit_pass_direction: String,
    /// Raw `look_direction` string (e.g. `"Right"`).
    pub look_direction: String,
    /// Raw `burst_id` string (e.g. `"t144_308011_iw2"`).
    pub burst_id: String,
    /// Raw `zero_doppler_start_time` (`YYYY-MM-DD HH:MM:SS.ffffff`).
    pub zero_doppler_start_time: String,
    /// Raw `zero_doppler_end_time` (same format).
    pub zero_doppler_end_time: String,
}

/// `input_burst_metadata` scalars used for provenance.
#[derive(Debug, Clone, Copy)]
pub struct CslcBurstMetadata {
    /// Slant-range pixel spacing (m), `range_pixel_spacing`.
    pub range_pixel_spacing_m: f64,
    /// Azimuth line time interval (s), `azimuth_time_interval`.
    pub azimuth_time_interval_s: f64,
    /// Burst center `[lon, lat]` in degrees, `center`.
    pub center_lonlat_deg: [f64; 2],
}

/// `/metadata/orbit` state vectors.
#[derive(Debug, Clone)]
pub struct CslcOrbit {
    /// Seconds since `reference_epoch`, one per state vector.
    pub time_s: Vec<f64>,
    /// ECEF positions (m), `[x, y, z]` per state vector.
    pub position_m: Vec<[f64; 3]>,
    /// ECEF velocities (m/s), `[x, y, z]` per state vector.
    pub velocity_mps: Vec<[f64; 3]>,
    /// Raw `reference_epoch` datetime string (same format as zero-doppler times).
    pub reference_epoch: String,
}

/// Read the `/identification` provenance scalars.
///
/// # Errors
/// `Err` when the file or any key is missing/unreadable (e.g. cropped granules).
pub fn read_cslc_identification(path: &Path) -> Result<CslcIdentification> {
    let file = hdf5::File::open(path)?;
    Ok(CslcIdentification {
        orbit_pass_direction: read_string(&file, "/identification/orbit_pass_direction")?,
        look_direction: read_string(&file, "/identification/look_direction")?,
        burst_id: read_string(&file, "/identification/burst_id")?,
        zero_doppler_start_time: read_string(&file, "/identification/zero_doppler_start_time")?,
        zero_doppler_end_time: read_string(&file, "/identification/zero_doppler_end_time")?,
    })
}

/// Read the `input_burst_metadata` provenance scalars.
///
/// # Errors
/// `Err` when the file or any key is missing/unreadable.
pub fn read_cslc_burst_metadata(path: &Path) -> Result<CslcBurstMetadata> {
    const GROUP: &str = "/metadata/processing_information/input_burst_metadata";
    let file = hdf5::File::open(path)?;
    let center = file
        .dataset(&format!("{GROUP}/center"))?
        .read_raw::<f64>()?;
    let [lon, lat] = center.as_slice() else {
        return Err(crate::IoError::Shape(format!(
            "{GROUP}/center: expected [lon, lat], got {} values",
            center.len()
        )));
    };
    Ok(CslcBurstMetadata {
        range_pixel_spacing_m: file
            .dataset(&format!("{GROUP}/range_pixel_spacing"))?
            .read_scalar::<f64>()?,
        azimuth_time_interval_s: file
            .dataset(&format!("{GROUP}/azimuth_time_interval"))?
            .read_scalar::<f64>()?,
        center_lonlat_deg: [*lon, *lat],
    })
}

/// Read the `/metadata/orbit` state vectors.
///
/// # Errors
/// `Err` when the file or any key is missing/unreadable, or the component arrays
/// disagree in length.
pub fn read_cslc_orbit(path: &Path) -> Result<CslcOrbit> {
    let file = hdf5::File::open(path)?;
    let time_s = file.dataset("/metadata/orbit/time")?.read_raw::<f64>()?;
    let position_m = read_xyz(&file, "position", time_s.len())?;
    let velocity_mps = read_xyz(&file, "velocity", time_s.len())?;
    Ok(CslcOrbit {
        time_s,
        position_m,
        velocity_mps,
        reference_epoch: read_string(&file, "/metadata/orbit/reference_epoch")?,
    })
}

/// Read `/metadata/orbit/{name}_{x,y,z}` into per-vector `[x, y, z]` triples.
fn read_xyz(file: &hdf5::File, name: &str, expected_len: usize) -> Result<Vec<[f64; 3]>> {
    let read = |axis: &str| -> Result<Vec<f64>> {
        Ok(file
            .dataset(&format!("/metadata/orbit/{name}_{axis}"))?
            .read_raw::<f64>()?)
    };
    let (x, y, z) = (read("x")?, read("y")?, read("z")?);
    if x.len() != expected_len || y.len() != expected_len || z.len() != expected_len {
        return Err(crate::IoError::Shape(format!(
            "orbit {name} arrays disagree with time length {expected_len}"
        )));
    }
    Ok((0..expected_len).map(|i| [x[i], y[i], z[i]]).collect())
}

/// Read a fixed-ASCII HDF5 string scalar (OPERA metadata strings are `|S<n>`).
fn read_string(file: &hdf5::File, path: &str) -> Result<String> {
    let value = file.dataset(path)?.read_scalar::<FixedAscii<64>>()?;
    Ok(value.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture(path: &Path) {
        let f = hdf5::File::create(path).unwrap();
        let ident = f.create_group("identification").unwrap();
        for (key, value) in [
            ("orbit_pass_direction", "Descending"),
            ("look_direction", "Right"),
            ("burst_id", "t144_308011_iw2"),
            ("zero_doppler_start_time", "2022-11-02 14:00:26.983904"),
            ("zero_doppler_end_time", "2022-11-02 14:00:30.087794"),
        ] {
            let ascii = FixedAscii::<64>::from_ascii(value).unwrap();
            ident
                .new_dataset::<FixedAscii<64>>()
                .create(key)
                .unwrap()
                .write_scalar(&ascii)
                .unwrap();
        }
        let orbit = f.create_group("metadata/orbit").unwrap();
        orbit
            .new_dataset_builder()
            .with_data(&[0.0_f64, 10.0])
            .create("time")
            .unwrap();
        let xyz = [
            ("position_x", 7.0e6),
            ("position_y", 7.0e6 + 1.0),
            ("position_z", 7.0e6 + 2.0),
            ("velocity_x", 7.5e3),
            ("velocity_y", 7.5e3 + 1.0),
            ("velocity_z", 7.5e3 + 2.0),
        ];
        for (key, value) in xyz {
            orbit
                .new_dataset_builder()
                .with_data(&[value, value + 1.0])
                .create(key)
                .unwrap();
        }
        let epoch = FixedAscii::<64>::from_ascii("2022-10-31 14:00:26.983904").unwrap();
        orbit
            .new_dataset::<FixedAscii<64>>()
            .create("reference_epoch")
            .unwrap()
            .write_scalar(&epoch)
            .unwrap();
        let burst = f
            .create_group("metadata/processing_information/input_burst_metadata")
            .unwrap();
        burst
            .new_dataset::<f64>()
            .create("range_pixel_spacing")
            .unwrap()
            .write_scalar(&2.329_562)
            .unwrap();
        burst
            .new_dataset::<f64>()
            .create("azimuth_time_interval")
            .unwrap()
            .write_scalar(&0.002_055_556)
            .unwrap();
        burst
            .new_dataset_builder()
            .with_data(&[-119.302, 36.684])
            .create("center")
            .unwrap();
    }

    #[test]
    fn reads_provenance_metadata_groups() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_cslc_metadata_contract.h5");
        let _ = std::fs::remove_file(&path);
        write_fixture(&path);

        let ident = read_cslc_identification(&path).unwrap();
        assert_eq!(ident.orbit_pass_direction, "Descending");
        assert_eq!(ident.look_direction, "Right");
        assert_eq!(ident.burst_id, "t144_308011_iw2");
        assert_eq!(ident.zero_doppler_start_time, "2022-11-02 14:00:26.983904");

        let burst = read_cslc_burst_metadata(&path).unwrap();
        assert!((burst.range_pixel_spacing_m - 2.329_562).abs() < 1e-9);
        assert!((burst.center_lonlat_deg[1] - 36.684).abs() < 1e-9);

        let orbit = read_cslc_orbit(&path).unwrap();
        assert_eq!(orbit.time_s, vec![0.0, 10.0]);
        assert_eq!(orbit.position_m[1], [7.0e6 + 1.0, 7.0e6 + 2.0, 7.0e6 + 3.0]);
        assert_eq!(orbit.velocity_mps[0], [7.5e3, 7.5e3 + 1.0, 7.5e3 + 2.0]);
        assert_eq!(orbit.reference_epoch, "2022-10-31 14:00:26.983904");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_groups_error_cleanly() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_cslc_metadata_missing.h5");
        let _ = std::fs::remove_file(&path);
        hdf5::File::create(&path).unwrap();
        assert!(read_cslc_identification(&path).is_err());
        assert!(read_cslc_burst_metadata(&path).is_err());
        assert!(read_cslc_orbit(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
