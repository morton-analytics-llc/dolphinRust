//! Atmospheric-correction stage: build per-date ionospheric + tropospheric range
//! delays (meters) on the frame grid and subtract them from the inverted
//! displacement time series, **before velocity** (per the pipeline contract).
//!
//! Opt-in: with no correction files configured this is a no-op and the output is
//! bit-identical to the uncorrected run. Enabling it requires
//! `input_options.wavelength` (the ionospheric delay is `1/f²`-scaled to the
//! configured carrier; the meters→phase conversion needs λ).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dolphin_core::config::CorrectionOptions;
use dolphin_corrections::ionosphere::{read_ionex, vtec_to_range_delay, SPEED_OF_LIGHT};
use dolphin_corrections::subtract_delay;
use dolphin_corrections::troposphere::{
    read_l4_netcdf, read_l4_total, resample_bilinear, DelayGrid,
};
use dolphin_io::grid_centroid_lonlat;
use ndarray::{Array3, Axis};

/// Per-date correction delay layers (meters, `(n_dates, rows, cols)`), returned
/// for the typed API and per-band COG output.
#[derive(Debug)]
pub struct CorrectionLayers {
    /// Ionospheric range delay, present when `ionosphere_files` were supplied.
    pub ionosphere: Option<Array3<f64>>,
    /// Tropospheric range delay, present when `troposphere_files` were supplied.
    pub troposphere: Option<Array3<f64>>,
}

impl CorrectionLayers {
    /// No corrections applied (the opt-out default).
    fn none() -> Self {
        Self {
            ionosphere: None,
            troposphere: None,
        }
    }
}

/// Build and subtract the configured corrections from `disp_rad` in place.
///
/// `date_files` are the per-date input granules (one per acquisition, in date
/// order) used to time-stamp the IONEX lookup; `epsg`/`gt` georeference the frame
/// grid. Returns the per-date delay layers for output.
///
/// # Errors
/// Returns `Err` if corrections are enabled without a wavelength, if a correction
/// file count does not match the acquisition count, or on read/subtract failure.
pub fn apply_corrections(
    opts: &CorrectionOptions,
    wavelength: Option<f64>,
    disp_rad: &mut Array3<f64>,
    date_files: &[PathBuf],
    epsg: u32,
    gt: [f64; 6],
) -> Result<CorrectionLayers> {
    if !opts.is_enabled() {
        return Ok(CorrectionLayers::none());
    }
    let wavelength =
        wavelength.context("atmospheric corrections require input_options.wavelength")?;
    let (bands, rows, cols) = disp_rad.dim();
    let n_dates = bands + 1;
    let freq = SPEED_OF_LIGHT / wavelength;

    let ionosphere = build_ionosphere(opts, date_files, n_dates, (rows, cols), gt, epsg, freq)?;
    let troposphere = build_troposphere(opts, n_dates, (rows, cols), gt, epsg)?;

    let total = sum_layers(
        ionosphere.as_ref(),
        troposphere.as_ref(),
        n_dates,
        (rows, cols),
    );
    subtract_delay(disp_rad, total.view(), wavelength)?;
    Ok(CorrectionLayers {
        ionosphere,
        troposphere,
    })
}

/// Sum the present delay layers (either may be absent) into one `(n_dates, rows,
/// cols)` total.
fn sum_layers(
    iono: Option<&Array3<f64>>,
    tropo: Option<&Array3<f64>>,
    n_dates: usize,
    (rows, cols): (usize, usize),
) -> Array3<f64> {
    let mut total = Array3::<f64>::zeros((n_dates, rows, cols));
    if let Some(a) = iono {
        total += a;
    }
    if let Some(a) = tropo {
        total += a;
    }
    total
}

/// Build the per-date ionospheric delay grid from IONEX TEC maps. IONEX is coarse
/// (2.5°×5°), so VTEC is sampled once at the frame centre per date and projected
/// to a uniform LOS range delay via the configured incidence angle and carrier.
fn build_ionosphere(
    opts: &CorrectionOptions,
    date_files: &[PathBuf],
    n_dates: usize,
    (rows, cols): (usize, usize),
    gt: [f64; 6],
    epsg: u32,
    freq: f64,
) -> Result<Option<Array3<f64>>> {
    if opts.ionosphere_files.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        opts.ionosphere_files.len() == n_dates,
        "expected {n_dates} ionosphere_files (one per date), got {}",
        opts.ionosphere_files.len()
    );
    let (lon, lat) = grid_centroid_lonlat(gt, rows, cols, epsg)?;
    let mut out = Array3::<f64>::zeros((n_dates, rows, cols));
    for (t, ionex_path) in opts.ionosphere_files.iter().enumerate() {
        let utc_sec = date_files.get(t).map_or(43200.0, |p| acq_utc_sec(p));
        let content = std::fs::read_to_string(ionex_path)
            .with_context(|| format!("reading IONEX {}", ionex_path.display()))?;
        let maps = read_ionex(&content).map_err(anyhow::Error::msg)?;
        let vtec = maps.value(utc_sec, lat, lon);
        let delay = vtec_to_range_delay(vtec, opts.incidence_angle_deg, freq);
        out.index_axis_mut(Axis(0), t).fill(delay);
    }
    Ok(Some(out))
}

/// Build the per-date tropospheric delay grid by resampling each OPERA L4 netCDF
/// onto the frame grid.
fn build_troposphere(
    opts: &CorrectionOptions,
    n_dates: usize,
    (rows, cols): (usize, usize),
    gt: [f64; 6],
    epsg: u32,
) -> Result<Option<Array3<f64>>> {
    if opts.troposphere_files.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        opts.troposphere_files.len() == n_dates,
        "expected {n_dates} troposphere_files (one per date), got {}",
        opts.troposphere_files.len()
    );
    // ZTD is a zenith delay; project to line-of-sight by 1/cos(incidence), the
    // same geometry the ionospheric term uses.
    let slant = 1.0 / opts.incidence_angle_deg.to_radians().cos();
    let mut out = Array3::<f64>::zeros((n_dates, rows, cols));
    for (t, nc) in opts.troposphere_files.iter().enumerate() {
        let grid = read_tropo(nc, &opts.troposphere_variable)?;
        check_epsg(grid.epsg, epsg, nc);
        let band = resample_bilinear(grid.data.view(), grid.geotransform, gt, (rows, cols));
        out.index_axis_mut(Axis(0), t).assign(&(band * slant));
    }
    Ok(Some(out))
}

/// Read a tropospheric delay grid: `"total"` sums the real OPERA L4
/// `hydrostatic_delay` + `wet_delay`, any other name reads that single variable.
fn read_tropo(nc: &Path, var: &str) -> Result<DelayGrid> {
    let grid = match var {
        "total" => read_l4_total(nc),
        other => read_l4_netcdf(nc, other),
    };
    grid.map_err(anyhow::Error::msg)
}

/// Warn (don't fail) if a tropospheric product carries a CRS differing from the
/// frame; resampling assumes a shared CRS.
fn check_epsg(src: Option<u32>, frame: u32, path: &Path) {
    if matches!(src, Some(e) if e != frame) {
        tracing::warn!(
            file = %path.display(),
            src_epsg = src,
            frame_epsg = frame,
            "tropospheric product CRS differs from frame; resampling assumes a shared CRS"
        );
    }
}

/// Seconds-of-day from a granule name's `…YYYYMMDDThhmmss…` stamp, else noon
/// (43200 s) when no time token is present (e.g. a date-only synthetic name).
fn acq_utc_sec(path: &Path) -> f64 {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return 43200.0;
    };
    let chars: Vec<char> = name.chars().collect();
    chars
        .windows(15)
        .find_map(parse_time_token)
        .unwrap_or(43200.0)
}

/// Parse `YYYYMMDDThhmmss` (15 chars) → seconds of day, if the window matches.
fn parse_time_token(w: &[char]) -> Option<f64> {
    if w[8] != 'T'
        || w.iter()
            .enumerate()
            .any(|(i, c)| i != 8 && !c.is_ascii_digit())
    {
        return None;
    }
    let n =
        |a: usize, b: usize| -> f64 { w[a..b].iter().collect::<String>().parse().unwrap_or(0.0) };
    Some(n(9, 11) * 3600.0 + n(11, 13) * 60.0 + n(13, 15))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Corrections off → no layers, displacement untouched (DoD #1).
    #[test]
    fn disabled_is_noop() {
        let opts = CorrectionOptions::default();
        let mut disp = Array3::from_shape_fn((2, 2, 2), |(t, r, c)| (t + r + c) as f64);
        let original = disp.clone();
        let layers = apply_corrections(
            &opts,
            Some(0.24),
            &mut disp,
            &[],
            32610,
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        )
        .unwrap();
        assert!(layers.ionosphere.is_none() && layers.troposphere.is_none());
        assert_eq!(disp, original);
    }

    /// Enabled without a wavelength is an error (can't convert meters → phase).
    #[test]
    fn enabled_without_wavelength_errors() {
        let opts = CorrectionOptions {
            troposphere_files: vec![PathBuf::from("/x.nc")],
            ..Default::default()
        };
        let mut disp = Array3::<f64>::zeros((1, 1, 1));
        let err = apply_corrections(&opts, None, &mut disp, &[], 32610, [0.0; 6]).unwrap_err();
        assert!(err.to_string().contains("wavelength"));
    }

    /// Time token parsing: OPERA-style stamp → seconds of day; date-only → noon.
    #[test]
    fn parses_acquisition_time() {
        let opera = Path::new("OPERA_L2_CSLC-S1_T027_20230914T132417Z_x.h5");
        let want = 13.0 * 3600.0 + 24.0 * 60.0 + 17.0;
        assert!((acq_utc_sec(opera) - want).abs() < 1e-9);
        assert!((acq_utc_sec(Path::new("cslc_20221119.h5")) - 43200.0).abs() < 1e-9);
    }
}
