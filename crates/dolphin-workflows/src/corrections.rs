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
use dolphin_corrections::geometry::{resolve_los_geometry, LosGeometry};
use dolphin_corrections::ionosphere::{read_ionex, vtec_to_range_delay, SPEED_OF_LIGHT};
use dolphin_corrections::subtract_delay;
use dolphin_corrections::troposphere::{
    ensure_finite_coverage, read_l4_netcdf_for_grid, read_l4_total_for_grid, resample_bilinear,
    warp_to_frame, DelayGrid,
};
use dolphin_io::{grid_centroid_lonlat, GeoInfo};
use ndarray::{Array2, Array3, Axis};

/// Per-date correction delay layers (meters, `(n_dates, rows, cols)`), returned
/// for the typed API and per-band COG output.
#[derive(Debug)]
pub struct CorrectionLayers {
    /// Ionospheric range delay, present when `ionosphere_files` were supplied.
    pub ionosphere: Option<Array3<f64>>,
    /// Tropospheric range delay, present when `troposphere_files` were supplied.
    pub troposphere: Option<Array3<f64>>,
    /// Per-pixel LOS geometry, present when `geometry_files` (CSLC-S1-STATIC) were
    /// supplied. Independent of the atmospheric terms — the front door for the GPS
    /// ground-truth harness's ENU→LOS projection. When present it also drives the
    /// per-pixel zenith→slant incidence for the iono/tropo delays.
    pub los_geometry: Option<LosGeometry>,
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
    let (bands, rows, cols) = disp_rad.dim();
    // LOS geometry is resolved independently of the atmospheric opt-in: a
    // geometry-only config (for the GPS harness) needs it even with no iono/tropo.
    let los_geometry = resolve_geometry(opts, epsg, gt, (rows, cols))?;
    if !opts.is_enabled() {
        return Ok(CorrectionLayers {
            ionosphere: None,
            troposphere: None,
            los_geometry,
        });
    }
    let wavelength =
        wavelength.context("atmospheric corrections require input_options.wavelength")?;
    let n_dates = bands + 1;
    let freq = SPEED_OF_LIGHT / wavelength;
    let los = los_geometry.as_ref();

    let geo = GeoInfo {
        epsg,
        geotransform: gt,
    };
    let ionosphere = build_ionosphere(opts, date_files, n_dates, (rows, cols), geo, freq, los)?;
    let troposphere = build_troposphere(opts, n_dates, (rows, cols), gt, epsg, los)?;

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
        los_geometry,
    })
}

/// Load + resolve per-pixel LOS geometry from the configured CSLC-S1-STATIC
/// granules (one per burst), or `None` when none are configured.
fn resolve_geometry(
    opts: &CorrectionOptions,
    epsg: u32,
    gt: [f64; 6],
    shape: (usize, usize),
) -> Result<Option<LosGeometry>> {
    if opts.geometry_files.is_empty() {
        return Ok(None);
    }
    let target_geo = GeoInfo {
        epsg,
        geotransform: gt,
    };
    let layers = opts
        .geometry_files
        .iter()
        .map(|p| {
            dolphin_io::geometry::read_los_layers_for_grid(p, "/data", target_geo, shape)
                .context("reading bounded CSLC-S1-STATIC geometry")
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok(Some(resolve_los_geometry(&layers, gt, epsg, shape)?))
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
    geo: GeoInfo,
    freq: f64,
    los: Option<&LosGeometry>,
) -> Result<Option<Array3<f64>>> {
    if opts.ionosphere_files.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        opts.ionosphere_files.len() == n_dates,
        "expected {n_dates} ionosphere_files (one per date), got {}",
        opts.ionosphere_files.len()
    );
    let (lon, lat) = grid_centroid_lonlat(geo.geotransform, rows, cols, geo.epsg)?;
    // Per-pixel incidence from geometry when supplied; else the scalar knob (whose
    // uniform layer is numerically identical to the pre-geometry `.fill(delay)`).
    let inc_grid = los.map(LosGeometry::incidence_deg);
    let mut out = Array3::<f64>::zeros((n_dates, rows, cols));
    for (t, ionex_path) in opts.ionosphere_files.iter().enumerate() {
        let utc_sec = date_files.get(t).map_or(43200.0, |p| acq_utc_sec(p));
        let content = std::fs::read_to_string(ionex_path)
            .with_context(|| format!("reading IONEX {}", ionex_path.display()))?;
        let maps = read_ionex(&content).map_err(anyhow::Error::msg)?;
        let vtec = maps.value(utc_sec, lat, lon);
        let layer = iono_delay_layer(
            vtec,
            freq,
            opts.incidence_angle_deg,
            inc_grid.as_ref(),
            (rows, cols),
        );
        out.index_axis_mut(Axis(0), t).assign(&layer);
    }
    Ok(Some(out))
}

/// One date's ionospheric delay layer: per-pixel incidence from `inc_grid` when
/// present, else the scalar `inc_scalar_deg` filled uniformly (bit-identical to the
/// pre-geometry path).
fn iono_delay_layer(
    vtec: f64,
    freq: f64,
    inc_scalar_deg: f64,
    inc_grid: Option<&Array2<f64>>,
    shape: (usize, usize),
) -> Array2<f64> {
    match inc_grid {
        Some(inc) => inc.mapv(|i| vtec_to_range_delay(vtec, i, freq)),
        None => Array2::from_elem(shape, vtec_to_range_delay(vtec, inc_scalar_deg, freq)),
    }
}

/// Build the per-date tropospheric delay grid by resampling each OPERA L4 netCDF
/// onto the frame grid.
fn build_troposphere(
    opts: &CorrectionOptions,
    n_dates: usize,
    (rows, cols): (usize, usize),
    gt: [f64; 6],
    epsg: u32,
    los: Option<&LosGeometry>,
) -> Result<Option<Array3<f64>>> {
    if opts.troposphere_files.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        opts.troposphere_files.len() == n_dates,
        "expected {n_dates} troposphere_files (one per date), got {}",
        opts.troposphere_files.len()
    );
    // ZTD is a zenith delay; project to line-of-sight by 1/cos(incidence) — per pixel
    // (1/up) from geometry when supplied, else the scalar knob (bit-identical fill).
    let slant = slant_grid(opts.incidence_angle_deg, los, (rows, cols));
    let mut out = Array3::<f64>::zeros((n_dates, rows, cols));
    for (t, nc) in opts.troposphere_files.iter().enumerate() {
        let grid = read_tropo_for_grid(nc, &opts.troposphere_variable, gt, epsg, (rows, cols))?;
        let band = resample_to_frame(&grid, gt, epsg, (rows, cols))?;
        out.index_axis_mut(Axis(0), t).assign(&(&band * &slant));
    }
    Ok(Some(out))
}

/// Per-pixel zenith→slant factor `1/cos(incidence)`: `1/up` from geometry when
/// present, else the scalar factor filled uniformly (numerically identical to the
/// pre-geometry `band * scalar_slant`).
fn slant_grid(
    inc_scalar_deg: f64,
    los: Option<&LosGeometry>,
    shape: (usize, usize),
) -> Array2<f64> {
    match los {
        Some(g) => g.up.mapv(|u| 1.0 / u),
        None => Array2::from_elem(shape, 1.0 / inc_scalar_deg.to_radians().cos()),
    }
}

/// Read a tropospheric delay grid: `"total"` sums the real OPERA L4
/// `hydrostatic_delay` + `wet_delay`, any other name reads that single variable.
fn read_tropo_for_grid(
    nc: &Path,
    var: &str,
    gt: [f64; 6],
    epsg: u32,
    shape: (usize, usize),
) -> Result<DelayGrid> {
    let grid = match var {
        "total" => read_l4_total_for_grid(nc, gt, epsg, shape),
        other => read_l4_netcdf_for_grid(nc, other, gt, epsg, shape),
    };
    grid.map_err(anyhow::Error::msg)
}

/// Resample a tropospheric delay grid onto the frame. When the source CRS matches
/// the frame this is the plain bilinear resample; when it differs (e.g. a global
/// EPSG:4326 OPERA L4 product onto a UTM frame) it reprojects via GDAL warp. With
/// no source CRS at all it fails closed.
fn resample_to_frame(
    grid: &DelayGrid,
    gt: [f64; 6],
    frame_epsg: u32,
    shape: (usize, usize),
) -> Result<ndarray::Array2<f64>> {
    let (rows, cols) = shape;
    match grid.epsg {
        Some(e) if e == frame_epsg => {
            let output = resample_bilinear(grid.data.view(), grid.geotransform, gt, (rows, cols));
            ensure_finite_coverage(&output).map_err(anyhow::Error::msg)?;
            Ok(output)
        }
        _ if grid.srs_wkt.is_some() || grid.epsg.is_some() => {
            let output =
                warp_to_frame(grid, gt, frame_epsg, (rows, cols)).map_err(anyhow::Error::msg)?;
            ensure_finite_coverage(&output).map_err(anyhow::Error::msg)?;
            Ok(output)
        }
        _ => Err(dolphin_corrections::CorrectionError::NoSourceCrs.into()),
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
    use dolphin_io::read_los_layers;
    use ndarray::Array3;

    /// Write a single-band EPSG:4326 OPERA-L4-format netCDF (variable `Band1`).
    fn write_4326_netcdf(path: &Path, field: &ndarray::Array2<f64>, gt: [f64; 6]) {
        use gdal::raster::Buffer;
        use gdal::spatial_ref::SpatialRef;
        use gdal::DriverManager;
        let (rows, cols) = field.dim();
        let mem = DriverManager::get_driver_by_name("MEM").unwrap();
        let mut src = mem
            .create_with_band_type::<f64, _>("", cols, rows, 1)
            .unwrap();
        src.set_geo_transform(&gt).unwrap();
        src.set_spatial_ref(&SpatialRef::from_epsg(4326).unwrap())
            .unwrap();
        {
            let mut band = src.rasterband(1).unwrap();
            let mut buf = Buffer::new((cols, rows), field.iter().copied().collect());
            band.write((0, 0), (cols, rows), &mut buf).unwrap();
        }
        let nc = DriverManager::get_driver_by_name("netCDF").unwrap();
        src.create_copy(&nc, path, &Default::default()).unwrap();
    }

    /// End-to-end (Phase 1): two synthesized **4326** OPERA-L4 netCDFs resampled
    /// through `build_troposphere` onto a **UTM 32610** frame land the analytic
    /// per-date zenith delay at known frame pixels — the warp dispatch, proven
    /// through the pipeline stage, not just the bare warp fn.
    #[test]
    fn build_troposphere_warps_4326_onto_utm_frame() {
        use gdal::spatial_ref::{AxisMappingStrategy, CoordTransform, SpatialRef};

        let tmp = std::env::temp_dir();
        let f0 = tmp.join("dolphin_tropo_warp_d0.nc");
        let f1 = tmp.join("dolphin_tropo_warp_d1.nc");
        // date0 = 1.0 constant; date1 = 1.0 + g(lon,lat), g linear in (lon,lat).
        let g = |lon: f64, lat: f64| 0.10 * (lon + 123.0) + 0.05 * (lat - 38.0);
        let src_gt = [-124.0, 0.1, 0.0, 39.0, 0.0, -0.1];
        let d0 = ndarray::Array2::<f64>::from_elem((21, 21), 1.0);
        let d1 = ndarray::Array2::from_shape_fn((21, 21), |(r, col)| {
            let lon = src_gt[0] + (col as f64 + 0.5) * src_gt[1];
            let lat = src_gt[3] + (r as f64 + 0.5) * src_gt[5];
            1.0 + g(lon, lat)
        });
        write_4326_netcdf(&f0, &d0, src_gt);
        write_4326_netcdf(&f1, &d1, src_gt);

        let (rows, cols) = (5_usize, 5_usize);
        let dst_gt = [495_000.0, 2_000.0, 0.0, 4_211_000.0, 0.0, -2_000.0];
        let opts = CorrectionOptions {
            troposphere_files: vec![f0.clone(), f1.clone()],
            troposphere_variable: "Band1".to_string(),
            incidence_angle_deg: 0.0, // slant = 1, so zenith delay lands unscaled
            ..Default::default()
        };
        let layers = build_troposphere(&opts, 2, (rows, cols), dst_gt, 32610, None)
            .unwrap()
            .expect("troposphere layers present");

        let mut utm = SpatialRef::from_epsg(32610).unwrap();
        let mut wgs = SpatialRef::from_epsg(4326).unwrap();
        utm.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
        wgs.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
        let ct = CoordTransform::new(&utm, &wgs).unwrap();
        for (r, cc) in [(0_usize, 0_usize), (2, 2), (4, 4)] {
            let x = dst_gt[0] + (cc as f64 + 0.5) * dst_gt[1];
            let y = dst_gt[3] + (r as f64 + 0.5) * dst_gt[5];
            let (mut xs, mut ys, mut zs) = ([x], [y], []);
            ct.transform_coords(&mut xs, &mut ys, &mut zs).unwrap();
            let expected = 1.0 + g(xs[0], ys[0]);
            assert!(
                (layers[(0, r, cc)] - 1.0).abs() < 5e-3,
                "date0 should be 1.0"
            );
            assert!(
                (layers[(1, r, cc)] - expected).abs() < 5e-3,
                "date1 ({r},{cc}): {} vs {expected}",
                layers[(1, r, cc)]
            );
        }
        let _ = std::fs::remove_file(&f0);
        let _ = std::fs::remove_file(&f1);
    }

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

    /// Serialize HDF5 access across parallel tests in this crate's test binary —
    /// `hdf5-metno` is not thread-safe (mirrors `dolphin_io`'s own test lock, which
    /// is `pub(crate)` and so unreachable from here).
    static HDF5_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn hdf5_guard() -> std::sync::MutexGuard<'static, ()> {
        HDF5_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Write a minimal CSLC-S1-STATIC HDF5 with *uniform* LOS (incidence θ, az 30°)
    /// so a per-pixel resolve reproduces the scalar-incidence path.
    fn write_uniform_static(path: &Path, inc_deg: f64, gt: [f64; 6], shape: (usize, usize)) {
        let (rows, cols) = shape;
        let inc = inc_deg.to_radians();
        let az = 30.0_f64.to_radians();
        let east = ndarray::Array2::from_elem((rows, cols), (-inc.sin() * az.sin()) as f32);
        let north = ndarray::Array2::from_elem((rows, cols), (-inc.sin() * az.cos()) as f32);
        let x: Vec<f64> = (0..cols)
            .map(|c| gt[0] + (c as f64 + 0.5) * gt[1])
            .collect();
        let y: Vec<f64> = (0..rows)
            .map(|r| gt[3] + (r as f64 + 0.5) * gt[5])
            .collect();
        let _ = std::fs::remove_file(path);
        let f = hdf5::File::create(path).unwrap();
        let g = f.create_group("data").unwrap();
        g.new_dataset_builder()
            .with_data(&east)
            .create("los_east")
            .unwrap();
        g.new_dataset_builder()
            .with_data(&north)
            .create("los_north")
            .unwrap();
        g.new_dataset_builder()
            .with_data(&x)
            .create("x_coordinates")
            .unwrap();
        g.new_dataset_builder()
            .with_data(&y)
            .create("y_coordinates")
            .unwrap();
        g.new_dataset::<i64>()
            .create("projection")
            .unwrap()
            .write_scalar(&32610_i64)
            .unwrap();
    }

    /// Bar #3: a STATIC product encoding a *uniform* incidence θ, driving the
    /// per-pixel iono+tropo path, reproduces the scalar `incidence_angle_deg = θ`
    /// path to the LOS **f32 quantization** floor (~5e-8; the product stores
    /// los_east/north as float32, so this is the tightest honest bound — not 1e-9,
    /// and NOT literal bit-equality). The exact-to-roundoff invariant is the *None*
    /// path (`from_elem(scalar)` == the old `fill(scalar)`), covered by
    /// `disabled_is_noop` / `build_troposphere_warps_*`. This test guards the *Some*
    /// path: a future "simplify" that breaks the geometry derivation blows 1e-6.
    #[test]
    fn uniform_geometry_matches_scalar_path() {
        let _hdf5 = hdf5_guard();
        let theta = 38.5_f64;
        let gt = [500_000.0, 60.0, 0.0, 4_000_000.0, 0.0, -60.0];
        let path = std::env::temp_dir().join("dolphin_static_uniform_bar3.h5");
        write_uniform_static(&path, theta, gt, (4, 4));
        let layers = vec![read_los_layers(&path, "/data").unwrap()];
        let los = resolve_los_geometry(&layers, gt, 32610, (4, 4)).unwrap();

        // Tropo slant: per-pixel 1/up vs scalar 1/cos(theta).
        let scalar = 1.0 / theta.to_radians().cos();
        let per_pixel = slant_grid(theta, Some(&los), (4, 4));
        for &v in per_pixel.iter() {
            assert!((v - scalar).abs() < 1e-6, "slant {v} vs {scalar}");
        }
        // Iono delay: per-pixel incidence vs scalar, at a representative TEC/freq.
        let (vtec, freq) = (25.0, SPEED_OF_LIGHT / 0.055);
        let inc_grid = los.incidence_deg();
        let pp = iono_delay_layer(vtec, freq, theta, Some(&inc_grid), (4, 4));
        let sc = iono_delay_layer(vtec, freq, theta, None, (4, 4));
        for (a, b) in pp.iter().zip(sc.iter()) {
            let tol = 1e-6 * b.abs().max(1.0);
            assert!((a - b).abs() < tol, "iono {a} vs {b}");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Bar #5: a geometry-only config (no iono/tropo, no wavelength) still resolves
    /// and returns `LosGeometry`, leaving displacement untouched — proves the gate
    /// decoupling from `is_enabled()`/wavelength.
    #[test]
    fn geometry_only_config_resolves_without_wavelength() {
        let _hdf5 = hdf5_guard();
        let gt = [500_000.0, 60.0, 0.0, 4_000_000.0, 0.0, -60.0];
        let path = std::env::temp_dir().join("dolphin_static_geom_only_bar5.h5");
        write_uniform_static(&path, 34.0, gt, (3, 3));
        let opts = CorrectionOptions {
            geometry_files: vec![path.clone()],
            ..Default::default()
        };
        let mut disp = Array3::from_shape_fn((2, 3, 3), |(t, r, c)| (t + r + c) as f64);
        let original = disp.clone();
        let layers = apply_corrections(&opts, None, &mut disp, &[], 32610, gt).unwrap();

        assert!(layers.ionosphere.is_none() && layers.troposphere.is_none());
        let los = layers.los_geometry.expect("geometry present");
        assert!((los.incidence_deg()[(1, 1)] - 34.0).abs() < 1e-3);
        assert_eq!(disp, original, "geometry-only must not touch displacement");
        let _ = std::fs::remove_file(&path);
    }

    /// A missing geometry file surfaces a contextual error (not a panic), naming the
    /// offending path — the `resolve_geometry` read-error path.
    #[test]
    fn missing_geometry_file_errors_with_context() {
        let opts = CorrectionOptions {
            geometry_files: vec![PathBuf::from("/nonexistent/static_geometry.h5")],
            ..Default::default()
        };
        let mut disp = Array3::<f64>::zeros((1, 2, 2));
        let err = apply_corrections(&opts, None, &mut disp, &[], 32610, [0.0; 6]).unwrap_err();
        assert!(
            err.to_string().contains("CSLC-S1-STATIC geometry")
                || format!("{err:#}").contains("static_geometry.h5"),
            "expected contextual geometry read error, got: {err:#}"
        );
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
