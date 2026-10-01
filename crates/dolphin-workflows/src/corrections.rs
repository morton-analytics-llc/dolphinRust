//! Atmospheric-correction stage: build per-date apparent LOS corrections
//! (meters, positive toward the sensor) on the frame grid and subtract them
//! from the inverted displacement time series, **before velocity**.
//!
//! Opt-in: with no correction files configured this is a no-op and the output is
//! bit-identical to the uncorrected run. Enabling it requires
//! `input_options.wavelength` (the ionospheric delay is `1/f²`-scaled to the
//! configured carrier; the meters→phase conversion needs λ).

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use dolphin_core::config::CorrectionOptions;
use dolphin_corrections::geometry::{
    resolve_los_geometry_with_options, LosCoverageOptions, LosGeometry,
    DEFAULT_MAX_OUTSIDE_STATIC_FRACTION,
};
use dolphin_corrections::ionosphere::{read_ionex, vtec_to_range_delay, SPEED_OF_LIGHT};
use dolphin_corrections::plate_motion::{plate_motion_range_delay_rate_grid, resolve_euler_pole};
use dolphin_corrections::solid_earth_tide::{tide_range_delay_grid, LonLatGrid};
use dolphin_corrections::subtract_delay;
use dolphin_corrections::troposphere::{
    ensure_finite_coverage, height_levels, read_l4_level_for_grid, read_l4_netcdf_for_grid,
    read_l4_total_for_grid, read_raster_for_grid, resample_bilinear, warp_to_frame, DelayGrid,
};
use dolphin_io::{grid_centroid_lonlat, GeoInfo};
use ndarray::{Array2, Array3, ArrayView2, Axis};

/// Per-date apparent LOS layers (meters toward sensor, `(n_dates, rows, cols)`), returned
/// for the typed API and per-band COG output.
#[derive(Debug)]
pub struct CorrectionLayers {
    /// Ionospheric phase advance in apparent LOS meters, present when
    /// `ionosphere_files` were supplied.
    pub ionosphere: Option<Array3<f64>>,
    /// Negative tropospheric path excess in apparent LOS meters, present when
    /// timed or per-date tropospheric inputs were supplied.
    pub troposphere: Option<Array3<f64>>,
    /// Per-pixel LOS geometry, present when `geometry_files` (CSLC-S1-STATIC) were
    /// supplied. Independent of the atmospheric terms — the front door for the GPS
    /// ground-truth harness's ENU→LOS projection. When present it also drives the
    /// per-pixel zenith→slant incidence for the iono/tropo delays.
    pub los_geometry: Option<LosGeometry>,
    /// Solid-earth-tide displacement projected toward the sensor, present when
    /// `correction_options.solid_earth_tide` is set.
    pub solid_earth_tide: Option<Array3<f64>>,
    /// Rigid-plate-motion equivalent range delay, present when
    /// `correction_options.plate_motion_model` is set.
    pub plate_motion: Option<Array3<f64>>,
}

/// Build and subtract the configured corrections from `disp_rad` in place.
///
/// `date_files` are the per-date input granules (one per acquisition, in date
/// order) used to time-stamp the IONEX lookup; `epsg`/`gt` georeference the frame
/// grid. `support` is the validity mask: terrain and geometry must be finite
/// there, and may be nodata anywhere else. `None` requires the whole grid.
/// Returns the per-date delay layers for output.
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
    support: Option<ArrayView2<'_, bool>>,
) -> Result<CorrectionLayers> {
    let (bands, rows, cols) = disp_rad.dim();
    ensure!(
        support.is_none_or(|mask| mask.dim() == (rows, cols)),
        "correction support shape differs from the displacement grid"
    );
    // LOS geometry is resolved independently of the atmospheric opt-in: a
    // geometry-only config (for the GPS harness) needs it even with no iono/tropo.
    let los_geometry = resolve_geometry(opts, epsg, gt, (rows, cols))?;
    if !opts.is_enabled() {
        return Ok(CorrectionLayers {
            ionosphere: None,
            troposphere: None,
            solid_earth_tide: None,
            plate_motion: None,
            los_geometry,
        });
    }
    let wavelength =
        wavelength.context("atmospheric corrections require input_options.wavelength")?;
    let n_dates = bands + 1;
    ensure!(
        opts.acquisition_utc.is_empty() || opts.acquisition_utc.len() == n_dates,
        "explicit acquisition UTC must cover every correction epoch"
    );
    let freq = SPEED_OF_LIGHT / wavelength;
    let los = los_geometry.as_ref();

    let geo = GeoInfo {
        epsg,
        geotransform: gt,
    };
    let ionosphere = build_ionosphere(opts, date_files, n_dates, (rows, cols), geo, freq, los)?;
    let troposphere = build_troposphere(opts, n_dates, (rows, cols), gt, epsg, los, support)?;
    let solid_earth_tide = build_solid_earth_tide(opts, date_files, (rows, cols), geo, los)?;
    let plate_motion = build_plate_motion(opts, date_files, (rows, cols), geo, los)?;

    let total = sum_layers(
        [
            ionosphere.as_ref(),
            troposphere.as_ref(),
            solid_earth_tide.as_ref(),
            plate_motion.as_ref(),
        ],
        n_dates,
        (rows, cols),
    );
    subtract_delay(disp_rad, total.view(), wavelength)?;
    Ok(CorrectionLayers {
        ionosphere,
        troposphere,
        solid_earth_tide,
        plate_motion,
        los_geometry,
    })
}

/// Apply per-group acquisition times using the same date-wise source ownership as stitching.
pub(crate) fn apply_corrections_with_ownership(
    groups: &[(u32, CorrectionOptions, Vec<PathBuf>)],
    wavelength: Option<f64>,
    displacement: &mut Array3<f64>,
    ownership: ndarray::ArrayView3<'_, u32>,
    geo: GeoInfo,
    support: Option<ArrayView2<'_, bool>>,
) -> Result<CorrectionLayers> {
    let (bands, rows, cols) = displacement.dim();
    ensure!(
        ownership.dim() == (bands + 1, rows, cols)
            && support.is_none_or(|mask| mask.dim() == (rows, cols)),
        "correction ownership or support shape mismatch"
    );
    let opts = &groups
        .first()
        .context("missing correction spatial groups")?
        .1;
    let wavelength = wavelength.context("corrections require wavelength")?;
    let los_geometry = resolve_geometry(opts, geo.epsg, geo.geotransform, (rows, cols))?;
    let mut combined = CorrectionLayers {
        ionosphere: None,
        troposphere: None,
        solid_earth_tide: None,
        plate_motion: None,
        los_geometry,
    };
    for (owner, options, files) in groups {
        let Some((r0, c0, r1, c1)) = owner_bounds(ownership, *owner, rows, cols) else {
            continue;
        };
        ensure!(
            options.acquisition_utc.len() == bands + 1,
            "spatial group acquisition UTC axis mismatch"
        );
        let mut gt = geo.geotransform;
        gt[0] += c0 as f64 * gt[1] + r0 as f64 * gt[2];
        gt[3] += c0 as f64 * gt[4] + r0 as f64 * gt[5];
        let local_geo = GeoInfo {
            epsg: geo.epsg,
            geotransform: gt,
        };
        let shape = (r1 - r0, c1 - c0);
        let local_los = combined.los_geometry.as_ref().map(|los| LosGeometry {
            east: los.east.slice(ndarray::s![r0..r1, c0..c1]).to_owned(),
            north: los.north.slice(ndarray::s![r0..r1, c0..c1]).to_owned(),
            up: los.up.slice(ndarray::s![r0..r1, c0..c1]).to_owned(),
        });
        let ionosphere = build_ionosphere(
            options,
            files,
            bands + 1,
            shape,
            local_geo,
            SPEED_OF_LIGHT / wavelength,
            local_los.as_ref(),
        )?;
        let local_support = support
            .as_ref()
            .map(|m| m.slice(ndarray::s![r0..r1, c0..c1]));
        let troposphere = build_troposphere(
            options,
            bands + 1,
            shape,
            gt,
            geo.epsg,
            local_los.as_ref(),
            local_support,
        )?;
        let tide = build_solid_earth_tide(options, files, shape, local_geo, local_los.as_ref())?;
        let plate_motion =
            build_plate_motion(options, files, shape, local_geo, local_los.as_ref())?;
        for (destination, source) in [
            (&mut combined.ionosphere, ionosphere),
            (&mut combined.troposphere, troposphere),
            (&mut combined.solid_earth_tide, tide),
            (&mut combined.plate_motion, plate_motion),
        ] {
            if let Some(source) = source {
                let destination = destination
                    .get_or_insert_with(|| Array3::from_elem((bands + 1, rows, cols), f64::NAN));
                for ((date, row, col), value) in source.indexed_iter() {
                    if ownership[(date, row + r0, col + c0)] == *owner {
                        destination[(date, row + r0, col + c0)] = *value;
                    }
                }
            }
        }
    }
    let total = sum_layers(
        [
            combined.ionosphere.as_ref(),
            combined.troposphere.as_ref(),
            combined.solid_earth_tide.as_ref(),
            combined.plate_motion.as_ref(),
        ],
        bands + 1,
        (rows, cols),
    );
    subtract_delay(displacement, total.view(), wavelength)?;
    Ok(combined)
}

/// The `(row_start, col_start, row_stop, col_stop)` bounding box of the pixels
/// `owner` occupies in `ownership` (across every date band), or `None` if it
/// owns nothing in this frame.
fn owner_bounds(
    ownership: ndarray::ArrayView3<'_, u32>,
    owner: u32,
    rows: usize,
    cols: usize,
) -> Option<(usize, usize, usize, usize)> {
    let mut bounds = (rows, cols, 0, 0);
    for ((_, row, col), &value) in ownership.indexed_iter() {
        if value == owner {
            bounds.0 = bounds.0.min(row);
            bounds.1 = bounds.1.min(col);
            bounds.2 = bounds.2.max(row + 1);
            bounds.3 = bounds.3.max(col + 1);
        }
    }
    let (r0, c0, r1, c1) = bounds;
    (r0 < r1 && c0 < c1).then_some(bounds)
}

/// Resolve the configured LOS geometry and discard it, purely to fail early.
///
/// Geometry is otherwise resolved in the corrections stage, which runs *after*
/// unwrapping and inversion. A frame the supplied CSLC-S1-STATIC granules do not
/// cover therefore surfaced ~90 minutes into a real 52-date run, having already
/// paid for every expensive stage. The check costs a couple of warps.
///
/// # Errors
/// Returns the same [`CorrectionError::GeometryCoverage`] /
/// [`CorrectionError::GeometryOverlapMismatch`] the corrections stage would.
pub fn verify_geometry_coverage(
    opts: &CorrectionOptions,
    epsg: u32,
    gt: [f64; 6],
    shape: (usize, usize),
) -> Result<()> {
    resolve_geometry(opts, epsg, gt, shape)
        .context("line-of-sight geometry precheck (before unwrapping)")?;
    Ok(())
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
            if let Some(group) = &opts.nisar_geometry_group {
                dolphin_io::geometry::read_nisar_los_layers_for_grid(
                    p,
                    group,
                    target_geo,
                    shape,
                    opts.nisar_ellipsoidal_dem_file.as_deref(),
                )
                .context("reading bounded NISAR LOS cube geometry")
            } else {
                dolphin_io::geometry::read_los_layers_for_grid(p, "/data", target_geo, shape)
                    .context("reading bounded CSLC-S1-STATIC geometry")
            }
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let coverage = LosCoverageOptions {
        max_outside_static_fraction: opts
            .max_outside_static_fraction
            .unwrap_or(DEFAULT_MAX_OUTSIDE_STATIC_FRACTION),
    };
    Ok(Some(resolve_los_geometry_with_options(
        &layers, gt, epsg, shape, coverage,
    )?))
}

/// Sum the present delay layers (any may be absent) into one `(n_dates, rows,
/// cols)` total.
fn sum_layers<const N: usize>(
    layers: [Option<&Array3<f64>>; N],
    n_dates: usize,
    (rows, cols): (usize, usize),
) -> Array3<f64> {
    layers.iter().flatten().fold(
        Array3::<f64>::zeros((n_dates, rows, cols)),
        |mut total, layer| {
            total += *layer;
            total
        },
    )
}

/// Per-date solid-earth-tide apparent LOS displacement (meters toward sensor).
///
/// Needs no external data file — only the acquisition UTC from each granule name
/// and the per-pixel LOS geometry. Both are hard requirements rather than
/// fallbacks: the tide is semidiurnal, so a defaulted acquisition time would be
/// wrong by up to half a cycle, and the tide is a 3-D vector whose LOS projection
/// needs the full unit vector, which the scalar `incidence_angle_deg` cannot give.
fn build_solid_earth_tide(
    opts: &CorrectionOptions,
    date_files: &[PathBuf],
    (rows, cols): (usize, usize),
    geo: GeoInfo,
    los: Option<&LosGeometry>,
) -> Result<Option<Array3<f64>>> {
    if !opts.solid_earth_tide {
        return Ok(None);
    }
    let los = los.context(
        "correction_options.solid_earth_tide requires geometry_files: projecting a 3-D \
         tidal displacement into line of sight needs the full LOS unit vector, which the \
         scalar incidence_angle_deg cannot supply",
    )?;
    let corners = dolphin_io::grid_corner_lonlat(geo.geotransform, rows, cols, geo.epsg)?;
    let lonlat = LonLatGrid::from_corners(corners, rows, cols);
    let mut out = Array3::<f64>::zeros((date_files.len(), rows, cols));
    for (t, path) in date_files.iter().enumerate() {
        let utc = opts
            .acquisition_utc
            .get(t)
            .map(chrono::DateTime::naive_utc)
            .or_else(|| acq_utc_datetime(path))
            .with_context(|| {
                format!(
                    "correction_options.solid_earth_tide needs each granule's acquisition time; \
                 {} carries no YYYYMMDDThhmmss token. The tide is semidiurnal, so defaulting \
                 the time would be wrong by up to half a cycle",
                    path.display()
                )
            })?;
        out.index_axis_mut(Axis(0), t)
            .assign(&tide_range_delay_grid(utc, &lonlat, los).mapv(|range| -range));
    }
    Ok(Some(out))
}

/// Per-date rigid-plate-motion equivalent range delay (meters) on the frame
/// grid.
///
/// Needs no external data file — only each acquisition's UTC (for elapsed
/// time since acquisition 0) and the per-pixel LOS geometry (for projecting
/// the plate's rigid velocity into line of sight), matching
/// `build_solid_earth_tide`'s requirements and for the same reason: a 3-D
/// rigid-rotation velocity cannot be projected from a scalar incidence.
/// Unlike the tide, the per-pixel term (`plate_motion_range_delay_rate_grid`)
/// depends only on position, not time, so it is computed once and scaled by
/// each date's elapsed time relative to acquisition 0.
fn build_plate_motion(
    opts: &CorrectionOptions,
    date_files: &[PathBuf],
    (rows, cols): (usize, usize),
    geo: GeoInfo,
    los: Option<&LosGeometry>,
) -> Result<Option<Array3<f64>>> {
    let Some(model) = &opts.plate_motion_model else {
        return Ok(None);
    };
    let los = los.context(
        "correction_options.plate_motion_model requires geometry_files: projecting the \
         plate's rigid surface velocity into line of sight needs the full LOS unit vector, \
         which the scalar incidence_angle_deg cannot supply",
    )?;
    let pole = resolve_euler_pole(model)?;
    let corners = dolphin_io::grid_corner_lonlat(geo.geotransform, rows, cols, geo.epsg)?;
    let lonlat = LonLatGrid::from_corners(corners, rows, cols);
    let rate_m_per_year = plate_motion_range_delay_rate_grid(pole, &lonlat, los);

    let reference_utc = plate_motion_acquisition_utc(opts, date_files, 0)?;
    let mut out = Array3::<f64>::zeros((date_files.len(), rows, cols));
    for t in 0..date_files.len() {
        let utc = plate_motion_acquisition_utc(opts, date_files, t)?;
        let elapsed_years = (utc - reference_utc).num_seconds() as f64 / (365.25 * 86_400.0);
        out.index_axis_mut(Axis(0), t)
            .assign(&(&rate_m_per_year * elapsed_years));
    }
    Ok(Some(out))
}

/// A granule's acquisition UTC for the plate-motion correction: explicit
/// `acquisition_utc[t]` if supplied, else the filename's `YYYYMMDDThhmmss`
/// token — only elapsed time since acquisition 0 matters here, since the
/// plate's rigid velocity is a constant rate, position-only.
fn plate_motion_acquisition_utc(
    opts: &CorrectionOptions,
    date_files: &[PathBuf],
    t: usize,
) -> Result<chrono::NaiveDateTime> {
    opts.acquisition_utc
        .get(t)
        .map(chrono::DateTime::naive_utc)
        .or_else(|| acq_utc_datetime(&date_files[t]))
        .with_context(|| {
            format!(
                "correction_options.plate_motion_model needs each granule's acquisition time; \
                 {} carries no YYYYMMDDThhmmss token",
                date_files[t].display()
            )
        })
}

/// Full acquisition UTC from a granule name's `YYYYMMDDThhmmss` token.
pub(crate) fn acq_utc_datetime(path: &Path) -> Option<chrono::NaiveDateTime> {
    let name = path.file_name().and_then(|s| s.to_str())?;
    let chars: Vec<char> = name.chars().collect();
    chars.windows(15).find_map(|w| {
        let token: String = w.iter().collect();
        chrono::NaiveDateTime::parse_from_str(&token, "%Y%m%dT%H%M%S").ok()
    })
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
        let utc = opts
            .acquisition_utc
            .get(t)
            .copied()
            .or_else(|| {
                date_files
                    .get(t)
                    .and_then(|path| acq_utc_datetime(path).map(|value| value.and_utc()))
            })
            .context("ionospheric correction requires full acquisition UTC")?;
        let content = std::fs::read_to_string(ionex_path)
            .with_context(|| format!("reading IONEX {}", ionex_path.display()))?;
        let maps = read_ionex(&content)?;
        let vtec = maps.value(utc, lat, lon)?;
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

fn troposphere_bracket(
    epochs: &[dolphin_core::config::TroposphereEpoch],
    utc: chrono::DateTime<chrono::Utc>,
) -> Result<(usize, usize, f64)> {
    ensure!(
        !epochs.is_empty() && epochs.windows(2).all(|p| p[0].epoch < p[1].epoch),
        "TROPO epochs must strictly increase"
    );
    ensure!(
        utc >= epochs[0].epoch && utc <= epochs[epochs.len() - 1].epoch,
        "acquisition outside TROPO temporal coverage"
    );
    let upper = epochs.partition_point(|entry| entry.epoch < utc);
    if epochs[upper].epoch == utc {
        return Ok((upper, upper, 0.0));
    }
    let lower = upper - 1;
    let seconds = (utc - epochs[lower].epoch)
        .num_microseconds()
        .context("TROPO interval overflow")? as f64;
    let span = (epochs[upper].epoch - epochs[lower].epoch)
        .num_microseconds()
        .context("TROPO span overflow")? as f64;
    Ok((lower, upper, seconds / span))
}

/// Build negative apparent LOS displacement from positive tropospheric path excess.
/// Resample each OPERA L4 netCDF onto the frame grid. Terrain and LOS
/// projection are judged on `support`
/// narrowed to finite geometry ([`geometry_support`]).
fn build_troposphere(
    opts: &CorrectionOptions,
    n_dates: usize,
    (rows, cols): (usize, usize),
    gt: [f64; 6],
    epsg: u32,
    los: Option<&LosGeometry>,
    support: Option<ArrayView2<'_, bool>>,
) -> Result<Option<Array3<f64>>> {
    if opts.troposphere_files.is_empty() && opts.troposphere_epochs.is_empty() {
        return Ok(None);
    }
    ensure!(
        opts.troposphere_files.is_empty() || opts.troposphere_epochs.is_empty(),
        "choose timed TROPO inputs or legacy per-date inputs, not both"
    );
    let support = geometry_support(support, los);
    let support = support.as_ref().map(Array2::view);
    if !opts.troposphere_epochs.is_empty() {
        ensure!(
            opts.acquisition_utc.len() == n_dates,
            "timed TROPO requires every acquisition UTC"
        );
        ensure!(
            opts.dem_file.is_some() && los.is_some(),
            "timed TROPO requires ellipsoidal terrain and per-pixel LOS"
        );
        let terrain = load_terrain(opts, gt, epsg, (rows, cols), support)?
            .context("missing TROPO terrain")?;
        let slant = slant_grid(opts.incidence_angle_deg, los, (rows, cols));
        ensure!(
            holds_on_support(&slant, support, |v| v.is_finite() && v >= 1.0),
            "invalid TROPO LOS projection"
        );
        let mut out = Array3::<f64>::zeros((n_dates, rows, cols));
        let mut cached = std::collections::BTreeMap::new();
        for (t, utc) in opts.acquisition_utc.iter().enumerate() {
            let (lower, upper, weight) = troposphere_bracket(&opts.troposphere_epochs, *utc)?;
            cached.retain(|index, _| *index == lower || *index == upper);
            for index in [lower, upper] {
                if let std::collections::btree_map::Entry::Vacant(entry) = cached.entry(index) {
                    entry.insert(tropo_at_terrain(
                        &opts.troposphere_epochs[index].path,
                        "total",
                        gt,
                        epsg,
                        (rows, cols),
                        &terrain,
                        support,
                    )?);
                }
            }
            let mut layer = out.index_axis_mut(Axis(0), t);
            ndarray::Zip::from(&mut layer)
                .and(&cached[&lower])
                .and(&cached[&upper])
                .and(&slant)
                .for_each(|value, &a, &b, &projection| {
                    *value = -(a * (1.0 - weight) + b * weight) * projection
                });
        }
        return Ok(Some(out));
    }
    anyhow::ensure!(
        opts.troposphere_files.len() == n_dates,
        "expected {n_dates} troposphere_files (one per date), got {}",
        opts.troposphere_files.len()
    );
    // ZTD is a zenith delay; project to line-of-sight by 1/cos(incidence) — per pixel
    // (1/up) from geometry when supplied, else the scalar knob (bit-identical fill).
    let slant = slant_grid(opts.incidence_angle_deg, los, (rows, cols));
    // The real L4 product resolves delay over 145 height levels, so a pixel's
    // delay must be taken at its terrain elevation; reading level 0 (-500 m)
    // over-corrects by ~2x at 2 km (issue #38). A DEM is therefore required
    // whenever the granule is height-resolved.
    let terrain = load_terrain(opts, gt, epsg, (rows, cols), support)?;
    let mut out = Array3::<f64>::zeros((n_dates, rows, cols));
    for (t, nc) in opts.troposphere_files.iter().enumerate() {
        let band = match terrain.as_ref() {
            Some(dem) => tropo_at_terrain(
                nc,
                &opts.troposphere_variable,
                gt,
                epsg,
                (rows, cols),
                dem,
                support,
            )?,
            None => {
                let grid =
                    read_tropo_for_grid(nc, &opts.troposphere_variable, gt, epsg, (rows, cols))?;
                resample_to_frame(&grid, gt, epsg, (rows, cols))?
            }
        };
        out.index_axis_mut(Axis(0), t).assign(&(-&band * &slant));
    }
    Ok(Some(out))
}

/// Terrain elevation on the frame grid, or `None` when no DEM is configured.
/// The DEM on the frame grid. Terrain must be finite on `support`; a nodata
/// pixel off it stands as NaN and yields a NaN delay there, which the validity
/// mask already excludes from the product.
fn load_terrain(
    opts: &CorrectionOptions,
    gt: [f64; 6],
    epsg: u32,
    shape: (usize, usize),
    support: Option<ArrayView2<'_, bool>>,
) -> Result<Option<Array2<f64>>> {
    let Some(dem) = opts.dem_file.as_ref() else {
        return Ok(None);
    };
    let grid = read_raster_for_grid(dem, gt, epsg, shape).map_err(anyhow::Error::msg)?;
    let frame = resample_terrain_to_frame(&grid, gt, epsg, shape)?;
    ensure!(
        holds_on_support(&frame, support, f64::is_finite),
        "terrain has nodata on valid support"
    );
    Ok(Some(frame))
}

/// Whether `ok` holds at every pixel of `support` — at every pixel when there
/// is no support mask.
fn holds_on_support(
    values: &Array2<f64>,
    support: Option<ArrayView2<'_, bool>>,
    ok: impl Fn(f64) -> bool,
) -> bool {
    values
        .indexed_iter()
        .all(|(index, &value)| support.is_some_and(|mask| !mask[index]) || ok(value))
}

/// Delay at each pixel's terrain elevation, linearly interpolated between the
/// two bracketing height levels of the L4 granule. Terrain must be finite and
/// inside the granule's height range on `support`; elsewhere the delay is NaN.
fn tropo_at_terrain(
    nc: &Path,
    var: &str,
    gt: [f64; 6],
    epsg: u32,
    shape: (usize, usize),
    terrain: &Array2<f64>,
    support: Option<ArrayView2<'_, bool>>,
) -> Result<Array2<f64>> {
    let vars: Vec<&str> = match var {
        "total" => vec!["hydrostatic_delay", "wet_delay"],
        other => vec![other],
    };
    let levels = height_levels(nc, vars[0]).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !levels.is_empty()
            && levels.iter().all(|v| v.is_finite())
            && levels.windows(2).all(|p| p[0] < p[1]),
        "L4 height levels must be finite and strictly increasing"
    );
    anyhow::ensure!(
        holds_on_support(terrain, support, |h| h.is_finite()
            && h >= levels[0]
            && h <= levels[levels.len() - 1]),
        "terrain is missing or outside L4 height coverage"
    );
    let (lo, hi) = bracketing_levels(&levels, terrain);
    let mut total = Array2::<f64>::zeros(shape);
    for name in vars {
        anyhow::ensure!(
            height_levels(nc, name)? == levels,
            "tropospheric variables have different height coordinates"
        );
        let mut planes = Vec::with_capacity(hi - lo + 1);
        for level in lo..=hi {
            let grid = read_l4_level_for_grid(nc, name, level, gt, epsg, shape)
                .map_err(anyhow::Error::msg)?;
            planes.push(resample_to_frame(&grid, gt, epsg, shape)?);
        }
        total += &interpolate_to_terrain(&levels[lo..=hi], &planes, terrain);
    }
    anyhow::ensure!(
        holds_on_support(&total, support, f64::is_finite),
        "missing terrain interpolation support"
    );
    Ok(total)
}

/// Inclusive band range covering the frame's terrain range, clamped to the
/// granule's own levels.
fn bracketing_levels(levels: &[f64], terrain: &Array2<f64>) -> (usize, usize) {
    let finite: Vec<f64> = terrain.iter().copied().filter(|v| v.is_finite()).collect();
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let last = levels.len() - 1;
    let lo = levels.iter().rposition(|&h| h <= min).unwrap_or(0);
    let hi = levels.iter().position(|&h| h >= max).unwrap_or(last);
    (lo.min(hi), hi.max(lo))
}

/// Per-pixel linear interpolation of the level planes onto terrain elevation.
fn interpolate_to_terrain(
    levels: &[f64],
    planes: &[Array2<f64>],
    terrain: &Array2<f64>,
) -> Array2<f64> {
    Array2::from_shape_fn(terrain.dim(), |ix| {
        let h = terrain[ix];
        if !h.is_finite() || h < levels[0] || h > levels[levels.len() - 1] {
            return f64::NAN;
        }
        if levels.len() == 1 {
            return planes[0][ix];
        }
        let upper = levels
            .iter()
            .position(|&level| level >= h)
            .unwrap_or(levels.len() - 1);
        let upper = upper.max(1);
        let lower = upper - 1;
        let span = levels[upper] - levels[lower];
        let weight = if span > 0.0 {
            (h - levels[lower]) / span
        } else {
            0.0
        };
        if weight == 0.0 {
            return planes[lower][ix];
        }
        if weight == 1.0 {
            return planes[upper][ix];
        }
        planes[lower][ix] * (1.0 - weight) + planes[upper][ix] * weight
    })
}

/// The validity support narrowed to pixels with finite LOS geometry; without
/// per-pixel geometry the caller's support stands. The phase-link mask knows
/// nothing of the STATIC footprint: a pixel outside every granule is admitted by
/// the coverage gate and masked at publication, but it has no slant factor and,
/// when the DEM stops at the granule edge, no terrain either — so terrain and
/// projection are judged only where geometry exists.
fn geometry_support(
    support: Option<ArrayView2<'_, bool>>,
    los: Option<&LosGeometry>,
) -> Option<Array2<bool>> {
    let Some(geometry) = los else {
        return support.map(|mask| mask.to_owned());
    };
    Some(Array2::from_shape_fn(geometry.up.dim(), |index| {
        support.is_none_or(|mask| mask[index]) && geometry.up[index].is_finite()
    }))
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
    let output = resample_terrain_to_frame(grid, gt, frame_epsg, shape)?;
    ensure_finite_coverage(&output).map_err(anyhow::Error::msg)?;
    Ok(output)
}

/// [`resample_to_frame`] without the whole-grid finiteness gate: DEM nodata is
/// judged on the validity support by the caller, not here.
fn resample_terrain_to_frame(
    grid: &DelayGrid,
    gt: [f64; 6],
    frame_epsg: u32,
    shape: (usize, usize),
) -> Result<ndarray::Array2<f64>> {
    match grid.epsg {
        Some(e) if e == frame_epsg => Ok(resample_bilinear(
            grid.data.view(),
            grid.geotransform,
            gt,
            shape,
        )),
        _ if grid.srs_wkt.is_some() || grid.epsg.is_some() => {
            warp_to_frame(grid, gt, frame_epsg, shape).map_err(anyhow::Error::msg)
        }
        _ => Err(dolphin_corrections::CorrectionError::NoSourceCrs.into()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn correction_epochs_follow_date_varying_burst_owner() {
        let _hdf5 = hdf5_guard();
        let utc = chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .to_utc();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let dir = std::env::temp_dir().join(format!("dolphin-owner-tropo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dem = dir.join("dem.nc");
        let geometry = dir.join("static.h5");
        let gt = [-0.5, 1.0, 0.0, 2.5, 0.0, -1.0];
        write_4326_netcdf(&dem, &Array2::from_elem((3, 3), 500.0), gt);
        write_uniform_static(&geometry, 30.0, gt, (3, 3));
        hdf5::File::open_rw(&geometry)
            .unwrap()
            .dataset("/data/projection")
            .unwrap()
            .write_scalar(&4326_i64)
            .unwrap();
        let options = CorrectionOptions {
            troposphere_epochs: vec![
                dolphin_core::config::TroposphereEpoch {
                    path: fixtures.join("tropo_lower.nc"),
                    epoch: utc,
                },
                dolphin_core::config::TroposphereEpoch {
                    path: fixtures.join("tropo_upper.nc"),
                    epoch: utc + chrono::Duration::hours(3),
                },
            ],
            dem_file: Some(dem),
            geometry_files: vec![geometry],
            ..Default::default()
        };
        let groups: Vec<_> = [0, 1]
            .into_iter()
            .map(|owner| {
                let mut options = options.clone();
                options.acquisition_utc = [0, 60, 120]
                    .map(|minutes| utc + chrono::Duration::minutes(minutes + 30 * i64::from(owner)))
                    .to_vec();
                (owner, options, vec![PathBuf::from("unused"); 3])
            })
            .collect();
        let owners = ndarray::array![[[0, 1]], [[1, 0]], [[0, 1]]];
        for wavelength in [0.05546576, 0.238403545] {
            let mut displacement = Array3::zeros((2, 1, 2));
            let output = apply_corrections_with_ownership(
                &groups,
                Some(wavelength),
                &mut displacement,
                owners.view(),
                GeoInfo {
                    epsg: 4326,
                    geotransform: [-0.5, 1.0, 0.0, 1.5, 0.0, -1.0],
                },
                None,
            )
            .unwrap();
            let meters =
                displacement.mapv(|phase| -wavelength * phase / (4.0 * std::f64::consts::PI));
            assert!((meters[(0, 0, 0)] - 0.825 / 30_f64.to_radians().cos()).abs() < 1e-7);
            assert!((meters[(0, 0, 1)] - 0.275 / 30_f64.to_radians().cos()).abs() < 1e-7);
            assert!(
                (output.troposphere.unwrap()[(0, 0, 1)] + 1.925 / 30_f64.to_radians().cos()).abs()
                    < 1e-7
            );
        }
    }

    #[test]
    fn timed_troposphere_combines_height_time_and_los() {
        use dolphin_core::config::TroposphereEpoch;
        let utc = chrono::DateTime::parse_from_rfc3339("2026-01-01T23:00:00Z")
            .unwrap()
            .to_utc();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let dem =
            std::env::temp_dir().join(format!("dolphin-timed-tropo-dem-{}.nc", std::process::id()));
        write_4326_netcdf(
            &dem,
            &Array2::from_elem((3, 3), 500.0),
            [-0.5, 1.0, 0.0, 2.5, 0.0, -1.0],
        );
        let opts = CorrectionOptions {
            acquisition_utc: vec![
                utc,
                utc + chrono::Duration::minutes(90),
                utc + chrono::Duration::hours(3),
            ],
            troposphere_epochs: vec![
                TroposphereEpoch {
                    path: fixtures.join("tropo_lower.nc"),
                    epoch: utc,
                },
                TroposphereEpoch {
                    path: fixtures.join("tropo_upper.nc"),
                    epoch: utc + chrono::Duration::hours(3),
                },
            ],
            dem_file: Some(dem.clone()),
            ..Default::default()
        };
        let los = LosGeometry {
            east: Array2::from_elem((1, 1), 0.6),
            north: Array2::zeros((1, 1)),
            up: Array2::from_elem((1, 1), 0.8),
        };
        let layers = build_troposphere(
            &opts,
            3,
            (1, 1),
            [0.5, 1.0, 0.0, 1.5, 0.0, -1.0],
            4326,
            Some(&los),
            None,
        )
        .unwrap()
        .unwrap();
        for (index, expected) in [2.0625, 3.09375, 4.125].into_iter().enumerate() {
            assert!((layers[(index, 0, 0)] + expected).abs() < 1e-10);
        }
        for wavelength in [0.05546576, 0.238403545] {
            let meters_per_radian = -wavelength / (4.0 * std::f64::consts::PI);
            // Stationary ground: increasing path excess produces apparent motion away.
            let mut displacement = Array3::from_shape_vec(
                (2, 1, 1),
                vec![-1.03125 / meters_per_radian, -2.0625 / meters_per_radian],
            )
            .unwrap();
            subtract_delay(&mut displacement, layers.view(), wavelength).unwrap();
            assert!(displacement
                .iter()
                .all(|phase| (phase * meters_per_radian).abs() < 1e-10));
        }
        std::fs::remove_file(dem).unwrap();
    }

    /// A DEM nodata pixel is judged on the validity support: off it, the delay
    /// there is NaN and the run goes on; on it, the run still fails closed.
    #[test]
    fn terrain_nodata_off_support_stands_as_nan() {
        let _hdf5 = hdf5_guard();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let gt = [-0.5, 1.0, 0.0, 1.5, 0.0, -1.0];
        let terrain = ndarray::array![[500.0, f64::NAN]];
        let support = ndarray::array![[true, false]];
        let delay = tropo_at_terrain(
            &fixtures.join("tropo_lower.nc"),
            "total",
            gt,
            4326,
            (1, 2),
            &terrain,
            Some(support.view()),
        )
        .expect("nodata off support is not an error");
        assert!(delay[(0, 0)].is_finite());
        assert!(delay[(0, 1)].is_nan());

        let whole_grid = tropo_at_terrain(
            &fixtures.join("tropo_lower.nc"),
            "total",
            gt,
            4326,
            (1, 2),
            &terrain,
            None,
        )
        .unwrap_err();
        assert!(whole_grid
            .to_string()
            .contains("outside L4 height coverage"));
        let on_support = tropo_at_terrain(
            &fixtures.join("tropo_lower.nc"),
            "total",
            gt,
            4326,
            (1, 2),
            &terrain,
            Some(ndarray::array![[true, true]].view()),
        )
        .unwrap_err();
        assert!(on_support
            .to_string()
            .contains("outside L4 height coverage"));
    }

    #[test]
    fn troposphere_time_brackets_are_dated_and_closed() {
        use dolphin_core::config::TroposphereEpoch;
        let start = chrono::DateTime::parse_from_rfc3339("2026-01-01T23:00:00Z")
            .unwrap()
            .to_utc();
        let epochs = vec![
            TroposphereEpoch {
                path: "a.nc".into(),
                epoch: start,
            },
            TroposphereEpoch {
                path: "b.nc".into(),
                epoch: start + chrono::Duration::hours(3),
            },
        ];
        assert_eq!(troposphere_bracket(&epochs, start).unwrap(), (0, 0, 0.0));
        assert_eq!(
            troposphere_bracket(&epochs, epochs[1].epoch).unwrap(),
            (1, 1, 0.0)
        );
        assert_eq!(
            troposphere_bracket(&epochs, start + chrono::Duration::minutes(90)).unwrap(),
            (0, 1, 0.5)
        );
        assert!(troposphere_bracket(&epochs, start - chrono::Duration::seconds(1)).is_err());
        assert!(troposphere_bracket(&epochs, start + chrono::Duration::days(1)).is_err());
        assert!(troposphere_bracket(&[epochs[0].clone(), epochs[0].clone()], start).is_err());
    }

    #[test]
    fn terrain_interpolation_rejects_missing_and_outside_support() {
        let levels = [0.0, 1000.0];
        let planes = vec![ndarray::array![[2.0]], ndarray::array![[1.0]]];
        for height in [f64::NAN, -1.0, 1001.0] {
            assert!(
                interpolate_to_terrain(&levels, &planes, &ndarray::array![[height]])[(0, 0)]
                    .is_nan()
            );
        }
    }

    /// Issue #38: a pixel's delay must come from its terrain elevation, not the
    /// granule's lowest level. Linear between bracketing levels, exact at a knot.
    #[test]
    fn delay_interpolates_to_terrain_elevation() {
        let levels = [0.0_f64, 1000.0, 2000.0, 3000.0];
        // Each plane is constant so the expected value is analytic.
        let planes: Vec<Array2<f64>> = [2.6_f64, 2.2, 1.8, 1.4]
            .iter()
            .map(|&v| Array2::from_elem((1, 4), v))
            .collect();
        let terrain = ndarray::array![[0.0_f64, 2000.0, 2500.0, 3000.0]];
        let out = interpolate_to_terrain(&levels, &planes, &terrain);
        // knots exact
        assert!((out[(0, 0)] - 2.6).abs() < 1e-12);
        assert!((out[(0, 1)] - 1.8).abs() < 1e-12);
        assert!((out[(0, 3)] - 1.4).abs() < 1e-12);
        // midway between 2000 m and 3000 m
        assert!((out[(0, 2)] - 1.6).abs() < 1e-12);
    }

    /// The level window must cover the frame's terrain range, so a 2.2-2.7 km
    /// frame reads the 2000 m and 3000 m levels and not the -500 m one.
    #[test]
    fn bracketing_levels_cover_the_terrain_range() {
        let levels = [-500.0_f64, 0.0, 1000.0, 2000.0, 3000.0, 4000.0];
        let terrain = ndarray::array![[2200.0_f64, 2700.0]];
        assert_eq!(bracketing_levels(&levels, &terrain), (3, 4));
    }

    use super::*;
    use dolphin_corrections::geometry::resolve_los_geometry;
    use dolphin_corrections::ionosphere::IonexError;
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
        let layers = build_troposphere(&opts, 2, (rows, cols), dst_gt, 32610, None, None)
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
            let expected = -(1.0 + g(xs[0], ys[0]));
            assert!(
                (layers[(0, r, cc)] + 1.0).abs() < 5e-3,
                "date0 apparent displacement should be -1.0"
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
            None,
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
        let err =
            apply_corrections(&opts, None, &mut disp, &[], 32610, [0.0; 6], None).unwrap_err();
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

    /// A frame the coverage gate admits (0.3% outside the one STATIC granule) runs
    /// timed TROPO instead of being refused: the phase-link support is narrowed to
    /// finite geometry before the terrain and LOS-projection checks, and the
    /// outside pixels come out as NaN delay for the validity mask to drop.
    #[test]
    fn outside_static_pixels_do_not_refuse_timed_troposphere() {
        let _hdf5 = hdf5_guard();
        let utc = chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .to_utc();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let dir = std::env::temp_dir().join(format!(
            "dolphin-outside-static-tropo-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dem = dir.join("dem.nc");
        let geometry = dir.join("static.h5");
        let gt = [0.0, 1.0 / 1024.0, 0.0, 1.0, 0.0, -1.0 / 1024.0];
        let (rows, cols) = (1000, 2);
        write_4326_netcdf(
            &dem,
            &Array2::from_elem((3, 3), 500.0),
            [-0.5, 1.0, 0.0, 2.5, 0.0, -1.0],
        );
        write_uniform_static(&geometry, 34.0, gt, (rows - 3, cols));
        hdf5::File::open_rw(&geometry)
            .unwrap()
            .dataset("/data/projection")
            .unwrap()
            .write_scalar(&4326_i64)
            .unwrap();
        let opts = CorrectionOptions {
            acquisition_utc: vec![
                utc,
                utc + chrono::Duration::minutes(90),
                utc + chrono::Duration::hours(3),
            ],
            troposphere_epochs: vec![
                dolphin_core::config::TroposphereEpoch {
                    path: fixtures.join("tropo_lower.nc"),
                    epoch: utc,
                },
                dolphin_core::config::TroposphereEpoch {
                    path: fixtures.join("tropo_upper.nc"),
                    epoch: utc + chrono::Duration::hours(3),
                },
            ],
            dem_file: Some(dem),
            geometry_files: vec![geometry],
            ..Default::default()
        };
        let support = Array2::from_elem((rows, cols), true);
        let mut displacement = Array3::zeros((2, rows, cols));
        let layers = apply_corrections(
            &opts,
            Some(0.05546576),
            &mut displacement,
            &[],
            4326,
            gt,
            Some(support.view()),
        )
        .expect("0.3% outside STATIC runs timed TROPO instead of being refused");
        let los = layers.los_geometry.expect("geometry configured");
        assert_eq!(los.outside_static().pixel_count, 3 * cols);
        let troposphere = layers.troposphere.expect("timed TROPO built");
        for row in 0..rows {
            let inside = row < rows - 3;
            assert_eq!(
                troposphere[(0, row, 0)].is_finite(),
                inside,
                "row {row}: delay finite only inside the STATIC footprint"
            );
            assert_eq!(displacement[(0, row, 0)].is_finite(), inside);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `correction_options.max_outside_static_fraction` reaches the resolver: the
    /// 0.3% corridor-edge case (a 1000×100 frame whose one STATIC granule stops
    /// three rows short) resolves under the default gate and is refused under a
    /// configured 0.1% one, with the outside pixels masked and counted.
    #[test]
    fn configured_outside_static_gate_reaches_the_resolver() {
        let _hdf5 = hdf5_guard();
        let gt = [500_000.0, 30.0, 0.0, 4_000_000.0, 0.0, -30.0];
        let path = std::env::temp_dir().join(format!(
            "dolphin_static_outside_gate_{}.h5",
            std::process::id()
        ));
        write_uniform_static(&path, 34.0, gt, (997, 100));
        let default_gate = CorrectionOptions {
            geometry_files: vec![path.clone()],
            ..Default::default()
        };
        let los = resolve_geometry(&default_gate, 32610, gt, (1000, 100))
            .expect("0.3% outside resolves under the default gate")
            .expect("geometry configured");
        assert_eq!(los.outside_static().pixel_count, 300);

        let tight_gate = CorrectionOptions {
            max_outside_static_fraction: Some(0.001),
            ..default_gate
        };
        let err = resolve_geometry(&tight_gate, 32610, gt, (1000, 100))
            .expect_err("0.3% outside is refused under a 0.1% gate");
        assert!(
            format!("{err:#}").contains("max_outside_static_fraction"),
            "expected the coverage gate, got: {err:#}"
        );
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
        let layers = apply_corrections(&opts, None, &mut disp, &[], 32610, gt, None).unwrap();

        assert!(layers.ionosphere.is_none() && layers.troposphere.is_none());
        let los = layers.los_geometry.expect("geometry present");
        assert!((los.incidence_deg()[(1, 1)] - 34.0).abs() < 1e-3);
        assert_eq!(disp, original, "geometry-only must not touch displacement");
        let _ = std::fs::remove_file(&path);
    }

    /// A malformed IONEX file reaches the caller as the typed `IonexError`, not
    /// a flattened message, so the failure class stays inspectable.
    #[test]
    fn malformed_ionex_keeps_its_typed_error() {
        let dir =
            std::env::temp_dir().join(format!("dolphin-malformed-ionex-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ionex = dir.join("garbage.INX");
        std::fs::write(&ionex, "not an ionex file\n").unwrap();
        let utc = chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .to_utc();
        let opts = CorrectionOptions {
            ionosphere_files: vec![ionex.clone(), ionex],
            acquisition_utc: vec![utc, utc + chrono::Duration::days(12)],
            ..Default::default()
        };
        let mut displacement = Array3::<f64>::zeros((1, 1, 1));
        let err = apply_corrections(
            &opts,
            Some(0.05546576),
            &mut displacement,
            &[],
            4326,
            [0.0, 1.0, 0.0, 1.0, 0.0, -1.0],
            None,
        )
        .unwrap_err();
        assert!(
            err.chain()
                .any(|cause| cause.downcast_ref::<IonexError>().is_some()),
            "expected a typed IonexError in the chain, got: {err:#}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
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
        let err =
            apply_corrections(&opts, None, &mut disp, &[], 32610, [0.0; 6], None).unwrap_err();
        assert!(
            err.to_string().contains("CSLC-S1-STATIC geometry")
                || format!("{err:#}").contains("static_geometry.h5"),
            "expected contextual geometry read error, got: {err:#}"
        );
    }

    /// The tide needs the whole timestamp, not just the seconds of day, and a
    /// date-only name has none to give.
    #[test]
    fn parses_full_acquisition_datetime() {
        let opera = Path::new("OPERA_L2_CSLC-S1_T027_20230914T132417Z_x.h5");
        let parsed = acq_utc_datetime(opera).expect("OPERA stamp");
        assert_eq!(parsed.to_string(), "2023-09-14 13:24:17");
        assert!(acq_utc_datetime(Path::new("cslc_20221119.h5")).is_none());
    }

    /// Issue #21, the default: the tide flag is off, so `apply_corrections` is a
    /// no-op on displacement and emits no tide layer.
    #[test]
    fn solid_earth_tide_is_off_by_default() {
        let opts = CorrectionOptions::default();
        assert!(!opts.is_enabled());
        let mut disp = Array3::from_shape_fn((2, 2, 2), |(t, r, c)| (t + r + c) as f64);
        let original = disp.clone();
        let layers = apply_corrections(&opts, None, &mut disp, &[], 32614, [0.0; 6], None).unwrap();
        assert!(layers.solid_earth_tide.is_none());
        assert_eq!(disp, original);
    }

    /// Enabling the tide without geometry fails closed and says why: a 3-D
    /// displacement cannot be projected into line of sight from a scalar incidence.
    #[test]
    fn solid_earth_tide_without_geometry_fails_closed() {
        let opts = CorrectionOptions {
            solid_earth_tide: true,
            ..Default::default()
        };
        let mut disp = Array3::<f64>::zeros((1, 2, 2));
        let gt = [500_000.0, 60.0, 0.0, 2_150_000.0, 0.0, -60.0];
        let files = vec![PathBuf::from("OPERA_L2_CSLC-S1_T005_20230104T004053Z_x.h5")];
        let err =
            apply_corrections(&opts, Some(0.055), &mut disp, &files, 32614, gt, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("geometry_files"),
            "expected the geometry requirement, got: {err:#}"
        );
    }

    /// A granule with no time stamp fails closed rather than defaulting to noon:
    /// the tide is semidiurnal, so a defaulted time can be wrong by half a cycle.
    #[test]
    fn solid_earth_tide_without_a_timestamp_fails_closed() {
        let _hdf5 = hdf5_guard();
        let gt = [500_000.0, 60.0, 0.0, 4_000_000.0, 0.0, -60.0];
        let path = std::env::temp_dir().join("dolphin_static_set_no_time.h5");
        write_uniform_static(&path, 34.0, gt, (3, 3));
        let opts = CorrectionOptions {
            solid_earth_tide: true,
            geometry_files: vec![path.clone()],
            ..Default::default()
        };
        let mut disp = Array3::<f64>::zeros((1, 3, 3));
        let files = vec![PathBuf::from("cslc_20230104.h5")];
        let err =
            apply_corrections(&opts, Some(0.055), &mut disp, &files, 32610, gt, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("semidiurnal"),
            "expected the timestamp requirement, got: {err:#}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// End to end: with geometry and timestamps the tide layer is built, has the
    /// physically expected magnitude, and is actually subtracted from the series.
    #[test]
    fn solid_earth_tide_is_built_and_subtracted() {
        let _hdf5 = hdf5_guard();
        let gt = [500_000.0, 60.0, 0.0, 4_000_000.0, 0.0, -60.0];
        let path = std::env::temp_dir().join("dolphin_static_set_applied.h5");
        write_uniform_static(&path, 34.0, gt, (3, 3));
        let opts = CorrectionOptions {
            solid_earth_tide: true,
            geometry_files: vec![path.clone()],
            ..Default::default()
        };
        // Two acquisitions 12 days and ~12 h of tidal phase apart, so the
        // differential does not cancel.
        let files = vec![
            PathBuf::from("OPERA_L2_CSLC-S1_T005_20230104T004053Z_x.h5"),
            PathBuf::from("OPERA_L2_CSLC-S1_T005_20230116T124053Z_x.h5"),
        ];
        let mut disp = Array3::<f64>::zeros((1, 3, 3));
        let layers =
            apply_corrections(&opts, Some(0.055), &mut disp, &files, 32610, gt, None).unwrap();

        let mut explicit = opts.clone();
        explicit.acquisition_utc = files
            .iter()
            .map(|p| acq_utc_datetime(p).unwrap().and_utc())
            .collect();
        let opaque = vec![PathBuf::from("opaque-a.h5"), PathBuf::from("opaque-b.h5")];
        let mut explicit_disp = Array3::<f64>::zeros((1, 3, 3));
        let explicit_layers = apply_corrections(
            &explicit,
            Some(0.055),
            &mut explicit_disp,
            &opaque,
            32610,
            gt,
            None,
        )
        .unwrap();
        assert_eq!(explicit_disp, disp);
        assert_eq!(explicit_layers.solid_earth_tide, layers.solid_earth_tide);
        let tide = layers.solid_earth_tide.expect("tide layer");
        assert_eq!(tide.dim(), (2, 3, 3));
        let peak = tide.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            (0.005..0.40).contains(&peak),
            "tide LOS delay {peak} m is outside the physical envelope"
        );
        assert!(
            disp.iter().any(|v| v.abs() > 1e-9),
            "the tide was built but never subtracted"
        );
        let los = layers.los_geometry.expect("tide geometry");
        let corners = dolphin_io::grid_corner_lonlat(gt, 3, 3, 32610).unwrap();
        let lonlat = LonLatGrid::from_corners(corners, 3, 3);
        let meters_per_radian = -0.055 / (4.0 * std::f64::consts::PI);
        let mut measured = Array3::from_shape_fn((1, 3, 3), |(_, row, col)| {
            let (lon, lat) = lonlat.at(row, col);
            let enu = files
                .iter()
                .map(|file| {
                    dolphin_corrections::solid_earth_tide::tide_displacement_enu(
                        acq_utc_datetime(file).unwrap(),
                        lon,
                        lat,
                        0.0,
                    )
                })
                .collect::<Vec<_>>();
            let tidal_motion = (enu[1][0] - enu[0][0]) * los.east[(row, col)]
                + (enu[1][1] - enu[0][1]) * los.north[(row, col)]
                + (enu[1][2] - enu[0][2]) * los.up[(row, col)];
            (0.003 + tidal_motion) / meters_per_radian
        });
        apply_corrections(&opts, Some(0.055), &mut measured, &files, 32610, gt, None).unwrap();
        assert!(measured
            .iter()
            .all(|phase| (phase * meters_per_radian - 0.003).abs() < 1e-10));
        let _ = std::fs::remove_file(&path);
    }

    /// Issue #103, the default: plate motion is unset, so `apply_corrections`
    /// is a no-op on displacement and emits no plate-motion layer.
    #[test]
    fn plate_motion_is_off_by_default() {
        let opts = CorrectionOptions::default();
        assert!(!opts.is_enabled());
        let mut disp = Array3::from_shape_fn((2, 2, 2), |(t, r, c)| (t + r + c) as f64);
        let original = disp.clone();
        let layers = apply_corrections(&opts, None, &mut disp, &[], 32614, [0.0; 6], None).unwrap();
        assert!(layers.plate_motion.is_none());
        assert_eq!(disp, original);
    }

    /// A named plate not in the built-in table is a contextual error, not a panic.
    #[test]
    fn plate_motion_unknown_plate_errors() {
        let _hdf5 = hdf5_guard();
        let gt = [500_000.0, 60.0, 0.0, 4_000_000.0, 0.0, -60.0];
        let path = std::env::temp_dir().join("dolphin_static_plate_unknown.h5");
        write_uniform_static(&path, 34.0, gt, (3, 3));
        let opts = CorrectionOptions {
            plate_motion_model: Some(dolphin_core::config::PlateMotionModel::Plate(
                "Atlantis".into(),
            )),
            geometry_files: vec![path.clone()],
            ..Default::default()
        };
        let files = vec![PathBuf::from("OPERA_L2_CSLC-S1_T005_20230104T004053Z_x.h5")];
        let mut disp = Array3::<f64>::zeros((1, 3, 3));
        let err =
            apply_corrections(&opts, Some(0.055), &mut disp, &files, 32610, gt, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("unknown plate motion model plate"),
            "expected the unknown-plate error, got: {err:#}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Enabling plate motion without geometry fails closed and says why: a
    /// rigid-rotation velocity cannot be projected into line of sight from a
    /// scalar incidence.
    #[test]
    fn plate_motion_without_geometry_fails_closed() {
        let opts = CorrectionOptions {
            plate_motion_model: Some(dolphin_core::config::PlateMotionModel::Plate(
                "Pacific".into(),
            )),
            ..Default::default()
        };
        let mut disp = Array3::<f64>::zeros((1, 2, 2));
        let gt = [500_000.0, 60.0, 0.0, 2_150_000.0, 0.0, -60.0];
        let files = vec![PathBuf::from("OPERA_L2_CSLC-S1_T005_20230104T004053Z_x.h5")];
        let err =
            apply_corrections(&opts, Some(0.055), &mut disp, &files, 32614, gt, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("geometry_files"),
            "expected the geometry requirement, got: {err:#}"
        );
    }

    /// End to end: with geometry and timestamps the plate-motion layer is
    /// built, is exactly zero at acquisition 0 (the elapsed-time reference),
    /// scales *linearly* with elapsed time (the signature that distinguishes
    /// a constant-rate correction from the tide's oscillatory one), and is
    /// actually subtracted from the series.
    #[test]
    fn plate_motion_is_built_and_subtracted() {
        let _hdf5 = hdf5_guard();
        let gt = [500_000.0, 60.0, 0.0, 4_000_000.0, 0.0, -60.0];
        let path = std::env::temp_dir().join("dolphin_static_plate_applied.h5");
        write_uniform_static(&path, 34.0, gt, (3, 3));
        let opts = CorrectionOptions {
            plate_motion_model: Some(dolphin_core::config::PlateMotionModel::Plate(
                "Pacific".into(),
            )),
            geometry_files: vec![path.clone()],
            ..Default::default()
        };
        // Acquisitions 1 and 2 years after acquisition 0.
        let files = vec![
            PathBuf::from("OPERA_L2_CSLC-S1_T005_20200104T004053Z_x.h5"),
            PathBuf::from("OPERA_L2_CSLC-S1_T005_20210103T184053Z_x.h5"),
            PathBuf::from("OPERA_L2_CSLC-S1_T005_20220103T124053Z_x.h5"),
        ];
        let mut disp = Array3::<f64>::zeros((2, 3, 3));
        let layers =
            apply_corrections(&opts, Some(0.055), &mut disp, &files, 32610, gt, None).unwrap();
        let plate_motion = layers.plate_motion.expect("plate motion layer");
        assert_eq!(plate_motion.dim(), (3, 3, 3));

        assert!(
            plate_motion
                .index_axis(Axis(0), 0)
                .iter()
                .all(|&v| v == 0.0),
            "delay at the elapsed-time reference (acquisition 0) must be exactly zero"
        );
        // Pacific-plate LOS speed at this incidence/azimuth is on the order of
        // tens of mm/yr, so two years is a physically plausible envelope.
        let year1 = plate_motion.index_axis(Axis(0), 1);
        let year2 = plate_motion.index_axis(Axis(0), 2);
        for ((_idx, &d1), &d2) in year1.indexed_iter().zip(year2.iter()) {
            assert!(
                (0.0..0.20).contains(&d1.abs()),
                "1-year plate motion delay {d1} m is outside the physical envelope"
            );
            // Constant rate: year 2 is (to within the ~day-level rounding of
            // the test's own hand-picked dates) exactly double year 1.
            let ratio = d2 / d1;
            assert!(
                (1.97..2.03).contains(&ratio),
                "plate motion should accumulate linearly: year1={d1}, year2={d2}, ratio={ratio}"
            );
        }
        assert!(
            disp.iter().any(|v| v.abs() > 1e-9),
            "plate motion was built but never subtracted"
        );
        let _ = std::fs::remove_file(&path);
    }
}
