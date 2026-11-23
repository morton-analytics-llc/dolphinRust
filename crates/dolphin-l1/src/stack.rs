//! Sentinel-1 burst stack: raw CDSE bursts in, geocoded CSLC HDF5 out.
//!
//! For every date of one burst id: download the burst and its POEORB, open
//! the burst, geocode it onto the requested grid with the ESD azimuth shift,
//! and write the OPERA-layout HDF5 the phase-linking path reads. The DEM is
//! read once per stack. With ESD on, the next burst along track (`k+1`) of
//! every date is fetched too, both bursts are geocoded onto a 20 m grid over
//! their overlap, and each date's shift is chained from the previous date
//! (short temporal baselines keep the overlap coherent):
//! `shift_i = shift_{i−1} + ESD(i−1, i)`, with the first date as reference.
//!
//! **Live operation.** CDSE publishes a new IW SLC 2–21 h after acquisition
//! (measured 2026-11-22, `validation/results/issue230_raw_front_end/latency.json`),
//! and a burst is downloadable as soon as its parent is. The host polls by
//! calling [`process_stack`] with `start` at its last processed acquisition:
//! dates whose output already exists are not rewritten (their ESD shift is
//! read back to anchor the chain), new dates are processed, and a date
//! written with a restituted orbit (RESORB, the only orbit for the first
//! ~3 weeks) is reprocessed once its precise POEORB exists.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};

use crate::cdse::{BurstId, BurstQuery, BurstRecord, CdseClient};
use crate::cslc_h5::{write_cslc, CslcProvenance};
use crate::dem::{Dem, LonLatBounds};
use crate::error::{L1Error, Result};
use crate::esd::{EsdEstimate, OverlapPair};
use crate::geocode::{geocode_burst, GeocodedBurst};
use crate::gim::{read_gim, GimClient};
use crate::grid::{GeoGrid, Projection};
use crate::orbit::Orbit;
use crate::safe::S1Burst;
use crate::static_layers::{static_layers, write_static};
use dolphin_corrections::ionosphere::IonexMaps;

/// ESD overlap grid posting (m).
const ESD_POSTING: f64 = 20.0;
/// Orbit state vectors kept either side of the burst.
const ORBIT_MARGIN_SECONDS: i64 = 60;
/// DEM margin around the burst footprint (deg).
const DEM_MARGIN_DEG: f64 = 0.1;

/// What to process.
#[derive(Debug, Clone)]
pub struct StackRequest {
    /// Burst to process.
    pub burst_id: BurstId,
    /// Polarisation, e.g. `VV`.
    pub polarization: String,
    /// First acquisition time (inclusive).
    pub start: DateTime<Utc>,
    /// Last acquisition time (exclusive).
    pub end: DateTime<Utc>,
    /// Output grid: the OPERA burst grid or a window of it.
    pub grid: GeoGrid,
    /// Estimate and apply ESD azimuth shifts.
    pub esd: bool,
    /// Raw-data cache (bursts, orbits, annotations).
    pub cache_dir: PathBuf,
    /// Where the CSLC HDF5 files go.
    pub output_dir: PathBuf,
    /// Keep downloaded bursts in `cache_dir` after use.
    pub keep_raw: bool,
    /// Also write `azimuth_carrier_phase` and `flattening_phase` layers.
    pub phase_layers: bool,
    /// Add the JPL GIM ionospheric delay to the sampling range (needs the
    /// Earthdata token).
    pub ionosphere: bool,
}

/// One written date.
#[derive(Debug, Clone)]
pub struct StackOutput {
    /// Zero-Doppler time of the burst's first line.
    pub sensing_start: DateTime<Utc>,
    /// The CSLC HDF5 file.
    pub path: PathBuf,
    /// ESD azimuth shift applied (s).
    pub azimuth_shift: f64,
    /// The ESD estimate against the previous date (`None` for the reference,
    /// with ESD off, or when the overlap was not coherent enough).
    pub esd: Option<EsdEstimate>,
}

/// One date's burst, orbit and provenance.
struct Acquisition {
    record: BurstRecord,
    safe: PathBuf,
    burst: S1Burst,
    orbit: Orbit,
    orbit_file: String,
    ionosphere: Option<IonexMaps>,
    gim_file: Option<String>,
}

impl Acquisition {
    fn load(
        client: &mut CdseClient,
        gim: Option<&GimClient>,
        record: BurstRecord,
        cache: &Path,
    ) -> Result<Self> {
        let safe = client.download_burst(&record, &cache.join("bursts"))?;
        let iw2 = match record.swath_identifier.as_str() {
            "IW2" => None,
            _ => Some(client.download_iw2_annotation(&record, &cache.join("annotations"))?),
        };
        let burst = S1Burst::open(&safe, &record.polarisation_channels, iw2.as_deref())?;
        let margin = Duration::seconds(ORBIT_MARGIN_SECONDS);
        let orbit_record = client.search_orbit(
            record.mission(),
            burst.sensing_start - margin,
            burst.sensing_stop() + margin,
        )?;
        let eof = client.download_orbit(&orbit_record, &cache.join("orbits"))?;
        let orbit = Orbit::from_eof(
            &eof,
            burst.sensing_start,
            burst.sensing_start,
            burst.sensing_stop(),
            margin,
        )?;
        let gim_path = gim
            .map(|g| g.fetch(burst.sensing_mid().date_naive(), &cache.join("ionex")))
            .transpose()?;
        Ok(Self {
            ionosphere: gim_path.as_deref().map(read_gim).transpose()?,
            gim_file: gim_path
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())),
            record,
            safe,
            burst,
            orbit,
            orbit_file: orbit_record.name,
        })
    }

    fn geocode(&self, dem: &Dem, grid: &GeoGrid, shift: f64) -> Result<GeocodedBurst> {
        geocode_burst(
            &self.burst,
            &self.orbit,
            dem,
            grid,
            shift,
            false,
            self.ionosphere.as_ref(),
        )
    }
}

/// A date with its next-along-track burst, for ESD.
struct DatePair {
    k: Acquisition,
    k1: Option<Acquisition>,
}

impl DatePair {
    fn remove_raw(&self) -> Result<()> {
        for acquisition in std::iter::once(&self.k).chain(self.k1.as_ref()) {
            if let Some(burst_dir) = acquisition.safe.parent().filter(|d| d.is_dir()) {
                std::fs::remove_dir_all(burst_dir)?;
            }
        }
        Ok(())
    }
}

/// Process every date of `request`'s burst. Returns the written files, oldest
/// first. Dates are fetched one at a time, so scratch holds one date's raw
/// bursts at once (they are removed after use unless `keep_raw`).
pub fn process_stack(client: &mut CdseClient, request: &StackRequest) -> Result<Vec<StackOutput>> {
    let records = find_dates(client, request)?;
    if records.is_empty() {
        return Err(L1Error::NotFound(format!(
            "no {} bursts in the window",
            request.burst_id.opera()
        )));
    }
    std::fs::create_dir_all(&request.output_dir)?;
    let mut outputs = Vec::with_capacity(records.len());
    let mut stack: Option<(Dem, Option<GeoGrid>)> = None;
    let mut previous: OverlapProducts = None;
    let mut shift = 0.0;
    let gim = request.ionosphere.then(GimClient::from_env).transpose()?;
    for (record, next) in records {
        let date = DatePair {
            k1: next
                .map(|n| Acquisition::load(client, gim.as_ref(), n, &request.cache_dir))
                .transpose()?,
            k: Acquisition::load(client, gim.as_ref(), record, &request.cache_dir)?,
        };
        if stack.is_none() {
            let esd_grid = match (&date.k1, request.esd) {
                (Some(k1), true) => {
                    Some(overlap_grid(&date.k.burst, &k1.burst, request.grid.epsg)?)
                }
                _ => None,
            };
            let dem = stack_dem(&date)?;
            write_static_once(&date.k, &dem, request)?;
            stack = Some((dem, esd_grid));
        }
        let Some((dem, esd_grid)) = stack.as_ref() else {
            unreachable!("stack context initialised above")
        };
        let output = process_date(
            &date,
            dem,
            esd_grid.as_ref(),
            request,
            &mut shift,
            &mut previous,
        )?;
        outputs.push(output);
        if !request.keep_raw {
            date.remove_raw()?;
        }
    }
    Ok(outputs)
}

fn process_date(
    date: &DatePair,
    dem: &Dem,
    esd_grid: Option<&GeoGrid>,
    request: &StackRequest,
    shift: &mut f64,
    previous: &mut OverlapProducts,
) -> Result<StackOutput> {
    let path = request.output_dir.join(output_name(request, &date.k.burst));
    let sensing_start = date.k.burst.sensing_start;
    if let Some(written_shift) = existing_output(&path, &date.k)? {
        *shift = written_shift;
        *previous = match (esd_grid, &date.k1) {
            (Some(grid), Some(k1)) => Some((
                date.k.geocode(dem, grid, *shift)?,
                k1.geocode(dem, grid, *shift)?,
            )),
            _ => None,
        };
        return Ok(StackOutput {
            sensing_start,
            path,
            azimuth_shift: *shift,
            esd: None,
        });
    }
    let (estimate, overlap) = match (esd_grid, &date.k1) {
        (Some(grid), Some(k1)) => esd_step(date, k1, dem, grid, shift, previous.take())?,
        _ => (None, None),
    };
    *previous = overlap;
    let geocoded = geocode_burst(
        &date.k.burst,
        &date.k.orbit,
        dem,
        &request.grid,
        *shift,
        request.phase_layers,
        date.k.ionosphere.as_ref(),
    )?;
    write_date(&path, &geocoded, &date.k, request, *shift)?;
    tracing::info!(file = %path.display(), shift = *shift, orbit = %date.k.orbit.orbit_type, "wrote CSLC");
    Ok(StackOutput {
        sensing_start,
        path,
        azimuth_shift: *shift,
        esd: estimate,
    })
}

type OverlapProducts = Option<(GeocodedBurst, GeocodedBurst)>;

/// Geocode this date's overlap, estimate its shift against the previous
/// date's overlap, update `shift`, and return this date's overlap products
/// (re-geocoded with the updated shift) for the next date.
fn esd_step(
    date: &DatePair,
    k1: &Acquisition,
    dem: &Dem,
    grid: &GeoGrid,
    shift: &mut f64,
    previous: OverlapProducts,
) -> Result<(Option<EsdEstimate>, OverlapProducts)> {
    let overlap_k = date.k.geocode(dem, grid, *shift)?;
    let overlap_k1 = k1.geocode(dem, grid, *shift)?;
    let Some((reference_k, reference_k1)) = previous else {
        return Ok((None, Some((overlap_k, overlap_k1))));
    };
    let pair = OverlapPair {
        reference_k: &reference_k,
        reference_k1: &reference_k1,
        secondary_k: &overlap_k,
        secondary_k1: &overlap_k1,
    };
    match pair.estimate() {
        Ok(mut estimate) => {
            *shift += estimate.azimuth_shift;
            let k = date.k.geocode(dem, grid, *shift)?;
            let k1 = k1.geocode(dem, grid, *shift)?;
            let check = OverlapPair {
                secondary_k: &k,
                secondary_k1: &k1,
                ..pair
            };
            estimate.residual_shift = check.estimate().map_or(f64::NAN, |e| e.azimuth_shift);
            Ok((Some(estimate), Some((k, k1))))
        }
        Err(error) => {
            tracing::warn!(%error, date = %date.k.burst.sensing_start, "ESD skipped; carrying the previous shift");
            Ok((None, Some((overlap_k, overlap_k1))))
        }
    }
}

/// Every burst of the window, each with its next-along-track burst when ESD
/// is on and CDSE has it.
fn find_dates(
    client: &CdseClient,
    request: &StackRequest,
) -> Result<Vec<(BurstRecord, Option<BurstRecord>)>> {
    let query = BurstQuery {
        burst_id: request.burst_id,
        polarization: request.polarization.clone(),
        start: request.start,
        end: request.end,
    };
    let records = client.search_bursts(&query)?;
    let next_records = match request.esd {
        true => client.search_bursts(&BurstQuery {
            burst_id: request.burst_id.next(),
            end: request.end + Duration::seconds(10),
            ..query
        })?,
        false => Vec::new(),
    };
    Ok(records
        .into_iter()
        .map(|record| {
            let next = next_records.iter().find(|n| {
                let gap = n.azimuth_time - record.azimuth_time;
                gap > Duration::zero() && gap < Duration::seconds(5)
            });
            (record.clone(), next.cloned())
        })
        .collect())
}

fn stack_dem(first: &DatePair) -> Result<Dem> {
    let mut points: Vec<(f64, f64)> = Vec::new();
    for acquisition in std::iter::once(&first.k).chain(first.k1.as_ref()) {
        points.extend(
            acquisition
                .burst
                .geolocation_grid()?
                .iter()
                .map(|p| (p.longitude, p.latitude)),
        );
    }
    Dem::glo30(LonLatBounds::around(points, DEM_MARGIN_DEG)?)
}

/// A 20 m grid in `epsg` over the overlap of bursts `k` and `k1`, from the
/// annotation tie points that bracket the overlap time span.
fn overlap_grid(k: &S1Burst, k1: &S1Burst, epsg: u32) -> Result<GeoGrid> {
    let line_time = |b: &S1Burst, line: usize| {
        b.sensing_start
            + Duration::nanoseconds((line as f64 * b.azimuth_time_interval * 1e9) as i64)
    };
    let (t_start, t_stop) = (
        line_time(k1, k1.first_valid_line),
        line_time(k, k.last_valid_line),
    );
    let ties: Vec<_> = k
        .geolocation_grid()?
        .into_iter()
        .chain(k1.geolocation_grid()?)
        .collect();
    let mut times: Vec<DateTime<Utc>> = ties.iter().map(|p| p.azimuth_time).collect();
    times.sort();
    times.dedup();
    let spacing = times
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or_else(Duration::zero);
    let (mut lon, mut lat): (Vec<f64>, Vec<f64>) = ties
        .iter()
        .filter(|p| p.azimuth_time >= t_start - spacing && p.azimuth_time <= t_stop + spacing)
        .map(|p| (p.longitude, p.latitude))
        .unzip();
    if lon.is_empty() {
        return Err(L1Error::Geometry("bursts k and k+1 do not overlap".into()));
    }
    Projection::new(epsg)?.to_projected(&mut lon, &mut lat)?;
    let snap = |v: f64, up: bool| match up {
        true => (v / ESD_POSTING).ceil() * ESD_POSTING,
        false => (v / ESD_POSTING).floor() * ESD_POSTING,
    };
    let fold = |v: &[f64], f: fn(f64, f64) -> f64, init: f64| v.iter().copied().fold(init, f);
    let (xmin, xmax) = (
        snap(fold(&lon, f64::min, f64::INFINITY), false),
        snap(fold(&lon, f64::max, f64::NEG_INFINITY), true),
    );
    let (ymin, ymax) = (
        snap(fold(&lat, f64::min, f64::INFINITY), false),
        snap(fold(&lat, f64::max, f64::NEG_INFINITY), true),
    );
    Ok(GeoGrid {
        epsg,
        x0: xmin + 0.5 * ESD_POSTING,
        y0: ymax - 0.5 * ESD_POSTING,
        dx: ESD_POSTING,
        dy: -ESD_POSTING,
        cols: ((xmax - xmin) / ESD_POSTING) as usize,
        rows: ((ymax - ymin) / ESD_POSTING) as usize,
    })
}

/// The ESD shift of an output that is already current: it exists and was
/// made with an orbit at least as good as the one available now.
fn existing_output(path: &Path, acquisition: &Acquisition) -> Result<Option<f64>> {
    if !path.is_file() {
        return Ok(None);
    }
    let file = hdf5::File::open(path)?;
    let orbit_type = file
        .dataset("metadata/orbit/orbit_type")?
        .read_scalar::<hdf5::types::FixedAscii<64>>()?;
    if orbit_type.as_str() != "POEORB" && acquisition.orbit.orbit_type == "POEORB" {
        return Ok(None);
    }
    let shift = file
        .dataset("metadata/processing_information/parameters/esd_azimuth_shift_seconds")?
        .read_scalar::<f64>()?;
    Ok(Some(shift))
}

/// Write the stack's static layers (LOS, layover/shadow) from its first
/// date, unless a static file for this burst grid is already there.
fn write_static_once(
    reference: &Acquisition,
    dem: &Dem,
    request: &StackRequest,
) -> Result<PathBuf> {
    let id = request.burst_id;
    let path = request.output_dir.join(format!(
        "STATIC-S1_T{:03}-{:06}-{}_{}.h5",
        id.relative_orbit,
        id.burst,
        id.swath.as_str(),
        reference.burst.sensing_start.format("%Y%m%d"),
    ));
    let prefix = format!(
        "STATIC-S1_T{:03}-{:06}-{}_",
        id.relative_orbit,
        id.burst,
        id.swath.as_str()
    );
    let existing = std::fs::read_dir(&request.output_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix))
        });
    if let Some(found) = existing {
        return Ok(found);
    }
    let layers = static_layers(&reference.burst, &reference.orbit, dem, &request.grid)?;
    write_static(&path, &layers, &id.opera())?;
    tracing::info!(file = %path.display(), "wrote static layers");
    Ok(path)
}

fn output_name(request: &StackRequest, burst: &S1Burst) -> String {
    let id = request.burst_id;
    format!(
        "CSLC-S1_T{:03}-{:06}-{}_{}_{}.h5",
        id.relative_orbit,
        id.burst,
        id.swath.as_str(),
        burst.sensing_start.format("%Y%m%dT%H%M%SZ"),
        request.polarization
    )
}

fn write_date(
    path: &Path,
    geocoded: &GeocodedBurst,
    acquisition: &Acquisition,
    request: &StackRequest,
    shift: f64,
) -> Result<()> {
    let ties = acquisition.burst.geolocation_grid()?;
    let n = ties.len() as f64;
    let center = [
        ties.iter().map(|p| p.longitude).sum::<f64>() / n,
        ties.iter().map(|p| p.latitude).sum::<f64>() / n,
    ];
    let burst_id = request.burst_id.opera();
    write_cslc(
        path,
        geocoded,
        &acquisition.burst,
        &acquisition.orbit,
        &CslcProvenance {
            burst_id: &burst_id,
            track: request.burst_id.relative_orbit,
            l1_slc: &acquisition.record.parent_product_name,
            orbit_file: &acquisition.orbit_file,
            gim_file: acquisition.gim_file.as_deref(),
            azimuth_shift: shift,
            center_lonlat: center,
        },
    )
}
