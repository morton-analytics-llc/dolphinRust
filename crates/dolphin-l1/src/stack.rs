//! Sentinel-1 burst stack: raw CDSE bursts in, geocoded CSLC HDF5 out.
//!
//! For every date of one burst id: download the burst, its POEORB and the
//! day's JPL GIM, geocode the burst onto the fixed burst grid
//! ([`crate::geocode`]), and write the OPERA-layout HDF5 the phase-linking
//! path reads. The DEM is read once per stack, and the static layers (LOS,
//! layover/shadow) are written once from the first date. Every date is
//! geocoded independently onto the same grid, as COMPASS does for OPERA
//! CSLC-S1; precise orbits and the timing corrections coregister the dates
//! (measured against OPERA on Houston: offsets below 1/16 pixel, per-date
//! correlation ≥ 0.99999), so no enhanced-spectral-diversity step is applied.
//!
//! **Live operation.** CDSE publishes a new IW SLC 2–21 h after acquisition
//! (`validation/results/issue230_raw_front_end/latency.json`), and a burst is
//! downloadable as soon as its parent is. The host polls by calling
//! [`process_stack`] with `start` at its last processed acquisition: dates
//! whose output exists with a precise orbit are skipped before any download,
//! new dates are processed, and a date written with a restituted orbit
//! (RESORB, the only orbit for the first ~3 weeks) is reprocessed once its
//! POEORB exists.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use dolphin_corrections::ionosphere::IonexMaps;

use crate::cdse::{BurstId, BurstQuery, BurstRecord, CdseClient};
use crate::cslc_h5::{write_cslc, CslcProvenance};
use crate::dem::{Dem, LonLatBounds};
use crate::error::{L1Error, Result};
use crate::geocode::{geocode_burst, GeocodedBurst};
use crate::gim::{read_gim, GimClient};
use crate::grid::GeoGrid;
use crate::orbit::Orbit;
use crate::safe::S1Burst;
use crate::static_layers::{static_layers, write_static, RadarGeometry};

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
    /// Raw-data cache (bursts, orbits, annotations, GIMs).
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

/// One date of the stack.
#[derive(Debug, Clone)]
pub struct StackOutput {
    /// Zero-Doppler time of the burst's first line (catalogue time for a
    /// date that was already current and skipped).
    pub sensing_start: DateTime<Utc>,
    /// The CSLC HDF5 file.
    pub path: PathBuf,
    /// `false` when the file was already current and left as it was.
    pub written: bool,
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

    fn geocode(&self, dem: &Dem, request: &StackRequest) -> Result<GeocodedBurst> {
        geocode_burst(
            &self.burst,
            &self.orbit,
            dem,
            &request.grid,
            request.phase_layers,
            self.ionosphere.as_ref(),
        )
    }

    fn remove_raw(&self) -> Result<()> {
        if let Some(burst_dir) = self.safe.parent().filter(|d| d.is_dir()) {
            std::fs::remove_dir_all(burst_dir)?;
        }
        Ok(())
    }
}

/// Process every date of `request`'s burst. Returns one entry per date,
/// oldest first. Dates are fetched one at a time, so scratch holds one
/// date's raw burst at once (removed after use unless `keep_raw`).
pub fn process_stack(client: &mut CdseClient, request: &StackRequest) -> Result<Vec<StackOutput>> {
    let query = BurstQuery {
        burst_id: request.burst_id,
        polarization: request.polarization.clone(),
        start: request.start,
        end: request.end,
    };
    let records = client.search_bursts(&query)?;
    if records.is_empty() {
        return Err(L1Error::NotFound(format!(
            "no {} bursts in the window",
            request.burst_id.opera()
        )));
    }
    std::fs::create_dir_all(&request.output_dir)?;
    let gim = request.ionosphere.then(GimClient::from_env).transpose()?;
    let mut dem: Option<Dem> = None;
    let mut outputs = Vec::with_capacity(records.len());
    for record in records {
        if let Some(path) = current_output(client, request, &record)? {
            outputs.push(StackOutput {
                sensing_start: record.azimuth_time,
                path,
                written: false,
            });
            continue;
        }
        let acquisition = Acquisition::load(client, gim.as_ref(), record, &request.cache_dir)?;
        if dem.is_none() {
            let d = stack_dem(&acquisition)?;
            write_static_once(&acquisition, &d, request)?;
            dem = Some(d);
        }
        let Some(dem) = dem.as_ref() else {
            unreachable!("DEM initialised above")
        };
        let path = request
            .output_dir
            .join(output_name(request, &acquisition.burst));
        let geocoded = acquisition.geocode(dem, request)?;
        write_date(&path, &geocoded, &acquisition, request)?;
        tracing::info!(file = %path.display(), orbit = %acquisition.orbit.orbit_type, "wrote CSLC");
        outputs.push(StackOutput {
            sensing_start: acquisition.burst.sensing_start,
            path,
            written: true,
        });
        if !request.keep_raw {
            acquisition.remove_raw()?;
        }
    }
    Ok(outputs)
}

/// The existing output for `record`'s date when it is current: written with
/// a precise orbit, or with a restituted one while no POEORB covers it yet.
fn current_output(
    client: &CdseClient,
    request: &StackRequest,
    record: &BurstRecord,
) -> Result<Option<PathBuf>> {
    let day = record.azimuth_time.format("%Y%m%d").to_string();
    let prefix = format!("{}_{day}T", name_prefix(request.burst_id));
    let found = std::fs::read_dir(&request.output_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix))
        });
    let Some(path) = found else {
        return Ok(None);
    };
    let orbit_type = hdf5::File::open(&path)?
        .dataset("metadata/orbit/orbit_type")?
        .read_scalar::<hdf5::types::FixedAscii<64>>()?;
    if orbit_type.as_str() == "POEORB" {
        return Ok(Some(path));
    }
    let margin = Duration::seconds(ORBIT_MARGIN_SECONDS);
    let best = client.search_orbit(
        record.mission(),
        record.azimuth_time - margin,
        record.azimuth_time + margin + Duration::seconds(4),
    )?;
    Ok((!best.is_precise()).then_some(path))
}

fn stack_dem(first: &Acquisition) -> Result<Dem> {
    let points: Vec<(f64, f64)> = first
        .burst
        .geolocation_grid()?
        .iter()
        .map(|p| (p.longitude, p.latitude))
        .collect();
    Dem::glo30(LonLatBounds::around(points, DEM_MARGIN_DEG)?)
}

fn name_prefix(id: BurstId) -> String {
    format!(
        "CSLC-S1_T{:03}-{:06}-{}",
        id.relative_orbit,
        id.burst,
        id.swath.as_str()
    )
}

/// Write the stack's static layers (LOS, layover/shadow) from its first
/// date, unless a static file for this burst is already there.
fn write_static_once(
    reference: &Acquisition,
    dem: &Dem,
    request: &StackRequest,
) -> Result<PathBuf> {
    let id = request.burst_id;
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
    let path = request.output_dir.join(format!(
        "{prefix}{}.h5",
        reference.burst.sensing_start.format("%Y%m%d")
    ));
    let layers = static_layers(
        &RadarGeometry::from_s1(&reference.burst),
        &reference.orbit,
        dem,
        &request.grid,
    )?;
    write_static(&path, &layers, &id.opera())?;
    tracing::info!(file = %path.display(), "wrote static layers");
    Ok(path)
}

fn output_name(request: &StackRequest, burst: &S1Burst) -> String {
    format!(
        "{}_{}_{}.h5",
        name_prefix(request.burst_id),
        burst.sensing_start.format("%Y%m%dT%H%M%SZ"),
        request.polarization
    )
}

fn write_date(
    path: &Path,
    geocoded: &GeocodedBurst,
    acquisition: &Acquisition,
    request: &StackRequest,
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
            center_lonlat: center,
        },
    )
}
