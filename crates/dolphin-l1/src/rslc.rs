//! NISAR L1 RSLC: catalogue search on ASF, windowed remote reads, and the
//! radar-grid metadata the geocoder needs.
//!
//! **Source.** `NISAR_L1_RSLC_PROVISIONAL_V1` (CMR `C2853145197-ASF`), one
//! HDF5 per track/frame/date, about 25 GB for a full 40+5 MHz dual-pol frame.
//! Nothing is downloaded whole: ASF answers the Earthdata bearer token with a
//! 303 to a CloudFront URL signed for GET (valid an hour), and GDAL's HDF5
//! driver reads that URL through `/vsicurl/` with HTTP range requests, so
//! only the metadata and the 512 × 512 chunks under the output grid are
//! transferred. The bearer token itself never reaches GDAL. A local RSLC
//! path works the same way.
//!
//! **What is read** (`/science/LSAR/RSLC`): the zero-Doppler time axis
//! (`swaths/zeroDopplerTime`, seconds since the epoch in its `units`), per
//! band (`frequencyA`, `frequencyB`) the slant-range axis, processed centre
//! frequency and range bandwidth, polarisations and valid-sample bounds of
//! each sub-swath; the native Doppler centroid LUT
//! (`metadata/processingInformation/parameters/frequencyX/dopplerCentroid`);
//! the orbit (`metadata/orbit`) and attitude (`metadata/attitude`); look
//! side and pass direction (`identification`).
//!
//! **Conventions checked on real granules (2026-09/10, track 5 frame 20):**
//! the image grid is zero-Doppler (geo2rdr solves `V·(S − P) = 0`), but the
//! data spectrum is not at zero Doppler: the native Doppler centroid is
//! 937–968 Hz at a 1520 Hz line rate, so the azimuth interpolation is done on
//! basebanded data ([`crate::nisar_geocode`]). The radar looks left. The
//! orbit in the RSLC is the MOE orbit, identical vector for vector to the
//! one JPL's GSLC carries. Azimuth line `i` is at `zeroDopplerTime[0] + i ·
//! zeroDopplerTimeSpacing`; range sample `j` at `slantRange[0] + j ·
//! slantRangeSpacing`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use gdal::cpl::CslStringList;
use gdal::raster::MDArray;
use gdal::{Dataset, DatasetOptions, GdalOpenFlags};
use ndarray::Array2;
use num_complex::Complex32;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, LOCATION, RANGE};
use reqwest::redirect::Policy;
use serde::Deserialize;

use crate::error::{product, L1Error, Result};
use crate::geometry::LookSide;
use crate::orbit::{parse_utc, Orbit, Vec3};
use crate::safe::read_complex_window;

/// CMR collection of the provisional NISAR L1 RSLC.
pub const RSLC_COLLECTION: &str = "C2853145197-ASF";
/// CMR collection of the NISAR ionosphere TEC ancillary (`NISAR_ANC_TEC`).
pub const TEC_COLLECTION: &str = "C3622219145-ASF";
const CMR_GRANULES: &str = "https://cmr.earthdata.nasa.gov/search/granules.json";
/// The only host the Earthdata bearer token is sent to.
const ASF_HOST: &str = "nisar.asf.earthdatacloud.nasa.gov";
const PRODUCT: &str = "/science/LSAR/RSLC";
const ATTEMPTS: u32 = 4;

/// A NISAR track/frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NisarFrame {
    /// Relative orbit (track), 1–173.
    pub track: u32,
    /// `'A'` ascending or `'D'` descending.
    pub direction: char,
    /// Frame number.
    pub frame: u32,
}

impl NisarFrame {
    /// `T005-A-F020`, used in output file names.
    pub fn label(self) -> String {
        format!("T{:03}-{}-F{:03}", self.track, self.direction, self.frame)
    }
}

/// One catalogue granule.
#[derive(Debug, Clone)]
pub struct GranuleRecord {
    /// Granule name (file stem).
    pub name: String,
    /// HTTPS download URL on ASF.
    pub url: String,
    /// Acquisition (or validity) start.
    pub start: DateTime<Utc>,
    /// Acquisition (or validity) end.
    pub end: DateTime<Utc>,
}

#[derive(Deserialize)]
struct CmrFeed {
    feed: CmrEntries,
}

#[derive(Deserialize)]
struct CmrEntries {
    entry: Vec<CmrEntry>,
}

#[derive(Deserialize)]
struct CmrEntry {
    title: String,
    time_start: String,
    time_end: String,
    links: Vec<CmrLink>,
}

#[derive(Deserialize)]
struct CmrLink {
    href: String,
}

impl CmrEntry {
    fn record(self, extension: &str) -> Result<GranuleRecord> {
        let url = self
            .links
            .iter()
            .map(|l| l.href.as_str())
            .find(|h| h.starts_with(&format!("https://{ASF_HOST}/")) && h.ends_with(extension))
            .ok_or_else(|| L1Error::NotFound(format!("{extension} link for {}", self.title)))?
            .to_string();
        Ok(GranuleRecord {
            start: parse_utc(&self.time_start)?,
            end: parse_utc(&self.time_end)?,
            name: self.title,
            url,
        })
    }
}

/// The RSLC granule-name fields used to pick one granule per acquisition:
/// (acquisition start token, product counter).
fn rslc_identity(name: &str) -> Option<(&str, u32)> {
    let fields: Vec<&str> = name.split('_').collect();
    Some((fields.get(11)?, fields.last()?.parse().ok()?))
}

/// ASF/CMR client holding the Earthdata bearer token.
pub struct AsfClient {
    http: Client,
    token: String,
}

impl AsfClient {
    /// A client with the token from [`crate::earthdata::bearer_token`].
    ///
    /// # Errors
    /// Missing credentials or an EDL failure.
    pub fn from_env() -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .redirect(Policy::none())
            .build()?;
        Ok(Self {
            http,
            token: crate::earthdata::bearer_token()?,
        })
    }

    fn cmr(&self, query: &[(&str, String)], extension: &str) -> Result<Vec<GranuleRecord>> {
        let collection = query.first().map_or("", |q| q.1.as_str());
        tracing::info!(url = CMR_GRANULES, collection, "CMR granule search");
        let feed: CmrFeed = self.retry("CMR", CMR_GRANULES, || {
            self.http.get(CMR_GRANULES).query(query).send()
        })?;
        feed.feed
            .entry
            .into_iter()
            .map(|e| e.record(extension))
            .collect()
    }

    /// RSLC granules of `frame` acquired in `[start, end)`, oldest first.
    /// When an acquisition was processed more than once, the granule with
    /// the highest product counter (the reprocessing) is kept.
    ///
    /// # Errors
    /// CMR failures.
    pub fn search_rslc(
        &self,
        frame: NisarFrame,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<GranuleRecord>> {
        let pattern = format!(
            "NISAR_L1_*_RSLC_*_{:03}_{}_{:03}_*",
            frame.track, frame.direction, frame.frame
        );
        let query = [
            ("collection_concept_id", RSLC_COLLECTION.to_string()),
            ("readable_granule_name", pattern),
            ("options[readable_granule_name][pattern]", "true".into()),
            (
                "temporal",
                format!("{},{}", start.to_rfc3339(), end.to_rfc3339()),
            ),
            ("sort_key", "start_date".into()),
            ("page_size", "2000".into()),
        ];
        let mut records = self.cmr(&query, ".h5")?;
        records.retain(|r| r.start >= start && r.start < end);
        records.sort_by(|a, b| {
            let (ka, kb) = (rslc_identity(&a.name), rslc_identity(&b.name));
            ka.map(|k| k.0)
                .cmp(&kb.map(|k| k.0))
                .then(kb.map(|k| k.1).cmp(&ka.map(|k| k.1)))
        });
        records.dedup_by(|later, kept| {
            rslc_identity(&later.name).map(|k| k.0) == rslc_identity(&kept.name).map(|k| k.0)
        });
        Ok(records)
    }

    /// The newest `NISAR_ANC_TEC` file whose validity covers
    /// `[start, end]`.
    ///
    /// # Errors
    /// [`L1Error::NotFound`] when no TEC file covers the span yet.
    pub fn search_tec(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<GranuleRecord> {
        let query = [
            ("collection_concept_id", TEC_COLLECTION.to_string()),
            (
                "temporal",
                format!("{},{}", start.to_rfc3339(), end.to_rfc3339()),
            ),
            ("page_size", "50".into()),
        ];
        // Names carry the creation time first: NISAR_ANC_TEC_<created>_<start>_<end>.
        self.cmr(&query, ".json")?
            .into_iter()
            .filter(|r| r.start <= start && r.end >= end)
            .max_by(|a, b| a.name.cmp(&b.name))
            .ok_or_else(|| L1Error::NotFound(format!("NISAR TEC file covering {start} .. {end}")))
    }

    /// A GET-signed CloudFront URL for an ASF download URL (valid ~1 h),
    /// readable without credentials.
    ///
    /// # Errors
    /// The URL is not on ASF, or ASF does not redirect.
    pub fn signed_url(&self, url: &str) -> Result<String> {
        if !url.starts_with(&format!("https://{ASF_HOST}/")) {
            return Err(L1Error::Product(format!(
                "{url} is not an ASF NISAR URL; the Earthdata token is not sent to it"
            )));
        }
        let response = self.retry_raw("ASF", url, || {
            self.http
                .get(url)
                .header(AUTHORIZATION, format!("Bearer {}", self.token))
                .header(RANGE, "bytes=0-0")
                .send()
        })?;
        response
            .headers()
            .get(LOCATION)
            .and_then(|v| v.to_str().ok())
            .filter(|l| l.starts_with("https://"))
            .map(str::to_string)
            .ok_or_else(|| L1Error::Service {
                service: "ASF",
                status: response.status().as_u16(),
                url: url.to_string(),
                body: "no signed redirect".into(),
            })
    }

    /// Download a small ASF file (TEC JSON) into `dir`, cached by name.
    ///
    /// # Errors
    /// HTTP or I/O failures.
    pub fn download(&self, record: &GranuleRecord, dir: &Path) -> Result<PathBuf> {
        let name = record
            .url
            .rsplit('/')
            .next()
            .ok_or_else(|| L1Error::Product(format!("no file name in {}", record.url)))?;
        let dest = dir.join(name);
        if dest.is_file() {
            return Ok(dest);
        }
        let signed = self.signed_url(&record.url)?;
        let response = self.retry_raw("ASF", &signed, || self.http.get(&signed).send())?;
        let bytes = response.bytes()?;
        std::fs::create_dir_all(dir)?;
        let part = dest.with_extension("part");
        std::fs::write(&part, &bytes)?;
        std::fs::rename(&part, &dest)?;
        tracing::info!(url = %record.url, "fetched from ASF");
        Ok(dest)
    }

    fn retry<T: serde::de::DeserializeOwned>(
        &self,
        service: &'static str,
        url: &str,
        send: impl Fn() -> reqwest::Result<reqwest::blocking::Response>,
    ) -> Result<T> {
        Ok(self.retry_raw(service, url, send)?.json()?)
    }

    fn retry_raw(
        &self,
        service: &'static str,
        url: &str,
        send: impl Fn() -> reqwest::Result<reqwest::blocking::Response>,
    ) -> Result<reqwest::blocking::Response> {
        let mut attempt = 1;
        loop {
            let error = match send() {
                Ok(r) if r.status().is_success() || r.status().is_redirection() => return Ok(r),
                Ok(r) => L1Error::Service {
                    service,
                    status: r.status().as_u16(),
                    url: url.split('?').next().unwrap_or(url).to_string(),
                    body: String::new(),
                },
                Err(e) => L1Error::Http(e),
            };
            let transient = match &error {
                L1Error::Service { status, .. } => *status == 429 || *status >= 500,
                _ => true,
            };
            if !transient || attempt >= ATTEMPTS {
                return Err(error);
            }
            tracing::warn!(%error, attempt, "{service} request failed; retrying");
            std::thread::sleep(Duration::from_secs(10) * attempt);
            attempt += 1;
        }
    }
}

/// A 2-D lookup table on (azimuth time, slant range), bilinear, clamped at
/// the edges.
#[derive(Debug, Clone)]
pub struct Lut2d {
    /// Azimuth times (s since the product epoch), increasing.
    pub times: Vec<f64>,
    /// Slant ranges (m), increasing.
    pub ranges: Vec<f64>,
    /// `values[time][range]`.
    pub values: Array2<f64>,
}

fn bracket(axis: &[f64], x: f64) -> (usize, f64) {
    let last = axis.len() - 1;
    if last == 0 {
        return (0, 0.0);
    }
    let i = axis.partition_point(|&a| a <= x).clamp(1, last) - 1;
    let w = ((x - axis[i]) / (axis[i + 1] - axis[i])).clamp(0.0, 1.0);
    (i, w)
}

impl Lut2d {
    /// Bilinear value at `(time, range)`; outside the axes the edge value.
    pub fn eval(&self, time: f64, range: f64) -> f64 {
        let (i, wi) = bracket(&self.times, time);
        let (j, wj) = bracket(&self.ranges, range);
        let at = |a: usize, b: usize| {
            self.values[[a.min(self.times.len() - 1), b.min(self.ranges.len() - 1)]]
        };
        let top = at(i, j) * (1.0 - wj) + at(i, j + 1) * wj;
        let bottom = at(i + 1, j) * (1.0 - wj) + at(i + 1, j + 1) * wj;
        top * (1.0 - wi) + bottom * wi
    }

    /// `true` when `(time, range)` lies within the table's axes.
    pub fn contains(&self, time: f64, range: f64) -> bool {
        let within = |axis: &[f64], x: f64| axis[0] <= x && x <= axis[axis.len() - 1];
        within(&self.times, time) && within(&self.ranges, range)
    }
}

/// One frequency band of an RSLC.
#[derive(Debug, Clone)]
pub struct RslcBand {
    /// `'A'` or `'B'`.
    pub frequency: char,
    /// Slant range of sample 0 (m).
    pub starting_range: f64,
    /// Slant-range spacing (m).
    pub range_spacing: f64,
    /// Range samples per line.
    pub samples: usize,
    /// Processed centre frequency (Hz).
    pub center_frequency: f64,
    /// Processed range bandwidth (Hz).
    pub range_bandwidth: f64,
    /// Polarisations present, e.g. `["HH", "HV"]`.
    pub polarizations: Vec<String>,
    /// Native Doppler centroid (Hz).
    pub doppler: Lut2d,
    /// Valid `[first, last)` range samples per line, one vector per
    /// sub-swath.
    pub valid_samples: Vec<Vec<[u32; 2]>>,
}

impl RslcBand {
    /// Radar wavelength (m).
    pub fn wavelength(&self) -> f64 {
        crate::safe::SPEED_OF_LIGHT / self.center_frequency
    }

    /// Sub-swath (1-based) holding `sample` on `line`, 0 when none does.
    pub fn subswath(&self, line: i64, sample: i64) -> u8 {
        let (Ok(line), Ok(sample)) = (usize::try_from(line), u32::try_from(sample)) else {
            return 0;
        };
        self.valid_samples
            .iter()
            .position(|swath| {
                swath
                    .get(line)
                    .is_some_and(|[a, b]| (*a..*b).contains(&sample))
            })
            .map_or(0, |k| (k + 1) as u8)
    }
}

/// Attitude quaternions as stored in the RSLC.
#[derive(Debug, Clone)]
pub struct Attitude {
    /// Times, seconds since the product epoch.
    pub times: Vec<f64>,
    /// Quaternions `[q0, q1, q2, q3]` per time.
    pub quaternions: Vec<[f64; 4]>,
}

/// An opened NISAR L1 RSLC.
pub struct Rslc {
    /// GDAL path of the HDF5 (local path or `/vsicurl/` signed URL).
    source: String,
    /// Granule id from `identification/granuleId`.
    pub granule_id: String,
    /// Look side (NISAR: left).
    pub look_side: LookSide,
    /// `Ascending` / `Descending`.
    pub pass_direction: String,
    /// Absolute orbit number.
    pub absolute_orbit: u32,
    /// Track (relative orbit).
    pub track: u32,
    /// Frame.
    pub frame: u32,
    /// Epoch that every time below counts seconds from.
    pub epoch: DateTime<Utc>,
    /// Zero-Doppler time of line 0 (s since `epoch`).
    pub first_line_time: f64,
    /// Line spacing (s).
    pub line_spacing: f64,
    /// Lines.
    pub lines: usize,
    /// Orbit (MOE/POE as embedded), on `epoch`.
    pub orbit: Orbit,
    /// Attitude.
    pub attitude: Attitude,
    /// Bands present, `frequencyA` first.
    pub bands: Vec<RslcBand>,
}

fn open_multidim(source: &str) -> Result<Dataset> {
    Ok(Dataset::open_ex(
        source,
        DatasetOptions {
            open_flags: GdalOpenFlags::GDAL_OF_MULTIDIM_RASTER | GdalOpenFlags::GDAL_OF_READONLY,
            allowed_drivers: Some(&["HDF5"]),
            ..Default::default()
        },
    )?)
}

/// Run `read` on the array at absolute `path`.
fn with_array<T>(
    dataset: &Dataset,
    path: &str,
    read: impl FnOnce(&MDArray) -> Result<T>,
) -> Result<T> {
    let (parent, name) = path
        .rsplit_once('/')
        .ok_or_else(|| L1Error::Product(format!("bad HDF5 path {path}")))?;
    let mut group = dataset.root_group()?;
    for part in parent.split('/').filter(|p| !p.is_empty()) {
        group = group.open_group(part, CslStringList::new())?;
    }
    let array = group
        .open_md_array(name, CslStringList::new())
        .map_err(|e| L1Error::Product(format!("{path}: {e}")))?;
    read(&array)
}

fn read_f64(dataset: &Dataset, path: &str) -> Result<Vec<f64>> {
    with_array(dataset, path, |a| {
        let shape: Vec<usize> = a.dimensions()?.iter().map(|d| d.size()).collect();
        Ok(a.read_as::<f64>(vec![0; shape.len()], shape)?)
    })
}

fn read_scalar(dataset: &Dataset, path: &str) -> Result<f64> {
    read_f64(dataset, path)?
        .first()
        .copied()
        .ok_or_else(|| L1Error::Product(format!("{path} is empty")))
}

fn read_strings(dataset: &Dataset, path: &str) -> Result<Vec<String>> {
    with_array(dataset, path, |a| Ok(a.read_as_string_array()?))
}

fn read_string(dataset: &Dataset, path: &str) -> Result<String> {
    read_strings(dataset, path)?
        .into_iter()
        .next()
        .ok_or_else(|| L1Error::Product(format!("{path} is empty")))
}

/// The epoch of a `seconds since <time>` units attribute.
fn units_epoch(dataset: &Dataset, path: &str) -> Result<DateTime<Utc>> {
    let units = with_array(dataset, path, |a| {
        Ok(a.attribute("units")?.read_as_string())
    })?;
    let since = units
        .strip_prefix("seconds since ")
        .ok_or_else(|| L1Error::Product(format!("{path} units {units:?}")))?;
    parse_utc(since)
}

fn rows3(flat: &[f64]) -> Vec<Vec3> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn read_orbit(dataset: &Dataset, epoch: DateTime<Utc>) -> Result<Orbit> {
    let group = format!("{PRODUCT}/metadata/orbit");
    let shift = crate::orbit::seconds_since(epoch, units_epoch(dataset, &format!("{group}/time"))?);
    Ok(Orbit {
        reference_epoch: epoch,
        times: read_f64(dataset, &format!("{group}/time"))?
            .into_iter()
            .map(|t| t + shift)
            .collect(),
        positions: rows3(&read_f64(dataset, &format!("{group}/position"))?),
        velocities: rows3(&read_f64(dataset, &format!("{group}/velocity"))?),
        orbit_type: read_string(dataset, &format!("{group}/orbitType"))?,
    })
}

fn read_attitude(dataset: &Dataset, epoch: DateTime<Utc>) -> Result<Attitude> {
    let group = format!("{PRODUCT}/metadata/attitude");
    let shift = crate::orbit::seconds_since(epoch, units_epoch(dataset, &format!("{group}/time"))?);
    Ok(Attitude {
        times: read_f64(dataset, &format!("{group}/time"))?
            .into_iter()
            .map(|t| t + shift)
            .collect(),
        quaternions: read_f64(dataset, &format!("{group}/quaternions"))?
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect(),
    })
}

fn read_band(dataset: &Dataset, frequency: char, epoch: DateTime<Utc>) -> Result<RslcBand> {
    let swath = format!("{PRODUCT}/swaths/frequency{frequency}");
    let doppler =
        format!("{PRODUCT}/metadata/processingInformation/parameters/frequency{frequency}");
    let ranges = read_f64(dataset, &format!("{swath}/slantRange"))?;
    let shift = crate::orbit::seconds_since(
        epoch,
        units_epoch(dataset, &format!("{doppler}/zeroDopplerTime"))?,
    );
    let doppler_times: Vec<f64> = read_f64(dataset, &format!("{doppler}/zeroDopplerTime"))?
        .into_iter()
        .map(|t| t + shift)
        .collect();
    let doppler_ranges = read_f64(dataset, &format!("{doppler}/slantRange"))?;
    let doppler_values = Array2::from_shape_vec(
        (doppler_times.len(), doppler_ranges.len()),
        read_f64(dataset, &format!("{doppler}/dopplerCentroid"))?,
    )
    .map_err(product)?;
    let subswaths = read_scalar(dataset, &format!("{swath}/numberOfSubSwaths"))? as usize;
    let valid_samples = (1..=subswaths)
        .map(|k| {
            Ok(
                read_f64(dataset, &format!("{swath}/validSamplesSubSwath{k}"))?
                    .chunks_exact(2)
                    .map(|c| [c[0] as u32, c[1] as u32])
                    .collect(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RslcBand {
        frequency,
        starting_range: ranges[0],
        range_spacing: read_scalar(dataset, &format!("{swath}/slantRangeSpacing"))?,
        samples: ranges.len(),
        center_frequency: read_scalar(dataset, &format!("{swath}/processedCenterFrequency"))?,
        range_bandwidth: read_scalar(dataset, &format!("{swath}/processedRangeBandwidth"))?,
        polarizations: read_strings(dataset, &format!("{swath}/listOfPolarizations"))?,
        doppler: Lut2d {
            times: doppler_times,
            ranges: doppler_ranges,
            values: doppler_values,
        },
        valid_samples,
    })
}

/// GDAL settings for windowed HTTP reads of the RSLC: size probes by GET
/// (the signed URL is valid for GET only), no directory listing, merged
/// consecutive ranges and 1 MiB requests with a 256 MiB block cache.
fn configure_vsicurl() -> Result<()> {
    for (key, value) in [
        ("CPL_VSIL_CURL_USE_HEAD", "NO"),
        ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"),
        ("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES"),
        ("CPL_VSIL_CURL_CHUNK_SIZE", "1048576"),
        ("CPL_VSIL_CURL_CACHE_SIZE", "268435456"),
        ("GDAL_HTTP_MAX_RETRY", "4"),
        ("GDAL_HTTP_RETRY_DELAY", "10"),
    ] {
        gdal::config::set_config_option(key, value)?;
    }
    Ok(())
}

impl Rslc {
    /// Open the RSLC behind an ASF granule through a signed URL.
    ///
    /// # Errors
    /// Signing, HTTP or product-layout failures.
    pub fn open_remote(client: &AsfClient, record: &GranuleRecord) -> Result<Self> {
        configure_vsicurl()?;
        tracing::info!(url = %record.url, "opening RSLC for range reads");
        let signed = client.signed_url(&record.url)?;
        Self::open_source(format!("/vsicurl/{signed}"))
    }

    /// Open a local RSLC file.
    ///
    /// # Errors
    /// Product-layout failures.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_source(path.display().to_string())
    }

    fn open_source(source: String) -> Result<Self> {
        let dataset = open_multidim(&source)?;
        let ident = "/science/LSAR/identification";
        let times_path = format!("{PRODUCT}/swaths/zeroDopplerTime");
        let epoch = units_epoch(&dataset, &times_path)?;
        let times = read_f64(&dataset, &times_path)?;
        let look = read_string(&dataset, &format!("{ident}/lookDirection"))?;
        let frequencies = read_strings(&dataset, &format!("{ident}/listOfFrequencies"))?;
        let bands = ['A', 'B']
            .into_iter()
            .filter(|f| frequencies.iter().any(|s| s.trim() == f.to_string()))
            .map(|f| read_band(&dataset, f, epoch))
            .collect::<Result<Vec<_>>>()?;
        let integer = |name: &str| -> Result<u32> {
            Ok(read_scalar(&dataset, &format!("{ident}/{name}"))? as u32)
        };
        Ok(Self {
            granule_id: read_string(&dataset, &format!("{ident}/granuleId"))?,
            look_side: LookSide::parse(&look)?,
            pass_direction: read_string(&dataset, &format!("{ident}/orbitPassDirection"))?,
            absolute_orbit: integer("absoluteOrbitNumber")?,
            track: integer("trackNumber")?,
            frame: integer("frameNumber")?,
            first_line_time: times[0],
            line_spacing: read_scalar(
                &dataset,
                &format!("{PRODUCT}/swaths/zeroDopplerTimeSpacing"),
            )?,
            lines: times.len(),
            orbit: read_orbit(&dataset, epoch)?,
            attitude: read_attitude(&dataset, epoch)?,
            epoch,
            bands,
            source,
        })
    }

    /// The band for `frequency`.
    ///
    /// # Errors
    /// [`L1Error::NotFound`] when the product has no such band.
    pub fn band(&self, frequency: char) -> Result<&RslcBand> {
        self.bands
            .iter()
            .find(|b| b.frequency == frequency)
            .ok_or_else(|| {
                L1Error::NotFound(format!("frequency{frequency} in {}", self.granule_id))
            })
    }

    /// Zero-Doppler time of fractional `line` (s since `epoch`).
    pub fn line_time(&self, line: f64) -> f64 {
        self.first_line_time + line * self.line_spacing
    }

    /// Absolute time of seconds since `epoch`.
    pub fn datetime(&self, t: f64) -> DateTime<Utc> {
        self.epoch + crate::orbit::duration_from_seconds(t)
    }

    /// Sensing start (line 0).
    pub fn sensing_start(&self) -> DateTime<Utc> {
        self.datetime(self.first_line_time)
    }

    /// Sensing stop (last line).
    pub fn sensing_stop(&self) -> DateTime<Utc> {
        self.datetime(self.line_time((self.lines - 1) as f64))
    }

    /// Read lines `[row0, row0 + rows)`, samples `[col0, col0 + cols)` of
    /// `frequency`/`polarization`.
    ///
    /// # Errors
    /// GDAL failures or a window outside the swath.
    pub fn read_window(
        &self,
        frequency: char,
        polarization: &str,
        (row0, col0): (usize, usize),
        (rows, cols): (usize, usize),
    ) -> Result<Array2<Complex32>> {
        let band = self.band(frequency)?;
        if row0 + rows > self.lines || col0 + cols > band.samples {
            return Err(L1Error::Product(format!(
                "window {row0}+{rows} × {col0}+{cols} exceeds the {} × {} swath",
                self.lines, band.samples
            )));
        }
        let subdataset = format!(
            "HDF5:\"{}\"://science/LSAR/RSLC/swaths/frequency{frequency}/{polarization}",
            self.source
        );
        read_complex_window(Path::new(&subdataset), row0, col0, rows, cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_is_bilinear_and_clamped() {
        let lut = Lut2d {
            times: vec![0.0, 10.0],
            ranges: vec![100.0, 200.0, 300.0],
            values: Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).unwrap(),
        };
        assert!((lut.eval(5.0, 150.0) - 5.5).abs() < 1e-12);
        assert!((lut.eval(-3.0, 400.0) - 2.0).abs() < 1e-12);
        assert!(lut.contains(10.0, 100.0));
        assert!(!lut.contains(10.1, 100.0));
    }

    /// Reprocessed acquisitions keep the highest product counter.
    #[test]
    fn rslc_identity_reads_start_and_counter() {
        let name = "NISAR_L1_PR_RSLC_031_005_A_020_4005_DHDH_A_20260906T125217_20260906T125252_P05023_N_F_J_002";
        assert_eq!(rslc_identity(name), Some(("20260906T125217", 2)));
    }
}
