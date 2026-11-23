//! Copernicus Data Space Ecosystem (CDSE): Sentinel-1 bursts and orbit files.
//!
//! - **Auth:** OIDC password grant against the CDSE realm with the public
//!   `cdse-public` client, minted once per run and refreshed before expiry
//!   (measured 2026-11-21: access token 1800 s, refresh token 3600 s).
//! - **Bursts:** the OData `Bursts` catalogue (no auth) returns one record per
//!   IW burst; `POST Bursts({Id})/$value` with the Bearer token answers 301 to
//!   the burst service, which must be POSTed again with the token re-attached.
//!   The body is a zip holding a one-burst SAFE (manifest, annotation,
//!   calibration, noise, measurement TIFF).
//! - **Orbits:** `AUX_POEORB` (precise) from the same catalogue, falling back to
//!   `AUX_RESORB` (restituted) when no precise file covers the window yet.
//!
//! Credentials come from `CDSE_USERNAME` / `CDSE_PASSWORD`; they are sent only
//! to the CDSE identity endpoint and never logged.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use reqwest::blocking::{Client, RequestBuilder, Response};
use reqwest::header::{AUTHORIZATION, LOCATION};
use reqwest::redirect::Policy;
use serde::Deserialize;

use crate::error::{L1Error, Result};

const TOKEN_URL: &str =
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token";
const CATALOGUE_URL: &str = "https://catalogue.dataspace.copernicus.eu/odata/v1";
const ZIPPER_URL: &str = "https://zipper.dataspace.copernicus.eu/odata/v1";
/// Refresh the access token this long before it expires.
const TOKEN_MARGIN: Duration = Duration::from_secs(120);
/// Attempts per download (transient 429/5xx answers are retried).
const DOWNLOAD_ATTEMPTS: u32 = 4;
/// Redirect hops followed by hand (so the Bearer token survives each hop).
const MAX_REDIRECTS: usize = 5;

/// Sentinel-1 IW sub-swath.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Swath {
    /// IW1 (near range).
    Iw1,
    /// IW2.
    Iw2,
    /// IW3 (far range).
    Iw3,
}

impl Swath {
    /// `"IW1"`, `"IW2"` or `"IW3"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Swath::Iw1 => "IW1",
            Swath::Iw2 => "IW2",
            Swath::Iw3 => "IW3",
        }
    }

    /// Parse `IW1`..`IW3` (case-insensitive).
    pub fn parse(text: &str) -> Result<Self> {
        match text.to_ascii_uppercase().as_str() {
            "IW1" => Ok(Swath::Iw1),
            "IW2" => Ok(Swath::Iw2),
            "IW3" => Ok(Swath::Iw3),
            other => Err(L1Error::Product(format!("unknown IW swath {other:?}"))),
        }
    }
}

/// An OPERA/ESA burst identifier: relative orbit, burst number, sub-swath.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BurstId {
    /// Relative orbit (track) number, 1..=175.
    pub relative_orbit: u32,
    /// ESA burst number along the track.
    pub burst: u32,
    /// IW sub-swath.
    pub swath: Swath,
}

impl BurstId {
    /// Parse `T100-213486-IW2`, `t100_213486_iw2` or `100_213486_IW2`.
    pub fn parse(text: &str) -> Result<Self> {
        let cleaned = text.trim().trim_start_matches(['T', 't']).replace('-', "_");
        let parts: Vec<&str> = cleaned.split('_').collect();
        let bad = || L1Error::Product(format!("burst id {text:?} is not T<track>-<burst>-IW<n>"));
        let [track, burst, swath] = parts.as_slice() else {
            return Err(bad());
        };
        Ok(Self {
            relative_orbit: track.parse().map_err(|_| bad())?,
            burst: burst.parse().map_err(|_| bad())?,
            swath: Swath::parse(swath)?,
        })
    }

    /// OPERA form, `t100_213486_iw2`.
    pub fn opera(&self) -> String {
        format!(
            "t{:03}_{:06}_{}",
            self.relative_orbit,
            self.burst,
            self.swath.as_str().to_ascii_lowercase()
        )
    }

    /// The next burst along track in the same sub-swath (burst numbers grow
    /// with azimuth time).
    pub fn next(&self) -> Self {
        Self {
            burst: self.burst + 1,
            ..*self
        }
    }
}

/// One burst record from the CDSE `Bursts` catalogue.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct BurstRecord {
    /// CDSE burst UUID (the download key).
    pub id: String,
    /// Burst name, e.g. `S1A-SLC-20230110T133459-213486-IW2-VV-367066`.
    pub name: String,
    /// Parent SAFE name.
    pub parent_product_name: String,
    /// Parent SAFE product UUID.
    pub parent_product_id: String,
    /// Burst number along the track.
    pub burst_id: u32,
    /// Relative orbit.
    pub relative_orbit_number: u32,
    /// `IW1`..`IW3`.
    pub swath_identifier: String,
    /// Polarisation, e.g. `VV`.
    pub polarisation_channels: String,
    /// Azimuth time of the first line of the burst.
    pub azimuth_time: DateTime<Utc>,
    /// Byte offset of the burst inside the parent measurement TIFF.
    pub byte_offset: u64,
    /// `ASCENDING` / `DESCENDING`.
    pub orbit_direction: String,
}

impl BurstRecord {
    /// Mission, e.g. `S1A`, from the burst name.
    pub fn mission(&self) -> &str {
        self.name.split('-').next().unwrap_or_default()
    }
}

/// An orbit file record from the CDSE catalogue.
#[derive(Debug, Clone)]
pub struct OrbitRecord {
    /// CDSE product UUID.
    pub id: String,
    /// File name, e.g. `S1A_OPER_AUX_POEORB_OPOD_..._V..._....EOF`.
    pub name: String,
    /// Validity start.
    pub valid_start: DateTime<Utc>,
    /// Validity stop.
    pub valid_stop: DateTime<Utc>,
}

impl OrbitRecord {
    /// `true` for a precise (POEORB) orbit, `false` for restituted (RESORB).
    pub fn is_precise(&self) -> bool {
        self.name.contains("POEORB")
    }
}

/// Burst search: one burst id and polarisation over a date window.
#[derive(Debug, Clone)]
pub struct BurstQuery {
    /// Burst to find.
    pub burst_id: BurstId,
    /// Polarisation, e.g. `VV`.
    pub polarization: String,
    /// Window start (inclusive).
    pub start: DateTime<Utc>,
    /// Window end (exclusive).
    pub end: DateTime<Utc>,
}

#[derive(Deserialize)]
struct ODataPage<T> {
    value: Vec<T>,
    #[serde(rename = "@odata.nextLink")]
    next_link: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ProductRow {
    id: String,
    name: String,
    content_date: ContentDate,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ContentDate {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

#[derive(Deserialize)]
struct NodeListing {
    result: Vec<NodeRow>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct NodeRow {
    name: String,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
    refresh_token: Option<String>,
    refresh_expires_in: Option<u64>,
}

struct Token {
    access: String,
    expires_at: Instant,
    refresh: Option<(String, Instant)>,
}

/// CDSE client: catalogue search plus authenticated burst and orbit download.
pub struct CdseClient {
    http: Client,
    username: String,
    password: String,
    token: Option<Token>,
}

impl CdseClient {
    /// Build a client from `CDSE_USERNAME` / `CDSE_PASSWORD`.
    pub fn from_env() -> Result<Self> {
        let username = std::env::var("CDSE_USERNAME")
            .map_err(|_| L1Error::MissingCredential("CDSE_USERNAME"))?;
        let password = std::env::var("CDSE_PASSWORD")
            .map_err(|_| L1Error::MissingCredential("CDSE_PASSWORD"))?;
        let http = Client::builder()
            .redirect(Policy::none())
            .timeout(Duration::from_secs(900))
            .connect_timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            http,
            username,
            password,
            token: None,
        })
    }

    /// Every burst matching `query`, oldest first.
    pub fn search_bursts(&self, query: &BurstQuery) -> Result<Vec<BurstRecord>> {
        let filter = format!(
            "BurstId eq {} and SwathIdentifier eq '{}' and RelativeOrbitNumber eq {} \
             and PolarisationChannels eq '{}' and ContentDate/Start ge {} and ContentDate/Start lt {}",
            query.burst_id.burst,
            query.burst_id.swath.as_str(),
            query.burst_id.relative_orbit,
            query.polarization,
            odata_time(query.start),
            odata_time(query.end),
        );
        let first = self.http.get(format!("{CATALOGUE_URL}/Bursts")).query(&[
            ("$filter", filter.as_str()),
            ("$orderby", "ContentDate/Start asc"),
            ("$top", "1000"),
        ]);
        let mut records: Vec<BurstRecord> = self.paged(first)?;
        // Re-processed parents can list the same burst twice; keep the first.
        records.dedup_by(|b, a| b.azimuth_time == a.azimuth_time);
        Ok(records)
    }

    /// The best orbit file covering `[start, stop]` for `mission` (`S1A`, `S1B`,
    /// `S1C`): precise when available, otherwise restituted.
    pub fn search_orbit(
        &self,
        mission: &str,
        start: DateTime<Utc>,
        stop: DateTime<Utc>,
    ) -> Result<OrbitRecord> {
        for kind in ["AUX_POEORB", "AUX_RESORB"] {
            let filter = format!(
                "Collection/Name eq 'SENTINEL-1' and startswith(Name,'{mission}_OPER_{kind}') \
                 and ContentDate/Start lt {} and ContentDate/End gt {}",
                odata_time(start),
                odata_time(stop),
            );
            let request = self.http.get(format!("{CATALOGUE_URL}/Products")).query(&[
                ("$filter", filter.as_str()),
                ("$orderby", "ContentDate/Start desc"),
                ("$top", "20"),
            ]);
            let rows: Vec<ProductRow> = self.paged(request)?;
            // The newest production of a validity window wins (names sort by
            // production time after the mission/type prefix).
            let best = rows.into_iter().max_by(|a, b| a.name.cmp(&b.name));
            if let Some(row) = best {
                return Ok(OrbitRecord {
                    id: row.id,
                    name: row.name,
                    valid_start: row.content_date.start,
                    valid_stop: row.content_date.end,
                });
            }
        }
        Err(L1Error::NotFound(format!(
            "no {mission} POEORB or RESORB covering {start} .. {stop}"
        )))
    }

    /// Download one burst and unpack its one-burst SAFE under `dest_dir`.
    /// Returns the `.SAFE` directory. An existing unpacked burst is reused.
    pub fn download_burst(&mut self, record: &BurstRecord, dest_dir: &Path) -> Result<PathBuf> {
        let unpack_dir = dest_dir.join(&record.name);
        if let Some(safe) = find_safe_dir(&unpack_dir)? {
            return Ok(safe);
        }
        std::fs::create_dir_all(dest_dir)?;
        let zip_path = dest_dir.join(format!("{}.zip", record.name));
        let url = format!("{CATALOGUE_URL}/Bursts({})/$value", record.id);
        self.download_authorized(&url, Method::Post, &zip_path)?;
        unzip(&zip_path, &unpack_dir)?;
        std::fs::remove_file(&zip_path)?;
        find_safe_dir(&unpack_dir)?
            .ok_or_else(|| L1Error::Product(format!("{} unpacked without a .SAFE", record.name)))
    }

    /// Download the parent product's IW2 annotation for `record`'s
    /// polarisation into `dest_dir` and return its path.
    ///
    /// The bistatic-delay correction is referenced to the IW2 mid-swath range,
    /// which a one-burst IW1 or IW3 SAFE does not carry. The single annotation
    /// file comes from the parent product's node tree, not the whole product.
    pub fn download_iw2_annotation(
        &mut self,
        record: &BurstRecord,
        dest_dir: &Path,
    ) -> Result<PathBuf> {
        let dest = dest_dir.join(format!(
            "{}-iw2-{}.xml",
            record.parent_product_name.trim_end_matches(".SAFE"),
            record.polarisation_channels.to_ascii_lowercase()
        ));
        if dest.is_file() {
            return Ok(dest);
        }
        std::fs::create_dir_all(dest_dir)?;
        let folder = format!(
            "{ZIPPER_URL}/Products({})/Nodes({})/Nodes(annotation)/Nodes",
            record.parent_product_id, record.parent_product_name
        );
        let listing: NodeListing = checked(self.http.get(&folder).send()?, "CDSE nodes")?.json()?;
        let wanted = format!(
            "-iw2-slc-{}-",
            record.polarisation_channels.to_ascii_lowercase()
        );
        let name = listing
            .result
            .into_iter()
            .map(|n| n.name)
            .find(|n| n.contains(&wanted) && n.ends_with(".xml"))
            .ok_or_else(|| {
                L1Error::NotFound(format!(
                    "{wanted} annotation in {}",
                    record.parent_product_name
                ))
            })?;
        let url = format!("{}({name})/$value", folder);
        self.download_authorized(&url, Method::Get, &dest)?;
        Ok(dest)
    }

    /// Download an orbit file into `dest_dir`; returns the `.EOF` path.
    pub fn download_orbit(&mut self, record: &OrbitRecord, dest_dir: &Path) -> Result<PathBuf> {
        let eof_name = record.name.trim_end_matches(".zip");
        let eof_path = dest_dir.join(eof_name);
        if eof_path.is_file() {
            return Ok(eof_path);
        }
        std::fs::create_dir_all(dest_dir)?;
        let part = dest_dir.join(format!("{eof_name}.part"));
        let url = format!("{ZIPPER_URL}/Products({})/$value", record.id);
        self.download_authorized(&url, Method::Get, &part)?;
        if is_zip(&part)? {
            extract_member_with_suffix(&part, ".EOF", &eof_path)?;
            std::fs::remove_file(&part)?;
        } else {
            std::fs::rename(&part, &eof_path)?;
        }
        Ok(eof_path)
    }

    fn paged<T: for<'de> Deserialize<'de>>(&self, first: RequestBuilder) -> Result<Vec<T>> {
        let mut out = Vec::new();
        let mut page: ODataPage<T> = checked(first.send()?, "CDSE catalogue")?.json()?;
        loop {
            out.append(&mut page.value);
            let Some(next) = page.next_link.take() else {
                return Ok(out);
            };
            page = checked(self.http.get(next).send()?, "CDSE catalogue")?.json()?;
        }
    }

    fn access_token(&mut self) -> Result<String> {
        let now = Instant::now();
        if let Some(token) = self
            .token
            .as_ref()
            .filter(|t| t.expires_at > now + TOKEN_MARGIN)
        {
            return Ok(token.access.clone());
        }
        let refresh = self
            .token
            .as_ref()
            .and_then(|t| t.refresh.clone())
            .filter(|(_, until)| *until > now + TOKEN_MARGIN);
        let response = match refresh {
            Some((refresh_token, _)) => self.http.post(TOKEN_URL).form(&[
                ("client_id", "cdse-public"),
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.as_str()),
            ]),
            None => self.http.post(TOKEN_URL).form(&[
                ("client_id", "cdse-public"),
                ("grant_type", "password"),
                ("username", self.username.as_str()),
                ("password", self.password.as_str()),
            ]),
        }
        .send()?;
        let body: TokenResponse = checked(response, "CDSE identity")?.json()?;
        let issued = Instant::now();
        let token = Token {
            access: body.access_token,
            expires_at: issued + Duration::from_secs(body.expires_in),
            refresh: body
                .refresh_token
                .zip(body.refresh_expires_in)
                .map(|(t, s)| (t, issued + Duration::from_secs(s))),
        };
        let access = token.access.clone();
        self.token = Some(token);
        Ok(access)
    }

    fn download_authorized(&mut self, url: &str, method: Method, dest: &Path) -> Result<()> {
        let mut last_error = None;
        for attempt in 0..DOWNLOAD_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(10 * u64::from(attempt)));
            }
            match self.try_download(url, method, dest) {
                Ok(()) => return Ok(()),
                Err(error) if is_transient(&error) => last_error = Some(error),
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| L1Error::NotFound(url.to_string())))
    }

    fn try_download(&mut self, url: &str, method: Method, dest: &Path) -> Result<()> {
        let mut current = url.to_string();
        for _ in 0..=MAX_REDIRECTS {
            let token = self.access_token()?;
            let request = match method {
                Method::Get => self.http.get(&current),
                Method::Post => self.http.post(&current),
            };
            let response = request
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .send()?;
            if response.status().is_redirection() {
                current = redirect_target(&response)?;
                continue;
            }
            let mut response = checked(response, "CDSE download")?;
            let part = dest.with_extension("download");
            let mut file = File::create(&part)?;
            response.copy_to(&mut file)?;
            file.flush()?;
            std::fs::rename(&part, dest)?;
            return Ok(());
        }
        Err(L1Error::Product(format!("too many redirects from {url}")))
    }
}

#[derive(Clone, Copy)]
enum Method {
    Get,
    Post,
}

/// RFC 3339 UTC timestamp in the form the CDSE OData filter accepts.
fn odata_time(time: DateTime<Utc>) -> String {
    time.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

pub(crate) fn redirect_target(response: &Response) -> Result<String> {
    let location = response
        .headers()
        .get(LOCATION)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| L1Error::Product("redirect without a Location header".into()))?;
    let base = response.url();
    Ok(base
        .join(location)
        .map_err(crate::error::product)?
        .to_string())
}

/// Pass a 2xx response through; turn anything else into [`L1Error::Service`].
pub(crate) fn checked(response: Response, service: &'static str) -> Result<Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status().as_u16();
    let url = response.url().to_string();
    let body: String = response
        .text()
        .unwrap_or_default()
        .chars()
        .take(300)
        .collect();
    Err(L1Error::Service {
        service,
        status,
        url,
        body,
    })
}

pub(crate) fn is_transient(error: &L1Error) -> bool {
    match error {
        L1Error::Service { status, .. } => *status == 429 || *status >= 500,
        L1Error::Http(e) => e.is_timeout() || e.is_connect() || e.is_body(),
        _ => false,
    }
}

fn is_zip(path: &Path) -> Result<bool> {
    let mut magic = [0u8; 2];
    let read = File::open(path)?.read(&mut magic)?;
    Ok(read == 2 && &magic == b"PK")
}

pub(crate) fn unzip(zip_path: &Path, dest: &Path) -> Result<()> {
    let mut archive = zip::ZipArchive::new(File::open(zip_path)?)?;
    std::fs::create_dir_all(dest)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let Some(relative) = entry.enclosed_name().map(Path::to_path_buf) else {
            continue;
        };
        let out = dest.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::io::copy(&mut entry, &mut File::create(&out)?)?;
    }
    Ok(())
}

fn extract_member_with_suffix(zip_path: &Path, suffix: &str, dest: &Path) -> Result<()> {
    let mut archive = zip::ZipArchive::new(File::open(zip_path)?)?;
    let name = archive
        .file_names()
        .find(|n| n.ends_with(suffix))
        .map(str::to_string)
        .ok_or_else(|| L1Error::Product(format!("{} holds no *{suffix}", zip_path.display())))?;
    let mut entry = archive.by_name(&name)?;
    std::io::copy(&mut entry, &mut File::create(dest)?)?;
    Ok(())
}

/// The first `*.SAFE` directory under `root` (depth ≤ 2), if `root` exists.
pub(crate) fn find_safe_dir(root: &Path) -> Result<Option<PathBuf>> {
    if !root.is_dir() {
        return Ok(None);
    }
    if root.extension().is_some_and(|e| e == "SAFE") && root.join("manifest.safe").is_file() {
        return Ok(Some(root.to_path_buf()));
    }
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "SAFE") && path.join("manifest.safe").is_file() {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_id_parses_opera_and_dashed_forms() {
        let dashed = BurstId::parse("T100-213486-IW2").unwrap();
        let opera = BurstId::parse("t100_213486_iw2").unwrap();
        assert_eq!(dashed, opera);
        assert_eq!(dashed.opera(), "t100_213486_iw2");
        assert_eq!(
            BurstId::parse("034_071054_IW3").unwrap().opera(),
            "t034_071054_iw3"
        );
        assert!(BurstId::parse("T100-IW2").is_err());
    }

    #[test]
    fn odata_time_is_millisecond_utc() {
        let t: DateTime<Utc> = "2023-01-10T13:34:58.375289Z".parse().unwrap();
        assert_eq!(odata_time(t), "2023-01-10T13:34:58.375Z");
    }
}
