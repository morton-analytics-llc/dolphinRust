//! ASF burst extractor: the fallback Sentinel-1 burst source.
//!
//! CMR's `SENTINEL-1_BURSTS` collection lists every IW burst ASF can cut, with
//! direct links to `…/{SAFE}/{IW}/{POL}/{index}.zip` on
//! `sentinel1-burst.asf.alaska.edu`. The zip is a one-burst SAFE in the same
//! layout CDSE serves, so both sources feed the same reader. The extractor
//! sits behind Earthdata Login: the first request redirects to EDL, which takes
//! HTTP Basic credentials (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`) and
//! returns with a session cookie. A burst that is not cut yet answers
//! `202 Accepted`; the request is repeated until it answers 200.
//! Measured 2026-11-22: CDSE and ASF bursts decode to identical samples.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use serde::Deserialize;

use crate::cdse::{checked, find_safe_dir, redirect_target, unzip, BurstQuery};
use crate::error::{L1Error, Result};

const CMR_GRANULES: &str = "https://cmr.earthdata.nasa.gov/search/granules.umm_json";
const EDL_HOST: &str = "urs.earthdata.nasa.gov";
/// Polls of a `202 Accepted` burst before giving up.
const ACCEPTED_POLLS: u32 = 30;
const MAX_REDIRECTS: usize = 8;

/// One ASF burst granule from CMR.
#[derive(Debug, Clone)]
pub struct AsfBurst {
    /// CMR granule UR, e.g. `S1_213486_IW2_20230110T133458_VV_95B4-BURST`.
    pub granule: String,
    /// Azimuth time of the first burst line.
    pub azimuth_time: DateTime<Utc>,
    /// Link to the one-burst SAFE zip.
    pub zip_url: String,
}

#[derive(Deserialize)]
struct CmrPage {
    items: Vec<CmrItem>,
}

#[derive(Deserialize)]
struct CmrItem {
    umm: CmrUmm,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct CmrUmm {
    #[serde(rename = "GranuleUR")]
    granule_ur: String,
    related_urls: Vec<CmrUrl>,
    additional_attributes: Vec<CmrAttribute>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct CmrUrl {
    #[serde(rename = "URL")]
    url: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct CmrAttribute {
    name: String,
    values: Vec<String>,
}

/// ASF burst client (CMR search plus EDL-authenticated download).
pub struct AsfBurstClient {
    http: Client,
    username: String,
    password: String,
}

impl AsfBurstClient {
    /// Build a client from `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.
    pub fn from_env() -> Result<Self> {
        let username = std::env::var("EARTHDATA_USERNAME")
            .map_err(|_| L1Error::MissingCredential("EARTHDATA_USERNAME"))?;
        let password = std::env::var("EARTHDATA_PASSWORD")
            .map_err(|_| L1Error::MissingCredential("EARTHDATA_PASSWORD"))?;
        let http = Client::builder()
            .redirect(Policy::none())
            .cookie_store(true)
            .timeout(Duration::from_secs(900))
            .connect_timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            http,
            username,
            password,
        })
    }

    /// Every ASF burst matching `query`, oldest first.
    pub fn search(&self, query: &BurstQuery) -> Result<Vec<AsfBurst>> {
        let id = query.burst_id;
        let burst_full = format!(
            "{:03}_{:06}_{}",
            id.relative_orbit,
            id.burst,
            id.swath.as_str()
        );
        let temporal = format!(
            "{},{}",
            query.start.format("%Y-%m-%dT%H:%M:%SZ"),
            query.end.format("%Y-%m-%dT%H:%M:%SZ")
        );
        let page: CmrPage = checked(
            self.http
                .get(CMR_GRANULES)
                .query(&[
                    ("short_name", "SENTINEL-1_BURSTS"),
                    ("attribute[]", &format!("string,BURST_ID_FULL,{burst_full}")),
                    (
                        "attribute[]",
                        &format!("string,POLARIZATION,{}", query.polarization),
                    ),
                    ("temporal[]", &temporal),
                    ("page_size", "2000"),
                    ("sort_key", "start_date"),
                ])
                .send()?,
            "CMR",
        )?
        .json()?;
        page.items
            .into_iter()
            .map(|item| parse_granule(item.umm))
            .collect()
    }

    /// Download one burst and unpack its SAFE under `dest_dir`.
    pub fn download(&self, burst: &AsfBurst, dest_dir: &Path) -> Result<PathBuf> {
        let unpack_dir = dest_dir.join(&burst.granule);
        if let Some(safe) = find_safe_dir(&unpack_dir)? {
            return Ok(safe);
        }
        std::fs::create_dir_all(dest_dir)?;
        let zip_path = dest_dir.join(format!("{}.zip", burst.granule));
        self.fetch(&burst.zip_url, &zip_path)?;
        unzip(&zip_path, &unpack_dir)?;
        std::fs::remove_file(&zip_path)?;
        find_safe_dir(&unpack_dir)?
            .ok_or_else(|| L1Error::Product(format!("{} unpacked without a .SAFE", burst.granule)))
    }

    fn fetch(&self, url: &str, dest: &Path) -> Result<()> {
        for _ in 0..ACCEPTED_POLLS {
            let mut response = self.follow(url)?;
            if response.status().as_u16() == 202 {
                std::thread::sleep(Duration::from_secs(5));
                continue;
            }
            let part = dest.with_extension("download");
            let mut file = File::create(&part)?;
            response.copy_to(&mut file)?;
            file.flush()?;
            std::fs::rename(&part, dest)?;
            return Ok(());
        }
        Err(L1Error::Product(format!(
            "{url} still not cut after {ACCEPTED_POLLS} polls"
        )))
    }

    /// GET `url`, following redirects by hand and answering the EDL login
    /// with Basic credentials (sent to the EDL host only).
    fn follow(&self, url: &str) -> Result<reqwest::blocking::Response> {
        let mut current = url.to_string();
        for _ in 0..MAX_REDIRECTS {
            let mut request = self.http.get(&current);
            if reqwest::Url::parse(&current).is_ok_and(|u| u.host_str() == Some(EDL_HOST)) {
                request = request.basic_auth(&self.username, Some(&self.password));
            }
            let response = request.send()?;
            if !response.status().is_redirection() {
                return checked(response, "ASF burst extractor");
            }
            current = redirect_target(&response)?;
        }
        Err(L1Error::Product(format!("too many redirects from {url}")))
    }
}

fn parse_granule(umm: CmrUmm) -> Result<AsfBurst> {
    let attribute = |name: &str| {
        umm.additional_attributes
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| a.values.first().cloned())
    };
    let azimuth_time = attribute("AZIMUTH_TIME")
        .ok_or_else(|| L1Error::Product(format!("{} has no AZIMUTH_TIME", umm.granule_ur)))?;
    let azimuth_time = format!("{}Z", azimuth_time.trim_end_matches('Z'))
        .parse()
        .map_err(crate::error::product)?;
    let zip_url = umm
        .related_urls
        .iter()
        .map(|u| u.url.clone())
        .find(|u| u.ends_with(".zip"))
        .ok_or_else(|| L1Error::Product(format!("{} has no burst zip link", umm.granule_ur)))?;
    Ok(AsfBurst {
        granule: umm.granule_ur,
        azimuth_time,
        zip_url,
    })
}
