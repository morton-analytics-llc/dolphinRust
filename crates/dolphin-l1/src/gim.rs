//! JPL global ionosphere maps (GIM) for the ionospheric range delay.
//!
//! OPERA CSLC-S1 shifts every sample's slant range by the line-of-sight
//! ionospheric delay of the JPL final GIM (COMPASS `ionosphere_delay`:
//! `K · TEC · 1e16 / f² / cos(θ_inc)`, TEC trilinear in time, latitude and
//! longitude at the ground point and the burst's mid time). At C band that
//! is 0.2-0.35 m (≈ 0.1 range pixel) and it changes from date to date, so
//! leaving it out misregisters every date by its own amount. The delay moves
//! the sampling position only; the flattening uses the geometric range, so
//! the ionospheric phase stays in the CSLC for the downstream correction.
//!
//! Files come from NASA CDDIS with the Earthdata bearer token
//! (`EARTHDATA_TOKEN`, or `GP_EARTHDATA_TOKEN`): the long name
//! `JPL0OPSFIN_YYYYDDD0000_01D_02H_GIM.INX.gz` from GPS week 2238
//! (2022-11-27) on, the short name `jplgDDD0.YYi.Z` (Unix `compress`) before.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Datelike, NaiveDate, Utc};
use dolphin_corrections::ionosphere::{read_ionex, IonexMaps};
use reqwest::blocking::Client;
use reqwest::header::AUTHORIZATION;

use crate::error::{product, L1Error, Result};

const CDDIS_URL: &str = "https://cddis.nasa.gov/archive/gnss/products/ionex";
/// First day of the long (IGS 2.0) GIM file names.
const LONG_NAMES_FROM: (i32, u32, u32) = (2022, 11, 27);
/// Ionospheric constant (m³/s²), COMPASS `K`.
const K_IONO: f64 = 40.31;
const ATTEMPTS: u32 = 4;

/// CDDIS GIM downloader.
pub struct GimClient {
    http: Client,
    token: String,
}

impl GimClient {
    /// A client with the Earthdata token from the environment.
    pub fn from_env() -> Result<Self> {
        let token = std::env::var("EARTHDATA_TOKEN")
            .or_else(|_| std::env::var("GP_EARTHDATA_TOKEN"))
            .map_err(|_| L1Error::MissingCredential("EARTHDATA_TOKEN"))?;
        let http = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self { http, token })
    }

    /// The decompressed IONEX file for `day`, cached under `cache_dir`.
    pub fn fetch(&self, day: NaiveDate, cache_dir: &Path) -> Result<PathBuf> {
        let (name, compressed) = file_name(day);
        let dest = cache_dir.join(&name);
        if dest.is_file() {
            return Ok(dest);
        }
        let url = format!(
            "{CDDIS_URL}/{}/{:03}/{compressed}",
            day.year(),
            day.ordinal()
        );
        let bytes = self.download(&url)?;
        let text = match compressed.rsplit('.').next() {
            Some("gz") => {
                let mut out = Vec::new();
                flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut out)?;
                out
            }
            _ => uncompress_z(&bytes)?,
        };
        std::fs::create_dir_all(cache_dir)?;
        let part = dest.with_extension("part");
        std::fs::write(&part, text)?;
        std::fs::rename(&part, &dest)?;
        Ok(dest)
    }

    fn download(&self, url: &str) -> Result<Vec<u8>> {
        let mut attempt = 1;
        loop {
            let outcome = self
                .http
                .get(url)
                .header(AUTHORIZATION, format!("Bearer {}", self.token))
                .send();
            let error = match outcome {
                Ok(response) if response.status().is_success() => {
                    return Ok(response.bytes()?.to_vec());
                }
                Ok(response) => L1Error::Service {
                    service: "CDDIS",
                    status: response.status().as_u16(),
                    url: url.to_string(),
                    body: String::new(),
                },
                Err(error) => L1Error::Http(error),
            };
            let transient = match &error {
                L1Error::Service { status, .. } => *status == 429 || *status >= 500,
                _ => true,
            };
            if !transient || attempt >= ATTEMPTS {
                return Err(error);
            }
            tracing::warn!(%error, attempt, "CDDIS GIM download failed; retrying");
            std::thread::sleep(Duration::from_secs(10) * attempt);
            attempt += 1;
        }
    }
}

/// `(decompressed name, archive name)` of the JPL final GIM for `day`.
pub fn file_name(day: NaiveDate) -> (String, String) {
    let (y, m, d) = LONG_NAMES_FROM;
    let long_from = NaiveDate::from_ymd_opt(y, m, d).unwrap_or(NaiveDate::MIN);
    if day >= long_from {
        let name = format!(
            "JPL0OPSFIN_{}{:03}0000_01D_02H_GIM.INX",
            day.year(),
            day.ordinal()
        );
        (name.clone(), format!("{name}.gz"))
    } else {
        let name = format!("jplg{:03}0.{:02}i", day.ordinal(), day.year() % 100);
        (name.clone(), format!("{name}.Z"))
    }
}

/// Read a decompressed IONEX file.
pub fn read_gim(path: &Path) -> Result<IonexMaps> {
    read_ionex(&std::fs::read_to_string(path)?).map_err(product)
}

/// COMPASS's line-of-sight ionospheric range delay (m) at `(lat, lon)`
/// degrees and `utc`, for incidence `incidence` (rad) and carrier `frequency`
/// (Hz). `None` where the map has no value.
pub fn slant_delay(
    maps: &IonexMaps,
    utc: DateTime<Utc>,
    lat: f64,
    lon: f64,
    incidence: f64,
    frequency: f64,
) -> Option<f64> {
    let tec = maps.value(utc, lat, lon).ok()?;
    Some(K_IONO * tec * 1e16 / (frequency * frequency) / incidence.cos())
}

/// Decode a Unix `compress` (`.Z`, LZW) stream, following ncompress 4.2's
/// decoder: codes are read LSB-first in groups of `n_bits` bytes, and a code
/// width change or a CLEAR code skips to the end of the current group.
pub fn uncompress_z(data: &[u8]) -> Result<Vec<u8>> {
    const CLEAR: u32 = 256;
    let [0x1f, 0x9d, flags, body @ ..] = data else {
        return Err(L1Error::Product("not a .Z (compress) stream".into()));
    };
    let max_bits = u32::from(flags & 0x1f);
    let block_mode = flags & 0x80 != 0;
    if !(9..=16).contains(&max_bits) {
        return Err(L1Error::Product(format!(
            ".Z stream with {max_bits}-bit codes"
        )));
    }
    let max_max_code = 1u32 << max_bits;
    let mut prefix = vec![0u16; max_max_code as usize];
    let mut suffix: Vec<u8> = (0..max_max_code).map(|c| (c & 0xff) as u8).collect();
    let total_bits = body.len() * 8;
    let (mut base, mut pos) = (0usize, 0usize);
    let mut n_bits = 9u32;
    let mut max_code = (1u32 << n_bits) - 1;
    let mut free_ent = if block_mode { 257 } else { 256 };
    let mut old: Option<u32> = None;
    let mut fin_char = 0u8;
    let mut out = Vec::with_capacity(body.len() * 3);
    let mut stack = Vec::new();
    // Round `pos` up to the end of the current n_bits-byte group and rebase.
    let realign = |base: &mut usize, pos: &mut usize, n_bits: u32| {
        let group = (n_bits as usize) << 3;
        if *pos > 0 {
            *pos = (*pos - 1) + (group - (*pos - 1 + group) % group);
        }
        *base += *pos;
        *pos = 0;
    };
    while base + pos + n_bits as usize <= total_bits {
        if free_ent > max_code && n_bits < max_bits {
            realign(&mut base, &mut pos, n_bits);
            n_bits += 1;
            max_code = if n_bits == max_bits {
                max_max_code
            } else {
                (1u32 << n_bits) - 1
            };
            continue;
        }
        let code = read_code(body, base + pos, n_bits);
        pos += n_bits as usize;
        let Some(previous) = old else {
            fin_char = (code & 0xff) as u8;
            out.push(fin_char);
            old = Some(code);
            continue;
        };
        if code == CLEAR && block_mode {
            free_ent = 256;
            realign(&mut base, &mut pos, n_bits);
            n_bits = 9;
            max_code = (1u32 << n_bits) - 1;
            continue;
        }
        let mut current = code;
        if current >= free_ent {
            if current > free_ent {
                return Err(L1Error::Product(".Z stream holds an invalid code".into()));
            }
            stack.push(fin_char);
            current = previous;
        }
        while current >= 256 {
            stack.push(suffix[current as usize]);
            current = u32::from(prefix[current as usize]);
        }
        fin_char = suffix[current as usize];
        stack.push(fin_char);
        out.extend(stack.drain(..).rev());
        if free_ent < max_max_code {
            prefix[free_ent as usize] = previous as u16;
            suffix[free_ent as usize] = fin_char;
            free_ent += 1;
        }
        old = Some(code);
    }
    Ok(out)
}

/// `n_bits` bits at bit offset `bit`, least significant first.
fn read_code(body: &[u8], bit: usize, n_bits: u32) -> u32 {
    let byte = bit >> 3;
    let mut word = 0u32;
    for k in 0..3 {
        word |= u32::from(body.get(byte + k).copied().unwrap_or(0)) << (8 * k);
    }
    (word >> (bit & 7)) & ((1u32 << n_bits) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file made by `compress` (codes widen from 9 to 12 bits) decodes to
    /// the original bytes.
    #[test]
    fn uncompress_z_matches_compress() {
        let compressed = include_bytes!("../tests/fixtures/lzw_sample.txt.Z");
        let original = include_bytes!("../tests/fixtures/lzw_sample.txt");
        assert_eq!(uncompress_z(compressed).unwrap(), original.to_vec());
        assert!(uncompress_z(b"not compressed").is_err());
    }

    #[test]
    fn gim_names_switch_at_gps_week_2238() {
        let old = NaiveDate::from_ymd_opt(2022, 1, 11).unwrap();
        assert_eq!(file_name(old).1, "jplg0110.22i.Z");
        let new = NaiveDate::from_ymd_opt(2025, 11, 16).unwrap();
        assert_eq!(
            file_name(new).1,
            "JPL0OPSFIN_20253200000_01D_02H_GIM.INX.gz"
        );
    }
}
