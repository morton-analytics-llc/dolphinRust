//! NISAR ionospheric timing corrections from the `NISAR_ANC_TEC` ancillary.
//!
//! JPL's IMAGEN team publishes one TEC file per day (ASF collection
//! `NISAR_TEC`, ~6 MB JSON): every 10 s along the orbit, the total TEC
//! (`totTecNr`/`totTecFr`, TECU) at the ionospheric piercing points of the
//! near and far swath edges and the topside TEC above the satellite
//! (`topTecNr`/`topTecFr`). It is an input to NISAR processing, like the orbit
//! and the DEM, not a derived SAR product. At L band the ionosphere moves the
//! image by a large fraction of a pixel (0.8–1.0 m of slant range on the
//! track-5 frame, a third of a range sample, and up to ±1e-4 s in azimuth,
//! 0.15 line), so it is part of the sampling geometry.
//!
//! The corrections follow ISCE3 0.25 (`isce3.atmosphere.tec_product`), which
//! JPL's GSLC applies:
//! - **slant range**: `ΔR = K · TEC_sub · 1e16 / f² / cos θ`, `TEC_sub =
//!   totTec − topTec` (the TEC below the satellite), θ the incidence angle on
//!   the ellipsoid at the near/far range, `K = 40.31`, `f` the band's processed
//!   centre frequency; tabulated at the TEC times × {near, far} range;
//! - **azimuth time**: a TEC gradient along the track shifts the azimuth
//!   focus, `Δt = −2K / (c · K_a · f) · dTEC_sub/dt · 1e16` with `K_a = −2 v²
//!   f / (c R)`, on the staggered grid between TEC samples.
//!
//! Both are added before sampling (`t + Δt`, `R + ΔR`); the flattening keeps
//! the geometric range, so the ionospheric phase stays in the product for
//! the downstream correction (split-spectrum in `dolphin pairs`).

use std::path::Path;

use ndarray::Array2;
use serde::Deserialize;

use crate::error::{product, L1Error, Result};
use crate::geometry::{dot, enu_basis, norm, rdr2geo, sub};
use crate::orbit::parse_utc;
use crate::rslc::{Lut2d, Rslc, RslcBand};
use crate::safe::SPEED_OF_LIGHT;

/// Ionospheric constant (m³/s²), ISCE3 `K`.
const K_IONO: f64 = 40.31;
const TECU: f64 = 1e16;
/// TEC samples kept either side of the radar grid (s), ISCE3 default.
const MARGIN_S: f64 = 40.0;

#[derive(Deserialize)]
struct TecFile {
    utc: Vec<String>,
    #[serde(rename = "totTecNr")]
    tot_near: Vec<f64>,
    #[serde(rename = "topTecNr")]
    top_near: Vec<f64>,
    #[serde(rename = "totTecFr")]
    tot_far: Vec<f64>,
    #[serde(rename = "topTecFr")]
    top_far: Vec<f64>,
}

/// Ionospheric timing corrections of one band.
#[derive(Debug, Clone)]
pub struct TecCorrections {
    /// Slant-range delay (m) on (azimuth time, range).
    pub range: Lut2d,
    /// Azimuth-time shift (s) on (azimuth time, range).
    pub azimuth: Lut2d,
}

/// The radar-grid span the TEC must cover.
struct Span {
    start: f64,
    stop: f64,
    near: f64,
    far: f64,
}

impl Span {
    fn new(rslc: &Rslc, band: &RslcBand) -> Self {
        Self {
            start: rslc.first_line_time,
            stop: rslc.line_time((rslc.lines - 1) as f64),
            near: band.starting_range,
            far: band.starting_range + (band.samples - 1) as f64 * band.range_spacing,
        }
    }
}

/// Indices of the TEC samples in `[lo, hi]`, checked to bracket the radar
/// grid (`pad` widens the grid for the staggered azimuth gradient).
fn kept(times: &[f64], lo: f64, hi: f64, span: &Span, pad: f64) -> Result<Vec<usize>> {
    let keep: Vec<usize> = (0..times.len())
        .filter(|&k| lo <= times[k] && times[k] <= hi)
        .collect();
    let covers = keep.len() >= 2
        && times[keep[0]] < span.start - pad
        && times[keep[keep.len() - 1]] > span.stop + pad;
    if !covers {
        return Err(L1Error::Product(
            "the TEC file does not cover the radar grid".into(),
        ));
    }
    Ok(keep)
}

fn incidence(rslc: &Rslc, t: f64, range: f64) -> Result<f64> {
    let ground = rdr2geo(t, range, &rslc.orbit, rslc.look_side, |_, _| Ok(0.0))?;
    let look = sub(rslc.orbit.state(t)?.position, ground.ecef);
    let up = enu_basis(ground.lat, ground.lon)[2];
    Ok((dot(look, up) / norm(look)).acos())
}

/// Read `path` and build `band`'s corrections for `rslc`.
///
/// # Errors
/// Unreadable JSON, or a file that does not cover the RSLC.
pub fn tec_corrections(path: &Path, rslc: &Rslc, band: &RslcBand) -> Result<TecCorrections> {
    let file: TecFile = serde_json::from_slice(&std::fs::read(path)?)?;
    let times = file
        .utc
        .iter()
        .map(|u| Ok(rslc.orbit.seconds(parse_utc(u)?)))
        .collect::<Result<Vec<f64>>>()?;
    let n = times.len();
    if n < 2
        || [&file.tot_near, &file.top_near, &file.tot_far, &file.top_far]
            .iter()
            .any(|v| v.len() != n)
    {
        return Err(L1Error::Product(format!(
            "{}: ragged TEC arrays",
            path.display()
        )));
    }
    let sub_tec = |k: usize| {
        [
            file.tot_near[k] - file.top_near[k],
            file.tot_far[k] - file.top_far[k],
        ]
    };
    let span = Span::new(rslc, band);
    let orbit = &rslc.orbit;
    let (orbit0, orbit1) = (orbit.times[0], orbit.times[orbit.times.len() - 1]);
    let f2 = band.center_frequency * band.center_frequency;
    let ranges = vec![span.near, span.far];

    let keep = kept(
        &times,
        orbit0.max(span.start - MARGIN_S),
        orbit1.min(span.stop + MARGIN_S),
        &span,
        0.0,
    )?;
    let mut delay = Vec::with_capacity(2 * keep.len());
    for &k in &keep {
        let tec = sub_tec(k);
        for (side, &range) in ranges.iter().enumerate() {
            let cos = incidence(rslc, times[k], range)?.cos();
            delay.push(K_IONO * tec[side] * TECU / f2 / cos);
        }
    }
    let range_lut = Lut2d {
        times: keep.iter().map(|&k| times[k]).collect(),
        ranges: ranges.clone(),
        values: Array2::from_shape_vec((keep.len(), 2), delay).map_err(product)?,
    };

    let spacing = (times[n - 1] - times[0]) / (n - 1) as f64;
    let half = spacing / 2.0;
    let margin = MARGIN_S.max(half);
    let keep = kept(
        &times,
        (orbit0 - half).max(span.start - margin),
        (orbit1 + half).min(span.stop + margin),
        &span,
        half,
    )?;
    let mut mids = Vec::with_capacity(keep.len() - 1);
    let mut shift = Vec::with_capacity(2 * (keep.len() - 1));
    for pair in keep.windows(2) {
        let mid = times[pair[1]] - half;
        let speed = norm(orbit.state(mid)?.velocity);
        let (a, b) = (sub_tec(pair[0]), sub_tec(pair[1]));
        for (side, &range) in ranges.iter().enumerate() {
            let gradient = (b[side] - a[side]) / spacing;
            let fm_rate = -2.0 * speed * speed * (band.center_frequency / SPEED_OF_LIGHT) / range;
            shift.push(
                -2.0 * K_IONO / (SPEED_OF_LIGHT * fm_rate * band.center_frequency)
                    * gradient
                    * TECU,
            );
        }
        mids.push(mid);
    }
    let azimuth_lut = Lut2d {
        values: Array2::from_shape_vec((mids.len(), 2), shift).map_err(product)?,
        times: mids,
        ranges,
    };
    Ok(TecCorrections {
        range: range_lut,
        azimuth: azimuth_lut,
    })
}
