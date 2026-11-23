//! Sentinel-1 IW SLC burst reader (one-burst SAFE from CDSE).
//!
//! The product annotation XML carries everything the geometry needs: burst
//! timing, range sampling, radar parameters, the Doppler-centroid and
//! azimuth-FM-rate polynomial lists, the valid-sample masks, and the downlink
//! values (PRF, rank, chirp rate) used by the bistatic and Doppler-shift
//! corrections. Field selection follows s1-reader v0.2.4, the reader behind
//! OPERA CSLC-S1, so the two processors start from the same numbers.
//! Measurement samples are CInt16, read through GDAL as complex f32.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use ndarray::Array2;
use num_complex::Complex32;

use crate::error::{product, L1Error, Result};
use crate::orbit::{parse_utc, seconds_since};

/// Speed of light (m/s).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// A Sentinel-1 range polynomial `f(τ) = Σ cᵢ (τ − τ₀)ⁱ` in two-way slant-range
/// time τ, annotated at azimuth time `azimuth_time`.
#[derive(Debug, Clone, PartialEq)]
pub struct RangePolynomial {
    /// Annotation azimuth time of this estimate.
    pub azimuth_time: DateTime<Utc>,
    /// Two-way range time origin τ₀ (s).
    pub t0: f64,
    /// Coefficients, constant term first.
    pub coefficients: Vec<f64>,
}

impl RangePolynomial {
    /// Evaluate at two-way range time `tau` (s).
    pub fn eval_time(&self, tau: f64) -> f64 {
        let x = tau - self.t0;
        self.coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, c| acc * x + c)
    }

    /// Evaluate at slant range `range` (m).
    pub fn eval_range(&self, range: f64) -> f64 {
        self.eval_time(2.0 * range / SPEED_OF_LIGHT)
    }
}

/// One Sentinel-1 IW burst: annotation values and the measurement path.
#[derive(Debug, Clone)]
pub struct S1Burst {
    /// `S1A`, `S1B`, `S1C`.
    pub mission: String,
    /// `IW1`..`IW3`.
    pub swath: String,
    /// `VV`, `VH`, `HH`, `HV`.
    pub polarization: String,
    /// IPF version, e.g. `3.52`.
    pub ipf_version: String,
    /// Absolute orbit number.
    pub absolute_orbit: u32,
    /// `Ascending` / `Descending`.
    pub pass_direction: String,
    /// Zero-Doppler azimuth time of the first burst line.
    pub sensing_start: DateTime<Utc>,
    /// Azimuth line spacing (s).
    pub azimuth_time_interval: f64,
    /// Burst lines.
    pub lines: usize,
    /// Burst samples.
    pub samples: usize,
    /// Two-way slant-range time of the first sample (s).
    pub slant_range_time: f64,
    /// Range sampling rate (Hz).
    pub range_sampling_rate: f64,
    /// Radar centre frequency (Hz).
    pub radar_frequency: f64,
    /// Azimuth antenna steering rate (rad/s).
    pub azimuth_steering_rate: f64,
    /// Raw-data PRF (Hz).
    pub prf_raw: f64,
    /// Rank: pulses in flight between transmit and echo.
    pub rank: u32,
    /// Transmit chirp ramp rate (Hz/s).
    pub range_chirp_rate: f64,
    /// Range processing bandwidth (Hz).
    pub range_bandwidth: f64,
    /// Range window name, e.g. `Hamming`.
    pub range_window_type: String,
    /// Range window coefficient.
    pub range_window_coefficient: f64,
    /// Mid-swath slant range of IW2 (m), the bistatic-delay reference.
    pub iw2_mid_range: f64,
    /// First line with valid samples.
    pub first_valid_line: usize,
    /// Last valid line (inclusive).
    pub last_valid_line: usize,
    /// First valid sample across the valid lines.
    pub first_valid_sample: usize,
    /// Last valid sample (inclusive) across the valid lines.
    pub last_valid_sample: usize,
    /// Doppler-centroid estimates (Hz vs range time), annotation order.
    pub doppler_centroids: Vec<RangePolynomial>,
    /// Azimuth FM rates (Hz/s vs range time), annotation order.
    pub azimuth_fm_rates: Vec<RangePolynomial>,
    /// Measurement GeoTIFF (CInt16).
    pub measurement: PathBuf,
    /// Annotation XML.
    pub annotation: PathBuf,
}

impl S1Burst {
    /// Read the burst in a one-burst `.SAFE` directory for `polarization`.
    ///
    /// `iw2_annotation` is the parent product's IW2 annotation, required for
    /// IW1/IW3 bursts (the bistatic correction uses the IW2 mid-swath range);
    /// IW2 bursts use their own annotation.
    pub fn open(safe: &Path, polarization: &str, iw2_annotation: Option<&Path>) -> Result<Self> {
        let annotation = find_file(&safe.join("annotation"), polarization, "xml")?;
        let measurement = find_file(&safe.join("measurement"), polarization, "tiff")?;
        let text = std::fs::read_to_string(&annotation)?;
        let doc = roxmltree::Document::parse(&text)?;
        let root = doc.root_element();
        let swath = text_at(root, "adsHeader/swath")?.to_string();
        let iw2_mid_range = match (swath.as_str(), iw2_annotation) {
            ("IW2", _) => mid_swath_range(root)?,
            (_, Some(path)) => {
                let iw2_text = std::fs::read_to_string(path)?;
                mid_swath_range(roxmltree::Document::parse(&iw2_text)?.root_element())?
            }
            (_, None) => {
                return Err(L1Error::Product(format!(
                    "{swath} burst needs the parent IW2 annotation for the bistatic reference"
                )))
            }
        };
        let mut burst = Self::from_annotation(root, iw2_mid_range)?;
        burst.measurement = measurement;
        burst.annotation = annotation;
        burst.ipf_version = manifest_ipf_version(safe)?;
        Ok(burst)
    }

    fn from_annotation(root: roxmltree::Node, iw2_mid_range: f64) -> Result<Self> {
        let num = |path: &str| -> Result<f64> {
            text_at(root, path)?.trim().parse::<f64>().map_err(product)
        };
        let downlink = "generalAnnotation/downlinkInformationList/downlinkInformation";
        let range_processing = "imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing";
        let bursts: Vec<_> = path_nodes(root, "swathTiming/burstList")?
            .children()
            .filter(|n| n.has_tag_name("burst"))
            .collect();
        let [burst_node] = bursts.as_slice() else {
            return Err(L1Error::Product(format!(
                "expected a one-burst SAFE, annotation lists {} bursts",
                bursts.len()
            )));
        };
        let valid = ValidWindow::from_burst(*burst_node)?;
        Ok(Self {
            mission: text_at(root, "adsHeader/missionId")?.to_string(),
            swath: text_at(root, "adsHeader/swath")?.to_string(),
            polarization: text_at(root, "adsHeader/polarisation")?.to_string(),
            ipf_version: String::new(),
            absolute_orbit: num("adsHeader/absoluteOrbitNumber")? as u32,
            pass_direction: text_at(root, "generalAnnotation/productInformation/pass")?.to_string(),
            sensing_start: parse_utc(text_at(*burst_node, "azimuthTime")?)?,
            azimuth_time_interval: num("imageAnnotation/imageInformation/azimuthTimeInterval")?,
            lines: num("swathTiming/linesPerBurst")? as usize,
            samples: num("swathTiming/samplesPerBurst")? as usize,
            slant_range_time: num("imageAnnotation/imageInformation/slantRangeTime")?,
            range_sampling_rate: num("generalAnnotation/productInformation/rangeSamplingRate")?,
            radar_frequency: num("generalAnnotation/productInformation/radarFrequency")?,
            azimuth_steering_rate: num("generalAnnotation/productInformation/azimuthSteeringRate")?
                .to_radians(),
            prf_raw: num(&format!("{downlink}/prf"))?,
            rank: num(&format!("{downlink}/downlinkValues/rank"))? as u32,
            range_chirp_rate: num(&format!("{downlink}/downlinkValues/txPulseRampRate"))?,
            range_bandwidth: num(&format!("{range_processing}/processingBandwidth"))?,
            range_window_type: text_at(root, &format!("{range_processing}/windowType"))?
                .to_string(),
            range_window_coefficient: num(&format!("{range_processing}/windowCoefficient"))?,
            iw2_mid_range,
            first_valid_line: valid.first_line,
            last_valid_line: valid.last_line,
            first_valid_sample: valid.first_sample,
            last_valid_sample: valid.last_sample,
            doppler_centroids: polynomial_list(
                root,
                "dopplerCentroid/dcEstimateList",
                "dataDcPolynomial",
            )?,
            azimuth_fm_rates: polynomial_list(
                root,
                "generalAnnotation/azimuthFmRateList",
                "azimuthFmRatePolynomial",
            )?,
            measurement: PathBuf::new(),
            annotation: PathBuf::new(),
        })
    }

    /// Wavelength (m).
    pub fn wavelength(&self) -> f64 {
        SPEED_OF_LIGHT / self.radar_frequency
    }

    /// Slant-range pixel spacing (m).
    pub fn range_pixel_spacing(&self) -> f64 {
        SPEED_OF_LIGHT / (2.0 * self.range_sampling_rate)
    }

    /// Slant range of the first sample (m).
    pub fn starting_range(&self) -> f64 {
        self.slant_range_time * SPEED_OF_LIGHT / 2.0
    }

    /// s1-reader's `sensing_mid`: `sensing_start + lines·ATI/2`, exact for
    /// odd line counts (half a line past the centre line's integer index).
    pub fn sensing_mid(&self) -> DateTime<Utc> {
        self.sensing_start + seconds(0.5 * self.lines as f64 * self.azimuth_time_interval)
    }

    /// s1-reader's `sensing_stop`: the zero-Doppler time of the last burst
    /// line, `sensing_start + (lines − 1)·ATI`.
    pub fn sensing_stop(&self) -> DateTime<Utc> {
        self.sensing_start + seconds((self.lines - 1) as f64 * self.azimuth_time_interval)
    }

    /// The annotation polynomial nearest in time to the burst middle.
    pub fn nearest<'a>(&self, list: &'a [RangePolynomial]) -> Result<&'a RangePolynomial> {
        let mid = self.sensing_start
            + seconds(0.5 * (self.lines - 1) as f64 * self.azimuth_time_interval);
        list.iter()
            .min_by(|a, b| {
                let da = seconds_since(mid, a.azimuth_time).abs();
                let db = seconds_since(mid, b.azimuth_time).abs();
                da.total_cmp(&db)
            })
            .ok_or_else(|| L1Error::Product("empty polynomial list".into()))
    }

    /// Doppler-centroid polynomial for this burst.
    pub fn doppler(&self) -> Result<&RangePolynomial> {
        self.nearest(&self.doppler_centroids)
    }

    /// Azimuth FM-rate polynomial for this burst.
    pub fn fm_rate(&self) -> Result<&RangePolynomial> {
        self.nearest(&self.azimuth_fm_rates)
    }

    /// The polynomials spanning the burst: the last one at or before its start
    /// through the first one at or after its end (at least two).
    pub fn spanning<'a>(&self, list: &'a [RangePolynomial]) -> &'a [RangePolynomial] {
        let start = self.sensing_start;
        let stop = self.sensing_stop();
        let first = list
            .iter()
            .rposition(|p| p.azimuth_time <= start)
            .unwrap_or(0);
        let last = list
            .iter()
            .position(|p| p.azimuth_time >= stop)
            .unwrap_or(list.len().saturating_sub(1))
            .max((first + 1).min(list.len().saturating_sub(1)));
        &list[first..=last]
    }

    /// Read the burst samples `[row0, row0 + rows) × [col0, col0 + cols)` as
    /// complex f32 (lines × samples).
    pub fn read_window(
        &self,
        row0: usize,
        col0: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Array2<Complex32>> {
        read_complex_window(&self.measurement, row0, col0, rows, cols)
    }

    /// IPF version as a number (`003.52` → 3.52).
    pub fn ipf_number(&self) -> Result<f64> {
        self.ipf_version.trim().parse::<f64>().map_err(product)
    }

    /// Read the whole burst.
    pub fn read_all(&self) -> Result<Array2<Complex32>> {
        self.read_window(0, 0, self.lines, self.samples)
    }
}

/// A [`Duration`] from seconds, rounded to the nanosecond.
fn seconds(value: f64) -> Duration {
    Duration::nanoseconds((value * 1e9).round() as i64)
}

/// One annotation geolocation-grid tie point.
#[derive(Debug, Clone, Copy)]
pub struct GeolocationPoint {
    /// Zero-Doppler azimuth time.
    pub azimuth_time: DateTime<Utc>,
    /// Two-way slant-range time (s).
    pub slant_range_time: f64,
    /// Latitude (deg).
    pub latitude: f64,
    /// Longitude (deg).
    pub longitude: f64,
    /// Height above the ellipsoid (m).
    pub height: f64,
}

impl S1Burst {
    /// The annotation geolocation grid (ESA tie points covering the burst).
    pub fn geolocation_grid(&self) -> Result<Vec<GeolocationPoint>> {
        let text = std::fs::read_to_string(&self.annotation)?;
        let doc = roxmltree::Document::parse(&text)?;
        let list = path_nodes(
            doc.root_element(),
            "geolocationGrid/geolocationGridPointList",
        )?;
        list.children()
            .filter(|n| n.has_tag_name("geolocationGridPoint"))
            .map(|point| {
                let num = |tag: &str| -> Result<f64> {
                    text_at(point, tag)?.trim().parse::<f64>().map_err(product)
                };
                Ok(GeolocationPoint {
                    azimuth_time: parse_utc(text_at(point, "azimuthTime")?)?,
                    slant_range_time: num("slantRangeTime")?,
                    latitude: num("latitude")?,
                    longitude: num("longitude")?,
                    height: num("height")?,
                })
            })
            .collect()
    }
}

/// Read a window of band 1 of a complex raster as complex f32.
pub fn read_complex_window(
    path: &Path,
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
) -> Result<Array2<Complex32>> {
    let dataset = gdal::Dataset::open(path)?;
    let band = dataset.rasterband(1)?;
    let mut buffer = vec![Complex32::new(0.0, 0.0); rows * cols];
    let to_int = |v: usize| i32::try_from(v).map_err(product);
    // SAFETY: `buffer` holds rows*cols CFloat32 values (8 bytes each, the
    // layout of Complex32) and GDAL writes exactly that window into it.
    let status = unsafe {
        gdal_sys::GDALRasterIO(
            band.c_rasterband(),
            gdal_sys::GDALRWFlag::GF_Read,
            to_int(col0)?,
            to_int(row0)?,
            to_int(cols)?,
            to_int(rows)?,
            buffer.as_mut_ptr().cast(),
            to_int(cols)?,
            to_int(rows)?,
            gdal_sys::GDALDataType::GDT_CFloat32,
            0,
            0,
        )
    };
    if status != gdal_sys::CPLErr::CE_None {
        return Err(L1Error::Product(format!(
            "GDALRasterIO failed on {}",
            path.display()
        )));
    }
    Array2::from_shape_vec((rows, cols), buffer).map_err(product)
}

struct ValidWindow {
    first_line: usize,
    last_line: usize,
    first_sample: usize,
    last_sample: usize,
}

impl ValidWindow {
    /// s1-reader's rule: the valid lines are those with `firstValidSample ≥ 0`;
    /// the sample window is the tighter of the first and last valid lines.
    fn from_burst(burst: roxmltree::Node) -> Result<Self> {
        let ints = |tag: &str| -> Result<Vec<i64>> {
            text_at(burst, tag)?
                .split_whitespace()
                .map(|v| v.parse::<i64>().map_err(product))
                .collect()
        };
        let first_samples = ints("firstValidSample")?;
        let last_samples = ints("lastValidSample")?;
        let first_line = first_samples
            .iter()
            .position(|&v| v >= 0)
            .ok_or_else(|| L1Error::Product("burst has no valid line".into()))?;
        let valid_lines = first_samples.iter().filter(|&&v| v >= 0).count();
        let last_line = first_line + valid_lines - 1;
        let as_index = |v: i64| usize::try_from(v).map_err(product);
        Ok(Self {
            first_line,
            last_line,
            first_sample: as_index(first_samples[first_line].max(first_samples[last_line]))?,
            last_sample: as_index(last_samples[first_line].min(last_samples[last_line]))?,
        })
    }
}

fn mid_swath_range(root: roxmltree::Node) -> Result<f64> {
    let tau = text_at(root, "imageAnnotation/imageInformation/slantRangeTime")?
        .trim()
        .parse::<f64>()
        .map_err(product)?;
    let samples = text_at(root, "swathTiming/samplesPerBurst")?
        .trim()
        .parse::<f64>()
        .map_err(product)?;
    let rate = text_at(
        root,
        "generalAnnotation/productInformation/rangeSamplingRate",
    )?
    .trim()
    .parse::<f64>()
    .map_err(product)?;
    let spacing = SPEED_OF_LIGHT / (2.0 * rate);
    Ok(tau * SPEED_OF_LIGHT / 2.0 + 0.5 * samples * spacing)
}

fn polynomial_list(
    root: roxmltree::Node,
    list: &str,
    coeff_tag: &str,
) -> Result<Vec<RangePolynomial>> {
    path_nodes(root, list)?
        .children()
        .filter(roxmltree::Node::is_element)
        .map(|entry| {
            let coefficients = match entry.children().find(|c| c.has_tag_name(coeff_tag)) {
                Some(node) => node
                    .text()
                    .unwrap_or_default()
                    .split_whitespace()
                    .map(|v| v.parse::<f64>().map_err(product))
                    .collect::<Result<Vec<_>>>()?,
                // IPF < 2.82 lists the FM-rate terms as separate c0/c1/c2 elements.
                None => ["c0", "c1", "c2"]
                    .iter()
                    .map(|t| text_at(entry, t)?.trim().parse::<f64>().map_err(product))
                    .collect::<Result<Vec<_>>>()?,
            };
            Ok(RangePolynomial {
                azimuth_time: parse_utc(text_at(entry, "azimuthTime")?)?,
                t0: text_at(entry, "t0")?
                    .trim()
                    .parse::<f64>()
                    .map_err(product)?,
                coefficients,
            })
        })
        .collect()
}

/// Read the IPF version from a SAFE manifest.
pub fn manifest_ipf_version(safe: &Path) -> Result<String> {
    let text = std::fs::read_to_string(safe.join("manifest.safe"))?;
    let doc = roxmltree::Document::parse(&text)?;
    doc.descendants()
        .find(|n| n.has_tag_name("software") && n.attribute("name") == Some("Sentinel-1 IPF"))
        .and_then(|n| n.attribute("version"))
        .map(str::to_string)
        .ok_or_else(|| L1Error::Product(format!("{} manifest has no IPF version", safe.display())))
}

fn path_nodes<'a, 'input>(
    root: roxmltree::Node<'a, 'input>,
    path: &str,
) -> Result<roxmltree::Node<'a, 'input>> {
    path.split('/').try_fold(root, |node, tag| {
        node.children()
            .find(|c| c.has_tag_name(tag))
            .ok_or_else(|| L1Error::Product(format!("annotation has no {path}")))
    })
}

fn text_at<'a>(root: roxmltree::Node<'a, '_>, path: &str) -> Result<&'a str> {
    path_nodes(root, path)?
        .text()
        .ok_or_else(|| L1Error::Product(format!("annotation {path} is empty")))
}

fn find_file(dir: &Path, polarization: &str, extension: &str) -> Result<PathBuf> {
    let tag = format!("-{}-", polarization.to_ascii_lowercase());
    std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.extension().is_some_and(|e| e == extension)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains(&tag))
        })
        .ok_or_else(|| {
            L1Error::NotFound(format!("{polarization} *.{extension} in {}", dir.display()))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_polynomial_evaluates_in_range_time() {
        let poly = RangePolynomial {
            azimuth_time: Utc::now(),
            t0: 5.0e-3,
            coefficients: vec![10.0, 2.0e3, -1.0e6],
        };
        let tau = 5.1e-3;
        let x: f64 = 1.0e-4;
        assert!((poly.eval_time(tau) - (10.0 + 2.0e3 * x - 1.0e6 * x * x)).abs() < 1e-12);
        let range = tau * SPEED_OF_LIGHT / 2.0;
        assert!((poly.eval_range(range) - poly.eval_time(tau)).abs() < 1e-9);
    }
}
