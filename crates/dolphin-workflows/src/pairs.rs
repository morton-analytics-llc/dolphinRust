//! Dated pairwise products over a planned set of acquisition pairs (`dolphin pairs`).
//!
//! `mode: coherence` writes `|Γ_ij|` for each planned pair `(i, j)`: the magnitude of the
//! off-diagonal of the sample coherence matrix from
//! [`dolphin_phaselink::covariance::estimate_stack_covariance`], with the same
//! `half_window` / `strides` semantics as phase linking. Each pair is estimated on its own
//! two-acquisition stack; the entry is element-wise identical to the `(i, j)` off-diagonal
//! of the full-stack estimate because the normalization only involves `i` and `j`.
//!
//! `mode: interferogram` (NISAR GSLC only) forms the `frequencyA` (main band) and
//! `frequencyB` (side band) interferograms `ref · conj(sec)`, multilooks A onto the B grid,
//! unwraps both with SNAPHU, aligns B's integer cycles to A within each A connected
//! component ([`align_side_band_cycles`]), estimates the ionosphere in the A
//! interferogram by split-spectrum ([`dolphin_corrections::split_spectrum`]), low-pass
//! filters it and
//! subtracts it from A. Per pair it writes `unwrapped` (A), `ionosphere`, `corrected`
//! (`unwrapped − ionosphere`), `coherence` (A, `half_window` with strides equal to the
//! A→B looks) and `connected_components` (A); once per run it writes `los_enu.tif`. A
//! product without `frequencyB` is refused with [`PairsError::MissingFrequencyB`].
//!
//! Nodata: an output pixel is NaN when any input sample it is formed from (in either
//! acquisition, and in either band) is NaN, and every COG carries GDAL nodata NaN.
//!
//! Outputs, under `work_directory`:
//! - `<layer>_<refYYYYMMDD>_<secYYYYMMDD>.tif` per pair: a COG carrying the
//!   `REFERENCE_ACQUISITION_ID`, `SECONDARY_ACQUISITION_ID`, `REFERENCE_DATE`,
//!   `SECONDARY_DATE`, `TEMPORAL_BASELINE_DAYS` and `LAYER` tags. Interferogram mode adds
//!   `REFERENCE_CRID`, `SECONDARY_CRID`, `CENTER_FREQUENCY_A_HZ`, `CENTER_FREQUENCY_B_HZ`,
//!   `WAVELENGTH_M`, `IONOSPHERE_METHOD` and `PHASE_SIGN`.
//! - `los_enu.tif` (interferogram mode): ground→sensor LOS unit vector, bands east,
//!   north, up, on the output grid.
//! - `pairs.json`: the run index listing every pair's IDs, dates, files and sha256;
//!   interferogram mode adds each pair's summary, including the per-component side-band
//!   cycle shift `k` and its pixel count.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use dolphin_core::config::{DisplacementWorkflow, InputType};
use dolphin_core::{BlockIndices, Cf32, Cf64, HalfWindow, Strides};
use dolphin_corrections::split_spectrum::{
    align_side_band_cycles, estimate_ionosphere, lowpass_ionosphere, BandPhase, ComponentCycles,
};
use dolphin_io::geometry::read_nisar_los_layers_for_grid;
use dolphin_io::nisar::{read_nisar_band_metadata, NisarBandMetadata};
use dolphin_io::{
    read_cslc_shape, read_cslc_window, read_geotransform, read_nisar_geotransform,
    read_nisar_window, transform_bounds, GeoInfo,
};
use dolphin_phaselink::covariance::estimate_stack_covariance;
use dolphin_unwrap::snaphu::{unwrap, UnwrapConfig};
use gdal::raster::{Buffer, RasterCreationOptions};
use gdal::spatial_ref::SpatialRef;
use gdal::{DriverManager, Metadata};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, Axis, Zip};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::burst::BurstGeo;
use crate::crop::plan_bounds;

/// File name of the per-run pair index written under `work_directory`.
pub const PAIRS_INDEX_FILENAME: &str = "pairs.json";

/// EPSG code of the `bounds` envelope.
const BOUNDS_EPSG: u32 = 4326;

/// Date format used in GDAL tags and `pairs.json`.
const TAG_DATE_FORMAT: &str = "%Y-%m-%d";

/// File name of the per-run LOS geometry written in interferogram mode.
pub const LOS_ENU_FILENAME: &str = "los_enu.tif";

/// `IONOSPHERE_METHOD` tag value.
const IONOSPHERE_METHOD: &str = "split_spectrum_main_side_band";

/// `PHASE_SIGN` tag value: the `tests/sign_convention.rs` interferogram order.
const PHASE_SIGN: &str = "reference_times_conj_secondary";

/// Main-band group name in a NISAR GSLC subdataset path.
const MAIN_BAND_GROUP: &str = "/frequencyA/";

/// Side-band group name in a NISAR GSLC subdataset path.
const SIDE_BAND_GROUP: &str = "/frequencyB/";

/// Coherence threshold reported in the `pairs.json` interferogram summary.
const SUMMARY_COHERENCE_THRESHOLD: f32 = 0.2;

/// Speed of light in vacuum (m/s), for `WAVELENGTH_M`.
const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;

/// Typed `dolphin pairs` refusals.
#[derive(Debug, thiserror::Error)]
pub enum PairsError {
    /// Interferogram mode needs the NISAR side band; the product has none.
    #[error(
        "acquisition {acquisition_id} has no frequencyB side band at {subdataset}; \
         split-spectrum ionosphere needs both range bands, so interferogram mode refuses it"
    )]
    MissingFrequencyB {
        /// Granule ID of the acquisition without a side band.
        acquisition_id: String,
        /// Side-band dataset path that was looked for.
        subdataset: String,
    },
}

/// `dolphin pairs` YAML config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairsConfig {
    /// Which pairwise product to compute.
    pub mode: PairsMode,
    /// Input-product reader selection.
    pub input_type: InputType,
    /// Complex grid dataset inside each input file (e.g. `/data/VV`).
    pub subdataset: String,
    /// Directory the pair COGs and `pairs.json` are written to.
    pub work_directory: PathBuf,
    /// The acquisitions available to the pair plan.
    pub acquisitions: Vec<PairAcquisition>,
    /// Which acquisition pairs to form.
    pub pair_plan: PairPlan,
    /// Covariance half-window, as in phase linking.
    pub half_window: HalfWindow,
    /// Output decimation, as in phase linking.
    pub strides: Strides,
    /// Optional EPSG:4326 `[W, S, E, N]` output envelope.
    #[serde(default)]
    pub bounds: Option<[f64; 4]>,
    /// Interferogram mode: NISAR radarGrid LOS cube group (e.g.
    /// `/science/LSAR/GSLC/metadata/radarGrid`) for `los_enu.tif`.
    #[serde(default)]
    pub nisar_geometry_group: Option<String>,
    /// Interferogram mode: ellipsoidal-height DEM (m) on the output CRS, used to
    /// interpolate the LOS cube.
    #[serde(default)]
    pub nisar_ellipsoidal_dem_file: Option<PathBuf>,
    /// Interferogram mode: low-pass cutoff wavelength (m) applied to the
    /// split-spectrum ionosphere estimate.
    #[serde(default)]
    pub ionosphere_lowpass_wavelength_m: Option<f64>,
}

impl PairsConfig {
    /// Parse a pairs config from a YAML string.
    ///
    /// # Errors
    /// Returns `Err` on malformed YAML, unknown fields or missing required fields.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).context("parsing pairs config")
    }
}

/// Pairwise product selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairsMode {
    /// Pair coherence magnitude `|Γ_ij|`.
    Coherence,
    /// NISAR split-spectrum ionosphere-corrected unwrapped interferogram.
    Interferogram,
}

/// One input acquisition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairAcquisition {
    /// Local path of the input file.
    pub path: PathBuf,
    /// Granule ID of the acquisition.
    pub acquisition_id: String,
    /// Acquisition time (UTC).
    pub acquisition_utc: DateTime<Utc>,
}

/// Pair-formation rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairPlan {
    /// Allowed temporal baselines in days (e.g. `[12]`).
    pub pair_class_days: Vec<f64>,
    /// Allowed deviation from a pair class, in days.
    pub tolerance_days: f64,
    /// Explicit `[reference_id, secondary_id]` pairs. When absent, every date-ordered
    /// pair whose baseline matches a class is formed. When present, each listed pair
    /// must match a class.
    #[serde(default)]
    pub pairs: Option<Vec<[String; 2]>>,
}

/// A planned pair, indexing into [`PairsConfig::acquisitions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedPair {
    /// Index of the earlier (reference) acquisition.
    pub reference: usize,
    /// Index of the later (secondary) acquisition.
    pub secondary: usize,
    /// Calendar-day difference between the two acquisition dates.
    pub temporal_baseline_days: i64,
}

/// The `pairs.json` run index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairsIndex {
    /// Mode the run was configured with.
    pub mode: PairsMode,
    /// One record per written pair, in plan order.
    pub pairs: Vec<PairRecord>,
    /// The per-run `los_enu.tif` (interferogram mode only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub los_enu: Option<PairFile>,
}

/// One pair in the run index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairRecord {
    /// Reference granule ID.
    pub reference_acquisition_id: String,
    /// Secondary granule ID.
    pub secondary_acquisition_id: String,
    /// Reference acquisition date (`YYYY-MM-DD`, UTC).
    pub reference_date: String,
    /// Secondary acquisition date (`YYYY-MM-DD`, UTC).
    pub secondary_date: String,
    /// Calendar-day difference between the two dates.
    pub temporal_baseline_days: i64,
    /// Files written for this pair.
    pub files: Vec<PairFile>,
    /// Interferogram-mode summary statistics (absent in coherence mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interferogram: Option<InterferogramSummary>,
}

/// Per-pair interferogram summary in `pairs.json`, over finite pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterferogramSummary {
    /// Fraction of finite coherence pixels with coherence ≥ 0.2; `None` if none are finite.
    pub coherence_fraction_at_least_0_2: Option<f64>,
    /// Median of the finite coherence pixels; `None` if none are finite.
    pub median_coherence: Option<f64>,
    /// Population standard deviation of the finite low-passed ionosphere pixels
    /// (radians); `None` if none are finite.
    pub ionosphere_std_rad: Option<f64>,
    /// Integer cycles removed from the frequencyB unwrapped phase per frequencyA
    /// connected component before the ionosphere estimate, ascending by component.
    pub side_band_cycle_alignment: Vec<SideBandCycleAlignment>,
}

/// One connected component's frequencyB→frequencyA cycle alignment in `pairs.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SideBandCycleAlignment {
    /// frequencyA connected-component label (as in the `connected_components` layer).
    pub component: u32,
    /// Cycles removed from frequencyB: `φ_B ← φ_B − 2π·k`.
    pub k: i64,
    /// Pixels with both bands valid, over which `k` was estimated.
    pub pixel_count: usize,
}

impl From<ComponentCycles> for SideBandCycleAlignment {
    fn from(entry: ComponentCycles) -> Self {
        Self {
            component: entry.component,
            k: entry.k,
            pixel_count: entry.pixel_count,
        }
    }
}

/// One written pair layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairFile {
    /// Layer name (also the `LAYER` tag and the file-name prefix).
    pub layer: String,
    /// File name, relative to `work_directory`.
    pub file: String,
    /// Hex sha256 of the file.
    pub sha256: String,
}

/// Where to read and how to place the output grid.
struct PairGrid {
    /// Native-resolution read window.
    source: BlockIndices,
    /// Output-pixel window kept from the strided estimate of `source`.
    trim: BlockIndices,
    /// Georeference of the trimmed output.
    geo: GeoInfo,
}

/// Run `dolphin pairs`: plan the pairs, write one product per pair and `pairs.json`.
///
/// # Errors
/// Returns `Err` on invalid config, an unsupported mode/input-type combination, I/O
/// failure, inconsistent acquisition grids, or bounds that do not fit the grid.
/// Interferogram mode returns [`PairsError::MissingFrequencyB`] for a product without
/// a side band.
pub fn run_pairs(cfg: &PairsConfig) -> Result<PairsIndex> {
    validate_config(cfg)?;
    let planned = plan_pairs(&cfg.acquisitions, &cfg.pair_plan)?;
    let index = match (cfg.mode, cfg.input_type) {
        (PairsMode::Coherence, InputType::OperaCslc) => run_coherence(cfg, &planned)?,
        (PairsMode::Coherence, InputType::NisarGslc) => bail!(
            "dolphin pairs coherence mode does not support input_type nisar_gslc; \
             only opera_cslc is implemented"
        ),
        (PairsMode::Interferogram, InputType::NisarGslc) => run_interferogram(cfg, &planned)?,
        (PairsMode::Interferogram, InputType::OperaCslc) => bail!(
            "dolphin pairs interferogram mode needs two range bands for split-spectrum \
             ionosphere; it supports input_type nisar_gslc only"
        ),
    };
    let index_path = cfg.work_directory.join(PAIRS_INDEX_FILENAME);
    std::fs::write(&index_path, serde_json::to_string_pretty(&index)?)
        .with_context(|| format!("writing {}", index_path.display()))?;
    Ok(index)
}

fn run_coherence(cfg: &PairsConfig, planned: &[PlannedPair]) -> Result<PairsIndex> {
    let grid = pair_grid(cfg)?;
    create_work_directory(cfg)?;
    let pairs = planned
        .iter()
        .map(|pair| write_coherence_pair(cfg, &grid, pair))
        .collect::<Result<Vec<_>>>()?;
    Ok(PairsIndex {
        mode: cfg.mode,
        pairs,
        los_enu: None,
    })
}

fn create_work_directory(cfg: &PairsConfig) -> Result<()> {
    std::fs::create_dir_all(&cfg.work_directory)
        .with_context(|| format!("creating {}", cfg.work_directory.display()))
}

/// Form the planned pairs from the acquisitions and the pair plan.
///
/// # Errors
/// Returns `Err` on invalid classes or tolerance, duplicate IDs or dates, an unknown or
/// reversed explicit pair, or an explicit pair outside every class.
pub fn plan_pairs(acquisitions: &[PairAcquisition], plan: &PairPlan) -> Result<Vec<PlannedPair>> {
    validate_plan(plan)?;
    validate_unique(acquisitions)?;
    let Some(explicit) = &plan.pairs else {
        return Ok(class_pairs(acquisitions, plan));
    };
    explicit
        .iter()
        .map(|[reference, secondary]| explicit_pair(acquisitions, plan, reference, secondary))
        .collect()
}

fn class_pairs(acquisitions: &[PairAcquisition], plan: &PairPlan) -> Vec<PlannedPair> {
    let mut order: Vec<usize> = (0..acquisitions.len()).collect();
    order.sort_by_key(|&index| acquisitions[index].acquisition_utc);
    order
        .iter()
        .enumerate()
        .flat_map(|(position, &reference)| {
            order[position + 1..]
                .iter()
                .map(move |&secondary| planned(acquisitions, reference, secondary))
        })
        .filter(|pair| matches_class(plan, pair.temporal_baseline_days))
        .collect()
}

fn explicit_pair(
    acquisitions: &[PairAcquisition],
    plan: &PairPlan,
    reference_id: &str,
    secondary_id: &str,
) -> Result<PlannedPair> {
    let find = |id: &str| {
        acquisitions
            .iter()
            .position(|acquisition| acquisition.acquisition_id == id)
            .with_context(|| format!("pair_plan.pairs names unknown acquisition_id {id}"))
    };
    let pair = planned(acquisitions, find(reference_id)?, find(secondary_id)?);
    ensure!(
        pair.temporal_baseline_days > 0,
        "pair [{reference_id}, {secondary_id}]: reference must be dated before secondary"
    );
    ensure!(
        matches_class(plan, pair.temporal_baseline_days),
        "pair [{reference_id}, {secondary_id}]: {}-day baseline matches no pair_class_days within tolerance",
        pair.temporal_baseline_days
    );
    Ok(pair)
}

fn planned(acquisitions: &[PairAcquisition], reference: usize, secondary: usize) -> PlannedPair {
    let days = acquisition_date(&acquisitions[secondary])
        .signed_duration_since(acquisition_date(&acquisitions[reference]))
        .num_days();
    PlannedPair {
        reference,
        secondary,
        temporal_baseline_days: days,
    }
}

fn matches_class(plan: &PairPlan, days: i64) -> bool {
    plan.pair_class_days
        .iter()
        .any(|&class| (days as f64 - class).abs() <= plan.tolerance_days)
}

fn acquisition_date(acquisition: &PairAcquisition) -> NaiveDate {
    acquisition.acquisition_utc.date_naive()
}

fn validate_plan(plan: &PairPlan) -> Result<()> {
    ensure!(
        !plan.pair_class_days.is_empty(),
        "pair_plan.pair_class_days must not be empty"
    );
    ensure!(
        plan.pair_class_days
            .iter()
            .all(|days| days.is_finite() && *days > 0.0),
        "pair_plan.pair_class_days must be finite and positive"
    );
    ensure!(
        plan.tolerance_days.is_finite() && plan.tolerance_days >= 0.0,
        "pair_plan.tolerance_days must be finite and non-negative"
    );
    Ok(())
}

/// IDs and dates must be unique: dates key the output file names.
fn validate_unique(acquisitions: &[PairAcquisition]) -> Result<()> {
    let mut ids = HashSet::new();
    let mut dates = HashSet::new();
    for acquisition in acquisitions {
        ensure!(
            !acquisition.acquisition_id.is_empty(),
            "acquisition_id must not be empty"
        );
        ensure!(
            ids.insert(acquisition.acquisition_id.as_str()),
            "duplicate acquisition_id {}",
            acquisition.acquisition_id
        );
        ensure!(
            dates.insert(acquisition_date(acquisition)),
            "acquisition {} shares its date with another acquisition",
            acquisition.acquisition_id
        );
    }
    Ok(())
}

fn validate_config(cfg: &PairsConfig) -> Result<()> {
    ensure!(!cfg.subdataset.is_empty(), "subdataset must not be empty");
    ensure!(
        cfg.acquisitions.len() >= 2,
        "at least two acquisitions are required"
    );
    ensure!(
        cfg.strides.y > 0 && cfg.strides.x > 0,
        "strides.y and strides.x must both be positive"
    );
    let Some([west, south, east, north]) = cfg.bounds else {
        return Ok(());
    };
    ensure!(
        [west, south, east, north].iter().all(|v| v.is_finite()) && west < east && south < north,
        "bounds must be finite EPSG:4326 [W, S, E, N] with W < E and S < N"
    );
    Ok(())
}

/// Shared grid of every acquisition, and the bounded read/trim plan if `bounds` is set.
fn pair_grid(cfg: &PairsConfig) -> Result<PairGrid> {
    let burst = common_burst_geo(cfg)?;
    let native = read_cslc_shape(&cfg.acquisitions[0].path, &cfg.subdataset)?;
    let Some(bounds) = cfg.bounds else {
        return Ok(PairGrid {
            source: block(0, native.0, 0, native.1),
            trim: block(0, burst.rows, 0, burst.cols),
            geo: burst.geo,
        });
    };
    let mut bounds_cfg = DisplacementWorkflow::default();
    bounds_cfg.output_options.bounds = Some((bounds[0], bounds[1], bounds[2], bounds[3]));
    bounds_cfg.output_options.bounds_epsg = Some(BOUNDS_EPSG);
    bounds_cfg.output_options.strides = cfg.strides;
    bounds_cfg.phase_linking.half_window = cfg.half_window;
    let plan = plan_bounds(&bounds_cfg, &[burst], 2)?.context("bounds produced no plan")?;
    let window = plan.windows[0].context("bounds do not intersect the acquisition grid")?;
    let [row, col] = plan.provenance.target_pixel_offset;
    let gt = burst.geo.geotransform;
    Ok(PairGrid {
        source: window.source,
        trim: plan.target_in_analysis,
        geo: GeoInfo {
            epsg: burst.geo.epsg,
            geotransform: [
                gt[0] + col as f64 * gt[1],
                gt[1],
                0.0,
                gt[3] + row as f64 * gt[5],
                0.0,
                gt[5],
            ],
        },
    })
}

/// Output-grid footprint shared by every acquisition; differing grids are rejected.
fn common_burst_geo(cfg: &PairsConfig) -> Result<BurstGeo> {
    let read = |acquisition: &PairAcquisition| -> Result<((usize, usize), GeoInfo)> {
        let shape = read_cslc_shape(&acquisition.path, &cfg.subdataset)
            .with_context(|| format!("reading grid of {}", acquisition.acquisition_id))?;
        let geo = read_geotransform(&acquisition.path, &cfg.subdataset)
            .with_context(|| format!("reading georeference of {}", acquisition.acquisition_id))?;
        Ok((shape, geo))
    };
    let (shape, geo) = read(&cfg.acquisitions[0])?;
    ensure!(geo.epsg != 0, "source georeference has no valid EPSG");
    for acquisition in &cfg.acquisitions[1..] {
        ensure!(
            read(acquisition)? == (shape, geo),
            "acquisition {} is not on the grid of {}",
            acquisition.acquisition_id,
            cfg.acquisitions[0].acquisition_id
        );
    }
    let (rows, cols) = cfg.strides.out_shape(shape);
    let gt = geo.geotransform;
    let (sx, sy) = (cfg.strides.x as f64, cfg.strides.y as f64);
    Ok(BurstGeo {
        geo: GeoInfo {
            epsg: geo.epsg,
            geotransform: [gt[0], gt[1] * sx, 0.0, gt[3], 0.0, gt[5] * sy],
        },
        rows,
        cols,
    })
}

fn write_coherence_pair(
    cfg: &PairsConfig,
    grid: &PairGrid,
    pair: &PlannedPair,
) -> Result<PairRecord> {
    let stack = read_pair_stack(cfg, &cfg.subdataset, read_cslc_window, grid.source, pair)?;
    let covariance = estimate_stack_covariance(stack.view(), cfg.half_window, cfg.strides, None)
        .map_err(anyhow::Error::msg)
        .context("estimating pair coherence")?;
    let nodata = nodata_mask(stack.view(), cfg.strides);
    let trim = grid.trim;
    let coherence = Array2::from_shape_fn((trim.height(), trim.width()), |(row, col)| {
        let (row, col) = (row + trim.row_start, col + trim.col_start);
        match nodata[(row, col)] {
            true => f32::NAN,
            false => covariance[(row, col, 0, 1)].norm() as f32,
        }
    });
    let mut record = pair_record(cfg, pair);
    let file = write_pair_layer(cfg, grid.geo, &record, "coherence", &coherence, &[])?;
    record.files.push(file);
    Ok(record)
}

/// Window reader shared by the OPERA and NISAR complex grids.
type WindowReader = fn(&Path, &str, BlockIndices) -> dolphin_io::Result<Array2<Cf32>>;

/// `(2, rows, cols)` stack of the reference and secondary over `source`.
fn read_pair_stack(
    cfg: &PairsConfig,
    subdataset: &str,
    reader: WindowReader,
    source: BlockIndices,
    pair: &PlannedPair,
) -> Result<Array3<Cf64>> {
    let mut stack = Array3::<Cf64>::zeros((2, source.height(), source.width()));
    for (layer, index) in [pair.reference, pair.secondary].into_iter().enumerate() {
        let acquisition = &cfg.acquisitions[index];
        let window = reader(&acquisition.path, subdataset, source)
            .with_context(|| format!("reading {} {subdataset}", acquisition.acquisition_id))?;
        Zip::from(stack.index_axis_mut(Axis(0), layer))
            .and(&window)
            .for_each(|dst, z| *dst = Cf64::new(f64::from(z.re), f64::from(z.im)));
    }
    Ok(stack)
}

/// Output-grid nodata: `true` where any sample of either acquisition inside the
/// output pixel's `looks` block is non-finite.
fn nodata_mask(stack: ArrayView3<Cf64>, looks: Strides) -> Array2<bool> {
    let (_, rows, cols) = stack.dim();
    Array2::from_shape_fn(looks.out_shape((rows, cols)), |(row, col)| {
        let (r0, c0) = (row * looks.y, col * looks.x);
        stack
            .slice(s![.., r0..r0 + looks.y, c0..c0 + looks.x])
            .iter()
            .any(|z| !z.is_finite())
    })
}

/// The pair's IDs and dates, with no files yet.
fn pair_record(cfg: &PairsConfig, pair: &PlannedPair) -> PairRecord {
    let reference = &cfg.acquisitions[pair.reference];
    let secondary = &cfg.acquisitions[pair.secondary];
    PairRecord {
        reference_acquisition_id: reference.acquisition_id.clone(),
        secondary_acquisition_id: secondary.acquisition_id.clone(),
        reference_date: acquisition_date(reference)
            .format(TAG_DATE_FORMAT)
            .to_string(),
        secondary_date: acquisition_date(secondary)
            .format(TAG_DATE_FORMAT)
            .to_string(),
        temporal_baseline_days: pair.temporal_baseline_days,
        files: Vec::new(),
        interferogram: None,
    }
}

/// Write one pair layer as a NaN-nodata COG with the six pair tags plus `extra_tags`.
fn write_pair_layer(
    cfg: &PairsConfig,
    geo: GeoInfo,
    record: &PairRecord,
    layer: &str,
    data: &Array2<f32>,
    extra_tags: &[(&str, &str)],
) -> Result<PairFile> {
    let file = format!(
        "{layer}_{}_{}.tif",
        record.reference_date.replace('-', ""),
        record.secondary_date.replace('-', "")
    );
    let baseline = record.temporal_baseline_days.to_string();
    let mut tags = vec![
        (
            "REFERENCE_ACQUISITION_ID",
            record.reference_acquisition_id.as_str(),
        ),
        (
            "SECONDARY_ACQUISITION_ID",
            record.secondary_acquisition_id.as_str(),
        ),
        ("REFERENCE_DATE", record.reference_date.as_str()),
        ("SECONDARY_DATE", record.secondary_date.as_str()),
        ("TEMPORAL_BASELINE_DAYS", baseline.as_str()),
        ("LAYER", layer),
    ];
    tags.extend_from_slice(extra_tags);
    let path = cfg.work_directory.join(&file);
    dolphin_io::geotiff::write_raster_with_metadata(
        &path,
        data.view(),
        geo.geotransform,
        Some(geo.epsg),
        Some(f64::NAN),
        &tags,
    )
    .with_context(|| format!("writing {}", path.display()))?;
    let sha256 = sha256_file(&path)?;
    Ok(PairFile {
        layer: layer.to_owned(),
        file,
        sha256,
    })
}

/// Validated interferogram-mode settings.
struct InterferogramSettings<'a> {
    geometry_group: &'a str,
    dem: &'a Path,
    lowpass_wavelength_m: f64,
}

/// The NISAR main/side-band read windows, the A→B looks and the output (B) grid.
struct DualBandGrid {
    side_subdataset: String,
    main_source: BlockIndices,
    side_source: BlockIndices,
    looks: Strides,
    geo: GeoInfo,
    main_band: NisarBandMetadata,
    side_band: NisarBandMetadata,
}

/// The four per-pair rasters besides `corrected`, on the output grid, NaN at nodata.
struct PairLayers {
    unwrapped: Array2<f32>,
    ionosphere: Array2<f32>,
    coherence: Array2<f32>,
    connected_components: Array2<f32>,
}

const SINGLE_LOOK: Strides = Strides { y: 1, x: 1 };

fn run_interferogram(cfg: &PairsConfig, planned: &[PlannedPair]) -> Result<PairsIndex> {
    let settings = interferogram_settings(cfg)?;
    let side_subdataset = side_band_subdataset(&cfg.subdataset)?;
    require_side_band(cfg, &side_subdataset)?;
    let grid = dual_band_grid(cfg, side_subdataset)?;
    create_work_directory(cfg)?;
    let pairs = planned
        .iter()
        .map(|pair| write_interferogram_pair(cfg, &grid, &settings, pair))
        .collect::<Result<Vec<_>>>()?;
    let los_enu = write_los_enu(cfg, &grid, &settings)?;
    Ok(PairsIndex {
        mode: cfg.mode,
        pairs,
        los_enu: Some(los_enu),
    })
}

fn interferogram_settings(cfg: &PairsConfig) -> Result<InterferogramSettings<'_>> {
    let geometry_group = cfg
        .nisar_geometry_group
        .as_deref()
        .context("interferogram mode requires nisar_geometry_group for los_enu.tif")?;
    let dem = cfg
        .nisar_ellipsoidal_dem_file
        .as_deref()
        .context("interferogram mode requires nisar_ellipsoidal_dem_file for los_enu.tif")?;
    let lowpass_wavelength_m = cfg
        .ionosphere_lowpass_wavelength_m
        .context("interferogram mode requires ionosphere_lowpass_wavelength_m")?;
    ensure!(
        lowpass_wavelength_m.is_finite() && lowpass_wavelength_m > 0.0,
        "ionosphere_lowpass_wavelength_m must be finite and positive"
    );
    Ok(InterferogramSettings {
        geometry_group,
        dem,
        lowpass_wavelength_m,
    })
}

/// `…/frequencyA/HH` → `…/frequencyB/HH`.
fn side_band_subdataset(subdataset: &str) -> Result<String> {
    ensure!(
        subdataset.matches(MAIN_BAND_GROUP).count() == 1,
        "interferogram mode: subdataset {subdataset} must name one frequencyA grid \
         (e.g. /science/LSAR/GSLC/grids/frequencyA/HH)"
    );
    Ok(subdataset.replacen(MAIN_BAND_GROUP, SIDE_BAND_GROUP, 1))
}

/// Refuse any acquisition whose product has no side-band grid.
fn require_side_band(cfg: &PairsConfig, side_subdataset: &str) -> Result<()> {
    for acquisition in &cfg.acquisitions {
        let present = dataset_exists(&acquisition.path, side_subdataset)
            .with_context(|| format!("opening {}", acquisition.path.display()))?;
        if !present {
            return Err(PairsError::MissingFrequencyB {
                acquisition_id: acquisition.acquisition_id.clone(),
                subdataset: side_subdataset.to_owned(),
            }
            .into());
        }
    }
    Ok(())
}

/// Whether every link along `dataset` exists, walking group by group so a missing
/// intermediate group is `false` rather than an HDF5 error.
fn dataset_exists(path: &Path, dataset: &str) -> Result<bool> {
    let file = hdf5::File::open(path)?;
    let mut group = file.group("/")?;
    let mut parts = dataset
        .split('/')
        .filter(|part| !part.is_empty())
        .peekable();
    while let Some(part) = parts.next() {
        if !group.link_exists(part) {
            return Ok(false);
        }
        if parts.peek().is_some() {
            group = group.group(part)?;
        }
    }
    Ok(true)
}

fn dual_band_grid(cfg: &PairsConfig, side_subdataset: String) -> Result<DualBandGrid> {
    let (main_shape, main_geo) = common_nisar_grid(cfg, &cfg.subdataset)?;
    let (side_shape, side_geo) = common_nisar_grid(cfg, &side_subdataset)?;
    let (looks, [row_offset, col_offset]) = band_alignment(main_geo, side_geo)?;
    ensure!(
        cfg.strides == looks,
        "interferogram mode: strides must equal the frequencyA→frequencyB looks \
         (y: {}, x: {}) so coherence lands on the frequencyB grid",
        looks.y,
        looks.x
    );
    let mut rows = covered(row_offset, looks.y, main_shape.0, side_shape.0);
    let mut cols = covered(col_offset, looks.x, main_shape.1, side_shape.1);
    if let Some(bounds) = cfg.bounds {
        let (bound_rows, bound_cols) = bounds_window(bounds, side_geo, side_shape)?;
        rows = rows.start.max(bound_rows.start)..rows.end.min(bound_rows.end);
        cols = cols.start.max(bound_cols.start)..cols.end.min(bound_cols.end);
    }
    ensure!(
        !rows.is_empty() && !cols.is_empty(),
        "the frequencyB grid has no pixels covered by frequencyA within bounds"
    );
    let to_main =
        |offset: i64, look: usize, index: usize| (offset + (index * look) as i64) as usize;
    let gt = side_geo.geotransform;
    Ok(DualBandGrid {
        main_source: block(
            to_main(row_offset, looks.y, rows.start),
            to_main(row_offset, looks.y, rows.end),
            to_main(col_offset, looks.x, cols.start),
            to_main(col_offset, looks.x, cols.end),
        ),
        side_source: block(rows.start, rows.end, cols.start, cols.end),
        looks,
        geo: GeoInfo {
            epsg: side_geo.epsg,
            geotransform: [
                gt[0] + cols.start as f64 * gt[1],
                gt[1],
                0.0,
                gt[3] + rows.start as f64 * gt[5],
                0.0,
                gt[5],
            ],
        },
        main_band: band_metadata(cfg, &cfg.subdataset)?,
        side_band: band_metadata(cfg, &side_subdataset)?,
        side_subdataset,
    })
}

/// Shape and georeference of `subdataset`, identical across every acquisition.
fn common_nisar_grid(cfg: &PairsConfig, subdataset: &str) -> Result<((usize, usize), GeoInfo)> {
    let read = |acquisition: &PairAcquisition| -> Result<((usize, usize), GeoInfo)> {
        let shape = read_cslc_shape(&acquisition.path, subdataset)
            .with_context(|| format!("reading {subdataset} of {}", acquisition.acquisition_id))?;
        let geo = read_nisar_geotransform(&acquisition.path, subdataset).with_context(|| {
            format!(
                "reading {subdataset} georeference of {}",
                acquisition.acquisition_id
            )
        })?;
        Ok((shape, geo))
    };
    let first = read(&cfg.acquisitions[0])?;
    ensure!(
        first.1.epsg != 0,
        "{subdataset} georeference has no valid EPSG"
    );
    for acquisition in &cfg.acquisitions[1..] {
        ensure!(
            read(acquisition)? == first,
            "acquisition {} {subdataset} is not on the grid of {}",
            acquisition.acquisition_id,
            cfg.acquisitions[0].acquisition_id
        );
    }
    Ok(first)
}

/// `centerFrequency` / `rangeBandwidth` of `subdataset`, identical across acquisitions.
fn band_metadata(cfg: &PairsConfig, subdataset: &str) -> Result<NisarBandMetadata> {
    let read = |acquisition: &PairAcquisition| {
        read_nisar_band_metadata(&acquisition.path, subdataset).with_context(|| {
            format!(
                "reading {subdataset} band metadata of {}",
                acquisition.acquisition_id
            )
        })
    };
    let first = read(&cfg.acquisitions[0])?;
    for acquisition in &cfg.acquisitions[1..] {
        ensure!(
            read(acquisition)? == first,
            "acquisition {} {subdataset} band frequencies differ from {}",
            acquisition.acquisition_id,
            cfg.acquisitions[0].acquisition_id
        );
    }
    Ok(first)
}

/// A→B looks and B's pixel-edge offset in A pixels `[row, col]`. B must be an integer
/// multilook of A on the same CRS.
fn band_alignment(main: GeoInfo, side: GeoInfo) -> Result<(Strides, [i64; 2])> {
    ensure!(
        main.epsg == side.epsg,
        "frequencyA EPSG {} differs from frequencyB EPSG {}",
        main.epsg,
        side.epsg
    );
    let (gm, gs) = (main.geotransform, side.geotransform);
    let whole = |value: f64| {
        let rounded = value.round();
        ((value - rounded).abs() < 1e-6).then_some(rounded as i64)
    };
    let alignment = (
        whole(gs[5] / gm[5]),
        whole(gs[1] / gm[1]),
        whole((gs[3] - gm[3]) / gm[5]),
        whole((gs[0] - gm[0]) / gm[1]),
    );
    match alignment {
        (Some(look_y), Some(look_x), Some(row), Some(col)) if look_y >= 1 && look_x >= 1 => Ok((
            Strides {
                y: look_y as usize,
                x: look_x as usize,
            },
            [row, col],
        )),
        _ => bail!(
            "frequencyB grid {gs:?} is not an integer multilook of the frequencyA grid {gm:?}"
        ),
    }
}

/// Side-band indices whose `look`-pixel main-band block lies inside the main grid.
fn covered(offset: i64, look: usize, main_len: usize, side_len: usize) -> std::ops::Range<usize> {
    let look = look as i64;
    let first = match offset >= 0 {
        true => 0,
        false => (-offset + look - 1) / look,
    };
    let last = ((main_len as i64 - offset).max(0) / look).min(side_len as i64);
    first as usize..(last as usize).max(first as usize)
}

/// Row and column ranges of `shape` on `geo` intersecting EPSG:4326 `bounds`.
fn bounds_window(
    bounds: [f64; 4],
    geo: GeoInfo,
    shape: (usize, usize),
) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let [west, south, east, north] = bounds;
    let (left, bottom, right, top) =
        transform_bounds((west, south, east, north), BOUNDS_EPSG, geo.epsg)
            .context("projecting bounds onto the frequencyB grid")?;
    let gt = geo.geotransform;
    let index = |value: f64, origin: f64, step: f64, len: usize| {
        ((value - origin) / step).clamp(0.0, len as f64)
    };
    let rows = index(top, gt[3], gt[5], shape.0).floor() as usize
        ..index(bottom, gt[3], gt[5], shape.0).ceil() as usize;
    let cols = index(left, gt[0], gt[1], shape.1).floor() as usize
        ..index(right, gt[0], gt[1], shape.1).ceil() as usize;
    Ok((rows, cols))
}

fn write_interferogram_pair(
    cfg: &PairsConfig,
    grid: &DualBandGrid,
    settings: &InterferogramSettings,
    pair: &PlannedPair,
) -> Result<PairRecord> {
    let main = read_pair_stack(
        cfg,
        &cfg.subdataset,
        read_nisar_window,
        grid.main_source,
        pair,
    )?;
    let side = read_pair_stack(
        cfg,
        &grid.side_subdataset,
        read_nisar_window,
        grid.side_source,
        pair,
    )?;
    let nodata = Zip::from(&nodata_mask(main.view(), grid.looks))
        .and(&nodata_mask(side.view(), SINGLE_LOOK))
        .map_collect(|&main_bad, &side_bad| main_bad || side_bad);
    let coherence = main_band_coherence(cfg, grid, main.view(), &nodata)?;
    let mut record = pair_record(cfg, pair);
    let scratch = cfg.work_directory.join(format!(
        ".snaphu_{}_{}",
        record.reference_date, record.secondary_date
    ));
    let main_ifg = multilook_interferogram(main.view(), grid.looks);
    let (main_unwrapped, components) =
        unwrap_band(&main_ifg, &coherence, &nodata, &scratch.join("frequencyA"))?;
    let side_ifg = multilook_interferogram(side.view(), SINGLE_LOOK);
    let (side_unwrapped, _) =
        unwrap_band(&side_ifg, &coherence, &nodata, &scratch.join("frequencyB"))?;
    std::fs::remove_dir_all(&scratch).with_context(|| format!("removing {}", scratch.display()))?;
    let (ionosphere, alignment) = split_spectrum_ionosphere(
        grid,
        settings,
        &main_unwrapped,
        &side_unwrapped,
        &components,
    )?;
    let layers = PairLayers {
        unwrapped: main_unwrapped.mapv(|value| value as f32),
        ionosphere: ionosphere.mapv(|value| value as f32),
        coherence,
        connected_components: components,
    };
    record.interferogram = Some(summarize(&layers.coherence, &layers.ionosphere, &alignment));
    record.files = write_interferogram_layers(cfg, grid, pair, &record, &layers)?;
    Ok(record)
}

/// Complex mean of `ref · conj(sec)` over each `looks` block.
fn multilook_interferogram(stack: ArrayView3<Cf64>, looks: Strides) -> Array2<Cf32> {
    let (_, rows, cols) = stack.dim();
    let count = (looks.y * looks.x) as f64;
    Array2::from_shape_fn(looks.out_shape((rows, cols)), |(row, col)| {
        let (r0, c0) = (row * looks.y, col * looks.x);
        let window = s![r0..r0 + looks.y, c0..c0 + looks.x];
        let sum = Zip::from(stack.index_axis(Axis(0), 0).slice(window))
            .and(stack.index_axis(Axis(0), 1).slice(window))
            .fold(Cf64::new(0.0, 0.0), |acc, &reference, &secondary| {
                acc + reference * secondary.conj()
            });
        let mean = sum / count;
        Cf32::new(mean.re as f32, mean.im as f32)
    })
}

/// `|Γ_AB|` of the main band with `half_window` and strides equal to the looks.
fn main_band_coherence(
    cfg: &PairsConfig,
    grid: &DualBandGrid,
    main: ArrayView3<Cf64>,
    nodata: &Array2<bool>,
) -> Result<Array2<f32>> {
    let covariance = estimate_stack_covariance(main, cfg.half_window, grid.looks, None)
        .map_err(anyhow::Error::msg)
        .context("estimating frequencyA pair coherence")?;
    let magnitude = Array2::from_shape_fn(nodata.dim(), |(row, col)| {
        covariance[(row, col, 0, 1)].norm() as f32
    });
    Ok(masked(&magnitude, nodata, f32::NAN))
}

/// `values` with every nodata pixel replaced by `fill`.
fn masked<T: Copy>(values: &Array2<T>, nodata: &Array2<bool>, fill: T) -> Array2<T> {
    Zip::from(values)
        .and(nodata)
        .map_collect(|&value, &bad| match bad {
            true => fill,
            false => value,
        })
}

/// SNAPHU-unwrap one band; nodata goes in as zero signal and comes out NaN.
/// Returns the unwrapped phase and the connected-component labels.
fn unwrap_band(
    wrapped: &Array2<Cf32>,
    coherence: &Array2<f32>,
    nodata: &Array2<bool>,
    scratch: &Path,
) -> Result<(Array2<f64>, Array2<f32>)> {
    std::fs::create_dir_all(scratch).with_context(|| format!("creating {}", scratch.display()))?;
    let result = unwrap(
        masked(wrapped, nodata, Cf32::new(0.0, 0.0)).view(),
        masked(coherence, nodata, 0.0).view(),
        &UnwrapConfig::default(),
        scratch,
    )
    .context("unwrapping with snaphu")?;
    let unwrapped = masked(&result.unwrapped.mapv(f64::from), nodata, f64::NAN);
    let components = masked(
        &result.conncomp.mapv(|label| label as f32),
        nodata,
        f32::NAN,
    );
    Ok((unwrapped, components))
}

/// Low-passed split-spectrum ionosphere in the main-band interferogram (radians), with
/// the side band first aligned to the main band's cycles per main-band component.
/// Components come from the main band: it is the written `connected_components` layer
/// and, multilooked from the wider band, the better-unwrapped of the two.
fn split_spectrum_ionosphere(
    grid: &DualBandGrid,
    settings: &InterferogramSettings,
    main_unwrapped: &Array2<f64>,
    side_unwrapped: &Array2<f64>,
    components: &Array2<f32>,
) -> Result<(Array2<f64>, Vec<ComponentCycles>)> {
    let main = BandPhase {
        unwrapped_phase: main_unwrapped.view(),
        center_frequency_hz: grid.main_band.center_frequency_hz,
    };
    let side = BandPhase {
        unwrapped_phase: side_unwrapped.view(),
        center_frequency_hz: grid.side_band.center_frequency_hz,
    };
    // Nodata labels are NaN and cast to 0, but their phase is NaN in both bands, so
    // they never enter a component's median.
    let labels = components.mapv(|label| label as u32);
    let aligned = align_side_band_cycles(&main, &side, labels.view())
        .context("aligning frequencyB cycles to frequencyA")?;
    let side = BandPhase {
        unwrapped_phase: aligned.unwrapped_phase.view(),
        ..side
    };
    let raw = estimate_ionosphere(&main, Some(&side)).context("split-spectrum ionosphere")?;
    let gt = grid.geo.geotransform;
    let pixel_spacing_m = (gt[1] * gt[5]).abs().sqrt();
    let lowpassed = lowpass_ionosphere(raw.view(), settings.lowpass_wavelength_m, pixel_spacing_m)
        .context("low-pass filtering the ionosphere")?;
    Ok((lowpassed, aligned.components))
}

/// Pair summary over finite pixels.
fn summarize(
    coherence: &Array2<f32>,
    ionosphere: &Array2<f32>,
    alignment: &[ComponentCycles],
) -> InterferogramSummary {
    let finite = |values: &Array2<f32>| -> Vec<f64> {
        values
            .iter()
            .filter(|value| value.is_finite())
            .map(|&value| f64::from(value))
            .collect()
    };
    let mut coherence = finite(coherence);
    coherence.sort_by(f64::total_cmp);
    let ionosphere = finite(ionosphere);
    let count = |values: &[f64]| values.len() as f64;
    let above = coherence
        .iter()
        .filter(|&&value| value >= f64::from(SUMMARY_COHERENCE_THRESHOLD))
        .count();
    let mean = ionosphere.iter().sum::<f64>() / count(&ionosphere);
    let variance = ionosphere
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count(&ionosphere);
    let half = coherence.len() / 2;
    let median = match (coherence.len(), coherence.len() % 2) {
        (0, _) => None,
        (_, 1) => Some(coherence[half]),
        _ => Some((coherence[half - 1] + coherence[half]) / 2.0),
    };
    InterferogramSummary {
        coherence_fraction_at_least_0_2: (!coherence.is_empty())
            .then(|| above as f64 / count(&coherence)),
        median_coherence: median,
        ionosphere_std_rad: (!ionosphere.is_empty()).then(|| variance.sqrt()),
        side_band_cycle_alignment: alignment.iter().copied().map(Into::into).collect(),
    }
}

fn write_interferogram_layers(
    cfg: &PairsConfig,
    grid: &DualBandGrid,
    pair: &PlannedPair,
    record: &PairRecord,
    layers: &PairLayers,
) -> Result<Vec<PairFile>> {
    let owned = interferogram_tags(cfg, grid, pair)?;
    let tags: Vec<(&str, &str)> = owned.iter().map(|(k, v)| (*k, v.as_str())).collect();
    // f32 arithmetic on the written values, so `corrected − (unwrapped − ionosphere)`
    // is exactly zero when re-read.
    let corrected = &layers.unwrapped - &layers.ionosphere;
    [
        ("unwrapped", &layers.unwrapped),
        ("ionosphere", &layers.ionosphere),
        ("corrected", &corrected),
        ("coherence", &layers.coherence),
        ("connected_components", &layers.connected_components),
    ]
    .into_iter()
    .map(|(layer, data)| write_pair_layer(cfg, grid.geo, record, layer, data, &tags))
    .collect()
}

fn interferogram_tags(
    cfg: &PairsConfig,
    grid: &DualBandGrid,
    pair: &PlannedPair,
) -> Result<Vec<(&'static str, String)>> {
    let reference = &cfg.acquisitions[pair.reference].acquisition_id;
    let secondary = &cfg.acquisitions[pair.secondary].acquisition_id;
    let main_hz = grid.main_band.center_frequency_hz;
    Ok(vec![
        (
            "REFERENCE_CRID",
            composite_release_id(reference)?.to_owned(),
        ),
        (
            "SECONDARY_CRID",
            composite_release_id(secondary)?.to_owned(),
        ),
        ("CENTER_FREQUENCY_A_HZ", main_hz.to_string()),
        (
            "CENTER_FREQUENCY_B_HZ",
            grid.side_band.center_frequency_hz.to_string(),
        ),
        ("WAVELENGTH_M", (SPEED_OF_LIGHT_M_S / main_hz).to_string()),
        ("IONOSPHERE_METHOD", IONOSPHERE_METHOD.to_owned()),
        ("PHASE_SIGN", PHASE_SIGN.to_owned()),
    ])
}

/// The NISAR composite release ID (CRID) token of a granule ID, e.g. `A12345`.
fn composite_release_id(acquisition_id: &str) -> Result<&str> {
    acquisition_id
        .split('_')
        .find(|token| {
            token.len() == 6
                && token.starts_with(|c: char| c.is_ascii_uppercase())
                && token[1..].bytes().all(|b| b.is_ascii_digit())
        })
        .with_context(|| {
            format!("acquisition_id {acquisition_id} carries no NISAR composite release ID (CRID)")
        })
}

/// `los_enu.tif`: the first acquisition's ground→sensor LOS on the output grid.
fn write_los_enu(
    cfg: &PairsConfig,
    grid: &DualBandGrid,
    settings: &InterferogramSettings,
) -> Result<PairFile> {
    let source = &cfg.acquisitions[0];
    let shape = (grid.side_source.height(), grid.side_source.width());
    let layers = read_nisar_los_layers_for_grid(
        &source.path,
        settings.geometry_group,
        grid.geo,
        shape,
        Some(settings.dem),
    )
    .with_context(|| format!("reading NISAR LOS geometry of {}", source.acquisition_id))?
    .context("NISAR LOS geometry does not intersect the output grid")?;
    let up = Zip::from(&layers.east)
        .and(&layers.north)
        .map_collect(|&east, &north| (1.0 - east * east - north * north).max(0.0).sqrt());
    let path = cfg.work_directory.join(LOS_ENU_FILENAME);
    let bands = [
        ("east", layers.east.view()),
        ("north", layers.north.view()),
        ("up", up.view()),
    ];
    let tags = [
        ("LAYER", "los_enu"),
        ("GEOMETRY_ACQUISITION_ID", source.acquisition_id.as_str()),
    ];
    write_multiband_cog(&path, grid.geo, &bands, &tags)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(PairFile {
        layer: "los_enu".to_owned(),
        file: LOS_ENU_FILENAME.to_owned(),
        sha256: sha256_file(&path)?,
    })
}

/// Multi-band f32 COG with per-band descriptions and NaN nodata.
fn write_multiband_cog(
    path: &Path,
    geo: GeoInfo,
    bands: &[(&str, ArrayView2<f64>)],
    tags: &[(&str, &str)],
) -> Result<()> {
    let (rows, cols) = bands.first().context("no bands to write")?.1.dim();
    let mem = DriverManager::get_driver_by_name("MEM")?;
    let mut source = mem.create_with_band_type::<f32, _>("", cols, rows, bands.len())?;
    source.set_geo_transform(&geo.geotransform)?;
    source.set_spatial_ref(&SpatialRef::from_epsg(geo.epsg)?)?;
    for (key, value) in tags {
        source.set_metadata_item(key, value, "")?;
    }
    for (number, (name, data)) in bands.iter().enumerate() {
        let mut band = source.rasterband(number + 1)?;
        band.set_no_data_value(Some(f64::NAN))?;
        band.set_description(name)?;
        let mut buffer = Buffer::new((cols, rows), data.iter().map(|&v| v as f32).collect());
        band.write((0, 0), (cols, rows), &mut buffer)?;
    }
    let cog = DriverManager::get_driver_by_name("COG")?;
    let options =
        RasterCreationOptions::from_iter(["COMPRESS=DEFLATE", "BLOCKSIZE=256", "OVERVIEWS=AUTO"]);
    source.create_copy(&cog, path, &options)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("hashing {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn block(row_start: usize, row_stop: usize, col_start: usize, col_stop: usize) -> BlockIndices {
    BlockIndices {
        row_start,
        row_stop,
        col_start,
        col_stop,
    }
}
