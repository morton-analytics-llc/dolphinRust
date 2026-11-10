//! Dated pairwise products over a planned set of acquisition pairs (`dolphin pairs`).
//!
//! `mode: coherence` writes `|Γ_ij|` for each planned pair `(i, j)`: the magnitude of the
//! off-diagonal of the sample coherence matrix from
//! [`dolphin_phaselink::covariance::estimate_stack_covariance`], with the same
//! `half_window` / `strides` semantics as phase linking. Each pair is estimated on its own
//! two-acquisition stack; the entry is element-wise identical to the `(i, j)` off-diagonal
//! of the full-stack estimate because the normalization only involves `i` and `j`.
//!
//! Outputs, under `work_directory`:
//! - `<layer>_<refYYYYMMDD>_<secYYYYMMDD>.tif` per pair: a COG carrying the
//!   `REFERENCE_ACQUISITION_ID`, `SECONDARY_ACQUISITION_ID`, `REFERENCE_DATE`,
//!   `SECONDARY_DATE`, `TEMPORAL_BASELINE_DAYS` and `LAYER` tags.
//! - `pairs.json`: the run index listing every pair's IDs, dates, files and sha256.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use dolphin_core::config::{DisplacementWorkflow, InputType};
use dolphin_core::{BlockIndices, Cf64, HalfWindow, Strides};
use dolphin_io::{read_cslc_shape, read_cslc_window, read_geotransform, GeoInfo};
use dolphin_phaselink::covariance::estimate_stack_covariance;
use ndarray::{Array2, Array3, Axis, Zip};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::burst::BurstGeo;
use crate::crop::plan_bounds;

/// File name of the per-run pair index written under `work_directory`.
pub const PAIRS_INDEX_FILENAME: &str = "pairs.json";

/// EPSG code of the `bounds` envelope.
const BOUNDS_EPSG: u32 = 4326;

/// Date format used in pair file names.
const FILE_DATE_FORMAT: &str = "%Y%m%d";

/// Date format used in GDAL tags and `pairs.json`.
const TAG_DATE_FORMAT: &str = "%Y-%m-%d";

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
/// Returns `Err` on invalid config, unsupported input type, I/O failure, inconsistent
/// acquisition grids, or bounds that do not fit the grid.
pub fn run_pairs(cfg: &PairsConfig) -> Result<PairsIndex> {
    validate_config(cfg)?;
    match cfg.input_type {
        InputType::OperaCslc => {}
        InputType::NisarGslc => bail!(
            "dolphin pairs does not support input_type nisar_gslc; only opera_cslc is implemented"
        ),
    }
    let planned = plan_pairs(&cfg.acquisitions, &cfg.pair_plan)?;
    let grid = pair_grid(cfg)?;
    std::fs::create_dir_all(&cfg.work_directory)
        .with_context(|| format!("creating {}", cfg.work_directory.display()))?;
    let pairs = planned
        .iter()
        .map(|pair| match cfg.mode {
            PairsMode::Coherence => write_coherence_pair(cfg, &grid, pair),
        })
        .collect::<Result<Vec<_>>>()?;
    let index = PairsIndex {
        mode: cfg.mode,
        pairs,
    };
    let index_path = cfg.work_directory.join(PAIRS_INDEX_FILENAME);
    std::fs::write(&index_path, serde_json::to_string_pretty(&index)?)
        .with_context(|| format!("writing {}", index_path.display()))?;
    Ok(index)
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
    let stack = read_pair_stack(cfg, grid.source, pair)?;
    let covariance = estimate_stack_covariance(stack.view(), cfg.half_window, cfg.strides, None)
        .map_err(anyhow::Error::msg)
        .context("estimating pair coherence")?;
    let trim = grid.trim;
    let coherence = Array2::from_shape_fn((trim.height(), trim.width()), |(row, col)| {
        covariance[(row + trim.row_start, col + trim.col_start, 0, 1)].norm() as f32
    });
    write_pair_layer(cfg, grid, pair, "coherence", &coherence)
}

/// `(2, rows, cols)` stack of the reference and secondary over `source`.
fn read_pair_stack(
    cfg: &PairsConfig,
    source: BlockIndices,
    pair: &PlannedPair,
) -> Result<Array3<Cf64>> {
    let mut stack = Array3::<Cf64>::zeros((2, source.height(), source.width()));
    for (layer, index) in [pair.reference, pair.secondary].into_iter().enumerate() {
        let acquisition = &cfg.acquisitions[index];
        let window = read_cslc_window(&acquisition.path, &cfg.subdataset, source)
            .with_context(|| format!("reading {}", acquisition.acquisition_id))?;
        Zip::from(stack.index_axis_mut(Axis(0), layer))
            .and(&window)
            .for_each(|dst, z| *dst = Cf64::new(f64::from(z.re), f64::from(z.im)));
    }
    Ok(stack)
}

fn write_pair_layer(
    cfg: &PairsConfig,
    grid: &PairGrid,
    pair: &PlannedPair,
    layer: &str,
    data: &Array2<f32>,
) -> Result<PairRecord> {
    let reference = &cfg.acquisitions[pair.reference];
    let secondary = &cfg.acquisitions[pair.secondary];
    let (ref_date, sec_date) = (acquisition_date(reference), acquisition_date(secondary));
    let file = format!(
        "{layer}_{}_{}.tif",
        ref_date.format(FILE_DATE_FORMAT),
        sec_date.format(FILE_DATE_FORMAT)
    );
    let record = PairRecord {
        reference_acquisition_id: reference.acquisition_id.clone(),
        secondary_acquisition_id: secondary.acquisition_id.clone(),
        reference_date: ref_date.format(TAG_DATE_FORMAT).to_string(),
        secondary_date: sec_date.format(TAG_DATE_FORMAT).to_string(),
        temporal_baseline_days: pair.temporal_baseline_days,
        files: Vec::new(),
    };
    let baseline = record.temporal_baseline_days.to_string();
    let tags = [
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
    let path = cfg.work_directory.join(&file);
    dolphin_io::geotiff::write_raster_with_metadata(
        &path,
        data.view(),
        grid.geo.geotransform,
        Some(grid.geo.epsg),
        None,
        &tags,
    )
    .with_context(|| format!("writing {}", path.display()))?;
    let sha256 = sha256_file(&path)?;
    Ok(PairRecord {
        files: vec![PairFile {
            layer: layer.to_owned(),
            file,
            sha256,
        }],
        ..record
    })
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
