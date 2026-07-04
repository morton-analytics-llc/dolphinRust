//! Geometry-provenance artifact (dolphinRust #1 / eo #120).
//!
//! Assembles per-run geometry provenance from real product metadata — never from
//! config knobs, filename guesses, or platform nominals — and writes it as
//! `geometry_provenance.json` beside the rasters. Absent provenance is explicit:
//! a `null` scalar always pairs with an `absent` entry carrying the reason, and
//! `decomposition_geometry_complete` is the fail-safe bit GroundPulse gates
//! asc/desc decomposition on. Design + per-field sources:
//! `md/design/geometry-provenance.md`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::NaiveDateTime;
use dolphin_core::config::{DisplacementWorkflow, InputType};
use dolphin_corrections::LosGeometry;
use dolphin_io::{
    read_cslc_burst_metadata, read_cslc_identification, read_cslc_orbit, CslcBurstMetadata,
    CslcIdentification, CslcOrbit,
};
use serde::{Deserialize, Serialize};

/// Artifact filename inside `work_directory`.
pub const GEOMETRY_PROVENANCE_FILENAME: &str = "geometry_provenance.json";

const SCHEMA: &str = "dolphinrust-geometry-provenance/1";
const METHOD_VERSION: &str = "1.0.0";
/// `temporal_coherence.tif` is the phase-linking quality raster (see
/// `write_outputs`); this key is relative to `work_directory`.
const PHASE_LINKING_COHERENCE_KEY: &str = "temporal_coherence.tif";
const DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.f";

const HEADING_SPREAD_GATE_DEG: f64 = 1.0;
const RANGE_SPACING_GATE_M: f64 = 1e-6;
const AZIMUTH_SPACING_GATE_M: f64 = 0.1;
const TIME_OF_DAY_GATE_S: f64 = 60.0;
/// Single-subswath incidence spreads are ~1.6° std; cross-subswath mixes exceed
/// 4° — a frame-level scalar incidence is not a safe decomposition input there.
const INCIDENCE_SPREAD_GATE_DEG: f64 = 3.0;

const WGS84_A_M: f64 = 6_378_137.0;
const WGS84_B_M: f64 = 6_356_752.314_245;

/// Machine-readable geometry provenance for one displacement run. Scalar `None`
/// always pairs with an `Absent` entry in `geometry_provenance.fields`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryProvenance {
    /// Schema identifier (`dolphinrust-geometry-provenance/1`).
    pub schema: String,
    /// Derivation method version.
    pub method_version: String,
    /// `ascending`/`descending`, normalized lowercase.
    pub orbit_direction: Option<String>,
    /// Spatial mean ellipsoidal incidence over the output grid, degrees.
    pub incidence_angle_deg: Option<f64>,
    /// Population std of the per-pixel incidence, degrees.
    pub incidence_angle_spread_deg: Option<f64>,
    /// Minimum per-pixel incidence, degrees.
    pub incidence_angle_min_deg: Option<f64>,
    /// Maximum per-pixel incidence, degrees.
    pub incidence_angle_max_deg: Option<f64>,
    /// Platform-velocity azimuth in the scene-center ENU frame, degrees clockwise
    /// from geographic north, `[0, 360)`.
    pub heading_deg: Option<f64>,
    /// Native slant-range pixel spacing, meters.
    pub native_range_spacing_m: Option<f64>,
    /// Ground-projected azimuth line spacing, meters.
    pub native_azimuth_spacing_m: Option<f64>,
    /// Zero-doppler mid-time seconds-of-day (UTC), mean across granules.
    pub acquisition_time_of_day_utc_s: Option<f64>,
    /// Artifact key of the phase-linking coherence raster, relative to
    /// `work_directory`.
    pub phase_linking_coherence: String,
    /// Fail-safe decomposition gate: `orbit_direction`, `incidence_angle_deg`, and
    /// `heading_deg` all sourced AND incidence spread within the gate.
    pub decomposition_geometry_complete: bool,
    /// Per-field source files/keys/method — the block eo persists as JSONB.
    pub geometry_provenance: ProvenanceBlock,
}

/// The nested provenance block naming source metadata keys + method version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBlock {
    /// Derivation method version (self-contained copy for JSONB persistence).
    pub method_version: String,
    /// Per-field provenance, keyed by the top-level scalar name.
    pub fields: BTreeMap<String, FieldProvenance>,
}

/// Where one exported field came from — or why it is absent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum FieldProvenance {
    /// Read/derived from real product metadata.
    Sourced {
        /// Granule filenames the value was read from.
        source_files: Vec<String>,
        /// HDF5 keys involved.
        source_keys: Vec<String>,
        /// Derivation method (versioned by `method_version`).
        method: String,
        /// Raw product string before normalization, where applicable.
        #[serde(skip_serializing_if = "Option::is_none")]
        raw_value: Option<String>,
        /// Caveat that does not invalidate sourcing (e.g. spread-gate breach).
        #[serde(skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    /// Not sourceable — never defaulted.
    Absent {
        /// Why (missing keys, inconsistency, unsupported input type, …).
        reason: String,
    },
}

/// Per-granule metadata bundle (all three groups read successfully).
struct GranuleMeta {
    name: String,
    ident: CslcIdentification,
    burst: CslcBurstMetadata,
    orbit: CslcOrbit,
}

/// Per-granule derived geometry.
struct GranuleGeometry {
    heading_deg: f64,
    azimuth_spacing_m: f64,
    time_of_day_s: f64,
}

/// Assemble the run's geometry provenance from the config's input granules and the
/// resolved per-pixel LOS geometry (when CSLC-S1-STATIC layers were supplied).
#[must_use]
pub fn assemble_geometry_provenance(
    cfg: &DisplacementWorkflow,
    los: Option<&LosGeometry>,
) -> GeometryProvenance {
    let mut fields = BTreeMap::new();
    let cslc = read_granules(cfg, &mut fields);
    let orbit_direction = orbit_direction(&cslc, &mut fields);
    let heading_deg = heading(&cslc, &mut fields);
    let native_range_spacing_m = range_spacing(&cslc, &mut fields);
    let native_azimuth_spacing_m = azimuth_spacing(&cslc, &mut fields);
    let acquisition_time_of_day_utc_s = time_of_day(&cslc, &mut fields);
    let incidence = incidence(cfg, los, &cslc, &mut fields);

    for (field, prov) in &fields {
        if let FieldProvenance::Absent { reason } = prov {
            tracing::warn!(field, reason, "geometry provenance field absent");
        }
    }
    let incidence_ok = incidence.is_some_and(|s| s.std_deg <= INCIDENCE_SPREAD_GATE_DEG);
    // Time-of-day rides on the same derivation as heading; when granules are
    // readable it can only be absent through its consistency gate, which signals a
    // mixed stack (e.g. an adjacent relative orbit slipped in) — withhold safety.
    GeometryProvenance {
        schema: SCHEMA.into(),
        method_version: METHOD_VERSION.into(),
        orbit_direction: orbit_direction.clone(),
        incidence_angle_deg: incidence.map(|s| s.mean_deg),
        incidence_angle_spread_deg: incidence.map(|s| s.std_deg),
        incidence_angle_min_deg: incidence.map(|s| s.min_deg),
        incidence_angle_max_deg: incidence.map(|s| s.max_deg),
        heading_deg,
        native_range_spacing_m,
        native_azimuth_spacing_m,
        acquisition_time_of_day_utc_s,
        phase_linking_coherence: PHASE_LINKING_COHERENCE_KEY.into(),
        decomposition_geometry_complete: orbit_direction.is_some()
            && heading_deg.is_some()
            && acquisition_time_of_day_utc_s.is_some()
            && incidence_ok,
        geometry_provenance: ProvenanceBlock {
            method_version: METHOD_VERSION.into(),
            fields,
        },
    }
}

/// Write the artifact into `dir` (unconditionally, overwriting any previous run's
/// file — a stale artifact would misattribute prior provenance to this run).
///
/// # Errors
/// Propagates serialization and filesystem errors; a failed write fails the run.
pub fn write_geometry_provenance(dir: &Path, prov: &GeometryProvenance) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let json = serde_json::to_string_pretty(prov)?;
    std::fs::write(dir.join(GEOMETRY_PROVENANCE_FILENAME), json + "\n")?;
    Ok(())
}

/// Fields whose provenance comes from the CSLC acquisition granules.
const CSLC_FIELDS: [&str; 5] = [
    "orbit_direction",
    "heading_deg",
    "native_range_spacing_m",
    "native_azimuth_spacing_m",
    "acquisition_time_of_day_utc_s",
];

/// `Sourced` entry with no raw value or note (the common case).
fn sourced(source_files: Vec<String>, source_keys: Vec<String>, method: &str) -> FieldProvenance {
    FieldProvenance::Sourced {
        source_files,
        source_keys,
        method: method.into(),
        raw_value: None,
        note: None,
    }
}

/// Mark `names` absent with a shared `reason`; returns `None` for tail-position use.
fn mark_absent<T>(
    fields: &mut BTreeMap<String, FieldProvenance>,
    names: &[&str],
    reason: &str,
) -> Option<T> {
    for name in names {
        fields.insert(
            (*name).into(),
            FieldProvenance::Absent {
                reason: reason.into(),
            },
        );
    }
    None
}

/// Read all three metadata groups from every CSLC granule. On any failure, mark
/// every CSLC-derived field absent (sourced requires the full stack readable) and
/// return `None`.
fn read_granules(
    cfg: &DisplacementWorkflow,
    fields: &mut BTreeMap<String, FieldProvenance>,
) -> Option<Vec<GranuleMeta>> {
    if cfg.input_options.input_type == InputType::NisarGslc {
        let reason = "NISAR GSLC geometry metadata mapping not implemented";
        return mark_absent(fields, &CSLC_FIELDS, reason);
    }
    if cfg.cslc_file_list.is_empty() {
        return mark_absent(fields, &CSLC_FIELDS, "cslc_file_list is empty");
    }
    let read = |path: &PathBuf| -> Result<GranuleMeta, String> {
        let name = granule_name(path);
        let context = |e: dolphin_io::IoError| format!("granule {name}: {e}");
        Ok(GranuleMeta {
            ident: read_cslc_identification(path).map_err(context)?,
            burst: read_cslc_burst_metadata(path).map_err(context)?,
            orbit: read_cslc_orbit(path).map_err(context)?,
            name: granule_name(path),
        })
    };
    match cfg.cslc_file_list.iter().map(read).collect() {
        Ok(granules) => Some(granules),
        Err(reason) => mark_absent(fields, &CSLC_FIELDS, &reason),
    }
}

fn granule_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into(),
    )
}

fn granule_names(granules: &[GranuleMeta]) -> Vec<String> {
    granules.iter().map(|g| g.name.clone()).collect()
}

/// `/identification/orbit_pass_direction`, case-insensitive, all granules agreeing.
fn orbit_direction(
    cslc: &Option<Vec<GranuleMeta>>,
    fields: &mut BTreeMap<String, FieldProvenance>,
) -> Option<String> {
    let granules = cslc.as_ref()?;
    let raw: Vec<&str> = granules
        .iter()
        .map(|g| g.ident.orbit_pass_direction.as_str())
        .collect();
    let normalized: Vec<String> = raw.iter().map(|v| v.to_ascii_lowercase()).collect();
    let value = &normalized[0];
    if !matches!(value.as_str(), "ascending" | "descending") {
        return mark_absent(
            fields,
            &["orbit_direction"],
            &format!("unrecognized orbit_pass_direction {:?}", raw[0]),
        );
    }
    if normalized.iter().any(|v| v != value) {
        return mark_absent(
            fields,
            &["orbit_direction"],
            &format!("orbit_pass_direction inconsistent across granules: {raw:?}"),
        );
    }
    fields.insert(
        "orbit_direction".into(),
        FieldProvenance::Sourced {
            source_files: granule_names(granules),
            source_keys: vec!["/identification/orbit_pass_direction".into()],
            method: "read scalar per granule, case-insensitive consistency".into(),
            raw_value: Some(raw[0].into()),
            note: None,
        },
    );
    Some(value.clone())
}

/// Keys behind the per-granule orbit-geometry derivation.
fn orbit_geometry_keys() -> Vec<String> {
    [
        "/metadata/orbit/time",
        "/metadata/orbit/velocity_x",
        "/metadata/orbit/velocity_y",
        "/metadata/orbit/velocity_z",
        "/metadata/orbit/position_x",
        "/metadata/orbit/position_y",
        "/metadata/orbit/position_z",
        "/metadata/orbit/reference_epoch",
        "/identification/zero_doppler_start_time",
        "/identification/zero_doppler_end_time",
        "/metadata/processing_information/input_burst_metadata/center",
    ]
    .map(String::from)
    .to_vec()
}

/// Derive per-granule geometry, or mark `fields` absent with the failing reason.
fn derived_geometry(
    cslc: &Option<Vec<GranuleMeta>>,
    fields: &mut BTreeMap<String, FieldProvenance>,
    for_fields: &[&str],
) -> Option<Vec<GranuleGeometry>> {
    let granules = cslc.as_ref()?;
    match granules.iter().map(granule_geometry).collect() {
        Ok(derived) => Some(derived),
        Err(reason) => mark_absent(fields, for_fields, &reason),
    }
}

/// Platform-velocity heading in scene-center ENU: circular mean across granules.
fn heading(
    cslc: &Option<Vec<GranuleMeta>>,
    fields: &mut BTreeMap<String, FieldProvenance>,
) -> Option<f64> {
    let derived = derived_geometry(cslc, fields, &["heading_deg"])?;
    let headings: Vec<f64> = derived.iter().map(|d| d.heading_deg).collect();
    let (mean, max_dev) = circular_mean_deg(&headings);
    if !mean.is_finite() || max_dev > HEADING_SPREAD_GATE_DEG {
        let reason = format!(
            "heading spread {max_dev:.3}° exceeds {HEADING_SPREAD_GATE_DEG}° gate (or non-finite)"
        );
        return mark_absent(fields, &["heading_deg"], &reason);
    }
    let method = "ECEF velocity linearly interpolated at mid zero-doppler time, rotated to ENU \
                  at scene center (geodetic lat); atan2(v_e, v_n); vector-sum circular mean \
                  across granules";
    fields.insert(
        "heading_deg".into(),
        sourced(granule_names(cslc.as_ref()?), orbit_geometry_keys(), method),
    );
    Some(mean)
}

/// `range_pixel_spacing`: a constant of the acquisition mode; exact agreement.
fn range_spacing(
    cslc: &Option<Vec<GranuleMeta>>,
    fields: &mut BTreeMap<String, FieldProvenance>,
) -> Option<f64> {
    let granules = cslc.as_ref()?;
    let values: Vec<f64> = granules
        .iter()
        .map(|g| g.burst.range_pixel_spacing_m)
        .collect();
    let (min, max) = min_max(&values);
    // `any(!finite)` (not `min.is_finite()`): f64::min/max swallow NaN, so a NaN
    // sample would otherwise pass the gate and serialize a sourced-looking null.
    if values.iter().any(|v| !v.is_finite()) || min <= 0.0 || max - min > RANGE_SPACING_GATE_M {
        let reason = format!(
            "range_pixel_spacing non-finite, non-positive, or inconsistent across granules \
             ({min}..{max})"
        );
        return mark_absent(fields, &["native_range_spacing_m"], &reason);
    }
    let keys =
        vec!["/metadata/processing_information/input_burst_metadata/range_pixel_spacing".into()];
    fields.insert(
        "native_range_spacing_m".into(),
        sourced(
            granule_names(granules),
            keys,
            "read scalar per granule, exact consistency (slant-range spacing)",
        ),
    );
    Some(values[0])
}

/// Ground-projected azimuth line spacing: mean across granules.
fn azimuth_spacing(
    cslc: &Option<Vec<GranuleMeta>>,
    fields: &mut BTreeMap<String, FieldProvenance>,
) -> Option<f64> {
    let derived = derived_geometry(cslc, fields, &["native_azimuth_spacing_m"])?;
    let values: Vec<f64> = derived.iter().map(|d| d.azimuth_spacing_m).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let max_dev = values.iter().map(|v| (v - mean).abs()).fold(0.0, f64::max);
    if !mean.is_finite() || mean <= 0.0 || max_dev > AZIMUTH_SPACING_GATE_M {
        let reason = format!(
            "azimuth spacing deviation {max_dev:.4} m exceeds {AZIMUTH_SPACING_GATE_M} m gate \
             (or non-finite/non-positive)"
        );
        return mark_absent(fields, &["native_azimuth_spacing_m"], &reason);
    }
    let mut keys = orbit_geometry_keys();
    keys.push("/metadata/processing_information/input_burst_metadata/azimuth_time_interval".into());
    let method = "azimuth_time_interval × |v_ecef| × r_earth(lat)/|r_platform| (ground-projected \
                  azimuth line spacing); mean across granules";
    fields.insert(
        "native_azimuth_spacing_m".into(),
        sourced(granule_names(cslc.as_ref()?), keys, method),
    );
    Some(mean)
}

/// Zero-doppler mid-time seconds-of-day: mean across granules.
fn time_of_day(
    cslc: &Option<Vec<GranuleMeta>>,
    fields: &mut BTreeMap<String, FieldProvenance>,
) -> Option<f64> {
    let derived = derived_geometry(cslc, fields, &["acquisition_time_of_day_utc_s"])?;
    let values: Vec<f64> = derived.iter().map(|d| d.time_of_day_s).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let max_dev = values.iter().map(|v| (v - mean).abs()).fold(0.0, f64::max);
    if !mean.is_finite() || max_dev > TIME_OF_DAY_GATE_S {
        let reason = format!(
            "zero-doppler time-of-day deviation {max_dev:.1} s exceeds {TIME_OF_DAY_GATE_S} s \
             gate (or non-finite)"
        );
        return mark_absent(fields, &["acquisition_time_of_day_utc_s"], &reason);
    }
    let keys = vec![
        "/identification/zero_doppler_start_time".into(),
        "/identification/zero_doppler_end_time".into(),
    ];
    fields.insert(
        "acquisition_time_of_day_utc_s".into(),
        sourced(
            granule_names(cslc.as_ref()?),
            keys,
            "mid zero-doppler time seconds-of-day (UTC); mean across granules",
        ),
    );
    Some(mean)
}

/// Incidence stats from the resolved per-pixel LOS — never from the
/// `correction_options.incidence_angle_deg` atmospheric knob.
fn incidence(
    cfg: &DisplacementWorkflow,
    los: Option<&LosGeometry>,
    cslc: &Option<Vec<GranuleMeta>>,
    fields: &mut BTreeMap<String, FieldProvenance>,
) -> Option<dolphin_corrections::geometry::IncidenceStats> {
    const FIELD: &[&str] = &["incidence_angle_deg"];
    if cfg.correction_options.geometry_files.is_empty() {
        let reason = "correction_options.geometry_files not supplied (no CSLC-S1-STATIC geometry)";
        return mark_absent(fields, FIELD, reason);
    }
    let Some(stats) = los.and_then(LosGeometry::incidence_stats) else {
        return mark_absent(fields, FIELD, "per-pixel LOS geometry not resolved");
    };
    if !(stats.mean_deg.is_finite() && stats.std_deg.is_finite()) {
        return mark_absent(fields, FIELD, "incidence statistics non-finite");
    }
    // A wrong-track/wrong-pass STATIC yields plausible incidence (up is
    // sign-insensitive) — cross-check its identity against the CSLC stack.
    let consistency_note = match cslc.as_ref() {
        Some(granules) => {
            if let Err(reason) =
                verify_static_consistency(&cfg.correction_options.geometry_files, granules)
            {
                return mark_absent(fields, FIELD, &reason);
            }
            None
        }
        None => Some(
            "consistency with CSLC stack unverified (CSLC identification unreadable); \
             decomposition gate is closed via the CSLC-derived fields anyway"
                .to_string(),
        ),
    };
    let spread_note = (stats.std_deg > INCIDENCE_SPREAD_GATE_DEG).then(|| {
        format!(
            "incidence spread {:.2}° > {INCIDENCE_SPREAD_GATE_DEG}° gate (multi-subswath \
             frame?) — decomposition_geometry_complete withheld",
            stats.std_deg
        )
    });
    let note = match (spread_note, consistency_note) {
        (Some(a), Some(b)) => Some(format!("{a}; {b}")),
        (a, b) => a.or(b),
    };
    fields.insert(
        "incidence_angle_deg".into(),
        FieldProvenance::Sourced {
            source_files: cfg
                .correction_options
                .geometry_files
                .iter()
                .map(|p| granule_name(p))
                .collect(),
            source_keys: vec!["/data/los_east".into(), "/data/los_north".into()],
            method: "mean/std/min/max over finite pixels of degrees(acos(los_up)), \
                 up = +sqrt(1−e²−n²), on the resolved output grid"
                .into(),
            raw_value: None,
            note,
        },
    );
    Some(stats)
}

/// Verify each CSLC-S1-STATIC granule belongs to the same pass and burst set as
/// the CSLC stack (STATIC products carry the same `/identification` group). A
/// mismatched STATIC would otherwise source plausible-but-wrong incidence.
fn verify_static_consistency(
    geometry_files: &[PathBuf],
    granules: &[GranuleMeta],
) -> Result<(), String> {
    let stack_passes: Vec<String> = granules
        .iter()
        .map(|g| g.ident.orbit_pass_direction.to_ascii_lowercase())
        .collect();
    let stack_bursts: Vec<String> = granules
        .iter()
        .map(|g| normalize_burst_id(&g.ident.burst_id))
        .collect();
    for path in geometry_files {
        let name = granule_name(path);
        let ident = read_cslc_identification(path)
            .map_err(|e| format!("STATIC granule {name}: identification unreadable ({e}) — cannot verify pass/burst consistency"))?;
        let pass = ident.orbit_pass_direction.to_ascii_lowercase();
        if !stack_passes.contains(&pass) {
            return Err(format!(
                "STATIC granule {name} orbit_pass_direction {:?} does not match the CSLC stack",
                ident.orbit_pass_direction
            ));
        }
        let burst = normalize_burst_id(&ident.burst_id);
        if !stack_bursts.contains(&burst) {
            return Err(format!(
                "STATIC granule {name} burst_id {burst:?} not in the CSLC stack's burst set"
            ));
        }
    }
    Ok(())
}

/// Normalize a burst id for comparison (`T144-308011-IW2` ≡ `t144_308011_iw2`).
fn normalize_burst_id(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace('-', "_")
}

/// Heading, azimuth spacing, and time-of-day for one granule.
fn granule_geometry(g: &GranuleMeta) -> Result<GranuleGeometry, String> {
    let context = |what: &str| format!("granule {}: {what}", g.name);
    let epoch = parse_datetime(&g.orbit.reference_epoch)
        .ok_or_else(|| context("unparseable orbit reference_epoch"))?;
    let start = parse_datetime(&g.ident.zero_doppler_start_time)
        .ok_or_else(|| context("unparseable zero_doppler_start_time"))?;
    let end = parse_datetime(&g.ident.zero_doppler_end_time)
        .ok_or_else(|| context("unparseable zero_doppler_end_time"))?;
    let mid = start + (end - start) / 2;
    let t_mid_s = seconds_between(epoch, mid);
    if g.orbit.time_s.len() < 2 {
        return Err(context("fewer than 2 orbit state vectors"));
    }
    // Out-of-span interpolation would silently clamp to an edge state vector and
    // produce a plausible-but-wrong sourced heading.
    let (t_first, t_last) = (g.orbit.time_s[0], g.orbit.time_s[g.orbit.time_s.len() - 1]);
    if !t_mid_s.is_finite() || t_mid_s < t_first || t_mid_s > t_last {
        return Err(context(&format!(
            "mid zero-doppler time {t_mid_s:.1}s outside orbit state-vector span \
             [{t_first:.1}, {t_last:.1}]s"
        )));
    }
    let velocity = interp3_clamped(&g.orbit.time_s, &g.orbit.velocity_mps, t_mid_s);
    let position = interp3_clamped(&g.orbit.time_s, &g.orbit.position_m, t_mid_s);
    let [lon_deg, lat_deg] = g.burst.center_lonlat_deg;
    let heading_deg = enu_heading_deg(velocity, lon_deg.to_radians(), lat_deg.to_radians());
    let azimuth_spacing_m = g.burst.azimuth_time_interval_s
        * norm3(velocity)
        * (geocentric_radius_m(lat_deg.to_radians()) / norm3(position));
    let midnight = mid
        .date()
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| context("bad date"))?;
    Ok(GranuleGeometry {
        heading_deg,
        azimuth_spacing_m,
        time_of_day_s: seconds_between(midnight, mid),
    })
}

fn parse_datetime(raw: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(raw.trim(), DATETIME_FORMAT).ok()
}

fn seconds_between(from: NaiveDateTime, to: NaiveDateTime) -> f64 {
    (to - from)
        .num_microseconds()
        .map_or(f64::NAN, |us| us as f64 / 1e6)
}

/// `numpy.interp`-style clamped linear interpolation of `[x, y, z]` samples.
fn interp3_clamped(t: &[f64], values: &[[f64; 3]], x: f64) -> [f64; 3] {
    let hi = t.partition_point(|&ti| ti < x).clamp(1, t.len() - 1);
    let lo = hi - 1;
    let span = t[hi] - t[lo];
    let frac = if span == 0.0 {
        0.0
    } else {
        ((x - t[lo]) / span).clamp(0.0, 1.0)
    };
    std::array::from_fn(|i| values[lo][i] + frac * (values[hi][i] - values[lo][i]))
}

/// Velocity azimuth (degrees clockwise from north, `[0, 360)`) in the ENU frame at
/// geodetic `(lon, lat)`.
fn enu_heading_deg(v: [f64; 3], lon_rad: f64, lat_rad: f64) -> f64 {
    let east = [-lon_rad.sin(), lon_rad.cos(), 0.0];
    let north = [
        -lat_rad.sin() * lon_rad.cos(),
        -lat_rad.sin() * lon_rad.sin(),
        lat_rad.cos(),
    ];
    let dot = |a: [f64; 3]| a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
    dot(east).atan2(dot(north)).to_degrees().rem_euclid(360.0)
}

/// WGS84 geocentric radius at geodetic latitude.
fn geocentric_radius_m(lat_rad: f64) -> f64 {
    let (a, b) = (WGS84_A_M, WGS84_B_M);
    let (cos, sin) = (lat_rad.cos(), lat_rad.sin());
    let num = (a * a * cos).powi(2) + (b * b * sin).powi(2);
    let den = (a * cos).powi(2) + (b * sin).powi(2);
    (num / den).sqrt()
}

fn norm3(v: [f64; 3]) -> f64 {
    v.iter().map(|c| c * c).sum::<f64>().sqrt()
}

/// Vector-sum circular mean (degrees, `[0, 360)`) and the max angular deviation of
/// any sample from it.
fn circular_mean_deg(headings: &[f64]) -> (f64, f64) {
    let (sin, cos) = headings
        .iter()
        .map(|h| h.to_radians().sin_cos())
        .fold((0.0, 0.0), |(s, c), (hs, hc)| (s + hs, c + hc));
    let mean = sin.atan2(cos).to_degrees().rem_euclid(360.0);
    let max_dev = headings
        .iter()
        .map(|h| ((h - mean).rem_euclid(360.0) + 540.0).rem_euclid(360.0) - 180.0)
        .fold(0.0, |acc: f64, d: f64| acc.max(d.abs()));
    (mean, max_dev)
}

fn min_max(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        })
}
