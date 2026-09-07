//! Multi-burst frame stitching.
//!
//! An OPERA frame is tiled by several bursts (e.g. `T064-135518-IW1/2/3`). Each
//! burst is phase-linked independently, then the per-date linked phase and
//! quality layers are mosaicked onto one frame grid before unwrapping (so phase
//! is continuous across burst seams). A later finite burst value replaces an
//! earlier one in overlaps, while later nodata does not erase existing finite
//! support. Because a frame's bursts share pixel posting and CRS, an
//! integer-offset paste onto the union grid is exact — no resampling. Bursts
//! with differing posting/CRS are rejected (reprojection is deferred).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use dolphin_core::config::InputType;
use dolphin_io::GeoInfo;
use ndarray::{Array2, Array3};

/// Pixel posting must match this closely (in CRS units) to stitch without resampling.
const POSTING_TOL: f64 = 1e-6;

/// Group CSLC file indices by burst id parsed from each filename, preserving
/// input order within a group. Files with no recognizable burst id fall into a
/// single `"single"` group, so single-burst stacks take the identity path.
///
/// The burst id is the OPERA token of the form `T###-######-IW#` (e.g.
/// `T064-135518-IW2`); matched without a regex by scanning `_`-delimited tokens.
#[must_use]
pub fn group_by_burst(files: &[PathBuf]) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, f) in files.iter().enumerate() {
        let id = burst_id(f).unwrap_or_else(|| "single".to_string());
        groups.entry(id).or_default().push(i);
    }
    groups
}

/// Resolve verified spatial groups, rejecting partial or incompatible metadata.
///
/// # Errors
/// Metadata must cover inputs exactly, preserve grids, and supply common ordered epochs.
pub fn workflow_groups(
    cfg: &dolphin_core::config::DisplacementWorkflow,
) -> Result<BTreeMap<String, Vec<usize>>> {
    use dolphin_core::config::InputType;
    let metadata = &cfg.input_options.acquisition_metadata;
    if metadata.is_empty() {
        return Ok(group_by_burst(&cfg.cslc_file_list));
    }
    ensure!(
        metadata.len() == cfg.cslc_file_list.len(),
        "acquisition metadata must cover every input"
    );
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, path) in cfg.cslc_file_list.iter().enumerate() {
        let matching: Vec<_> = metadata.iter().filter(|m| m.path == *path).collect();
        ensure!(
            matching.len() == 1,
            "each input needs exactly one acquisition metadata record"
        );
        let m = matching[0];
        ensure!(
            !m.spatial_group.is_empty() && !m.grid_id.is_empty(),
            "spatial group and grid identity are required"
        );
        groups.entry(m.spatial_group.clone()).or_default().push(i);
    }
    ensure!(
        cfg.input_options.input_type != InputType::NisarGslc || groups.len() == 1,
        "NISAR requires one spatial group per run"
    );
    let mut common_dates = None;
    let mut common_utc = None;
    for indices in groups.values() {
        let records: Vec<_> = indices
            .iter()
            .map(|&i| {
                metadata
                    .iter()
                    .find(|m| m.path == cfg.cslc_file_list[i])
                    .ok_or_else(|| anyhow::anyhow!("missing acquisition metadata"))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            records.iter().all(|m| m.grid_id == records[0].grid_id),
            "spatial group changes grid identity"
        );
        let dates: Vec<_> = records.iter().map(|m| m.acquisition_utc).collect();
        ensure!(
            dates.windows(2).all(|w| w[0] < w[1]),
            "acquisition UTC must be unique and increasing within each group"
        );
        if !cfg.correction_options.ionosphere_files.is_empty()
            || cfg.correction_options.solid_earth_tide
        {
            if let Some(expected) = &common_utc {
                ensure!(expected == &dates, "temporal corrections require identical UTC across spatial groups; per-burst correction evaluation is required");
            } else {
                common_utc = Some(dates.clone());
            }
        }
        let epoch_dates: Vec<_> = dates.iter().map(chrono::DateTime::date_naive).collect();
        if let Some(expected) = &common_dates {
            ensure!(
                expected == &epoch_dates,
                "spatial groups must share complete acquisition epochs"
            );
        } else {
            common_dates = Some(epoch_dates);
        }
    }
    Ok(groups)
}

/// Resolve configured layover/shadow masks onto the active burst groups.
///
/// An empty mask list is the identity path. OPERA inputs require one mask with
/// an unambiguous burst id for every active burst. Inputs without burst ids and
/// non-OPERA inputs support exactly one mask for their single group.
pub(crate) fn resolve_layover_shadow_masks(
    input_type: InputType,
    groups: &BTreeMap<String, Vec<usize>>,
    mask_files: &[PathBuf],
) -> Result<BTreeMap<String, Option<PathBuf>>> {
    if mask_files.is_empty() {
        return Ok(groups.keys().cloned().map(|id| (id, None)).collect());
    }

    let is_single_group = groups.len() == 1 && groups.contains_key("single");
    if input_type != InputType::OperaCslc || is_single_group {
        ensure!(
            groups.len() == 1,
            "layover/shadow masks for non-OPERA inputs require exactly one active group; found {}",
            groups.len()
        );
        ensure!(
            mask_files.len() == 1,
            "layover_shadow_mask_files must contain exactly one file for a single/non-OPERA group; found {}",
            mask_files.len()
        );
        let group = groups
            .keys()
            .next()
            .expect("group count was checked above")
            .clone();
        return Ok(BTreeMap::from([(group, Some(mask_files[0].clone()))]));
    }

    ensure!(
        !groups.is_empty() && !groups.contains_key("single"),
        "OPERA CSLC filenames must contain burst ids when layover_shadow_mask_files is configured"
    );

    let mut resolved = BTreeMap::new();
    for mask in mask_files {
        let ids = opera_burst_ids(mask);
        ensure!(
            ids.len() == 1,
            "layover/shadow mask '{}' must contain exactly one OPERA burst id; found {}",
            mask.display(),
            ids.len()
        );
        let id = &ids[0];
        ensure!(
            groups.contains_key(id),
            "layover/shadow mask '{}' names burst '{}' which is not active",
            mask.display(),
            id
        );
        ensure!(
            resolved.insert(id.clone(), Some(mask.clone())).is_none(),
            "multiple layover/shadow masks were provided for active burst '{id}'"
        );
    }

    for id in groups.keys() {
        ensure!(
            resolved.contains_key(id),
            "no layover/shadow mask was provided for active burst '{id}'"
        );
    }
    Ok(resolved)
}

/// Extract the `T###-######-IW#` burst id from a filename, if present.
fn burst_id(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    name.split('_')
        .find(|t| t.starts_with('T') && t.contains("-IW"))
        .map(str::to_string)
}

fn opera_burst_ids(path: &Path) -> Vec<String> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    let bytes = name.as_bytes();
    const BURST_ID_LEN: usize = 15;
    let mut ids = Vec::new();
    for start in 0..bytes.len().saturating_sub(BURST_ID_LEN - 1) {
        let end = start + BURST_ID_LEN;
        if bytes[start] != b'T'
            || !bytes[start + 1..start + 4].iter().all(u8::is_ascii_digit)
            || bytes[start + 4] != b'-'
            || !bytes[start + 5..start + 11].iter().all(u8::is_ascii_digit)
            || &bytes[start + 11..start + 14] != b"-IW"
            || !bytes[start + 14].is_ascii_digit()
        {
            continue;
        }
        let starts_at_boundary = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let ends_at_boundary = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if starts_at_boundary && ends_at_boundary {
            ids.push(name[start..end].to_string());
        }
    }
    ids
}

/// One burst's grid footprint on the output CRS.
#[derive(Debug, Clone, Copy)]
pub struct BurstGeo {
    /// Georeferencing (EPSG + geotransform) of the burst grid.
    pub geo: GeoInfo,
    /// Burst row count.
    pub rows: usize,
    /// Burst column count.
    pub cols: usize,
}

/// The stitched frame grid covering the union of all burst footprints.
#[derive(Debug, Clone, Copy)]
pub struct FrameGrid {
    /// Frame georeferencing (origin at the upper-left of the union).
    pub geo: GeoInfo,
    /// Frame row count.
    pub rows: usize,
    /// Frame column count.
    pub cols: usize,
}

/// Compute the union frame grid from the burst footprints.
///
/// # Errors
/// Returns `Err` if the bursts disagree on pixel posting or CRS (would need
/// resampling/reprojection, which is deferred).
pub fn frame_grid(bursts: &[BurstGeo]) -> Result<FrameGrid> {
    let first = bursts.first().ok_or_else(|| anyhow::anyhow!("no bursts"))?;
    let dx = first.geo.geotransform[1];
    let dy = first.geo.geotransform[5];
    let epsg = first.geo.epsg;
    for b in bursts {
        ensure!(
            (b.geo.geotransform[1] - dx).abs() < POSTING_TOL
                && (b.geo.geotransform[5] - dy).abs() < POSTING_TOL
                && b.geo.epsg == epsg,
            "bursts differ in posting/CRS; reprojection is not supported in v1"
        );
        let col_offset = (b.geo.geotransform[0] - first.geo.geotransform[0]) / dx;
        let row_offset = (first.geo.geotransform[3] - b.geo.geotransform[3]) / -dy;
        ensure!(
            (col_offset - col_offset.round()).abs() < POSTING_TOL
                && (row_offset - row_offset.round()).abs() < POSTING_TOL,
            "burst origins are not aligned to the common output pixel grid"
        );
    }
    let xmin = reduce(bursts, f64::min, |b| b.geo.geotransform[0]);
    let ymax = reduce(bursts, f64::max, |b| b.geo.geotransform[3]);
    let xmax = reduce(bursts, f64::max, |b| {
        b.geo.geotransform[0] + b.cols as f64 * dx
    });
    let ymin = reduce(bursts, f64::min, |b| {
        b.geo.geotransform[3] + b.rows as f64 * dy
    });
    Ok(FrameGrid {
        geo: GeoInfo {
            epsg,
            geotransform: [xmin, dx, 0.0, ymax, 0.0, dy],
        },
        rows: ((ymax - ymin) / -dy).round() as usize,
        cols: ((xmax - xmin) / dx).round() as usize,
    })
}

/// Upper-left `(row, col)` offset of a burst within the frame grid.
#[must_use]
pub fn burst_offset(frame: &FrameGrid, burst: &BurstGeo) -> (usize, usize) {
    let dx = frame.geo.geotransform[1];
    let dy = frame.geo.geotransform[5];
    let col = ((burst.geo.geotransform[0] - frame.geo.geotransform[0]) / dx).round() as usize;
    let row = ((frame.geo.geotransform[3] - burst.geo.geotransform[3]) / -dy).round() as usize;
    (row, col)
}

/// Paste a burst's 2-D layer onto `frame` at `(row_off, col_off)` (last-on-top).
pub fn paste2<T: Clone>(frame: &mut Array2<T>, burst: &Array2<T>, offset: (usize, usize)) {
    let (ro, co) = offset;
    let (br, bc) = burst.dim();
    frame
        .slice_mut(ndarray::s![ro..ro + br, co..co + bc])
        .assign(burst);
}

/// Paste a burst's 3-D cube (band, row, col) onto `frame` at `(row_off, col_off)`.
pub fn paste3<T: Clone>(frame: &mut Array3<T>, burst: &Array3<T>, offset: (usize, usize)) {
    let (ro, co) = offset;
    let (_, br, bc) = burst.dim();
    frame
        .slice_mut(ndarray::s![.., ro..ro + br, co..co + bc])
        .assign(burst);
}

/// min/max reduction over a burst-derived scalar.
fn reduce(bursts: &[BurstGeo], op: fn(f64, f64) -> f64, key: impl Fn(&BurstGeo) -> f64) -> f64 {
    bursts.iter().map(&key).fold(key(&bursts[0]), op)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opera_groups() -> BTreeMap<String, Vec<usize>> {
        BTreeMap::from([
            ("T064-135518-IW1".to_string(), vec![0, 2]),
            ("T064-135518-IW2".to_string(), vec![1, 3]),
        ])
    }

    fn geo(ox: f64, oy: f64, rows: usize, cols: usize) -> BurstGeo {
        BurstGeo {
            geo: GeoInfo {
                epsg: 32611,
                geotransform: [ox, 30.0, 0.0, oy, 0.0, -30.0],
            },
            rows,
            cols,
        }
    }

    #[test]
    fn groups_by_opera_burst_id() {
        let files = vec![
            PathBuf::from("OPERA_L2_CSLC-S1_T064-135518-IW1_20221119T0_x.h5"),
            PathBuf::from("OPERA_L2_CSLC-S1_T064-135518-IW2_20221119T0_x.h5"),
            PathBuf::from("OPERA_L2_CSLC-S1_T064-135518-IW1_20221201T0_x.h5"),
        ];
        let g = group_by_burst(&files);
        assert_eq!(g.len(), 2);
        assert_eq!(g["T064-135518-IW1"], vec![0, 2]);
        assert_eq!(g["T064-135518-IW2"], vec![1]);
    }

    #[test]
    fn undated_names_collapse_to_single_group() {
        let files = vec![
            PathBuf::from("cslc_20221119.h5"),
            PathBuf::from("cslc_20221201.h5"),
        ];
        let g = group_by_burst(&files);
        assert_eq!(g.len(), 1);
        assert_eq!(g["single"], vec![0, 1]);
    }

    #[test]
    fn empty_mask_list_is_identity_for_every_active_group() {
        let resolved =
            resolve_layover_shadow_masks(InputType::OperaCslc, &opera_groups(), &[]).unwrap();
        assert_eq!(
            resolved,
            BTreeMap::from([
                ("T064-135518-IW1".to_string(), None),
                ("T064-135518-IW2".to_string(), None),
            ])
        );
    }

    #[test]
    fn opera_mask_mapping_is_independent_of_list_order() {
        let groups = opera_groups();
        let first = vec![
            PathBuf::from("masks/T064-135518-IW2_mask.tif"),
            PathBuf::from("masks/T064-135518-IW1_mask.tif"),
        ];
        let second = vec![first[1].clone(), first[0].clone()];
        let first_resolved =
            resolve_layover_shadow_masks(InputType::OperaCslc, &groups, &first).unwrap();
        let second_resolved =
            resolve_layover_shadow_masks(InputType::OperaCslc, &groups, &second).unwrap();
        assert_eq!(first_resolved, second_resolved);
        assert_eq!(
            first_resolved["T064-135518-IW1"],
            Some(PathBuf::from("masks/T064-135518-IW1_mask.tif"))
        );
        assert_eq!(
            first_resolved["T064-135518-IW2"],
            Some(PathBuf::from("masks/T064-135518-IW2_mask.tif"))
        );
    }

    #[test]
    fn single_group_accepts_one_mask_without_a_burst_id() {
        let groups = BTreeMap::from([("single".to_string(), vec![0, 1])]);
        let mask = PathBuf::from("layover-shadow.tif");
        let resolved = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &groups,
            std::slice::from_ref(&mask),
        )
        .unwrap();
        assert_eq!(resolved["single"], Some(mask));
    }

    #[test]
    fn non_opera_group_accepts_one_mask_without_a_burst_id() {
        let groups = BTreeMap::from([("nisar".to_string(), vec![0, 1])]);
        let mask = PathBuf::from("layover-shadow.tif");
        let resolved = resolve_layover_shadow_masks(
            InputType::NisarGslc,
            &groups,
            std::slice::from_ref(&mask),
        )
        .unwrap();
        assert_eq!(resolved["nisar"], Some(mask));
    }

    #[test]
    fn rejects_missing_opera_mask() {
        let error = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &opera_groups(),
            &[PathBuf::from("T064-135518-IW1_mask.tif")],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("no layover/shadow mask was provided for active burst 'T064-135518-IW2'"));
    }

    #[test]
    fn rejects_duplicate_opera_mask() {
        let error = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &opera_groups(),
            &[
                PathBuf::from("a_T064-135518-IW1_mask.tif"),
                PathBuf::from("b_T064-135518-IW1_mask.tif"),
            ],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("multiple layover/shadow masks were provided for active burst"));
    }

    #[test]
    fn rejects_extra_opera_mask() {
        let error = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &opera_groups(),
            &[
                PathBuf::from("T064-135518-IW1_mask.tif"),
                PathBuf::from("T064-135518-IW2_mask.tif"),
                PathBuf::from("T064-135518-IW3_mask.tif"),
            ],
        )
        .unwrap_err();
        assert!(error.to_string().contains("which is not active"));
    }

    #[test]
    fn rejects_unparseable_opera_mask() {
        let error = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &opera_groups(),
            &[PathBuf::from("mask_without_burst_id.tif")],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("must contain exactly one OPERA burst id; found 0"));
    }

    #[test]
    fn rejects_mask_with_multiple_opera_burst_ids() {
        let error = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &opera_groups(),
            &[PathBuf::from("T064-135518-IW1_T064-135518-IW2_mask.tif")],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("must contain exactly one OPERA burst id; found 2"));
    }

    #[test]
    fn rejects_mixed_parseable_and_unparseable_opera_groups() {
        let groups = BTreeMap::from([
            ("T064-135518-IW1".to_string(), vec![0]),
            ("single".to_string(), vec![1]),
        ]);
        let error = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &groups,
            &[PathBuf::from("T064-135518-IW1_mask.tif")],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("OPERA CSLC filenames must contain burst ids"));
    }

    #[test]
    fn rejects_multiple_masks_for_single_group() {
        let groups = BTreeMap::from([("single".to_string(), vec![0, 1])]);
        let error = resolve_layover_shadow_masks(
            InputType::OperaCslc,
            &groups,
            &[PathBuf::from("a.tif"), PathBuf::from("b.tif")],
        )
        .unwrap_err();
        assert!(error.to_string().contains("exactly one file"));
    }

    #[test]
    fn rejects_multiple_active_non_opera_groups() {
        let groups = BTreeMap::from([("a".to_string(), vec![0]), ("b".to_string(), vec![1])]);
        let error = resolve_layover_shadow_masks(
            InputType::NisarGslc,
            &groups,
            &[PathBuf::from("mask.tif")],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("require exactly one active group"));
    }

    #[test]
    fn frame_grid_unions_two_adjacent_bursts() {
        // burst B sits to the right of and below A, overlapping by 5 px.
        let a = geo(1000.0, 2000.0, 20, 30); // x: 1000..1900, y: 1400..2000
        let b = geo(1000.0 + 25.0 * 30.0, 2000.0 - 15.0 * 30.0, 20, 30); // shifted (25,15) px
        let frame = frame_grid(&[a, b]).unwrap();
        // union spans cols 0..(25+30)=55, rows 0..(15+20)=35
        assert_eq!(frame.cols, 55);
        assert_eq!(frame.rows, 35);
        assert_eq!(burst_offset(&frame, &a), (0, 0));
        assert_eq!(burst_offset(&frame, &b), (15, 25));
    }

    #[test]
    fn rejects_mismatched_posting() {
        let a = geo(0.0, 0.0, 10, 10);
        let mut b = geo(300.0, 0.0, 10, 10);
        b.geo.geotransform[1] = 20.0; // different dx
        assert!(frame_grid(&[a, b]).is_err());
    }

    #[test]
    fn rejects_subpixel_misaligned_burst_origin() {
        let a = geo(0.0, 0.0, 10, 10);
        let b = geo(315.0, 0.0, 10, 10);
        let error = frame_grid(&[a, b]).unwrap_err();
        assert!(error.to_string().contains("origins are not aligned"));
    }

    #[test]
    fn paste_places_burst_block() {
        let mut frame = Array2::<f64>::zeros((4, 4));
        let burst = Array2::from_shape_fn((2, 2), |(i, j)| (i * 2 + j + 1) as f64);
        paste2(&mut frame, &burst, (1, 1));
        assert_eq!(frame[(1, 1)], 1.0);
        assert_eq!(frame[(2, 2)], 4.0);
        assert_eq!(frame[(0, 0)], 0.0);
    }
}
