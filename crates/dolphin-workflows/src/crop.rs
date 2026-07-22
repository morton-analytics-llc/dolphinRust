//! AOI target/analysis/read-window planning for bounded displacement runs.

use dolphin_core::config::{DisplacementWorkflow, UnwrapMethod};
use dolphin_core::BlockIndices;
use dolphin_io::transform_bounds;
use serde::{Deserialize, Serialize};

use crate::burst::{burst_offset, frame_grid, BurstGeo, FrameGrid};

/// Versioned finite-halo policy. Changing any dependency-cone rule requires a
/// new value because bounded products are AOI-local processing versions.
pub const HALO_POLICY_VERSION: &str = "dolphinrust-aoi-halo/1";
/// Explicit processing method for products whose phase linking and unwrap were
/// evaluated only on a bounded dependency domain.
pub const AOI_PROCESSING_METHOD: &str = "bounded_aoi_local";
/// Version of the bounded processing/provenance contract.
pub const AOI_PROCESSING_VERSION: &str = "1.0.0";

const MIN_TARGET_AXIS_PIXELS: usize = 2;
const MIN_MULTIBURST_OVERLAP_PIXELS: usize = 4;
const TOPHU_OVERLAP_PIXELS: usize = 16;

/// Typed failures for requested bounded processing.
#[derive(Debug, thiserror::Error)]
pub enum BoundsError {
    /// Bounds or CRS were invalid.
    #[error("invalid output bounds: {0}")]
    Invalid(String),
    /// Requested target does not intersect the source frame.
    #[error("requested output bounds do not intersect the source frame")]
    NoIntersection,
    /// Snapped target is too small for a scientifically meaningful product.
    #[error("bounded target is too small: {rows}x{cols} pixels; minimum is 2x2")]
    TooSmall {
        /// Target rows after outward snapping.
        rows: usize,
        /// Target columns after outward snapping.
        cols: usize,
    },
    /// Burst grids cannot form a common frame.
    #[error("inconsistent bounded burst grids: {0}")]
    InconsistentGrid(String),
    /// The expanded domain is not completely covered by source bursts.
    #[error("bounded analysis domain has {uncovered} uncovered output pixels")]
    IncompleteCoverage {
        /// Count of output pixels with no source burst.
        uncovered: usize,
    },
    /// A multi-burst crop removed the overlap needed to reconcile its seam.
    #[error(
        "bounded multi-burst analysis retains only {overlap_pixels} overlap pixels; at least 4 are required"
    )]
    InsufficientOverlap {
        /// Pixels covered by at least two included bursts.
        overlap_pixels: usize,
    },
}

/// Requested and actual geometry written into `geometry_provenance.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessingBoundsProvenance {
    /// Explicitly distinguishes AOI-local processing from a full-frame crop.
    pub processing_method: String,
    /// Version of the AOI-local processing contract.
    pub processing_method_version: String,
    /// Bounds exactly as requested in the workflow config.
    pub requested_target_bounds: [f64; 4],
    /// EPSG of `requested_target_bounds`.
    pub requested_bounds_epsg: u32,
    /// Outward-snapped target bounds on the output grid.
    pub actual_output_bounds: [f64; 4],
    /// Expanded analysis/read envelope on the output grid.
    pub actual_analysis_bounds: [f64; 4],
    /// Union of source read windows expressed on the output grid. Under halo
    /// policy v1 it equals the fully covered analysis envelope.
    pub actual_read_bounds: [f64; 4],
    /// EPSG shared by actual output and analysis bounds.
    pub output_epsg: u32,
    /// Target upper-left `[row, col]` offset in the full frame output grid.
    pub target_pixel_offset: [usize; 2],
    /// Analysis upper-left `[row, col]` offset in the full frame output grid.
    pub analysis_pixel_offset: [usize; 2],
    /// Analysis halo `[rows, cols]` in output pixels.
    pub analysis_halo_pixels: [usize; 2],
    /// Versioned dependency-cone policy.
    pub halo_policy_version: String,
    /// Actual native-resolution source windows read for each included burst.
    pub native_reads: Vec<NativeReadProvenance>,
}

/// One included burst's actual native-resolution read receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeReadProvenance {
    /// Stable zero-based burst position in the planner input.
    pub burst_index: usize,
    /// Native `[row_start, col_start, row_stop, col_stop]` window.
    pub pixel_window: [usize; 4],
    /// Native window bounds `[left, bottom, right, top]` in `output_epsg`.
    pub bounds: [f64; 4],
}

/// One source burst's bounded full-resolution read window.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BurstWindow {
    pub(crate) source: BlockIndices,
}

/// Plan shared by input reads, analysis-domain processing, final trim, and
/// provenance emission.
#[derive(Debug, Clone)]
pub(crate) struct BoundedPlan {
    pub(crate) windows: Vec<Option<BurstWindow>>,
    pub(crate) target_in_analysis: BlockIndices,
    pub(crate) provenance: ProcessingBoundsProvenance,
}

/// Return `None` for the legacy full-frame path, else build a fail-closed plan.
pub(crate) fn plan_bounds(
    cfg: &DisplacementWorkflow,
    bursts: &[BurstGeo],
    acquisitions_per_burst: usize,
) -> Result<Option<BoundedPlan>, BoundsError> {
    let Some(requested) = cfg.output_options.bounds else {
        return Ok(None);
    };
    let requested_epsg = cfg
        .output_options
        .bounds_epsg
        .ok_or_else(|| BoundsError::Invalid("bounds_epsg is required when bounds is set".into()))?;
    let frame =
        frame_grid(bursts).map_err(|error| BoundsError::InconsistentGrid(error.to_string()))?;
    validate_frame(&frame)?;
    if cfg
        .output_options
        .epsg
        .is_some_and(|epsg| epsg != frame.geo.epsg)
    {
        return Err(BoundsError::Invalid(
            "output reprojection is not supported; output EPSG must match the source frame".into(),
        ));
    }
    let transformed = transform_bounds(requested, requested_epsg, frame.geo.epsg)
        .map_err(|error| BoundsError::Invalid(error.to_string()))?;
    let target = snapped_block(transformed, &frame)?;
    if target.height() < MIN_TARGET_AXIS_PIXELS || target.width() < MIN_TARGET_AXIS_PIXELS {
        return Err(BoundsError::TooSmall {
            rows: target.height(),
            cols: target.width(),
        });
    }
    let halo = analysis_halo(cfg, acquisitions_per_burst);
    let analysis = expand(target, halo, (frame.rows, frame.cols));
    let windows = burst_windows(bursts, &frame, analysis, cfg.output_options.strides);
    validate_coverage(bursts, &frame, analysis, &windows)?;
    let target_in_analysis = BlockIndices {
        row_start: target.row_start - analysis.row_start,
        row_stop: target.row_stop - analysis.row_start,
        col_start: target.col_start - analysis.col_start,
        col_stop: target.col_stop - analysis.col_start,
    };
    let native_reads = windows
        .iter()
        .enumerate()
        .filter_map(|(burst_index, window)| {
            window.map(|window| NativeReadProvenance {
                burst_index,
                pixel_window: [
                    window.source.row_start,
                    window.source.col_start,
                    window.source.row_stop,
                    window.source.col_stop,
                ],
                bounds: native_window_bounds(
                    &bursts[burst_index],
                    window.source,
                    cfg.output_options.strides,
                ),
            })
        })
        .collect();
    Ok(Some(BoundedPlan {
        windows,
        target_in_analysis,
        provenance: ProcessingBoundsProvenance {
            processing_method: AOI_PROCESSING_METHOD.into(),
            processing_method_version: AOI_PROCESSING_VERSION.into(),
            requested_target_bounds: tuple_array(requested),
            requested_bounds_epsg: requested_epsg,
            actual_output_bounds: tuple_array(block_bounds(&frame, target)),
            actual_analysis_bounds: tuple_array(block_bounds(&frame, analysis)),
            actual_read_bounds: tuple_array(block_bounds(&frame, analysis)),
            output_epsg: frame.geo.epsg,
            target_pixel_offset: [target.row_start, target.col_start],
            analysis_pixel_offset: [analysis.row_start, analysis.col_start],
            analysis_halo_pixels: [halo.0, halo.1],
            halo_policy_version: HALO_POLICY_VERSION.into(),
            native_reads,
        },
    }))
}

fn native_window_bounds(
    burst: &BurstGeo,
    block: BlockIndices,
    strides: dolphin_core::Strides,
) -> [f64; 4] {
    let gt = burst.geo.geotransform;
    let dx = gt[1] / strides.x as f64;
    let dy = gt[5] / strides.y as f64;
    [
        gt[0] + block.col_start as f64 * dx,
        gt[3] + block.row_stop as f64 * dy,
        gt[0] + block.col_stop as f64 * dx,
        gt[3] + block.row_start as f64 * dy,
    ]
}

fn validate_frame(frame: &FrameGrid) -> Result<(), BoundsError> {
    let gt = frame.geo.geotransform;
    if frame.geo.epsg == 0 || gt[1] <= 0.0 || gt[5] >= 0.0 || gt[2] != 0.0 || gt[4] != 0.0 {
        return Err(BoundsError::Invalid(
            "source frame needs a nonzero EPSG and north-up, non-rotated posting".into(),
        ));
    }
    Ok(())
}

fn snapped_block(
    bounds: (f64, f64, f64, f64),
    frame: &FrameGrid,
) -> Result<BlockIndices, BoundsError> {
    let (left, bottom, right, top) = bounds;
    let gt = frame.geo.geotransform;
    let raw = BlockIndices {
        row_start: ((gt[3] - top) / -gt[5]).floor().max(0.0) as usize,
        row_stop: ((gt[3] - bottom) / -gt[5]).ceil().max(0.0) as usize,
        col_start: ((left - gt[0]) / gt[1]).floor().max(0.0) as usize,
        col_stop: ((right - gt[0]) / gt[1]).ceil().max(0.0) as usize,
    };
    let clamped = BlockIndices {
        row_start: raw.row_start.min(frame.rows),
        row_stop: raw.row_stop.min(frame.rows),
        col_start: raw.col_start.min(frame.cols),
        col_stop: raw.col_stop.min(frame.cols),
    };
    if clamped.row_start >= clamped.row_stop || clamped.col_start >= clamped.col_stop {
        return Err(BoundsError::NoIntersection);
    }
    Ok(clamped)
}

fn analysis_halo(cfg: &DisplacementWorkflow, acquisitions: usize) -> (usize, usize) {
    let strides = cfg.output_options.strides;
    let half = cfg.phase_linking.half_window;
    let depth = acquisitions.div_ceil(cfg.phase_linking.ministack_size.max(1));
    let dependency =
        |h: usize, stride: usize| (h + depth.saturating_sub(1) * (h + stride)).div_ceil(stride);
    let unwrap = match cfg.unwrap_options.unwrap_method {
        UnwrapMethod::Tophu => (TOPHU_OVERLAP_PIXELS, TOPHU_OVERLAP_PIXELS),
        _ => cfg.unwrap_options.snaphu_options.tile_overlap,
    };
    let preprocess = if cfg.unwrap_options.run_interpolation {
        cfg.unwrap_options.preprocess_options.max_radius
    } else {
        0
    };
    (
        dependency(half.y, strides.y) + unwrap.0 + preprocess,
        dependency(half.x, strides.x) + unwrap.1 + preprocess,
    )
}

fn expand(block: BlockIndices, halo: (usize, usize), shape: (usize, usize)) -> BlockIndices {
    BlockIndices {
        row_start: block.row_start.saturating_sub(halo.0),
        row_stop: (block.row_stop + halo.0).min(shape.0),
        col_start: block.col_start.saturating_sub(halo.1),
        col_stop: (block.col_stop + halo.1).min(shape.1),
    }
}

fn burst_windows(
    bursts: &[BurstGeo],
    frame: &FrameGrid,
    analysis: BlockIndices,
    strides: dolphin_core::Strides,
) -> Vec<Option<BurstWindow>> {
    bursts
        .iter()
        .map(|burst| {
            let (row, col) = burst_offset(frame, burst);
            let footprint = BlockIndices {
                row_start: row,
                row_stop: row + burst.rows,
                col_start: col,
                col_stop: col + burst.cols,
            };
            let intersection = intersect(analysis, footprint)?;
            let local = BlockIndices {
                row_start: intersection.row_start - row,
                row_stop: intersection.row_stop - row,
                col_start: intersection.col_start - col,
                col_stop: intersection.col_stop - col,
            };
            Some(BurstWindow {
                source: BlockIndices {
                    row_start: local.row_start * strides.y,
                    row_stop: local.row_stop * strides.y,
                    col_start: local.col_start * strides.x,
                    col_stop: local.col_stop * strides.x,
                },
            })
        })
        .collect()
}

fn validate_coverage(
    bursts: &[BurstGeo],
    frame: &FrameGrid,
    analysis: BlockIndices,
    windows: &[Option<BurstWindow>],
) -> Result<(), BoundsError> {
    let mut coverage = vec![0_u8; analysis.height() * analysis.width()];
    let mut included = 0_usize;
    for (burst, window) in bursts.iter().zip(windows) {
        if window.is_none() {
            continue;
        }
        included += 1;
        let (row, col) = burst_offset(frame, burst);
        let footprint = BlockIndices {
            row_start: row,
            row_stop: row + burst.rows,
            col_start: col,
            col_stop: col + burst.cols,
        };
        let Some(overlap) = intersect(analysis, footprint) else {
            continue;
        };
        for r in overlap.row_start..overlap.row_stop {
            for c in overlap.col_start..overlap.col_stop {
                let index = (r - analysis.row_start) * analysis.width() + c - analysis.col_start;
                coverage[index] = coverage[index].saturating_add(1);
            }
        }
    }
    let uncovered = coverage.iter().filter(|&&count| count == 0).count();
    if uncovered > 0 {
        return Err(BoundsError::IncompleteCoverage { uncovered });
    }
    let overlap_pixels = coverage.iter().filter(|&&count| count > 1).count();
    if included > 1 && overlap_pixels < MIN_MULTIBURST_OVERLAP_PIXELS {
        return Err(BoundsError::InsufficientOverlap { overlap_pixels });
    }
    Ok(())
}

fn intersect(a: BlockIndices, b: BlockIndices) -> Option<BlockIndices> {
    let block = BlockIndices {
        row_start: a.row_start.max(b.row_start),
        row_stop: a.row_stop.min(b.row_stop),
        col_start: a.col_start.max(b.col_start),
        col_stop: a.col_stop.min(b.col_stop),
    };
    (block.row_start < block.row_stop && block.col_start < block.col_stop).then_some(block)
}

fn block_bounds(frame: &FrameGrid, block: BlockIndices) -> (f64, f64, f64, f64) {
    let gt = frame.geo.geotransform;
    (
        gt[0] + block.col_start as f64 * gt[1],
        gt[3] + block.row_stop as f64 * gt[5],
        gt[0] + block.col_stop as f64 * gt[1],
        gt[3] + block.row_start as f64 * gt[5],
    )
}

fn tuple_array(bounds: (f64, f64, f64, f64)) -> [f64; 4] {
    [bounds.0, bounds.1, bounds.2, bounds.3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use dolphin_core::{HalfWindow, Strides};
    use dolphin_io::GeoInfo;

    fn burst(x: f64, source_cols: usize, strides: Strides) -> BurstGeo {
        BurstGeo {
            geo: GeoInfo {
                epsg: 32611,
                geotransform: [
                    x,
                    30.0 * strides.x as f64,
                    0.0,
                    2_000.0,
                    0.0,
                    -30.0 * strides.y as f64,
                ],
            },
            rows: 60 / strides.y,
            cols: source_cols / strides.x,
        }
    }

    fn config(strides: Strides) -> DisplacementWorkflow {
        let mut cfg = DisplacementWorkflow::default();
        cfg.output_options.strides = strides;
        cfg.output_options.bounds_epsg = Some(32611);
        cfg.phase_linking.half_window = HalfWindow { y: 2, x: 4 };
        cfg.phase_linking.ministack_size = 5;
        cfg
    }

    #[test]
    fn single_burst_snaps_outward_and_expands_at_1x2() {
        let mut cfg = config(Strides { y: 1, x: 2 });
        cfg.output_options.bounds = Some((1_125.0, 710.0, 1_595.0, 1_615.0));
        let plan = plan_bounds(&cfg, &[burst(1_000.0, 40, cfg.output_options.strides)], 7)
            .unwrap()
            .unwrap();
        assert_eq!(plan.target_in_analysis.height(), 31);
        assert_eq!(plan.target_in_analysis.width(), 8);
        assert!(plan.windows[0].unwrap().source.width() < 40);
        assert_eq!(plan.provenance.halo_policy_version, HALO_POLICY_VERSION);
        assert_eq!(plan.provenance.processing_method, AOI_PROCESSING_METHOD);
        assert_eq!(
            plan.provenance.processing_method_version,
            AOI_PROCESSING_VERSION
        );
        assert_eq!(plan.provenance.native_reads.len(), 1);
        assert_eq!(
            plan.provenance.native_reads[0].pixel_window,
            [
                plan.windows[0].unwrap().source.row_start,
                plan.windows[0].unwrap().source.col_start,
                plan.windows[0].unwrap().source.row_stop,
                plan.windows[0].unwrap().source.col_stop,
            ]
        );
    }

    #[test]
    fn multiburst_crop_keeps_seam_at_3x6() {
        let mut cfg = config(Strides { y: 3, x: 6 });
        let bursts = [
            burst(1_000.0, 60, cfg.output_options.strides),
            burst(2_440.0, 60, cfg.output_options.strides),
        ];
        let frame = frame_grid(&bursts).unwrap();
        cfg.output_options.bounds = Some((2_200.0, 500.0, 2_900.0, 1_700.0));
        let plan = plan_bounds(&cfg, &bursts, 9).unwrap().unwrap();
        assert!(plan.windows.iter().all(Option::is_some));
        assert_eq!(plan.provenance.output_epsg, frame.geo.epsg);
    }

    #[test]
    fn no_intersection_and_too_small_are_typed() {
        let mut cfg = config(Strides { y: 1, x: 2 });
        cfg.output_options.bounds = Some((5_000.0, 5_000.0, 6_000.0, 6_000.0));
        assert!(matches!(
            plan_bounds(&cfg, &[burst(1_000.0, 40, cfg.output_options.strides)], 5),
            Err(BoundsError::NoIntersection)
        ));
        cfg.output_options.bounds = Some((1_001.0, 1_971.0, 1_029.0, 1_999.0));
        assert!(matches!(
            plan_bounds(&cfg, &[burst(1_000.0, 40, cfg.output_options.strides)], 5),
            Err(BoundsError::TooSmall { .. })
        ));
    }

    #[test]
    fn invalid_crs_is_a_typed_bounds_error() {
        let mut cfg = config(Strides { y: 1, x: 2 });
        cfg.output_options.bounds = Some((-123.0, 37.0, -122.0, 38.0));
        cfg.output_options.bounds_epsg = Some(999_999);
        assert!(matches!(
            plan_bounds(&cfg, &[burst(1_000.0, 40, cfg.output_options.strides)], 5),
            Err(BoundsError::Invalid(_))
        ));
    }

    #[test]
    fn multi_burst_without_overlap_fails_explicitly() {
        let mut cfg = config(Strides { y: 1, x: 2 });
        let bursts = [
            burst(1_000.0, 40, cfg.output_options.strides),
            burst(2_200.0, 40, cfg.output_options.strides),
        ];
        cfg.output_options.bounds = Some((1_900.0, 500.0, 2_500.0, 1_700.0));
        assert!(matches!(
            plan_bounds(&cfg, &bursts, 5),
            Err(BoundsError::InsufficientOverlap { .. })
        ));
    }

    #[test]
    fn processing_bounds_provenance_round_trips() {
        let mut cfg = config(Strides { y: 1, x: 2 });
        cfg.output_options.bounds = Some((1_120.0, 800.0, 1_600.0, 1_600.0));
        let provenance = plan_bounds(&cfg, &[burst(1_000.0, 40, cfg.output_options.strides)], 7)
            .unwrap()
            .unwrap()
            .provenance;
        let json = serde_json::to_string(&provenance).unwrap();
        let parsed: ProcessingBoundsProvenance = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, provenance);
    }
}
