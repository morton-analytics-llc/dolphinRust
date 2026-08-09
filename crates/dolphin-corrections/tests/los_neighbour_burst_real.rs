//! Real-data gate for issue #39: the along-track neighbour STATIC the LOS mosaic
//! needs.
//!
//! The three-station GNSS frame (`mmx1_icmx_mxtx_common`, burst `T005_008704_IW1`)
//! extends past burst 008704's valid LOS, so resolving geometry from 008704 alone is
//! a coverage error; adding the along-track neighbour 008705 — whose burst is by
//! definition not in the CSLC stack — completes it. This gate proves both halves on
//! the real granules and measures the overlap agreement the relaxed provenance rule
//! now rests on. Skips (passes as a no-op) when the local real fixtures are absent,
//! mirroring the other real-data gates.

use std::path::Path;

use dolphin_corrections::geometry::resolve_los_geometry;
use dolphin_corrections::troposphere::{warp_to_frame, DelayGrid};
use dolphin_corrections::CorrectionError;
use dolphin_io::{read_geotransform, read_los_layers, LosLayers};
use ndarray::{Array2, Zip};

const FIXTURE: &str = "../../validation/real_data/gps_mmx1/cropped/mmx1_icmx_mxtx_common";
const FRAME_CSLC: &str = "cslc/OPERA_L2_CSLC-S1_T005-008704-IW1_\
20230104T004053Z_20240805T201701Z_S1A_VV_v1.1.h5";
const STATIC_OWN: &str = "static/OPERA_L2_CSLC-S1-STATIC_T005-008704-IW1_20140403_S1A_v1.0.h5";
const STATIC_NEIGHBOUR: &str =
    "static/OPERA_L2_CSLC-S1-STATIC_T005-008705-IW1_20140403_S1A_v1.0.h5";

#[test]
fn neighbour_burst_static_completes_coverage_and_agrees_in_overlap() {
    let root = Path::new(FIXTURE);
    let (frame, own, neighbour) = (
        root.join(FRAME_CSLC),
        root.join(STATIC_OWN),
        root.join(STATIC_NEIGHBOUR),
    );
    if !frame.exists() || !own.exists() || !neighbour.exists() {
        eprintln!("gps_mmx1 three-station fixtures absent; skipping #39 real-data gate");
        return;
    }

    let geo = read_geotransform(&frame, "/data/VV").expect("read frame geotransform");
    let shape = dolphin_io::read_cslc(&frame, "/data/VV")
        .expect("read frame grid")
        .dim();
    eprintln!(
        "frame: epsg={} shape={}x{} gt={:?}",
        geo.epsg, shape.0, shape.1, geo.geotransform
    );

    let own_layers = read_los_layers(&own, "/data").expect("read 008704 LOS");
    let neighbour_layers = read_los_layers(&neighbour, "/data").expect("read 008705 LOS");

    // Half 1: the processed burst alone does not cover the frame — this is the
    // shortfall that made the STATIC identity rule bite.
    let err = resolve_los_geometry(
        std::slice::from_ref(&own_layers),
        geo.geotransform,
        geo.epsg,
        shape,
    )
    .expect_err("008704 alone unexpectedly covered the whole frame");
    assert!(
        matches!(err, CorrectionError::GeometryCoverage(_)),
        "expected a coverage error, got {err}"
    );
    eprintln!("008704 alone: {err}");

    // Half 2: adding the along-track neighbour completes coverage and passes the
    // overlap-agreement gate that first-covered-burst-wins now rests on.
    let los = resolve_los_geometry(
        &[own_layers.clone(), neighbour_layers.clone()],
        geo.geotransform,
        geo.epsg,
        shape,
    )
    .expect("008704 + 008705 failed to resolve");
    let stats = los.incidence_stats().expect("incidence stats");
    eprintln!(
        "008704 + 008705: incidence mean={:.4}° std={:.4}° min={:.4}° max={:.4}°",
        stats.mean_deg, stats.std_deg, stats.min_deg, stats.max_deg
    );

    // Independent measurement of the overlap agreement (the gate inside
    // resolve_los_geometry is derived separately, so this is a cross-check, not a
    // restatement): both granules warped onto the frame, compared where both are valid.
    let mut diffs = overlap_angles_deg(&own_layers, &neighbour_layers, geo, shape);
    assert!(
        !diffs.is_empty(),
        "the two bursts do not overlap on this frame — the agreement claim is untested"
    );
    diffs.sort_by(f64::total_cmp);
    let (median, p99, max) = (
        diffs[diffs.len() / 2],
        diffs[diffs.len() * 99 / 100],
        diffs[diffs.len() - 1],
    );
    eprintln!(
        "overlap: {} px, LOS difference median={median:.4}° p99={p99:.4}° max={max:.4}°",
        diffs.len()
    );
    assert!(
        median <= 1.0,
        "overlapping same-track bursts disagree by a median {median:.4}° — \
         first-covered-burst-wins would be arbitrary"
    );

    // The correction this fixes: the scalar fallback projects zenith→slant with
    // 1/cos(37°), the resolved per-pixel geometry with 1/up.
    let scalar_slant = 1.0 / 37.0_f64.to_radians().cos();
    let mean_slant = 1.0 / stats.mean_deg.to_radians().cos();
    eprintln!(
        "zenith→slant: scalar 37° = {scalar_slant:.4}, resolved mean {:.4}° = {mean_slant:.4} \
         ({:.1}% error avoided)",
        stats.mean_deg,
        100.0 * (scalar_slant - mean_slant) / mean_slant
    );
}

/// Angle in degrees between the two bursts' LOS at every frame pixel where both
/// warp to valid (non-`(0,0)`, finite) data.
fn overlap_angles_deg(
    a: &LosLayers,
    b: &LosLayers,
    geo: dolphin_io::GeoInfo,
    shape: (usize, usize),
) -> Vec<f64> {
    let warp = |data: &Array2<f64>, src: dolphin_io::GeoInfo| {
        let grid = DelayGrid {
            data: data.clone(),
            geotransform: src.geotransform,
            epsg: Some(src.epsg),
            srs_wkt: None,
        };
        warp_to_frame(&grid, geo.geotransform, geo.epsg, shape).expect("warp LOS component")
    };
    let (ae, an) = (warp(&a.east, a.geo), warp(&a.north, a.geo));
    let (be, bn) = (warp(&b.east, b.geo), warp(&b.north, b.geo));

    let mut diffs = Vec::new();
    Zip::from(&ae)
        .and(&an)
        .and(&be)
        .and(&bn)
        .for_each(|&aev, &anv, &bev, &bnv| {
            let valid = [aev, anv, bev, bnv].iter().all(|v| v.is_finite())
                && (aev != 0.0 || anv != 0.0)
                && (bev != 0.0 || bnv != 0.0);
            if valid {
                diffs.push(angle_between_deg([aev, anv], [bev, bnv]));
            }
        });
    diffs
}

fn angle_between_deg(a: [f64; 2], b: [f64; 2]) -> f64 {
    let up = |[e, n]: [f64; 2]| (1.0_f64 - e * e - n * n).max(0.0).sqrt();
    let dot = a[0] * b[0] + a[1] * b[1] + up(a) * up(b);
    dot.clamp(-1.0, 1.0).acos().to_degrees()
}
