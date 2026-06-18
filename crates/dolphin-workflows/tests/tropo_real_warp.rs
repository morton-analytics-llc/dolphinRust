//! Best-effort real-data gate for the tropospheric 4326→UTM warp (Phase 1).
//!
//! Warps the real global EPSG:4326 `OPERA_L4_TROPO-ZENITH_V1` granule onto a real
//! UTM CSLC frame and records the applied-correction magnitude. Skips (passes as a
//! no-op) when the local real fixtures are absent, mirroring the other real-data
//! gates — never a stub, never a faked number.

use std::path::Path;

use dolphin_corrections::troposphere::{read_l4_total, warp_to_frame};
use dolphin_io::read_geotransform;

const L4_REAL: &str = "../../validation/real_data/tropo/opera_l4_tropo.nc";
const FRAME_CSLC: &str = "../../validation/real_data/cropped_mexico/\
OPERA_L2_CSLC-S1_T005-008704-IW1_20230410T004052Z_20240806T201045Z_S1A_VV_v1.1.h5";

#[test]
fn real_l4_warps_onto_real_utm_frame() {
    let l4 = Path::new(L4_REAL);
    let frame = Path::new(FRAME_CSLC);
    if !l4.exists() || !frame.exists() {
        eprintln!("real fixtures absent; skipping real tropo warp gate");
        return;
    }

    // Real UTM frame geocoding from the CSLC grid.
    let geo = read_geotransform(frame, "/data/VV").expect("read CSLC geotransform");
    let cf32 = dolphin_io::read_cslc(frame, "/data/VV").expect("read CSLC grid");
    let (rows, cols) = cf32.dim();
    eprintln!(
        "frame: epsg={} shape={rows}x{cols} gt={:?}",
        geo.epsg, geo.geotransform
    );

    // Real global 4326 zenith total delay (hydrostatic + wet), warped onto the frame.
    let grid = read_l4_total(l4).expect("read real L4 total");
    eprintln!(
        "L4 src epsg={:?} wkt_present={}",
        grid.epsg,
        grid.srs_wkt.is_some()
    );
    let warped = warp_to_frame(&grid, geo.geotransform, geo.epsg, (rows, cols))
        .expect("warp real L4 onto UTM frame");

    // Applied-correction magnitude over the frame. Sentinel-1 IW incidence ~39°.
    let valid: Vec<f64> = warped
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.1 && *v < 6.0)
        .collect();
    assert!(
        valid.len() > (rows * cols) / 2,
        "most warped pixels should be physical ZTD; got {}/{}",
        valid.len(),
        rows * cols
    );
    let mean_ztd = valid.iter().sum::<f64>() / valid.len() as f64;
    let (mn, mx) = valid
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let slant = 1.0 / (39.0_f64).to_radians().cos();
    eprintln!(
        "APPLIED TROPO: zenith mean={mean_ztd:.4} m (min={mn:.4}, max={mx:.4}); \
         slant(39°)≈{:.4} m over {rows}x{cols} frame",
        mean_ztd * slant
    );
    assert!(
        (1.5..3.5).contains(&mean_ztd),
        "mean zenith ZTD {mean_ztd} m should be meters-scale"
    );
}
