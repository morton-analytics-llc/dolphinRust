//! Real-data gate for issue #21: how large is the solid earth tide on a frame we
//! actually process, and does it leak into velocity?
//!
//! Runs the tide model over the three-station GNSS fixture's real acquisition
//! times and real per-pixel LOS geometry, and reports (a) the absolute LOS tide,
//! (b) the differential relative to acquisition 0 — which is what
//! `subtract_delay` applies — (c) the spatial gradient across the frame, and
//! (d) the rate the differential would contribute to a linear velocity fit if it
//! were left in. Skips (passes as a no-op) when the local real fixtures are
//! absent, mirroring the other real-data gates.

use std::path::Path;

use chrono::NaiveDateTime;
use dolphin_corrections::geometry::resolve_los_geometry;
use dolphin_corrections::solid_earth_tide::{tide_range_delay_grid, LonLatGrid};
use dolphin_io::{grid_corner_lonlat, read_geotransform, read_los_layers};

const FIXTURE: &str = "../../validation/real_data/gps_mmx1/cropped/mmx1_icmx_mxtx_common";
const STATIC_GRANULES: [&str; 2] = [
    "static/OPERA_L2_CSLC-S1-STATIC_T005-008704-IW1_20140403_S1A_v1.0.h5",
    "static/OPERA_L2_CSLC-S1-STATIC_T005-008705-IW1_20140403_S1A_v1.0.h5",
];

#[test]
fn solid_earth_tide_on_the_real_gnss_frame() {
    let root = Path::new(FIXTURE);
    let cslc_dir = root.join("cslc");
    if !cslc_dir.exists() || STATIC_GRANULES.iter().any(|g| !root.join(g).exists()) {
        eprintln!("gps_mmx1 three-station fixtures absent; skipping #21 real-data gate");
        return;
    }

    let mut granules: Vec<_> = std::fs::read_dir(&cslc_dir)
        .expect("read cslc dir")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "h5"))
        .collect();
    granules.sort();
    let times: Vec<NaiveDateTime> = granules.iter().filter_map(|p| acq_utc(p)).collect();
    assert_eq!(
        times.len(),
        granules.len(),
        "every fixture granule must carry a timestamp"
    );

    let frame = &granules[0];
    let geo = read_geotransform(frame, "/data/VV").expect("read frame geotransform");
    let shape = dolphin_io::read_cslc(frame, "/data/VV")
        .expect("read frame grid")
        .dim();
    let layers: Vec<_> = STATIC_GRANULES
        .iter()
        .map(|g| read_los_layers(&root.join(g), "/data").expect("read LOS"))
        .collect();
    let los = resolve_los_geometry(&layers, geo.geotransform, geo.epsg, shape)
        .expect("resolve LOS geometry");
    let corners = grid_corner_lonlat(geo.geotransform, shape.0, shape.1, geo.epsg)
        .expect("frame corner lon/lat");
    let lonlat = LonLatGrid::from_corners(corners, shape.0, shape.1);
    eprintln!(
        "frame: epsg={} shape={}x{} corners(lon,lat)={corners:?}",
        geo.epsg, shape.0, shape.1
    );

    // Frame-mean LOS tide per date, and the frame's own spatial spread.
    let mut means = Vec::new();
    let mut max_gradient_mm = 0.0_f64;
    for (t, &utc) in times.iter().enumerate() {
        let grid = tide_range_delay_grid(utc, &lonlat, &los);
        let mean = grid.iter().sum::<f64>() / grid.len() as f64;
        let (lo, hi) = grid
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        max_gradient_mm = max_gradient_mm.max((hi - lo) * 1000.0);
        eprintln!(
            "  {utc}: LOS tide mean {:+.2} mm, across-frame spread {:.3} mm",
            mean * 1000.0,
            (hi - lo) * 1000.0
        );
        means.push(mean);
        assert!(
            mean.abs() < 0.4,
            "date {t}: LOS tide {mean} m is outside the physical envelope"
        );
    }

    // The differential relative to acquisition 0 is what is actually applied.
    let differentials: Vec<f64> = means.iter().map(|m| m - means[0]).collect();
    let peak_differential_mm = differentials
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs() * 1000.0));
    eprintln!("peak differential vs acquisition 0: {peak_differential_mm:.2} mm");
    eprintln!("largest across-frame spread on any date: {max_gradient_mm:.3} mm");

    // What the uncorrected differential contributes to a linear rate.
    let days: Vec<f64> = times
        .iter()
        .map(|t| (*t - times[0]).num_seconds() as f64 / 86_400.0)
        .collect();
    let rate_mm_per_yr = linear_rate(&days, &differentials) * 1000.0;
    eprintln!("velocity leak if left uncorrected: {rate_mm_per_yr:+.3} mm/yr");

    assert!(
        peak_differential_mm > 1.0,
        "differential tide {peak_differential_mm} mm is below the InSAR error budget — \
         this frame does not motivate the correction"
    );
}

/// Slope of an unweighted degree-1 fit, per year.
fn linear_rate(days: &[f64], values: &[f64]) -> f64 {
    let n = days.len() as f64;
    let (sx, sy) = (days.iter().sum::<f64>(), values.iter().sum::<f64>());
    let sxx: f64 = days.iter().map(|x| x * x).sum();
    let sxy: f64 = days.iter().zip(values).map(|(x, y)| x * y).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx) * 365.25
}

fn acq_utc(path: &Path) -> Option<NaiveDateTime> {
    let name = path.file_name()?.to_str()?;
    let chars: Vec<char> = name.chars().collect();
    chars.windows(15).find_map(|w| {
        let token: String = w.iter().collect();
        NaiveDateTime::parse_from_str(&token, "%Y%m%dT%H%M%S").ok()
    })
}
