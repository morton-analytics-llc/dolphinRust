//! Real-data contracts for the Level-1 front end (issue #230).
//!
//! These download raw Sentinel-1 data, so they are `#[ignore]`d and run on
//! demand:
//!
//! ```text
//! DOLPHIN_L1_ENV=../../.env DOLPHIN_L1_CACHE=/path/to/cache \
//!   DOLPHIN_L1_REFERENCE=/path/to/validation/real_data \
//!   DOLPHIN_L1_BURST_DB=/path/to/opera-burst-bbox-only.sqlite3 \
//!   cargo test -p dolphin-l1 --test real_data -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use chrono::{DateTime, Duration, Utc};
use dolphin_l1::{BurstId, BurstQuery, CdseClient};

fn setup() -> PathBuf {
    if let Ok(env) = std::env::var("DOLPHIN_L1_ENV") {
        dolphin_l1::load_env_file(std::path::Path::new(&env)).unwrap();
    }
    PathBuf::from(std::env::var("DOLPHIN_L1_CACHE").expect("set DOLPHIN_L1_CACHE"))
}

fn t100_query() -> BurstQuery {
    let start: DateTime<Utc> = "2023-01-10T00:00:00Z".parse().unwrap();
    BurstQuery {
        burst_id: BurstId::parse("T100-213486-IW2").unwrap(),
        polarization: "VV".into(),
        start,
        end: start + Duration::days(1),
    }
}

fn measurement_tiff(safe: &std::path::Path) -> PathBuf {
    std::fs::read_dir(safe.join("measurement"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "tiff"))
        .unwrap()
}

/// Phase 1 contract: the CDSE burst for T100-213486-IW2 on 2023-01-10 is the
/// burst OPERA processed (same parent SAFE and first-line time), the POEORB
/// found is the one OPERA used and covers the burst.
#[test]
#[ignore = "downloads raw Sentinel-1 data from CDSE"]
fn phase1_cdse_burst_and_orbit() {
    let cache = setup();
    let query = t100_query();
    let mut cdse = CdseClient::from_env().unwrap();
    let records = cdse.search_bursts(&query).unwrap();
    assert_eq!(records.len(), 1, "{records:?}");
    let record = &records[0];
    assert_eq!(
        record.parent_product_name,
        "S1A_IW_SLC__1SDV_20230110T133437_20230110T133504_046722_0599DA_95B4.SAFE"
    );
    assert_eq!(
        record.azimuth_time.to_rfc3339(),
        "2023-01-10T13:34:58.375289+00:00"
    );

    let safe = cdse.download_burst(record, &cache.join("bursts")).unwrap();
    for sub in ["annotation", "measurement"] {
        assert!(safe.join(sub).is_dir(), "{} has no {sub}/", safe.display());
    }
    let start = record.azimuth_time - Duration::seconds(60);
    let orbit = cdse
        .search_orbit(record.mission(), start, start + Duration::seconds(130))
        .unwrap();
    assert!(orbit.is_precise());
    assert_eq!(
        orbit.name,
        "S1A_OPER_AUX_POEORB_OPOD_20230130T081807_V20230109T225942_20230111T005942.EOF"
    );
    let eof = cdse.download_orbit(&orbit, &cache.join("orbits")).unwrap();
    assert!(std::fs::read_to_string(&eof).unwrap().contains("<OSV>"));

    let shape = gdal::Dataset::open(measurement_tiff(&safe))
        .unwrap()
        .raster_size();
    assert_eq!(shape, (25997, 1513));
    println!(
        "phase1: CDSE {} | orbit {} | shape {:?}",
        record.name, orbit.name, shape
    );
}

fn only_safe(dir: PathBuf) -> PathBuf {
    let burst_dir = std::fs::read_dir(dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::read_dir(burst_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "SAFE"))
        .unwrap()
}

/// Phase 2 contract: the burst reader recovers the values OPERA's s1-reader
/// recorded for the same burst (T100-213486-IW2, 2023-01-10; OPERA CSLC
/// `input_burst_metadata` / `burst_location_parameters`), and the CInt16
/// measurement decodes to the full burst.
#[test]
#[ignore = "reads the raw bursts downloaded by phase1"]
fn phase2_safe_reader_matches_opera_burst_metadata() {
    use dolphin_l1::safe::S1Burst;
    let cache = setup();
    let cdse = S1Burst::open(&only_safe(cache.join("bursts")), "VV", None).unwrap();
    let close =
        |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    assert_eq!((cdse.lines, cdse.samples), (1513, 25997));
    assert_eq!(
        (
            cdse.first_valid_line,
            cdse.last_valid_line,
            cdse.first_valid_sample,
            cdse.last_valid_sample
        ),
        (26, 1489, 706, 25147)
    );
    assert_eq!(
        cdse.sensing_start.to_rfc3339(),
        "2023-01-10T13:34:58.375289+00:00"
    );
    assert_eq!(cdse.ipf_version, "003.52");
    assert_eq!(cdse.rank, 8);
    close(cdse.starting_range(), 845_776.713_492_821_3, 1e-12);
    close(cdse.range_pixel_spacing(), 2.329_562_114_715_323, 1e-12);
    close(cdse.azimuth_time_interval, 0.002_055_556_299_999_998, 1e-12);
    close(cdse.prf_raw, 1_451.627_112_193_99, 1e-12);
    close(cdse.range_chirp_rate, 779_281_727_512.048_1, 1e-12);
    close(cdse.azimuth_steering_rate, 0.017_101_841_240_789_263, 1e-12);
    close(cdse.iw2_mid_range, 876_057.526_640_948_5, 1e-12);
    close(cdse.wavelength(), 0.055_465_76, 1e-7);
    close(cdse.range_bandwidth, 48.3e6, 1e-12);
    let doppler = cdse.doppler().unwrap();
    for (got, want) in doppler
        .coefficients
        .iter()
        .zip([9.295423, -1.010618e3, -1.161456e7])
    {
        close(*got, want, 1e-7);
    }
    let fm = cdse.fm_rate().unwrap();
    for (got, want) in
        fm.coefficients
            .iter()
            .zip([-2.196_644_10e3, 4.019_450_77e5, -6.534_465_54e7])
    {
        close(*got, want, 1e-8);
    }

    let cdse_samples = cdse.read_all().unwrap();
    assert_eq!(cdse_samples.dim(), (1513, 25997));
    let nonzero = cdse_samples.iter().filter(|v| v.norm_sqr() > 0.0).count();
    assert!(
        nonzero > 1513 * 25997 / 2,
        "only {nonzero} non-zero samples"
    );
    println!(
        "phase2: {:?} burst, {nonzero} non-zero samples",
        cdse_samples.dim()
    );
}

/// Reference data root (`validation/real_data` of a dolphinRust checkout,
/// or a copy of it) from `DOLPHIN_L1_REFERENCE`.
fn reference_dir() -> PathBuf {
    PathBuf::from(std::env::var("DOLPHIN_L1_REFERENCE").expect("set DOLPHIN_L1_REFERENCE"))
}

fn t100_static() -> PathBuf {
    reference_dir().join(
        "t100_213486_iw2_p105_smel/cropped/shared/static/\
         OPERA_L2_CSLC-S1-STATIC_T100-213486-IW2_20140403_S1A_v1.0.h5",
    )
}

/// Phase 3 contract on T100-213486-IW2 2023-01-10 with its POEORB:
/// - geo2rdr of ESA's annotation tie points lands on their annotated azimuth
///   and range times (orbit interpolation + zero-Doppler solve);
/// - GLO-30 + EGM2008 ellipsoidal heights match the GNSS heights of P105 and
///   SMEL to within monument height plus DEM error (the geoid alone is ~17 m
///   here, so a missing or wrong-sign geoid fails);
/// - ground-to-satellite LOS east/north on OPERA's static-layer grid matches
///   OPERA's own `los_east` / `los_north`.
#[test]
#[ignore = "reads phase1 downloads and GLO-30/EGM2008 over HTTP"]
fn phase3_geometry_dem_and_los_match_references() {
    use dolphin_l1::dem::{Dem, LonLatBounds};
    use dolphin_l1::geometry::{geo2rdr, llh_to_ecef};
    use dolphin_l1::orbit::Orbit;
    use dolphin_l1::safe::{S1Burst, SPEED_OF_LIGHT};

    let cache = setup();
    let burst = S1Burst::open(&only_safe(cache.join("bursts")), "VV", None).unwrap();
    let eof = std::fs::read_dir(cache.join("orbits"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let orbit = Orbit::from_eof(
        &eof,
        burst.sensing_start,
        burst.sensing_start,
        burst.sensing_stop(),
        Duration::seconds(60),
    )
    .unwrap();

    let grid = burst.geolocation_grid().unwrap();
    let (mut worst_az, mut worst_rg) = (0.0_f64, 0.0_f64);
    for point in &grid {
        let target = llh_to_ecef(
            point.latitude.to_radians(),
            point.longitude.to_radians(),
            point.height,
        );
        let annotated = orbit.seconds(point.azimuth_time);
        let radar = geo2rdr(target, &orbit, annotated).unwrap();
        worst_az = worst_az.max((radar.azimuth_time - annotated).abs());
        worst_rg =
            worst_rg.max((radar.slant_range - point.slant_range_time * SPEED_OF_LIGHT / 2.0).abs());
    }
    println!(
        "phase3: {} tie points, max |Δaz| {worst_az:.3e} s, max |Δrange| {worst_rg:.3} m",
        grid.len()
    );
    assert!(worst_az < 2.0e-5, "azimuth {worst_az}");
    assert!(worst_rg < 0.05, "range {worst_rg}");

    let bounds =
        LonLatBounds::around(grid.iter().map(|p| (p.longitude, p.latitude)), 0.05).unwrap();
    let dem = Dem::glo30(bounds).unwrap();
    for (station, lon, lat, gnss_height) in [
        ("P105", -112.504_087_744_9, 39.387_544_054_9, 1_431.482_52),
        ("SMEL", -112.844_926_651_0, 39.425_644_204_6, 1_422.144_77),
    ] {
        let dem_height = dem.height(lon, lat).unwrap();
        println!("phase3: {station} GNSS {gnss_height:.2} m, DEM {dem_height:.2} m");
        assert!((dem_height - gnss_height).abs() < 4.0, "{station}");
    }

    let t_mid = orbit.seconds(burst.sensing_mid());
    let (worst, compared) = los_difference_vs_opera(&dem, &orbit, t_mid);
    println!("phase3: LOS vs OPERA static over {compared} pixels, max |Δ| {worst:.2e}");
    assert!(compared > 1000);
    assert!(worst < 2.0e-4, "LOS {worst}");
}

/// Max |Δ| of LOS east/north between our geometry and OPERA's static layers on
/// a sparse sample of the T100 static grid, and the number of pixels compared.
fn los_difference_vs_opera(
    dem: &dolphin_l1::dem::Dem,
    orbit: &dolphin_l1::orbit::Orbit,
    t_mid: f64,
) -> (f64, usize) {
    use dolphin_l1::geometry::{geo2rdr, line_of_sight_enu, llh_to_ecef};
    use dolphin_l1::grid::Projection;
    let file = hdf5::File::open(t100_static()).unwrap();
    let x: Vec<f64> = file
        .dataset("data/x_coordinates")
        .unwrap()
        .read_raw()
        .unwrap();
    let y: Vec<f64> = file
        .dataset("data/y_coordinates")
        .unwrap()
        .read_raw()
        .unwrap();
    let east = file
        .dataset("data/los_east")
        .unwrap()
        .read_2d::<f32>()
        .unwrap();
    let north = file
        .dataset("data/los_north")
        .unwrap()
        .read_2d::<f32>()
        .unwrap();
    let projection = Projection::new(32612).unwrap();
    let mut worst = 0.0_f64;
    let mut compared = 0;
    for (r, &yy) in y.iter().enumerate().step_by(25) {
        let cols: Vec<usize> = (0..x.len()).step_by(97).collect();
        let mut lon: Vec<f64> = cols.iter().map(|&c| x[c]).collect();
        let mut lat = vec![yy; cols.len()];
        projection.to_lonlat(&mut lon, &mut lat).unwrap();
        for (k, &c) in cols.iter().enumerate() {
            if !east[[r, c]].is_finite() {
                continue;
            }
            let h = dem.height(lon[k], lat[k]).unwrap();
            let (la, lo) = (lat[k].to_radians(), lon[k].to_radians());
            let target = llh_to_ecef(la, lo, h);
            let radar = geo2rdr(target, orbit, t_mid).unwrap();
            let los = line_of_sight_enu(target, la, lo, radar.platform.position);
            let d = (los[0] - f64::from(east[[r, c]]))
                .abs()
                .max((los[1] - f64::from(north[[r, c]])).abs());
            worst = worst.max(d);
            compared += 1;
        }
    }
    (worst, compared)
}

/// Azimuth Doppler centroid (Hz) of `data[rows, cols]` from the lag-one
/// correlation along azimuth.
fn lag_one_doppler(
    data: ndarray::ArrayView2<num_complex::Complex32>,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    line_interval: f64,
) -> f64 {
    let mut sum = num_complex::Complex64::new(0.0, 0.0);
    for r in rows {
        for c in cols.clone() {
            let (a, b) = (data[[r + 1, c]], data[[r, c]]);
            sum += num_complex::Complex64::new(f64::from(a.re), f64::from(a.im))
                * num_complex::Complex64::new(f64::from(b.re), -f64::from(b.im));
        }
    }
    sum.arg() / (2.0 * std::f64::consts::PI * line_interval)
}

/// Phase 4 contract (TOPS deramp) on T100-213486-IW2 2023-01-10: the raw
/// burst's local azimuth Doppler sweeps through the burst (TOPS steering),
/// and after deramping every 64-line block sits within 15 Hz of the annotated
/// geometric Doppler centroid. A wrong carrier sign or rate leaves a sweep.
#[test]
#[ignore = "reads the raw burst downloaded by phase1"]
fn phase4_deramp_centres_azimuth_spectrum() {
    use dolphin_l1::geocode::DerampedBurst;
    use dolphin_l1::orbit::Orbit;
    use dolphin_l1::safe::S1Burst;
    use dolphin_l1::tops::TopsCarrier;

    let cache = setup();
    let burst = S1Burst::open(&only_safe(cache.join("bursts")), "VV", None).unwrap();
    let eof = std::fs::read_dir(cache.join("orbits"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let margin = Duration::seconds(60);
    let orbit = Orbit::from_eof(
        &eof,
        burst.sensing_start,
        burst.sensing_start,
        burst.sensing_stop(),
        margin,
    )
    .unwrap();
    let carrier = TopsCarrier::new(&burst, &orbit).unwrap();
    let deramped = DerampedBurst::read(&burst, &carrier).unwrap();
    let raw = burst
        .read_window(
            deramped.row0,
            deramped.col0,
            deramped.samples.nrows(),
            deramped.samples.ncols(),
        )
        .unwrap();
    let cols = 12_000 - deramped.col0..12_512 - deramped.col0;
    let range = burst.starting_range() + 12_256.0 * burst.range_pixel_spacing();
    let geometric = burst.doppler().unwrap().eval_range(range);
    let dt = burst.azimuth_time_interval;
    let (mut worst, mut raw_min, mut raw_max) = (0.0_f64, f64::INFINITY, f64::NEG_INFINITY);
    for start in (0..deramped.samples.nrows() - 65).step_by(64) {
        let rows = start..start + 64;
        let f = lag_one_doppler(deramped.samples.view(), rows.clone(), cols.clone(), dt);
        let f_raw = lag_one_doppler(raw.view(), rows, cols.clone(), dt);
        worst = worst.max((f - geometric).abs());
        raw_min = raw_min.min(f_raw);
        raw_max = raw_max.max(f_raw);
    }
    println!(
        "phase4: deramped |f - f_geo| <= {worst:.1} Hz (f_geo {geometric:.1} Hz); raw sweeps {raw_min:.0}..{raw_max:.0} Hz"
    );
    assert!(raw_max - raw_min > 200.0, "raw burst shows no TOPS sweep");
    assert!(worst < 15.0, "deramped Doppler off by {worst} Hz");
}

/// Houston contract (T034-071054-IW3, the OPERA crop window): two dates
/// built from raw CDSE bursts give the same interferogram as OPERA's crop
/// of the same dates. Over pixels where OPERA's 11×11 coherence is ≥ 0.9,
/// the circular std of `arg(I_ours · conj(I_opera))` is ≤ 0.1 rad (the
/// receipt in `validation/results/issue230_raw_front_end/` measures
/// ≤ 0.018 rad over 14 pairs).
#[test]
#[ignore = "downloads two raw Houston bursts from CDSE and their GIMs"]
fn houston_iw3_interferogram_matches_opera() {
    use dolphin_l1::grid::GeoGrid;
    use dolphin_l1::stack::{process_stack, StackRequest};
    use num_complex::Complex64;

    let cache = setup();
    let crop = reference_dir().join("t034_071054_iw3_houston_crop/cslc");
    let burst_db =
        PathBuf::from(std::env::var("DOLPHIN_L1_BURST_DB").expect("set DOLPHIN_L1_BURST_DB"));
    let burst_id = BurstId::parse("T034-071054-IW3").unwrap();
    let out = cache.join("houston_contract");
    let _ = std::fs::remove_dir_all(&out);
    let start: DateTime<Utc> = "2022-01-23T00:00:00Z".parse().unwrap();
    let request = StackRequest {
        burst_id,
        polarization: "VV".into(),
        start,
        end: start + Duration::days(13),
        grid: GeoGrid::from_burst_db(&burst_db, &burst_id)
            .unwrap()
            .window(2704, 1266, 699, 3996)
            .unwrap(),
        cache_dir: cache.clone(),
        output_dir: out,
        keep_raw: false,
        phase_layers: false,
        ionosphere: true,
    };
    let mut cdse = CdseClient::from_env().unwrap();
    let outputs = process_stack(&mut cdse, &request).unwrap();
    assert_eq!(outputs.len(), 2);
    let read = |path: &std::path::Path| {
        hdf5::File::open(path)
            .unwrap()
            .dataset("data/VV")
            .unwrap()
            .read_2d::<num_complex::Complex32>()
            .unwrap()
    };
    let opera_of = |day: &str| {
        std::fs::read_dir(&crop)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains(&format!("IW3_{day}T"))
            })
            .unwrap()
    };
    let (o1, o2) = (read(&outputs[0].path), read(&outputs[1].path));
    let (r1, r2) = (read(&opera_of("20220123")), read(&opera_of("20220204")));
    let (rows, cols) = o1.dim();
    let c64 = |v: num_complex::Complex32| Complex64::new(f64::from(v.re), f64::from(v.im));
    let mut resultant = Complex64::new(0.0, 0.0);
    let mut compared = 0usize;
    for r in (5..rows - 5).step_by(3) {
        for c in (5..cols - 5).step_by(3) {
            let (mut num, mut p1, mut p2) = (Complex64::new(0.0, 0.0), 0.0, 0.0);
            for i in r - 5..=r + 5 {
                for j in c - 5..=c + 5 {
                    let (a, b) = (c64(r1[[i, j]]), c64(r2[[i, j]]));
                    num += a * b.conj();
                    p1 += a.norm_sqr();
                    p2 += b.norm_sqr();
                }
            }
            let coherence = num.norm() / (p1 * p2).sqrt();
            let ours = c64(o1[[r, c]]) * c64(o2[[r, c]]).conj();
            let opera = c64(r1[[r, c]]) * c64(r2[[r, c]]).conj();
            let delta = ours * opera.conj();
            if coherence >= 0.9 && delta.norm() > 0.0 && delta.norm().is_finite() {
                resultant += delta / delta.norm();
                compared += 1;
            }
        }
    }
    let circular_std = (-2.0 * (resultant.norm() / compared as f64).ln()).sqrt();
    println!(
        "houston: {compared} coherent pixels, interferogram circular std {circular_std:.4} rad"
    );
    assert!(compared > 1000, "only {compared} coherent pixels");
    assert!(circular_std <= 0.1, "circular std {circular_std}");
}
