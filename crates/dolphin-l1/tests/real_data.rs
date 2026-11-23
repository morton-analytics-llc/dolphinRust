//! Real-data contracts for the Level-1 front end (issue #230).
//!
//! These download raw Sentinel-1 data, so they are `#[ignore]`d and run on
//! demand:
//!
//! ```text
//! DOLPHIN_L1_ENV=../../.env DOLPHIN_L1_CACHE=/path/to/cache \
//!   cargo test -p dolphin-l1 --test real_data -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use chrono::{DateTime, Duration, Utc};
use dolphin_l1::asf::AsfBurstClient;
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
/// found is the one OPERA used and covers the burst, and the ASF fallback burst
/// decodes to the same samples.
#[test]
#[ignore = "downloads raw Sentinel-1 data from CDSE and ASF"]
fn phase1_cdse_burst_orbit_and_asf_fallback() {
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

    let asf = AsfBurstClient::from_env().unwrap();
    let asf_bursts = asf.search(&query).unwrap();
    assert_eq!(asf_bursts.len(), 1);
    let asf_safe = asf
        .download(&asf_bursts[0], &cache.join("asf_bursts"))
        .unwrap();
    let shape = |p: PathBuf| {
        let ds = gdal::Dataset::open(p).unwrap();
        (ds.raster_size(), ds.raster_count())
    };
    let cdse_shape = shape(measurement_tiff(&safe));
    assert_eq!(
        cdse_shape,
        shape(measurement_tiff(&asf_safe)),
        "burst shapes differ"
    );
    println!(
        "phase1: CDSE {} | ASF {} | orbit {} | shape {:?}",
        record.name, asf_bursts[0].granule, orbit.name, cdse_shape
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
/// `input_burst_metadata` / `burst_location_parameters`), and the CDSE and ASF
/// bursts decode to identical complex samples.
#[test]
#[ignore = "reads the raw bursts downloaded by phase1"]
fn phase2_safe_reader_matches_opera_burst_metadata() {
    use dolphin_l1::safe::S1Burst;
    let cache = setup();
    let cdse = S1Burst::open(&only_safe(cache.join("bursts")), "VV", None).unwrap();
    let asf = S1Burst::open(&only_safe(cache.join("asf_bursts")), "VV", None).unwrap();
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
    assert_eq!(asf.doppler().unwrap().coefficients, doppler.coefficients);
    assert_eq!(asf.fm_rate().unwrap().coefficients, fm.coefficients);

    let cdse_samples = cdse.read_all().unwrap();
    let asf_samples = asf.read_all().unwrap();
    assert_eq!(cdse_samples.dim(), (1513, 25997));
    assert!(cdse_samples == asf_samples, "CDSE and ASF bursts differ");
    let nonzero = cdse_samples.iter().filter(|v| v.norm_sqr() > 0.0).count();
    assert!(
        nonzero > 1513 * 25997 / 2,
        "only {nonzero} non-zero samples"
    );
    println!(
        "phase2: CDSE == ASF over {:?}; {nonzero} non-zero samples",
        cdse_samples.dim()
    );
}
