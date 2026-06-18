//! Gated real-data sign test (matches the IONEX / NISAR real-data gates).
//!
//! Gated on `SIGN_REF_PROD_IFG` pointing at a run directory holding `config_rust.yaml`
//! (a real OPERA CSLC stack) and `work_oracle/timeseries/<ref>_<sec>.tif` (a full
//! production `dolphin run`). Skips cleanly when unset. Runs the **fixed** dolphinRust
//! pipeline on the real stack and confirms its displacement *sign* matches the
//! production displacement on the deforming F38502/Corcoran subsidence bowl.
//!
//! Receipt (recorded in VALIDATION.md): on the longest-baseline pair the demeaned,
//! coherence-gated correlation between dolphinRust and dolphin-production displacement
//! is **≈ +0.99 after the fix** (was **≈ −0.97 before**, when `unwrap_pair` formed
//! `sec·conj(ref)`). The reverse order is an exact pixelwise negation, so a negative
//! correlation here is the v1.0–v1.2 inverted-sign regression.
//!
//! Reproduce:
//! ```sh
//! source validation/creds.sh
//! oracle/.venv/bin/python validation/fetch_real.py --burst T144_308015_IW2 --n 15 \
//!     --start 2016-07-01 --end 2017-02-01            # F38502/Corcoran bowl
//! oracle/.venv/bin/python validation/crop_real.py --size 1024 --out /tmp/cv_cropped
//! validation/run_real.sh <rundir>                    # full dolphin run -> work_oracle/
//! SIGN_REF_PROD_IFG=<rundir> cargo test -p dolphin-workflows --test sign_real_data -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_io::read_raster;
use dolphin_workflows::run_displacement;
use ndarray::Array2;

const COH_GATE: f64 = 0.7;

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// The longest-baseline production displacement raster `YYYYMMDD_YYYYMMDD.tif`
/// under `work_oracle/timeseries/` (the strongest-signal date, lexicographically
/// last; `velocity.tif` and `conncomp_*` are excluded by the date-pair pattern).
fn production_last_date(timeseries: &Path) -> Option<PathBuf> {
    let is_pair = |stem: &str| {
        stem.split_once('_').is_some_and(|(a, b)| {
            a.len() == 8 && b.len() == 8 && a.chars().chain(b.chars()).all(|c| c.is_ascii_digit())
        })
    };
    std::fs::read_dir(timeseries)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tif"))
        .filter(|p| p.file_stem().and_then(|s| s.to_str()).is_some_and(is_pair))
        .max()
}

/// Demeaned Pearson correlation over the masked pixels.
fn masked_corr(a: &Array2<f64>, b: &Array2<f64>, keep: &Array2<bool>) -> f64 {
    let pick = |x: &Array2<f64>| -> Vec<f64> {
        x.iter()
            .zip(keep.iter())
            .filter_map(|(&v, &k)| k.then_some(v))
            .collect()
    };
    let (av, bv) = (pick(a), pick(b));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb) = (mean(&av), mean(&bv));
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&x, &y) in av.iter().zip(bv.iter()) {
        num += (x - ma) * (y - mb);
        da += (x - ma).powi(2);
        db += (y - mb).powi(2);
    }
    num / (da.sqrt() * db.sqrt())
}

#[test]
fn rust_displacement_sign_matches_production_on_corcoran_bowl() {
    let Ok(rundir) = std::env::var("SIGN_REF_PROD_IFG") else {
        eprintln!("skipping real-data sign test: set SIGN_REF_PROD_IFG to a run dir");
        return;
    };
    if !snaphu_available() {
        eprintln!("skipping real-data sign test: snaphu not on PATH");
        return;
    }
    let rundir = PathBuf::from(rundir);
    let prod = production_last_date(&rundir.join("work_oracle/timeseries"))
        .expect("a production YYYYMMDD_YYYYMMDD.tif under work_oracle/timeseries");
    eprintln!("production reference: {}", prod.display());

    // Run the fixed pipeline on the real stack; isolate outputs from work_rust.
    let mut cfg = DisplacementWorkflow::from_yaml(
        &std::fs::read_to_string(rundir.join("config_rust.yaml")).unwrap(),
    )
    .unwrap();
    cfg.work_directory = std::env::temp_dir().join("dolphin_sign_real");
    let out = run_displacement(&cfg).unwrap();

    // dolphin's timeseries rasters are meters with an identity geotransform
    // (top-origin); dolphinRust writes north-up COGs (dy<0), so a vertical flip
    // reconciles row order before the per-pixel comparison. Orientation is
    // orthogonal to the per-pixel sign under test.
    let prod_m = read_raster::<f32>(&prod).unwrap().data.mapv(f64::from);
    let last = out.displacement.dim().0 - 1;
    let rust_m = out
        .displacement
        .index_axis(ndarray::Axis(0), last)
        .to_owned();
    let rust_flipped: Array2<f64> =
        Array2::from_shape_fn(rust_m.dim(), |(r, c)| rust_m[(rust_m.dim().0 - 1 - r, c)]);

    assert_eq!(prod_m.dim(), rust_flipped.dim(), "grid dims match");
    let coh = &out.temporal_coherence;
    let keep = Array2::from_shape_fn(prod_m.dim(), |(r, c)| {
        let cr = coh.dim().0 - 1 - r; // coherence shares the rust orientation
        prod_m[(r, c)].is_finite() && rust_flipped[(r, c)].is_finite() && coh[(cr, c)] > COH_GATE
    });
    let n = keep.iter().filter(|&&k| k).count();
    assert!(n > 10_000, "enough coherent pixels to correlate (got {n})");

    let corr = masked_corr(&prod_m, &rust_flipped, &keep);
    eprintln!("F38502/Corcoran sign correlation (rust vs production, coh>{COH_GATE}): {corr:+.4} on {n} px");
    assert!(
        corr > 0.5,
        "dolphinRust displacement sign must match dolphin production (corr {corr:+.4}); \
         a negative correlation is the v1.0–v1.2 sec·conj(ref) inverted-sign regression \
         (pre-fix corr ≈ −0.97)"
    );
}
