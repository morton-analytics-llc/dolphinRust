//! Always-on sign-convention regression guard.
//!
//! The interferogram is formed `ref · conj(sec)` (`displacement.rs::unwrap_pair`,
//! dolphin production `interferogram.py`). The opposite order `sec · conj(ref)`
//! globally inverts the LOS displacement *and* velocity sign — the regression that
//! silently shipped in v1.0.0–v1.2.0 (the sign-sensitive oracle carried the same
//! inversion, so the contracts could not see it). This guard pins the convention
//! against an analytically-known sign, with **no network and no oracle fixture**.
//!
//! Construction: a noise-free single-burst stack carrying a positive, monotonic,
//! cycle-free LOS phase ramp `+0.3·t·(col/cols)` — range that *increases* in time
//! away from a zero-phase reference column. Under `ref · conj(sec)` the recovered
//! displacement at the far column is **positive**; under the reverted order it is
//! negative. Reverting `unwrap_pair` to `pl[j]·conj(pl[i])` flips this assertion
//! red (verified by hand: flip the order locally → test fails → flip back).

use std::path::{Path, PathBuf};

use dolphin_core::config::DisplacementWorkflow;
use dolphin_core::types::{HalfWindow, Strides};
use dolphin_core::Cf32;
use dolphin_workflows::run_displacement;
use ndarray::Array2;

const ROWS: usize = 24;
const COLS: usize = 40;
const N: usize = 5;
const SENTINEL1_WAVELENGTH_M: f64 = 0.055_465_76;
const REF_ROW: usize = ROWS / 2;

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

/// Positive, monotonic, cycle-free LOS ramp: range grows with acquisition index
/// and with column, anchored at zero on column 0 (the reference column).
fn sample(t: usize, col: usize) -> Cf32 {
    let phase = 0.3 * t as f64 * (col as f64 / COLS as f64);
    Cf32::from_polar(1.0, phase as f32)
}

/// Write the synthetic OPERA-CSLC stack as `/data/VV` complex grids in dated
/// `cslc_YYYYMMDD.h5` files (12-day cadence, the names dolphin's date parser reads).
fn write_stack(dir: &Path) -> Vec<PathBuf> {
    let base = chrono::NaiveDate::from_ymd_opt(2022, 11, 19).unwrap();
    (0..N)
        .map(|t| {
            let stamp = (base + chrono::Duration::days(t as i64 * 12)).format("%Y%m%d");
            let path = dir.join(format!("cslc_{stamp}.h5"));
            let grid = Array2::from_shape_fn((ROWS, COLS), |(_, j)| sample(t, j));
            let file = hdf5::File::create(&path).unwrap();
            let group = file.create_group("data").unwrap();
            let ds = group
                .new_dataset::<Cf32>()
                .shape((ROWS, COLS))
                .create("VV")
                .unwrap();
            ds.write(&grid).unwrap();
            let x = (0..COLS)
                .map(|col| 500_015.0 + col as f64 * 30.0)
                .collect::<Vec<_>>();
            let y = (0..ROWS)
                .map(|row| 4_200_015.0 - row as f64 * 30.0)
                .collect::<Vec<_>>();
            group
                .new_dataset_builder()
                .with_data(&x)
                .create("x_coordinates")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&y)
                .create("y_coordinates")
                .unwrap();
            group
                .new_dataset::<i64>()
                .create("projection")
                .unwrap()
                .write_scalar(&32611_i64)
                .unwrap();
            path
        })
        .collect()
}

#[test]
fn displacement_sign_matches_ref_conj_sec_convention() {
    if !snaphu_available() {
        eprintln!("skipping sign guard: snaphu not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("dolphin_sign_guard");
    std::fs::create_dir_all(&dir).unwrap();
    let files = write_stack(&dir);

    let mut cfg = DisplacementWorkflow {
        cslc_file_list: files,
        work_directory: dir.clone(),
        ..Default::default()
    };
    cfg.input_options.subdataset = Some("/data/VV".to_string());
    cfg.input_options.wavelength = Some(SENTINEL1_WAVELENGTH_M);
    cfg.phase_linking.ministack_size = 15;
    cfg.phase_linking.half_window = HalfWindow { y: 2, x: 2 };
    cfg.output_options.strides = Strides { y: 1, x: 1 };
    cfg.interferogram_network.reference_idx = Some(0);
    // Reference the series to the zero-phase column so the sign at the far column
    // survives spatial referencing (a uniform offset would otherwise demean away).
    cfg.timeseries_options.reference_point = Some((REF_ROW, 0));

    let out = run_displacement(&cfg).unwrap();

    // displacement bands are dates 1..N; the last band carries the largest ramp.
    let last = out.displacement.dim().0 - 1;
    let far = out.displacement[(last, REF_ROW, COLS - 1)];
    let near = out.displacement[(last, REF_ROW, 0)];
    eprintln!("sign guard: disp[far]={far:+.6e} m, disp[ref-col]={near:+.6e} m");

    // Injected range *increases* away from the reference column; under the
    // production `ref · conj(sec)` order the recovered displacement is positive
    // there. The reverted `sec · conj(ref)` order makes this negative.
    assert!(
        far > 1e-4,
        "displacement at the far column must be positive under ref·conj(sec) \
         (got {far:+.6e} m); a negative value means unwrap_pair was reverted to \
         sec·conj(ref) — the v1.0–v1.2 inverted-sign regression"
    );
    // Monotonic in time: the sign holds across every date, not just the last.
    for t in 0..=last {
        let v = out.displacement[(t, REF_ROW, COLS - 1)];
        assert!(
            v > 0.0,
            "displacement[{t}] far column sign must be positive, got {v:+.6e}"
        );
    }
}
