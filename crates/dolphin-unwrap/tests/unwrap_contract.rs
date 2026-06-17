//! Phase-9 (unwrapping) contract test.
//!
//! The Rust SNAPHU wrapper must reproduce a direct SNAPHU invocation: same
//! wrapped ifg + correlation in, same unwrapped phase + connected components
//! out. Skips when fixtures or the `snaphu` binary are absent.

use std::path::{Path, PathBuf};

use dolphin_core::Cf32;
use dolphin_unwrap::{unwrap, UnwrapConfig};
use ndarray::{Array2, ArrayView2};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

fn snaphu_available() -> bool {
    std::process::Command::new("snaphu")
        .arg("--help")
        .output()
        .is_ok()
}

fn max_abs_err(a: ArrayView2<f32>, b: ArrayView2<f32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn unwrap_matches_direct_snaphu() {
    let dir = fixtures();
    if !dir.join("unw_oracle.npy").exists() {
        eprintln!("skipping unwrap oracle: no fixtures");
        return;
    }
    if !snaphu_available() {
        eprintln!("skipping unwrap oracle: snaphu not on PATH");
        return;
    }
    let ifg: Array2<Cf32> = ndarray_npy::read_npy(dir.join("unw_ifg.npy")).unwrap();
    let corr: Array2<f32> = ndarray_npy::read_npy(dir.join("unw_corr.npy")).unwrap();
    let unw_o: Array2<f32> = ndarray_npy::read_npy(dir.join("unw_oracle.npy")).unwrap();
    let cc_o: Array2<u32> = ndarray_npy::read_npy(dir.join("unw_conncomp.npy")).unwrap();

    let scratch = std::env::temp_dir().join("dolphinrust_unwrap_test");
    std::fs::create_dir_all(&scratch).unwrap();

    let out = unwrap(ifg.view(), corr.view(), &UnwrapConfig::default(), &scratch).unwrap();

    assert!(
        max_abs_err(out.unwrapped.view(), unw_o.view()) < 1e-3,
        "unwrapped phase differs"
    );
    assert_eq!(out.conncomp, cc_o, "connected components differ");
}
