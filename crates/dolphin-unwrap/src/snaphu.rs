//! SNAPHU subprocess wrapper (port of dolphin's `unwrap/` SNAPHU dispatch).
//!
//! dolphin contains no unwrapping math — it drives the external SNAPHU solver,
//! and so do we. The wrapped interferogram and correlation are written as flat
//! binary, SNAPHU is invoked, and the unwrapped phase + connected-component
//! labels are read back. Flat-binary I/O assumes a little-endian host (matches
//! SNAPHU's native-endian format and numpy's `.tofile`).

use std::path::Path;
use std::process::Command;

use dolphin_core::Cf32;
use ndarray::{Array2, ArrayView2};

/// SNAPHU cost mode.
#[derive(Debug, Clone, Copy)]
pub enum CostMode {
    Smooth,
    Defo,
    Topo,
}

/// SNAPHU initialization method.
#[derive(Debug, Clone, Copy)]
pub enum InitMethod {
    Mcf,
    Mst,
}

/// SNAPHU invocation configuration.
#[derive(Debug, Clone)]
pub struct UnwrapConfig {
    pub cost: CostMode,
    pub init: InitMethod,
    pub ntiles: (usize, usize),
    pub tile_overlap: (usize, usize),
    pub nproc: usize,
    pub snaphu_path: String,
}

impl Default for UnwrapConfig {
    fn default() -> Self {
        Self {
            cost: CostMode::Smooth,
            init: InitMethod::Mcf,
            ntiles: (1, 1),
            tile_overlap: (0, 0),
            nproc: 1,
            snaphu_path: "snaphu".to_string(),
        }
    }
}

/// Unwrapped phase + connected-component labels.
pub struct UnwrapResult {
    pub unwrapped: Array2<f32>,
    pub conncomp: Array2<u32>,
}

/// Errors from the SNAPHU dispatch.
#[derive(Debug, thiserror::Error)]
pub enum UnwrapError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("snaphu failed: {0}")]
    Snaphu(String),
    #[error("output shape mismatch: {0}")]
    Shape(String),
}

/// Convenience alias for fallible unwrapping.
pub type Result<T> = std::result::Result<T, UnwrapError>;

/// Unwrap a wrapped interferogram with SNAPHU, writing scratch files in `scratch`.
///
/// # Errors
/// Returns `Err` if scratch I/O fails, SNAPHU exits non-zero, or the outputs are
/// the wrong size.
pub fn unwrap(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    cfg: &UnwrapConfig,
    scratch: &Path,
) -> Result<UnwrapResult> {
    let (rows, cols) = wrapped.dim();
    let ifg_path = scratch.join("ifg.c8");
    let corr_path = scratch.join("corr.f4");
    let unw_path = scratch.join("unw.f4");
    let cc_path = scratch.join("conncomp.u4");

    std::fs::write(&ifg_path, complex_bytes(wrapped))?;
    std::fs::write(&corr_path, f32_bytes(correlation))?;
    run_snaphu(cfg, &ifg_path, &corr_path, &unw_path, &cc_path, cols)?;

    let unwrapped = read_f32(&unw_path, (rows, cols))?;
    let conncomp = read_u32(&cc_path, (rows, cols))?;
    Ok(UnwrapResult {
        unwrapped,
        conncomp,
    })
}

/// Build and run the SNAPHU command, erroring on a non-zero exit.
fn run_snaphu(
    cfg: &UnwrapConfig,
    ifg: &Path,
    corr: &Path,
    unw: &Path,
    cc: &Path,
    cols: usize,
) -> Result<()> {
    let mut cmd = Command::new(&cfg.snaphu_path);
    cmd.arg(cost_flag(cfg.cost)).arg(init_flag(cfg.init));
    cmd.args(["-C", "CONNCOMPOUTTYPE UINT"]);
    cmd.args(["-C", "OUTFILEFORMAT FLOAT_DATA"]);
    cmd.args(["-C", "CORRFILEFORMAT FLOAT_DATA"]);
    cmd.arg("-c").arg(corr).arg("-o").arg(unw).arg("-g").arg(cc);
    add_tiling(&mut cmd, cfg);
    cmd.arg(ifg).arg(cols.to_string());

    let output = cmd.output()?;
    if output.status.success() {
        return Ok(());
    }
    Err(UnwrapError::Snaphu(
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

/// Add multi-tile flags when more than one tile is requested.
fn add_tiling(cmd: &mut Command, cfg: &UnwrapConfig) {
    if cfg.ntiles == (1, 1) {
        return;
    }
    let (nrow, ncol) = cfg.ntiles;
    let (rov, cov) = cfg.tile_overlap;
    cmd.args([
        "--tile",
        &nrow.to_string(),
        &ncol.to_string(),
        &rov.to_string(),
        &cov.to_string(),
    ]);
    cmd.args(["--nproc", &cfg.nproc.to_string()]);
}

fn cost_flag(cost: CostMode) -> &'static str {
    match cost {
        CostMode::Smooth => "-s",
        CostMode::Defo => "-d",
        CostMode::Topo => "-t",
    }
}

fn init_flag(init: InitMethod) -> &'static str {
    match init {
        InitMethod::Mcf => "--mcf",
        InitMethod::Mst => "--mst",
    }
}

/// Serialize a complex array as interleaved little-endian `(re, im)` floats.
fn complex_bytes(arr: ArrayView2<Cf32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(arr.len() * 8);
    arr.iter().for_each(|z| {
        bytes.extend_from_slice(&z.re.to_le_bytes());
        bytes.extend_from_slice(&z.im.to_le_bytes());
    });
    bytes
}

/// Serialize an f32 array as little-endian floats.
fn f32_bytes(arr: ArrayView2<f32>) -> Vec<u8> {
    arr.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Read a row-major little-endian f32 raster of the given shape.
fn read_f32(path: &Path, shape: (usize, usize)) -> Result<Array2<f32>> {
    let bytes = std::fs::read(path)?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Array2::from_shape_vec(shape, values).map_err(|e| UnwrapError::Shape(e.to_string()))
}

/// Read a row-major little-endian u32 raster of the given shape.
fn read_u32(path: &Path, shape: (usize, usize)) -> Result<Array2<u32>> {
    let bytes = std::fs::read(path)?;
    let values: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Array2::from_shape_vec(shape, values).map_err(|e| UnwrapError::Shape(e.to_string()))
}
