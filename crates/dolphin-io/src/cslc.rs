//! OPERA/NISAR CSLC reading from HDF5 (port of the CSLC paths in `io/`).
//!
//! Complex SLCs live in HDF5 datasets (e.g.
//! `/science/SENTINEL1/CSLC/grids/VV`). `hdf5-metno` reads `Complex<f32>` via the
//! h5py-compatible `(r, i)` compound layout, so no manual decoding is needed.

use std::path::Path;

use dolphin_core::Cf32;
use ndarray::{Array2, Array3, Axis};

use crate::error::{IoError, Result};

/// Read a 2D complex CSLC dataset at `dataset` from the HDF5 file at `path`.
pub fn read_cslc(path: &Path, dataset: &str) -> Result<Array2<Cf32>> {
    let file = hdf5::File::open(path)?;
    Ok(file.dataset(dataset)?.read_2d::<Cf32>()?)
}

/// Read a date-ordered set of CSLC files into an `(n_slc, rows, cols)` stack.
///
/// `files` are `(path, dataset)` pairs; the caller supplies them already sorted
/// by acquisition date (`VRTStack`'s date-sort lives in the pipeline phase).
pub fn read_cslc_stack(files: &[(std::path::PathBuf, String)]) -> Result<Array3<Cf32>> {
    let layers = files
        .iter()
        .map(|(path, dataset)| read_cslc(path, dataset))
        .collect::<Result<Vec<_>>>()?;
    let views: Vec<_> = layers.iter().map(Array2::view).collect();
    ndarray::stack(Axis(0), &views).map_err(|e| IoError::Shape(e.to_string()))
}
