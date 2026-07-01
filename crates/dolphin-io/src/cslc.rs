//! OPERA/NISAR CSLC reading from HDF5 (port of the CSLC paths in `io/`).
//!
//! Complex SLCs live in HDF5 datasets (e.g.
//! `/science/SENTINEL1/CSLC/grids/VV`). `hdf5-metno` reads `Complex<f32>` via the
//! h5py-compatible `(r, i)` compound layout, so no manual decoding is needed.

use std::path::Path;

use dolphin_core::{BlockIndices, Cf32};
use ndarray::{s, Array2, Array3, Axis};

use crate::error::{IoError, Result};

/// Read a 2D complex CSLC dataset at `dataset` from the HDF5 file at `path`.
pub fn read_cslc(path: &Path, dataset: &str) -> Result<Array2<Cf32>> {
    let file = hdf5::File::open(path)?;
    Ok(file.dataset(dataset)?.read_2d::<Cf32>()?)
}

/// Read the `(rows, cols)` shape of the HDF5 dataset `dataset` from metadata
/// only — no sample data is loaded. Used to size block-tiled processing before
/// any windowed read. Works for any 2D complex grid (OPERA or NISAR).
pub fn read_cslc_shape(path: &Path, dataset: &str) -> Result<(usize, usize)> {
    let file = hdf5::File::open(path)?;
    let shape = file.dataset(dataset)?.shape();
    match shape.as_slice() {
        &[rows, cols] => Ok((rows, cols)),
        _ => Err(IoError::Shape(format!(
            "expected a 2D dataset at {dataset}, got shape {shape:?}"
        ))),
    }
}

/// Read a rectangular `block` window of the complex CSLC dataset `dataset` —
/// only the `[row_start, row_stop) x [col_start, col_stop)` sub-rectangle is
/// pulled from disk via HDF5 hyperslab selection, never the whole grid.
///
/// The result is bit-identical to [`read_cslc`] sliced to the same window; it is
/// the load-bearing read for block-tiled phase linking, which keeps only one
/// tile (block + halo) resident per burst instead of the full stack.
pub fn read_cslc_window(path: &Path, dataset: &str, block: BlockIndices) -> Result<Array2<Cf32>> {
    let file = hdf5::File::open(path)?;
    Ok(file
        .dataset(dataset)?
        .read_slice_2d::<Cf32, _>(s![block.rows(), block.cols()])?)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a single complex dataset at `/data/VV` for the window contract test.
    fn write_vv_fixture(path: &Path, full: &Array2<Cf32>) {
        let _ = std::fs::remove_file(path);
        let file = hdf5::File::create(path).unwrap();
        let group = file.create_group("data").unwrap();
        group
            .new_dataset_builder()
            .with_data(full)
            .create("VV")
            .unwrap();
    }

    /// Contract: a windowed read returns exactly the same samples as a full read
    /// sliced to the same rectangle — bit-identical, including an off-origin,
    /// non-edge-aligned window. This is the invariant the block-tiled reader
    /// relies on for whole-burst bit-identity.
    #[test]
    fn window_read_matches_full_read_sliced() {
        let _hdf5 = crate::test_hdf5_lock::guard();
        let path = std::env::temp_dir().join("dolphin_cslc_window_contract.h5");
        let full = Array2::from_shape_fn((40, 50), |(r, c)| {
            Cf32::new(r as f32 - 3.0, (c * 2) as f32 + 0.5)
        });
        write_vv_fixture(&path, &full);

        let block = BlockIndices {
            row_start: 7,
            row_stop: 29,
            col_start: 11,
            col_stop: 44,
        };
        let win = read_cslc_window(&path, "/data/VV", block).unwrap();
        let expected = full.slice(s![block.rows(), block.cols()]).to_owned();
        assert_eq!(win, expected);

        // Sanity: the whole-grid read also agrees with the source.
        assert_eq!(read_cslc(&path, "/data/VV").unwrap(), full);
        let _ = std::fs::remove_file(&path);
    }
}
