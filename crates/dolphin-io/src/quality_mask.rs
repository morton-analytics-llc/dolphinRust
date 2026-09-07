//! Conservative native sensor mask sampling on an integer-refined output grid.
use std::path::Path;

use ndarray::{s, Array2};

use crate::{GeoInfo, IoError, Result};

/// Read a bounded mask window. Returns validity and full-pixel source coverage.
/// OPERA STATIC uses 0=good; NISAR GSLC uses subswath IDs 1..254=good.
/// Every native subpixel must pass; partially covered output pixels fail closed.
///
/// # Errors
/// Rejects missing masks, incompatible CRS, shape, posting, or registration.
pub fn read_native_quality_mask(
    path: &Path,
    dataset: &str,
    nisar: bool,
    target: GeoInfo,
    shape: (usize, usize),
) -> Result<(Array2<bool>, Array2<bool>)> {
    let geo = if nisar {
        crate::read_nisar_geotransform(path, dataset)?
    } else {
        crate::read_geotransform(path, dataset)?
    };
    let sg = geo.geotransform;
    let tg = target.geotransform;
    let scales = [tg[5] / sg[5], tg[1] / sg[1]];
    let offsets = [(tg[3] - sg[3]) / sg[5], (tg[0] - sg[0]) / sg[1]];
    if geo.epsg != target.epsg
        || sg[1] <= 0.0
        || sg[5] >= 0.0
        || [sg[2], sg[4], tg[2], tg[4]].iter().any(|x| x.abs() > 1e-6)
        || scales
            .iter()
            .any(|x| !x.is_finite() || *x < 1.0 || (x - x.round()).abs() > 1e-6)
        || offsets
            .iter()
            .any(|x| !x.is_finite() || (x - x.round()).abs() > 1e-6)
    {
        return Err(IoError::Geo(
            "native quality mask requires the same CRS and integer-aligned refinement".into(),
        ));
    }
    let file = hdf5::File::open(path)?;
    let ds = file.dataset(dataset)?;
    let dims = ds.shape();
    if dims.len() != 2 {
        return Err(IoError::Shape(
            "native quality mask must be two-dimensional".into(),
        ));
    }
    let mut valid = Array2::from_elem(shape, true);
    let mut covered = Array2::from_elem(shape, false);
    let step = [scales[0].round() as isize, scales[1].round() as isize];
    let origin = [offsets[0].round() as isize, offsets[1].round() as isize];
    let start = [origin[0].max(0) as usize, origin[1].max(0) as usize];
    let stop = [
        (origin[0] + shape.0 as isize * step[0]).min(dims[0] as isize),
        (origin[1] + shape.1 as isize * step[1]).min(dims[1] as isize),
    ];
    if stop[0] <= start[0] as isize || stop[1] <= start[1] as isize {
        return Ok((valid, covered));
    }
    let values =
        ds.read_slice_2d::<f64, _>(s![start[0]..stop[0] as usize, start[1]..stop[1] as usize])?;
    for ((r, c), good) in valid.indexed_iter_mut() {
        let lo = [
            origin[0] + r as isize * step[0],
            origin[1] + c as isize * step[1],
        ];
        let hi = [lo[0] + step[0], lo[1] + step[1]];
        if lo[0] >= 0 && lo[1] >= 0 && hi[0] <= dims[0] as isize && hi[1] <= dims[1] as isize {
            covered[(r, c)] = true;
            *good = values
                .slice(s![
                    (lo[0] as usize - start[0])..(hi[0] as usize - start[0]),
                    (lo[1] as usize - start[1])..(hi[1] as usize - start[1])
                ])
                .iter()
                .all(|&v| {
                    if nisar {
                        v > 0.0 && v < 255.0 && v.fract() == 0.0
                    } else {
                        v == 0.0
                    }
                });
        }
    }
    Ok((valid, covered))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_quality_mask_respects_sensor_codes_and_all_subpixels() {
        let _lock = crate::test_hdf5_lock::guard();
        for nisar in [false, true] {
            let path = std::env::temp_dir().join(format!("native_quality_mask_{nisar}.h5"));
            {
                let file = hdf5::File::create(&path).unwrap();
                let g = file.create_group("data").unwrap();
                let good = if nisar { 1u8 } else { 0u8 };
                let mut values = Array2::from_elem((4, 4), good);
                values[(1, 1)] = if nisar { 255 } else { 1 };
                values[(2, 2)] = if nisar { 0 } else { 3 };
                g.new_dataset_builder()
                    .with_data(&values)
                    .create("mask")
                    .unwrap();
                g.new_dataset_builder()
                    .with_data(&[0.5, 1.5, 2.5, 3.5])
                    .create(if nisar {
                        "xCoordinates"
                    } else {
                        "x_coordinates"
                    })
                    .unwrap();
                g.new_dataset_builder()
                    .with_data(&[3.5, 2.5, 1.5, 0.5])
                    .create(if nisar {
                        "yCoordinates"
                    } else {
                        "y_coordinates"
                    })
                    .unwrap();
                g.new_dataset::<i64>()
                    .create("projection")
                    .unwrap()
                    .write_scalar(&32611)
                    .unwrap();
            }
            let target = GeoInfo {
                epsg: 32611,
                geotransform: [0.0, 2.0, 0.0, 4.0, 0.0, -2.0],
            };
            let (valid, covered) =
                read_native_quality_mask(&path, "/data/mask", nisar, target, (2, 2)).unwrap();
            assert_eq!(valid, ndarray::arr2(&[[false, true], [true, false]]));
            assert!(covered.iter().all(|v| *v));
            let mut shifted = target;
            shifted.geotransform[0] += 0.5;
            assert!(read_native_quality_mask(&path, "/data/mask", nisar, shifted, (2, 2)).is_err());
            assert!(
                read_native_quality_mask(&path, "/data/missing", nisar, target, (2, 2)).is_err()
            );
            std::fs::remove_file(path).unwrap();
        }
    }
}
