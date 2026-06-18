//! Real-data validation of the NISAR reader (DoD #5, best effort).
//!
//! Gated on `NISAR_REAL_H5` pointing to a real NISAR GSLC granule (the full HH
//! grid is billions of pixels, so a corner block is sliced rather than fully
//! read). Skips when the env var is unset. Proves the reader handles the real
//! `{r: f32, i: f32}` compound and the real NISAR geocoding metadata
//! (`xCoordinates`/`yCoordinates` + `projection.epsg_code`), not just the
//! synthesized fixture.

use dolphin_core::Cf32;
use dolphin_io::read_nisar_geotransform;
use ndarray::s;

const HH: &str = "/science/LSAR/GSLC/grids/frequencyA/HH";

#[test]
fn reads_real_nisar_granule() {
    let Ok(path) = std::env::var("NISAR_REAL_H5") else {
        eprintln!("skipping nisar real-data: set NISAR_REAL_H5 to a NISAR GSLC .h5");
        return;
    };

    // A center block of the complex grid decodes from the {r,i} f32 compound.
    // (The geocoded grid's corners are NaN fill outside the swath footprint, so
    // sample the middle to land on real imagery.)
    let file = hdf5::File::open(&path).unwrap();
    let hh = file.dataset(HH).unwrap();
    let (rows, cols) = (hh.shape()[0], hh.shape()[1]);
    let (r0, c0) = (rows / 2, cols / 2);
    let block = hh
        .read_slice_2d::<Cf32, _>(s![r0..r0 + 256, c0..c0 + 256])
        .unwrap();
    assert_eq!(block.dim(), (256, 256));
    let finite = block
        .iter()
        .filter(|z| z.re.is_finite() && z.im.is_finite())
        .count();
    eprintln!("real NISAR HH center 256x256 block: {finite}/65536 finite samples");
    assert!(finite > 0, "center block has real (finite) complex samples");

    // The custom geotransform/EPSG reader works on the real geocoding metadata.
    let geo = read_nisar_geotransform(std::path::Path::new(&path), HH).unwrap();
    eprintln!(
        "real NISAR geo: epsg={} origin=({:.1},{:.1}) posting=({:.3},{:.3})",
        geo.epsg,
        geo.geotransform[0],
        geo.geotransform[3],
        geo.geotransform[1],
        geo.geotransform[5]
    );
    assert!(geo.epsg >= 32601 && geo.epsg <= 32760, "plausible UTM EPSG");
    assert!(geo.geotransform[1] > 0.0, "dx > 0");
    assert!(geo.geotransform[5] < 0.0, "dy < 0");
    assert!(
        geo.geotransform[1] < 1000.0 && geo.geotransform[5] > -1000.0,
        "posting within ~1km"
    );
}
