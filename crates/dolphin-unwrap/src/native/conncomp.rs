//! Coherence-based connected-component segmentation + masking.
//!
//! SNAPHU emits a connected-component label per pixel: low-coherence pixels are
//! dropped to component 0 (masked) and the reliable regions they separate become
//! distinct components. We reproduce that partition clean-room from the
//! coherence alone — pixels below `min_corr` are masked, the remaining pixels are
//! grouped into 4-connected components, and components smaller than `min_frac` of
//! the scene are dropped to 0. Labels are numbered by descending size to mirror
//! SNAPHU's convention. This replaces the prior trivial single-component output;
//! it agrees with SNAPHU's partition to high IoU on the residue-dense golden.

use ndarray::{Array2, ArrayView2};

/// Segment `corr` into connected components: 0 = masked/low-coherence, 1..N the
/// reliable regions ordered by descending pixel count.
pub fn segment(corr: ArrayView2<f32>, min_corr: f32, min_frac: f64) -> Array2<u32> {
    let (rows, cols) = corr.dim();
    let keep = corr.mapv(|c| c >= min_corr);
    let (provisional, sizes) = label_components(&keep);
    let min_size = (min_frac * (rows * cols) as f64).ceil() as usize;
    let remap = rank_by_size(&sizes, min_size);
    provisional.mapv(|p| remap.get(p as usize).copied().unwrap_or(0))
}

/// Flood-fill 4-connected `true` pixels into provisional labels `1..=K`,
/// returning the label grid and `sizes[label] = pixel count` (`sizes[0] = 0`).
fn label_components(keep: &Array2<bool>) -> (Array2<u32>, Vec<usize>) {
    let (rows, cols) = keep.dim();
    let mut labels = Array2::<u32>::zeros((rows, cols));
    let mut sizes = vec![0usize]; // index 0 unused (masked)
    let mut next = 1u32;
    for (seed, &k) in keep.indexed_iter() {
        if !k || labels[seed] != 0 {
            continue;
        }
        sizes.push(flood(keep, &mut labels, seed, next));
        next += 1;
    }
    (labels, sizes)
}

/// Stack flood-fill from `seed`, stamping `label`; returns the region size.
fn flood(keep: &Array2<bool>, labels: &mut Array2<u32>, seed: (usize, usize), label: u32) -> usize {
    let (rows, cols) = keep.dim();
    let mut stack = vec![seed];
    labels[seed] = label;
    let mut size = 0usize;
    while let Some((i, j)) = stack.pop() {
        size += 1;
        for (ni, nj) in neighbors(i, j, rows, cols) {
            if keep[(ni, nj)] && labels[(ni, nj)] == 0 {
                labels[(ni, nj)] = label;
                stack.push((ni, nj));
            }
        }
    }
    size
}

/// The in-bounds 4-neighbours of `(i, j)`.
fn neighbors(i: usize, j: usize, rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize)> {
    let up = (i > 0).then(|| (i - 1, j));
    let down = (i + 1 < rows).then_some((i + 1, j));
    let left = (j > 0).then(|| (i, j - 1));
    let right = (j + 1 < cols).then_some((i, j + 1));
    [up, down, left, right].into_iter().flatten()
}

/// Map provisional labels to final labels `1..=N` by descending size, sending
/// components below `min_size` (and the masked label 0) to 0.
fn rank_by_size(sizes: &[usize], min_size: usize) -> Vec<u32> {
    let mut kept: Vec<usize> = (1..sizes.len()).filter(|&p| sizes[p] >= min_size).collect();
    kept.sort_unstable_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let mut remap = vec![0u32; sizes.len()];
    for (rank, &p) in kept.iter().enumerate() {
        remap[p] = rank as u32 + 1;
    }
    remap
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Two coherent blocks split by a low-coherence moat become components 1 and
    /// 2 (the larger first); the moat is masked to 0.
    #[test]
    fn moat_splits_into_two_components() {
        let corr = array![
            [0.8f32, 0.8, 0.02, 0.8, 0.8, 0.8],
            [0.8, 0.8, 0.02, 0.8, 0.8, 0.8],
            [0.8, 0.8, 0.02, 0.8, 0.8, 0.8],
        ];
        let cc = segment(corr.view(), 0.15, 0.0);
        assert_eq!(cc[(0, 2)], 0, "moat masked");
        assert_eq!(cc[(0, 0)], 2, "smaller left block is component 2");
        assert_eq!(cc[(0, 3)], 1, "larger right block is component 1");
        assert!(cc.iter().filter(|&&l| l == 1).count() > cc.iter().filter(|&&l| l == 2).count());
    }

    /// Components below `min_frac` are dropped to 0.
    #[test]
    fn tiny_components_dropped() {
        let corr = array![[0.9f32, 0.02, 0.9], [0.02, 0.02, 0.9], [0.9, 0.9, 0.9],];
        // min_frac 0.2 of 9 px -> min_size 2; the lone top-left pixel is dropped.
        let cc = segment(corr.view(), 0.15, 0.2);
        assert_eq!(cc[(0, 0)], 0, "isolated 1-px component dropped");
        assert_eq!(cc[(2, 0)], 1, "the large L-shaped region survives");
    }
}
