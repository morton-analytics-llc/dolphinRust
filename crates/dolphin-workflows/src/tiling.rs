//! Block-tiling plan for memory-bounded phase linking.
//!
//! `link_one_burst` reads and phase-links one burst a tile at a time so peak
//! memory is bounded by a tile (block + halo) instead of the whole stack and its
//! per-pixel `N×N` coherence cube.
//!
//! ## Why not `StridedBlockManager`
//! dolphin-core's [`dolphin_core::StridedBlockManager`] deliberately leaves the
//! `output_margin` border (`round(half_window / strides)` output pixels on every
//! edge) as nodata — the dolphin convention. The current whole-burst path does
//! the opposite: covariance clamps each border pixel's window *inward* so it
//! stays full-size (`covariance::window_origin`), so every output pixel is
//! filled. Dropping the border to nodata would change those pixels and break the
//! whole-burst bit-identity contract. So instead of the manager's nodata-margin
//! tiling, [`plan_tiles`] covers the **full** output grid and chooses each tile's
//! read window so the existing covariance, run per tile, reproduces the global
//! (clamped) window for every output pixel bit-for-bit.

use dolphin_core::{BlockIndices, HalfWindow, Strides};

/// One processing tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePlan {
    /// Full-resolution input rectangle to read (across all epochs), padded by the
    /// window halo and clamped to the burst.
    pub read: BlockIndices,
    /// Global output rectangle this tile writes (decimated grid).
    pub out: BlockIndices,
    /// First local-output row of the tile's kernel output that maps to `out`
    /// (drops the leading halo).
    pub local_row0: usize,
    /// First local-output column of the tile's kernel output that maps to `out`.
    pub local_col0: usize,
}

/// Plan the tiles covering the full output grid of a `(rows, cols)` burst,
/// decimated by `strides`, with a halo of `depth` half-windows. `out_block` is
/// the output tile size (in output pixels); tiles partition the output grid
/// exactly once.
///
/// `depth` is the number of ministacks: sequential phase linking carries a
/// compressed SLC forward from each ministack into the next, and a compressed
/// SLC is itself a window-based product, so a written pixel's true data
/// dependency cone is `depth · half_window`, not one half_window. Reading only a
/// single-half-window halo silently corrupts pixels near interior tile seams
/// (their carried compressed SLCs were computed with a tile-clamped window).
#[must_use]
pub fn plan_tiles(
    shape: (usize, usize),
    strides: Strides,
    half: HalfWindow,
    depth: usize,
    out_block: (usize, usize),
) -> Vec<TilePlan> {
    let (rows, cols) = shape;
    let (out_rows, out_cols) = strides.out_shape((rows, cols));
    let row_tiles = tile_starts(out_rows, out_block.0.max(1));
    let col_tiles = tile_starts(out_cols, out_block.1.max(1));
    row_tiles
        .iter()
        .flat_map(|&(og0, og1)| {
            col_tiles
                .iter()
                .map(move |&(oc0, oc1)| (og0, og1, oc0, oc1))
        })
        .map(|(og0, og1, oc0, oc1)| {
            let (r0, r1, lr) = axis_window(og0, og1, strides.y, half.y, depth, rows);
            let (c0, c1, lc) = axis_window(oc0, oc1, strides.x, half.x, depth, cols);
            TilePlan {
                read: BlockIndices {
                    row_start: r0,
                    row_stop: r1,
                    col_start: c0,
                    col_stop: c1,
                },
                out: BlockIndices {
                    row_start: og0,
                    row_stop: og1,
                    col_start: oc0,
                    col_stop: oc1,
                },
                local_row0: lr,
                local_col0: lc,
            }
        })
        .collect()
}

/// Contiguous `[start, stop)` output-pixel tiles of width `block` covering `len`.
fn tile_starts(len: usize, block: usize) -> Vec<(usize, usize)> {
    (0..len)
        .step_by(block)
        .map(|s| (s, (s + block).min(len)))
        .collect()
}

/// Global window origin (top-left, full-res) of output pixel `g`, clamped inward
/// at the border to keep the window full-size — mirrors `covariance::window_origin`.
fn origin(g: usize, s: usize, h: usize, dim: usize) -> usize {
    let win = 2 * h + 1;
    (s / 2 + g * s).saturating_sub(h).min(dim - win)
}

/// Per-axis read range + local trim for output pixels `[og0, og1)`, padded to a
/// `depth`-deep ministack dependency cone.
///
/// Returns `(read_start, read_stop, local_start)`. The base window padding (one
/// half-window) is extended by `(depth - 1)·h` on each side to cover the
/// compressed-SLC chain. `read_start` is snapped down to a multiple of the stride
/// so the tile's local output grid aligns to the global one; both ends are
/// clamped to the burst so the edge (inward-clamping) covariance still matches.
fn axis_window(
    og0: usize,
    og1: usize,
    s: usize,
    h: usize,
    depth: usize,
    dim: usize,
) -> (usize, usize, usize) {
    let win = 2 * h + 1;
    // Each compressed-SLC hop reads through the strided `upsample_nearest`, whose
    // center rounds outward by up to one stride; so each of the `depth-1` hops
    // beyond the base window needs `h + s` of clearance, not just `h`.
    let extra = depth.saturating_sub(1) * (h + s);
    let lo = origin(og0, s, h, dim).saturating_sub(extra);
    let hi = (origin(og1 - 1, s, h, dim) + win + extra).max(s * og1);
    // Snap both ends to stride multiples so the tile's decimated grid aligns to
    // the global one *and* its `out_shape` is an exact `1/s` of the read height —
    // the compressed-SLC `upsample_nearest` infers its look factor as
    // `read_len / out_len`, which only equals `s` (matching the whole-burst
    // upsample phase) when the read length is a stride multiple.
    let read_start = (lo / s) * s;
    let read_stop = hi.div_ceil(s).saturating_mul(s).min((dim / s) * s);
    (read_start, read_stop, og0 - read_start / s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert a single tile is well-formed: stride-aligned read, window fits, and
    /// the trim lands inside the tile's kernel output. Returns the (row, col)
    /// output pixels it writes, for the coverage tally.
    fn check_tile(p: &TilePlan, strides: Strides, half: HalfWindow) -> Vec<(usize, usize)> {
        assert_eq!(
            p.read.row_start % strides.y,
            0,
            "read row not stride-aligned"
        );
        assert_eq!(
            p.read.col_start % strides.x,
            0,
            "read col not stride-aligned"
        );
        assert!(
            p.read.height() > 2 * half.y,
            "read block shorter than window"
        );
        assert!(
            p.read.width() > 2 * half.x,
            "read block narrower than window"
        );
        let (lor, loc) = strides.out_shape((p.read.height(), p.read.width()));
        assert!(
            p.local_row0 + p.out.height() <= lor,
            "row trim out of bounds"
        );
        assert!(
            p.local_col0 + p.out.width() <= loc,
            "col trim out of bounds"
        );
        p.out
            .rows()
            .flat_map(|r| p.out.cols().map(move |c| (r, c)))
            .collect()
    }

    fn assert_case(
        shape: (usize, usize),
        strides: Strides,
        half: HalfWindow,
        depth: usize,
        block: (usize, usize),
    ) {
        let (out_rows, out_cols) = strides.out_shape(shape);
        let mut covered = vec![0u32; out_rows * out_cols];
        plan_tiles(shape, strides, half, depth, block)
            .iter()
            .flat_map(|p| check_tile(p, strides, half))
            .for_each(|(r, c)| covered[r * out_cols + c] += 1);
        assert!(covered.iter().all(|&n| n == 1), "pixel covered != once");
    }

    /// Every output pixel is written by exactly one tile, and each tile's written
    /// region fits inside its kernel output (trim is in-bounds) — across a mix of
    /// strides, half-windows, and block sizes (including non-divisible edges).
    #[test]
    fn tiles_cover_output_exactly_once_and_trims_fit() {
        assert_case(
            (40, 50),
            Strides { y: 1, x: 1 },
            HalfWindow { y: 7, x: 14 },
            1,
            (16, 16),
        );
        assert_case(
            (40, 50),
            Strides { y: 2, x: 3 },
            HalfWindow { y: 5, x: 11 },
            3,
            (8, 8),
        );
        assert_case(
            (97, 61),
            Strides { y: 1, x: 1 },
            HalfWindow { y: 2, x: 2 },
            2,
            (13, 9),
        );
        assert_case(
            (97, 61),
            Strides { y: 3, x: 2 },
            HalfWindow { y: 4, x: 6 },
            4,
            (5, 5),
        );
    }
}
