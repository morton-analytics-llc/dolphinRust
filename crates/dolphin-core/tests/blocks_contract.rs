//! Contract tests for the `StridedBlockManager` tiling scheme.
//!
//! The load-bearing invariants (crate CLAUDE.md): every output pixel in the
//! valid interior is covered exactly once, the nodata border is never covered,
//! strides are honored, and the halo/trim slices recover the un-padded region.

use dolphin_core::blocks::{StridedBlockManager, TileBlocks};
use dolphin_core::{HalfWindow, Strides};

type Case = ((usize, usize), (usize, usize), Strides, HalfWindow);

fn cases() -> Vec<Case> {
    vec![
        (
            (100, 100),
            (32, 32),
            Strides { y: 1, x: 1 },
            HalfWindow { y: 5, x: 5 },
        ),
        (
            (100, 100),
            (40, 40),
            Strides { y: 1, x: 1 },
            HalfWindow { y: 7, x: 14 },
        ),
        (
            (120, 240),
            (32, 64),
            Strides { y: 3, x: 6 },
            HalfWindow { y: 7, x: 14 },
        ),
        (
            (90, 90),
            (30, 30),
            Strides { y: 3, x: 3 },
            HalfWindow { y: 4, x: 4 },
        ),
        (
            (64, 64),
            (64, 64),
            Strides { y: 1, x: 1 },
            HalfWindow { y: 0, x: 0 },
        ),
        (
            (200, 150),
            (50, 50),
            Strides { y: 2, x: 2 },
            HalfWindow { y: 5, x: 11 },
        ),
    ]
}

#[test]
fn interior_covered_exactly_once_and_border_never() {
    cases().into_iter().for_each(assert_partition);
}

#[test]
fn strides_honored_and_input_in_bounds() {
    cases().into_iter().for_each(assert_strides);
}

#[test]
fn input_trim_recovers_unpadded_region() {
    cases().into_iter().for_each(assert_trims);
}

fn assert_partition((arr, block, strides, half): Case) {
    let mgr = StridedBlockManager::new(arr, block, strides, half);
    let (out_rows, out_cols) = mgr.output_shape();
    let (margin_y, margin_x) = mgr.output_margin();

    let mut counts = vec![vec![0u32; out_cols]; out_rows];
    let covered = mgr
        .iter_blocks()
        .into_iter()
        .flat_map(|t| pixels(t.out_block));
    covered.for_each(|(r, c)| counts[r][c] += 1);

    let last_row = out_rows.saturating_sub(margin_y);
    let last_col = out_cols.saturating_sub(margin_x);
    let interior = |r, c| r >= margin_y && r < last_row && c >= margin_x && c < last_col;
    let cells = (0..out_rows).flat_map(|r| (0..out_cols).map(move |c| (r, c)));
    cells.for_each(|(r, c)| {
        let expected = u32::from(interior(r, c));
        assert_eq!(counts[r][c], expected, "pixel ({r},{c}) in {arr:?}");
    });
}

fn assert_strides((arr, block, strides, half): Case) {
    let mgr = StridedBlockManager::new(arr, block, strides, half);
    mgr.iter_blocks().into_iter().for_each(|tile| {
        let inp = tile.input_no_padding;
        assert_eq!(inp.row_start, tile.out_block.row_start * strides.y);
        assert_eq!(inp.row_stop, tile.out_block.row_stop * strides.y);
        assert_eq!(inp.col_start, tile.out_block.col_start * strides.x);
        assert_eq!(inp.col_stop, tile.out_block.col_stop * strides.x);
        assert!(tile.input_block.row_stop <= arr.0);
        assert!(tile.input_block.col_stop <= arr.1);
    });
}

fn assert_trims((arr, block, strides, half): Case) {
    let mgr = StridedBlockManager::new(arr, block, strides, half);
    mgr.iter_blocks()
        .into_iter()
        .for_each(|t| assert_recovers(&t));
}

/// Every output-grid pixel in a block, as `(row, col)` pairs.
fn pixels(b: dolphin_core::BlockIndices) -> impl Iterator<Item = (usize, usize)> {
    b.rows().flat_map(move |r| b.cols().map(move |c| (r, c)))
}

/// The input trim applied to the read block reproduces the un-padded region.
fn assert_recovers(tile: &TileBlocks) {
    let rows = tile.input_trim.rows(tile.input_block.height());
    let cols = tile.input_trim.cols(tile.input_block.width());
    assert_eq!(
        tile.input_block.row_start + rows.start,
        tile.input_no_padding.row_start
    );
    assert_eq!(
        tile.input_block.row_start + rows.end,
        tile.input_no_padding.row_stop
    );
    assert_eq!(
        tile.input_block.col_start + cols.start,
        tile.input_no_padding.col_start
    );
    assert_eq!(
        tile.input_block.col_start + cols.end,
        tile.input_no_padding.col_stop
    );
}
