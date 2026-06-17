//! Blocked/decimated raster tiling — Rust port of dolphin's `io/_blocks.py`.
//!
//! [`StridedBlockManager`] tiles a raster for sliding-window block processing
//! (covariance, SHP). Each tile yields five regions:
//!
//! * `out_block` — the decimated output region this tile writes,
//! * `out_trim` — slice into the decimated kernel output that drops the halo,
//! * `input_block` — the full-res region to read (padded by the window halo),
//! * `input_no_padding` — the full-res region matching `out_block` (no halo),
//! * `input_trim` — slice into the read block that recovers `input_no_padding`.
//!
//! The halo equals `strides * round(half_window / strides)`; the output-grid
//! border of `round(half_window / strides)` pixels is never written (edge
//! pixels lack full window support and stay nodata).

use std::ops::Range;

use crate::types::{HalfWindow, Strides};

/// Slices for 2D array access (concrete start/stop, full resolution or output grid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIndices {
    /// First row in the block (inclusive).
    pub row_start: usize,
    /// One past the last row in the block (exclusive).
    pub row_stop: usize,
    /// First column in the block (inclusive).
    pub col_start: usize,
    /// One past the last column in the block (exclusive).
    pub col_stop: usize,
}

impl BlockIndices {
    /// Row range `[row_start, row_stop)`.
    #[must_use]
    pub fn rows(&self) -> Range<usize> {
        self.row_start..self.row_stop
    }

    /// Column range `[col_start, col_stop)`.
    #[must_use]
    pub fn cols(&self) -> Range<usize> {
        self.col_start..self.col_stop
    }

    /// Number of rows spanned.
    #[must_use]
    pub fn height(&self) -> usize {
        self.row_stop - self.row_start
    }

    /// Number of columns spanned.
    #[must_use]
    pub fn width(&self) -> usize {
        self.col_stop - self.col_start
    }
}

/// A trimming slice: drop `start` pixels from the front and `end_trim` from the
/// back of a computed block. `end_trim == 0` means "to the end".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trim {
    /// Rows to drop from the front of the block.
    pub row_start: usize,
    /// Rows to drop from the back of the block.
    pub row_end_trim: usize,
    /// Columns to drop from the front of the block.
    pub col_start: usize,
    /// Columns to drop from the back of the block.
    pub col_end_trim: usize,
}

impl Trim {
    /// Row range into a block of `len` rows.
    #[must_use]
    pub fn rows(&self, len: usize) -> Range<usize> {
        self.row_start..len - self.row_end_trim
    }

    /// Column range into a block of `len` columns.
    #[must_use]
    pub fn cols(&self, len: usize) -> Range<usize> {
        self.col_start..len - self.col_end_trim
    }
}

/// The five regions describing one processing tile (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileBlocks {
    /// Decimated output region this tile writes.
    pub out_block: BlockIndices,
    /// Slice into the decimated kernel output that drops the halo.
    pub out_trim: Trim,
    /// Full-resolution region to read, padded by the window halo.
    pub input_block: BlockIndices,
    /// Full-resolution region matching `out_block`, without the halo.
    pub input_no_padding: BlockIndices,
    /// Slice into the read block that recovers `input_no_padding`.
    pub input_trim: Trim,
}

/// Generate block indices tiling `arr_shape` with `block_shape`-sized tiles.
///
/// Port of dolphin's `iter_blocks`. `overlaps` shrinks the step between tiles;
/// `start_offsets`/`end_margin` skip a border on the leading/trailing edges.
/// Returns an empty `Vec` for degenerate (zero-size) blocks or margins that
/// consume the whole array.
#[must_use]
pub fn iter_blocks(
    arr_shape: (usize, usize),
    block_shape: (usize, usize),
    overlaps: (usize, usize),
    start_offsets: (usize, usize),
    end_margin: (usize, usize),
) -> Vec<BlockIndices> {
    let (height, width) = block_shape;
    if height == 0 || width == 0 {
        return Vec::new();
    }
    let last_row = arr_shape.0.saturating_sub(end_margin.0);
    let last_col = arr_shape.1.saturating_sub(end_margin.1);
    let row_starts = cursors(start_offsets.0, height, overlaps.0, last_row);
    let col_starts = cursors(start_offsets.1, width, overlaps.1, last_col);

    row_starts
        .iter()
        .flat_map(|&row| col_starts.iter().map(move |&col| (row, col)))
        .map(|(row, col)| BlockIndices {
            row_start: row,
            row_stop: (row + height).min(last_row),
            col_start: col,
            col_stop: (col + width).min(last_col),
        })
        .collect()
}

/// Tiling cursor positions from `start` up to `last`, stepping by `size` and
/// backing off by `overlap` whenever another tile still fits.
fn cursors(start: usize, size: usize, overlap: usize, last: usize) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut cur = start;
    while cur < last {
        positions.push(cur);
        let next = cur + size;
        cur = if next < last { next - overlap } else { next };
    }
    positions
}

/// `round(num / den)` with round-half-to-even, matching Python's `round`.
fn round_half_even(num: usize, den: usize) -> usize {
    let quotient = num / den;
    let twice_rem = 2 * (num % den);
    match twice_rem.cmp(&den) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal if quotient.is_multiple_of(2) => quotient,
        std::cmp::Ordering::Equal => quotient + 1,
    }
}

/// Handles slicing/trimming of overlapping, decimated processing blocks.
///
/// Port of dolphin's `StridedBlockManager`.
#[derive(Debug, Clone, Copy)]
pub struct StridedBlockManager {
    arr_shape: (usize, usize),
    block_shape: (usize, usize),
    strides: Strides,
    half_window: HalfWindow,
}

impl StridedBlockManager {
    /// Build a manager for an `arr_shape` raster, processed in `block_shape`
    /// full-resolution tiles, decimated by `strides`, with a `half_window` halo.
    #[must_use]
    pub fn new(
        arr_shape: (usize, usize),
        block_shape: (usize, usize),
        strides: Strides,
        half_window: HalfWindow,
    ) -> Self {
        Self {
            arr_shape,
            block_shape,
            strides,
            half_window,
        }
    }

    /// Output-grid shape after striding.
    #[must_use]
    pub fn output_shape(&self) -> (usize, usize) {
        self.strides.out_shape(self.arr_shape)
    }

    /// Output-grid border (in output pixels) ignored on every edge as nodata.
    #[must_use]
    pub fn output_margin(&self) -> (usize, usize) {
        (
            round_half_even(self.half_window.y, self.strides.y),
            round_half_even(self.half_window.x, self.strides.x),
        )
    }

    /// Extra full-resolution padding each input block carries for the halo.
    fn input_padding(&self) -> (usize, usize) {
        let margin = self.output_margin();
        (self.strides.y * margin.0, self.strides.x * margin.1)
    }

    /// Constant output trim that drops the nodata halo from each kernel output.
    fn out_trim(&self) -> Trim {
        let margin = self.output_margin();
        Trim {
            row_start: margin.0,
            row_end_trim: margin.0,
            col_start: margin.1,
            col_end_trim: margin.1,
        }
    }

    /// Iterate the processing tiles covering the raster interior.
    #[must_use]
    pub fn iter_blocks(&self) -> Vec<TileBlocks> {
        let out_trim = self.out_trim();
        let margin = self.output_margin();
        let padding = self.input_padding();
        let out_block_shape = self.strides.out_shape(self.block_shape);

        iter_blocks(self.output_shape(), out_block_shape, (0, 0), margin, margin)
            .into_iter()
            .map(|out_block| self.expand(out_block, out_trim, padding))
            .collect()
    }

    /// Expand one output block into the full five-region tile.
    fn expand(
        &self,
        out_block: BlockIndices,
        out_trim: Trim,
        padding: (usize, usize),
    ) -> TileBlocks {
        let input_no_padding = dilate(out_block, self.strides);
        let input_block = pad(input_no_padding, padding);
        let input_trim = full_res_trim(input_no_padding, input_block);
        TileBlocks {
            out_block,
            out_trim,
            input_block,
            input_no_padding,
            input_trim,
        }
    }
}

/// Grow output-grid indices to full resolution by multiplying through strides.
fn dilate(block: BlockIndices, strides: Strides) -> BlockIndices {
    BlockIndices {
        row_start: block.row_start * strides.y,
        row_stop: block.row_stop * strides.y,
        col_start: block.col_start * strides.x,
        col_stop: block.col_stop * strides.x,
    }
}

/// Pad a block by `(row, col)` margins on every side.
fn pad(block: BlockIndices, margins: (usize, usize)) -> BlockIndices {
    BlockIndices {
        row_start: block.row_start - margins.0,
        row_stop: block.row_stop + margins.0,
        col_start: block.col_start - margins.1,
        col_stop: block.col_stop + margins.1,
    }
}

/// Trim that recovers `inner` from the padded `outer` block.
fn full_res_trim(inner: BlockIndices, outer: BlockIndices) -> Trim {
    Trim {
        row_start: inner.row_start - outer.row_start,
        row_end_trim: outer.row_stop - inner.row_stop,
        col_start: inner.col_start - outer.col_start,
        col_end_trim: outer.col_stop - inner.col_stop,
    }
}
