//! Shared windowing helpers for the SHP tests.
//!
//! Center-pixel windows are clamped to the image bounds (port of dolphin's
//! `utils._get_slices`): at borders the window shrinks rather than shifting, and
//! neighbor offsets index from the clamped start — matching dolphin's `is_shp`
//! layout exactly.

use ndarray::{Array2, Array4};

/// A window clamped to image bounds: `[r_start, r_end) x [c_start, c_end)`.
#[derive(Debug, Clone, Copy)]
pub struct Window {
    pub r_start: usize,
    pub r_end: usize,
    pub c_start: usize,
    pub c_end: usize,
}

/// Clamp a `half`-radius window around `center` to the `(rows, cols)` bounds.
#[must_use]
pub fn clamped_window(
    center: (usize, usize),
    half: dolphin_core::HalfWindow,
    shape: (usize, usize),
) -> Window {
    Window {
        r_start: center.0.saturating_sub(half.y),
        r_end: (center.0 + half.y + 1).min(shape.0),
        c_start: center.1.saturating_sub(half.x),
        c_end: (center.1 + half.x + 1).min(shape.1),
    }
}

/// Iterate window cells as `(row, col, row_offset, col_offset)`, where offsets
/// are measured from the clamped window start.
pub fn neighbor_grid(w: Window) -> impl Iterator<Item = (usize, usize, usize, usize)> {
    (w.r_start..w.r_end)
        .flat_map(move |r| (w.c_start..w.c_end).map(move |c| (r, c, r - w.r_start, c - w.c_start)))
}

/// Assemble per-pixel `(win_h, win_w)` slabs into an `(out_rows, out_cols, win_h, win_w)` array.
#[must_use]
pub fn stack_slabs(slabs: Vec<Array2<bool>>, shape: (usize, usize, usize, usize)) -> Array4<bool> {
    let flat: Vec<bool> = slabs
        .into_iter()
        .flat_map(IntoIterator::into_iter)
        .collect();
    Array4::from_shape_vec(shape, flat).expect("slab assembly shape mismatch")
}
