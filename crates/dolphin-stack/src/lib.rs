//! Ministack planning for the Ansari et al. (2017) sequential estimator —
//! port of `dolphin/stack.py` `MiniStackPlanner`. Pure planning logic, no
//! numerics.
//!
//! A stack of `num_slc` real SLCs is partitioned into `ministack_size` batches.
//! Each ministack compresses to one SLC, carried forward as the leading
//! element(s) of later ministacks (up to `max_num_compressed`). The
//! [`CompressedSlcPlan`] sets the reference-index convention.
#![warn(missing_docs)]

use dolphin_core::config::CompressedSlcPlan;

/// One planned ministack: prepended compressed SLCs followed by real SLCs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniStack {
    /// Global zero-based block identifier, including any resumed prefix.
    pub block_id: usize,
    /// Number of prior compressed SLCs prepended to this ministack.
    pub num_compressed: usize,
    /// Index (into the real SLC list) of this ministack's first real SLC.
    pub real_start: usize,
    /// Number of real SLCs in this ministack.
    pub num_real: usize,
    /// Reference index for phase-linking output (may be -1 = last).
    pub output_reference_idx: isize,
    /// Reference index for compressed-SLC creation (may be -1 = last).
    pub compressed_reference_idx: isize,
}

impl MiniStack {
    /// Total SLCs in the ministack (compressed + real).
    #[must_use]
    pub fn size(&self) -> usize {
        self.num_compressed + self.num_real
    }

    /// Global block IDs whose compressed SLCs are carried into this block.
    ///
    /// The range is oldest to newest and already reflects
    /// [`MiniStackPlanner::max_num_compressed`] eviction.
    pub fn carried_parent_ids(&self) -> std::ops::Range<usize> {
        self.block_id - self.num_compressed..self.block_id
    }

    /// Resolve and validate the phase-link output reference for this block.
    ///
    /// # Errors
    /// Returns `Err` when the reference is less than `-1` or outside the
    /// combined compressed-plus-real stack.
    pub fn resolved_output_reference_idx(&self) -> Result<usize, &'static str> {
        resolve_reference_index(self.output_reference_idx, self.size())
    }

    /// Resolve and validate the compression reference for this block.
    ///
    /// # Errors
    /// Returns `Err` when the reference is less than `-1` or outside the
    /// combined compressed-plus-real stack.
    pub fn resolved_compressed_reference_idx(&self) -> Result<usize, &'static str> {
        resolve_reference_index(self.compressed_reference_idx, self.size())
    }
}

/// Resolve a dolphin-style reference index against `len`.
///
/// `-1` selects the last element. Other negative values, an empty input, and
/// non-negative indices outside the input fail before a consumer can index.
///
/// # Errors
/// Returns `Err` when `reference_idx` cannot identify an element of `len`.
pub fn resolve_reference_index(reference_idx: isize, len: usize) -> Result<usize, &'static str> {
    if len == 0 {
        return Err("cannot resolve a reference in an empty ministack");
    }
    if reference_idx == -1 {
        return Ok(len - 1);
    }
    let resolved =
        usize::try_from(reference_idx).map_err(|_| "reference index must be -1 or non-negative")?;
    if resolved >= len {
        return Err("reference index is outside the ministack");
    }
    Ok(resolved)
}

/// Plans the sequence of ministacks for a stack of real SLCs.
#[derive(Debug, Clone, Copy)]
pub struct MiniStackPlanner {
    /// Number of real SLCs in the full stack.
    pub num_slc: usize,
    /// Cap on the number of compressed SLCs carried into any ministack.
    pub max_num_compressed: usize,
    /// Default phase-linking output reference index.
    pub output_reference_idx: isize,
    /// Compressed-SLC carry-forward convention.
    pub compressed_slc_plan: CompressedSlcPlan,
}

impl MiniStackPlanner {
    /// Partition the stack into ministacks of `ministack_size` real SLCs each,
    /// resolving the compressed carry-forward and reference indices.
    ///
    /// # Errors
    /// Returns `Err` if `ministack_size < 2` (dolphin's minimum).
    pub fn plan(&self, ministack_size: usize) -> Result<Vec<MiniStack>, &'static str> {
        self.plan_with_offset(ministack_size, 0)
    }

    /// Plan ministacks for a stack that **resumes** an earlier run, where
    /// `batch_offset` ministacks have already been sealed and compressed. The
    /// batch index (which sets `num_compressed` and the reference indices) is
    /// shifted by `batch_offset` so the carried-compressed accounting continues
    /// the prior sequence, while `real_start` stays relative to this (tail) stack.
    /// `plan` is the special case `batch_offset = 0`.
    ///
    /// The resumed tail must begin on a ministack boundary — guaranteed because a
    /// sealed ministack is always full, so `batch_offset · ministack_size` real
    /// SLCs precede the tail.
    ///
    /// # Errors
    /// Returns `Err` if `ministack_size < 2` (dolphin's minimum).
    pub fn plan_with_offset(
        &self,
        ministack_size: usize,
        batch_offset: usize,
    ) -> Result<Vec<MiniStack>, &'static str> {
        if ministack_size < 2 {
            return Err("cannot create ministacks with size < 2");
        }
        if self.max_num_compressed == 0
            && self.num_slc > 0
            && (batch_offset > 0 || self.num_slc > ministack_size)
        {
            return Err("multi-ministack plans require at least one carried compressed SLC");
        }
        let ministacks = (0..self.num_slc)
            .step_by(ministack_size)
            .enumerate()
            .map(|(batch, start)| self.batch(batch + batch_offset, start, ministack_size))
            .collect::<Vec<_>>();
        for ministack in &ministacks {
            ministack.resolved_output_reference_idx()?;
            ministack.resolved_compressed_reference_idx()?;
        }
        Ok(ministacks)
    }

    /// Build the `batch`-th ministack starting at real-SLC index `start`.
    /// `batch` equals the number of compressed SLCs produced so far.
    fn batch(&self, batch: usize, start: usize, ministack_size: usize) -> MiniStack {
        let num_real = ministack_size.min(self.num_slc - start);
        let num_compressed = batch.min(self.max_num_compressed);
        let (output_reference_idx, compressed_reference_idx) = self.references(num_compressed);
        MiniStack {
            block_id: batch,
            num_compressed,
            real_start: start,
            num_real,
            output_reference_idx,
            compressed_reference_idx,
        }
    }

    /// Resolve `(output_reference_idx, compressed_reference_idx)` for a ministack
    /// carrying `num_compressed` compressed SLCs, per the plan.
    fn references(&self, num_compressed: usize) -> (isize, isize) {
        let ncc = num_compressed as isize;
        match self.compressed_slc_plan {
            CompressedSlcPlan::AlwaysFirst => {
                (self.output_reference_idx, self.output_reference_idx)
            }
            CompressedSlcPlan::FirstPerMinistack => (self.output_reference_idx, ncc),
            CompressedSlcPlan::LastPerMinistack => (ncc - 1, -1),
        }
    }
}
