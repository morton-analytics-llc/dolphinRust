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
        let ministacks = (0..self.num_slc)
            .step_by(ministack_size)
            .enumerate()
            .map(|(batch, start)| self.batch(batch + batch_offset, start, ministack_size))
            .collect();
        Ok(ministacks)
    }

    /// Build the `batch`-th ministack starting at real-SLC index `start`.
    /// `batch` equals the number of compressed SLCs produced so far.
    fn batch(&self, batch: usize, start: usize, ministack_size: usize) -> MiniStack {
        let num_real = ministack_size.min(self.num_slc - start);
        let num_compressed = batch.min(self.max_num_compressed);
        let (output_reference_idx, compressed_reference_idx) = self.references(num_compressed);
        MiniStack {
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
