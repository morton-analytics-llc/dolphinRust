//! Ministack planning — port of `dolphin/stack.py`.
//!
//! `MiniStackPlanner`/`MiniStackInfo`/`CompressedSlcInfo`: partition the SLC
//! archive into batches of `ministack_size`, plan compressed-SLC carry-forward
//! (`ALWAYS_FIRST`) and the `max_num_compressed` cap for the Ansari et al.
//! (2017) sequential estimator. Pure planning logic, no numerics.
