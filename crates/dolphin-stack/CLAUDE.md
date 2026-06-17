# dolphin-stack — ministack planning (reference: `dolphin/stack.py`)

## Domain
Ansari et al. (2017) sequential estimator — pure planning logic, no numerics.
- `MiniStackPlanner` partitions N dates into `ministack_size` (default 15) batches.
- Each ministack compresses to one SLC (dolphin-phaselink), carried forward as the first
  element of the next ministack (`CompressedSlcPlan::AlwaysFirst`).
- `max_num_compressed` (default 10) caps accumulated compressed SLCs.

## Contracts
- Unit-test the planner against the published sequential scheme for several
  `(N, ministack_size)` combinations.
