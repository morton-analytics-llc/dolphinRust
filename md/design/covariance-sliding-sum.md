# Design — covariance sliding-sum (separable box filter)

## Goal
Cut the phase-linking covariance cost (now the 72–84% e2e bottleneck's dominant
sub-stage) by replacing the direct per-window Hermitian sum with a **row-separable
box-sum** over the per-pair cross-products `p_ij = z_i · conj(z_j)`. Overlapping
windows (dolphin defaults ~23×11 window, 6×3 strides) recompute each sample's
products ~14×; the separable kernel removes the horizontal redundancy.

## Scope (locked)
- **Unmasked rectangular path only** (`neighbors: None`) — confirmed the entire
  production path (`run_sequential` → `link_and_compress` → `engine.link(.., None, ..)`,
  and `phase_link_tiled`). The SHP-masked path (`neighbors: Some`) keeps the current
  direct per-pixel kernel unchanged.
- **This pass: horizontal separability** (~3.8× = win_w/strides.x) via per-output-row
  vertical partial sums reused across the row's output cols, each window summed
  directly in fixed left-to-right order. Vertical cross-row incremental (the remaining
  ~3.7×) is an explicit follow-up, NOT in this pass — keep it simple.

  > **Implementation note (deviation from first draft):** the original draft used
  > horizontal **prefix-subtraction** (`hpref[c0+win_w] - hpref[c0]`). That carries a
  > row-wide accumulator whose FP rounding depends on the block's column count, which
  > breaks `tiled_phase_link_is_bit_identical_to_whole_burst` (~1e-16, block width ≠
  > whole-burst width). Replaced with a **direct fixed-order window sum** over the
  > shared vertical sums (`window_sum` folds `vsum[c0..c0+win_w]` left-to-right): a
  > window's numerator depends only on its own samples, so tiled==whole and
  > fused==staged stay bit-identical while the ~3.8× vertical-reuse win is retained.

## Correctness bar (locked)
- Sliding vs direct: **tolerance ~1e-4 coherence** (not bit-identical — running sums
  reorder FP accumulation and subtract). This matches the crate contract
  (coherence ~1e-4, phase ~1e-3). Approved explicitly.
- fused vs staged: **stays bit-identical** — achieve this by routing BOTH the staged
  (`estimate_stack_covariance`) and fused (`link_fused`) unmasked paths through the
  ONE shared sliding kernel. `fused_contract.rs`'s bit-identical assertions compare
  fused-vs-staged, so identical kernel ⇒ identical bits ⇒ still green.
- The `direct` kernel is retained as the tolerance oracle (test-only consumer) and as
  the masked-path implementation.

## Memory (load-bearing)
Do NOT reintroduce the `nslc²·area` cube that Lever-1 fusion just removed. The
separable kernel holds only per-output-row buffers: `vsum[npairs][cols]` and
`hpref[npairs][cols+1]` (npairs = nslc(nslc+1)/2), one set per rayon row-task.
Parallelize over **output rows** (each row independent: own r0, own buffers).

## Kernel (per output row `orow`, parallel over rows)
1. `r0 = window_origin_row(orow)` — depends only on `orow` (clamped inward, matches
   current `window_origin`). Refactor `window_origin` into `window_origin_row` +
   `window_origin_col` axis helpers so there is one source of clamp truth.
2. `vsum[p][c] = Σ_{r=r0..r0+win_h} finite_or_zero(z_i[r][c])·conj(finite_or_zero(z_j[r][c]))`
   for every input column `c` and every Hermitian pair `p=(i≤j)`.
3. For each output col `ocol`: `c0 = window_origin_col(ocol)`;
   `numer[p] = Σ_{c=c0}^{c0+win_w-1} vsum[p][c]` summed left-to-right (fixed order,
   block-width-independent); expand to the Hermitian `n×n` matrix (`numer[j][i]=conj`),
   then reuse the existing `normalize`.

The `finite_or_zero` masking and the `normalize` (amplitude floor `AMP_FLOOR=1e-6`)
must match `coh_mat` exactly. Only the accumulation ORDER differs.

## Files
- `crates/dolphin-phaselink/src/covariance.rs`
  - Rename the current assemble path to `pub fn estimate_stack_covariance_direct(..)`
    (same signature) — masked-path impl + test oracle.
  - `estimate_stack_covariance`: `neighbors.is_some()` → direct; else → sliding.
  - Add `sliding` kernel + `window_origin_row` / `window_origin_col` helpers; keep
    functions small (≤ ~40 lines, ≤3 nesting — hook-enforced).
- `crates/dolphin-phaselink/src/fused.rs`
  - Refactor `fused_pixel` so the estimator/quality/crlb/closure consumers take a
    precomputed coherence matrix (`fused_from_coh(coh, params)`).
  - `link_fused`: masked → current flat per-pixel path (calls `pixel_coh`); unmasked →
    parallel over output rows, sliding numerators → `fused_from_coh`. Identical `idx`
    ordering / packing so output layout is unchanged.
- `crates/dolphin-phaselink/src/lib.rs` — re-export `estimate_stack_covariance_direct`
  if the test needs the crate-root path (test imports `covariance::` directly, so may
  not be required).

## Tests / verify (definition of done)
- `crates/dolphin-phaselink/tests/covariance_sliding_contract.rs` (already written,
  currently red) goes GREEN.
- `crates/dolphin-phaselink/tests/fused_contract.rs` stays GREEN (bit-identical).
- `cargo fmt`, `cargo clippy --all-targets -- -D warnings` clean.
- `cargo test -p dolphin-phaselink` and `cargo test -p dolphin-workflows` all green
  (the tiled bit-identical test `tiled_phase_link_is_bit_identical_to_whole_burst`
  must still pass — both tiled and whole go through the same sliding kernel).
