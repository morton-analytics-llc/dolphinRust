# Compressed-SLC reference indices, audited against dolphin v0.42.9

**Date:** 2026-11-09 · **Issue:** #176 · **Upstream:** dolphin PR #750 (`6edd7e1`, v0.42.9),
PR #588 (`df68577`, v0.39.0) · **Script:** `oracle/ref_idx_experiment.py`

## Result

1. **dolphinRust does not have the v0.42.9 bug.** No code path can put a compressed SLC
   anywhere but the front of a ministack, so "number of compressed SLCs − 1" is always the
   position of the last one. A contract test now pins that.
2. **The audit found a different gap in the same logic, filed as #192.** For the
   `always_first` plan dolphinRust references every ministack to the *oldest* carried
   compressed SLC. That is dolphin v0.35.0's rule. dolphin changed it to the *newest* in
   v0.39.0 because the old rule was "much noisier", and the v0.42.9 fix is a correction to
   that newer default. In dolphin's own kernels the old rule is 1.4–4× worse from the third
   ministack on wherever coherence decays slowly without a floor.

## 1. The v0.42.9 bug and why it cannot occur here

dolphin's wrapped-phase workflow picked the output reference as

```python
num_ccslc = sum(is_compressed)
ref_idx = cfg.phase_linking.output_reference_idx or max(0, num_ccslc - 1)
```

over a file list **sorted by date**, in which compressed SLCs are recognised by filename. A
count is only a position if every compressed SLC sorts first. PR #745 ("compressed SLC
forward", since reverted) let one fall between real SLCs, and PR #750 replaced the count with
a scan for the last compressed entry.

dolphinRust never has that list:

- **Compressed SLCs are not inputs.** There is no filename test and no date sort that mixes
  them with real acquisitions. A search of `crates/` for `is_compressed`, a `"compressed"`
  filename match and `first_real_slc_idx` finds only the phase-linking kernel's own
  parameter. The workflow rejects an update whose new granule repeats a prior date
  (`update_repeating_a_prior_acquisition_date_fails_before_cslc_io`, #140).
- **One function builds every combined ministack.** `sequential.rs::assemble` writes the
  last `num_compressed` carried compressed SLCs to positions `0..num_compressed` and the
  ministack's real SLCs after them. `drive` and `drive_with_covariance_capture` are its only
  callers, for full runs, resumed runs and updates alike.
- **Every index derived from the count is used against that layout:**
  `fused_params` sets `crlb_reference_idx = num_compressed − 1` (dolphin's
  `max(first_real_slc_idx − 1, 0)`) and `average_coherence_start_idx = num_compressed`;
  `link_and_compress` slices real dates and SHP samples from `num_compressed..`.
- **The plans.** `always_first` and `first_per_ministack` resolve to positions inside the
  same front-loaded layout. `last_per_ministack` carries no compressed SLC at all since #164
  (it is refused beyond one ministack), so the issue's worry about a compressed SLC "placed
  last" does not arise: that plan's name describes which *date* is the reference, not where a
  compressed SLC sits.

Contract test: `sequential::tests::compressed_slcs_always_lead_the_combined_ministack`. For
`always_first` and `first_per_ministack`, with and without `max_num_compressed` truncation
and on a resumed plan, it tags compressed and real layers, assembles every planned ministack,
and checks that the last compressed position found by scanning (dolphin's fix) equals the
count-derived index the driver uses.

## 2. The reference rule itself is v0.35's (#192)

`MiniStackPlanner::references` gives `always_first` the pair
`(output_reference_idx, output_reference_idx)`, and an unset
`phase_linking.output_reference_idx` becomes 0. Index 0 of a combined ministack is the oldest
carried compressed SLC. dolphin v0.35.0 did the same, and the planner and sequential oracle
fixtures pin it.

dolphin v0.39.0 (PR #588) moved both indices to `max(0, num_ccslc − 1)`, the newest compressed
SLC, with the note: "We want the `output_reference_idx` to move forward one each time when
doing `ALWAYS_FIRST`. Otherwise, the results look much noisier (though still roughly
correct)".

### Measurement

`oracle/ref_idx_experiment.py` runs one simulated stack through dolphin's `run_phase_linking`
and `compress` (v0.42.0 env) twice, changing only the two reference indices. Homogeneous
distributed scatterers with coherence `(1 − γ∞)·exp(−Δt/τ) + γ∞`; 60 dates at 12 days;
ministack 15; five compressed SLCs carried; half-window 5; 34×34 interior pixels; 5 seeds.
The true phase history is known, so the error is measured directly.

RMS phase error against truth, rad, by ministack. Ministacks 1 and 2 are identical under the
two rules by construction, because at most one compressed SLC is carried.

| τ (days), γ∞ | rule | ms 1 | ms 2 | ms 3 | ms 4 |
|---|---|---:|---:|---:|---:|
| 180, 0 | oldest (dolphinRust, dolphin v0.35) | 0.066 | 0.118 | 0.185 | 0.354 |
| | newest (dolphin ≥ 0.39) | 0.066 | 0.118 | 0.134 | 0.161 |
| 90, 0 | oldest | 0.104 | 0.212 | 0.529 | 1.270 |
| | newest | 0.104 | 0.212 | 0.254 | 0.316 |
| 45, 0.05 | oldest | 0.173 | 0.321 | 0.465 | 0.595 |
| | newest | 0.173 | 0.321 | 0.335 | 0.371 |
| 30, 0.1 | oldest | 0.189 | 0.271 | 0.253 | 0.274 |
| | newest | 0.189 | 0.271 | 0.247 | 0.269 |
| 20, 0 | oldest | 0.633 | 1.418 | 1.722 | 1.854 |
| | newest | 0.633 | 1.418 | 1.732 | 1.766 |

Spread across pixels of each ministack's mean error (rad), which is the per-pixel datum offset
a whole ministack carries and so the size of the step at its junction:

| τ (days), γ∞ | rule | ms 3 | ms 4 |
|---|---|---:|---:|
| 180, 0 | oldest / newest | 0.170 / 0.118 | 0.345 / 0.149 |
| 90, 0 | oldest / newest | 0.512 / 0.228 | 1.266 / 0.299 |
| 45, 0.05 | oldest / newest | 0.445 / 0.296 | 0.581 / 0.340 |
| 30, 0.1 | oldest / newest | 0.222 / 0.205 | 0.243 / 0.229 |
| 20, 0 | oldest / newest | 1.637 / 1.691 | 1.833 / 1.714 |

The old rule loses where coherence decays slowly and has no floor: the oldest compressed SLC
has decorrelated against the new dates and the newest has not. With a coherence floor (30 d,
γ∞ = 0.1) the two rules are within 2–8%. With no coherence left (20 d, γ∞ = 0) both fail.

### What is and is not established

- **Measured:** the tables, in dolphin's kernels.
- **Measured in Rust, one stack:** dolphinRust's `run_sequential` (`rect`, EMI, the same
  ministack and window settings) on the seed-0 stack of the τ = 90 d, γ∞ = 0 case gives
  0.1032, 0.2178, 0.4994 and 0.8990 rad by ministack. dolphin's kernels on that stack give
  0.1032, 0.2178, 0.4995, 0.8990 under the oldest rule and 0.1032, 0.2178, 0.2590, 0.3302
  under the newest. So the Rust driver is the "oldest" row, 1.9× and 2.7× worse than the
  newest rule in ministacks 3 and 4 there. The other rows of the tables are dolphin's
  kernels only; that the Rust driver follows them rests on that one check and on
  `sequential_contract`, which pins it to `oracle/gen_sequential.py` (`ALWAYS_FIRST`,
  `output_reference_idx=0`, v0.35.0).
- **Not established:** that this causes the junction steps of #184 and eo #758. #184 found
  that dolphin v0.35.0 "produces the same junction statistics on the same real crop". That
  comparison has the old rule on both sides, so it neither supports nor rules this out.

The fix and its consequences (the covariance operator requires `output_reference_idx = 0`;
every multi-ministack product changes from the third ministack on) are in #192. It was not
made under #176: it changes production output and needs the #184 junction statistics measured
before and after on real data.
