# Issue #207 test design: why half the frame steps a whole cycle at junctions 45 and 75

**Frozen 2026-11-12**, in the commit that adds this file, after the #192 inputs were
re-fetched and before any arm below was run or scored. It freezes the measurements, the
statistics and the decision rules. Engine configuration is that of #192
(`validation/issue192_houston_recipe.json`) except for the one variable each arm changes.
Nothing below is changed after results are seen.

## Finding under test

On Houston burst T034-071054-IW3 (#192 receipt
`validation/results/issue192_reference_rule/houston_t034_071054_iw3_20261110/receipt.json`)
the engine's `ministack_junctions` diagnostic reports that 51% (index 45) and 37% (index 75)
of evaluable cells step a non-zero whole number of cycles from their connected component's
median, against 0.03–4% at the other junctions. The #184 window rule flags ~400 cells.
Velocity agrees with OPERA DISP-S1 (median |diff| 0.41 mm/yr).

Three candidate causes:

- **(a)** the step field at those dates really spans more than a cycle across the frame (two
  populations, or one smooth field wider than a cycle), so the component median sits away
  from much of the frame and the diagnostic reports a real signal as unwrap integers;
- **(b)** an unwrapping disagreement at those junctions: the native unwrapper, or
  connected-component handling;
- **(c)** a defect in the diagnostic's arithmetic (indexing, labels, median, rounding).

## Held fixed

Same 91 granules (plan `catalog_sha256` 9226039c…, window row0 2704, col0 1266, 699 × 3996),
same recipe engine configuration, binary built from main 7721d4e, arm `always_first` unless
stated. #192 showed the two `always_first` rules give the same fractions, so the reference
rule is not a variable here.

## Statistics

For acquisition index `k` (1..90), `step_k` is the per-cell series difference
`phi_k − phi_{k−1}` in cycles (the inverted series; from the published
`displacement_*.tif` converted at λ/2 per cycle, with acquisition 0 as zero). Spatial
referencing subtracts one constant per date and cancels in every statistic below; the
configuration applies no correction.

- **J(k)** — the diagnostic as documented: fraction of cells with
  `|step_k − median(step_k over the cell's component)| > 0.5`, component labels from the
  `(k−1, k)` interferogram's `conncomp_*.tif`.
- **J1(k)** — J(k) with every finite cell in one component.
- **Itoh(k)** — fraction of 4-neighbour cell pairs, both finite and in the same component,
  with `|Δ step_k| > 0.5` cycle. A whole-cycle split of the frame into two populations by
  an unwrapping error requires a boundary of such jumps; a correctly unwrapped field
  wider than a cycle needs none.
- **L(k)** — `cycle_steps.per_date[k].fraction_over_half_cycle` from the engine's own
  provenance: the same `> 0.5` test against a same-component neighbourhood median
  (the #184 junction neighbourhood at junctions, 5 × 5 elsewhere).
- **Range(k)** — 98th minus 2nd percentile of `step_k` over finite cells, cycles.

## Measurements, in order

**M0 — reproduction.** Arm A0 (`always_first`, native) on the fetched inputs. Pass: the
engine's per-junction `fraction_integer_cycle` is within 0.01 of the #192 `new` arm at all
six junctions. If not, stop and report; nothing else is scored.

**M1 — the diagnostic's arithmetic, (c).** Recompute J(k) at the six junctions offline from
A0's published rasters and compare with the engine's provenance and with its
`junction_step_cycles_KK.tif`. Pass: |offline − engine| ≤ 0.005 for every junction, and
the median per-cell |offline step − median − junction raster| ≤ 0.01 cycle. A failure is
cause (c).

**M2 — is it junction-specific?** J(k), J1(k), Itoh(k), L(k) and Range(k) at all 90 steps
of A0. Reported as the junction values against the distribution over the 84 non-junction
steps (median, 95th percentile, max).

**M3 — the unwrapper, (b).** Arm A1 = A0 with `unwrap_options.unwrap_method: snaphu`, the
one change. The native unwrapper is ruled out as the cause if A1's engine
`fraction_integer_cycle` at 45 and at 75 is within 0.05 of A0's; it is the cause if A1 is
below 0.05 at both while A0 is not.

**M4 — an independent processor, (a) vs (b).** OPERA DISP-S1 frame F08882 v1.0 (the
granules behind the #192 velocity reference), range-read over this crop, resampled onto the
output grid by nearest neighbour as the #192 scorer does. For every `k` whose two
acquisitions share one OPERA reference date (84 of 90 steps; 45 and 75 qualify, since
OPERA's references change after 45 and 75), the OPERA step is the difference of the two
granules' `displacement`, on cells with `recommended_mask = 1` in both and finite in A0.
The sign that relates the two products is fixed as the sign of the median Pearson
correlation between A0 and OPERA steps over the non-junction comparable steps.

- **J_O(k)** — J1(k) of the OPERA step.
- **I(k)** — fraction of common cells with
  `|step_A0 − step_O − median(step_A0 − step_O)| > 0.5` cycle: a whole-cycle
  disagreement between the two processors.

**M5 — component handling, (b).** J(k) against J1(k) at 45 and 75 in A0: if labels split the
frame, the per-component median is the variable; if `J1 ≈ J` the labels are not involved.

## Decision rule

Taken in this order; the first that holds is the finding.

1. M0 fails → no finding; report the reproduction gap.
2. M1 fails → **(c)**.
3. I(45) > 0.05 or I(75) > 0.05, while the median non-junction I ≤ 0.05 → **(b)**: A0
   disagrees with an independent unwrapping by whole cycles at those junctions. M3 and M5
   then say which engine step (native unwrapper if M3 says so, component handling if M5
   does).
4. I(45) ≤ 0.05 and I(75) ≤ 0.05, and J_O(45), J_O(75) are each at least half of A0's
   J1 at the same step → **(a)**: OPERA, unwrapped independently, carries the same
   whole-cycle spread across the frame at those dates, so the spread is in the data and
   the diagnostic's component-median reference misreports it.
5. Anything else → no finding; report the measurements as they stand.

## What each outcome triggers

- **(c)** fix the arithmetic; the corrected engine must reproduce M1 offline values.
- **(b)** fix the located engine step; Houston must then be not worse than #192 on the
  frozen velocity metrics (median |velocity − OPERA| ≤ 0.414 + 0.1 mm/yr, fraction within
  1.5 mm/yr ≥ 0.935 − 0.01), and J at 45 and 75 must fall to the non-junction 95th
  percentile from M2.
- **(a)** correct the diagnostic so it measures each cell's junction step against a
  reference that a field wider than a cycle cannot move: the same-component neighbourhood
  median at the #184 junction half-window. Acceptance for the corrected diagnostic, on A0
  re-run with the fix: the per-junction integer fraction at 45 and 75 is no larger than
  the 95th percentile of the same statistic over non-junction steps from M2 (L(k)), and
  the existing unit test of a single injected cycle at a junction still reports exactly that
  cell. Published displacement must be bit-identical, since only the diagnostic changes.

Receipt: `validation/results/issue207_junction_steps/houston_t034_071054_iw3_<date>/receipt.json`.

## Addendum, 2026-11-12, after M0 and M1 were first computed

Recorded here rather than edited above, because it was found by running the scorer. Two
defects in the **scorer**, not in the engine, and one in how M1 was written:

1. The scorer converted published displacement to cycles at `+λ/2`; the engine publishes
   `−λ/(4π)` m per radian, so a cycle is `−λ/2`. With the wrong sign every offline step
   was negated and the per-cell M1 difference read ~2 × |step − median| (0.998 cycle at 45).
   Fixed in the scorer.
2. M1 compared the offline fraction, computed on the 127,692 cells the published product
   keeps, with the provenance fraction, computed by the engine on the ~130,000 cells of the
   raw series before the publication mask. The difference between the two populations
   shifts each component median by a constant (0.0088 cycle at 45; the per-cell difference
   is that constant at the median and at the 99th percentile alike), and moves the fraction
   at 45 from 0.5004 to 0.5070. M1 now compares the offline fraction with the engine's own
   `junction_step_cycles_KK.tif` fraction on the same cells; the provenance value is kept in
   the receipt beside it. The thresholds (0.005 on the fraction, 0.01 cycle per cell) are
   unchanged.

## Result, 2026-11-12

Receipt `validation/results/issue207_junction_steps/houston_t034_071054_iw3_20261112/receipt.json`.
Finding by the decision rule: **(a)**.

- M0: the engine reproduces #192 exactly (0.5070 at 45, 0.3739 at 75).
- M1 (after the addendum): offline and engine agree to ≤ 0.0024 in fraction and ≤ 0.0088
  cycle per cell. Pass.
- M2: the large fractions are not junction-specific. Non-junction steps 44, 46, 47, 74, 76
  read 0.415, 0.638, 0.224, 0.260, 0.378. Step 46 has the widest field (4.25 cycles from
  the 2nd to the 98th percentile). Itoh(k) stays ≤ 0.0007 at every step, so no boundary of
  whole-cycle jumps divides the frame.
- M3: SNAPHU in place of the native unwrapper gives 0.536 and 0.415 (native: 0.507, 0.374).
  So the native unwrapper is ruled out.
- M4: OPERA DISP-S1 steps correlate with A0 at r = 0.998 (45) and 0.998 (75), and OPERA's
  own J_O is 0.497 and 0.362. Only 0.08% and 0.05% of cells differ from A0 by a whole cycle.
- M5: J = J1 at 45 and 75, so the component labels play no part.

Fix: the junction diagnostic now measures against the same-component neighbourhood median at
the #184 junction half-window (6 px here). Published displacement is bit-identical, and the
#192 velocity metrics are unchanged to the last digit. The fraction at 45 is now 0.00059 and
at 75 0.00044, in line with the other junctions (0.00021–0.00080).

**The acceptance bound fails and stays as written.** 0.00059 and 0.00044 are above the
non-junction p95 of L (0.000083). That holds like for like too: with the junction window
applied at every step, the non-junction p95 is 0.000153 and five of six junctions exceed it.
The cells left over are exactly the #184-flagged cells at every junction (24–101 each). That
excess is a separate finding, filed as #216.
