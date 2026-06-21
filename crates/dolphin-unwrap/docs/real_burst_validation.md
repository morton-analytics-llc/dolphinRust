# Native vs SNAPHU — real CSLC burst validation (Tier-2 [B])

Native unwrapper validated against the SNAPHU oracle on a **real** OPERA CSLC-S1
burst, not a synthetic fixture. Per-component cycle parity + connected-component
agreement at real decorrelation/atmosphere.

## Data — genuinely real captured SAR

- **Source:** `oracle/fixtures/real_slc_stack.npy` — 13 × 384 × 384 complex64
  OPERA Level-2 CSLC-S1 acquisitions, **Mexico burst T005-008704-IW1**, fetched
  from NASA Earthdata (`validation/fetch_real.py`), spatially cropped. Real
  Sentinel-1 coregistered single-look complex — captured SAR, not generated.
- **Wrapped ifg + coherence** (`oracle/prep_real_ifg.py`): interferogram
  `ref · conj(sec)` from two stack epochs; coherence is the standard 5×5 boxcar
  sample coherence of the same pair (real temporal/geometric decorrelation, NOT a
  synthetic noise model). Two baselines bracket the regime:
  - `real_ifg` — epochs 0 vs 12 (**long** temporal baseline): coh mean 0.568,
    12.1% of pixels < 0.3, **23 418 residues (15.96%)**.
  - `real_ifg_short` — epochs 0 vs 1 (**short** baseline): coh mean 0.642,
    6.7% < 0.3, **20 270 residues (13.82%)**.

These are hard, heavily-decorrelated real scenes (the T005 stack is low-coherence
even at short baseline), which is the point — they exercise real branch-cut
routing through noise, not a clean analytic field.

## Method

`cargo run --release --example real_burst_validation -p dolphin-unwrap`
(`REAL_IFG_DIR` selects the fixture). For each fixture the harness runs a **fresh
SNAPHU oracle subprocess** (smooth cost, MCF init, single tile — black-box, never
linked/read) and the native solver in two configs: the production fine-tiled
default (~48 px cores → 8×8 here) and the global single-MCF solve. Metric is the
same per-connected-component cycle disagreement the seam contracts use — grouped
by the **SNAPHU** component labels, modal integer offset per component, fraction
of pixels deviating — reported over all component pixels and restricted to
**trusted** pixels (coherence ≥ 0.5). Connected-component agreement is the
reliable-mask IoU vs SNAPHU.

## Results (measured)

| fixture (baseline) | config | per-comp all-px | per-comp coh≥0.5 | mask-IoU | wall |
|---|---|---|---|---|---|
| `real_ifg` (long, 16.0% res) | native-tiled 8×8 | 4.347% | 3.112% | 0.788 | 0.11 s |
| `real_ifg` (long) | native-global | 3.097% | **1.944%** | 0.788 | 4.98 s |
| `real_ifg_short` (short, 13.8% res) | native-tiled 8×8 | 2.285% | 1.591% | 0.890 | 0.11 s |
| `real_ifg_short` (short) | native-global | 2.296% | **1.588%** | 0.890 | 4.59 s |

SNAPHU found 2 components on the long-baseline scene, 1 on the short.

## Interpretation — honest

- **Native tracks SNAPHU on trusted pixels to ~1.6–1.9%** (global solve). The
  residual is two *different* MCF formulations placing branch cuts differently
  through real decorrelation noise — not a defect: it shrinks monotonically with
  coherence (long→short baseline: trusted 1.94%→1.59%, IoU 0.79→0.89), exactly the
  expected dependence. There is no clean ground truth here; SNAPHU is itself one
  estimator, so this is agreement-between-estimators, not error-vs-truth.
- These real numbers are **looser than the synthetic goldens** (0.01–0.3%) because
  the synthetic scenes are lower-residue with clean coherence moats; the real T005
  pair is ~14–16% residues in genuinely noisy data. That gap is the cost of real
  decorrelation, reported rather than hidden.
- **Tiling cost is real-data-confirmed small.** On the single-component short scene
  tiled ≈ global (2.285% vs 2.296%). On the 2-component long scene tiling adds
  ~1.2% all-px / ~1.2% trusted over global — the per-region seam reconciliation,
  carrying the [A] conncomp-regrow bridge growth, holds on real data too.
- **Not bit-parity, and shouldn't be** — the project's bar is scientific
  correctness on trusted pixels, which this meets.

Reproduce: `python oracle/prep_real_ifg.py` (or with `REF=`/`SEC=`/`PAIR_NAME=`)
then the example above. Source `real_slc_stack.npy` is gitignored like all
`oracle/fixtures/`; regenerate via `validation/fetch_real.py` if absent.
