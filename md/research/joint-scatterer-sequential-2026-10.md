# Sequential Joint-Scatterer InSAR (S-JSInSAR): scope (issue #110)

**Date:** 2026-10-31
**Question:** What would adopting S-JSInSAR's "unified PS+DS sequential estimation" involve in
dolphinRust? This doc is scope only, per the issue. It does not commit to a build.
**Answer:** The issue misreads the paper. S-JSInSAR has no joint PS+DS estimator. Its three
parts either conflict with dolphinRust's architecture or duplicate work already tracked
elsewhere.

## 1. What the paper does (read from the full text)

Zhang, Duan, Hu, Chai, Yun, Lv, "The Sequential Joint-Scatterer InSAR for Sentinel-1 Long-Term
Deformation Estimation", *Remote Sens.* 18(2):329, published 2026-01-19, doi:10.3390/rs18020329.
Metadata came from Crossref; the PDF came from mdpi-res.com, since mdpi.com returned 403.

- **"Joint scatterer" means a spatial patch, not PS+DS.** The JS vector stacks the pixels of an
  l1×l2 sub-window (3×3 in the experiment) for every date (§2.1, Eq. 1). SHPs are found by a
  two-sample KS test on the patch-mean amplitude vector (Eqs. 2–3). The covariance is computed
  over JS vectors (Eqs. 4–5), KS-weighted (Eq. 6), and phase-linked by MLE (Eq. 7). JS-InSAR [27]
  is "an extension of SqueeSAR".
- **The batches are scene-wide subsets with overlap, not compressed SLCs.** One average JS
  coherence matrix, built from bright patches, sets the subset boundaries. Each subset has at
  least 20 images, and adjacent subsets overlap by 4–6 images (§2.2). In the experiment that gave
  4 subsets of 35/34/49/38 scenes. Each subset is solved independently. The subset deformation
  series are then joined by least squares on a joint design matrix, with the overlaps tying them
  together (§2.3).
- **PS and DS meet only in a PSI network.** PS candidates come from the amplitude dispersion
  index. A Delaunay network is built on PS alone, then a local DS network inside each triangle,
  referenced to PS (§2.3, Fig. 4). This is classic PSI integration. The paper's introduction
  traces DS-into-PSI integration to Ferretti's 2011 DS method (SqueeSAR). There is no single estimator handling both.
- **Reported numbers** (145 S1 ascending images, Tangshan, 2018-01-22 to 2023-11-16, i7 with
  128 GB RAM):
  - Run time: PSI 1.9 h, JS-InSAR 28.4 h, S-JSInSAR 12.5 h, their "Sequential-Estimator" 16.7 h.
  - Points: PSI 1,846,443; SE 2,939,011; S-JSInSAR 3,172,065.
  - Against 20 leveling points: RMSE 2.88 mm/y, max error 9.71 mm/y.
  - Their SE is not dolphin's, and no leveling comparison is reported for SE.

## 2. Scope: what each part would mean for dolphinRust

| S-JSInSAR part | dolphinRust counterpart | fit |
|---|---|---|
| JS patch vectors (3×3 composite pixel) | per-pixel SHP covariance over a (7,14) half-window, `dolphin-shp` + `dolphin-phaselink` | Spatial pre-averaging inside an SHP window that already averages. It changes the estimand, and there is no reference toolchain to validate against. |
| KS on patch-mean amplitude | `ShpMethod::Ks` / `Glrt` on per-pixel amplitudes | Modest. It would be a new `ShpMethod`, not an architectural change. |
| Overlapping subsets, joined by least squares | compressed-SLC sequential estimator; v1.4 NRT update bit-identical to a full rerun | **Conflicts.** The joint adjustment re-solves earlier subsets whenever a new one arrives, so the series is no longer feed-forward. That breaks the NRT contract (`nrt_incremental_contract.rs`, `nrt_displacement_contract.rs`). |
| Coherence-guided subset boundaries | fixed `ministack_size` | Covered by #108 (`adaptive-ministack-2026-10.md`), which is deferred. |
| PS + DS in a two-layer Delaunay PSI network | wrapped ifg network → SNAPHU/native unwrap → SBAS inversion; PS-fill is not wired | A different downstream architecture. The useful, parity-backed piece is wiring PS selection and fill, which is scoped in #107 (`nrt-incremental-ps-2026-10.md`). |

No measurement was run. There is nothing in dolphinRust to compare against without first
building one of these parts, and the issue says not to.

## 3. Conclusion — scope only; recommendation: do not build

There is no "unified PS+DS sequential estimator" to adopt. Of the parts the paper does have:
- the transferable one (scene-wide adaptive subsets) is #108;
- the PS integration dolphinRust actually lacks is #107;
- the remaining parts (JS patch vectors, overlapping least-squares batches, a PSI network)
  either break the NRT feed-forward contract or replace the downstream architecture, and no
  reference toolchain (dolphin, MintPy, ISCE3, the DISP-S1 SAS) implements them.

Keep the issue's re-entry gate, with one clarification:
- (a) a reference toolchain ships JS-vector covariance or overlap-joined batches; **or**
- (b) after #107 lands, a validated AOI shows that dolphinRust's PS-fill + DS output has
  measurably lower point density or higher GNSS/leveling error than a PSI-network baseline on
  the same stack.

If (b) fires, the first item to scope is a `ShpMethod::KsPatch` variant. It is the only part
that stays within the current estimator.
