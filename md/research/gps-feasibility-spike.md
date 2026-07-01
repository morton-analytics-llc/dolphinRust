# GPS ground-truth feasibility spike — verdict

**Date:** 2026-06-30
**Gate:** per `gps-ground-truth-scoping.md` §Recommended action — spike ICMX/MMX1 first;
if sign+order-of-magnitude agree, build the full harness on a **new GPS-colocated AOI**
(via `fetch_real.py`/`crop_real.py`), not retrofit onto T005. Full build 1–2 wk, gated on
data/geometry, not algorithm.
**Fixture window:** 2023-01-04 (epoch 0) .. 2023-06-09 (epoch 12), burst T005-008704-IW1.

## 1. What the spike found

Both sides are available. GNSS acquired live/clean; InSAR product already on disk.

**GNSS (NGL 24h final, IGS20 tenv3 — brief's IGS14 path 404s, correct layout is
`/gps_timeseries/IGS20/tenv3/IGS20/`). Endpoint ENU→LOS, S1 descending, incidence ~39°
(negative = range increase / motion away from sat):**

| Station | offset | in-window vertical | expected LOS (desc, i≈39°) | signal quality |
|---|---|---|---|---|
| **ICMX** (6.2 km) | 6.2 km | ~−0.8 mm (stalled) | **−4.9 mm** (H-dominated; flips +4.2 mm ascending) | near noise floor; **not representative** — full-record −26 mm/yr, but Jan–Jun 2023 was an anomalous dry→wet aquifer stall (−1.9 mm/yr) |
| **MMX1** (9.3 km) | 9.3 km | **−124 mm** (~−29 cm/yr) | **−102.6 mm** (robustly negative all geometries, −92→−108 mm) | high-SNR linear ramp; matches published 26–31 cm/yr |

**InSAR (present on disk, meters LOS, `-λ/4π·phase`, negative = subsidence; EPSG:32614,
384² crop):**
- Rust `validation/runs/real_mexico_T005/work_rust/displacement_11.tif`
- Oracle `.../work_oracle/timeseries/20230104_20230609.tif`
- AOI-center 5×5 mean: **Rust −15.1 mm, Oracle −10.3 mm**; crop mean −14.9 / −9.5 mm;
  deepest pixel −51.9 / −36.8 mm. Rust↔oracle bias −5.5 mm, RMS 8.2 mm; velocity TLS
  slope 1.026, corr 0.865. → **unambiguous subsidence, ≈−23 to −35 mm/yr LOS at AOI center.**

**The gap for a direct comparison:** both stations are **outside** the 1.92 km crop and the
full burst is **not on disk** (all 13 h5 are 384² cropped granules). InSAR *at the station
pixels* was not derivable this spike — the crop never reaches ICMX/MMX1. So the spike
compares GNSS-at-station vs InSAR-at-AOI-center, which are **different subsidence
compartments**, not a co-located pixel check.

## 2. Sign / magnitude agreement assessment

**Sign: AGREE.** Every observable is negative/subsiding — MMX1 GNSS (−102 mm LOS), the
InSAR AOI (−10 to −15 mm), and (weakly) ICMX. No sign inversion; consistent with the
post-v1.3.0 convention (`ifg-sign-inversion` memory).

**Magnitude: order-of-magnitude PLAUSIBLE, but not co-located.** MMX1's −10 cm LOS and the
AOI's −1 to −1.5 cm LOS differ by ~7–10×, but this is **expected heterogeneity** (3–4× rate
variation between adjacent compartments over ~10 km; MMX1 sits in the airport bowl, the AOI
in a different cell) — not an InSAR error. ICMX contributes essentially nothing this window
(stalled to noise floor). The two GNSS stations themselves disagree ~150× in-window,
empirically confirming the brief's core caveat: **this field is too heterogeneous at 6–9 km
offset to serve as point-truth for the 1.92 km crop.**

Net: the spike is **not** a negative result (data acquired, signs agree), and **not** a
clean positive (no co-located pixel; magnitudes are cross-compartment, not comparable).

## 3. Recommendation

# GO — build on a NEW GPS-colocated AOI, not T005.

**Rationale:** signs agree across GNSS and InSAR and MMX1 is a strong, unambiguous
subsidence target (−10 cm LOS, high-SNR), clearing the brief's "sign + order-of-magnitude
promising" gate — but T005's 6–9 km offset in a 3–4×-heterogeneous field permanently caps a
T005 retrofit at a weak regional check, so the payoff only materializes on a station-centered
crop.

**Concrete next steps (per brief, in order):**
1. **Per-pixel LOS geometry** is the hard prerequisite: `dolphin-workflows/src/corrections.rs`
   carries only a scalar `incidence_angle_deg` — no per-pixel incidence/heading reader exists,
   and the asc-vs-desc ambiguity here is unresolved because track direction was never read from
   the burst HDF5. Add an OPERA CSLC local-incidence + orbit-heading reader before any
   mm-tolerance ENU→LOS projection.
   > **DONE (unmerged, on working tree — see `md/design/per-pixel-los-geometry.md`).** Reads the
   > per-pixel LOS unit vector from the **CSLC-S1-STATIC** companion (not the main granule, which
   > has no LOS rasters), resolves `LosGeometry{east,north,up}` onto the frame grid, exposed on
   > `DisplacementOutput.los_geometry`. Asc/desc is resolved for free (the signed LOS vector
   > encodes it — no separate heading read). ENU→LOS: `d_los = d_e·east + d_n·north + d_u·up`
   > (ground→sensor; sign reconciliation with the `−λ/4π·φ` displacement convention is this
   > harness's job). Deferred: iono ground→ionospheric-shell mapping.
2. **Pick a GPS-colocated AOI** centered on MMX1 (19.432, −99.068) — strong linear signal —
   as the primary, ICMX as a stalled/near-null control. Re-run `fetch_real.py --burst
   T005_008704_IW1 ...` → `crop_real.py --burst T005 --row0 <r> --col0 <c> --size <n>` with a
   window on the station UTM coords, then the displacement pipeline on the new crop.

**Single blocking acquisition to make a T005-level check conclusive** (if pursued before the
new AOI): the **full T005-008704-IW1 burst** — needs `GP_EARTHDATA_TOKEN`
(`source validation/creds.sh`); Earthdata token availability in-sandbox is **unverified** and
is the one hard external dependency. Without it, InSAR at the station pixels cannot be derived.
