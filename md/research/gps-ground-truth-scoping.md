# Scoping brief: absolute GNSS ground-truth validation for the native unwrapper

**Date:** 2026-06-30
**Scope:** research only — no pipeline code, no test stubs written.
**Question:** can we close the "no absolute ground truth" gap on the Mexico T005 burst fixture, and is it worth doing now?

## Executive summary

The Mexico T005 fixture sits in the single most extreme, best-instrumented subsidence
bowl in North America (Mexico City, near the airport/Iztapalapa-Xochimilco lacustrine
zone), which is a lucky break — but the exact 1.92 km × 1.92 km crop used in the fixture
has **no GNSS station inside it or within ~6 km of it that was operating during the
fixture's Jan–Jun 2023 window**. The two nearest live stations in 2023 (ICMX, MMX1) are
6–9 km away, in a city where subsidence rate varies by 2–3× over such distances due to
differential clay-layer compaction — so they are not valid point-truth for this specific
crop without accepting a wide, honestly-reported tolerance. Building the harness is a
**1–2 week effort**, gated almost entirely on data acquisition and geometry (not code),
and the payoff is a **regional/pattern-level absolute check (TLS slope, sign, order of
magnitude)**, not a pixel-level mm-tolerance validation. Recommend: **defer building the
full harness now; do a 1-day feasibility spike first** to pull ICMX/MMX1 tenv3 series and
eyeball whether the sign/magnitude even roughly tracks the InSAR LOS estimate before
investing in the comparison pipeline.

## Key findings

1. **Exact AOI is pinned via three independently-cross-checked sources.** The fixture
   (`oracle/fixtures/real_slc_stack.npy`, consumed by both
   `crates/dolphin-unwrap/examples/real_burst_validation.rs` and
   `oracle/gen_phaselink_real.py`) is the crop at `validation/real_data/cropped_mexico`,
   produced by `validation/crop_real.py --burst T005 --row0 3656 --col0 3488 --size 384`
   (`VALIDATION.md:522`). Its geotransform is read directly off a granule in that same
   directory in `crates/dolphin-workflows/tests/tropo_real_warp.rs:16`:
   `gt=[485770, 5, 0, 2143510, 0, -10]`, EPSG:32614 (UTM 14N).

2. **Computed footprint** (UTM→WGS84, this session): a 1920 m × 1920 m square,
   NW (-99.13552, 19.38569), NE (-99.11723, 19.38571), SW (-99.13549, 19.35099),
   SE (-99.11721, 19.35100), **center (-99.12636, 19.36835)**. This is southern Mexico
   City — the Iztapalapa/airport-adjacent lacustrine subsidence zone, not the historic
   center or Xochimilco proper.

3. **Acquisition dates and burst ID are explicit in VALIDATION.md:207-209**: OPERA
   CSLC-S1 burst **T005-008704-IW1**, 13 acquisitions, **2023-01-04 through 2023-06-09**,
   12-day cadence (Sentinel-1A only cadence post-2021 loss of S1B — consistent).
   `real_burst_validation.rs` uses epoch 0 vs 12 (full span, 2023-01-04 vs ~2023-06-09)
   for the "long baseline" fixture and epoch 0 vs 1 (12 days) for "short baseline."
   Granule filename confirms one epoch precisely:
   `OPERA_L2_CSLC-S1_T005-008704-IW1_20230410T004052Z_..._S1A_VV_v1.1.h5`.

4. **No GNSS station falls inside or within ~6 km of the AOI with data covering
   Jan–Jun 2023.** Query of NGL's full station holdings file
   (`https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt`, 23,605 stations) for the
   Mexico City box found candidates `UIGF`/`UJAL`/`UNIP`/`IGCU`/`TNGF`/`SSNX` at 7–12 km,
   but **all of them stopped recording between 2013 and 2022** — none span 2023. Only two
   stations within a reasonable radius were live during the fixture window:
   **ICMX** (19.4056, -99.1709; INEGI RGNA network per secondary sources; 2017–2026,
   active) at **6.2 km**, and **MMX1** (19.4317, -99.0684; FAA CORS at Mexico City
   International Airport; 2008–2026, active) at **9.3 km**.

5. **MMX1 is independently documented as recording 26–31 cm/yr subsidence** (2010–2018),
   "the most rapid subsidence ever recorded by continuous GNSS in an urban area" per the
   GOM20 reference-frame literature. This is strong secondary confirmation that the
   region has a real, large, well-characterized deformation signal — good for a
   sign/order-of-magnitude check — but it also proves the field is **highly spatially
   heterogeneous** (published studies distinguish >9 distinct GPS-monitored subsidence
   compartments within a ~20 km radius, with rates varying 3–4× between adjacent sites).
   A station 6–9 km away is not a safe point-proxy for the specific 1.92 km AOI without
   an explicit, generous tolerance.

6. **TLALOCNet (UNAVCO/EarthScope-standard Mexico backbone) is not usefully close.**
   TLALOCNet's ~40+25 stations are built to Plate Boundary Observatory standards and
   distributed via `tlalocnet.udg.mx` / UNAVCO GSAC, but the network is designed for
   regional tectonics/atmosphere, not urban subsidence — its Mexico City-area density is
   low, and none of its published station IDs (site search) landed inside the AOI box
   in the searches performed. It's a viable secondary/regional-frame source for ITRF
   reference-frame stability, not a direct AOI proxy.

7. **The pipeline has no per-pixel LOS geometry today.** `dolphin-workflows/src/corrections.rs`
   uses a single scalar `incidence_angle_deg` (currently a config default, not read
   per-pixel from OPERA CSLC metadata) for tropo/iono scaling. A GNSS-to-LOS projection
   needs per-pixel incidence *and* heading/azimuth — OPERA CSLC-S1 HDF5 products do carry
   this (local incidence angle + orbit-derivable heading), but dolphinRust does not
   currently read or expose it anywhere in the codebase (confirmed via grep across
   `dolphin-workflows`, `dolphin-corrections`, `dolphin-io`).

## Implications for Muse

*(Framing note: this is dolphinRust, not the Muse SaaS product — the brief format is
reused as requested. Implications below are for dolphinRust/GroundPulse.)*

- **Don't build a per-pixel mm-tolerance GPS validation on this specific fixture.** The
  AOI was chosen for phase-linking/unwrapping stress-testing (low coherence, complex
  branch cuts), not for GPS co-location. Retrofitting ground truth onto it means
  validating against a *different physical location* than the stations, which weakens
  the claim rather than strengthening it.
- **If ground-truth validation is wanted, it should target station-colocated AOIs**,
  not this fixture. The right move is to pick a *new* crop window centered on ICMX or
  MMX1 (or a TLALOCNet urban site) and re-run the existing `fetch_real.py` /
  `crop_real.py` pipeline — same tooling, different `--row0/--col0`. That is a much
  more defensible ground-truth harness than force-fitting GPS onto T005.
- **Per-pixel incidence/azimuth ingestion is a prerequisite regardless of which AOI is
  chosen.** Right now `incidence_angle_deg` is a scalar knob for tropo/iono, adequate
  for those corrections but not precise enough for GNSS ENU→LOS projection at mm
  tolerance if the goal is anything beyond order-of-magnitude/sign checks. This is a
  small, well-scoped follow-on task (read OPERA CSLC local-incidence-angle + orbit
  heading, expose as a per-pixel or per-burst-constant field) independent of the GPS
  work itself.
- **Reference-point handling is a real design decision, not a detail.** InSAR
  displacement is inherently relative to a reference pixel/date; GPS gives absolute ENU.
  Any comparison must either (a) double-difference — GPS_station_LOS(t2)-GPS_station_LOS(t1)
  minus InSAR-implied same, both relative to the same reference epoch, or (b) if the
  reference pixel itself is near a *second* stable GPS station, use that as the absolute
  anchor. Given finding 4-5, option (b) is not available at T005; only (a) is feasible,
  and only as a regional/pattern check.

## Competitive context

Ground-truthing InSAR against GNSS is the field-standard validation method (used by
JPL/Caltech for OPERA DISP-S1 itself, and by academic Mexico City subsidence literature
e.g. the "Long-term subsidence in Mexico City... revealed by five SAR sensors" and
"Assessing subsidence of Mexico City from InSAR and LandSat ETM+ with CGPS and SVM"
papers found in this research). The published academic standard for Mexico City
specifically uses **city-scale correlation (up to 0.98) between InSAR velocity fields
and 9 nearby GPS stations**, not pixel-colocated mm-level agreement — because urban
subsidence in this basin is genuinely too spatially heterogeneous for tighter claims.
OPERA's own DISP-S1 cal/val plan uses dedicated corner reflectors and co-located
continuous GNSS at *purpose-selected* validation sites (e.g. Southern California
Plate Boundary Observatory sites), not retrofit onto arbitrary algorithm-test fixtures.
This matches the recommendation above: pick a GPS-colocated site if absolute validation
is the goal; don't retrofit.

## Technical considerations (Rust/AWS stack)

- **GNSS data formats needed:** NGL's `tenv3` daily ENU position time series
  (`https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/<STA>.tenv3`, plain-text,
  documented at `geodesy.unr.edu/gps_timeseries/README_tenv3.txt`) is the simplest
  ingest — no new format support needed beyond a small text parser (not a dependency,
  a ~50-line parser). UNAVCO/TLALOCNet data via GSAC would need RINEX or GAGE
  Web Services JSON — more integration surface for no AOI benefit here.
- **No new AWS/infra footprint.** This is a validation-harness concern
  (`validation/` + `oracle/` Python tooling, same pattern as existing real-data fixtures),
  not a production pipeline change. It would not touch ECS Fargate, S3, Bedrock, or
  Transcribe — it's purely an offline correctness gate, same tier as
  `real_burst_validation.rs`.
- **New Rust surface, if any:** a small ENU→LOS projection helper
  (`los = -sin(incidence)*cos(heading-90)*e_east - ... `, standard unit-vector dot
  product) belongs in `dolphin-corrections` alongside the existing tropo/iono geometry
  code, and should reuse whatever per-pixel incidence/azimuth reader is eventually added
  (finding 7) rather than hardcoding a new scalar.
- **Tolerance/metric, if pursued on a colocated AOI:** double-differenced GNSS LOS
  displacement between the two InSAR epochs, compared to the InSAR pixel(s) nearest the
  station (average over a small window, e.g. 3×3 to 5×5, to reduce speckle/unwrap
  noise — same boxcar philosophy already used for coherence in `prep_real_ifg.py`).
  Given C-band InSAR's own per-epoch atmospheric/noise floor is commonly cited at
  **5-15 mm** in the literature for a single interferometric pair (not stacked/time-series
  averaged), and this project's own real-data tier already reports **RMS ≤0.011 rad
  (~0.1 mm equivalent phase) / max 0.08 rad cross-engine noise** but that's InSAR-vs-InSAR,
  not InSAR-vs-independent-truth — a realistic "pass" bar for a single-pair GNSS check is
  **agreement within ~1-2 cm on displacement, or TLS slope within ~10-15% on a
  multi-epoch series**, not mm-level. Claiming mm-level absolute pass/fail against a
  6-9 km-offset station would overstate confidence; say so explicitly if this is built.

## Cost and effort estimate

| Path | Effort | Notes |
|---|---|---|
| Feasibility spike (pull ICMX+MMX1 tenv3, plot vs published T005 velocity/displacement, sanity-check sign+order of magnitude) | **0.5–1 day** | No new AOI, no code in the pipeline — pure analysis script, answers "is this even worth pursuing." |
| Full harness on a *new*, GPS-colocated AOI (pick station, refetch/crop stack via existing `fetch_real.py`/`crop_real.py`, build ENU→LOS projector, wire comparison + tolerance) | **1–2 weeks** | Bulk of the time is data reacquisition (new Earthdata fetch + crop) and building/validating the geometry projection, not algorithm work. |
| Full harness retrofit onto existing T005 AOI as-is | **Not recommended** | Would produce a weak, easily-challenged result (6-9 km offset in a heterogeneous field) for the same effort as doing it right on a colocated AOI. |

No recurring cost — GNSS data (NGL, UNAVCO/TLALOCNet) is free and public; no AWS spend.

## Recommended action

**Defer full harness build. Do the 1-day feasibility spike first**, order of operations:

1. Pull `ICMX.tenv3` and `MMX1.tenv3` from NGL for Jan-Jun 2023, compute the LOS-projected
   double-difference between the two InSAR epoch dates (approximate incidence 39°, S1A
   descending/ascending heading for track T005 — confirm track direction from the OPERA
   burst metadata already in the fixture's HDF5 before assuming).
2. Compare sign and rough magnitude against the InSAR LOS displacement at the pixel(s)
   nearest each station's true coordinates (even though they're outside the crop —
   re-derive from the full burst, not just the 384² crop, since the crop doesn't reach
   the stations).
3. If sign and order-of-magnitude agree: worth investing the 1-2 week harness, but build
   it on a **new GPS-colocated crop**, not T005. If they disagree or are noise-dominated:
   stop — that's a valid, useful negative result confirming the "no ground truth"
   caveat should stay explicit in VALIDATION.md rather than be quietly closed with a
   misleading comparison.
4. Regardless of outcome, the per-pixel incidence/heading reader (finding 7) is small
   and independently useful (tropo/iono correction accuracy) — fine to schedule
   separately of the GPS decision.

## Sources

**Authoritative:**
- `crates/dolphin-unwrap/examples/real_burst_validation.rs`,
  `oracle/prep_real_ifg.py`, `oracle/gen_phaselink_real.py`, `validation/crop_real.py`,
  `validation/fetch_real.py`, `VALIDATION.md`, `crates/dolphin-workflows/tests/tropo_real_warp.rs`
  — read directly, ground the AOI/date/geotransform findings (this repo).
- Nevada Geodetic Laboratory `DataHoldings.txt` master station file (geodesy.unr.edu) —
  fetched live this session; authoritative coordinate/date-range source for all NGL
  station distance calculations.
- Nevada Geodetic Laboratory `README_tenv3.txt` and `ICMX.sta` station page — format and
  metadata confirmation, fetched live.

**Secondary (blog/paper-level, treat findings as directional not definitive):**
- Springer "Assessing subsidence of Mexico City from InSAR and LandSat ETM+ with CGPS
  and SVM" (paywalled abstract/search-snippet only — did not access full text; station
  list (MOCS, MPAA, MRRA, UPEC, UCHI, UIGF, UGOL, UJAL, UTEO) came from search snippets,
  not verified against the paper's own coordinate table).
- GOM20 reference frame paper (MDPI) — MMX1 subsidence-rate claim (26-31 cm/yr) via
  secondary web-search summary, not the primary PDF.
- TLALOCNet overview (UNAVCO/ResearchGate summaries) — network description only, no
  station-level AOI match confirmed.

## Confidence level

**Overall: medium-high on facts about this repo (AOI, dates, geotransform — high
confidence, cross-checked across 3 independent files); medium on GNSS coverage
conclusion.** The "no station within 6 km with 2023 coverage" finding is well-supported
by the authoritative NGL master list (23,605 stations, directly queried), so I'm
confident in the *negative* result. Lower confidence on the exact station-list-to-paper
match for MOCS/MPAA/etc. (secondary sources only, code/station identity not verified
against primary literature) — treat that specific 9-station list as illustrative of
"academic studies exist," not as a verified candidate roster. The subsidence-rate/
heterogeneity claim (finding 5) is well-corroborated across multiple independent
secondary sources but I did not access a primary peer-reviewed PDF directly.
