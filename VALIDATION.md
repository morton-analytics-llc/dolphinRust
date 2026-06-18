# dolphinRust — end-to-end validation against Python `dolphin`

Proves dolphinRust reproduces the Python `dolphin` it replaces, end to end, within the
PLAYBOOK §"Correctness & validation strategy" tolerances (physical, not bit-exact). Run
date: **2026-06-16**. Reproduce with `validation/run.sh <speckle>`.

## Environment (pinned)

| Component | Version | Role |
|---|---|---|
| Python `dolphin` | **0.35.0** (`e567e55`) | reference oracle (resolves Open question #1) |
| Python `dolphin` | **0.42.0** | forward oracle — **only** for the v1.2.0 CRLB + closure layers (`oracle/.venv-v042`) |
| dolphinRust | 1.0.0 | engine under test |
| GDAL | 3.12.2 | raster I/O (both) |
| HDF5 | 2.1.1 | CSLC read (both) |
| SNAPHU — oracle | `snaphu-py` 0.4.1 (`import snaphu`) | dolphin's unwrap backend |
| SNAPHU — dolphinRust | binary `snaphu v2.0.7` (`/opt/homebrew/bin/snaphu`) | dolphin-unwrap shells out |
| Python | 3.11.14 (`oracle/.venv`, built with `uv`) | oracle env |
| rustc | 1.94.1 | release build |

Oracle env rebuild recipe: see auto-memory `oracle-env`. The two engines use **different
SNAPHU implementations** (snaphu-py wheel vs Stanford binary) — an intended consequence of
the "shell out for unwrapping" architecture decision, exercised here for the first time.

## Data paths: synthetic tier + real OPERA tier

**Synthetic tier.** `validation/gen_stack.py` emits a deterministic single-burst stack
(fixed seed 21): N=5 acquisitions, 48×64, complex64 at `/data/VV`, files named
`cslc_YYYYMMDD.h5` (1-day cadence) so dolphin's date parser accepts them. Signal is a smooth
range ramp growing linearly in time (`0.3·t·x/cols`), kept small (|ifg phase| < ~1.2 rad) so
SNAPHU is cycle-free and the comparison isolates the estimators, not integer-cycle
disagreements. A `--speckle` knob sets per-SLC complex noise; the sweep below uses it to
characterize the divergence.

**Real OPERA tier (new).** Genuine OPERA L2 CSLC-S1 v1.1 granules from ASF, fetched with the
Earthdata bearer token (`validation/fetch_real.py`, `creds.sh`), cropped to a 384×384 window
(`crop_real.py`) so both engines run quickly on identical pixels. Bursts sampled: a coastal
burst (T063-133231-IW1) and three land bursts — Mojave (T071-151223-IW1), Las Vegas
(T173-370304-IW2), and Central Valley/Corcoran (T144-308011-IW2), each a 5- or 9-acquisition,
12-day-cadence stack. See "Real-data results" below.

## One config, both engines — compatibility: PASS

The config is a genuine `dolphin config`-generated `DisplacementWorkflow` YAML (the canonical
pydantic schema, ~15 KB), one per engine differing only in `work_directory`. **dolphinRust
parses and runs it unchanged** through the full pipeline (`#[serde(default)]` + ignore-unknown;
`snaphu_options`/`tophu_options` are modeled and round-trip, the spurt/whirlwind solver blocks
it does not model are absorbed). This is the drop-in-config requirement and it holds.

## Tiers run

Both tiers ran — SNAPHU is present, so the pipeline executed end to end including unwrap.

- **Tier A** (no SNAPHU): not needed as a fallback; covered as a subset of Tier B.
- **Tier B** (full): phase-linking → ifg network → SNAPHU unwrap → timeseries → velocity,
  on both engines. Note: the single-reference network (`reference_idx=0`) means dolphin
  *skips* the inversion solve ("only single reference interferograms exist") — the unwrapped
  phases are the displacement; dolphinRust runs its SBAS-L2 solve on the same single-ref
  network, which reduces to the same quantity.

## Per-stage results

### A. Per-kernel oracle agreement (PLAYBOOK §Correctness secondary check)

Each numerical kernel carries a contract test against a `.npy` fixture generated from
dolphin v0.35.0 (`oracle/gen_*.py`). All green (`cargo test --workspace`, clippy/fmt clean):

| Kernel / crate | Contract suite | Tests | Result |
|---|---|---|---|
| blocks, config (`dolphin-core`) | blocks_contract, config_contract | 3, 4 | PASS |
| phase-linking EVD/EMI (`dolphin-phaselink`) | phaselink_contract | 7 | PASS (`\|⟨v,v_oracle⟩\|`>0.999, cov<1e-4) |
| quality / temp_coh (`dolphin-phaselink`) | quality_contract | 6 | PASS |
| CRLB σ + closure phase (`dolphin-phaselink`) | quality_v042_contract | 6 | PASS vs **v0.42.0** (σ + closure max \|Δ\| <1e-4; singular-Γ NaN matches) |
| SHP GLRT/KS (`dolphin-shp`) | shp_contract | 5 | PASS |
| PS selection (`dolphin-ps`) | ps_contract | 4 | PASS |
| ministack planner + sequential (`dolphin-stack`/`dolphin-workflows`) | planner_contract, sequential_contract | 3, 2 | PASS (incl. multi-ministack stitched temp_coh + concatenated CRLB/closure <1e-3 vs **v0.42.0**) |
| network + SBAS L2 **and L1/ADMM** (`dolphin-timeseries`) | timeseries_contract | 6 | PASS (L1 vs dolphin oracle <1.5e-6) |
| filters (`dolphin-filtering`) | filtering_contract | 4 | PASS |
| I/O round-trip (`dolphin-io`) | io_contract | 5 | PASS |
| SNAPHU dispatch (`dolphin-unwrap`) | unwrap_contract | 1 | PASS |
| tophu multi-scale unwrap (`dolphin-unwrap`) | tophu_contract + lib units | 2 + 4 | PASS (ramp within SNAPHU envelope; planted inter-tile 2π jump resolved) |
| pipeline (`dolphin-workflows`) | displacement_contract | 1 | PASS |

### B. End-to-end CLI equivalence (new — full `dolphin run` vs dolphinRust)

Displacement series + velocity, compared on the common finite mask after removing a per-date
constant (the global phase reference the spec permits). dolphin auto-picks a spatial
reference point and masks low-coherence/edge pixels to nodata; dolphinRust references only
temporally (date 0) and fills all pixels — handled by demeaning on the shared mask. **Stated
physical tolerance: corr ≥ 0.95 and demeaned per-pixel RMS ≤ 0.10 rad (< 0.016 cycle).**

> **Sign correction (2026-06-17).** This section originally read "Sign is +1 (engines agree
> in sign after demean)." That agreement was **real but blind**: the oracle generator
> (`oracle/gen_displacement.py`) formed the ifg in the *same* reversed order
> (`sec·conj(ref)`) that dolphinRust used, so both engines were inverted in lockstep relative
> to dolphin **production** (`interferogram.py` forms `ref·conj(sec)`). The contracts proved
> Rust agreed with a flipped oracle, not with production. v1.0.0–v1.2.0 therefore shipped
> displacement *and* velocity with a globally inverted LOS sign. Fixed in v1.3.0 (commit
> `e1db05a`); proven against production on real data in "Interferogram sign convention" below.

Speckle sweep (max |deviation| in rad), strongest-signal date `displacement[3]`:

| speckle | disp corr | disp RMS (rad) | disp max (rad) | verdict |
|---|---|---|---|---|
| 0.00 (pure algorithm) | 1.0000 | 5.1e-4 | 1.1e-3 | PASS |
| 0.005 | 0.9997 | 5.6e-3 | 1.9e-2 | PASS |
| 0.05 (realistic) | 0.9761 | 5.1e-2 | 2.0e-1 | PASS |

Full per-date table at realistic speckle 0.05:

| stage | n | corr | RMS (rad) | max (rad) | pass |
|---|---|---|---|---|---|
| displacement[0] 20221120 | 1595 | 0.7501 | 5.0e-2 | 1.9e-1 | corr-FAIL¹ |
| displacement[1] 20221121 | 1595 | 0.9202 | 4.9e-2 | 1.7e-1 | corr-FAIL¹ |
| displacement[2] 20221122 | 1595 | 0.9594 | 5.0e-2 | 1.9e-1 | PASS |
| displacement[3] 20221123 | 1595 | 0.9761 | 5.1e-2 | 2.0e-1 | PASS |
| velocity (pattern) | 1595 | 0.9656 | — | — | PASS² |

¹ The **RMS floor is constant (~0.050 rad) across all dates**; correlation only dips on the
early dates because their signal is weakest (low SNR), not because agreement worsens. See
divergence #1.
² Velocity now matches on **absolute scale** (affine slope a=0.9997 at speckle 0.05; a=1.0000
  noise-free) after the real-baseline fix — see divergence #2 (resolved).

## Divergences (with hypotheses)

1. **Displacement residual ≈ speckle (sanctioned eigensolver divergence) — not a bug.**
   Demeaned RMS scales linearly with input speckle: 5.1e-4 → 5.6e-3 → 5.1e-2 rad as speckle
   goes 0.0 → 0.005 → 0.05. At zero speckle the two independent pipelines agree to max
   **1.1e-3 rad** (corr 1.0000). Hypothesis: the only end-to-end divergence is the
   per-pixel difference between faer's direct self-adjoint eigendecomposition and dolphin's
   JAX power/inverse iteration (PLAYBOOK Open question #2) — bounded in the
   eigenvector-overlap metric (>0.999, confirmed by `phaselink_contract`) but realized as a
   phase difference proportional to per-pixel speckle. Physical and within tolerance.

2. **Velocity absolute scale — RESOLVED (was: wrong for non-12-day cadence).**
   Previously `oracle_velocity = 12.0004 · rust_velocity` because
   `displacement.rs` hardcoded `DT_DAYS = 12.0` and never read acquisition dates. **Fixed
   (Workstream A1):** `dolphin-workflows::dates::decimal_days` parses the real acquisition
   dates from the CSLC filenames (`input_options.cslc_date_fmt`) and feeds real decimal-day
   baselines to `build_network` / `estimate_velocity`. The affine fit `oracle = a·rust + b`
   now gives **a = 1.0000 / 0.9994 / 0.9997** at speckle 0.0 / 0.005 / 0.05 — absolute scale
   matches within ±0.02 across all tiers. (`b` is dolphin's spatial reference-pixel offset,
   removed by the demean; a raw median ratio is meaningless against it, so `compare.py` now
   reports the affine slope.) The typed API additionally exposes `velocity_mm_yr`, converting
   LOS phase rate via `−λ/4π` (config wavelength, else Sentinel-1 default). Contract tests:
   `displacement::tests::recovers_injected_rate_in_mm_per_yr`, `rate_is_independent_of_cadence`.

3. **Per-ministack temporal-coherence "stitching" is `numpy.nanmean`, not a richer
   algorithm — clarified.** dolphin v0.41 added *retention + spatial stitching* of the
   per-ministack `temporal_coherence_<dates>` files, but the **consumed** full-span layer
   stays `temporal_coherence_average` = `numpy.nanmean(A, axis=0)` across ministacks
   (`workflows/sequential.py::_average_or_rename`, fed to unwrapping/reference/quality in
   `displacement.py`). There is no separate scalar reduction to adopt; the v0.35 source even
   carries a `# Currently ignoring to not stitch:` marker on the per-ministack files. Our fix
   replaces a plain mean with that NaN-aware mean (`sequential.rs::stitch_temp_coh`): equal on
   all-finite layers (prior parity held), but a pixel masked/decorrelated (NaN) in some
   ministacks now averages only the finite ones instead of being diluted toward zero. Oracle
   `oracle/gen_stitch_v042.py` composes the v0.42 kernels over a 2-ministack stack exactly as
   `sequential.rs` does; `stitching_and_quality_match_oracle_multiministack` confirms stitched
   temp_coh + concatenated CRLB + closure all agree <1e-3. This closes the CRLB/closure
   many-ministack concatenation caveat (all three layers now combine per-ministack results
   with consistent NaN semantics).

4. **tophu multi-scale unwrap now beats raw SNAPHU on low-coherence scenes — measured on
   the frozen scenes.** On the same 512×512 subsidence-bowl scenes with known truth under
   vegetation-style coherence loss where raw SNAPHU genuinely struggles (gross-cycle-error
   0.13–0.17), tophu is now **≤ raw SNAPHU on all three metrics on both scenes**:
   discontinuities −9 % on both, gross-cycle-error −10 % on the steep+decorr-ring scene, and
   rms ≤ raw on both (−3.5 % gentle, −0.7 % steep). The earlier honest *loss* (this same item)
   had two named causes; both were fixed in the algorithm — (a) a **coherence-weighted** coarse
   multilook with low-trust blocks masked + filled, so decorrelated phasors no longer poison
   the coarse anchor, and (b) **overlap-region inter-tile cycle reconciliation** via a
   maximum-reliability spanning forest plus a **feathered tile merge**, replacing the per-tile
   snap-to-coarse that injected the cross-tile cycle errors. The scenes, noise model, seeds and
   metric definitions are byte-for-byte unchanged from the loss measurement — only the
   algorithm changed; the win was **not** manufactured by tuning the scene or a tolerance.
   SNAPHU remains the default (simpler, sufficient for small/coherent scenes). Numbers,
   margins + reproduction in `bench/UNWRAP.md`.

## Real-data results (OPERA CSLC, both engines)

Both engines ran the **full pipeline on genuine OPERA CSLC-S1 granules** — config
compatibility on real data: **PASS** (the same `dolphin config` YAML drives both, unchanged).
Findings, consistent across all four bursts sampled:

- **Engine agreement — PASS.** The demeaned per-pixel displacement residual between the two
  independent engines is **≤ 0.008 rad** (velocity residual ≤ 0.031 rad/yr) on real data —
  well inside the sanctioned eigensolver+SNAPHU divergence envelope (≤ 0.10 rad) established
  on synthetic data. The engines produce near-identical output on real OPERA scenes.
- **Velocity magnitude agreement — PASS.** On the Central Valley window both engines report
  the same velocity spread (oracle std 0.020, rust std 0.020 rad/yr; comparable ranges); the
  per-field means differ only by dolphin's spatial-reference-pixel offset. Rust does **not**
  fabricate signal — a direct check that the engines see the same deformation magnitude.
- **Temporal coherence agreement — PASS.** Rust vs oracle `temporal_coherence_average` agree
  to ~0.01 on valid pixels (e.g. 0.598 vs 0.608).

### Velocity absolute scale on a real deforming scene (Mexico City) — CONFIRMED

The narrow B4 follow-up — *velocity absolute scale under strong real signal* — is now
closed against a genuine subsidence scene. Earlier samples sat at the cross-engine noise
floor because high coherence selects *stable* ground, and a small window over a *broad*
subsidence bowl is nearly uniform, so demeaning removes the signal and the scale regression
is ill-conditioned (corr ~0.16, slope meaningless). The fix is **scene + window selection
on the velocity gradient**, not just on coherence.

Scene: OPERA CSLC-S1 burst **T005-008704-IW1** (Mexico City), 13 acquisitions
**2023-01-04 … 2023-06-09**, 12-day cadence, cropped 384² at full-burst (row0 3656,
col0 3488), EPSG 32614, λ=0.0555 m. The window was located by a cumulative-phase-gradient
scan over the decimated burst (highest differential-deformation window with mean
phase-linking temporal coherence **0.99**), so the within-window velocity rises clearly
above the floor (corr **0.865** vs 0.16 on stable windows).

| metric | value | reading |
|---|---|---|
| velocity corr (oracle vs rust) | **0.865** | common deformation pattern, unambiguous |
| OLS slope `oracle = a·rust + b` | 0.884 | biased low — errors-in-variables attenuation¹ |
| **TLS (orthogonal) slope** | **1.026 – 1.030** | **absolute scale matches within ~3 %** |
| magnitude (velocity std) | oracle 0.096 / rust 0.094 mm/yr | magnitudes agree |
| engine displacement RMS / max | ≤ 0.011 / 0.080 rad | inside the ≤0.10 rad sanctioned envelope |

¹ OLS regresses oracle on the *noisy* rust velocity, so cross-engine noise in the regressor
biases the slope toward zero. The symmetric **total-least-squares** slope removes that bias
and is the honest scale metric; it is **stable at ≈1.03 across all coherence gates**
(0.0/0.6/0.8/0.9), confirming the match is not a coherence artifact. Reproduced by
`validation/velocity_scale.py`.

The strict end-to-end table threshold (corr ≥ 0.95) is *not* reached here: within-window
differential subsidence over a broad bowl is modest, and per-pixel cross-engine unwrap noise
(two independent SNAPHU builds) caps pattern agreement at ~0.86 — a larger 768² window made
it worse (corr 0.30) as the extra area introduced integer-cycle unwrap divergence. So the
scale is confirmed (TLS ≈ 1.03, magnitudes match), while bit-level pattern parity on real
data remains noise-limited — consistent with the **synthetic tier (a = 1.0000)**, which stays
the controlled-magnitude scale anchor. This **closes the documented B4 gap**.

## GPU backend validation (first-class, 2026-06-17)

The GPU compute backend (`wgpu`/Metal, f32) validated against the CPU (`faer`, f64)
reference and the dolphin v0.35.0 oracle on the **real** Mexico stack (13 acqs, 384²,
`half_window` 11×5). Backend: Apple M2 Pro, Metal (integrated). Full numbers + method
in [bench/GPU.md](bench/GPU.md); reproduce with
`cargo run -p dolphin-phaselink --release --example gpu_bench`.

**Accuracy — all 147,456 pixels (no coherent-only masking):**

| Comparison | overlap ≥ | max Δφ |
|---|---|---|
| EVD GPU(f32) vs CPU(f64) | 1.0000 | 0.176 mm |
| EMI **raw** GPU(f32) vs CPU(f64) | 0.343 | 13.85 mm (π-rad tail) |
| EMI **hybrid** GPU vs CPU(f64) | 0.9991 | **0.607 mm** |
| EMI hybrid GPU vs oracle | 0.9888 | 5.35 mm |
| EMI CPU(f64) vs oracle | 0.9888 | 5.35 mm |

- EVD is sub-mm everywhere. Raw f32 EMI has a π-rad tail on ill-conditioned pixels;
  the **hybrid** (GPU flags near-degenerate / borderline-PD pixels, host recomputes
  the 5.9% minority on f64 `faer`) makes EMI **sub-mm across every pixel — no tail.**
- The hybrid-vs-oracle and CPU-vs-oracle max are identical (5.35 mm, one degenerate
  pixel): the GPU hybrid tracks the f64 CPU exactly; the residual is a Rust-CPU vs
  dolphin difference, not a GPU error.
- EMI is run-to-run **deterministic** (bit-identical) at 384²/nslc 13 and nslc 32.

**Speed — end-to-end (covariance + phase-link + transfer + hybrid recompute), honest:**
on this *integrated* M2 Pro the GPU is **0.66× on the real 384² stack (slower than CPU)**
and ~1.09× on clean synthetic stacks above the ~192² crossover. The readback + f64
round-trip + 5.9% CPU recompute outweigh the kernel saving on integrated silicon with a
strong `faer`+`rayon` CPU baseline. **The first-class value is correctness + portability:**
the same WGSL runs unchanged on discrete NVIDIA/AMD, where the FP32 headroom is. On Apple
silicon, prefer `compute_backend = auto` (CPU below the crossover) or `cpu`.

**Backend selection + fallback:** GPU compiled into the default build; runtime-selected via
`worker_settings.compute_backend` (`auto`/`cpu`/`gpu`); no adapter / unsupported nslc /
`no-gpu` build → automatic CPU fallback with a warning, never a panic (contract:
`engine_contract::no_adapter_falls_back_to_cpu_without_panic`).

## NISAR / L-band ingest validation (2026-06-17, v1.3.0 part 1)

Real-data validation of the NISAR/L-band ingest path (NISAR_INGEST_PROMPT.md). **The
reader is validated against a real granule; full multi-date displacement is deferred.**

**Granule.** `NISAR_L2_PR_GSLC_010_165_D_100_2005_DHDH_M_20260120T155930_..._001.h5`
(collection `NISAR_L2_GSLC_BETA_V1`, 7.24 GB), fetched from ASF/Earthdata
(`nisar.asf.earthdatacloud.nasa.gov`) with the `validation/creds.sh` bearer token. The
collection has 23,450 granules — real NISAR data is plentiful and reachable.

**De-risk correction (load-bearing).** The prompt's central assumption — NISAR is a
*complex-int16* compound — **does not hold**. `h5dump` on the real granule shows
`/science/LSAR/GSLC/grids/frequencyA/HH` is `H5T_COMPOUND { F32 "r"; F32 "i" }`
(complex64), i.e. the **same h5py `(r,i)` layout as OPERA CSLC**. `xCoordinates`/
`yCoordinates` are F64; `projection.epsg_code` is an I64 attribute (= 32736). Consequence:
the complex read needs no NISAR-specific decoding (reads as `Cf32`); the *only* NISAR-specific
code is the geocoding-metadata geotransform reader. The reader was corrected from int16 to
f32 accordingly. **(Flagged for sign-off — this overrides the prompt's "don't re-litigate
int16" instruction because the real data contradicts it.)**

**Reader result — PASS** (`dolphin-io` test `nisar_real_data`, gated on `NISAR_REAL_H5`):

| Check | Result |
|---|---|
| HH complex decode (center 256×256 block, `{r,i}` f32 → `Cf32`) | **65536/65536 finite** samples |
| EPSG (from `projection.epsg_code` attribute) | **32736** (WGS84 / UTM 36S) |
| Grid origin (upper-left) | (290880.0, 7611840.0) m |
| Pixel posting (dx, dy) | (10.0, −5.0) m |

(The grid corners are NaN fill outside the swath footprint — the geocoded grid is larger
than the imaged area — so the validation samples the grid center.)

**Deferred — full real-data displacement.** A real velocity/displacement product needs ≥2
co-located repeat-pass acquisitions over one frame (~15 GB+ to download, plus coherent
repeat coverage). Not attempted this run; the end-to-end pipeline is instead proven on a
synthesized multi-acquisition NISAR stack (`nisar_e2e_contract`). **Where to look next:** CMR
`short_name=NISAR_L2_GSLC_BETA_V1` filtered to a single `track/frame` with ≥2 dates.

**Limitation — atmospheric correction.** This is a geometrically-correct but
*atmospherically-uncorrected* L-band product. Ionospheric delay is ~16× the C-band effect
and is mandatory for a *usable* L-band displacement product; ionospheric + tropospheric
corrections are the separate later half of v1.3.0 (added below).

## Atmospheric corrections validation (2026-06-17, v1.3.0 part 2)

Ionospheric + tropospheric corrections (`ATMO_CORRECTIONS_PROMPT.md`), in the new
`dolphin-corrections` crate. Corrections produce a per-acquisition range delay in meters,
subtracted (relative to date 0) from the inverted LOS-phase series before velocity, off by
default. **Correction math is fixture-proven; both real sources were reachable and
validated on real data this run.**

**Ionosphere — TEC/IONEX → L-band delay (`1/f²`).** Closed-form `delay = vtec·1e16·K/f²`
(`K = 40.31`, Yunjun et al. 2022 / Chen & Zebker 2012), scaled to the *configured* carrier.
Contract green vs the closed-form relation; the L/C ratio is `(f_C/f_L)²`.

- **Real IONEX — PASS** (`dolphin-corrections` test `real_ionex_parses_to_physical_delay`,
  gated on `IONEX_REAL`). Fetched a real IGS final GIM from CDDIS with the
  `validation/creds.sh` bearer token (the `~/.netrc` is stale; the token authorizes):
  `https://cddis.nasa.gov/archive/gnss/products/ionex/2023/001/IGS0OPSFIN_20230010000_01D_02H_GIM.INX.gz`.
  Parsed to a `(13, 71, 73)` VTEC cube (2-hourly, 2.5°×5°). Equatorial-noon VTEC = **56.5
  TECU** → **L-band range delay 14.40 m** vs C-band 0.78 m = **18.5×** — i.e. ~14 m of
  apparent LOS displacement if uncorrected. This is why the correction is mandatory at
  L-band, confirmed on real data, not just analytically.

**Troposphere — OPERA L4 netCDF ingest + 4326→UTM warp.** GDAL `NETCDF:` ingest, then a
**reprojecting resample**: same-CRS grids take the bilinear `resample_bilinear` path,
cross-CRS grids (global EPSG:4326 product → UTM frame) take `warp_to_frame` (GDAL bilinear
`reproject`), zenith→slant by `1/cos(inc)`. Contract green vs a synthesized L4-format netCDF
fixture (`ingests_synthesized_l4_netcdf`, written via GDAL's netCDF driver).

- **4326→UTM warp contract — PASS** (`warps_4326_field_onto_utm_frame`,
  `build_troposphere_warps_4326_onto_utm_frame`). A synthesized EPSG:4326 delay field linear
  in (lon, lat) warps onto a UTM (32610) frame and recovers the analytic delay at known frame
  pixels to `< 5e-3 m` (bilinear of a field linear in the source's uniform-degree index space
  is exact). Proven both at the bare-warp level and end-to-end through the `build_troposphere`
  pipeline stage. The old CRS-mismatch `warn!` path is gone — a known mismatch now reprojects.

- **Real OPERA L4 — PASS** (test `real_opera_l4_total_is_physical`, gated on `OPERA_L4_REAL`).
  Collection `OPERA_L4_TROPO-ZENITH_V1` (CMR: **15,274 granules**), fetched from ASF
  (`cumulus.asf.earthdatacloud.nasa.gov`, ~2.0 GB) with the bearer token. **Real-product
  facts discovered:** the total zenith delay is **two** variables — `hydrostatic_delay` +
  `wet_delay` (meters), not a single `troposphere` field; the grid is a **global EPSG:4326**
  raster (2560×5120, 0.07°, time-stepped bands), with **no EPSG authority code** on the CRS
  and a `9.96921e36` no-data fill. The reader reads + sums both (`read_l4_total`); centre
  total ZTD = **2.79 m** (hydrostatic mean 2.38 m + wet). `troposphere_variable` defaults to
  `"total"`; the divergence from the prompt's single-`troposphere` assumption is documented.

- **Real full-frame tropo application — PASS** (test `real_l4_warps_onto_real_utm_frame`,
  `crates/dolphin-workflows/tests/tropo_real_warp.rs`, skips when local real fixtures absent).
  The real global EPSG:4326 `OPERA_L4_TROPO-ZENITH_V1` granule warps onto the **real Mexico
  City UTM frame** (EPSG:32614, 384², `gt=[485770, 5, 0, 2143510, 0, -10]`, read from a real
  cropped CSLC via `read_geotransform`). Applied zenith delay over the frame: **mean 2.553 m**
  (min 2.548, max 2.558); slant at a Sentinel-1 IW incidence of 39° ≈ **3.285 m**. Lower than
  the sea-level centre (2.79 m) because Mexico City sits at ~2.2 km altitude — physically
  consistent. The L4 variables carry no embedded CRS through GDAL's NETCDF driver, so the
  reader assigns EPSG:4326 when the geotransform spans geographic-degree ranges (the global
  plate-carrée product); a projected CRS-less grid stays unset rather than mislabeled.

**RAiDER fallback — deferred (not installed).** `python -c "import RAiDER"` fails and no
`raider.py` on `PATH` here, so the fallback is **gated behind `raider_available()`** (like
SNAPHU) and returns `RaiderUnavailable` rather than being stubbed; the subprocess + GDAL
ingest path is implemented for when it is installed. The OPERA L4 path is primary.

**Apply stage / typed API.** `subtract_delay` removes the per-date delay (relative to date 0)
in radians via `φ = d·(-4π/λ)`; exact-subtraction + zero-delay-identity + constant-delay-
cancels contracts green. Layers surface on `DisplacementOutput.{ionosphere_delay,
troposphere_delay}` and as `ionosphere_NN.tif` / `troposphere_NN.tif` COGs. A dolphin
`correction_options` YAML (`ionosphere_files`/`geometry_files`/`dem_file`) round-trips
(`dolphin_correction_options_round_trips`); corrections are off by default (output unchanged).

**Reproduce.**

```sh
source validation/creds.sh
# IONEX (real, ~170 KB)
curl -sL -H "Authorization: Bearer $GP_EARTHDATA_TOKEN" -o /tmp/gim.inx.gz \
  https://cddis.nasa.gov/archive/gnss/products/ionex/2023/001/IGS0OPSFIN_20230010000_01D_02H_GIM.INX.gz
gunzip -f /tmp/gim.inx.gz
IONEX_REAL=/tmp/gim.inx cargo test -p dolphin-corrections real_ionex -- --nocapture
# OPERA L4 (real, ~2 GB) — granule URL from CMR short_name=OPERA_L4_TROPO-ZENITH_V1
OPERA_L4_REAL=/path/opera_l4_tropo.nc cargo test -p dolphin-corrections real_opera_l4 -- --nocapture
```

## Interferogram sign convention (2026-06-17, v1.3.0)

The interferogram is formed `ref·conj(sec)` = `pl[i]·conj(pl[j])` for pair `(i=ref, j=sec)`
(`displacement.rs::unwrap_pair`), matching dolphin production `interferogram.py`. The earlier
`sec·conj(ref)` order **globally inverted the LOS displacement and velocity sign of every
release v1.0.0–v1.2.0**. It was invisible because `oracle/gen_displacement.py` carried the
identical inversion, so the sign-sensitive contracts proved Rust agreed with a *flipped*
oracle, not with production. This section brings the fix to the IONEX/NISAR real-data bar:
an always-on analytic guard plus a gated real-data test against a full production `dolphin run`.

**Always-on analytic guard — PASS** (`dolphin-workflows` test
`sign_convention::displacement_sign_matches_ref_conj_sec_convention`, no network, no oracle
fixture). A noise-free single-burst stack carries a positive, monotonic, cycle-free LOS ramp
(range *increasing* in time away from a zero-phase reference column). Under `ref·conj(sec)`
the recovered displacement at the far column is **+4.65 mm** (positive); reverting `unwrap_pair`
to `sec·conj(ref)` makes it **−4.65 mm** (exact negation) and the test **goes red** — verified
by flipping the order locally, watching it fail, and flipping back. This locks the convention
in CI regardless of data availability.

**Gated real-data test — PASS** (`dolphin-workflows` test
`sign_real_data::rust_displacement_sign_matches_production_on_corcoran_bowl`, gated on
`SIGN_REF_PROD_IFG`, skips when unset — same pattern as `real_ionex_parses_to_physical_delay`
/ `reads_real_nisar_granule`). Scene: OPERA CSLC-S1 frame **F38502 / burst T144-308015-IW2**,
the **Corcoran / Tulare-basin subsidence bowl** (lon −119.443, lat 36.021), 15 acquisitions
**2016-07-24 … 2017-01-20**, 12-day cadence, cropped 1024². dolphinRust runs the fixed pipeline
on the real stack; its displacement on the longest-baseline date (`20170120`) is compared
against the production `dolphin run` displacement (`work_oracle/timeseries/20160724_20170120.tif`),
demeaned, coherence-gated, with a vertical flip reconciling row order (dolphin's `timeseries/`
rasters carry an identity geotransform; dolphinRust writes north-up COGs — orientation is
orthogonal to per-pixel sign).

| measurement | before fix (`sec·conj(ref)`) | after fix (`ref·conj(sec)`) |
|---|---|---|
| rust vs production displacement corr, coh>0.7 | **−0.95** | **+0.95** (live test, 323,107 px) |
| rust vs production displacement corr, coh>0.9 | −0.99 | **+0.99** |
| production unwrapped ifg vs `arg(ref·conj(sec))` (strong px) | −1.0000 | **+1.0000** |
| bowl-pixel velocity sign (subsidence) | +0.136 (wrong: uplift) | **−0.136** (correct: subsidence) |

The two ifg orders are an exact pixelwise negation (`arg(conj z) = −arg(z)`), so the production
unwrapped ifg correlating **+1.0000** with `arg(ref·conj(sec))` and **−1.0000** with the reverse
is the conclusive localization. The eo-relevant `velocity_mm_yr` (subsidence vs uplift, which
drives risk tiers) now carries the correct sign.

**Reproduce.**

```sh
source validation/creds.sh
oracle/.venv/bin/python validation/fetch_real.py --burst T144_308015_IW2 --n 15 \
    --start 2016-07-01 --end 2017-02-01                       # F38502/Corcoran bowl
oracle/.venv/bin/python validation/crop_real.py --size 1024 --out /tmp/cv_cropped
validation/run_real.sh validation/runs/real_F38502_T144_bowl  # full dolphin run -> work_oracle/
SIGN_REF_PROD_IFG=validation/runs/real_F38502_T144_bowl \
  cargo test -p dolphin-workflows --test sign_real_data -- --nocapture
# always-on guard (no data needed):
cargo test -p dolphin-workflows --test sign_convention -- --nocapture
```

## Phase-bias / non-closure correction (2026-06-18, v1.4.0 Phase 4)

Michaelides et al. (RSE 2022). **Not in Python dolphin — this leads the oracle**, so there
is no parity target; validated by an analytic fixture (exact known-answer) plus a measured
reduction in non-closure. `dolphin-phaselink::phasebias`; opt-in via
`phase_linking.correct_phase_bias` (**off by default**, so default output is unchanged and the
oracle/sign contracts are untouched).

Model (first-order). The nearest-neighbour closure of the coherence matrix satisfies
`Ξ_k = β_k + β_{k+1} = B_{k+2} − B_k`, where `β_n` is the connection-1 fading bias and `B_n` the
cumulative bias the linked phase carries. A per-pixel **constant bias velocity**
`β̄ = mean_k(Ξ_k)/2`, cumulative `B_n = n·β̄`, is subtracted from the linked series
(`θ_n ← θ_n·e^{−j n β̄}`) before the interferogram network. One parameter per pixel — removes
the systematic part without over-fitting noise. Time-varying bias is a documented future
refinement.

- **Analytic contract — PASS** (`constant_bias_is_recovered_exactly`, `residual_is_zero_for_constant_bias`):
  a constant injected bias `c` gives closures `Ξ = 2c`; the estimate is exactly `c`, the
  corrected series `θ_n = φ_n + n·c` recovers `φ_n` to `< 1e-9`, and the residual closure is `0`.
- **Measured non-closure reduction — PASS** (`reduces_nonclosure_on_long_noisy_series`): a
  100-date series whose closures are a constant systematic bias plus deterministic zero-mean
  noise has its mean non-closure cut **0.800 → 0.095 rad (8.4×)** after the correction.
- **End-to-end wiring — PASS** (`phase_bias_correction_runs_end_to_end`, snaphu-gated): enabling
  `correct_phase_bias` runs through unwrap + SBAS and yields a finite displacement of the right
  shape. Default-off parity is the existing `end_to_end_displacement_matches_oracle` contract.

```sh
cargo test -p dolphin-phaselink --lib phasebias -- --nocapture   # analytic + reduction
```

## Open / pending

- **Real-data velocity absolute scale under strong signal** — RESOLVED (2026-06-17, v1.1.0):
  confirmed on Mexico City burst T005-008704-IW1 to TLS slope ≈1.03 with matching magnitude;
  see "Velocity absolute scale on a real deforming scene" above.
- **Velocity cadence fix** (divergence #2) — RESOLVED above; a `dolphin-workflows` change.
- **Per-stage CLI intermediates** (linked-phase SLCs, temp_coh, unwrapped ifgs) are not
  persisted by the dolphinRust CLI, so end-to-end comparison is on displacement + velocity;
  those intermediates are covered by the §A contract tests against the same oracle.

## Reproduce

Synthetic tier:

```sh
oracle/.venv/bin/python validation/gen_stack.py --outdir /tmp/d --speckle 0.05  # stack
validation/run.sh 0.05      # gen config, run both engines, compare
validation/run.sh 0.0       # noise-free (pure-algorithm) agreement
```

Real OPERA tier (needs the Earthdata token in `.env`):

```sh
source validation/creds.sh
oracle/.venv/bin/python validation/fetch_real.py --burst T144_308011_IW2 --n 9   # download
oracle/.venv/bin/python validation/scan_coherence.py                              # find window
oracle/.venv/bin/python validation/crop_real.py --row0 <r> --col0 <c> --size 384  # crop
validation/run_real.sh                                                            # both engines + compare
```

Real-data velocity absolute-scale check (Mexico City strong-signal scene):

```sh
source validation/creds.sh
oracle/.venv/bin/python validation/fetch_real.py --burst T005_008704_IW1 --n 13 \
    --start 2023-01-01 --end 2023-07-15                                           # download
oracle/.venv/bin/python validation/crop_real.py --burst T005 \
    --row0 3656 --col0 3488 --size 384 --out validation/real_data/cropped_mexico  # crop on the gradient
validation/run_real.sh validation/real_data/cropped_mexico real_mexico_T005       # both engines
oracle/.venv/bin/python validation/velocity_scale.py \
    --run validation/runs/real_mexico_T005                                        # OLS + TLS scale
```
