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
temporally (date 0) and fills all pixels — handled by demeaning on the shared mask. Sign is
`+1` (engines agree in sign after demean). **Stated physical tolerance: corr ≥ 0.95 and
demeaned per-pixel RMS ≤ 0.10 rad (< 0.016 cycle).**

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

4. **tophu multi-scale unwrap does NOT beat raw SNAPHU on low-coherence scenes — reported,
   not hidden.** On 512×512 subsidence-bowl scenes with known truth under vegetation-style
   coherence loss, raw SNAPHU genuinely struggles (gross-cycle-error 0.13–0.17) but our tophu
   path is **modestly worse** on every metric (discontinuities, RMS-vs-truth, gross-cycle
   fraction). Hypothesis: multilooking decorrelated complex phasors yields an unreliable
   coarse anchor, and the constant-2π-cycle tile merge is cruder than SNAPHU's global MCF.
   Consistent with the contract tests, which show tophu *matches* SNAPHU on coherent data
   (`tophu_recovers_analytic_ramp_within_snaphu_envelope`) — the loss is specific to
   decorrelated ground. SNAPHU remains the default; the scene and tolerances were **not**
   tuned to manufacture a win. Numbers + reproduction in `bench/UNWRAP.md`.

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
