# GPU phase-linking — first-class backend (accuracy & speed)

A `wgpu`/Metal GPU compute backend for dolphinRust's **own** phase-linking kernel,
compiled into the **default build** and runtime-selected (`worker_settings.compute_backend
= auto | cpu | gpu`). The GPU runs single precision (`f32` — Apple GPUs are
f32-only); the CPU (`faer`, `f64`) path stays the correctness reference and the
automatic fallback. The same WGSL runs on Apple Metal (here) and on discrete
NVIDIA/AMD (Vulkan/DX12) via wgpu, unchanged.

- **Machine:** Apple M2 Pro, Metal backend (integrated GPU, unified memory).
- **Kernel:** one GPU thread per output pixel; covariance + EVD/EMI in-shader,
  complex as `vec2<f32>`. EVD is power iteration; EMI is shifted power iteration
  for the least eigenvector (no per-iteration solve).
- **Reproduce:** `cargo run -p dolphin-phaselink --release --example gpu_bench`
  (accuracy needs the real-stack oracle fixtures:
  `oracle/.venv/bin/python oracle/gen_phaselink_real.py`).

All numbers below are measured on this machine. Nothing is assumed.

---

## 1. Accuracy (priority)

Real OPERA CSLC-S1 stack — cropped Mexico burst T005-008704-IW1, 13 acquisitions,
384×384, `half_window` (11,5), strides (1,1), `reference_idx` 0. The dolphin
v0.35.0 oracle (`process_coherence_matrices`) is generated on the identical
coherence input. Displacement: Sentinel-1 C-band, λ/4π ≈ **4.414 mm/rad**. Phase
deltas wrapped to (−π, π]; "overlap" is the min per-pixel eigenvector overlap
`|⟨v_gpu, v_cpu⟩|`. **All rows are over every pixel** (no coherent-only masking).

| Comparison (all 147,456 px)          | overlap ≥ | median Δφ        | p99 Δφ            | max Δφ              |
|--------------------------------------|-----------|------------------|-------------------|---------------------|
| **EVD** GPU(f32) vs CPU(f64)         | **1.0000** | 6.0e-8 rad (~0)  | 3.6e-7 rad (~0)   | 4.0e-2 rad (**0.176 mm**) |
| **EMI raw** GPU(f32) vs CPU(f64)     | 0.343     | 1.2e-7 rad (~0)  | 1.3e-2 rad (0.057 mm) | 3.14 rad (**13.85 mm**) |
| **EMI hybrid** GPU vs CPU(f64)       | **0.9991** | 1.2e-7 rad (~0)  | 2.0e-4 rad (0.001 mm) | 1.4e-1 rad (**0.607 mm**) |
| **EMI hybrid** GPU vs oracle         | 0.9888    | 1.2e-7 rad (~0)  | 2.1e-4 rad (0.001 mm) | 1.21 rad (5.35 mm) |
| **EMI CPU**(f64) vs oracle           | 0.9888    | 5.5e-8 rad (~0)  | 9.1e-7 rad (~0)   | 1.21 rad (5.35 mm) |

**Verdict.**
- **EVD is production-accurate on the GPU** — across *every* pixel the f32 EVD
  eigenvector matches the f64 CPU result to overlap 1.0000 and a worst-case
  **0.176 mm** (median sub-nanometre).
- **EMI raw f32 has a π-rad tail** (max 13.85 mm) on ill-conditioned / near-degenerate
  pixels — the spike's known limitation.
- **The hybrid removes the tail.** The GPU EMI kernel flags pixels whose least
  eigenvector is ill-conditioned — small bottom eigengap (recovered via one
  Hotelling-deflated power pass), a Rayleigh quotient indicating wrong-mode
  capture, low mean coherence, or a **borderline Cholesky pivot** (where the f32
  vs f64 PD decision, EMI vs EVD, can differ). The host recomputes that flagged
  minority (**5.9% = 8,685 px**) on the f64 `faer` path. Result: **EMI matches the
  CPU reference to a worst-case 0.607 mm across all 147,456 px — no π-rad tail.**
- The `hybrid vs oracle` max (5.35 mm) equals the `CPU vs oracle` max: that pixel
  is a **Rust-CPU vs dolphin** difference (a degenerate eigenvector both pick
  differently), not a GPU error — the GPU hybrid tracks the CPU reference exactly.

Fixture parity (`tests/gpu_contract.rs`): analytic + synthetic-DS EVD/EMI match
the CPU path to overlap > 0.999 and ≤ 5e-7 rad; SHP-masked covariance matches the
dolphin SHP oracle to 5.4e-7; β regularization matches CPU sub-mm; EMI is
**run-to-run deterministic** (bit-identical) at 384²/nslc 13 and at nslc 32.

---

## 2. Speed — END-TO-END (honest)

Wall-clock of the **full** path — covariance **+** phase-linking **+** host↔device
transfer **+** the hybrid's CPU recompute — i.e. exactly what `run_displacement`
runs per ministack. nslc = 13, median of 5 warm reps. (The earlier spike timed the
phase-linking *kernel only*, covariance excluded; those numbers are not comparable.)

**Real Mexico 384² stack:** CPU **2.65 s**, GPU **4.01 s** → **0.66× (GPU slower).**

| synthetic size | CPU (s) | GPU (s) | speedup |
|---------------:|--------:|--------:|--------:|
|           64²  |  0.061  |  0.079  |  0.77×  |
|          128²  |  0.235  |  0.247  |  0.95×  |
|          192²  |  0.566  |  0.535  |  1.06×  |
|          256²  |  1.019  |  0.936  |  1.09×  |
|          384²  |  2.257  |  2.074  |  1.09×  |
|          512²  |  4.035  |  3.689  |  1.09×  |

- **On this integrated M2 Pro, the end-to-end GPU win is marginal at best, and on
  the real stack the GPU is *slower* (0.66×).** The honest reasons: (a) the GPU
  covariance result (~200 MB at 384²) is read back and round-tripped f32→f64→f32
  through the hybrid; (b) the hybrid recomputes 5.9% of pixels on the CPU; (c)
  this is an *integrated* GPU sharing the M2 Pro's memory bandwidth, and the CPU
  baseline is strong (`faer` + `rayon` across all cores). On clean synthetic
  stacks (fewer flagged pixels) the GPU edges ahead at **≥ 192²**, but the margin
  is thin.
- **The payoff is portability, not this machine.** The same WGSL ports unchanged
  to a discrete NVIDIA/AMD GPU with far more FP32 throughput, dedicated bandwidth,
  and headroom — that is the motivating next test. On integrated Apple silicon,
  prefer `compute_backend = auto` (CPU below the crossover) or `cpu`.
- Wall-clock carries ±10–20% run-to-run variance (thermal/scheduling); the
  ordering and crossover are stable across runs.

Known optimizations not yet taken (would help the integrated case, required on
none): keep the coherence on-device between the covariance and EMI dispatches
(eliminate the 200 MB readback + f64 round-trip); recompute flagged pixels from
the f32 coherence instead of upcasting the whole stack.

---

## 3. Implementation notes & limits

- **`MAX_NSLC = 32`** (was 16). The EMI per-pixel scratch (Γ, its Cholesky factor,
  Γ⁻¹) lives in **threadgroup memory**, sized by pipeline-override constants
  (`WG`, `GAM_LEN = WG·nslc²`) so `2·WG·nslc²·4 B` fits a 24 KiB budget (WG ≈ 18 at
  nslc 13, 3 at nslc 32). Per-thread *private* scratch spilled out of registers at
  nslc 32 and produced run-to-run **nondeterministic** output at 384²; threadgroup
  memory never spills, so EMI is deterministic at every size. nslc > 32 returns a
  clean error and the backend falls back to CPU.
- **Hybrid reliability flags** (tuned on the real Mexico stack; generous — a false
  positive only costs a CPU recompute): bottom relative eigengap `< 0.07·λ_max`,
  Rayleigh `> 0.5·λ_max`, mean coherence `< 0.10`, min Cholesky pivot `< 1e-4`.
- **GPU covariance** supports the **SHP neighbor mask** and the EMI **β
  regularization** + `zero_correlation_threshold` (the spike was rectangular-window,
  β = 0 only).
- **Backend selection + fallback.** `auto` uses the GPU at/above the ~128²
  crossover and the CPU below; `gpu`/`cpu` are honored; **no adapter / unsupported
  / `no-gpu` build → automatic CPU fallback with a warning, never a panic.**

## 4. Bottom line

- **Accurate?** Yes. **EVD** is sub-mm (≤ 0.176 mm) across a full real stack;
  **EMI** with the hybrid is sub-mm (≤ 0.607 mm) across *every* pixel — the π-rad
  tail is gone. The GPU tracks the f64 CPU reference, which tracks the dolphin
  oracle.
- **Faster?** Not on this integrated M2 Pro: end-to-end it is **0.66× on the real
  stack** and only ~1.09× on clean synthetic stacks above ~192². The first-class
  value here is **correctness + portability**: the kernel is production-accurate
  and runs unchanged on discrete hardware, where the FP32 headroom is.
