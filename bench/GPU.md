# GPU phase-linking spike — accuracy & speed (R4)

A wgpu/Metal GPU path for dolphinRust's **own** phase-linking kernel, behind the
`gpu` cargo feature on `dolphin-phaselink`. The GPU runs single precision (`f32`
— Apple GPUs are f32-only); the CPU (`faer`, `f64`) path stays the default and
the correctness reference. Two questions, answered in order: **(1) still
accurate? (2) faster than the CPU path?**

- **Machine:** Apple M2 Pro, Metal backend (integrated GPU, unified memory).
- **Kernel:** one GPU thread per output pixel; covariance + EVD/EMI in-shader,
  complex as `vec2<f32>`. Eigensolvers are power / shifted-power iteration
  (dolphin's iterative approach), not a direct decomposition.
- **Reproduce:** `cargo run -p dolphin-phaselink --features gpu --release --example gpu_bench`
  (accuracy needs the real-stack oracle fixtures: `oracle/.venv/bin/python oracle/gen_phaselink_real.py`).

All numbers below are measured on this machine in this spike. Nothing is assumed.

---

## 1. Accuracy (priority)

Real OPERA CSLC-S1 stack — cropped Mexico burst T005-008704-IW1, 13 acquisitions,
384×384, `half_window` (11,5), strides (1,1), `reference_idx` 0. The dolphin
v0.35.0 oracle (`process_coherence_matrices`) is generated on the identical
coherence input. Displacement conversion: Sentinel-1 C-band, λ/4π ≈ **4.414
mm/rad**. Phase deltas are wrapped to (−π, π]; "overlap" is the per-pixel
eigenvector overlap `|⟨v_gpu, v_cpu⟩|` (its minimum over the reported pixels).

| Comparison                | overlap ≥ | median Δφ          | p99 Δφ            | max Δφ            |
|---------------------------|-----------|--------------------|-------------------|-------------------|
| **EVD** GPU(f32) vs CPU(f64), all px | **1.0000** | 6.0e-8 rad (2.6e-7 mm) | 3.6e-7 rad (~0 mm) | 4.0e-2 rad (**0.18 mm**) |
| **EMI** GPU(f32) vs CPU(f64), all px | 0.345 | 1.2e-7 rad (5.3e-7 mm) | 1.2e-2 rad (0.05 mm) | 3.14 rad (13.9 mm) |
| **EMI** GPU(f32) vs CPU(f64), coherent (γ̄>0.6, 40% of px) | 0.345 | 1.2e-7 rad (5.3e-7 mm) | 2.7e-2 rad (0.12 mm) | 3.13 rad (13.8 mm) |
| **EMI** GPU(f32) vs oracle, coherent | 0.345 | 1.8e-7 rad (7.9e-7 mm) | 2.7e-2 rad (0.12 mm) | 3.13 rad (13.8 mm) |
| **EMI** CPU(f64) vs oracle, coherent | 1.0000 | 3.0e-8 rad (1.3e-7 mm) | 3.6e-7 rad (~0 mm) | 4.7e-5 rad (~0 mm) |

The CPU-vs-oracle row is the control: the Rust CPU path reproduces dolphin
v0.35.0 to ~3e-8 rad, so any GPU delta below is f32-vs-f64, not an algorithm
difference.

**Verdict.**
- **EVD is production-accurate on the GPU.** Across *every* pixel of the real
  stack the f32 EVD eigenvector matches the f64 CPU result to overlap 1.0000 and
  the referenced phase to a **worst-case 0.18 mm** (median sub-nanometre). Sub-mm
  with margin, vs both CPU and (transitively) the oracle.
- **EMI is sub-mm for the bulk but has an ill-conditioned tail.** Median and p99
  are sub-0.12 mm — for the large majority of pixels GPU f32 EMI is sub-mm. But a
  minority of pixels diverge by up to π rad (13.9 mm): these are decorrelated /
  near-singular-Γ pixels where the *least* eigenvector is numerically ill-defined,
  so the GPU's iterative solver and faer's direct decomposition pick different
  vectors in a near-degenerate subspace. The low overlap floor (0.345) is these
  pixels. They are exactly the pixels masked out by temporal-coherence
  thresholding downstream — but per the honesty rule: **GPU EMI is *not*
  uniformly sub-mm on a real stack; EVD is.**

Fixture parity (analytic + synthetic DS, `tests/gpu_contract.rs`): EVD and EMI
both match the CPU path to overlap > 0.999 and ≤ 5e-7 rad, and the EMI→EVD
fallback on a singular Γ fires identically to the CPU path.

---

## 2. Speed

Phase-linking wall-clock (the estimator stage only; covariance excluded, both
engines consume the same coherence stack), nslc = 13, synthetic KMS coherence,
median of 5 warm reps. CPU = `process_coherence_matrices` (faer + rayon over all
cores); GPU = `process_coherence_matrices_gpu` (EMI).

| stack size | CPU (s) | GPU (s) | speedup |
|-----------:|--------:|--------:|--------:|
|       64²  |  0.021  |  0.024  |  0.88×  |
|      128²  |  0.086  |  0.061  |  1.41×  |
|      192²  |  0.191  |  0.128  |  1.49×  |
|      256²  |  0.344  |  0.218  |  1.58×  |
|      384²  |  0.760  |  0.491  |  1.55×  |
|      512²  |  1.351  |  0.830  |  1.63×  |

- **Crossover:** the GPU starts winning at **≈ 128²** pixels. Below that (64²),
  dispatch + readback overhead dominates and the **CPU wins** (0.88×).
- **At scale (≥ 256²): ~1.5–1.6× faster.** A real but **modest** win. The honest
  reasons: this is an *integrated* GPU sharing the M2 Pro's memory bandwidth, and
  the CPU baseline is strong (faer's tuned selfadjoint EVD across all cores). A
  discrete NVIDIA GPU — to which the same WGSL ports unchanged — would have far
  more FP32 throughput and headroom; that is the motivating next test.
- Wall-clock timings carry ±10–20% run-to-run variance (thermal / scheduling);
  the speedup ordering and crossover are stable across runs.

---

## 3. Implementation notes & limits

- **`MAX_NSLC = 16`.** The EMI kernel holds Γ, its Cholesky factor, and Γ⁻¹ in
  per-thread `private` scratch (nslc² each). At `MAX_NSLC = 32` that scratch
  (~8 KB/thread × tens of thousands of threads) spilled and produced
  **run-to-run nondeterministic** EMI output at 384² (a whole-dispatch failure
  mode — median jumping from 1e-7 to 1.2 rad between identical runs). Dropping to
  16 (covers ministack ≤ 16; this stack is 13) cut the scratch 4× and made EMI
  **fully deterministic**. `nslc > 16` returns a clean error. EVD never hit this
  (it keeps only length-nslc iterate vectors).
- **EMI least-eigenvector method:** M = Γ⁻¹⊙C is PSD (Schur product of PSD
  factors), so we power-iterate M for λ_max, then power-iterate λ_max·I − M whose
  dominant mode is M's least mode. A loose Gershgorin shift was tried first and
  failed to converge (compressed spectrum); the tight λ_max shift fixed it on
  well-conditioned matrices but cannot rescue genuinely near-degenerate ones (the
  EMI tail above).
- **Not modeled in the GPU path (spike scope):** SHP masking in covariance
  (rectangular window only), the `β > 0` Γ regularization, and a GPU covariance
  *speed* comparison (covariance is GPU-accelerated and matches CPU to 4.5e-7,
  but only phase-linking speed is measured here).

## 4. Bottom line

- **Accurate?** Yes for **EVD** — sub-mm (≤0.18 mm) across a full real stack, f32
  vs f64. **EMI** is sub-mm for the bulk but has a π-rad tail on ill-conditioned
  pixels; use EVD on the GPU, or keep EMI on the f64 CPU (the default).
- **Faster?** Yes above ~128² — **~1.5–1.6×** on this integrated M2 Pro GPU,
  crossover ≈128², CPU wins below. Headroom is on discrete hardware, where the
  same kernel runs unchanged.
