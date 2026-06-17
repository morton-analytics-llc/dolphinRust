# dolphinRust — GPU as a first-class backend (dynamic workflow loop)

Promote the GPU phase-linking spike (branch `gpu-phaselink`, `bench/GPU.md`) into a
**production, first-class compute backend**: compiled into the default build, runtime-
selected, correct on *all* pixels, feature-complete, wired end-to-end, and validated. The
CPU (faer, f64) path stays the correctness reference and the automatic fallback. Same WGSL
runs on Apple Metal (here) and discrete NVIDIA (later) via wgpu.

---

## Prompt

Make the GPU backend first-class. Start from branch `gpu-phaselink`; work on a new branch
`gpu-first-class`. Dynamic, self-paced loop, contract test first; gate every step on
`cargo fmt` + `cargo clippy --all-targets -- -D warnings` + `cargo test` + `cargo doc
--no-deps`. **Push nothing without my sign-off.** Honesty rule: the CPU path is the
reference — report the real end-to-end speedup and the real accuracy delta; if first-class
GPU is only marginally faster (or slower) on this integrated Apple GPU, say so plainly (the
payoff is portability to discrete NVIDIA). Do not weaken a tolerance to pass.

Read first: `bench/GPU.md` (spike findings + limits), `crates/dolphin-phaselink/CLAUDE.md`,
`CLAUDE.md`, `ROADMAP.md`.

The spike proved EVD is GPU-accurate and the kernel ports; first-class means closing its
documented gaps. Required work:

1. **EMI correctness on GPU — the gating item.** The spike's EMI has a π-rad tail on
   ill-conditioned / near-degenerate pixels. First-class is not allowed to ship that. Make
   GPU EMI match CPU EMI within tolerance on **every** pixel of a real stack via a hybrid:
   the GPU kernel flags non-converged / near-degenerate pixels (small eigengap, low
   coherence, iteration cap hit), and those pixels are recomputed on the f64 CPU path.
   Result must show **no π-rad tail** — max Δφ sub-mm across all pixels vs CPU-only EMI.
   ⛔ If a hybrid can't get EMI all-pixel-accurate, stop and report rather than ship it.

2. **Lift `MAX_NSLC`.** 16 is too small (ministack + compressed SLCs exceed it). Support at
   least nslc 32 with a *proper* fix for the per-thread scratch spill (threadgroup/shared
   memory or chunked Cholesky), not a cap — and prove EMI stays **deterministic** at 384²
   and the new size. Validate run-to-run identical output.

3. **GPU covariance feature parity.** Support the **SHP neighbor-array mask** and the **β
   regularization** on the GPU (spike was rectangular-window, β=0 only). Without SHP the GPU
   can't do real DS processing. Contract-test GPU vs CPU covariance with SHP + β on.

4. **Runtime backend selection + fallback.** Compile wgpu into the **default build** (GPU is
   first-class, not an opt-in feature). Add a config knob (`worker_settings.gpu_enabled` to
   match dolphin, or `compute_backend: auto|cpu|gpu`): `auto` uses GPU above the ~128²
   crossover and CPU below; explicit modes honored. **No GPU adapter / unsupported → automatic
   CPU fallback with a warning, never a crash.** Provide a `no-gpu` build feature for hosts
   that can't link wgpu. CPU remains the default-safe correctness reference.

5. **Wire end-to-end.** `dolphin_workflows::run_displacement` uses the selected backend
   through the real pipeline — not just the kernel crate. One config, CPU or GPU, same result.

6. **Validate end-to-end.** Full GPU pipeline vs CPU vs the dolphin v0.35.0 oracle on a real
   stack (`source validation/creds.sh` if a fetch is needed): accuracy across **all** pixels
   (no tail) and **end-to-end** wall-clock (include covariance + host↔device + hybrid-fallback
   overhead, not just the kernel). Update `bench/GPU.md` and `VALIDATION.md` with honest
   numbers and the crossover.

7. **Docs + release hygiene.** README + `docs/usage.md`: the GPU backend, how to select it,
   the fallback behavior, the f32-vs-f64 accuracy note, and platform support. `CHANGELOG`
   entry. Update `ROADMAP.md` (GPU moves from R4-deferred to shipped).

Update `STATUS.md` as items land. Otherwise don't debate directions — state the load-bearing
assumption in one line and proceed.

**Definition of Done:**
- [ ] GPU compiled into the default build; backend runtime-selectable; **no-adapter →
      automatic CPU fallback** (tested by simulating no adapter), never a panic.
- [ ] GPU **EMI matches CPU EMI sub-mm on every pixel** of a real stack — no π-rad tail —
      via the hybrid CPU fallback for ill-conditioned pixels; EVD likewise.
- [ ] `MAX_NSLC` ≥ 32 with the scratch spill properly fixed; EMI deterministic at 384².
- [ ] GPU covariance supports SHP masking + β, matching CPU within f32 tolerance.
- [ ] `run_displacement` runs end-to-end on the GPU backend; full-pipeline accuracy vs CPU +
      oracle within tolerance on a real stack; end-to-end speedup + crossover in `bench/GPU.md`.
- [ ] Gates green (default build *and* `no-gpu`): fmt, clippy -D warnings, test, doc.
- [ ] README/usage/CHANGELOG/ROADMAP updated; committed on `gpu-first-class`; unpushed.

---

## Launching with elevated permissions

Two steps. **Step 2 is a slash command typed inside Claude Code — not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
source validation/creds.sh          # for real-stack end-to-end validation
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read GPU_FIRSTCLASS_PROMPT.md and promote the GPU backend to first-class per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git unattended. `/loop` with no interval =
dynamic self-pacing. It stops if GPU EMI can't be made all-pixel-accurate (it reports rather
than shipping a π-rad tail) or if no Metal adapter initializes.
