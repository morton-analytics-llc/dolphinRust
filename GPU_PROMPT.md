# dolphinRust — GPU phase-linking spike (dynamic workflow loop)

Add a GPU-accelerated phase-linking path for **dolphinRust's own kernel** on the **Apple
Metal** GPU (this dev machine), and answer two questions, in order: **(1) is it still
accurate, (2) is it faster than our CPU path.** This is *not* about matching Python dolphin's
GPU mode — ignore that entirely. Pulls the R4 GPU item forward as a spike; the CPU (faer)
path stays the default and the correctness reference.

---

## Prompt

Build a GPU phase-linking path behind a `gpu` cargo feature on `dolphin-phaselink`, using
**wgpu** (runs on Metal here; the same WGSL kernel ports to NVIDIA later — no rewrite). Work
as a dynamic, self-paced loop, contract test first, each step gated by `cargo fmt` +
`cargo clippy --all-targets -- -D warnings` (with and without `--features gpu`) + `cargo test`
+ `cargo doc --no-deps`. Commit on a branch `gpu-phaselink`; **push nothing without my
sign-off.** Honesty rule: report the real accuracy delta and real speedup — if the GPU path
falls outside tolerance or is *slower*, say so plainly; do not weaken a tolerance to pass.

Read first: `CLAUDE.md`, `crates/dolphin-phaselink/CLAUDE.md`, `ROADMAP.md` (R4), `bench/`.

Design constraints (decided — don't re-litigate):
- **Apple GPUs are f32-only.** The GPU kernel runs in single precision; the CPU path stays
  f64 and is the reference. GPU-vs-CPU will not be bit-identical — measure the difference,
  don't assume it. (dolphin's own GPU path is f32 too, so this is expected.)
- **One GPU thread per output pixel.** Covariance + eigenvector per pixel; complex as
  `vec2<f32>` in WGSL. Apple unified memory — hand the SLC stack to the GPU, no PCIe copy.
- **Eigensolver in-shader = power / inverse iteration** (dolphin's algorithm, simplest to
  port), not a direct decomposition. EVD (largest eigenvector) first; EMI second.
- CPU remains the default feature; the default build must be unchanged.

Loop:
1. **Scaffold + adapter check.** Add the `gpu` feature + optional `wgpu` dep. Enumerate
   adapters and confirm a Metal device is reachable on this machine. ⛔ If no GPU adapter
   initializes, stop and tell me.
2. **GPU covariance** (the #1 hot path). WGSL compute shader + host wgpu setup (buffers, bind
   groups, dispatch). Contract test: GPU covariance vs CPU covariance within f32 tolerance on
   a fixture.
3. **GPU EVD eigenvector** (power iteration in-shader). Contract test: eigenvector overlap
   `|⟨v_gpu, v_cpu⟩|` > 0.999 and referenced phase within the existing envelope, on the
   analytic + synthetic DS fixtures. Then EMI (inverse iteration) if EVD lands cleanly.
4. **Wire a `gpu`-gated phase-linking entry** mirroring the CPU one (same inputs/outputs).
5. **Accuracy (priority).** GPU(f32) vs CPU(f64) vs the dolphin v0.35.0 oracle on a real
   stack (reuse an already-fetched stack under `validation/`/`bench/` if present, else
   `source validation/creds.sh` and fetch one). Record the measured accuracy delta — in
   millimeters as well as radians — in `bench/GPU.md`. State whether it stays sub-mm.
6. **Speed.** Extend `bench/bench.py` (or a sibling) with a GPU mode; compare GPU vs CPU
   phase-linking on a **large** stack (≥384², larger is better — GPUs need scale to win).
   Record the speedup and the crossover point (stack size below which CPU wins) in
   `bench/GPU.md`. Be explicit about where the GPU loses.

Update `ROADMAP.md`/`STATUS.md` notes as it lands. Otherwise don't debate directions — state
the load-bearing assumption in one line and proceed.

**Definition of Done:**
- [ ] `gpu` feature builds and runs on Apple Metal (wgpu); the default (CPU) build is unchanged.
- [ ] GPU phase-linking matches the CPU path within tolerance (eigenvector overlap > 0.999,
      phase within envelope) on fixtures **and** a real stack; the f32 accuracy delta vs CPU
      and vs the oracle is documented in `bench/GPU.md`, in mm and rad.
- [ ] `bench/GPU.md` reports the real GPU-vs-CPU phase-linking speedup on a large stack, with
      the crossover point and where the GPU loses — honest numbers.
- [ ] Gates green for both the default build and `--features gpu` (fmt, clippy -D warnings,
      test, doc). Committed on branch `gpu-phaselink`; unpushed.

---

## Launching with elevated permissions

Two steps. **Step 2 is a slash command typed inside Claude Code — not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
source validation/creds.sh          # in case accuracy validation needs to fetch a real stack
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read GPU_PROMPT.md and build + validate the GPU phase-linking path per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git unattended. `/loop` with no interval =
dynamic self-pacing. It stops if no Metal GPU adapter initializes, or if GPU accuracy falls
outside the documented tolerance (it reports rather than fudges).
