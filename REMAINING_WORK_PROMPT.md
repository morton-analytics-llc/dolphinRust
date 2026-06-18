# dolphinRust — finish v1.3.0 + complete v1.4.0 (dynamic workflow loop)

The remaining roadmap, worked **one phase at a time** in a self-paced dynamic loop: finish
v1.3.0 (the deferred tropospheric warp → tag), then complete v1.4.0 (NRT incremental updates,
performance, phase-bias, 3D-unwrap interface → tag). This is the project's dynamic/iterative
flow: take the next phase, write its contract (red) first, make it green, land it, move on.

Current state (origin/main 469c4d2): v1.0.0–v1.2.0 tagged; v1.3.0 part 1 (NISAR/L-band ingest)
+ part 2 (atmospheric corrections) landed but **untagged** because tropo application to a UTM
frame is deferred. The ifg sign convention is fixed and guarded (`tests/sign_convention.rs`) —
**every phase below must preserve it; that guard must stay green.**

---

## Global rules (apply to every phase)

- Contract test FIRST (red), then make it green. A phase is done only when its contract is green.
- Gate every step: `cargo fmt` + `cargo clippy --all-targets -- -D warnings` + `cargo test` +
  `cargo doc --no-deps`, on **default (== gpu)** and **`--features no-gpu`**.
- One branch per landed unit (`v1.4-phasebias`, `v1.4-unwrap-iface`). **Run your own checkpoints
  and land autonomously** — when a unit's gates + contracts are green (verified, not assumed),
  merge `--no-ff` to `main`, push, and continue to the next phase without stopping for sign-off.
  Tag `v1.4.0` once Phase 5 and Phase 2b are both in. Only stop and report on a genuine blocker
  (an ambiguous contract, a benchmark that won't clear its bar, an unreachable source) or when the
  whole sprint is done.
- Honesty rule: report real numbers. Never weaken a tolerance, fake a benchmark win, or stub a
  dependency. If something can't meet its bar, stop and report with a hypothesis.
- Elevate genuine blockers: if a phase needs an architecture/contract decision not answerable
  from the code or ROADMAP, stop and ask — don't guess.
- Keep `STATUS.md`/`ROADMAP.md`/`CHANGELOG.md`/`docs/` current as each phase lands. State the
  load-bearing assumption in one line and proceed; don't debate directions.

Read first: `CLAUDE.md`, `ROADMAP.md` (v1.3.0 + v1.4.0), `VALIDATION.md`, `bench/README.md` +
`bench/results.json` (the committed dolphinRust-vs-dolphin baseline), and the per-crate
`CLAUDE.md` of whichever crate a phase touches.

---

## Phase 1 — Tropospheric 4326→UTM warp (finishes v1.3.0)

The OPERA L4 tropo product is a global EPSG:4326 grid; applying it to a UTM NISAR/DISP-S1 frame
needs a 4326→UTM warp the bilinear resampler doesn't do (it `warn!`s on CRS mismatch). Add the
warp (GDAL warp/reproject) so the real global grid resamples onto the frame grid and the per-date
tropo delay is actually subtracted end-to-end. Contract: a synthesized 4326 L4 fixture warped
onto a UTM frame yields the known delay at known frame pixels; the CRS-mismatch `warn!` path is
gone. Best-effort real-data: warp the real `OPERA_L4_TROPO-ZENITH_V1` granule onto a real UTM
frame and record the applied-correction magnitude in `VALIDATION.md` (or defer-with-receipts if
unreachable). **Done bar:** tropo correction applies end-to-end on a UTM frame (fixture-proven;
real if reachable); gates green. **Then, on sign-off: tag `v1.3.0`** (both parts + the warp).

## Phase 2 — NRT incremental ministack updates (v1.4.0)

A streaming mode that folds a newly arrived acquisition into an existing time series using the
carried **compressed SLC**, without reprocessing the whole stack. This is the operational lead
over batch dolphin and the payoff of the speed edge. Contract/exit: an incremental update of an
existing series with one new acquisition matches a **full rerun** of the extended stack to
physical tolerance (state it). ⛔ If the compressed-SLC carry/handoff contract is ambiguous from
the phase-linking code, stop and ask before designing it. **Done bar:** incremental update ==
full rerun within tolerance; contract green; gates green.

*(Phase 2 status: the incremental phase-linking **core** is ✅ done and bit-identical to a full
rerun — `run_sequential_resumable` / `update_sequential`, merged to main. Phase 2b below carries
it to a usable front door.)*

## Phase 2b — End-to-end NRT front door (v1.4.0, REQUIRED before tag)

The incremental core is not operationally "done" until eo can call it: wire an end-to-end
`update_displacement` (incremental phase-linking core + the non-causal downstream recompute) and a
CLI streaming entry point, so a newly arrived acquisition produces an updated displacement product
without re-phase-linking the sealed history. Sequence this **after Phase 3** so it builds on the
optimized streaming-I/O path. **Done bar:** an end-to-end incremental update from a new acquisition
matches a full `run_displacement` of the extended stack (within the same tolerance as Phase 2);
exposed via a public API + CLI entry; contract green; gates green. **This is a required v1.4 unit —
not a gated-phase deferral; v1.4.0 does not tag without it.**

## Phase 3 — Performance optimization vs the baseline (v1.4.0)

Beat the committed pre-R1 baseline (`bench/results.json`, dolphinRust vs Python dolphin v0.35.0).
Candidate work: faer small-matrix (N×N covariance, N≈10–30) tuning, `EagerLoader`-style block
prefetch, streaming I/O, thread-pool/BLAS contention. Re-run the **existing** bench harness
(don't alter the scenes to flatter results) and publish the measured multiple over Python dolphin
in `bench/`. **Done bar:** a documented, reproduced speedup vs the baseline with honest numbers
(if a given optimization doesn't help, say so and drop it); no accuracy regression (the oracle
contracts + sign guard stay green); gates green.

## Phase 4 — Phase-bias / non-closure correction (v1.4.0)

Michaelides et al., RSE 2022. **Not in Python dolphin** — this leads the oracle, so there is no
oracle parity; validate by analytic fixture + a measured **reduction in non-closure** on a long
synthetic series, using the closure-phase layer already shipped in v1.2.0. CPU path. **Done bar:**
analytic contract green; phase-bias correction measurably reduces non-closure on a long series
(numbers recorded); gates green.

## Phase 5 — 3D-unwrap-ready dispatch interface (v1.4.0)

Abstract the unwrap backend so a spurt-style 3D spatiotemporal solver can drop in later without a
refactor. Interface/trait only — **do not port spurt.** SNAPHU and tophu implement the trait;
behavior is unchanged. **Done bar:** the unwrap dispatch is behind a trait both existing backends
implement, output bit-identical to today; gates green.

---

## Final (on sign-off, after Phase 5 lands)

Tag **v1.4.0**. Update ROADMAP exit notes: NRT validated against a full rerun; published speedup
vs baseline; phase-bias reduces non-closure; unwrap interface ready for a 3D backend. Note any
remaining deferrals (GPU CRLB, NISAR multi-date real displacement, spurt port) honestly.

**Overall Definition of Done:**
- [ ] Phase 1: tropo warp end-to-end on a UTM frame; `v1.3.0` tagged (on sign-off).
- [ ] Phase 2: NRT incremental update == full rerun within tolerance; contract green. ✅ (core)
- [ ] Phase 2b: end-to-end `update_displacement` + CLI streaming entry; matches a full
      `run_displacement` of the extended stack; public API/CLI exposed. **Required before v1.4.0 tag.**
- [ ] Phase 3: documented, reproduced speedup vs the committed baseline; no accuracy regression.
- [ ] Phase 4: phase-bias correction reduces non-closure (measured); analytic contract green.
- [ ] Phase 5: unwrap dispatch behind a trait; SNAPHU + tophu implement it; output unchanged.
- [ ] Sign guard + all oracle contracts green throughout; gates green (default/gpu/no-gpu) at
      every landing; `v1.4.0` tagged (on sign-off). Each unit merged `--no-ff` only after my
      sign-off; nothing pushed or tagged unilaterally.

---

## Launching with elevated permissions

Make sure the previous CLI session is closed first (a dirty tree or stale lock causes the
empty-diff/uncommitted-change failures we've already hit once). Then two steps — **step 2 is a
slash command typed inside Claude Code, not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
git status            # confirm clean before dispatching
source validation/creds.sh          # real OPERA-L4 / NISAR best-effort fetches
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read REMAINING_WORK_PROMPT.md and work the phases in order per its Definition of Done, stopping for sign-off before each merge/tag
```

`--dangerously-skip-permissions` runs cargo/git/pip/curl unattended. `/loop` with no interval =
dynamic self-pacing across phases. It stops and reports at each unit's completion (for merge/tag
sign-off) and on any genuine blocker (ambiguous compressed-SLC carry contract, a benchmark that
doesn't win, an unreachable real source) — rather than guessing or faking a result.
