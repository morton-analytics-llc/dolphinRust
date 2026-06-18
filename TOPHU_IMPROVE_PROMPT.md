# dolphinRust — make tophu actually beat SNAPHU (dynamic loop)

The v1.2.0 unwrap work landed on branch `v1.2-unwrap` but with an honest miss: our tophu
multi-scale unwrapper is *correct* yet **modestly worse than raw SNAPHU on low-coherence
scenes** (`bench/UNWRAP.md`). Decision: **improve tophu so it genuinely beats raw SNAPHU on
the existing fixed scenes before this ships** — don't ship a method that loses to the default.
The diagnosis in `bench/UNWRAP.md` already names the two causes; this loop fixes them.

The stitching fix on the same branch is a clean win and stays — it ships with tophu once tophu
wins. Nothing on `v1.2-unwrap` is pushed yet.

---

## Prompt

Continue on branch `v1.2-unwrap`. Improve the tophu multi-scale unwrapper in `dolphin-unwrap`
so it measurably beats raw single-pass SNAPHU on the low-coherence benchmark scenes. Dynamic,
self-paced loop, contract test first; gate every step on `cargo fmt` + `cargo clippy
--all-targets -- -D warnings` + `cargo test` + `cargo doc --no-deps`. Commit on `v1.2-unwrap`;
**push nothing without my sign-off.**

⛔ **Integrity guard — this is the whole point of the exercise.** The benchmark scenes and
their parameters in `crates/dolphin-unwrap/tests/tophu_bench.rs` are **frozen**. Do **not**
edit the scene generators, coherence maps, noise model, seeds, sizes, or metric definitions to
make tophu look better. The win must come from the *algorithm*, measured on the **unchanged**
scenes. If after honest effort tophu still doesn't beat SNAPHU on those scenes, **stop and
report** with numbers + hypothesis — do not tune the scene, weaken a tolerance, or relax a
metric. A truthful "still loses" is an acceptable outcome of this loop; a manufactured win is
not.

Read first: `bench/UNWRAP.md` (the measured loss + the named causes + the "likely improvement
path" already sketched there), `crates/dolphin-unwrap/src/tophu.rs`,
`crates/dolphin-unwrap/src/snaphu.rs`, `crates/dolphin-unwrap/tests/tophu_bench.rs` and
`tophu_contract.rs`, `crates/dolphin-unwrap/CLAUDE.md`.

The two diagnosed causes to fix (from `bench/UNWRAP.md`):
1. **Coarse anchor poisoned by decorrelation.** The coarse pass multilooks complex phasors over
   `downsample_factor` blocks; in low-γ regions those phasors are near-random, so the coarse
   phase is unreliable and the per-tile 2π anchor lands on the wrong cycle. → **Coherence-weight
   the coarse multilook** (weight each phasor by coherence, or amplitude²·coherence), and where
   a coarse block falls below a coherence floor, treat it as untrusted (mask + fill from
   trusted neighbours rather than anchoring to garbage).
2. **Mean-offset tile merge is cruder than SNAPHU's global MCF.** Each tile is reconciled with a
   single constant 2π offset to the coarse solution. → **Replace it with a proper inter-tile
   integer-cycle reconciliation**: estimate the relative integer-cycle offset between adjacent
   tiles from their *overlap region* (robust/median over coherent overlap pixels), then solve
   the consistent set of per-tile offsets across the tile-adjacency graph (least-cost /
   spanning-tree over the overlap graph), instead of anchoring each tile independently to a
   noisy coarse field.

Loop:
1. **Coherence-weighted coarse pass.** Implement weighting + low-γ masking/fill in the coarse
   multilook. Add/extend a contract test: the coarse phase on a partly-decorrelated synthetic
   field tracks truth better than the unweighted version (a *unit* comparison, not the frozen
   bench scene).
2. **Overlap-based tile merge.** Replace the constant-to-coarse offset with overlap-region
   inter-tile offset estimation + a graph solve for globally consistent per-tile cycles. Keep
   the existing contract tests green (`merge_resolves_planted_2pi_jump`,
   `tophu_recovers_analytic_ramp_within_snaphu_envelope`, `tophu_coarse_pass_round_trips_ramp`);
   add one for a planted offset across a 2×2 tile grid (consistency around a loop).
3. **Re-measure on the frozen scenes.** Re-run `tophu_bench` on the **unmodified** scenes.
   Rewrite the results table + conclusion in `bench/UNWRAP.md` with the new honest numbers. The
   target: tophu ≤ raw SNAPHU on **all three** metrics (discont, rms, gross-cycle-err-frac) on
   **both** scenes, with at least one scene showing a clear improvement (not noise). State the
   margin plainly.
4. **If it wins:** update `CHANGELOG`/`ROADMAP`/`STATUS`/`docs/usage.md` to reflect tophu now
   beating SNAPHU on low-coherence scenes (replace the "prefer SNAPHU" guidance with the
   measured win + when to use each). If it does **not** win after honest effort, leave the
   honest-loss writeup in place and stop with a report — don't touch the scenes.

State the load-bearing assumption in one line and proceed; don't debate directions.

**Definition of Done (the win case):**
- [ ] Coherence-weighted coarse multilook + low-γ masking/fill implemented; unit contract shows
      the coarse phase tracks truth better than unweighted.
- [ ] Overlap-region inter-tile cycle reconciliation + global graph solve replaces the
      constant-2π-to-coarse merge; all prior tophu contract tests green + a 2×2-grid
      loop-consistency test green.
- [ ] On the **frozen, unedited** `tophu_bench` scenes, tophu ≤ raw SNAPHU on all three metrics
      on both scenes, with a clear (non-noise) improvement on at least one; `bench/UNWRAP.md`
      rewritten with the new numbers and the margin stated honestly.
- [ ] Gates green (default, `--features gpu`, `--features no-gpu`): fmt, clippy -D warnings,
      test, doc.
- [ ] Docs/CHANGELOG/ROADMAP/STATUS updated to the measured win; committed on `v1.2-unwrap`;
      unpushed.

**Acceptable alternative outcome:** if tophu still cannot beat SNAPHU on the frozen scenes,
stop and report the new numbers + hypothesis with the scenes untouched and tolerances intact.
Do not ship a faked win.

---

## Launching with elevated permissions

First make sure the previous CLI session is fully closed (it left shells running) so they don't
hold the cargo lock or a dirty tree. Then, two steps — **step 2 is a slash command typed inside
Claude Code, not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
git checkout v1.2-unwrap          # continue on the existing branch
source validation/creds.sh
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read TOPHU_IMPROVE_PROMPT.md and make tophu beat raw SNAPHU on the frozen low-coherence scenes per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git/snaphu unattended. `/loop` with no interval =
dynamic self-pacing. It stops and reports — scenes untouched — if tophu still can't beat SNAPHU
after honest effort, rather than tuning the benchmark to manufacture a win. SNAPHU (`snaphu` on
`PATH`, v2.0.7 via Homebrew) is required to measure the comparison.
