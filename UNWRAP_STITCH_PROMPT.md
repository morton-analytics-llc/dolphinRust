# dolphinRust — tophu multi-scale unwrapping + per-ministack coherence stitching (dynamic loop)

The other half of ROADMAP v1.2.0 (the quality half — CRLB + closure — shipped on `main`).
Two items:

1. **tophu-style multi-scale tiled unwrapping** — OPERA's *production* unwrapper. Raw SNAPHU
   degrades on large, low-coherence (vegetated) scenes; tophu adds a coarse-resolution
   initialization feeding tiled SNAPHU, then merges. SNAPHU stays the simple default path.
2. **Per-ministack temporal-coherence stitching** (dolphin v0.41) — today
   `temporal_coherence` is ministack-averaged, which is exact for single-ministack stacks but
   only approximate across many ministacks. Proper per-ministack stitching closes that gap —
   **and closes the open caveat from the CRLB/closure work** (those layers are computed per
   ministack and concatenated; this is what makes the many-ministack frame correct).

---

## Prompt

Add tophu-style multi-scale unwrapping to `dolphin-unwrap` and per-ministack temporal-coherence
stitching to the workflow, validated against the dolphin/tophu reference. Dynamic, self-paced
loop, contract test first; gate every step on `cargo fmt` + `cargo clippy --all-targets --
-D warnings` + `cargo test` + `cargo doc --no-deps`. Commit on branch `v1.2-unwrap`; **push
nothing without my sign-off.** Honesty rule: report real behaviour vs the reference — if tophu
doesn't beat raw SNAPHU on the test scene, say so with a hypothesis; don't tune the scene to
manufacture a win, and don't weaken a tolerance.

Read first: `crates/dolphin-unwrap/CLAUDE.md` and `src/snaphu.rs` (the existing SNAPHU
subprocess wrapper — tophu drives SNAPHU per tile, it doesn't replace it),
`crates/dolphin-workflows/CLAUDE.md` + `src/sequential.rs` (where ministack temporal coherence
is produced/averaged today), `CLAUDE.md`, `ROADMAP.md` (v1.2.0), `VALIDATION.md`.

Load-bearing decisions (don't re-litigate):
- **tophu is a heuristic orchestration over SNAPHU, not new unwrap math.** Implement the
  coarse→fine strategy in Rust calling the existing SNAPHU wrapper per tile. The reference is
  tophu's *algorithm* and its *result quality*, **not bit-parity** — tophu's own tiling/merge
  is non-unique, so the contract is comparative (fewer residues / unwrap discontinuities than
  raw SNAPHU on a low-coherence scene), plus internal-consistency checks, not exact match.
- **SNAPHU stays the default.** tophu is opt-in via config. The default path is unchanged.
- **CPU path.** No GPU here.

Loop:
1. **tophu multi-scale unwrap.** In `dolphin-unwrap`: (a) coarse pass — downsample the wrapped
   interferogram + correlation by `downsample_factor`, unwrap the coarse grid with the existing
   SNAPHU wrapper; (b) upsample the coarse unwrapped phase to full res as a per-tile
   initialization/offset reference; (c) tile the full-res grid into overlapping tiles
   (`ntiles`), unwrap each tile via the SNAPHU wrapper (parallelize tiles with `rayon`);
   (d) merge tiles, reconciling integer-cycle (2π) offsets between adjacent tiles against the
   coarse solution. Contract tests as you go: coarse pass round-trips a known ramp; the
   tile-merge resolves a planted inter-tile 2π jump; the full path on an analytic ramp recovers
   it to the SNAPHU envelope.
2. **Prove the win.** On a **large, low-coherence** scene where raw SNAPHU produces unwrapping
   errors (reuse a fetched stack under `validation/`/`bench/` if one qualifies; else synthesize
   a low-coherence scene with a known phase field — don't cherry-pick parameters to fake a
   margin), measure tophu vs raw SNAPHU: residue count / number of unwrap discontinuities /
   RMS-vs-truth. Record honest numbers in `VALIDATION.md` (or `bench/UNWRAP.md`). If tophu does
   **not** beat SNAPHU, report it with a hypothesis rather than tuning the scene.
3. **Per-ministack coherence stitching.** Replace the ministack-averaged `temporal_coherence`
   with dolphin v0.41 per-ministack stitching across the full frame. Contract test vs the
   oracle on a **multi-ministack** stack (`ministack_size` < n_slc so ≥2 ministacks form).
   **Then re-verify CRLB + closure on that same multi-ministack stack** — this is the layer
   that closes their concatenation caveat; confirm they hold (or report the delta).
4. **Surface + wire.** Config: match dolphin/tophu names so a real dolphin YAML round-trips —
   check the actual `unwrap_options` field names (`unwrap_method`/`UnwrapMethod`, `ntiles`,
   `downsample_factor`, `n_parallel_tiles` or whatever v0.4x uses) and match them; selecting
   the tophu method routes through it. `run_displacement` uses tophu when selected; default
   (SNAPHU) unchanged. The stitched coherence flows into the existing
   `DisplacementOutput.temporal_coherence` + its COG (no schema change).
5. **Docs + hygiene.** `docs/usage.md` + README: the tophu method, when to use it
   (large/low-coherence scenes), its config; note the stitching fix in CHANGELOG. `CHANGELOG`
   + `ROADMAP` (mark the v1.2.0 unwrap+stitching half landed → v1.2.0 complete). Update
   `STATUS.md` as items land. Don't debate directions — state the load-bearing assumption in
   one line and proceed.

**Definition of Done:**
- [ ] tophu multi-scale unwrap implemented in `dolphin-unwrap` over the existing SNAPHU wrapper
      (coarse init → overlapping tiled SNAPHU → 2π-reconciled merge); contract tests green
      (ramp recovery, planted inter-tile 2π jump resolved).
- [ ] Measured tophu-vs-raw-SNAPHU comparison on a large low-coherence scene recorded with
      honest numbers (residues / discontinuities / RMS); if tophu doesn't win, that's reported,
      not hidden.
- [ ] Per-ministack temporal-coherence stitching replaces the average; contract test vs oracle
      on a ≥2-ministack stack green; **CRLB + closure re-verified on that multi-ministack stack**
      (caveat closed or delta reported).
- [ ] Config flags match dolphin/tophu names; a real dolphin YAML still round-trips; SNAPHU
      remains the default path and the default build is behaviourally unchanged.
- [ ] Gates green (default, `--features gpu`, `--features no-gpu`): fmt, clippy -D warnings,
      test, doc.
- [ ] README/usage/CHANGELOG/ROADMAP/STATUS updated; committed on `v1.2-unwrap`; unpushed.

---

## Launching with elevated permissions

Two steps. **Step 2 is a slash command typed inside Claude Code — not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
source validation/creds.sh          # in case the low-coherence test needs a real stack
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read UNWRAP_STITCH_PROMPT.md and add tophu multi-scale unwrapping + per-ministack coherence stitching per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git/snaphu/pip unattended. `/loop` with no interval
= dynamic self-pacing. It stops and reports if tophu can't beat raw SNAPHU on the test scene, or
if the stitching diverges from the oracle (reports rather than fudging). SNAPHU (`snaphu` on
`PATH`) is required for the unwrap tests; they skip cleanly if it's absent, but the tophu win
can't be measured without it.
