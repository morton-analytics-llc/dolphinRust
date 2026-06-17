# dolphinRust — phase-linking quality layers: CRLB + closure phase (dynamic loop)

The next enhancement (ROADMAP v1.2.0, quality half): add the two per-pixel quality rasters
dolphin produces but dolphinRust doesn't yet — **CRLB uncertainty** and **sequential
closure-phase**. CRLB is the product driver: GroundPulse scores asset risk from velocity +
a `confidence_score`, and per-pixel CRLB σ is the physical uncertainty that's currently
missing from that. Closure phase is the non-closure diagnostic and the prerequisite signal
for the later phase-bias work. Both were added upstream in dolphin v0.40–v0.41, so this also
moves the oracle forward. (tophu unwrapping + per-ministack coherence stitching are the
*other* half of v1.2.0 — a separate later loop, not this one.)

---

## Prompt

Add CRLB and sequential closure-phase quality layers to `dolphin-phaselink`, validated
against a newer dolphin oracle. Dynamic, self-paced loop, contract test first; gate every
step on `cargo fmt` + `cargo clippy --all-targets -- -D warnings` + `cargo test` +
`cargo doc --no-deps`. Commit on branch `v1.2-quality`; **push nothing without my sign-off.**
Honesty rule: report real accuracy vs the oracle; if a layer can't match within tolerance,
say so with a hypothesis — don't weaken a tolerance.

Read first: `crates/dolphin-phaselink/CLAUDE.md` (CRLB = Fisher-information σ; closure =
triplet non-closure — the domain math is there), `CLAUDE.md`, `ROADMAP.md` (v1.2.0),
`VALIDATION.md`.

1. **Move the oracle forward — carefully.** The pinned oracle is dolphin **v0.35.0**, which
   has neither layer. Stand up a second oracle venv at **v0.42.0** (CRLB landed v0.40,
   closure v0.41, plus a v0.42 singular-matrix CRLB fix) and use it **only** to validate the
   two new layers. Keep existing kernels validated at v0.35.0. ⛔ Do **not** silently re-tune
   existing kernels to chase v0.36–v0.42 default changes (25 km filter boundary, auto
   reference-point, default sigma, larger half-window, nearest-3). If generating the v0.42.0
   oracle surfaces divergence in any *existing* kernel, stop and report it — a full oracle
   bump + reconciliation is a separate decision for me.

2. **CRLB uncertainty raster.** Per-pixel phase-estimate σ from the Fisher information of the
   coherence model (dolphin `crlb.py`). CPU (`faer`, f64) path — GPU CRLB is a later
   follow-up; note it, don't build it here. Handle the singular/ill-conditioned Γ case (the
   v0.42 fix). Contract test vs the v0.42.0 oracle within a stated tolerance.

3. **Sequential closure-phase raster.** Per ministack, the wrapped sum of phase around
   nearest-neighbour triplets (dolphin v0.41). Contract test vs the v0.42.0 oracle.

4. **Surface them.** Add to the typed `DisplacementOutput` (per-pixel CRLB σ and closure
   raster) and write both as COG, sharing the grid CRS/geotransform. Gate with config flags
   matching dolphin's names (`write_crlb`, `write_closure_phase` or the actual v0.4x names —
   check and match so a real dolphin YAML round-trips).

5. **Wire end-to-end.** `run_displacement` produces the layers when enabled; one config, same
   result. Keep them off by default if dolphin defaults them off (match dolphin).

6. **Docs + hygiene.** README + `docs/usage.md`: the new layers, their units, and the CRLB →
   GroundPulse `confidence_score` connection (documentation only — no `../eo` edits).
   `CHANGELOG` + `ROADMAP` (mark the v1.2.0 quality half landed; note tophu/stitching remain).

Update `STATUS.md` as items land. Otherwise don't debate directions — state the load-bearing
assumption in one line and proceed.

**Definition of Done:**
- [ ] v0.42.0 oracle stood up and used for the new layers; pin recorded in `VALIDATION.md`;
      existing kernels still validated at v0.35.0 (any divergence from the bump reported, not
      silently absorbed).
- [ ] **CRLB** per-pixel σ raster matches the v0.42.0 oracle within tolerance (incl. the
      singular-Γ case); contract test green.
- [ ] **Closure-phase** raster matches the v0.42.0 oracle within tolerance; contract test green.
- [ ] Both in the typed API + written as COG; config flags match dolphin; `run_displacement`
      produces them end-to-end; a real dolphin YAML still round-trips.
- [ ] Gates green (default, `--features gpu`, `--features no-gpu`): fmt, clippy -D warnings,
      test, doc.
- [ ] README/usage/CHANGELOG/ROADMAP/STATUS updated (incl. the CRLB→confidence_score note);
      committed on `v1.2-quality`; unpushed.

---

## Launching with elevated permissions

Two steps. **Step 2 is a slash command typed inside Claude Code — not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
source validation/creds.sh          # in case oracle validation needs a real stack
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read QUALITY_LAYERS_PROMPT.md and add CRLB + closure-phase quality layers per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git/pip unattended. `/loop` with no interval =
dynamic self-pacing. It stops if the v0.42.0 oracle bump surfaces divergence in existing
kernels, or if a new layer can't match the oracle within tolerance (it reports rather than
fudging).
