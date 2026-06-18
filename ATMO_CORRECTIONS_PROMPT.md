# dolphinRust — atmospheric corrections: ionospheric + tropospheric (dynamic loop, v1.3.0 part 2)

Second half of ROADMAP v1.3.0 — what makes L-band actually *usable*. NISAR/L-band ingest
landed (part 1): the pipeline now produces a geometrically-correct but **atmospherically-
uncorrected** L-band displacement product. The ionosphere is ~16× the C-band effect at L-band,
so ionospheric correction is **mandatory** for usable NISAR displacement, not optional;
tropospheric correction is the user-expected companion. This loop adds both as a correction
stage that subtracts per-date delay from the displacement time series.

External-data reality (the reason for the best-effort structure below): IONEx TEC maps, the
OPERA L4 tropospheric product, and RAiDER (a Python subprocess) all live behind network/tooling
that may not be reachable in this run. The **correction math + apply stage** must be proven on
analytic/synthesized fixtures regardless; **real-source fetch is best-effort and deferred-with-
receipts if unreachable** — exactly like the NISAR real-data gate. No stubs, no faked sources.

---

## Prompt

Add ionospheric and tropospheric atmospheric corrections to the displacement pipeline,
subtracting per-date delay from the time series. Dynamic, self-paced loop, contract test first;
gate every step on `cargo fmt` + `cargo clippy --all-targets -- -D warnings` + `cargo test` +
`cargo doc --no-deps`. Commit on branch `v1.3-atmo`; **push nothing without my sign-off.**
Honesty rule: a correction is only "validated against real data" if a real IONEx / OPERA-L4 /
RAiDER output was actually obtained — otherwise validate the math on a fixture and mark the
real-source gate deferred with where you looked. Never stub a source or `fetch('/TODO')`.

Read first: `crates/dolphin-workflows/src/displacement.rs` (the stage order: read → phase-link →
network → unwrap → invert → velocity → COGs — corrections subtract from the inverted time
series, before velocity), `crates/dolphin-core/src/config.rs` (config surface + where wavelength
lives), `crates/dolphin-io/src/nisar.rs` (the L-band reader from part 1; λ ≈ 0.24 m, f ≈ 1.257
GHz), `CLAUDE.md`, `ROADMAP.md` (v1.3.0), `VALIDATION.md` (the NISAR record + deferral style to
mirror).

Load-bearing decisions (state once, don't re-litigate):
- **New crate `dolphin-corrections`** with its own `CLAUDE.md` carrying the atmospheric-delay
  math (per the project's dispersed-domain convention) — IONEx parsing + TEC→range-delay,
  netCDF L4 ingest, RAiDER dispatch. If a single existing crate is clearly the better home,
  state why in one line and use it.
- **Corrections subtract from the inverted displacement time series**, per date, converted to
  the raster's units (radians or meters) consistently with how displacement is expressed.
  Both default **off** (match dolphin — corrections are opt-in product layers); enabling them
  applies the subtraction and writes the per-date correction rasters as COGs alongside the
  displacement.
- **Ionosphere is frequency-scaled.** TEC→delay ∝ 1/f²; use the NISAR L-band frequency, not a
  C-band constant. This is the term that dwarfs everything at L-band.
- **CPU path; no GPU.** No new solver — this is delay modelling + raster subtraction.

Loop:
1. **Correction framework + apply stage.** The typed correction layer(s) on `DisplacementOutput`
   + the subtract-from-time-series stage in `run_displacement`, gated by config flags matching
   dolphin's names (check the real `correction`/`troposphere_files`/`ionosphere_files` option
   names so a dolphin YAML round-trips; where v0.35.0 lacks a field, add the minimal one and
   document the divergence). Contract: with corrections off, output is bit-identical to today;
   with a synthesized known correction raster supplied, the subtraction is exact.
2. **Ionospheric (TEC/IONEx).** Parse IONEx TEC maps; convert slant TEC → L-band range delay
   (1/f² scaling) → phase; interpolate to the frame grid + acquisition time. Contract test
   against the **closed-form** TEC→delay relation on a known TEC value (analytic — always
   provable). Then best-effort: fetch a real IONEx file (IGS/CDDIS; `source validation/creds.sh`
   — Earthdata/CDDIS bearer may work) and apply it to a real frame; if unreachable, record the
   gate deferred with where you looked.
3. **Tropospheric — OPERA L4 ingest.** Ingest the free public OPERA L4 tropospheric netCDF
   (aligned to DISP-S1 frames) via GDAL/netCDF; resample to the frame grid; subtract. Contract
   on a synthesized L4-format netCDF fixture with a known field. Best-effort: fetch a real L4
   product (ASF/PODAAC) and validate against it; else defer-with-receipts.
4. **Tropospheric — RAiDER fallback** (scenes without an L4 product). Subprocess dispatch +
   GDAL ingest of RAiDER's output. ⛔ Check `RAiDER` is installed first (e.g. `python -c "import
   RAiDER"` / `command -v raider.py`); if absent, **do not stub it** — gate the path behind its
   availability, document the dependency like SNAPHU, and mark this gate deferred. The L4 path
   is the primary, RAiDER is the fallback.
5. **Validate against an OPERA correction layer (the roadmap exit).** If any real correction
   layer (L4 troposphere or an OPERA-provided ionosphere layer) is obtainable, diff dolphinRust's
   correction against it and record the agreement in `VALIDATION.md`. If none is reachable this
   run, record deferred-with-receipts — don't fake the comparison.

Update `STATUS.md`/`ROADMAP.md` (mark v1.3.0 complete only if both corrections are at least
fixture-validated and wired; note any real-source deferrals); `docs/usage.md` + README: the
correction config, units, the L-band ionosphere note, and the RAiDER optional dependency;
`CHANGELOG`. Don't debate directions — state the load-bearing assumption in one line and proceed.

**Definition of Done:**
- [ ] `dolphin-corrections` (or justified existing home) with a per-crate `CLAUDE.md`; the
      apply stage subtracts per-date correction from the time series, off by default; with
      corrections off, `run_displacement` output is unchanged (contract proves it).
- [ ] **Ionospheric** TEC/IONEx → L-band delay (1/f²) implemented; contract green vs the
      closed-form relation; real IONEx applied to a real frame **or** gate deferred-with-receipts.
- [ ] **Tropospheric** OPERA-L4 netCDF ingest + subtract; contract green vs a synthesized
      L4-format fixture; real L4 validated **or** deferred-with-receipts. RAiDER fallback wired
      behind an availability check (documented dependency, not stubbed) or deferred.
- [ ] Correction layers in the typed API + written as COGs; config flags match dolphin; a real
      dolphin YAML round-trips; the per-pixel subtraction is exact vs a known correction raster.
- [ ] Validated against a real OPERA correction layer **or** that gate recorded deferred-with-
      receipts (the roadmap exit, honestly handled).
- [ ] Gates green (default, `--features gpu`, `--features no-gpu`): fmt, clippy -D warnings,
      test, doc. Docs/CHANGELOG/ROADMAP/STATUS updated. Committed on `v1.3-atmo`; unpushed.

---

## Launching with elevated permissions

Make sure the previous CLI session is closed first (stale shells can hold the cargo lock). Then
two steps — **step 2 is a slash command typed inside Claude Code, not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
source validation/creds.sh          # IONEx (CDDIS) + OPERA-L4 (ASF/PODAAC) best-effort fetch
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read ATMO_CORRECTIONS_PROMPT.md and add ionospheric + tropospheric corrections per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git/pip/curl unattended. `/loop` with no interval =
dynamic self-pacing. It validates the correction *math* on fixtures regardless of network, and
records each real-source gate (IONEx, OPERA L4, RAiDER) as deferred-with-receipts rather than
faking it if the source/tool isn't reachable. RAiDER, if absent, is gated and documented like
SNAPHU — never stubbed.
