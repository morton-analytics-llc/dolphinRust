# dolphinRust — NISAR / L-band RSLC ingest (dynamic loop, v1.3.0 part 1)

First half of ROADMAP v1.3.0. NISAR is the growth surface: L-band DISP-NI penetrates canopy
where C-band fails — exactly eo's forested pipeline/dam assets. This loop gets a **NISAR RSLC
stack reading end-to-end into displacement**. Atmospheric corrections (ionospheric/tropospheric)
are a **separate later loop** — not this one. (Ionosphere is ~16× the C-band effect and is
mandatory for *usable* L-band, so this loop produces a geometrically-correct displacement
product, not yet an atmospherically-corrected one — say so in the docs.)

The hard de-risk here is the **HDF5 reader**: NISAR RSLC is complex-int16 compound types in a
different product-group layout than OPERA S1 CSLC, and GDAL's HDF5 driver returns an identity
geotransform for it (already noted in `crates/dolphin-io/CLAUDE.md`). That ergonomics risk
blocks everything L-band, so it goes first.

---

## Prompt

Add a NISAR RSLC reader to `dolphin-io` and the L-band parameters needed to run a NISAR stack
through the existing pipeline. Dynamic, self-paced loop, contract test first; gate every step on
`cargo fmt` + `cargo clippy --all-targets -- -D warnings` + `cargo test` + `cargo doc --no-deps`.
Commit on branch `v1.3-nisar`; **push nothing without my sign-off.** Honesty rule: real-data
validation only counts on a real/sample NISAR granule — if none is reachable, say so and mark
that gate deferred; don't synthesize a fixture and call it real-data validation.

Read first: `crates/dolphin-io/CLAUDE.md` + `src/cslc.rs` (the OPERA S1 CSLC reader to mirror)
+ `src/geo.rs` (geotransform handling), `crates/dolphin-core/CLAUDE.md` + `src/config.rs`
(wavelength + input options), `crates/dolphin-phaselink/CLAUDE.md` (where wavelength/spectral
params enter covariance), `CLAUDE.md`, `ROADMAP.md` (v1.3.0), `VALIDATION.md`.

Load-bearing decisions (don't re-litigate):
- **The reader is the de-risk — prove it on a synthesized NISAR-format fixture first.** Build a
  small NISAR-layout HDF5 fixture generator (complex-int16 compound RSLC dataset in the NISAR
  product group structure, with the coordinate/projection metadata GDAL ignores) with a **known
  answer**, and make the reader's contract test pass against it. This contract is provable
  regardless of real-data availability.
- **Custom geotransform reader.** GDAL's HDF5 driver returns identity for NISAR — read the grid
  origin/spacing/EPSG from the NISAR coordinate datasets directly (mirror what `geo.rs` does for
  OPERA where needed). The complex-int16 compound is decoded to `Cf32` on read (the pipeline is
  f32 complex throughout).
- **L-band is a parameter change, not new algorithm.** NISAR L-band λ ≈ 0.24 m vs S1 C-band
  0.055 m. Thread the wavelength + any L-band spectral parameter through config →
  covariance/phase-linking → the `−λ/4π` velocity scaling so a NISAR stack produces correct
  mm/yr. No new solver.
- **Atmospheric correction is out of scope here** — this loop ends at a geometrically-correct,
  *atmospherically-uncorrected* L-band displacement product. Note the limitation in docs.

Loop:
1. **NISAR fixture + reader.** Fixture generator (synthesized NISAR-layout HDF5, complex-int16
   compound, NISAR group paths, coordinate/projection metadata). Reader in `dolphin-io` that
   opens it, decodes complex-int16 → `Cf32`, and returns the grid + a correct geotransform/EPSG.
   ⛔ If `hdf5-metno` can't handle the complex-int16 compound type ergonomically, **stop and
   report** — that's the blocking risk and a real architecture question for me.
2. **Contract test** vs the fixture: pixel values, grid shape, and geotransform/EPSG match the
   known answer.
3. **Config + product detection.** Let a NISAR stack be configured (product type / subdataset
   path / date parsing for NISAR granule names). Match dolphin/DISP-NI field names where they
   exist so a DISP-NI-style config round-trips; where dolphin v0.35.0 has no equivalent, add the
   minimal field and document it as a forward divergence.
4. **L-band parameters end-to-end.** Wavelength + L-band spectral params flow through
   covariance/phase-linking and the velocity scaling. Contract/unit test: the `−λ/4π` scaling
   and any spectral term use the NISAR λ, not the S1 default.
5. **End-to-end on the fixture stack.** `run_displacement` consumes a multi-acquisition
   synthesized NISAR stack and produces a displacement product (typed output + COGs), grid/EPSG
   correct. This proves the wiring without needing real data.
6. **Real-data gate (best effort).** Attempt to obtain a real or NASA-sample NISAR RSLC granule
   (`source validation/creds.sh`; try the known NISAR/ASF sample-product endpoints). If reachable,
   run a real stack and record what comes out in `VALIDATION.md`. If **not** reachable, record
   the gate as **deferred — no real/sample NISAR granule available as of this run**, with where
   you looked. Do not fake it.

Update `STATUS.md`/`ROADMAP.md` as items land; `docs/usage.md` + README: NISAR input
requirements, the L-band wavelength, and the explicit "no atmospheric correction yet" limitation.
`CHANGELOG`. Don't debate directions — state the load-bearing assumption in one line and proceed.

**Definition of Done:**
- [ ] NISAR RSLC reader in `dolphin-io` decodes complex-int16 compound → `Cf32` and returns a
      correct custom geotransform/EPSG; contract test green vs a synthesized NISAR-layout fixture.
- [ ] A NISAR stack is configurable (product/subdataset/date-parse); config round-trips; field
      names match DISP-NI where they exist, divergences documented.
- [ ] L-band λ + spectral params flow end-to-end; velocity uses the NISAR λ (test proves it).
- [ ] `run_displacement` produces a displacement product from a multi-acquisition NISAR fixture
      stack (typed output + COGs, grid/EPSG correct).
- [ ] Real/sample NISAR granule validated end-to-end **or** that gate explicitly recorded as
      deferred (with where you looked) — not faked.
- [ ] Gates green (default, `--features gpu`, `--features no-gpu`): fmt, clippy -D warnings,
      test, doc. Docs/CHANGELOG/ROADMAP/STATUS updated. Committed on `v1.3-nisar`; unpushed.

---

## Launching with elevated permissions

Make sure the previous CLI session is closed first (stale shells can hold the cargo lock). Then
two steps — **step 2 is a slash command typed inside Claude Code, not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
source validation/creds.sh          # for the best-effort real/sample NISAR fetch
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read NISAR_INGEST_PROMPT.md and add the NISAR/L-band RSLC ingest path per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git/pip/curl unattended. `/loop` with no interval =
dynamic self-pacing. It stops and reports if `hdf5-metno` can't handle the NISAR complex-int16
compound type (the blocking risk), and records the real-data gate as deferred rather than faking
it if no NISAR granule is reachable.
