# dolphinRust → ready-to-ship v1 (dynamic workflow loop)

One prompt, one bar: **make THIS repo complete, documented, and shippable as a library that
GroundPulse (`../eo`) can depend on.** Scope is dolphinRust only — do **not** edit `../eo`;
wiring it into eo is downstream, and the job here is to make that wiring obvious and safe
via a stable API + docs. "Compiles" and "matches on synthetic data" already passed and left
gaps; this loop is done only when the Definition of Done below is fully checked.

**The only acceptable end state is ready-to-ship (the Definition of Done).** Do not stop at
an intermediary milestone, do not declare partial success, and do not emit throwaway
intermediary prompts/reports as deliverables. Keep iterating until every box is checked.

---

## Prompt

Convert dolphinRust into a ready-to-ship v1: a complete, documented library eo can adopt.
Work the workstreams below as a dynamic, self-paced loop — one coherent change at a time,
contract test first, each change gated by `cargo fmt` + `cargo clippy --all-targets --
-D warnings` + `cargo test` + `cargo doc --no-deps` (no warnings), and for numerical changes
re-running `validation/run.sh`. Never weaken a tolerance or edit product code to force a
pass; a divergence is a finding to report.

Read first: `CLAUDE.md`, `PLAYBOOK.md`, `VALIDATION.md`. For the consumer's needs, read
`../eo`'s `CLAUDE.md` and `gp-displacement` / `gp-storage` (read-only — to shape the API and
output contract), but make **no changes** to eo.

**Honesty rule (non-negotiable):** README / VALIDATION / STATUS state exactly what ran — real
vs synthetic data, which tiers, engine versions, measured tolerances. Never claim a
validation or capability that did not execute. (A prior run wrote a false "validated against
dolphin" README; do not repeat it.)

### Workstream A — correctness eo depends on
1. **Velocity units.** Velocity currently carries the 12-day cadence (clean slope = 12.00)
   instead of a physical rate. Convert to mm/yr via the real temporal baselines; add a
   contract test (noise-free recovered rate == injected mm/yr); re-run all validation tiers —
   velocity must match the oracle on **absolute scale**, not just correlation. eo stores
   `velocity_mm_yr` for risk scoring, so this is load-bearing.
2. **L1 inversion.** dolphin defaults to L1/ADMM; we have only L2. Add it (Phase 6b),
   make the method config-driven, validate L1-vs-L1 so a default-config run matches dolphin.
3. **Multi-burst stitching.** eo processes frame mosaics — implement burst stitching so a
   multi-burst frame runs end to end.

### Workstream B — real-data validation
4. Obtain a small real OPERA S1 CSLC stack (a few cropped bursts). If Earthdata/ASF creds
   are required and absent ⛔ pause and ask for creds or a local path. Run both engines; add
   a **real-data tier** to `validation/` + `VALIDATION.md` (displacement + velocity in mm/yr).
   Divergence beyond the sanctioned eigensolver noise ⛔ stop and report with a hypothesis.

### Workstream C — shippable library surface
5. **Stable public API.** Present one clear entry point (`dolphin_workflows::run_displacement`
   or similar) returning a typed result — displacement cube, velocity, temporal coherence,
   acquisition dates, CRS + geotransform — so eo can call it directly and bridge via
   `spawn_blocking`. The CLI must be a thin wrapper over that same API. Keep it synchronous
   and runtime-agnostic (no tokio in the library path).
6. **Output contract.** Write the products eo consumes — COG/GeoTIFF for displacement,
   coherence, and velocity with correct CRS/geotransform — and also return them in memory.
   Document the exact schema (bands, dtype, units, nodata).
7. **API docs.** Add `#![warn(missing_docs)]` to every crate; document all public items;
   `cargo doc --no-deps` clean.

### Workstream D — documentation to learn how to use it
8. **README**: accurate status (no false claims), one-screen overview, system requirements
   (GDAL, HDF5, SNAPHU), and a quickstart for both the CLI and the library (a short Rust
   snippet calling `run_displacement`).
9. **`docs/usage.md`**: the real how-to — install + system deps; input requirements (CSLC
   layout/paths); config (dolphin-compatible YAML, key parameters with defaults); running via
   CLI and via Rust (including the exact `spawn_blocking` pattern eo will use, as a code
   sample); the output schema; known limitations/deferrals; how to run validation.
10. **Runnable example** under `examples/` that produces a result from the synthetic stack
    generator, so a new user gets output in one command.

### Workstream E — release readiness
11. Every crate's `Cargo.toml`: `description`, `keywords`, `categories`, `license`,
    `repository`, `readme`. Add `CHANGELOG.md` (v1.0.0). All gates green: fmt, clippy
    -D warnings, test, `cargo doc --no-deps`. Run `cargo publish --dry-run` per publishable
    crate (or document why a crate is private) to confirm packaging.

Commit on a branch. **Do not push or merge to any remote without my sign-off.** Otherwise do
not debate directions: state the load-bearing assumption in one line and proceed.

### Definition of Done — ready to ship (ALL must hold)
- [ ] Velocity (mm/yr, absolute scale) **and** displacement match Python dolphin v0.35.0 on a
      **real** OPERA CSLC stack within tolerance; L1 default matches; a multi-burst frame runs.
- [ ] `validation/run.sh` + `VALIDATION.md` cover synthetic tiers **and** a real-data tier,
      all passing (velocity scale included), deviations recorded.
- [ ] One documented, synchronous public API entry returns typed displacement/velocity/
      coherence + CRS/geotransform; CLI is a thin wrapper; output COG/GeoTIFF schema documented.
- [ ] `#![warn(missing_docs)]` on all crates; `cargo doc --no-deps` clean.
- [ ] README quickstart (CLI + library) + system requirements; `docs/usage.md` integration
      guide incl. the eo `spawn_blocking` call pattern; a runnable `examples/` program.
- [ ] Release metadata complete on every crate; `CHANGELOG.md`; `cargo publish --dry-run` clean.
- [ ] Gates green: fmt, clippy -D warnings, test, doc.
- [ ] README / VALIDATION / STATUS reflect exactly what ran; deferrals listed; no fabricated
      claims. Committed on a branch; nothing pushed without sign-off.

---

## Launching with elevated permissions

Two steps. **Step 2 is a slash command typed inside Claude Code — not a shell command.**

1. In your terminal (this one line only):

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read V1_PROMPT.md and convert dolphinRust to a ready-to-ship v1 per its Definition of Done
```

`--dangerously-skip-permissions` lets it run cargo/git/conda/pip unattended. `/loop` with no
interval = dynamic self-pacing. It pauses and asks if Earthdata creds are needed or if
real-data output diverges beyond the sanctioned eigensolver noise.
