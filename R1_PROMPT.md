# dolphinRust — baseline + v1.1.0 (dynamic workflow loop)

Drives the pre-R1 speed baseline and the v1.1.0 release per ROADMAP.md. Dynamic, self-paced,
contract-first. One hard stop: **do not modify `../eo` without my sign-off.**

---

## Prompt

Execute the **baseline benchmark** and **v1.1.0** from ROADMAP.md. Work as a dynamic loop —
one coherent change at a time, each gated by `cargo fmt` + `cargo clippy --all-targets --
-D warnings` + `cargo test` + `cargo doc --no-deps`, and validated against the pinned dolphin
oracle (v0.35.0) where numerical. Commit on a branch `v1.1` (never `main`); **push nothing
without my sign-off**. Honesty rule: report real measured numbers — never fabricate a
benchmark or validation result.

Read first: `ROADMAP.md` (Baseline + v1.1.0 sections), `CLAUDE.md`, `VALIDATION.md`,
`validation/creds.sh`.

Do these in order:

1. **Baseline speed benchmark** (the pre-R1 item). Time Python dolphin v0.35.0 vs dolphinRust
   on the existing `validation/` stacks — per-frame wall-clock and phase-linking throughput,
   warm and cold (capture dolphin's JAX JIT warm-up separately). Commit a reproducible
   `bench/` (script + a results table in `bench/README.md`). State the speedup honestly,
   including where Rust loses if it does.

2. **Close the velocity-scale residual** (ROADMAP R1 / VALIDATION.md B4). `source
   validation/creds.sh`, authenticate with the bearer token —
   `asf_search.ASFSession().auth_with_token(os.environ["GP_EARTHDATA_TOKEN"])` (NOT `~/.netrc`,
   which is stale). Fetch one PS-rich, high-coherence subsidence scene (Mexico City ~
   lon −99.25..−98.95 lat 19.25..19.55, or Las Vegas Valley ~ lon −115.4..−115.0 lat
   36.0..36.35), ~10–15 dates. Run both engines; confirm dolphinRust velocity tracks the
   oracle at real magnitude. Add the result to `VALIDATION.md`. ⛔ If the granule *download*
   (not a URS ping) returns 401/403, stop and tell me — that's a data-permission issue.

3. **Auto reference-point selection** (center-of-mass, dolphin v0.36.0). Implement,
   contract-test against the oracle, wire into the workflow config.

4. **eo integration.** ⛔ STOP and ask me before editing `../eo`: confirm sign-off, the crate
   name (`gp-dolphin`?), and the SNAPHU install in eo's worker image. Then, on an `eo` branch:
   a crate calling `run_displacement` via `spawn_blocking`, wired as a `gp-tasks` task that
   lands a COG via `gp-storage` + summary rows in PostGIS, and confirm a real dolphin YAML
   parses unchanged. Run one frame end-to-end. Commit on the eo branch; do not push/merge.

Update `STATUS.md`/`ROADMAP.md` checkboxes as items land. Otherwise don't debate directions —
state the load-bearing assumption in one line and proceed.

**Definition of Done (v1.1.0):**
- [ ] `bench/` committed with a real, reproducible dolphinRust-vs-dolphin speed table.
- [ ] Velocity absolute scale confirmed on a real deforming scene; `VALIDATION.md` updated.
- [ ] Auto reference-point matches the oracle; contract test green.
- [ ] eo integration either runs one frame end-to-end (if signed off) or is paused at the
      documented gate awaiting your sign-off.
- [ ] Gates green (fmt, clippy -D warnings, test, doc); committed on branch `v1.1`; unpushed.

---

## Launching with elevated permissions

Two steps. **Step 2 is a slash command typed inside Claude Code — not a shell command.**

1. In your terminal — load the working Earthdata token, then launch:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
source validation/creds.sh          # exports the verified bearer token
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read R1_PROMPT.md and execute the baseline benchmark + v1.1.0 per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git/conda/pip/asf_search unattended. `/loop` with
no interval = dynamic self-pacing. It pauses before touching `../eo`, or if a real granule
download fails auth.
