# dolphinRust — end-to-end validation prompt

Paste the Prompt section below into a Claude Code CLI session started with elevated
permissions (see "Launching" at the bottom). It validates that dolphinRust reproduces the
Python `dolphin` it is replacing, end to end, within the §Correctness tolerances — closing
the two gaps left after the build: no pinned reference oracle, and no SNAPHU.

---

## Prompt

Validate dolphinRust end to end against the Python `dolphin` it is the optimized rebuild of.
Goal: prove that, on identical inputs and config, dolphinRust's displacement output agrees
with dolphin's within the tolerances in PLAYBOOK.md §"Correctness & validation strategy".
Code-existence and unit tests already pass — this is about real-pipeline equivalence.

Read first: `CLAUDE.md`, `PLAYBOOK.md` (§Correctness, §GroundPulse integration, the phase
list), and `STATUS.md`. Do not modify product code to make validation pass — if outputs
diverge, that is a finding to report, not a thing to paper over. Touch only validation
scaffolding under `validation/` plus `VALIDATION.md`. Idiomatic-Rust and no-stub rules in
CLAUDE.md still apply; the PostToolUse hook is active.

Work this sequence; stop and ASK me at any blocker marked ⛔ rather than guessing:

1. **Environment preflight.** Verify and record versions:
   - `gdal-config --version`, `h5cc -showconfig | head` (known present).
   - `command -v snaphu` — the unwrapper binary our `dolphin-unwrap` shells out to. If
     missing, try `conda install -c conda-forge snaphu` (or document the install you used).
     If it cannot be installed here ⛔ pause and tell me; otherwise continue and run a
     SNAPHU-less tier (see step 5).
   - A Python env with dolphin: create an isolated env (conda preferred for the GDAL/HDF5
     stack) and install dolphin. **Pin the exact version** you install and record it in
     `VALIDATION.md` — this resolves Open question #1; if you have no basis to choose, use
     the latest stable release and say so.

2. **Test stack.** Obtain a *small* CSLC stack (a few bursts, cropped) that both
   implementations can ingest from one config:
   - Preferred: a real OPERA S1 CSLC mini-stack. If it needs Earthdata/ASF credentials and
     none are available here ⛔ pause and ask me for creds or a local path.
   - Fallback: synthesize a CSLC stack with a known deformation signal (a small generator
     under `validation/`, fixed seed, emitting the HDF5/GeoTIFF layout both tools read).
     Label results "synthetic-input equivalence" and note that real-data validation is
     still pending. State which path you took.

3. **One config, both engines.** Write a single dolphin displacement-workflow YAML and
   confirm dolphinRust accepts it unchanged (that compatibility is a requirement — if it
   does not parse, that is a finding). Use a fixed `work_directory` per engine.

4. **Run the oracle.** `dolphin run <config>` (Python) → reference outputs.

5. **Run dolphinRust.** `cargo run --release -p dolphin-cli -- run --config <config>`.
   - Tier A (no SNAPHU): set `run_unwrap: false` (or equivalent) in both; validate through
     phase linking / wrapped phase / quality layers / ministack / network.
   - Tier B (SNAPHU present): full pipeline including unwrap + timeseries inversion.
   Run whichever tiers the environment supports; record which ran.

6. **Compare, per §Correctness.** Write a comparison script under `validation/` (Python,
   load both with numpy/rasterio) reporting, per stage and end to end:
   - phase quantities compared modulo 2π and up to a global phase reference;
   - coherence / temp_coh to `atol≈1e-4`; eigenvector agreement as `|⟨v_rust, v_oracle⟩|`;
   - displacement time series and velocity to a stated physical tolerance.
   Emit a pass/fail table.

7. **Report.** Write `VALIDATION.md`: pinned dolphin version, GDAL/HDF5/SNAPHU versions,
   data path (real vs synthetic), which tiers ran, the per-stage pass/fail table with the
   numeric max-deviations, and any divergences with a one-line hypothesis each. Tick the
   validation state in `STATUS.md`. Commit (Co-Authored-By trailer per CLAUDE.md). Do not
   push unless I ask.

**Done:** `VALIDATION.md` exists with a per-stage pass/fail table against a pinned dolphin
oracle on a documented stack; every stage either passes its tolerance or has a recorded,
explained divergence; `cargo test`/`clippy`/`fmt` still green.

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
/loop read VALIDATE_PROMPT.md and execute its end-to-end validation sequence
```

`--dangerously-skip-permissions` lets it run conda/pip/cargo/git unattended. `/loop` with
no interval = dynamic self-pacing. It will pause and ask if SNAPHU can't be installed,
Earthdata creds are needed, or the dolphin version needs your sign-off.
