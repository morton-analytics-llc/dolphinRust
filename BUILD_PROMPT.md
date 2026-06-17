# dolphinRust — autonomous build prompt (v1.0.0, dynamic workflow)

Paste the prompt below into a Claude Code CLI session started with elevated permissions
(see "Launching" at the bottom). It drives the complete build dynamically — one phase at a
time, contract-first — until v1.0.0.

---

## Prompt

Build dolphinRust to its first complete build (v1.0.0): end-to-end OPERA InSAR displacement
from a CSLC stack. This is an **optimized Rust rebuild** of the Python `dolphin` (the
algorithm reference — NOT a line-by-line port), and the drop-in replacement for the Python
dolphin that GroundPulse (`../eo`) is adopting.

Read these before acting, every session:
- `CLAUDE.md` (root) — workflow, idiomatic-Rust rules, contract-first rule, how to work with me.
- `PLAYBOOK.md` — the phased plan (Phases 0–10), build-priority DAG, correctness/validation
  strategy, GroundPulse integration, and the elevated/open questions.
- The target crate's own `CLAUDE.md` before writing in it (domain math + contract tolerances).
- `STATUS.md` — the phase checklist; it is your memory across sessions.

Work this loop, **dynamically self-paced, one phase at a time**:
1. Open `STATUS.md`; pick the next unchecked phase whose dependencies are satisfied (per the
   PLAYBOOK build-priority DAG).
2. Write its **contract test FIRST (red)** — an analytic fixture and/or a dolphin oracle,
   to the tolerances in that crate's `CLAUDE.md`. Correctness, not bit-exactness.
3. Implement the smallest code that turns the test green. Idiomatic Rust: minimal
   branching and nesting (≤3 levels), early returns and guard clauses, `match` over if/else
   ladders, iterator chains over manual loops, `Result`-based errors (no `unwrap` on the
   happy path), human-readable names. The PostToolUse hook pushes back on volume/nesting —
   heed it; do not silence it.
4. Gate before moving on: `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, and
   `cargo test --workspace` all clean.
5. Tick the phase in `STATUS.md`, commit (with the Co-Authored-By trailer from CLAUDE.md),
   then continue to the next phase. Do not batch phases; finish and verify one before the next.

**Stop and ASK me (do not guess)** when:
- An "Elevated question" in PLAYBOOK.md becomes load-bearing for the current phase (e.g.
  packaging before Phase 10).
- A required external dependency is missing — run the environment preflight at the top of
  the phase: system GDAL/HDF5 (Phase 8), the SNAPHU binary (Phase 9), or a Python env with
  the pinned dolphin for oracle generation. If missing, report exactly what's missing and pause.
- The dolphin reference version is not yet pinned (Open question #1) — needed before any
  oracle-based validation.

Otherwise do not debate directions: make the call, state the load-bearing assumption in one
line, proceed.

**Done (v1.0.0):** every phase checked in `STATUS.md`; `dolphin run --config <yaml>` produces
a displacement time series on a real CSLC burst stack matching the dolphin oracle within the
§Correctness tolerances; clippy/fmt/tests green; README status updated.

---

## Launching with elevated permissions

Interactive, dynamic (recommended) — the model self-paces the loop:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
claude --dangerously-skip-permissions
# then, in the session:
/loop <paste the Prompt section above>
```

`--dangerously-skip-permissions` bypasses every tool-permission prompt so the loop runs
unattended (cargo, git, file writes). `/loop` with no interval = dynamic self-pacing.

Headless / one-shot:

```sh
claude --dangerously-skip-permissions -p "$(sed -n '/^## Prompt$/,/^## Launching/p' BUILD_PROMPT.md)"
```

Less-elevated alternative (still prompts for shell commands): `--permission-mode acceptEdits`.
