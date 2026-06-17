# dolphinRust — project guide for Claude Code

Ground-up Rust **rebuild** of the OPERA InSAR / DISP-S1 displacement pipeline, for
performance. This is **not a port**: the Python `dolphin` library is the algorithm
reference (the scientific spec), not a line-by-line target. We choose the data layouts,
solvers, and parallelism that are fastest in Rust, and validate that results are
*scientifically correct* — against analytic fixtures and dolphin as a reference oracle, to
physically-meaningful tolerances, not bit-exactness. **v1.0.0 is the first complete build**
— end-to-end displacement from a real CSLC burst stack.

Domain knowledge is **dispersed**: each crate carries its own `CLAUDE.md` with the InSAR
math and conventions specific to it. Read the crate's `CLAUDE.md` before working in it. The
phased roadmap and parity strategy live in [PLAYBOOK.md](PLAYBOOK.md).

## Workflow — dynamic, contract-first

- Build **phase by phase** per PLAYBOOK.md. The flow is dynamic/iterative, not the staged
  /design→/implement pipeline: take the next phase, write its contract, make it pass, move on.
- **Test/contract-driven.** For every kernel, write the contract test FIRST (red) — a
  fixture with a known analytic answer, and/or golden data from dolphin used as an oracle.
  A kernel is "done" only when its contract test is green. Code existence is not done.
- One phase at a time. Don't scaffold ahead of the phase you are proving.

## Idiomatic Rust (enforced)

- **Minimal branching and nesting.** Early returns and guard clauses over nested `if`;
  `match` over `if/else` ladders; iterator chains over manual index loops. Keep nesting
  ≤3 levels — the PostToolUse hook pushes back past that, and `clippy::cognitive_complexity`
  warns.
- **Small, single-purpose functions.** Simplicity-first: three similar lines beat a clever
  abstraction; a helper must be called 3+ times to exist. The hook pushes back on large
  writes; `clippy::too_many_lines` warns.
- **Result-based errors in library code** — no `unwrap`/`expect`/`panic!` on the happy
  path; propagate with `?` and crate error enums. (`unwrap` is fine in tests.)
- **Human-readable naming.** Descriptive names, no cryptic abbreviations. Mirror dolphin's
  scientific terms where it aids parity (`temporal_coherence`, `half_window`, `ministack`).
- **Linters stay green continuously.** `cargo fmt` runs on every Rust edit via hook; keep
  `cargo clippy --all-targets -- -D warnings` clean before completing a phase.

## Verify before claiming done

`cargo check --workspace`, `cargo clippy`, then `cargo test` including the phase's
parity/contract test. Report failures with output; never claim success unverified.

## Working with me

- **Elevate genuine blockers.** When a decision changes the architecture or a contract and
  is not answerable from the code or the playbook, ask. Standing strategic questions are
  tracked under "Elevated questions" in PLAYBOOK.md.
- **Do not debate directions unless I ask.** Make the call, state the load-bearing
  assumption in one line, proceed. No menus of alternatives.

## Host app

GroundPulse (`../eo`) is adopting the Python `dolphin` now; dolphinRust is its **optimized
Rust drop-in replacement** — same algorithms and workflow surface, faster. It is consumed
as a synchronous library; GroundPulse bridges to its tokio runtime via `spawn_blocking`.
The cleanest correctness oracle is GP's own production dolphin output. See PLAYBOOK.md
§GroundPulse integration.
