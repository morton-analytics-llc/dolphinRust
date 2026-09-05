---
name: backlog-pipeline
description: Carry one human-approved dolphinRust issue through a red-to-green analytic or oracle contract test, verification, and a PR merged on green CI. Use for scheduled contract-first backlog work.
---

# dolphinRust backlog pipeline

Require `backlog-ready`, non-red `main`, no open `automation-pr`, and an isolated worktree. Process one issue and never treat candidate provenance as approval.

Define the analytic fixture or pinned dolphin oracle, tolerance, and expected failure. Write and run the contract test red before production code. Implement the smallest idiomatic Rust change, then run the contract green, `cargo check --workspace`, `cargo clippy --all-targets -- -D warnings`, and relevant workspace tests with unmasked exit codes.

Open one PR labeled `automation-pr` with provenance, `Closes #N`, red-to-green evidence, commands/results, numeric risks, and unrun external gates. Merge it once CI is green — enable auto-merge rather than waiting — and delete the branch. Do not ask for merge approval.

Still stop before release, crate publication, the GroundPulse submodule bump, and deployment. Those reach outside this repo; merging does not.
