# dolphinRust Codex guidance

dolphinRust is the active contract-first Rust displacement engine consumed by GroundPulse. Read `CLAUDE.md` and preserve the dynamic, idiomatic Rust architecture.

Every analytic or oracle change starts with a specific red contract/parity test and is complete only when that test turns green. Then run `cargo check --workspace`, `cargo clippy --all-targets -- -D warnings`, and the relevant test set. Report external-data or numeric-oracle limitations exactly.

Backlog automation accepts only Ryan-applied `backlog-ready`, skips red `main` and an existing `automation-pr`, works in an isolated worktree, and opens at most one verified `automation-pr`. Stop before merge, release, publication, or the GroundPulse submodule bump.
