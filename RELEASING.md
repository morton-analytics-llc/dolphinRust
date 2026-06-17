# Releasing dolphinRust

## Packaging status

Every crate carries complete publish metadata (`description`, `keywords`, `categories`,
`license`, `repository`, `homepage`, `readme`) and a pinned internal dependency version, so
the workspace is publishable. Verified:

```sh
cargo publish --dry-run -p dolphin-core   # PASS — packages + verifies cleanly
```

`dolphin-core` is the dependency root and has no internal dependencies, so its dry-run runs
end to end. The other 11 crates depend on internal crates by `path` **and** `version`; until
those dependencies are actually on crates.io, `cargo publish --dry-run` / `cargo package` for
a downstream crate reports:

```
no matching package named `dolphin-core` found — location searched: crates.io index
```

This is the standard unpublished-workspace chicken-and-egg (the index lookup, not a manifest
or file-list defect): cargo verifies each dependency exists on the registry, and they don't
yet. The fix is to publish in dependency order; each crate becomes available for the next.

## How eo consumes dolphinRust

GroundPulse (`../eo`) depends on dolphinRust as a **git (or path) workspace dependency**, not
from crates.io, e.g.:

```toml
dolphin-workflows = { git = "https://github.com/morton-analytics-llc/dolphinRust", tag = "v1.0.0" }
```

So crates.io publication is **optional** for the eo integration. If/when publishing to
crates.io is desired, publish in this topological order (each waits for the previous to
appear on the index):

1. `dolphin-core`
2. `dolphin-shp`, `dolphin-ps`, `dolphin-stack`, `dolphin-filtering`, `dolphin-unwrap`,
   `dolphin-phaselink`, `dolphin-timeseries`, `dolphin-io` (all depend only on `dolphin-core`)
3. `dolphin-ingest` (depends on `dolphin-io`)
4. `dolphin-workflows` (depends on all of the above)
5. `dolphin-cli` (depends on `dolphin-core`, `dolphin-workflows`)

## Cutting a release

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --no-deps --workspace
git tag v1.0.0 && git push origin v1.0.0   # only after sign-off
```
