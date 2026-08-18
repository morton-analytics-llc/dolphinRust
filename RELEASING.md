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

GroundPulse (`../eo`) vendors dolphinRust as the `vendor/dolphinRust` git submodule. The
superproject records the release tag's target commit as a gitlink; git does not store the tag
name itself. Verify the pin resolves to the intended tag before committing it:

```sh
git -C ../eo/vendor/dolphinRust fetch --tags origin
git -C ../eo/vendor/dolphinRust checkout v1.5.0
git -C ../eo/vendor/dolphinRust describe --exact-match --tags HEAD
git -C ../eo add vendor/dolphinRust
```

Crates.io publication is therefore **optional** for the eo integration. If/when publishing to
crates.io is desired, publish in this topological order (each waits for the previous to
appear on the index):

1. `dolphin-core`
2. `dolphin-shp`, `dolphin-ps`, `dolphin-stack`, `dolphin-filtering`, `dolphin-unwrap`,
   `dolphin-phaselink`, `dolphin-timeseries`, `dolphin-io` (all depend only on `dolphin-core`)
3. `dolphin-ingest` (depends on `dolphin-io`)
4. `dolphin-workflows` (depends on all of the above)
5. `dolphin-cli` (depends on `dolphin-core`, `dolphin-workflows`)

## Cutting a release

1. Move the accumulated changelog entries from Unreleased to the version/date and open a new
   empty Unreleased section.
2. Set `[workspace.package].version` and every internal workspace dependency requirement to
   the release version, then regenerate `Cargo.lock`.
3. Run the complete local release checks:

```sh
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --no-deps --workspace
oracle/.venv/bin/python -m compileall -q validation
oracle/.venv/bin/python -m unittest discover -s validation/tests
git diff --check
```

4. Merge through a green PR, then require green `main` CI on the exact release commit.
5. Create and push an annotated tag from that commit, create the GitHub Release, and verify the
   remote peeled tag target:

```sh
git switch main
git pull --ff-only origin main
git tag -a v1.5.0 -m "Release v1.5.0"
git push origin v1.5.0
gh release create v1.5.0 --title v1.5.0 --notes-file /path/to/release-notes.md
git describe --exact-match --tags HEAD
git ls-remote --tags origin refs/tags/v1.5.0 'refs/tags/v1.5.0^{}'
```

Crates.io publication is a separate operation; a GitHub release does not authorize it.
