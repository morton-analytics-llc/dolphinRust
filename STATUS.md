# dolphinRust — build status

Target: **v1.0.0** (first complete build). Update this file as phases land — it is the
single source of truth for build progress across sessions. Phase details in PLAYBOOK.md.

## Phases (build in dependency order, per PLAYBOOK.md DAG)
- [x] 0 — Foundation (`dolphin-core`): types, `StridedBlockManager`, config, error
- [x] 1 — Covariance + EMI/EVD phase linking (`dolphin-phaselink`) ★
- [x] 2 — SHP selection (`dolphin-shp`)
- [x] 3 — PS selection (`dolphin-ps`)
- [x] 4 — Quality layers (`dolphin-phaselink`): temp_coh + compressed SLC done;
      **CRLB/closure deferred** — absent in pinned dolphin v0.35.0 (off the v1.0.0 critical path)
- [ ] 5 — Ministack sequencing (`dolphin-stack` + `workflows::sequential`)
- [ ] 6 — Interferogram network + SBAS L2 inversion (`dolphin-timeseries`)
- [ ] 7 — Filters (`dolphin-filtering`)
- [ ] 8 — I/O layer + S3 read-staging (`dolphin-io` + `dolphin-ingest`) — needs GDAL/HDF5
- [ ] 9 — Unwrapping dispatch (`dolphin-unwrap`) — needs SNAPHU binary
- [ ] 10 — Pipeline orchestration + CLI (`dolphin-workflows` + `dolphin-cli`)

## Awaiting input (see PLAYBOOK.md questions)
- ~~Pin the dolphin reference version~~ — **pinned: `v0.35.0` (`e567e55`)**.
- Packaging: workspace member of `eo` vs. separate crate dependency — before Phase 10.

## Scaffold (done)
- [x] Workspace, 12 crates, builds clean (`cargo check`, `clippy`, `fmt`)
- [x] Claude Code setup: root + per-crate `CLAUDE.md`, PostToolUse hook, workspace lints
- [x] PLAYBOOK.md, README.md
