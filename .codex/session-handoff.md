# Session handoff — 2026-07-09

## Summary

Implemented, reviewed, verified, and committed the MMX1/ICMX GNSS ground-truth harness for dolphinRust. The harness now provides a reproducible acquisition/crop/run/score path for independent GNSS validation of native MCF and SNAPHU without treating the Python dolphin oracle as truth. A separate contract-first automation-assets commit landed afterward. Before this EOD write, local `main` was clean, had no PR, and was two commits ahead of `origin/main`; the only current worktree addition is the untracked local `.codex` handoff chain.

This is the first `.codex` handoff for this repository.

## Completed

- Added the tracked MMX1 recipe and implementation plan.
- Extended real-data acquisition to select the exact 13 T005 epochs, fetch the matching CSLC-S1-STATIC product, validate downloaded HDF5 burst/pass identity, hash inputs, and support safe dry-run/static-only preflight.
- Extended cropping to build a 384x384 MMX1 core fixture and a shared MMX1/ICMX frame while preserving CSLC/STATIC metadata and pinning burst/date contracts.
- Added NGL IGS20 tenv3 parsing, bounded interpolation, ENU-to-LOS projection, spatial-reference cancellation, fixed-window sampling, OLS/TLS metrics, and JSON/CSV/SVG artifacts.
- Added the controlled native/SNAPHU runner with config-identity checks, per-backend receipts, sourced-geometry validation, and meaningful exit codes.
- Reviewed and fixed:
  - lost receipts when one backend was unavailable or invalid;
  - exit code 0 on a scientific validation failure;
  - missing downloaded-HDF burst/pass validation;
  - mixed-station/non-finite GNSS acceptance and incomplete cache provenance;
  - acceptance of stale fixtures with the wrong burst or acquisition dates.
- Committed the harness as `2d99348 feat(validation): add MMX1 GNSS ground-truth harness`.
- A subsequent commit, `29bb709 chore(codex): add contract-first automation assets`, added `AGENTS.md`, `.agents/skills`, and Claude skill symlinks.

## In progress

- The full 13-CSLC acquisition, both fixture crops, and native/SNAPHU scoring runs have not been executed.
- Initial acceptance thresholds remain provisional: endpoint residual <=20 mm, TLS slope 0.85-1.15, and correlation >=0.90.
- Atmospheric inputs are not part of the first comparison. If corrections remain off, residual error includes atmosphere and cannot be attributed solely to unwrapping.

## Verification

Completed and passing:

- `oracle/.venv/bin/python -m unittest discover -s validation/tests -p 'test_gps_*.py'` — 22 tests.
- `oracle/.venv/bin/python -m compileall -q validation`.
- Live ASF catalog dry-run — exactly 13 declared CSLC epochs plus one matching STATIC; expected transfer 3.55 GB.
- Authenticated STATIC acquisition preflight — 161.6 MB product transferred, HDF5 datasets and SHA-256 validated.
- Live GNSS preflight — MMX1 endpoint -101.839 mm, ICMX +2.424 mm, MMX1-minus-ICMX -104.262 mm; MMX1 has one declared interpolation on 2023-03-05.
- Missing-fixture runner smoke — structured `not_evaluable`, exit code 2.
- `cargo check --workspace`.
- `cargo clippy --all-targets -- -D warnings`.
- `cargo test --workspace`.
- `git diff --check`.

Not run:

- Full live MMX1 core/native/SNAPHU gate.
- Full shared-frame GNSS scoring.
- Push, PR, or deployment.

## Open questions

1. Where should enough disk headroom be created for the 3.55 GB source stack plus crops and four backend work directories? The host was at about 99% utilization with roughly 12 GiB free.
2. Should the provisional comparison bars become hard release gates or remain report-only for the first real run?
3. Should 13-date ionosphere/troposphere acquisition be included before attributing any residual specifically to the unwrappers?
4. Should the two local commits be pushed directly to `origin/main` or sent through a PR?

## Next actions

1. Free or attach sufficient storage before downloading the full stack.
2. Run the full acquisition and MMX1 core gate:
   ```sh
   source validation/creds.sh
   oracle/.venv/bin/python validation/fetch_real.py --recipe validation/gps_mmx1.json --with-static
   oracle/.venv/bin/python validation/crop_real.py --recipe validation/gps_mmx1.json --fixture mmx1_core
   oracle/.venv/bin/python validation/run_gps_ground_truth.py --recipe validation/gps_mmx1.json --fixture mmx1_core --build
   ```
3. If the core gate is valid, build and score the shared frame:
   ```sh
   oracle/.venv/bin/python validation/crop_real.py --recipe validation/gps_mmx1.json --fixture mmx1_icmx_common
   oracle/.venv/bin/python validation/run_gps_ground_truth.py --recipe validation/gps_mmx1.json --fixture mmx1_icmx_common --score
   ```
4. Review `gps_ground_truth.json`, CSV, and SVG; keep native and SNAPHU judgments separate from shared atmospheric/pipeline residual.
5. Decide push/PR strategy for local `main`.

## References

- Branch: `main`; upstream: `origin/main`; current state: ahead 2, with only `.codex/` untracked from this EOD write.
- No open or associated pull request.
- Commits: `2d99348` (GPS harness), `29bb709` (automation assets).
- Plan: `md/plans/gps-mmx1-aoi-harness.md`.
- Recipe: `validation/gps_mmx1.json`.
- Runner: `validation/run_gps_ground_truth.py`.
- Scorer: `validation/gps_ground_truth.py`.
- Evidence/status: `VALIDATION.md`.
