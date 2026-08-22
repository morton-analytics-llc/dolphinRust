# Issue #50 layover/shadow mask intake

**Source:** live GitHub issue #50, refreshed 2026-08-22 UTC:
`fix(config): layover_shadow_mask_files is inert — declared, defaulted, never read`.
**Plan:** `md/plans/issue-50-layover-shadow-mask-2026-08-22.md`.
**Queue scope:** #50 is the only open dolphinRust issue; there are no open pull requests.

| ID | Canonical requirement | Disposition |
|---|---|---|
| DR-050-MASK | Make `layover_shadow_mask_files` operational or fail explicitly when populated. The selected contract is one aligned binary mask per processed burst, applied before covariance/phase linking, with no resampling or silent fallback. | **Scheduled — T01-T05** |
| DR-050-GUARD | Prevent another accepted-but-inert public config field. Every public config field must have an explicit, mechanically complete disposition; unsupported non-default values must fail before processing. | **Scheduled — T06** |
| GP-050-TRUTH | Reconcile GroundPulse's caller and customer-facing statement that layover/shadow masks can null pixels. dolphinRust support alone does not make that production claim true. | **Deferred — destination: a dedicated `eo` integration issue titled “Wire released dolphinRust layover/shadow masks into GroundPulse.” Re-enter after the engine PR is merged and a release/pin is selected; acceptance requires STATIC-mask extraction, per-burst mapping, caller wiring, and a fresh terminal-artifact receipt.** |

## Coverage audit

Every requirement in issue #50 has a disposition. DR-050-MASK maps to T01-T05 and
DR-050-GUARD maps to T06. GP-050-TRUTH is not part of the dolphinRust implementation PR;
its destination and re-entry gate are explicit. dolphinRust has no UI. The paired
customer-facing work is GP-050-TRUTH rather than an invented engine UI task.
