# Idea Scout Ledger

Gate-tracked backlog for dolphinRust. `idea-scout` writes **Build after X** candidates here as
`## DEFERRED` entries with a re-entry gate; `backlog-pipeline` reads them, evaluates the gate, and
on a successful PR moves the entry to `## SHIPPED`. Items approved for immediate build live on
GitHub as issues labeled `backlog-ready` instead — not here.

This ledger is only for items that genuinely need gate-tracking (uncertain evidence, a blocking
prior phase, an upstream product). It is not a general TODO list — PLAYBOOK.md owns the phase
roadmap, and memory owns milestones/deferrals.

Entry format:

```
### D{n} — {short title}
- **Source**: {competitive / ecosystem / inbound issue #}
- **Issue**: #{n}  (enhancement-labeled, NOT yet backlog-ready)
- **Re-entry gate**: {the "X" — a verifiable-now check or an external/human condition}
- **Design sketch**: {one or two lines, or a md/design/ path}
- **Added**: {date} by {manual | scheduled scout run}
```

## DEFERRED

### D1 — Degenerate all-non-finite input window silently yields temporal_coherence=1.0 / displacement=0.0
- **Source**: inbound (cross-repo signal, `../eo` `dolphin-safety-report.md` Finding #2)
- **Issue**: #8  (enhancement-labeled, NOT yet backlog-ready)
- **Re-entry gate**: run the pinned dolphin v0.35.0 oracle against an all-NaN synthetic SLC
  stack and record its actual output. If Python dolphin also returns temp_coh=1.0/disp=0.0
  on all-NaN input, this becomes a forward-divergence architecture decision (add an explicit
  nodata guard) rather than a parity bug fix — elevate that call per PLAYBOOK's "elevate
  genuine blockers" policy rather than deciding it in an unattended run. If dolphin fails
  loudly/differently, this is a direct parity bug, fix to match the oracle.
- **Design sketch**: `dolphin-phaselink/src/covariance.rs::finite_or_zero` +
  `coherence_entry` (AMP_FLOOR underflow → 0+0j matrix) combined with
  `quality.rs::temp_coh_single`/`pair_diff` (phase-only, `arg(0+0j)==0.0`) reproduces the
  observed 1.0/0.0 exactly on a fully-degenerate window — traced, not yet fixed.
- **Added**: 2026-07-20 by scheduled scout run

## SHIPPED

_(none yet)_
