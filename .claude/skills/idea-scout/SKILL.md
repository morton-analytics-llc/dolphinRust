---
name: idea-scout
description: Source new work for dolphinRust from InSAR-processing ecosystem scanning (dolphin parity gaps, perf levers, OPERA/NISAR cadence) and inbound GitHub signal, triage each one, and write approved candidates into the backlog ledger so backlog-pipeline can pick them up. Loop-compatible — upstream of backlog-pipeline, not a replacement for it. Runs full monthly; `inbound` mode weekly.
argument-hint: "[inbound | topic-focus]  (inbound = triage GitHub signal only, for the weekly run)"
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Idea Scout

Sources and triages new work for dolphinRust, then hands qualified items to `backlog-pipeline`
via a GitHub issue label or the ledger. Generation and execution are separate skills on purpose
— they run on different cadences (competitive/ecosystem scan monthly, inbound triage weekly,
pipeline continuously).

dolphinRust is the **algorithm library** (a Rust rebuild of OPERA `dolphin`/DISP-S1), not the
product — so "ideas" here are parity gaps against the dolphin reference, perf levers, and
ecosystem shifts (OPERA/NISAR cadence), plus cross-repo pull from the host app `../eo`. There is
no `market_research/` corpus; ground the scan in the sources below, not training-data recall.

## Mode

- **`inbound`** (weekly run): skip Step 1A. Do only Step 1B (inbound signal) + dedup + triage +
  disposition. Cheap, deterministic, no web agents. This is the "feature requests" cadence.
- **default / topic-focus** (monthly run): full Step 1A + 1B. A non-`inbound` argument scopes the
  scan (e.g. `perf`, `unwrap`, `nisar`).

## Step 1 — Source candidates (run the two 1A agents in parallel)

**A. Competitive / ecosystem scan** *(skip in `inbound` mode)*

Ground both agents in dolphinRust's actual reference frame first: `PLAYBOOK.md` (phase roadmap,
Optimization log, Out-of-scope, Elevated/Open questions), `CHANGELOG.md`, each crate's
`CLAUDE.md`, and the memory index (milestones + deferrals). Do not let an agent fall back to
generic training-data competitors.

- **researcher agent** — "dolphinRust is a performance-focused Rust rebuild of OPERA `dolphin`
  (the DISP-S1 InSAR displacement pipeline); the Python `dolphin` library is its scientific spec
  (parity oracle pinned at v0.35.0). Scan for (1) capabilities in current `dolphin` releases
  (through the latest tag) NOT yet in dolphinRust — new solvers, corrections, quality layers,
  product schemas; (2) OPERA DISP-S1 / DISP-NI product and cadence changes, NISAR L-band data
  availability, Sentinel-1 continuity news, and relevant algorithm papers (phase-linking, MCF/
  MInSAR unwrapping, atmospheric correction) that open or close a feature window. Return 3-5
  concrete gaps, each one sentence: {capability} — {why it matters for scientific parity or a
  GroundPulse production run}." Scope to `$ARGUMENTS` if a topic-focus is given.
- **competitive-analyst agent** — "dolphinRust's peer set is the InSAR-processing toolchain:
  MintPy, ISCE3, GMTSAR, LiCSBAS, StaMPS, and the unwrappers SNAPHU/tophu. What do those ship —
  algorithmically or in performance — that dolphinRust doesn't, that would matter to a user
  running a real CSLC burst stack to displacement? Return 3-5 concrete gaps, each one sentence:
  {capability} — {why a user choosing a processor would care}." Scope to `$ARGUMENTS` if given.

**B. Inbound signal** *(always)*
```
gh issue list --state open --label enhancement --json number,title,body,createdAt
```
Also pull cross-repo pull from the host app — eo issues that name dolphinRust / gp-dolphin and
imply a change here (issue #1 was one):
```
gh issue list --repo morton-analytics-llc/eo --state open --search "dolphin OR gp-dolphin OR provenance" --json number,title,body 2>/dev/null
```
These are already-articulated asks, not generated ideas — keep them a distinct source in output.

## Step 2 — Dedup against known state

For every candidate, check for existing coverage and drop matches:

- `PLAYBOOK.md` — a phase that already delivers it (`done`), an **Out of scope (initial)** entry,
  or an **Elevated/Open question** that already owns the decision. An out-of-scope item is only a
  fresh candidate if new evidence changes the rationale.
- `CHANGELOG.md` + memory index (`MEMORY.md` milestones) — already shipped under another name.
- `md/intake/idea-scout-ledger.md` — SHIPPED / DEFERRED / OUT-OF-SCOPE entry on the same
  capability. Don't re-propose a documented deferral without new evidence.
- Open GitHub issues — an existing issue already tracks it.

## Step 3 — Triage survivors

For each survivor, invoke `/feature-request` — it runs product-manager + chief-of-staff +
competitive-analyst in parallel (plus scoping when buildable) and produces **Build now / Build
after X / Defer / Decline**. Invoke once per candidate when few; inline the same agent calls as
one batch when triaging many is cheaper. Weigh parity value (does dolphin have it?) and whether a
real GroundPulse run needs it — not novelty.

## Step 4 — Write disposition

Every candidate gets one of three dispositions — no silent drops. Prefer GitHub issues as the
record; the ledger is only for gate-tracked items.

- **Build now** → self-sourced idea: `/feature-request`'s issue-creation step creates one with
  `enhancement`; accept it. Inbound issue: use it directly. Either way ensure the label exists
  and apply it:
  ```
  gh label list --json name -q '.[].name' | grep -qx backlog-ready \
    || gh label create backlog-ready --color 0E8A16 --description "Approved for autonomous backlog-pipeline pickup"
  gh issue edit {n} --add-label backlog-ready
  ```
  `backlog-ready` is the signal `/backlog-pipeline next` scans for; no ledger entry needed.
- **Build after X** → create/use the issue with `enhancement`, but do **not** apply
  `backlog-ready` — "X" is a real unmet condition (a prior phase, a bench result, an upstream
  product like DISP-NI). Add a `## DEFERRED` entry in `md/intake/idea-scout-ledger.md`
  referencing the issue number, with the re-entry gate = "X". A future scout pass promotes it
  (apply `backlog-ready`, remove the ledger entry) once the gate clears.
- **Decline** → self-sourced with no prior record: no issue. Inbound issue: comment the rationale
  and label `wontfix` — never close silently. If it touches the host-app surface (export, share,
  demo, read-only viewing) write the one-sentence reason it's not needed rather than defaulting
  to decline — those are often GroundPulse-sales-critical even when parity-first triage ranks
  them low.

## Step 5 — Stop

Report what was sourced, deduped, and dispositioned. Do **not** run `/design` or enter the build
pipeline — that boundary belongs to `backlog-pipeline`. Creating issues and applying labels is
shared, visible state: if this run was triggered automatically (scheduled cron or `/loop`), say
so in the report so a human reading issue history later understands the provenance.

## Output

```
## Idea Scout — {date}  [mode: full | inbound]

### Sourced
- Competitive/ecosystem: {N candidates}   (skipped in inbound mode)
- Inbound (GitHub, this repo + eo): {N candidates}

### Deduped away
- {candidate} — already covered by {PLAYBOOK phase / Out-of-scope / ledger entry / issue #}

### Triaged
| Idea | Source | Decision | Gate / reason |
|---|---|---|---|
| ... | competitive / ecosystem / inbound | Build now / Build after X / Decline | ... |

### Issues / ledger updated
{N issues created or labeled backlog-ready, N ledger DEFERRED entries added}

### Provenance
{manual, or "automated — scheduled {weekly inbound | monthly full} run"}

### Next
`/backlog-pipeline next` picks up any `backlog-ready`-labeled issue on its next run.
```
