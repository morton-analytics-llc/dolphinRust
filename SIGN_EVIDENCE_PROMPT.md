# dolphinRust — backfill the ifg sign-convention evidence, then land v1.3.0 part 2 (dynamic loop)

The atmo loop landed two correctness commits on `v1.3-atmo` alongside the corrections feature:
the interferogram-formation fix `ref·conj(sec)` (`e1db05a`) and its oracle companion
(`2c85a79`). The fix is **correct** — pairs are `(ref, sec)`, dolphin's `interferogram.py`
forms `ref·conj(sec)`, and the old `sec·conj(ref)` order globally inverted the displacement
*and velocity* sign of **every release v1.0.0–v1.2.0**. It was invisible because the oracle
generator carried the identical inversion, so the contracts proved Rust agreed with a flipped
oracle, not with production (VALIDATION.md even records "Sign is +1, engines agree").

The gap: the decisive evidence — F38502/Corcoran **corr −1.0000** vs real production — exists
only in a commit message and a code comment. It has **no committed reproducible test and no
VALIDATION.md record**, unlike every other real-data claim here (IONEx, NISAR each have a
gated test + receipts). This loop brings the sign fix up to that bar, **then lands**.

---

## Prompt

On branch `v1.3-atmo`, backfill committed evidence for the ifg sign-convention fix to the same
standard as the IONEx/NISAR real-data gates, then merge to `main` and push. Dynamic, self-paced
loop, contract test first; gate every step on `cargo fmt` + `cargo clippy --all-targets --
-D warnings` + `cargo test` + `cargo doc --no-deps`. Honesty rule: only land if the real-data
comparison actually reproduces the correct sign — if the re-fetch fails or the correlation is
not unambiguously positive after the fix, **stop and report**, do not push.

Read first: `crates/dolphin-workflows/src/displacement.rs` (`unwrap_pair`, the fixed
`pl[i]·conj(pl[j])` formation), `crates/dolphin-timeseries/src/network.rs` (`single_reference`
emits `(min,max)=(ref,sec)`), `oracle/gen_displacement.py` (the fixed companion),
`VALIDATION.md` (the IONEx/NISAR record + reproduce-command style to mirror), `validation/`
(`creds.sh`, `fetch_real.py`, `crop_real.py` — the existing real-OPERA fetch tooling).

Two artifacts, in order:

1. **Always-on analytic sign-regression guard** (no network — the permanent guard the bug
   lacked). A contract test on a tiny synthetic 2+-acquisition stack with an *analytically
   known* LOS displacement sign: a known monotonic range change between date 0 and a later
   date, run through `run_displacement` (or the `unwrap_pair`→invert path), asserting the output
   displacement sign matches dolphin's `ref·conj(sec)` convention (positive where the physical
   model says positive). It must **fail if `unwrap_pair` is reverted to `sec·conj(ref)`** —
   verify that by flipping it locally, watching the test go red, and flipping back. This locks
   the convention regardless of data availability. Put it where the displacement contracts live.

2. **Gated real-data sign test + VALIDATION.md receipts** (matches IONEx/NISAR). Re-fetch the
   real OPERA stack the atmo loop used (`source validation/creds.sh`; F38502/Corcoran —
   Central Valley/Corcoran burst, reuse `fetch_real.py`/`crop_real.py`). Run dolphinRust's
   referenced ifg and compare its sign/correlation against the **production unwrapped ifg** (a
   full `dolphin run`, or the OPERA-provided unwrapped layer). Add a test gated on an env var
   (e.g. `SIGN_REF_PROD_IFG` / the stack path) that skips cleanly when unset — same pattern as
   `real_ionex_parses_to_physical_delay` / `reads_real_nisar_granule`. Record in VALIDATION.md:
   the granule/burst ID, **the measured correlation before the fix (≈ −1.0000) and after
   (≈ +1.0000)**, why the old contract was blind (lockstep-inverted oracle), and the exact
   reproduce command. ⛔ If the granule can't be re-fetched this run, do **not** fake it — write
   the always-on guard (artifact 1), record the real-data gate as deferred-with-receipts, and
   **stop before pushing** so I can decide.

3. **Correct the record + note the impact.** Fix VALIDATION.md line ~89 ("Sign is +1, engines
   agree") to explain it reflected a lockstep-inverted oracle, now corrected. Add a `CHANGELOG`
   entry stating plainly that **v1.0.0–v1.2.0 produced displacement/velocity with inverted LOS
   sign vs dolphin; fixed in this release** — and that the eo-relevant `velocity_mm_yr` sign
   (subsidence vs uplift, which drives risk tiers) is now correct. Update `STATUS.md`.

4. **Land.** With both gates green (the analytic guard always; the real-data test passing on the
   re-fetched stack) and fmt/clippy/test/doc green on default + no-gpu, merge `v1.3-atmo` into
   `main` with `--no-ff` and push. **Do not tag v1.3.0** — the tropospheric correction's
   4326→UTM warp is still deferred (real-frame tropo isn't end-to-end yet), so v1.3.0 is not
   complete; leave tagging for after that lands.

State the load-bearing assumption in one line and proceed; don't debate directions.

**Definition of Done:**
- [ ] Always-on analytic sign-regression guard committed; proven to go red if `unwrap_pair` is
      reverted to `sec·conj(ref)` (state that you checked).
- [ ] Real-data sign test (gated, skips when unset) committed **and** VALIDATION.md records the
      F38502/Corcoran before(−1)/after(+1) correlation with receipts + reproduce command — OR,
      if the granule is unreachable, the guard ships, the real gate is deferred-with-receipts,
      and the loop **stops before pushing**.
- [ ] VALIDATION.md "Sign is +1" line corrected; CHANGELOG states the v1.0.0–v1.2.0 inverted-sign
      regression is fixed (incl. the `velocity_mm_yr`/eo-risk implication); STATUS updated.
- [ ] Gates green (default, `--features gpu`, `--features no-gpu`): fmt, clippy -D warnings,
      test, doc.
- [ ] If the real-data test passed: `v1.3-atmo` merged `--no-ff` into `main` and pushed; **no
      v1.3.0 tag** (tropo warp still deferred). If it could not: nothing pushed, stopped with a
      report.

---

## Launching with elevated permissions

Make sure the previous CLI session is closed first. Then two steps — **step 2 is a slash command
typed inside Claude Code, not a shell command.**

1. In your terminal:

```sh
cd /Users/ryanemorton/Documents/GitHub/dolphinRust
git checkout v1.3-atmo
source validation/creds.sh          # required — re-fetches the F38502/Corcoran stack
claude --dangerously-skip-permissions
```

2. Wait for the Claude Code prompt, then type at it:

```
/loop read SIGN_EVIDENCE_PROMPT.md and backfill the ifg sign-convention evidence then land per its Definition of Done
```

`--dangerously-skip-permissions` runs cargo/git/curl unattended. `/loop` with no interval =
dynamic self-pacing. It pushes **only** if the real-data sign comparison reproduces the correct
(+1) sign on the re-fetched stack; otherwise it ships the always-on guard, defers the real gate
with receipts, and stops without pushing. It does **not** tag v1.3.0 (tropo warp deferred).
