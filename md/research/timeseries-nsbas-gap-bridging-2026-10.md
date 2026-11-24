# NSBAS gap-bridging for the SBAS inversion (#111)

**Date:** 2026-10-31 · **Base commit:** `25ab07a` · **Issue:** #111
**Question:** should `dolphin-timeseries` add an NSBAS-style temporal constraint so a
network split by an acquisition gap still inverts?

## Result

**Defer.** The re-entry gate in #111 has not been met, and nothing in the production path can
meet it today. GroundPulse runs an index-based network (`max_bandwidth = 2`), and an
index-based network always spans an acquisition gap. On a synthetic gap network, a denser
index network beat NSBAS with fewer interferograms. NSBAS also carries a bias that the gate
would have to justify: its linear constraint pulls the post-gap segment by −7.1 mm when the
ground has a seasonal signal.

The probe found a separate problem. The current inversion does not fail cleanly on a
disconnected network: L1 panics, and weighted L2 publishes arbitrary offsets for about half
the pixels. That needs a bug fix whatever happens to #111 (see "Bugs found").

## Where gaps can come from in dolphinRust

- `build_network` (`crates/dolphin-timeseries/src/network.rs:26`) unions four modes.
  `reference_idx` and `max_bandwidth` select pairs by index distance, so the network is
  connected whatever the acquisition dates are. Only `max_temporal_baseline` (day distance) and
  explicit `indexes` can split the network at a calendar gap.
- Per-pixel censoring cannot split it either. The unwrap support mask is 2-D and applies to
  every interferogram alike (`crates/dolphin-workflows/src/displacement.rs:4864-4883`), and
  loop-closure QC masks a whole pixel (`loop_closure.rs:366`). So every pixel inverts the
  configured network or nothing. dolphin is different here: it masks unwrapped pixels per
  interferogram by connected component during L2 inversion (`dolphin/timeseries.py` v0.35.0,
  `run`, "Masking unw pixels during inversion using connected components"), which can split a
  single pixel's network.
- Production: GroundPulse sets `reference_idx = None`, `max_bandwidth = Some(2)`,
  `max_temporal_baseline = None` (`eo/crates/gp-dolphin/src/lib.rs:866-868, 925-927`). The
  GNSS harness sets only `max_bandwidth` (`validation/run_gps_ground_truth.py:192-208`).
  Neither can produce a gap-disconnected network.

So the case #111 targets only happens through a configuration that no current caller uses.
Whether a gap breaks the network is fixed by config and dates, so it can be decided at the
`network` stage (`displacement.rs:516`), before unwrapping.

## What the literature and reference implementations do

- **NSBAS (Doin et al. 2011, §4, eqs. 2–4).** The pixel solve adds one equation per
  acquisition, `φ_k − φs_k − αΔB⊥_k = E_k`. `φs` is either a parameterised function of time,
  whose parameters are solved jointly, or a series kept smooth by Laplacian rows (Cavalié et al.
  2007). The total weight on these rows is "set low enough … in order for lines (4) to only
  impact the relative phase delay between separated image subsets (Lopez-Quiroz et al., 2009)".
  Source: author PDF, isterre.fr/IMG/pdf/505_DOIN.pdf (fetched 2026-10-31).
- **López-Quiroz et al. 2009**, *J. Appl. Geophys.* 69(1):1–15, doi:10.1016/j.jappgeo.2009.02.006.
  The bibliographic record is verified (IP Paris research portal). The UNAM full-text link now
  returns 404, so the method claim above is attributed through Doin 2011, which cites it for
  exactly this point.
- **LiCSBAS** (`LiCSBAS_lib/LiCSBAS_inv_lib.py::invert_nsbas`, comet-licsar/LiCSBAS
  `a145e6c`). Unknowns are the `n_im−1` increments plus `v` and `c`. The bottom block is
  `γ·[tril(ones, k=−1) | −t | −1]`, which ties each cumulative value to `v·t + c` (linear
  only). Default `--gamma 0.0001` (`bin/LiCSBAS13_sb_inv.py:70, 238`), with time in years
  (`:616`) and displacement in mm. The issue calls this a "second-derivative temporal-smoothness
  penalty". It isn't: LiCSBAS constrains to a linear model, and the Laplacian form is the other
  option Doin describes.
- **MintPy** (`src/mintpy/ifgram_inversion.py`, insarlab/MintPy `cc44148`). No regularisation.
  A weighted inversion on a rank-deficient network raises "Input network of interferograms is
  not fully connected!" (`check_design_matrix`, ~l.366-388). The unweighted path warns and
  takes the SVD minimum-norm velocity solution (Berardino 2002 default, `minNormVelocity`).
- **dolphin** (v0.35.0 and `main`, `timeseries.py`). L2 uses `jnp.linalg.lstsq`, which gives the
  minimum-norm solution with no warning (the rank check is commented out). L1 uses
  `jax.scipy.linalg.cholesky(A.T @ A)`. dolphin has no NSBAS equivalent.

## Synthetic prototype

**Setup.** 12-day cadence, 45 nominal acquisitions, acquisitions 18–23 removed. That leaves an
84-day gap (day 204 → 288) and 39 dates. There are 2000 Monte-Carlo pixels with per-date
atmosphere N(0, 3 mm) and per-interferogram noise N(0, 1 mm). Seed `20261031`. Two truths:
**S1** linear −30 mm/yr; **S2** linear −30 mm/yr plus an 8 mm annual sinusoid (phase 0.7 rad).
Networks: `max_temporal_baseline = 48 d` (**tb48**, 136 ifgs, `rank(A) = 37` of 38 columns:
disconnected at the gap), `max_temporal_baseline = 96 d` (**tb96**, 243 ifgs) and
`max_bandwidth = 3` (**bw3**, 111 ifgs). Both of the last two are connected.

"Current" rows are dolphinRust's own `build_network`, `get_incidence_matrix`, `invert_stack`
and `invert_stack_l1` called from a scratch binary. Rust and Python pair lists were asserted
equal. The other rows are numpy `lstsq` on the tb48 `A`:

- NSBAS linear: `[A 0; γ(E | −t | −1)] x = [Δφ; 0]`, with `E` the identity minus the gauge
  column, t in years, mm.
- NSBAS linear+annual: the same with `−sin 2πt, −cos 2πt` columns added.
- Second-derivative: `[A; γ·Δ̄²·D₂]`, with `D₂` the non-uniform second difference scaled to be
  dimensionless at 12-day spacing.

**Metrics.** *Gap offset error*: the mean post-gap error minus the mean pre-gap error. This is
the quantity the network cannot determine. *Rate error*: the OLS slope of the estimate minus the
OLS slope of the truth. *Pre-gap shape*: RMSE within the connected first segment after removing
its mean, which shows whether the constraint distorts data the network does constrain.

**Commands** (scratch dir, not kept in the repo):

```sh
python3 synth.py run_S1 20261031 S1 && python3 synth.py run_S2 20261031 S2
HDF5_DIR=/opt/homebrew/opt/hdf5 CARGO_TARGET_DIR=$SCRATCH/target cargo build --release  # probe/, path-dep on crates/dolphin-timeseries
$SCRATCH/target/release/gap_probe run_S1 && $SCRATCH/target/release/gap_probe run_S2
python3 analyze.py run_S1 && python3 analyze.py run_S2
```

### S2 (linear + seasonal)

| method | ifgs | valid px | ts RMSE mm | gap offset err RMS (mean) mm | rate err RMS (bias) mm/yr | pre-gap shape mm |
|---|---|---|---|---|---|---|
| bw3, L2 (Rust) | 111 | 2000 | 4.37 | 1.37 (+0.05) | 1.58 (+0.04) | 2.96 |
| tb96, L2 (Rust) | 243 | 2000 | 4.24 | 1.19 (+0.04) | 1.28 (+0.04) | 2.92 |
| tb48 current L2 uniform (Rust) | 136 | **0** | – | – | – | – |
| tb48 current L2 coherence-weighted (Rust) | 136 | **979** | 30.59 | 41.64 (+33.40) | 41.42 (+33.22) | 2.94 |
| tb48 current L1 (Rust) | 136 | **panic** | – | – | – | – |
| tb48 min-norm `lstsq` (what dolphin L2 returns) | 136 | 2000 | 26.39 | 35.78 (+35.66) | 35.61 (+35.49) | 2.94 |
| tb48 NSBAS linear γ = 1e-4, 1e-2 | 136 | 2000 | 7.02 | 7.47 (−7.10) | 7.60 (−7.07) | 2.94 |
| tb48 NSBAS linear γ = 0.1 | 136 | 2000 | 7.01 | 7.49 (−7.12) | 7.63 (−7.10) | 2.94 |
| tb48 NSBAS linear γ = 1 | 136 | 2000 | 6.76 | 8.39 (−8.06) | 8.68 (−8.22) | 3.10 |
| tb48 NSBAS linear γ = 10 | 136 | 2000 | 7.93 | 8.76 (−8.38) | 9.13 (−8.60) | 3.69 |
| tb48 NSBAS linear+annual γ = 1e-4, 0.1 | 136 | 2000 | 6.34 | 6.74 (−0.07) | 6.87 (−0.08) | 2.94 |
| tb48 second-derivative γ = 0.1 | 136 | 2000 | 18.47 | 24.49 (−1.08) | 24.42 (−1.08) | 2.92 |
| tb48 second-derivative γ = 1 | 136 | 2000 | 12.74 | 16.69 (−0.92) | 16.67 (−0.93) | 2.17 |

### S1 (linear only), the case that favours NSBAS

| method | ifgs | gap offset err RMS (mean) mm | rate err RMS (bias) mm/yr | pre-gap shape mm |
|---|---|---|---|---|
| bw3, L2 (Rust) | 111 | 1.37 (+0.05) | 1.58 (+0.04) | 2.96 |
| tb48 NSBAS linear γ ≤ 0.1 | 136 | 2.34 (−0.04) | 2.79 (−0.04) | 2.94 |
| tb48 NSBAS linear γ = 10 | 136 | 2.55 (−0.05) | 3.05 (−0.06) | 0.56 |
| tb48 NSBAS linear+annual γ ≤ 0.1 | 136 | 6.74 (−0.07) | 6.87 (−0.08) | 2.94 |
| tb48 second-derivative γ = 0.1 | 136 | 24.47 (−0.48) | 24.40 (−0.48) | 2.92 |
| tb48 current L2 weighted (Rust) | 136 | 42.20 (+33.42), 979 px | 41.98 (+33.25) | 2.94 |

With no noise, NSBAS-linear at γ = 1e-4 gives a gap offset error of +0.000 mm on S1 and
**−7.058 mm on S2**. The S2 bias comes from the model, not the noise.

### What the numbers say

1. **Index-based densification beats NSBAS here, and costs fewer interferograms.** bw3 uses 111
   ifgs against tb48's 136, and its offset error is lower (1.37 vs 2.34 mm RMS even in S1). The
   premise in #111, that NSBAS is "cheaper per interferogram" than densifying, does not hold
   for `max_bandwidth`, because index pairs only cross the gap where they are needed. One
   caveat goes against bw3: this noise model gives gap-spanning pairs the same noise as short
   ones. In real data the 96-day bridging pairs lose coherence, and that is the one case where
   NSBAS could win. That is the measurement the gate asks for.
2. **NSBAS is only as good as its temporal model, and the bias cannot be seen in the output.**
   A linear constraint against a seasonal ground moved the post-gap segment by −7.1 mm, and
   the rate by −7.1 mm/yr, on every pixel in the same direction. Adding the annual term removes
   the bias but triples the scatter (6.7 mm RMS), because four parameters are fitted across a
   gap with no data. Neither fit leaves a residual that flags the problem.
3. **γ has no effect until it starts to smooth.** Results are identical from 1e-4 to 0.1: the
   rows only set the null-space component, as Doin intends. From γ ≥ 1 they also change the
   connected segment (pre-gap shape 2.94 → 0.56 mm in S1, where truth is linear; 2.94 → 3.69 mm
   in S2, where it isn't), so they are imposing the model, not bridging the gap.
4. **A second-derivative penalty applied directly to φ, as #111's sketch proposes, is the worst
   option tested** (16–24 mm offset RMS). Across a data-free gap it extrapolates each segment's
   end curvature, and the noise at the edge dates dominates. Doin applies the Laplacian to a
   separate smoothed series `φs`, not to `φ`, and this test does not cover that form.
5. **The current code fails messily.** See below.

## Bugs found (separate from #111)

1. **L1 panics on a disconnected network.** `crates/dolphin-timeseries/src/inversion.rs:363`,
   `.expect("AtA not SPD (rank-deficient network)")`, reproduced:
   `CholeskyError { non_positive_definite_minor: 37 }`. L1 is dolphin's default method, and
   the panic happens after unwrapping, the most expensive stage.
2. **Weighted L2 publishes arbitrary offsets on a disconnected network.**
   `solve_pixel_with_covariance` (`inversion.rs:459`) trusts any Cholesky that succeeds. With
   non-integer precisions, rounding left a tiny positive pivot on an exactly singular normal
   matrix for 979 of 2000 pixels. Those pixels return finite series with post-gap values up to
   244 mm (median |value| 8.4 mm), mixed with NaN pixels, and nothing marks them. The
   uniform-weight path returns NaN everywhere (Cholesky fails exactly).
   `fixed_l2_pixel_map` on the same pixel returns `Ok` with `condition_number = inf`
   (`inversion.rs:554, 596`). Its only consumer gates on finite condition number
   (`spatial_covariance.rs:558`), so that half is caught downstream.
3. **No connectivity check before unwrapping.** `finish_displacement` checks only `!pairs.is_empty()`
   (`displacement.rs:517`). A date that no pair covers is silently dropped by
   `get_incidence_matrix` (`inversion.rs:229-243`). Reproduced: 6 dates, `max_temporal_baseline
   = 48`, pairs `[(0,1),(0,2),(1,2),(4,5)]` give `A` with 4 columns, not 5. The run then fails
   at the velocity fit with "velocity dates do not match the displacement series"
   (`displacement.rs:1302`), after unwrap and inversion.

All three are fixed by the same small change, outside #111: at the `network` stage, reject a
pair set whose graph over `0..n_dates` is not a single connected component that covers every
date. The error message should name the gap dates and suggest `max_bandwidth`. MintPy's
weighted path does the same thing (raises on a not-fully-connected network). This diverges from
dolphin, which returns min-norm or NaN, but only by turning a silent wrong answer into an
error, which the repo already does elsewhere (fail-closed statuses in `PixelL2MapStatus`).

## Conclusion: DEFER

**Reason.** No current caller can build a gap-disconnected network: production and the GNSS
harness use index-based networks, and per-pixel masks are spatial only. On the synthetic case,
index densification gave lower error with fewer interferograms. NSBAS would add a temporal
model bias (−7.1 mm here) that the gate would have to accept knowingly. #111 stays deferred with
its gate unchanged, and its "second-derivative" design sketch should be dropped in favour of the
parametric form if it is ever reopened.

**Evidence that would reopen it:** a real stack (GroundPulse or the GNSS harness) where
interferograms bridging an acquisition gap of ≥ 2 missed cycles unwrap with conncomp-0 or
loop-closure failure over most of the AOI at `max_bandwidth ≥ 2`, so densifying does not
connect the network. Even then, adopt only if NSBAS's gap-offset error against GNSS beats a
"break the series at the gap and report two segments" baseline. If reopened, the starting
design is LiCSBAS's form with an optional annual term, config-gated and off by default, with
the rows added only for pixels or networks that are actually rank-deficient. Connected
networks must stay bit-identical to today.
