//! Pluggable phase-unwrapping backend for the interferogram network.
//!
//! The pipeline dispatches unwrapping through the [`UnwrapBackend`] trait. Its
//! signature is **network-level** — it receives the linked phase history and the
//! date pairs, not pre-formed independent 2D interferograms — so a future
//! spurt-style **3D spatiotemporal** solver can implement the same trait and
//! unwrap the whole stack jointly without any pipeline change. The two shipped
//! backends ([`SnaphuBackend`], [`TophuBackend`]) are 2D: they form each ifg and
//! unwrap it independently, exactly as before, so their output is unchanged.

use std::path::Path;

use anyhow::{Context, Result};
use dolphin_core::{Cf32, Cf64};
use dolphin_unwrap::native::{unwrap_native, NativeConfig};
use dolphin_unwrap::{
    unwrap_multiscale, unwrap_with_corr, write_correlation, TophuConfig, UnwrapConfig,
};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;

/// A phase-unwrapping backend for the interferogram network: maps the linked
/// phase `pl` `(n_dates, rows, cols)` and the `(i, j)` date `pairs` to the
/// unwrapped phase per ifg `(n_pairs, rows, cols)` in radians.
///
/// 2D backends unwrap each ifg independently; a 3D backend may use the full
/// spatiotemporal structure of `pl` + `pairs`. Implement this trait to add a
/// backend — no other pipeline code changes.
pub trait UnwrapBackend: Send + Sync {
    /// Unwrap the whole network, returning `(n_pairs, rows, cols)` radians.
    ///
    /// # Errors
    /// Backend-specific (solver failure, scratch I/O, stacking).
    fn unwrap_network(
        &self,
        pl: ArrayView3<Cf64>,
        pairs: &[(usize, usize)],
        correlation: ArrayView2<f32>,
        scratch: &Path,
    ) -> Result<Array3<f64>>;
}

/// Single-pass SNAPHU (the default backend).
pub struct SnaphuBackend(pub UnwrapConfig);

/// tophu coarse→fine multi-scale over the SNAPHU per-tile solver.
pub struct TophuBackend(pub TophuConfig);

/// Clean-room in-process native unwrapper (MCF branch cuts). No subprocess and
/// no scratch round-trip: each ifg is unwrapped from in-memory arrays, so the
/// per-pair `par_iter` parallelizes with neither a fork nor flat-binary I/O.
pub struct NativeUnwrapBackend(pub NativeConfig);

impl UnwrapBackend for SnaphuBackend {
    fn unwrap_network(
        &self,
        pl: ArrayView3<Cf64>,
        pairs: &[(usize, usize)],
        correlation: ArrayView2<f32>,
        scratch: &Path,
    ) -> Result<Array3<f64>> {
        // #3: the correlation is identical across every pair — serialize it once
        // into the shared scratch and reuse the file for all ifgs instead of
        // re-writing corr.f4 per pair.
        let corr_path = write_correlation(scratch, correlation)?;
        unwrap_each_ifg(
            pl,
            pairs,
            correlation,
            scratch,
            |ifg, _corr, pair_scratch| {
                Ok(unwrap_with_corr(ifg, &corr_path, &self.0, pair_scratch)?
                    .unwrapped
                    .mapv(f64::from))
            },
        )
    }
}

impl UnwrapBackend for TophuBackend {
    fn unwrap_network(
        &self,
        pl: ArrayView3<Cf64>,
        pairs: &[(usize, usize)],
        correlation: ArrayView2<f32>,
        scratch: &Path,
    ) -> Result<Array3<f64>> {
        unwrap_each_ifg(
            pl,
            pairs,
            correlation,
            scratch,
            |ifg, corr, pair_scratch| {
                Ok(unwrap_multiscale(ifg, corr, &self.0, pair_scratch)?
                    .unwrapped
                    .mapv(f64::from))
            },
        )
    }
}

impl UnwrapBackend for NativeUnwrapBackend {
    fn unwrap_network(
        &self,
        pl: ArrayView3<Cf64>,
        pairs: &[(usize, usize)],
        correlation: ArrayView2<f32>,
        _scratch: &Path,
    ) -> Result<Array3<f64>> {
        // In-process: form each ifg and unwrap from memory — no scratch dirs,
        // no subprocess. `par_iter().collect()` keeps the stack in `pairs` order.
        let layers = pairs
            .par_iter()
            .map(|&pair| solve_native(pl, pair, correlation, &self.0))
            .collect::<Result<Vec<_>>>()?;
        let views: Vec<_> = layers.iter().map(Array2::view).collect();
        ndarray::stack(Axis(0), &views).context("stacking unwrapped ifgs")
    }
}

/// Form one ifg from the linked phase and unwrap it with the native solver.
fn solve_native(
    pl: ArrayView3<Cf64>,
    pair: (usize, usize),
    correlation: ArrayView2<f32>,
    cfg: &NativeConfig,
) -> Result<Array2<f64>> {
    let ifg = form_ifg(pl, pair);
    let out = unwrap_native(ifg.view(), correlation, cfg).context("native unwrap")?;
    Ok(out.unwrapped.mapv(f64::from))
}

/// Form each ifg from the linked phase and unwrap it with a 2D solver, stacking
/// the results in `pairs` order. Shared by the 2D backends.
fn unwrap_each_ifg(
    pl: ArrayView3<Cf64>,
    pairs: &[(usize, usize)],
    correlation: ArrayView2<f32>,
    scratch: &Path,
    solve: impl Fn(ArrayView2<Cf32>, ArrayView2<f32>, &Path) -> Result<Array2<f64>> + Sync,
) -> Result<Array3<f64>> {
    // Solve pairs concurrently; `par_iter().collect()` is order-stable, so the
    // stack matches `pairs` order regardless of completion order. Each pair gets
    // its own scratch subdir so the fixed-name SNAPHU files never collide.
    let layers = pairs
        .par_iter()
        .enumerate()
        .map(|(idx, &pair)| unwrap_one_pair(pl, pair, correlation, scratch, idx, &solve))
        .collect::<Result<Vec<_>>>()?;
    let views: Vec<_> = layers.iter().map(Array2::view).collect();
    ndarray::stack(Axis(0), &views).context("stacking unwrapped ifgs")
}

/// Unwrap a single pair into its own scratch subdir `pair_NNNN`, isolating the
/// fixed-name SNAPHU scratch files so pairs can be solved in parallel.
fn unwrap_one_pair(
    pl: ArrayView3<Cf64>,
    pair: (usize, usize),
    correlation: ArrayView2<f32>,
    scratch: &Path,
    idx: usize,
    solve: &(impl Fn(ArrayView2<Cf32>, ArrayView2<f32>, &Path) -> Result<Array2<f64>> + Sync),
) -> Result<Array2<f64>> {
    let pair_scratch = scratch.join(format!("pair_{idx:04}"));
    std::fs::create_dir_all(&pair_scratch)?;
    solve(form_ifg(pl, pair).view(), correlation, &pair_scratch)
}

/// Form the wrapped ifg `(i, j)` as `exp(j∠(pl_i · conj(pl_j)))` — dolphin's
/// production convention `ref · conj(sec)` (`interferogram.py`, `_create_vrt_conj`):
/// for the single-reference network `i` is the reference/earlier date and `j` the
/// secondary/later one. The opposite order globally inverts the displacement sign
/// (guarded by `tests/sign_convention.rs`); keep `pl_i · conj(pl_j)`.
fn form_ifg(pl: ArrayView3<Cf64>, (i, j): (usize, usize)) -> Array2<Cf32> {
    let (_, rows, cols) = pl.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let z = pl[(i, r, c)] * pl[(j, r, c)].conj();
        Cf32::from_polar(1.0, z.arg() as f32)
    })
}
