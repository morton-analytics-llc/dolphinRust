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
use dolphin_phaselink::{interpolate, INTERPOLATION_ALPHA, INTERPOLATION_NUM_NEIGHBORS};
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
    ) -> Result<UnwrapNetworkOutput>;
}

/// Network-aligned unwrap products. Both cubes follow `pairs` order.
pub struct UnwrapNetworkOutput {
    /// Unwrapped phase, one band per interferogram.
    pub unwrapped: Array3<f64>,
    /// Connected-component labels in the same band order.
    pub connected_components: Array3<u32>,
}

/// Single-pass SNAPHU (the default backend).
pub struct SnaphuBackend(pub UnwrapConfig);

/// tophu coarse→fine multi-scale over the SNAPHU per-tile solver.
pub struct TophuBackend(pub TophuConfig);

/// Clean-room in-process native unwrapper (MCF branch cuts). No subprocess and
/// no scratch round-trip: each ifg is unwrapped from in-memory arrays, so the
/// per-pair `par_iter` parallelizes with neither a fork nor flat-binary I/O.
pub struct NativeUnwrapBackend(pub NativeConfig);

/// Pre-unwrap interpolation of low-quality wrapped phase, as dolphin v0.35.0's
/// `unwrap/_unwrap.py` runs it when `unwrap_options.run_interpolation` is set.
pub(crate) struct InterpolationPlan<'a> {
    /// Phase similarity deciding the coherent-pixel mask. `None` falls back to
    /// each interferogram's own correlation — dolphin's similarity test
    /// *replaces* the correlation test rather than combining with it.
    pub similarity: Option<ArrayView2<'a, f64>>,
    /// Pixels below this (similarity, else correlation) are interpolated.
    pub cutoff: f64,
    /// Neighbour search radius in pixels.
    pub max_radius: usize,
}

/// Network unwrap products plus, when interpolation ran, the per-pixel number of
/// interferograms whose wrapped phase was replaced before unwrapping.
pub(crate) struct CorrelatedNetworkUnwrap {
    pub output: UnwrapNetworkOutput,
    pub interpolated_count: Option<Array2<u32>>,
}

pub(crate) fn unwrap_phase_correlated_network(
    backend: &dyn UnwrapBackend,
    pl: ArrayView3<Cf64>,
    pairs: &[(usize, usize)],
    support: ArrayView2<bool>,
    interpolation: Option<&InterpolationPlan>,
    scratch: &Path,
) -> Result<CorrelatedNetworkUnwrap> {
    let mut layers = Vec::with_capacity(pairs.len());
    let mut interpolated_count = interpolation.map(|_| Array2::<u32>::zeros(support.dim()));
    let batch_size = rayon::current_num_threads().max(1);
    for (batch_index, batch) in pairs.chunks(batch_size).enumerate() {
        let batch_layers = batch
            .par_iter()
            .enumerate()
            .map(|(index, &pair)| {
                let ifg = form_ifg(pl, pair);
                let correlation = interferogram_correlation(ifg.view(), support);
                let index = batch_index * batch_size + index;
                let pair_scratch = scratch.join(format!("ifg_{index:04}"));
                std::fs::create_dir_all(&pair_scratch)?;
                let Some(plan) = interpolation else {
                    let output =
                        backend.unwrap_network(pl, &[pair], correlation.view(), &pair_scratch)?;
                    return Ok((first_layer(output), None));
                };
                unwrap_interpolated(backend, ifg.view(), correlation.view(), plan, &pair_scratch)
            })
            .collect::<Result<Vec<_>>>()?;
        for (layer, replaced) in batch_layers {
            if let (Some(count), Some(replaced)) = (interpolated_count.as_mut(), replaced) {
                ndarray::Zip::from(count)
                    .and(&replaced)
                    .for_each(|count, &replaced| *count += u32::from(replaced));
            }
            layers.push(layer);
        }
    }
    Ok(CorrelatedNetworkUnwrap {
        output: stack_layers(layers)?,
        interpolated_count,
    })
}

type UnwrapLayer = (Array2<f64>, Array2<u32>);

fn first_layer(output: UnwrapNetworkOutput) -> UnwrapLayer {
    (
        output.unwrapped.index_axis_move(Axis(0), 0),
        output.connected_components.index_axis_move(Axis(0), 0),
    )
}

/// dolphin's interpolate → unwrap → `transfer_ambiguities` sequence for one ifg.
/// The interpolated phase reaches only the unwrapper, with the ifg's original
/// correlation (dolphin v0.35.0 passes the original `corr_filename`); every
/// pixel's output is its own wrapped phase plus the solved whole-cycle ambiguity.
/// Replaced pixels lose their connected-component label: their ambiguity came
/// from neighbours, so the interpolation never adds unwrap support (and a
/// phase-driven solver such as SNAPHU would otherwise label the smoothed gap).
/// This stands in for dolphin's post-transfer SNAPHU component regrow, which is
/// not ported. Also returns which pixels were replaced.
fn unwrap_interpolated(
    backend: &dyn UnwrapBackend,
    ifg: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    plan: &InterpolationPlan,
    scratch: &Path,
) -> Result<(UnwrapLayer, Option<Array2<bool>>)> {
    // dolphin compares the float32 rasters against the cutoff in float32.
    let cutoff = plan.cutoff as f32;
    let weights = match plan.similarity {
        Some(similarity) => similarity.mapv(|value| value as f32 >= cutoff),
        None => correlation.mapv(|value| value >= cutoff),
    }
    .mapv(|coherent| f32::from(u8::from(coherent)));
    let interpolated = interpolate(
        ifg,
        weights.view(),
        cutoff,
        INTERPOLATION_NUM_NEIGHBORS,
        plan.max_radius,
        INTERPOLATION_ALPHA,
    );
    let replaced = ndarray::Zip::from(&ifg)
        .and(&weights)
        .map_collect(|&value, &weight| value != Cf32::new(0.0, 0.0) && weight < cutoff);
    let output = backend.unwrap_network(
        single_pair_stack(interpolated.view()).view(),
        &[(0, 1)],
        correlation,
        scratch,
    )?;
    let (estimate, mut components) = first_layer(output);
    ndarray::Zip::from(&mut components)
        .and(&replaced)
        .for_each(|label, &replaced| {
            if replaced {
                *label = 0;
            }
        });
    Ok((
        (transfer_ambiguities(ifg, estimate.view()), components),
        Some(replaced),
    ))
}

/// A two-date linked-phase stack whose single pair `(0, 1)` forms exactly `ifg`,
/// so a prepared interferogram can go through the network-level backend trait.
fn single_pair_stack(ifg: ArrayView2<Cf32>) -> Array3<Cf64> {
    let (rows, cols) = ifg.dim();
    Array3::from_shape_fn((2, rows, cols), |(date, row, col)| match date {
        0 => {
            let value = ifg[(row, col)];
            Cf64::new(f64::from(value.re), f64::from(value.im))
        }
        _ => Cf64::new(1.0, 0.0),
    })
}

/// dolphin `transfer_ambiguities`: round the unwrapped estimate's offset from the
/// original wrapped phase to whole cycles (numpy rounds half to even) and add
/// them to the wrapped phase. Nodata pixels keep the solver's value.
fn transfer_ambiguities(ifg: ArrayView2<Cf32>, estimate: ArrayView2<f64>) -> Array2<f64> {
    ndarray::Zip::from(&ifg)
        .and(&estimate)
        .map_collect(|&value, &unwrapped| {
            if value == Cf32::new(0.0, 0.0) {
                return unwrapped;
            }
            let wrapped = f64::from(value.im.atan2(value.re));
            let cycles = ((unwrapped - wrapped) / std::f64::consts::TAU).round_ties_even();
            wrapped + std::f64::consts::TAU * cycles
        })
}

/// Upstream workflow phase correlation: Gaussian sigma 11/3, truncated at four
/// sigma, with nearest boundary extension. Missing samples have zero weight;
/// correlation never supplies structural support.
fn interferogram_correlation(ifg: ArrayView2<Cf32>, support: ArrayView2<bool>) -> Array2<f32> {
    assert_eq!(ifg.dim(), support.dim());
    let mut real = Array2::zeros(ifg.dim());
    let mut imaginary = Array2::zeros(ifg.dim());
    let mut weights = Array2::zeros(ifg.dim());
    for (point, &value) in ifg.indexed_iter() {
        let value = Cf64::new(f64::from(value.re), f64::from(value.im));
        if support[point] && value.re.is_finite() && value.im.is_finite() && value.norm_sqr() > 0.0
        {
            let unit = value / value.norm();
            real[point] = unit.re;
            imaginary[point] = unit.im;
            weights[point] = 1.0;
        }
    }
    let sigma = 11.0 / 3.0;
    let mut kernel: Vec<f64> = (-15..=15)
        .map(|offset| (-f64::from(offset * offset) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f64 = kernel.iter().sum();
    kernel.iter_mut().for_each(|weight| *weight /= total);
    let filtered_real = gaussian_nearest(real.view(), &kernel);
    let filtered_imaginary = gaussian_nearest(imaginary.view(), &kernel);
    let filtered_weights = gaussian_nearest(weights.view(), &kernel);
    Array2::from_shape_fn(ifg.dim(), |point| {
        if weights[point] == 0.0 || filtered_weights[point] == 0.0 {
            0.0
        } else {
            (filtered_real[point].hypot(filtered_imaginary[point]) / filtered_weights[point])
                .clamp(0.0, 1.0) as f32
        }
    })
}

fn gaussian_nearest(values: ArrayView2<f64>, kernel: &[f64]) -> Array2<f64> {
    let (rows, cols) = values.dim();
    let vertical = Array2::from_shape_fn((rows, cols), |(row, col)| {
        kernel
            .iter()
            .zip(-15..=15)
            .map(|(&weight, delta)| {
                weight * values[(row.saturating_add_signed(delta).min(rows - 1), col)]
            })
            .sum::<f64>()
    });
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        kernel
            .iter()
            .zip(-15..=15)
            .map(|(&weight, delta)| {
                weight * vertical[(row, col.saturating_add_signed(delta).min(cols - 1))]
            })
            .sum()
    })
}

impl UnwrapBackend for SnaphuBackend {
    fn unwrap_network(
        &self,
        pl: ArrayView3<Cf64>,
        pairs: &[(usize, usize)],
        correlation: ArrayView2<f32>,
        scratch: &Path,
    ) -> Result<UnwrapNetworkOutput> {
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
                let out = unwrap_with_corr(ifg, &corr_path, &self.0, pair_scratch)?;
                Ok((out.unwrapped.mapv(f64::from), out.conncomp))
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
    ) -> Result<UnwrapNetworkOutput> {
        unwrap_each_ifg(
            pl,
            pairs,
            correlation,
            scratch,
            |ifg, corr, pair_scratch| {
                let out = unwrap_multiscale(ifg, corr, &self.0, pair_scratch)?;
                Ok((out.unwrapped.mapv(f64::from), out.conncomp))
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
    ) -> Result<UnwrapNetworkOutput> {
        // In-process: form each ifg and unwrap from memory — no scratch dirs,
        // no subprocess. `par_iter().collect()` keeps the stack in `pairs` order.
        let layers = pairs
            .par_iter()
            .map(|&pair| solve_native(pl, pair, correlation, &self.0))
            .collect::<Result<Vec<_>>>()?;
        stack_layers(layers)
    }
}

/// Form one ifg from the linked phase and unwrap it with the native solver.
fn solve_native(
    pl: ArrayView3<Cf64>,
    pair: (usize, usize),
    correlation: ArrayView2<f32>,
    cfg: &NativeConfig,
) -> Result<(Array2<f64>, Array2<u32>)> {
    let ifg = form_ifg(pl, pair);
    let out = unwrap_native(ifg.view(), correlation, cfg).context("native unwrap")?;
    Ok((out.unwrapped.mapv(f64::from), out.conncomp))
}

/// Form each ifg from the linked phase and unwrap it with a 2D solver, stacking
/// the results in `pairs` order. Shared by the 2D backends.
fn unwrap_each_ifg(
    pl: ArrayView3<Cf64>,
    pairs: &[(usize, usize)],
    correlation: ArrayView2<f32>,
    scratch: &Path,
    solve: impl Fn(ArrayView2<Cf32>, ArrayView2<f32>, &Path) -> Result<(Array2<f64>, Array2<u32>)>
        + Sync,
) -> Result<UnwrapNetworkOutput> {
    // Solve pairs concurrently; `par_iter().collect()` is order-stable, so the
    // stack matches `pairs` order regardless of completion order. Each pair gets
    // its own scratch subdir so the fixed-name SNAPHU files never collide.
    let layers = pairs
        .par_iter()
        .enumerate()
        .map(|(idx, &pair)| unwrap_one_pair(pl, pair, correlation, scratch, idx, &solve))
        .collect::<Result<Vec<_>>>()?;
    stack_layers(layers)
}

fn stack_layers(layers: Vec<(Array2<f64>, Array2<u32>)>) -> Result<UnwrapNetworkOutput> {
    let phase_views: Vec<_> = layers.iter().map(|layer| layer.0.view()).collect();
    let component_views: Vec<_> = layers.iter().map(|layer| layer.1.view()).collect();
    Ok(UnwrapNetworkOutput {
        unwrapped: ndarray::stack(Axis(0), &phase_views).context("stacking unwrapped ifgs")?,
        connected_components: ndarray::stack(Axis(0), &component_views)
            .context("stacking connected components")?,
    })
}

/// Unwrap a single pair into its own scratch subdir `pair_NNNN`, isolating the
/// fixed-name SNAPHU scratch files so pairs can be solved in parallel.
fn unwrap_one_pair(
    pl: ArrayView3<Cf64>,
    pair: (usize, usize),
    correlation: ArrayView2<f32>,
    scratch: &Path,
    idx: usize,
    solve: &(impl Fn(ArrayView2<Cf32>, ArrayView2<f32>, &Path) -> Result<(Array2<f64>, Array2<u32>)>
          + Sync),
) -> Result<(Array2<f64>, Array2<u32>)> {
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
        if z.norm_sqr() == 0.0 {
            Cf32::new(0.0, 0.0)
        } else {
            Cf32::from_polar(1.0, z.arg() as f32)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_correlation_matches_independent_gaussian_fixtures() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/ifg_correlation.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let shape = (
                case["shape"][0].as_u64().unwrap() as usize,
                case["shape"][1].as_u64().unwrap() as usize,
            );
            let real = case["ifg_re"].as_array().unwrap();
            let imaginary = case["ifg_im"].as_array().unwrap();
            let ifg = Array2::from_shape_vec(
                shape,
                real.iter()
                    .zip(imaginary)
                    .map(|(r, i)| {
                        Cf32::new(
                            r.as_f64().unwrap_or(f64::NAN) as f32,
                            i.as_f64().unwrap_or(f64::NAN) as f32,
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let support = Array2::from_shape_vec(
                shape,
                case["support"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_bool().unwrap())
                    .collect(),
            )
            .unwrap();
            let actual = interferogram_correlation(ifg.view(), support.view());
            for (index, (value, expected)) in actual
                .iter()
                .zip(case["expected"].as_array().unwrap())
                .enumerate()
            {
                assert!(
                    (f64::from(*value) - expected.as_f64().unwrap()).abs() < 1e-6,
                    "case {} pixel {index}: {value} != {expected}",
                    case["name"]
                );
            }
        }
    }

    #[test]
    fn missing_phase_never_contributes_a_unit_phasor() {
        let mut ifg = Array2::from_elem((9, 11), Cf32::new(0.0, 2.0));
        let mut support = Array2::from_elem(ifg.dim(), true);
        ifg[(4, 4)] = Cf32::new(f32::NAN, f32::NAN);
        ifg[(4, 5)] = Cf32::new(0.0, 0.0);
        ifg[(4, 6)] = Cf32::new(1.0, 0.0);
        support[(4, 6)] = false;
        let actual = interferogram_correlation(ifg.view(), support.view());
        for (point, &value) in actual.indexed_iter() {
            if point.0 == 4 && (4..=6).contains(&point.1) {
                assert_eq!(value, 0.0);
            } else {
                assert!((value - 1.0).abs() < 1e-6);
            }
        }
        support.fill(false);
        assert!(interferogram_correlation(ifg.view(), support.view())
            .iter()
            .all(|&value| value == 0.0));
    }

    /// Returns each received ifg's wrapped phase plus one cycle, and records the
    /// phase it was handed at `PROBE`.
    struct PhasePlusCycle(std::sync::Mutex<Vec<f64>>);

    const PROBE: (usize, usize) = (4, 4);

    impl UnwrapBackend for PhasePlusCycle {
        fn unwrap_network(
            &self,
            pl: ArrayView3<Cf64>,
            pairs: &[(usize, usize)],
            _correlation: ArrayView2<f32>,
            _scratch: &Path,
        ) -> Result<UnwrapNetworkOutput> {
            let ifg = form_ifg(pl, pairs[0]);
            self.0.lock().unwrap().push(f64::from(ifg[PROBE].arg()));
            let unwrapped = ifg.mapv(|z| f64::from(z.arg()) + std::f64::consts::TAU);
            Ok(UnwrapNetworkOutput {
                unwrapped: unwrapped.insert_axis(Axis(0)),
                connected_components: Array3::from_elem((1, ifg.nrows(), ifg.ncols()), 1),
            })
        }
    }

    /// A smooth 0.9 rad field with a decorrelated probe pixel at +2.0 rad and a
    /// nodata pixel, both below the similarity cutoff.
    fn probe_scene() -> (Array3<Cf64>, Array2<f64>) {
        let mut pl = Array3::from_elem((2, 9, 9), Cf64::new(1.0, 0.0));
        pl.index_axis_mut(Axis(0), 0)
            .fill(Cf64::from_polar(1.0, 0.9));
        pl[(0, PROBE.0, PROBE.1)] = Cf64::from_polar(1.0, 0.9 + 2.0);
        pl[(0, 0, 0)] = Cf64::new(0.0, 0.0);
        let mut similarity = Array2::from_elem((9, 9), 0.8);
        similarity[PROBE] = 0.1;
        similarity[(0, 0)] = 0.1;
        (pl, similarity)
    }

    fn run_probe(
        pl: &Array3<Cf64>,
        interpolation: Option<&InterpolationPlan>,
    ) -> (CorrelatedNetworkUnwrap, Vec<f64>) {
        let backend = PhasePlusCycle(std::sync::Mutex::new(Vec::new()));
        let scratch = std::env::temp_dir().join(format!(
            "dolphin_interpolation_probe_{}_{}",
            std::process::id(),
            interpolation.map_or(0.0, |plan| plan.cutoff)
        ));
        let support = Array2::from_elem((9, 9), true);
        let output = unwrap_phase_correlated_network(
            &backend,
            pl.view(),
            &[(0, 1)],
            support.view(),
            interpolation,
            &scratch,
        )
        .unwrap();
        std::fs::remove_dir_all(scratch).unwrap();
        (output, backend.0.into_inner().unwrap())
    }

    /// The unwrapper sees the neighbour phase at a decorrelated pixel; the output
    /// there is the pixel's own wrapped phase plus the transferred cycle, it
    /// carries no component label, and the provenance count marks it (and not
    /// the nodata pixel).
    #[test]
    fn interpolation_feeds_unwrapper_and_transfers_ambiguities_to_original_phase() {
        let (pl, similarity) = probe_scene();
        let plan = InterpolationPlan {
            similarity: Some(similarity.view()),
            cutoff: 0.3,
            max_radius: 51,
        };
        let (output, seen) = run_probe(&pl, Some(&plan));
        assert!((seen[0] - 0.9).abs() < 1e-6, "unwrapper saw {}", seen[0]);
        let own = f64::from(Cf32::from_polar(1.0, (0.9 + 2.0_f64) as f32).arg());
        let probe = output.output.unwrapped[(0, PROBE.0, PROBE.1)];
        assert!(
            (probe - (own + std::f64::consts::TAU)).abs() < 1e-6,
            "{probe}"
        );
        assert_eq!(output.output.connected_components[(0, PROBE.0, PROBE.1)], 0);
        assert_eq!(output.output.connected_components[(0, 2, 2)], 1);
        let count = output.interpolated_count.unwrap();
        assert_eq!(count[PROBE], 1);
        assert_eq!(count.sum(), 1, "only the valid low-similarity pixel counts");
    }

    /// Interpolation that replaces nothing (cutoff 0) reproduces the
    /// interpolation-off output bit for bit, and off records no provenance.
    #[test]
    fn interpolation_off_leaves_network_output_unchanged() {
        let (pl, similarity) = probe_scene();
        let (off, seen_off) = run_probe(&pl, None);
        assert!(off.interpolated_count.is_none());
        assert!((seen_off[0] - (0.9 + 2.0)).abs() < 1e-6);
        let plan = InterpolationPlan {
            similarity: Some(similarity.view()),
            cutoff: 0.0,
            max_radius: 51,
        };
        let (noop, _) = run_probe(&pl, Some(&plan));
        assert_eq!(noop.output.unwrapped, off.output.unwrapped);
        assert_eq!(
            noop.output.connected_components,
            off.output.connected_components
        );
        assert_eq!(noop.interpolated_count.unwrap().sum(), 0);
    }

    struct EchoCorrelation;

    impl UnwrapBackend for EchoCorrelation {
        fn unwrap_network(
            &self,
            _pl: ArrayView3<Cf64>,
            pairs: &[(usize, usize)],
            correlation: ArrayView2<f32>,
            scratch: &Path,
        ) -> Result<UnwrapNetworkOutput> {
            assert_eq!(pairs.len(), 1);
            assert!(scratch.is_dir());
            Ok(UnwrapNetworkOutput {
                unwrapped: correlation.mapv(f64::from).insert_axis(Axis(0)),
                connected_components: Array3::from_elem(
                    (1, correlation.nrows(), correlation.ncols()),
                    1,
                ),
            })
        }
    }

    #[test]
    fn workflow_dispatches_pair_specific_correlation_in_network_order() {
        let pl = Array3::from_shape_fn((3, 33, 35), |(date, row, col)| {
            let phase = if date == 2 {
                0.7 * row as f64 + 0.8 * col as f64
            } else {
                date as f64 * 0.4
            };
            Cf64::from_polar(1.0, phase)
        });
        let support = Array2::from_elem((33, 35), true);
        let pairs = [(0, 2), (0, 1), (1, 2)];
        let scratch =
            std::env::temp_dir().join(format!("dolphin_pair_correlation_{}", std::process::id()));
        let output = unwrap_phase_correlated_network(
            &EchoCorrelation,
            pl.view(),
            &pairs,
            support.view(),
            None,
            &scratch,
        )
        .unwrap()
        .output;
        assert!(output.unwrapped[(0, 16, 17)] < 0.01);
        assert!((output.unwrapped[(1, 16, 17)] - 1.0).abs() < 1e-6);
        assert!(output.unwrapped[(2, 16, 17)] < 0.01);
        assert!(support.iter().all(|&valid| valid));
        let mut missing = pl.clone();
        missing[(0, 5, 5)] = Cf64::new(0.0, 0.0);
        missing[(2, 7, 7)] = Cf64::new(f64::NAN, f64::NAN);
        let output = unwrap_phase_correlated_network(
            &EchoCorrelation,
            missing.view(),
            &pairs,
            support.view(),
            None,
            &scratch,
        )
        .unwrap()
        .output;
        assert_eq!(output.unwrapped[(0, 5, 5)], 0.0);
        assert_eq!(output.unwrapped[(1, 5, 5)], 0.0);
        assert_eq!(output.unwrapped[(0, 7, 7)], 0.0);
        assert_eq!(output.unwrapped[(2, 7, 7)], 0.0);
        std::fs::remove_dir_all(scratch).unwrap();
    }
}
