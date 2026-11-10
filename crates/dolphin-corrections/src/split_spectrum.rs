//! Split-spectrum ionosphere estimation from two NISAR range bands.
//!
//! Main/side-band method (Gomba et al. 2016; Fattahi et al. 2017; Liang et al.
//! 2019): NISAR acquires a wide main band (`frequencyA`, e.g. 40 MHz) and a
//! narrow side band (`frequencyB`, 5 MHz) at a different carrier. Each band's
//! unwrapped interferometric phase is modelled as a non-dispersive term
//! (troposphere + deformation + geometry, `∝ f`) plus a dispersive ionospheric
//! term (`∝ 1/f`):
//!
//! ```text
//! φ(f) = a·f + b/f
//! ```
//!
//! Solving the two-band system for the dispersive term at the reference
//! frequency `f₀` gives
//!
//! ```text
//! φ_iono(f₀) = b/f₀ = f_A·f_B / (f₀·(f_A² − f_B²)) · (φ_B·f_A − φ_A·f_B)
//! ```
//!
//! **Reference frequency.** Here `f₀ = f_A`, the main-band carrier: the
//! estimate is the ionospheric phase *in the frequencyA interferogram*, so
//! `φ_A − φ_iono` is the non-dispersive frequencyA phase. With `r = f_B/f_A`
//! the formula reduces to `φ_iono = r·(φ_B − r·φ_A)/(1 − r²)`, which is what is
//! evaluated (it avoids forming `f²` products of ~1e18).
//!
//! **Sign convention.** Linear in the inputs, so `φ_iono` carries the sign of
//! the input phases. Both bands must be formed `ref · conj(sec)` — the pipeline
//! convention pinned by `dolphin-workflows/tests/sign_convention.rs` — and the
//! result is in that same convention, in radians.
//!
//! **Unwrapping errors.** A `k·2π` slip in `φ_B` adds the constant
//! `k · 2π·r/(1 − r²)` ([`side_band_cycle_offset`]) to `φ_iono` over the
//! affected connected component. For NISAR-like carriers (`r ≈ 1.03`) that is
//! ≈ −111 rad per cycle, so the two bands must be unwrapped consistently and the
//! estimate low-pass filtered within each connected component
//! ([`lowpass_ionosphere`]) before use.
//!
//! **Inter-band cycle alignment.** Unwrapping the two bands independently leaves
//! each connected component with its own integer offset between them.
//! [`align_side_band_cycles`] removes it before estimation: per component `c`,
//! `k_c = round(median((φ_B − r·φ_A) / 2π))` over pixels where both bands are
//! finite, and `φ_B ← φ_B − 2π·k_c`. This fixes the side band to the main
//! band's cycles only; the main band's own per-component offset against ground
//! truth remains and must be anchored downstream.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use dolphin_filtering::masked_gaussian_lowpass;
use ndarray::{s, Array2, ArrayView2, Zip};
use thiserror::Error;

/// Split-spectrum failures.
#[derive(Debug, Error)]
pub enum SplitSpectrumError {
    /// The side-band (`frequencyB`) phase was not supplied. A main-band-only
    /// estimate is not possible, so the output is refused rather than faked.
    #[error(
        "split-spectrum ionosphere needs the frequencyB (side-band) unwrapped phase; \
         the product has no side band, so the estimate is refused"
    )]
    MissingSideBand,
    /// The two bands are on different grids.
    #[error(
        "main-band phase shape {main:?} != side-band shape {side:?}; resample \
         frequencyA onto the frequencyB grid first"
    )]
    ShapeMismatch {
        /// Main-band `(rows, cols)`.
        main: (usize, usize),
        /// Side-band `(rows, cols)`.
        side: (usize, usize),
    },
    /// The connected-component labels are on a different grid from the phase.
    #[error("component label shape {labels:?} != phase shape {phase:?}")]
    LabelShapeMismatch {
        /// Label `(rows, cols)`.
        labels: (usize, usize),
        /// Phase `(rows, cols)`.
        phase: (usize, usize),
    },
    /// A center frequency is non-finite, non-positive, or the two are equal
    /// (the two-band system is singular).
    #[error(
        "center frequencies must be finite, positive and distinct: \
         main {main_hz} Hz, side {side_hz} Hz"
    )]
    InvalidFrequencies {
        /// Main-band center frequency (Hz).
        main_hz: f64,
        /// Side-band center frequency (Hz).
        side_hz: f64,
    },
    /// The `dolphin-filtering` low-pass rejected its parameters.
    #[error("low-pass filter: {0}")]
    Filter(&'static str),
}

/// One band's unwrapped interferometric phase (radians, `ref · conj(sec)`) and
/// its center frequency.
#[derive(Debug, Clone, Copy)]
pub struct BandPhase<'a> {
    /// Unwrapped phase in radians. NaN marks nodata and propagates.
    pub unwrapped_phase: ArrayView2<'a, f64>,
    /// Band center frequency in Hz (NISAR `centerFrequency`).
    pub center_frequency_hz: f64,
}

/// Ionospheric phase in the main-band interferogram (`f₀ = f_A`), radians,
/// from main (`frequencyA`) and side (`frequencyB`) unwrapped phase.
///
/// # Errors
/// [`SplitSpectrumError::MissingSideBand`] when `side` is `None`;
/// [`SplitSpectrumError::ShapeMismatch`] when the grids differ;
/// [`SplitSpectrumError::InvalidFrequencies`] for unusable center frequencies.
pub fn estimate_ionosphere(
    main: &BandPhase,
    side: Option<&BandPhase>,
) -> Result<Array2<f64>, SplitSpectrumError> {
    let side = side.ok_or(SplitSpectrumError::MissingSideBand)?;
    let (main_dim, side_dim) = (main.unwrapped_phase.dim(), side.unwrapped_phase.dim());
    if main_dim != side_dim {
        return Err(SplitSpectrumError::ShapeMismatch {
            main: main_dim,
            side: side_dim,
        });
    }
    let r = frequency_ratio(main.center_frequency_hz, side.center_frequency_hz)?;
    let scale = r / (1.0 - r * r);
    Ok(Zip::from(&main.unwrapped_phase)
        .and(&side.unwrapped_phase)
        .map_collect(|&phi_a, &phi_b| scale * (phi_b - r * phi_a)))
}

/// Constant added to [`estimate_ionosphere`] by one `+2π` unwrapping slip in
/// the side-band phase: `2π·r/(1 − r²)`, `r = f_B/f_A` (radians per cycle).
///
/// # Errors
/// [`SplitSpectrumError::InvalidFrequencies`] for unusable center frequencies.
pub fn side_band_cycle_offset(main_hz: f64, side_hz: f64) -> Result<f64, SplitSpectrumError> {
    let r = frequency_ratio(main_hz, side_hz)?;
    Ok(TAU * r / (1.0 - r * r))
}

/// Integer side-band cycles removed from one connected component by
/// [`align_side_band_cycles`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentCycles {
    /// Connected-component label.
    pub component: u32,
    /// Cycles removed from the side band in this component: `φ_B ← φ_B − 2π·k`.
    pub k: i64,
    /// Pixels with both bands finite, over which `k` was estimated.
    pub pixel_count: usize,
}

/// Side-band phase on the main band's integer cycles, and the per-component shifts.
#[derive(Debug, Clone)]
pub struct AlignedSideBand {
    /// `φ_B − 2π·k_c` (radians); pixels of a component without any finite
    /// pixel pair are returned unchanged.
    pub unwrapped_phase: Array2<f64>,
    /// One entry per label with at least one finite pixel pair, ascending by label.
    pub components: Vec<ComponentCycles>,
}

/// Align the side-band unwrapped phase to the main band's integer cycles within
/// each connected component: `k_c = round(median((φ_B − r·φ_A) / 2π))`,
/// `r = f_B/f_A`, over the component's pixels where both bands are finite.
///
/// # Errors
/// [`SplitSpectrumError::ShapeMismatch`] / [`SplitSpectrumError::LabelShapeMismatch`]
/// when the grids differ; [`SplitSpectrumError::InvalidFrequencies`] for unusable
/// center frequencies.
pub fn align_side_band_cycles(
    main: &BandPhase,
    side: &BandPhase,
    labels: ArrayView2<u32>,
) -> Result<AlignedSideBand, SplitSpectrumError> {
    let (main_dim, side_dim) = (main.unwrapped_phase.dim(), side.unwrapped_phase.dim());
    if main_dim != side_dim {
        return Err(SplitSpectrumError::ShapeMismatch {
            main: main_dim,
            side: side_dim,
        });
    }
    let r = frequency_ratio(main.center_frequency_hz, side.center_frequency_hz)?;
    if labels.dim() != side_dim {
        return Err(SplitSpectrumError::LabelShapeMismatch {
            labels: labels.dim(),
            phase: side_dim,
        });
    }
    let components: Vec<ComponentCycles> = cycle_residuals_by_component(main, side, labels, r)
        .into_iter()
        .map(|(component, mut cycles)| ComponentCycles {
            component,
            k: median(&mut cycles).round() as i64,
            pixel_count: cycles.len(),
        })
        .collect();
    let shift: BTreeMap<u32, f64> = components
        .iter()
        .map(|entry| (entry.component, TAU * entry.k as f64))
        .collect();
    let unwrapped_phase = Zip::from(&side.unwrapped_phase)
        .and(&labels)
        .map_collect(|&phase, label| phase - shift.get(label).copied().unwrap_or(0.0));
    Ok(AlignedSideBand {
        unwrapped_phase,
        components,
    })
}

/// `(φ_B − r·φ_A) / 2π` at every pixel with both bands finite, grouped by label.
fn cycle_residuals_by_component(
    main: &BandPhase,
    side: &BandPhase,
    labels: ArrayView2<u32>,
    r: f64,
) -> BTreeMap<u32, Vec<f64>> {
    let mut residuals: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    Zip::from(&main.unwrapped_phase)
        .and(&side.unwrapped_phase)
        .and(&labels)
        .for_each(|&phi_a, &phi_b, &label| {
            let cycles = (phi_b - r * phi_a) / TAU;
            if cycles.is_finite() {
                residuals.entry(label).or_default().push(cycles);
            }
        });
    residuals
}

/// Median of a non-empty slice (mean of the two middle values for even length).
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let half = values.len() / 2;
    match values.len() % 2 {
        1 => values[half],
        _ => (values[half - 1] + values[half]) / 2.0,
    }
}

/// Minimum normalized weight of [`lowpass_ionosphere`]: a pixel whose Gaussian
/// neighbourhood holds less than a quarter of the kernel mass on finite pixels of its
/// own component is NaN. A straight component edge sits at ½ and a convex corner at ¼,
/// so both are kept; slivers narrower than about `0.6σ` and islands much smaller than
/// the filter scale are dropped, since their estimate rests on a handful of pixels.
pub const LOWPASS_MIN_WEIGHT: f64 = 0.25;

/// Gaussian low-pass of an ionosphere estimate within each connected component, by
/// normalized convolution ([`dolphin_filtering::masked_gaussian_lowpass`], the
/// `wavelength_cutoff_m` half-power point of `filter_long_wavelength`), with
/// `pixel_spacing_m = [y, x]` so `σ_y` and `σ_x` follow the grid's own spacings.
///
/// Each label is filtered over its own finite pixels only, so per-component offsets
/// never bleed across component edges, and non-finite pixels carry no weight (a valid
/// `0.0` is data, not nodata). Pixels below [`LOWPASS_MIN_WEIGHT`], and non-finite
/// pixels, are NaN.
///
/// # Errors
/// [`SplitSpectrumError::LabelShapeMismatch`] when `components` differs in shape;
/// [`SplitSpectrumError::Filter`] for an unusable cutoff or spacing.
pub fn lowpass_ionosphere(
    ionosphere: ArrayView2<f64>,
    components: ArrayView2<u32>,
    wavelength_cutoff_m: f64,
    pixel_spacing_m: [f64; 2],
) -> Result<Array2<f64>, SplitSpectrumError> {
    if components.dim() != ionosphere.dim() {
        return Err(SplitSpectrumError::LabelShapeMismatch {
            labels: components.dim(),
            phase: ionosphere.dim(),
        });
    }
    let mut lowpassed = Array2::from_elem(ionosphere.dim(), f64::NAN);
    for (label, bounds) in component_bounds(ionosphere, components) {
        let window = s![bounds[0]..bounds[1], bounds[2]..bounds[3]];
        let values = ionosphere.slice(window);
        let labels = components.slice(window);
        let valid = Zip::from(&values)
            .and(&labels)
            .map_collect(|value, &other| value.is_finite() && other == label);
        let filtered = masked_gaussian_lowpass(
            values,
            valid.view(),
            wavelength_cutoff_m,
            pixel_spacing_m,
            LOWPASS_MIN_WEIGHT,
        )
        .map_err(SplitSpectrumError::Filter)?;
        Zip::from(lowpassed.slice_mut(window))
            .and(&filtered)
            .and(&valid)
            .for_each(|out, &value, &ok| {
                if ok {
                    *out = value;
                }
            });
    }
    Ok(lowpassed)
}

/// `[row_start, row_stop, col_start, col_stop]` of each label's finite pixels.
fn component_bounds(
    values: ArrayView2<f64>,
    components: ArrayView2<u32>,
) -> BTreeMap<u32, [usize; 4]> {
    let mut bounds: BTreeMap<u32, [usize; 4]> = BTreeMap::new();
    Zip::indexed(&values)
        .and(&components)
        .for_each(|(row, col), value, &label| {
            if !value.is_finite() {
                return;
            }
            let entry = bounds.entry(label).or_insert([row, row + 1, col, col + 1]);
            *entry = [
                entry[0].min(row),
                entry[1].max(row + 1),
                entry[2].min(col),
                entry[3].max(col + 1),
            ];
        });
    bounds
}

/// `f_B/f_A`, rejecting non-finite, non-positive or equal frequencies.
fn frequency_ratio(main_hz: f64, side_hz: f64) -> Result<f64, SplitSpectrumError> {
    let usable = |f: f64| f.is_finite() && f > 0.0;
    if !usable(main_hz) || !usable(side_hz) || main_hz == side_hz {
        return Err(SplitSpectrumError::InvalidFrequencies { main_hz, side_hz });
    }
    Ok(side_hz / main_hz)
}
