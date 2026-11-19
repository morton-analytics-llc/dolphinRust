//! Whirlwind backend: Scott Staniewicz's Rust minimum-cost-flow unwrapper
//! (`whirlwind-core`, BSD-3-Clause OR Apache-2.0), called in-process.
//!
//! Phase comes from `whirlwind_core::unwrap_linear`, the crate's verified
//! single-tile linear solver with Carballo statistical costs, called directly
//! so the `WHIRLWIND_UNWRAP_SOLVER` environment switch cannot change it.
//! Components come from `components_snaphu`, whirlwind's port of SNAPHU's
//! connected-component grow, so no SNAPHU subprocess is needed (dolphin's
//! Python wrapper shells out to SNAPHU for this step).

use ndarray::{Array2, ArrayView2, Zip};
use whirlwind_core::cost::PhaseGradWindow;
use whirlwind_core::SnaphuConnCompParams;

pub use whirlwind_core::UnwrapError as WhirlwindError;

use crate::snaphu::UnwrapResult;
use dolphin_core::Cf32;

/// Whirlwind solver settings.
#[derive(Debug, Clone, PartialEq)]
pub struct WhirlwindConfig {
    /// Effective number of looks of the correlation estimate, for the
    /// Carballo cost model.
    pub nlooks: f32,
}

impl Default for WhirlwindConfig {
    /// dolphin's `unwrap_whirlwind` passes `clip(nlooks / 2, 1, 20)` with its
    /// default `nlooks = 5`.
    fn default() -> Self {
        Self { nlooks: 2.5 }
    }
}

/// Unwrap one interferogram with whirlwind.
///
/// Pixels whose interferogram or correlation is non-finite or exactly zero are
/// masked out of the solve and get component 0.
///
/// # Errors
/// Shape mismatch between `wrapped` and `correlation`, or a grid under 2x2.
pub fn unwrap_whirlwind(
    wrapped: ArrayView2<Cf32>,
    correlation: ArrayView2<f32>,
    cfg: &WhirlwindConfig,
) -> Result<UnwrapResult, WhirlwindError> {
    let mut mask = Array2::from_elem(wrapped.dim(), false);
    Zip::from(&mut mask)
        .and(wrapped)
        .and(correlation)
        .for_each(|valid, z, &corr| {
            *valid = z.re.is_finite() && z.im.is_finite() && *z != Cf32::new(0.0, 0.0);
            *valid &= corr.is_finite() && corr > 0.0;
        });
    let correlation = correlation.mapv(|corr| if corr.is_finite() { corr } else { 0.0 });
    let window = PhaseGradWindow::default();
    let unwrapped = whirlwind_core::unwrap_linear(
        wrapped,
        correlation.view(),
        cfg.nlooks,
        Some(mask.view()),
        window,
    )?;
    let conncomp = whirlwind_core::components_snaphu(
        wrapped,
        correlation.view(),
        cfg.nlooks,
        unwrapped.view(),
        Some(mask.view()),
        SnaphuConnCompParams {
            phase_grad_window: window,
            ..SnaphuConnCompParams::default()
        },
    )?;
    Ok(UnwrapResult {
        unwrapped,
        conncomp,
    })
}
