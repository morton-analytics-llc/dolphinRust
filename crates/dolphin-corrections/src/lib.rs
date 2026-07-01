//! Atmospheric phase corrections (ionospheric + tropospheric) for the dolphinRust
//! displacement pipeline.
//!
//! Both corrections produce a **per-acquisition range delay in meters** on the
//! frame grid; [`apply::subtract_delay`] removes the per-date delay (referenced to
//! acquisition 0) from the LOS-phase displacement time series, before velocity.
//! Both are **opt-in** (off by default): the workflow enables them only when
//! correction files are supplied (matching dolphin, where corrections are auxiliary
//! product layers).
//!
//! - [`ionosphere`] — IONEX TEC maps → `1/f²`-scaled L-band range delay. The
//!   dominant atmospheric term at L-band (~18× C-band for the same TEC).
//! - [`troposphere`] — OPERA L4 netCDF ingest (non-dispersive delay), resampled to
//!   the frame grid; [`raider`] is the gated fallback.
//!
//! See `CLAUDE.md` in this crate for the delay math.
#![warn(missing_docs)]

pub mod apply;
pub mod error;
pub mod geometry;
pub mod ionosphere;
pub mod raider;
pub mod troposphere;

pub use apply::subtract_delay;
pub use error::{CorrectionError, Result};
pub use geometry::{resolve_los_geometry, LosGeometry};
pub use ionosphere::{read_ionex, vtec_to_range_delay, IonexMaps, K_IONO, SPEED_OF_LIGHT};
pub use troposphere::{read_l4_netcdf, resample_bilinear, DelayGrid};
