//! SBAS network inversion — port of `dolphin/timeseries.py`.
//!
//! Builds the `(n_ifgs, n_dates-1)` incidence matrix and solves `A φ = Δφ`:
//! weighted L2 least squares (via `faer`) and L1/ADMM (dolphin's default LAD).
//! Plus interferogram-network construction and linear velocity estimation.
//! Block-parallel.
#![warn(missing_docs)]

pub mod inversion;
pub mod network;

pub use inversion::{
    estimate_velocity, get_incidence_matrix, invert_stack, invert_stack_l1, L1Config,
};
pub use network::{build_network, NetworkConfig};
