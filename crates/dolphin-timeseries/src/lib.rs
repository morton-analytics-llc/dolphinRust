//! SBAS network inversion — port of `dolphin/timeseries.py`.
//!
//! Builds the `(n_ifgs, n_dates-1)` incidence matrix and solves `A φ = Δφ`:
//! weighted L2 least squares first (via `faer`), L1/ADMM deferred. Plus
//! interferogram-network construction and linear velocity estimation.
//! Block-parallel.

pub mod inversion;
pub mod network;

pub use inversion::{estimate_velocity, get_incidence_matrix, invert_stack};
pub use network::{build_network, NetworkConfig};
