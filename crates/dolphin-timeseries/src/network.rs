//! Interferogram-network construction (port of `interferogram.py` `Network`).
//!
//! From a date-sorted SLC stack, form interferogram index pairs `(i, j)` with
//! `i < j` per the requested modes, then sort and dedupe (dolphin's
//! `sorted(set(...))`). Modes combine: any set mode contributes its pairs.

use std::collections::BTreeSet;

/// Interferogram-network selection. Any non-`None` mode contributes pairs.
#[derive(Debug, Clone, Default)]
pub struct NetworkConfig {
    /// Single-reference network: all pairs `(reference, date)`.
    pub reference_idx: Option<usize>,
    /// Nearest-`max_bandwidth` pairs by index distance.
    pub max_bandwidth: Option<usize>,
    /// Pairs within a temporal baseline (in days).
    pub max_temporal_baseline: Option<f64>,
    /// Explicit `(early, later)` index pairs.
    pub indexes: Option<Vec<(usize, usize)>>,
}

/// Build the sorted, deduped interferogram index pairs for `n_dates` dates with
/// decimal-day positions `dates_days`.
#[must_use]
pub fn build_network(
    n_dates: usize,
    dates_days: &[f64],
    cfg: &NetworkConfig,
) -> Vec<(usize, usize)> {
    let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
    if let Some(reference_idx) = cfg.reference_idx {
        pairs.extend(single_reference(n_dates, reference_idx));
    }
    if let Some(bandwidth) = cfg.max_bandwidth {
        pairs.extend(all_pairs(n_dates).filter(|&(i, j)| j - i <= bandwidth));
    }
    if let Some(max_baseline) = cfg.max_temporal_baseline {
        pairs.extend(
            all_pairs(n_dates).filter(|&(i, j)| dates_days[j] - dates_days[i] <= max_baseline),
        );
    }
    if let Some(indexes) = &cfg.indexes {
        pairs.extend(indexes.iter().copied());
    }
    pairs.into_iter().collect()
}

/// All ordered index pairs `(i, j)` with `i < j`.
fn all_pairs(n_dates: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n_dates).flat_map(move |i| ((i + 1)..n_dates).map(move |j| (i, j)))
}

/// Single-reference pairs `(min, max)` of `reference_idx` with every other date.
fn single_reference(n_dates: usize, reference_idx: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n_dates)
        .filter(move |&d| d != reference_idx)
        .map(move |d| (reference_idx.min(d), reference_idx.max(d)))
}
