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

/// Group the `n_dates` dates into the connected components of the pair graph,
/// each component's dates ascending and components ordered by first date. A
/// network an SBAS inversion can solve has exactly one component.
#[must_use]
pub fn network_components(n_dates: usize, pairs: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n_dates).collect();
    for &(i, j) in pairs.iter().filter(|&&(i, j)| i < n_dates && j < n_dates) {
        let (a, b) = (root(&mut parent, i), root(&mut parent, j));
        parent[a.max(b)] = a.min(b);
    }
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut slot = vec![usize::MAX; n_dates];
    for date in 0..n_dates {
        let base = root(&mut parent, date);
        if slot[base] == usize::MAX {
            slot[base] = components.len();
            components.push(Vec::new());
        }
        components[slot[base]].push(date);
    }
    components
}

/// Union-find root of `node`, halving the path on the way.
fn root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

#[cfg(test)]
mod component_tests {
    use super::*;

    #[test]
    fn a_gap_wider_than_the_temporal_baseline_splits_the_network() {
        let days = [0.0, 12.0, 24.0, 108.0, 120.0];
        let cfg = NetworkConfig {
            max_temporal_baseline: Some(48.0),
            ..NetworkConfig::default()
        };
        let pairs = build_network(5, &days, &cfg);
        assert_eq!(
            network_components(5, &pairs),
            vec![vec![0, 1, 2], vec![3, 4]]
        );
        let bridged = NetworkConfig {
            max_bandwidth: Some(2),
            ..NetworkConfig::default()
        };
        let pairs = build_network(5, &days, &bridged);
        assert_eq!(network_components(5, &pairs), vec![vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn a_date_no_pair_touches_is_its_own_component() {
        assert_eq!(
            network_components(4, &[(0, 1), (1, 3)]),
            vec![vec![0, 1, 3], vec![2]]
        );
    }
}
