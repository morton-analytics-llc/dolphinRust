//! Validation-only temporal covariance experiment harness for issue #53.
//!
//! This module records a preregistered synthetic grid and comparator results. It
//! deliberately has no corrected-sigma output and therefore cannot authorize
//! inferential serving. Promotion requires an external coverage artifact and
//! independent review.

use serde::{Deserialize, Serialize};

/// Stable schema identifier for the issue #53 validation receipt.
pub const TEMPORAL_VALIDATION_SCHEMA: &str = "dolphinrust-temporal-covariance-validation/1";

/// One preregistered synthetic factor cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValidationCell {
    /// Number of ordered acquisitions.
    pub date_count: usize,
    /// AR(1) correlation used to generate residuals.
    pub rho: f64,
    /// Whether one acquisition is removed before fitting.
    pub missing_date: bool,
    /// Ratio between largest and smallest measurement variance.
    pub variance_ratio: f64,
    /// Standard deviation of the spatial-reference contribution.
    pub reference_noise: f64,
}

/// Status for one synthetic validation cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationCellStatus {
    /// Cell was generated and both comparator point estimates were finite.
    Evaluated,
    /// Cell is retained in the receipt but cannot support a fit.
    NotEvaluable,
}

/// Comparator result for one synthetic cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValidationCellResult {
    /// Preregistered factors.
    pub cell: ValidationCell,
    /// Cell disposition.
    pub status: ValidationCellStatus,
    /// OLS slope point estimate in units per acquisition-day.
    pub ols_slope: Option<f64>,
    /// Oracle GLS slope point estimate using the generating covariance.
    pub oracle_gls_slope: Option<f64>,
    /// Plugin covariance result is intentionally withheld until #53 is accepted.
    pub plugin_slope: Option<f64>,
}

/// Validation-only receipt. `promotion_status` is always blocked here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalValidationReceipt {
    /// Schema identifier.
    pub schema: String,
    /// Deterministic generator seed.
    pub seed: u64,
    /// Number of attempted cells.
    pub attempted_cells: usize,
    /// Number of evaluated cells.
    pub evaluated_cells: usize,
    /// Cell-level comparator results.
    pub cells: Vec<ValidationCellResult>,
    /// Explicit promotion boundary.
    pub promotion_status: String,
}

/// Return the frozen issue #53 factor grid.
#[must_use]
pub fn preregistered_grid() -> Vec<ValidationCell> {
    [4, 8]
        .into_iter()
        .flat_map(|date_count| {
            [0.0, 0.8].into_iter().flat_map(move |rho| {
                [false, true].into_iter().flat_map(move |missing_date| {
                    [1.0, 4.0].into_iter().flat_map(move |variance_ratio| {
                        [0.0, 1.0]
                            .into_iter()
                            .map(move |reference_noise| ValidationCell {
                                date_count,
                                rho,
                                missing_date,
                                variance_ratio,
                                reference_noise,
                            })
                    })
                })
            })
        })
        .collect()
}

/// Run the deterministic validation harness without producing inferential sigma.
#[must_use]
pub fn run_preregistered_validation(seed: u64) -> TemporalValidationReceipt {
    let cells = preregistered_grid()
        .into_iter()
        .map(|cell| evaluate_cell(cell, seed))
        .collect::<Vec<_>>();
    let evaluated_cells = cells
        .iter()
        .filter(|result| result.status == ValidationCellStatus::Evaluated)
        .count();
    TemporalValidationReceipt {
        schema: TEMPORAL_VALIDATION_SCHEMA.to_owned(),
        seed,
        attempted_cells: cells.len(),
        evaluated_cells,
        cells,
        promotion_status: "blocked_pending_coverage_and_independent_review".to_owned(),
    }
}

fn evaluate_cell(cell: ValidationCell, seed: u64) -> ValidationCellResult {
    if cell.date_count < 4 || (cell.missing_date && cell.date_count < 5) {
        return ValidationCellResult {
            cell,
            status: ValidationCellStatus::NotEvaluable,
            ols_slope: None,
            oracle_gls_slope: None,
            plugin_slope: None,
        };
    }
    let mut x = Vec::with_capacity(cell.date_count);
    let mut y = Vec::with_capacity(cell.date_count);
    let mut previous_noise = 0.0;
    for index in 0..cell.date_count {
        if cell.missing_date && index == cell.date_count / 2 {
            continue;
        }
        let t = index as f64;
        let innovation = deterministic_noise(seed ^ (index as u64), 0.0);
        let noise =
            cell.rho * previous_noise + (1.0 - cell.rho * cell.rho).max(0.0).sqrt() * innovation;
        previous_noise = noise;
        x.push(t);
        y.push(0.25 * t + noise * cell.variance_ratio.sqrt() + cell.reference_noise * 0.1);
    }
    let ols_slope = slope(&x, &y);
    let oracle_gls_slope =
        oracle_gls_slope(&x, &y, cell.rho, cell.variance_ratio, cell.reference_noise);
    let status = if ols_slope.is_finite() && oracle_gls_slope.is_finite() {
        ValidationCellStatus::Evaluated
    } else {
        ValidationCellStatus::NotEvaluable
    };
    ValidationCellResult {
        cell,
        status,
        ols_slope: (status == ValidationCellStatus::Evaluated).then_some(ols_slope),
        oracle_gls_slope: (status == ValidationCellStatus::Evaluated).then_some(oracle_gls_slope),
        plugin_slope: None,
    }
}

fn slope(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let x_mean = x.iter().sum::<f64>() / x.len() as f64;
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let denominator = x.iter().map(|value| (value - x_mean).powi(2)).sum::<f64>();
    if denominator <= 0.0 {
        return f64::NAN;
    }
    x.iter()
        .zip(y)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>()
        / denominator
}

#[allow(clippy::needless_range_loop)]
fn oracle_gls_slope(
    x: &[f64],
    y: &[f64],
    rho: f64,
    variance_ratio: f64,
    reference_noise: f64,
) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let n = x.len();
    let mut covariance = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in 0..n {
            let lag = row.abs_diff(column) as i32;
            covariance[row][column] = rho.powi(lag)
                + if row == column {
                    1.0 + (variance_ratio - 1.0) * row as f64 / (n - 1) as f64
                        + reference_noise.powi(2)
                } else {
                    0.0
                };
        }
    }
    let Some(inverse) = invert_matrix(covariance) else {
        return f64::NAN;
    };
    let design = [vec![1.0; n], x.to_vec()];
    let mut normal = [[0.0; 2]; 2];
    let mut rhs = [0.0; 2];
    for left in 0..2 {
        for right in 0..2 {
            for row in 0..n {
                for column in 0..n {
                    normal[left][right] +=
                        design[left][row] * inverse[row][column] * design[right][column];
                }
            }
        }
        for row in 0..n {
            for column in 0..n {
                rhs[left] += design[left][row] * inverse[row][column] * y[column];
            }
        }
    }
    let determinant = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0];
    if determinant.abs() > f64::EPSILON {
        (normal[0][0] * rhs[1] - normal[0][1] * rhs[0]) / determinant
    } else {
        f64::NAN
    }
}

#[allow(clippy::needless_range_loop)]
fn invert_matrix(mut matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for index in 0..n {
        inverse[index][index] = 1.0;
    }
    for pivot in 0..n {
        let (pivot_row, pivot_value) = (pivot..n)
            .map(|row| (row, matrix[row][pivot].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))?;
        if pivot_value <= f64::EPSILON {
            return None;
        }
        matrix.swap(pivot, pivot_row);
        inverse.swap(pivot, pivot_row);
        let scale = matrix[pivot][pivot];
        for column in 0..n {
            matrix[pivot][column] /= scale;
            inverse[pivot][column] /= scale;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in 0..n {
                matrix[row][column] -= factor * matrix[pivot][column];
                inverse[row][column] -= factor * inverse[pivot][column];
            }
        }
    }
    Some(inverse)
}

fn deterministic_noise(seed: u64, rho: f64) -> f64 {
    let mut state = seed | 1;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let white = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0;
    white * (1.0 - rho * rho).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{preregistered_grid, run_preregistered_validation, ValidationCellStatus};

    #[test]
    fn receipt_covers_frozen_grid_and_blocks_promotion() {
        let receipt = run_preregistered_validation(0x53_u64);
        assert_eq!(
            receipt.schema,
            "dolphinrust-temporal-covariance-validation/1"
        );
        assert_eq!(receipt.attempted_cells, preregistered_grid().len());
        assert!(receipt.evaluated_cells > 0);
        assert_eq!(
            receipt.promotion_status,
            "blocked_pending_coverage_and_independent_review"
        );
        assert!(receipt.cells.iter().all(|cell| cell.plugin_slope.is_none()));
    }

    #[test]
    fn insufficient_date_cells_fail_closed() {
        let receipt = run_preregistered_validation(0x53_u64);
        assert!(receipt
            .cells
            .iter()
            .any(|cell| cell.status == ValidationCellStatus::NotEvaluable));
    }
}
