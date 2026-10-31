#!/usr/bin/env python
"""Decompose the GNSS harness's two InSAR rate estimators from the pair-sweep receipts.

Issues #129 and #126. Everything here is recomputed from what a scored run already
wrote down — no pipeline output is re-read beyond the run grid's geotransform:

- `window_samples.json`: the 1/3/5/7-pixel station series of the 2026-09-10 run;
- `pair_scores*.json`: the scored pair table at the recipe gate and at 0.80;
- `headline_receipt.json`: the per-station `velocity.tif` samples;
- the cached tenv3 files and the burst's CSLC-S1-STATIC LOS, which `score_common_frame`
  itself read.

Before any number is reported, the receipts are reproduced: the series estimator and
the GNSS rate to float precision, the `velocity.tif` difference to float32 precision.
If they do not reproduce, this exits rather than decomposing a different estimator.

    oracle/.venv/bin/python validation/gnss_estimator_gap.py \\
      --recipe validation/gps_mmx1_2018.json \\
      --sweep validation/results/gps_mmx1_2018/pair_sweep_20260910 \\
      --fixture-root validation/real_data/gps_mmx1_2018/cropped/mmx1_2018_los_common \\
      --gnss-cache validation/real_data/gps_mmx1_2018/gnss \\
      --grid <any displacement_NN.tif on the fixture's output grid> \\
      --output estimator_gap.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gps_ground_truth as gps  # noqa: E402

OMEGA = 2.0 * math.pi / 365.25
# Axis order of the factorial; 0 is the velocity.tif setting, 1 the series setting.
AXES = {
    "spatial_window": ("1x1 station pixel", "5x5 window mean"),
    "temporal_model": ("rate + annual sin/cos", "rate only"),
    "gauge_epoch": ("excluded (post-gauge)", "included as a zero"),
    "gnss_gap_epochs": ("kept", "dropped (common GNSS only)"),
}


def design(days: np.ndarray, seasonal: bool) -> np.ndarray:
    columns = [np.ones_like(days), days]
    if seasonal:
        columns += [np.sin(OMEGA * days), np.cos(OMEGA * days)]
    return np.column_stack(columns)


def ols(days: np.ndarray, values: np.ndarray, seasonal: bool) -> dict[str, float]:
    """Rate in mm/yr with its IID standard error, plus the residual diagnostics."""
    matrix = design(days, seasonal)
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    residual = values - matrix @ coefficients
    dof = len(values) - matrix.shape[1]
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.inv(matrix.T @ matrix)
    return {
        "rate_mm_yr": float(coefficients[1] * 365.25),
        "se_mm_yr": float(math.sqrt(covariance[1, 1]) * 365.25),
        "rss": float(residual @ residual),
        "dof": dof,
        "residual_rms_mm": math.sqrt(variance),
        "lag1_rho": float(np.corrcoef(residual[:-1], residual[1:])[0, 1]),
        "annual_amplitude_mm": float(np.hypot(coefficients[2], coefficients[3])) if seasonal else None,
    }


def annual_term_test(days: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    """Nested F test: does the residual series carry an annual term?"""
    linear, seasonal = ols(days, residual, False), ols(days, residual, True)
    f_value = ((linear["rss"] - seasonal["rss"]) / 2) / (seasonal["rss"] / seasonal["dof"])
    return {"F": f_value, "df": [2, seasonal["dof"]], "p": float(stats.f.sf(f_value, 2, seasonal["dof"]))}


def shapley(cells: dict[tuple[int, ...], float]) -> list[float]:
    """Exact Shapley share of each axis in cells[1,1,1,1] - cells[0,0,0,0]."""
    count = len(next(iter(cells)))
    shares = [0.0] * count
    orders = list(itertools.permutations(range(count)))
    for order in orders:
        state = [0] * count
        for axis in order:
            before = cells[tuple(state)]
            state[axis] = 1
            shares[axis] += cells[tuple(state)] - before
    return [share / len(orders) for share in shares]


def linear_fit(x: np.ndarray, y: np.ndarray, intercept: bool) -> dict[str, Any]:
    """OLS with nominal standard errors; callers state the real degrees of freedom."""
    matrix = np.column_stack([np.ones_like(x[:, 0]), x]) if intercept else x
    coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ coefficients
    dof = len(y) - matrix.shape[1]
    out: dict[str, Any] = {"coefficients": coefficients.tolist(), "nominal_dof": dof}
    if dof > 0:
        variance = float(residual @ residual) / dof
        out["standard_errors"] = np.sqrt(np.diag(variance * np.linalg.inv(matrix.T @ matrix))).tolist()
    else:
        out["standard_errors"] = None
    total = float(((y - y.mean()) ** 2).sum()) if intercept else float((y**2).sum())
    out["r2"] = 1.0 - float(residual @ residual) / total if total > 0 else None
    return out


def station_plane(
    pairs: dict[str, float], stations: list[str], east_north_km: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Per-station residuals from the pair table (zero-mean gauge), then a plane F test."""
    rows = []
    for label in pairs:
        primary, control = label.split("-")
        row = np.zeros(len(stations))
        row[stations.index(primary)], row[stations.index(control)] = 1.0, -1.0
        rows.append(row)
    matrix = np.vstack([*rows, np.ones(len(stations))])
    values = np.array([*pairs.values(), 0.0])
    per_station, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    fitted = matrix[:-1] @ per_station
    observed = values[:-1]
    misfit = observed - fitted
    out: dict[str, Any] = {
        "per_station_mm_yr": dict(zip(stations, per_station.tolist())),
        "pair_table_r2": 1.0 - float(misfit @ misfit) / float(((observed - observed.mean()) ** 2).sum()),
        "pair_table_misfit_rms_mm_yr": float(np.sqrt(np.mean(misfit**2))),
    }
    coords = np.vstack([east_north_km[s] for s in stations])
    coords = coords - coords.mean(axis=0)
    dof = len(stations) - 3
    out["plane_dof"] = dof
    if dof < 1:
        out["plane"] = "untestable: a plane has 3 parameters and fits this many stations exactly"
        return out
    plane = linear_fit(coords, per_station, intercept=True)
    rss_full = float(((per_station - np.column_stack([np.ones(len(stations)), coords]) @ np.array(plane["coefficients"])) ** 2).sum())
    rss_null = float(((per_station - per_station.mean()) ** 2).sum())
    f_value = ((rss_null - rss_full) / 2) / (rss_full / dof)
    out["plane"] = {
        "east_gradient_mm_yr_per_km": plane["coefficients"][1],
        "east_gradient_se": plane["standard_errors"][1],
        "north_gradient_mm_yr_per_km": plane["coefficients"][2],
        "north_gradient_se": plane["standard_errors"][2],
        "r2": plane["r2"],
        "F": f_value,
        "df": [2, dof],
        "p": float(stats.f.sf(f_value, 2, dof)),
    }
    return out


def pair_geometry_fits(
    residuals: dict[str, float], east_north_km: dict[str, np.ndarray], stations: list[str]
) -> dict[str, Any]:
    """#126 measurement 2 on the pair table itself, with its real degrees of freedom."""
    delta = np.vstack([east_north_km[p] - east_north_km[c] for p, c in (k.split("-") for k in residuals)])
    baseline = np.hypot(delta[:, 0], delta[:, 1])
    value = np.array(list(residuals.values()))
    independent = len(stations) - 1
    # A ramp flips sign with pair orientation, so its baseline signature is in |R|.
    versus_baseline = linear_fit(baseline[:, None], np.abs(value), intercept=True)
    # A tilt is R = g . (x_primary - x_control): no intercept, orientation-invariant.
    tilt = linear_fit(delta, value, intercept=False)
    gradient = np.array(tilt["coefficients"])
    return {
        "pairs": len(value),
        "independent_differential_series": independent,
        "abs_residual_vs_baseline": {
            "slope_mm_yr_per_km": versus_baseline["coefficients"][1],
            "slope_se_nominal": None if versus_baseline["standard_errors"] is None else versus_baseline["standard_errors"][1],
            "intercept_mm_yr": versus_baseline["coefficients"][0],
            "r2": versus_baseline["r2"],
            "nominal_dof": versus_baseline["nominal_dof"],
            "independent_dof": independent - 2,
        },
        "tilt_vs_bearing": {
            "east_gradient_mm_yr_per_km": tilt["coefficients"][0],
            "north_gradient_mm_yr_per_km": tilt["coefficients"][1],
            "standard_errors_nominal": tilt["standard_errors"],
            "gradient_magnitude_mm_yr_per_km": float(np.hypot(*gradient)),
            "gradient_azimuth_deg": float(math.degrees(math.atan2(gradient[0], gradient[1])) % 360),
            "r2_uncentred": tilt["r2"],
            "nominal_dof": tilt["nominal_dof"],
            "independent_dof": independent - 2,
        },
    }


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--gnss-cache", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text())
    windows = json.loads((args.sweep / "window_samples.json").read_text())
    headline = json.loads((args.sweep / "headline_receipt.json").read_text())
    gates = {
        "recipe_gate_0.90": json.loads((args.sweep / "pair_scores.json").read_text()),
        "diagnostic_gate_0.80": json.loads((args.sweep / "pair_scores_relaxed.json").read_text()),
    }
    manifest = json.loads((args.fixture_root / "fixture_manifest.json").read_text())
    static = next((args.fixture_root / "static").glob(f"*{recipe['burst_filename_id']}*.h5"))
    if headline["dates"] != recipe["expected_dates"]:
        raise SystemExit("headline receipt dates differ from the recipe")
    dates = [dt.date.fromisoformat(value) for value in recipe["expected_dates"]]
    days = np.array([(date - dates[0]).days for date in dates], dtype=float)
    stations = sorted(windows["stations"])
    series = {
        s: {w: np.asarray(windows["stations"][s]["samples"][w]["series_mm"]) for w in ("1", "3", "5", "7")}
        for s in stations
    }

    # GNSS LOS per station, exactly as score_common_frame builds it.
    available: dict[str, np.ndarray] = {}
    aligned: dict[str, list] = {}
    los: dict[str, np.ndarray] = {}
    with rasterio.open(args.grid) as grid:
        for s in stations:
            station = recipe["stations"][s]
            pixel = list(gps.station_pixel(grid, station["longitude"], station["latitude"]))
            if pixel != windows["stations"][s]["pixel"]:
                raise SystemExit(f"{s}: grid puts the station at {pixel}, receipt at {windows['stations'][s]['pixel']}")
            los[s] = gps.los_at_output_pixel(static, grid, *pixel)
            records = gps.parse_tenv3((args.gnss_cache / f"{s}.tenv3").read_text())
            aligned[s] = gps.align_records_available(records, dates, recipe["max_interpolation_gap_days"])
            available[s] = np.array([item is not None for item in aligned[s]])

    def gnss_series(s: str, mask: np.ndarray) -> np.ndarray:
        out = np.full(len(dates), np.nan)
        out[mask] = gps.gnss_los_series([a for a, keep in zip(aligned[s], mask) if keep], los[s])
        return out

    def insar_rate(primary: str, control: str, bits: tuple[int, ...], mask: np.ndarray) -> float:
        spatial, model, gauge, gaps = bits
        window = "5" if spatial else "1"
        values = series[primary][window] - series[control][window]
        keep = mask.copy() if gaps else np.ones(len(dates), dtype=bool)
        keep[0] = bool(gauge)
        return ols(days[keep], values[keep], seasonal=not model)["rate_mm_yr"]

    report: dict[str, Any] = {
        "schema": "dolphinrust-gnss-estimator-gap/1",
        "inputs": {
            "run_commit": windows["commit"],
            "fixture": windows["fixture"],
            "model": headline["model"],
            "sweep": str(args.sweep),
            "grid_geotransform_source": str(args.grid),
        },
        "axes": {name: {"velocity_tif": a, "series": b} for name, (a, b) in AXES.items()},
        "reproduction": {},
        "issue_129": {},
        "issue_126": {},
    }

    pair_results: dict[str, dict[str, Any]] = {}
    worst = {"series_rate": 0.0, "gnss_rate": 0.0, "raster_rate": 0.0}
    for label, scored in gates["diagnostic_gate_0.80"]["pairs"].items():
        if scored["status"] != "scored":
            continue
        primary, control = label.split("-")
        mask = available[primary] & available[control]
        gnss = gnss_series(primary, mask) - gnss_series(control, mask)
        cells = {bits: insar_rate(primary, control, bits, mask) for bits in itertools.product((0, 1), repeat=4)}
        gnss_linear = ols(days[mask], gnss[mask], False)["rate_mm_yr"]
        worst["series_rate"] = max(worst["series_rate"], abs(cells[(1, 1, 1, 1)] - scored["insar_velocity_mm_yr"]))
        worst["gnss_rate"] = max(worst["gnss_rate"], abs(gnss_linear - scored["gnss_velocity_mm_yr"]))
        worst["raster_rate"] = max(worst["raster_rate"], abs(cells[(0, 0, 0, 0)] - scored["insar_velocity_raster_mm_yr"]))

        matched = {}
        for window in ("5", "1"):
            residual = (series[primary][window] - series[control][window] - gnss)[mask]
            for model, seasonal in (("linear", False), ("seasonal", True)):
                fit = ols(days[mask], residual, seasonal)
                matched[f"{window}x{window}_{model}"] = {
                    "residual_rate_mm_yr": fit["rate_mm_yr"],
                    "se_iid_mm_yr": fit["se_mm_yr"],
                    "lag1_rho": fit["lag1_rho"],
                    "residual_rms_mm": fit["residual_rms_mm"],
                    "annual_amplitude_mm": fit["annual_amplitude_mm"],
                }
            matched[f"{window}x{window}_annual_term_F_test"] = annual_term_test(days[mask], residual)
            # Under the null of no annual term both rates are unbiased and the linear one
            # is efficient, so var(difference) = var(seasonal) - var(linear).
            lin, sea = matched[f"{window}x{window}_linear"], matched[f"{window}x{window}_seasonal"]
            spread = math.sqrt(max(sea["se_iid_mm_yr"] ** 2 - lin["se_iid_mm_yr"] ** 2, 0.0))
            matched[f"{window}x{window}_linear_minus_seasonal"] = {
                "mm_yr": lin["residual_rate_mm_yr"] - sea["residual_rate_mm_yr"],
                "z": (lin["residual_rate_mm_yr"] - sea["residual_rate_mm_yr"]) / spread,
            }
        matched["gnss_rate_linear_mm_yr"] = gnss_linear
        matched["gnss_rate_seasonal_mm_yr"] = ols(days[mask], gnss[mask], True)["rate_mm_yr"]

        gap = cells[(1, 1, 1, 1)] - cells[(0, 0, 0, 0)]
        pair_results[label] = {
            "common_gnss_epochs": int(mask.sum()),
            "velocity_tif_rate_mm_yr": cells[(0, 0, 0, 0)],
            "series_rate_mm_yr": cells[(1, 1, 1, 1)],
            "gap_series_minus_velocity_tif_mm_yr": gap,
            "shapley_share_mm_yr": dict(zip(AXES, shapley(cells))),
            "single_axis_switch_from_velocity_tif_mm_yr": {
                name: cells[tuple(1 if i == k else 0 for i in range(4))] - cells[(0, 0, 0, 0)]
                for k, name in enumerate(AXES)
            },
            "cells": {"".join(map(str, bits)): value for bits, value in cells.items()},
            "residual_vs_gnss": {
                "raster_mismatched_design_mm_yr": scored["difference_raster_mm_yr"],
                "series_matched_linear_mm_yr": scored["difference_mm_yr"],
            },
            "matched_design": matched,
        }
    report["reproduction"] = {
        "max_abs_deviation_mm_yr": worst,
        "note": "velocity.tif is float32; a raster deviation under 1e-3 mm/yr is the cast",
    }
    if worst["series_rate"] > 1e-6 or worst["gnss_rate"] > 1e-6 or worst["raster_rate"] > 1e-3:
        raise SystemExit(f"receipts do not reproduce: {worst}")
    strict = [k for k, v in gates["recipe_gate_0.90"]["pairs"].items() if v["status"] == "scored"]
    report["issue_129"] = {"recipe_gate_pairs": strict, "pairs": pair_results}

    # Geometry: projected coordinates from the fixture manifest (EPSG from the manifest).
    east_north_km = {
        s: np.array([manifest["stations"][s]["projected_x"], manifest["stations"][s]["projected_y"]]) / 1000.0
        for s in manifest["stations"]
    }
    baselines = {}
    for a, b in [("MMX1", "ICMX"), ("SSNX", "TNGF")]:
        tenv = {}
        for s in (a, b):
            records = gps.parse_tenv3((args.gnss_cache / f"{s}.tenv3").read_text())
            ref = min(records, key=lambda r: abs((r.date - dates[0]).days))
            tenv[s] = (ref.latitude, ref.longitude, ref.date.isoformat())
        delta_m = (east_north_km[a] - east_north_km[b]) * 1000.0
        baselines[f"{a}-{b}"] = {
            "recipe_great_circle_km": haversine_km(
                (recipe["stations"][a]["latitude"], recipe["stations"][a]["longitude"]),
                (recipe["stations"][b]["latitude"], recipe["stations"][b]["longitude"]),
            ),
            "tenv3_great_circle_km": haversine_km(tenv[a][:2], tenv[b][:2]),
            "tenv3_reference_dates": [tenv[a][2], tenv[b][2]],
            "fixture_projected_km": float(np.hypot(*delta_m) / 1000.0),
            "projected_epsg": manifest["epsg"],
        }
    with rasterio.open(args.grid) as grid:
        posting = (abs(grid.transform.a), abs(grid.transform.e))
    delta_m = (east_north_km["SSNX"] - east_north_km["TNGF"]) * 1000.0
    separation_px = (abs(delta_m[0]) / posting[0], abs(delta_m[1]) / posting[1])
    baselines["SSNX-TNGF"]["posting_m"] = list(posting)
    baselines["SSNX-TNGF"]["separation_pixels_x_y"] = list(separation_px)
    baselines["SSNX-TNGF"]["primary_windows_overlap"] = bool(
        all(sep < recipe["primary_window"] for sep in separation_px)
    )
    report["issue_126"]["baselines"] = baselines

    fits = {}
    for gate, table in gates.items():
        scored_pairs = {k: v for k, v in table["pairs"].items() if v["status"] == "scored"}
        gate_stations = sorted({s for k in scored_pairs for s in k.split("-")})
        fits[gate] = {"stations": gate_stations}
        estimators = {
            "headline_matched_linear": {k: v["difference_mm_yr"] for k, v in scored_pairs.items()},
            "matched_seasonal": {
                k: pair_results[k]["matched_design"]["5x5_seasonal"]["residual_rate_mm_yr"] for k in scored_pairs
            },
            "raster_mismatched": {k: v["difference_raster_mm_yr"] for k, v in scored_pairs.items()},
        }
        for estimator, residuals in estimators.items():
            fits[gate][estimator] = {
                "pair_fits": pair_geometry_fits(residuals, east_north_km, gate_stations),
                "station_field": station_plane(residuals, gate_stations, east_north_km),
            }
    report["issue_126"]["geometry_fits"] = fits

    # Measurement 3: window-size spread converted to a rate, under each estimator.
    sweep = {}
    for s in stations:
        rows = {}
        for label, fit_days, seasonal in (
            ("linear_all_epochs", slice(None), False),
            ("velocity_tif_estimator", slice(1, None), True),
        ):
            rates = {w: ols(days[fit_days], series[s][w][fit_days], seasonal)["rate_mm_yr"] for w in series[s]}
            rows[label] = {"rates_mm_yr": rates, "spread_mm_yr": max(rates.values()) - min(rates.values())}
        rows["mean_5x5_within_window_sd_mm"] = float(
            np.mean(windows["stations"][s]["samples"]["5"]["std_mm"])
        )
        sweep[s] = rows
    pair_sweep = {}
    for label in pair_results:
        primary, control = label.split("-")
        mask = available[primary] & available[control]
        rates = {
            w: ols(days[mask], (series[primary][w] - series[control][w])[mask], False)["rate_mm_yr"]
            for w in ("1", "3", "5", "7")
        }
        pair_sweep[label] = {"headline_rates_mm_yr": rates, "spread_mm_yr": max(rates.values()) - min(rates.values())}
    report["issue_126"]["window_sweep"] = {"per_station": sweep, "per_pair_headline_estimator": pair_sweep}

    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
