#!/usr/bin/env python
"""Independent GNSS-to-InSAR comparison primitives for public station pairs."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import rowcol, xy


class NotEvaluable(RuntimeError):
    """The validation could not produce scientific evidence."""


@dataclass(frozen=True)
class Tenv3Record:
    station: str
    date: dt.date
    east_m: float
    north_m: float
    up_m: float
    sigma_e_m: float
    sigma_n_m: float
    sigma_u_m: float
    latitude: float
    longitude: float
    height_m: float


@dataclass(frozen=True)
class AlignedRecord:
    record: Tenv3Record
    quality: str


@dataclass(frozen=True)
class WindowStats:
    mean: float
    std: float
    valid_count: int
    total_count: int


def json_safe(value: Any) -> Any:
    """Convert numpy values and nonfinite abstentions to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_tenv3(text: str) -> list[Tenv3Record]:
    """Parse the 23-column NGL tenv3 format documented by README_tenv3.txt."""
    records: list[Tenv3Record] = []
    seen: set[dt.date] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("site "):
            continue
        columns = stripped.split()
        if len(columns) != 23:
            raise ValueError(f"tenv3 line {line_number} must have 23 columns; found {len(columns)}")
        try:
            date = dt.datetime.strptime(columns[1].upper(), "%y%b%d").date()
            values = [float(value) for value in columns[2:]]
        except ValueError as error:
            raise ValueError(f"invalid tenv3 line {line_number}: {error}") from error
        if date in seen:
            raise ValueError(f"duplicate tenv3 date: {date}")
        seen.add(date)
        # Integer and fractional coordinate columns are summed before differencing.
        # This remains continuous if NGL reinitializes an integer component.
        east_m = float(columns[7]) + float(columns[8])
        north_m = float(columns[9]) + float(columns[10])
        up_m = float(columns[11]) + float(columns[12])
        numeric = [east_m, north_m, up_m, *values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite tenv3 value on line {line_number}")
        records.append(
            Tenv3Record(
                station=columns[0],
                date=date,
                east_m=east_m,
                north_m=north_m,
                up_m=up_m,
                sigma_e_m=float(columns[14]),
                sigma_n_m=float(columns[15]),
                sigma_u_m=float(columns[16]),
                latitude=float(columns[20]),
                longitude=float(columns[21]),
                height_m=float(columns[22]),
            )
        )
    if not records:
        raise ValueError("tenv3 series contains no data rows")
    stations = {record.station for record in records}
    if len(stations) != 1:
        raise ValueError(f"tenv3 series mixes station IDs: {sorted(stations)}")
    return sorted(records, key=lambda record: record.date)


def interpolate_record(before: Tenv3Record, after: Tenv3Record, date: dt.date) -> Tenv3Record:
    span = (after.date - before.date).days
    fraction = (date - before.date).days / span

    def lerp(a: float, b: float) -> float:
        return a + fraction * (b - a)

    return Tenv3Record(
        station=before.station,
        date=date,
        east_m=lerp(before.east_m, after.east_m),
        north_m=lerp(before.north_m, after.north_m),
        up_m=lerp(before.up_m, after.up_m),
        sigma_e_m=lerp(before.sigma_e_m, after.sigma_e_m),
        sigma_n_m=lerp(before.sigma_n_m, after.sigma_n_m),
        sigma_u_m=lerp(before.sigma_u_m, after.sigma_u_m),
        latitude=lerp(before.latitude, after.latitude),
        longitude=lerp(before.longitude, after.longitude),
        height_m=lerp(before.height_m, after.height_m),
    )


def align_records(
    records: Sequence[Tenv3Record], dates: Sequence[dt.date], max_gap_days: int
) -> list[AlignedRecord]:
    by_date = {record.date: record for record in records}
    ordered = sorted(records, key=lambda record: record.date)
    aligned: list[AlignedRecord] = []
    for date in dates:
        if date in by_date:
            aligned.append(AlignedRecord(by_date[date], "exact"))
            continue
        before = next((record for record in reversed(ordered) if record.date < date), None)
        after = next((record for record in ordered if record.date > date), None)
        if before is None or after is None:
            raise NotEvaluable(f"cannot extrapolate GNSS series to {date}")
        span = (after.date - before.date).days
        if span > max_gap_days:
            raise NotEvaluable(
                f"GNSS gap of {span} days around {date} exceeds {max_gap_days}-day limit"
            )
        aligned.append(AlignedRecord(interpolate_record(before, after, date), "interpolated"))
    return aligned


def align_records_available(
    records: Sequence[Tenv3Record], dates: Sequence[dt.date], max_gap_days: int
) -> list[AlignedRecord | None]:
    """Align dates independently so unavailable truth remains an abstention."""
    output: list[AlignedRecord | None] = []
    for date in dates:
        try:
            output.append(align_records(records, [date], max_gap_days)[0])
        except NotEvaluable:
            output.append(None)
    return output


def project_enu(enu_m: np.ndarray, los: np.ndarray) -> float:
    enu = np.asarray(enu_m, dtype=float)
    vector = np.asarray(los, dtype=float)
    if enu.shape != (3,) or vector.shape != (3,):
        raise ValueError("ENU and LOS must each contain east, north, up")
    if not np.all(np.isfinite(enu)) or not np.all(np.isfinite(vector)):
        raise ValueError("ENU and LOS must be finite")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"LOS vector is not unit norm: {norm}")
    return float(np.dot(enu, vector))


def spatial_difference(primary: np.ndarray, control: np.ndarray) -> np.ndarray:
    primary_values = np.asarray(primary, dtype=float)
    control_values = np.asarray(control, dtype=float)
    if primary_values.shape != control_values.shape:
        raise ValueError("station series must share a common frame and shape")
    return primary_values - control_values


def window_stats(
    data: np.ndarray,
    row: int,
    col: int,
    size: int,
    minimum_finite_fraction: float,
) -> WindowStats:
    if size <= 0 or size % 2 == 0:
        raise ValueError("sample window size must be a positive odd integer")
    if not 0 < minimum_finite_fraction <= 1:
        raise ValueError("minimum finite fraction must be in (0, 1]")
    half = size // 2
    if row - half < 0 or col - half < 0 or row + half >= data.shape[0] or col + half >= data.shape[1]:
        raise NotEvaluable(f"{size}x{size} station window is outside the raster")
    values = np.asarray(data[row - half : row + half + 1, col - half : col + half + 1], dtype=float)
    finite = values[np.isfinite(values)]
    required = math.ceil(values.size * minimum_finite_fraction)
    if finite.size < required:
        raise NotEvaluable(
            f"station window has {finite.size}/{values.size} finite pixels; requires {required}"
        )
    return WindowStats(float(finite.mean()), float(finite.std()), int(finite.size), int(values.size))


def prepend_reference_epoch(bands: Sequence[np.ndarray]) -> np.ndarray:
    if not bands:
        raise ValueError("at least one displacement band is required")
    arrays = [np.asarray(band, dtype=float) for band in bands]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("displacement rasters do not share a shape")
    return np.stack([np.zeros(shape, dtype=float), *arrays])


def tls_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    centered = np.column_stack([x - x.mean(), y - y.mean()])
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] == 0 or abs(vh[0, 0]) < 1e-12:
        raise NotEvaluable("TLS slope is undefined for a constant series")
    slope = float(vh[0, 1] / vh[0, 0])
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def compute_metrics(
    truth_mm: np.ndarray, estimate_mm: np.ndarray, thresholds: dict[str, float]
) -> dict[str, Any]:
    truth = np.asarray(truth_mm, dtype=float)
    estimate = np.asarray(estimate_mm, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 1 or truth.size < 3:
        raise NotEvaluable("metric series must be one-dimensional, equal, and contain at least 3 epochs")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise NotEvaluable("metric series contains non-finite values")
    residual = estimate - truth
    correlation = float(np.corrcoef(truth, estimate)[0, 1])
    if not math.isfinite(correlation):
        raise NotEvaluable("time-series correlation is undefined")
    ols_slope, ols_intercept = np.polyfit(truth, estimate, 1)
    tls_slope, tls_intercept = tls_line(truth, estimate)
    sign_agrees = bool(np.signbit(truth[-1]) == np.signbit(estimate[-1]) and truth[-1] != 0 and estimate[-1] != 0)
    endpoint_residual = float(residual[-1])
    passed = (
        sign_agrees
        and abs(endpoint_residual) <= thresholds["endpoint_error_mm"]
        and thresholds["tls_slope_min"] <= tls_slope <= thresholds["tls_slope_max"]
        and correlation >= thresholds["correlation_min"]
    )
    return {
        "status": "pass" if passed else "fail",
        "endpoint_sign_agrees": sign_agrees,
        "endpoint_truth_mm": float(truth[-1]),
        "endpoint_estimate_mm": float(estimate[-1]),
        "endpoint_residual_mm": endpoint_residual,
        "mae_mm": float(np.mean(np.abs(residual))),
        "rmse_mm": float(np.sqrt(np.mean(residual**2))),
        "correlation": correlation,
        "ols_slope": float(ols_slope),
        "ols_intercept_mm": float(ols_intercept),
        "tls_slope": tls_slope,
        "tls_intercept_mm": tls_intercept,
    }


def fetch_text(url: str, destination: Path) -> tuple[str, dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = destination.exists()
    if cache_hit:
        text = destination.read_text()
    else:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        text = response.text
        destination.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return text, {
        "url": url,
        "path": str(destination),
        "sha256": digest,
        "bytes": len(text.encode()),
        "cache_hit": cache_hit,
        "observed_at": datetime.now(UTC).isoformat(),
        "cache_modified_at": datetime.fromtimestamp(
            destination.stat().st_mtime, UTC
        ).isoformat(),
    }


def station_pixel(dataset: rasterio.DatasetReader, longitude: float, latitude: float) -> tuple[int, int]:
    transformer = Transformer.from_crs(4326, dataset.crs, always_xy=True)
    projected_x, projected_y = transformer.transform(longitude, latitude)
    row, col = rowcol(dataset.transform, projected_x, projected_y)
    if row < 0 or col < 0 or row >= dataset.height or col >= dataset.width:
        raise NotEvaluable("station coordinate lies outside displacement raster")
    return int(row), int(col)


def interpolate_grid(data: np.ndarray, x: np.ndarray, y: np.ndarray, x_value: float, y_value: float) -> float:
    x_order = np.argsort(x)
    y_order = np.argsort(y)
    xs = x[x_order]
    ys = y[y_order]
    values = data[np.ix_(y_order, x_order)]
    if not (xs[0] <= x_value <= xs[-1] and ys[0] <= y_value <= ys[-1]):
        raise NotEvaluable("output station pixel lies outside STATIC geometry")
    col_hi = int(np.searchsorted(xs, x_value, side="right"))
    row_hi = int(np.searchsorted(ys, y_value, side="right"))
    col_hi = min(max(col_hi, 1), len(xs) - 1)
    row_hi = min(max(row_hi, 1), len(ys) - 1)
    col_lo, row_lo = col_hi - 1, row_hi - 1
    x0, x1 = xs[col_lo], xs[col_hi]
    y0, y1 = ys[row_lo], ys[row_hi]
    tx = 0.0 if x1 == x0 else (x_value - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y_value - y0) / (y1 - y0)
    return float(
        (1 - tx) * (1 - ty) * values[row_lo, col_lo]
        + tx * (1 - ty) * values[row_lo, col_hi]
        + (1 - tx) * ty * values[row_hi, col_lo]
        + tx * ty * values[row_hi, col_hi]
    )


def los_at_output_pixel(
    static_path: Path, dataset: rasterio.DatasetReader, row: int, col: int
) -> np.ndarray:
    output_x, output_y = xy(dataset.transform, row, col, offset="center")
    with h5py.File(static_path, "r") as product:
        x = product["/data/x_coordinates"][:].astype(float)
        y = product["/data/y_coordinates"][:].astype(float)
        static_epsg = int(product["/data/projection"][()])
        east_data = product["/data/los_east"][:].astype(float)
        north_data = product["/data/los_north"][:].astype(float)
    if dataset.crs.to_epsg() != static_epsg:
        transformer = Transformer.from_crs(dataset.crs, static_epsg, always_xy=True)
        output_x, output_y = transformer.transform(output_x, output_y)
    east = interpolate_grid(east_data, x, y, output_x, output_y)
    north = interpolate_grid(north_data, x, y, output_x, output_y)
    if not (math.isfinite(east) and math.isfinite(north)) or (east == 0 and north == 0):
        raise NotEvaluable("STATIC LOS is invalid at station output pixel")
    up = math.sqrt(max(0.0, 1.0 - east * east - north * north))
    los = np.array([east, north, up])
    norm = float(np.linalg.norm(los))
    if abs(norm - 1.0) > 1e-5:
        raise NotEvaluable(f"STATIC LOS unit norm is invalid at station: {norm}")
    return los


def load_displacement_cube(work_directory: Path, expected_epochs: int) -> tuple[np.ndarray, dict[str, Any]]:
    files = sorted(work_directory.glob("displacement_[0-9][0-9].tif"))
    if len(files) != expected_epochs - 1:
        raise NotEvaluable(
            f"expected {expected_epochs - 1} displacement rasters in {work_directory}; found {len(files)}"
        )
    bands: list[np.ndarray] = []
    metadata: dict[str, Any] | None = None
    for path in files:
        with rasterio.open(path) as dataset:
            band = dataset.read(1).astype(float)
            current = {
                "shape": [dataset.height, dataset.width],
                "crs": str(dataset.crs),
                "transform": list(dataset.transform)[:6],
            }
            if metadata is None:
                metadata = current
            elif current != metadata:
                raise NotEvaluable(f"displacement raster grid differs: {path}")
            bands.append(band)
    return prepend_reference_epoch(bands), metadata or {}


def unwrap_component_counts(work_directory: Path) -> list[int]:
    """Count positive connected-component labels in each interferogram raster."""
    counts = []
    for path in sorted(work_directory.glob("conncomp_[0-9][0-9].tif")):
        with rasterio.open(path) as dataset:
            labels = np.unique(dataset.read(1))
        counts.append(int(np.sum(labels > 0)))
    return counts


def raster_delta_summary(first: Path, second: Path) -> dict[str, float]:
    """Summarize finite first-minus-second raster differences."""
    with rasterio.open(first) as left, rasterio.open(second) as right:
        if (left.shape, left.crs, left.transform) != (right.shape, right.crs, right.transform):
            raise NotEvaluable("weighted/unweighted velocity grids differ")
        delta = left.read(1).astype(float) - right.read(1).astype(float)
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        raise NotEvaluable("weighted/unweighted velocity delta has no finite pixels")
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "max_abs": float(np.max(np.abs(finite))),
    }


def infer_reference_pixel(cube: np.ndarray, coherence_path: Path) -> dict[str, Any]:
    finite = np.all(np.isfinite(cube), axis=0)
    zero = np.all(np.abs(cube) <= 1e-12, axis=0)
    candidates = np.argwhere(finite & zero)
    if candidates.size == 0:
        return {"status": "not_inferred", "reason": "no all-epoch exact-zero pixel"}
    if coherence_path.exists():
        with rasterio.open(coherence_path) as dataset:
            coherence = dataset.read(1)
        scores = np.array([coherence[row, col] for row, col in candidates], dtype=float)
        scores[~np.isfinite(scores)] = -np.inf
        selected = int(np.argmax(scores))
    else:
        selected = 0
    row, col = candidates[selected]
    return {
        "status": "inferred_from_all_epoch_zero",
        "pixel": [int(row), int(col)],
        "candidate_count": int(len(candidates)),
    }


def gnss_los_series(aligned: Sequence[AlignedRecord], los: np.ndarray) -> np.ndarray:
    coordinates = np.array(
        [[item.record.east_m, item.record.north_m, item.record.up_m] for item in aligned]
    )
    delta = coordinates - coordinates[0]
    return np.array([project_enu(vector, los) for vector in delta]) * 1000.0


def gnss_los_sigma_series(aligned: Sequence[AlignedRecord], los: np.ndarray) -> np.ndarray:
    """Project independent ENU one-sigma errors into referenced LOS millimeters."""
    component_sigma = np.array(
        [
            math.sqrt(
                (item.record.sigma_e_m * los[0]) ** 2
                + (item.record.sigma_n_m * los[1]) ** 2
                + (item.record.sigma_u_m * los[2]) ** 2
            )
            for item in aligned
        ]
    )
    relative = np.sqrt(component_sigma**2 + component_sigma[0] ** 2) * 1000.0
    relative[0] = 0.0
    return relative


def sample_cube(
    cube_m: np.ndarray,
    row: int,
    col: int,
    sizes: Sequence[int],
    minimum_finite_fraction: float,
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for size in sizes:
        stats = [window_stats(band, row, col, size, minimum_finite_fraction) for band in cube_m]
        output[size] = {
            "series_mm": [item.mean * 1000.0 for item in stats],
            "std_mm": [item.std * 1000.0 for item in stats],
            "valid_count": [item.valid_count for item in stats],
            "total_count": stats[0].total_count,
        }
    return output


def sample_uncertainty_layers(
    work_directory: Path,
    row: int,
    col: int,
    window: int,
    minimum_finite_fraction: float,
    expected_epochs: int,
    wavelength_m: float,
) -> dict[str, Any]:
    """Sample CRLB, posterior, and velocity uncertainty from one station pixel."""
    mm_per_rad = abs(wavelength_m / (4.0 * math.pi)) * 1000.0

    def sample_files(files: Sequence[Path], transform) -> list[float]:
        values = []
        for path in files:
            with rasterio.open(path) as dataset:
                stats = window_stats(
                    transform(dataset.read(1).astype(float)),
                    row,
                    col,
                    window,
                    minimum_finite_fraction,
                )
            values.append(stats.mean)
        return values

    crlb_files = sorted(work_directory.glob("crlb_sigma_[0-9][0-9].tif"))
    posterior_files = sorted(
        work_directory.glob("displacement_variance_[0-9][0-9].tif")
    )
    if len(crlb_files) != expected_epochs:
        raise NotEvaluable(
            f"expected {expected_epochs} CRLB rasters in {work_directory}; found {len(crlb_files)}"
        )
    if len(posterior_files) != expected_epochs - 1:
        raise NotEvaluable(
            f"expected {expected_epochs - 1} posterior rasters in {work_directory}; found {len(posterior_files)}"
        )
    crlb = sample_files(crlb_files, lambda data: np.abs(data) * mm_per_rad)
    posterior = [0.0, *sample_files(posterior_files, lambda data: np.sqrt(np.maximum(data, 0.0)) * 1000.0)]
    velocity_path = work_directory / "velocity_sigma.tif"
    if not velocity_path.exists():
        raise NotEvaluable(f"missing velocity uncertainty: {velocity_path}")
    velocity_sigma = sample_files([velocity_path], lambda data: np.abs(data) * 1000.0)[0]
    return {
        "crlb_sigma_mm": crlb,
        "posterior_sigma_mm": posterior,
        "velocity_sigma_mm_yr": velocity_sigma,
    }


def uncertainty_reliability(
    residual_mm: np.ndarray,
    gnss_sigma_mm: np.ndarray,
    crlb_sigma_mm: np.ndarray,
    posterior_sigma_mm: np.ndarray,
) -> dict[str, Any]:
    """Coverage and interval score for independent uncertainty alternatives."""
    residual = np.asarray(residual_mm, dtype=float)
    gnss = np.asarray(gnss_sigma_mm, dtype=float)
    crlb = np.asarray(crlb_sigma_mm, dtype=float)
    posterior = np.asarray(posterior_sigma_mm, dtype=float)
    if not (residual.shape == gnss.shape == crlb.shape == posterior.shape):
        raise ValueError("uncertainty and residual series must share a shape")
    methods = {
        "crlb_only": np.sqrt(gnss**2 + crlb**2),
        "posterior_only": np.sqrt(gnss**2 + posterior**2),
    }
    levels = {"68": (0.68, 1.0), "90": (0.90, 1.6448536269514722), "95": (0.95, 1.959963984540054)}
    output: dict[str, Any] = {}
    for name, sigma in methods.items():
        finite = np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0)
        intervals: dict[str, Any] = {}
        for label, (nominal, z) in levels.items():
            evaluated = int(np.sum(finite))
            lower = -z * sigma[finite]
            upper = z * sigma[finite]
            observed = residual[finite]
            alpha = 1.0 - nominal
            score = (
                upper
                - lower
                + (2.0 / alpha) * (lower - observed) * (observed < lower)
                + (2.0 / alpha) * (observed - upper) * (observed > upper)
            )
            intervals[label] = {
                    "nominal": nominal,
                    "covered": int(np.sum(np.abs(observed) <= z * sigma[finite])),
                    "evaluated": evaluated,
                    "abstained": int(residual.size - evaluated),
                    "coverage": float(np.mean(np.abs(observed) <= z * sigma[finite]))
                    if evaluated
                    else None,
                    "coverage_error": float(
                        abs(np.mean(np.abs(observed) <= z * sigma[finite]) - nominal)
                    )
                    if evaluated
                    else None,
                    "mean_width_mm": float(np.mean(upper - lower)) if evaluated else None,
                    "mean_interval_score": float(np.mean(score)) if evaluated else None,
                }
        output[name] = {
            "sigma_mm": sigma.tolist(),
            "intervals": intervals,
        }
    return output


def comparison_contract(recipe: dict[str, Any]) -> dict[str, str]:
    comparison = recipe.get("comparison")
    if comparison is None:
        comparison = {
            "id": "MMX1_minus_ICMX_common_frame",
            "fixture": "mmx1_icmx_common",
            "primary_station": "MMX1",
            "control_station": "ICMX",
        }
    required = {"id", "fixture", "primary_station", "control_station"}
    missing = required - set(comparison)
    if missing:
        raise NotEvaluable(f"comparison contract is missing: {sorted(missing)}")
    station_ids = {comparison["primary_station"], comparison["control_station"]}
    if len(station_ids) != 2 or not station_ids.issubset(recipe["stations"]):
        raise NotEvaluable("comparison stations must be distinct recipe stations")
    return {key: str(comparison[key]) for key in required}


def write_reliability_artifacts(path: Path, payload: dict[str, Any]) -> None:
    """Write flat CSV and dependency-free SVG coverage artifacts."""
    engines = payload["engines"]
    reliability = {
        "schema": "dolphinrust-uncertainty-reliability/1",
        "comparison": payload["comparison"],
        "dates": payload["dates"],
        "gnss_date_quality": payload["gnss_date_quality"],
        "limitations": payload["limitations"],
        "engines": {
            name: {
                "gnss_projected_sigma_mm": data["gnss_projected_sigma_mm"],
                "velocity_comparison": data["velocity_comparison"],
                "uncertainty_reliability": data["uncertainty_reliability"],
            }
            for name, data in engines.items()
        },
    }
    (path / "uncertainty_reliability.json").write_text(
        json.dumps(reliability, indent=2, allow_nan=False) + "\n"
    )
    with (path / "uncertainty_reliability.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["engine", "method", "interval_percent", "nominal", "covered", "evaluated", "coverage"])
        for engine, data in engines.items():
            for method, result in data["uncertainty_reliability"].items():
                for level, interval in result["intervals"].items():
                    writer.writerow([engine, method, level, interval["nominal"], interval["covered"], interval["evaluated"], interval["coverage"]])
    width, height, margin = 980, 540, 70
    entries = [
        (engine, method, level, result["coverage"])
        for engine, data in engines.items()
        for method, method_data in data["uncertainty_reliability"].items()
        for level, result in method_data["intervals"].items()
    ]
    bar_width = (width - 2 * margin) / max(1, len(entries))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="70" y="28" font-family="sans-serif" font-size="18">{payload["comparison"]} uncertainty interval coverage</text>',
    ]
    colors = {"crlb_only": "#2563eb", "posterior_only": "#dc2626"}
    for index, (engine, method, level, coverage) in enumerate(entries):
        value = 0.0 if coverage is None else coverage
        x = margin + index * bar_width
        y = height - margin - value * (height - 2 * margin)
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bar_width - 1):.1f}" height="{value * (height - 2 * margin):.1f}" fill="{colors[method]}"/>')
        lines.append(f'<text transform="translate({x + bar_width / 2:.1f},{height-margin+8}) rotate(65)" font-family="sans-serif" font-size="8">{engine}:{method}:{level}</text>')
    lines.extend([
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#6b7280"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#6b7280"/>',
        '</svg>',
    ])
    (path / "uncertainty_reliability.svg").write_text("\n".join(lines) + "\n")


def write_csv(path: Path, dates: Sequence[str], gnss: np.ndarray, engines: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "gnss_diff_mm", *[f"{engine}_insar_diff_mm" for engine in engines]])
        for index, date in enumerate(dates):
            writer.writerow([date, float(gnss[index]), *[engine["insar_diff_mm"][index] for engine in engines.values()]])


def write_svg(
    path: Path,
    dates: Sequence[str],
    gnss: np.ndarray,
    engines: dict[str, dict[str, Any]],
    comparison: str = "station-pair",
) -> None:
    series = [("GNSS", np.asarray(gnss), "#111827")]
    colors = ["#2563eb", "#dc2626"]
    series.extend((name, np.asarray(data["insar_diff_mm"]), colors[index % len(colors)]) for index, (name, data) in enumerate(engines.items()))
    all_values = np.concatenate([values for _, values, _ in series])
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size == 0:
        raise NotEvaluable("plot has no finite station-pair values")
    low, high = float(finite_values.min()), float(finite_values.max())
    if high == low:
        high += 1.0
    width, height, margin = 900, 520, 60
    plot_width, plot_height = width - 2 * margin, height - 2 * margin

    def points(values: np.ndarray) -> str:
        coords = []
        for index, value in enumerate(values):
            if not math.isfinite(float(value)):
                continue
            x_value = margin + plot_width * index / max(1, len(values) - 1)
            y_value = margin + plot_height * (high - float(value)) / (high - low)
            coords.append(f"{x_value:.1f},{y_value:.1f}")
        return " ".join(coords)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#6b7280"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#6b7280"/>',
        f'<text x="{margin}" y="28" font-family="sans-serif" font-size="18">{comparison} LOS displacement (mm)</text>',
        f'<text x="8" y="{margin}" font-family="sans-serif" font-size="12">{high:.1f}</text>',
        f'<text x="8" y="{height-margin}" font-family="sans-serif" font-size="12">{low:.1f}</text>',
    ]
    for index, (name, values, color) in enumerate(series):
        lines.append(f'<polyline points="{points(values)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{width-margin-120}" y="{margin+18*index}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>')
    lines.append(f'<text x="{margin}" y="{height-15}" font-family="sans-serif" font-size="11">{dates[0]} to {dates[-1]}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def score_common_frame(
    recipe: dict[str, Any],
    fixture_manifest: dict[str, Any],
    work_directories: dict[str, Path],
    static_path: Path,
    cache_directory: Path,
    output_directory: Path,
    wavelength_m: float,
) -> dict[str, Any]:
    comparison = comparison_contract(recipe)
    if fixture_manifest.get("fixture") != comparison["fixture"]:
        raise NotEvaluable("magnitude scoring requires the declared shared station frame")
    primary_station = comparison["primary_station"]
    control_station = comparison["control_station"]
    station_ids = [primary_station, control_station]
    dates = [dt.date.fromisoformat(value) for value in recipe["expected_dates"]]
    aligned_by_station: dict[str, list[AlignedRecord | None]] = {}
    source_provenance: dict[str, Any] = {}
    for station_id in station_ids:
        station = recipe["stations"][station_id]
        text, source = fetch_text(station["tenv3_url"], cache_directory / f"{station_id}.tenv3")
        metadata_text, metadata_source = fetch_text(station["metadata_url"], cache_directory / f"{station_id}.sta.html")
        records = parse_tenv3(text)
        if records[0].station != station_id:
            raise NotEvaluable(
                f"GNSS source for {station_id} contains station {records[0].station}"
            )
        reference = min(records, key=lambda record: abs((record.date - dates[0]).days))
        if abs(reference.latitude - station["latitude"]) > 0.002 or abs(reference.longitude - station["longitude"]) > 0.002:
            raise NotEvaluable(f"{station_id} authoritative coordinates conflict with recipe")
        aligned_by_station[station_id] = align_records_available(
            records, dates, recipe["max_interpolation_gap_days"]
        )
        source_provenance[station_id] = {"tenv3": source, "metadata": metadata_source, "metadata_bytes": len(metadata_text.encode())}

    common_gnss = np.array(
        [
            all(aligned_by_station[station_id][index] is not None for station_id in station_ids)
            for index in range(len(dates))
        ],
        dtype=bool,
    )
    required_fraction = float(recipe.get("minimum_gnss_fraction", 1.0))
    if float(np.mean(common_gnss)) < required_fraction:
        raise NotEvaluable(
            f"common GNSS availability {float(np.mean(common_gnss)):.3f} is below {required_fraction:.3f}"
        )
    if not common_gnss[0]:
        raise NotEvaluable("the reference acquisition lacks common GNSS truth")

    engines: dict[str, dict[str, Any]] = {}
    station_geometry: dict[str, Any] | None = None
    gnss_diff: np.ndarray | None = None
    for engine, work_directory in work_directories.items():
        cube, grid = load_displacement_cube(work_directory, len(dates))
        first_raster = sorted(work_directory.glob("displacement_[0-9][0-9].tif"))[0]
        with rasterio.open(first_raster) as dataset:
            station_data: dict[str, Any] = {}
            gnss_station: dict[str, np.ndarray] = {}
            gnss_station_sigma: dict[str, np.ndarray] = {}
            for station_id in station_ids:
                station = recipe["stations"][station_id]
                row, col = station_pixel(dataset, station["longitude"], station["latitude"])
                los = los_at_output_pixel(static_path, dataset, row, col)
                samples = sample_cube(
                    cube,
                    row,
                    col,
                    recipe["sample_windows"],
                    recipe["minimum_finite_fraction"],
                )
                coherence_path = work_directory / "temporal_coherence.tif"
                coherence = None
                if coherence_path.exists():
                    with rasterio.open(coherence_path) as coherence_dataset:
                        coherence = window_stats(
                            coherence_dataset.read(1), row, col, recipe["primary_window"], recipe["minimum_finite_fraction"]
                        ).mean
                station_data[station_id] = {
                    "pixel": [row, col],
                    "los_east_north_up": los.tolist(),
                    "temporal_coherence": coherence,
                    "samples": {str(size): data for size, data in samples.items()},
                    "uncertainty": sample_uncertainty_layers(
                        work_directory,
                        row,
                        col,
                        recipe["primary_window"],
                        recipe["minimum_finite_fraction"],
                        len(dates),
                        wavelength_m,
                    ),
                }
                available = [
                    item
                    for index, item in enumerate(aligned_by_station[station_id])
                    if common_gnss[index] and item is not None
                ]
                station_series = np.full(len(dates), np.nan)
                station_sigma = np.full(len(dates), np.nan)
                station_series[common_gnss] = gnss_los_series(available, los)
                station_sigma[common_gnss] = gnss_los_sigma_series(available, los)
                gnss_station[station_id] = station_series
                gnss_station_sigma[station_id] = station_sigma
            current_gnss_diff = spatial_difference(
                gnss_station[primary_station], gnss_station[control_station]
            )
            if gnss_diff is None:
                gnss_diff = current_gnss_diff
                station_geometry = station_data
            elif not np.allclose(current_gnss_diff, gnss_diff, atol=1e-6, equal_nan=True):
                raise NotEvaluable("backend output grids produce different station LOS geometry")
            primary = str(recipe["primary_window"])
            insar_diff = spatial_difference(
                np.asarray(station_data[primary_station]["samples"][primary]["series_mm"]),
                np.asarray(station_data[control_station]["samples"][primary]["series_mm"]),
            )
            gnss_diff_sigma = np.sqrt(
                gnss_station_sigma[primary_station] ** 2
                + gnss_station_sigma[control_station] ** 2
            )
            crlb_diff_sigma = np.sqrt(
                np.asarray(station_data[primary_station]["uncertainty"]["crlb_sigma_mm"]) ** 2
                + np.asarray(station_data[control_station]["uncertainty"]["crlb_sigma_mm"]) ** 2
            )
            posterior_diff_sigma = np.sqrt(
                np.asarray(station_data[primary_station]["uncertainty"]["posterior_sigma_mm"]) ** 2
                + np.asarray(station_data[control_station]["uncertainty"]["posterior_sigma_mm"]) ** 2
            )
            days = np.array([(date - dates[0]).days for date in dates], dtype=float)
            insar_velocity = float(
                np.polyfit(days[common_gnss], insar_diff[common_gnss], 1)[0] * 365.25
            )
            gnss_velocity = float(
                np.polyfit(days[common_gnss], current_gnss_diff[common_gnss], 1)[0]
                * 365.25
            )
            velocity_sigma = math.sqrt(
                station_data[primary_station]["uncertainty"]["velocity_sigma_mm_yr"] ** 2
                + station_data[control_station]["uncertainty"]["velocity_sigma_mm_yr"] ** 2
            )
            engines[engine] = {
                "work_directory": str(work_directory),
                "grid": grid,
                "spatial_reference": infer_reference_pixel(
                    cube, work_directory / "temporal_coherence.tif"
                ),
                "stations": station_data,
                "insar_diff_mm": insar_diff.tolist(),
                "metrics": compute_metrics(
                    gnss_diff[common_gnss], insar_diff[common_gnss], recipe["thresholds"]
                ),
                "unwrap_component_counts": unwrap_component_counts(work_directory),
                "gnss_projected_sigma_mm": gnss_diff_sigma.tolist(),
                "uncertainty_reliability": uncertainty_reliability(
                    insar_diff - current_gnss_diff,
                    gnss_diff_sigma,
                    crlb_diff_sigma,
                    posterior_diff_sigma,
                ),
                "velocity_comparison": {
                    "insar_velocity_mm_yr": insar_velocity,
                    "insar_sigma_mm_yr": velocity_sigma,
                    "gnss_velocity_mm_yr": gnss_velocity,
                    "difference_mm_yr": insar_velocity - gnss_velocity,
                },
            }
    if gnss_diff is None:
        raise NotEvaluable("no backend outputs supplied for scoring")
    statuses = [engine["metrics"]["status"] for engine in engines.values()]
    overall = "pass" if statuses and all(status == "pass" for status in statuses) else "fail"
    backend_difference = None
    if "native" in engines and "snaphu" in engines:
        difference = np.asarray(engines["native"]["insar_diff_mm"]) - np.asarray(
            engines["snaphu"]["insar_diff_mm"]
        )
        finite_difference = difference[np.isfinite(difference)]
        if finite_difference.size == 0:
            raise NotEvaluable("native/SNAPHU station-pair difference is not finite")
        backend_difference = {
            "series_mm": difference.tolist(),
            "endpoint_mm": float(finite_difference[-1]),
            "mae_mm": float(np.mean(np.abs(finite_difference))),
            "rmse_mm": float(np.sqrt(np.mean(finite_difference**2))),
        }
    weighting_ab: dict[str, Any] = {}
    for backend in ["native", "snaphu"]:
        unweighted = f"{backend}_unweighted"
        if backend in engines and unweighted in engines:
            weighting_ab[backend] = {
                "velocity_delta": raster_delta_summary(
                    Path(engines[backend]["work_directory"]) / "velocity.tif",
                    Path(engines[unweighted]["work_directory"]) / "velocity.tif",
                ),
                "component_counts_weighted": engines[backend]["unwrap_component_counts"],
                "component_counts_unweighted": engines[unweighted]["unwrap_component_counts"],
                "gnss_metrics_weighted": engines[backend]["metrics"],
                "gnss_metrics_unweighted": engines[unweighted]["metrics"],
                "reference_weighted": engines[backend]["spatial_reference"],
                "reference_unweighted": engines[unweighted]["spatial_reference"],
            }
    payload = {
        "schema": "dolphinrust-gps-ground-truth/1",
        "status": overall,
        "comparison": comparison["id"],
        "units": "millimeters_ground_to_sensor_los",
        "sign_convention": "negative_is_motion_away_from_sensor",
        "dates": recipe["expected_dates"],
        "thresholds": recipe["thresholds"],
        "gnss_diff_mm": gnss_diff.tolist(),
        "gnss_date_quality": {
            station_id: [
                item.quality if item is not None else "unavailable" for item in aligned
            ]
            for station_id, aligned in aligned_by_station.items()
        },
        "common_gnss_fraction": float(np.mean(common_gnss)),
        "gnss_sources": source_provenance,
        "station_geometry": station_geometry,
        "engines": engines,
        "native_minus_snaphu": backend_difference,
        "weighted_unweighted_ab": weighting_ab,
        "limitations": [
            "one Sentinel-1 burst pair of stations",
            f"{len(dates)} acquisition epochs",
            "atmospheric corrections are disabled",
            "coverage samples are temporally correlated and are not an independent population",
        ],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / "gps_ground_truth.csv", recipe["expected_dates"], gnss_diff, engines)
    write_svg(
        output_directory / "gps_ground_truth.svg",
        recipe["expected_dates"],
        gnss_diff,
        engines,
        comparison["id"],
    )
    safe_payload = json_safe(payload)
    (output_directory / "gps_ground_truth.json").write_text(
        json.dumps(safe_payload, indent=2, allow_nan=False) + "\n"
    )
    write_reliability_artifacts(output_directory, safe_payload)
    return safe_payload
