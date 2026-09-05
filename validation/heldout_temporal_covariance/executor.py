"""Bounded one-shot primitives for one frozen held-out cluster."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import requests
import rasterio
from pyproj import Transformer

if __package__ and __package__.startswith("validation."):
    from validation.gps_ground_truth import (
        AlignedRecord,
        Tenv3Record,
        gnss_los_covariance_series,
        gnss_los_series,
        parse_tenv3,
    )
else:
    from gps_ground_truth import (
        AlignedRecord,
        Tenv3Record,
        gnss_los_covariance_series,
        gnss_los_series,
        parse_tenv3,
    )

from .cohort import canonical_digest, validate_manifest
from .runner import run_production_temporal_estimator


class Attrition(RuntimeError):
    """A preregistered scientific attrition disposition for one cluster."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


NGL_TENV3_BASE_URL = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20"
NGL_BYTE_CAP = 64 * 1024 * 1024
JSON_BYTE_CAP = 1024 * 1024
FACTOR_BYTE_CAP = 1024 * 1024 * 1024
FACTOR_SLICE_BYTE_CAP = 64 * 1024 * 1024
RASTER_BYTE_CAP = 4 * 1024 * 1024 * 1024
FACTOR_EVIDENCE_FILES = {
    "approximation_receipt_digest": "referenced_displacement_covariance_approximation_receipt.json",
    "resource_receipt_digest": "referenced_displacement_covariance_resource_receipt.json",
    "review_receipt_digest": "referenced_displacement_covariance_review_receipt.json",
    "method_manifest_digest": "referenced_displacement_covariance_method_manifest.json",
}
FACTOR_REQUIRED_EVIDENCE = (
    "referenced_displacement_covariance_approximation_result.json",
    "referenced_displacement_covariance_preregistration.json",
    "referenced_displacement_covariance_design.md",
    "referenced_displacement_covariance_producer_binary",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _hash_open_file(source: Any) -> str:
    source.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    source.seek(0)
    return digest.hexdigest()


def _open_snapshot(path: Path, byte_cap: int) -> tuple[Any, os.stat_result, str]:
    try:
        source = path.open("rb")
        before = os.fstat(source.fileno())
        if before.st_size <= 0 or before.st_size > byte_cap:
            source.close()
            raise ValueError("artifact size is outside its byte cap")
        digest = _hash_open_file(source)
        if _descriptor_identity(before) != _descriptor_identity(os.fstat(source.fileno())):
            source.close()
            raise ValueError("artifact changed while it was hashed")
        return source, before, digest
    except OSError as error:
        raise ValueError("artifact cannot be opened") from error


def _hdf5_text(group: h5py.Group, name: str) -> str:
    value = group[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8").strip("\x00")
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.tobytes().decode("utf-8").strip("\x00")
    return str(value)


def _bare_digest(value: Any, field: str) -> str:
    text = str(value)
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise Attrition("factor_identity_mismatch", f"{field} is not a SHA-256")
    return text


def select_manifest_cluster(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    cluster_id: str,
) -> dict[str, Any]:
    validate_manifest(manifest, preregistration)
    selected = manifest["frozen_clusters"] + manifest["surplus_clusters"]
    matches = [candidate for candidate in selected if candidate["candidate_id"] == cluster_id]
    if len(matches) != 1:
        raise ValueError("cluster ID is not one exact frozen primary or surplus cluster")
    return dict(matches[0])


def exact_common_dates(
    candidate: Mapping[str, Any],
    acquisition_dates: Sequence[dt.date],
    records_by_station: Mapping[str, Sequence[Tenv3Record]],
    preregistration: Mapping[str, Any],
) -> list[dt.date]:
    dates = list(acquisition_dates)
    if dates != sorted(set(dates)):
        raise Attrition("dates_not_strictly_increasing", "acquisition dates are not unique and increasing")
    if (
        len(dates) != candidate["epoch_count"]
        or not dates
        or dates[0].isoformat() != candidate["date_start"]
        or dates[-1].isoformat() != candidate["date_end"]
    ):
        raise Attrition("factor_identity_mismatch", "acquisition dates differ from frozen metadata")
    station_ids = candidate["station_ids"]
    if set(records_by_station) != set(station_ids):
        raise Attrition("gnss_solution_missing", "GNSS station identities differ from the frozen pair")
    available = [
        {record.date for record in records_by_station[station_id]}
        for station_id in station_ids
    ]
    common = [date for date in dates if all(date in values for values in available)]
    minimum = preregistration["eligibility"]["minimum_common_epochs"]
    if len(common) < minimum or not common or common[0] != dates[0]:
        raise Attrition("insufficient_dates", "exact common GNSS/acquisition epochs do not meet the frozen minimum")
    gaps = [(right - left).days for left, right in zip(common, common[1:])]
    if gaps:
        minimum_gap, maximum_median_gap = preregistration["eligibility"]["median_gap_days"]
        if not minimum_gap <= median(gaps) <= maximum_median_gap or max(gaps) > preregistration["eligibility"]["maximum_gap_days"]:
            raise Attrition("unsupported_cadence", "exact common epochs violate the frozen cadence bounds")
    return common


def fit_slope_with_covariance(
    dates: Sequence[dt.date],
    values: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
) -> dict[str, Any]:
    if len(dates) < 2 or list(dates) != sorted(set(dates)):
        raise ValueError("slope dates must be unique and strictly increasing")
    observed = np.asarray(values, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if observed.shape != (len(dates),) or matrix.shape != (len(dates), len(dates)):
        raise ValueError("slope values/covariance do not match the common dates")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(matrix)):
        raise ValueError("slope inputs contain non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError("slope covariance is not symmetric")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if float(np.min(np.linalg.eigvalsh(matrix))) < -scale * 1e-10:
        raise ValueError("slope covariance is not positive semidefinite")
    years = np.array([(date - dates[0]).days / 365.25 for date in dates], dtype=float)
    design = np.column_stack((np.ones(len(dates)), years))
    inverse = np.linalg.inv(design.T @ design)
    projection = inverse @ design.T
    coefficients = projection @ observed
    slope_weights = projection[1]
    variance = float(slope_weights @ matrix @ slope_weights)
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("slope variance is invalid")
    return {
        "slope": float(coefficients[1]),
        "intercept": float(coefficients[0]),
        "variance": variance,
        "design_sha256": canonical_digest(
            {
                "dates": [date.isoformat() for date in dates],
                "years_from_epoch_zero": years.tolist(),
                "method": "unweighted_ols_with_full_covariance_propagation_v1",
            }
        ),
    }


def fit_origin_anchored_gls(
    dates: Sequence[dt.date],
    values: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
) -> dict[str, Any]:
    """Fit the preregistered post-gauge station-pair slope by GLS."""

    observed = np.asarray(values, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if (
        len(dates) < 2
        or list(dates) != sorted(set(dates))
        or observed.shape != (len(dates),)
        or matrix.shape != (len(dates), len(dates))
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("origin-anchored GLS inputs are invalid")
    if observed[0] != 0.0 or np.any(matrix[0] != 0.0) or np.any(matrix[:, 0] != 0.0):
        raise ValueError("origin-anchored GLS gauge row is not exact zero")
    post = matrix[1:, 1:]
    if not np.allclose(post, post.T, rtol=0.0, atol=1e-10):
        raise ValueError("origin-anchored GLS covariance is not symmetric")
    scale = max(1.0, float(np.max(np.abs(post))))
    eigenvalues = np.linalg.eigvalsh(post)
    if eigenvalues[0] <= scale * 1e-12:
        raise ValueError("origin-anchored GLS covariance is not positive definite")
    years = np.array(
        [(date - dates[0]).days / 365.25 for date in dates[1:]], dtype=float
    )
    solved_design = np.linalg.solve(post, years)
    denominator = float(years @ solved_design)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("origin-anchored GLS information is not positive")
    slope = float(solved_design @ observed[1:] / denominator)
    return {
        "slope": slope,
        "intercept": 0.0,
        "variance": 1.0 / denominator,
        "design_sha256": canonical_digest(
            {
                "dates": [date.isoformat() for date in dates],
                "years_from_epoch_zero": [0.0, *years.tolist()],
                "method": "origin_anchored_gls_full_covariance_v1",
            }
        ),
    }


def station_pair_gnss_observation(
    first_records: Sequence[Tenv3Record],
    second_records: Sequence[Tenv3Record],
    dates: Sequence[dt.date],
    first_los: np.ndarray,
    second_los: np.ndarray,
) -> dict[str, Any]:
    aligned: list[list[AlignedRecord]] = []
    for records in (first_records, second_records):
        by_date = {record.date: record for record in records}
        if any(date not in by_date for date in dates):
            raise Attrition("gnss_solution_missing", "an exact common GNSS epoch is missing")
        aligned.append([AlignedRecord(by_date[date], "exact") for date in dates])
    first_series = gnss_los_series(aligned[0], first_los)
    second_series = gnss_los_series(aligned[1], second_los)
    difference = first_series - second_series
    try:
        covariance = gnss_los_covariance_series(
            aligned[0], first_los
        ) + gnss_los_covariance_series(aligned[1], second_los)
        fitted = fit_origin_anchored_gls(dates, difference, covariance)
    except ValueError as error:
        raise Attrition("gnss_covariance_missing", str(error)) from error
    return {
        "slope_mm_year": fitted["slope"],
        "slope_variance": fitted["variance"],
        "design_sha256": fitted["design_sha256"],
        "difference_series_sha256": canonical_digest(difference.tolist()),
        "covariance_sha256": canonical_digest(covariance.tolist()),
        "covariance_projection": "u_transpose_C_u",
    }


def fetch_ngl_pair(
    candidate: Mapping[str, Any],
    session: requests.Session | None = None,
    base_url: str = NGL_TENV3_BASE_URL,
) -> tuple[dict[str, list[Tenv3Record]], dict[str, Any]]:
    client = session or requests.Session()
    records_by_station: dict[str, list[Tenv3Record]] = {}
    sources: dict[str, Any] = {}
    for station_id in candidate["station_ids"]:
        url = f"{base_url.rstrip('/')}/{station_id}.tenv3"
        try:
            response = client.get(url, timeout=60, stream=True)
            response.raise_for_status()
        except requests.RequestException as error:
            raise Attrition("gnss_solution_missing", f"NGL solution retrieval failed for {station_id}") from error
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None:
            try:
                if int(declared_length) > NGL_BYTE_CAP:
                    raise Attrition("gnss_solution_missing", f"NGL solution exceeds byte cap for {station_id}")
            except ValueError as error:
                raise Attrition("gnss_solution_missing", f"NGL content length is invalid for {station_id}") from error
        chunks: list[bytes] = []
        transferred = 0
        for chunk in response.iter_content(64 * 1024):
            transferred += len(chunk)
            if transferred > NGL_BYTE_CAP:
                raise Attrition("gnss_solution_missing", f"NGL solution exceeds byte cap for {station_id}")
            chunks.append(chunk)
        payload = b"".join(chunks)
        try:
            text = payload.decode("utf-8")
            records = parse_tenv3(text)
        except (UnicodeDecodeError, ValueError) as error:
            code = "gnss_covariance_missing" if "covariance" in str(error) or "correlation" in str(error) else "gnss_solution_missing"
            raise Attrition(code, f"NGL solution is invalid for {station_id}") from error
        if records[0].station != station_id:
            raise Attrition("gnss_solution_missing", f"NGL solution station identity differs for {station_id}")
        records_by_station[station_id] = records
        sources[station_id] = {
            "url": url,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return records_by_station, sources


def _grid(group: h5py.Group) -> dict[str, int]:
    return {
        field: int(group.attrs[field])
        for field in ("row_start", "col_start", "rows", "cols", "stride_y", "stride_x")
    }


def station_pixel_from_grid(
    longitude: float,
    latitude: float,
    crs: str,
    geotransform: Sequence[float],
    rows: int,
    cols: int,
) -> tuple[int, int]:
    """Map one GNSS coordinate to the exact production raster pixel."""

    if len(geotransform) != 6 or rows <= 0 or cols <= 0:
        raise Attrition("factor_identity_mismatch", "production grid identity is invalid")
    try:
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(
            longitude, latitude
        )
    except (ValueError, TypeError) as error:
        raise Attrition("factor_identity_mismatch", "production CRS is invalid") from error
    x0, a, b, y0, d, e = [float(value) for value in geotransform]
    determinant = a * e - b * d
    if not all(math.isfinite(value) for value in (x, y, x0, a, b, y0, d, e)) or determinant == 0.0:
        raise Attrition("factor_identity_mismatch", "production affine is invalid")
    col_value = (e * (x - x0) - b * (y - y0)) / determinant
    row_value = (-d * (x - x0) + a * (y - y0)) / determinant
    row, col = math.floor(row_value), math.floor(col_value)
    if row < 0 or col < 0 or row >= rows or col >= cols:
        raise Attrition("difference_covariance_unavailable", "GNSS station lies outside the production factor grid")
    return row, col


def _bounded_json(path: Path, byte_cap: int = JSON_BYTE_CAP) -> tuple[dict[str, Any], bytes]:
    try:
        source, before, _ = _open_snapshot(path, byte_cap)
        with source:
            payload = source.read(byte_cap + 1)
            after = os.fstat(source.fileno())
        if len(payload) > byte_cap or _descriptor_identity(before) != _descriptor_identity(after):
            raise ValueError("JSON changed while it was read")
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Attrition("factor_identity_mismatch", f"bounded JSON is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise Attrition("factor_identity_mismatch", f"bounded JSON is not an object: {path.name}")
    return value, payload


def _validate_factor_evidence(directory: Path, metadata: h5py.Group) -> None:
    for digest_name, file_name in FACTOR_EVIDENCE_FILES.items():
        value, payload = _bounded_json(directory / file_name)
        del value
        expected = _bare_digest(_hdf5_text(metadata, digest_name), digest_name)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise Attrition("difference_covariance_uncalibrated", f"factor evidence hash differs: {file_name}")
    for file_name in FACTOR_REQUIRED_EVIDENCE:
        path = directory / file_name
        try:
            size = path.stat().st_size
        except OSError as error:
            raise Attrition("difference_covariance_uncalibrated", f"factor evidence is missing: {file_name}") from error
        cap = 512 * 1024 * 1024 if file_name.endswith("producer_binary") else JSON_BYTE_CAP
        if size <= 0 or size > cap:
            raise Attrition("difference_covariance_uncalibrated", f"factor evidence size is invalid: {file_name}")


def _raster_sample(
    path: Path,
    pixel: tuple[int, int],
    *,
    rows: int,
    cols: int,
    crs: str,
    geotransform: Sequence[float],
) -> tuple[float, str]:
    try:
        source, before, digest = _open_snapshot(path, RASTER_BYTE_CAP)
        with source:
            with rasterio.open(source) as dataset:
                if (
                    dataset.count != 1
                    or dataset.height != rows
                    or dataset.width != cols
                    or dataset.crs is None
                    or dataset.crs.to_string() != rasterio.crs.CRS.from_string(crs).to_string()
                    or tuple(dataset.transform.to_gdal()) != tuple(float(value) for value in geotransform)
                ):
                    raise Attrition("factor_identity_mismatch", "production raster grid differs from the factor")
                row, col = pixel
                value = float(dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
            after = os.fstat(source.fileno())
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise Attrition("factor_identity_mismatch", "production raster changed while it was read")
    except Attrition:
        raise
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        raise Attrition("difference_covariance_unavailable", "production raster is unreadable") from error
    if not math.isfinite(value):
        raise Attrition("difference_covariance_unavailable", "production raster sample is non-finite")
    return value, digest


def derive_product_observations(
    product_directory: Path,
    candidate: Mapping[str, Any],
    records_by_station: Mapping[str, Sequence[Tenv3Record]],
    factor_header: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive pixels, LOS, and station-pair displacement from production bytes."""

    receipt, receipt_bytes = _bounded_json(product_directory / "fixed_cube_receipt.json")
    acquisition_days = [float(value) for value in factor_header["acquisition_days"]]
    rows = int(factor_header["full_grid"]["rows"])
    cols = int(factor_header["full_grid"]["cols"])
    geotransform = [float(value) for value in factor_header["geotransform"]]
    crs = str(factor_header["crs"])
    epsg = rasterio.crs.CRS.from_string(crs).to_epsg()
    fixed = preregistration["fixed_cube_binding"]
    required_receipt = {
        "contract_version": fixed["contract_version"],
        "acquisition_days": acquisition_days,
        "rows": rows,
        "cols": cols,
        "geotransform": geotransform,
        "epsg": epsg,
        "geometry_source": fixed["geometry_source"],
        "los_rasters": ["los_east.tif", "los_north.tif", "los_up.tif"],
        "acquisition_days_sha256": "sha256:"
        + hashlib.sha256(
            b"".join(struct.pack("<d", value) for value in acquisition_days)
        ).hexdigest(),
        "velocity_estimator": fixed["velocity_estimator"],
        "inference_status": fixed["inference_status"],
        "corrected_velocity_raster": None,
        "corrected_sigma_raster": None,
        "validity_mask_raster": fixed["validity_mask_raster"],
        "geometry_provenance": fixed["geometry_provenance"],
    }
    if any(receipt.get(field) != value for field, value in required_receipt.items()):
        raise Attrition("factor_identity_mismatch", "fixed-cube receipt differs from factor scope")
    if factor_header["units"] != fixed["displacement_units"]:
        raise Attrition(
            "factor_identity_mismatch",
            "factor units differ from the frozen fixed-cube displacement units",
        )
    station_pixels: dict[str, tuple[int, int]] = {}
    station_los: dict[str, np.ndarray] = {}
    raster_hashes: dict[str, str] = {
        "fixed_cube_receipt.json": hashlib.sha256(receipt_bytes).hexdigest()
    }
    _, geometry_bytes = _bounded_json(product_directory / receipt["geometry_provenance"])
    raster_hashes[receipt["geometry_provenance"]] = hashlib.sha256(
        geometry_bytes
    ).hexdigest()
    for station_id in candidate["station_ids"]:
        records = records_by_station.get(station_id)
        if not records:
            raise Attrition("gnss_solution_missing", "GNSS station solution is empty")
        record = records[0]
        pixel = station_pixel_from_grid(
            record.longitude,
            record.latitude,
            crs,
            geotransform,
            rows,
            cols,
        )
        station_pixels[station_id] = pixel
        validity, validity_digest = _raster_sample(
            product_directory / receipt["validity_mask_raster"],
            pixel,
            rows=rows,
            cols=cols,
            crs=crs,
            geotransform=geotransform,
        )
        if validity != 1.0:
            raise Attrition(
                "difference_covariance_unavailable",
                "GNSS station lies outside the fixed-cube validity mask",
            )
        raster_hashes[receipt["validity_mask_raster"]] = validity_digest
        components = []
        for name in receipt["los_rasters"]:
            value, digest = _raster_sample(
                product_directory / name,
                pixel,
                rows=rows,
                cols=cols,
                crs=crs,
                geotransform=geotransform,
            )
            components.append(value)
            raster_hashes[name] = digest
        vector = np.asarray(components, dtype=float)
        if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-5:
            raise Attrition("sourced_los_invalid", "fixed-cube LOS vector is not unit norm")
        station_los[station_id] = vector
    if receipt.get("reference_point") != list(
        station_pixels[candidate["station_ids"][1]]
    ):
        raise Attrition(
            "factor_identity_mismatch",
            "fixed-cube reference point is not the derived control station pixel",
        )
    displacement_paths = sorted(product_directory.glob("displacement_[0-9][0-9].tif"))
    if len(displacement_paths) != len(acquisition_days) - 1:
        raise Attrition("factor_identity_mismatch", "fixed-cube displacement count differs from acquisition dates")
    first_id, second_id = candidate["station_ids"]
    difference = [0.0]
    for path in displacement_paths:
        first, digest = _raster_sample(
            path,
            station_pixels[first_id],
            rows=rows,
            cols=cols,
            crs=crs,
            geotransform=geotransform,
        )
        second, second_digest = _raster_sample(
            path,
            station_pixels[second_id],
            rows=rows,
            cols=cols,
            crs=crs,
            geotransform=geotransform,
        )
        if digest != second_digest:
            raise Attrition("factor_identity_mismatch", "production raster changed between station reads")
        raster_hashes[path.name] = digest
        difference.append((first - second) * 1000.0)
    return {
        "acquisition_days": acquisition_days,
        "station_pixels": station_pixels,
        "station_los": station_los,
        "insar_difference_mm": difference,
        "source_sha256": canonical_digest(raster_hashes),
        "raster_hashes": raster_hashes,
    }


def _block_factor(
    blocks: h5py.Group, pixel: tuple[int, int]
) -> tuple[np.ndarray, int, float | None]:
    row, col = pixel
    for name in sorted(blocks):
        block = blocks[name]
        grid = _grid(block["target_grid"])
        row_delta = row - grid["row_start"]
        col_delta = col - grid["col_start"]
        if row_delta < 0 or col_delta < 0:
            continue
        if row_delta % grid["stride_y"] or col_delta % grid["stride_x"]:
            continue
        grid_row = row_delta // grid["stride_y"]
        grid_col = col_delta // grid["stride_x"]
        if grid_row >= grid["rows"] or grid_col >= grid["cols"]:
            continue
        index = grid_row * grid["cols"] + grid_col
        if int(block["status"][index]) != 0:
            raise Attrition("difference_covariance_unavailable", "station pixel factor status is not valid")
        realized_rank = int(block["rank_by_target"][index])
        factor_dataset = block["difference_factor"]
        if len(factor_dataset.shape) != 3 or factor_dataset.shape[1] * factor_dataset.shape[2] * 8 > FACTOR_SLICE_BYTE_CAP:
            raise Attrition("difference_covariance_unavailable", "station pixel factor slice exceeds its byte cap")
        factor = np.asarray(factor_dataset[index], dtype=float)
        if realized_rank <= 0 or realized_rank > factor.shape[1]:
            raise Attrition("difference_covariance_invalid", "station pixel factor rank is invalid")
        if not np.all(factor[:, realized_rank:] == 0.0):
            raise Attrition("difference_covariance_invalid", "station pixel factor rank padding is nonzero")
        factor = factor[:, :realized_rank]
        condition = None
        if "condition_number" in block:
            condition = float(block["condition_number"][index])
        return factor, realized_rank, condition
    raise Attrition("difference_covariance_unavailable", "station pixel is absent from production factor blocks")


def read_production_factor_header(
    product_directory: Path,
    candidate: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    factor_path = product_directory / preregistration["factor_binding"]["output_factor"]["artifact_hdf5"]
    manifest_path = product_directory / preregistration["factor_binding"]["output_factor"]["artifact_manifest"]
    manifest, manifest_bytes = _bounded_json(manifest_path)
    try:
        source, before, factor_sha256 = _open_snapshot(factor_path, FACTOR_BYTE_CAP)
        with source, h5py.File(source, "r") as factor:
            required = preregistration["factor_binding"]["output_factor"]
            if (
                manifest.get("schema_version") != 3
                or manifest.get("method") != required["method"]
                or manifest.get("method_version") != required["method_version"]
                or manifest.get("hdf5_file") != factor_path.name
                or manifest.get("hdf5_bytes") != before.st_size
                or manifest.get("hdf5_sha256") != factor_sha256
                or manifest.get("burst_id") != candidate["burst_id"]
                or manifest.get("calibration_scope") != required["calibration_status"]
                or int(factor.attrs.get("schema_version", -1)) != required["schema_version"]
                or int(factor.attrs.get("method_version", -1)) != required["method_version"]
                or int(factor.attrs.get("gauge_date_index", -1)) != 0
                or int(factor.attrs.get("calibration_scope", -1)) != 1
                or int(factor.attrs.get("complete", 0)) != 1
            ):
                raise Attrition("factor_identity_mismatch", "factor header/manifest differs from frozen scope")
            metadata = factor["metadata"]
            _validate_factor_evidence(product_directory, metadata)
            acquisition_days = np.asarray(metadata["acquisition_days"][:], dtype=float)
            geotransform = np.asarray(metadata["geotransform"][:], dtype=float)
            full_grid = _grid(factor["full_grid"])
            header = {
                "acquisition_days": acquisition_days.tolist(),
                "geotransform": geotransform.tolist(),
                "crs": _hdf5_text(metadata, "crs"),
                "units": _hdf5_text(metadata, "units"),
                "full_grid": full_grid,
                "reference_pixel": [
                    int(metadata.attrs.get("reference_row", -1)),
                    int(metadata.attrs.get("reference_col", -1)),
                ],
                "factor_sha256": factor_sha256,
                "factor_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
            opened = os.fstat(source.fileno())
    except Attrition:
        raise
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise Attrition("difference_covariance_invalid", "production factor header is unreadable") from error
    if _descriptor_identity(before) != _descriptor_identity(opened):
        raise Attrition("factor_identity_mismatch", "factor changed while it was opened")
    days = header["acquisition_days"]
    if (
        not days
        or days[0] != 0.0
        or days != sorted(set(days))
        or not all(math.isfinite(value) for value in days)
        or len(header["geotransform"]) != 6
        or not header["crs"]
        or header["units"] not in {"meters", "millimeters"}
        or header["full_grid"]["row_start"] != 0
        or header["full_grid"]["col_start"] != 0
    ):
        raise Attrition("factor_identity_mismatch", "factor grid/date identity is invalid")
    return header


def read_production_difference_factor(
    factor_path: Path,
    factor_manifest_path: Path,
    candidate: Mapping[str, Any],
    target_pixel: tuple[int, int],
    control_pixel: tuple[int, int],
    common_dates: Sequence[dt.date],
    preregistration: Mapping[str, Any],
    operator_path: Path | None = None,
    operator_manifest_path: Path | None = None,
) -> dict[str, Any]:
    factor_manifest, factor_manifest_bytes = _bounded_json(factor_manifest_path)
    try:
        factor_source, factor_stat, factor_sha256 = _open_snapshot(
            factor_path, FACTOR_BYTE_CAP
        )
    except ValueError as error:
        raise Attrition(
            "difference_covariance_unavailable", "factor artifact is unavailable"
        ) from error
    required_output = preregistration["factor_binding"]["output_factor"]
    if factor_path.name != required_output["artifact_hdf5"] or factor_manifest_path.name != required_output["artifact_manifest"]:
        factor_source.close()
        raise Attrition("factor_identity_mismatch", "production factor filenames differ from frozen scope")
    required_manifest = {
        "schema_version": 3,
        "method": required_output["method"],
        "method_version": required_output["method_version"],
        "hdf5_file": required_output["artifact_hdf5"],
        "hdf5_bytes": factor_stat.st_size,
        "hdf5_sha256": factor_sha256,
        "burst_id": candidate["burst_id"],
        "calibration_scope": required_output["calibration_status"],
    }
    if not isinstance(factor_manifest, Mapping) or any(
        factor_manifest.get(field) != value for field, value in required_manifest.items()
    ):
        factor_source.close()
        raise Attrition("factor_identity_mismatch", "production factor provenance differs from frozen scope")
    if operator_path is None or operator_manifest_path is None:
        operator_sha256 = _bare_digest(factor_manifest.get("operator_sha256"), "operator_sha256")
        operator_manifest_sha256 = _bare_digest(
            factor_manifest.get("operator_manifest_sha256"), "operator_manifest_sha256"
        )
    else:
        required_operator = preregistration["factor_binding"]["input_operator"]
        if operator_path.name != required_operator["artifact_hdf5"] or operator_manifest_path.name != required_operator["artifact_manifest"]:
            factor_source.close()
            raise Attrition("factor_identity_mismatch", "operator filenames differ from frozen scope")
        try:
            operator_manifest, operator_manifest_bytes = _bounded_json(
                operator_manifest_path
            )
        except Attrition:
            factor_source.close()
            raise
        try:
            operator_source, operator_stat, operator_sha256 = _open_snapshot(
                operator_path, FACTOR_BYTE_CAP
            )
        except ValueError as error:
            factor_source.close()
            raise Attrition(
                "factor_identity_mismatch", "operator artifact is missing"
            ) from error
        with operator_source:
            operator_after = os.fstat(operator_source.fileno())
        if _descriptor_identity(operator_stat) != _descriptor_identity(operator_after):
            factor_source.close()
            raise Attrition("factor_identity_mismatch", "operator changed while read")
        operator_manifest_sha256 = hashlib.sha256(operator_manifest_bytes).hexdigest()
        operator_required = {
            "schema_version": 1,
            "method": required_operator["method"],
            "method_version": required_operator["method_version"],
            "gauge_date_index": 0,
            "hdf5_file": required_operator["artifact_hdf5"],
            "hdf5_bytes": operator_stat.st_size,
            "hdf5_sha256": operator_sha256,
        }
        if not isinstance(operator_manifest, Mapping) or any(
            operator_manifest.get(field) != value
            for field, value in operator_required.items()
        ):
            factor_source.close()
            raise Attrition("factor_identity_mismatch", "operator provenance differs from frozen scope")

    try:
        with factor_source, h5py.File(factor_source, "r") as factor:
            if (
                int(factor.attrs.get("schema_version", -1)) != required_output["schema_version"]
                or int(factor.attrs.get("method_version", -1)) != required_output["method_version"]
                or int(factor.attrs.get("gauge_date_index", -1)) != 0
                or int(factor.attrs.get("calibration_scope", -1)) != 1
                or int(factor.attrs.get("complete", 0)) != 1
            ):
                raise Attrition("difference_covariance_uncalibrated", "factor header is not complete calibrated schema v4")
            metadata = factor["metadata"]
            _validate_factor_evidence(factor_path.parent, metadata)
            if (
                _hdf5_text(metadata, "method") != required_output["method"]
                or _hdf5_text(metadata, "burst_id") != candidate["burst_id"]
            ):
                raise Attrition("factor_identity_mismatch", "factor method or burst identity differs")
            if (
                int(metadata.attrs.get("reference_row", -1)) != control_pixel[0]
                or int(metadata.attrs.get("reference_col", -1)) != control_pixel[1]
            ):
                raise Attrition(
                    "factor_identity_mismatch",
                    "production factor reference is not the frozen control station pixel",
                )
            units = _hdf5_text(metadata, "units")
            if units not in {"meters", "millimeters"}:
                raise Attrition("difference_covariance_invalid", "factor units are unsupported")
            crs = _hdf5_text(metadata, "crs")
            if not crs:
                raise Attrition("factor_identity_mismatch", "factor CRS identity is missing")
            acquisition_days = np.asarray(metadata["acquisition_days"][:], dtype=float)
            if (
                acquisition_days.ndim != 1
                or acquisition_days.size == 0
                or not np.all(np.isfinite(acquisition_days))
                or not np.all(acquisition_days == np.floor(acquisition_days))
                or acquisition_days.tolist()
                != sorted(set(acquisition_days.tolist()))
                or acquisition_days[0] != 0.0
            ):
                raise Attrition(
                    "factor_identity_mismatch",
                    "factor acquisition days are not exact increasing dates",
                )
            ordered = np.asarray(metadata["ordered_date_indices"][:], dtype=int)
            if ordered.tolist() != list(range(len(acquisition_days))):
                raise Attrition("factor_identity_mismatch", "factor date ordering is not the exact acquisition order")
            acquisition_dates = [
                dt.date.fromisoformat(candidate["date_start"]) + dt.timedelta(days=float(day))
                for day in acquisition_days
            ]
            if (
                len(acquisition_dates) != candidate["epoch_count"]
                or acquisition_dates[-1].isoformat() != candidate["date_end"]
            ):
                raise Attrition("factor_identity_mismatch", "factor dates differ from frozen metadata")
            try:
                common_indices = [acquisition_dates.index(date) for date in common_dates]
            except ValueError as error:
                raise Attrition("factor_identity_mismatch", "common date is absent from factor") from error
            difference_factor, target_rank, target_condition = _block_factor(
                factor["blocks"], target_pixel
            )
            if difference_factor.shape[0] != len(acquisition_dates):
                raise Attrition(
                    "difference_covariance_invalid",
                    "station factor dates disagree with the acquisition set",
                )
            difference_factor = difference_factor[common_indices]
            covariance = difference_factor @ difference_factor.T
            if units == "meters":
                covariance *= 1_000_000.0
            if not np.all(np.isfinite(covariance)):
                raise Attrition("difference_covariance_invalid", "factor covariance is non-finite")
            geotransform = np.asarray(metadata["geotransform"][:], dtype=float).tolist()
            full_grid = _grid(factor["full_grid"])
            if full_grid["row_start"] != 0 or full_grid["col_start"] != 0:
                raise Attrition("factor_identity_mismatch", "factor full grid is not origin based")
            digest_fields = {
                "reference_signature_sha256": "reference_signature_digest",
                "source_replay_sha256": "source_replay_digest",
                "l2_map_sha256": "l2_map_digest",
                "mask_sha256": "mask_digest",
                "source_model_sha256": "source_model_digest",
                "effective_looks_sha256": "effective_looks_digest",
                "support_sha256": "support_digest",
                "correction_order_sha256": "correction_order_digest",
                "unwrap_branch_sha256": "unwrap_branch_digest",
                "burst_ownership_sha256": "burst_ownership_digest",
                "runtime_resource_receipt_sha256": "runtime_resource_receipt_digest",
                "approximation_receipt_sha256": "approximation_receipt_digest",
                "resource_receipt_sha256": "resource_receipt_digest",
                "review_receipt_sha256": "review_receipt_digest",
                "method_manifest_sha256": "method_manifest_digest",
                "calibration_scope_sha256": "calibration_scope_digest",
            }
            scope_hashes = {
                output: _bare_digest(_hdf5_text(metadata, source), source)
                for output, source in digest_fields.items()
            }
            factor_after = os.fstat(factor_source.fileno())
    except Attrition:
        raise
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise Attrition("difference_covariance_invalid", "production factor is unreadable") from error
    if _descriptor_identity(factor_stat) != _descriptor_identity(factor_after):
        raise Attrition("factor_identity_mismatch", "factor changed while it was read")
    if any(
        condition is not None and (not math.isfinite(condition) or condition > 1.0e8)
        for condition in (target_condition,)
    ):
        raise Attrition("total_covariance_ill_conditioned", "station factor condition exceeds the production limit")

    required = preregistration["factor_binding"]
    scope = {
        "target_station_id": candidate["station_ids"][0],
        "control_station_id": candidate["station_ids"][1],
        "target_station_pixel": list(target_pixel),
        "control_station_pixel": list(control_pixel),
        "schema_version": required_output["schema_version"],
        "method_version": required_output["method_version"],
        "method": required_output["method"],
        "calibration_scope": required_output["calibration_status"],
        "common_dates_sha256": canonical_digest([date.isoformat() for date in common_dates]),
        "acquisition_days_sha256": canonical_digest(acquisition_days.tolist()),
        "geotransform_sha256": canonical_digest(geotransform),
        "window": {
            "row_start": min(target_pixel[0], control_pixel[0]),
            "row_end": max(target_pixel[0], control_pixel[0]) + 1,
            "col_start": min(target_pixel[1], control_pixel[1]),
            "col_end": max(target_pixel[1], control_pixel[1]) + 1,
        },
        "overlap": target_pixel == control_pixel,
        "distance": math.hypot(target_pixel[0] - control_pixel[0], target_pixel[1] - control_pixel[1]),
        **scope_hashes,
        "burst_id": candidate["burst_id"],
        "grid_sha256": canonical_digest(full_grid),
        "units": units,
    }
    binding = {
        "operation": required["operation"],
        "input_operator": required["input_operator"],
        "output_factor": required["output_factor"],
        "marginal_rss_combination_allowed": False,
        "mode": required["mode"],
        "reference_specific": required["reference_specific"],
        "stitched_burst_count": required["stitched_burst_count"],
        "operator_sha256": operator_sha256,
        "operator_manifest_sha256": operator_manifest_sha256,
        "persisted_factor_sha256": factor_sha256,
        "persisted_factor_manifest_sha256": hashlib.sha256(factor_manifest_bytes).hexdigest(),
        "factor_sha256": canonical_digest(difference_factor.tolist()),
        "scope_sha256": canonical_digest(scope),
        "calibrated_scope_match": "calibrated_scope_match",
        "scope": scope,
    }
    return {
        "covariance": covariance,
        "binding": binding,
        "target_rank": target_rank,
        "header": {
            "acquisition_days": acquisition_days.tolist(),
            "geotransform": geotransform,
            "crs": crs,
            "units": units,
            "full_grid": full_grid,
        },
    }


def _base_fragment(
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    freeze_receipt_sha256: str,
    run_identity_sha256: str,
    product_identity_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "dolphinrust.temporal_covariance.heldout_cluster_fragment",
        "schema_version": 1,
        "outcomes_present": True,
        "one_shot_unblinding": True,
        "generation_id": preregistration["generation_id"],
        "preregistration_sha256": canonical_digest(preregistration),
        "manifest_sha256": canonical_digest(manifest),
        "freeze_receipt_sha256": freeze_receipt_sha256,
        "run_identity_sha256": _bare_digest(
            run_identity_sha256, "run_identity_sha256"
        ),
        "product_identity_sha256": _bare_digest(
            product_identity_sha256, "product_identity_sha256"
        ),
        "cluster_id": candidate["candidate_id"],
        "station_ids": candidate["station_ids"],
        "burst_id": candidate["burst_id"],
        "site_id": candidate["site_id"],
    }


def _attrition_fragment(
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    attrition: Attrition,
    freeze_receipt_sha256: str,
    run_identity_sha256: str,
    product_identity_sha256: str,
) -> dict[str, Any]:
    if attrition.code not in preregistration["attrition"]["allowed_codes"]:
        raise ValueError(f"executor produced an unregistered attrition code: {attrition.code}")
    return {
        **_base_fragment(
            candidate,
            manifest,
            preregistration,
            freeze_receipt_sha256,
            run_identity_sha256,
            product_identity_sha256,
        ),
        "status": "not_evaluable",
        "reason_code": attrition.code,
        "reason": attrition.detail,
    }


def run_product_cluster(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    cluster_id: str,
    product_directory: Path,
    rust_batch_path: Path,
    *,
    freeze_receipt_sha256: str,
    run_identity_sha256: str,
    product_identity_sha256: str,
    ngl_session: requests.Session | None = None,
    ngl_base_url: str = NGL_TENV3_BASE_URL,
) -> dict[str, Any]:
    """Execute one frozen cluster from production artifacts and live GNSS bytes."""

    candidate = select_manifest_cluster(manifest, preregistration, cluster_id)
    try:
        records_by_station, ngl_sources = fetch_ngl_pair(
            candidate, ngl_session, ngl_base_url
        )
        factor_header = read_production_factor_header(
            product_directory, candidate, preregistration
        )
        start = dt.date.fromisoformat(candidate["date_start"])
        acquisition_dates = [
            start + dt.timedelta(days=float(day))
            for day in factor_header["acquisition_days"]
        ]
        common_dates = exact_common_dates(
            candidate, acquisition_dates, records_by_station, preregistration
        )
        product = derive_product_observations(
            product_directory,
            candidate,
            records_by_station,
            factor_header,
            preregistration,
        )
        first_id, second_id = candidate["station_ids"]
        if factor_header["reference_pixel"] != list(product["station_pixels"][second_id]):
            raise Attrition(
                "factor_identity_mismatch",
                "production factor reference is not the derived control station pixel",
            )
        factor_spec = preregistration["factor_binding"]
        factor = read_production_difference_factor(
            product_directory / factor_spec["output_factor"]["artifact_hdf5"],
            product_directory / factor_spec["output_factor"]["artifact_manifest"],
            candidate,
            product["station_pixels"][first_id],
            product["station_pixels"][second_id],
            common_dates,
            preregistration,
            product_directory / factor_spec["input_operator"]["artifact_hdf5"],
            product_directory / factor_spec["input_operator"]["artifact_manifest"],
        )
        if factor["header"] != {
            key: factor_header[key]
            for key in ("acquisition_days", "geotransform", "crs", "units", "full_grid")
        }:
            raise Attrition("factor_identity_mismatch", "factor header changed between validated reads")
        common_indices = [acquisition_dates.index(date) for date in common_dates]
        try:
            temporal = run_production_temporal_estimator(
                rust_batch_path,
                cluster_id,
                [factor_header["acquisition_days"][index] for index in common_indices],
                [product["insar_difference_mm"][index] for index in common_indices],
                factor["covariance"],
                preregistration,
            )
        except ValueError as error:
            raise Attrition("temporal_estimator_abstained", str(error)) from error
        gnss = station_pair_gnss_observation(
            records_by_station[first_id],
            records_by_station[second_id],
            common_dates,
            product["station_los"][first_id],
            product["station_los"][second_id],
        )
        combined = temporal["slope_variance"] + gnss["slope_variance"]
        if not math.isfinite(combined) or combined <= 0:
            raise Attrition(
                "total_covariance_not_positive_definite",
                "combined station-pair slope variance is not positive",
            )
        solution_sha256 = canonical_digest(
            {station_id: source["sha256"] for station_id, source in ngl_sources.items()}
        )
        los_sha256 = canonical_digest(
            {
                station_id: product["station_los"][station_id].tolist()
                for station_id in candidate["station_ids"]
            }
        )
        return {
            **_base_fragment(
                candidate,
                manifest,
                preregistration,
                freeze_receipt_sha256,
                run_identity_sha256,
                product_identity_sha256,
            ),
            "status": "pass",
            "common_dates": [date.isoformat() for date in common_dates],
            "common_dates_sha256": canonical_digest(
                [date.isoformat() for date in common_dates]
            ),
            "gnss_provenance": {
                "solution_sources": ngl_sources,
                "solution_sha256": solution_sha256,
                "coordinate_frame": "ENU",
                "los_source": "run_specific_sourced_los_components",
                "los_sha256": los_sha256,
                "station_los_vectors": {
                    station_id: product["station_los"][station_id].tolist()
                    for station_id in candidate["station_ids"]
                },
                "projection_convention": "signed_ground_to_sensor_los_dot_enu",
                "epoch_zero_reference_sha256": canonical_digest(
                    {"date": common_dates[0].isoformat(), "stations": candidate["station_ids"]}
                ),
                "covariance_projection": "u_transpose_C_u",
            },
            "difference_covariance": factor["binding"],
            "estimator": {
                "method": temporal["method"],
                "method_version": temporal["method_version"],
                "binary_sha256": temporal["binary_sha256"],
                "request_sha256": temporal["request_sha256"],
                "response_sha256": temporal["response_sha256"],
                "insar_source_sha256": product["source_sha256"],
                "insar_design_sha256": canonical_digest(
                    [factor_header["acquisition_days"][index] for index in common_indices]
                ),
                "gnss_design_sha256": gnss["design_sha256"],
                "resource": temporal["resource"],
            },
            "observation": {
                "insar_slope_difference": temporal["slope_mm_year"],
                "gnss_slope_difference": gnss["slope_mm_year"],
                "insar_difference_variance": temporal["slope_variance"],
                "gnss_slope_variance": gnss["slope_variance"],
                "sensor_cross_covariance": 0.0,
                "baseline_sigma": {
                    level: math.sqrt(
                        temporal["baseline_sigma"] ** 2 + gnss["slope_variance"]
                    )
                    for level in ("68", "90", "95")
                },
            },
        }
    except Attrition as attrition:
        return _attrition_fragment(
            candidate,
            manifest,
            preregistration,
            attrition,
            freeze_receipt_sha256,
            run_identity_sha256,
            product_identity_sha256,
        )


def write_one_shot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"one-shot output already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"one-shot output already exists: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)
