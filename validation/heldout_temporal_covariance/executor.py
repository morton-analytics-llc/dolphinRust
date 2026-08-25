"""Bounded one-shot primitives for one frozen held-out cluster."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import requests
from pyproj import Transformer

if __package__ and __package__.startswith("validation."):
    from validation.crop_real import Window
    from validation.gps_ground_truth import (
        AlignedRecord,
        Tenv3Record,
        gnss_los_covariance_series,
        gnss_los_series,
        interpolate_grid,
        parse_tenv3,
    )
    from validation.remote_hdf5_crop import RemoteCropError, crop_remote_hdf5
else:
    from crop_real import Window
    from gps_ground_truth import (
        AlignedRecord,
        Tenv3Record,
        gnss_los_covariance_series,
        gnss_los_series,
        interpolate_grid,
        parse_tenv3,
    )
    from remote_hdf5_crop import RemoteCropError, crop_remote_hdf5

from .cohort import canonical_digest, validate_manifest


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        fitted = fit_slope_with_covariance(dates, difference, covariance)
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


def los_at_static_crop(static_path: Path, record: Tenv3Record) -> np.ndarray:
    try:
        with h5py.File(static_path, "r") as product:
            x = np.asarray(product["/data/x_coordinates"][:], dtype=float)
            y = np.asarray(product["/data/y_coordinates"][:], dtype=float)
            projection = int(product["/data/projection"][()])
            east_data = np.asarray(product["/data/los_east"][:], dtype=float)
            north_data = np.asarray(product["/data/los_north"][:], dtype=float)
        projected_x, projected_y = Transformer.from_crs(
            4326, projection, always_xy=True
        ).transform(record.longitude, record.latitude)
        east = interpolate_grid(east_data, x, y, projected_x, projected_y)
        north = interpolate_grid(north_data, x, y, projected_x, projected_y)
    except (OSError, KeyError, ValueError) as error:
        raise Attrition("sourced_los_missing", "STATIC LOS crop cannot be sampled at the GNSS station") from error
    horizontal = east * east + north * north
    if not math.isfinite(horizontal) or horizontal <= 0 or horizontal > 1.0 + 1e-10:
        raise Attrition("sourced_los_invalid", "STATIC LOS horizontal components are invalid")
    vector = np.array([east, north, math.sqrt(max(0.0, 1.0 - horizontal))])
    if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-5:
        raise Attrition("sourced_los_invalid", "STATIC LOS vector is not unit norm")
    return vector


def _grid(group: h5py.Group) -> dict[str, int]:
    return {
        field: int(group.attrs[field])
        for field in ("row_start", "col_start", "rows", "cols", "stride_y", "stride_x")
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
    try:
        if factor_path.stat().st_size > FACTOR_BYTE_CAP or factor_manifest_path.stat().st_size > JSON_BYTE_CAP:
            raise Attrition("difference_covariance_unavailable", "factor artifact exceeds its byte cap")
    except OSError as error:
        raise Attrition("difference_covariance_unavailable", "factor artifact is missing") from error
    try:
        factor_manifest_bytes = factor_manifest_path.read_bytes()
        factor_manifest = json.loads(factor_manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Attrition("difference_covariance_unavailable", "factor provenance is unreadable") from error
    factor_sha256 = _sha256_file(factor_path)
    required_output = preregistration["factor_binding"]["output_factor"]
    if factor_path.name != required_output["artifact_hdf5"] or factor_manifest_path.name != required_output["artifact_manifest"]:
        raise Attrition("factor_identity_mismatch", "production factor filenames differ from frozen scope")
    required_manifest = {
        "schema_version": 3,
        "method": required_output["method"],
        "method_version": required_output["method_version"],
        "hdf5_file": required_output["artifact_hdf5"],
        "hdf5_bytes": factor_path.stat().st_size,
        "hdf5_sha256": factor_sha256,
        "burst_id": candidate["burst_id"],
        "calibration_scope": required_output["calibration_status"],
    }
    if not isinstance(factor_manifest, Mapping) or any(
        factor_manifest.get(field) != value for field, value in required_manifest.items()
    ):
        raise Attrition("factor_identity_mismatch", "production factor provenance differs from frozen scope")
    if operator_path is None or operator_manifest_path is None:
        operator_sha256 = _bare_digest(factor_manifest.get("operator_sha256"), "operator_sha256")
        operator_manifest_sha256 = _bare_digest(
            factor_manifest.get("operator_manifest_sha256"), "operator_manifest_sha256"
        )
    else:
        required_operator = preregistration["factor_binding"]["input_operator"]
        if operator_path.name != required_operator["artifact_hdf5"] or operator_manifest_path.name != required_operator["artifact_manifest"]:
            raise Attrition("factor_identity_mismatch", "operator filenames differ from frozen scope")
        try:
            if operator_manifest_path.stat().st_size > JSON_BYTE_CAP:
                raise Attrition("factor_identity_mismatch", "operator provenance exceeds its byte cap")
        except OSError as error:
            raise Attrition("factor_identity_mismatch", "operator artifact is missing") from error
        operator_sha256 = _sha256_file(operator_path)
        operator_manifest_sha256 = _sha256_file(operator_manifest_path)
        try:
            operator_manifest = json.loads(operator_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Attrition("factor_identity_mismatch", "operator provenance is unreadable") from error
        operator_required = {
            "schema_version": 1,
            "method": required_operator["method"],
            "method_version": required_operator["method_version"],
            "gauge_date_index": 0,
            "hdf5_file": required_operator["artifact_hdf5"],
            "hdf5_bytes": operator_path.stat().st_size,
            "hdf5_sha256": operator_sha256,
        }
        if not isinstance(operator_manifest, Mapping) or any(
            operator_manifest.get(field) != value
            for field, value in operator_required.items()
        ):
            raise Attrition("factor_identity_mismatch", "operator provenance differs from frozen scope")

    try:
        with h5py.File(factor_path, "r") as factor:
            if (
                int(factor.attrs.get("schema_version", -1)) != required_output["schema_version"]
                or int(factor.attrs.get("method_version", -1)) != required_output["method_version"]
                or int(factor.attrs.get("gauge_date_index", -1)) != 0
                or int(factor.attrs.get("calibration_scope", -1)) != 1
                or int(factor.attrs.get("complete", 0)) != 1
            ):
                raise Attrition("difference_covariance_uncalibrated", "factor header is not complete calibrated schema v4")
            metadata = factor["metadata"]
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
            }
            scope_hashes = {
                output: _bare_digest(_hdf5_text(metadata, source), source)
                for output, source in digest_fields.items()
            }
    except Attrition:
        raise
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise Attrition("difference_covariance_invalid", "production factor is unreadable") from error
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
    }


def _base_fragment(
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    freeze_receipt_sha256: str,
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
) -> dict[str, Any]:
    if attrition.code not in preregistration["attrition"]["allowed_codes"]:
        raise ValueError(f"executor produced an unregistered attrition code: {attrition.code}")
    return {
        **_base_fragment(
            candidate, manifest, preregistration, freeze_receipt_sha256
        ),
        "status": "not_evaluable",
        "reason_code": attrition.code,
        "reason": attrition.detail,
    }


def run_one_cluster(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    input_spec: Mapping[str, Any],
    output_path: Path,
    aggregate_output_path: Path,
    *,
    allow_one_shot_unblinding: bool,
    freeze_receipt_sha256: str,
    static_session: requests.Session,
    ngl_session: requests.Session | None = None,
    ngl_base_url: str = NGL_TENV3_BASE_URL,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"one-shot output already exists: {output_path}")
    if aggregate_output_path.exists():
        raise FileExistsError(
            f"aggregate outcome artifact already exists: {aggregate_output_path}"
        )
    if not allow_one_shot_unblinding:
        raise PermissionError("explicit one-shot unblinding authorization is required")
    expected_fields = {
        "schema",
        "schema_version",
        "cluster_id",
        "acquisition_dates",
        "station_pixels",
        "insar_difference_mm",
        "baseline_sigma",
        "static_source",
        "factor",
        "manifest_sha256",
        "freeze_receipt_sha256",
        "insar_source_sha256",
        "estimator_receipt_sha256",
    }
    if set(input_spec) != expected_fields or input_spec.get("schema") != "dolphinrust.temporal_covariance.heldout_cluster_input" or input_spec.get("schema_version") != 1:
        raise ValueError("held-out cluster input fields/schema do not match version 1")
    candidate = select_manifest_cluster(
        manifest, preregistration, str(input_spec["cluster_id"])
    )
    if input_spec["manifest_sha256"] != canonical_digest(manifest):
        raise ValueError("cluster input manifest identity mismatch")
    if input_spec["freeze_receipt_sha256"] != freeze_receipt_sha256:
        raise ValueError("cluster input freeze receipt identity mismatch")
    _bare_digest(freeze_receipt_sha256, "freeze_receipt_sha256")
    insar_source_sha256 = _bare_digest(
        input_spec["insar_source_sha256"], "insar_source_sha256"
    )
    estimator_receipt_sha256 = _bare_digest(
        input_spec["estimator_receipt_sha256"], "estimator_receipt_sha256"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        acquisition_dates = [
            dt.date.fromisoformat(value) for value in input_spec["acquisition_dates"]
        ]
    except (TypeError, ValueError) as error:
        raise ValueError("cluster acquisition_dates must be ISO dates") from error
    station_pixels = input_spec["station_pixels"]
    if not isinstance(station_pixels, Mapping) or set(station_pixels) != set(candidate["station_ids"]):
        raise ValueError("station pixels must exactly match the frozen station pair")
    parsed_pixels: dict[str, tuple[int, int]] = {}
    for station_id, value in station_pixels.items():
        if not isinstance(value, list) or len(value) != 2 or any(not isinstance(item, int) or item < 0 for item in value):
            raise ValueError("station pixels must contain nonnegative row/column integers")
        parsed_pixels[station_id] = (value[0], value[1])
    static = input_spec["static_source"]
    if not isinstance(static, Mapping) or set(static) != {
        "url",
        "file_name",
        "windows",
        "maximum_transfer_bytes_per_station",
    }:
        raise ValueError("STATIC source fields do not match the bounded crop schema")
    maximum_transfer = static["maximum_transfer_bytes_per_station"]
    if not isinstance(maximum_transfer, int) or not 0 < maximum_transfer <= 64 * 1024 * 1024:
        raise ValueError("STATIC crop transfer cap must be between 1 and 67108864 bytes")
    windows = static["windows"]
    if not isinstance(windows, Mapping) or set(windows) != set(candidate["station_ids"]):
        raise ValueError("STATIC crop windows must exactly match the station pair")
    factor_spec = input_spec["factor"]
    if not isinstance(factor_spec, Mapping) or set(factor_spec) != {
        "hdf5_path",
        "manifest_path",
        "operator_path",
        "operator_manifest_path",
    }:
        raise ValueError("production factor input fields are incomplete")

    try:
        records_by_station, ngl_sources = fetch_ngl_pair(
            candidate, ngl_session, ngl_base_url
        )
        common_dates = exact_common_dates(
            candidate, acquisition_dates, records_by_station, preregistration
        )
        with tempfile.TemporaryDirectory(
            prefix="dolphinrust-heldout-cluster-", dir=output_path.parent
        ) as directory:
            temporary_root = Path(directory)
            station_los: dict[str, np.ndarray] = {}
            crop_receipts: dict[str, Any] = {}
            for station_id in candidate["station_ids"]:
                window_value = windows[station_id]
                if not isinstance(window_value, Mapping) or set(window_value) != {
                    "row0",
                    "col0",
                    "height",
                    "width",
                }:
                    raise ValueError("STATIC crop window fields are invalid")
                try:
                    window = Window(**{field: int(window_value[field]) for field in window_value})
                    crop_path = temporary_root / f"{station_id}.static.h5"
                    crop_receipt_path = temporary_root / f"{station_id}.static.receipt.json"
                    crop_receipts[station_id] = crop_remote_hdf5(
                        url=str(static["url"]),
                        expected_file_name=str(static["file_name"]),
                        destination=crop_path,
                        receipt_path=crop_receipt_path,
                        product_type="static",
                        window=window,
                        source_catalog_sha256=candidate["metadata_hashes"]["burst_metadata_sha256"],
                        session=static_session,
                        max_transfer_bytes=maximum_transfer,
                    )
                except (RemoteCropError, TypeError, ValueError) as error:
                    raise Attrition("sourced_los_missing", f"bounded STATIC crop failed for {station_id}") from error
                station_record = next(
                    record
                    for record in records_by_station[station_id]
                    if record.date == common_dates[0]
                )
                station_los[station_id] = los_at_static_crop(crop_path, station_record)

            first_id, second_id = candidate["station_ids"]
            gnss = station_pair_gnss_observation(
                records_by_station[first_id],
                records_by_station[second_id],
                common_dates,
                station_los[first_id],
                station_los[second_id],
            )
            factor = read_production_difference_factor(
                Path(factor_spec["hdf5_path"]),
                Path(factor_spec["manifest_path"]),
                candidate,
                parsed_pixels[first_id],
                parsed_pixels[second_id],
                common_dates,
                preregistration,
                Path(factor_spec["operator_path"]),
                Path(factor_spec["operator_manifest_path"]),
            )
            insar_series = np.asarray(input_spec["insar_difference_mm"], dtype=float)
            if insar_series.shape != (len(acquisition_dates),):
                raise ValueError("InSAR difference series must match all acquisition dates")
            common_indices = [acquisition_dates.index(date) for date in common_dates]
            try:
                insar = fit_slope_with_covariance(
                    common_dates,
                    insar_series[common_indices],
                    factor["covariance"],
                )
            except ValueError as error:
                raise Attrition("total_covariance_not_positive_definite", str(error)) from error
            baseline_sigma = input_spec["baseline_sigma"]
            if not isinstance(baseline_sigma, Mapping) or set(baseline_sigma) != {"68", "90", "95"} or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0
                for value in baseline_sigma.values()
            ):
                raise ValueError("baseline_sigma must contain positive finite 68/90/95 values")
            combined = insar["variance"] + gnss["slope_variance"]
            if not math.isfinite(combined) or combined <= 0:
                raise Attrition("total_covariance_not_positive_definite", "combined station-pair slope variance is not positive")
            solution_sha256 = canonical_digest(
                {station_id: source["sha256"] for station_id, source in ngl_sources.items()}
            )
            los_sha256 = canonical_digest(
                {
                    station_id: crop_receipts[station_id]["output"]["sha256"]
                    for station_id in candidate["station_ids"]
                }
            )
            fragment = {
                **_base_fragment(
                    candidate,
                    manifest,
                    preregistration,
                    freeze_receipt_sha256,
                ),
                "status": "evaluable",
                "common_dates": [date.isoformat() for date in common_dates],
                "common_dates_sha256": canonical_digest(
                    [date.isoformat() for date in common_dates]
                ),
                "station_pair_provenance": {
                    "solution_sources": ngl_sources,
                    "solution_sha256": solution_sha256,
                    "coordinate_frame": "ENU",
                    "los_source": "run_specific_sourced_los_components",
                    "los_sha256": los_sha256,
                    "station_los_vectors": {
                        station_id: station_los[station_id].tolist()
                        for station_id in candidate["station_ids"]
                    },
                    "los_crop_receipts": crop_receipts,
                    "projection_convention": "signed_ground_to_sensor_los_dot_enu",
                    "epoch_zero_reference_sha256": canonical_digest(
                        {
                            "date": common_dates[0].isoformat(),
                            "stations": candidate["station_ids"],
                        }
                    ),
                    "covariance_projection": "u_transpose_C_u",
                },
                "difference_covariance": factor["binding"],
                "estimator": {
                    "method": "unweighted_ols_with_full_covariance_propagation_v1",
                    "estimator_receipt_sha256": estimator_receipt_sha256,
                    "insar_source_sha256": insar_source_sha256,
                    "insar_difference_series_sha256": canonical_digest(
                        insar_series.tolist()
                    ),
                    "insar_design_sha256": insar["design_sha256"],
                    "gnss_design_sha256": gnss["design_sha256"],
                },
                "observation": {
                    "insar_slope_difference": insar["slope"],
                    "gnss_slope_difference": gnss["slope_mm_year"],
                    "insar_difference_variance": insar["variance"],
                    "gnss_slope_variance": gnss["slope_variance"],
                    "sensor_cross_covariance": 0.0,
                    "baseline_sigma": dict(baseline_sigma),
                },
            }
    except Attrition as attrition:
        fragment = _attrition_fragment(
            candidate,
            manifest,
            preregistration,
            attrition,
            freeze_receipt_sha256,
        )
    if aggregate_output_path.exists():
        raise FileExistsError(
            f"aggregate outcome artifact already exists: {aggregate_output_path}"
        )
    write_one_shot(output_path, fragment)
    return fragment


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
    finally:
        temporary_path.unlink(missing_ok=True)
