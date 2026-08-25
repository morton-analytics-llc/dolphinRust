"""Immutable cohort execution and production-estimator bindings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import fcntl
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cohort import canonical_digest, validate_manifest


LEDGER_SCHEMA = "dolphinrust.temporal_covariance.heldout_run_ledger"
RECEIPT_SCHEMA = "dolphinrust.temporal_covariance.heldout_receipt"
MAX_ESTIMATOR_STDOUT_BYTES = 1024 * 1024
PRODUCT_FILE_BYTE_CAP = 4 * 1024 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def product_identity_sha256(
    product_directory: Path, preregistration: Mapping[str, Any]
) -> str:
    """Hash the exact local product bytes used by a held-out cluster."""

    root = product_directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("held-out product root is not a directory")
    factor = preregistration["factor_binding"]
    names = {
        "fixed_cube_receipt.json",
        "los_east.tif",
        "los_north.tif",
        "los_up.tif",
        "velocity_validity_mask.tif",
        "geometry_provenance.json",
        factor["input_operator"]["artifact_hdf5"],
        factor["input_operator"]["artifact_manifest"],
        factor["output_factor"]["artifact_hdf5"],
        factor["output_factor"]["artifact_manifest"],
        "referenced_displacement_covariance_approximation_receipt.json",
        "referenced_displacement_covariance_resource_receipt.json",
        "referenced_displacement_covariance_review_receipt.json",
        "referenced_displacement_covariance_method_manifest.json",
        "referenced_displacement_covariance_approximation_result.json",
        "referenced_displacement_covariance_preregistration.json",
        "referenced_displacement_covariance_design.md",
        "referenced_displacement_covariance_producer_binary",
    }
    names.update(path.name for path in root.glob("displacement_[0-9][0-9].tif"))
    if not any(name.startswith("displacement_") for name in names):
        raise ValueError("held-out product has no displacement rasters")
    identities: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = root / name
        resolved = path.resolve(strict=True)
        if resolved.parent != root or not resolved.is_file():
            raise ValueError("held-out product contains an external or missing artifact")
        with resolved.open("rb") as source:
            before = os.fstat(source.fileno())
            if before.st_size <= 0 or before.st_size > PRODUCT_FILE_BYTE_CAP:
                raise ValueError("held-out product artifact exceeds its byte cap")
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(source.fileno())
        stat_identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )
        if stat_identity(before) != stat_identity(after):
            raise ValueError("held-out product changed while it was hashed")
        identities[name] = {"bytes": before.st_size, "sha256": digest.hexdigest()}
    return canonical_digest(identities)


def validate_product_run_plan(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Path]:
    if (
        set(plan) != {"schema", "schema_version", "clusters"}
        or plan.get("schema")
        != "dolphinrust.temporal_covariance.heldout_run_plan"
        or plan.get("schema_version") != 1
        or not isinstance(plan.get("clusters"), list)
    ):
        raise ValueError("held-out run plan schema/fields differ from version 1")
    expected = {
        value["candidate_id"]
        for value in manifest["frozen_clusters"] + manifest["surplus_clusters"]
    }
    roots: dict[str, Path] = {}
    for entry in plan["clusters"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "cluster_id",
            "product_directory",
        }:
            raise ValueError("held-out run plan cluster fields are invalid")
        cluster_id = entry["cluster_id"]
        if cluster_id not in expected or cluster_id in roots:
            raise ValueError("held-out run plan contains unknown or duplicate clusters")
        root = Path(entry["product_directory"]).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("held-out product directory is not a directory")
        roots[cluster_id] = root
    if set(roots) != expected:
        raise ValueError("held-out run plan does not contain exact 96+20 clusters")
    if len(set(roots.values())) != len(roots):
        raise ValueError("held-out run plan reuses a product directory across clusters")
    return roots


def _atomic_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


class CohortRunLedger:
    """Exclusive durable authorization for one frozen unblinding identity."""

    def __init__(self, path: Path, payload: dict[str, Any], lock: Any) -> None:
        self.path = path
        self.payload = payload
        self._lock = lock

    @classmethod
    def acquire(
        cls, path: Path, identity: Mapping[str, Any]
    ) -> "CohortRunLedger":
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("cohort run identity must be a nonempty object")
        identity_value = dict(identity)
        identity_sha256 = canonical_digest(identity_value)
        payload = {
            "schema": LEDGER_SCHEMA,
            "schema_version": 1,
            "state": "running",
            "identity": identity_value,
            "identity_sha256": identity_sha256,
            "receipt_sha256": None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        lock = lock_path.open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.close()
            raise PermissionError("frozen unblinding run is already active") from error
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                lock.close()
                raise PermissionError("existing unblinding ledger is unreadable") from error
            if (
                not isinstance(existing, dict)
                or existing.get("schema") != LEDGER_SCHEMA
                or existing.get("schema_version") != 1
                or existing.get("identity_sha256") != identity_sha256
                or existing.get("identity") != identity_value
            ):
                lock.close()
                raise PermissionError("existing unblinding ledger has a different identity")
            if existing.get("state") == "complete":
                lock.close()
                raise PermissionError("frozen outcomes were already completed")
            if existing.get("state") != "running" or existing.get("receipt_sha256") is not None:
                lock.close()
                raise PermissionError("existing unblinding ledger state is invalid")
            return cls(path, existing, lock)
        except OSError:
            lock.close()
            raise
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return cls(path, payload, lock)

    def complete(self, receipt_sha256: str) -> None:
        if (
            not isinstance(receipt_sha256, str)
            or len(receipt_sha256) != 64
            or any(value not in "0123456789abcdef" for value in receipt_sha256)
        ):
            raise ValueError("receipt_sha256 must be a lowercase SHA-256")
        if self._lock.closed:
            raise PermissionError("unblinding ledger lock is not held")
        current = json.loads(self.path.read_text(encoding="utf-8"))
        if current != self.payload or current.get("state") != "running":
            raise PermissionError("unblinding ledger changed while locked")
        current["state"] = "complete"
        current["receipt_sha256"] = receipt_sha256
        _atomic_replace(self.path, current)
        self.payload = current

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()


def _not_used(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": candidate["candidate_id"],
        "station_ids": candidate["station_ids"],
        "burst_id": candidate["burst_id"],
        "site_id": candidate["site_id"],
        "status": "not_used",
    }


def assemble_heldout_receipt(
    preregistration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fragments: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble exact primary-first, lexical-surplus outcome accounting."""

    validate_manifest(manifest, preregistration)
    primary = list(manifest["frozen_clusters"])
    surplus = sorted(manifest["surplus_clusters"], key=lambda value: value["candidate_id"])
    candidates = {value["candidate_id"]: value for value in primary + surplus}
    by_id: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        cluster_id = fragment.get("cluster_id")
        if cluster_id not in candidates or cluster_id in by_id:
            raise ValueError("cluster fragments contain an unknown or duplicate identity")
        if fragment.get("status") not in {"pass", "fail", "not_evaluable"}:
            raise ValueError("cluster fragment has an invalid executable status")
        by_id[str(cluster_id)] = dict(fragment)
    primary_ids = [value["candidate_id"] for value in primary]
    if any(cluster_id not in by_id for cluster_id in primary_ids):
        raise ValueError("every frozen primary must be executed before assembly")
    attrited = [
        cluster_id
        for cluster_id in primary_ids
        if by_id[cluster_id]["status"] == "not_evaluable"
    ]
    used_surplus: list[str] = []
    attempted_surplus: list[str] = []
    for candidate in surplus:
        if len(used_surplus) >= len(attrited):
            break
        cluster_id = candidate["candidate_id"]
        if cluster_id not in by_id:
            raise ValueError("required lexical surplus cluster was not executed")
        attempted_surplus.append(cluster_id)
        if by_id[cluster_id]["status"] != "not_evaluable":
            used_surplus.append(cluster_id)
    if len(used_surplus) != len(attrited):
        raise ValueError("executed surplus cannot fill frozen primary attrition")
    unused_surplus = [
        candidate["candidate_id"]
        for candidate in surplus
        if candidate["candidate_id"] not in used_surplus
    ]
    unexpected = set(by_id) - set(primary_ids) - set(attempted_surplus)
    if unexpected:
        raise ValueError("outcomes were fetched for unused surplus clusters")
    clusters = [by_id[cluster_id] for cluster_id in primary_ids]
    clusters.extend(
        by_id[cluster_id]
        if cluster_id in by_id
        else _not_used(candidates[cluster_id])
        for cluster_id in [candidate["candidate_id"] for candidate in surplus]
    )
    required_hashes = set(preregistration["receipt_hash_fields"])
    if set(hashes) != required_hashes:
        raise ValueError("cohort receipt implementation hashes are incomplete")
    normalized_hashes = dict(hashes)
    for name, value in normalized_hashes.items():
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"cohort receipt hash is invalid: {name}")
    evaluable = [value for value in clusters if value["status"] in {"pass", "fail"}]
    bundle_fields = (
        "operator_sha256",
        "operator_manifest_sha256",
        "persisted_factor_sha256",
        "persisted_factor_manifest_sha256",
    )
    for field in bundle_fields:
        normalized_hashes[field] = canonical_digest(
            {
                value["cluster_id"]: value["difference_covariance"][field]
                for value in evaluable
                if "difference_covariance" in value
            }
        )
    normalized_hashes["gnss_catalog_sha256"] = canonical_digest(
        {
            value["cluster_id"]: value["gnss_provenance"]["solution_sha256"]
            for value in evaluable
            if "gnss_provenance" in value
        }
    )
    reasons = {
        cluster_id: by_id[cluster_id]["reason_code"]
        for cluster_id in by_id
        if by_id[cluster_id]["status"] == "not_evaluable"
    }
    return {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "outcomes_present": True,
        "one_shot_unblinding": True,
        "cohort_id": preregistration["cohort_id"],
        "generation_id": preregistration["generation_id"],
        "preregistration_sha256": canonical_digest(preregistration),
        "manifest_sha256": canonical_digest(manifest),
        "scope_hash": canonical_digest(preregistration["field_scope"]),
        "calibrated_scope_match": True,
        "factor_binding": preregistration["factor_binding"],
        "hashes": normalized_hashes,
        "cluster_counts": {
            "primary": len(primary),
            "surplus": len(surplus),
            "executed": len(by_id),
            "evaluable": len(evaluable),
        },
        "attrition": {
            "attrited_primary_ids": attrited,
            "used_surplus_ids": used_surplus,
            "unused_surplus_ids": unused_surplus,
            "reasons_by_cluster": reasons,
        },
        "clusters": clusters,
    }


def run_production_temporal_estimator(
    binary_path: Path,
    cluster_id: str,
    acquisition_days: Sequence[float],
    observations_mm: Sequence[float],
    difference_covariance_mm2: np.ndarray,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen Rust complete-refit estimator and validate its selection."""

    before = binary_path.stat()
    binary_sha256 = _sha256_file(binary_path)
    after = binary_path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise ValueError("temporal estimator binary changed while hashing")
    binding = preregistration.get("estimator_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("binary") != "temporal_covariance_batch"
        or binding.get("schema") != "dolphinrust-temporal-covariance-batch/4"
        or binding.get("execution_path") != "fixed_factor"
        or binding.get("method") != "complete_refit_bootstrap"
        or binding.get("method_version") != 1
        or binding.get("baseline") != "conditional_wls"
        or not isinstance(binding.get("options"), Mapping)
    ):
        raise ValueError("frozen temporal estimator identity is incomplete")
    options = binding["options"]
    options = dict(options)
    options["bootstrap_seed"] = int(canonical_digest(cluster_id)[:16], 16)
    covariance = np.asarray(difference_covariance_mm2, dtype=float)
    days = [float(value) for value in acquisition_days]
    observations = [float(value) for value in observations_mm]
    if covariance.shape != (len(days), len(days)) or len(observations) != len(days):
        raise ValueError("temporal estimator inputs have inconsistent dimensions")
    request = {
        "execution_path": "fixed_factor",
        "cell_id": cluster_id,
        "cell_index": 0,
        "outer_seed_index": 0,
        "seed_sha256": canonical_digest(
            {"generation_id": preregistration["generation_id"], "cluster_id": cluster_id}
        ),
        "seed": options["bootstrap_seed"],
        "days": days,
        "options": options,
        "fixed_factor": {
            "observations": observations,
            "difference_covariance": covariance.tolist(),
        },
        "production_path": None,
    }
    completed = subprocess.run(
        [str(binary_path)],
        input=json.dumps(request, sort_keys=True, allow_nan=False) + "\n",
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("production temporal estimator failed")
    if len(completed.stdout.encode()) > MAX_ESTIMATOR_STDOUT_BYTES:
        raise ValueError("production temporal estimator output exceeds its byte cap")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("production temporal estimator output is invalid") from error
    fit = response.get("fit") if isinstance(response, Mapping) else None
    if (
        response.get("schema") != "dolphinrust-temporal-covariance-batch/4"
        or response.get("execution_path") != "fixed_factor"
        or response.get("fixed_factor_status") != "evaluated"
        or response.get("emitted") is not True
        or response.get("failed") is not False
        or not isinstance(fit, Mapping)
    ):
        raise ValueError("production temporal estimator abstained or changed schema")
    selected = fit.get("complete_refit_bootstrap")
    baseline = fit.get("conditional_wls")
    if not isinstance(selected, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("production estimator comparators are incomplete")
    attempts = binding.get("bootstrap_replicates", 200)
    minimum_successes = binding.get("bootstrap_minimum_successes", 198)
    if (
        fit.get("status") != "evaluated"
        or fit.get("bootstrap_attempts") != attempts
        or fit.get("bootstrap_successes", 0) < minimum_successes
        or selected.get("status") != "evaluated"
        or selected.get("attempted_replicates") != attempts
        or selected.get("successful_replicates") != fit.get("bootstrap_successes")
        or baseline.get("status") != "evaluated"
    ):
        raise ValueError("production estimator does not satisfy frozen bootstrap accounting")
    slope = selected.get("point_estimate")
    standard_error = selected.get("standard_error_diagnostic")
    baseline_standard_error = baseline.get("standard_error_diagnostic")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in (slope, standard_error, baseline_standard_error)
    ) or standard_error <= 0 or baseline_standard_error <= 0:
        raise ValueError("production estimator returned invalid selected uncertainty")
    return {
        "method": binding.get("method", "complete_refit_bootstrap"),
        "method_version": binding.get("method_version", 1),
        "slope_mm_year": slope * 365.25,
        "slope_variance": (standard_error * 365.25) ** 2,
        "baseline_sigma": baseline_standard_error * 365.25,
        "binary_sha256": binary_sha256,
        "request_sha256": canonical_digest(request),
        "response_sha256": canonical_digest(response),
        "fit": fit,
        "resource": response.get("resource"),
    }
