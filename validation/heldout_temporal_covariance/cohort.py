"""Frozen metadata-only cohort discovery and manifest validation."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "dolphinrust.temporal_covariance.heldout_cohort"
SCHEMA_VERSION = 1
HASH_FIELDS = ("catalog_sha256", "burst_metadata_sha256", "gnss_station_metadata_sha256")
CLUSTER_KEYS = ("burst_id", "orbit_id", "footprint_id", "site_id")
PROTECTED_SITES = {"fresno"}
PROTECTED_STATIONS = {"MMX1", "ICMX", "MXMX", "MXTM", "SSNX", "TNGF", "UJAL", "UNVA", "UTAC"}
PROTECTED_BURSTS = {
    "T005_008704_IW1",
    "T035_073270_IW1",
    "T100_213486_IW2",
    "T137_292316_IW3",
    "T137_292329_IW1",
    "T150_320250_IW3",
}
OUTCOME_FIELDS = {
    "displacement",
    "velocity",
    "sigma",
    "residual",
    "gnss_series",
    "insar_series",
    "outcome",
    "coverage",
    "interval_score",
}
RECORD_FIELDS = {
    "candidate_id",
    "source_kind",
    "burst_id",
    "orbit_id",
    "footprint_id",
    "site_id",
    "frame_id",
    "station_ids",
    "date_start",
    "date_end",
    "epoch_count",
    "metadata_hashes",
    "query_digest",
}
MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "cohort_id",
    "status",
    "outcomes_present",
    "preregistration_sha256",
    "candidate_query_digest",
    "selection_algorithm",
    "selection_outcome_blind",
    "candidate_pool",
    "frozen_clusters",
    "surplus_clusters",
    "excluded_after_selection",
}


class CohortValidationError(ValueError):
    """The metadata or frozen cohort violates the preregistered contract."""


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _date(value: Any) -> date:
    if not isinstance(value, str):
        raise CohortValidationError("candidate dates must be ISO strings")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise CohortValidationError("candidate dates must be valid ISO dates") from error


def _lower(value: Any) -> str:
    return str(value).strip().lower()


def _reject_outcomes(record: Mapping[str, Any]) -> None:
    if set(record) & OUTCOME_FIELDS:
        raise CohortValidationError("metadata discovery cannot inspect outcome fields")


def validate_candidate(record: Mapping[str, Any], preregistration: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise CohortValidationError("candidate must be an object")
    _reject_outcomes(record)
    query = preregistration["candidate_query"].get("query")
    if not isinstance(query, Mapping) or preregistration["candidate_query"].get("query_digest") != canonical_digest(query):
        raise CohortValidationError("frozen candidate query identity is not a SHA-256 of its query")
    missing = RECORD_FIELDS - set(record)
    if missing:
        raise CohortValidationError("candidate is missing metadata fields: %s" % ", ".join(sorted(missing)))
    if set(record) - RECORD_FIELDS:
        raise CohortValidationError("candidate contains fields outside the metadata schema")
    if record["source_kind"] != "catalog_metadata":
        raise CohortValidationError("candidate source_kind must be catalog_metadata")
    stations = record["station_ids"]
    if not isinstance(stations, list) or len(stations) != 2 or len(set(stations)) != 2 or stations != sorted(stations):
        raise CohortValidationError("candidate must contain two sorted distinct station IDs")
    if any(not isinstance(station, str) or not station for station in stations):
        raise CohortValidationError("candidate station IDs must be non-empty strings")
    if _lower(record["site_id"]) in preregistration["exclusions"]["site_ids"]:
        raise CohortValidationError("candidate site is protected from the outer holdout")
    if _lower(record["site_id"]) in PROTECTED_SITES:
        raise CohortValidationError("candidate site is Fresno or an equivalent protected site")
    if set(stations) & set(preregistration["exclusions"]["station_ids"]):
        raise CohortValidationError("candidate station is in the exposed-data exclusion set")
    if set(stations) & PROTECTED_STATIONS:
        raise CohortValidationError("candidate station is in the exposed-data station set")
    if record["burst_id"] in preregistration["exclusions"]["burst_ids"] or record["burst_id"] in PROTECTED_BURSTS:
        raise CohortValidationError("candidate burst is in the exposed-data exclusion set")
    if _date(record["date_start"]) >= _date(record["date_end"]):
        raise CohortValidationError("candidate date range must be increasing")
    if not isinstance(record["epoch_count"], int) or record["epoch_count"] < preregistration["eligibility"]["minimum_declared_epochs"]:
        raise CohortValidationError("candidate has too few declared epochs")
    hashes = record["metadata_hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(HASH_FIELDS) or any(not _is_hash(hashes[field]) for field in HASH_FIELDS):
        raise CohortValidationError("candidate metadata hashes are incomplete or invalid")
    if record["query_digest"] != preregistration["candidate_query"]["query_digest"]:
        raise CohortValidationError("candidate was not produced by the frozen metadata query")


def discover_candidates(records: Iterable[Mapping[str, Any]], preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Filter a supplied catalog snapshot without opening data or selecting on outcomes."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for record in records:
        candidate_id = str(record.get("candidate_id", "<missing>")) if isinstance(record, Mapping) else "<invalid>"
        try:
            validate_candidate(record, preregistration)
        except CohortValidationError as error:
            rejected.append({"candidate_id": candidate_id, "reason": str(error)})
        else:
            accepted.append(dict(record))
    accepted.sort(key=lambda candidate: candidate["candidate_id"])
    return {
        "query_digest": preregistration["candidate_query"]["query_digest"],
        "metadata_only": True,
        "bulk_fetch_performed": False,
        "candidates": accepted,
        "rejected": rejected,
    }


def _disjoint(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if any(left[key] == right[key] for key in CLUSTER_KEYS):
        return False
    return not set(left["station_ids"]) & set(right["station_ids"])


def _greedy_disjoint(candidates: Sequence[Mapping[str, Any]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        if len(chosen) < count and all(_disjoint(candidate, prior) for prior in chosen):
            chosen.append(dict(candidate))
        else:
            excluded.append(dict(candidate))
    return chosen, excluded


def build_manifest(discovery: Mapping[str, Any], preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze lexical, outcome-blind cluster membership from a metadata discovery result."""

    if discovery.get("metadata_only") is not True or discovery.get("bulk_fetch_performed") is not False:
        raise CohortValidationError("manifest input must be metadata-only discovery")
    candidates = discovery.get("candidates")
    if not isinstance(candidates, list):
        raise CohortValidationError("discovery candidates must be a list")
    for candidate in candidates:
        validate_candidate(candidate, preregistration)
    ordered_candidates = sorted(candidates, key=lambda item: item["candidate_id"])
    required = preregistration["power"]["maximum_required_evaluable_clusters"]
    surplus_count = preregistration["attrition"]["frozen_surplus_clusters"]
    selected_and_surplus, excluded = _greedy_disjoint(ordered_candidates, required + surplus_count)
    selected = selected_and_surplus[:required]
    surplus = selected_and_surplus[required:]
    status = "frozen_metadata_only" if len(selected) == required and len(surplus) == surplus_count else "not_evaluable_candidate_pool"
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "cohort_id": preregistration["cohort_id"],
        "status": status,
        "outcomes_present": False,
        "preregistration_sha256": canonical_digest(preregistration),
        "candidate_query_digest": discovery["query_digest"],
        "selection_algorithm": "lexical_candidate_id_greedy_disjoint_v1",
        "selection_outcome_blind": True,
        "candidate_pool": [dict(candidate) for candidate in ordered_candidates],
        "frozen_clusters": [dict(candidate) for candidate in selected],
        "surplus_clusters": [dict(candidate) for candidate in surplus],
        "excluded_after_selection": [
            candidate["candidate_id"]
            for candidate in excluded
        ],
    }


def validate_manifest(manifest: Mapping[str, Any], preregistration: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise CohortValidationError("manifest must be an object")
    if set(manifest) != MANIFEST_FIELDS:
        raise CohortValidationError("manifest fields do not match the metadata-only schema")
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise CohortValidationError("manifest schema/version mismatch")
    if manifest.get("outcomes_present") is not False or manifest.get("selection_outcome_blind") is not True:
        raise CohortValidationError("manifest must remain metadata-only and outcome-blind")
    if manifest.get("preregistration_sha256") != canonical_digest(preregistration):
        raise CohortValidationError("manifest is stale for this preregistration")
    if manifest.get("candidate_query_digest") != preregistration["candidate_query"]["query_digest"]:
        raise CohortValidationError("manifest candidate query scope mismatch")
    candidate_pool = manifest.get("candidate_pool")
    frozen = manifest.get("frozen_clusters")
    surplus = manifest.get("surplus_clusters")
    if not all(isinstance(value, list) for value in (candidate_pool, frozen, surplus)):
        raise CohortValidationError("manifest candidate and cluster fields must be lists")
    for candidate in candidate_pool:
        validate_candidate(candidate, preregistration)
    candidate_ids = [candidate["candidate_id"] for candidate in candidate_pool]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CohortValidationError("manifest candidate IDs must be unique")
    pool_ids = {candidate["candidate_id"] for candidate in candidate_pool}
    cluster_ids = [candidate["candidate_id"] for candidate in frozen + surplus]
    if len(cluster_ids) != len(set(cluster_ids)) or not set(cluster_ids) <= pool_ids:
        raise CohortValidationError("frozen clusters must be unique members of the candidate pool")
    for index, candidate in enumerate(frozen + surplus):
        if any(not _disjoint(candidate, other) for other in frozen + surplus if other is not candidate and other["candidate_id"] != candidate["candidate_id"]):
            raise CohortValidationError("frozen and surplus clusters are not disjoint")
    required = preregistration["power"]["maximum_required_evaluable_clusters"]
    surplus_count = preregistration["attrition"]["frozen_surplus_clusters"]
    expected_status = "frozen_metadata_only" if len(frozen) == required and len(surplus) == surplus_count else "not_evaluable_candidate_pool"
    if manifest.get("status") != expected_status:
        raise CohortValidationError("manifest status does not match frozen pool sufficiency")
    expected = build_manifest(
        {
            "query_digest": preregistration["candidate_query"]["query_digest"],
            "metadata_only": True,
            "bulk_fetch_performed": False,
            "candidates": candidate_pool,
        },
        preregistration,
    )
    frozen_fields = (
        "cohort_id",
        "status",
        "selection_algorithm",
        "candidate_pool",
        "frozen_clusters",
        "surplus_clusters",
        "excluded_after_selection",
    )
    if any(manifest.get(field) != expected[field] for field in frozen_fields):
        raise CohortValidationError("manifest does not match the exact lexical disjoint freeze")
