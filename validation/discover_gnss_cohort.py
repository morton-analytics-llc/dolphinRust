#!/usr/bin/env python
"""Discover a blind public GNSS/OPERA cohort without inspecting InSAR outcomes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import asf_search as asf
import requests

from heldout_temporal_covariance.cohort import (
    build_manifest,
    canonical_digest,
    validate_candidate,
    validate_manifest,
)

HOLDINGS_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
CANDIDATE_FIELDS = {
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


@dataclass(frozen=True)
class Station:
    station_id: str
    latitude: float
    longitude: float
    height_m: float
    first_date: dt.date
    last_date: dt.date
    solution_count: int


@dataclass(frozen=True)
class Exclusions:
    station_ids: frozenset[str]
    burst_ids: frozenset[str]
    site_ids: frozenset[str]


class CachedCatalogResult:
    def __init__(self, feature: dict[str, Any]) -> None:
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            raise ValueError("cached ASF result is not a GeoJSON feature")
        self._feature = feature
        self.properties = properties

    def geojson(self) -> dict[str, Any]:
        return self._feature


class MetadataSearchCache:
    SCHEMA = "dolphinrust.asf_metadata_search_cache"
    VERSION = 2
    CATALOG_PROPERTIES = (
        "operaBurstID",
        "startTime",
        "fileName",
        "fileID",
        "flightDirection",
        "pathNumber",
        "orbit",
        "groupID",
        "processingDate",
        "productVersion",
    )

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if path is not None and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != self.SCHEMA:
                raise ValueError("ASF metadata cache schema/version mismatch")
            version = payload.get("schema_version")
            if version == 1:
                self._migrate_legacy(payload)
            elif version == self.VERSION:
                self._load_entries()
            else:
                raise ValueError("ASF metadata cache schema/version mismatch")

    @property
    def entry_directory(self) -> Path | None:
        if self.path is None:
            return None
        return self.path.with_name(self.path.name + ".entries")

    def _normalize_feature(self, feature: dict[str, Any]) -> dict[str, Any]:
        result = CachedCatalogResult(feature)
        missing = [
            field
            for field in self.CATALOG_PROPERTIES
            if result.properties.get(field) is None
        ]
        if missing:
            raise ValueError(
                "ASF result is missing cache metadata fields: " + ", ".join(missing)
            )
        return {
            "type": "Feature",
            "geometry": result.geojson()["geometry"],
            "properties": {
                field: result.properties[field] for field in self.CATALOG_PROPERTIES
            },
        }

    def _entry(self, request: dict[str, Any], features: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = sorted(
            (self._normalize_feature(feature) for feature in features),
            key=lambda feature: (
                feature["properties"]["startTime"],
                feature["properties"]["fileName"],
                canonical_digest(feature),
            ),
        )
        return {
            "request": request,
            "results": normalized,
            "results_sha256": canonical_digest(normalized),
        }

    def _validate_entry(self, key: str, entry: dict[str, Any]) -> None:
        request = entry.get("request")
        features = entry.get("results")
        if not isinstance(request, dict) or canonical_digest(request) != key:
            raise ValueError("ASF metadata cache request identity mismatch")
        if not isinstance(features, list) or entry.get("results_sha256") != canonical_digest(features):
            raise ValueError("ASF metadata cache result hash mismatch")
        for feature in features:
            self._normalize_feature(feature)

    def _migrate_legacy(self, payload: dict[str, Any]) -> None:
        legacy = payload.get("entries")
        if not isinstance(legacy, dict):
            raise ValueError("ASF metadata cache entries are invalid")
        for key, value in legacy.items():
            request = value.get("request") if isinstance(value, dict) else None
            features = value.get("results") if isinstance(value, dict) else None
            if not isinstance(request, dict) or not isinstance(features, list):
                raise ValueError("ASF metadata cache legacy entry is invalid")
            entry = self._entry(request, features)
            self._validate_entry(key, entry)
            self.entries[key] = entry
            self._persist_entry(key)
        self._persist_index()

    def _load_entries(self) -> None:
        directory = self.entry_directory
        if directory is None or not directory.is_dir():
            raise ValueError("ASF metadata cache entry directory is missing")
        for entry_path in sorted(directory.glob("*.json")):
            key = entry_path.stem
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            self._validate_entry(key, entry)
            self.entries[key] = entry

    def _request(self, station: Station, start: str, end: str) -> dict[str, Any]:
        return {
            "dataset": "OPERA-S1",
            "processing_level": "CSLC",
            "station_id": station.station_id,
            "latitude": station.latitude,
            "longitude": station.longitude,
            "start": start,
            "end": end,
            "maximum_results": 500,
        }

    def search(self, station: Station, start: str, end: str) -> list[Any]:
        request = self._request(station, start, end)
        key = canonical_digest(request)
        if key in self.entries:
            self.hits += 1
            entry = self.entries[key]
            if entry.get("request") != request:
                raise ValueError("ASF metadata cache request identity mismatch")
            features = entry.get("results")
            if not isinstance(features, list) or entry.get("results_sha256") != canonical_digest(features):
                raise ValueError("ASF metadata cache result hash mismatch")
            return [CachedCatalogResult(feature) for feature in features]
        self.misses += 1
        features = [result.geojson() for result in search_station(station, start, end)]
        self.entries[key] = self._entry(request, features)
        self._persist_entry(key)
        self._persist_index()
        return [CachedCatalogResult(feature) for feature in self.entries[key]["results"]]

    def prefetch(
        self,
        stations: Sequence[Station],
        start: str,
        end: str,
        workers: int,
    ) -> None:
        if workers <= 0 or workers > 8:
            raise ValueError("metadata prefetch workers must be between 1 and 8")
        requests_by_key: dict[str, tuple[Station, dict[str, Any]]] = {}
        for station in stations:
            request = self._request(station, start, end)
            requests_by_key[canonical_digest(request)] = (station, request)
        missing: dict[str, tuple[Station, dict[str, Any]]] = {}
        for key, value in requests_by_key.items():
            if key in self.entries:
                self._validate_entry(key, self.entries[key])
                self.hits += 1
            else:
                missing[key] = value
        if not missing:
            return

        def fetch(station: Station, request: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            return request, [result.geojson() for result in search_station(station, start, end)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch, station, request): key
                for key, (station, request) in missing.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                request, features = future.result()
                self.entries[key] = self._entry(request, features)
                self.misses += 1
                self._persist_entry(key)
        self._persist_index()

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _persist_entry(self, key: str) -> None:
        directory = self.entry_directory
        if directory is not None:
            self._atomic_write(directory / f"{key}.json", self.entries[key])

    def _persist_index(self) -> None:
        if self.path is not None:
            self._atomic_write(
                self.path,
                {
                    "schema": self.SCHEMA,
                    "schema_version": self.VERSION,
                    "entry_directory": self.entry_directory.name,
                },
            )


def load_exclusions(path: Path) -> Exclusions:
    payload = json.loads(path.read_text())
    values = payload.get("exclusions")
    if not isinstance(values, dict):
        raise ValueError("held-out preregistration is missing exclusions")

    def identifiers(field: str) -> frozenset[str]:
        entries = values.get(field)
        if not isinstance(entries, list) or not all(
            isinstance(entry, str) and entry for entry in entries
        ):
            raise ValueError(f"held-out exclusions.{field} is invalid")
        if field == "site_ids":
            return frozenset(entry.strip().lower() for entry in entries)
        return frozenset(entry.strip().upper() for entry in entries)

    return Exclusions(
        station_ids=identifiers("station_ids"),
        burst_ids=identifiers("burst_ids"),
        site_ids=identifiers("site_ids"),
    )


def exclude_stations(
    stations: Sequence[Station], exclusions: Exclusions
) -> list[Station]:
    excluded = {station_id.strip().upper() for station_id in exclusions.station_ids}
    return [
        station
        for station in stations
        if station.station_id.strip().upper() not in excluded
    ]


def cohort_site_id(
    burst_id: str, first_station_id: str, second_station_id: str
) -> str:
    return (
        f"{burst_id.lower()}_{first_station_id.lower()}_"
        f"{second_station_id.lower()}"
    )


def select_shared_burst(
    shared: Sequence[tuple[str, list[str]]],
    used_bursts: set[str],
    first_station_id: str,
    second_station_id: str,
    exclusions: Exclusions,
) -> tuple[str, list[str]] | tuple[None, None]:
    excluded_bursts = {
        excluded.strip().upper() for excluded in exclusions.burst_ids
    }
    excluded_sites = {excluded.strip().lower() for excluded in exclusions.site_ids}
    used = {used.strip().upper() for used in used_bursts}
    for burst, dates in shared:
        site_id = cohort_site_id(burst, first_station_id, second_station_id)
        canonical_burst = burst.strip().upper()
        if (
            canonical_burst not in used
            and canonical_burst not in excluded_bursts
            and site_id.strip().lower() not in excluded_sites
        ):
            return burst, dates
    return None, None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def station_metadata(station: Station) -> dict[str, Any]:
    return {
        "station_id": station.station_id,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "height_m": station.height_m,
        "first_date": station.first_date.isoformat(),
        "last_date": station.last_date.isoformat(),
        "solution_count": station.solution_count,
    }


def parse_holdings(text: str) -> list[Station]:
    stations: list[Station] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("Sta "):
            continue
        fields = line.split()
        if len(fields) < 11:
            raise ValueError(f"holdings line {line_number} has {len(fields)} fields")
        try:
            longitude = float(fields[2])
            if longitude > 180.0:
                longitude -= 360.0
            stations.append(
                Station(
                    station_id=fields[0],
                    latitude=float(fields[1]),
                    longitude=longitude,
                    height_m=float(fields[3]),
                    first_date=dt.date.fromisoformat(fields[7]),
                    last_date=dt.date.fromisoformat(fields[8]),
                    solution_count=int(fields[10]),
                )
            )
        except ValueError as error:
            raise ValueError(f"invalid holdings line {line_number}: {error}") from error
    if not stations:
        raise ValueError("NGL holdings catalog is empty")
    return stations


def haversine_km(first: Station, second: Station) -> float:
    lat1, lat2 = math.radians(first.latitude), math.radians(second.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(second.longitude - first.longitude)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(value))


def candidate_pairs(
    stations: Sequence[Station],
    start: dt.date,
    end: dt.date,
    min_distance_km: float,
    max_distance_km: float,
    bounds: tuple[float, float, float, float],
) -> list[tuple[Station, Station, float]]:
    west, south, east, north = bounds
    eligible = [
        station
        for station in stations
        if station.first_date <= start
        and station.last_date >= end
        and west <= station.longitude <= east
        and south <= station.latitude <= north
    ]
    buckets: dict[tuple[int, int], list[Station]] = {}
    for station in eligible:
        key = (math.floor(station.latitude), math.floor(station.longitude))
        buckets.setdefault(key, []).append(station)
    pairs: list[tuple[Station, Station, float]] = []
    seen: set[tuple[str, str]] = set()
    for first in eligible:
        lat_cell = math.floor(first.latitude)
        lon_cell = math.floor(first.longitude)
        for dlat in (-1, 0, 1):
            for dlon in (-2, -1, 0, 1, 2):
                for second in buckets.get((lat_cell + dlat, lon_cell + dlon), []):
                    pair_id = tuple(sorted((first.station_id, second.station_id)))
                    if first.station_id == second.station_id or pair_id in seen:
                        continue
                    seen.add(pair_id)
                    distance = haversine_km(first, second)
                    if min_distance_km <= distance <= max_distance_km:
                        ordered = sorted((first, second), key=lambda value: value.station_id)
                        pairs.append((ordered[0], ordered[1], distance))
    return sorted(
        pairs,
        key=lambda item: (
            -min(item[0].solution_count, item[1].solution_count),
            item[2],
            item[0].station_id,
            item[1].station_id,
        ),
    )


def search_station(station: Station, start: str, end: str) -> list[Any]:
    return list(
        asf.search(
            dataset="OPERA-S1",
            processingLevel="CSLC",
            intersectsWith=f"POINT({station.longitude} {station.latitude})",
            start=start,
            end=end,
            maxResults=500,
        )
    )


def results_by_burst(results: Sequence[Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for result in results:
        burst = result.properties.get("operaBurstID")
        start_time = result.properties.get("startTime")
        if not isinstance(burst, str) or not isinstance(start_time, str):
            continue
        date = start_time[:10]
        current = output.setdefault(burst, {})
        prior = current.get(date)
        if prior is None or str(result.properties.get("fileName")) > str(prior.properties.get("fileName")):
            current[date] = result
    return output


def result_geometry(result: Any) -> dict[str, Any]:
    feature = result.geojson()
    geometry = feature.get("geometry") if isinstance(feature, dict) else None
    if not isinstance(geometry, dict):
        raise ValueError("ASF result is missing GeoJSON geometry")
    return geometry


def frame_identity(properties: dict[str, Any]) -> tuple[str, str]:
    direction = properties.get("flightDirection")
    path_number = properties.get("pathNumber")
    burst_id = properties.get("operaBurstID")
    group_id = properties.get("groupID")
    if not isinstance(direction, str) or not isinstance(path_number, int):
        raise ValueError("ASF result is missing flight direction or relative orbit")
    if not isinstance(burst_id, str) or not burst_id:
        raise ValueError("ASF result is missing burst identity")
    if not isinstance(group_id, str):
        raise ValueError("ASF result is missing scene group metadata")
    parts = group_id.split("_")
    if len(parts) < 6 or not parts[2].isdigit() or not parts[3].isdigit():
        raise ValueError("ASF scene group metadata is not recognized")
    orbit_id = f"{direction.lower()}-r{path_number:03d}"
    return orbit_id, f"{orbit_id}-burst-{burst_id.lower()}"


def burst_metadata(result: Any) -> dict[str, Any]:
    properties = result.properties
    required = (
        "operaBurstID",
        "startTime",
        "fileName",
        "fileID",
        "flightDirection",
        "pathNumber",
        "orbit",
        "groupID",
        "processingDate",
        "productVersion",
    )
    missing = [field for field in required if properties.get(field) is None]
    if missing:
        raise ValueError("ASF result is missing metadata fields: " + ", ".join(missing))
    return {
        **{field: properties[field] for field in required},
        "geometry": result_geometry(result),
    }


def catalog_candidate(
    first: Station,
    second: Station,
    burst: str,
    dates: Sequence[str],
    results: dict[str, Any],
    catalog_sha256: str,
    query_digest: str,
) -> dict[str, Any]:
    if not dates or set(dates) != set(results):
        raise ValueError("candidate dates and ASF metadata do not match")
    metadata = [burst_metadata(results[date]) for date in dates]
    identities = {frame_identity(entry) for entry in metadata}
    if len(identities) != 1:
        raise ValueError("candidate ASF epochs do not share an orbit/frame")
    orbit_id, frame_id = identities.pop()
    geometry_digests = sorted(
        {canonical_digest(entry["geometry"]) for entry in metadata}
    )
    station_ids = sorted((first.station_id, second.station_id))
    site_id = cohort_site_id(burst, station_ids[0], station_ids[1])
    candidate = {
        "candidate_id": site_id,
        "source_kind": "catalog_metadata",
        "burst_id": burst,
        "orbit_id": orbit_id,
        "footprint_id": "sha256-" + canonical_digest(geometry_digests),
        "site_id": site_id,
        "frame_id": frame_id,
        "station_ids": station_ids,
        "date_start": dates[0],
        "date_end": dates[-1],
        "epoch_count": len(dates),
        "metadata_hashes": {
            "catalog_sha256": catalog_sha256,
            "burst_metadata_sha256": canonical_digest(metadata),
            "gnss_station_metadata_sha256": canonical_digest(
                sorted(
                    (station_metadata(first), station_metadata(second)),
                    key=lambda value: value["station_id"],
                )
            ),
        },
        "query_digest": query_digest,
    }
    if set(candidate) != CANDIDATE_FIELDS:
        raise AssertionError("candidate schema construction is incomplete")
    return candidate


def validate_runtime_bounds(
    bounds: tuple[float, float, float, float],
    preregistration: dict[str, Any],
) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError("runtime geographic bounds must be an increasing WGS84 extent")
    frozen_query = preregistration["candidate_query"]["query"]
    allowed = frozen_query.get("geographic_bounds", [-180.0, -90.0, 180.0, 90.0])
    if (
        not isinstance(allowed, list)
        or len(allowed) != 4
        or not all(isinstance(value, (int, float)) for value in allowed)
    ):
        raise ValueError("preregistered geographic bounds are invalid")
    allowed_west, allowed_south, allowed_east, allowed_north = map(float, allowed)
    if west < allowed_west or south < allowed_south or east > allowed_east or north > allowed_north:
        raise ValueError("runtime geographic bounds exceed the preregistered geography")
    return bounds


def runtime_query(args: argparse.Namespace, preregistration: dict[str, Any]) -> dict[str, Any]:
    bounds = validate_runtime_bounds(
        (args.west, args.south, args.east, args.north), preregistration
    )
    return {
        "criteria": {
            "target_sites": args.target_sites,
            "minimum_epochs": args.min_epochs,
            "minimum_span_days": args.min_span_days,
            "station_pair_distance_km": [args.min_distance_km, args.max_distance_km],
            "date_window": [args.start, args.end],
            "geographic_bounds": list(bounds),
            "maximum_pairs_examined": args.max_pairs,
        },
        "candidate_query_digest": preregistration["candidate_query"]["query_digest"],
    }


def runtime_query_digest(query: dict[str, Any]) -> str:
    return canonical_digest(query)


def discover(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    runtime = runtime_query(args, preregistration)
    bounds = tuple(runtime["criteria"]["geographic_bounds"])
    exclusions = load_exclusions(args.preregistration)
    response = requests.get(HOLDINGS_URL, timeout=60)
    response.raise_for_status()
    stations = exclude_stations(parse_holdings(response.text), exclusions)
    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    pairs = candidate_pairs(
        stations,
        start_date,
        end_date,
        args.min_distance_km,
        args.max_distance_km,
        bounds,
    )
    selected: list[dict[str, Any]] = []
    examined: list[dict[str, Any]] = []
    used_bursts: set[str] = set()
    search_cache = MetadataSearchCache(getattr(args, "metadata_cache", None))
    bounded_pairs = pairs[: args.max_pairs]
    stations_by_id = {
        station.station_id: station
        for first, second, _ in bounded_pairs
        for station in (first, second)
    }
    search_cache.prefetch(
        [stations_by_id[key] for key in sorted(stations_by_id)],
        args.start,
        args.end,
        workers=getattr(args, "metadata_workers", 8),
    )
    for first, second, distance in bounded_pairs:
        first_bursts = results_by_burst(search_cache.search(first, args.start, args.end))
        second_bursts = results_by_burst(search_cache.search(second, args.start, args.end))
        shared: list[tuple[str, list[str]]] = []
        for burst in sorted(set(first_bursts) & set(second_bursts)):
            dates = sorted(set(first_bursts[burst]) & set(second_bursts[burst]))
            span = (dt.date.fromisoformat(dates[-1]) - dt.date.fromisoformat(dates[0])).days if dates else 0
            if len(dates) >= args.min_epochs and span >= args.min_span_days:
                shared.append((burst, dates))
        examined_entry = {
            "stations": [first.station_id, second.station_id],
            "distance_km": distance,
            "eligible_bursts": [burst for burst, _ in shared],
            "excluded_bursts": [
                burst for burst, _ in shared if burst in exclusions.burst_ids
            ],
        }
        examined.append(examined_entry)
        if not shared:
            continue
        shared.sort(key=lambda item: (-len(item[1]), item[0]))
        burst, dates = select_shared_burst(
            shared,
            used_bursts,
            first.station_id,
            second.station_id,
            exclusions,
        )
        if burst is None or dates is None:
            continue
        candidate = catalog_candidate(
            first,
            second,
            burst,
            dates,
            {date: first_bursts[burst][date] for date in dates},
            sha256_text(response.text),
            preregistration["candidate_query"]["query_digest"],
        )
        validate_candidate(candidate, preregistration)
        selected.append(candidate)
        used_bursts.add(burst)
        if len(selected) == args.target_sites:
            break
    return {
        "schema": "dolphinrust.temporal_covariance.heldout_discovery",
        "schema_version": 1,
        "status": "eligible" if len(selected) == args.target_sites else "not_evaluable",
        "query_digest": preregistration["candidate_query"]["query_digest"],
        "metadata_only": True,
        "bulk_fetch_performed": False,
        "selection_outcome_blind": True,
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(args.preregistration.read_bytes()).hexdigest(),
        },
        "exclusions": {
            "station_ids": sorted(exclusions.station_ids),
            "burst_ids": sorted(exclusions.burst_ids),
            "site_ids": sorted(exclusions.site_ids),
        },
        "criteria": runtime["criteria"],
        "holdings_source": {
            "url": HOLDINGS_URL,
            "sha256": sha256_text(response.text),
            "bytes": len(response.content),
        },
        "runtime_query": runtime,
        "runtime_query_digest": runtime_query_digest(runtime),
        "candidates": selected,
        "rejected": [],
        "examined_pairs": examined,
        "metadata_cache": {
            "persisted": search_cache.path is not None,
            "entry_count": len(search_cache.entries),
            "hits": search_cache.hits,
            "misses": search_cache.misses,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--target-sites", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--min-span-days", type=int, default=330)
    parser.add_argument("--min-distance-km", type=float, default=1.0)
    parser.add_argument("--max-distance-km", type=float, default=30.0)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--west", type=float, default=-170.0)
    parser.add_argument("--south", type=float, default=5.0)
    parser.add_argument("--east", type=float, default=-50.0)
    parser.add_argument("--north", type=float, default=75.0)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).with_name(
            "temporal_covariance_heldout_preregistration.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--metadata-cache", type=Path)
    parser.add_argument("--metadata-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = discover(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if args.manifest_output is not None:
        preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
        manifest = build_manifest(payload, preregistration)
        validate_manifest(manifest, preregistration)
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "candidate_count": len(payload["candidates"]),
                "output": str(args.output),
                "manifest_output": str(args.manifest_output) if args.manifest_output else None,
            },
            indent=2,
        )
    )
    if payload["status"] != "eligible":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
