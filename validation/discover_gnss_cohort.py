#!/usr/bin/env python
"""Discover a blind public GNSS/OPERA cohort without inspecting InSAR outcomes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import asf_search as asf
import requests

from fetch_real import result_hdf5_bytes
from gps_ground_truth import NotEvaluable, align_records, parse_tenv3

HOLDINGS_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
TENV3_ROOT = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20"
STATION_ROOT = "https://geodesy.unr.edu/NGLStationPages/stations"


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
        return frozenset(entries)

    return Exclusions(
        station_ids=identifiers("station_ids"),
        burst_ids=identifiers("burst_ids"),
        site_ids=identifiers("site_ids"),
    )


def exclude_stations(
    stations: Sequence[Station], exclusions: Exclusions
) -> list[Station]:
    return [
        station
        for station in stations
        if station.station_id not in exclusions.station_ids
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
    for burst, dates in shared:
        site_id = cohort_site_id(burst, first_station_id, second_station_id)
        if (
            burst not in used_bursts
            and burst not in exclusions.burst_ids
            and site_id not in exclusions.site_ids
        ):
            return burst, dates
    return None, None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
                        pairs.append((first, second, distance))
    # One metadata-blind pair per 5-degree cell. Prefer longer records, then shorter separation.
    by_region: dict[tuple[int, int], tuple[Station, Station, float]] = {}
    for pair in pairs:
        first, second, distance = pair
        region = (
            math.floor(((first.latitude + second.latitude) / 2) / 5),
            math.floor(((first.longitude + second.longitude) / 2) / 5),
        )
        incumbent = by_region.get(region)
        rank = (min(first.solution_count, second.solution_count), -distance, pair[0].station_id, pair[1].station_id)
        if incumbent is None:
            by_region[region] = pair
            continue
        incumbent_rank = (
            min(incumbent[0].solution_count, incumbent[1].solution_count),
            -incumbent[2],
            incumbent[0].station_id,
            incumbent[1].station_id,
        )
        if rank > incumbent_rank:
            by_region[region] = pair
    return sorted(by_region.values(), key=lambda item: (-min(item[0].solution_count, item[1].solution_count), item[0].station_id, item[1].station_id))


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


def aligned_fraction(station: Station, dates: Sequence[str], max_gap_days: int) -> tuple[float, dict[str, Any]]:
    url = f"{TENV3_ROOT}/{station.station_id}.tenv3"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    records = parse_tenv3(response.text)
    aligned = 0
    quality: dict[str, str] = {}
    for value in dates:
        date = dt.date.fromisoformat(value)
        try:
            match = align_records(records, [date], max_gap_days)[0]
        except NotEvaluable:
            quality[value] = "unavailable"
        else:
            aligned += 1
            quality[value] = match.quality
    return aligned / len(dates), {
        "url": url,
        "sha256": sha256_text(response.text),
        "bytes": len(response.content),
        "quality": quality,
    }


def discover(args: argparse.Namespace) -> dict[str, Any]:
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
        (args.west, args.south, args.east, args.north),
    )
    selected: list[dict[str, Any]] = []
    examined: list[dict[str, Any]] = []
    used_bursts: set[str] = set()
    for first, second, distance in pairs[: args.max_pairs]:
        first_bursts = results_by_burst(search_station(first, args.start, args.end))
        second_bursts = results_by_burst(search_station(second, args.start, args.end))
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
        first_fraction, first_source = aligned_fraction(first, dates, args.max_gap_days)
        second_fraction, second_source = aligned_fraction(second, dates, args.max_gap_days)
        common_fraction = sum(
            first_source["quality"][date] != "unavailable"
            and second_source["quality"][date] != "unavailable"
            for date in dates
        ) / len(dates)
        if common_fraction < args.min_gnss_fraction:
            examined_entry["gnss_fraction"] = {
                first.station_id: first_fraction,
                second.station_id: second_fraction,
                "common": common_fraction,
            }
            continue
        results = [first_bursts[burst][date] for date in dates]
        estimated_bytes = sum(result_hdf5_bytes(result) or 0 for result in results)
        site_id = cohort_site_id(burst, first.station_id, second.station_id)
        selected.append(
            {
                "site_id": site_id,
                "burst_id": burst,
                "burst_filename_id": burst.replace("_", "-"),
                "start": dates[0],
                "end": dates[-1],
                "expected_dates": dates,
                "epoch_count": len(dates),
                "span_days": (dt.date.fromisoformat(dates[-1]) - dt.date.fromisoformat(dates[0])).days,
                "distance_km": distance,
                "common_gnss_fraction": common_fraction,
                "estimated_cslc_bytes": estimated_bytes,
                "stations": {
                    station.station_id: {
                        "latitude": station.latitude,
                        "longitude": station.longitude,
                        "tenv3_url": f"{TENV3_ROOT}/{station.station_id}.tenv3",
                        "metadata_url": f"{STATION_ROOT}/{station.station_id}.sta",
                        "availability_fraction": fraction,
                        "source": source,
                    }
                    for station, fraction, source in [
                        (first, first_fraction, first_source),
                        (second, second_fraction, second_source),
                    ]
                },
                "comparison": {
                    "id": f"{first.station_id}_minus_{second.station_id}",
                    "primary_station": first.station_id,
                    "control_station": second.station_id,
                },
            }
        )
        used_bursts.add(burst)
        if len(selected) == args.target_sites:
            break
    total_bytes = sum(site["estimated_cslc_bytes"] for site in selected)
    return {
        "schema": "dolphinrust-gnss-cohort-feasibility/1",
        "status": "eligible" if len(selected) == args.target_sites else "not_evaluable",
        "selection_blinded_to_insar_outcomes": True,
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(args.preregistration.read_bytes()).hexdigest(),
        },
        "exclusions": {
            "station_ids": sorted(exclusions.station_ids),
            "burst_ids": sorted(exclusions.burst_ids),
            "site_ids": sorted(exclusions.site_ids),
        },
        "criteria": {
            "target_sites": args.target_sites,
            "minimum_epochs": args.min_epochs,
            "minimum_span_days": args.min_span_days,
            "minimum_gnss_fraction": args.min_gnss_fraction,
            "maximum_interpolation_gap_days": args.max_gap_days,
            "station_pair_distance_km": [args.min_distance_km, args.max_distance_km],
            "date_window": [args.start, args.end],
            "geographic_bounds": [args.west, args.south, args.east, args.north],
            "maximum_pairs_examined": args.max_pairs,
        },
        "holdings_source": {
            "url": HOLDINGS_URL,
            "sha256": sha256_text(response.text),
            "bytes": len(response.content),
        },
        "selected_sites": selected,
        "selected_site_count": len(selected),
        "estimated_cslc_bytes": total_bytes,
        "estimated_cslc_gb": total_bytes / 1e9,
        "examined_pairs": examined,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--target-sites", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--min-span-days", type=int, default=330)
    parser.add_argument("--min-gnss-fraction", type=float, default=0.9)
    parser.add_argument("--max-gap-days", type=int, default=4)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = discover(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_site_count": payload["selected_site_count"],
                "estimated_cslc_gb": payload["estimated_cslc_gb"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if payload["status"] != "eligible":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
