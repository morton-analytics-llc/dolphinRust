#!/usr/bin/env python
"""Fetch real OPERA CSLC-S1 data used by validation fixtures.

The legacy mode remains unchanged::

    source validation/creds.sh
    oracle/.venv/bin/python validation/fetch_real.py --burst T063_133231_IW1 --n 5

The GNSS ground-truth mode uses a tracked recipe, resolves the exact declared
epochs, and optionally downloads the matching per-burst CSLC-S1-STATIC product::

    oracle/.venv/bin/python validation/fetch_real.py \
      --recipe validation/gps_mmx1.json --dry-run --with-static
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asf_search as asf
import h5py

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "real_data"
DATE_RE = re.compile(r"_(20\d{6})T")


def require_token(environment: Mapping[str, str] = os.environ) -> str:
    """Return the Earthdata bearer token or fail without exposing its value."""
    token = environment.get("GP_EARTHDATA_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "GP_EARTHDATA_TOKEN is missing; source validation/creds.sh and check .env"
        )
    return token


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text())
    if recipe.get("schema") != "dolphinrust-gps-ground-truth-recipe/1":
        raise ValueError(f"unsupported GPS recipe schema in {path}")
    return recipe


def result_date(result: Any) -> str:
    start = result.properties.get("startTime")
    if isinstance(start, str) and len(start) >= 10:
        return start[:10]
    name = str(result.properties.get("fileName", ""))
    match = DATE_RE.search(name)
    if match:
        stamp = match.group(1)
        return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    raise ValueError(f"result has no acquisition date: {name or '<unnamed>'}")


def select_expected_cslcs(
    results: Sequence[Any], expected_dates: Sequence[str], burst_filename_id: str
) -> list[Any]:
    """Select exactly one CSLC for each declared date, in declared order."""
    by_date: dict[str, list[Any]] = {date: [] for date in expected_dates}
    wrong_burst: list[str] = []
    for result in results:
        props = result.properties
        if props.get("processingLevel") != "CSLC":
            continue
        name = str(props.get("fileName", ""))
        date = result_date(result)
        if date not in by_date:
            continue
        if burst_filename_id not in name:
            wrong_burst.append(name)
            continue
        by_date[date].append(result)
    if wrong_burst:
        raise ValueError(f"CSLC burst mismatch: {wrong_burst[0]}")
    missing = [date for date, matches in by_date.items() if not matches]
    duplicates = [date for date, matches in by_date.items() if len(matches) > 1]
    if missing:
        raise ValueError(f"missing declared CSLC dates: {', '.join(missing)}")
    if duplicates:
        raise ValueError(f"duplicate CSLC results for dates: {', '.join(duplicates)}")
    return [by_date[date][0] for date in expected_dates]


def select_static_result(results: Sequence[Any], burst_filename_id: str) -> Any:
    matches = [
        result
        for result in results
        if result.properties.get("processingLevel") == "CSLC-STATIC"
        and burst_filename_id in str(result.properties.get("fileName", ""))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one matching CSLC-STATIC for {burst_filename_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def is_valid_h5(path: Path, required_datasets: Sequence[str] = ("/data/VV",)) -> bool:
    if not path.exists() or path.stat().st_size <= 1_000_000:
        return False
    with path.open("rb") as stream:
        if stream.read(4) != b"\x89HDF":
            return False
    try:
        with h5py.File(path, "r") as product:
            return all(dataset in product for dataset in required_datasets)
    except OSError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hdf5_text(product: h5py.File, path: str) -> str:
    value = product[path][()]
    if isinstance(value, bytes):
        return value.decode().strip("\x00")
    if hasattr(value, "tobytes"):
        return value.tobytes().decode().strip("\x00")
    return str(value)


def product_identity(path: Path) -> dict[str, str]:
    try:
        with h5py.File(path, "r") as product:
            return {
                "burst_id": hdf5_text(product, "/identification/burst_id"),
                "orbit_pass_direction": hdf5_text(
                    product, "/identification/orbit_pass_direction"
                ),
            }
    except (KeyError, OSError, UnicodeDecodeError) as error:
        raise RuntimeError(f"product identification is unreadable: {path}: {error}") from error


def normalize_burst_id(value: str) -> str:
    return value.lower().replace("-", "_")


def validate_product_identities(
    cslc_paths: Sequence[Path], static_paths: Sequence[Path], expected_burst: str
) -> dict[str, Any]:
    identities = {
        "cslc": [product_identity(path) for path in cslc_paths],
        "static": [product_identity(path) for path in static_paths],
    }
    all_identities = [*identities["cslc"], *identities["static"]]
    expected = normalize_burst_id(expected_burst)
    wrong_burst = [
        identity["burst_id"]
        for identity in all_identities
        if normalize_burst_id(identity["burst_id"]) != expected
    ]
    if wrong_burst:
        raise RuntimeError(
            f"downloaded product burst identity does not match {expected_burst}: {wrong_burst[0]}"
        )
    passes = {identity["orbit_pass_direction"].lower() for identity in all_identities}
    unrecognized = passes - {"ascending", "descending"}
    if unrecognized:
        raise RuntimeError(
            f"downloaded product has unrecognized orbit pass: {sorted(unrecognized)}"
        )
    if len(passes) > 1:
        raise RuntimeError(f"CSLC/STATIC orbit pass identities disagree: {sorted(passes)}")
    return {**identities, "orbit_pass_direction": next(iter(passes), None)}


def result_name(result: Any) -> str:
    name = result.properties.get("fileName")
    if not isinstance(name, str) or not name:
        raise ValueError("ASF result is missing fileName")
    return name


def result_manifest_entry(result: Any, path: Path | None = None) -> dict[str, Any]:
    props = result.properties
    entry: dict[str, Any] = {
        "file_name": result_name(result),
        "processing_level": props.get("processingLevel"),
        "start_time": props.get("startTime"),
        "source": props.get("url") or props.get("browse"),
    }
    if path is not None:
        entry.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entry


def result_hdf5_bytes(result: Any) -> int | None:
    products = result.properties.get("bytes", {})
    if not isinstance(products, dict):
        return None
    sizes = [
        item.get("bytes")
        for item in products.values()
        if isinstance(item, dict) and item.get("format") == "HDF5"
    ]
    return int(sizes[0]) if sizes and isinstance(sizes[0], int) else None


def download_result(
    result: Any,
    out: Path,
    session: asf.ASFSession,
    required_datasets: Sequence[str],
    prior_hash: str | None = None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    destination = out / result_name(result)
    if destination.exists():
        if not is_valid_h5(destination, required_datasets):
            raise RuntimeError(f"existing product is invalid; refusing overwrite: {destination}")
        current_hash = sha256_file(destination)
        if prior_hash is not None and current_hash != prior_hash:
            raise RuntimeError(f"existing product hash differs from manifest: {destination}")
        print(f"  have  {destination.name}")
        return destination
    print(f"  fetch {destination.name} ...", flush=True)
    result.download(path=str(out), session=session)
    if not is_valid_h5(destination, required_datasets):
        raise RuntimeError(f"downloaded product failed HDF5 validation: {destination}")
    print(f"        {destination.stat().st_size / 1e6:.1f} MB [ok]")
    return destination


def prior_hashes(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    entries = [*manifest.get("cslc", []), *manifest.get("static", [])]
    return {
        entry["file_name"]: entry["sha256"]
        for entry in entries
        if entry.get("file_name") and entry.get("sha256")
    }


def search_recipe_products(recipe: Mapping[str, Any], with_static: bool) -> tuple[list[Any], list[Any]]:
    common = {
        "dataset": "OPERA-S1",
        "operaBurstID": recipe["burst_id"],
        "maxResults": 100,
    }
    cslc_results = asf.search(
        **common,
        processingLevel="CSLC",
        start=recipe["start"],
        end=recipe["end"],
    )
    cslcs = select_expected_cslcs(
        cslc_results, recipe["expected_dates"], recipe["burst_filename_id"]
    )
    static = []
    if with_static:
        static_results = asf.search(**common, processingLevel="CSLC-STATIC")
        static = [select_static_result(static_results, recipe["burst_filename_id"])]
    return cslcs, static


def run_recipe(args: argparse.Namespace) -> None:
    recipe_path = args.recipe.resolve()
    recipe = load_recipe(recipe_path)
    with_static = args.with_static or args.static_only
    cslcs, static = search_recipe_products(recipe, with_static)
    print(f"burst {recipe['burst_id']}: resolved {len(cslcs)} declared CSLC epochs")
    for result in cslcs:
        print(f"  {result_date(result)}  {result_name(result)}")
    for result in static:
        print(f"  STATIC      {result_name(result)}")
    expected_bytes = sum(result_hdf5_bytes(result) or 0 for result in [*cslcs, *static])
    print(f"expected HDF5 transfer: {expected_bytes / 1e9:.2f} GB")
    if args.dry_run:
        return

    token = require_token()
    session = asf.ASFSession().auth_with_token(token)
    out = args.out.resolve() if args.out else OUT / "gps_mmx1" / "source"
    manifest_path = out / "acquisition_manifest.json"
    known_hashes = prior_hashes(manifest_path)
    cslc_paths = [] if args.static_only else [
        download_result(
            result,
            out / "cslc",
            session,
            ("/data/VV", "/data/x_coordinates", "/data/y_coordinates", "/data/projection"),
            known_hashes.get(result_name(result)),
        )
        for result in cslcs
    ]
    static_paths = [
        download_result(
            result,
            out / "static",
            session,
            (
                "/data/los_east",
                "/data/los_north",
                "/data/x_coordinates",
                "/data/y_coordinates",
                "/data/projection",
            ),
            known_hashes.get(result_name(result)),
        )
        for result in static
    ]
    identities = validate_product_identities(
        cslc_paths, static_paths, recipe["burst_id"]
    )
    manifest = {
        "schema": "dolphinrust-gps-acquisition/1",
        "created_at": datetime.now(UTC).isoformat(),
        "recipe": str(recipe_path),
        "burst_id": recipe["burst_id"],
        "expected_dates": recipe["expected_dates"],
        "complete": len(cslc_paths) == len(recipe["expected_dates"]) and len(static_paths) == 1,
        "identification": identities,
        "cslc": [result_manifest_entry(result, path) for result, path in zip(cslcs, cslc_paths)],
        "static": [result_manifest_entry(result, path) for result, path in zip(static, static_paths)],
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


def run_legacy(args: argparse.Namespace) -> None:
    out = args.out.resolve() if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    token = require_token()
    session = asf.ASFSession().auth_with_token(token)
    results = asf.search(
        dataset="OPERA-S1",
        processingLevel="CSLC",
        operaBurstID=args.burst,
        start=args.start,
        end=args.end,
    )
    chosen = sorted(results, key=lambda result: result.properties["startTime"])[: args.n]
    print(f"burst {args.burst}: {len(results)} granules, taking {len(chosen)}")
    for result in chosen:
        download_result(result, out, session, ("/data/VV",))
    files = sorted(path.name for path in out.glob("OPERA_*.h5") if is_valid_h5(path))
    print(f"\nready: {len(files)} CSLC files in {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burst", default="T063_133231_IW1")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--start", default="2023-06-01")
    parser.add_argument("--end", default="2023-10-01")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--with-static", action="store_true")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="download only the small STATIC companion (credential/preflight gate)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recipe:
        run_recipe(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
