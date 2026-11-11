#!/usr/bin/env python
"""Issue #192 real-data validation: the two `always_first` reference rules,
like for like, on one cropped Houston burst.

Protocol (criteria frozen before any data was fetched):
`md/research/issue-192-validation-protocol-2026-11.md`. Everything the scoring
uses is in `validation/issue192_houston_recipe.json`.

    source validation/creds.sh        # or --env-file <a .env holding the token>
    PY=oracle/.venv/bin/python
    $PY validation/issue192_reference_rule.py plan  --root RUN
    $PY validation/issue192_reference_rule.py fetch --root RUN
    $PY validation/issue192_reference_rule.py run   --root RUN --bin target/release/dolphin
    $PY validation/issue192_reference_rule.py score --root RUN \\
        --opera-velocity <opera_F08882_T34_velocity_mm_yr.tif> --receipt <out.json>

`fetch` range-reads only the crop window of each granule. `run` runs the same
binary twice; the two configs differ in `phase_linking.compressed_slc_plan` only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
RECIPE = ROOT / "issue192_houston_recipe.json"
PRODUCED = re.compile(r"_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_S1")
SCALE_MAD = 1.4826


def load_recipe() -> dict[str, Any]:
    return json.loads(RECIPE.read_text())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def session(env_file: Path | None):
    from fetch_real import authenticated_session

    if env_file is None:
        return authenticated_session()
    return authenticated_session({}, env_file=env_file)


# ------------------------------------------------------------------ plan


def select_granules(recipe: dict[str, Any]) -> list[dict[str, str]]:
    """One granule per expected date: the latest production timestamp."""
    import asf_search as asf

    dates = recipe["expected_dates"]
    results = asf.search(
        dataset="OPERA-S1",
        processingLevel="CSLC",
        operaBurstID=[recipe["burst_id"]],
        start=f"{dates[0]}T00:00:00Z",
        end=f"{dates[-1]}T23:59:59Z",
    )
    by_date: dict[str, tuple[str, str, str]] = {}
    for item in results:
        name = item.properties["fileName"]
        match = PRODUCED.search(name)
        if match is None or recipe["burst_filename_id"] not in name:
            continue
        acquired, produced = match.groups()
        date = f"{acquired[:4]}-{acquired[4:6]}-{acquired[6:8]}"
        if date in dates and (date not in by_date or produced > by_date[date][0]):
            by_date[date] = (produced, name, item.properties["url"])
    missing = [date for date in dates if date not in by_date]
    if missing:
        raise SystemExit(f"no CSLC granule for {len(missing)} expected dates: {missing[:5]}")
    return [{"date": date, "file_name": by_date[date][1], "url": by_date[date][2]} for date in dates]


def crop_window(recipe: dict[str, Any], url: str, http) -> dict[str, int]:
    """Rows/cols of the recipe's UTM bounds on the burst grid (grid vectors only)."""
    import fsspec
    import h5py

    first = http.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True)
    first.raise_for_status()
    headers = {key: value for key, value in http.headers.items() if key.lower() == "authorization"}
    filesystem = fsspec.filesystem("http", headers=headers, client_kwargs={"cookies": http.cookies.get_dict()})
    with filesystem.open(first.url, "rb", block_size=64 * 1024, cache_type="readahead") as remote:
        with h5py.File(remote, "r") as source:
            x = source["/data/x_coordinates"][:]
            y = source["/data/y_coordinates"][:]
            epsg = int(source["/data/projection"][()])
    bounds = recipe["bounds_utm_m"]
    if epsg != recipe["epsg"]:
        raise SystemExit(f"burst grid is EPSG:{epsg}, recipe says {recipe['epsg']}")
    cols = np.flatnonzero((x >= bounds["west"]) & (x < bounds["east"]))
    rows = np.flatnonzero((y > bounds["south"]) & (y <= bounds["north"]))
    if cols.size == 0 or rows.size == 0:
        raise SystemExit("recipe bounds do not intersect the burst grid")
    strides = recipe["engine"]["strides"]
    height = rows.size - rows.size % strides["y"]
    width = cols.size - cols.size % strides["x"]
    return {"row0": int(rows[0]), "col0": int(cols[0]), "height": int(height), "width": int(width)}


def plan(args: argparse.Namespace) -> None:
    recipe = load_recipe()
    granules = select_granules(recipe)
    window = crop_window(recipe, granules[0]["url"], session(args.env_file))
    catalog = json.dumps(granules, sort_keys=True).encode()
    payload = {
        "recipe_sha256": sha256_path(RECIPE),
        "catalog_sha256": hashlib.sha256(catalog).hexdigest(),
        "window": window,
        "granules": granules,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"granules": len(granules), "window": window}, indent=2))


# ----------------------------------------------------------------- fetch


def fetch(args: argparse.Namespace) -> None:
    import shutil

    from crop_real import Window
    from remote_hdf5_crop import crop_remote_hdf5

    recipe = load_recipe()
    planned = json.loads((args.root / "plan.json").read_text())
    http = session(args.env_file)
    cslc = args.root / "cslc"
    receipts = args.root / "crop_receipts"
    cslc.mkdir(exist_ok=True)
    receipts.mkdir(exist_ok=True)
    for granule in planned["granules"]:
        destination = cslc / granule["file_name"]
        if destination.exists():
            continue
        free_gib = shutil.disk_usage(args.root).free / 2**30
        if free_gib < args.min_free_gib:
            raise SystemExit(f"stopping: {free_gib:.1f} GiB free is under the {args.min_free_gib} GiB floor")
        crop_remote_hdf5(
            url=granule["url"],
            expected_file_name=granule["file_name"],
            destination=destination,
            receipt_path=receipts / f"{granule['file_name']}.json",
            product_type="cslc",
            window=Window(**planned["window"]),
            source_catalog_sha256=planned["catalog_sha256"],
            session=http,
            max_transfer_bytes=recipe["max_transfer_bytes_per_granule"],
        )
        print(granule["date"], destination.stat().st_size, flush=True)


# ------------------------------------------------------------------- run


def arm_config(recipe: dict[str, Any], files: list[Path], work: Path, plan_name: str) -> dict[str, Any]:
    engine = recipe["engine"]
    return {
        "cslc_file_list": [str(path) for path in files],
        "work_directory": str(work),
        "input_options": {"subdataset": "/data/VV", "wavelength": engine["wavelength_m"]},
        "phase_linking": {
            "ministack_size": engine["ministack_size"],
            "max_num_compressed": engine["max_num_compressed"],
            "half_window": engine["half_window"],
            "shp_method": engine["shp_method"],
            "compressed_slc_plan": plan_name,
            "write_crlb": False,
        },
        "output_options": {"strides": engine["strides"]},
        "interferogram_network": {"max_bandwidth": engine["max_bandwidth"]},
        "unwrap_options": {"unwrap_method": engine["unwrap_method"]},
        "timeseries_options": {
            "use_coherence_weights": False,
            "mask_unwrap_loop_errors": engine["mask_unwrap_loop_errors"],
            "reference_selection": engine["reference_selection"],
            "reference_coherence_floor": engine["reference_coherence_floor"],
            "write_cycle_step_diagnostics": True,
        },
    }


def run(args: argparse.Namespace) -> None:
    from ruamel.yaml import YAML

    recipe = load_recipe()
    planned = json.loads((args.root / "plan.json").read_text())
    files = [args.root / "cslc" / granule["file_name"] for granule in planned["granules"]]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} cropped granules are missing; run fetch first")
    overrides = json.loads(args.config_overrides) if args.config_overrides else {}
    identity = {"binary": str(args.bin), "binary_sha256": sha256_path(args.bin), "config_overrides": overrides}
    for arm, plan_name in recipe["arms"].items():
        work = args.root / f"work_{arm}"
        config = arm_config(recipe, files, work, plan_name)
        for section, values in overrides.items():
            config.setdefault(section, {}).update(values)
        config_path = args.root / f"config_{arm}.yaml"
        with config_path.open("w") as handle:
            YAML().dump(config, handle)
        started = dt.datetime.now(dt.UTC)
        with (args.root / f"run_{arm}.log").open("w") as log:
            subprocess.run([str(args.bin), "run", "--config", str(config_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
        identity[arm] = {
            "compressed_slc_plan": plan_name,
            "config_sha256": sha256_path(config_path),
            "elapsed_seconds": (dt.datetime.now(dt.UTC) - started).total_seconds(),
        }
        print(arm, identity[arm], flush=True)
    (args.root / "run_identity.json").write_text(json.dumps(identity, indent=2) + "\n")


# ----------------------------------------------------------------- score


def load_cube_mm(work: Path, epochs: int) -> tuple[np.ndarray, Any, Any]:
    """Published displacement, mm, with the zero reference epoch prepended."""
    import rasterio

    files = sorted(work.glob("displacement_[0-9]*.tif"))
    if len(files) != epochs - 1:
        raise SystemExit(f"{work}: {len(files)} displacement rasters, expected {epochs - 1}")
    bands = []
    for path in files:
        with rasterio.open(path) as dataset:
            bands.append(dataset.read(1).astype(float) * 1000.0)
            transform, crs = dataset.transform, dataset.crs
    return np.stack([np.zeros_like(bands[0]), *bands]), transform, crs


def high_pass(field: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """`field` minus its mask-normalised Gaussian mean (NaN outside `mask`)."""
    from scipy.ndimage import gaussian_filter

    weights = gaussian_filter(mask.astype(float), sigma, mode="constant")
    smooth = gaussian_filter(np.where(mask, field, 0.0), sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mask, field - smooth / weights, np.nan)


def step_roughness(cube: np.ndarray, mask: np.ndarray, sigma: float) -> list[float]:
    """Per date step k (1..n-1): 1.4826 * MAD of the high-passed step, mm."""
    values = []
    for k in range(1, cube.shape[0]):
        residual = high_pass(cube[k] - cube[k - 1], mask, sigma)[mask]
        values.append(float(SCALE_MAD * np.median(np.abs(residual - np.median(residual)))))
    return values


def ols_rate(years: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Least-squares slope along axis 0 (per year)."""
    centred = years - years.mean()
    shape = (-1,) + (1,) * (series.ndim - 1)
    return (centred.reshape(shape) * (series - series.mean(axis=0))).sum(axis=0) / (centred**2).sum()


def station_series(cube: np.ndarray, mask: np.ndarray, transform, station: dict[str, Any], recipe: dict[str, Any]):
    col, row = (int(np.floor(v)) for v in ~transform * (station["easting_m"], station["northing_m"]))
    half = recipe["station_window_px"] // 2
    rows = slice(row - half, row + half + 1)
    cols = slice(col - half, col + half + 1)
    inside = 0 <= row - half and 0 <= col - half and row + half < mask.shape[0] and col + half < mask.shape[1]
    kept = mask[rows, cols] if inside else np.zeros((0, 0), bool)
    if kept.sum() < recipe["station_min_pixels"]:
        return None, int(kept.sum())
    return np.median(cube[:, rows, cols][:, kept], axis=1), int(kept.sum())


def provenance_junctions(work: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    provenance = json.loads((work / "geometry_provenance.json").read_text())
    junctions = {item["epoch"]: item for item in provenance["ministack_junctions"]["junctions"]}
    affected = [junctions[epoch] for epoch in recipe["junction_epochs"]["affected"]]
    mean = lambda key: float(np.mean([item[key] for item in affected]))
    return {
        "fraction_integer_cycle": mean("fraction_integer_cycle"),
        "fraction_linked_over_quarter_cycle": mean("fraction_linked_over_quarter_cycle"),
        "per_junction": list(junctions.values()),
        "cycle_step_flagged_pixels": provenance["cycle_steps"]["flagged_pixels"],
    }


def score(args: argparse.Namespace) -> None:
    import rasterio
    from rasterio.warp import Resampling, reproject

    recipe = load_recipe()
    if sha256_path(args.opera_velocity) != recipe["opera_source"]["sha256"]:
        raise SystemExit("OPERA velocity raster differs from the recipe's frozen sha256")
    dates = [dt.date.fromisoformat(value) for value in recipe["expected_dates"]]
    years = np.array([(date - dates[0]).days / 365.25 for date in dates])
    cubes = {}
    for arm in recipe["arms"]:
        cubes[arm], transform, crs = load_cube_mm(args.root / f"work_{arm}", len(dates))
    mask = np.all([np.isfinite(cube).all(axis=0) for cube in cubes.values()], axis=0)
    if not mask.any():
        raise SystemExit("no pixel is finite on every date in both arms")

    opera = np.full(mask.shape, np.nan, dtype="float32")
    with rasterio.open(args.opera_velocity) as source:
        reproject(
            rasterio.band(source, 1), opera, dst_transform=transform, dst_crs=crs,
            src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest,
        )
    opera_mask = mask & np.isfinite(opera)

    affected = recipe["junction_epochs"]["affected"]
    junction_steps = {*affected, *recipe["junction_epochs"]["control"]}
    tolerance = recipe["tolerances"]
    arms: dict[str, Any] = {}
    for arm, cube in cubes.items():
        roughness = step_roughness(cube, mask, recipe["high_pass_sigma_px"])  # index k-1 holds step k
        velocity = ols_rate(years, cube)
        difference = (velocity - opera)[opera_mask]
        centred = difference - np.median(difference)
        stations = {}
        for name, station in recipe["stations"].items():
            series, pixels = station_series(cube, mask, transform, station, recipe)
            stations[name] = {"pixels": pixels, "series": series}
        pairs = {}
        for a, b in itertools.combinations(sorted(recipe["stations"]), 2):
            if stations[a]["series"] is None or stations[b]["series"] is None:
                continue
            insar = float(ols_rate(years, stations[a]["series"] - stations[b]["series"]))
            gnss = recipe["stations"][a]["gnss_los_rate_vs_thsu_mm_yr"] - recipe["stations"][b]["gnss_los_rate_vs_thsu_mm_yr"]
            pairs[f"{a}-{b}"] = {"insar_mm_yr": insar, "gnss_mm_yr": gnss, "error_mm_yr": insar - gnss}
        arms[arm] = {
            "junction_roughness_mm": float(np.mean([roughness[k - 1] for k in affected])),
            "junction_roughness_by_epoch_mm": {str(k): roughness[k - 1] for k in sorted(junction_steps)},
            "non_junction_roughness_median_mm": float(
                np.median([value for k, value in enumerate(roughness, start=1) if k not in junction_steps])
            ),
            **provenance_junctions(args.root / f"work_{arm}", recipe),
            "opera_median_abs_mm_yr": float(np.median(np.abs(centred))),
            "opera_fraction_within": float(np.mean(np.abs(centred) <= tolerance["opera_within_mm_yr"])),
            "station_pixels": {name: value["pixels"] for name, value in stations.items()},
            "gnss_pairs": pairs,
            "gnss_mean_abs_mm_yr": float(np.mean([abs(p["error_mm_yr"]) for p in pairs.values()])) if pairs else None,
        }

    old, new = arms["old"], arms["new"]
    checks = {
        "junction_roughness_mm": new["junction_roughness_mm"]
        <= old["junction_roughness_mm"] * (1 + tolerance["junction_roughness_relative"]),
        "fraction_integer_cycle": new["fraction_integer_cycle"]
        <= old["fraction_integer_cycle"] + tolerance["junction_fraction_absolute"],
        "fraction_linked_over_quarter_cycle": new["fraction_linked_over_quarter_cycle"]
        <= old["fraction_linked_over_quarter_cycle"] + tolerance["junction_fraction_absolute"],
        "opera_median_abs_mm_yr": new["opera_median_abs_mm_yr"]
        <= old["opera_median_abs_mm_yr"] + tolerance["opera_median_abs_mm_yr"],
        "opera_fraction_within": new["opera_fraction_within"]
        >= old["opera_fraction_within"] - tolerance["opera_fraction_within"],
    }
    if old["gnss_mean_abs_mm_yr"] is not None:
        checks["gnss_mean_abs_mm_yr"] = new["gnss_mean_abs_mm_yr"] <= old["gnss_mean_abs_mm_yr"] + tolerance["gnss_mean_abs_mm_yr"]
    reduced = new["junction_roughness_mm"] <= old["junction_roughness_mm"] * (1 - recipe["reduction_threshold_relative"])
    receipt = {
        "schema": "dolphinrust.issue192-reference-rule-receipt/1",
        "recipe_sha256": sha256_path(RECIPE),
        "scorer_sha256": sha256_path(Path(__file__)),
        "protocol": recipe["protocol"],
        "run_identity": json.loads((args.root / "run_identity.json").read_text()),
        "plan": {k: v for k, v in json.loads((args.root / "plan.json").read_text()).items() if k != "granules"},
        "dates": len(dates),
        "output_grid": {"shape": list(mask.shape), "transform": list(transform)[:6], "crs": str(crs)},
        "pixels": {"grid": int(mask.size), "common_valid": int(mask.sum()), "with_opera": int(opera_mask.sum())},
        "arms": arms,
        "not_worse": checks,
        "gnss": "scored" if "gnss_mean_abs_mm_yr" in checks else "not evaluable: fewer than two stations measured",
        "junction_steps_reduced": bool(reduced),
        "verdict": "pass" if all(checks.values()) else "fail",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("pixels", "not_worse", "gnss", "junction_steps_reduced", "verdict")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("plan", plan), ("fetch", fetch), ("run", run), ("score", score)):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--root", type=Path, required=True, help="run directory (kept out of git)")
        if name in ("plan", "fetch"):
            command.add_argument("--env-file", type=Path, help="a .env holding the Earthdata bearer token")
        if name == "fetch":
            command.add_argument("--min-free-gib", type=float, default=20.0)
        if name == "run":
            command.add_argument("--bin", type=Path, required=True)
            command.add_argument("--config-overrides", help="JSON merged into both arms' configs (configuration may be swept; criteria may not)")
        if name == "score":
            command.add_argument("--opera-velocity", type=Path, required=True)
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args.handler(args)


if __name__ == "__main__":
    main()
