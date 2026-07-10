#!/usr/bin/env python
"""Crop full OPERA products into small, georeferenced validation fixtures.

Legacy pixel-window mode remains available. Recipe mode builds the MMX1 core or
the shared MMX1/ICMX comparison frame and crops the CSLC-S1-STATIC companion by
geographic bounds on its own grid.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pyproj import Transformer

from fetch_real import load_recipe, sha256_file

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "real_data"
OUT = SRC / "cropped"
ACQUISITION_DATE_RE = re.compile(r"_(20\d{6})T")


@dataclass(frozen=True)
class Window:
    row0: int
    col0: int
    height: int
    width: int

    @property
    def row1(self) -> int:
        return self.row0 + self.height

    @property
    def col1(self) -> int:
        return self.col0 + self.width


def projected_to_pixel(
    x_value: float, y_value: float, x_coordinates: np.ndarray, y_coordinates: np.ndarray
) -> tuple[int, int]:
    if x_coordinates.ndim != 1 or y_coordinates.ndim != 1:
        raise ValueError("coordinate arrays must be one-dimensional")
    col = int(np.argmin(np.abs(x_coordinates - x_value)))
    row = int(np.argmin(np.abs(y_coordinates - y_value)))
    return row, col


def validate_window(window: Window, shape: tuple[int, int]) -> Window:
    rows, cols = shape
    if (
        window.row0 < 0
        or window.col0 < 0
        or window.height <= 0
        or window.width <= 0
        or window.row1 > rows
        or window.col1 > cols
    ):
        raise ValueError(f"crop window {window} falls outside source shape {shape}")
    return window


def centered_window(
    row: int, col: int, size: int, shape: tuple[int, int]
) -> Window:
    return validate_window(Window(row - size // 2, col - size // 2, size, size), shape)


def window_for_projected_bounds(
    bounds: tuple[float, float, float, float],
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    margin_pixels: int = 0,
) -> Window:
    left, bottom, right, top = bounds
    if left > right or bottom > top:
        raise ValueError(f"invalid projected bounds: {bounds}")
    corners = [
        projected_to_pixel(left, bottom, x_coordinates, y_coordinates),
        projected_to_pixel(left, top, x_coordinates, y_coordinates),
        projected_to_pixel(right, bottom, x_coordinates, y_coordinates),
        projected_to_pixel(right, top, x_coordinates, y_coordinates),
    ]
    rows = [row for row, _ in corners]
    cols = [col for _, col in corners]
    window = Window(
        min(rows) - margin_pixels,
        min(cols) - margin_pixels,
        max(rows) - min(rows) + 1 + 2 * margin_pixels,
        max(cols) - min(cols) + 1 + 2 * margin_pixels,
    )
    return validate_window(window, (len(y_coordinates), len(x_coordinates)))


def window_projected_bounds(
    window: Window, x_coordinates: np.ndarray, y_coordinates: np.ndarray
) -> tuple[float, float, float, float]:
    dx = float(np.median(np.diff(x_coordinates)))
    dy = float(np.median(np.diff(y_coordinates)))
    x0 = float(x_coordinates[window.col0] - dx / 2.0)
    x1 = float(x_coordinates[window.col1 - 1] + dx / 2.0)
    y0 = float(y_coordinates[window.row0] - dy / 2.0)
    y1 = float(y_coordinates[window.row1 - 1] + dy / 2.0)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(path, "r") as product:
        x = product["/data/x_coordinates"][:].astype(float)
        y = product["/data/y_coordinates"][:].astype(float)
        epsg = int(product["/data/projection"][()])
    return x, y, epsg


def copy_group_if_present(source: h5py.File, destination: h5py.File, name: str) -> None:
    if name in source:
        source.copy(name, destination)


def crop_product(
    source_path: Path,
    destination_path: Path,
    product_type: str,
    window: Window,
) -> dict[str, Any]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if product_type == "cslc":
        layers = ["VV"]
    elif product_type == "static":
        layers = ["los_east", "los_north"]
    else:
        raise ValueError(f"unsupported product type: {product_type}")
    with h5py.File(source_path, "r") as source:
        x = source["/data/x_coordinates"][:]
        y = source["/data/y_coordinates"][:]
        validate_window(window, (len(y), len(x)))
        with h5py.File(destination_path, "w") as destination:
            for key, value in source.attrs.items():
                destination.attrs[key] = value
            copy_group_if_present(source, destination, "identification")
            copy_group_if_present(source, destination, "metadata")
            data = destination.create_group("data")
            for key, value in source["/data"].attrs.items():
                data.attrs[key] = value
            for layer in layers:
                dataset = source[f"/data/{layer}"]
                cropped = dataset[window.row0 : window.row1, window.col0 : window.col1]
                output = data.create_dataset(layer, data=cropped)
                for key, value in dataset.attrs.items():
                    output.attrs[key] = value
            for name, values in [
                ("x_coordinates", x[window.col0 : window.col1]),
                ("y_coordinates", y[window.row0 : window.row1]),
            ]:
                output = data.create_dataset(name, data=values)
                for key, value in source[f"/data/{name}"].attrs.items():
                    output.attrs[key] = value
            projection = source["/data/projection"]
            output_projection = data.create_dataset("projection", data=projection[()])
            for key, value in projection.attrs.items():
                output_projection.attrs[key] = value
    cropped_x, cropped_y, epsg = read_grid(destination_path)
    return {
        "source": str(source_path),
        "destination": str(destination_path),
        "source_sha256": sha256_file(source_path),
        "destination_sha256": sha256_file(destination_path),
        "product_type": product_type,
        "epsg": epsg,
        "window": asdict(window),
        "shape": [len(cropped_y), len(cropped_x)],
        "bounds": list(window_projected_bounds(Window(0, 0, len(cropped_y), len(cropped_x)), cropped_x, cropped_y)),
    }


def crop_product_to_bounds(
    source_path: Path,
    destination_path: Path,
    product_type: str,
    bounds: tuple[float, float, float, float],
    margin_pixels: int = 0,
) -> dict[str, Any]:
    x, y, _ = read_grid(source_path)
    window = window_for_projected_bounds(bounds, x, y, margin_pixels)
    return crop_product(source_path, destination_path, product_type, window)


def station_projected(
    station: dict[str, Any], epsg: int
) -> tuple[float, float]:
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    return transformer.transform(station["longitude"], station["latitude"])


def fixture_window(
    recipe: dict[str, Any], fixture_name: str, x: np.ndarray, y: np.ndarray, epsg: int
) -> tuple[Window, dict[str, dict[str, Any]]]:
    fixture = recipe["fixtures"][fixture_name]
    stations: dict[str, dict[str, Any]] = {}
    for station_id, station in recipe["stations"].items():
        projected_x, projected_y = station_projected(station, epsg)
        row, col = projected_to_pixel(projected_x, projected_y, x, y)
        stations[station_id] = {
            "longitude": station["longitude"],
            "latitude": station["latitude"],
            "projected_x": projected_x,
            "projected_y": projected_y,
            "source_row": row,
            "source_col": col,
        }
    if fixture["mode"] == "center":
        center = stations[fixture["center_station"]]
        window = centered_window(center["source_row"], center["source_col"], fixture["size"], (len(y), len(x)))
    elif fixture["mode"] == "stations":
        selected = [stations[station_id] for station_id in fixture["station_ids"]]
        margin = int(fixture["margin_pixels"])
        rows = [station["source_row"] for station in selected]
        cols = [station["source_col"] for station in selected]
        window = validate_window(
            Window(
                min(rows) - margin,
                min(cols) - margin,
                max(rows) - min(rows) + 1 + 2 * margin,
                max(cols) - min(cols) + 1 + 2 * margin,
            ),
            (len(y), len(x)),
        )
    else:
        raise ValueError(f"unsupported fixture mode: {fixture['mode']}")
    for station in stations.values():
        station["crop_row"] = station["source_row"] - window.row0
        station["crop_col"] = station["source_col"] - window.col0
    return window, stations


def assert_cslc_grids(files: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    if not files:
        raise ValueError("no source CSLC granules; run fetch_real.py first")
    reference = read_grid(files[0])
    for path in files[1:]:
        grid = read_grid(path)
        if grid[2] != reference[2] or not np.array_equal(grid[0], reference[0]) or not np.array_equal(grid[1], reference[1]):
            raise ValueError(f"CSLC grid differs from first acquisition: {path}")
    return reference


def validate_cslc_files(files: list[Path], recipe: dict[str, Any]) -> None:
    burst = recipe["burst_filename_id"]
    if any(burst not in path.name for path in files):
        raise ValueError(f"source CSLC list contains a file outside burst {burst}")
    dates: list[str] = []
    for path in files:
        match = ACQUISITION_DATE_RE.search(path.name)
        if match is None:
            raise ValueError(f"source CSLC filename has no acquisition date: {path.name}")
        stamp = match.group(1)
        dates.append(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}")
    if dates != recipe["expected_dates"]:
        raise ValueError(
            f"source CSLC dates do not match recipe: got {dates}, "
            f"expected {recipe['expected_dates']}"
        )


def run_recipe(recipe_path: Path, fixture_name: str, source_root: Path | None, out: Path | None) -> None:
    recipe = load_recipe(recipe_path)
    source = source_root or SRC / "gps_mmx1" / "source"
    cslcs = sorted((source / "cslc").glob(f"OPERA_L2_CSLC-S1_{recipe['burst_filename_id']}_*.h5"))
    static_files = sorted((source / "static").glob(f"OPERA_L2_CSLC-S1-STATIC_{recipe['burst_filename_id']}_*.h5"))
    if len(static_files) != 1:
        raise ValueError(f"expected exactly one source STATIC; found {len(static_files)}")
    validate_cslc_files(cslcs, recipe)
    if recipe["burst_filename_id"] not in static_files[0].name:
        raise ValueError("source STATIC burst does not match recipe")
    x, y, epsg = assert_cslc_grids(cslcs)
    window, stations = fixture_window(recipe, fixture_name, x, y, epsg)
    bounds = window_projected_bounds(window, x, y)
    destination = out or SRC / "gps_mmx1" / "cropped" / fixture_name
    cslc_out = destination / "cslc"
    static_out = destination / "static"
    entries = [
        crop_product(path, cslc_out / path.name, "cslc", window)
        for path in cslcs
    ]
    static_entry = crop_product_to_bounds(
        static_files[0], static_out / static_files[0].name, "static", bounds, margin_pixels=1
    )
    manifest = {
        "schema": "dolphinrust-gps-fixture/1",
        "recipe": str(recipe_path.resolve()),
        "fixture": fixture_name,
        "burst_id": recipe["burst_id"],
        "expected_dates": recipe["expected_dates"],
        "epsg": epsg,
        "source_window": asdict(window),
        "projected_bounds": list(bounds),
        "shape": [window.height, window.width],
        "estimated_complex_pixels": window.height * window.width * len(cslcs),
        "stations": stations,
        "cslc": entries,
        "static": [static_entry],
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"fixture {fixture_name}: {window.height}x{window.width}, EPSG:{epsg}")
    print(f"bounds: {bounds}")
    print(f"manifest: {manifest_path}")


def run_legacy(args: argparse.Namespace) -> None:
    source = args.source or SRC
    out = args.out or OUT
    files = sorted(path for path in source.glob("OPERA_L2_CSLC-S1_T*.h5") if args.burst in path.name)
    if not files:
        raise SystemExit("no source granules — run validation/fetch_real.py first")
    window = Window(args.row0, args.col0, args.size, args.size)
    for path in files:
        info = crop_product(path, out / path.name, "cslc", window)
        print(f"{path.name}: {tuple(info['shape'])} epsg={info['epsg']} -> {out / path.name}")
    print(f"cropped {len(files)} files to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row0", type=int, default=2200)
    parser.add_argument("--col0", type=int, default=9700)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--burst", default="", help="substring filter in legacy mode")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--fixture", choices=["mmx1_core", "mmx1_icmx_common"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recipe:
        if not args.fixture:
            raise SystemExit("--fixture is required with --recipe")
        run_recipe(args.recipe, args.fixture, args.source, args.out)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
