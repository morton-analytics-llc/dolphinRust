#!/usr/bin/env python
"""Run and score the MMX1/ICMX GNSS validation matrix.

This runner creates native and SNAPHU Rust configs from one dolphin-generated
base, asserts that their scientific settings are identical, runs each backend,
and optionally scores the shared MMX1/ICMX frame against NGL GNSS.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from ruamel.yaml import YAML as RuamelYAML

import gps_ground_truth as gps
from crop_real import validate_cslc_files
from fetch_real import load_recipe, sha256_file

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "oracle" / ".venv" / "bin"
RUST_BIN = ROOT / "target" / "release" / "dolphin"
SAFE_YAML = RuamelYAML(typ="safe")


def build_backend_configs(
    base: dict[str, Any], static_path: Path, run_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    configs = []
    for backend in ["native", "snaphu"]:
        config = copy.deepcopy(base)
        config["work_directory"] = str(run_root / f"work_{backend}")
        config.setdefault("unwrap_options", {})["unwrap_method"] = backend
        config.setdefault("correction_options", {})["geometry_files"] = [str(static_path)]
        configs.append(config)
    return configs[0], configs[1]


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["work_directory"] = "<backend-work-directory>"
    normalized.setdefault("unwrap_options", {})["unwrap_method"] = "<backend>"
    return normalized


def assert_backend_config_identity(
    native: dict[str, Any], snaphu: dict[str, Any]
) -> None:
    if scientific_config(native) != scientific_config(snaphu):
        raise ValueError("native and SNAPHU scientific settings differ")


def result_payload(
    status: str, reason: str | None, context: dict[str, Any], engines: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema": "dolphinrust-gps-ground-truth/1",
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "context": context,
        "engines": engines,
    }
    if reason:
        payload["reason"] = reason
    return payload


def matrix_status(engine_receipts: dict[str, dict[str, Any]]) -> str:
    statuses = {receipt.get("status") for receipt in engine_receipts.values()}
    if "error" in statuses:
        return "error"
    if "not_evaluable" in statuses:
        return "not_evaluable"
    return "complete"


def exit_code_for_status(status: str) -> int:
    if status in {"complete", "pass"}:
        return 0
    if status == "not_evaluable":
        return 2
    return 1


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = SAFE_YAML.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"YAML config is not a mapping: {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    yaml = RuamelYAML()
    yaml.default_flow_style = False
    with path.open("w") as stream:
        yaml.dump(value, stream)


def generate_base_config(cslcs: list[Path], path: Path, work_directory: Path) -> dict[str, Any]:
    command = [
        str(VENV / "dolphin"),
        "config",
        "--slc-files",
        *[str(item) for item in cslcs],
        "-sds",
        "/data/VV",
        "--work-directory",
        str(work_directory),
        "-ms",
        "15",
        "-o",
        str(path),
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return load_yaml(path)


def ensure_rust_binary(build: bool) -> None:
    if build or not RUST_BIN.exists():
        subprocess.run(
            ["cargo", "build", "--release", "-p", "dolphin-cli"], cwd=ROOT, check=True
        )
    if not RUST_BIN.exists():
        raise RuntimeError(f"Rust CLI is unavailable: {RUST_BIN}")


def run_backend(backend: str, config_path: Path, log_path: Path) -> dict[str, Any]:
    if backend == "snaphu" and shutil.which("snaphu") is None:
        raise gps.NotEvaluable("SNAPHU executable is unavailable on PATH")
    started = datetime.now(UTC)
    with log_path.open("w") as log:
        process = subprocess.run(
            [str(RUST_BIN), "run", "--config", str(config_path)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = (datetime.now(UTC) - started).total_seconds()
    if process.returncode != 0:
        raise RuntimeError(
            f"{backend} pipeline failed with exit {process.returncode}; see {log_path}"
        )
    return {"status": "complete", "elapsed_seconds": elapsed, "log": str(log_path)}


def verify_output(work_directory: Path, expected_epochs: int) -> dict[str, Any]:
    displacement = sorted(work_directory.glob("displacement_[0-9][0-9].tif"))
    if len(displacement) != expected_epochs - 1:
        raise gps.NotEvaluable(
            f"{work_directory} has {len(displacement)} displacement rasters; expected {expected_epochs - 1}"
        )
    provenance_path = work_directory / "geometry_provenance.json"
    if not provenance_path.exists():
        raise gps.NotEvaluable(f"missing geometry provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    field = provenance.get("geometry_provenance", {}).get("fields", {}).get("incidence_angle_deg", {})
    if provenance.get("incidence_angle_deg") is None or field.get("status") != "sourced":
        raise gps.NotEvaluable(
            f"LOS geometry is not sourced in {provenance_path}: {field.get('reason', 'unknown reason')}"
        )
    with rasterio.open(displacement[-1]) as dataset:
        values = dataset.read(1)
        finite_fraction = float(np.isfinite(values).mean())
        grid = {
            "shape": [dataset.height, dataset.width],
            "crs": str(dataset.crs),
            "transform": list(dataset.transform)[:6],
        }
    if finite_fraction == 0:
        raise gps.NotEvaluable(f"no finite displacement pixels in {displacement[-1]}")
    return {
        "displacement_files": [str(path) for path in displacement],
        "geometry_provenance": str(provenance_path),
        "incidence_angle_deg": provenance["incidence_angle_deg"],
        "finite_fraction_final": finite_fraction,
        "grid": grid,
    }


def verify_core_station(
    work_directory: Path, recipe: dict[str, Any]
) -> dict[str, Any]:
    final_path = sorted(work_directory.glob("displacement_[0-9][0-9].tif"))[-1]
    station = recipe["stations"]["MMX1"]
    with rasterio.open(final_path) as dataset:
        row, col = gps.station_pixel(dataset, station["longitude"], station["latitude"])
        stats = gps.window_stats(
            dataset.read(1),
            row,
            col,
            recipe["primary_window"],
            recipe["minimum_finite_fraction"],
        )
    return {"pixel": [row, col], "final_window": as_json(stats)}


def as_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_json(item) for key, item in value.__dict__.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_fixture_contract(
    manifest: dict[str, Any],
    recipe: dict[str, Any],
    fixture: str,
    cslcs: list[Path],
    static_path: Path,
) -> None:
    if manifest.get("schema") != "dolphinrust-gps-fixture/1":
        raise gps.NotEvaluable("fixture manifest schema is unsupported")
    if manifest.get("fixture") != fixture:
        raise gps.NotEvaluable("fixture manifest name does not match requested fixture")
    if manifest.get("burst_id") != recipe["burst_id"]:
        raise gps.NotEvaluable("fixture manifest burst does not match recipe")
    if manifest.get("expected_dates") != recipe["expected_dates"]:
        raise gps.NotEvaluable("fixture manifest dates do not match recipe")
    try:
        validate_cslc_files(cslcs, recipe)
    except ValueError as error:
        raise gps.NotEvaluable(str(error)) from error
    if recipe["burst_filename_id"] not in static_path.name:
        raise gps.NotEvaluable("fixture STATIC burst does not match recipe")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    recipe = load_recipe(args.recipe)
    fixture_root = args.fixture_root or ROOT / "validation" / "real_data" / "gps_mmx1" / "cropped" / args.fixture
    manifest_path = fixture_root / "fixture_manifest.json"
    if not manifest_path.exists():
        raise gps.NotEvaluable(f"fixture manifest is missing: {manifest_path}")
    fixture_manifest = json.loads(manifest_path.read_text())
    cslcs = sorted((fixture_root / "cslc").glob("OPERA_L2_CSLC-S1_T*.h5"))
    static_files = sorted((fixture_root / "static").glob("OPERA_L2_CSLC-S1-STATIC_*.h5"))
    if len(cslcs) != len(recipe["expected_dates"]):
        raise gps.NotEvaluable(f"expected {len(recipe['expected_dates'])} cropped CSLCs; found {len(cslcs)}")
    if len(static_files) != 1:
        raise gps.NotEvaluable(f"expected one cropped STATIC; found {len(static_files)}")
    validate_fixture_contract(
        fixture_manifest, recipe, args.fixture, cslcs, static_files[0]
    )
    run_root = args.run_root or ROOT / "validation" / "runs" / "gps_mmx1" / args.fixture
    run_root.mkdir(parents=True, exist_ok=True)
    base_path = run_root / "config_base.yaml"
    base = generate_base_config(cslcs, base_path, run_root / "work_base")
    native, snaphu = build_backend_configs(base, static_files[0].resolve(), run_root)
    assert_backend_config_identity(native, snaphu)
    configs = {"native": native, "snaphu": snaphu}
    config_paths: dict[str, Path] = {}
    for backend, config in configs.items():
        path = run_root / f"config_{backend}.yaml"
        write_yaml(path, config)
        config_paths[backend] = path
    ensure_rust_binary(args.build)
    engine_receipts: dict[str, Any] = {}
    context = {
        "commit": git_commit(),
        "recipe": str(args.recipe.resolve()),
        "recipe_sha256": sha256_file(args.recipe),
        "fixture": args.fixture,
        "fixture_manifest": str(manifest_path),
        "fixture_manifest_sha256": sha256_file(manifest_path),
        "static": str(static_files[0]),
    }
    for backend in ["native", "snaphu"]:
        work_directory = Path(configs[backend]["work_directory"])
        receipt: dict[str, Any] = {
            "config": str(config_paths[backend]),
            "config_sha256": sha256_file(config_paths[backend]),
            "log": str(run_root / f"{backend}.log"),
        }
        try:
            if not args.no_run:
                if work_directory.exists():
                    shutil.rmtree(work_directory)
                receipt.update(
                    run_backend(
                        backend, config_paths[backend], run_root / f"{backend}.log"
                    )
                )
            else:
                receipt["status"] = "reused"
            receipt["output"] = verify_output(
                work_directory, len(recipe["expected_dates"])
            )
            if args.fixture == "mmx1_core":
                receipt["core_station"] = verify_core_station(work_directory, recipe)
        except gps.NotEvaluable as error:
            receipt.update({"status": "not_evaluable", "reason": str(error)})
        except RuntimeError as error:
            receipt.update({"status": "error", "reason": str(error)})
        engine_receipts[backend] = receipt
    status = matrix_status(engine_receipts)
    if status != "complete":
        reasons = [
            f"{backend}: {receipt.get('reason', receipt['status'])}"
            for backend, receipt in engine_receipts.items()
            if receipt["status"] in {"not_evaluable", "error"}
        ]
        payload = result_payload(
            status, "; ".join(reasons), context, engine_receipts
        )
        (run_root / "run_receipt.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        return payload
    if args.score:
        if args.fixture != "mmx1_icmx_common":
            raise gps.NotEvaluable("--score requires --fixture mmx1_icmx_common")
        payload = gps.score_common_frame(
            recipe,
            fixture_manifest,
            {backend: Path(configs[backend]["work_directory"]) for backend in configs},
            static_files[0],
            ROOT / "validation" / "real_data" / "gps_mmx1" / "gnss",
            run_root,
        )
        payload["context"] = context
        payload["run_receipts"] = engine_receipts
        (run_root / "gps_ground_truth.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    payload = result_payload("complete", None, context, engine_receipts)
    (run_root / "run_receipt.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, default=ROOT / "validation" / "gps_mmx1.json")
    parser.add_argument("--fixture", choices=["mmx1_core", "mmx1_icmx_common"], required=True)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="reuse existing backend outputs")
    parser.add_argument("--build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = execute(args)
    except (gps.NotEvaluable, RuntimeError) as error:
        status = "not_evaluable" if isinstance(error, gps.NotEvaluable) else "error"
        payload = result_payload(
            status,
            str(error),
            {"fixture": args.fixture, "recipe": str(args.recipe)},
            {},
        )
        print(json.dumps(payload, indent=2))
        raise SystemExit(exit_code_for_status(status)) from error
    print(json.dumps({"status": payload["status"], "fixture": args.fixture}, indent=2))
    exit_code = exit_code_for_status(payload["status"])
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
