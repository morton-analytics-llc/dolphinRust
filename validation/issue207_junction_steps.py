#!/usr/bin/env python
"""Issue #207: why half the Houston frame steps a whole cycle at junctions 45 and 75.

Test design (frozen before any arm was run):
`md/research/issue-207-junction-step-test-design-2026-11.md`. Inputs are the #192 ones
(`issue192_reference_rule.py plan` and `fetch`); this script adds the independent OPERA
DISP-S1 steps and the scorer.

    PY=oracle/.venv/bin/python
    $PY validation/issue192_reference_rule.py plan  --root RUN
    $PY validation/issue192_reference_rule.py fetch --root RUN
    $PY validation/issue207_junction_steps.py arms  --root RUN --bin target/release/dolphin
    $PY validation/issue207_junction_steps.py opera --root RUN
    $PY validation/issue207_junction_steps.py score --root RUN --receipt <out.json>

`arms` runs the #192 driver into `RUN/a0` (native unwrapper) and `RUN/a1`
(`unwrap_options.unwrap_method: snaphu`, the one change); both share `RUN/cslc`.
After the fix, the #192 driver's `run` into `RUN/a2` with the fixed binary, then

    $PY validation/issue207_junction_steps.py fixed --root RUN --receipt <out.json>

adds the corrected diagnostic's numbers to the same receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DESIGN = "md/research/issue-207-junction-step-test-design-2026-11.md"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
DISP_S1_COLLECTION = "C3294057315-ASF"
OPERA_FRAME = "F08882"
ARMS = {"a0": {}, "a1": {"unwrap_options": {"unwrap_method": "snaphu"}}}
ARM_PLAN = "work_new"  # the #192 driver's `always_first` arm


def recipe() -> dict[str, Any]:
    return json.loads((ROOT / "issue192_houston_recipe.json").read_text())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ arms


def arms(args: argparse.Namespace) -> None:
    for name, overrides in ARMS.items():
        arm_root = args.root / name
        arm_root.mkdir(exist_ok=True)
        for shared in ("cslc", "plan.json"):
            link = arm_root / shared
            if not link.exists():
                link.symlink_to((args.root / shared).resolve())
        command = [sys.executable, str(ROOT / "issue192_reference_rule.py"), "run", "--root", str(arm_root), "--bin", str(args.bin)]
        if overrides:
            command += ["--config-overrides", json.dumps(overrides)]
        subprocess.run(command, check=True)


# ----------------------------------------------------------------- opera


class RangeReader(io.RawIOBase):
    """Seekable file over HTTP range requests on a pre-signed URL."""

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.bytes_read = url, 0, 0
        self.size = int(self._get(0, 0).headers["Content-Range"].split("/")[1])

    def _get(self, start: int, end: int):
        import requests

        for attempt in range(6):
            response = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if response.status_code == 206:
                return response
            time.sleep(2**attempt)
        raise RuntimeError(f"range read failed with HTTP {response.status_code}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset + {0: 0, 1: self.pos, 2: self.size}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.pos)
        if count <= 0:
            return 0
        data = self._get(self.pos, self.pos + count - 1).content
        buffer[: len(data)] = data
        self.pos += len(data)
        self.bytes_read += len(data)
        return len(data)


def signed_url(url: str, token: str) -> str:
    """ASF answers a bearer request with a redirect to a signed URL; a refusal under
    burst load is transient, so back off and retry."""
    import requests

    for attempt in range(8):
        response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, allow_redirects=False, timeout=60)
        if response.status_code in (301, 302, 303, 307) and "Location" in response.headers:
            return response.headers["Location"]
        time.sleep(min(300, 15 * 2**attempt))
    raise RuntimeError(f"ASF did not issue a signed URL (last HTTP {response.status_code})")


def opera_granules(dates: list[str]) -> list[dict[str, str]]:
    """One DISP-S1 granule per secondary date: the newest production of each
    (reference, secondary) pair, the rule behind the #192 velocity reference."""
    import requests

    response = requests.get(
        CMR,
        params={
            "collection_concept_id": DISP_S1_COLLECTION,
            "page_size": 2000,
            "temporal": f"{dates[0]}T00:00:00Z,{dates[-1]}T23:59:59Z",
            "readable_granule_name": f"OPERA_L3_DISP-S1_IW_{OPERA_FRAME}_*",
            "options[readable_granule_name][pattern]": "true",
        },
        timeout=60,
    )
    response.raise_for_status()
    newest: dict[tuple[str, str], dict[str, str]] = {}
    for item in response.json()["items"]:
        umm = item["umm"]
        name = umm["GranuleUR"]
        parts = name.split("_")
        reference, secondary, produced = parts[6], parts[7], parts[9]
        if not name.endswith("v1.0_" + produced) or f"{secondary[:4]}-{secondary[4:6]}-{secondary[6:8]}" not in dates:
            continue
        url = next(link["URL"] for link in umm["RelatedUrls"] if link["Type"] == "GET DATA" and link["URL"].startswith("https") and link["URL"].endswith(".nc"))
        key = (reference, secondary)
        if key not in newest or produced > newest[key]["produced"]:
            newest[key] = {"granule": name, "reference": reference, "secondary": secondary, "produced": produced, "url": url}
    by_secondary: dict[str, dict[str, str]] = {}
    for granule in sorted(newest.values(), key=lambda g: (g["secondary"], g["reference"])):
        by_secondary[granule["secondary"][:8]] = granule  # one per date: the latest reference
    chosen = [by_secondary.get(date.replace("-", "")) for date in dates]
    missing = [date for date, granule in zip(dates, chosen) if granule is None]
    if missing:
        raise SystemExit(f"no DISP-S1 granule for {len(missing)} dates: {missing[:5]}")
    return chosen


def output_grid(root: Path) -> tuple[Any, Any, tuple[int, int]]:
    import rasterio

    with rasterio.open(root / "a0" / ARM_PLAN / "displacement_00.tif") as dataset:
        return dataset.transform, dataset.crs, dataset.shape


def opera(args: argparse.Namespace) -> None:
    import h5py
    import rasterio
    from rasterio.warp import Resampling, reproject

    sys.path.insert(0, str(ROOT))
    from fetch_real import require_token

    token = require_token()
    dates = recipe()["expected_dates"]
    granules = opera_granules(dates)
    transform, crs, shape = output_grid(args.root)
    out = args.root / "opera"
    out.mkdir(exist_ok=True)
    west, north = transform.c, transform.f
    east, south = west + shape[1] * transform.a, north + shape[0] * transform.e
    for index, granule in enumerate(granules):
        target = out / f"{index:02d}.npz"
        if target.exists():
            continue
        reader = RangeReader(signed_url(granule["url"], token))
        with h5py.File(io.BufferedReader(reader, buffer_size=1 << 20), "r") as source:
            gt = [float(value) for value in source["spatial_ref"].attrs["GeoTransform"].split()]
            x, y = source["x"][:], source["y"][:]
            cols = np.flatnonzero((x >= west - 60) & (x <= east + 60))
            rows = np.flatnonzero((y >= south - 60) & (y <= north + 60))
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            displacement = source["displacement"][r0:r1, c0:c1].astype(np.float64)
            mask = source["recommended_mask"][r0:r1, c0:c1].astype(np.float64)
        src_transform = rasterio.Affine(gt[1], 0.0, gt[0] + c0 * gt[1], 0.0, gt[5], gt[3] + r0 * gt[5])
        on_grid = {}
        for key, values in (("displacement_m", displacement), ("recommended_mask", mask)):
            band = np.full(shape, np.nan)
            reproject(values, band, src_transform=src_transform, src_crs=crs, dst_transform=transform, dst_crs=crs,
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
            on_grid[key] = band
        np.savez_compressed(target, **on_grid, granule=granule["granule"], reference=granule["reference"], bytes_read=reader.bytes_read)
        print(index, granule["granule"], reader.bytes_read, flush=True)


# ----------------------------------------------------------------- score


def published_series_cycles(work: Path, wavelength: float, n_dates: int) -> np.ndarray:
    """Published displacement as phase cycles, acquisition 0 prepended as zero.
    `displacement_NN.tif` is series band NN, acquisition NN + 1. The engine
    publishes `-lambda / (4 pi)` metres per radian, so a cycle is `-lambda / 2`."""
    import rasterio

    bands = []
    for band in range(n_dates - 1):
        with rasterio.open(work / f"displacement_{band:02d}.tif") as dataset:
            bands.append(dataset.read(1).astype(np.float64))
    metres = np.stack([np.zeros_like(bands[0]), *bands])
    return metres / (-wavelength / 2.0)


def step_labels(work: Path, pairs: list[tuple[int, int]], k: int, finite: np.ndarray) -> np.ndarray:
    """Labels of the `(k-1, k)` interferogram, else of the first ending at `k`
    (the engine's `step_components`), as published in `conncomp_*.tif`."""
    import rasterio

    band = next((i for i, pair in enumerate(pairs) if pair == (k - 1, k)), None)
    if band is None:
        band = next((i for i, (_, second) in enumerate(pairs) if second == k), None)
    if band is None:
        return finite.astype(np.uint32)
    with rasterio.open(work / f"conncomp_{band:02d}.tif") as dataset:
        return dataset.read(1).astype(np.uint32)


def centred_by_label(step: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """`step` minus the upper median of its label's finite values (NaN where unlabelled)."""
    out = np.full(step.shape, np.nan)
    for label in np.unique(labels[labels > 0]):
        inside = (labels == label) & np.isfinite(step)
        if inside.any():
            values = np.sort(step[inside])
            out[inside] = step[inside] - values[values.size // 2]
    return out


def frac_over_half(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(np.abs(finite) > 0.5)) if finite.size else float("nan")


def itoh(step: np.ndarray, labels: np.ndarray) -> float:
    jumps, pairs = 0, 0
    for axis in (0, 1):
        a = step[:-1, :] if axis == 0 else step[:, :-1]
        b = step[1:, :] if axis == 0 else step[:, 1:]
        la = labels[:-1, :] if axis == 0 else labels[:, :-1]
        lb = labels[1:, :] if axis == 0 else labels[:, 1:]
        ok = np.isfinite(a) & np.isfinite(b) & (la == lb) & (la > 0)
        pairs += int(ok.sum())
        jumps += int((np.abs(a - b)[ok] > 0.5).sum())
    return jumps / pairs if pairs else float("nan")


def summary(values: dict[int, float], junctions: list[int]) -> dict[str, Any]:
    others = np.array([value for k, value in values.items() if k not in junctions and np.isfinite(value)])
    return {
        "junctions": {str(k): values[k] for k in junctions if k in values},
        "non_junction": {
            "steps": int(others.size),
            "median": float(np.median(others)),
            "p95": float(np.percentile(others, 95)),
            "max": float(others.max()),
            "argmax": int(max((k for k in values if k not in junctions), key=lambda k: values[k] if np.isfinite(values[k]) else -1)),
        },
    }


def provenance(work: Path) -> dict[str, Any]:
    return json.loads((work / "geometry_provenance.json").read_text())


def network_pairs(n_dates: int, bandwidth: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_dates) for j in range(i + 1, min(i + bandwidth, n_dates - 1) + 1)]


def arm_statistics(work: Path, frozen: dict[str, Any], n_dates: int, junctions: list[int]) -> dict[str, Any]:
    import rasterio

    engine = frozen["engine"]
    series = published_series_cycles(work, engine["wavelength_m"], n_dates)
    pairs = network_pairs(n_dates, engine["max_bandwidth"])
    if len(pairs) != len(list(work.glob("conncomp_*.tif"))):
        raise SystemExit(f"{work}: {len(pairs)} planned pairs but a different number of conncomp rasters")
    record = provenance(work)
    engine_junctions = {item["epoch"]: item for item in record["ministack_junctions"]["junctions"]}
    local = {item["epoch"]: item["fraction_over_half_cycle"] for item in record["cycle_steps"]["per_date"]}
    stats: dict[str, dict[int, float]] = {key: {} for key in ("J", "J1", "itoh", "range")}
    offline: dict[str, Any] = {}
    steps: dict[int, np.ndarray] = {}
    for k in range(1, n_dates):
        step = series[k] - series[k - 1]
        steps[k] = step
        finite = np.isfinite(step)
        labels = step_labels(work, pairs, k, finite)
        centred = centred_by_label(step, labels)
        stats["J"][k] = frac_over_half(centred)
        stats["J1"][k] = frac_over_half(centred_by_label(step, finite.astype(np.uint32)))
        stats["itoh"][k] = itoh(step, labels)
        stats["range"][k] = float(np.subtract(*np.percentile(step[finite], [98, 2])))
        if k in junctions:
            with rasterio.open(work / f"junction_step_cycles_{k:02d}.tif") as dataset:
                raster = dataset.read(1).astype(np.float64)
            both = np.isfinite(raster) & np.isfinite(centred)
            offline[str(k)] = {
                "offline_fraction_integer_cycle": frac_over_half(np.where(both, centred, np.nan)),
                "engine_fraction_integer_cycle": frac_over_half(np.where(both, raster, np.nan)),
                "engine_provenance_fraction_integer_cycle": engine_junctions[k]["fraction_integer_cycle"],
                "engine_provenance_evaluable_pixels": engine_junctions[k]["evaluable_pixels"],
                "cells_compared": int(both.sum()),
                "median_abs_cell_difference_cycles": float(np.median(np.abs(centred[both] - raster[both]))),
            }
    return {
        "series": series,
        "steps": steps,
        "engine_junctions": engine_junctions,
        "offline": offline,
        "statistics": {key: summary(values, junctions) for key, values in stats.items()},
        "local_L": summary(local, junctions),
        "raw": stats,
    }


def opera_statistics(root: Path, a0: dict[str, Any], n_dates: int, junctions: list[int], wavelength: float) -> dict[str, Any]:
    stacks = [np.load(root / "opera" / f"{index:02d}.npz") for index in range(n_dates)]
    references = [str(stack["reference"]) for stack in stacks]
    cycles = [np.where(stack["recommended_mask"] == 1, stack["displacement_m"], np.nan) / (wavelength / 2.0) for stack in stacks]
    comparable = [k for k in range(1, n_dates) if references[k] == references[k - 1]]
    correlation, opera_steps = {}, {}
    for k in comparable:
        ours, theirs = a0["steps"][k], cycles[k] - cycles[k - 1]
        common = np.isfinite(ours) & np.isfinite(theirs)
        opera_steps[k] = (theirs, common)
        correlation[k] = float(np.corrcoef(ours[common], theirs[common])[0, 1])
    sign = float(np.sign(np.median([correlation[k] for k in comparable if k not in junctions])))
    J_O, I = {}, {}
    for k in comparable:
        theirs, common = opera_steps[k]
        theirs = sign * np.where(common, theirs, np.nan)
        J_O[k] = frac_over_half(centred_by_label(theirs, common.astype(np.uint32)))
        difference = np.where(common, a0["steps"][k] - theirs, np.nan)
        I[k] = frac_over_half(centred_by_label(difference, common.astype(np.uint32)))
    return {
        "granules": [str(stack["granule"]) for stack in stacks],
        "bytes_read": int(sum(int(stack["bytes_read"]) for stack in stacks)),
        "comparable_steps": comparable,
        "sign": sign,
        "correlation": summary(correlation, junctions),
        "J_O": summary(J_O, junctions),
        "I": summary(I, junctions),
        "raw": {"J_O": J_O, "I": I, "correlation": correlation},
    }


def decide(m0: bool, m1: bool, a0: dict[str, Any], o: dict[str, Any]) -> str:
    if not m0:
        return "no finding: M0 reproduction failed"
    if not m1:
        return "c"
    I, J_O, J1 = o["raw"]["I"], o["raw"]["J_O"], a0["raw"]["J1"]
    if (I[45] > 0.05 or I[75] > 0.05) and o["I"]["non_junction"]["median"] <= 0.05:
        return "b"
    if I[45] <= 0.05 and I[75] <= 0.05 and J_O[45] >= 0.5 * J1[45] and J_O[75] >= 0.5 * J1[75]:
        return "a"
    return "no finding"


def score(args: argparse.Namespace) -> None:
    frozen = recipe()
    n_dates = len(frozen["expected_dates"])
    wavelength = frozen["engine"]["wavelength_m"]
    receipt_192 = json.loads((ROOT / "results/issue192_reference_rule/houston_t034_071054_iw3_20261110/receipt.json").read_text())
    expected = {item["epoch"]: item["fraction_integer_cycle"] for item in receipt_192["arms"]["new"]["per_junction"]}
    junctions = sorted(expected)
    a0 = arm_statistics(args.root / "a0" / ARM_PLAN, frozen, n_dates, junctions)
    a1_record = provenance(args.root / "a1" / ARM_PLAN)
    a1 = {item["epoch"]: item["fraction_integer_cycle"] for item in a1_record["ministack_junctions"]["junctions"]}
    m0 = {str(k): {"engine": a0["engine_junctions"][k]["fraction_integer_cycle"], "issue192": expected[k]} for k in junctions}
    m0_pass = all(abs(v["engine"] - v["issue192"]) <= 0.01 for v in m0.values())
    m1_pass = all(
        abs(v["offline_fraction_integer_cycle"] - v["engine_fraction_integer_cycle"]) <= 0.005
        and v["median_abs_cell_difference_cycles"] <= 0.01
        for v in a0["offline"].values()
    )
    a0_engine = {k: a0["engine_junctions"][k]["fraction_integer_cycle"] for k in junctions}
    m3 = {
        "a0_native": {str(k): a0_engine[k] for k in junctions},
        "a1_snaphu": {str(k): a1[k] for k in junctions},
        "native_ruled_out": all(abs(a1[k] - a0_engine[k]) <= 0.05 for k in (45, 75)),
        "native_is_cause": all(a1[k] < 0.05 for k in (45, 75)) and not all(a0_engine[k] < 0.05 for k in (45, 75)),
    }
    m5 = {str(k): {"J": a0["raw"]["J"][k], "J1": a0["raw"]["J1"][k]} for k in (45, 75)}
    o = opera_statistics(args.root, a0, n_dates, junctions, wavelength)
    finding = decide(m0_pass, m1_pass, a0, o)
    receipt = {
        "schema": "dolphinrust.issue207-junction-steps-receipt/1",
        "design": DESIGN,
        "scorer_sha256": sha256_path(Path(__file__)),
        "recipe_sha256": sha256_path(ROOT / "issue192_houston_recipe.json"),
        "plan": {k: v for k, v in json.loads((args.root / "plan.json").read_text()).items() if k != "granules"},
        "run_identity": {arm: json.loads((args.root / arm / "run_identity.json").read_text()) for arm in ARMS},
        "arm_overrides": ARMS,
        "M0_reproduction": {"per_junction": m0, "pass": m0_pass},
        "M1_diagnostic_arithmetic": {"per_junction": a0["offline"], "pass": m1_pass},
        "M2_all_steps": {**a0["statistics"], "L": a0["local_L"]},
        "M3_unwrapper_swap": m3,
        "M4_opera": {k: v for k, v in o.items() if k != "raw"},
        "M5_components": m5,
        "per_step": {
            "J": a0["raw"]["J"], "J1": a0["raw"]["J1"], "itoh": a0["raw"]["itoh"], "range": a0["raw"]["range"],
            "L": {item["epoch"]: item["fraction_over_half_cycle"] for item in provenance(args.root / "a0" / ARM_PLAN)["cycle_steps"]["per_date"]},
            "J_O": o["raw"]["J_O"], "I": o["raw"]["I"], "correlation": o["raw"]["correlation"],
        },
        "finding": finding,
        "scored_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("M0_reproduction", "M1_diagnostic_arithmetic", "M3_unwrapper_swap", "M5_components", "finding")}, indent=2))


def local_residuals(step: np.ndarray, labels: np.ndarray, half: int, min_neighbours: int) -> np.ndarray:
    """Step minus the upper median of its finite same-label neighbours within
    `half` (the engine's `neighbourhood_residuals`); NaN where not judged."""
    from numpy.lib.stride_tricks import sliding_window_view

    size = 2 * half + 1
    windows = sliding_window_view(np.pad(step, half, constant_values=np.nan), (size, size))
    window_labels = sliding_window_view(np.pad(labels, half, constant_values=0), (size, size))
    values = np.where(window_labels == labels[:, :, None, None], windows, np.nan).reshape(*step.shape, -1).copy()
    values[:, :, size * size // 2] = np.nan
    count = np.isfinite(values).sum(-1)
    ordered = np.sort(values, -1)
    median = np.take_along_axis(ordered, np.minimum(count // 2, size * size - 1)[..., None], -1)[..., 0]
    residual = step - median
    residual[(count < min_neighbours) | (labels == 0) | ~np.isfinite(step)] = np.nan
    return residual


def fixed(args: argparse.Namespace) -> None:
    """The corrected diagnostic (design: outcome (a)) on arm A2 = A0 rebuilt with the fix."""
    import rasterio

    frozen = recipe()
    n_dates = len(frozen["expected_dates"])
    receipt = json.loads(args.receipt.read_text())
    junctions = [int(k) for k in receipt["M0_reproduction"]["per_junction"]]
    before, after = args.root / "a0", args.root / "a2"
    identical = {}
    for plan in ("work_new", "work_old"):
        files = sorted(p.name for p in (before / plan).glob("displacement_[0-9]*.tif"))
        identical[plan] = len(files) == n_dates - 1 and all(
            np.array_equal(rasterio.open(before / plan / name).read(1), rasterio.open(after / plan / name).read(1), equal_nan=True)
            for name in files
        )
    record = provenance(after / ARM_PLAN)
    cycle = record["cycle_steps"]
    fractions = {item["epoch"]: item["fraction_integer_cycle"] for item in record["ministack_junctions"]["junctions"]}
    frozen_bound = receipt["M2_all_steps"]["L"]["non_junction"]["p95"]

    series = published_series_cycles(after / ARM_PLAN, frozen["engine"]["wavelength_m"], n_dates)
    pairs = network_pairs(n_dates, frozen["engine"]["max_bandwidth"])
    like_for_like = {}
    for k in range(1, n_dates):
        step = series[k] - series[k - 1]
        labels = step_labels(after / ARM_PLAN, pairs, k, np.isfinite(step))
        like_for_like[k] = frac_over_half(local_residuals(step, labels, cycle["junction_half_window"], cycle["min_neighbours"]))
    like_summary = summary(like_for_like, junctions)

    overlap = {}
    for k in junctions:
        with rasterio.open(after / ARM_PLAN / f"junction_step_cycles_{k:02d}.tif") as dataset:
            residual = dataset.read(1).astype(np.float64)
        with rasterio.open(after / ARM_PLAN / f"cycle_step_{k:02d}.tif") as dataset:
            flagged = dataset.read(1) != 0
        beyond = np.isfinite(residual) & (np.abs(residual) > 0.5)
        overlap[str(k)] = {
            "cells_beyond_half_cycle": int(beyond.sum()),
            "of_which_184_flagged": int((beyond & flagged).sum()),
            "184_flagged_cells": int(flagged.sum()),
        }
    receipt["fix"] = {
        "run_identity": json.loads((after / "run_identity.json").read_text()),
        "published_displacement_bit_identical": identical,
        "junction_half_window": record["ministack_junctions"]["half_window"],
        "fraction_integer_cycle_before": {str(k): receipt["M3_unwrapper_swap"]["a0_native"][str(k)] for k in junctions},
        "fraction_integer_cycle_after": {str(k): fractions[k] for k in junctions},
        "acceptance_frozen": {
            "bound_p95_L_non_junction": frozen_bound,
            "45": fractions[45] <= frozen_bound,
            "75": fractions[75] <= frozen_bound,
        },
        "like_for_like_junction_window_all_steps": like_summary,
        "residual_vs_184_flags": overlap,
        "cycle_step_flagged_pixels": cycle["flagged_pixels"],
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt["fix"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("arms", arms), ("opera", opera), ("score", score), ("fixed", fixed)):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--root", type=Path, required=True, help="run directory (kept out of git)")
        if name == "arms":
            command.add_argument("--bin", type=Path, required=True)
        if name in ("score", "fixed"):
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args.handler(args)


if __name__ == "__main__":
    main()
