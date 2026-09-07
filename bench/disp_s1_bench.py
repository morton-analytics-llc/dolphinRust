#!/usr/bin/env python
"""dolphinRust end-to-end ministack bench, staged to compare against published
DISP-S1 production wall-clock (issue #104).

Runs `target/release/dolphin run` once over a real OPERA CSLC-S1 ministack and
records per-stage wall-clock and peak RSS, so the result lines up against the
stage breakdown in Staniewicz et al., "Near-Real-Time InSAR Phase Estimation for
Large-Scale Surface Displacement Monitoring" (arXiv:2511.12051): median 6.7
hours per 15-date ministack in DISP-S1 North America production, ~80% of it in
unwrapping, ~90 min in phase-linking + PS selection.

Every number written here is measured in this run. The paper's numbers are
quoted, never re-derived, and the two are NOT the same experiment: this is one
burst on one laptop, theirs is full-frame production on a cluster. The output
JSON carries both the measured config and the machine spec so the comparison can
be read with that difference in view rather than normalized away.

Stage timings come from the library's own `timed(...)` tracing events
(`crates/dolphin-workflows/src/displacement.rs`), the same source `bench/bench.py`
parses. Peak RSS is `/usr/bin/time -l` on the whole process, which is
authoritative; the per-stage `peak_rss_kib` breadcrumbs are the in-process
high-water at each stage boundary.

Run:
  oracle/.venv/bin/python bench/disp_s1_bench.py
  oracle/.venv/bin/python bench/disp_s1_bench.py --dates 15 --out bench/disp_s1.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "oracle" / ".venv" / "bin"
RUST_BIN = ROOT / "target" / "release" / "dolphin"

# The largest real burst-scale stack committed under validation/real_data.
DEFAULT_STACK = (
    ROOT
    / "validation"
    / "real_data"
    / "gps_mmx1_2018"
    / "cropped"
    / "mmx1_2018_los_common"
    / "cslc"
)
SUBDATASET = "/data/VV"

# DISP-S1 production geometry: half-window 11x5, strides 6x3, 15-date ministack.
HALF_WINDOW = (11, 5)
STRIDES = (6, 3)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# tracing fmt: stage="unwrap" event="complete" elapsed_s=12.3 rss_kib=.. peak_rss_kib=..
# elapsed_s is Rust f64 Display, so sub-millisecond stages arrive in scientific
# notation (`elapsed_s=1.666e-6`). Matching only [\d.] silently turns 1.666e-6
# into 1.666 seconds, which is how a microsecond stage fakes a 4.9 s cost.
_STAGE = re.compile(
    r'stage="(?P<stage>\w+)".*?event="complete".*?'
    r'elapsed_s=(?P<elapsed>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
)
# /usr/bin/time -l on macOS: "  <bytes>  maximum resident set size"
_MAXRSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)


def select_dates(stack: Path, count: int) -> list[Path]:
    """The first `count` acquisitions in date order."""
    files = sorted(stack.glob("OPERA_*.h5"))
    if len(files) < count:
        raise SystemExit(f"{stack} holds {len(files)} acquisitions; need {count}")
    return files[:count]


def raster_shape(granule: Path) -> tuple[int, int]:
    import h5py

    found: list[tuple[int, int]] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and name.endswith("VV"):
            found.append(obj.shape)

    with h5py.File(granule, "r") as handle:
        handle.visititems(visit)
    if not found:
        raise SystemExit(f"no 2-D VV dataset in {granule}")
    return found[0]


def gen_config(files: list[Path], work_dir: Path, out: Path) -> None:
    subprocess.run(
        [
            str(VENV / "dolphin"), "config",
            "--slc-files", *map(str, files),
            "-sds", SUBDATASET,
            "--work-directory", str(work_dir),
            "-ms", str(len(files)),
            "-hw", str(HALF_WINDOW[0]), str(HALF_WINDOW[1]),
            "-s", str(STRIDES[0]), str(STRIDES[1]),
            "-o", str(out),
        ],
        check=True,
        capture_output=True,
    )
    # dolphin writes threads_per_worker from its own host core count; dolphinRust
    # models the field for YAML compatibility only and rejects anything but 1,
    # threading its own work through rayon instead. Rewriting it changes no
    # dolphinRust behaviour.
    out.write_text(
        re.sub(r"^(\s*threads_per_worker:).*$", r"\1 1", out.read_text(), flags=re.M)
    )


def run_once(config: Path, log: Path) -> dict[str, object]:
    """One full `dolphin run` under /usr/bin/time -l; returns stages + peak RSS."""
    env = {**os.environ, "RUST_LOG": "info"}
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-l", str(RUST_BIN), "run", "--config", str(config)],
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - start
    text = _ANSI.sub("", proc.stdout + proc.stderr)
    log.write_text(text)
    if proc.returncode != 0:
        raise SystemExit(f"dolphin run failed ({proc.returncode}); see {log}")

    # The library's per-stage rss_kib/peak_rss_kib breadcrumbs read 0 on this
    # host, so whole-process /usr/bin/time -l max-RSS is the only real memory
    # number here and per-stage RSS is deliberately not reported.
    stages = [
        {"stage": m.group("stage"), "elapsed_s": float(m.group("elapsed"))}
        for m in _STAGE.finditer(text)
    ]
    maxrss = _MAXRSS.search(text)
    return {
        "wall_s": wall,
        "stages": stages,
        "peak_rss_gib": int(maxrss.group(1)) / 1024**3 if maxrss else None,
    }


def machine() -> dict[str, object]:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "platform": platform.platform(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "physical_cores": sysctl("hw.physicalcpu"),
        "logical_cores": sysctl("hw.logicalcpu"),
        "memory_gib": round(int(sysctl("hw.memsize") or 0) / 1024**3, 1),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default (all cores)"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--dates", type=int, default=15)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "disp_s1.json")
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "bench" / "runs" / "disp_s1"
    )
    args = parser.parse_args()

    if not RUST_BIN.exists():
        raise SystemExit("build first: cargo build --release -p dolphin-cli")

    files = select_dates(args.stack, args.dates)
    rows, cols = raster_shape(files[0])

    run_dir = args.run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    work = run_dir / "work"
    work.mkdir(parents=True)
    config = run_dir / "config.yaml"
    gen_config(files, work, config)

    print(
        f"### {args.dates} dates x {rows}x{cols} "
        f"({rows * cols / 1e6:.2f} Mpix), half_window={HALF_WINDOW}, strides={STRIDES}"
    )
    reps = []
    for rep in range(args.reps):
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        out = run_once(config, run_dir / f"run{rep}.log")
        print(f"  rep {rep}: {out['wall_s']:.1f} s")
        reps.append(out)

    # cold = first invocation, warm = median of the rest (bench.py's convention).
    warm = reps[1:] or reps
    result = {
        "cold_wall_s": reps[0]["wall_s"],
        "wall_s": statistics.median(r["wall_s"] for r in warm),
        "peak_rss_gib": statistics.median(
            r["peak_rss_gib"] for r in warm if r["peak_rss_gib"]
        ),
        "reps": args.reps,
        "all_wall_s": [r["wall_s"] for r in reps],
        "stages": [
            {
                "stage": name,
                "elapsed_s": statistics.median(
                    s["elapsed_s"] for r in warm for s in r["stages"] if s["stage"] == name
                ),
            }
            for name in [s["stage"] for s in reps[0]["stages"]]
        ],
    }

    payload = {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dolphinrust_version": subprocess.run(
            [str(RUST_BIN), "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "stack": str(args.stack.relative_to(ROOT)),
        "burst_id": files[0].name.split("_")[3],
        "dates": args.dates,
        "date_range": [files[0].name.split("_")[4][:8], files[-1].name.split("_")[4][:8]],
        "rows": rows,
        "cols": cols,
        "megapixels": round(rows * cols / 1e6, 3),
        "half_window_xy": list(HALF_WINDOW),
        "strides_xy": list(STRIDES),
        "machine": machine(),
        **result,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nwarm wall  {result['wall_s']:.1f} s "
          f"({result['wall_s'] / 60:.2f} min); cold {result['cold_wall_s']:.1f} s")
    if result["peak_rss_gib"]:
        print(f"peak RSS   {result['peak_rss_gib']:.2f} GiB")
    print(f"\n{'stage':<24}{'seconds':>10}{'% total':>10}")
    for s in result["stages"]:
        share = 100 * s["elapsed_s"] / result["wall_s"]
        print(f"{s['stage']:<24}{s['elapsed_s']:>10.2f}{share:>9.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
