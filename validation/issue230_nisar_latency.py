"""Issue #230, NISAR: how current is each source?

For the last N acquisitions of one NISAR track/frame, compare acquisition
start with when ASF's CMR first recorded
- the L1 RSLC (the raw input),
- the NISAR_ANC_TEC file covering it (the ionosphere ancillary `dolphin
  nisar` needs; `--no-ionosphere` runs without it), and
- JPL's L2 GSLC for the same acquisition (missing = not produced).

The raw path can run once both the RSLC and the TEC exist. Standard library
only; no credentials (CMR is public).

    python3 -I validation/issue230_nisar_latency.py --track 5 --direction A \\
        --frame 20 [--count 12] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
RSLC = "C2853145197-ASF"
GSLC = "C2854332392-ASF"
TEC = "C3622219145-ASF"


def get(params: dict) -> list[dict]:
    query = urllib.parse.urlencode(params, doseq=True)
    with urllib.request.urlopen(f"{CMR}?{query}", timeout=60) as response:
        return json.load(response)["items"]


def parse(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def inserted(item: dict) -> datetime:
    dates = item["umm"].get("ProviderDates", [])
    insert = [d["Date"] for d in dates if d["Type"] in ("Insert", "Create")]
    return parse(min(insert) if insert else item["meta"]["revision-date"])


def start(item: dict) -> datetime:
    return parse(item["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])


def end(item: dict) -> datetime:
    return parse(item["umm"]["TemporalExtent"]["RangeDateTime"]["EndingDateTime"])


def hours(a: datetime, b: datetime) -> float:
    return round((b - a).total_seconds() / 3600, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    key = f"{args.track:03d}_{args.direction}_{args.frame:03d}"
    pattern = {"options[readable_granule_name][pattern]": "true", "page_size": 2000}
    rslc = get({"collection_concept_id": RSLC, "readable_granule_name": f"NISAR_L1_*_RSLC_*_{key}_*", **pattern})
    gslc = get({"collection_concept_id": GSLC, "readable_granule_name": f"NISAR_L2_*_GSLC_*_{key}_*", **pattern})
    by_start: dict[str, dict] = {}
    for item in sorted(rslc, key=start):
        by_start.setdefault(item["umm"]["GranuleUR"].split("_")[11], item)
    gslc_first: dict[str, datetime] = {}
    for item in gslc:
        token = item["umm"]["GranuleUR"].split("_")[11]
        gslc_first[token] = min(gslc_first.get(token, inserted(item)), inserted(item))
    rows = []
    for token, item in sorted(by_start.items())[-args.count :]:
        acquired = start(item)
        tecs = get(
            {
                "collection_concept_id": TEC,
                "temporal": f"{(acquired - timedelta(minutes=1)).isoformat()},{(end(item) + timedelta(minutes=1)).isoformat()}",
                "page_size": 50,
            }
        )
        covering = [t for t in tecs if start(t) <= acquired - timedelta(minutes=1) and end(t) >= end(item)]
        tec_at = min((inserted(t) for t in covering), default=None)
        rslc_at = inserted(item)
        raw_ready = max(rslc_at, tec_at) if tec_at else None
        rows.append(
            {
                "acquired": acquired.isoformat(),
                "rslc_h": hours(acquired, rslc_at),
                "tec_h": hours(acquired, tec_at) if tec_at else None,
                "raw_path_ready_h": hours(acquired, raw_ready) if raw_ready else None,
                "gslc_h": hours(acquired, gslc_first[token]) if token in gslc_first else None,
            }
        )
    report = {"frame": key, "rows": rows}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
