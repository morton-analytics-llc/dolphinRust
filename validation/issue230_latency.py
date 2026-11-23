"""Issue #230: how current is each source?

For the last N acquisitions of the Houston (T034-071054-IW3) and T100
(T100-213486-IW2) bursts, compare acquisition time with
- when CDSE published the parent IW SLC (the raw burst is cut from it on
  demand, so this is when the burst becomes downloadable), and
- when ASF's CMR first recorded the OPERA CSLC-S1 granule for the same burst
  and date (missing = OPERA has not produced it).

Standard library only; no credentials (both catalogues are public).

    python3 -I validation/issue230_latency.py [--count 12] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
OPERA_CSLC = "OPERA_L2_CSLC-S1_V1"
BURSTS = {
    "T034-071054-IW3": (34, 71054, "IW3"),
    "T100-213486-IW2": (100, 213486, "IW2"),
}


def get_json(url: str, params: dict | list) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    with urllib.request.urlopen(f"{url}?{query}", timeout=60) as response:
        return json.load(response)


def parse(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def cdse_bursts(track: int, burst: int, swath: str, count: int) -> list[dict]:
    flt = (
        f"BurstId eq {burst} and SwathIdentifier eq '{swath}' and RelativeOrbitNumber eq {track} "
        "and PolarisationChannels eq 'VV'"
    )
    page = get_json(
        f"{CATALOGUE}/Bursts",
        {"$filter": flt, "$orderby": "ContentDate/Start desc", "$top": str(count)},
    )
    return page["value"]


def cdse_publication(product_id: str) -> datetime:
    row = get_json(f"{CATALOGUE}/Products({product_id})", {})
    return parse(row["PublicationDate"])


def opera_insert_time(label: str, start: datetime) -> datetime | None:
    day = start.strftime("%Y%m%d")
    page = get_json(
        CMR,
        [
            ("short_name", OPERA_CSLC),
            ("native_id", f"OPERA_L2_CSLC-S1_{label}_{day}T*"),
            ("options[native_id][pattern]", "true"),
            ("page_size", "20"),
        ],
    )
    inserts = [
        parse(d["Date"])
        for item in page.get("items", [])
        for d in item["umm"].get("ProviderDates", [])
        if d["Type"] == "Insert"
    ]
    return min(inserts) if inserts else None


def hours(a: datetime, b: datetime | None) -> float | None:
    return None if b is None else round((b - a).total_seconds() / 3600.0, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results/issue230_raw_front_end/latency.json",
    )
    args = parser.parse_args()
    measured = datetime.now(timezone.utc)
    report = {"measured_at": measured.isoformat(), "bursts": {}}
    for label, (track, burst, swath) in BURSTS.items():
        rows = []
        for record in cdse_bursts(track, burst, swath, args.count):
            acquired = parse(record["AzimuthTime"])
            published = cdse_publication(record["ParentProductId"])
            opera = opera_insert_time(label, acquired)
            rows.append(
                {
                    "acquired": acquired.isoformat(),
                    "slc": record["ParentProductName"],
                    "cdse_published": published.isoformat(),
                    "cdse_latency_h": hours(acquired, published),
                    "opera_cslc_inserted": opera.isoformat() if opera else None,
                    "opera_latency_h": hours(acquired, opera),
                    "opera_missing_after_h": None if opera else hours(acquired, measured),
                }
            )
        report["bursts"][label] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for label, rows in report["bursts"].items():
        print(label)
        for row in rows:
            print(
                f"  {row['acquired'][:16]}  CDSE {row['cdse_latency_h']:>7} h  "
                f"OPERA {row['opera_latency_h'] if row['opera_latency_h'] is not None else 'none (' + str(row['opera_missing_after_h']) + ' h)'}"
            )


if __name__ == "__main__":
    main()
