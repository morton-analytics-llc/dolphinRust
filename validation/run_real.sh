#!/usr/bin/env bash
# Real-data validation: one dolphin config, both engines, on a cropped real OPERA
# CSLC-S1 burst time series (fetch_real.py + crop_real.py must have run).
#
#   validation/run_real.sh
#
# Same structure as run.sh but the inputs are genuine OPERA granules (single
# burst T063-133231-IW1, 5 acquisitions, 12-day cadence), cropped to a window so
# both engines run quickly. Compares displacement + velocity (absolute scale).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/oracle/.venv/bin"
RUST_BIN="$ROOT/target/release/dolphin"
# Optional args: <cropped_dir> <run_tag> (default: the T144 coastal/land burst).
DATA="${1:-$ROOT/validation/real_data/cropped}"
RUN="$ROOT/validation/runs/${2:-real_T144-308011-IW2}"

[ -n "$(ls "$DATA"/OPERA_*.h5 2>/dev/null)" ] || {
  echo "no cropped real data — run fetch_real.py then crop_real.py" >&2; exit 1; }

echo "### real-data validation -> $RUN"
rm -rf "$RUN"; mkdir -p "$RUN/work_oracle" "$RUN/work_rust"

gen_config() {  # <work_dir> <outfile>
  "$VENV/dolphin" config \
    --slc-files "$DATA"/OPERA_*.h5 \
    -sds /data/VV \
    --work-directory "$1" \
    -ms 15 \
    -o "$2" >/dev/null
}
gen_config "$RUN/work_oracle" "$RUN/config.yaml"
gen_config "$RUN/work_rust" "$RUN/config_rust.yaml"

echo "### oracle: dolphin run"
"$VENV/dolphin" run "$RUN/config.yaml" > "$RUN/oracle.log" 2>&1

echo "### rust: dolphin run"
"$RUST_BIN" run --config "$RUN/config_rust.yaml" > "$RUN/rust.log" 2>&1

"$VENV/python" "$ROOT/validation/compare.py" \
  --oracle "$RUN/work_oracle" --rust "$RUN/work_rust" \
  --label "real_T144-308011-IW2" --json-out "$RUN/result.json"
