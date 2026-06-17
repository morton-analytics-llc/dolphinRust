#!/usr/bin/env bash
# End-to-end validation driver: one dolphin config, both engines, compare.
#
#   validation/run.sh <speckle>
#
# Synthesizes a date-named CSLC stack, generates a real dolphin DisplacementWorkflow
# YAML via `dolphin config`, confirms dolphinRust parses it unchanged, runs the Python
# oracle and the Rust engine into per-engine work dirs, then diffs the outputs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/oracle/.venv/bin"
RUST_BIN="$ROOT/target/release/dolphin"
SPECKLE="${1:-0.05}"
TAG="speckle_${SPECKLE}"
RUN="$ROOT/validation/runs/$TAG"
DATA="$RUN/data"

echo "### validation run: speckle=$SPECKLE -> $RUN"
rm -rf "$RUN"; mkdir -p "$DATA"

"$VENV/python" "$ROOT/validation/gen_stack.py" --outdir "$DATA" --speckle "$SPECKLE"

# Real dolphin DisplacementWorkflow config (the canonical schema), single source of truth.
# Generated per engine so each gets its own work_directory; identical otherwise. That the
# Rust engine consumes a genuine `dolphin config` YAML unchanged is the compatibility claim.
gen_config() {  # <work_dir> <outfile>
  "$VENV/dolphin" config \
    --slc-files "$DATA"/cslc_*.h5 \
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
  --label "$TAG" --json-out "$RUN/result.json"
