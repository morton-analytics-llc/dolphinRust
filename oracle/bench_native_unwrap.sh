#!/usr/bin/env bash
# Phase-7 benchmark: native in-process MCF vs SNAPHU subprocess.
# CPU sweep (wall ms) + max-RSS comparison. macOS /usr/bin/time -l for RSS.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=target/release/examples/unwrap_bench
cargo build --release --example unwrap_bench -p dolphin-workflows >/dev/null 2>&1
NCPU=$(sysctl -n hw.ncpu)
ROWS=${ROWS:-512}; COLS=${COLS:-512}
export ROWS COLS

echo "# grid=${ROWS}x${COLS}  ncpu=${NCPU}"
echo "# === CPU sweep (wall_ms) ==="
for B in snaphu native; do
  for E in 12 30; do
    for T in 1 2 4 8 "$NCPU"; do
      BACKEND=$B EPOCHS=$E RAYON_NUM_THREADS=$T "$BIN"
    done
  done
done

echo "# === max RSS (30 epochs, 8 threads) ==="
for B in snaphu native; do
  # macOS BSD time -l prints 'maximum resident set size' in BYTES on stderr.
  OUT=$( { BACKEND=$B EPOCHS=30 RAYON_NUM_THREADS=8 /usr/bin/time -l "$BIN" ; } 2>&1 )
  RSS=$(echo "$OUT" | awk '/maximum resident set size/ {print $1}')
  MB=$(echo "scale=1; $RSS/1048576" | bc)
  echo "backend=$B max_rss_bytes=$RSS max_rss_mb=$MB"
done
