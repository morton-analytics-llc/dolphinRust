#!/usr/bin/env bash
# PUSH-2: production-concurrency throughput, native MCF vs SNAPHU subprocess.
#
# Two questions, measured not asserted:
#   A) CPU SECONDS PER FRAME (user+sys via /usr/bin/time -l, which rolls up reaped
#      SNAPHU children). At full core saturation frames/hour = NCPU*3600/cpu_s, so
#      the lower-CPU backend wins aggregate throughput regardless of thread split.
#   B) ACTUAL frames/hour at a concurrency operating point: run K frames as K
#      concurrent processes (GP's per-job model), each with T=NCPU/K threads, so
#      total threads ~= NCPU. Batch wall -> frames/hour.
#
# A "frame" = one unwrap_network over EPOCHS-1 ifgs at ROWS x COLS, DENSE (real
# residue density ~2.6% at 1024^2). Env: ROWS COLS EPOCHS (frame size), TILES
# (native tile grid), KS (concurrency points for part B).
# No `set -e`: the bench uses `[ cond ] && action` guards that return non-zero
# when the condition is false, which would abort under -e. -u/-o pipefail stay.
set -uo pipefail
cd "$(dirname "$0")/.."

BIN=target/release/examples/unwrap_bench
cargo build --release --example unwrap_bench -p dolphin-workflows >/dev/null 2>&1
NCPU=$(sysctl -n hw.ncpu)
ROWS=${ROWS:-1024}; COLS=${COLS:-1024}; EPOCHS=${EPOCHS:-8}
TILES=${TILES:-"4 8"}
export ROWS COLS EPOCHS DENSE=1

# Parse "real user sys" from the first line of /usr/bin/time -l, + max RSS (MB).
timed() { # args: env... BIN ; echoes "wall_s cpu_s rss_mb"
  local out
  out=$( { /usr/bin/time -l "$@" >/dev/null ; } 2>&1 )
  awk '
    /real/ && /user/ && /sys/ { real=$1; user=$3; sys=$5 }
    /maximum resident set size/ { rss=$1 }
    END { printf "%.1f %.1f %.0f", real, user+sys, rss/1048576 }' <<<"$out"
}

echo "# box: ${NCPU} cores   frame: ${ROWS}x${COLS} dense, $((EPOCHS-1)) ifgs"
echo "# === A) per-frame wall / CPU-s / RSS (8 threads) ==="
printf "%-22s %8s %8s %8s %12s\n" backend wall_s cpu_s rss_mb frames_per_hr_sat
for spec in "snaphu:0" $(for t in $TILES; do echo "native:$t"; done); do
  b=${spec%%:*}; tile=${spec##*:}
  read -r wall cpu rss < <(RAYON_NUM_THREADS=8 BACKEND=$b TILE=$tile timed "$BIN")
  fph=$(echo "scale=0; $NCPU*3600/$cpu" | bc)
  label="$b"; [ "$tile" = 0 ] || label="$b(tile$tile)"
  printf "%-22s %8s %8s %8s %12s\n" "$label" "$wall" "$cpu" "$rss" "$fph"
done

echo "# === B) frames/hour at concurrency K (each frame NCPU/K threads) ==="
printf "%-22s %4s %4s %8s %12s\n" backend K T batch_s frames_per_hr
NTOTAL=${NTOTAL:-6}   # frames per batch point (>= max K so a wave fully saturates)
for spec in "snaphu:0" $(for t in $TILES; do echo "native:$t"; done); do
  b=${spec%%:*}; tile=${spec##*:}
  label="$b"; [ "$tile" = 0 ] || label="$b(tile$tile)"
  for K in ${KS:-"1 2 3 4 6"}; do
    T=$(( NCPU / K )); [ "$T" -lt 1 ] && T=1
    # Launch frames in waves of K (bash 3.2 has no `wait -n`); each wave is a
    # barrier, a slight conservative bias applied equally to both backends.
    start=$(date +%s.%N)
    done_frames=0
    while [ "$done_frames" -lt "$NTOTAL" ]; do
      this=$K; [ $((done_frames + this)) -gt "$NTOTAL" ] && this=$((NTOTAL - done_frames))
      for ((w=0; w<this; w++)); do
        RAYON_NUM_THREADS=$T BACKEND=$b TILE=$tile "$BIN" >/dev/null 2>&1 &
      done
      wait
      done_frames=$((done_frames + this))
    done
    end=$(date +%s.%N)
    batch=$(echo "$end - $start" | bc)
    fph=$(echo "scale=0; $NTOTAL*3600/$batch" | bc)
    printf "%-22s %4s %4s %8.1f %12s\n" "$label" "$K" "$T" "$batch" "$fph"
  done
done
