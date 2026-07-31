#!/usr/bin/env bash
# Force page-cache reclaim when buff/cache is high. On unified-memory boxes
# (GB10) heavy page cache depresses large-model decode 20-25% — measured
# 2026-07-16: Qwen3-30B q6 tg128 57.4 with 114GB cache vs 74.0 clean, same
# binary + file. Used as PRE_RUN_CMD by gb10_two_way.sh. No sudo needed:
# allocating + touching anonymous memory forces the kernel to drop clean
# cache pages. Skips when cache is already low (threshold GB, default 40).
THRESH_GB="${1:-40}"
cache_kb=$(awk '/^Cached:/{print $2}' /proc/meminfo)
if [ "$cache_kb" -lt $((THRESH_GB*1024*1024)) ]; then exit 0; fi
avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
target_gb=$(( avail_kb/1024/1024 - 10 ))
[ "$target_gb" -gt 100 ] && target_gb=100
[ "$target_gb" -lt 8 ] && exit 0
echo "    (reclaiming page cache: $((cache_kb/1024/1024))GB cached, touching ${target_gb}GB)"
python3 - "$target_gb" <<'PY'
import ctypes, sys
n = int(sys.argv[1]) // 2
chunks = []
try:
    for _ in range(n):
        b = ctypes.create_string_buffer(2*1024**3)
        ctypes.memset(b, 1, 2*1024**3)
        chunks.append(b)
except MemoryError:
    pass
PY
