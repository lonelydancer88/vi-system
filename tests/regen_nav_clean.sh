#!/bin/bash
# 只重跑"比 prices.parquet 旧"的净值缓存档，再重画曲线（可反复跑，幂等）。
cd /Users/hpl/WorkBuddy/2026-09-07-15-54-23 || exit 1
PY=/Users/hpl/.workbuddy/binaries/python/versions/3.13.12/bin/python3
P=data/real_universe/prices.parquet
for K in 30 5 3 equal; do
  C=out/_navcache/nav_$K.csv
  if [ -f "$C" ] && [ "$C" -nt "$P" ]; then echo "[skip] $K 已是最新"; continue; fi
  echo "[dump] $K ..."
  $PY tests/regen_nav_curve.py --dump $K || echo "[fail] $K"
done
echo "[plot] 重画曲线"
$PY tests/regen_nav_curve.py --plot
echo "[$(date)] DONE"
