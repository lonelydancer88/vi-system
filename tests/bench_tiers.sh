#!/bin/bash
# 补跑三档（与回测报告同配置 --max-holdings N，会自动放宽单票/行业上限）对沪深300 的
# 超额年化与胜率，用于刷新 README「效果速览」表。
cd /Users/hpl/WorkBuddy/2026-09-07-15-54-23 || exit 1
PY=/Users/hpl/.workbuddy/binaries/python/versions/3.13.12/bin/python3
DB="--db data/real_universe"
for N in 30 5 3; do
  echo "[$(date)] benchmark --max-holdings $N 开始"
  $PY -m vi_system.cli $DB backtest --benchmark sh000300 --max-holdings "$N" \
      > "/tmp/bt_bench_$N.log" 2>&1
  echo "[$(date)] benchmark --max-holdings $N 完成"
done
echo "[$(date)] ALL DONE"
