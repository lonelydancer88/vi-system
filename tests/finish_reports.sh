#!/bin/bash
# 续跑：净值曲线 + 三档对沪深300 基准对比（任务被回收后重跑本脚本即可续上，
# 已完成的 bt_bench_N.log 会覆盖重算，幂等）。
cd /Users/hpl/WorkBuddy/2026-09-07-15-54-23 || exit 1
PY=/Users/hpl/.workbuddy/binaries/python/versions/3.13.12/bin/python3
DB="--db data/real_universe"

echo "[$(date)] A 净值曲线 + 图表"
$PY tests/regen_charts.py
echo "[$(date)] B 基准对比 top30"
$PY -m vi_system.cli $DB backtest --benchmark sh000300 --max-holdings 30 > /tmp/bt_bench_30.log 2>&1
echo "[$(date)] C 基准对比 top5"
$PY -m vi_system.cli $DB backtest --benchmark sh000300 --max-holdings 5 > /tmp/bt_bench_5.log 2>&1
echo "[$(date)] D 基准对比 top3"
$PY -m vi_system.cli $DB backtest --benchmark sh000300 --max-holdings 3 > /tmp/bt_bench_3.log 2>&1
echo "[$(date)] ALL DONE"
