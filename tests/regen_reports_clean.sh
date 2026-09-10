#!/bin/bash
# 干净行情重建后，重跑三档回测报告 + 净值曲线。
# 注意：托管后台任务约 20 分钟会被回收，靠本脚本重跑续上（每步自带日志）。
cd /Users/hpl/WorkBuddy/2026-09-07-15-54-23 || exit 1
PY=/Users/hpl/.workbuddy/binaries/python/versions/3.13.12/bin/python3
DB="--db data/real_universe"

echo "[$(date)] 1/5 backtest CLI（含沪深300基准）"
$PY -m vi_system.cli $DB backtest --benchmark sh000300
echo "[$(date)] 2/5 报告 top5"
$PY tests/gen_backtest_report.py --holdings 5
echo "[$(date)] 3/5 报告 top3"
$PY tests/gen_backtest_report.py --holdings 3
echo "[$(date)] 4/5 报告 top30"
$PY tests/gen_backtest_report.py --holdings 30
echo "[$(date)] 5/5 净值曲线 + 图表"
$PY tests/regen_charts.py
echo "[$(date)] ALL DONE"
