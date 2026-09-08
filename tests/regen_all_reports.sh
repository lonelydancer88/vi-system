#!/bin/bash
# 600938 股本修复后，全量重生成 out/ 报告与图表。
cd /Users/hpl/WorkBuddy/2026-09-07-15-54-23 || exit 1
DB="--db data/real_universe"
set -e
echo "[$(date)] 1/9 backtest (default)" 
python3 -m vi_system.cli $DB backtest > /tmp/r1.log 2>&1
echo "[$(date)] 2/9 backtest --benchmark sh000300"
python3 -m vi_system.cli $DB backtest --benchmark sh000300 > /tmp/r2.log 2>&1
echo "[$(date)] 3/9 backtest --max-holdings 3"
python3 -m vi_system.cli $DB backtest --max-holdings 3 > /tmp/r3.log 2>&1
echo "[$(date)] 4/9 backtest --benchmark sh000300 --max-holdings 3"
python3 -m vi_system.cli $DB backtest --benchmark sh000300 --max-holdings 3 > /tmp/r4.log 2>&1
echo "[$(date)] 5/9 backtest --split"
python3 -m vi_system.cli $DB backtest --split > /tmp/r5.log 2>&1
echo "[$(date)] 6/9 trades"
python3 -m vi_system.cli $DB trades > /tmp/r6.log 2>&1
echo "[$(date)] 7/9 trades --max-holdings 3"
python3 -m vi_system.cli $DB trades --max-holdings 3 > /tmp/r7.log 2>&1
echo "[$(date)] 8/9 reasons"
python3 -m vi_system.cli $DB reasons > /tmp/r8.log 2>&1
echo "[$(date)] 9/9 reasons --max-holdings 3"
python3 -m vi_system.cli $DB reasons --max-holdings 3 > /tmp/r9.log 2>&1
echo "[$(date)] A nav-curve + positions"
python3 tests/regen_charts.py > /tmp/ra.log 2>&1
echo "[$(date)] ALL DONE"
