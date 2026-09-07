#!/bin/bash
# 等待行情抓取结束 → 补算市值(mktcap) → 校验 → 跑大宇宙真实回测。
# 用法：bash tests/wait_and_backtest.sh
cd /Users/hpl/WorkBuddy/2026-09-07-15-54-23 || exit 1

echo "[wait] 等待抓取进程结束 ..."
# [f]etch_real.py 的方括号写法避免 pgrep 匹配到自身命令行
while pgrep -f "[f]etch_real.py" > /dev/null; do sleep 60; done
echo "[wait] 抓取已结束，开始补算市值 ..."

python3 tests/attach_shares.py

STAT=$(python3 -c "
import pandas as pd
p = pd.read_parquet('data/real_universe/prices.parquet')
n = p['code'].nunique()
rate = p['mktcap'].notna().mean() if len(p) else 0
print(f'{n} {rate:.3f}')
")
N=$(echo "$STAT" | cut -d' ' -f1)
RATE=$(echo "$STAT" | cut -d' ' -f2)
echo "[wait] 行情覆盖 $N / 300 只，mktcap 有效率 $RATE"

if [ "$N" -ge 280 ] && [ "$(echo "$RATE >= 0.9" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "[wait] 数据充足，开始真实回测 ..."
    python -m vi_system.cli backtest --db data/real_universe --split
    echo "[wait] 回测结束"
else
    echo "[wait] 数据不达标（需 >=280 只且 mktcap 有效率 >=0.9），跳过回测"
fi
