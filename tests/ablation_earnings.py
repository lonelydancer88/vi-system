"""隔离实验：季度化业绩变差，是「利润(TTM)算错/有噪」还是「调仓频率」导致的？

三个对照（都基于同一份 facts，仅改因子口径/频率）：
  A. [5,9]    + 年报利润 : 旧基线（已知 top5 ≈ +12.3%）
  B. [5,9,11] + TTM 利润 : 当前实现（刚出的报告，top5 ≈ +11.3%）
  C. [5,9,11] + 年报利润 : 频率提高但利润仍用最新年报（隔离「频率」单一变量）

判读：
  - C ≈ A 且 C >> B  → TTM 利润口径在拖后腿（利润预测有误/有噪）
  - C ≈ B 且 C << A  → 「频率」本身在拖后腿，利润口径没问题
"""
import sys, copy, importlib
sys.path.insert(0, "/Users/hpl/WorkBuddy/2026-09-07-15-54-23")

import pandas as pd
import numpy as np

import vi_system.pipeline.metrics as metrics
from vi_system.config import load_config
from vi_system.data import store as store_mod
from vi_system.backtest import engine as bt

ROOT = "/Users/hpl/WorkBuddy/2026-09-07-15-54-23/data/real_universe"
st = store_mod.Store(ROOT)
cfg = load_config()


def _ttm_from_annual(full, flow_fields):
    """cur 的流量字段强制用「最新年报」值（不跨季度），等价于『只用年报利润』。

    资产负债表时点值仍取最新披露期（与 B 组一致），故 B/C 的唯一差异就是
    利润等流量字段的口径（TTM vs 年报），正好隔离『利润预测』这一变量。
    """
    rep = full[full["period"].astype(str).str.endswith(("1231", "0630", "0930", "0331"))]
    annual = rep[rep["period"].astype(str).str.endswith("1231")]
    if annual.empty:
        return {}
    latest_annual = annual.sort_values("announce_date").iloc[-1]
    latest_period = str(rep.sort_values("announce_date")["period"].iloc[-1])
    cols = [c for c in flow_fields if c in rep.columns]
    return {latest_period: {f: latest_annual.get(f, np.nan) for f in cols}}


def set_mode(mode):
    importlib.reload(metrics)  # 复位为原始(真实 TTM)实现
    if mode == "annual":
        metrics._ttm_for_periods = _ttm_from_annual


def run(months, tag):
    c = cfg.with_value("backtest.rebalance_months", list(months)).with_value("backtest.rebalance_day", 15)
    r = bt.run_backtest(st, c, "2013-01-01", "2026-09-07", benchmark="equal", max_holdings=5)
    s = r["stats"]
    print(f"[{tag}] months={months}  periods={s['periods']}  "
          f"年化={s['cagr']*100:+.2f}%  超额={s['excess_cagr']*100:+.2f}%  "
          f"总收益={s['total_return']*100:+.1f}%  回撤={s['max_drawdown']*100:.2f}%  "
          f"夏普={s['sharpe']:.2f}  胜率={s['win_rate_vs_bench']*100:.1f}%")
    return s


if __name__ == "__main__":
    print("=== 隔离实验：频率 vs 利润口径（top5 档）===")
    set_mode("ttm")      # 原始真实实现
    run([5, 9],       "A 旧基线 [5,9]+年报")
    run([5, 9, 11],   "B 当前   [5,9,11]+TTM")
    set_mode("annual")
    run([5, 9, 11],   "C 对照   [5,9,11]+年报")
