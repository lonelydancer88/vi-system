"""真实数据端到端冒烟测试：用腾讯自选股抓一批 A 股，跑通全链路。

用法：
    python -m tests.real_smoke
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vi_system.config import load_config
from vi_system.data.westock import build_westock_store
from vi_system.pipeline import universe, metrics, vetoes, factors, valuation
from vi_system.monitor.alerts import weekly_report, thesis_alerts, price_alerts
from vi_system.backtest.engine import run_backtest

# 跨行业蓝筹 + 一只地产（测试金融/地产覆盖表）。真实代码，仅用于验证适配器。
UNIVERSE = [
    "600519.SH",  # 贵州茅台 白酒
    "000858.SZ",  # 五粮液   白酒
    "000333.SZ",  # 美的集团 家电
    "000651.SZ",  # 格力电器 家电
    "600036.SH",  # 招商银行 银行
    "601398.SH",  # 工商银行 银行
    "600276.SH",  # 恒瑞医药 医药
    "600585.SH",  # 海螺水泥 制造业
    "603288.SH",  # 海天味业 食品
    "000002.SZ",  # 万科A    地产
]


def main():
    cfg = load_config()
    import pathlib
    tmp = str(pathlib.Path(__file__).resolve().parent.parent / "data" / "real")
    pathlib.Path(tmp).mkdir(parents=True, exist_ok=True)
    print(f"[build] 抓取 {len(UNIVERSE)} 只真实数据 → {tmp}")

    st = build_westock_store(
        tmp, UNIVERSE, start="2015-01-01", end="2026-09-07",
        periods=52, with_pledge=True, verbose=True,
    )
    print("\n[store] summary:", st.summary())

    asof = "2026-04-30"  # 2025 年报已公告（4 月前）
    print(f"\n[screen] asof={asof}")
    u = universe.build_universe(st, asof, cfg)
    print("  宇宙:", u.shape, "| 行业:", u["industry"].value_counts().to_dict())
    print("  流动性过滤后:", u.shape)

    m = metrics.compute_metrics(st.facts_asof(asof), st.market_asof(asof), u)
    print("  指标表:", m.shape)
    cols = ["code", "ep", "bp", "gpa", "roic", "accruals", "altman_z",
            "beneish_m", "ocf_to_ni_5y", "pledge_ratio", "interest_coverage"]
    print(m[cols].to_string())

    passed, rejected = vetoes.apply_vetoes(m, cfg)
    print(f"\n[vetoes] 通过 {passed.shape[0]} / 否决 {rejected.shape[0]}")
    if not rejected.empty:
        print(vetoes.veto_summary(rejected).to_string())

    f = factors.score_factors(passed, cfg)
    print(f"\n[factors] 过 AND 闸 {int(f['passes_gate'].sum())} / {len(f)}")
    print(f[["code", "value_pct", "quality_pct", "safety_pct",
             "total_score", "passes_gate"]].sort_values("total_score", ascending=False).to_string())

    v = valuation.valuate(f, cfg)
    print("\n[valuation] 逆向 DCF 隐含增长率 & 结论")
    print(v[["code", "ep", "g_implied", "v_mid", "verdict"]].sort_values("g_implied").to_string())

    # 监控周报
    rep = weekly_report(thesis_alerts(m, cfg), price_alerts(v), None)
    print("\n[monitor]\n" + rep)

    # 回测（小规模，验证 point-in-time 链路在真实数据上不崩）
    print("\n[backtest] 2017-2026 样本...")
    r = run_backtest(st, cfg, "2017-01-01", "2026-06-30")
    s = r["stats"]
    print("  年化策略 {:.2%} | 基准 {:.2%} | 超额 {:.2%} | 最大回撤 {:.2%} | 夏普 {:.2f}".format(
        s.get("cagr", float("nan")), s.get("bench_cagr", float("nan")),
        s.get("excess_cagr", float("nan")), s.get("max_drawdown", float("nan")),
        s.get("sharpe", float("nan"))))
    print("  平均持股数 {:.1f} | 平均换手 {:.2%}".format(
        s.get("avg_holdings", float("nan")), s.get("avg_turnover", float("nan"))))
    print("\n完成。")


if __name__ == "__main__":
    main()
