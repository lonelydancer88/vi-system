"""重生成 out/nav-curve.png 与 out/positions-100w-2026-05-15.md（修复后口径）。

用法: python3 tests/regen_charts.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# macOS 中文字体（缺失时图内中文会变方块）
for _f in ("/System/Library/Fonts/PingFang.ttc",
           "/System/Library/Fonts/Hiragino Sans GB.ttc",
           "/System/Library/Fonts/STHeiti Medium.ttc"):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest.engine import run_backtest

DB = ROOT / "data" / "real_universe"
OUT = ROOT / "out"
ASOF = "2026-05-15"
CAPITAL = 1_000_000.0

store = Store(DB)
cfg = load_config()

r30 = run_backtest(store, cfg, "2013-01-01", "2026-12-31",
                   label="30只", benchmark="sh000300")
r3 = run_backtest(store, cfg, "2013-01-01", "2026-12-31",
                  label="3只", benchmark="sh000300", max_holdings=3)
r5 = run_backtest(store, cfg, "2013-01-01", "2026-12-31",
                  label="5只", benchmark="sh000300", max_holdings=5)
req = run_backtest(store, cfg, "2013-01-01", "2026-12-31",
                   label="等权", benchmark="equal")

s30, s3, s5 = r30["stats"], r3["stats"], r5["stats"]
eff = r30["effective_start"]

# --------------------------------------------------------------- 净值曲线
fig, ax = plt.subplots(figsize=(11, 6))
nav30 = r30["nav"]
nav3 = r3["nav"].reindex(nav30.index)
nav5 = r5["nav"].reindex(nav30.index)
ax.plot(nav30.index, nav30["nav"], lw=2.0, color="#c0392b",
        label=f"策略·30只（年化 {s30['cagr']:+.1%}）")
ax.plot(nav30.index, nav3["nav"], lw=1.6, color="#e67e22",
        label=f"策略·最多3只（年化 {s3['cagr']:+.1%}）")
ax.plot(nav30.index, nav5["nav"], lw=1.6, color="#27ae60",
        label=f"策略·最多5只（年化 {s5['cagr']:+.1%}）")
ax.plot(nav30.index, nav30["bench"], lw=1.6, color="#2980b9",
        label=f"沪深300·价格回报（年化 {s30['bench_cagr']:+.1%}）")
ax.plot(nav30.index, req["nav"]["bench"].reindex(nav30.index), lw=1.2,
        ls="--", color="#7f8c8d",
        label=f"等权全市场基准（年化 {req['stats']['bench_cagr']:+.1%}）")
ax.axhline(1.0, color="gray", lw=0.6, alpha=0.6)
ax.set_title(f"净值曲线（{eff} 起，剔除期初空仓期；分红再投口径，费用后）", fontsize=12)
ax.set_ylabel("净值（起点=1）")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "nav-curve.png", dpi=150)
plt.close(fig)
print("→ out/nav-curve.png")

# --------------------------------------------------------------- 手数参考表
px = store.load_prices()
px["date"] = pd.to_datetime(px["date"])
pidx = px.set_index(["date", "code"])


def price_at(d, c):
    try:
        row = pidx.loc[(pd.Timestamp(d), c)]
        raw = float(row["close_raw"])
        return raw if raw > 0 else float("nan")
    except Exception:
        return float("nan")


def holdings_at(result, d):
    for h in result["holdings"]:
        if pd.Timestamp(h["date"]) == pd.Timestamp(d):
            return h["weights"]
    return None


uni = store.load_universe().set_index("code")["name"].to_dict()
lines = [
    "# 建仓手数参考表（z 版评分｜本金 100 万元）", "",
    "> 调仓日：2026-05-15｜真实价 close_raw｜1 手=100 股向下取整", "",
]
for title, res in (("30只版", r30), ("最多3只版", r3), ("最多5只版", r5)):
    w = holdings_at(res, ASOF)
    lines += [f"## {title}", "",
              "| 名称 | 权重 | 价 | 目标手数 | 占用资金(元) |",
              "|------|------|------|---------|------------|"]
    if w is None:
        lines.append("| —（该期无持仓） | | | | |")
    else:
        for c, wt in w.sort_values(ascending=False).items():
            p = price_at(ASOF, c)
            if math.isfinite(p) and p > 0:
                lots = int(wt * CAPITAL / p / 100)
                cost = lots * 100 * p
                lines.append(f"| {uni.get(c, c)} | {wt:.1%} | {p:.2f} | "
                             f"{lots if lots > 0 else '—'} | "
                             f"{f'{cost:,.0f}' if cost > 0 else '—'} |")
            else:
                lines.append(f"| {uni.get(c, c)} | {wt:.1%} | — | — | — |")
    lines.append("")
lines += ["---", "",
          f"> 30只版年化 {s30['cagr']:+.2%}｜最多3只版年化 {s3['cagr']:+.2%}｜最多5只版年化 {s5['cagr']:+.2%}"
          f"｜沪深300（价格回报）{s30['bench_cagr']:+.2%}"
          f"｜30只版超额 {s30['excess_cagr']:+.2%}/IR {s30['information_ratio']:.2f}。", ""]
(OUT / "positions-100w-2026-05-15.md").write_text("\n".join(lines), encoding="utf-8")
print("→ out/positions-100w-2026-05-15.md")
print(f"30只: cagr={s30['cagr']:.2%} excess_hs300={s30['excess_cagr']:+.2%} "
      f"IR={s30['information_ratio']:.2f} 持仓={s30['avg_holdings']:.1f} 换手={s30['avg_turnover']:.1%}")
print(f"3只:  cagr={s3['cagr']:.2%} excess_hs300={s3['excess_cagr']:+.2%}")
print(f"5只:  cagr={s5['cagr']:.2%} excess_hs300={s5['excess_cagr']:+.2%}")
