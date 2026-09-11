"""重生成 out/nav-curve.png（五条累计净值对比曲线）。

用法: python3 tests/regen_charts.py

历史说明：本脚本曾顺带产出 out/positions-100w-2026-05-15.md（一张写死调仓日的
建仓手数参考表）。该表与「持仓建议」「回测报告」的手数内容重复，且调仓日固定在
2026-05-15 早已过期，故一并移除，本脚本现在只负责净值曲线。
需要手数表请用 tests/gen_position_advice.py。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "nav-curve.png", dpi=150)
plt.close(fig)
print("→ out/nav-curve.png")
print(f"30只: cagr={s30['cagr']:.2%} excess_hs300={s30['excess_cagr']:+.2%} "
      f"IR={s30['information_ratio']:.2f} 持仓={s30['avg_holdings']:.1f} 换手={s30['avg_turnover']:.1%}")
print(f"3只:  cagr={s3['cagr']:.2%} excess_hs300={s3['excess_cagr']:+.2%}")
print(f"5只:  cagr={s5['cagr']:.2%} excess_hs300={s5['excess_cagr']:+.2%}")
