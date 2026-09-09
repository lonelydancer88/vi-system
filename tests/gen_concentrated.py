"""生成集中持股版（最多 N 只）组合报告。

为什么不能直接拿默认配置跑：
  默认单票上限 8%，5 只等权就是 20%/只 —— 会被 _apply_caps 压回每只 8%，
  剩下 60%+ 变现金。那不是"集中"，那是变相空仓。
  故对集中组合沿用回测 max_holdings 的既有放宽惯例：
    单票上限 = max(原值, min(0.35, 0.9/N × 1.1))
    N <= 5 时行业上限放宽到 0.5

产出：out/portfolio-top{N}-{asof}.md，每个策略一个文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest import engine as bt
from vi_system.portfolio.constructor import build_portfolio


ASOF = "2026-09-09"
DB = "data/real_universe"
OUT = Path("out")
SIZES = [5, 3]


def concentrated_cfg(cfg, n: int):
    """按回测 max_holdings 惯例放宽集中组合的持仓上限。"""
    po = cfg.section("portfolio") or {}
    mpos = max(float(po.get("max_position", 0.08)), min(0.35, 0.9 / n * 1.1))
    out = cfg.with_value("portfolio.target_size", [1, n])
    out = out.with_value("portfolio.max_position", round(mpos, 4))
    out = out.with_value("portfolio.max_industry", 0.5 if n <= 5 else float(po.get("max_industry", 0.25)))
    return out, mpos


def report(pf: pd.DataFrame, scored: pd.DataFrame, n: int, mpos: float, asof: str) -> str:
    if pf.empty:
        return f"# 集中组合 Top{n}（{asof}）\n\n无持仓（候选不足或被全部排除）\n"

    # 估值信息回捞，便于判断"买的是不是好价格"
    extra = scored.set_index("code")
    pf = pf.copy()
    for col in ["verdict", "g_implied", "mktcap", "v_mid"]:
        pf[col] = pf["code"].map(extra[col]) if col in extra.columns else np.nan

    cash = max(0.0, 1 - pf["weight"].sum())
    lines = [
        f"# 集中组合 Top{n}（{asof}）",
        "",
        f"- 目标持股：最多 {n} 只",
        f"- 单票上限：{mpos:.1%}（默认 8%，集中组合按回测惯例放宽）",
        f"- 行业上限：50%（N≤5 放宽）",
        f"- 实际持仓：{len(pf)} 只",
        f"- 现金比例：{cash:.1%}",
        f"- 估值回灌：已启用（verdict=已到卖点 不作为买入标的）",
        "",
        "## 行业分布",
        "",
        "| 行业 | 权重 |",
        "|------|------|",
    ]
    iw = pf.groupby("industry")["weight"].sum().sort_values(ascending=False)
    for k, v in iw.items():
        lines.append(f"| {k} | {v:.2%} |")

    lines += [
        "",
        "## 持仓明细",
        "",
        "| 代码 | 名称 | 行业 | 权重 | 总分 | 估值结论 | 隐含增长 g* | 市值(亿) |",
        "|------|------|------|------|------|----------|------------|----------|",
    ]
    for _, r in pf.iterrows():
        g = r.get("g_implied", np.nan)
        g = f"{g * 100:.1f}%" if np.isfinite(g) else "—"
        mc = r.get("mktcap", np.nan)
        mc = f"{mc / 1e8:.0f}" if np.isfinite(mc) else "—"
        lines.append(
            f"| {r['code']} | {r.get('name', '')} | {r['industry']} | {r['weight']:.2%} "
            f"| {r['total_score']:.2f} | {r.get('verdict', '—')} | {g} | {mc} |"
        )

    # 与 Top30 基准版的差异提示
    lines += [
        "",
        "## 说明",
        "",
        f"- 权重 = 总分 × 流动性平方根调整后，经单票/行业上限迭代收敛，再乘 (1 − 最低现金 10%)。",
        f"- 集中组合天然放大单票风险：Top{n} 的波动会显著高于 Top30 分散版，",
        f"  此处仅为「若只持有 {n} 只」的选股结果，不等同于建议的仓位管理方式。",
        "",
    ]
    return "\n".join(lines)


def main():
    cfg = load_config(None)
    st = Store(DB)
    scored, _, uni = bt.screen_at(st, ASOF, cfg, with_valuation=True)
    if scored.empty:
        print("无候选")
        return 1
    print(f"宇宙 {len(uni)} → 通过排雷 {len(scored)} → 过 AND 门槛 {int(scored['passes_gate'].sum())}")

    for n in SIZES:
        ccfg, mpos = concentrated_cfg(cfg, n)
        pf = build_portfolio(scored, ccfg)
        OUT.mkdir(exist_ok=True)
        path = OUT / f"portfolio-top{n}-{ASOF}.md"
        path.write_text(report(pf, scored, n, mpos, ASOF), encoding="utf-8")
        cash = max(0.0, 1 - pf["weight"].sum()) if not pf.empty else 1.0
        print(f"→ {path}  持仓 {len(pf)} 只，现金 {cash:.1%}，单票上限 {mpos:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
