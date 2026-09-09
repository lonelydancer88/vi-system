"""生成「当前持仓建议」：按估值结论分组的持仓明细 + 100 万资金手数参考。

用法
----
    PYTHONPATH=. python3 tests/gen_position_advice.py [--asof 2026-09-10] [--capital 1000000]

与 out/portfolio-*.md 的区别：
  portfolio 只给权重；本文件补上三支柱 z、DCF 三情景估值、熊市下行空间，
  并按 verdict 分组（已到买点 / 合理区间 / 已到卖点），便于判断"现在该不该动手"。
  另附 100 万资金的建仓手数，可直接照单执行。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest import engine as bt
from vi_system.pipeline import vetoes as _vetoes
from vi_system.portfolio.constructor import build_portfolio

DB = "data/real_universe"
OUT = Path("out")

COLS = [("code", "代码"), ("name", "名称"), ("industry", "行业"), ("weight", "权重"),
        ("value_z", "Vz"), ("quality_z", "Qz"), ("safety_z", "Sz"),
        ("total_score", "总分"), ("mktcap", "市值亿"), ("g_implied", "g*"),
        ("v_mid", "中枢亿"), ("buy_point", "买点亿"), ("sell_point", "卖点亿"),
        ("downside_bear", "熊市下行")]


def _fmt(col, val):
    """col 必须是英文字段名（COLS 里的 key），不是中文表头。"""
    if col == "weight":
        return f"{val:.1%}"
    if col in ("value_z", "quality_z", "safety_z", "total_score"):
        return f"{val:.2f}" if np.isfinite(val) else "—"
    if col in ("mktcap", "v_mid", "buy_point", "sell_point"):
        return f"{val / 1e8:,.0f}" if np.isfinite(val) else "—"
    if col in ("g_implied", "downside_bear"):
        return f"{val * 100:.1f}%" if np.isfinite(val) else "—"
    return str(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-09-10")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    args = ap.parse_args()

    cfg = load_config()
    st = Store(DB)
    scored, _, uni = bt.screen_at(st, args.asof, cfg, with_valuation=True)
    if scored.empty:
        print("无候选")
        return 1
    pf = build_portfolio(scored, cfg)
    if pf.empty:
        print("组合为空")
        return 1

    # 估值/因子信息回捞；现价用于算手数
    extra = scored.set_index("code")
    mk = st.market_asof(args.asof).set_index("code")
    px_date = str(pd.to_datetime(mk["asof_date"].max()).date()) if "asof_date" in mk else "—"

    warns = _vetoes.warn_dead_rules(scored, cfg)
    n_gate = int(scored["passes_gate"].sum())
    cash = max(0.0, 1 - pf["weight"].sum())

    lines = [
        f"# 当前持仓建议（{args.asof}）", "",
        f"- 数据截面：**{args.asof}**；行情收盘价取 **{px_date}**（库内最新交易日）",
        f"- 选股管线：宇宙 {len(uni)} → 通过排雷 {len(scored)} → 过 AND 门 **{n_gate} 只**"
        f" → 组合 {len(pf)} 只，现金 {cash:.0%}",
        "- 权重来自 L6 组合构建（分数×流动性，单票≤8%、单行业≤25%、强制留≥10%现金）",
        "- **买入侧已启用估值回灌**：verdict=已到卖点 不作为买入标的（rules v1.3）",
    ]
    for w in warns:
        lines.append(f"- ⚠️ {w.strip()}")
    lines.append("")

    # 按 verdict 分组
    pf = pf.copy()
    pf["verdict"] = pf["code"].map(extra["verdict"])
    for v in ("已到买点", "合理区间", "已到卖点", "无法估值"):
        sub = pf[pf["verdict"] == v].copy()
        if sub.empty:
            continue
        lines += [f"## {v}（{len(sub)} 只）", ""]
        head = "| " + " | ".join(c[1] for c in COLS) + " | 现价 | 手数 |"
        lines += [head, "|" + "|".join([":---"] * (len(COLS) + 2)) + "|"]
        for _, r in sub.iterrows():
            row = extra.loc[r["code"]]
            cells = [_fmt(k, r["code"] if k == "code" else (r["weight"] if k == "weight" else row.get(k))) for k, c in COLS]
            price = float(mk.loc[r["code"], "close_raw"]) if r["code"] in mk.index else np.nan
            lots = ""
            if np.isfinite(price) and price > 0:
                sh = int(args.capital * r["weight"] / price / 100) * 100
                lots = f"{sh // 100}" if sh >= 100 else "0"
            cells += [f"{price:.2f}" if np.isfinite(price) else "—", lots]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## 说明", "",
        f"- 手数按本金 {args.capital:,.0f} 元、以 {px_date} 收盘价估算，1 手 = 100 股，已向下取整。",
        "- g\\* 为市场隐含永续增长率（越低越悲观、安全边际越足）；熊市下行为 v_bear 相对现价的空间。",
        "- 三支柱 z 为行业内中性值，>0 即跑赢行业均值；总分 = 0.4·价值 + 0.4·质量 + 0.2·安全。",
        "- **组合只看排名、不看择时**：本建议是「若今天按模型建仓」的截面结果，非盘中择时信号。",
        "",
    ]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"持仓建议-{args.asof}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {path}  持仓 {len(pf)} 只，现金 {cash:.0%}，行情口径 {px_date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
