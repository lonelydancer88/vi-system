"""大宇宙排雷通过率统计（财务数据已全量抓完，纯本地、无需行情）。

对 data/real_universe 里 807 只（沪深300+中证500+造假股），在 2025-05-15
这个真实调仓时点，用 point-in-time 财务数据算 L3 排雷指标，统计被拦比例与原因。

目的：验证排雷规则在真实大宇宙是否“误杀过多”——若拦掉一半说明阈值过严，
若只拦 5% 说明规则有效且宽松（这正是我们想要的：早拦造假，少错杀好公司）。
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.pipeline.metrics import _metrics_for_code
from vi_system.pipeline.vetoes import apply_vetoes

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
ASOF = "2025-05-15"
PERIODS = 12  # 取近 12 期（≈3 年）用于 5 年 OCF/NI 等滚动指标


def main():
    cfg = load_config()
    st = Store(str(ROOT / "data" / "real_universe"))
    print(f"[load] facts_asof({ASOF}, periods={PERIODS}) ...", flush=True)
    panel = st.facts_asof(ASOF, periods=PERIODS)
    print(f"  面板 shape={panel.shape}, codes={panel['code'].nunique()}", flush=True)

    rows = []
    skipped = 0
    for code in panel["code"].unique():
        g = panel[panel["code"] == code]
        if g.empty or len(g) < 2:
            skipped += 1
            continue
        # 市值代理：账面净资产（无行情时）。仅影响 Altman 的市值分量与估值。
        eq = g.iloc[-1].get("total_equity")
        mktcap = float(eq) if pd.notna(eq) and eq and eq > 0 else 1.0
        try:
            m = _metrics_for_code(g, mktcap)
        except Exception as e:
            skipped += 1
            continue
        m["code"] = code
        rows.append(m)

    m_all = pd.DataFrame(rows)
    print(f"[metrics] 成功 {len(m_all)} 只，跳过 {skipped} 只", flush=True)

    # 必须在 apply_vetoes 之前把 industry 注入 m_all，否则 veto 里的
    # skip_industries / industry_overrides 全因 industry=NaN 而失效
    uni = pd.read_parquet(ROOT / "data" / "real_universe" / "universe.parquet")
    code2ind = dict(zip(uni["code"].astype(str), uni.get("industry", [""] * len(uni))))
    m_all = m_all.copy()
    m_all["industry"] = m_all["code"].map(code2ind).fillna("")

    passed, rejected = apply_vetoes(m_all, cfg)
    total = len(m_all)
    passed_n = len(passed)
    vetoed_n = total - passed_n
    print(f"[vetoes] 通过 {passed_n} / 拦截 {vetoed_n}", flush=True)

    # 各否决项触发数（一只可触发多项，rejected 每行一条）
    rule_counts = rejected["rule"].value_counts() if not rejected.empty else pd.Series(dtype=int)
    rule_desc = dict(zip(rejected["rule"], rejected["rule_desc"])) if not rejected.empty else {}

    stats = []
    for rule, n in rule_counts.items():
        desc = rule_desc.get(rule, rule)
        stats.append((rule, desc, int(n), f"{100*n/total:.1f}%"))
    stats.sort(key=lambda x: -x[2])

    # 按行业看通过率
    rejected_codes = set(rejected["code"]) if not rejected.empty else set()
    m_all["vetoed"] = m_all["code"].isin(rejected_codes)
    ind = m_all.groupby("industry").agg(n=("code", "size"), vetoed=("vetoed", "sum"))
    ind["pass_rate"] = (1 - ind["vetoed"] / ind["n"]).round(3)
    ind = ind.sort_values("n", ascending=False)

    lines = [
        f"# 大宇宙排雷通过率统计（{ASOF} 时点，{total} 只有效样本）",
        "",
        f"- 原始宇宙：807 只（沪深300+中证500+造假股）",
        f"- 有效样本：{total} 只（财务不足 2 期或计算异常跳过 {skipped} 只）",
        f"- **通过排雷：{passed_n} 只（{100*passed_n/total:.1f}%）**",
        f"- **被排雷拦截：{vetoed_n} 只（{100*vetoed_n/total:.1f}%）**",
        "",
        "## 各否决项触发数（一只可触发多项）",
        "",
        "| 规则 | 含义 | 触发数 | 占样本% |",
        "|---|---|---|---|",
    ]
    for rule, desc, n, p in stats:
        lines.append(f"| {rule} | {desc} | {n} | {p} |")

    lines += [
        "",
        "## 分行业通过率（按样本量排序，前 25）",
        "",
        "| 行业 | 样本数 | 拦截数 | 通过率 |",
        "|---|---|---|---|",
    ]
    for name, r in ind.head(25).iterrows():
        lines.append(f"| {name} | {int(r['n'])} | {int(r['vetoed'])} | {r['pass_rate']*100:.1f}% |")

    out = ROOT / "out" / "universe_veto_stats.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
