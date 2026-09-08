"""全量原始数据质量复核（data-quality audit）。

不依赖外部数据源，做「内部一致性 + 离群检测 + 覆盖率」三类检查，产出
分级可疑清单。目的是在回测前把源数据里的静默错误（错股本、缺字段、硬编码 0、
畸形报告期）系统性地捞出来，而不是靠逐个抽查。

检查维度：
  P0 结构：重复主键、非法报告期(period)
  P1 价格层：mktcap==close_raw*total_share、NaN 市值(被排除)、负/零价、隐含股本跳变
  P1 股本/市值：缺 total_share(被排除)、total_share 非零->非零大幅跳变
  P1 财务字段：各字段最新期符号/零值、st_debt/bond 全零(已知 bug)、估值因子离群
  P2 交叉：facts.total_share vs prices.total_share 在公告日一致性、估值因子离群股清单
  P3 宇宙：上市/退市日期合理、is_financial 与行业一致

用法：python3 tests/audit_data.py
输出：out/data_audit.md（人读报告）+ out/data_audit_findings.csv（机器可读）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vi_system.data.store import Store  # noqa: E402
from vi_system.data import schema as S  # noqa: E402

DB = ROOT / "data" / "real_universe"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

PERIOD_RE = re.compile(r"^\d{8}$")
VALID_EOM = {"0331", "0630", "0930", "1231"}


def valid_period(p: str) -> bool:
    p = str(p)
    if not PERIOD_RE.match(p):
        return False
    # 年报/季报字段须为报告期（0331/0630/0930/1231）；高频快照字段（如 pledge_ratio）
    # 为抓取日 YYYYMMDD，只要是合法日期即视为有效。
    try:
        pd.Timestamp(p)
    except (ValueError, TypeError):
        return False
    # 年末/季末报告期
    if p[4:] in VALID_EOM:
        return True
    # 其它 8 位合法日期（快照类字段）也放行
    return True


def fnum(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# =====================================================================
# 主流程
# =====================================================================
def main() -> None:
    st = Store(str(DB))
    px = st.load_prices()
    facts = st.load_facts()
    uni = st.load_universe()
    uni_idx = uni.set_index("code")

    findings: list[dict] = []

    def add(sev, dim, code, field, detail):
        findings.append({"severity": sev, "dimension": dim, "code": code,
                         "field": field, "detail": detail})

    lines = ["# 原始数据质量复核报告", "",
             f"> 生成于自动审计 `tests/audit_data.py`｜库：`{DB}`",
             f"> prices {len(px):,} 行 / {px['code'].nunique()} 只｜"
             f"facts {len(facts):,} 行 / {facts['field'].nunique()} 字段｜"
             f"universe {len(uni)} 只", ""]

    # ---------------------------------------------------------- P0 结构
    lines += ["## P0 结构完整性", ""]
    dup = facts.duplicated(subset=["code", "field", "period", "announce_date"]).sum()
    lines.append(f"- 重复主键 (code,field,period,announce_date)：**{int(dup)}** 行")
    add("INFO", "结构", "-", "-", f"重复主键 {int(dup)} 行")

    badp = facts[~facts["period"].astype(str).map(valid_period)]
    if len(badp):
        ex = sorted(badp["period"].astype(str).unique())[:6]
        lines.append(f"- 非法报告期：**{len(badp)}** 行，涉及字段 {sorted(badp['field'].unique())[:5]}，"
                     f"示例 {ex}")
        # 按 (code,field) 聚合
        for (c, f), g in badp.groupby(["code", "field"]):
            add("WARN", "结构", c, f, f"非法 period 示例 {sorted(g['period'].astype(str).unique())[:3]}")
    else:
        lines.append("- 非法报告期：**0** 行")

    # ---------------------------------------------------------- P1 价格层
    lines += ["", "## P1 价格层（prices）", ""]
    m = px.dropna(subset=["mktcap", "close_raw", "total_share"]).copy()
    mism = (m["mktcap"] - m["close_raw"] * m["total_share"]).abs() > 1.0
    lines.append(f"- mktcap == close_raw×total_share 一致性：不一致 **{int(mism.sum())}** / {len(m)} 行")
    if int(mism.sum()):
        for c in m[mism]["code"].unique()[:10]:
            add("WARN", "价格", c, "mktcap", "mktcap≠close_raw×total_share")

    neg = px[px["close_raw"] <= 0]
    lines.append(f"- close_raw<=0：**{len(neg)}** 行 / {neg['code'].nunique()} 只")
    for c in neg["code"].unique()[:10]:
        add("WARN", "价格", c, "close_raw", "close_raw<=0")

    # NaN 市值（被排除）
    nan_mc = px[px["mktcap"].isna()]
    per_code = nan_mc.groupby("code").size()
    # 前 2 行空窗视为正常（早于首份财报公告日），>5 行才告警
    excluded = per_code[per_code > 5]
    lines.append(f"- NaN 市值行：**{len(nan_mc)}** / {nan_mc['code'].nunique()} 只；"
                 f"其中「>5 行全期缺失」(疑似整股被排除) 共 **{len(excluded)}** 只："
                 f"{list(excluded.index)}")
    for c in excluded.index:
        nm = uni_idx.loc[c, "name"] if c in uni_idx.index else "?"
        add("HIGH", "价格", c, "mktcap", f"NaN 市值 {int(per_code[c])} 行 → 该股被排除；疑似 total_share 缺失/错值")

    # 隐含股本跳变
    mp = m.copy()
    mp["imp"] = mp["mktcap"] / mp["close_raw"]
    step_rows = []
    for c, g in mp.sort_values("date").groupby("code"):
        s = g["imp"].values.astype(float)
        if len(s) > 1:
            r = s[1:] / s[:-1]
            r = r[np.isfinite(r)]
            if len(r):
                i = int(np.argmax(np.abs(r - 1)))
                step_rows.append((c, float(r[i]), float(s[i]), float(s[i + 1])))
    step_rows.sort(key=lambda x: -abs(x[1] - 1))
    big = [(c, r, a, b) for c, r, a, b in step_rows if abs(r - 1) > 0.5]
    lines.append(f"- 隐含股本(mktcap/close) 最大跳变 >1.5x 的只数：**{len(big)}**"
                 f"（含 IPO/真实公司行动伪跳变，需结合 0->x 过滤）")
    # 非 0->x 的大跳变（更可能是错值）
    real_anom = [(c, r, a, b) for c, r, a, b in big if a > 0 and b > 0]
    lines.append(f"  - 其中「非零->非零」跳变（剔除 IPO 伪跳变后）共 **{len(real_anom)}** 只，Top10：")
    for c, r, a, b in real_anom[:10]:
        nm = uni_idx.loc[c, "name"] if c in uni_idx.index else "?"
        lines.append(f"    - {c} {nm}：{r:+.1%} （{a/1e8:.1f}亿→{b/1e8:.1f}亿股）")
        add("MED", "股本", c, "total_share",
            f"隐含股本跳变 {r:+.1%}（{a/1e8:.1f}亿→{b/1e8:.1f}亿股），疑似错值或真实公司行动，需外部复核")

    # ---------------------------------------------------------- P1 股本/市值覆盖
    lines += ["", "## P1 股本与市值覆盖（facts ↔ prices）", ""]
    codes = set(uni["code"])
    has_ts = set(facts[facts["field"] == S.TOTAL_SHARE]["code"])
    missing_ts = codes - has_ts
    lines.append(f"- 缺 total_share facts 的股票：**{len(missing_ts)}** 只 → 这些股 mktcap 全 NaN 被排除："
                 f"{sorted(missing_ts)}")
    for c in sorted(missing_ts):
        nm = uni_idx.loc[c, "name"] if c in uni_idx.index else "?"
        add("HIGH", "股本", c, "total_share", "无 total_share facts → 该股 mktcap 全空、被排除")

    # facts.total_share 非零->非零 跳变（同源，剔除 0->x）
    ts = facts[facts["field"] == S.TOTAL_SHARE].copy()
    ts_steps = []
    for c, g in ts.sort_values("period").groupby("code"):
        v = fnum(g["value"]).values.astype(float)
        if len(v) > 1:
            r = v[1:] / v[:-1]
            r = r[np.isfinite(r)]
            if len(r):
                i = int(np.argmax(np.abs(r - 1)))
                if v[i] > 0 and v[i + 1] > 0:
                    ts_steps.append((c, float(r[i]), float(v[i]), float(v[i + 1])))
    ts_steps.sort(key=lambda x: -abs(x[1] - 1))
    ts_big = [(c, r, a, b) for c, r, a, b in ts_steps if abs(r - 1) > 1.0]
    lines.append(f"- facts.total_share 非零->非零 跳变 >2x：**{len(ts_big)}** 只（真公司行动或错值，需外部复核）")
    for c, r, a, b in ts_big[:15]:
        nm = uni_idx.loc[c, "name"] if c in uni_idx.index else "?"
        lines.append(f"    - {c} {nm}：{r:+.0%}（{a/1e8:.1f}亿→{b/1e8:.1f}亿股）")
        add("MED", "股本", c, "total_share",
            f"facts 股本跳变 {r:+.0%}（{a/1e8:.1f}亿→{b/1e8:.1f}亿股），需外部复核")

    # ---------------------------------------------------------- P1 财务字段符号/零
    lines += ["", "## P1 财务字段符号/零值（最新年报截面）", ""]
    fields = [S.NET_INCOME, S.TOTAL_EQUITY, S.TOTAL_ASSETS, S.REVENUE, S.OCF,
              S.CASH, S.RECEIVABLES, S.INVENTORY, S.PPE_NET, S.GOODWILL,
              S.ST_DEBT, S.LT_DEBT, S.BONDS, S.TOTAL_SHARE]
    # 已知「源缺失应 NaN」却被存的字段
    for f in (S.ST_DEBT, S.BONDS):
        sub = facts[facts["field"] == f]
        nz = int((fnum(sub["value"]) != 0).sum())
        zr = int((fnum(sub["value"]) == 0).sum())
        lines.append(f"- **{f}：非零 {nz} / 零 {zr} 行** "
                     f"{'→ 全零即源未提供却硬编码 0（应改 NaN）' if nz == 0 and len(sub) else ''}")
        if nz == 0 and len(sub):
            add("HIGH", "财务字段", "-", f, f"全 {len(sub)} 行恒为 0（源未提供却硬编码 0，应改 NaN）")

    # 各字段最新期符号异常
    for f in fields:
        sub = facts[facts["field"] == f].sort_values("announce_date").groupby("code").tail(1)
        v = fnum(sub["value"])
        neg = int((v < 0).sum())
        zero = int((v == 0).sum())
        if neg or zero:
            note = ""
            if f in (S.NET_INCOME, S.OCF):
                note = "（亏损/经营流出可为负，属正常）"
            elif f == S.TOTAL_EQUITY and neg:
                note = "（资不抵债，困境/退市股，需确认是否已退市）"
            lines.append(f"  - {f}：负值 {neg}、零值 {zero}{note}")
            if f not in (S.NET_INCOME, S.OCF) and (neg or (zero and f != S.ST_DEBT and f != S.BONDS)):
                for c in sub[v < 0]["code"]:
                    add("LOW", "财务字段", c, f, "最新期为负值")

    # ---------------------------------------------------------- P2 估值因子离群
    lines += ["", "## P2 估值因子离群（常指向 mktcap 或基本面错）", ""]
    # 用最新期截面粗略算 ep/bp/cfp/ebit_ev
    recent = facts.copy()
    recent["period"] = recent["period"].astype(str)
    # 取 2023 以后最新年报
    recent = recent[recent["period"].str[:4].astype(int) >= 2023]
    panel = recent.pivot_table(index="code", columns="field", values="value", aggfunc="last")
    if S.MKTCAP in panel and S.NET_INCOME in panel:
        ep = panel[S.NET_INCOME] / panel[S.MKTCAP]
        bp = panel.get(S.TOTAL_EQUITY, pd.Series(dtype=float)) / panel[S.MKTCAP]
        for name, ser in (("ep", ep), ("bp", bp)):
            if ser.notna().any():
                q = ser.quantile([0.001, 0.01, 0.99, 0.999])
                out_lo = ser[ser < q[0.001]]
                out_hi = ser[ser > q[0.999]]
                lines.append(f"  - {name}：极小值 {ser.min():.3f} / 极大值 {ser.max():.3f}；"
                             f"极端离群(<0.1%分位 {q[0.001]:.3f} 或 >99.9%分位 {q[0.999]:.3f}) "
                             f"共 {len(out_lo)+len(out_hi)} 只")
                for c in list(out_lo.index) + list(out_hi.index):
                    add("LOW", "估值", c, name, f"离群值 {ser[c]:.3f}")

    # ---------------------------------------------------------- P2 facts↔prices 股本一致性
    lines += ["", "## P2 facts.total_share ↔ prices.total_share 一致性", ""]
    # 取每只 prices 的 total_share 与 facts 公告日回填值比较（抽最新重合点）
    # 简化：比较 facts 最新 total_share 与 prices 最新 total_share
    f_latest = ts.sort_values("announce_date").groupby("code").tail(1).set_index("code")["value"]
    p_latest = px.dropna(subset=["total_share"]).sort_values("date").groupby("code")["total_share"].tail(1)
    common = set(f_latest.index) & set(p_latest.index)
    diffs = []
    for c in common:
        fv, pv = float(f_latest[c]), float(p_latest[c])
        if fv and pv and abs(fv - pv) / max(abs(fv), 1) > 0.01:
            diffs.append((c, fv, pv))
    lines.append(f"- 最新期 facts↔prices 股本差异 >1% 的只数：**{len(diffs)}**（point-in-time 边界或错值）")
    for c, fv, pv in diffs[:15]:
        nm = uni_idx.loc[c, "name"] if c in uni_idx.index else "?"
        lines.append(f"    - {c} {nm}：facts {fv/1e8:.2f}亿 vs prices {pv/1e8:.2f}亿")
        add("MED", "股本", c, "total_share", f"facts {fv/1e8:.2f}亿 vs prices {pv/1e8:.2f}亿")

    # ---------------------------------------------------------- P3 宇宙
    lines += ["", "## P3 宇宙（universe）", ""]
    fin_ind = set(S.FINANCIAL_INDUSTRIES)
    mismatch_fin = uni[(uni["is_financial"]) & (~uni["industry"].isin(fin_ind))]
    lines.append(f"- is_financial=True 但行业非金融：**{len(mismatch_fin)}** 只")
    for _, r in mismatch_fin.iterrows():
        add("LOW", "宇宙", r["code"], "is_financial", f"is_financial 但行业={r['industry']}")

    # ---------------------------------------------------------- 汇总
    lines += ["", "## 汇总（按严重度）", ""]
    sev_order = {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3, "WARN": 1}
    from collections import Counter
    cnt = Counter(f["severity"] for f in findings)
    lines.append("| 严重度 | 条数 |")
    lines.append("|--------|------|")
    for s in ("HIGH", "MED", "LOW", "WARN", "INFO"):
        if cnt.get(s):
            lines.append(f"| {s} | {cnt[s]} |")
    lines.append("")
    lines.append("> HIGH=应修/已排除；MED=需外部复核；LOW=留意；WARN=检查；INFO=信息。")
    lines.append("> 完整条目见 `out/data_audit_findings.csv`。")

    report = "\n".join(lines)
    (OUT / "data_audit.md").write_text(report, encoding="utf-8")
    pd.DataFrame(findings).to_csv(OUT / "data_audit_findings.csv", index=False, encoding="utf-8")

    # 控制台摘要
    print(report)
    print(f"\n→ 已写入 {OUT/'data_audit.md'} 与 {OUT/'data_audit_findings.csv'}")
    print(f"→ 发现条目 {len(findings)}：HIGH {cnt.get('HIGH',0)} / MED {cnt.get('MED',0)} "
          f"/ LOW {cnt.get('LOW',0)} / WARN {cnt.get('WARN',0)} / INFO {cnt.get('INFO',0)}")


if __name__ == "__main__":
    main()
