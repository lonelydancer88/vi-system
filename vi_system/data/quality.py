"""数据质量闸门。

三条铁律：
  1. 缺失即缺失 —— 标 NaN，**禁止插值、禁止用行业均值补、禁止凭印象填**
  2. 重述留痕 —— 同一 (code, field, period) 的多个版本全部保留，回测用「当时可见的最新版」
  3. 异常可解释 —— 报表恒等式（资产=负债+权益）偏差过大的期次打标，不静默通过
"""

from __future__ import annotations

import pandas as pd

from .schema import TOTAL_ASSETS, TOTAL_LIAB, TOTAL_EQUITY, REQUIRED_FOR_FACTORS


def restatement_report(facts: pd.DataFrame) -> pd.DataFrame:
    """列出存在重述（同一 code/field/period 多个公告版本）的记录。"""
    if facts.empty:
        return pd.DataFrame(columns=["code", "field", "period", "n_versions", "value_range"])
    g = facts.groupby(["code", "field", "period"])["value"]
    rep = g.agg(n_versions="count", vmin="min", vmax="max").reset_index()
    rep = rep[rep["n_versions"] > 1].copy()
    if rep.empty:
        return rep
    rep["value_range"] = rep["vmax"] - rep["vmin"]
    return rep.drop(columns=["vmin", "vmax"]).sort_values("n_versions", ascending=False)


def coverage_report(panel: pd.DataFrame, fields: list[str] | None = None) -> pd.DataFrame:
    """各字段在 panel 中的覆盖率。覆盖率低说明数据源有问题，先看这个再谈因子。"""
    if panel.empty:
        return pd.DataFrame(columns=["field", "coverage", "n"])
    fields = fields or [c for c in panel.columns]
    rows = []
    n = len(panel)
    for f in fields:
        if f not in panel.columns:
            rows.append({"field": f, "coverage": 0.0, "n": 0})
        else:
            rows.append({"field": f, "coverage": float(panel[f].notna().mean()), "n": int(panel[f].notna().sum())})
    out = pd.DataFrame(rows).sort_values("coverage")
    out["total"] = n
    return out


def balance_sheet_check(panel: pd.DataFrame, tol: float = 0.02) -> pd.DataFrame:
    """资产 = 负债 + 所有者权益 恒等式检查，返回超差的期次。"""
    need = [TOTAL_ASSETS, TOTAL_LIAB, TOTAL_EQUITY]
    if panel.empty or not all(c in panel.columns for c in need):
        return pd.DataFrame()
    sub = panel.dropna(subset=need).copy()
    if sub.empty:
        return sub
    lhs = sub[TOTAL_ASSETS]
    rhs = sub[TOTAL_LIAB] + sub[TOTAL_EQUITY]
    sub["gap_pct"] = (lhs - rhs).abs() / lhs.abs().clip(lower=1)
    bad = sub[sub["gap_pct"] > tol]
    return bad[["code", "period", TOTAL_ASSETS, TOTAL_LIAB, TOTAL_EQUITY, "gap_pct"]]


def mark_missing(panel: pd.DataFrame, required: list[str] | None = None) -> pd.DataFrame:
    """给因子计算所需字段不全的期次打标 _complete=False。

    注意：这里只做标记，不做填补。缺失的标的该期不参与打分，
    而不是用一个漂亮的假数字混进去。
    """
    required = required or REQUIRED_FOR_FACTORS
    p = panel.copy()
    have = [c for c in required if c in p.columns]
    if not have:
        p["_complete"] = False
        return p
    p["_complete"] = p[have].notna().all(axis=1)
    return p


def staleness_report(panel: pd.DataFrame, asof: pd.Timestamp, max_days: int = 400) -> pd.DataFrame:
    """数据新鲜度：只看每只股票**最新**报告期的公告日至决策日的间隔。

    历史期次天然"过期"，把它们算进来只会制造噪音。
    """
    if panel.empty or "announce_date" not in panel.columns:
        return pd.DataFrame()
    p = panel.copy()
    p = p.sort_values("period").groupby("code", as_index=False).tail(1)
    p["days_since_announce"] = (pd.Timestamp(asof) - pd.to_datetime(p["announce_date"])).dt.days
    stale = p[p["days_since_announce"] > max_days]
    return stale[["code", "period", "announce_date", "days_since_announce"]]


def run_all_checks(store, asof) -> dict:
    """一次性跑完所有质量检查，返回报告字典。"""
    facts = store.load_facts()
    panel = store.facts_asof(asof)
    return {
        "summary": store.summary(),
        "restatements": restatement_report(facts),
        "coverage": coverage_report(panel),
        "balance_sheet_violations": balance_sheet_check(panel),
        "stale_periods": staleness_report(panel, asof),
    }
