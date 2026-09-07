"""L7 监控层：论文报警 vs 价格报警，职责分离。

核心纪律（这是本层的全部意义）：
  **价格下跌本身永远不触发卖出，只触发"检查论文是否还成立"。**

大多数投资者的失败不是选错股，而是在论文没破的时候因为价格波动卖掉了对的股票。
所以两类报警走两个完全独立的通道：
  - 论文报警（基本面恶化）→ 强制重估 + 人工复核
  - 价格报警（触及买/卖点）→ 按规则机械执行

另含因子拥挤度监控：因子不是买了躺赢，拥挤 → 失效（A股 2026H1 红利低波就是实例）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


# ==================================================================== 论文报警
def thesis_alerts(
    metrics: pd.DataFrame,
    cfg: Config,
    prev: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """基本面恶化报警。

    prev 传入上一期指标表时，额外检测「跳变」类信号（质押率跳升等）。
    """
    if metrics.empty:
        return pd.DataFrame(columns=["code", "name", "alert", "detail", "value"])

    mcfg = cfg.section("monitor").get("thesis_alerts", {})
    rows = []

    def _flag(row, col, test, name, unit=".3f"):
        v = row.get(col)
        if v is None or pd.isna(v):
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if test(v):
            rows.append({
                "code": row.get("code"), "name": row.get("name"),
                "alert": name, "detail": f"{col}={v:{unit}}", "value": v,
            })

    prev_map = {}
    if prev is not None and not prev.empty:
        prev_map = prev.set_index("code").to_dict("index")

    for _, r in metrics.iterrows():
        _flag(r, "beneish_m", lambda v: v > mcfg.get("beneish_m_score_max", -1.78),
              "M-Score 翻红（财务操纵嫌疑）")
        # Altman Z 软报警同样只在"净借款人"时成立：净现金/低杠杆公司的 Z 会系统性偏低，
        # 并非破产信号（与排雷层硬否决的 net_debt_ebitda>0 前提保持一致）。
        az_min = mcfg.get("altman_z_score_min", 1.8)
        nd = r.get("net_debt_ebitda")
        if (pd.notna(r.get("altman_z")) and r.get("altman_z") < az_min
                and pd.notna(nd) and nd > 0):
            rows.append({
                "code": r.get("code"), "name": r.get("name"),
                "alert": "Z-Score 跌破安全线（破产风险）",
                "detail": f"altman_z={r.get('altman_z'):.3f}", "value": float(r.get("altman_z")),
            })
        _flag(r, "ocf_to_ni_5y", lambda v: v < mcfg.get("ocf_to_ni_min", 0.50),
              "现金流含量恶化（纸面利润）")
        _flag(r, "pledge_ratio", lambda v: v > 0.40, "质押比例偏高")

        p = prev_map.get(r.get("code"))
        if p is not None:
            jump = mcfg.get("pledge_ratio_jump", 0.15)
            pv, cv = p.get("pledge_ratio"), r.get("pledge_ratio")
            if pd.notna(pv) and pd.notna(cv) and (cv - pv) > jump:
                rows.append({
                    "code": r.get("code"), "name": r.get("name"),
                    "alert": "质押率单期跳升",
                    "detail": f"{pv:.1%} → {cv:.1%}", "value": float(cv - pv),
                })
            pa, ca = p.get("audit_nonstd", 0), r.get("audit_nonstd", 0)
            if ca and not pa:
                rows.append({
                    "code": r.get("code"), "name": r.get("name"),
                    "alert": "审计意见由标准转为非标", "detail": "审计意见变化", "value": 1.0,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"审计意见由标准转为非标": 0, "M-Score 翻红（财务操纵嫌疑）": 1,
             "Z-Score 跌破安全线（破产风险）": 2, "质押率单期跳升": 3,
             "现金流含量恶化（纸面利润）": 4, "质押比例偏高": 5}
    return out.sort_values("alert", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)


# ==================================================================== 价格报警
def price_alerts(valued: pd.DataFrame) -> pd.DataFrame:
    """触及买点/卖点的机械信号。仅按规则执行，不做判断。"""
    if valued.empty or "verdict" not in valued.columns:
        return pd.DataFrame()
    out = valued[valued["verdict"].isin(["已到买点", "已到卖点"])].copy()
    cols = ["code", "name", "industry", "mktcap", "buy_point", "sell_point", "verdict"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


# ==================================================================== 拥挤度
def crowding_metrics(store, holdings: list[str], asof, cfg: Config) -> dict:
    """三维度拥挤度：交易端 / 估值端 / 集中度。

    说明：完整的拥挤度监控还需要研报集中度与资金流数据；
    这里实现的是能从行情与估值数据直接算出的部分，其余留接口。
    """
    cfg_c = cfg.section("monitor").get("crowding", {})
    lookback = cfg_c.get("lookback_days", 250)
    prices = store.load_prices()
    if prices.empty or not holdings:
        return {}

    asof = pd.to_datetime(asof)
    px = prices[(prices["date"] <= asof) & (prices["code"].isin(holdings))]
    if px.empty:
        return {}

    # 交易端：近20日成交额 / 近250日成交额
    recent = px.sort_values("date").groupby("code")["amount"].apply(lambda s: s.tail(20).mean())
    base = px.sort_values("date").groupby("code")["amount"].apply(lambda s: s.tail(lookback).mean())
    ratio = (recent / base.replace(0, np.nan)).dropna()
    turnover_ratio = float(ratio.median()) if len(ratio) else np.nan

    # 集中度：前20%持仓贡献的成交额占比
    amt = px.groupby("code")["amount"].mean().sort_values(ascending=False)
    k = max(1, int(len(amt) * 0.2))
    concentration = float(amt.head(k).sum() / amt.sum()) if amt.sum() > 0 else np.nan

    res = {
        "turnover_ratio_20d_250d": turnover_ratio,
        "amount_concentration_top20pct": concentration,
        "n_holdings": len(holdings),
    }

    low_t = cfg_c.get("turnover_percentile_low", 30) / 100.0
    if np.isfinite(turnover_ratio):
        res["turnover_alert"] = turnover_ratio < low_t
    if np.isfinite(concentration):
        res["concentration_alert"] = concentration > 0.45
    return res


def weekly_report(thesis: pd.DataFrame, price: pd.DataFrame,
                  crowding: dict | None = None) -> str:
    lines = ["# 周报（L7）", ""]
    lines += ["## 论文报警（需人工复核）", ""]
    if thesis is None or thesis.empty:
        lines += ["无。", ""]
    else:
        lines += ["| 代码 | 名称 | 报警 | 详情 |", "|------|------|------|------|"]
        for _, r in thesis.iterrows():
            lines.append(f"| {r['code']} | {r['name']} | {r['alert']} | {r['detail']} |")
        lines.append("")
    lines += ["## 价格报警（机械执行）", ""]
    if price is None or price.empty:
        lines += ["无。", ""]
    else:
        lines += ["| 代码 | 名称 | 判定 |", "|------|------|------|"]
        for _, r in price.iterrows():
            lines.append(f"| {r['code']} | {r['name']} | {r['verdict']} |")
        lines.append("")
    if crowding:
        lines += ["## 拥挤度", "", "| 指标 | 值 | 报警 |", "|------|------|------|"]
        _alert_key = {
            "turnover_ratio_20d_250d": "turnover_alert",
            "amount_concentration_top20pct": "concentration_alert",
        }
        for k, v in crowding.items():
            if k.endswith("_alert"):
                continue
            val = f"{v:.3f}" if isinstance(v, float) else str(v)
            alert = "⚠️" if crowding.get(_alert_key.get(k, ""), False) else ""
            lines.append(f"| {k} | {val} | {alert} |")
        lines.append("")
    return "\n".join(lines)
