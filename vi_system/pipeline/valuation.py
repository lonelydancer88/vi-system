"""L5 估值层：从"排序"到"可买"。

核心方法不是"算目标价"，而是**逆向 DCF**：
解出当前股价隐含的永续增长率 g*，回答"市场现在假设了什么"。
    P = FCF0 × (1+g) / (r - g)   →   g* = (P × r - FCF0) / (P + FCF0)

为什么用 g* 而不是目标价：
  目标价依赖"我认为未来会怎样"，g* 只依赖公开价格与一个折现率假设。
  "市场假设了什么"远比"应该值多少"稳健 —— 这是估值层最重要的方法论选择。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def implied_growth(price: float, fcf0: float, r: float) -> float:
    """Gordon 模型下求解市场隐含的永续增长率。

    price : 当前市值（或股价，与 fcf0 同口径）
    fcf0  : 当期自由现金流
    r     : 折现率
    """
    if not all(np.isfinite(x) for x in (price, fcf0, r)):
        return np.nan
    if fcf0 <= 0 or price <= 0:
        return np.nan
    g = (price * r - fcf0) / (price + fcf0)
    if not np.isfinite(g) or g >= r:
        return np.nan
    return float(g)


def dcf_value(fcf0: float, growth: float, years: int,
              terminal: float, discount: float) -> float:
    """两阶段 DCF：前 N 年按 growth 增长，之后永续 terminal。"""
    if not np.isfinite(fcf0) or fcf0 <= 0:
        return np.nan
    if not np.isfinite(discount) or discount <= terminal:
        return np.nan
    pv = 0.0
    f = fcf0
    for t in range(1, years + 1):
        f *= (1 + growth)
        pv += f / (1 + discount) ** t
    tv = f * (1 + terminal) / (discount - terminal)
    pv += tv / (1 + discount) ** years
    return float(pv)


def interpret_g(g: float) -> str:
    if not np.isfinite(g):
        return "无法计算（自由现金流为负）"
    if g < 0.02:
        return "市场假设近乎零增长 → 极度悲观，安全边际大概率充足"
    if g < 0.06:
        return "市场假设温和增长 → 合理区间"
    if g < 0.12:
        return "市场假设较高增长 → 需要业绩兑现"
    return "市场假设极高增长 → 你在为完美定价，危险"


def valuate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """对因子层输出的候选表做估值。

    新增列：fcf0, fcf_basis, g_implied, v_bull, v_base, v_bear, v_mid,
            buy_point, sell_point, downside_bear, verdict
    """
    if df.empty:
        return df
    vcfg = cfg.section("valuation")
    r = vcfg.get("discount_rate", 0.10)
    years = int(vcfg.get("forecast_years", 5))
    buy_disc = vcfg.get("buy_discount", 0.70)
    sell_mult = vcfg.get("sell_multiple", 1.50)
    max_down = vcfg.get("max_downside_bear", 0.25)
    sc = vcfg.get("scenarios", {})

    out = df.copy()
    fcf0 = out["ocf"] - out["capex"].fillna(0) if {"ocf", "capex"}.issubset(out.columns) else pd.Series(np.nan, index=out.index)
    # 现金流缺失时用净利润的 80% 兜底，并标注
    fallback = fcf0.isna() | (fcf0 <= 0)

    # OCF 口径失真行业（银行 / 房地产 / 非银金融）：经营现金流里塞着客户存款、预售款
    # （合同负债）等递延项，不是可自由支配的现金流。用它当永续 FCF 折现会系统性高估估值
    # —— 实证：滨江集团 OCF 102 亿 / 净利仅 18 亿，FCF 达净利 6.3 倍，v_mid 给到市值 7.8 倍。
    # 这些行业强制改用净利润口径，并以 fcf_basis=ni_forced 标注，便于下游识别。
    ni_ratio = float(vcfg.get("fcf_ni_ratio", 0.8))
    ni_inds = list(vcfg.get("fcf_ni_industries", []) or [])
    forced = (out["industry"].isin(ni_inds)
              if ("industry" in out.columns and ni_inds)
              else pd.Series(False, index=out.index))

    if "net_income" in out.columns:
        fcf0 = fcf0.where(~fallback, out["net_income"] * ni_ratio)
        fcf0 = fcf0.where(~forced, out["net_income"] * ni_ratio)
    out["fcf0"] = fcf0
    out["fcf_is_fallback"] = fallback | forced
    out["fcf_basis"] = np.where(forced, "ni_forced",
                                np.where(fallback, "ni_fallback", "ocf"))

    mkt = out["mktcap"]
    out["g_implied"] = [
        implied_growth(p, f, r) for p, f in zip(mkt, fcf0)
    ]

    def _v(key):
        s = sc.get(key, {})
        return [
            dcf_value(f, s.get("growth", 0.08), years,
                      s.get("terminal", 0.03), s.get("discount", r))
            for f in fcf0
        ]

    out["v_bull"] = _v("bull")
    out["v_base"] = _v("base")
    out["v_bear"] = _v("bear")
    out["v_mid"] = out[["v_bull", "v_base", "v_bear"]].median(axis=1)

    out["buy_point"] = out["v_mid"] * buy_disc
    out["sell_point"] = out["v_mid"] * sell_mult
    out["downside_bear"] = (out["v_bear"] - mkt) / mkt

    def _verdict(row):
        if not np.isfinite(row.get("v_mid", np.nan)):
            return "无法估值"
        if row["mktcap"] <= row["buy_point"]:
            return "已到买点"
        if row["mktcap"] >= row["sell_point"]:
            return "已到卖点"
        return "合理区间"

    out["verdict"] = out.apply(_verdict, axis=1)
    out["downside_ok"] = out["downside_bear"] > -max_down
    out["g_comment"] = out["g_implied"].apply(interpret_g)
    return out


def valuation_report(df: pd.DataFrame, top: int = 25) -> str:
    if df.empty:
        return "# 估值报告\n\n无数据"
    cols = ["rank", "code", "name", "industry", "mktcap", "g_implied",
            "v_mid", "buy_point", "sell_point", "downside_bear", "verdict"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].head(top).copy()
    for c in ["mktcap", "v_mid", "buy_point", "sell_point"]:
        if c in sub.columns:
            sub[c] = (sub[c] / 1e8).round(2)
    for c in ["g_implied", "downside_bear"]:
        if c in sub.columns:
            sub[c] = (sub[c] * 100).round(2)
    lines = ["# 估值报告（L5）", "",
             "- 市值/估值单位：亿元；g* 与下行空间：%", "",
             sub.to_markdown(index=False)]
    return "\n".join(lines)
