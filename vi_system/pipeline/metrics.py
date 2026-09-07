"""原始指标计算层（L3/L4 的公共底座）。

产出一张"每只股票一行"的指标表，供排雷层与因子层消费。

设计原则：
  - **只用年报**（period 以 1231 结尾）。季报与年报混用会让五年中位数失真。
  - **市值用决策日的**（market 表），不用财报里的历史市值 —— 估值永远相对当下价格。
  - **缺失即 NaN**，不做任何填补。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import *  # noqa: F401,F403
from ..data.schema import (
    REVENUE, COGS, SGA, EBIT, INTEREST_EXPENSE, NET_INCOME, TAX, DEPRECIATION,
    TOTAL_ASSETS, CURRENT_ASSETS, CURRENT_LIAB, TOTAL_LIAB, TOTAL_EQUITY,
    CASH, RECEIVABLES, INVENTORY, PPE_NET, GOODWILL, ST_DEBT, LT_DEBT, BONDS,
    RETAINED_EARNINGS, TOTAL_SHARE, OCF, CAPEX, DIVIDEND_PAID, DIVIDEND,
    EQUITY_ISSUED, BUYBACK, MKTCAP, PLEDGE_RATIO, AUDIT_OPINION,
    NPL_RATIO, PROVISION_COVERAGE, CET1, CASH_SHORT_DEBT,
)

TAX_RATE = 0.25


def _d(a, b):
    """安全除法：分母为 0/NaN 时返回 NaN（绝不返回 inf 或 0）。"""
    if a is None or b is None:
        return np.nan
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def _v(row, field, default=np.nan):
    """从一行里取字段，缺失返回 NaN。"""
    if field not in row.index:
        return default
    x = row[field]
    if x is None:
        return default
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


# ==================================================================== 单组计算
def _beneish_m(cur: pd.Series, prev: pd.Series) -> float:
    """Beneish M-Score（8 变量）。值越大越可疑，> -1.78 视为预警。"""
    ar_t, s_t = _v(cur, RECEIVABLES), _v(cur, REVENUE)
    ar_p, s_p = _v(prev, RECEIVABLES), _v(prev, REVENUE)
    dsri = _d(_d(ar_t, s_t), _d(ar_p, s_p))

    gm_t = _d(s_t - _v(cur, COGS), s_t)
    gm_p = _d(s_p - _v(prev, COGS), s_p)
    gmi = _d(gm_p, gm_t)

    ta_t, ta_p = _v(cur, TOTAL_ASSETS), _v(prev, TOTAL_ASSETS)
    aq_t = 1 - _d(_v(cur, CURRENT_ASSETS) + _v(cur, PPE_NET), ta_t)
    aq_p = 1 - _d(_v(prev, CURRENT_ASSETS) + _v(prev, PPE_NET), ta_p)
    aqi = _d(aq_t, aq_p)

    sgi = _d(s_t, s_p)

    dep_t, dep_p = _v(cur, DEPRECIATION), _v(prev, DEPRECIATION)
    ppe_t, ppe_p = _v(cur, PPE_NET), _v(prev, PPE_NET)
    dr_t = _d(dep_t, dep_t + ppe_t)
    dr_p = _d(dep_p, dep_p + ppe_p)
    depi = _d(dr_p, dr_t)
    # 腾讯接口不提供折旧行项 → depi 无法计算。取中性值 1.0（既不偏多也不偏空），
    # 避免整个 Beneish 舞弊屏因单字段缺失而整体失效。仅影响 0.115 权重那一维。
    if not np.isfinite(depi):
        depi = 1.0

    sgai = _d(_d(_v(cur, SGA), s_t), _d(_v(prev, SGA), s_p))

    tata = _d(_v(cur, NET_INCOME) - _v(cur, OCF), ta_t)

    lev_t = _d(_v(cur, CURRENT_LIAB) + _v(cur, LT_DEBT), ta_t)
    lev_p = _d(_v(prev, CURRENT_LIAB) + _v(prev, LT_DEBT), ta_p)
    lvgi = _d(lev_t, lev_p)

    coefs = [(0.920, dsri), (0.528, gmi), (0.404, aqi), (0.892, sgi),
             (0.115, depi), (-0.172, sgai), (4.679, tata), (-0.327, lvgi)]
    # 除 depi 外，其余 7 维任一分量为非有限值则无法计算（数据缺失）
    if any(not np.isfinite(v) for c, v in coefs if abs(c - 0.115) > 1e-9):
        return np.nan
    return -4.84 + sum(c * v for c, v in coefs)


def _altman_z(cur: pd.Series, mktcap: float) -> float:
    """Altman Z-Score。Z < 1.8 为破产风险区（金融股不适用）。"""
    ta = _v(cur, TOTAL_ASSETS)
    a = _d(_v(cur, CURRENT_ASSETS) - _v(cur, CURRENT_LIAB), ta)
    b = _d(_v(cur, RETAINED_EARNINGS), ta)
    c = _d(_v(cur, EBIT), ta)
    d = _d(mktcap, _v(cur, TOTAL_LIAB))
    e = _d(_v(cur, REVENUE), ta)
    if any(not np.isfinite(x) for x in (a, b, c, d, e)):
        return np.nan
    return 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e


def _metrics_for_code(g: pd.DataFrame, mktcap: float) -> dict:
    """对单只股票的年报序列计算全部指标。"""
    # 先留一份全量：质押等「时点快照」数据的 period 是时间戳（如 202609040101）
    # 而非年报，会被下面的年报过滤丢弃，需单独回退取用（见 pledge_ratio）。
    full = g.sort_values("period")
    g = full[full["period"].astype(str).str.endswith("1231")]
    if g.empty:
        return {}
    cur = g.iloc[-1]
    prev = g.iloc[-2] if len(g) > 1 else None
    hist5 = g.tail(5)

    ta = _v(cur, TOTAL_ASSETS)
    eq = _v(cur, TOTAL_EQUITY)
    ni = _v(cur, NET_INCOME)
    rev = _v(cur, REVENUE)
    ocf = _v(cur, OCF)
    debt = (_v(cur, ST_DEBT, 0.0) or 0.0) + (_v(cur, LT_DEBT, 0.0) or 0.0) + (_v(cur, BONDS, 0.0) or 0.0)
    cash = _v(cur, CASH, 0.0) or 0.0
    ebit = _v(cur, EBIT)

    # 质押快照：period 为时间戳而非年报，年报行取不到 → 从全量里取最新一条
    if PLEDGE_RATIO in full.columns:
        _pl = full[full[PLEDGE_RATIO].notna()]
        pledge_val = _v(_pl.iloc[-1], PLEDGE_RATIO) if not _pl.empty else np.nan
    else:
        pledge_val = np.nan

    # -------------------------------------------------- 便宜（Value）
    ev = mktcap + (_v(cur, TOTAL_LIAB, 0.0) or 0.0) - cash
    div = _v(cur, DIVIDEND_PAID, np.nan)
    if not np.isfinite(div):
        div = _v(cur, DIVIDEND, np.nan)

    m = {
        "ep": _d(ni, mktcap),
        "bp": _d(eq, mktcap),
        "cfp": _d(ocf, mktcap),
        "ebit_ev": _d(ebit, ev),
        "dividend_yield": _d(div, mktcap),

        # -------------------------------------------------- 质量（Quality）
        "gpa": _d(rev - _v(cur, COGS), ta),
        "roic": _d(ebit * (1 - TAX_RATE), eq + debt - cash) if np.isfinite(ebit) else np.nan,
        "accruals": _d(ni - ocf, ta),
        "net_payout": _d(
            (_v(cur, DIVIDEND_PAID, 0.0) or 0.0) + (_v(cur, BUYBACK, 0.0) or 0.0)
            - (_v(cur, EQUITY_ISSUED, 0.0) or 0.0), mktcap),

        # -------------------------------------------------- 安全（Safety）
        # 利息覆盖倍数：腾讯把"财务费用"整体给出，现金充裕公司该值为负（净利息收入）。
        # 此时公司没有利息负担，覆盖倍数应视为极高而非负数 —— 否则会被杠杆否决误杀。
        "interest_coverage": (
            99.0 if (not np.isfinite(_v(cur, INTEREST_EXPENSE)) or _v(cur, INTEREST_EXPENSE) <= 0)
            else _d(ebit, _v(cur, INTEREST_EXPENSE))
        ),
        "net_debt_ebitda": _d(debt - cash, ebit + _v(cur, DEPRECIATION, 0.0)),
        "earnings_volatility": _d(hist5[NET_INCOME].std(), abs(hist5[NET_INCOME].mean()))
        if NET_INCOME in hist5 else np.nan,
        "revenue_volatility": _d(hist5[REVENUE].std(), abs(hist5[REVENUE].mean()))
        if REVENUE in hist5 else np.nan,

        # -------------------------------------------------- 排雷用
        "ocf_to_ni_5y": _d(hist5[OCF].sum(), hist5[NET_INCOME].sum())
        if {OCF, NET_INCOME}.issubset(hist5.columns) else np.nan,
        "goodwill_to_equity": _d(_v(cur, GOODWILL), eq),
        # 质押为高频快照，年报行里恒为 NaN → 回退取全量内的最新快照。
        # 其 announce_date 是抓取时刻，早期回测时点会被 facts_asof 正常过滤掉，不产生前视。
        "pledge_ratio": pledge_val,
        "audit_nonstd": float(_v(cur, AUDIT_OPINION, 0.0) or 0.0),
        "audit_nonstd_3y": float(g.tail(3)[AUDIT_OPINION].max())
        if AUDIT_OPINION in g.columns and g.tail(3)[AUDIT_OPINION].notna().any() else 0.0,
        "altman_z": _altman_z(cur, mktcap),
        "beneish_m": _beneish_m(cur, prev) if prev is not None else np.nan,

        # -------------------------------------------------- 行业特化
        "npl_ratio": _v(cur, NPL_RATIO),
        "provision_coverage": _v(cur, PROVISION_COVERAGE),
        "cet1": _v(cur, CET1),
        "cash_short_debt": _v(cur, CASH_SHORT_DEBT),

        # -------------------------------------------------- 元信息
        "period": cur["period"],
        "announce_date": cur.get("announce_date"),
        "n_years": len(g),
        "mktcap": mktcap,
        "net_income": ni,
        "ocf": ocf,
        "capex": _v(cur, CAPEX),
        "revenue": rev,
        "total_share": _v(cur, TOTAL_SHARE),
    }

    # 盈利能力 5 年变化（QMJ 的 Growth 块）
    gpa_now = m["gpa"]
    if len(g) >= 5:
        gpa_old = _d(_v(g.iloc[-5], REVENUE) - _v(g.iloc[-5], COGS), _v(g.iloc[-5], TOTAL_ASSETS))
        ratio = (gpa_now / gpa_old) if (np.isfinite(gpa_old) and gpa_old != 0) else np.nan
        # 关键：gpa_old 为负时 ratio 可能为负，而「负数的 0.2 次幂」在 Python 里返回
        # **复数**，会让整列变成 complex128，后续 groupby.rank 直接崩
        # （真实数据里有亏损企业才触发，合成数据从未暴露过）。
        # 比值 <= 0 时几何增长率无定义 → 置 NaN（打分时按中性处理）。
        m["profit_growth_5y"] = float(ratio ** 0.2 - 1) if (np.isfinite(ratio) and ratio > 0) else np.nan
    else:
        m["profit_growth_5y"] = np.nan

    # 5 年股本膨胀
    if len(g) >= 5 and TOTAL_SHARE in g.columns:
        sh_now = _v(cur, TOTAL_SHARE)
        sh_old = _v(g.iloc[-5], TOTAL_SHARE)
        m["share_dilution_5y"] = _d(sh_now, sh_old) - 1 if np.isfinite(sh_now) and np.isfinite(sh_old) else np.nan
    else:
        m["share_dilution_5y"] = np.nan

    return m


# ==================================================================== 主入口
def compute_metrics(
    panel: pd.DataFrame, market: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    """计算全宇宙指标表。

    panel   : store.facts_asof(asof) 的财务宽表（多期）
    market  : store.market_asof(asof) 的行情快照
    universe: 宇宙层输出
    """
    if panel.empty or market.empty or universe.empty:
        return pd.DataFrame()

    codes = set(universe["code"])
    p = panel[panel["code"].isin(codes)].copy()
    if p.empty:
        return pd.DataFrame()

    mkt = market.set_index("code")["mktcap"].to_dict()
    rows = []
    for code, g in p.groupby("code", sort=False):
        mc = mkt.get(code, np.nan)
        if not np.isfinite(mc) or mc <= 0:
            continue
        m = _metrics_for_code(g, float(mc))
        if not m:
            continue
        m["code"] = code
        rows.append(m)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("code")

    meta = universe.set_index("code")[["name", "industry", "avg_amount_60d",
                                       "listing_years", "is_financial"]]
    out = meta.join(out, how="inner")
    return out.reset_index().rename(columns={"index": "code"})
