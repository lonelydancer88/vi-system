"""合成数据生成器。

用途：在没有 tushare token 的情况下，端到端跑通全流程并做冒烟测试。
**生成的数据仅供流程验证，绝不用于任何真实结论。**

设计要点：
  - 财报数据带真实感的时间滞后（年报次年 4 月披露）
  - 价格对内在价值均值回归，使"便宜+高质量"在统计上确实有正超额
    （否则回测跑出来的是纯噪音，冒烟测试失去意义）
  - 故意掺入"坏公司"：高应计、负现金流、高质押、非标审计意见
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import *  # noqa: F401,F403  (字段常量)
from .schema import (
    MKTCAP, CLOSE, AUDIT_OPINION, PLEDGE_RATIO,
    NUMERIC_FIELDS, AUDIT_STANDARD, AUDIT_NONSTANDARD_MARK,
)

INDUSTRIES = [
    "白酒", "家电", "化学制品", "医疗器械", "银行", "证券",
]
FINANCIAL = ["银行", "证券"]


@dataclass
class GenConfig:
    n_stocks: int = 150
    fund_start_year: int = 2007
    fund_end_year: int = 2026
    price_start: str = "2010-01-01"
    price_end: str = "2026-12-31"
    steps_per_year: int = 250
    seed: int = 42
    bad_company_ratio: float = 0.12


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def generate(cfg: GenConfig | None = None):
    """返回 (universe_df, facts_df, prices_df)。"""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)

    years = list(range(cfg.fund_start_year, cfg.fund_end_year + 1))
    n_years = len(years)

    # ---------------------------------------------------------- 公司属性
    codes = [f"{600000 + i}.SH" if i % 2 == 0 else f"{1 + i:06d}.SZ"
             for i in range(cfg.n_stocks)]
    industry = rng.choice(INDUSTRIES, size=cfg.n_stocks)
    quality = rng.normal(0, 1, size=cfg.n_stocks)               # 质量潜变量
    growth_mu = rng.normal(0.05, 0.07, size=cfg.n_stocks)
    margin_base = np.clip(0.04 + 0.05 * _sigmoid(quality) + rng.normal(0, 0.02, cfg.n_stocks), 0.005, 0.35)
    debt_ratio = np.clip(0.30 + rng.normal(0, 0.12, cfg.n_stocks), 0.02, 0.75)
    payout = np.clip(0.30 + 0.15 * _sigmoid(quality) + rng.normal(0, 0.08, cfg.n_stocks), 0.0, 0.8)
    shares = rng.uniform(1e8, 2e9, size=cfg.n_stocks)
    is_bad = rng.random(cfg.n_stocks) < cfg.bad_company_ratio
    list_year = rng.choice([1996, 2000, 2004, 2008, 2012, 2015, 2019, 2023], size=cfg.n_stocks)

    # ---------------------------------------------------------- 逐年基本面
    rev = np.zeros((cfg.n_stocks, n_years))
    ni = np.zeros_like(rev)
    ta = np.zeros_like(rev)
    eq = np.zeros_like(rev)
    ocf = np.zeros_like(rev)
    debt = np.zeros_like(rev)
    gd = np.zeros_like(rev)
    revenue0 = rng.uniform(2e8, 2e10, size=cfg.n_stocks)

    for t in range(n_years):
        if t == 0:
            rev[:, t] = revenue0
        else:
            g = growth_mu + rng.normal(0, 0.10, cfg.n_stocks)
            rev[:, t] = np.maximum(rev[:, t - 1] * (1 + g), 1e7)
        m = margin_base * (1 + rng.normal(0, 0.18, cfg.n_stocks))
        ni[:, t] = rev[:, t] * np.clip(m, -0.2, 0.45)
        ta[:, t] = rev[:, t] * rng.uniform(0.8, 2.5, size=cfg.n_stocks)
        debt[:, t] = ta[:, t] * debt_ratio * (1 + rng.normal(0, 0.15, cfg.n_stocks))
        if t == 0:
            eq[:, t] = ta[:, t] - debt[:, t]
        else:
            eq[:, t] = np.maximum(eq[:, t - 1] + ni[:, t] * (1 - payout), ta[:, t] * 0.05)
        # 经营现金流：质量越高，现金含量越高；坏公司有大量应计
        cash_conv = np.clip(0.80 + 0.35 * _sigmoid(quality) + rng.normal(0, 0.18, cfg.n_stocks), -0.3, 1.9)
        cash_conv = np.where(is_bad, cash_conv - 0.75, cash_conv)
        ocf[:, t] = ni[:, t] * cash_conv
        gd[:, t] = np.where(is_bad, ta[:, t] * rng.uniform(0.05, 0.45), ta[:, t] * rng.uniform(0, 0.08))

    # ---------------------------------------------------------- 年度内在价值 → 年价格路径
    # 内在价值：盈利 × (10~18倍，质量越高倍数越高)；换算成**每股**价
    mult = 10 + 8 * _sigmoid(quality)
    intrinsic = np.maximum(ni, 1e6) * mult[:, None]
    intrinsic = np.maximum(intrinsic, ta * 0.4)
    intrinsic_ps = intrinsic / shares[:, None]          # 每股内在价值

    p0 = intrinsic_ps[:, 0] * rng.uniform(0.5, 1.6, size=cfg.n_stocks)
    logp_year = np.zeros((cfg.n_stocks, n_years))
    logp_year[:, 0] = np.log(np.maximum(p0, 0.1))
    for t in range(1, n_years):
        gap = np.log(np.maximum(intrinsic_ps[:, t - 1], 1e-2)) - logp_year[:, t - 1]
        # 均值回归 + 漂移 + 噪音
        logp_year[:, t] = (
            logp_year[:, t - 1] + 0.35 * np.clip(gap, -1.5, 1.5)
            + 0.02 + rng.normal(0, 0.22, cfg.n_stocks)
        )

    # ---------------------------------------------------------- 展开成日频
    dates = pd.bdate_range(cfg.price_start, cfg.price_end)
    n_days = len(dates)
    year_idx = np.clip(
        ((dates.year - cfg.fund_start_year) * 1.0).to_numpy(), 0, n_years - 1
    )
    f_lo = np.floor(year_idx).astype(int)
    f_hi = np.minimum(f_lo + 1, n_years - 1)
    frac = year_idx - f_lo

    price_rows = []
    sigma_d = 0.020
    for i in range(cfg.n_stocks):
        base = logp_year[i, f_lo] * (1 - frac) + logp_year[i, f_hi] * frac
        noise = np.concatenate([[0.0], np.cumsum(rng.normal(0, sigma_d, n_days - 1))])
        noise = noise - noise * np.linspace(0, 1, n_days) * 0.0  # 保留随机游走成分
        logp = base + noise
        # 上市前无行情
        listed = dates.year >= list_year[i]
        price = np.where(listed, np.exp(logp), np.nan)
        price = np.maximum(price, 0.5)

        turnover = rng.uniform(0.004, 0.03)
        volume = shares[i] * turnover * rng.uniform(0.5, 1.8, n_days)
        volume = np.where(listed, volume, np.nan)

        div_yield = np.clip((ni[i, f_lo] * payout[i]) / (price * shares[i]), 0, 0.12)
        adj = np.cumprod(1 + np.nan_to_num(div_yield) / 250)

        df = pd.DataFrame({
            "code": codes[i],
            "date": dates,
            "close_raw": price,
            "close_adj": price * adj,
            "volume": volume,
            "amount": volume * price,
            "total_share": shares[i],
            "mktcap": price * shares[i],
        }).dropna(subset=["close_raw"])
        price_rows.append(df)
    prices_df = pd.concat(price_rows, ignore_index=True)

    # ---------------------------------------------------------- 财务事实表（长表）
    fact_rows = []
    for t, y in enumerate(years):
        # 年报次年 4 月随机日披露
        ann = pd.Timestamp(year=y + 1, month=4, day=int(rng.integers(1, 29)))
        for i in range(cfg.n_stocks):
            if list_year[i] > y:
                continue
            px = np.exp(logp_year[i, t])
            vals = {
                REVENUE: rev[i, t],
                COGS: rev[i, t] * (1 - np.clip(margin_base[i] * 1.6, 0.05, 0.9)),
                SGA: rev[i, t] * rng.uniform(0.05, 0.18),
                EBIT: ni[i, t] * rng.uniform(1.1, 1.4),
                INTEREST_EXPENSE: np.maximum(debt[i, t] * 0.045, 1e5),
                NET_INCOME: ni[i, t],
                NET_INCOME_TOTAL: ni[i, t] * 1.02,
                TAX: np.maximum(ni[i, t] * 0.2, 0),
                DEPRECIATION: ta[i, t] * 0.04,
                TOTAL_ASSETS: ta[i, t],
                CURRENT_ASSETS: ta[i, t] * 0.45,
                CURRENT_LIAB: ta[i, t] * 0.30,
                TOTAL_LIAB: debt[i, t] + ta[i, t] * 0.12,
                TOTAL_EQUITY: eq[i, t],
                CASH: ta[i, t] * rng.uniform(0.05, 0.25),
                RECEIVABLES: rev[i, t] * rng.uniform(0.15, 0.45),
                INVENTORY: rev[i, t] * rng.uniform(0.10, 0.35),
                PPE_NET: ta[i, t] * rng.uniform(0.20, 0.50),
                GOODWILL: gd[i, t],
                ST_DEBT: debt[i, t] * 0.4,
                LT_DEBT: debt[i, t] * 0.5,
                BONDS: debt[i, t] * 0.1,
                RETAINED_EARNINGS: eq[i, t] * 0.5,
                TOTAL_SHARE: shares[i],
                OCF: ocf[i, t],
                CAPEX: ta[i, t] * rng.uniform(0.02, 0.10),
                DIVIDEND_PAID: np.maximum(ni[i, t] * payout[i], 0),
                DIVIDEND: np.maximum(ni[i, t] * payout[i], 0),
                EQUITY_ISSUED: ta[i, t] * rng.uniform(0, 0.03),
                BUYBACK: 0.0,
                MKTCAP: px * shares[i],
                CLOSE: px,
                PLEDGE_RATIO: float(rng.uniform(0.4, 0.8) if is_bad[i] else rng.uniform(0, 0.25)),
                NPL_RATIO: float(rng.uniform(0.02, 0.06)) if industry[i] == "银行" else np.nan,
                PROVISION_COVERAGE: float(rng.uniform(1.0, 2.5)) if industry[i] == "银行" else np.nan,
                CET1: float(rng.uniform(0.08, 0.14)) if industry[i] == "银行" else np.nan,
                CASH_SHORT_DEBT: float(rng.uniform(0.4, 2.0)) if industry[i] == "房地产开发" else np.nan,
            }
            for f, v in vals.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                fact_rows.append({
                    "code": codes[i], "field": f, "period": f"{y}1231",
                    "announce_date": ann, "value": float(v),
                    "source": "synthetic", "fetch_time": pd.Timestamp.now(),
                })
            # 审计意见（文本字段用数值编码后存 value：0=标准, 1=非标）
            op = 1.0 if (is_bad[i] and rng.random() < 0.35) else 0.0
            fact_rows.append({
                "code": codes[i], "field": AUDIT_OPINION, "period": f"{y}1231",
                "announce_date": ann, "value": op,
                "source": "synthetic", "fetch_time": pd.Timestamp.now(),
            })
    facts_df = pd.DataFrame(fact_rows)

    # ---------------------------------------------------------- 宇宙表
    universe_df = pd.DataFrame({
        "code": codes,
        "name": [f"公司{i:03d}" for i in range(cfg.n_stocks)],
        "industry": industry,
        "list_date": [pd.Timestamp(year=int(y), month=1, day=1) for y in list_year],
        "delist_date": [pd.NaT] * cfg.n_stocks,
        "is_financial": [ind in FINANCIAL for ind in industry],
    })
    return universe_df, facts_df, prices_df


def build_demo_store(root: str, cfg: GenConfig | None = None, overwrite: bool = True):
    """生成合成数据并写入 Store，返回 Store 实例。

    overwrite=True 时先清空旧文件 —— 否则旧数据与新数据会被追加混在一起，
    同一 (code, period) 出现两个不同来源的值（这是实测踩过的坑）。
    """
    from .store import Store

    uni, facts, prices = generate(cfg)
    st = Store(root)
    if overwrite:
        for p in (st.facts_path, st.prices_path, st.universe_path):
            if p.exists():
                p.unlink()
    st.save_universe(uni)
    st.save_facts(facts)
    st.save_prices(prices)
    return st
