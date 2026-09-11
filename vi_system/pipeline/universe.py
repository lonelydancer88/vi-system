"""L1 宇宙层：定义"我的池塘"。

两条硬要求：
  1. **点时宇宙** —— 按决策日当时的状态过滤，含后来退市的公司。
     不含退市股的回测 = 幸存者偏差 = 必然高估。
  2. 流动性门槛 —— 既是可交易性要求，也是回测可执行性的前提。
"""

from __future__ import annotations

import pandas as pd

from ..config import Config


def build_universe(store, asof: str | pd.Timestamp, cfg: Config) -> pd.DataFrame:
    """返回 asof 时点的可投资宇宙。

    返回列：code, name, industry, list_date, listing_years, avg_amount_60d, mktcap, is_financial
    """
    u = store.load_universe()
    if u.empty:
        return pd.DataFrame()

    asof = pd.to_datetime(asof)
    ucfg = cfg.section("universe")

    # --- 点时过滤：已上市，且（未退市 或 退市日晚于决策日）
    listed = u["list_date"].isna() | (u["list_date"] <= asof)
    alive = u["delist_date"].isna() | (u["delist_date"] > asof)
    df = u[listed & alive].copy()
    if not ucfg.get("include_delisted", True):
        df = df[u["delist_date"].isna()]

    # --- 上市年限
    df["listing_years"] = (asof - df["list_date"]).dt.days / 365.25
    df = df[df["listing_years"] >= ucfg.get("min_listing_years", 3)]

    # --- 名称排除（ST / 退市）
    for pat in ucfg.get("exclude_name_patterns", []):
        df = df[~df["name"].astype(str).str.contains(pat.replace("*", ""), na=False)]

    # --- 板块排除（北交所 8/4/9 开头）
    for board in ucfg.get("exclude_boards", []):
        if board == "BJ":
            df = df[~df["code"].str[:3].isin(["430", "830", "831", "832", "833",
                                              "834", "835", "836", "837", "838",
                                              "839", "870", "871", "872", "873"])]

    # --- 行业排除（OCF 口径失真 / 财报科目与实业不可比）
    #     实证：银行/非银/房地产 的 OCF 含客户存款、预售款（合同负债）等递延项，
    #     ocf−capex 不是可自由支配现金流 → FCF、accruals 等衍生指标系统性失真。
    ex_inds = list(ucfg.get("exclude_industries", []) or [])
    if ex_inds:
        df = df[~df["industry"].astype(str).isin(ex_inds)]

    # --- 合并行情：市值 + 流动性
    mkt = store.market_asof(asof)
    if mkt.empty:
        return pd.DataFrame()
    df = df.merge(mkt, on="code", how="inner")
    df = df[df["avg_amount_60d"] >= ucfg.get("min_avg_amount_60d", 0)]

    cols = ["code", "name", "industry", "list_date", "listing_years",
            "avg_amount_60d", "mktcap", "close_raw", "close_adj", "is_financial"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def universe_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    return {
        "n": len(df),
        "n_industry": int(df["industry"].nunique()),
        "mktcap_sum": float(df["mktcap"].sum()),
        "median_amount": float(df["avg_amount_60d"].median()),
    }
