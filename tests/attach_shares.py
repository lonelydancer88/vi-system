"""补算市值（point-in-time 股本回填）。

背景：fetch_real.py 只调了 WeStockFetcher.prices()，而 prices() 里
total_share / mktcap 是刻意留空的（line 344-345）——市值必须由
attach_shares() 用财报里的股本按**公告日**回填后再算 close × total_share，
否则用最新股本会抹平历史增发稀释，引入前视偏差。

后果：mktcap 恒为 NaN → compute_metrics 里 `if not np.isfinite(mc) or mc <= 0: continue`
把所有股票跳过 → 回测结果为空。

本脚本一次性补算并覆写 prices.parquet（save_prices 按 (code,date) 去重保留最新）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from vi_system.data.store import Store
from vi_system.data.westock import WeStockFetcher


def main():
    st = Store(str(ROOT / "data" / "real_universe"))
    prices = st.load_prices()
    facts = st.load_facts()
    print(f"[in] prices {len(prices)} 行 / {prices['code'].nunique()} 只；facts {len(facts)} 行", flush=True)
    print(f"[in] 补算前 mktcap 有效 {prices['mktcap'].notna().sum()} 行", flush=True)

    merged = WeStockFetcher().attach_shares(prices, facts)

    # 清洗两类异常，否则会污染回测
    # 1) close_raw <= 0：腾讯对部分股票的不复权字段返回的是「涨跌额」而非价格
    #    （如 sh600039 2018-02-06 close_raw=-0.02，而 close_adj=28.03 才是真价）
    bad_px = int((merged["close_raw"] <= 0).sum())
    merged.loc[merged["close_raw"] <= 0, "mktcap"] = np.nan
    # 2) 市值超出合理区间：A 股最大约 3 万亿，留 5 万亿上限
    #    （如 sh600941 股本被接口给成 4618 亿股 → 算出 51.7 万亿市值）
    bad_big = int((merged["mktcap"] > 5e12).sum())
    merged.loc[merged["mktcap"] > 5e12, "mktcap"] = np.nan
    merged.loc[merged["mktcap"] < 0, "mktcap"] = np.nan
    print(f"[clean] 清洗：负/零价格 {bad_px} 行、超 5 万亿 {bad_big} 行", flush=True)

    n = len(merged)
    after = int(merged["mktcap"].notna().sum())
    print(f"[out] 补算后 mktcap 有效 {after} / {n} 行 ({100*after/max(1,n):.1f}%)", flush=True)

    # 分年统计：财报公告日最早只到 2017-01，此前无法回填股本 → 回测须从 2018 起
    yr = pd.to_datetime(merged["date"]).dt.year
    by_year = merged.groupby(yr)["mktcap"].agg(n="size", ok=lambda s: s.notna().sum())
    by_year["rate"] = (by_year["ok"] / by_year["n"]).round(3)
    print("[year] 各年市值有效率：", flush=True)
    print(by_year.to_string(), flush=True)

    # 合理性检查：A 股市值通常落在 10 亿 ~ 5 万亿
    if after:
        mc = merged.loc[merged["mktcap"].notna(), "mktcap"]
        ok = mc.between(1e8, 5e13).sum()
        print(f"[check] 市值落在 1亿~50万亿 区间: {ok} / {after} ({100*ok/after:.1f}%)", flush=True)
        print(f"[check] 市值中位数 {mc.median():.4g}｜最小 {mc.min():.4g}｜最大 {mc.max():.4g}", flush=True)

    st.save_prices(merged)
    print("[done] 已覆写 prices.parquet", flush=True)


if __name__ == "__main__":
    main()
