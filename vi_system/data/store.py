"""数据落盘与 point-in-time 查询。

存储：Parquet 分区文件（facts / prices / universe）。
核心能力：给定决策日 asof，只返回 announce_date <= asof 的财务数据 —— 这是防前视偏差的地基。

重述处理：同一 (code, field, period) 可能因财报更正出现多行，
取 asof 时点可见的「最新版本」（announce_date 最大者），历史版本保留在库中供审计。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import (
    FACTS_COLUMNS,
    PRICES_COLUMNS,
    UNIVERSE_COLUMNS,
    MKTCAP,
    CLOSE,
)


class Store:
    """基于 Parquet 的本地数据仓库。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.facts_path = self.root / "facts.parquet"
        self.prices_path = self.root / "prices.parquet"
        self.universe_path = self.root / "universe.parquet"
        self.index_path = self.root / "index_prices.parquet"
        # 回测会按调仓日反复查询同一份数据，不缓存的话每个截面都要重读一次全表
        self._facts_cache: pd.DataFrame | None = None
        self._prices_cache: pd.DataFrame | None = None

    # ================================================================ 写入
    def save_facts(self, df: pd.DataFrame) -> None:
        df = self._normalize(df, FACTS_COLUMNS)
        df["period"] = df["period"].astype(str)
        df["announce_date"] = pd.to_datetime(df["announce_date"])
        if self.facts_path.exists():
            old = pd.read_parquet(self.facts_path)
            df = pd.concat([old, df], ignore_index=True)
            # 同一 (code, field, period, announce_date) 只保留一条
            df = df.drop_duplicates(
                subset=["code", "field", "period", "announce_date"], keep="last"
            )
        df.to_parquet(self.facts_path, index=False)
        self._facts_cache = None

    def save_prices(self, df: pd.DataFrame) -> None:
        df = self._normalize(df, PRICES_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        if self.prices_path.exists():
            old = pd.read_parquet(self.prices_path)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=["code", "date"], keep="last")
        df.to_parquet(self.prices_path, index=False)
        self._prices_cache = None

    def save_universe(self, df: pd.DataFrame) -> None:
        df = self._normalize(df, UNIVERSE_COLUMNS)
        for c in ("list_date", "delist_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        df.to_parquet(self.universe_path, index=False)

    # ---------------------------------------------------------- 指数基准行情
    def save_index_prices(self, df: pd.DataFrame) -> None:
        df = self._normalize(df, ["code", "date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        if self.index_path.exists():
            old = pd.read_parquet(self.index_path)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=["code", "date"], keep="last")
        df.to_parquet(self.index_path, index=False)

    def load_index_prices(self) -> pd.DataFrame:
        if not self.index_path.exists():
            return pd.DataFrame(columns=["code", "date", "close"])
        df = pd.read_parquet(self.index_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def _normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        return df[columns].copy()

    # ================================================================ 读取
    def has_data(self) -> bool:
        return self.facts_path.exists() and self.prices_path.exists()

    def load_facts(self) -> pd.DataFrame:
        if self._facts_cache is None:
            if not self.facts_path.exists():
                return pd.DataFrame(columns=FACTS_COLUMNS)
            df = pd.read_parquet(self.facts_path)
            df["announce_date"] = pd.to_datetime(df["announce_date"])
            self._facts_cache = df
        return self._facts_cache

    def load_prices(self) -> pd.DataFrame:
        if self._prices_cache is None:
            if not self.prices_path.exists():
                return pd.DataFrame(columns=PRICES_COLUMNS)
            df = pd.read_parquet(self.prices_path)
            df["date"] = pd.to_datetime(df["date"])
            self._prices_cache = df
        return self._prices_cache

    def load_universe(self) -> pd.DataFrame:
        if not self.universe_path.exists():
            return pd.DataFrame(columns=UNIVERSE_COLUMNS)
        return pd.read_parquet(self.universe_path)

    # ==================================================== point-in-time 核心
    def facts_asof(self, asof: str | pd.Timestamp, periods: int = 24) -> pd.DataFrame:
        """返回截至 asof 可见的财务宽表（panel）。

        规则：
          1. 只保留 announce_date <= asof
          2. 同一 (code, field, period) 取 announce_date 最大的版本（最新重述）
          3. 只保留最近 `periods` 个报告期（控制计算量）
          4. pivot 成 (code, period) × field

        注：`periods` 默认 24（≈6 年 × 4 期），保证季报接入后仍有 ≥5 个年报供
        5 年指标（hist5 / profit_growth_5y / share_dilution_5y）使用，不会被季度
        稀释成 5 个季度。仅年报时 24 自然覆盖全部历史，行为不变。
        """
        facts = self.load_facts()
        if facts.empty:
            return pd.DataFrame()

        asof = pd.to_datetime(asof)
        visible = facts[facts["announce_date"] <= asof]
        if visible.empty:
            return pd.DataFrame()

        # 最近 N 个报告期
        recent = sorted(visible["period"].unique())[-periods:]
        visible = visible[visible["period"].isin(recent)]

        # 同一 (code, field, period) 取最新版本
        visible = visible.sort_values("announce_date")
        latest = visible.groupby(
            ["code", "field", "period"], as_index=False
        ).tail(1)

        panel = latest.pivot_table(
            index=["code", "period"], columns="field", values="value", aggfunc="last"
        )
        panel.columns.name = None
        panel = panel.reset_index()

        # 附上该报告期的公告日期，供上层判断数据新鲜度
        ann = (
            latest.groupby(["code", "period"], as_index=False)["announce_date"]
            .max()
            .rename(columns={"announce_date": "announce_date"})
        )
        panel = panel.merge(ann, on=["code", "period"], how="left")
        return panel

    def market_asof(self, asof: str | pd.Timestamp) -> pd.DataFrame:
        """返回 asof 当日（或之前最近交易日）的行情快照。

        含：收盘价（原始/后复权）、市值、近60交易日日均成交额。
        """
        prices = self.load_prices()
        if prices.empty:
            return pd.DataFrame()

        asof = pd.to_datetime(asof)
        hist = prices[prices["date"] <= asof]
        if hist.empty:
            return pd.DataFrame()

        last_date = hist["date"].max()
        snap = hist[hist["date"] == last_date].copy()

        # 近 60 个交易日日均成交额。
        # 按精确的「最近 60 个交易日」窗口计算（此前用 100 自然日窗口 ≈ 68 个
        # 交易日，与字段名/配置注释的「60 交易日」不符，导致阈值口径偏松）。
        # 只在窗口日期上过滤再分组 —— 对全历史 groupby 会让每个调仓截面都
        # 付出 O(全表) 的代价，回测里会被放大几十倍。
        recent_dates = set(
            pd.DatetimeIndex(sorted(hist["date"].unique()))[-60:]
        )
        window = hist[hist["date"].isin(recent_dates)]
        amt = (
            window.groupby("code")["amount"]
            .mean()
            .rename("avg_amount_60d")
            .reset_index()
        )
        snap = snap.merge(amt, on="code", how="left")
        snap["asof_date"] = last_date
        return snap[["code", "close_raw", "close_adj", "mktcap",
                     "avg_amount_60d", "asof_date", "volume", "amount"]]

    def price_panel(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        """返回区间内的后复权收盘价宽表：index=date, columns=code。"""
        prices = self.load_prices()
        if prices.empty:
            return pd.DataFrame()
        mask = (prices["date"] >= pd.to_datetime(start)) & (
            prices["date"] <= pd.to_datetime(end)
        )
        sub = prices.loc[mask, ["date", "code", "close_adj"]]
        return sub.pivot(index="date", columns="code", values="close_adj").sort_index()

    # ================================================================ 工具
    def rebalance_dates(
        self, months: list[int], day: int, start: str, end: str
    ) -> list[pd.Timestamp]:
        """生成调仓日：指定月份的指定日（若非交易日则取之前最近交易日）。"""
        prices = self.load_prices()
        if prices.empty:
            return []
        all_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
        out = []
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            for m in months:
                target = pd.Timestamp(year=year, month=m, day=day)
                prior = all_dates[all_dates <= target]
                if len(prior):
                    d = prior[-1]
                    if pd.Timestamp(start) <= d <= pd.Timestamp(end):
                        out.append(d)
        return sorted(set(out))

    def summary(self) -> dict:
        if not self.has_data():
            return {"status": "empty"}
        f = self.load_facts()
        p = self.load_prices()
        u = self.load_universe()
        return {
            "status": "ok",
            "facts_rows": len(f),
            "prices_rows": len(p),
            "universe_rows": len(u),
            "codes": int(p["code"].nunique()) if len(p) else 0,
            "price_range": (
                f"{p['date'].min().date()} ~ {p['date'].max().date()}" if len(p) else "-"
            ),
            "facts_range": (
                f"{f['period'].min()} ~ {f['period'].max()}" if len(f) else "-"
            ),
        }
