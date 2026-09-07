"""腾讯自选股（WeStock）数据通道。

**这是推荐的数据源**，理由只有一个但足够充分：
`finance` 接口直接返回 `InfoPublDate`（财报公告日）。

point-in-time 是整套系统的地基 —— 没有公告日就无法防前视偏差，
而前视偏差能把回测收益虚增 60% 以上。tushare 需要额外拼 announce_date，
这里一步到位。

    westock-data finance sh600519 --num 12 --raw
      → {"sections": [income[], balance[], cashflow[]]}，每项带 InfoPublDate

命令入口：npx -y westock-data-skillhub@1.0.5 <cmd>
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import *  # noqa: F401,F403
from .schema import (
    REVENUE, COGS, SGA, EBIT, INTEREST_EXPENSE, NET_INCOME, NET_INCOME_TOTAL,
    DEPRECIATION, TOTAL_ASSETS, CURRENT_ASSETS, CURRENT_LIAB, TOTAL_LIAB,
    TOTAL_EQUITY, CASH, RECEIVABLES, INVENTORY, PPE_NET, GOODWILL, ST_DEBT,
    LT_DEBT, BONDS, RETAINED_EARNINGS, TOTAL_SHARE, OCF, CAPEX,
    DIVIDEND_PAID, MKTCAP, CLOSE, PLEDGE_RATIO, AUDIT_OPINION,
    NPL_RATIO, PROVISION_COVERAGE, CET1,
    FACTS_COLUMNS, PRICES_COLUMNS, UNIVERSE_COLUMNS,
)

PKG = "westock-data-skillhub@1.0.5"

# 优先用已缓存的 node bin 直调，跳过 npx 每次解析开销（长任务提速关键）。
# 缓存目录 ~/.npm/_npx/<hash>/node_modules/.bin/westock-data-skillhub
def _cached_bin() -> str | None:
    base = Path.home() / ".npm" / "_npx"
    if not base.exists():
        return None
    cands = []
    for d in base.iterdir():
        p = d / "node_modules" / ".bin" / "westock-data-skillhub"
        if p.exists():
            try:
                cands.append((p.stat().st_mtime, str(p)))
            except OSError:
                continue
    return max(cands)[1] if cands else None


BIN = _cached_bin()


# ---------------------------------------------------------------- 代码归一化
# 腾讯接口代码格式为 sh600519 / sz000858（小写市场前缀，无点）。
# 系统内部统一用 600519.SH / 000858.SZ，这里做双向兼容。
def _norm_code(code: str) -> str:
    c = (code or "").strip().lower()
    if c.endswith((".sh", ".sz")):
        c = c[:-3]
    if c.startswith(("sh", "sz")):
        return c
    if c.startswith("6"):
        return "sh" + c
    if c.startswith(("0", "3", "2")):
        return "sz" + c
    return c

# ---------------------------------------------------------------- 字段映射
# 说明：部分字段腾讯接口不提供（商誉、折旧摊销、审计意见、增发），
# 缺失即缺失 —— 排雷规则遇到 NaN 会自动跳过该条，不会误判为通过。
INCOME_MAP = {
    "TotalOperatingRevenue": REVENUE,
    "OperatingCost": COGS,
    "OperatingExpense": None,          # 销售费用，与 TotalAdminExpense 合并为 SGA
    "TotalAdminExpense": None,
    "FinancialExpense": INTEREST_EXPENSE,
    "NPParentCompanyOwners": NET_INCOME,
    "TotalProfit": NET_INCOME_TOTAL,
}
BALANCE_MAP = {
    "EBIT": EBIT,
    "TotalCurrentAssets": CURRENT_ASSETS,
    "TotalNonCurrentAssets": None,     # 与流动资产相加得总资产
    "TotalCurrentLiability": CURRENT_LIAB,
    "TotalLiability": TOTAL_LIAB,
    "SEWithoutMI": TOTAL_EQUITY,       # 归母股东权益
    "CashEquivalents": CASH,
    "BillAccReceivable": RECEIVABLES,
    "Inventories": INVENTORY,
    "TotalFixedAsset": PPE_NET,
    "InterestBearDebt": LT_DEBT,       # 有息负债总额
    "RetainedProfit": RETAINED_EARNINGS,
    "PaidInCapital": TOTAL_SHARE,      # 实收资本（元）≈ 股本（面值 1 元）
}
CASHFLOW_MAP = {
    "NetOperateCashFlow": OCF,
    "FCFF": None,                      # 用于反推资本开支：capex = OCF - FCFF
}


class WeStockFetcher:
    """腾讯自选股数据抓取封装。"""

    def __init__(self, timeout: int = 180):
        self.timeout = timeout

    # ------------------------------------------------------------ 底层调用
    def _run(self, args: list[str], retries: int = 3) -> str:
        cmd = ([BIN] if BIN else ["npx", "-y", PKG]) + args
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=self.timeout)
            except subprocess.TimeoutExpired as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
                continue
            out = (r.stdout or "").strip()
            # 接口偶发限流/瞬时错误会返回空或错误文本，重试即可恢复
            if out and not out.startswith("执行失败") and "未知命令" not in out[:40]:
                return out
            last_err = RuntimeError(out[:200] or r.stderr[:200])
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"westock-data 失败(重试{retries}次): {' '.join(args[:4])} :: {last_err}")

    def _json(self, args: list[str]) -> Any:
        return json.loads(self._run(args + ["--raw"]))

    # ------------------------------------------------------------ 宇宙
    def profile(self, codes: Iterable[str]) -> pd.DataFrame:
        """公司基本信息：名称、行业、上市日期。

        注意：腾讯接口对逗号分隔的多码 profile 返回空字段，必须逐只调用。
        """
        codes = list(codes)
        rows = []
        for code in codes:
            code = _norm_code(code)
            try:
                resp = self._json(["profile", code])
            except Exception:
                continue
            data = resp.get("data") if isinstance(resp, dict) else resp
            if isinstance(data, dict):
                data = [data]
            for r in data or []:
                rows.append({
                    "code": r.get("code") or code,
                    "name": r.get("name"),
                    "industry": (r.get("industry") or "未知").strip(),
                    "list_date": pd.to_datetime(r.get("listedDate"), errors="coerce"),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=UNIVERSE_COLUMNS)
        df["delist_date"] = pd.NaT
        df["is_financial"] = df["industry"].isin(FINANCIAL_INDUSTRIES)
        return df[UNIVERSE_COLUMNS]

    def sector_constituents(self, sector_code: str) -> list[str]:
        """板块/行业成份股代码列表。"""
        data = self._json(["sector", "constituent", sector_code])
        if isinstance(data, dict):
            data = data.get("list") or data.get("items") or []
        out = []
        for r in data or []:
            c = r.get("code") or r.get("secuCode") or r.get("symbol")
            if c:
                out.append(c)
        return out

    # ------------------------------------------------------------ 财报
    def financials(self, codes: Iterable[str], num: int = 12) -> pd.DataFrame:
        """抓取三大报表 → 长表 facts（带 announce_date）。

        num 为期数，按报告期倒序。12 期 ≈ 3 年（含季报）或 12 年（仅年报）。
        """
        rows: list[dict] = []
        for code in codes:
            code = _norm_code(code)
            try:
                data = self._json(["finance", code, "--num", str(num)])
            except Exception:
                continue
            sections = data.get("sections") if isinstance(data, dict) else data
            if not isinstance(sections, list) or len(sections) < 3:
                continue

            # 部分接口版本把每个 section 返回成 dict（以报告期为 key），
            # 统一规整成「元素列表」。
            def _as_list(s):
                if isinstance(s, list):
                    return s
                if isinstance(s, dict):
                    return list(s.values())
                return []

            income, balance, cashflow = (
                _as_list(sections[0]), _as_list(sections[1]), _as_list(sections[2]))

            # 同一报告期可能有多条（不同披露版本），取公告日最新的
            def _latest(sec):
                d: dict = {}
                for r in sec or []:
                    if not isinstance(r, dict):
                        continue
                    p = str(r.get("EndDate") or r.get("date") or "")[:10].replace("-", "")
                    if not p:
                        continue
                    ann = r.get("InfoPublDate")
                    if p not in d or (ann and str(ann) > str(d[p].get("InfoPublDate") or "")):
                        d[p] = r
                return d

            inc, bal, cfs = _latest(income), _latest(balance), _latest(cashflow)
            for period in sorted(set(inc) | set(bal) | set(cfs)):
                # 只用年报（period 以 1231 结尾）。季报与年报混用会让 5 年中位数失真，
                # 且 ROE/增长会退化成季度值。这与合成数据的 schema 完全一致。
                if not str(period).endswith("1231"):
                    continue
                i, b, c = inc.get(period, {}), bal.get(period, {}), cfs.get(period, {})
                ann = (i.get("InfoPublDate") or b.get("InfoPublDate")
                       or c.get("InfoPublDate"))
                if not ann:
                    continue
                ann = pd.to_datetime(str(ann)[:10])
                vals: dict[str, float] = {}

                def _f(src, key):
                    v = src.get(key)
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return np.nan

                for k, std in INCOME_MAP.items():
                    if std:
                        vals[std] = _f(i, k)
                for k, std in BALANCE_MAP.items():
                    if std:
                        vals[std] = _f(b, k)
                for k, std in CASHFLOW_MAP.items():
                    if std:
                        vals[std] = _f(c, k)

                # 派生字段
                ca, nca = _f(b, "TotalCurrentAssets"), _f(b, "TotalNonCurrentAssets")
                vals[TOTAL_ASSETS] = (ca + nca) if np.isfinite(ca) and np.isfinite(nca) else np.nan
                sga = _f(i, "OperatingExpense")
                adm = _f(i, "TotalAdminExpense")
                vals[SGA] = (sga + adm) if np.isfinite(sga) and np.isfinite(adm) else \
                    (sga if np.isfinite(sga) else (adm if np.isfinite(adm) else np.nan))
                ocf, fcff = _f(c, "NetOperateCashFlow"), _f(c, "FCFF")
                vals[CAPEX] = (ocf - fcff) if np.isfinite(ocf) and np.isfinite(fcff) else np.nan
                vals[ST_DEBT] = 0.0
                vals[BONDS] = 0.0
                vals[NPL_RATIO] = np.nan
                vals[PROVISION_COVERAGE] = np.nan
                vals[CET1] = np.nan

                for f, v in vals.items():
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        continue
                    rows.append({
                        "code": code, "field": f, "period": period,
                        "announce_date": ann, "value": float(v),
                        "source": "westock", "fetch_time": pd.Timestamp.now(),
                    })
        if not rows:
            return pd.DataFrame(columns=FACTS_COLUMNS)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ 行情
    def _fetch_kline(self, code: str, start: str, end: str,
                     adj: str | None = None) -> pd.DataFrame:
        """拉取日线行情，自动向后翻页突破 250 行硬上限。

        腾讯 kline 单次最多返回 --end 往前 250 个交易日，且忽略 --start。
        因此以 --end 为游标逐段前移，拼出完整历史。adj=None 为不复权，
        "hfq" 为后复权。
        """
        code = _norm_code(code)
        chunks = []
        cur_end = pd.Timestamp(end)
        stop = pd.Timestamp(start) - pd.Timedelta(days=5)
        guard = 0
        while guard < 300:
            guard += 1
            args = ["kline", code, "--period", "day",
                    "--start", start, "--end", cur_end.strftime("%Y-%m-%d")]
            if adj:
                args += ["--fq", adj]
            try:
                js = self._json(args)
            except Exception:
                break
            if not js:
                break
            df = pd.DataFrame(js)
            if df.empty:
                break
            df["date"] = pd.to_datetime(df["date"])
            chunks.append(df)
            first = df["date"].min()
            if first <= stop:
                break
            cur_end = first - pd.Timedelta(days=1)
        if not chunks:
            return pd.DataFrame()
        full = pd.concat(chunks, ignore_index=True).drop_duplicates("date").sort_values("date")
        return full

    def prices(self, code: str, start: str, end: str) -> pd.DataFrame:
        """日线，同时取不复权与后复权（--fq hfq）。自动翻页取全历史。"""
        code = _norm_code(code)
        raw = self._fetch_kline(code, start, end, adj=None)
        if raw.empty:
            return pd.DataFrame(columns=PRICES_COLUMNS)
        adj = self._fetch_kline(code, start, end, adj="hfq")

        def _s(df, col):
            s = df.set_index("date")[col].rename(col) if not df.empty else pd.Series(dtype=float)
            return s

        r = _s(raw, "last")
        a = _s(adj, "last") if not adj.empty else r
        vol = _s(raw, "volume")
        amt = _s(raw, "amount")
        df = pd.concat([r.rename("close_raw"), a.rename("close_adj"),
                        vol, amt], axis=1).reset_index()
        df["code"] = code
        # 腾讯 volume 单位为手，amount 为元
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["total_share"] = np.nan
        df["mktcap"] = np.nan
        return df[PRICES_COLUMNS].dropna(subset=["close_raw"])

    def index_prices(self, code: str, start: str, end: str) -> pd.DataFrame:
        """指数日线（如沪深300 `sh000300`、中证500 `sz399905`）。

        返回 [code, date, close]，close 为指数收盘点。指数无股本/市值，
        后复权不适用，故取不复权（指数点本身就是价格回报口径，忽略股息；
        这是与个股基准对比时的已知偏差，已在报告中标注）。
        """
        code = _norm_code(code)
        raw = self._fetch_kline(code, start, end, adj=None)
        if raw.empty:
            return pd.DataFrame(columns=["code", "date", "close"])
        df = pd.DataFrame({
            "code": code,
            "date": pd.to_datetime(raw["date"]),
            "close": pd.to_numeric(raw["last"], errors="coerce"),
        }).dropna(subset=["close"])
        return df.sort_values("date").reset_index(drop=True)

    def attach_shares(self, prices: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
        """用财报里的 PaidInCapital（股本）按公告日回填总股本，并计算市值。

        市值必须用**当时可知**的股本，不能一律用最新股本 —— 否则增发稀释的历史被抹平。
        """
        if prices.empty or facts.empty:
            return prices
        sh = facts[facts["field"] == TOTAL_SHARE].copy()
        if sh.empty:
            return prices
        sh["announce_date"] = pd.to_datetime(sh["announce_date"])
        out = []
        for code, g in prices.groupby("code"):
            s = sh[sh["code"] == code][["announce_date", "value"]].sort_values("announce_date")
            if s.empty:
                g["total_share"] = np.nan
                g["mktcap"] = np.nan
                out.append(g)
                continue
            g = g.sort_values("date").drop(
                columns=["total_share", "mktcap"], errors="ignore")
            merged = pd.merge_asof(
                g, s.rename(columns={"announce_date": "date", "value": "total_share"}),
                on="date", direction="backward",
            )
            if "total_share" not in merged.columns:
                merged["total_share"] = np.nan
            merged["mktcap"] = merged["close_raw"] * merged["total_share"]
            out.append(merged)
        return pd.concat(out, ignore_index=True)

    # ------------------------------------------------------------ 事件类
    def pledge(self, codes: Iterable[str]) -> pd.DataFrame:
        """大股东质押比例 → facts（PLEDGE_RATIO）。"""
        rows = []
        for code in codes:
            code = _norm_code(code)
            try:
                data = self._json(["risk", code, "--types", "pledge"])
            except Exception:
                continue
            if not data:
                continue
            r = data[0] if isinstance(data, list) else data
            v = r.get("质押比例")
            if v is None:
                continue
            try:
                v = float(str(v).rstrip("%")) / 100.0
            except ValueError:
                continue
            d = r.get("日期") or r.get("date")
            rows.append({
                "code": code, "field": PLEDGE_RATIO,
                "period": str(d)[:10].replace("-", "") + "0101" if d else "99991231",
                "announce_date": pd.to_datetime(str(d)[:10]) if d else pd.Timestamp.now(),
                "value": v, "source": "westock", "fetch_time": pd.Timestamp.now(),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=FACTS_COLUMNS)

    def dividends(self, code: str) -> pd.DataFrame:
        """分红数据（用于计算股息率 / 净派息）。"""
        try:
            data = self._json(["dividend", code])
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame(data or [])

    def buybacks(self, code: str) -> pd.DataFrame:
        try:
            data = self._json(["buyback", code])
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame(data or [])


# ==================================================================== 一键建库
def build_westock_store(
    root: str,
    codes: list[str],
    start: str,
    end: str,
    periods: int = 52,
    with_pledge: bool = True,
    fetcher: WeStockFetcher | None = None,
    verbose: bool = True,
):
    """抓取真实数据并写入 Store。

    注意：**腾讯接口不提供退市股名单**。因此本函数建的库适合"当下选股"，
    用于历史回测会引入幸存者偏差 —— 回测需另外补退市名单。
    """
    from .store import Store

    f = fetcher or WeStockFetcher()
    st = Store(root)
    for p in (st.facts_path, st.prices_path, st.universe_path):
        if p.exists():
            p.unlink()

    if verbose:
        print(f"[1/4] 公司基本信息 {len(codes)} 只 ...")
    uni = f.profile(codes)
    st.save_universe(uni)

    if verbose:
        print(f"[2/4] 财报（{periods} 期）...")
    facts = f.financials(codes, num=periods)
    if with_pledge:
        facts = pd.concat([facts, f.pledge(codes)], ignore_index=True)
    st.save_facts(facts)

    if verbose:
        print("[3/4] 行情 ...")
    px = []
    for i, c in enumerate(codes):
        if verbose and (i + 1) % 20 == 0:
            print(f"      {i + 1}/{len(codes)}")
        try:
            d = f.prices(c, start, end)
        except Exception:
            continue
        if not d.empty:
            px.append(d)
    prices = pd.concat(px, ignore_index=True) if px else pd.DataFrame(columns=PRICES_COLUMNS)

    if verbose:
        print("[4/4] 回填股本与市值 ...")
    prices = f.attach_shares(prices, facts)
    st.save_prices(prices)

    if verbose:
        print("完成：", st.summary())
    return st
