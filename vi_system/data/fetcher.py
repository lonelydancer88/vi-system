"""tushare 数据抓取器（真实数据通道）。

仅在配置了 TUSHARE_TOKEN 时可用；无 token 时用 data.synthetic 生成合成数据跑通流程。

核心约束：财报数据必须带 announce_date（公告日期），不是报告期日期。
没有 announce_date 的回测 = 前视偏差 = 结果虚增。
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from .schema import *  # noqa: F401,F403
from .schema import FACTS_COLUMNS, AUDIT_OPINION

# tushare 字段名 → 内部标准字段名
INCOME_MAP = {
    "total_revenue": REVENUE,
    "oper_cost": COGS,
    "sell_exp": None,               # 需与 admin_exp 合并为 SGA
    "admin_exp": None,
    "operate_profit": None,
    "ebit": EBIT,
    "fin_exp": INTEREST_EXPENSE,
    "income_tax": TAX,
    "n_income_attr_p": NET_INCOME,
    "n_income": NET_INCOME_TOTAL,
    "total_profit": None,
}
BALANCE_MAP = {
    "total_assets": TOTAL_ASSETS,
    "total_cur_assets": CURRENT_ASSETS,
    "total_cur_liab": CURRENT_LIAB,
    "total_liab": TOTAL_LIAB,
    "total_hldr_eqy_exc_min_int": TOTAL_EQUITY,
    "money_cap": CASH,
    "notes_receiv": None,
    "accounts_receiv": RECEIVABLES,
    "inventories": INVENTORY,
    "fix_assets": PPE_NET,
    "goodwill": GOODWILL,
    "st_borrow": ST_DEBT,
    "lt_borrow": LT_DEBT,
    "bond_payable": BONDS,
    "undistr_porfit": RETAINED_EARNINGS,
    "total_share": TOTAL_SHARE,
}
CASHFLOW_MAP = {
    "n_cashflow_act": OCF,
    "c_pay_acq_const_fiolta": CAPEX,
    "div_pay": DIVIDEND_PAID,
    "depr_fa_coga_dpba": DEPRECIATION,
    "cash_recy_cap_contrib": EQUITY_ISSUED,
}


class TushareFetcher:
    """tushare pro 抓取封装。"""

    def __init__(self, token: str | None = None, sleep: float = 0.35):
        try:
            import tushare as ts  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "未安装 tushare。执行 `pip install tushare`，"
                "或改用 data.synthetic.build_demo_store() 生成合成数据跑通流程。"
            ) from e
        token = token or __import__("os").environ.get("TUSHARE_TOKEN")
        if not token:
            raise ValueError(
                "缺少 tushare token：设置环境变量 TUSHARE_TOKEN 或传入 token 参数。"
            )
        self.pro = ts.pro_api(token)
        self.sleep = sleep

    # ------------------------------------------------------------ 宇宙
    def fetch_universe(self) -> pd.DataFrame:
        basic = self.pro.stock_basic(
            exchange="", list_status="L",
            fields="ts_code,symbol,name,area,industry,list_date,delist_date,market",
        )
        df = pd.DataFrame({
            "code": basic["ts_code"],
            "name": basic["name"],
            "industry": basic["industry"].fillna("未知"),
            "list_date": pd.to_datetime(basic["list_date"], errors="coerce"),
            "delist_date": pd.to_datetime(basic.get("delist_date"), errors="coerce"),
        })
        df["is_financial"] = df["industry"].isin(FINANCIAL_INDUSTRIES)
        return df

    # ------------------------------------------------------------ 行情
    def fetch_prices(self, codes: Iterable[str], start: str, end: str) -> pd.DataFrame:
        """后复权日线。逐只抓取（tushare 批量接口对频率敏感）。"""
        rows = []
        for code in codes:
            try:
                px = self.pro.pro_bar(
                    ts_code=code, start_date=start, end_date=end, adj="hfq", freq="D"
                )
            except Exception:
                time.sleep(self.sleep)
                continue
            if px is None or px.empty:
                continue
            px["trade_date"] = pd.to_datetime(px["trade_date"])
            px = px.sort_values("trade_date")
            raw = self.pro.daily(
                ts_code=code, start_date=start, end_date=end,
                fields="trade_date,close",
            )
            share = self.pro.daily_basic(
                ts_code=code, start_date=start, end_date=end,
                fields="trade_date,total_share,total_mv",
            )
            if raw is not None and not raw.empty:
                raw["trade_date"] = pd.to_datetime(raw["trade_date"])
                px = px.merge(raw.rename(columns={"close": "close_raw"}),
                              on="trade_date", how="left")
            else:
                px["close_raw"] = px["close"]
            if share is not None and not share.empty:
                share["trade_date"] = pd.to_datetime(share["trade_date"])
                px = px.merge(share, on="trade_date", how="left")
            else:
                px["total_share"] = pd.NA
                px["total_mv"] = pd.NA

            rows.append(pd.DataFrame({
                "code": code,
                "date": px["trade_date"],
                "close_raw": pd.to_numeric(px["close_raw"], errors="coerce"),
                "close_adj": pd.to_numeric(px["close"], errors="coerce"),
                "volume": pd.to_numeric(px.get("vol"), errors="coerce") * 100,
                "amount": pd.to_numeric(px.get("amount"), errors="coerce") * 1000,
                "total_share": pd.to_numeric(px.get("total_share"), errors="coerce"),
                "mktcap": pd.to_numeric(px.get("total_mv"), errors="coerce") * 10000,
            }))
            time.sleep(self.sleep)
        if not rows:
            return pd.DataFrame(columns=FACTS_COLUMNS and PRICES_COLUMNS)
        return pd.concat(rows, ignore_index=True).dropna(subset=["close_adj"])

    # ------------------------------------------------------------ 财报
    def fetch_financials(
        self, codes: Iterable[str], start_period: str, end_period: str
    ) -> pd.DataFrame:
        """抓取三大报表，转成长表 facts（带 announce_date）。"""
        out = []
        for code in codes:
            for api, mapping in (
                ("income", INCOME_MAP),
                ("balancesheet", BALANCE_MAP),
                ("cashflow", CASHFLOW_MAP),
            ):
                try:
                    fn = getattr(self.pro, api)
                    df = fn(
                        ts_code=code, start_date=start_period, end_date=end_period,
                        fields="ts_code,ann_date,f_ann_date,end_date,report_type,"
                               + ",".join(k for k in mapping if k),
                    )
                except Exception:
                    time.sleep(self.sleep)
                    continue
                if df is None or df.empty:
                    continue
                # 只保留合并报表（report_type=1）
                if "report_type" in df.columns:
                    df = df[df["report_type"].astype(str).isin(["1", "1.0"])]
                df = df.drop_duplicates(subset=["end_date"], keep="last")
                for tushare_name, std_name in mapping.items():
                    if std_name is None or tushare_name not in df.columns:
                        continue
                    for _, r in df.iterrows():
                        ann = r.get("f_ann_date") or r.get("ann_date")
                        if not ann or pd.isna(ann):
                            continue
                        out.append({
                            "code": code,
                            "field": std_name,
                            "period": str(r["end_date"]),
                            "announce_date": pd.to_datetime(str(ann)),
                            "value": pd.to_numeric(r[tushare_name], errors="coerce"),
                            "source": "tushare",
                            "fetch_time": pd.Timestamp.now(),
                        })
                time.sleep(self.sleep)
        return pd.DataFrame(out).dropna(subset=["value"])

    def fetch_pledge(self, codes: Iterable[str]) -> pd.DataFrame:
        """大股东质押比例（tushare pledge_stat）。"""
        rows = []
        for code in codes:
            try:
                df = self.pro.pledge_stat(ts_code=code)
            except Exception:
                time.sleep(self.sleep)
                continue
            if df is None or df.empty:
                continue
            last = df.sort_values("end_date").tail(1)
            r = last.iloc[0]
            rows.append({
                "code": code, "field": PLEDGE_RATIO,
                "period": str(r["end_date"]),
                "announce_date": pd.to_datetime(str(r["end_date"])) + pd.Timedelta(days=30),
                "value": float(r.get("pledge_ratio", 0)) / 100.0,
                "source": "tushare", "fetch_time": pd.Timestamp.now(),
            })
            time.sleep(self.sleep)
        return pd.DataFrame(rows)

    def fetch_audit(self, codes: Iterable[str]) -> pd.DataFrame:
        """审计意见：0=标准无保留，1=非标。"""
        rows = []
        for code in codes:
            try:
                df = self.pro.fina_audit(ts_code=code)
            except Exception:
                time.sleep(self.sleep)
                continue
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                op = str(r.get("audit_result", ""))
                val = 0.0 if "标准" in op else 1.0
                ann = r.get("ann_date") or r.get("end_date")
                rows.append({
                    "code": code, "field": AUDIT_OPINION,
                    "period": str(r["end_date"]),
                    "announce_date": pd.to_datetime(str(ann)),
                    "value": val, "source": "tushare",
                    "fetch_time": pd.Timestamp.now(),
                })
            time.sleep(self.sleep)
        return pd.DataFrame(rows)
