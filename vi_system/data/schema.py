"""标准化字段定义。

系统内部有两套数据形态：
  1) facts（长表，落盘）：(code, field, period, announce_date, value, ...)
  2) panel（宽表，计算）：一行 = (code, period)，列 = 各财务字段 + announce_date

两者通过 store.panel_from_facts / facts_from_panel 转换。
字段命名统一为英文 snake_case，避免不同数据源命名差异污染上层逻辑。
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------- 利润表
REVENUE = "revenue"                    # 营业总收入
COGS = "cogs"                          # 营业成本
SGA = "sga"                            # 销售+管理费用
EBIT = "ebit"                          # 息税前利润
INTEREST_EXPENSE = "interest_expense"  # 利息支出（财务费用代理）
NET_INCOME = "net_income"              # 归母净利润
NET_INCOME_TOTAL = "net_income_total"  # 净利润（含少数股东）
TAX = "income_tax"                     # 所得税
DEPRECIATION = "depreciation"          # 折旧摊销

# ---------------------------------------------------------------- 资产负债表
TOTAL_ASSETS = "total_assets"
CURRENT_ASSETS = "current_assets"
CURRENT_LIAB = "current_liab"
TOTAL_LIAB = "total_liab"
TOTAL_EQUITY = "total_equity"          # 归母净资产
CASH = "cash"
RECEIVABLES = "receivables"            # 应收票据及应收账款
INVENTORY = "inventory"
PPE_NET = "ppe_net"                    # 固定资产净额（含在建/无形，代理 NetPPE）
GOODWILL = "goodwill"
ST_DEBT = "st_debt"                    # 短期借款
LT_DEBT = "lt_debt"                    # 长期借款
BONDS = "bonds_payable"                # 应付债券
RETAINED_EARNINGS = "retained_earnings"
TOTAL_SHARE = "total_share"            # 总股本（股）

# ---------------------------------------------------------------- 现金流量表
OCF = "ocf"                            # 经营活动现金流净额
CAPEX = "capex"                        # 购建固定/无形资产支付的现金
DIVIDEND_PAID = "dividend_paid"        # 分配股利支付的现金
EQUITY_ISSUED = "equity_issued"        # 吸收投资收到的现金（增发代理）
BUYBACK = "buyback"                    # 回购支付的现金

# ---------------------------------------------------------------- 市场/其他
MKTCAP = "mktcap"
CLOSE = "close"
DIVIDEND = "dividend"                  # 当期现金分红总额
AUDIT_OPINION = "audit_opinion"        # 审计意见（字符串编码）
PLEDGE_RATIO = "pledge_ratio"          # 大股东质押比例
EV = "ev"                              # 企业价值（派生）

# ---------------------------------------------------------------- 银行专用
NPL_RATIO = "npl_ratio"
PROVISION_COVERAGE = "provision_coverage"
CET1 = "cet1"
CASH_SHORT_DEBT = "cash_short_debt"    # 地产：现金/短债

#: 所有可出现在 facts 表中的数值字段
NUMERIC_FIELDS: Final[list[str]] = [
    REVENUE, COGS, SGA, EBIT, INTEREST_EXPENSE, NET_INCOME, NET_INCOME_TOTAL,
    TAX, DEPRECIATION,
    TOTAL_ASSETS, CURRENT_ASSETS, CURRENT_LIAB, TOTAL_LIAB, TOTAL_EQUITY,
    CASH, RECEIVABLES, INVENTORY, PPE_NET, GOODWILL,
    ST_DEBT, LT_DEBT, BONDS, RETAINED_EARNINGS, TOTAL_SHARE,
    OCF, CAPEX, DIVIDEND_PAID, EQUITY_ISSUED, BUYBACK,
    MKTCAP, CLOSE, DIVIDEND, PLEDGE_RATIO,
    NPL_RATIO, PROVISION_COVERAGE, CET1, CASH_SHORT_DEBT,
]

#: 字符串型字段（审计意见等）
TEXT_FIELDS: Final[list[str]] = [AUDIT_OPINION]

#: 计算因子所需的最小字段集；缺任一则该标的该期不参与打分
REQUIRED_FOR_FACTORS: Final[list[str]] = [
    NET_INCOME, TOTAL_EQUITY, TOTAL_ASSETS, REVENUE, COGS, OCF, MKTCAP,
]

#: 金融/地产行业（走特化规则）
FINANCIAL_INDUSTRIES: Final[list[str]] = ["银行", "保险", "证券", "多元金融"]
REAL_ESTATE_INDUSTRIES: Final[list[str]] = ["房地产"]

#: 审计意见编码
AUDIT_STANDARD: Final[str] = "标准无保留"
AUDIT_NONSTANDARD_MARK: Final[str] = "非标"

FACTS_COLUMNS: Final[list[str]] = [
    "code", "field", "period", "announce_date", "value", "source", "fetch_time",
]

PRICES_COLUMNS: Final[list[str]] = [
    "code", "date", "close_raw", "close_adj", "volume", "amount",
    "total_share", "mktcap",
]

UNIVERSE_COLUMNS: Final[list[str]] = [
    "code", "name", "industry", "list_date", "delist_date", "is_financial",
]

REPORT_PERIODS: Final[list[str]] = ["0331", "0630", "0930", "1231"]


def period_key(year: int, month: int) -> str:
    """把 (年, 月) 归一到报告期字符串，如 (2025, 12) -> '20251231'。"""
    if month <= 3:
        return f"{year}0331"
    if month <= 6:
        return f"{year}0630"
    if month <= 9:
        return f"{year}0930"
    return f"{year}1231"


def period_year(period: str) -> int:
    return int(str(period)[:4])
