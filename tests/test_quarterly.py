"""季报双轨单测：验证 metrics 层在接入季报/中报/三季报后

  - 当期轨（ep/cfp/gpa/roic/accruals 等）使用最新披露期的资产负债表(点数据)
    + 利润表/现金流 TTM（trailing 12m），而非最新季报的累计(YTD)值。
  - 年报轨（profit_growth_5y / share_dilution_5y / ocf_to_ni_5y / n_years）仍只用年报。
  - 仅年报时退化为旧行为（恒等），由 correctness_test 的年报路径覆盖。

用手工 fixture（已知 YTD 数值）精确断言 TTM 数值，不依赖外部数据源。
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from vi_system.pipeline import metrics as M
from vi_system.data.schema import (
    REVENUE, COGS, NET_INCOME, OCF, TOTAL_ASSETS, TOTAL_EQUITY, MKTCAP,
)

fails = []


def check(name, cond, extra=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}  {extra}")
        fails.append(name)


def _panel():
    """一只股票：年报 2021-2024 + 季报 2024Q1..Q3 + 2025Q1..Q3。
    YTD 数值刻意设定，使 TTM 可手算核对。
    """
    rows = []
    def add(period, ann, rev, ni, ocf, ta, eq):
        rows.append({
            "code": "sh000001", "period": period, "announce_date": pd.Timestamp(ann),
            REVENUE: float(rev), NET_INCOME: float(ni), OCF: float(ocf),
            TOTAL_ASSETS: float(ta), TOTAL_EQUITY: float(eq),
        })
    add("20211231", "2022-04-10", 800, 80, 100, 1000, 600)
    add("20221231", "2023-04-10", 900, 90, 110, 1100, 650)
    add("20231231", "2024-04-10", 950, 95, 120, 1200, 700)
    add("20240331", "2024-04-30", 200, 20, 30, 1200, 700)
    add("20240630", "2024-08-30", 500, 50, 70, 1220, 710)
    add("20240930", "2024-10-30", 750, 75, 100, 1230, 720)
    add("20241231", "2025-04-10", 1000, 100, 130, 1300, 750)
    add("20250331", "2025-04-30", 250, 25, 40, 1300, 750)
    add("20250630", "2025-08-30", 600, 60, 85, 1320, 760)
    add("20250930", "2025-10-30", 850, 85, 110, 1330, 770)
    return pd.DataFrame(rows)


def _run(latest_periods):
    panel = _panel()
    if latest_periods is not None:
        panel = panel[panel["period"].isin(latest_periods)]
    market = pd.DataFrame([{
        "code": "sh000001", MKTCAP: 10000.0, "close_raw": 10.0, "close_adj": 10.0,
    }])
    universe = pd.DataFrame([{
        "code": "sh000001", "name": "T", "industry": "测试",
        "avg_amount_60d": 1e8, "listing_years": 20, "is_financial": False,
    }])
    return M.compute_metrics(panel, market, universe).iloc[0]


def test_ttm_current_track():
    print("\n[A] 当期轨使用 TTM（最新披露 2025Q3）")
    m = _run(None)  # 全量 → latest = 20250930
    # TTM 手算：rev = 850 + 1000 - 750 = 1100
    check("TTM 营收 = 850 + 1000 - 750 = 1100", abs(m["revenue"] - 1100) < 1e-6, f"got {m['revenue']}")
    # TTM 净利 = 85 + 100 - 75 = 110 → ep = 110 / 10000
    check("TTM 净利流入 ep = 110/10000", abs(m["ep"] - 110 / 10000) < 1e-9, f"got {m['ep']}")
    # TTM OCF = 110 + 130 - 100 = 140 → cfp = 140 / 10000
    check("TTM OCF 流入 cfp = 140/10000", abs(m["cfp"] - 140 / 10000) < 1e-9, f"got {m['cfp']}")
    # accruals = (TTM NI - TTM OCF) / 最新资产负债表总资产(1330) = (110-140)/1330
    check("accruals 用 TTM 差 / 最新总资产",
          abs(m["accruals"] - (110 - 140) / 1330) < 1e-9, f"got {m['accruals']}")
    # 资产负债表用最新点数据：accruals 分母=TOTAL_ASSETS(2025Q3=1330) 而非 2024 年报 1300，
    # 上面等式仅当分母为 1330 成立，间接验证资产负债表取最新披露期。


def _v_cur_assets(m):
    return m.get("period")


def test_annual_track_untouched():
    print("\n[B] 年报轨不受影响（仅年报计数 / 5年指标）")
    m = _run(None)
    # 最新披露为 2025Q3，年报序列仍为 2021-2024 共 4 个 → n_years = 4
    check("n_years = 4（年报计数，不含季报）", m["n_years"] == 4, f"got {m['n_years']}")
    # 2021-2024 仅 4 个年报 < 5 → profit_growth_5y 应为 NaN
    check("profit_growth_5y = NaN（年报不足 5）",
          m["profit_growth_5y"] is None or (isinstance(m["profit_growth_5y"], float)
                                            and np.isnan(m["profit_growth_5y"])),
          f"got {m['profit_growth_5y']}")
    # meta period 反映最新披露期（季报），便于报告披露新鲜度
    check("meta period = 20250930（最新披露期）",
          str(m["period"]) == "20250930", f"got {m['period']}")


def test_ttm_not_ytd():
    print("\n[C] 当期指标是 TTM 而非 YTD（防旧口径回退）")
    m = _run(None)
    # 若错误地用 YTD(2025Q3 净利=85) 则 ep=85/10000；正确应为 110/10000
    check("ep 不是 YTD(85/10000) 而是 TTM(110/10000)",
          abs(m["ep"] - 110 / 10000) < 1e-9 and abs(m["ep"] - 85 / 10000) > 1e-9)


def test_annual_only_identity():
    print("\n[D] 仅年报时退化为恒等（季报缺失不报错）")
    m = _run(["20211231", "20221231", "20231231", "20241231"])
    # 仅年报：latest=20241231，TTM 恒等 → 营收=1000, ep=100/10000
    check("仅年报：营收=1000（恒等）", abs(m["revenue"] - 1000) < 1e-6, f"got {m['revenue']}")
    check("仅年报：ep=100/10000（恒等）", abs(m["ep"] - 100 / 10000) < 1e-9, f"got {m['ep']}")
    check("仅年报：n_years=4", m["n_years"] == 4, f"got {m['n_years']}")


if __name__ == "__main__":
    test_ttm_current_track()
    test_annual_track_untouched()
    test_ttm_not_ytd()
    test_annual_only_identity()
    print()
    if fails:
        print(f"[FAIL] test_quarterly: {len(fails)} 项失败 -> {fails}")
        sys.exit(1)
    print("[PASS] test_quarterly ✅")
