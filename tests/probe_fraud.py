"""可行性探针：历史爆雷股在腾讯接口是否仍有数据（决定 option A 能否成立）。"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
from vi_system.data.westock import WeStockFetcher

# 历史爆雷/造假/退市案例
CASES = [
    "600518.SH",  # 康美药业（300亿造假，现ST）
    "002450.SZ",  # 康得新（119亿造假）
    "300104.SZ",  # 乐视网（已退市）
    "002069.SZ",  # 獐子岛（扇贝跑路）
    "002680.SZ",  # 长生生物（疫苗造假，已退市）
    "600086.SH",  # 东方金钰
    "600781.SH",  # 辅仁药业
    "000651.SZ",  # 格力（对照：应有数据）
]

f = WeStockFetcher(timeout=120)
print(f"{'code':10} {'profile':8} {'fin_rows':10} {'price_rows':11} {'sample_period'}")
for c in CASES:
    t0 = time.time()
    try:
        prof = f.profile([c])
        pname = prof.iloc[0]["name"] if not prof.empty else "(none)"
    except Exception as e:
        pname = f"ERR:{e}"
    try:
        fin = f.financials([c], num=52)
        nfin = len(fin)
        per = sorted(set(fin["period"]))[-1] if nfin else "-"
    except Exception as e:
        nfin, per = -1, f"ERR:{e}"
    try:
        px = f.prices(c, "2010-01-01", "2026-09-07")
        npx = len(px)
    except Exception as e:
        npx = -1
    dt = time.time() - t0
    print(f"{c:10} {str(pname)[:8]:8} {nfin:<10} {npx:<11} {per}  ({dt:.1f}s)")
