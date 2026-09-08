"""清理确认的数据卫生问题（不改变回测结果，仅去脏）。

1) facts 中 ST_DEBT / BONDS 两字段全部为 0.0（源 westock 不提供、曾被硬编码 0）。
   这些行既无信息量又违反「缺失=NaN」原则，直接删除。metrics 层对这些字段本就
   _v(..., 0.0) 默认补 0，删除后行为等价（债务=lt_debt，短债/应付债视为未知）。
   真实短债/应付债须由 fetcher.py(tushare) 重抓补全。

2) pledge_ratio 393 行 period 存成畸形值 '202609040101'（12 位，来自 announce_date
   2026-09-04）。规整为合法 8 位日期 '20260904'（该字段为高频快照，非年报，指标层已
   用全量回退取数，不影响使用，仅净化数据）。

用法：python3 tests/cleanup_data.py   （先备份 *.bak_clean）
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
sys.path.insert(0, str(ROOT))

import pandas as pd
from vi_system.data.store import Store
from vi_system.data.schema import ST_DEBT, BONDS, PLEDGE_RATIO

st = Store(str(ROOT / "data" / "real_universe"))
fp = st.facts_path
shutil.copy2(fp, fp.with_name(fp.name + ".bak_clean"))

facts = st.load_facts()
before = len(facts)

# 1) 删除 ST_DEBT / BONDS（全 0.0，无意义）
drop_mask = facts["field"].isin([ST_DEBT, BONDS])
n_drop = int(drop_mask.sum())
facts = facts[~drop_mask].copy()

# 2) pledge_ratio period 规整为 announce_date 的 YYYYMMDD
pr = facts["field"] == PLEDGE_RATIO
if pr.any():
    ad = pd.to_datetime(facts.loc[pr, "announce_date"])
    facts.loc[pr, "period"] = ad.dt.strftime("%Y%m%d")
    n_pledge = int(pr.sum())
else:
    n_pledge = 0

# 直接落盘（绕过 save_facts 的 append+dedup，避免旧行复活）
facts = facts.reset_index(drop=True)
facts.to_parquet(fp, index=False)
st._facts_cache = None

print(f"[clean] 起始 {before} 行 → 删除 ST_DEBT/BONDS {n_drop} 行 → 净 {len(facts)} 行")
print(f"[clean] pledge_ratio 规整 {n_pledge} 行 period → announce_date(YYYYMMDD)")

# 验证
v = st.load_facts()
assert v["field"].isin([ST_DEBT, BONDS]).sum() == 0, "ST_DEBT/BONDS 仍残留！"
bad = v[v["field"] == PLEDGE_RATIO]
assert bad["period"].astype(str).str.len().eq(8).all(), "pledge period 仍有非 8 位！"
print(f"[verify] ST_DEBT/BONDS=0 行；pledge period 均为 8 位 ✓")
