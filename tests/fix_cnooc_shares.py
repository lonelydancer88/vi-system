"""修正中国海油(sh600938) total_share 并重算市值。

问题（review 2026-09-08 抽查确认）：
  facts 里 sh600938 的 total_share：FY2018-21 记 430.81 亿、FY2022+ 记 751.80 亿，
  均与真实股本不符：
    - 真实上市前股本（2021-11-15，中财网股本结构）：446.47 亿股 = 44,647,456,000
    - FY2022 年报（2022-12-31）47,566,764,000
    - 2024-09 回购后至 2025-12-31（2025 年度末期股息公告披露）：47,529,953,984 ≈ 475.30 亿股
  后果一：mktcap = close_raw × total_share 被高估 ~1.58×（2.5 万亿 vs 真实 ~1.6 万亿），
    估值类因子（ep/bp/cfp/ebit_ev/股息率）系统性偏悲观；
  后果二：share_dilution_5y 算成 +74.5%（430.8→751.8 亿的假膨胀）> 0.5 上限，
    600938 自 2025-04 满足上市年限起就被排雷误杀，从未入选。

修复：按报告期映射真股本改写 facts 中 sh600938 的 total_share，再按 attach_shares
的 point-in-time 逻辑（merge_asof 公告日向后取）重算该股 prices 的 mktcap。

关键坑（初版 bug）：merge_asof(g, s) 时 g、s 都带 code 列（非 key），
pandas 会把 code 整体丢弃 → m 没有 code → concat 进去的是 NaN-code 垃圾行，
既没替换旧值又污染全表。修正：s 先 drop code（g 已含同一 code），merge 后 m 保留 g 的 code。

用法: python3 tests/fix_cnooc_shares.py   （先备份 bak_cnooc，已入库，落在 .gitignore 的 bak* 模式内）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import shutil

from vi_system.data.store import Store
from vi_system.data.schema import TOTAL_SHARE, PRICES_COLUMNS

CODE = "sh600938"

# 真实股本（来源：中财网股本结构 + 中国海洋石油 2025 年度末期股息公告）
REAL_BY_PERIOD = {
    "20181231": 44_647_456_000,
    "20191231": 44_647_456_000,
    "20201231": 44_647_456_000,
    "20211231": 44_647_456_000,
    "20221231": 47_566_764_000,
    "20231231": 47_529_953_984,
    "20241231": 47_529_953_984,
    "20251231": 47_529_953_984,
}


def main() -> None:
    st = Store(str(ROOT / "data" / "real_universe"))
    p = st.prices_path
    f = st.facts_path
    # 备份（bak_cnooc 落在 .gitignore 的 bak* 模式内）
    shutil.copy2(p, p.with_name(p.name + ".bak_cnooc"))
    shutil.copy2(f, f.with_name(f.name + ".bak_cnooc"))
    print("[bak] 已备份 → *.bak_cnooc", flush=True)

    facts = st.load_facts()
    px = st.load_prices()

    # ---- 1) 修正 facts 的 total_share ----
    mask = (facts["code"] == CODE) & (facts["field"] == TOTAL_SHARE)
    submask = mask & facts["period"].isin(REAL_BY_PERIOD)
    n_rows = int(submask.sum())
    unmapped = sorted(facts.loc[mask & ~submask, "period"].unique())
    facts.loc[submask, "value"] = facts.loc[submask, "period"].map(REAL_BY_PERIOD)
    st.save_facts(facts)
    print(f"[facts] 改写 {n_rows} 行 → 真实股本；未映射 period {unmapped} 保持原值", flush=True)

    # ---- 2) 干净重写该股 prices（清掉旧值与 NaN 垃圾行，重算 mktcap）----
    # 2a. 丢弃当前 sh600938 与任何 code 为 NaN 的脏行
    clean = px[px["code"].notna() & (px["code"] != CODE)].copy()
    junk = len(px) - len(clean)
    print(f"[prices] 丢弃旧 {CODE} 行与 NaN 脏行共 {junk} 行（clean 余 {len(clean)}）", flush=True)

    # 2b. 重新从「修正后 facts」回填 total_share → mktcap（point-in-time）
    facts2 = st.load_facts()
    sh = facts2[facts2["field"] == TOTAL_SHARE][["code", "announce_date", "value"]].copy()
    sh["announce_date"] = pd.to_datetime(sh["announce_date"])
    g = px[px["code"] == CODE].sort_values("date").drop(
        columns=["total_share", "mktcap"], errors="ignore")
    # 关键：s 不带入 code，避免 merge_asof 把 code 整列丢弃（见文件头注释）
    s = (sh[sh["code"] == CODE]
         .sort_values("announce_date")
         .rename(columns={"announce_date": "date", "value": "total_share"})
         .drop(columns=["code"]))
    if s.empty:
        print("[err] 无 total_share 事实，中止", flush=True)
        return
    m = pd.merge_asof(g, s, on="date", direction="backward")
    # 兜底：极早期无公告值则向前取最近一个已公告值
    m["total_share"] = m["total_share"].bfill()
    m["mktcap"] = m["close_raw"] * m["total_share"]
    # 与 attach_shares 相同的清洗
    m.loc[m["mktcap"] > 5e12, "mktcap"] = np.nan
    m.loc[m["mktcap"] < 0, "mktcap"] = np.nan
    m = m[PRICES_COLUMNS]

    out = pd.concat([clean, m], ignore_index=True)
    # 直接落盘（bypass save_prices 的 append+dedup，避免再次堆积脏行）
    out = out[PRICES_COLUMNS]
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(st.prices_path, index=False)
    st._prices_cache = None

    ok = int(m["mktcap"].notna().sum())
    print(f"[prices] 重写 {CODE} {len(m)} 行（有效 mktcap {ok}），全表 {len(out)} 行", flush=True)

    # ---- 3) 验证 ----
    vpx = st.load_prices()
    vg = vpx[vpx["code"] == CODE].sort_values("date")
    latest = vg.iloc[-1]
    imp = latest["mktcap"] / latest["close_raw"] / 1e8
    print(f"[verify] 最新日 {latest['date'].date()} close={latest['close_raw']:.2f} "
          f"mktcap={latest['mktcap']/1e8:.0f}亿 隐含股本={imp:.2f}亿股（期望≈475.3）", flush=True)
    print(f"[verify] NaN-code 脏行残留: {int(vpx['code'].isna().sum())}", flush=True)
    for y, s_ in vg.assign(y=pd.to_datetime(vg["date"]).dt.year).groupby("y"):
        r = s_.iloc[len(s_) // 2]
        print(f"  {y}: close={r['close_raw']:.2f} mktcap={r['mktcap']/1e8:.0f}亿 "
              f"隐含股本={r['mktcap']/r['close_raw']/1e8:.1f}亿股", flush=True)


if __name__ == "__main__":
    main()
