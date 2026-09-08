"""修正「非 1 元面值」个股的 total_share / mktcap（根因：数据源返回注册资本(元)而非股数）。

背景（review 2026-09-08 逐股核对发现）：
  westock/腾讯源对**非 1 元面值**的 A 股，返回的 total_share 实为「注册资本(元)」而非
  「股本(股)」。两者关系：注册资本(元) = 总股本(股) × 每股面值。
  因此存储值 = 真实股数 × 面值。该错误同时污染 facts 与 prices（同源），跳变检测无法发现
  （facts 与 prices 隐含股本用的是同一错误数）。

  已知非 1 元面值 A 股（金杜律所列）：
    紫金矿业 601899 面值0.1 → 存储×10=真；洛阳钼业 603993 面值0.2 → 存储×5=真；
    福莱特 601865 面值0.25 → ×4；复旦微电 688385 面值0.1 → ×10；
    中芯国际 688981 面值$0.004 / 华润微 688396 面值HK$1 / 百济神州·格科微·九号 面值外币
    → 无干净倍数，需逐年真值映射（中芯/华润在 prices 中，已修正；外币红筹未在 prices，
      不影响回测，留作已知局限）。

  本脚本修正对象（均在 universe/facts 中）：
    sh601899 紫金矿业  ×10
    sh603993 洛阳钼业  ×5
    sh688981 中芯国际  逐年真值（A股2020上市，2017/2018用≈50亿占位，无A股价无影响）
    sh688396 华润微    逐年真值
    sh601865 福莱特    ×4  (仅 facts，不在 prices)
    sh688385 复旦微电  ×10 (仅 facts，不在 prices)

用法: python3 tests/fix_face_value_shares.py
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

# ---- 修正定义 ----
# factor: 存储值 × factor = 真实股数（注册资本元 / 面值）
# map:     period(8位) -> 真实股数（来自年报/HK披露）
CORRECTIONS = {
    "sh601899": {"kind": "factor", "factor": 10.0},   # 面值 0.1
    "sh603993": {"kind": "factor", "factor": 5.0},    # 面值 0.2
    "sh688981": {"kind": "map", "shares": {
        "20171231": 5.00e8, "20181231": 5.00e8, "20191231": 5_056_868_912,
        "20201231": 7_703_507_527, "20211231": 7_903_856_555, "20221231": 7_912_664_696,
        "20231231": 7_946_555_760, "20241231": 7_976_149_966, "20251231": 8_000_408_035,
    }},
    "sh688396": {"kind": "map", "shares": {
        "20161231": 8.79e8, "20171231": 8.79e8, "20181231": 8.79e8, "20191231": 8.79e8,
        "20201231": 1_215_925_195, "20211231": 1_320_091_861, "20221231": 1_320_091_861,
        "20231231": 1_320_091_861, "20241231": 1_323_517_004, "20251231": 1_327_529_398,
    }},
    "sh601865": {"kind": "factor", "factor": 4.0},     # 面值 0.25
    "sh688385": {"kind": "factor", "factor": 10.0},    # 面值 0.1
}


def fix_facts(st: Store, code: str, corr: dict) -> int:
    facts = st.load_facts()
    mask = (facts["code"] == code) & (facts["field"] == TOTAL_SHARE)
    n = int(mask.sum())
    if corr["kind"] == "factor":
        facts.loc[mask, "value"] = facts.loc[mask, "value"] * corr["factor"]
    else:
        mp = corr["shares"]
        sub = mask & facts["period"].isin(mp)
        facts.loc[sub, "value"] = facts.loc[sub, "period"].map(mp)
        unmapped = sorted(set(facts.loc[mask, "period"]) - set(mp))
        if unmapped:
            print(f"  [warn] {code} 未映射 period: {unmapped}")
    st.save_facts(facts)
    return n


def recompute_code(st: Store, code: str) -> None:
    """仅重写该股 prices 的 total_share/mktcap（与 pipeline attach_shares 同逻辑，保留 code）。"""
    facts = st.load_facts()
    px = st.load_prices()
    rest = px[px["code"] != code].copy()

    sh = facts[facts["field"] == TOTAL_SHARE]
    s = (sh[sh["code"] == code][["announce_date", "value"]]
         .copy())
    s["announce_date"] = pd.to_datetime(s["announce_date"])
    s = (s.sort_values("announce_date")
         .rename(columns={"announce_date": "date", "value": "total_share"}))
    if s.empty:
        print(f"  [skip] {code} facts 无 total_share，跳过 prices 重写")
        return
    g = (px[px["code"] == code].sort_values("date")
         .drop(columns=["total_share", "mktcap"], errors="ignore"))
    m = pd.merge_asof(g, s, on="date", direction="backward")
    m["total_share"] = m["total_share"].bfill()
    m["mktcap"] = m["close_raw"] * m["total_share"]
    m.loc[m["mktcap"] > 5e12, "mktcap"] = np.nan
    m.loc[m["mktcap"] < 0, "mktcap"] = np.nan
    m = m[PRICES_COLUMNS]

    out = pd.concat([rest, m], ignore_index=True)[PRICES_COLUMNS]
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(st.prices_path, index=False)
    st._prices_cache = None


def main() -> None:
    st = Store(str(ROOT / "data" / "real_universe"))
    p, f = st.prices_path, st.facts_path
    shutil.copy2(p, p.with_name(p.name + ".bak_facevalue"))
    shutil.copy2(f, f.with_name(f.name + ".bak_facevalue"))
    print("[bak] 已备份 → *.bak_facevalue", flush=True)

    px = st.load_prices()
    in_prices = set(px["code"])

    for code, corr in CORRECTIONS.items():
        n = fix_facts(st, code, corr)
        print(f"[facts] {code}: 改写 {n} 行 total_share ({corr['kind']})", flush=True)
        if code in in_prices:
            recompute_code(st, code)
            print(f"[prices] {code}: 重写 mktcap", flush=True)

    # 验证
    print("\n=== 验证（最新日 隐含股本 / mktcap）===")
    vpx = st.load_prices()
    vfacts = st.load_facts()
    ts = vfacts[vfacts["field"] == TOTAL_SHARE]
    expect = {
        "sh601899": 265.9, "sh603993": 213.9, "sh688981": 80.0, "sh688396": 13.28,
        "sh601865": 23.4, "sh688385": 8.24,
    }
    for code, exp in expect.items():
        g = vpx[vpx["code"] == code].sort_values("date")
        if g.empty:
            sub = ts[ts["code"] == code].sort_values("announce_date")
            lv = sub.iloc[-1]["value"] / 1e8 if not sub.empty else None
            print(f"  {code}: (不在prices) facts最新股本={lv:.2f}亿 期望≈{exp}亿")
            continue
        r = g.iloc[-1]
        imp = r["mktcap"] / r["close_raw"] / 1e8 if r["mktcap"] else None
        mc = r["mktcap"] / 1e8 if r["mktcap"] else None
        print(f"  {code}: date={r['date'].date()} close={r['close_raw']:.2f} "
              f"隐含股本={imp:.2f}亿 期望≈{exp}亿 mktcap={mc:.0f}亿")
    print(f"\n[done] NaN-code 脏行残留: {int(vpx['code'].isna().sum())}")


if __name__ == "__main__":
    main()
