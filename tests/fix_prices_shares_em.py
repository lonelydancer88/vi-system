"""只重算 prices 的 total_share / mktcap（东财股本历史）。

与 fix_shares_from_em.py 的 prices 段一致，但独立运行：
bad_all（需全区间覆盖的失真股）取自**原始审计结果** out/share_audit-2026-09-08.csv，
因为 facts 已被改写，重新计算比值会被抹平而误判。
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from vi_system.data.schema import PRICES_COLUMNS

PRICES = "data/real_universe/prices.parquet"
HIST = "data/real_universe/em_share_history.parquet"
AUDIT = "out/share_audit-2026-09-08.csv"
TOL = 0.20


def main():
    hist = pd.read_parquet(HIST)
    hist["date"] = pd.to_datetime(hist["date"])
    hist = (hist.sort_values(["code", "date"])
                .drop_duplicates(["code", "date"], keep="last")
                [["code", "date", "total_shares"]])

    audit = pd.read_csv(AUDIT)
    bad_all = set(audit.loc[audit["比值"].sub(1).abs() > TOL, "code"])
    print(f"[hist] {len(hist)} 条变动 / {hist['code'].nunique()} 只")
    print(f"[bad_all] 原始偏差>{TOL:.0%} 需全区间覆盖：{len(bad_all)} 只")

    shutil.copy2(PRICES, PRICES + ".bak_emshare")
    print(f"[bak] {PRICES} → {PRICES}.bak_emshare")

    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    all_codes = set(px["code"])
    target = [c for c in hist["code"].unique() if c in all_codes]
    print(f"[prices] 待重算 {len(target)} 只，其余 {len(all_codes)-len(target)} 只保持原值")

    parts = [px[~px["code"].isin(target)]]
    for code in target:
        s = (hist[hist["code"].eq(code)].sort_values("date")[["date", "total_shares"]]
             .rename(columns={"total_shares": "total_share"}))
        cur = px[px["code"].eq(code)].sort_values("date")
        m = pd.merge_asof(cur.drop(columns=["total_share", "mktcap"], errors="ignore"),
                          s, on="date", direction="backward")
        if code in bad_all:
            m["total_share"] = m["total_share"].bfill()
        m["total_share"] = m["total_share"].fillna(
            pd.Series(cur["total_share"].values, index=m.index))
        m["mktcap"] = m["close_raw"] * m["total_share"]
        m.loc[m["mktcap"] > 5e12, "mktcap"] = np.nan
        m.loc[m["mktcap"] < 0, "mktcap"] = np.nan
        parts.append(m)

    out = pd.concat(parts, ignore_index=True)
    out = out[[c for c in PRICES_COLUMNS if c in out.columns]]
    out.to_parquet(PRICES, index=False)
    print(f"[prices] 已重算并写回 {len(out):,} 行")

    # ------------------------------------------------------------------ 验证
    import subprocess
    px2 = pd.read_parquet(PRICES)
    last = px2[px2["date"] == px2["date"].max()].set_index("code")
    check = list(audit.loc[audit["比值"].sub(1).abs() > 0.05, "code"])[:12] + ["sz002415"]
    url = "http://qt.gtimg.cn/q=" + ",".join(check)
    raw = subprocess.run(["curl", "-s", "--max-time", "20", url],
                         capture_output=True).stdout.decode("gbk", "ignore")
    real = {}
    for line in raw.split(";"):
        line = line.strip()
        if not line.startswith("v_"):
            continue
        c = line[2:line.index("=")]
        f = line[line.index('"') + 1:line.rindex('"')].split("~")
        try:
            real[c] = (f[1], float(f[73]))
        except Exception:
            pass
    print("\n=== 验证：最新交易日隐含股本 vs 腾讯真实 ===")
    ok = 0
    for c in check:
        if c not in real or c not in last.index:
            continue
        nm, sh = real[c]
        imp = last.loc[c, "mktcap"] / last.loc[c, "close_raw"]
        flag = "OK" if abs(imp / sh - 1) < 0.02 else "!!"
        ok += flag == "OK"
        print(f"  {flag} {c} {nm:<6} 隐含 {imp/1e8:>8.2f}亿  真实 {sh/1e8:>8.2f}亿  比值 {imp/sh:>6.3f}")
    print(f"\n验证通过 {ok}/{len(check)} | NaN-code 脏行：{int(px2['code'].isna().sum())}")


if __name__ == "__main__":
    main()
