"""用东财 F10 股本变动历史修复 point-in-time 总股本（第三类股本 bug）。

问题
----
库内 facts.total_share 在「大比例送转 / 增发 / 借壳」后停在旧值：
分众传媒 3.28 亿(真实 144.42 亿)、中国移动 4684 亿(真实 216.91 亿) 等，
297 只中 34 只偏差 >5%。该值长期不变，故跳变检测抓不到。

修复
----
1. facts ：按 period 对齐「该期末之前最后一次股本变动」的真实值，改写 total_share。
   （pipeline 的 attach_shares 会按 announce_date 做 merge_asof，故改 value 即可保持
     point-in-time 语义。）
2. prices：按交易日 merge_asof 回填 total_share，重算 mktcap = close_raw × total_share。
3. 仅覆盖东财有数据且与库内偏差 >TOL 的记录，其余保留原值。

备份：*.bak_emshare（受 .gitignore 中 /data/*/*.parquet.bak* 忽略）
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from vi_system.data.schema import PRICES_COLUMNS

FACTS = "data/real_universe/facts.parquet"
PRICES = "data/real_universe/prices.parquet"
HIST = "data/real_universe/em_share_history.parquet"
TOL = 0.02          # 偏差 >2% 才覆盖


def _backup(path: str):
    dst = path + ".bak_emshare"
    shutil.copy2(path, dst)
    print(f"[bak] {path} → {dst}")


def main():
    hist = pd.read_parquet(HIST)
    hist["date"] = pd.to_datetime(hist["date"])
    # 同一天多条（不同原因）取最后一条
    hist = (hist.sort_values(["code", "date"])
                .drop_duplicates(["code", "date"], keep="last")
                [["code", "date", "total_shares"]])
    print(f"[hist] {len(hist)} 条变动，{hist['code'].nunique()} 只")

    _backup(FACTS)
    _backup(PRICES)

    # ---------------------------------------------------------------- facts
    ft = pd.read_parquet(FACTS)
    is_ts = ft["field"].eq("total_share")
    period = pd.to_datetime(ft["period"].astype(str), format="%Y%m%d", errors="coerce")
    ft["_pd"] = period

    # 东财只给「有记录的变动」（最多 20 条/只），早期区间是否可信取决于库内失真程度：
    #   - 库内最新 vs 东财最新 偏差 >50% → 库内整体不可信，早期也用东财最早值回填；
    #   - 否则（多为近年小幅行权/回购导致记录偏晚）→ 早期保留库内原值。
    lib_last = (ft[is_ts].sort_values("period").groupby("code").tail(1)
                .set_index("code")["value"])
    em_last = hist.sort_values("date").groupby("code").tail(1).set_index("code")["total_shares"]
    ratio_latest = (lib_last / em_last).dropna()
    bad_all = set(ratio_latest[(ratio_latest - 1).abs() > 0.20].index)
    print(f"[判定] 库内整体失真（偏差>50%）需全区间覆盖：{len(bad_all)} 只")

    changed = 0
    for code, g in hist.groupby("code"):
        sub = ft.index[is_ts & ft["code"].eq(code)]
        if len(sub) == 0:
            continue
        s = g.sort_values("date")[["date", "total_shares"]]
        tgt = ft.loc[sub, ["_pd"]].sort_values("_pd")
        m = pd.merge_asof(tgt, s, left_on="_pd", right_on="date", direction="backward")
        new = pd.Series(m["total_shares"].values)
        if code in bad_all:
            new = new.bfill()
        new = new.values
        old = ft.loc[tgt.index, "value"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(old) & (old > 0) & np.isfinite(new), new / old, 1.0)
        mask = np.isfinite(new) & (np.abs(ratio - 1) > TOL)
        if mask.any():
            ft.loc[tgt.index[mask], "value"] = new[mask]
            changed += int(mask.sum())
    ft = ft.drop(columns=["_pd"])
    ft.to_parquet(FACTS, index=False)
    print(f"[facts] 改写 {changed} 行 total_share")

    # --------------------------------------------------------------- prices
    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    all_codes = set(px["code"])
    target = [c for c in hist["code"].unique() if c in all_codes]
    print(f"[prices] 待重算 {len(target)} 只（其余 {len(all_codes)-len(target)} 只保持原值）")
    rest = px[~px["code"].isin(target)]
    fixed_parts = [rest]
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
        fixed_parts.append(m)
    out = pd.concat(fixed_parts, ignore_index=True)
    out = out[[c for c in PRICES_COLUMNS if c in out.columns]]
    out.to_parquet(PRICES, index=False)
    print(f"[prices] 重算 {len(target)} 只的 total_share/mktcap")

    # ---------------------------------------------------------------- 验证
    px2 = pd.read_parquet(PRICES)
    print("\n=== 验证（最新交易日隐含股本 vs 腾讯真实值）===")
    import subprocess
    last = px2[px2["date"] == px2["date"].max()].set_index("code")
    check = ["sz002027", "sh600233", "sh600941", "sz002600", "sz000301", "sz002415"]
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
    for c in check:
        if c not in real or c not in last.index:
            continue
        nm, sh = real[c]
        imp = last.loc[c, "mktcap"] / last.loc[c, "close_raw"]
        print(f"  {c} {nm:<6} 隐含 {imp/1e8:>8.2f}亿  真实 {sh/1e8:>8.2f}亿  "
              f"比值 {imp/sh:>6.3f}")
    print("\n[done] NaN-code 脏行:", int(px2["code"].isna().sum()))


if __name__ == "__main__":
    main()
