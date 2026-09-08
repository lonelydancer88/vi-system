"""从东方财富 F10 抓取「股本变动历史」，用于修复 point-in-time 总股本。

背景
----
库内 facts.total_share 存在第三类错误：大比例送转/增发/借壳后股本停在旧值
（如分众传媒库内 3.28 亿股 vs 真实 144.42 亿股）。该错误值长期不变，故跳变检测抓不到。
腾讯/东财实时接口只给**当前**股本，而回测需要**历年**股本，故需 F10 的股本变动历史。

数据源
------
东财 F10：https://emweb.securities.eastmoney.com/PC_HSF10/CapitalStockStructure/PageAjax?code=SZ002027&page=N
- gbjg  : 当前股本结构（TOTAL_SHARES）
- lngbbd: 股本变动历史，每页 20 条，倒序；page=1 最新，page=N 更早

产出
----
data/real_universe/em_share_history.parquet
列：code, date(变动日), total_shares, reason
"""
from __future__ import annotations

import json
import subprocess
import time

import pandas as pd

import json

OUT = "data/real_universe/em_share_history.parquet"
BASE = ("https://emweb.securities.eastmoney.com/PC_HSF10/"
        "CapitalStockStructure/PageAjax?code={code}&page={page}")
CUTOFF = "2015-12-31"      # 覆盖到回测起点(2017-05)之前即可
MAX_PAGE = 6               # 每页 20 条，6 页 = 120 条变动，足够覆盖 10 年


def _get(url: str, timeout: int = 20):
    r = subprocess.run(["curl", "-s", "--max-time", str(timeout), url],
                       capture_output=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def fetch_code(code: str) -> list[dict]:
    """抓取单只股票的股本变动历史（翻页直到覆盖 CUTOFF）。"""
    mkt = "SH" if code.startswith("sh") else "SZ"
    sym = f"{mkt}{code[2:]}"
    rows: list[dict] = []
    for page in range(1, MAX_PAGE + 1):
        j = _get(BASE.format(code=sym, page=page))
        chg = j.get("lngbbd") or []
        if not chg:
            break
        for r in chg:
            shares = r.get("TOTAL_SHARES")
            date = (r.get("END_DATE") or "")[:10]
            if not shares or not date:
                continue
            rows.append({"code": code, "date": date,
                         "total_shares": float(shares),
                         "reason": r.get("CHANGE_REASON") or ""})
        if min((r.get("END_DATE") or "9999")[:10] for r in chg) <= CUTOFF:
            break
        time.sleep(0.12)
    # 当前股本（gbjg）作为最新锚点
    j = _get(BASE.format(code=sym, page=1))
    gbjg = j.get("gbjg") or []
    if gbjg and gbjg[0].get("TOTAL_SHARES"):
        rows.append({"code": code, "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                     "total_shares": float(gbjg[0]["TOTAL_SHARES"]), "reason": "当前(锚点)"})
    return rows


def main(codes: list[str] | None = None):
    if codes is None:
        px = pd.read_parquet("data/real_universe/prices.parquet")
        codes = sorted(set(px["code"]))
    print(f"待抓取 {len(codes)} 只 …")
    all_rows: list[dict] = []
    for i, c in enumerate(codes, 1):
        try:
            all_rows.extend(fetch_code(c))
        except Exception as e:                      # 单只失败不中断全量
            print(f"  [warn] {c} 失败: {e}")
        if i % 25 == 0 or i == len(codes):
            print(f"  进度 {i}/{len(codes)}，累计 {len(all_rows)} 条", flush=True)
            pd.DataFrame(all_rows).to_parquet(OUT, index=False)   # 增量落盘
        time.sleep(0.1)

    df = (pd.DataFrame(all_rows)
          .drop_duplicates(["code", "date"])
          .sort_values(["code", "date"])
          .reset_index(drop=True))
    df.to_parquet(OUT, index=False)
    print(f"\n完成：{len(df)} 条，{df['code'].nunique()} 只 → {OUT}")
    earliest = df.groupby("code")["date"].min()
    print("最早变动日 中位数:", earliest.median(), "| 晚于 2017-01-01 的只数:",
          int((earliest > "2017-01-01").sum()))
    return df


if __name__ == "__main__":
    main()
