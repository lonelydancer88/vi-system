"""重建干净的 `close_adj`（后复权/含分红口径）。

问题根因（已实证）
----------------
`vi_system/data/westock.py` 经 `_fetch_kline(adj="hfq")` 落库的 `close_adj` 是脏数据：
在最新交易日 `close_adj/close_raw` 不等于 1（锚点错乱），且每个交易日漂移约 1%
（>1% 跳变占 24%~38% 的交易日）。实测根因是**跨源价差**——
westock 的 raw 与腾讯 gtimg 的 raw 每天差约 0.5%，而 westock 抓 hfq 时又单独翻页、
两次快照锚点错位，导致 `close_adj/close_raw` 出现伪漂移。此漂移直接污染回测 NAV
（引擎收益全吃 close_adj）与报告「含分红」列。

修复方案
--------
改用腾讯 gtimg fqkline 的**同源** raw + qfq 序列构建干净复权因子：
  factor_t = gtimg_qfq_t / gtimg_raw_t        # 同源 → 仅在除权除息日跳变（日漂仅 0.06%）
  close_adj_new_t = close_raw_t(现有 westock) × factor_t
这样 new close_adj 与现有 close_raw **同源**、干净，且收益率正确（锚点会在收益比中抵消）。

用法
----
  python tests/fix_adj_from_em.py            # 抓因子 → 检查点 parquet
  python tests/fix_adj_from_em.py --merge    # 把因子乘入 prices.parquet 的 close_adj（先备份）

检查点：`data/real_universe/prices_adj_factor.parquet`（受 .gitignore 忽略）
断点续跑：重跑自动跳过已完成的 code。
"""
from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
import json

import numpy as np
import pandas as pd

PRICES = "data/real_universe/prices.parquet"
CKPT = "data/real_universe/prices_adj_factor.parquet"
END = "2026-09-10"
BEG_FLOOR = "2017-01-01"   # 回测首期 2017-05-15 之前即可；更早的 close_adj 不被使用
WORKERS = 6                # 并发线程数。实测 12 线程会被腾讯限流（后段大批失败），6 较稳。
BATCH = 25                 # 每批多少只；批完立即落盘并释放内存（避免 OOM 被 SIGKILL）
                           # 注意：单只内部 6 次请求是串行的（翻页依赖上一页），
                           # 故总耗时 ≈ 只数×6×0.46s/线程数；建议用 --limit 分多轮跑完。


def _fetch_kline(code: str, fq: str, beg: str, end: str) -> list:
    """抓单页 gtimg 日线。fq ∈ {'', 'qfq', 'hfq'}。返回 [[date,o,c,h,l,v], ...]。"""
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={code},day,{beg},{end},800,{fq}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/0.5"})
    for attempt in range(3):
        if DELAY:
            time.sleep(DELAY)
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=25).read())
            dd = d.get("data")
            if not isinstance(dd, dict):
                return []
            inner = dd.get(code, {})
            if not isinstance(inner, dict):
                return []
            key = {"": "day", "qfq": "qfqday", "hfq": "hfqday"}[fq]
            return inner.get(key) or []
        except Exception:
            time.sleep(1.0 + attempt)
    return []


def _fetch_series(code: str, fq: str, beg: str, end: str) -> dict:
    """翻页抓全段，返回 {date_str: close}。"""
    out: dict[str, float] = {}
    cur_end = end
    for _ in range(15):
        kl = _fetch_kline(code, fq, beg, cur_end)
        if not kl:
            break
        dates = []
        for r in kl:
            try:
                ts = pd.to_datetime(r[0])
            except Exception:
                continue
            if ts < pd.Timestamp(beg):
                continue
            out[ts.strftime("%Y-%m-%d")] = float(r[2])   # close
            dates.append(ts)
        if not dates or min(dates) <= pd.Timestamp(beg) + pd.Timedelta(days=1):
            break
        prev = (min(dates) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        cur_end = prev
        time.sleep(0.03)
    return out


def compute_factor(code: str, min_date: pd.Timestamp) -> dict:
    """同源抓 raw+qfq，返回干净复权因子 {date_str: factor}。"""
    beg = max(min_date.strftime("%Y-%m-%d"), BEG_FLOOR)
    raw = _fetch_series(code, "", beg, END)
    qfq = _fetch_series(code, "qfq", beg, END)
    fac: dict[str, float] = {}
    for d, r in raw.items():
        if d in qfq and r and qfq[d]:
            fac[d] = qfq[d] / r
    return fac


def _load_ckpt() -> pd.DataFrame:
    if __import__("os").path.exists(CKPT):
        return pd.read_parquet(CKPT)
    return pd.DataFrame(columns=["code", "date", "factor"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="把因子乘入 prices.parquet 的 close_adj")
    ap.add_argument("--limit", type=int, default=0, help="仅抓前 N 只（调试）")
    args = ap.parse_args()

    if args.merge:
        merge()
        return

    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    codes = (px.groupby("code")["date"].agg(["min", "max"]).reset_index())
    ckpt = _load_ckpt()
    done = set(ckpt["code"].unique()) if not ckpt.empty else set()
    print(f"[ckpt] 已完成 {len(done)} 只")

    targets = [c for c in codes.to_dict("records") if c["code"] not in done]
    if args.limit:
        targets = targets[: args.limit]
    print(f"[plan] 待抓 {len(targets)} 只（并发 {WORKERS} 线程，批大小 {BATCH}）")

    failed: list[str] = []

    def _work(t):
        code = t["code"]
        try:
            return code, compute_factor(code, t["min"])
        except Exception:
            return code, None

    from concurrent.futures import ThreadPoolExecutor
    import gc
    t0 = time.time()
    for start in range(0, len(targets), BATCH):
        batch = targets[start:start + BATCH]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(_work, batch))
        rows = []
        for code, fac in results:
            if not fac:
                failed.append(code)
                continue
            rows.append(pd.DataFrame(
                [{"code": code, "date": k, "factor": v} for k, v in fac.items()]))
        if rows:
            new = pd.concat(rows, ignore_index=True)
            cur = _load_ckpt()
            out = (pd.concat([cur, new], ignore_index=True)
                     .drop_duplicates(["code", "date"], keep="last"))
            out.to_parquet(CKPT, index=False)
            del new, out, cur, rows
            gc.collect()
        done_n = start + len(batch)
        print(f"  [{done_n}/{len(targets)}] 耗时 {time.time()-t0:.0f}s  "
              f"失败 {len(failed)}", flush=True)

    ck = _load_ckpt()
    print(f"[done] 检查点总行数 {len(ck)}，覆盖 {ck['code'].nunique()} 只")
    if failed:
        print(f"[warn] 失败 {len(failed)} 只（重跑自动重试）: {failed[:10]}"
              f"{'...' if len(failed) > 10 else ''}")


def merge():
    if not __import__("os").path.exists(CKPT):
        print("[merge] 无检查点，先抓取")
        return
    ckpt = pd.read_parquet(CKPT)
    ckpt = ckpt.dropna(subset=["factor"])
    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    ck = ckpt.copy()
    ck["date"] = pd.to_datetime(ck["date"])

    dst = PRICES + ".bak_adjfix"
    shutil.copy2(PRICES, dst)
    print(f"[bak] {PRICES} -> {dst}")

    merged = px.merge(ck[["code", "date", "factor"]], on=["code", "date"], how="left")
    mask = merged["factor"].notna() & (merged["factor"] > 0)
    merged.loc[mask, "close_adj"] = merged.loc[mask, "close_raw"] * merged.loc[mask, "factor"]
    print(f"[merge] 覆盖 {int(mask.sum())} 行 close_adj（共 {len(merged)} 行）")

    # 校验：新 close_adj/close_raw = factor，应 piecewise-constant（仅在除权日跳变）
    import random
    samp = random.sample(list(merged[mask]["code"].unique()),
                         min(6, merged[mask]["code"].nunique()))
    for code in samp:
        s = merged[merged["code"] == code].sort_values("date").copy()
        s["ratio"] = s["close_adj"] / s["close_raw"]
        chg = s["ratio"].pct_change().abs().dropna()
        print(f"  {code}: ratio均值 {s['ratio'].mean():.3f}  "
              f"日跳变最大 {chg.max()*100:.2f}%  >1%占比 {(chg>0.01).mean()*100:.1f}%")

    merged.to_parquet(PRICES, index=False)
    print(f"[done] 写回 prices.parquet；NaN close_adj 行: {int(merged['close_adj'].isna().sum())}")


if __name__ == "__main__":
    main()
