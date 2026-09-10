"""从新浪(sina)独立重建 prices.parquet 的 close_raw / close_adj。

背景
----
gtimg 直连虽干净但被腾讯按累计请求持久限流（卡在 575/802）。新浪与本机代理
(127.0.0.1:63462) 互通、且与腾讯是**完全独立的第二源**，不受其限流牵制。

新浪两个明文端点（已验证可用、全历史、无 250/640 行截断）：
- 不复权日线: money.finance.sina.com.cn/quotes_service/api/json_v2.php/
               CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen=8000
- 后复权因子: finance.sina.com.cn/realstock/company/{code}/hfq.js
               → var sh600519hfq={"total":N,"data":[{"d":日期,"f":因子}]}
  后复权价 = 不复权收盘 × 因子(取该日及之前最近一次除权日的因子)

设计
----
- 每票 2 请求（raw + hfq），单线程低速(~1.5 req/s)避免新浪限流。
- 断点续传：检查点 prices_sina_clean.parquet 按 (code,date) 累积，重跑自动跳过。
- 失败自适应退避：连续失败超阈值长休，避免被封。
- --merge：把干净值整体替换回 prices 的 close_raw/close_adj，重算 mktcap，
  完全失败票隔离到 prices_sina_excluded.parquet。

用法
----
  python tests/rebuild_prices_sina.py            # 抓取 → 检查点（可重复跑，续传）
  python tests/rebuild_prices_sina.py --cap 5    # 小批量试跑
  python tests/rebuild_prices_sina.py --check    # 打印检查点覆盖
  python tests/rebuild_prices_sina.py --merge    # 合并干净值回 prices
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = "data/real_universe"
PRICES = os.path.join(ROOT, "prices.parquet")
CKPT = os.path.join(ROOT, "prices_sina_clean.parquet")          # (code,date,close_raw,close_adj)
GT_CKPT = os.path.join(ROOT, "prices_gtimg_clean.parquet")      # 腾讯检查点（用于并集去重）
EXCLUDED = os.path.join(ROOT, "prices_sina_excluded.parquet")   # 完全抓取失败的票

FLOOR = "2014-01-01"   # 回测首期 2017-05-15，留足动量/流动性回溯窗口
END = "2026-09-10"
DATALEN = 4000         # 减小返回体积：4000 行≈2013→今，覆盖回测窗口且传输更快
WORKERS = 1            # 单线程：最低并发，避免新浪限流（用户要求调低）
BATCH = 20             # 每批票数；批完即落盘防 OOM
REQ_SLEEP = 0.5        # 每请求间隔 0.5s
CAP = 0                # 每轮最多抓 N 只（0=全部），由 --cap 覆盖
ONLY_MISSING = False   # 只补"新浪∪腾讯"并集仍缺的票（最小化请求量，降低再限流风险）

_UA = {"User-Agent": "Mozilla/5.0",
       "Referer": "https://finance.sina.com.cn/"}


def _get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("gbk", "ignore")


def _sina_raw(code: str) -> dict:
    """返 {date_str: close}，全历史不复权收盘。"""
    url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen={DATALEN}")
    arr = json.loads(_get(url))
    out: dict[str, float] = {}
    for r in arr:
        try:
            out[str(r["day"])] = float(r["close"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


def _sina_hfq_factor(code: str) -> dict | None:
    """返 {ex_date_str: 后复权因子}，无分红返回 None。"""
    txt = _get(f"https://finance.sina.com.cn/realstock/company/{code}/hfq.js")
    m = re.search(r"=\s*(\{.*\})\s*;?", txt, re.S)
    if not m:
        return None
    obj = json.loads(m.group(1))
    data = obj.get("data") or []
    if not data:
        return None
    return {str(d["d"]): float(d["f"]) for d in data}


def clean_one(code: str) -> list[dict] | None:
    """抓单票 raw+hfq 因子 → 行列表；任一序列失败返回 None（整票隔离）。"""
    raw = _sina_raw(code)
    if not raw:
        return None
    fac = _sina_hfq_factor(code)
    fac_dates = sorted(fac) if fac else []
    fac_df = (pd.DataFrame({"d": pd.to_datetime(fac_dates),
                            "f": [fac[k] for k in fac_dates]}).sort_values("d")
              if fac else pd.DataFrame(columns=["d", "f"]))
    rows = []
    for d, c in raw.items():
        if c <= 0:
            continue
        cd = pd.Timestamp(d)
        if cd < pd.Timestamp(FLOOR):
            continue
        if fac_df.empty:
            adj = c
        else:
            fv = fac_df.loc[fac_df["d"] <= cd, "f"]
            if fv.empty:
                fv = fac_df["f"].iloc[0]          # 早于首个除权日 → 用首个因子
            adj = c * float(fv.iloc[-1])
        rows.append({"code": code, "date": d,
                     "close_raw": round(c, 4), "close_adj": round(adj, 4)})
    return rows or None


def _atomic_write(df: pd.DataFrame, path: str) -> None:
    """原子落盘：先写 .tmp 再 os.replace。

    关键：进程被 kill 时若正写 parquet，会留下无 footer 的损坏文件（整个检查点报废）。
    原子替换保证主文件永远是"上一次完整状态"，最多丢当前批次。
    """
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _load_ckpt() -> pd.DataFrame:
    if not os.path.exists(CKPT):
        return pd.DataFrame(columns=["code", "date", "close_raw", "close_adj"])
    try:
        return pd.read_parquet(CKPT)
    except Exception:
        # 检查点损坏 → 隔离留证，本轮从头重抓（宁可重抓也不能读到坏数据）
        os.replace(CKPT, CKPT + ".corrupt")
        print(f"[warn] 检查点损坏，已隔离为 {CKPT}.corrupt，本轮从头重抓", flush=True)
        return pd.DataFrame(columns=["code", "date", "close_raw", "close_adj"])


def _done_codes() -> set:
    if not os.path.exists(CKPT):
        return set()
    try:
        return set(pd.read_parquet(CKPT, columns=["code"])["code"].unique())
    except Exception:
        return set()


def main_fetch():
    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    meta = px.groupby("code")["date"].min().reset_index()
    done = _done_codes()
    if ONLY_MISSING:
        # 只补"新浪∪腾讯"并集仍缺的票：请求量最小化，最不容易再触发限流。
        # 单只票的 raw/adj 来自同一源（自洽），跨股票混源不影响回测收益。
        gt: set = set()
        if os.path.exists(GT_CKPT):
            try:
                gt = set(pd.read_parquet(GT_CKPT, columns=["code"])["code"].unique())
            except Exception:
                gt = set()
        print(f"[only-missing] 腾讯已覆盖 {len(gt)} 只，并入已完成集合", flush=True)
        done = done | gt
    targets = [r["code"] for _, r in meta.iterrows() if r["code"] not in done]
    if CAP > 0:
        targets = targets[:CAP]
    print(f"[plan] 全宇宙 {len(meta)} 只，已完成 {len(done)}，本轮待抓 {len(targets)} "
          f"（并发 {WORKERS}，批 {BATCH}，间隔 {REQ_SLEEP}s）", flush=True)

    failed: list[str] = []
    consec_fail = 0
    sleep_stage = 0
    t0 = time.time()

    def _work(code):
        try:
            return code, clean_one(code)
        except Exception:
            return code, None

    for start in range(0, len(targets), BATCH):
        batch = targets[start:start + BATCH]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(_work, batch))
        rows = []
        batch_fail = 0
        for code, rec in results:
            if not rec:
                failed.append(code)
                batch_fail += 1
                consec_fail += 1
                continue
            consec_fail = 0
            rows.append(pd.DataFrame(rec))
        if rows:
            new = pd.concat(rows, ignore_index=True)
            cur = _load_ckpt()
            out = (pd.concat([cur, new], ignore_index=True)
                     .drop_duplicates(["code", "date"], keep="last"))
            _atomic_write(out, CKPT)
            del new, out, cur, rows
            gc.collect()
        done_n = start + len(batch)
        if consec_fail >= 8:
            backoff = min(120 * (2 ** sleep_stage), 1800)
            print(f"  [!] 连续失败 {consec_fail}，疑似限流，休眠 {backoff}s（第{sleep_stage+1}次退避）", flush=True)
            time.sleep(backoff)
            consec_fail = 0
            sleep_stage += 1
        else:
            sleep_stage = 0
        print(f"  [{done_n}/{len(targets)}] 耗时 {time.time()-t0:.0f}s "
              f"本批失败 {batch_fail} 累计失败 {len(failed)}", flush=True)

    ck = _load_ckpt()
    print(f"[done] 检查点总行数 {len(ck)}，覆盖 {ck['code'].nunique()} 只")
    if failed:
        print(f"[warn] 失败 {len(failed)} 只: {failed[:20]}")


def main_check():
    px = pd.read_parquet(PRICES)
    ck = _load_ckpt() if os.path.exists(CKPT) else None
    if ck is None or ck.empty:
        print("检查点为空，尚未抓取。")
        return
    all_codes = set(px["code"].unique())
    done = set(ck["code"].unique())
    print(f"全宇宙 {len(all_codes)} 只，已抓取 {len(done)} 只，待抓 {len(all_codes-done)} 只")
    if done:
        c = sorted(done)[0]
        sub = ck[ck.code == c]
        print(f"样例 {c}: {len(sub)} 行, raw {sub.close_raw.min():.2f}~{sub.close_raw.max():.2f}, "
              f"adj {sub.close_adj.min():.2f}~{sub.close_adj.max():.2f}, 负值="
              f"{int((sub.close_raw<=0).sum()+(sub.close_adj<=0).sum())}")


def main_merge():
    if not os.path.exists(CKPT):
        print("[merge] 无检查点，先抓取")
        return
    ck = pd.read_parquet(CKPT).dropna(subset=["close_raw", "close_adj"])
    ck["date"] = pd.to_datetime(ck["date"])

    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    raw_n = len(px)

    bak = PRICES + ".bak_westock"
    if not os.path.exists(bak):
        shutil.copy2(PRICES, bak)
        print(f"[bak] {PRICES} -> {bak}")

    clean_codes = set(ck["code"].unique())
    px_codes = set(px["code"].unique())
    failed_codes = px_codes - clean_codes
    if failed_codes:
        exc = px[px["code"].isin(failed_codes)]
        exc.to_parquet(EXCLUDED, index=False)
        px = px[~px["code"].isin(failed_codes)].copy()
        print(f"[isolate] 移除 {len(failed_codes)} 只完全失败票（存 {EXCLUDED}）: "
              f"{sorted(failed_codes)[:20]}")
    else:
        print("[isolate] 无完全失败票")

    merged = px.merge(ck[["code", "date", "close_raw", "close_adj"]],
                      on=["code", "date"], how="left", suffixes=("", "_new"))
    cov = merged["close_raw_new"].notna()
    print(f"[merge] 对齐覆盖 {int(cov.sum())}/{len(merged)} 行 "
          f"（未覆盖 {(~cov).sum()} 行保留原值，多为 2014 年前）")
    merged.loc[cov, "close_raw"] = merged.loc[cov, "close_raw_new"]
    merged.loc[cov, "close_adj"] = merged.loc[cov, "close_adj_new"]
    merged = merged.drop(columns=["close_raw_new", "close_adj_new"])

    ts = pd.to_numeric(merged["total_share"], errors="coerce")
    merged["mktcap"] = merged["close_raw"] * ts

    neg = int(((merged["close_raw"] <= 0) | (merged["close_adj"] <= 0)).sum())
    print(f"[validate] 负价行数: {neg}（应为 0）")

    g = merged[merged.code == "sh600482"].sort_values("date")
    if not g.empty:
        b = g[g.date <= "2017-05-15"].iloc[-1]
        s = g[g.date <= "2026-05-15"].iloc[-1]
        pr = s.close_raw / b.close_raw - 1
        inc = s.close_adj / b.close_adj - 1
        print(f"[sanity] 中国动力 价格={pr*100:+.1f}% 含分红={inc*100:+.1f}% "
              f"（含分红应 > 价格）")

    win = merged[merged.date >= "2017-05-15"]
    win_neg = int(((win["close_raw"] <= 0) | (win["close_adj"] <= 0)).sum())
    print(f"[validate] 回测窗口内负价行数: {win_neg}")

    merged.to_parquet(PRICES, index=False)
    print(f"[done] 写回 {PRICES}（{raw_n}→{len(merged)} 行，"
          f"剩余 {merged['code'].nunique()} 只）")

    uni_path = os.path.join(ROOT, "universe.parquet")
    if os.path.exists(uni_path) and failed_codes:
        u = pd.read_parquet(uni_path)
        before = len(u)
        u = u[~u["code"].isin(failed_codes)].copy()
        u.to_parquet(uni_path, index=False)
        print(f"[universe] {before}→{len(u)} 行（移除隔离票）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="合并干净值回 prices 并隔离失败票")
    ap.add_argument("--check", action="store_true", help="打印检查点覆盖情况")
    ap.add_argument("--cap", type=int, default=0, help="每轮最多抓 N 只（0=全部）")
    ap.add_argument("--only-missing", action="store_true",
                    help="只补新浪∪腾讯并集仍缺的票（最小请求量，防再限流）")
    args = ap.parse_args()
    global CAP, ONLY_MISSING
    CAP = args.cap
    ONLY_MISSING = args.only_missing
    if args.check:
        main_check()
    elif args.merge:
        main_merge()
    else:
        main_fetch()


if __name__ == "__main__":
    main()
