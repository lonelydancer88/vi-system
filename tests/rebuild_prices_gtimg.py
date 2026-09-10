"""从干净源 gtimg 直连重建 prices.parquet 的 close_raw / close_adj。

背景
----
westock 落库的行情是脏的：不复权价 `close_raw` 对约 80 只票出现**负值**（股价不可能为负），
后复权价 `close_adj` 对全部票带每日 ~1%~3% 伪漂移（跨源锚点错位）。这些脏值直接污染回测
NAV 与「含分红」收益列，并曾导致中国动力「含分红 < 价格」的悖论、以及个别票 90 倍暴涨。

本脚本绕过 westock，直接从腾讯 gtimg（westock 的同源干净上游）抓取 raw + hfq 日线，
用干净值**整体替换** close_raw / close_adj，从而把数据做对。

设计
----
- 每票抓两序列：raw（不复权）与 hfq（后复权），按日期对齐 → (close_raw, close_adj)。
- gtimg 单次最多返回 2000 行，故每序列至多翻 6 页即可覆盖 2014-今。
- 断点续传：检查点 prices_gtimg_clean.parquet 按 (code,date) 累积，重跑自动跳过已完成票。
- 限流 + IP 封禁自适应退避：连续失败超阈值则长休，避免被腾讯拉黑。
- --merge：把干净值合并回 prices（先备份），重算 mktcap；完全失败的票从 prices 移除实现隔离。

用法
----
  python tests/rebuild_prices_gtimg.py            # 抓取 → 检查点（可重复跑，续传）
  python tests/rebuild_prices_gtimg.py --merge    # 合并干净值回 prices，并隔离失败票
  python tests/rebuild_prices_gtimg.py --check    # 打印检查点覆盖情况
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = "data/real_universe"
PRICES = os.path.join(ROOT, "prices.parquet")
CKPT = os.path.join(ROOT, "prices_gtimg_clean.parquet")        # (code,date,close_raw,close_adj)
EXCLUDED = os.path.join(ROOT, "prices_excluded.parquet")       # 完全抓取失败的票（隔离记录）

FLOOR = "2014-01-01"   # 回测首期 2017-05-15，留足动量/流动性回溯窗口
END = "2026-09-10"
CNT = 2000             # gtimg 单次上限（>2000 会返回错误结构）
WORKERS = 1            # 单线程：速率 ≈ 1 req/s（实测该速率不被限流），远低于 2 并发快取触发的封锁
BATCH = 20             # 每批票数；批完即落盘并释放内存（防 OOM/被 SIGKILL）
DELAY = 0.15           # 翻页间小间隔
REQ_SLEEP = 1.0        # 每请求间隔 1.0s → 单线程约 1 req/s，安全不触发限流
CAP = 0                 # 每轮最多抓 N 只（0=全部），由 --cap 覆盖


def _fetch_kline(code: str, fq: str, beg: str, end: str) -> list:
    """抓单页 gtimg 日线。fq ∈ {'','qfq','hfq'}。返回 [[date,o,c,h,l,v], ...]。

    注意：`fqkline/get` 端点已被腾讯下线（HTTP 501），改用同源的 `newfqkline/get`。
    """
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"
           f"?param={code},day,{beg},{end},{CNT},{fq}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
    for att in range(2):
        try:
            raw = urllib.request.urlopen(req, timeout=10).read()
            d = json.loads(raw)
            dd = d.get("data")
            if not isinstance(dd, dict):
                return []
            inner = dd.get(code, {})
            if not isinstance(inner, dict):
                return []
            key = {"": "day", "qfq": "qfqday", "hfq": "hfqday"}[fq]
            if REQ_SLEEP:
                time.sleep(REQ_SLEEP)
            return inner.get(key) or []
        except Exception:
            time.sleep(1.0 + att)
    return []


def _fetch_series(code: str, fq: str, beg: str, end: str) -> dict:
    """翻页抓全段，返回 {date_str: close}。"""
    out: dict[str, float] = {}
    cur_end = end
    for _ in range(6):
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
            try:
                out[ts.strftime("%Y-%m-%d")] = float(r[2])
            except (TypeError, ValueError):
                continue
            dates.append(ts)
        if not dates:
            break
        if min(dates) <= pd.Timestamp(beg) + pd.Timedelta(days=1):
            break
        cur_end = (min(dates) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if DELAY:
            time.sleep(DELAY)
    return out


def clean_one(code: str, min_date) -> list[dict]:
    """抓取单票干净 raw+hfq，返回行列表。"""
    beg = max(str(min_date)[:10], FLOOR)
    raw = _fetch_series(code, "", beg, END)
    hfq = _fetch_series(code, "hfq", beg, END)
    rows = []
    for d in raw:
        if d in hfq and np.isfinite(raw[d]) and np.isfinite(hfq[d]) and raw[d] > 0 and hfq[d] > 0:
            rows.append({"code": code, "date": d,
                         "close_raw": round(raw[d], 4), "close_adj": round(hfq[d], 4)})
    return rows


def _atomic_write(df: pd.DataFrame, path: str) -> None:
    """原子落盘：先写 .tmp 再 os.replace（防 kill 时写坏检查点，见 sina 版注释）。"""
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _load_ckpt() -> pd.DataFrame:
    if not os.path.exists(CKPT):
        return pd.DataFrame(columns=["code", "date", "close_raw", "close_adj"])
    try:
        return pd.read_parquet(CKPT)
    except Exception:
        os.replace(CKPT, CKPT + ".corrupt")
        print(f"[warn] 检查点损坏，已隔离为 {CKPT}.corrupt", flush=True)
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
    targets = [{"code": r["code"], "min": r["date"]}
               for _, r in meta.iterrows() if r["code"] not in done]
    # --cap：每轮只抓前 N 只，干净退出（避免长进程被杀），靠检查点续跑
    if CAP > 0:
        targets = targets[:CAP]
    print(f"[plan] 全宇宙 {len(meta)} 只，已完成 {len(done)}，本轮待抓 {len(targets)} "
          f"（并发 {WORKERS}，批 {BATCH}，间隔 {REQ_SLEEP}s）", flush=True)

    failed: list[str] = []
    consec_fail = 0
    sleep_stage = 0
    t0 = time.time()

    def _work(t):
        try:
            return t["code"], clean_one(t["code"], t["min"])
        except Exception:
            return t["code"], None

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
        # 连续失败过多 → 疑似被限流/封禁，指数退避降温（最长 15 分钟）
        if consec_fail >= 8:
            backoff = min(120 * (2 ** sleep_stage), 1800)
            print(f"  [!] 连续失败 {consec_fail}，疑似限流，休眠 {backoff}s 等待出口重置（第{sleep_stage+1}次退避）", flush=True)
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
    # 抽查一只
    if done:
        c = sorted(done)[0]
        sub = ck[ck.code == c]
        print(f"样例 {c}: {len(sub)} 行, raw {sub.close_raw.min():.2f}~{sub.close_raw.max():.2f}, "
              f"adj {sub.close_adj.min():.2f}~{sub.close_adj.max():.2f}, 负值="
              f"{(sub.close_raw<=0).sum()+(sub.close_adj<=0).sum()}")


def main_merge():
    if not os.path.exists(CKPT):
        print("[merge] 无检查点，先抓取")
        return
    ck = pd.read_parquet(CKPT)
    ck = ck.dropna(subset=["close_raw", "close_adj"])
    ck["date"] = pd.to_datetime(ck["date"])

    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    raw_n = len(px)

    # 备份
    bak = PRICES + ".bak_westock"
    shutil.copy2(PRICES, bak)
    print(f"[bak] {PRICES} -> {bak}")

    # 完全失败的票（在 prices 中但检查点完全无该 code）→ 隔离
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

    # 合并干净 close_raw / close_adj（按 code,date 对齐覆盖）
    merged = px.merge(ck[["code", "date", "close_raw", "close_adj"]],
                      on=["code", "date"], how="left",
                      suffixes=("", "_new"))
    cov = merged["close_raw_new"].notna()
    print(f"[merge] 对齐覆盖 {int(cov.sum())}/{len(merged)} 行 "
          f"（未覆盖 {(~cov).sum()} 行保留原值，多为 2014 年前）")
    merged.loc[cov, "close_raw"] = merged.loc[cov, "close_raw_new"]
    merged.loc[cov, "close_adj"] = merged.loc[cov, "close_adj_new"]
    merged = merged.drop(columns=["close_raw_new", "close_adj_new"])

    # 重算 mktcap = close_raw * total_share
    ts = pd.to_numeric(merged["total_share"], errors="coerce")
    merged["mktcap"] = merged["close_raw"] * ts

    # 校验
    neg = ((merged["close_raw"] <= 0) | (merged["close_adj"] <= 0)).sum()
    print(f"[validate] 负价行数: {int(neg)}（应为 0）")
    # 中国动力 sanity
    g = merged[merged.code == "sh600482"].sort_values("date")
    if not g.empty:
        b = g[g.date <= "2025-11-14"].iloc[-1]
        s = g[g.date <= "2026-05-15"].iloc[-1]
        pr = s.close_raw / b.close_raw - 1
        inc = s.close_adj / b.close_adj - 1
        print(f"[sanity] 中国动力 价格={pr*100:+.1f}% 含分红={inc*100:+.1f}% "
              f"（含分红应 > 价格）")

    # 回测窗口内（2017-05-15 起）不应再出现负价
    win = merged[merged.date >= "2017-05-15"]
    win_neg = ((win["close_raw"] <= 0) | (win["close_adj"] <= 0)).sum()
    print(f"[validate] 回测窗口内负价行数: {int(win_neg)}")

    merged.to_parquet(PRICES, index=False)
    print(f"[done] 写回 {PRICES}（{raw_n}→{len(merged)} 行，"
          f"剩余 {merged['code'].nunique()} 只）")

    # 同步更新 universe（删除被隔离票，保持 universe 与 prices 一致）
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
    ap.add_argument("--cap", type=int, default=0, help="每轮最多抓 N 只（0=全部），用于低速分批避开限流")
    args = ap.parse_args()
    global CAP
    CAP = args.cap
    if args.check:
        main_check()
    elif args.merge:
        main_merge()
    else:
        main_fetch()


if __name__ == "__main__":
    main()
