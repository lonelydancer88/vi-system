"""真实大宇宙数据抓取（带断点续传）。

抓取并落盘到 data/real_universe/：
  - facts.parquet   : 全部 807 只（沪深300+中证500+造假股）三大报表 + 质押
  - prices.parquet  : 沪深300（300 只）日线（原始+后复权，翻页突破250行）

支持断点续传：已落盘的代码自动跳过，中断后重跑不重复抓取。
用法：
  python tests/fetch_real.py              # 全量（财务 + 沪深300行情）
  python tests/fetch_real.py --skip-prices   # 只抓财务
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
from vi_system.data.westock import WeStockFetcher
from vi_system.data.store import Store

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23/data/real_universe")
CODES = json.loads((ROOT / "codes.json").read_text(encoding="utf-8"))
ALL = CODES["all"]
HS300 = CODES["hs300"]
PRICE_START = "2013-01-01"
PRICE_END = "2026-09-07"


def _done_codes(store: Store, which: str) -> set:
    if which == "facts":
        if not store.facts_path.exists():
            return set()
        df = pd.read_parquet(store.facts_path)
        return set(df["code"].unique())
    else:
        if not store.prices_path.exists():
            return set()
        df = pd.read_parquet(store.prices_path)
        return set(df["code"].unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prices", action="store_true")
    args = ap.parse_args()

    f = WeStockFetcher(timeout=180)
    st = Store(str(ROOT))
    log = open(ROOT / "fetch.log", "a", encoding="utf-8")

    # ============================ 1) 财务（全部 807 只）
    done_fin = _done_codes(st, "facts")
    todo = [c for c in ALL if c not in done_fin]
    print(f"[财务] 已完成 {len(done_fin)} / 共 {len(ALL)}，待抓 {len(todo)}", flush=True)
    t0 = time.time()
    for i, code in enumerate(todo):
        try:
            fin = f.financials([code], num=52)
            if not fin.empty:
                st.save_facts(fin)
            try:
                pl = f.pledge([code])
                if not pl.empty:
                    st.save_facts(pl)
            except Exception:
                pass
        except Exception as e:
            log.write(f"[warn] fin {code}: {e}\n")
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"   财务进度 {i+1}/{len(todo)} ({rate:.1f}/s)", flush=True)
    print(f"[财务] 完成，累计 {len(_done_codes(st,'facts'))} 只", flush=True)

    if args.skip_prices:
        log.close()
        return

    # ============================ 2) 行情（沪深300，300 只，并发翻页）
    done_px = _done_codes(st, "prices")
    todo_px = [c for c in HS300 if c not in done_px]
    print(f"[行情] 已完成 {len(done_px)} / 共 {len(HS300)}，待抓 {len(todo_px)}", flush=True)
    t0 = time.time()
    wx = threading.Lock()
    completed = 0
    failed = []

    def _worker(code):
        try:
            d = f.prices(code, PRICE_START, PRICE_END)
            return code, d
        except Exception as e:
            return code, None

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_worker, c): c for c in todo_px}
        for fu in as_completed(futs):
            code, d = fu.result()
            completed += 1
            if d is not None and not d.empty:
                with wx:
                    st.save_prices(d)
            else:
                failed.append(code)
            if completed % 10 == 0:
                el = time.time() - t0
                rate = completed / el if el else 0
                eta = (len(todo_px) - completed) / rate / 60 if rate else 0
                print(f"   行情进度 {completed}/{len(todo_px)} ({rate:.2f}/s, ETA {eta:.0f}min, 失败{len(failed)})",
                      flush=True)
    if failed:
        log.write(f"[warn] 行情失败 {len(failed)} 只: {failed}\n")

    # 市值回填：prices() 里 total_share/mktcap 是刻意留空的，必须用财报股本按
    # 公告日 point-in-time 回填后再算 close × total_share。漏掉这一步 → mktcap
    # 恒 NaN → compute_metrics 跳过所有股票 → 回测结果为空。
    try:
        px = st.load_prices()
        fx = st.load_facts()
        merged = f.attach_shares(px, fx)
        st.save_prices(merged)
        print(f"[市值] 已回填：mktcap 有效 {int(merged['mktcap'].notna().sum())} / {len(merged)} 行", flush=True)
    except Exception as e:
        log.write(f"[warn] attach_shares: {e}\n")

    # ---------------------------------------------------------------- 股本修正
    # 第三类股本 bug：westock 的 total_share 在大比例送转/增发/借壳后会停在旧值
    # （分众 3.28 亿 vs 真实 144.42 亿，差 44 倍）。**每次重抓 facts 都会把此前
    # 修好的股本覆盖回退**，故必须抓完立即重跑修正 —— 否则市值全错、价值因子失真，
    # 且不会报错（静默失效）。历史教训：2026-09-09 即因此中招。
    try:
        proj = ROOT.parent
        env = dict(os.environ, PYTHONPATH=str(proj))
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "fix_shares_from_em.py")],
            cwd=str(proj), env=env, capture_output=True, text=True, timeout=1800,
        )
        if r.returncode == 0:
            print("[股本] 已重跑 fix_shares_from_em（防重抓覆盖回退）", flush=True)
        else:
            log.write(f"[warn] fix_shares_from_em 退出码 {r.returncode}: {r.stderr[-500:]}\n")
            print(f"[股本] ⚠️ 修正失败（退出码 {r.returncode}），市值可能失真！", flush=True)
    except Exception as e:
        log.write(f"[warn] fix_shares_from_em: {e}\n")
        print(f"[股本] ⚠️ 修正异常：{e}", flush=True)

    print(f"[行情] 完成，累计 {len(_done_codes(st,'prices'))} 只", flush=True)
    print("全部抓取完成。", flush=True)
    log.close()


if __name__ == "__main__":
    main()
