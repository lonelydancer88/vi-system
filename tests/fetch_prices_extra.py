"""补齐行情：抓取 codes.json 中 ALL 里 prices.parquet 缺失的票，
对齐历史跨度(2013-01-01 ~ 2026-09-07)，带断点续传 + 市值回填。

与 fetch_real.py 只抓沪深300不同，本脚本补齐 ALL(807) 中缺失行情的票
（中证500/中证1000 等），使 L1 宇宙能从 ~295 扩展到全 807。

特性：
  - 断点续传：每次重算 todo = ALL − prices 已有，中断重跑不重复抓取。
  - 批量落盘：每 BATCH 只 save_prices 一次，降低 parquet 反复读写开销。
  - save 容错：单次落盘失败仅记日志，不杀进程。
  - 进度日志：每完成一只写 fetch_prices_progress.log，便于观测/重启。
  - 市值回填：抓完后用财报股本 point-in-time 回填 mktcap/total_share。
"""
from __future__ import annotations

import json
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
CODES = json.load(open(ROOT / "codes.json"))
ALL = CODES["all"]
PRICE_START = "2013-01-01"
PRICE_END = "2026-09-07"
BATCH = 20


def _safe_save(st, df, log):
    try:
        st.save_prices(df)
        return True
    except Exception as e:
        log.write(f"[err] save_prices failed: {e}\n")
        log.flush()
        return False


def main():
    f = WeStockFetcher(timeout=180)
    st = Store(str(ROOT))
    log = open(ROOT / "fetch_prices_extra.log", "a", encoding="utf-8")
    prog = open(ROOT / "fetch_prices_progress.log", "a", encoding="utf-8")
    done = set(st.load_prices()["code"].unique()) if st.prices_path.exists() else set()
    todo = [c for c in ALL if c not in done]
    print(f"[补齐行情] 已有 {len(done)} 只, 待抓 {len(todo)} 只", flush=True)
    t0 = time.time()
    completed = 0
    failed = []
    buf = []
    wx = threading.Lock()

    def _w(code):
        try:
            return code, f.prices(code, PRICE_START, PRICE_END)
        except Exception:
            return code, None

    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_w, c): c for c in todo}
            for fu in as_completed(futs):
                code, d = fu.result()
                completed += 1
                ok = d is not None and not d.empty
                prog.write(f"{time.strftime('%H:%M:%S')} {completed}/{len(todo)} {code} {'OK' if ok else 'FAIL'}\n")
                prog.flush()
                if ok:
                    with wx:
                        buf.append(d)
                        if len(buf) >= BATCH:
                            if _safe_save(st, pd.concat(buf, ignore_index=True), log):
                                buf.clear()
                else:
                    failed.append(code)
                if completed % 10 == 0:
                    el = time.time() - t0
                    rate = completed / el if el else 0
                    eta = (len(todo) - completed) / rate / 60 if rate else 0
                    print(f"  进度 {completed}/{len(todo)} ({rate:.2f}/s, ETA {eta:.0f}min, 失败{len(failed)})",
                          flush=True)
    finally:
        if buf:
            _safe_save(st, pd.concat(buf, ignore_index=True), log)
        if failed:
            log.write(f"[warn] 失败 {len(failed)}: {failed}\n")
        # 市值回填（用财报股本 point-in-time 回填 mktcap/total_share）
        try:
            px = st.load_prices()
            fx = st.load_facts()
            merged = f.attach_shares(px, fx)
            if _safe_save(st, merged, log):
                print(f"[市值] 回填完成 mktcap有效 {int(merged['mktcap'].notna().sum())}/{len(merged)}",
                      flush=True)
        except Exception as e:
            log.write(f"[warn] attach_shares: {e}\n")
        print(f"[补齐] 完成 累计 prices {len(st.load_prices()['code'].unique())} 只", flush=True)
        log.close()
        prog.close()


if __name__ == "__main__":
    main()
