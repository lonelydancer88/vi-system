"""季报/中报/三季报全量补抓（append 到 data/real_universe/facts.parquet）。

前提：
  - vi_system/data/westock.py 的 financials() 现已保留 0331/0630/0930/1231 全部报告期
    （原始 YTD 累计值落盘，TTM 折算在 metrics 层完成）。
  - store.save_facts 已做 append-merge + (code,field,period,announce_date) 去重。

续传逻辑：某 code 一旦 facts 中已存在任意季报期(0331/0630/0930)即视为已完成，
跳过；故首次跑会补抓全部 807 只的季报，中途中断重跑只补未完成的。

用法：
  python tests/fetch_quarterly.py
"""
from __future__ import annotations
import json
import sys
import time
import shutil
from pathlib import Path
import argparse

import pandas as pd

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23/data/real_universe")
sys.path.insert(0, "/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
from vi_system.data.westock import WeStockFetcher
from vi_system.data.store import Store

st = Store(str(ROOT))
CODES = json.loads((ROOT / "codes.json").read_text(encoding="utf-8"))
ALL = CODES["all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="试点：只抓前 N 只（0=全量）")
    args = ap.parse_args()
    # 备份现有 facts（.bak 受 .gitignore 忽略，安全）
    src = ROOT / "facts.parquet"
    if src.exists():
        bak = ROOT / "facts.parquet.bak"
        if not bak.exists():
            shutil.copy(src, bak)
            print(f"[备份] facts.parquet -> facts.parquet.bak ({src.stat().st_size/1e6:.1f}MB)")

    # 已含季报的 code 集合（一次加载，避免 807 次读盘）
    df = st.load_facts()
    q_mask = df["period"].astype(str).str.endswith(("0331", "0630", "0930"))
    done = set(df[q_mask]["code"].unique())
    todo = [c for c in ALL if c not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[季报] 已完成 {len(done)} / 共 {len(ALL)}，待抓 {len(todo)}", flush=True)

    f = WeStockFetcher(timeout=180)
    t0 = time.time()
    n_ok = 0
    for i, code in enumerate(todo):
        fin = None
        for attempt in range(4):
            try:
                fin = f.financials([code], num=52)
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        if fin is not None and not fin.empty:
            st.save_facts(fin)
            n_ok += 1
        # 质押快照：幂等更新（已有则去重跳过）
        try:
            pl = f.pledge([code])
            if pl is not None and not pl.empty:
                st.save_facts(pl)
        except Exception:
            pass
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            rate = (i + 1) / el if el else 0
            eta = (len(todo) - (i + 1)) / rate / 60 if rate else 0
            print(f"   进度 {i+1}/{len(todo)} ({rate:.2f}/s, ETA {eta:.0f}min, 成功{n_ok})", flush=True)

    print(f"[季报] 完成：成功补抓 {n_ok} 只，累计季报 code = {len(done) + n_ok}", flush=True)


if __name__ == "__main__":
    main()
