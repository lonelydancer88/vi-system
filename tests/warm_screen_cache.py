"""预热点：把回测所有调仓日 + 持仓截面日的 screen_at 结果落盘缓存。

动机：回测对 27 个调仓日各算一遍 L1→L5；30/5/3/equal 四档 + 指数基准 +
trades/reasons + screen 多次重跑，串行累计 ~40min，必撞本环境「~20min 硬杀长进程」。
落盘缓存后，四档只需第一次「真算」，其余秒级读盘。

用法：
    python3 tests/warm_screen_cache.py                      # 预热点 + 退出
    python3 tests/warm_screen_cache.py --asof 2026-09-07    # 额外预热某截面日
    python3 tests/warm_screen_cache.py --db data/real_universe
（可加 --background 以 start_new_session=True 脱离会话运行，日志写 out/_warm.log）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest import engine as bt
from vi_system.pipeline.screen_cache import warm, clear

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "data/real_universe"
OUT_LOG = ROOT / "out" / "_warm.log"


def main():
    ap = argparse.ArgumentParser(description="预热 screen_at 落盘缓存")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--asof", default=None,
                    help="额外预热的截面日（默认取 prices.parquet 最新交易日）")
    ap.add_argument("--background", action="store_true",
                    help="脱离会话后台跑（长任务防被会话轮次杀掉）")
    ap.add_argument("--no-clear", action="store_true",
                    help="保留已有缓存只补算缺失项（快，但代码级改动不会生效）")
    args = ap.parse_args()

    if args.background and os.environ.get("_WARM_DETACHED") != "1":
        OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "_WARM_DETACHED": "1", "PYTHONPATH": str(ROOT)}
        log = open(OUT_LOG, "a", encoding="utf-8")
        fwd = ["--db", args.db] + (["--asof", args.asof] if args.asof else []) \
            + (["--no-clear"] if args.no_clear else [])
        subprocess.Popen([sys.executable, __file__] + fwd,
                         cwd=ROOT, env=env, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"已在后台预热，日志：{OUT_LOG}")
        return

    db = Path(args.db)
    if not db.is_absolute():
        db = ROOT / db
    cfg = load_config()
    st = Store(db)

    if args.asof:
        extra = args.asof
    else:
        import pandas as pd

        p = pd.read_parquet(db / "prices.parquet", columns=["date"])
        extra = str(pd.to_datetime(p["date"]).max().date())

    bc = cfg.section("backtest")
    months = bc.get("rebalance_months", [5, 9])
    day = bc.get("rebalance_day", 15)
    dates = st.rebalance_dates(months, day, "2013-01-01", "2026-12-31")
    asofs = sorted({str(d.date()) for d in dates} | {extra})

    def log(msg):
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    # 默认清空重算：缓存 key 只含「数据 mtime + rules.yaml 指纹」，**不含代码版本**。
    # 若改了 screening 代码（如排雷/评分逻辑）而 data/config 未动，key 不变 ->
    # 旧缓存会被静默复用，报告反映的是旧代码结果。故默认清空，保证代码级改动也生效。
    # 确认代码未变、只想补算几个截面日时，用 --no-clear 走快速路径。
    log(f"开始预热点：{len(asofs)} 个 asof" + ("（先清旧缓存）" if not args.no_clear else "（保留已有缓存）"))
    if not args.no_clear:
        n = clear(st)
        log(f"已清旧缓存文件 {n} 个")
        bt.clear_screen_cache()
    t0 = time.time()
    warm(st, asofs, cfg, with_valuation=True, log=log)
    log(f"预热点完成：耗时 {time.time()-t0:.0f}s，缓存目录 {db / '.screen_cache'}")


if __name__ == "__main__":
    main()
