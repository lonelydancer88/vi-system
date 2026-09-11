"""预热点：把回测所有调仓日 + 持仓截面日的 screen_at 结果落盘缓存。

动机：回测对 27 个调仓日各算一遍 L1→L5；30/5/3/equal 四档 + 指数基准 +
trades/reasons + screen 多次重跑，串行累计 ~40min，必撞本环境「~20min 硬杀长进程」。
落盘缓存后，四档只需第一次「真算」，其余秒级读盘。

用法：
    python3 tests/warm_screen_cache.py            # 预热点 + 退出
（以 start_new_session=True 脱离会话运行，日志写 out/_warm.log，供外部轮询）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest import engine as bt
from vi_system.pipeline.screen_cache import warm, clear

DB = Path(__file__).resolve().parent.parent / "data" / "real_universe"
OUT_LOG = Path(__file__).resolve().parent.parent / "out" / "_warm.log"


def main():
    cfg = load_config()
    st = Store(DB)
    bc = cfg.section("backtest")
    months = bc.get("rebalance_months", [5, 9])
    day = bc.get("rebalance_day", 15)
    dates = st.rebalance_dates(months, day, "2013-01-01", "2026-12-31")
    asofs = [str(d.date()) for d in dates] + ["2026-09-09"]

    def log(msg):
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    log(f"开始预热点：{len(asofs)} 个 asof（清旧缓存再算）")
    n = clear(st)
    log(f"已清旧缓存文件 {n} 个")
    bt.clear_screen_cache()
    t0 = time.time()
    warm(st, asofs, cfg, with_valuation=True, log=log)
    log(f"预热点完成：耗时 {time.time()-t0:.0f}s，缓存目录 {DB / '.screen_cache'}")


if __name__ == "__main__":
    main()
