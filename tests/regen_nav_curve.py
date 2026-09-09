"""重生成 out/nav-curve.png（可拆分并行版）。

为什么不用 tests/regen_charts.py：
  它在一个进程里**串行**跑 4 次回测（30/3/5 只 + 等权全市场）。宇宙扩到 780 只后
  单次约 9.5 分钟，串行 ~38 分钟，必撞本环境「约 20 分钟硬杀长进程」的限制（实测被杀）。
  Python 线程受 GIL 限制无法并行 CPU 密集任务，故改为**多进程拆分**：
    阶段一：4 个独立进程各跑一次回测，把 nav/bench 序列缓存到 out/_navcache_*.csv
    阶段二：读取 4 份缓存画图（秒级）

用法
----
    # 阶段一：4 条命令可同时跑（各自独立进程，互不阻塞）
    python3 tests/regen_nav_curve.py --dump 30
    python3 tests/regen_nav_curve.py --dump 5
    python3 tests/regen_nav_curve.py --dump 3
    python3 tests/regen_nav_curve.py --dump equal
    # 阶段二：四份缓存齐了再画图
    python3 tests/regen_nav_curve.py --plot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("/System/Library/Fonts/PingFang.ttc",
           "/System/Library/Fonts/Hiragino Sans GB.ttc",
           "/System/Library/Fonts/STHeiti Medium.ttc"):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest.engine import run_backtest

DB = ROOT / "data" / "real_universe"
OUT = ROOT / "out"
CACHE = OUT / "_navcache"

# label -> (run_backtest kwargs)
SPECS = {
    "30": {"label": "30只", "benchmark": "sh000300"},
    "5": {"label": "5只", "benchmark": "sh000300", "max_holdings": 5},
    "3": {"label": "3只", "benchmark": "sh000300", "max_holdings": 3},
    "equal": {"label": "等权", "benchmark": "equal"},
}


def dump(key: str) -> None:
    spec = SPECS[key]
    store, cfg = Store(DB), load_config()
    r = run_backtest(store, cfg, "2013-01-01", "2026-12-31", **spec)
    if "error" in r:
        print(f"[error] {key}: {r['error']}")
        sys.exit(1)
    nav = r["nav"]
    nav.to_csv(CACHE / f"nav_{key}.csv", index=True)
    meta = {
        "label": spec["label"],
        "cagr": r["stats"]["cagr"],
        "bench_cagr": r["stats"]["bench_cagr"],
        "effective_start": str(r["effective_start"]),
    }
    (CACHE / f"meta_{key}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[dump] {key} 完成：年化 {meta['cagr']:+.1%}，基准 {meta['bench_cagr']:+.1%}，{len(nav)} 期")


def plot() -> None:
    keys = ["30", "5", "3", "equal"]
    miss = [k for k in keys if not (CACHE / f"nav_{k}.csv").exists()]
    if miss:
        print(f"[error] 缺少缓存 {miss}，请先跑 --dump")
        sys.exit(1)
    import pandas as pd
    data, metas = {}, {}
    for k in keys:
        df = pd.read_csv(CACHE / f"nav_{k}.csv", index_col=0, parse_dates=True)
        data[k] = df
        metas[k] = json.loads((CACHE / f"meta_{k}.json").read_text(encoding="utf-8"))

    base = data["30"].index
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(base, data["30"]["nav"].reindex(base), lw=2.0, color="#c0392b",
            label=f"策略·30只（年化 {metas['30']['cagr']:+.1%}）")
    ax.plot(base, data["5"]["nav"].reindex(base), lw=1.6, color="#27ae60",
            label=f"策略·最多5只（年化 {metas['5']['cagr']:+.1%}）")
    ax.plot(base, data["3"]["nav"].reindex(base), lw=1.6, color="#e67e22",
            label=f"策略·最多3只（年化 {metas['3']['cagr']:+.1%}）")
    ax.plot(base, data["30"]["bench"].reindex(base), lw=1.6, color="#2980b9",
            label=f"沪深300·价格回报（年化 {metas['30']['bench_cagr']:+.1%}）")
    ax.plot(base, data["equal"]["bench"].reindex(base), lw=1.2, ls="--", color="#7f8c8d",
            label=f"等权全市场基准（年化 {metas['equal']['bench_cagr']:+.1%}）")
    ax.axhline(1.0, color="gray", lw=0.6, alpha=0.6)
    ax.set_title(f"净值曲线（{metas['30']['effective_start']} 起，剔除期初空仓期；"
                 f"分红再投口径，费用后）", fontsize=12)
    ax.set_ylabel("净值（起点=1）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "nav-curve.png", dpi=150)
    plt.close(fig)
    print("→ out/nav-curve.png（已重生成）")


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    if "--plot" in sys.argv:
        plot()
    elif "--dump" in sys.argv:
        dump(sys.argv[sys.argv.index("--dump") + 1])
    else:
        print(__doc__)
