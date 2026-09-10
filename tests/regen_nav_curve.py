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
from vi_system.backtest.engine import run_tier

DB = ROOT / "data" / "real_universe"
OUT = ROOT / "out"
CACHE = OUT / "_navcache"

# label -> run_tier kwargs（统一入口：估值 verdict 回灌，与回测报告/CLI 同口径）
SPECS = {
    "30": {"label": "30只", "benchmark": "sh000300"},
    "5": {"label": "5只", "benchmark": "sh000300", "max_holdings": 5},
    "3": {"label": "3只", "benchmark": "sh000300", "max_holdings": 3},
    "equal": {"label": "等权", "benchmark": "equal"},
}


def _run_and_save(store, cfg, key: str) -> None:
    spec = SPECS[key]
    r = run_tier(store, cfg, **spec)
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
    print(f"[dump] {key} 完成：年化 {meta['cagr']:+.1%}，基准 {meta['bench_cagr']:+.1%}，{len(nav)} 期",
          flush=True)


def dump(key: str) -> None:
    store, cfg = Store(DB), load_config()
    _run_and_save(store, cfg, key)


def dump_all() -> None:
    """单进程跑完 4 档：共享 engine 的筛选缓存（三档筛选只做一遍）。

    比 4 个独立进程快约 3 倍，也避免并行抢 CPU 被回收。
    """
    import time as _t
    from vi_system.backtest import engine as _eng
    store, cfg = Store(DB), load_config()
    t0 = _t.time()
    for key in ["30", "5", "3", "equal"]:
        _run_and_save(store, cfg, key)
        print(f"  [进度] 累计 {_t.time()-t0:.0f}s，筛选缓存 {len(_eng._SCREEN_CACHE)} 期", flush=True)
    print(f"[dump-all] 全部完成，耗时 {_t.time()-t0:.0f}s", flush=True)


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


def plot_from_reports(asof: str = "2026-09-07") -> None:
    """直接从已生成的回测报告解析净值，不再重跑回测。

    为什么需要这条路径：`--dump` 要跑 4 次回测（各 ~10-15min），在本环境
    「约 20 分钟硬杀长进程」下实测全部被杀、缓存为空。而回测报告的每期标题里
    已经带了「组合收益 / 基准 / 累计净值」，足以还原曲线 —— 秒级完成。

    报告格式：`## 第 N 期：YYYY-MM-DD（组合收益 ±x%，基准 ±y%，超额 ±z%，累计净值 1.xxx）`
    """
    import re
    import pandas as pd

    pat = re.compile(
        r"## 第 \d+ 期：(\d{4}-\d{2}-\d{2})（组合收益 ([+-][\d.]+)%，"
        r"基准 ([+-][\d.]+)%，超额 ([+-][\d.]+)%，累计净值 ([\d.]+)）"
    )
    series, bench_nav = {}, None
    for key, fname in (("30", "top30"), ("5", "top5"), ("3", "top3")):
        path = OUT / f"回测报告-{fname}-{asof}.md"
        if not path.exists():
            print(f"[warn] 缺 {path.name}，跳过")
            continue
        rows = pat.findall(path.read_text(encoding="utf-8"))
        if not rows:
            print(f"[warn] {path.name} 未解析到期数据，跳过")
            continue
        dates = pd.to_datetime([r[0] for r in rows])
        strat = pd.Series([float(r[4]) for r in rows], index=dates)
        bret = pd.Series([float(r[2]) / 100.0 for r in rows], index=dates)
        series[key] = strat
        if bench_nav is None:  # 等权基准三份报告一致，取第一份即可
            bench_nav = (1 + bret).cumprod()
    if not series:
        print("[error] 无可用报告")
        sys.exit(1)

    # --- 期度对齐（关键，错位一期会让曲线整体偏移）
    # 回测调仓日共 41 个，实算起点 2017-05-15 = dates[12]；报告共 28 期。
    # 报告标题里的日期是**期初**，而「累计净值」是**期末**值。
    # 例：第1期标签 2017-05-15、净值 1.120，实际是 2017-05-15→2017-09-15 的收益。
    # 故净值点应画在「下一个调仓日」上，并在起点补一个 1.0。
    from vi_system.data.store import Store
    from vi_system.config import load_config
    _cfg = load_config()
    _bc = _cfg.section("backtest")
    _dates = Store(str(DB)).rebalance_dates(
        _bc.get("rebalance_months", [5, 9]), _bc.get("rebalance_day", 15),
        "2013-01-01", "2026-12-31")
    _dstr = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in _dates]
    # 每期标签日期 → 下一个调仓日（期末）
    end_dates, origin = [], None
    for i, (k, s) in enumerate(series.items()):
        idxs = [_dstr.index(pd.Timestamp(d).strftime("%Y-%m-%d")) for d in s.index]
        if origin is None:
            origin = _dates[idxs[0]]          # 实算起点（1.0 所在日）
        ends = pd.DatetimeIndex([_dates[j + 1] for j in idxs])
        series[k] = pd.Series(s.values, index=ends)
        end_dates = ends
    if bench_nav is not None:
        bench_nav = pd.Series(bench_nav.values, index=end_dates)

    # --- 沪深300：报告用的是等权基准，指数需单独从库里取并对齐同口径
    hs_nav = None
    try:
        idx = Store(str(DB)).load_index_prices()
        s = idx[idx["code"] == "sh000300"].set_index("date")["close"].sort_index()

        def _close_at(d):
            sub = s[s.index <= pd.Timestamp(d)]
            return float(sub.iloc[-1]) if len(sub) else float("nan")

        c0 = _close_at(origin)
        hs_nav = pd.Series([_close_at(d) / c0 for d in end_dates], index=end_dates)
    except Exception as e:
        print(f"[warn] 沪深300 取数失败：{e}")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"30": "#c0392b", "5": "#27ae60", "3": "#e67e22"}
    names = {"30": "策略·30只", "5": "策略·最多5只", "3": "策略·最多3只"}

    def _with_origin(s):
        """在起点补 1.0，使所有曲线同从 1 出发。"""
        return pd.concat([pd.Series([1.0], index=[origin]), s])

    for key in ("30", "5", "3"):
        if key not in series:
            continue
        s = _with_origin(series[key])
        cagr = s.iloc[-1] ** (1 / (len(s) / 3)) - 1
        ax.plot(s.index, s.values, lw=2.0 if key == "30" else 1.6,
                color=colors[key], label=f"{names[key]}（年化 {cagr:+.1%}）")
    if hs_nav is not None:
        s = _with_origin(hs_nav)
        cagr = s.iloc[-1] ** (1 / (len(s) / 3)) - 1
        ax.plot(s.index, s.values, lw=1.6, color="#2980b9",
                label=f"沪深300·价格回报（年化 {cagr:+.1%}）")
    if bench_nav is not None:
        s = _with_origin(bench_nav)
        cagr = s.iloc[-1] ** (1 / (len(s) / 3)) - 1
        ax.plot(s.index, s.values, lw=1.2, ls="--", color="#7f8c8d",
                label=f"等权全市场基准（年化 {cagr:+.1%}）")
    ax.axhline(1.0, color="gray", lw=0.6, alpha=0.6)
    ax.set_title(f"净值曲线（{pd.Timestamp(origin).date()} 起，剔除期初空仓期；"
                 f"分红再投口径，费用后；由回测报告还原）", fontsize=12)
    ax.set_ylabel("净值（起点=1）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "nav-curve.png", dpi=150)
    plt.close(fig)
    print(f"→ out/nav-curve.png（由 {len(series)} 份回测报告还原，{len(series[list(series)[0]])} 期）")


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    if "--plot" in sys.argv:
        plot()
    elif "--from-reports" in sys.argv:
        plot_from_reports()
    elif "--dump-all" in sys.argv:
        dump_all()
    elif "--dump" in sys.argv:
        dump(sys.argv[sys.argv.index("--dump") + 1])
    else:
        print(__doc__)
