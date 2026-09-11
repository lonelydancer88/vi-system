"""反事实检验：本系统是否「用未来数据预测今天」（后视镜 / look-ahead bias）。

方法
----
对策略注入一个**人为的前视**：把每个调仓日的财务可见截止日 `asof` 往后推
LOOKAHEAD 天（默认 365，等于偷看未来一整年的财报——年报 + 中报 + 季报）。
然后跑三档回测，与「干净版」（不带前视、命中既有磁盘缓存）逐档对比。

判读逻辑
--------
- 若注入未来数据后年化 / 超额**大幅跳升** → 说明未来财务对这套策略极有价值，
  而干净版根本没用它 → 干净版没有前视、其业绩是「保守/低估」的。
- 若注入后**几乎没变** → 要么策略是噪声，要么干净版本身已泄漏（需进一步查）。

防坑
----
screen 缓存 key = root名 + asof + 数据mtime + rules指纹，**不含 facts 内容**。
若注入版命中干净版缓存，测试作废。故注入版必须绕过磁盘缓存（patch get_screen
直接调 compute()），并在切换前清空 engine._SCREEN_CACHE。
"""
from __future__ import annotations

import pandas as pd
import sys

sys.path.insert(0, ".")

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest import engine as bt
from vi_system.pipeline import screen_cache as sc_mod

LOOKAHEAD = 365  # 偷看未来天数


class LookAheadStore(Store):
    """facts_asof 截止日 +LOOKAHEAD 天，等价于每个调仓日偷看下一周期财报。"""

    def facts_asof(self, asof, periods=24):
        return super().facts_asof(
            pd.to_datetime(asof) + pd.Timedelta(days=LOOKAHEAD), periods
        )


def _no_disk_get_screen(store, asof, cfg, compute, with_valuation=True):
    """绕过磁盘缓存，强制用当前（可能被 LookAhead 改写的）store 重算。"""
    return compute()


TIERS = [
    ("默认30只", None),
    ("Top5", 5),
    ("Top3", 3),
]


def run_all(store, label):
    cfg = load_config()
    rows = []
    for name, mh in TIERS:
        r = bt.run_tier(store, cfg, max_holdings=mh, benchmark="equal", label=name)
        s = r["stats"]
        rows.append({
            "档位": name,
            "年化": s["cagr"],
            "基准年化(等权)": s["bench_cagr"],
            "年化超额": s["excess_cagr"],
            "最大回撤": s["max_drawdown"],
            "夏普": s["sharpe"],
            "信息比率": s["information_ratio"],
            "换手": s["avg_turnover"],
        })
        print(f"  [{label}] {name}: 年化={s['cagr']:.2%} 超额={s['excess_cagr']:.2%} "
              f"回撤={s['max_drawdown']:.2%} 夏普={s['sharpe']:.2f} IR={s['information_ratio']:.2f}")
    return rows


def main():
    db = "data/real_universe"

    print(">>> 干净版（不带前视，走磁盘缓存）...")
    clean = run_all(Store(db), "clean")

    # 切换注入版：清进程内缓存 + 绕过磁盘缓存
    bt.clear_screen_cache()
    sc_mod.get_screen = _no_disk_get_screen
    print(f"\n>>> 注入版（facts_asof 截止日 +{LOOKAHEAD} 天，偷看未来一整年财报）...")
    ahead = run_all(LookAheadStore(db), "ahead")

    print("\n" + "=" * 70)
    print(f"对比（注入版 − 干净版）：前视偏移 = +{LOOKAHEAD} 天")
    print("=" * 70)
    print(f"{'档位':<10}{'年化Δ':>10}{'超额Δ':>10}{'回撤Δ':>10}{'夏普Δ':>8}{'IRΔ':>8}")
    for c, a in zip(clean, ahead):
        dc = a["年化"] - c["年化"]
        de = a["年化超额"] - c["年化超额"]
        dm = a["最大回撤"] - c["最大回撤"]
        ds = a["夏普"] - c["夏普"]
        di = a["信息比率"] - c["信息比率"]
        print(f"{c['档位']:<10}{dc:>+9.2%}{de:>+9.2%}{dm:>+9.2%}{ds:>+8.2f}{di:>+8.2f}")

    print("\n判读：若 Δ 显著为正 → 未来财务对策略极有价值，干净版未用 → 无前视/业绩保守；"
          "若 Δ≈0 → 需进一步排查是否已泄漏。")


if __name__ == "__main__":
    main()
