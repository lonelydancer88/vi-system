"""生成「今天建仓买什么」清单：选股信号 + 真实收盘价 + 手数。

用法
----
    PYTHONPATH=. python3 tests/gen_position_list.py              # 用最新交易日收盘价
    PYTHONPATH=. python3 tests/gen_position_list.py --capital 500000

要点
----
1. 选股信号用 `screen_at(asof)`：point-in-time，只用公告日之前可得的财报。
2. 成交价**不用库内 close_raw**（实测与真实收盘有 ~0.4% 随机噪声），
   改用腾讯行情批量接口 `qt.gtimg.cn` 的当日收盘价（f[3]）。
3. 5 只版必须复刻 `run_backtest(max_holdings=5)` 的 concentrated override：
   target_size=[1,5]、max_position=max(原值, min(0.35, 0.9/5×1.1))、max_industry=0.5，
   否则口径与回测不一致。
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

import numpy as np
import pandas as pd

from vi_system.backtest import engine as bt
from vi_system.cli import _cfg, _store
from vi_system.portfolio import constructor as portfolio_mod

QT = "http://qt.gtimg.cn/q={codes}"


def fetch_quotes(codes: list[str]) -> dict[str, dict]:
    """腾讯行情批量接口：GBK，`~` 分隔；f[3]=现价 f[4]=昨收 f[45]=总市值(亿) f[73]=总股本。"""
    out: dict[str, dict] = {}
    for i in range(0, len(codes), 60):
        url = QT.format(codes=",".join(codes[i:i + 60]))
        raw = subprocess.run(["curl", "-s", "--max-time", "25", url],
                             capture_output=True).stdout.decode("gbk", "ignore")
        for line in raw.split(";"):
            line = line.strip()
            if not line.startswith("v_"):
                continue
            code = line[2:line.index("=")]
            f = line[line.index('"') + 1:line.rindex('"')].split("~")
            try:
                out[code] = {"name": f[1], "close": float(f[3]),
                             "prev": float(f[4]), "shares": float(f[73])}
            except Exception:
                pass
    return out


def mkcfg(base, hi: int | None):
    """复刻 run_backtest 对 max_holdings 的配置 override。"""
    if not hi:
        return base, "默认版（20~30 只）"
    po = base.section("portfolio") or {}
    mpos = max(float(po.get("max_position", 0.08)), min(0.35, 0.9 / hi * 1.1))
    ov = {"portfolio.target_size": [1, hi],
          "portfolio.max_position": round(mpos, 4),
          "portfolio.max_industry": 0.5 if hi <= 5 else float(po.get("max_industry", 0.25))}
    c = base
    for k, v in ov.items():
        c = c.with_value(k, v)
    return c, f"最多 {hi} 只版"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/real_universe")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--holdings", default="none,5",
                    help="逗号分隔，none=默认版，如 none,5")
    args = ap.parse_args()

    cfg, st = _cfg(None), _store(args.db)
    asof = str(pd.read_parquet(f"{args.db}/prices.parquet")["date"].max().date())
    scored, rejected, uni = bt.screen_at(st, asof, cfg, with_valuation=True)
    quotes = fetch_quotes(sorted(set(scored["code"])))
    zcols = [c for c in ("value_z", "quality_z", "safety_z", "rank") if c in scored.columns]
    zmap = scored.set_index("code")[zcols].to_dict("index")

    L = [f"# 建仓清单（数据截至 {asof} 收盘）", "",
         f"> 选股信号 `screen_at(asof={asof})`：宇宙 **{len(uni)}** → 排雷后 **{len(scored)}** "
         f"→ 过 AND 门 **{int(scored['passes_gate'].sum())}**。",
         "> 价格取腾讯行情当日**真实收盘价**（非库内 close_raw）；手数按 1 手=100 股向下取整。",
         f"> 本金 {args.capital:,.0f} 元。"]
    for h in args.holdings.split(","):
        hi = None if h.strip() in ("", "none") else int(h)
        c, tag = mkcfg(cfg, hi)
        pf = portfolio_mod.build_portfolio(scored, c)
        if pf.empty:
            L += ["", f"## {tag}：无标的（无人过 AND 门槛，按纪律空仓）"]
            continue
        d = pf.copy()
        d["价"] = d["code"].map(lambda x: quotes.get(x, {}).get("close", np.nan))
        d["涨跌"] = d["code"].map(
            lambda x: (quotes[x]["close"] / quotes[x]["prev"] - 1) * 100
            if x in quotes and quotes[x]["prev"] else np.nan)
        d["手数"] = (d["weight"] * args.capital / d["价"] / 100).fillna(0).astype(int)
        d["占用"] = d["手数"] * 100 * d["价"]
        for z in ("value_z", "quality_z", "safety_z", "rank"):
            if z in zcols:
                d[z] = d["code"].map(lambda x: zmap.get(x, {}).get(z, np.nan))
        L += ["---", "", f"## {tag}：{len(d)} 只，现金 {max(0.0, 1-pf['weight'].sum()):.1%}", "",
              "| # | 代码 | 名称 | 行业 | 权重 | 总分 | 价值z | 质量z | 安全z | 排名 | 收盘价 | 涨跌 | 手数 | 占用(元) |",
              "|---|------|------|------|------|------|-------|-------|-------|------|-------|------|------|---------|"]
        for i, (_, r) in enumerate(d.iterrows(), 1):
            zs = " | ".join(f"{r[z]:+.2f}" for z in ("value_z", "quality_z", "safety_z") if z in d.columns)
            rk = f"{int(r['rank'])}" if "rank" in d.columns and np.isfinite(r.get("rank", np.nan)) else "—"
            L.append(f"| {i} | {r['code']} | {r['name']} | {r['industry']} | {r['weight']:.2%} | "
                     f"{r['total_score']:.2f} | {zs} | {rk} | {r['价']:.2f} | {r['涨跌']:+.2f}% | "
                     f"{r['手数']} | {r['占用']:,.0f} |")
        L += ["", f"合计占用 **{d['占用'].sum():,.0f} 元**，"
                  f"剩余现金 **{args.capital - d['占用'].sum():,.0f} 元**。", ""]
    L += ["---", "", "## 风险提示", "",
          "- 机械信号输出，未做人工基本面复核；不构成投资建议。",
          "- 模型历史跑输等权基准，请据此判断预期。",
          "- 信号与成交同用当日收盘价，比 t+1 开盘口径略偏乐观。", ""]
    dst = pathlib.Path("out") / f"建仓清单-{asof}.md"
    dst.write_text("\n".join(L), encoding="utf-8")
    print(f"→ {dst}")


if __name__ == "__main__":
    main()
