"""生成「回测报告」：逐期交易动作 / 持仓 / 买入原因 / 卖出原因。

用法
----
    PYTHONPATH=. python3 tests/gen_backtest_report.py            # 5 只版
    PYTHONPATH=. python3 tests/gen_backtest_report.py --holdings 30

区别于 backtest.md（只有业绩统计）/ trades.md（台账）/ reasons.md（仅变动）：
本报告把三期信息合到一期一节的叙述式报告 —— 每期给出
  ① 调仓动作（买入/卖出逐笔，含买入/卖出原因 = L4 三支柱 z + 排名 + L5 估值）
  ② 期末持仓（权重 / 现价 / 建仓价 / 价格收益 / 持有收益(含分红)）
信号与原因均取「该调仓日同期」的因子面板，与回测口径一致、可追溯。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vi_system.backtest import engine as bt          # noqa: E402
from vi_system.cli import _cfg, _store                # noqa: E402


def _pct(x, nd=1):
    """小数 → 百分数字符串（如 0.1229 → '+12.3%'）。非数值返回 ''。"""
    try:
        return "" if x is None or not np.isfinite(float(x)) else f"{float(x) * 100:+.{nd}f}%"
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/real_universe")
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--holdings", type=int, default=5,
                    help="组合持股上限（回测引擎自动放宽单票/行业上限）")
    ap.add_argument("--capital", type=float, default=1_000_000)
    args = ap.parse_args()

    cfg, st = _cfg(None), _store(args.db)
    tag = "" if args.holdings in (None, 0) else f"-top{args.holdings}"
    r = bt.run_backtest(st, cfg, args.start, args.end, label="full",
                        max_holdings=args.holdings if args.holdings else None,
                        with_panel=True, with_valuation=True)
    if r.get("error"):
        print("回测失败：", r["error"])
        sys.exit(1)

    st_ = r["stats"]
    nav_last = r["nav"].iloc[-1]
    bench_total = float(nav_last["bench"]) - 1.0
    by_date = bt._panel_by_date(r.get("panel"))
    by_rej = bt._panel_rej_by_date(r.get("panel_rej"))
    ledger = bt.trade_ledger(r, st)
    eff = r.get("effective_start", "")

    L = [f"# 回测报告（最多持有 {args.holdings} 只）", "",
         f"> 数据口径：`{args.db}`；实算起点 **{eff}**（剔除期初空仓期）；"
         f"调仓 5/9/11 月 15 日（年报+一季报 / 中报 / 三季报披露后），共 {len(ledger)} 期。",
         f"> 净值/收益用后复权价（含分红再投）；价格取真实不复权收盘价 `close_raw`。"
         "**收益列一律后复权口径**（与净值一致，避免跨送转除权把收益算错）。",
         "> **买入/卖出原因**：与该调仓日同期截面 —— L4 三支柱为行业内中性 z（>0 即跑赢行业均值），"
         "排名越小越优；L5 估值为两阶段 DCF 三情景（v_mid=三情景中位数、买点=v_mid×0.70、卖点=v_mid×1.50，"
         "单位亿元）。未进候选池的卖出会给出具体排雷/流动性原因。", ""]

    # ---------------- 业绩概览
    bench = st_.get("benchmark", "等权基准")
    L += ["---", "", "## 业绩概览", "",
          f"| 指标 | 策略 | {bench} |", "|------|------|------|",
          f"| 年化收益 | **{_pct(st_.get('cagr'))}** | {_pct(st_.get('bench_cagr'))} |",
          f"| 年化超额 | {_pct(st_.get('excess_cagr'))} | — |",
          f"| 区间总收益 | {_pct(st_.get('total_return'))} | {_pct(bench_total)} |",
          f"| 最大回撤 | {_pct(st_.get('max_drawdown'), 2)} | {_pct(st_.get('bench_max_drawdown'), 2)} |",
          f"| 夏普 | {st_.get('sharpe', float('nan')):.2f} | — |",
          f"| 信息比率 | {st_.get('information_ratio', float('nan')):.2f} | — |",
          f"| 胜率(对基准) | {st_.get('win_rate_vs_bench', float('nan')):.1%} | — |",
          f"| 平均换手 | {st_.get('avg_turnover', float('nan')):.1%} | — |",
          f"| 平均持仓数 | {st_.get('avg_holdings', float('nan')):.1f} | — |", "",
          "**净值曲线**", "",
          "![净值曲线（30只/5只/3只/等权/沪深300 对比）](./nav-curve.png)",
          "> 图中含 30只 / 5只 / 3只 / 等权全市场 / 沪深300 五条累计净值；本报告对应**策略 5 只**档。", ""]

    # ---------------- 逐期
    rec = r["records"].set_index("date") if r.get("records") is not None else None
    n = 0
    for blk in ledger:
        d = pd.Timestamp(blk["date"])
        meta = ""
        if rec is not None and d in rec.index:
            rr = rec.loc[d]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            meta = (f"组合收益 {_pct(rr.get('port_ret'))}，"
                    f"基准 {_pct(rr.get('bench_ret'))}，"
                    f"超额 {_pct(rr.get('excess'))}，"
                    f"累计净值 {float(rr.get('nav', np.nan)):.3f}")
        n += 1
        L += ["---", "", f"## 第 {n} 期：{d.date()}（{meta}）", ""]

        trades = [t for t in blk["trades"] if t["action"] != "持有"]
        # ---------- 买入/卖出动作与原因
        if trades:
            L += ["**调仓动作**", "",
                  "| 动作 | 名称 | 行业 | 上期权重 | 目标权重 | 变动 | 买入/卖出原因 |",
                  "|------|------|------|---------|---------|------|-------------|"]
            for t in sorted(trades, key=lambda x: ({"建仓": 0, "清仓": 1,
                                                    "增持": 2, "减持": 3}.get(x["action"], 9),
                                                    -abs(x["w_chg"]))):
                chg = f"{t['w_chg']:+.1%}"
                why = bt._reason_cell(by_date, d, t["code"], t["action"], by_rej, st)
                w = t.get("w_new", 0.0)
                cap_note = ""
                if t["action"] in ("建仓", "增持") and np.isfinite(t.get("price", np.nan)) and t["price"] > 0:
                    lots = int(w * args.capital / t["price"] / 100)
                    cap_note = f"（{lots} 手≈{lots * 100 * t['price']:,.0f} 元）" if lots > 0 else ""
                L.append(f"| {t['action']} | {t['name']} | {t.get('industry', '')} | "
                         f"{t['w_prev']:.1%} | {t['w_new']:.1%} | {chg} | {why}{cap_note} |")
            L.append("")

        # ---------- 当期卖出（清仓/减持）已实现盈亏
        sells = [t for t in trades if t["action"] in ("清仓", "减持")]
        if sells:
            L += ["**当期卖出（已实现盈亏）**", "",
                  "| 名称 | 行业 | 动作 | 卖出价 | 建仓日期 | 建仓成本 | 价格收益 | 已实现收益(含分红) |",
                  "|------|------|------|--------|---------|---------|---------|-----------|"]
            for t in sorted(sells, key=lambda x: x["action"]):
                prc = t.get("price")
                cst = t.get("cost")
                rtv = t.get("ret")
                rpv = t.get("ret_price")
                prc_f = f"{prc:.2f}" if (prc is not None and np.isfinite(prc)) else "—"
                cst_f = f"{cst:.2f}" if (cst is not None and np.isfinite(cst)) else "—"
                rtv_f = f"{rtv:+.1%}" if (rtv is not None and np.isfinite(rtv)) else "—"
                rpv_f = f"{rpv:+.1%}" if (rpv is not None and np.isfinite(rpv)) else "—"
                ed_f = t.get("entry_date") or "—"
                L.append(f"| {t['name']} | {t.get('industry', '')} | {t['action']} | "
                         f"{prc_f} | {ed_f} | {cst_f} | {rpv_f} | {rtv_f} |")
            L.append("")

        # ---------- 期末持仓
        snap = blk.get("snap", [])
        if snap:
            L += ["**期末持仓**", "",
                  "| 名称 | 行业 | 权重 | 现价 | 建仓日期 | 建仓价 | 价格收益 | 持有收益(含分红) |",
                  "|------|------|------|------|---------|--------|---------|------------|"]
            for s in sorted(snap, key=lambda x: -x["w"]):
                cst = s.get("cost")
                prc = s.get("price")
                rtv = s.get("ret")
                rpv = s.get("ret_price")
                cst_f = f"{cst:.2f}" if (cst is not None and np.isfinite(cst)) else "—"
                prc_f = f"{prc:.2f}" if (prc is not None and np.isfinite(prc)) else "—"
                rtv_f = f"{rtv:+.1%}" if (rtv is not None and np.isfinite(rtv)) else "—"
                rpv_f = f"{rpv:+.1%}" if (rpv is not None and np.isfinite(rpv)) else "—"
                ed_f = s.get("entry_date") or "—"
                L.append(f"| {s['name']} | {s.get('industry', '')} | {s['w']:.1%} | "
                         f"{prc_f} | {ed_f} | {cst_f} | {rpv_f} | {rtv_f} |")
            L.append("")
        else:
            L += ["（空仓——当期无人过买入门槛，按纪律不动手。）", ""]

    L += ["---", "",
          "**口径附注**：信号与成交同为调仓日收盘（未实现 t+1）；基准为等权全市场"
          "（价格回报，不含股息）；「建仓价/卖出价」为**真实不复权成交价**（对照行情用），"
          "「已实现收益/持有收益(含分红)」为**后复权口径**（含分红再投，与净值 NAV 一致——"
          "持仓跨送转/增发除权不会被未复权价差误判）；同表另列「价格收益」= 不复权口径"
          "（现价 vs 建仓价，不含分红），与行情软件默认一致，二者之差即持有期分红贡献；"
          "L5 估值在经营现金流缺失时以净利润×80%兜底，银行/地产类可能失真，买入前请人工复核。", ""]

    dst = pathlib.Path("out") / f"回测报告{tag}-{str(pd.Timestamp(r['dates'][-1]).date())}.md"
    dst.write_text("\n".join(L), encoding="utf-8")
    print(f"→ {dst}（{len(ledger)} 期）")


if __name__ == "__main__":
    main()
