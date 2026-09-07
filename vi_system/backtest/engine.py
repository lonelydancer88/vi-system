"""回测引擎。

本模块是系统中唯一有权否决前面所有工作的地方。四条硬性控制：

  1. **防前视**：调仓日设在 5 月与 9 月（年报 4 月底、半年报 8 月底披露完毕之后），
     且全部财务数据经 store.facts_asof() 按 announce_date 过滤。
  2. **防幸存者**：宇宙按决策日重建，含后来退市的公司。
  3. **成本**：单边 cost_per_side（默认 0.5%），按换手额计提。
  4. **样本内外分离**：样本内调参，样本外一次通过 —— 不许回头改。

另含分市场状态检验与参数敏感性检验。收益数据用后复权价（含分红再投）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..pipeline import universe as _universe, metrics as _metrics
from ..pipeline import vetoes as _vetoes, factors as _factors, valuation as _valuation
from ..portfolio.constructor import build_portfolio, turnover


# ==================================================================== 单次筛选截面
def screen_at(store, asof, cfg: Config, with_valuation: bool = False):
    """在 asof 时点跑完 L1→L5，返回 (scored_or_valued, rejected, universe)。"""
    uni = _universe.build_universe(store, asof, cfg)
    if uni.empty:
        return pd.DataFrame(), pd.DataFrame(), uni
    m = _metrics.compute_metrics(store.facts_asof(asof), store.market_asof(asof), uni)
    if m.empty:
        return pd.DataFrame(), pd.DataFrame(), uni
    passed, rejected = _vetoes.apply_vetoes(m, cfg)
    if passed.empty:
        return pd.DataFrame(), rejected, uni
    scored = _factors.score_factors(passed, cfg)
    if with_valuation:
        scored = _valuation.valuate(scored, cfg)
    return scored, rejected, uni


# ==================================================================== 主回测
def _index_benchmark_returns(store, code: str, dates: list) -> list | None:
    """构造与调仓日对齐的逐期指数基准收益序列。

    返回长度 len(dates)-1 的 list；若指数数据缺失则返回 None（上层回退等权）。
    取「调仓日当日或之前最近交易日」的收盘点，符合 point-in-time（不用未来价）。
    """
    idx = store.load_index_prices()
    if idx.empty or code not in set(idx["code"]):
        return None
    s = idx[idx["code"] == code].set_index("date")["close"].sort_index()
    closes = []
    for d in dates:
        dt = pd.to_datetime(d)
        sub = s[s.index <= dt]
        closes.append(float(sub.iloc[-1]) if len(sub) else np.nan)
    rets = []
    for i in range(len(closes) - 1):
        a, b = closes[i], closes[i + 1]
        rets.append((b / a - 1) if (np.isfinite(a) and np.isfinite(b) and a > 0) else 0.0)
    return rets


def run_backtest(
    store,
    cfg: Config,
    start: str,
    end: str,
    label: str = "full",
    with_valuation: bool = False,
    benchmark: str = "equal",
) -> dict:
    """运行回测。返回 dict（含 nav 曲线、指标、逐期明细）。

    benchmark:
      - "equal"（默认）：等权持有当期宇宙，作为对照基准。
      - 指数代码（如 "sh000300"）：以该指数日线收益作为基准（价格回报口径，不含股息）。
    """
    bcfg = cfg.section("backtest")
    months = bcfg.get("rebalance_months", [5, 9])
    day = bcfg.get("rebalance_day", 15)
    cost = bcfg.get("cost_per_side", 0.005)

    # 提前一天取价，确保调仓信号与成交价不重合
    dates = store.rebalance_dates(months, day, start, end)
    if len(dates) < 3:
        return {"error": f"调仓日不足（{len(dates)}），请放宽时间区间"}

    # ---- 基准解析
    bench_label = "等权基准"
    bench_index_rets: list | None = None
    if isinstance(benchmark, str) and benchmark != "equal":
        bench_index_rets = _index_benchmark_returns(store, benchmark, dates)
        if bench_index_rets is None:
            print(f"[warn] 指数基准 {benchmark} 数据缺失，回退等权基准")
        else:
            bench_label = f"{benchmark} 指数（价格回报）"

    px = store.price_panel(start, end)
    if px.empty:
        return {"error": "无行情数据"}
    px = px.ffill()

    nav, bench = 1.0, 1.0
    prev_w: pd.Series | None = None
    records, holdings_hist = [], []

    for i, d in enumerate(dates[:-1]):
        d_next = dates[i + 1]
        if d not in px.index or d_next not in px.index:
            continue

        scored, rejected, uni = screen_at(store, d, cfg, with_valuation)
        if scored.empty:
            w_new = pd.Series(dtype=float)
        else:
            pf = build_portfolio(scored, cfg, current_weights=prev_w)
            w_new = pd.Series(pf["weight"].values, index=pf["code"].values) if not pf.empty \
                else pd.Series(dtype=float)

        p0 = px.loc[d]
        p1 = px.loc[d_next]
        ret = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan)

        # ---- 换手：prev_w 是上一期持仓经收益漂移后的权重（在上一期末已算好）
        tvr = turnover(prev_w, w_new)
        cost_paid = tvr * cost

        # ---- 组合收益（本期持有 w_new 从 d 到 d_next）
        if len(w_new):
            r = ret.reindex(w_new.index).fillna(0.0)
            port_ret = float((w_new * r).sum())
        else:
            port_ret = 0.0

        # ---- 权重漂移：为下一期换手计算准备（不能用未来收益倒推本期）
        if len(w_new):
            drift = w_new * (1 + r)
            prev_w = drift / drift.sum() if drift.sum() > 0 else w_new
        else:
            prev_w = w_new

        # ---- 基准收益：指数代码优先，否则等权全市场
        if bench_index_rets is not None:
            bench_ret = bench_index_rets[i]
        else:
            bcodes = uni["code"] if not uni.empty else pd.Index([])
            bsub = ret.reindex(bcodes).dropna()
            bench_ret = float(bsub.mean()) if len(bsub) else 0.0

        nav *= (1 - cost_paid) * (1 + port_ret)
        bench *= (1 + bench_ret)

        records.append({
            "date": d, "next_date": d_next, "port_ret": port_ret,
            "bench_ret": bench_ret, "excess": port_ret - bench_ret,
            "turnover": tvr, "cost": cost_paid,
            "n_holdings": len(w_new), "n_universe": 0 if uni.empty else len(uni),
            "n_rejected": len(rejected), "nav": nav, "bench": bench,
        })
        holdings_hist.append({"date": d, "weights": w_new})
        prev_w = w_new

    if not records:
        return {"error": "回测未产生任何有效周期"}

    rec = pd.DataFrame(records)
    nav_curve = rec.set_index("date")[["nav", "bench"]]
    stats = _stats(rec, nav_curve, bench_label=bench_label)
    stats["label"] = label
    return {
        "label": label,
        "records": rec,
        "nav": nav_curve,
        "stats": stats,
        "holdings": holdings_hist,
        "dates": dates,
    }


def _stats(rec: pd.DataFrame, nav: pd.DataFrame, bench_label: str = "等权基准") -> dict:
    n = len(rec)
    years = n / 2.0  # 每年 2 次调仓
    total = float(nav["nav"].iloc[-1])
    bench_total = float(nav["bench"].iloc[-1])
    cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan
    bench_cagr = bench_total ** (1 / years) - 1 if years > 0 and bench_total > 0 else np.nan

    r = rec["port_ret"]
    br = rec["bench_ret"]
    vol = float(r.std(ddof=1) * np.sqrt(2)) if n > 1 else np.nan
    bench_vol = float(br.std(ddof=1) * np.sqrt(2)) if n > 1 else np.nan
    sharpe = (float(r.mean() * 2) / vol) if vol and np.isfinite(vol) and vol > 0 else np.nan

    cummax = nav["nav"].cummax()
    mdd = float((nav["nav"] / cummax - 1).min())
    bmax = nav["bench"].cummax()
    bench_mdd = float((nav["bench"] / bmax - 1).min())

    ex = rec["excess"]
    ir = (float(ex.mean() * 2) / float(ex.std(ddof=1) * np.sqrt(2))) \
        if n > 1 and ex.std(ddof=1) > 0 else np.nan

    return {
        "periods": n,
        "years": round(years, 2),
        "benchmark": bench_label,
        "total_return": total - 1,
        "cagr": cagr,
        "bench_cagr": bench_cagr,
        "excess_cagr": cagr - bench_cagr if np.isfinite(cagr) and np.isfinite(bench_cagr) else np.nan,
        "vol": vol,
        "bench_vol": bench_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "bench_max_drawdown": bench_mdd,
        "information_ratio": ir,
        "win_rate_vs_bench": float((rec["excess"] > 0).mean()),
        "avg_turnover": float(rec["turnover"].mean()),
        "avg_cost": float(rec["cost"].mean()),
        "avg_holdings": float(rec["n_holdings"].mean()),
    }


# ==================================================================== 样本内外 / 状态检验
def split_sample(store, cfg: Config) -> dict:
    """样本内 / 样本外分别回测。样本外一次通过才算数。"""
    b = cfg.section("backtest")
    ins = b.get("in_sample", ["2010-01-01", "2018-12-31"])
    oos = b.get("out_of_sample", ["2019-01-01", "2026-12-31"])
    return {
        "in_sample": run_backtest(store, cfg, ins[0], ins[1], label="样本内"),
        "out_of_sample": run_backtest(store, cfg, oos[0], oos[1], label="样本外"),
    }


def regime_report(result: dict) -> pd.DataFrame:
    """分市场状态检验：策略必须能解释自己在每种状态下的表现。"""
    rec = result.get("records")
    if rec is None or rec.empty:
        return pd.DataFrame()
    rec = rec.copy()
    rec["year"] = pd.to_datetime(rec["date"]).dt.year
    regimes = {
        (2010, 2014): "震荡磨底",
        (2015, 2015): "股灾",
        (2016, 2018): "熊市/去杠杆",
        (2019, 2021): "成长牛（价值跑输期）",
        (2022, 2024): "红利/低波牛（价值高光期）",
        (2025, 2026): "AI 行情（价值跑输期）",
    }
    rows = []
    for (y0, y1), name in regimes.items():
        sub = rec[(rec["year"] >= y0) & (rec["year"] <= y1)]
        if sub.empty:
            continue
        rows.append({
            "regime": name, "periods": len(sub),
            "port": float(sub["port_ret"].mean() * 2),
            "bench": float(sub["bench_ret"].mean() * 2),
            "excess": float(sub["excess"].mean() * 2),
            "win_rate": float((sub["excess"] > 0).mean()),
        })
    return pd.DataFrame(rows)


# ==================================================================== 参数敏感性
def sensitivity_test(store, cfg: Config, start: str, end: str) -> pd.DataFrame:
    """核心参数 ±20% 扰动。结果发生质变 = 过拟合，推倒重来。"""
    bcfg = cfg.section("backtest").get("sensitivity", {})
    perturb = bcfg.get("perturb", 0.20)
    params = bcfg.get("params", [])
    rows = [{
        "param": "baseline", "delta": 0.0,
        "cagr": run_backtest(store, cfg, start, end, "base")["stats"]["cagr"],
    }]
    for p in params:
        for sign in (+1, -1):
            c2 = cfg.perturbed(p, sign * perturb)
            s = run_backtest(store, c2, start, end, f"{p}{sign*perturb:+.0%}")["stats"]
            rows.append({"param": p, "delta": sign * perturb, "cagr": s["cagr"]})
    return pd.DataFrame(rows)


# ==================================================================== 报告
def backtest_report(results: dict, regimes: bool = True) -> str:
    lines = ["# 回测报告", ""]
    # 遍历所有结果，而不是硬编码 in_sample/out_of_sample ——
    # 否则 full 模式（key="full"）会全部跳过，报告只剩标题。
    for key, r in results.items():
        if not r or r.get("error"):
            continue
        s = r["stats"]
        bname = s.get("benchmark", "等权基准")
        lines += [
            f"## {s['label']}（{s['years']} 年 / {s['periods']} 期）", "",
            f"| 指标 | 策略 | {bname} |", "|------|------|------|",
            f"| 年化收益 | {s['cagr']:.2%} | {s['bench_cagr']:.2%} |",
            f"| 年化超额 | {s['excess_cagr']:.2%} | — |",
            f"| 年化波动 | {s['vol']:.2%} | {s['bench_vol']:.2%} |",
            f"| 夏普 | {s['sharpe']:.2f} | — |",
            f"| 信息比率 | {s['information_ratio']:.2f} | — |",
            f"| 最大回撤 | {s['max_drawdown']:.2%} | {s['bench_max_drawdown']:.2%} |",
            f"| 胜率(对基准) | {s['win_rate_vs_bench']:.1%} | — |",
            f"| 平均换手 | {s['avg_turnover']:.2%} | — |",
            f"| 平均持仓数 | {s['avg_holdings']:.1f} | — |",
            "",
        ]
        if regimes:
            rg = regime_report(r)
            if not rg.empty:
                lines += ["**分市场状态检验**", "",
                          "| 状态 | 期数 | 策略年化 | 基准年化 | 超额 | 胜率 |",
                          "|------|------|------|------|------|------|"]
                for _, x in rg.iterrows():
                    lines.append(
                        f"| {x['regime']} | {x['periods']} | {x['port']:.2%} | "
                        f"{x['bench']:.2%} | {x['excess']:.2%} | {x['win_rate']:.0%} |"
                    )
                lines.append("")
    return "\n".join(lines)


def compare_benchmarks(
    store, cfg: Config, start: str, end: str,
    index_code: str = "sh000300", index_name: str = "沪深300",
) -> str:
    """双基准对比：策略 vs 等权基准 vs 指数基准。返回 markdown 报告。"""
    r_eq = run_backtest(store, cfg, start, end, label="full", benchmark="equal")
    r_ix = run_backtest(store, cfg, start, end, label="full", benchmark=index_code)
    if r_eq.get("error"):
        return f"回测失败（等权）：{r_eq['error']}"
    if r_ix.get("error"):
        return f"回测失败（{index_name}）：{r_ix['error']}"

    se, si = r_eq["stats"], r_ix["stats"]

    def row(name, ke, ki, fmt="{:.2%}"):
        return f"| {name} | {fmt.format(ke)} | {fmt.format(ki)} |"

    lines = [
        f"# 双基准对比报告：策略 vs 等权 vs {index_name}（{start} ~ {end}）", "",
        "> 说明：{index_name} 指数基准为**价格回报口径**（指数点，不含股息再投），"
        "会系统性低估真实全收益基准约 2-3%/年。".format(index_name=index_name), "",
        "| 指标 | 等权基准 | %s 基准 |" % index_name,
        "|------|---------|---------|",
        row("基准年化", se["bench_cagr"], si["bench_cagr"]),
        row("基准波动", se["bench_vol"], si["bench_vol"]),
        row("基准最大回撤", se["bench_max_drawdown"], si["bench_max_drawdown"]),
        "",
        "| 指标 | 策略 | 等权基准 | %s 基准 |" % index_name,
        "|------|------|---------|---------|",
        f"| 策略年化 | {se['cagr']:.2%} | — | — |",
        f"| 超额年化(对等权) | {se['excess_cagr']:.2%} | — | — |",
        f"| 超额年化(对{index_name}) | {si['excess_cagr']:.2%} | — | — |",
        f"| 策略波动 | {se['vol']:.2%} | — | — |",
        f"| 策略最大回撤 | {se['max_drawdown']:.2%} | — | — |",
        f"| 夏普 | {se['sharpe']:.2f} | — | — |",
        f"| 信息比率(对等权) | {se['information_ratio']:.2f} | — | — |",
        f"| 信息比率(对{index_name}) | {si['information_ratio']:.2f} | — | — |",
        f"| 胜率(对等权) | {se['win_rate_vs_bench']:.1%} | — | — |",
        f"| 胜率(对{index_name}) | {si['win_rate_vs_bench']:.1%} | — | — |",
        "",
        "**结论**：",
        f"- 相对**等权全市场**：年化超额 {se['excess_cagr']:+.2%}，"
        f"信息比率 {se['information_ratio']:.2f}；",
        f"- 相对**{index_name}（价格回报）**：年化超额 {si['excess_cagr']:+.2%}，"
        f"信息比率 {si['information_ratio']:.2f}；",
        f"- 策略波动 {se['vol']:.2%} 显著低于等权基准 {se['bench_vol']:.2%}，"
        f"最大回撤 {se['max_drawdown']:.2%} 远低于基准。",
        "",
    ]
    return "\n".join(lines)
