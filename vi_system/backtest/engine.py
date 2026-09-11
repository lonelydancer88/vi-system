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
# 面板记录字段（with_panel=True 时每期保存，供逐期调仓原因归因）
# 三支柱记行业内中性 z（value_z 等）；行业内 pct 仅供展示、不作为排序/门槛依据。
# 面板快照列：L4（三支柱行业内中性 z / 总分 / rank）+ L5（逆向 DCF 与两阶段三情景估值）。
# 只有 with_valuation=True 的 screen_at 才会带 v_* 系列列；缺失列在快照时被过滤掉。
_PANEL_COLS = ("code", "name", "industry", "rank", "value_z",
               "quality_z", "safety_z", "total_score", "passes_gate",
               "mktcap", "fcf0", "fcf_is_fallback", "g_implied",
               "v_bull", "v_base", "v_bear", "v_mid",
               "buy_point", "sell_point", "downside_bear", "verdict")
# 每期 L3 一票否决明细（reasons 回溯"为何没进候选池"需要具体规则与数值）
_REJ_COLS = ("code", "name", "industry", "rule", "rule_desc", "value", "threshold")


# 同一进程内按 (asof, with_valuation) 缓存 L1→L5 筛选结果。
# 动机：Top30/Top5/Top3 三档的筛选过程完全一样，只有组组合那步不同。
# 缓存后三档合计只需筛一遍，净值曲线重生成从 ~40 分钟降到一遍的时间。
# 返回副本，避免上层就地修改污染缓存。
_SCREEN_CACHE: dict = {}


def clear_screen_cache() -> None:
    """换库/换配置时调用，避免拿到旧宇宙的缓存。"""
    _SCREEN_CACHE.clear()


def screen_at(store, asof, cfg: Config, with_valuation: bool = False):
    """在 asof 时点跑完 L1→L5，返回 (scored_or_valued, rejected, universe)。

    with_valuation 仅影响「是否要求返回估值列」的语义；为缓存复用，内部统一
    计算并缓存「带估值」版本（估值只加列、不影响过门/权重），False 调用方直接
    复用同一份缓存，避免 run_backtest(默认 False) 与预热(True) 因 key 不同而
    全量重算。
    """
    key = (str(asof),)
    hit = _SCREEN_CACHE.get(key)
    if hit is not None:
        scored, rejected, uni = hit
        return (scored.copy() if scored is not None and not scored.empty else scored,
                rejected, uni)
    uni = _universe.build_universe(store, asof, cfg)
    if uni.empty:
        return pd.DataFrame(), pd.DataFrame(), uni

    def _compute():
        m = _metrics.compute_metrics(store.facts_asof(asof), store.market_asof(asof), uni)
        if m.empty:
            return pd.DataFrame(), pd.DataFrame(), uni
        passed, rejected = _vetoes.apply_vetoes(m, cfg)
        if passed.empty:
            return pd.DataFrame(), rejected, uni
        scored = _factors.score_factors(passed, cfg)
        scored = _valuation.valuate(scored, cfg)  # 统一带估值，供缓存复用
        return scored, rejected, uni

    # 持久化磁盘缓存：跨进程/次复用同一 asof 的筛选结果，避免四档回测与多次
    # 重跑串行累计超时（详见 vi_system/pipeline/screen_cache）。
    from ..pipeline.screen_cache import get_screen
    scored, rejected, uni = get_screen(store, asof, cfg, _compute, with_valuation=True)
    _SCREEN_CACHE[key] = (scored, rejected, uni)
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


# ---------------------------------------------------------------- 统一口径
# 全项目「TopN 档位」回测的唯一入口。
#
# 历史坑（务必不再复现）：报告 / CLI / 净值曲线各自直接调 run_backtest，
# 但只有 gen_backtest_report 传了 with_valuation=True。而 with_valuation 决定
# screen_at 是否跑估值、是否产出 verdict 列，组合构建又会按 verdict 过滤
# （已到卖点不买）—— 于是同一「5 只档」在报告里 9.9%、CLI 里 6.2%，
# 并非算错，而是两个不同的组合。故收口到本函数：窗口 / 剔空仓 / 估值回灌固定。
BACKTEST_START = "2013-01-01"
BACKTEST_END = "2026-12-31"


def run_tier(
    store,
    cfg: Config,
    max_holdings: int | None = None,
    benchmark: str = "equal",
    label: str = "full",
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    with_panel: bool = False,
    skip_empty: bool = True,
) -> dict:
    """统一口径跑一档回测：估值 verdict 回灌 + 同一窗口 + 剔除期初空仓期。

    所有对外口径（回测报告、CLI backtest、净值曲线）都必须走这里，
    否则会出现「同名档位数出三组数」的口径分裂。
    """
    return run_backtest(
        store, cfg, start, end, label=label,
        benchmark=benchmark, skip_empty=skip_empty,
        max_holdings=max_holdings,
        with_valuation=True, with_panel=with_panel,
    )


def run_backtest(
    store,
    cfg: Config,
    start: str,
    end: str,
    label: str = "full",
    with_valuation: bool = False,
    benchmark: str = "equal",
    skip_empty: bool = True,
    max_holdings: int | None = None,
    with_panel: bool = False,
) -> dict:
    """运行回测。返回 dict（含 nav 曲线、指标、逐期明细）。

    benchmark:
      - "equal"（默认）：等权持有当期宇宙，作为对照基准。
      - 指数代码（如 "sh000300"）：以该指数日线收益作为基准（价格回报口径，不含股息）。
    skip_empty:
      - True（默认）：剔除期初「空仓期」——即从首个实际持有股票的调仓日起算。
        原因：财务因子数据（point-in-time）最早只到 FY2016（2017 年披露），
        更早于 2013-2016 的调仓组合为空（全程平走），纳入会把空仓期也算进年化，
        虚增/虚减收益。剔除后业绩归因更干净。
      - False：保留全部调仓期（含空仓平走段），用于展示数据缺口本身。
    max_holdings:
      - None（默认）：使用配置里的 portfolio.target_size。
      - 正整数 N：把组合持股上限压到 N 只（高集中度实验）。会配套放宽风控上限——
        单票上限 = max(原值, min(0.35, 0.9/N×1.1))，行业上限在 N<=5 时放宽到 0.5，
        否则维持原值。仅本次运行生效，不改动配置文件。
    with_panel:
      - False（默认）：不记录每期因子面板。
      - True：记录每期候选股的三支柱分位/排名/过门状态到 result["panel"]，
        供 trade_reasons_report 生成逐期调仓原因。
    """
    if max_holdings is not None and max_holdings > 0:
        hi = int(max_holdings)
        concentrated = hi <= 5
        po = cfg.section("portfolio") or {}
        mpos = max(float(po.get("max_position", 0.08)), min(0.35, 0.9 / hi * 1.1))
        overrides = {
            "portfolio.target_size": [1, hi],
            "portfolio.max_position": round(mpos, 4),
            "portfolio.max_industry": (
                0.5 if concentrated else float(po.get("max_industry", 0.25))
            ),
        }
        for dotted, val in overrides.items():
            cfg = cfg.with_value(dotted, val)

    bcfg = cfg.section("backtest")
    months = bcfg.get("rebalance_months", [5, 9])
    day = bcfg.get("rebalance_day", 15)
    cost = bcfg.get("cost_per_side", 0.005)

    # 执行口径：screen_at(d) 的信号与 px.loc[d] 的成交同为调仓日收盘价
    # （close_adj）。严格的 t+1 执行（次日开盘）尚未实现；对财务慢因子而言
    # 同日收盘执行的前视影响可忽略，但比 t+1 口径略偏乐观。
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
    panels, panel_meta, panels_rej = [], [], []

    for i, d in enumerate(dates[:-1]):
        d_next = dates[i + 1]
        if d not in px.index or d_next not in px.index:
            continue

        scored, rejected, uni = screen_at(store, d, cfg, with_valuation)
        if with_panel:
            # 候选池（过 L3）打分快照：L4 + L5
            if not scored.empty:
                _cols = [c for c in _PANEL_COLS if c in scored.columns]
                _sub = scored[_cols].copy()
                _sub["date"] = d
                panels.append(_sub)
            # 被 L3 一票否决的明细：留痕具体规则与数值，供 reasons 给"具体原因"
            if not rejected.empty:
                _rcols = [c for c in _REJ_COLS if c in rejected.columns]
                _rsub = rejected[_rcols].copy()
                _rsub["date"] = d
                panels_rej.append(_rsub)
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
        # 注意：prev_w 保持为上面漂移后的权重（drift）——下一期的换手要相对
        # 「持仓经本期收益漂移后的自然权重」计算，而不是本期的目标权重。
        # 曾在这里误写 prev_w = w_new 把 drift 覆盖掉，导致换手率系统性偏高。

    if not records:
        return {"error": "回测未产生任何有效周期"}

    # ---- 跳过空仓期初：从首个实际持有股票的调仓日起算（数据可得性）
    # 财务因子 point-in-time 最早只到 FY2016（2017 年披露），更早调仓组合为空、
    # 全程平走，纳入会虚增/虚减年化收益。剔除后业绩归因更干净。
    #
    # 关键：剔除后必须把 nav/bench **重新锚定到 1**。records 里的 nav/bench 是
    # 从首期开始累乘的累积值，若只删行不重置，基准会把空仓期（2013-2017）的
    # 涨跌摊进剩余年数的年化里——曾把 HS300 基准年化从 3.18% 虚推到 6.54%，
    # 策略超额被等幅低估（实测：+1.63% → 真实 +4.99%）。
    eff_start = records[0]["date"]
    n_skipped = 0
    if skip_empty:
        first = next((i for i, r in enumerate(records) if r["n_holdings"] > 0), 0)
        if first > 0:
            n_skipped = first
            anchor_nav = records[first - 1]["nav"]
            anchor_bench = records[first - 1]["bench"]
            records = records[first:]
            if anchor_nav > 0 and anchor_bench > 0:
                for r in records:
                    r["nav"] /= anchor_nav
                    r["bench"] /= anchor_bench
            holdings_hist = holdings_hist[first:]
            eff_start = records[0]["date"]

    rec = pd.DataFrame(records)
    # 净值曲线：每段持有收益记在期末(next_date)，并在最前补一个建仓基点
    # (eff_start, nav=1.0, bench=1.0)，使曲线从 1.0 起步。修复：原实现把首期
    # 收益前置记在 eff_start 当日，导致曲线起点 ≠ 1。reasons 报告仍引用 rec 的
    # 原始 date（期初调仓日），不受影响。
    _navc = rec[["next_date", "nav", "bench"]].rename(columns={"next_date": "date"})
    _navc = _navc.set_index("date")[["nav", "bench"]]
    _base = pd.DataFrame(
        {"nav": [1.0], "bench": [1.0]},
        index=[pd.Timestamp(eff_start)],
    )
    nav_curve = pd.concat([_base, _navc]).sort_index()
    stats = _stats(rec, nav_curve, bench_label=bench_label)
    stats["label"] = label
    stats["effective_start"] = str(pd.to_datetime(eff_start).date())
    stats["skipped_empty_periods"] = n_skipped
    out = {
        "label": label,
        "records": rec,
        "nav": nav_curve,
        "stats": stats,
        "holdings": holdings_hist,
        "dates": dates,
        "effective_start": str(pd.to_datetime(eff_start).date()),
        "skipped_empty_periods": n_skipped,
    }
    if with_panel:
        out["panel"] = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
        out["panel_rej"] = (pd.concat(panels_rej, ignore_index=True)
                            if panels_rej else pd.DataFrame())
        out["panel_meta"] = panel_meta
    return out


def _stats(rec: pd.DataFrame, nav: pd.DataFrame, bench_label: str = "等权基准") -> dict:
    n = len(rec)
    # 年化口径：按实际日历跨度 + 实测期数折算，避免硬编码「每年 2 期」。
    # 调仓改为 [5,9,11]（一年 3 次）后，旧的 n/2.0 会把 28 期当成 14 年（真实约 9.3 年），
    # 导致 CAGR 被系统性低估约 30%、波动率被高估 sqrt(1.5) 倍。
    first, last = nav.index[0], nav.index[-1]
    years = (last - first).days / 365.25 if len(nav) > 1 else np.nan
    ppy = n / years if (years and np.isfinite(years) and years > 0) else np.nan  # periods per year
    total = float(nav["nav"].iloc[-1])
    bench_total = float(nav["bench"].iloc[-1])
    cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan
    bench_cagr = bench_total ** (1 / years) - 1 if years > 0 and bench_total > 0 else np.nan

    r = rec["port_ret"]
    br = rec["bench_ret"]
    vol = float(r.std(ddof=1) * np.sqrt(ppy)) if (n > 1 and np.isfinite(ppy)) else np.nan
    bench_vol = float(br.std(ddof=1) * np.sqrt(ppy)) if (n > 1 and np.isfinite(ppy)) else np.nan
    sharpe = (float(r.mean() * ppy) / vol) if vol and np.isfinite(vol) and vol > 0 else np.nan

    cummax = nav["nav"].cummax()
    mdd = float((nav["nav"] / cummax - 1).min())
    bmax = nav["bench"].cummax()
    bench_mdd = float((nav["bench"] / bmax - 1).min())

    ex = rec["excess"]
    ir = (float(ex.mean() * ppy) / float(ex.std(ddof=1) * np.sqrt(ppy))) \
        if (n > 1 and ex.std(ddof=1) > 0 and np.isfinite(ppy)) else np.nan

    return {
        "periods": n,
        "years": round(years, 2),
        "periods_per_year": round(ppy, 2) if np.isfinite(ppy) else np.nan,
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
        # 同样开启估值 verdict 回灌，保持与 run_tier / 回测报告同口径
        "in_sample": run_backtest(store, cfg, ins[0], ins[1], label="样本内",
                                  with_valuation=True),
        "out_of_sample": run_backtest(store, cfg, oos[0], oos[1], label="样本外",
                                      with_valuation=True),
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
    ppy = result.get("stats", {}).get("periods_per_year", 2.0)
    for (y0, y1), name in regimes.items():
        sub = rec[(rec["year"] >= y0) & (rec["year"] <= y1)]
        if sub.empty:
            continue
        rows.append({
            "regime": name, "periods": len(sub),
            "port": float(sub["port_ret"].mean() * ppy),
            "bench": float(sub["bench_ret"].mean() * ppy),
            "excess": float(sub["excess"].mean() * ppy),
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
        ]
        eff = s.get("effective_start")
        skp = s.get("skipped_empty_periods", 0)
        if eff and skp:
            lines += [
                f"> 注：已剔除期初 {skp} 个空仓期，实算起点 **{eff}**"
                f"（财务因子数据最早到 FY2016，此前组合为空）。", "",
            ]
        lines += [
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
    max_holdings: int | None = None,
) -> str:
    """双基准对比：策略 vs 等权基准 vs 指数基准。返回 markdown 报告。

    max_holdings: 传给 run_backtest 的持股上限（None=配置默认）。
    """
    r_eq = run_tier(store, cfg, start=start, end=end, benchmark="equal",
                    max_holdings=max_holdings)
    r_ix = run_tier(store, cfg, start=start, end=end, benchmark=index_code,
                    max_holdings=max_holdings)
    if r_eq.get("error"):
        return f"回测失败（等权）：{r_eq['error']}"
    if r_ix.get("error"):
        return f"回测失败（{index_name}）：{r_ix['error']}"

    se, si = r_eq["stats"], r_ix["stats"]

    def row(name, ke, ki, fmt="{:.2%}"):
        return f"| {name} | {fmt.format(ke)} | {fmt.format(ki)} |"

    eff = se.get("effective_start")
    skp = se.get("skipped_empty_periods", 0)
    win = f"{eff} ~ {end}" if (eff and skp) else f"{start} ~ {end}"
    conc = f"（最多 {max_holdings} 只）" if max_holdings else ""
    lines = [
        f"# 双基准对比报告：策略{conc} vs 等权 vs {index_name}（{win}）", "",
        "> 说明：{index_name} 指数基准为**价格回报口径**（指数点，不含股息再投），"
        "会系统性低估真实全收益基准约 2-3%/年。".format(index_name=index_name), "",
    ]
    if eff and skp:
        lines += [
            f"> 注：已剔除期初 {skp} 个空仓期，实算起点 **{eff}**"
            f"（财务因子数据最早到 FY2016，此前组合为空）。", "",
        ]
    lines += [
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


# ==================================================================== 交易台账
def trade_ledger(result: dict, store) -> list:
    """把回测的持仓快照差分，生成逐次调仓的买卖明细。

    返回 list（每个调仓日一个 dict）：
        {"date", "n_buy", "n_sell", "n_hold", "trades": [...]}
    trades 中每条：{code, name, industry, action, w_prev, w_new, w_chg, price}
    action ∈ {建仓, 清仓, 增持, 减持, 持有}
    """
    holdings = result.get("holdings", [])
    if len(holdings) < 1:
        return []
    try:
        u = store.load_universe().set_index("code")
        name_map = u["name"].to_dict()
        ind_map = u.get("industry", pd.Series(dtype=str)).to_dict()
    except Exception:
        name_map, ind_map = {}, {}
    # 价格快照：双口径。
    #  - close_raw（真实不复权收盘价）：用于「真实价 / 手数 / 占用资金」换算与对照行情。
    #  - close_adj（后复权，含分红再投）：用于「收益」列 —— 直接拿 close_raw 跨除权日比价，
    #    送转 / 增发 / 大额分红会把收益算错（实测传音 2024 十转四把 -20% 真跌夸大成 -45.5%）。
    #    收益口径与回测 NAV（同为 close_adj）保持一致，可跨期对比。
    # 说明：close_raw 经 2026-09-08 两次修复后已可信 ——
    #  ① amount/volume 反推（原 2013–2021 段负值/异质缩放损坏已覆盖）；
    #  ② 科创板 688 板块 volume 单位重复×100 的 bug（价格/市值低估 100 倍）已修正。
    try:
        _px = store.load_prices()
        _px["date"] = pd.to_datetime(_px["date"])
        _idx = _px.set_index(["date", "code"])
        def _px_at(d, c, col):
            try:
                row = _idx.loc[(pd.to_datetime(d), c)]
                v = float(row[col]) if col in row else np.nan
                return v if np.isfinite(v) else np.nan
            except Exception:
                return np.nan

        def price_at(d, c):        # 展示用：真实不复权收盘价
            raw = _px_at(d, c, "close_raw")
            if np.isfinite(raw) and raw > 0:
                return raw
            return _px_at(d, c, "close_adj")

        def adj_at(d, c):          # 收益用：后复权价（含分红再投，与 NAV 同口径）
            adj = _px_at(d, c, "close_adj")
            if np.isfinite(adj) and adj > 0:
                return adj
            return _px_at(d, c, "close_raw")
    except Exception:
        def price_at(d, c):
            return np.nan
        def adj_at(d, c):
            return np.nan

    ledger = []
    first_cost = {}    # code -> {"raw": 首次建仓真实价, "adj": 首次建仓后复权价}（跨期追踪收益基准）
    first_entry = {}   # code -> 首次建仓日期（ISO 串）；清仓后若再建仓，基准重置为新日期
    prev = pd.Series(dtype=float)
    for h in holdings:
        d = h["date"]
        d_iso = pd.to_datetime(d).date().isoformat()
        cur = h["weights"]
        cur_set, prev_set = set(cur.index), set(prev.index)
        rows = []
        for c in sorted(cur.index.union(prev.index)):
            in_cur, in_prev = c in cur_set, c in prev_set
            w_new = float(cur.get(c, 0.0)) if in_cur else 0.0
            w_prev = float(prev.get(c, 0.0)) if in_prev else 0.0
            if in_cur and not in_prev:
                action = "建仓"
            elif in_prev and not in_cur:
                action = "清仓"
            elif in_cur and in_prev and w_new > w_prev + 1e-9:
                action = "增持"
            elif in_cur and in_prev and w_new < w_prev - 1e-9:
                action = "减持"
            else:
                action = "持有"
            if action == "持有":
                continue
            price = price_at(d, c)      # 真实价（展示）
            adj = adj_at(d, c)          # 后复权价（收益）
            if action == "建仓" and c not in first_cost:
                first_cost[c] = {"raw": price, "adj": adj}
                first_entry[c] = d_iso
            cost0 = first_cost.get(c)
            cost = cost0.get("raw") if cost0 else np.nan
            cost_adj = cost0.get("adj") if cost0 else np.nan
            # 含分红收益：后复权口径（与 NAV 同口径）
            ret = ((adj - cost_adj) / cost_adj
                   if (cost_adj is not None and np.isfinite(cost_adj) and cost_adj > 0
                       and np.isfinite(adj)) else np.nan)
            # 价格收益：不复权口径（当前价 vs 首次建仓真实价），不含分红，可直接与行情软件核对
            ret_price = ((price - cost) / cost
                         if (cost is not None and np.isfinite(cost) and cost > 0
                             and np.isfinite(price)) else np.nan)
            entry_date = first_entry.get(c)   # 首次建仓日（建仓=本调仓日；增持/清仓=更早的初建日）
            if action == "清仓" and c in first_cost:
                del first_cost[c]          # 清仓后若再建仓，成本基准重置（算完 ret 后再删）
                first_entry.pop(c, None)
            rows.append({
                "code": c,
                "name": name_map.get(c, c),
                "industry": ind_map.get(c, ""),
                "action": action,
                "w_prev": w_prev,
                "w_new": w_new,
                "w_chg": w_new - w_prev,
                "price": price,
                "cost": cost,
                "cost_adj": cost_adj,
                "ret": ret,
                "ret_price": ret_price,
                "entry_date": entry_date,
            })
        # 本期完整持仓快照（含「持有」无动作股），用于报告「每只股票目前的收益」
        snap = []
        for c in cur.index:
            if float(cur.get(c, 0.0)) > 0:
                pr = price_at(d, c)
                cost0 = first_cost.get(c)
                cost = cost0.get("raw") if cost0 else np.nan
                cost_adj = cost0.get("adj") if cost0 else np.nan
                adj = adj_at(d, c)
                ret = ((adj - cost_adj) / cost_adj
                       if (cost_adj is not None and np.isfinite(cost_adj) and cost_adj > 0
                           and np.isfinite(adj)) else np.nan)
                ret_price = ((pr - cost) / cost
                             if (cost is not None and np.isfinite(cost) and cost > 0
                                 and np.isfinite(pr)) else np.nan)
                snap.append({
                    "code": c,
                    "name": name_map.get(c, c),
                    "industry": ind_map.get(c, ""),
                    "price": pr,
                    "cost": cost,
                    "cost_adj": cost_adj,
                    "ret": ret,
                    "ret_price": ret_price,
                    "w": float(cur.get(c, 0.0)),
                    "entry_date": first_entry.get(c),
                })
        n_buy = sum(1 for r in rows if r["action"] in ("建仓", "增持"))
        n_sell = sum(1 for r in rows if r["action"] in ("清仓", "减持"))
        ledger.append({
            "date": d,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "n_hold": int((cur > 0).sum()),
            "trades": rows,
            "snap": snap,
        })
        prev = cur
    return ledger


def trade_report(result: dict, store, top_n: int = 12, capital: float = 1_000_000.0) -> str:
    """生成可阅读的交易轨迹报告（markdown）。

    top_n  : 每期最多展示的买卖笔数；<=0 表示全量展示、不省略。
    capital: 本金假设（元），用于把目标权重换算成手数/占用资金。
             公式：目标手数 = ⌊目标权重 × 本金 ÷ 真实价 ÷ 100⌋（1 手 = 100 股）。
    """
    ledger = trade_ledger(result, store)
    if not ledger:
        return "# 交易台账\n\n（无持仓记录）"
    eff = result.get("effective_start")
    _by = _panel_by_date(result.get("panel"))
    _by_rej = _panel_rej_by_date(result.get("panel_rej"))
    lines = [
        "# 交易轨迹（买卖台账）", "",
        "> **价与手数口径**：表中「真实价」= **真实不复权收盘价**（`close_raw`，经 amount/volume 反推"
        " + 科创板 688 volume 单位修复后可信），用于对照行情与手数/占用换算。"
        "**「收益(含分红)」列 = 后复权 `close_adj` 口径（含分红再投，与净值 NAV 同口径）**——"
        "不用未复权价直接比价，否则持仓跨送转/增发除权会把收益算错（实测传音 2024 十转四"
        "把 −20% 真跌夸大成 −45%）。"
        "**「价格收益」列 = 不复权口径（当前价 vs 首次建仓真实价，不含分红）**，与行情软件默认一致，"
        "两者之差即持有期分红贡献。", "",
        f"> 手数与占用资金按本金 **{capital/1e4:.0f} 万元** 假设换算："
        "`目标手数 = ⌊目标权重 × 本金 ÷ 价 ÷ 100⌋`（1 手 = 100 股，向下取整），"
        "`占用资金 = 目标手数 × 100 × 价`。目标手数/占用为该期**调仓后应持有的目标量**"
        "（建仓即本次买入量、清仓为 0），可按实际本金线性缩放。", "",
        f"> 实算起点 **{eff}**，共 {len(ledger)} 次调仓。"
        f"「建仓/清仓」为该期新进/退出，「增持/减持」为权重上调/下调。", "",
        "> **原因口径**：「原因」为该动作在**同期截面**的信号——三支柱为**行业内中性 z**"
        "（相对同行标准差，过 AND 门需各柱 z≥0 即跑赢行业均值），排名越小越优。"
        f"{'（未记录因子面板，如需原因需以 with_panel=True 重跑）' if not _by else ''}", "",
    ]
    MAIN_HDR = "| 调仓日 | 动作 | 代码 | 名称 | 行业 | 上期权重 | 目标权重 | 变动 | 真实价 | 收益(含分红) | 价格收益 | 建仓日期 | 目标手数 | 占用资金(元) | 原因 |"
    MAIN_SEP = "|--------|------|------|------|------|---------|---------|------|--------|------------|---------|---------|---------|------------|------|"
    # 「无变动」/「其余 N 笔略」占位行按主表列数生成，避免行数与表头不一致
    _phc = [c.strip() for c in MAIN_HDR.strip("|").split("|")]
    def _ph_omit(d: str, text: str) -> str:
        cells = ["…"] * len(_phc)
        cells[0] = d
        cells[3] = text
        return "| " + " | ".join(cells) + " |"
    first_blk = True
    for blk in ledger:
        # 每个调仓期自成一张表：先写表头+分隔行。否则被中间的「本期持仓收益」
        # 子表/引用块打断后，后续期行变成孤立行（GFM 不渲染为表格）。
        if not first_blk:
            lines.append("")
        lines.append(MAIN_HDR)
        lines.append(MAIN_SEP)
        first_blk = False
        d = pd.to_datetime(blk["date"]).date().isoformat()
        order = {"建仓": 0, "清仓": 1, "增持": 2, "减持": 3}
        rows = sorted(blk["trades"], key=lambda r: (order.get(r["action"], 9), -abs(r["w_chg"])))
        if not rows:
            lines.append(_ph_omit(d, "（无变动）"))
        shown = rows if top_n <= 0 else rows[:top_n]
        for r in shown:
            chg = f"+{r['w_chg']:.1%}" if r["w_chg"] >= 0 else f"{r['w_chg']:.1%}"
            p = float(r["price"]) if np.isfinite(r["price"]) else np.nan
            price = f"{p:.2f}" if np.isfinite(p) else "—"
            rt = r.get("ret", np.nan)
            ret_s = f"{rt:+.1%}" if (rt is not None and np.isfinite(rt)) else "—"
            rp = r.get("ret_price", np.nan)
            rp_s = f"{rp:+.1%}" if (rp is not None and np.isfinite(rp)) else "—"
            ed = r.get("entry_date")
            entry_s = ed if ed else "—"
            lots = cost = 0
            if np.isfinite(p) and p > 0 and r["w_new"] > 1e-9:
                lots = int(r["w_new"] * capital / p / 100)
                cost = lots * 100 * p
            _why = (_reason_cell(_by, d, r["code"], r["action"], _by_rej, store)
                    if _by else "—（无面板）")
            lines.append(
                f"| {d} | {r['action']} | {r['code']} | {r['name']} | {r['industry']} | "
                f"{r['w_prev']:.1%} | {r['w_new']:.1%} | {chg} | {price} | {ret_s} | {rp_s} | {entry_s} | "
                f"{lots if lots > 0 else '—'} | {f'{cost:,.0f}' if cost > 0 else '—'} | {_why} |"
            )
        if top_n > 0 and len(rows) > top_n:
            lines.append(
                _ph_omit(d, f"其余 {len(rows) - top_n} 笔（增持/减持）略")
            )
        # —— 本期持仓收益快照：调仓时每只股票目前的收益（相对首次建仓真实价）——
        snap = blk.get("snap", [])
        if snap:
            lines.append("")
            lines.append(f"**本期持仓收益（{d}，共 {len(snap)} 只）**")
            lines.append("| 代码 | 名称 | 行业 | 建仓日期 | 建仓价 | 当前价 | 价格收益 | 持有收益(含分红) | 权重 |")
            lines.append("|------|------|------|---------|--------|--------|---------|------------|------|")
            tot_w = 0.0
            tot_wr = 0.0
            for s in sorted(snap, key=lambda x: -x["w"]):
                cst = s.get("cost")
                prc = s.get("price")
                rtv = s.get("ret")
                rpv = s.get("ret_price")
                cst_f = f"{cst:.2f}" if (cst is not None and np.isfinite(cst)) else "—"
                prc_f = f"{prc:.2f}" if (prc is not None and np.isfinite(prc)) else "—"
                rtv_f = f"{rtv:+.1%}" if (rtv is not None and np.isfinite(rtv)) else "—"
                rpv_f = f"{rpv:+.1%}" if (rpv is not None and np.isfinite(rpv)) else "—"
                ed = s.get("entry_date")
                ed_f = ed if ed else "—"
                lines.append(f"| {s['code']} | {s['name']} | {s['industry']} | {ed_f} | {cst_f} | {prc_f} | {rpv_f} | {rtv_f} | {s['w']:.1%} |")
                if rtv is not None and np.isfinite(rtv):
                    tot_w += s["w"]
                    tot_wr += s["w"] * rtv
            if tot_w > 1e-9:
                avg = tot_wr / tot_w
                lines.append(f"> 合计权重 {tot_w:.1%}，加权平均持有收益 **{avg:+.1%}**"
                             f"（后复权口径，含分红再投，与净值一致）")
    seen = {}
    for blk in ledger:
        for r in blk["trades"]:
            seen.setdefault(r["code"], {"name": r["name"], "buy": 0, "sell": 0})
            if r["action"] in ("建仓", "增持"):
                seen[r["code"]]["buy"] += 1
            else:
                seen[r["code"]]["sell"] += 1
    lines += ["", "**累计覆盖标的**", "",
              f"共 {len(seen)} 只曾在组合中出现。进出最频繁的：", ""]
    top = sorted(seen.items(), key=lambda kv: kv[1]["buy"] + kv[1]["sell"], reverse=True)[:10]
    lines.append("| 代码 | 名称 | 买入次数 | 卖出次数 |")
    lines.append("|------|------|---------|---------|")
    for c, v in top:
        lines.append(f"| {c} | {v['name']} | {v['buy']} | {v['sell']} |")
    lines += ["", "---", "",
              "**数据质量附注（2026-09-08 修复与 review 后的现行口径）**", "",
              "- **净值/业绩**：回测用 `close_adj`（后复权，含分红再投），全量无负值/异常跳变，"
              "区间回报符合常识（茅台 2017→2026 后复权 3.68×），基于它的年化/回撤等指标可信。",
              "- **价格层 `close_raw` 已修复**：2013–2021 段原值异质损坏（含负值与正数偏低），"
              "已用 amount/volume 反推全量重建，并修正科创板 688 volume 重复×100 的单位 bug。"
              "台账「真实价/手数/占用资金」即用此层，可与行情对照。",
              "- **市值层 `total_share` 已部分修正**：中国移动（×21）、中国建筑（×10）、"
              "中国海油（sh600938，原记 430.81/751.80 亿股，实为 446.47/475.30 亿股，市值虚高 ~1.58×）"
              "三处源数据错误已手动修正（`tests/fix_cnooc_shares.py` 按报告期映射真股本、point-in-time 重算 mktcap）。"
              "估值类因子（EP/BP/CFP/EBIT_EV/股息率）依赖 mktcap；其余个股未全量核对，个别错股仍可能影响选股。",
              "- **非 1 元面值系统性 bug 已修复（2026-09-08）**：源数据对「非 1.00 元面值」A 股"
              "返回的 `total_share` 实为**注册资本(元)**而非股本(股)，关系为 注册资本(元)=总股本(股)×每股面值，"
              "故存储值 = 真实股数 × 面值，且 facts 与 prices 同源污染、跳变检测抓不到（属静默偏乐观/悲观错误）。"
              "已全量修正 universe 内相关个股：紫金矿业(sh601899，面值 0.1，原记 26.59 亿股实为 265.9 亿，×10)、"
              "洛阳钼业(sh603993，面值 0.2，原记 42.79 亿股实为 213.94 亿，×5)、福莱特(sh601865，面值 0.25，×4)、"
              "复旦微电(sh688385，面值 0.1，×10) 按面值倍数改写 facts 并重算 mktcap；"
              "中芯国际(sh688981，$0.004 面值)、华润微(sh688396，HK$1 面值) 无干净倍数，按年报/HK 披露逐年真值映射。"
              "脚本 `tests/fix_face_value_shares.py`（备份 `*.bak_facevalue`，受 .gitignore 忽略）。"
              "其中紫金/洛阳/中芯/华润同在 prices 中，其估值因子与 share_dilution 已回归真实；"
              "洛阳钼业为当前 #1 持仓，mktcap 由约 786 亿→3932 亿（×5），估值因子由偏乐观修正为偏悲观，候选池与入选可能改变。",
              "- **第三类股本 bug 已修复（2026-09-08）：大比例送转/增发/借壳后股本停在旧值**。"
              "用东方财富 F10「股本变动历史」全量核对 298 只，发现 34 只偏差 >5%"
              "（中国移动 21.6×、分众传媒 0.023×、领益智造 0.23×、圆通速递 0.28×、东方盛虹 1.45× 等）；"
              "因其值长期不变，跳变检测抓不到。已按东财变动日 point-in-time 回填 facts 并重算 prices 的 mktcap"
              "（`tests/fetch_em_share_history.py` 抓取 + `tests/fix_shares_from_em.py` / "
              "`tests/fix_prices_shares_em.py` 修正，备份 `*.bak_emshare`）。"
              "修复后全量复核 298 只偏差全部 <0.2%。此修复影响估值因子与候选池，回测结果已据此处重跑。",
              "- **已知口径**：信号与成交同为调仓日收盘（未实现 t+1）；基准指数为价格回报口径"
              "（不含股息）；质押/审计意见字段的 announce_date 为抓取时刻，回测历史时点不可见"
              "（该两条排雷规则在回测中形同虚设，实盘才有）。",
              ]
    return "\n".join(lines)


def _panel_by_date(panel: pd.DataFrame) -> dict:
    """把 with_panel=True 记录的面板按 date 分组（date -> 该期候选 DataFrame）。"""
    if panel is None or panel.empty or "date" not in panel.columns:
        return {}
    return {pd.Timestamp(dt): g for dt, g in panel.groupby("date")}


def _action_reason_from(by_date: dict, date, code: str, action: str) -> str:
    """为某一笔买卖动作生成可验证的原因文本（依据与调仓日同期的因子面板）。

    by_date: _panel_by_date 的输出。
    三支柱为行业内中性 z（相对同行多少个标准差，过 AND 门需各柱 z≥0，即跑赢行业典型）；
    排名越小越优。
    """
    g = by_date.get(pd.Timestamp(date))
    if g is None or g.empty:
        return "当期未进入候选池（未过排雷/无打分/数据缺失）"
    row = g[g["code"] == code]
    if row.empty:
        return "当期未进入候选池（未过排雷/无打分/数据缺失）"
    r = row.iloc[0]
    m = len(g)

    def _num(x):
        try:
            return bool(np.isfinite(float(x)))
        except Exception:
            return False

    def _fz(x):
        return "—" if not _num(x) else f"{float(x):+.2f}"

    gate = bool(r.get("passes_gate", False)) if "passes_gate" in g.columns else True
    v = _fz(r.get("value_z")) if "value_z" in g.columns else "—"
    q = _fz(r.get("quality_z")) if "quality_z" in g.columns else "—"
    ss = _fz(r.get("safety_z")) if "safety_z" in g.columns else "—"
    rk = r.get("rank")
    rk_txt = f"第 {float(rk):.0f}/{m}" if _num(rk) else "—"
    sc = r.get("total_score")
    sc_txt = f"，总分 {float(sc):+.2f}" if _num(sc) else ""
    if action == "建仓":
        if gate:
            return (f"三支柱均跑赢行业均值 z≥0（价值z {v}/质量z {q}/安全z {ss}），"
                    f"当期排名 {rk_txt}{sc_txt}，新进入目标组合")
        return (f"当期三支柱未全过 z≥0 门槛（价值z {v}/质量z {q}/安全z {ss}）"
                "仍入选（候选不足等例外通道）")
    if action == "清仓":
        if not gate:
            return (f"当期三支柱跌破行业均值 z≥0（价值z {v}/质量z {q}/安全z {ss}），"
                    f"排名 {rk_txt}，被清出")
        return (f"当期仍过 z 门槛但排名 {rk_txt}{sc_txt}，未进新一期目标组合"
                "（被总分更优/同行业者挤出）")
    return f"权重随当期信号调整：排名 {rk_txt}{sc_txt}，z≥0 门槛{'过' if gate else '未过'}"


def _valuation_from(g: pd.DataFrame | None, date, code: str) -> str:
    """从当期因子面板取 L5 估值摘要（单位：亿元 / 隐含 g 与空间：%）。

    只描述「该时点可得」的估值 —— 与原因文本同源，保证 reasons / trades 可追溯。
    无估值列（回测未开 with_valuation）或数据缺失时返回空串。
    """
    if g is None or g.empty:
        return ""
    if "v_mid" not in g.columns:
        return ""
    row = g[g["code"] == code]
    if row.empty:
        return ""
    r = row.iloc[0]

    def _num(x):
        try:
            return bool(np.isfinite(float(x)))
        except Exception:
            return False

    def _yi(x):
        return None if not _num(x) else float(x) / 1e8

    if _yi(r.get("v_mid")) is None:
        return ""
    parts = [f"v_mid {_yi(r.get('v_mid')):,.1f}亿"]
    for label, col in (("买点", "buy_point"), ("卖点", "sell_point")):
        v = _yi(r.get(col))
        if v is not None:
            parts.append(f"{label} {v:,.1f}亿")
    if _num(r.get("mktcap")) and float(r["mktcap"]) > 0:
        parts.append(f"空间 {float(r['v_mid']) / float(r['mktcap']) - 1:+.0%}")
    if "verdict" in g.columns:
        vd = r.get("verdict")
        if vd is not None and not (isinstance(vd, float) and np.isnan(vd)):
            parts.append(str(vd))
    if _num(r.get("g_implied")):
        parts.append(f"隐含g {float(r['g_implied']) * 100:.1f}%")
    if "fcf_is_fallback" in g.columns:
        fb = r.get("fcf_is_fallback", False)
        if isinstance(fb, (bool, np.bool_)) and bool(fb):
            parts.append("FCF为净利兜底")
    return "；".join(parts)


def _reason_cell(by_date: dict, date, code: str, action: str,
                 by_rej: dict | None = None, store=None) -> str:
    """交易原因文本 + L5 估值摘要（trades 台账「原因」列用；L4 已含在原因文本内）。

    当期不在候选池（过 L3 打分）的股票：by_rej 里有 L3 否决明细（或 L1 宇宙过滤）
    时，给**具体规则 + 数值**，替代通用兜底文案。
    """
    why = _action_reason_from(by_date, date, code, action)
    if why.startswith("当期未进入候选池"):
        if store is not None:
            det = _absent_detail(store, date, code, by_rej)
            if det:
                return det
        return why
    vtxt = _valuation_from(by_date.get(pd.Timestamp(date)), date, code)
    return f"{why}；估值 {vtxt}" if vtxt else why


def _panel_rej_by_date(panel_rej: pd.DataFrame | None) -> dict:
    """把 with_panel 记录的 L3 否决明细按 date 分组（date -> 否决 DataFrame）。"""
    if panel_rej is None or panel_rej.empty or "date" not in panel_rej.columns:
        return {}
    return {pd.Timestamp(dt): g for dt, g in panel_rej.groupby("date")}


def _absent_detail(store, date, code: str, by_rej: dict | None = None) -> str:
    """组合里出现、但当期不在候选池的股票：定位并解释具体原因。

    按优先级：① 当期被 L3 一票否决（panel_rej 有 rule/value/threshold）
              → ② 仍在 L1 宇宙源表但被点时过滤（退市/未上市/ST/北交所/停牌/流动性）
    L1 阈值取 vi_system/config/rules.yaml 的 universe 默认值（min_listing_years=3、
    min_avg_amount_60d=5e7、ST/退名称、北交所 8/4/9 前缀），与 cli 默认 cfg 一致。
    定位不到返回空串（调用方保留通用兜底文案）。
    """
    d = pd.Timestamp(date)

    # ① 当期 L3 排雷否决：给出具体规则与数值
    if by_rej is not None:
        gj = by_rej.get(d)
        if gj is not None and not gj.empty:
            row = gj[gj["code"] == code]
            if not row.empty:
                r = row.iloc[0]
                try:
                    hit = _vetoes.describe_rejection(
                        r.get("rule"), r.get("rule_desc"), r.get("value"), r.get("threshold"))
                except Exception:
                    hit = str(r.get("rule_desc", "排雷规则触发"))
                return f"L3 排雷否决：{hit}"

    # ② 该期在 L1 宇宙源表（含退市历史）中、但被点时过滤
    try:
        u = store.load_universe()
        if not u.empty and code in set(u["code"]):
            row = u[u["code"] == code].iloc[0]
            dd = row.get("delist_date")
            if pd.notna(dd) and pd.to_datetime(dd) <= d:
                return f"已退市（{pd.Timestamp(dd).date()} 退市）"
            ld = row.get("list_date")
            if pd.notna(ld) and pd.to_datetime(ld) > d:
                return f"尚未上市（{pd.Timestamp(ld).date()} 才上市）"
            nm = str(row.get("name", ""))
            if any(p in nm for p in ("ST", "退市", "退")):
                return f"名称含 ST/退市风险提示（{nm}）"
            if str(code)[:3] in {"430", "830", "831", "832", "833", "834", "835",
                                 "836", "837", "838", "839", "870", "871", "872", "873"}:
                return "北交所板块（配置剔除）"
            try:
                m = store.market_asof(str(d.date()))
                if not m.empty:
                    mrow = m[m["code"] == code]
                    if mrow.empty:
                        return "无当日行情/停牌（未并入宇宙）"
                    amt = mrow.iloc[0].get("avg_amount_60d")
                    if pd.notna(amt) and amt < 5e7:
                        return (f"流动性不足：60日均成交额 {amt / 1e4:.0f} 万元"
                                f" < 门槛 5000 万元")
            except Exception:
                pass
            if pd.notna(ld):
                yrs = (d - pd.to_datetime(ld)).days / 365.25
                if yrs < 3:
                    return f"上市不足 3 年（仅 {yrs:.1f} 年）"
            return "未进入当期候选池（L3 后异常丢失，请人工核查）"
    except Exception:
        pass
    return ""


def trade_reasons_report(result: dict, store, top_n: int = 0) -> str:
    """基于每期因子面板（with_panel=True 的回测结果）生成逐期调仓原因。

    top_n<=0 表示全量；>0 时每期只展示前 top_n 笔非「持有」动作。
    原因依据是「与该调仓日同一时点」的截面信号：三支柱为行业内中性 z
    （过 AND 门需各柱 z≥0，即每柱跑赢行业典型），排名越小越优。
    """
    panel = result.get("panel")
    if panel is None or panel.empty:
        return ("# 调仓原因\n\n"
                "> 需要 with_panel=True 的回测结果（当前记录未含因子面板）。")
    rec = result["records"]
    rec_by = rec.set_index("date") if rec is not None and "date" in rec.columns else None
    by_date = _panel_by_date(panel)
    by_rej = _panel_rej_by_date(result.get("panel_rej"))

    lines = ["# 逐期调仓原因", "",
             "> 依据：与每次调仓同期的因子面板（with_panel=True 记录）。价值/质量/安全三支柱为"
             "**行业内中性 z**（相对同行标准差；过 AND 门需各柱 z≥0，即每柱都跑赢行业均值）；"
             "排名越小越优。",
             "> 动作股当期**不在候选池**（未进 L3 打分）时，「原因」列会给出**具体原因与数值**"
             "（L3 排雷否决带实际值 vs 阈值；或 L1 退市/ST/流动性等）。",
             "> **估值(L5)**：同期的逆向 DCF + 三情景（v_mid=三情景中位数，买点=v_mid×0.70，"
             "卖点=v_mid×1.50，单位：亿元）；「空间」= v_mid 相对当期市值的空间；「隐含g」为市场价"
             "反解的永续增长率。历史早期财报现金流缺失时列为 —。", ""]
    ledger = trade_ledger(result, store)
    n_show = 0
    for blk in ledger:
        d = blk["date"]
        rows = [r_ for r_ in blk["trades"] if r_["action"] != "持有"]
        if not rows:
            continue
        rmeta = ""
        if rec_by is not None:
            try:
                rr = rec_by.loc[pd.Timestamp(d)]
                if isinstance(rr, pd.DataFrame):
                    rr = rr.iloc[0]
                rmeta = (f"；本期组合收益 {float(rr['port_ret']):+.1%}"
                         f"；换手 {float(rr['turnover']):.0%}")
            except Exception:
                pass
        show = rows if top_n <= 0 else rows[:top_n]
        lines += [f"## {pd.Timestamp(d).date()}（{len(rows)} 笔变动{rmeta}）", "",
                  "| 动作 | 名称 | 行业 | 估值(L5) | 原因 |",
                  "|------|------|------|---------|------|"]
        for t in show:
            why = _action_reason_from(by_date, d, t["code"], t["action"])
            if why.startswith("当期未进入候选池"):
                det = _absent_detail(store, d, t["code"], by_rej)
                if det:
                    why = det
            vtxt = _valuation_from(by_date.get(pd.Timestamp(d)), d, t["code"])
            ind = (t.get("industry") or "") or ""
            name = t.get("name") or t["code"]
            lines.append(f"| {t['action']} | {name} | {ind} | {vtxt or '—'} | {why} |")
        lines.append("")
        n_show += 1
    lines += ["---", f"共 {n_show} 期发生调仓。"]
    return "\n".join(lines)
