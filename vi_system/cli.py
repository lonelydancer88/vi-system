#!/usr/bin/env python3
"""价值投资选股系统 —— 命令行入口。

    python -m vi_system.cli demo                  # 生成合成数据（无 token 也能跑）
    python -m vi_system.cli quality               # 数据质量闸门
    python -m vi_system.cli screen --asof 2024-05-15
    python -m vi_system.cli backtest --split      # 样本内 + 样本外
    python -m vi_system.cli sensitivity
    python -m vi_system.cli monitor --asof 2024-05-15
    python -m vi_system.cli log-add --code X --action 买入 ...
    python -m vi_system.cli review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data.store import Store
from .data import quality as quality_mod
from .backtest import engine as bt
from .pipeline import valuation as valuation_mod
from .portfolio import constructor as portfolio_mod
from .monitor import alerts as alerts_mod
from .journal import decision_log as journal_mod

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "vi_db"
DEFAULT_OUT = ROOT / "out"


def _store(db: str) -> Store:
    return Store(db)


def _cfg(path: str | None):
    return load_config(path)


def _write(text: str, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(text, encoding="utf-8")
    print(f"→ 已写入 {p}")


# ==================================================================== 子命令
def cmd_demo(args):
    from .data.synthetic import GenConfig, build_demo_store
    st = build_demo_store(
        args.db,
        GenConfig(n_stocks=args.n_stocks, seed=args.seed,
                  fund_start_year=args.fund_start, fund_end_year=args.fund_end,
                  price_start=args.price_start),
    )
    print("合成数据库已生成：", st.summary())
    print("⚠️  合成数据仅用于流程验证，绝不用于任何真实结论。")


def cmd_fetch(args):
    from .data.fetcher import TushareFetcher
    st = _store(args.db)
    f = TushareFetcher(args.token)
    uni = f.fetch_universe()
    st.save_universe(uni)
    print(f"宇宙：{len(uni)} 只")
    codes = uni["code"].tolist()
    if args.limit:
        codes = codes[: args.limit]
    prices = f.fetch_prices(codes, args.start, args.end)
    st.save_prices(prices)
    print(f"行情：{len(prices)} 行")
    facts = f.fetch_financials(codes, args.start_period, args.end_period)
    st.save_facts(facts)
    print(f"财报：{len(facts)} 行")
    try:
        st.save_facts(f.fetch_pledge(codes))
        st.save_facts(f.fetch_audit(codes))
    except Exception as e:
        print("质押/审计数据获取失败（不影响主流程）：", e)


def cmd_quality(args):
    st = _store(args.db)
    rep = quality_mod.run_all_checks(st, args.asof)
    print("库概况：", rep["summary"])
    for k in ("restatements", "balance_sheet_violations", "stale_periods"):
        d = rep[k]
        print(f"\n== {k}: {len(d)} 条")
        if len(d):
            print(d.head(10).to_string())
    print("\n== 字段覆盖率（最低 10 项）")
    print(rep["coverage"].head(10).to_string())


def cmd_screen(args):
    cfg = _cfg(args.config)
    st = _store(args.db)
    scored, rejected, uni = bt.screen_at(st, args.asof, cfg, with_valuation=True)
    if scored.empty:
        print("无候选（宇宙为空或全部被排雷否决）")
        return
    # 排雷防线健康度：失效规则必须显式告警，避免静默失效给人虚假安全感
    from .pipeline.vetoes import warn_dead_rules, rule_coverage
    cov = rule_coverage(scored, cfg)
    for w in warn_dead_rules(scored, cfg):
        print(w)
    print(f"宇宙 {len(uni)} → 通过排雷 {len(scored)} → 过 AND 门槛 {int(scored['passes_gate'].sum())}")
    _write(_veto_md(rejected), Path(args.out), f"vetoes-{args.asof}.md")
    _write(valuation_mod.valuation_report(scored, args.top), Path(args.out), f"valuation-{args.asof}.md")
    pf = portfolio_mod.build_portfolio(scored, cfg)
    _write(portfolio_mod.portfolio_report(pf, cfg), Path(args.out), f"portfolio-{args.asof}.md")
    print(f"组合：{len(pf)} 只，现金 {max(0.0, 1 - pf['weight'].sum()):.1%}")


def _veto_md(rejected: pd.DataFrame) -> str:
    from .pipeline.vetoes import veto_summary
    if rejected.empty:
        return "# 排雷报告\n\n无否决记录"
    s = veto_summary(rejected)
    lines = ["# 排雷报告（L3）", "", f"共否决 {len(rejected)} 只", "",
             "| 规则 | 说明 | 数量 | 占比 |", "|------|------|------|------|"]
    for _, r in s.iterrows():
        lines.append(f"| {r['rule']} | {r['rule_desc']} | {r['n']} | {r['share']:.1%} |")
    lines += ["", "## 明细", "", "| 代码 | 名称 | 行业 | 触发 | 实际值 | 阈值 |",
              "|------|------|------|---------|--------|------|"]
    for _, r in rejected.iterrows():
        v = r["value"]
        v = f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(f"| {r['code']} | {r['name']} | {r['industry']} | {r['rule_desc']} | {v} | {r['threshold']} |")
    return "\n".join(lines)


def cmd_portfolio(args):
    cfg = _cfg(args.config)
    st = _store(args.db)
    scored, _, _ = bt.screen_at(st, args.asof, cfg, with_valuation=True)
    from .pipeline.vetoes import warn_dead_rules
    for w in warn_dead_rules(scored, cfg):
        print(w)
    pf = portfolio_mod.build_portfolio(scored, cfg)
    print(portfolio_mod.portfolio_report(pf, cfg))
    if args.out:
        pf.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"→ 已写入 {args.out}")


def cmd_backtest(args):
    cfg = _cfg(args.config)
    st = _store(args.db)
    if args.split:
        res = bt.split_sample(st, cfg)
        print(bt.backtest_report(res))
        _write(bt.backtest_report(res), Path(args.out), "backtest-split.md")
    else:
        r = bt.run_backtest(st, cfg, args.start, args.end, label=args.label)
        if r.get("error"):
            print("回测失败：", r["error"])
            return
        print(bt.backtest_report({"full": r}))
        _write(bt.backtest_report({"full": r}), Path(args.out), "backtest.md")

    # 指数基准对比（如 --benchmark sh000300）
    if args.benchmark:
        idx_name = {"sh000300": "沪深300", "sz399905": "中证500"}.get(
            args.benchmark, args.benchmark)
        try:
            rep = bt.compare_benchmarks(
                st, cfg, args.start, args.end,
                index_code=args.benchmark, index_name=idx_name)
            print("\n" + rep)
            _write(rep, Path(args.out), "backtest-hs300.md")
        except Exception as e:
            print("[warn] 指数基准对比失败：", e)


def cmd_fetch_index(args):
    """抓取并存储指数日线（如沪深300 sh000300），供回测作基准。"""
    from .data.westock import WeStockFetcher
    st = _store(args.db)
    f = WeStockFetcher()
    print(f"抓取指数 {args.code} 行情 {args.start} ~ {args.end} ...")
    df = f.index_prices(args.code, args.start, args.end)
    if df.empty:
        print("未获取到数据（检查代码格式：指数需带市场前缀，如 sh000300）")
        return
    st.save_index_prices(df)
    print(f"已存储 {len(df)} 行 → {st.index_path}")
    print(df.head(3).to_string(index=False))


def cmd_sensitivity(args):
    cfg = _cfg(args.config)
    st = _store(args.db)
    df = bt.sensitivity_test(st, cfg, args.start, args.end)
    print(df.to_string(index=False))
    print("\n判据：扰动后年化收益发生质变（如由正转负）＝ 过拟合，需推倒重来。")


def cmd_monitor(args):
    cfg = _cfg(args.config)
    st = _store(args.db)
    scored, _, _ = bt.screen_at(st, args.asof, cfg, with_valuation=True)
    from .pipeline.vetoes import warn_dead_rules
    for w in warn_dead_rules(scored, cfg):
        print(w)
    if scored.empty:
        print("无候选")
        return
    th = alerts_mod.thesis_alerts(scored, cfg)
    pr = alerts_mod.price_alerts(scored)
    cw = alerts_mod.crowding_metrics(st, scored["code"].tolist(), args.asof, cfg)
    print(alerts_mod.weekly_report(th, pr, cw))


def cmd_log_add(args):
    cfg = _cfg(args.config)
    log = journal_mod.DecisionLog(Path(args.db).parent / "decisions.jsonl", cfg)
    rec = log.add(args.code, args.name or args.code, args.action, args.price,
                  args.weight, None, args.thesis, args.confidence)
    print("已记录：", rec["ts"], rec["code"], rec["action"])


def cmd_review(args):
    cfg = _cfg(args.config)
    log = journal_mod.DecisionLog(Path(args.db).parent / "decisions.jsonl", cfg)
    print(journal_mod.review_report(log))


# ==================================================================== 参数解析
def main(argv=None):
    p = argparse.ArgumentParser(prog="vi_system", description="价值投资选股系统")
    p.add_argument("--db", default=str(DEFAULT_DB), help="数据目录")
    p.add_argument("--config", default=None, help="规则配置文件路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="生成合成数据")
    d.add_argument("--n-stocks", type=int, default=150)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--fund-start", type=int, default=2007)
    d.add_argument("--fund-end", type=int, default=2026)
    d.add_argument("--price-start", default="2010-01-01")
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fetch", help="抓取真实数据（需 tushare token）")
    f.add_argument("--token", default=None)
    f.add_argument("--start", default="20100101")
    f.add_argument("--end", default="20261231")
    f.add_argument("--start-period", default="20050101")
    f.add_argument("--end-period", default="20261231")
    f.add_argument("--limit", type=int, default=0)
    f.set_defaults(func=cmd_fetch)

    q = sub.add_parser("quality", help="数据质量闸门")
    q.add_argument("--asof", default="2024-05-15")
    q.set_defaults(func=cmd_quality)

    s = sub.add_parser("screen", help="跑 L1→L6 输出报告")
    s.add_argument("--asof", default="2024-05-15")
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.set_defaults(func=cmd_screen)

    pf = sub.add_parser("portfolio", help="构建组合")
    pf.add_argument("--asof", default="2024-05-15")
    pf.add_argument("--out", default=None)
    pf.set_defaults(func=cmd_portfolio)

    b = sub.add_parser("backtest", help="回测")
    b.add_argument("--split", action="store_true", help="样本内/样本外分别跑")
    b.add_argument("--start", default="2013-01-01")
    b.add_argument("--end", default="2026-12-31")
    b.add_argument("--label", default="full")
    b.add_argument("--benchmark", default=None,
                   help="指数基准代码（如 sh000300），生成双基准对比报告")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.set_defaults(func=cmd_backtest)

    fi = sub.add_parser("fetch-index", help="抓取并存储指数日线（如沪深300 sh000300）")
    fi.add_argument("--code", default="sh000300", help="指数代码，需带市场前缀")
    fi.add_argument("--start", default="2013-01-01")
    fi.add_argument("--end", default="2026-12-31")
    fi.set_defaults(func=cmd_fetch_index)

    sn = sub.add_parser("sensitivity", help="参数敏感性检验")
    sn.add_argument("--start", default="2013-01-01")
    sn.add_argument("--end", default="2026-12-31")
    sn.set_defaults(func=cmd_sensitivity)

    m = sub.add_parser("monitor", help="周报：论文/价格报警 + 拥挤度")
    m.add_argument("--asof", default="2024-05-15")
    m.set_defaults(func=cmd_monitor)

    la = sub.add_parser("log-add", help="记录决策")
    la.add_argument("--code", required=True)
    la.add_argument("--name", default=None)
    la.add_argument("--action", required=True)
    la.add_argument("--price", type=float, default=0.0)
    la.add_argument("--weight", type=float, default=None)
    la.add_argument("--thesis", required=True)
    la.add_argument("--confidence", default="中")
    la.set_defaults(func=cmd_log_add)

    rv = sub.add_parser("review", help="复盘报告")
    rv.set_defaults(func=cmd_review)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
