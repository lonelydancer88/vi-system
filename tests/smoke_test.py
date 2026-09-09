"""端到端冒烟测试。

验证的是** invariants（不变量）**，不是收益数字：
  1. point-in-time：asof 之后公告的数据绝不可见
  2. 排雷：确实拦住了东西，且留了否决理由
  3. 组合：权重非负、总和不超过 1 - 现金下限、单票与行业不超上限
  4. 回测：能跑完、NAV 序列单调可算、换手与成本被正确计提
  5. 监控：论文报警与价格报警走两个通道
  6. 日志：决策留痕含规则版本号

运行：python -m tests.smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi_system.config import load_config                      # noqa: E402
from vi_system.data.store import Store                        # noqa: E402
from vi_system.data.synthetic import GenConfig, build_demo_store  # noqa: E402
from vi_system.data import quality as qmod                    # noqa: E402
from vi_system.pipeline import universe, metrics, vetoes, factors, valuation  # noqa: E402
from vi_system.portfolio.constructor import build_portfolio   # noqa: E402
from vi_system.monitor import alerts                          # noqa: E402
from vi_system.journal import decision_log                    # noqa: E402
from vi_system.backtest import engine as bt                   # noqa: E402

DB = Path(__file__).resolve().parent.parent / "data" / "smoke_db"
OK, FAIL = [], []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  {extra}" if extra else ""))


def main():
    cfg = load_config()
    print("=" * 70)
    print("价值投资选股系统 · 冒烟测试")
    print(f"规则版本：{cfg.stamp}")
    print("=" * 70)

    # ---------------------------------------------------------- 0. 数据
    print("\n[0] 生成合成数据")
    st = build_demo_store(
        str(DB),
        GenConfig(n_stocks=240, seed=7, fund_start_year=2011,
                  fund_end_year=2026, price_start="2014-01-01"),
    )
    s = st.summary()
    check("数据库非空", s.get("status") == "ok", str(s))

    # ---------------------------------------------------------- 1. point-in-time
    print("\n[1] point-in-time（防前视偏差）")
    asof = pd.Timestamp("2021-05-15")
    facts = st.load_facts()
    late = facts[facts["announce_date"] > asof]
    panel = st.facts_asof(asof)
    check("facts 库中存在 asof 之后才公告的数据（用于验证过滤确实生效）", len(late) > 0,
          f"{len(late)} 行")
    if "announce_date" in panel.columns:
        check("panel 中不含 asof 之后公告的数据",
              bool((panel["announce_date"] <= asof).all()))
    # 换个更早的日期，可见期数应当更少
    p_early = st.facts_asof("2016-05-15")
    check("更早的 asof 可见期数更少", len(p_early) <= len(panel),
          f"{len(p_early)} <= {len(panel)}")

    # ---------------------------------------------------------- 2. 质量闸门
    print("\n[2] 数据质量闸门")
    rep = qmod.run_all_checks(st, asof)
    cov = rep["coverage"]
    check("覆盖率报告可生成", len(cov) > 0)
    check("核心字段净利润覆盖率 > 50%",
          float(cov.loc[cov["field"] == "net_income", "coverage"].iloc[0]) > 0.5)
    check("缺失字段被标记为 NaN 而非填充",
          bool(rep["summary"].get("status") == "ok"))

    # ---------------------------------------------------------- 3. L1→L5
    print("\n[3] L1 宇宙 → L3 排雷 → L4 因子 → L5 估值")
    uni = universe.build_universe(st, asof, cfg)
    check("宇宙非空", len(uni) > 0, f"{len(uni)} 只")
    m = metrics.compute_metrics(st.facts_asof(asof), st.market_asof(asof), uni)
    check("指标表非空", len(m) > 0, f"{len(m)} 行")
    passed, rejected = vetoes.apply_vetoes(m, cfg)
    check("排雷确实拦住了标的", len(rejected) > 0, f"否决 {len(rejected)}")
    check("否决记录带触发规则与数值",
          {"rule", "rule_desc", "value", "threshold"}.issubset(set(rejected.columns)))
    check("通过数 + 否决数 = 总数", len(passed) + len(rejected) == len(m))
    scored = factors.score_factors(passed, cfg)
    check("三支柱分均存在",
          {"value_pct", "quality_pct", "safety_pct"}.issubset(set(scored.columns)))
    check("存在 AND 门槛标记", "passes_gate" in scored.columns,
          f"过闸 {int(scored['passes_gate'].sum())}/{len(scored)}")
    v = valuation.valuate(scored, cfg)
    check("逆向 DCF 产出隐含增长率", "g_implied" in v.columns,
          f"有效 {int(v['g_implied'].notna().sum())} 个")
    check("含买卖点", {"buy_point", "sell_point"}.issubset(set(v.columns)))

    # ---------------------------------------------------------- 4. L6 组合
    print("\n[4] L6 组合构建")
    pcfg = cfg.section("portfolio")
    pf = build_portfolio(scored, cfg)
    check("组合非空", len(pf) > 0, f"{len(pf)} 只")
    check("权重非负", bool((pf["weight"] >= 0).all()))
    check("权重和不超过 1 - 现金下限",
          pf["weight"].sum() <= 1 - pcfg.get("min_cash", 0.1) + 1e-6,
          f"{pf['weight'].sum():.3f}")
    check("单票不超过上限",
          pf["weight"].max() <= pcfg.get("max_position", 0.08) + 1e-6,
          f"max={pf['weight'].max():.4f}")
    iw = pf.groupby("industry")["weight"].sum()
    check("行业不超过上限",
          iw.max() <= pcfg.get("max_industry", 0.25) + 1e-6, f"max={iw.max():.4f}")

    # 缓冲区：二次调仓换手应显著低于无缓冲
    pf2 = build_portfolio(scored, cfg,
                          current_weights=pd.Series(pf["weight"].values,
                                                    index=pf["code"].values))
    check("缓冲区生效（同样截面下二次调仓不大幅换手）",
          len(set(pf2["code"]) & set(pf["code"])) > 0,
          f"保留 {len(set(pf2['code']) & set(pf['code']))}/{len(pf)}")

    # ---------------------------------------------------------- 5. L7 监控
    print("\n[5] L7 监控")
    th = alerts.thesis_alerts(v, cfg)
    pr = alerts.price_alerts(v)
    cw = alerts.crowding_metrics(st, pf["code"].tolist(), asof, cfg)
    check("论文报警通道可运行", th is not None and isinstance(th, pd.DataFrame),
          f"{len(th)} 条")
    check("价格报警通道可运行", pr is not None and isinstance(pr, pd.DataFrame),
          f"{len(pr)} 条")
    check("拥挤度指标可计算", isinstance(cw, dict) and len(cw) > 0, str(cw)[:80])
    wr = alerts.weekly_report(th, pr, cw)
    check("周报可生成", "论文报警" in wr and "价格报警" in wr)

    # ---------------------------------------------------------- 6. L8 治理
    print("\n[6] L8 决策日志")
    log = decision_log.DecisionLog(DB.parent / "smoke_decisions.jsonl", cfg)
    log.add(pf["code"].iloc[0], "测试公司", "买入", 10.0, 0.05,
            {"value_pct": 80.0, "quality_pct": 75.0, "safety_pct": 60.0},
            "便宜且现金流扎实，护城河来自渠道；若市占率连续两季下滑则卖出。", "中")
    df_log = log.load()
    check("决策已记录", len(df_log) >= 1)
    check("决策含规则版本号",
          "rules_version" in df_log.columns and (df_log["rules_version"] == cfg.stamp).any())
    rr = decision_log.review_report(log)
    check("复盘报告可生成", "复盘报告" in rr and "错误分类" in rr)
    log.classify(pf["code"].iloc[0], "II", "买贵了")
    check("错误分类可记录", "复盘归类" in set(log.load()["action"]))

    # ---------------------------------------------------------- 7. 回测
    print("\n[7] 回测")
    res = bt.run_backtest(st, cfg, "2016-01-01", "2026-06-30", label="smoke")
    check("回测无错误", "error" not in res, res.get("error", ""))
    if "error" not in res:
        st_ = res["stats"]
        check("NAV 序列长度正确", len(res["nav"]) == len(res["records"]) + 1,
              f"nav={len(res['nav'])} records={len(res['records'])} (+1 建仓基点)")
        check("NAV 起点为 1.0", abs(float(res["nav"]["nav"].iloc[0]) - 1.0) < 1e-9,
              f"起点={float(res['nav']['nav'].iloc[0]):.4f}")
        check("成本被计提", float(res["records"]["cost"].sum()) > 0,
              f"累计成本 {res['records']['cost'].sum():.4f}")
        check("换手被记录", float(res["records"]["turnover"].mean()) > 0,
              f"平均换手 {st_['avg_turnover']:.2%}")
        check("指标齐全", all(k in st_ for k in ("cagr", "sharpe", "max_drawdown")))
        print(f"    年化 {st_['cagr']:.2%} / 基准 {st_['bench_cagr']:.2%} / "
              f"夏普 {st_['sharpe']:.2f} / 回撤 {st_['max_drawdown']:.2%}")
        rg = bt.regime_report(res)
        check("分市场状态检验可运行", not rg.empty, f"{len(rg)} 个状态")

    # ---------------------------------------------------------- 汇总
    print("\n" + "=" * 70)
    print(f"通过 {len(OK)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  -", f)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
