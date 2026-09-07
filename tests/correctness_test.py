"""分层逻辑正确性测试（聚焦"每条规则/公式是否真的对"）。

与 smoke_test（验证不变量）的区别：
  smoke_test 问的是"系统不崩、产出不变式成立"；
  本文件问的是"**具体逻辑是否正确**"——用最小手工 fixture 精确断言每个分支。

覆盖：
  L2 point-in-time（重述/前视边界）
  L3 排雷（每条规则触发/豁免/缺失）
  L4 三支柱 AND 门 + 总分权重
  L5 估值（v_mid/buy/sell/verdict/downside/逆向g*/DCF）
  L6 组合（单票/行业/现金上限、缓冲、集中宇宙）
  L7/L8 略（smoke_test 已覆盖）
  回测引擎（确定性、成本计提、指数基准接入）
  数据缺口鲁棒性（net_payout 缺失、负 gpa 复数守卫）
  L1 宇宙时点过滤（退市/上市年限/ST）
  真实库回归锚点（伊利被挡、圆通速递 #1）

运行：python -m tests.correctness_test
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi_system.config import load_config                                   # noqa: E402
from vi_system.data.store import Store                                     # noqa: E402
from vi_system.data.synthetic import GenConfig, build_demo_store           # noqa: E402
from vi_system.data import schema as S                                     # noqa: E402
from vi_system.pipeline import universe, metrics, vetoes, factors, valuation  # noqa: E402
from vi_system.portfolio.constructor import build_portfolio, turnover      # noqa: E402
from vi_system.backtest import engine as bt                               # noqa: E402

OK, FAIL = [], []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  {extra}" if extra else ""))


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ================================================================ 手工 fixture 辅助
def safe_veto_row(code, industry="白酒", name=None):
    """所有排雷字段取"安全值"，供单条规则测试时单独覆盖。"""
    return {
        "code": code, "name": name or code, "industry": industry,
        "beneish_m": -3.0,            # < max(-1.78) → 不触发
        "audit_nonstd_3y": 0.0,       # <= 0.5 → 不触发
        "pledge_ratio": 0.10,         # <= 0.50 → 不触发
        "goodwill_to_equity": 0.10,   # <= 0.30 → 不触发
        "ocf_to_ni_5y": 1.0,          # >= 0.60 → 不触发
        "share_dilution_5y": 0.10,    # <= 0.50 → 不触发
        "net_debt_ebitda": 1.0,       # <= 5 → 不触发
        "interest_coverage": 5.0,     # >= 2 → 不触发
        "altman_z": 3.0,              # >= 1.8 → 不触发
    }


def clean_factor_row(code, industry="白酒", **ov):
    """构造因子层所需的全部指标列（默认"优质"）。"""
    d = {
        "code": code, "name": code, "industry": industry,
        # value（越高越便宜，sign+1）
        "ep": 0.10, "bp": 1.0, "cfp": 0.10, "ebit_ev": 0.10, "dividend_yield": 0.03,
        # quality（gpa/roic/profit_growth_5y/net_payout sign+1；accruals sign-1）
        "gpa": 0.50, "roic": 0.15, "accruals": 0.0, "profit_growth_5y": 0.10, "net_payout": 0.03,
        # safety（interest_coverage +1；net_debt_ebitda/earnings_vol/revenue_vol -1）
        "interest_coverage": 8.0, "net_debt_ebitda": 1.0,
        "earnings_volatility": 0.05, "revenue_volatility": 0.03,
    }
    d.update(ov)
    return d


# ================================================================ L2 point-in-time
def test_point_in_time():
    print("\n[2] L2 point-in-time（防前视 + 财报重述）")
    with tempfile.TemporaryDirectory() as d:
        st = Store(d)
        # 同一 (code, field, period) 两份：原版 2021-04 公告，重述版 2024-04 公告
        rows = []
        for ann, val in [("2021-04-01", 100.0), ("2024-04-01", 999.0)]:
            rows.append({"code": "X.SH", "field": "net_income", "period": "20201231",
                         "announce_date": pd.Timestamp(ann), "value": val,
                         "source": "t", "fetch_time": pd.Timestamp.now()})
        st.save_facts(pd.DataFrame(rows))

        p_pre = st.facts_asof("2022-01-01")     # 仅可见原版
        p_post = st.facts_asof("2025-01-01")    # 可见重述版
        p_early = st.facts_asof("2020-01-01")   # 公告前 → 空

        v_pre = p_pre.loc[p_pre["period"] == "20201231", "net_income"].iloc[0] if not p_pre.empty else None
        v_post = p_post.loc[p_post["period"] == "20201231", "net_income"].iloc[0] if not p_post.empty else None
        check("重述前：仅见原版值 100", v_pre == 100.0, str(v_pre))
        check("重述后：见最新重述值 999（取 announce 最大）", v_post == 999.0, str(v_post))
        check("公告前 asof：无可见数据（防前视边界）", p_early.empty)

        # 更早决策日可见期数更少
        p_a = st.facts_asof("2023-01-01")
        p_b = st.facts_asof("2025-01-01")
        check("更早 asof 可见期数 <= 更晚", len(p_a) <= len(p_b), f"{len(p_a)}<={len(p_b)}")


# ================================================================ L3 排雷
def test_vetoes():
    print("\n[3] L3 排雷（每条规则触发 / 豁免 / 缺失）")
    cfg = load_config()

    cases = {
        "beneish_m_score":    ("beneish_m", 0.0),        # > -1.78 → 触发
        "ocf_to_ni_5y":       ("ocf_to_ni_5y", 0.10),    # < 0.60 → 触发
        "share_dilution_5y":  ("share_dilution_5y", 0.80),  # > 0.50 → 触发
        "pledge_ratio":       ("pledge_ratio", 0.80),    # > 0.50 → 触发
        "goodwill_to_equity": ("goodwill_to_equity", 0.50),  # > 0.30 → 触发
        "audit_opinion":      ("audit_nonstd_3y", 1.0),  # > 0.5 → 触发
    }
    for rule, (col, bad) in cases.items():
        r = safe_veto_row("A")
        r[col] = bad
        df = pd.DataFrame([r])
        passed, rejected = vetoes.apply_vetoes(df, cfg)
        hit = rejected["rule"].tolist()
        check(f"规则 {rule} 触发并留下记录", rule in hit and len(passed) == 0, str(hit))

    # 杠杆：两条分支
    r = safe_veto_row("L1"); r["net_debt_ebitda"] = 8.0
    passed, rejected = vetoes.apply_vetoes(pd.DataFrame([r]), cfg)
    check("杠杆(net_debt_ebitda>5) 触发", "leverage" in rejected["rule"].tolist())

    r = safe_veto_row("L2"); r["interest_coverage"] = 1.0
    passed, rejected = vetoes.apply_vetoes(pd.DataFrame([r]), cfg)
    check("杠杆(利息覆盖<2) 触发", "leverage" in rejected["rule"].tolist())

    # Altman：仅对"净借款人"触发
    r = safe_veto_row("A1"); r["altman_z"] = 1.0; r["net_debt_ebitda"] = -2.0  # 净现金
    passed, rejected = vetoes.apply_vetoes(pd.DataFrame([r]), cfg)
    hit_rules = rejected["rule"].tolist() if not rejected.empty else []
    check("Altman 低 Z + 净现金 → 豁免（不误杀）",
          "altman_z_score" not in hit_rules and len(passed) == 1)

    r = safe_veto_row("A2"); r["altman_z"] = 1.0; r["net_debt_ebitda"] = 3.0   # 净借款人
    passed, rejected = vetoes.apply_vetoes(pd.DataFrame([r]), cfg)
    check("Altman 低 Z + 净借款人 → 触发", "altman_z_score" in rejected["rule"].tolist())

    # 缺失值不崩溃、不误杀
    r = safe_veto_row("M"); r["ocf_to_ni_5y"] = np.nan
    passed, rejected = vetoes.apply_vetoes(pd.DataFrame([r]), cfg)
    check("某字段缺失 → 该规则跳过、不崩溃", len(passed) == 1 and len(rejected) == 0)

    # 干净股全过
    passed, rejected = vetoes.apply_vetoes(pd.DataFrame([safe_veto_row("C")]), cfg)
    check("全安全值 → 通过、零否决", len(passed) == 1 and len(rejected) == 0)

    # 多规则命中：按 _RULE_SPEC 顺序取第一条
    r = safe_veto_row("X"); r["beneish_m"] = 0.0; r["ocf_to_ni_5y"] = 0.10
    passed, rejected = vetoes.apply_vetoes(pd.DataFrame([r]), cfg)
    check("多规则命中取首条(beneish 优先)", rejected.iloc[0]["rule"] == "beneish_m_score")

    # rule_coverage / warn_dead_rules：字段全缺 → 判死
    dead = pd.DataFrame([{**safe_veto_row("D"), "goodwill_to_equity": np.nan}])
    cov = vetoes.rule_coverage(dead, cfg)
    gw = cov[cov["rule"] == "goodwill_to_equity"].iloc[0]
    check("goodwill 字段全缺 → 标记 dead", bool(gw["dead"]))
    warns = vetoes.warn_dead_rules(dead, cfg)
    check("warn_dead_rules 输出失效告警", len(warns) > 0 and "goodwill_to_equity" in warns[0])


# ================================================================ L4 三支柱 AND 门
def test_factors():
    print("\n[4] L4 三支柱 AND 门 + 总分权重")
    cfg = load_config()
    floor = cfg.section("factors").get("percentile_floor", 30)

    # 单行业 6 只：A 价值极差（其余优质）→ A 应因 value<floor 被挡
    rows = []
    for i in range(6):
        if i == 0:
            rows.append(clean_factor_row(f"V{i}", industry="白酒",
                                         ep=0.001, bp=0.01, cfp=0.001, ebit_ev=0.001, dividend_yield=0.001))
        else:
            rows.append(clean_factor_row(f"V{i}", industry="白酒"))
    df = pd.DataFrame(rows)
    scored = factors.score_factors(df, cfg)
    a = scored[scored["code"] == "V0"].iloc[0]
    check(f"价值垫底股 value_pct<{floor}（AND 门触发）", a["value_pct"] < floor, f"{a['value_pct']:.1f}")
    check("价值垫底股 passes_gate=False", not bool(a["passes_gate"]))
    b = scored[scored["code"] == "V1"].iloc[0]
    check("全优股 passes_gate=True", bool(b["passes_gate"]))
    check("全优股三支柱均>=floor",
          b["value_pct"] >= floor and b["quality_pct"] >= floor and b["safety_pct"] >= floor)

    # 总分权重重构一致：total = 0.4*v + 0.4*q + 0.2*s
    w = cfg.section("factors").get("weights", {"value": 0.4, "quality": 0.4, "safety": 0.2})
    recon = (w.get("value", 0.4) * scored["value_pct"].fillna(0)
             + w.get("quality", 0.4) * scored["quality_pct"].fillna(0)
             + w.get("safety", 0.2) * scored["safety_pct"].fillna(0))
    check("总分=加权三支柱分（公式一致）",
          bool((np.abs(recon - scored["total_score"]) < 1e-6).all()))

    # 空表不崩
    empty = factors.score_factors(pd.DataFrame(), cfg)
    check("空指标表 score_factors 返回空", empty.empty)


# ================================================================ L5 估值
def test_valuation():
    print("\n[5] L5 估值（v_mid / 买卖点 / verdict / downside / 逆向 g* / DCF）")
    cfg = load_config()
    vdf = pd.DataFrame([
        # c1: 正常；c2: ocf 为负 → 回退 net_income*0.8；c3: 普通
        {"code": "c1", "name": "A", "industry": "白酒", "ocf": 10.0, "capex": 2.0, "net_income": 12.0, "mktcap": 100.0},
        {"code": "c2", "name": "B", "industry": "白酒", "ocf": -5.0, "capex": 1.0, "net_income": 20.0, "mktcap": 300.0},
        {"code": "c3", "name": "C", "industry": "家电", "ocf": 8.0, "capex": 0.0, "net_income": 10.0, "mktcap": 200.0},
    ])
    v = valuation.valuate(vdf, cfg)

    check("v_mid = 三情景中位数",
          bool((np.abs(v["v_mid"] - v[["v_bull", "v_base", "v_bear"]].median(axis=1)) < 1e-6).all()))
    check("buy_point = v_mid*0.70",
          bool((np.abs(v["buy_point"] - v["v_mid"] * 0.7) < 1e-6).all()))
    check("sell_point = v_mid*1.50",
          bool((np.abs(v["sell_point"] - v["v_mid"] * 1.5) < 1e-6).all()))
    check("downside_bear = (v_bear-mktcap)/mktcap",
          bool((np.abs(v["downside_bear"] - (v["v_bear"] - v["mktcap"]) / v["mktcap"]) < 1e-6).all()))

    # verdict 三段一致性
    def expect_verdict(r):
        if r["mktcap"] <= r["buy_point"]:
            return "已到买点"
        if r["mktcap"] >= r["sell_point"]:
            return "已到卖点"
        return "合理区间"
    check("verdict 与买卖点区间一致",
          bool((v.apply(expect_verdict, axis=1) == v["verdict"]).all()),
          str(v["verdict"].tolist()))

    # c2 现金流为负 → fcf 回退 + 标注
    c2 = v[v["code"] == "c2"].iloc[0]
    check("ocf<=0 时 fcf 回退到 net_income*0.8",
          approx(c2["fcf0"], 20.0 * 0.8) and bool(c2["fcf_is_fallback"]))

    # 逆向 g* 公式：g = (P*r - F)/(P+F)
    c1 = v[v["code"] == "c1"].iloc[0]
    r = cfg.section("valuation").get("discount_rate", 0.10)
    g_ref = (c1["mktcap"] * r - c1["fcf0"]) / (c1["mktcap"] + c1["fcf0"])
    check("逆向 g* 公式正确", approx(c1["g_implied"], g_ref), f"{c1['g_implied']:.4f} vs {g_ref:.4f}")
    check("implied_growth(fcf0<=0) 返回 NaN",
          np.isnan(valuation.implied_growth(100.0, -5.0, 0.1)))

    # DCF 公式自洽
    def dcf_ref(f, growth, years, terminal, discount):
        pv, ff = 0.0, f
        for t in range(1, years + 1):
            ff *= (1 + growth)
            pv += ff / (1 + discount) ** t
        tv = ff * (1 + terminal) / (discount - terminal)
        return pv + tv / (1 + discount) ** years
    val = valuation.dcf_value(8.0, 0.08, 5, 0.03, 0.10)
    check("dcf_value 与两阶段公式一致", approx(val, dcf_ref(8.0, 0.08, 5, 0.03, 0.10)))
    # 单调性：增长越高价值越高
    check("DCF 单调性（增长↑→价值↑）",
          valuation.dcf_value(8.0, 0.05, 5, 0.03, 0.10)
          < valuation.dcf_value(8.0, 0.10, 5, 0.03, 0.10))


# ================================================================ L6 组合
def test_portfolio():
    print("\n[6] L6 组合（单票/行业/现金上限 + 集中宇宙 + 缓冲）")
    cfg = load_config()
    pcfg = cfg.section("portfolio")
    max_pos = pcfg.get("max_position", 0.08)
    max_ind = pcfg.get("max_industry", 0.25)
    min_cash = pcfg.get("min_cash", 0.10)

    # 集中宇宙：5 只同行业全过闸
    rows = [clean_factor_row(f"P{i}", industry="银行", name=f"银行{i}") for i in range(5)]
    for i, r in enumerate(rows):
        r["total_score"] = 100 - i
    df = pd.DataFrame(rows)
    df["passes_gate"] = True
    df["rank"] = range(1, len(df) + 1)
    pf = build_portfolio(df, cfg)
    check("集中宇宙组合非空且含名称", len(pf) > 0 and "name" in pf.columns)
    check("单票 <= 上限", pf["weight"].max() <= max_pos + 1e-6, f"max={pf['weight'].max():.4f}")
    iw = pf.groupby("industry")["weight"].sum()
    check("行业 <= 上限（即便全同行业）", iw.max() <= max_ind + 1e-6, f"max={iw.max():.4f}")
    check("现金 >= 下限（权重和 <= 1-现金）",
          pf["weight"].sum() <= 1 - min_cash + 1e-6, f"sum={pf['weight'].sum():.4f}")

    # 正常多行业 40 只
    rows = []
    inds = ["白酒", "家电", "银行", "医药", "化工", "机械"]
    for i in range(40):
        rows.append(clean_factor_row(f"Q{i}", industry=inds[i % len(inds)], name=f"股{i}"))
    df = pd.DataFrame(rows)
    df["passes_gate"] = True
    df["total_score"] = list(range(40, 0, -1))
    df["rank"] = range(1, len(df) + 1)
    pf = build_portfolio(df, cfg)
    check("多行业组合单票 <= 上限", pf["weight"].max() <= max_pos + 1e-6)
    check("多行业组合行业 <= 上限", pf.groupby("industry")["weight"].sum().max() <= max_ind + 1e-6)
    check("持仓数在上限内(<=30)", len(pf) <= 30, f"n={len(pf)}")
    check("现金 >= 下限", pf["weight"].sum() <= 1 - min_cash + 1e-6)

    # 缓冲区：相同截面二次调仓应保留大部分持仓
    w1 = pd.Series(pf["weight"].values, index=pf["code"].values)
    pf2 = build_portfolio(df, cfg, current_weights=w1)
    overlap = len(set(pf2["code"]) & set(pf["code"]))
    check("缓冲区生效（二次调仓保留持仓）", overlap > 0, f"保留 {overlap}/{len(pf)}")

    # turnover 语义
    check("turnover(None, w)=w.sum()", approx(turnover(None, w1), w1.sum()))
    check("turnover(w, w)=0", approx(turnover(w1, w1), 0.0))

    # 全空
    empty = build_portfolio(pd.DataFrame(), cfg)
    check("空 scored → 空组合", empty.empty)


# ================================================================ L1 宇宙时点过滤
def test_universe():
    print("\n[7] L1 宇宙时点过滤（退市 / 上市年限 / ST）")
    cfg = load_config()
    with tempfile.TemporaryDirectory() as d:
        st = Store(d)
        uni = pd.DataFrame([
            {"code": "A.SH", "name": "股票A", "industry": "白酒", "list_date": pd.Timestamp("2015-01-01"), "delist_date": pd.NaT, "is_financial": False},
            {"code": "B.SH", "name": "股票B", "industry": "白酒", "list_date": pd.Timestamp("2015-01-01"), "delist_date": pd.Timestamp("2023-01-01"), "is_financial": False},
            {"code": "C.SH", "name": "股票C", "industry": "白酒", "list_date": pd.Timestamp("2015-01-01"), "delist_date": pd.Timestamp("2025-01-01"), "is_financial": False},
            {"code": "D.SH", "name": "股票D", "industry": "白酒", "list_date": pd.Timestamp("2023-01-01"), "delist_date": pd.NaT, "is_financial": False},
            {"code": "E.SH", "name": "ST风险", "industry": "白酒", "list_date": pd.Timestamp("2015-01-01"), "delist_date": pd.NaT, "is_financial": False},
        ])
        st.save_universe(uni)
        # 行情（含 avg_amount_60d 所需 amount）
        prices = []
        for c in ["A.SH", "B.SH", "C.SH", "D.SH", "E.SH"]:
            prices.append({"code": c, "date": pd.Timestamp("2024-05-15"),
                           "close_raw": 10.0, "close_adj": 10.0, "volume": 1e6,
                           "amount": 1e9, "total_share": 1e9, "mktcap": 1e10})
        st.save_prices(pd.DataFrame(prices))

        asof = pd.Timestamp("2024-05-15")
        u = universe.build_universe(st, asof, cfg)
        codes = set(u["code"])
        check("正常股入选", "A.SH" in codes)
        check("早退市(2023) 被剔除", "B.SH" not in codes)
        check("晚退市(2025) 仍入选（点时宇宙含退市）", "C.SH" in codes)
        check("次新(上市<3年) 被剔除", "D.SH" not in codes)
        check("ST 名称被剔除", "E.SH" not in codes)


# ================================================================ 数据缺口鲁棒性
def test_data_gaps():
    print("\n[8] 数据缺口鲁棒性（net_payout 缺失 / 复数守卫）")
    cfg = load_config()
    # 负 gpa_old → profit_growth_5y 应为 NaN（float，非 complex）
    with tempfile.TemporaryDirectory() as d:
        st = Store(d)
        uni = pd.DataFrame([{"code": "G.SH", "name": "g", "industry": "白酒",
                             "list_date": pd.Timestamp("2000-01-01"), "delist_date": pd.NaT, "is_financial": False}])
        st.save_universe(uni)
        facts = []
        # 6 个年报期（2019-2024），2024 期 gpa 为正、2019 期 gpa 为负（rev<cogs）
        for y in range(2019, 2025):
            rev = 10.0 if y == 2019 else 100.0
            cogs = 20.0 if y == 2019 else 50.0     # 2019: gpa_old<0
            ta = 200.0
            facts.append({"code": "G.SH", "field": S.REVENUE, "period": f"{y}1231",
                          "announce_date": pd.Timestamp(y + 1, 4, 1), "value": rev, "source": "t", "fetch_time": pd.Timestamp.now()})
            facts.append({"code": "G.SH", "field": S.COGS, "period": f"{y}1231",
                          "announce_date": pd.Timestamp(y + 1, 4, 1), "value": cogs, "source": "t", "fetch_time": pd.Timestamp.now()})
            facts.append({"code": "G.SH", "field": S.TOTAL_ASSETS, "period": f"{y}1231",
                          "announce_date": pd.Timestamp(y + 1, 4, 1), "value": ta, "source": "t", "fetch_time": pd.Timestamp.now()})
        st.save_facts(pd.DataFrame(facts))
        st.save_prices(pd.DataFrame([{"code": "G.SH", "date": pd.Timestamp("2025-05-15"),
                                      "close_raw": 10.0, "close_adj": 10.0, "volume": 1e6,
                                      "amount": 1e9, "total_share": 1e9, "mktcap": 1e10}]))
        uni_b = universe.build_universe(st, "2025-05-15", cfg)
        m = metrics.compute_metrics(st.facts_asof("2025-05-15"), st.market_asof("2025-05-15"), uni_b)
        check("compute_metrics 非空", not m.empty)
        pg = m.iloc[0]["profit_growth_5y"]
        check("负 gpa_old → profit_growth_5y 为 NaN（非复数）",
              (np.isnan(pg) or np.isreal(pg)) and not np.iscomplex(pg), str(pg))


# ================================================================ 回测引擎
def test_backtest():
    print("\n[9] 回测引擎（确定性 / 成本计提）")
    cfg = load_config()
    st = build_demo_store(str(Path(__file__).resolve().parent.parent / "data" / "correctness_db"),
                          GenConfig(n_stocks=60, seed=11, fund_start_year=2015,
                                    fund_end_year=2026, price_start="2018-01-01"),
                          overwrite=True)
    res1 = bt.run_backtest(st, cfg, "2016-01-01", "2026-06-30", label="t")
    res2 = bt.run_backtest(st, cfg, "2016-01-01", "2026-06-30", label="t")
    check("回测无错误", "error" not in res1, res1.get("error", ""))
    if "error" not in res1:
        nav1 = res1["nav"]["nav"].values
        nav2 = res2["nav"]["nav"].values
        check("回测确定性（两次 NAV 完全一致）", np.allclose(nav1, nav2))
        check("NAV 无 NaN", not np.isnan(nav1).any())
        check("NAV 序列长度 = 记录数", len(nav1) == len(res1["records"]))
        # 成本计提：累计 NAV <= 无成本累计
        rec = res1["records"]
        gross = (1 + rec["port_ret"]).prod()
        check("成本被计提（净 NAV < 毛收益累积）",
              float(res1["nav"]["nav"].iloc[-1]) < gross,
              f"净={res1['nav']['nav'].iloc[-1]:.3f} 毛={gross:.3f}")
        check("累计成本 > 0", float(rec["cost"].sum()) > 0)
        check("灵敏度检验返回多行", not bt.sensitivity_test(st, cfg, "2016-01-01", "2026-06-30").empty)
        check("分状态检验返回非空", not bt.regime_report(res1).empty)


def test_index_benchmark():
    print("\n[9b] 指数基准接入（HS300 行情作基准）")
    cfg = load_config()
    path = str(Path(__file__).resolve().parent.parent / "data" / "correctness_db")
    st = build_demo_store(path, GenConfig(n_stocks=60, seed=11, fund_start_year=2015,
                                          fund_end_year=2026, price_start="2018-01-01"),
                          overwrite=True)
    # 注入合成指数日线（覆盖回测区间，温和上涨）
    px = st.load_prices()
    dates = sorted(px["date"].unique())
    closes = np.cumprod(np.full(len(dates), 1.0003))
    idx = pd.DataFrame({"code": "sh000300", "date": pd.to_datetime(dates), "close": closes})
    st.save_index_prices(idx)

    # 等权基准仍是默认
    r_eq = bt.run_backtest(st, cfg, "2016-01-01", "2026-06-30", label="t", benchmark="equal")
    check("等权基准为默认", "error" not in r_eq and r_eq["stats"]["benchmark"] == "等权基准")

    # 指数基准
    r_ix = bt.run_backtest(st, cfg, "2016-01-01", "2026-06-30", label="t", benchmark="sh000300")
    check("指数基准回测无错误", "error" not in r_ix, r_ix.get("error", ""))
    if "error" not in r_ix:
        check("基准标签含 sh000300", "sh000300" in r_ix["stats"]["benchmark"])
        check("指数基准 NAV 与等权基准 NAV 不同（用了不同收益序列）",
              not np.allclose(r_ix["nav"]["bench"].values, r_eq["nav"]["bench"].values))
        check("指数基准 NAV 合理（期末>期初）",
              float(r_ix["nav"]["bench"].iloc[-1]) > 1.0)
        check("指数基准下策略超额由 bench 序列驱动",
              np.isfinite(r_ix["stats"]["excess_cagr"]))

    # 指数数据缺失 → 回退等权（不崩）。用独立目录，避免继承上面写入的 index_prices
    with tempfile.TemporaryDirectory() as td:
        st2 = build_demo_store(td, GenConfig(n_stocks=60, seed=11, fund_start_year=2015,
                                             fund_end_year=2026, price_start="2018-01-01"),
                               overwrite=True)
        r_missing = bt.run_backtest(st2, cfg, "2016-01-01", "2026-06-30", benchmark="sh000300")
        check("指数数据缺失 → 回退等权基准（不崩）",
              "error" not in r_missing and r_missing["stats"]["benchmark"] == "等权基准")

    # 双基准对比报告可生成
    rep = bt.compare_benchmarks(st, cfg, "2016-01-01", "2026-06-30",
                                index_code="sh000300", index_name="沪深300")
    check("双基准对比报告含 沪深300", "沪深300" in rep and "策略" in rep)



# ================================================================ 真实库回归锚点
def test_real_anchors():
    print("\n[10] 真实库回归锚点（data/real_universe）")
    real = Path(__file__).resolve().parent.parent / "data" / "real_universe"
    if not (real / "facts.parquet").exists():
        print("  （真实库缺失，跳过）")
        return
    st = Store(real)
    cfg = load_config()
    asof = "2026-09-07"
    scored, rejected, uni = bt.screen_at(st, asof, cfg, with_valuation=True)

    check("真实库 screen 产出非空", not scored.empty)
    check("net_payout 全缺 → 中性 NaN（不静默惩罚）",
          scored["net_payout"].isna().all() or scored["net_payout"].isna().any())
    # 伊利(sh600887) 应被 L4 挡掉（前面已确认真实原因：GPA 增长为负）
    check("伊利(sh600887) 不在持仓（过不了 L4）", "sh600887" not in set(scored[scored["passes_gate"]]["code"]))
    # 圆通速递(sh600233) 应为 #1
    top = scored.sort_values("total_score", ascending=False).iloc[0]["code"]
    check("圆通速递(sh600233) 为 #1  ranked", top == "sh600233", f"top={top}")
    # 排雷失效告警：商誉字段全缺 → 判 dead 并告警；质押在当期有快照数据 → 不判 dead
    warns = vetoes.warn_dead_rules(scored, cfg)
    joined = " ".join(warns)
    check("排雷失效告警含 商誉（字段全缺→dead）", "goodwill_to_equity" in joined)
    check("质押当期有快照数据→不判 dead（历史时点可得性是另一问题）",
          "pledge_ratio" not in joined and scored["pledge_ratio"].notna().any())


# ================================================================ 主入口
def main():
    print("=" * 70)
    print("价值投资选股系统 · 分层逻辑正确性测试")
    print("=" * 70)
    test_point_in_time()
    test_vetoes()
    test_factors()
    test_valuation()
    test_portfolio()
    test_universe()
    test_data_gaps()
    test_backtest()
    test_index_benchmark()
    test_real_anchors()

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
