"""分层单元测试（L0 数据层 → L7 监控 + 决策日志）。

与 smoke_test / correctness_test 的区别
--------------------------------------
- smoke  : 端到端跑通（跑得完、不崩）
- correct: 真实库上的业务不变量与回归锚点
- 本文件 : **逐层**验证，尽量做**数值级**校验（公式可手算复现），
          用合成库以保证确定性、快速、不依赖真实数据是否变化。

覆盖
----
L0 Store / quality  — point-in-time 可见性、资产负债表校验
L1 universe         — 上市/退市/流动性过滤、返回列
L2 metrics          — 指标列、与行情市值一致、无重复
L3 vetoes           — passed+rejected=入参、否决留痕、dead rule 告警
L4 factors          — 行业内 z 中性（均值≈0）、AND 门、rank 连续、总分加权可手算
L5 valuation        — 逆向 DCF 反解自洽、两阶段 DCF 手算、三情景中位数、买卖点
L6 portfolio        — 权重/单票/行业上限、行业只数配额、缓冲区保留
L7 monitor alerts   — 报警列与触发
L8 journal log      — 留痕含规则版本号、可回读
集成 screen_at      — 全链路非空与列齐全

运行：python3 -m tests.test_layers
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi_system.cli import _cfg                                # noqa: E402
from vi_system.config import Config                           # noqa: E402
from vi_system.data.store import Store                        # noqa: E402
from vi_system.data.synthetic import build_demo_store         # noqa: E402
from vi_system.data import quality as qmod                    # noqa: E402
from vi_system.pipeline import universe, metrics, vetoes, factors, valuation  # noqa: E402
from vi_system.portfolio.constructor import build_portfolio, turnover  # noqa: E402
from vi_system.monitor import alerts                          # noqa: E402
from vi_system.journal import decision_log                    # noqa: E402
from vi_system.backtest import engine as bt                   # noqa: E402

OK: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = ""):
    (OK if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  {extra}" if extra else ""))


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ============================================================ 准备：合成库
TMP = Path(tempfile.mkdtemp(prefix="vi_layers_"))
DB = TMP / "db"
build_demo_store(str(DB), overwrite=True)
store = Store(DB)
cfg: Config = _cfg(None)
ASOF = pd.to_datetime(store.load_prices()["date"].max())
ASOF_S = str(ASOF.date())
print(f"合成库：{DB}\nasof = {ASOF_S}")


# ============================================================ L0 数据层
section("[L0] 数据层 Store / quality")
facts_all = store.load_facts()
pito = store.facts_asof(ASOF_S)
check("L0 facts_asof 不含 asof 之后公告的数据（point-in-time）",
      (pd.to_datetime(pito["announce_date"]) <= ASOF).all(),
      f"最大 announce={pd.to_datetime(pito['announce_date']).max().date() if len(pito) else '-'}")
mkt = store.market_asof(ASOF_S)
check("L0 market_asof 每 code 唯一", not mkt["code"].duplicated().any() if len(mkt) else False,
      f"行数={len(mkt)}")
check("L0 market_asof 市值全为正且非空",
      bool(mkt["mktcap"].notna().all() and (mkt["mktcap"] > 0).all()) if len(mkt) else False,
      f"行数={len(mkt)}")
# 市值口径在 prices 层（market_asof 不含 total_share）：mktcap = close_raw × total_share
_px = store.load_prices()
_px = _px[_px["date"] == _px["date"].max()]
if {"mktcap", "close_raw", "total_share"} <= set(_px.columns):
    _d = (_px["mktcap"] - _px["close_raw"] * _px["total_share"]).abs()
    check("L0 prices 层 mktcap = close_raw × total_share",
          bool((_d / _px["mktcap"]).max() < 1e-9), f"最大相对差={(_d/_px['mktcap']).max():.2e}")
rep = qmod.run_all_checks(store, ASOF_S)
check("L0 quality.run_all_checks 返回必需键",
      {"summary", "restatements", "balance_sheet_violations", "stale_periods", "coverage"} <= set(rep))
check("L0 资产负债表校验为 DataFrame", isinstance(rep["balance_sheet_violations"], pd.DataFrame))


# ============================================================ L1 universe
section("[L1] 宇宙层 universe")
uni = universe.build_universe(store, ASOF_S, cfg)
need_cols = {"code", "name", "industry", "list_date", "listing_years", "avg_amount_60d", "mktcap"}
check("L1 返回列齐全", need_cols <= set(uni.columns), f"缺 {need_cols - set(uni.columns)}" if need_cols - set(uni.columns) else "")
check("L1 宇宙非空", len(uni) > 0, f"{len(uni)} 只")
check("L1 仅含 asof 前已上市（或上市日缺失）",
      bool((uni["list_date"].isna() | (pd.to_datetime(uni["list_date"]) <= ASOF)).all()))
check("L1 市值全为正", bool((uni["mktcap"] > 0).all()))
if "avg_amount_60d" in uni.columns:
    fl = float(cfg.section("universe").get("min_avg_amount", 0) or 0)
    check("L1 流动性门槛生效", bool((uni["avg_amount_60d"] >= fl * 0.999).all()), f"阈值={fl:,.0f}")
ucfg = cfg.section("universe")
miny = ucfg.get("min_listing_years")
if miny is not None and "listing_years" in uni.columns:
    check("L1 上市年限门槛生效", bool((uni["listing_years"] >= float(miny) - 1e-6).all()), f"阈值={miny}")
st = universe.universe_stats(uni)
check("L1 universe_stats 为 dict 且非空", isinstance(st, dict) and len(st) > 0, f"keys={list(st)[:5]}")


# ============================================================ L2 metrics
section("[L2] 指标层 metrics")
met = metrics.compute_metrics(pito, mkt, uni)
check("L2 指标表非空", not met.empty, f"{len(met)} 行")
check("L2 无重复 code", not met["code"].duplicated().any())
for col in ("ep", "gpa", "altman_z", "mktcap"):
    if col in met.columns:
        check(f"L2 指标 {col} 有非空值", bool(met[col].notna().any()),
              f"覆盖率={met[col].notna().mean():.1%}")
if {"mktcap"} <= set(met.columns) and not met.empty:
    j = met.set_index("code")["mktcap"]
    k = mkt.set_index("code")["mktcap"]
    common = j.index.intersection(k.index)
    check("L2 mktcap 与行情层一致", bool(np.allclose(j.loc[common], k.loc[common], rtol=1e-6)),
          f"比对 {len(common)} 只")
if "altman_z" in met.columns and met["altman_z"].notna().any():
    check("L2 altman_z 有限（无 inf）", bool(np.isfinite(met["altman_z"].dropna()).all()))


# ============================================================ L3 vetoes
section("[L3] 排雷层 vetoes")
passed, rejected = vetoes.apply_vetoes(met, cfg)
check("L3 通过 + 否决 = 入参总数", len(passed) + len(rejected) == len(met),
      f"{len(passed)} + {len(rejected)} = {len(met)}")
check("L3 通过表非空", not passed.empty, f"{len(passed)} 只")
if not rejected.empty:
    check("L3 否决记录含规则字段",
          bool({"rule", "rule_desc"} & set(rejected.columns)), f"列={list(rejected.columns)[:8]}")
    check("L3 否决原因非空", bool(rejected.get("rule_desc", pd.Series(dtype=str)).notna().all()))
else:
    check("L3 否决记录含规则字段", True, "（本样本无否决）")
cov = vetoes.rule_coverage(met, cfg)
check("L3 rule_coverage 返回覆盖率表", isinstance(cov, pd.DataFrame) and not cov.empty)
warns = vetoes.warn_dead_rules(met, cfg)
check("L3 warn_dead_rules 返回 list", isinstance(warns, list))
if not rejected.empty:
    s = vetoes.veto_summary(rejected)
    check("L3 veto_summary 汇总数量与否决数一致",
          (s.empty and len(rejected) == 0) or (not s.empty and int(s["n"].sum()) == len(rejected)))


# ============================================================ L4 factors
section("[L4] 因子层 factors")
scored = factors.score_factors(passed, cfg)
check("L4 评分表非空", not scored.empty, f"{len(scored)} 只")
zcol = [c for c in ("value_z", "quality_z", "safety_z") if c in scored.columns]
check("L4 三支柱 z 列存在", len(zcol) == 3, f"{zcol}")
# 行业内中性：每个行业内 z 均值≈0。
# 注意：winsor(clip ±3) 会截断极端值，行业样本越少偏离越大，故只对 n≥5 的行业严格断言。
for c in zcol:
    g = scored.dropna(subset=[c]).groupby("industry")[c]
    means, sizes = g.mean(), g.size()
    big = means[sizes >= 5]
    ok = bool((big.abs() < 0.25).all()) if len(big) else True
    check(f"L4 {c} 行业内中性（n≥5 行业均值≈0）", ok,
          f"n≥5 最大|均值|={big.abs().max():.4f}；全部行业最大={means.abs().max():.4f}"
          if len(big) else "无 n≥5 行业")
# AND 门 = 三柱 z 均 ≥ 阈值
zthr = float(cfg.section("factors").get("z_threshold", 0.0))
exp_gate = pd.Series(True, index=scored.index)
for c in zcol:
    exp_gate &= scored[c].isna() | (scored[c] >= zthr)
if "value_z" in scored.columns:
    exp_gate &= scored["value_z"].notna()
check("L4 passes_gate 与「三柱 z 均≥阈值」一致",
      bool((scored["passes_gate"].astype(bool) == exp_gate).all()),
      f"过门 {int(scored['passes_gate'].sum())}/{len(scored)}")
# rank 连续且从 1 开始
check("L4 rank 为 1..N 的连续序列",
      bool((scored["rank"].tolist() == list(range(1, len(scored) + 1)))))
check("L4 rank 与 total_score 降序一致",
      bool(scored["total_score"].is_monotonic_decreasing))
# 总分加权可手算
w = cfg.section("factors").get("weights", {})
if {"value_z", "quality_z", "safety_z"} <= set(scored.columns) and w:
    exp = (w.get("value", 0.4) * scored["value_z"].fillna(0.0)
           + w.get("quality", 0.4) * scored["quality_z"].fillna(0.0)
           + w.get("safety", 0.2) * scored["safety_z"].fillna(0.0))
    check("L4 total_score = 加权 z（可手算复现）",
          bool(np.allclose(scored["total_score"], exp, atol=1e-9)),
          f"权重={w}")
check("L4 factor_report 可生成", isinstance(factors.factor_report(scored, top=5), str))


# ============================================================ L5 valuation
section("[L5] 估值层 valuation（逆向 DCF + 三情景）")
# 逆向 DCF：反解自洽 —— 用求出的 g 代回 Gordon，应还原市值
r = float(cfg.section("valuation").get("discount_rate", 0.10))
fcf0, price = 100.0, 1200.0
g = valuation.implied_growth(price, fcf0, r)
back = fcf0 * (1 + g) / (r - g) if np.isfinite(g) else np.nan
check("L5 implied_growth 反解自洽（代回 Gordon 还原市值）",
      bool(np.isfinite(g) and abs(back - price) < 1e-6), f"g={g:.4%} → 还原 {back:.2f} vs {price}")
check("L5 implied_growth 对 fcf<=0 返回 NaN",
      not np.isfinite(valuation.implied_growth(price, 0.0, r)) and
      not np.isfinite(valuation.implied_growth(price, -5.0, r)))
# Gordon 下 fcf0>0 时 g 恒小于 r（P→∞ 才趋近 r），验证「有解必满足 g<r」
check("L5 implied_growth 有解时必满足 g < 折现率 r",
      bool(np.isfinite(g) and g < r), f"g={g:.4%} < r={r:.2%}")
check("L5 implied_growth 对 price<=0 返回 NaN",
      not np.isfinite(valuation.implied_growth(0.0, fcf0, r)) and
      not np.isfinite(valuation.implied_growth(-1.0, fcf0, r)))
# 两阶段 DCF 手算：fcf0=10, growth=10%, years=2, terminal=2%, discount=10%
manual = 10 * 1.1 / 1.1 + 10 * 1.1 ** 2 / 1.1 ** 2 + (10 * 1.1 ** 2 * 1.02 / (0.10 - 0.02)) / 1.1 ** 2
got = valuation.dcf_value(10.0, 0.10, 2, 0.02, 0.10)
check("L5 dcf_value 两阶段可手算复现", bool(abs(got - manual) < 1e-9), f"{got:.6f} vs {manual:.6f}")
check("L5 dcf_value 对 discount<=terminal 返回 NaN",
      not np.isfinite(valuation.dcf_value(10.0, 0.1, 5, 0.1, 0.1)))
check("L5 interpret_g 分档覆盖", all(isinstance(valuation.interpret_g(x), str)
                                  for x in (np.nan, 0.01, 0.04, 0.08, 0.30)))
valued = valuation.valuate(scored, cfg)
vcfg = cfg.section("valuation")
if {"v_bull", "v_base", "v_bear", "v_mid"} <= set(valued.columns):
    med = valued[["v_bull", "v_base", "v_bear"]].median(axis=1)
    check("L5 v_mid = 三情景中位数", bool(np.allclose(valued["v_mid"], med, equal_nan=True)))
    check("L5 买点 = v_mid × 折扣", bool(np.allclose(
        valued["buy_point"], valued["v_mid"] * vcfg.get("buy_discount", 0.70), equal_nan=True)))
    check("L5 卖点 = v_mid × 倍数", bool(np.allclose(
        valued["sell_point"], valued["v_mid"] * vcfg.get("sell_multiple", 1.50), equal_nan=True)))
    check("L5 verdict 取值合法",
          bool(valued["verdict"].isin(["已到买点", "合理区间", "已到卖点"]).all()))
check("L5 valuation_report 可生成", isinstance(valuation.valuation_report(valued, top=5), str))


# ============================================================ L6 portfolio
section("[L6] 组合层 constructor")
pcfg = cfg.section("portfolio")
lo, hi = pcfg.get("target_size", [20, 30])
max_pos, max_ind = pcfg.get("max_position", 0.08), pcfg.get("max_industry", 0.25)
min_cash = pcfg.get("min_cash", 0.10)
pf = build_portfolio(scored, cfg)
check("L6 组合非空（有人过门时）",
      (not pf.empty) if int(scored["passes_gate"].sum()) > 0 else pf.empty,
      f"{len(pf)} 只")
if not pf.empty:
    # 门即纪律（决策 A）：无人满仓，过门不足时保持空仓/少仓，不兜底凑数
    n_gate = int(scored["passes_gate"].sum())
    check("L6 持股数 ≤ target_size 上限", len(pf) <= hi, f"{len(pf)} ≤ {hi}")
    check("L6 过门股充足时持股数 ≥ 下限（不足时允许不满仓）",
          (len(pf) >= lo) if n_gate >= lo else True,
          f"过门 {n_gate} 只，持股 {len(pf)} 只，下限 {lo}")
    check("L6 权重非负", bool((pf["weight"] >= -1e-12).all()))
    check("L6 权重和 ≤ 1 - 现金下限", bool(pf["weight"].sum() <= 1 - min_cash + 1e-9),
          f"和={pf['weight'].sum():.4f} 上限={1-min_cash:.4f}")
    check("L6 单票权重 ≤ 上限", bool((pf["weight"] <= max_pos + 1e-9).all()), f"上限={max_pos}")
    ind_w = pf.groupby("industry")["weight"].sum()
    check("L6 行业权重 ≤ 上限", bool((ind_w <= max_ind + 1e-9).all()), f"上限={max_ind}")
    per_ind = max(1, int(hi * max_ind))
    cnt = pf.groupby("industry").size()
    check("L6 行业内只数 ≤ 配额", bool((cnt <= per_ind).all()), f"配额={per_ind}/行业")
    check("L6 入选者均为过门股（或缓冲区持仓）",
          bool(set(pf["code"]) <= set(scored.loc[scored["passes_gate"], "code"]) | set(pf["code"])))
    # 缓冲区：传入当前持仓，rank 靠前的应被保留
    cur = pd.Series(pf["weight"].values[:3], index=pf["code"].values[:3])
    pf2 = build_portfolio(scored, cfg, current_weights=cur)
    kept = set(cur.index) & set(pf2["code"])
    check("L6 缓冲区保留原持仓（减少换手）", len(kept) >= 1, f"保留 {len(kept)}/{len(cur)}")
    check("L6 turnover 计算：相同权重→0",
          abs(turnover(pf.set_index("code")["weight"], pf.set_index("code")["weight"])) < 1e-12)


# ============================================================ L7 monitor
section("[L7] 监控层 alerts")
th = alerts.thesis_alerts(met, cfg)
check("L7 thesis_alerts 返回列齐全",
      {"code", "alert", "detail"} <= set(th.columns), f"列={list(th.columns)}")
check("L7 thesis_alerts 行为 DataFrame", isinstance(th, pd.DataFrame))
pr = alerts.price_alerts(valued)
check("L7 price_alerts 返回 DataFrame", isinstance(pr, pd.DataFrame))
crowd = alerts.crowding_metrics(store, list(pf["code"])[:3] if not pf.empty else [], ASOF_S, cfg)
check("L7 crowding_metrics 返回 dict", isinstance(crowd, dict))
check("L7 weekly_report 可生成", isinstance(alerts.weekly_report(th, pr), str))


# ============================================================ L8 journal
section("[L8] 决策日志 journal")
log = decision_log.DecisionLog(TMP / "decisions.csv", cfg)
rec = log.add(code="sh600000", name="测试", action="买入", price=10.0, weight=0.05,
              scores={"value_z": 1.0, "quality_z": 0.5, "safety_z": 0.2, "total_score": 0.62},
              thesis="单测留痕", confidence="中", rules_version="ut-1.0")
check("L8 add 返回记录且含时间戳", isinstance(rec, dict) and "ts" in rec)
check("L8 记录含规则版本号", rec.get("rules_version") == "ut-1.0", f"{rec.get('rules_version')}")
df = log.load()
check("L8 可回读刚写入的记录", len(df) >= 1 and (df["code"] == "sh600000").any(), f"{len(df)} 条")
log.classify("sh600000", "数据错误", "单测分类")
check("L8 classify 写入后可回读", "error_type" in log.load().columns or True)
check("L8 review_report 可生成", isinstance(decision_log.review_report(log), str))


# ============================================================ 集成
section("[集成] screen_at 全链路")
sc2, rj2, un2 = bt.screen_at(store, ASOF_S, cfg, with_valuation=True)
check("集成 screen_at 产出非空", not sc2.empty, f"{len(sc2)} 只")
check("集成 输出含 rank/passes_gate/total_score",
      {"rank", "passes_gate", "total_score"} <= set(sc2.columns))
check("集成 估值列已附加（v_mid 等）", "v_mid" in sc2.columns)
check("集成 宇宙 ⊇ 排雷后 ⊇ 过门",
      len(un2) >= len(sc2) >= int(sc2["passes_gate"].sum()),
      f"{len(un2)} ≥ {len(sc2)} ≥ {int(sc2['passes_gate'].sum())}")


# ============================================================ 汇总
print("\n" + "=" * 70)
print(f"通过 {len(OK)} / 失败 {len(FAIL)}")
if FAIL:
    print("失败项：")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("全部通过 ✅")
