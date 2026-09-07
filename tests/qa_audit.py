"""QA 审计：以测试人员角度检查系统是否有问题。

重点不是"能不能跑通"（冒烟测试已覆盖），而是：
  T1 前视偏差 —— 调仓日是否只用「已公告」数据（价值投资系统的命门）
  T2 行情前视 —— 是否用了未来价格
  T3 调仓日   —— 是否落在配置的月份/日期（年报/半年报披露后）
  T4 排雷规则覆盖 —— 哪些规则因数据缺口恒跳过（形同虚设却不报错）
  T5 边界条件 —— 空宇宙 / 早期时点 / 极小子宇宙 是否崩溃
  T6 组合约束 —— 权重上限、权重和、行业配额
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
sys.path.insert(0, str(ROOT))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.pipeline.universe import build_universe
from vi_system.pipeline.metrics import compute_metrics
from vi_system.pipeline.vetoes import apply_vetoes
from vi_system.pipeline import valuation as valuation_mod
from vi_system.portfolio import constructor as portfolio_mod

ASOFS = ["2019-05-15", "2021-05-15", "2023-05-15", "2025-05-15"]
# 排雷真正依赖的指标列（对应 vetoes._RULE_SPEC + 复合规则）
VETO_COLS = [
    "beneish_m", "audit_nonstd_3y", "pledge_ratio", "goodwill_to_equity",
    "ocf_to_ni_5y", "share_dilution_5y", "net_debt_ebitda",
    "interest_coverage", "altman_z",
]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'✅' if ok else '❌'} {name}" + (f"  {detail}" if detail else ""), flush=True)


def main():
    st = Store(str(ROOT / "data" / "real_universe"))
    cfg = load_config()

    # ---------------------------------------------------------------- T1
    print("\n[T1] 前视偏差：facts_asof 是否只用已公告数据", flush=True)
    leak_total = 0
    for asof in ASOFS:
        try:
            f = st.facts_asof(asof, periods=12)
            if f.empty:
                continue
            ann = pd.to_datetime(f["announce_date"])
            leak = int((ann > pd.Timestamp(asof)).sum())
            leak_total += leak
            if leak:
                worst = ann.max().date()
                check(f"  {asof} 无未来数据", False,
                      f"越界 {leak} 行，最大公告日 {worst} > {asof}")
            else:
                check(f"  {asof} 无未来数据", True, f"最大公告日 {ann.max().date()}")
        except Exception as e:
            check(f"  {asof} 无未来数据", False, f"异常 {type(e).__name__}: {e}")
    check("T1 前视偏差（财务）", leak_total == 0, f"累计越界 {leak_total} 行")

    # ---------------------------------------------------------------- T2
    print("\n[T2] 行情前视：market_asof 是否用了未来价格", flush=True)
    t2_ok = True
    for asof in ASOFS:
        try:
            m = st.market_asof(asof)
            if m.empty:
                continue
            col = "asof_date" if "asof_date" in m.columns else None
            if col:
                d = pd.to_datetime(m[col])
                bad = int((d > pd.Timestamp(asof)).sum())
                if bad:
                    t2_ok = False
                    check(f"  {asof} 行情日期", False, f"{bad} 行晚于调仓日")
                else:
                    check(f"  {asof} 行情日期", True, f"asof_date 最大 {d.max().date()}")
            else:
                check(f"  {asof} 行情日期", True, "无 asof_date 列，跳过")
        except Exception as e:
            t2_ok = False
            check(f"  {asof} 行情日期", False, f"异常 {type(e).__name__}: {e}")
    check("T2 行情前视", t2_ok)

    # ---------------------------------------------------------------- T3
    print("\n[T3] 调仓日是否落在配置月份/日期", flush=True)
    try:
        b = cfg.section("backtest")
        months = b.get("rebalance_months", [5, 9])
        day = b.get("rebalance_day", 15)
        rd = st.rebalance_dates(months, day, "2018-01-01", "2026-09-07")
        rd = list(pd.to_datetime(pd.Series(rd)))
        bad_m = [d for d in rd if d.month not in months]
        # 设计是「指定日若非交易日则前移到之前最近交易日」，故日期应 <= day，不得晚于 day
        bad_d = [d for d in rd if d.day > day]
        check("  调仓日月份", not bad_m,
              f"{len(rd)} 个调仓日，月份异常 {len(bad_m)} 个")
        check("  调仓日日期", not bad_d, f"日期异常 {len(bad_d)} 个（应为每月 {day} 日）")
        check("T3 调仓日", not bad_m and not bad_d)
    except Exception as e:
        check("T3 调仓日", False, f"异常 {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- T4
    print("\n[T4] 排雷规则覆盖率：哪些规则因数据缺口恒跳过", flush=True)
    try:
        asof = "2025-05-15"
        u = build_universe(st, asof, cfg)
        mm = compute_metrics(st.facts_asof(asof, periods=12), st.market_asof(asof), u)
        check("  指标表非空", not mm.empty, f"{len(mm)} 只")
        dead = []
        for c in VETO_COLS:
            if c not in mm.columns:
                dead.append((c, 0.0))
                print(f"    ⚠️ {c}: 列不存在", flush=True)
                continue
            rate = float(mm[c].notna().mean())
            flag = "🔴 恒缺失" if rate == 0 else ("🟡 严重不足" if rate < 0.3 else "")
            print(f"    {c}: 有效率 {rate:.1%} {flag}", flush=True)
            if rate == 0:
                dead.append((c, rate))
        # 区分失效原因：是接口根本没给字段，还是给了但历史时点取不到
        f_all = st.load_facts()
        fld_map = {"goodwill_to_equity": "goodwill", "pledge_ratio": "pledge_ratio"}
        for c, _ in dead:
            fld = fld_map.get(c, c)
            n = int((f_all["field"] == fld).sum())
            if n == 0:
                print(f"    → {c}: 库中无字段「{fld}」，接口不提供 → 规则永久失效", flush=True)
            else:
                ann = pd.to_datetime(f_all[f_all["field"] == fld]["announce_date"])
                print(f"    → {c}: 库中有 {n} 条，但 announce_date 集中在 "
                      f"{ann.min().date()}~{ann.max().date()}（当前快照，历史时点取不到）", flush=True)
        check("T4 排雷规则无「恒缺失」字段", not dead,
              f"恒缺失(形同虚设): {[d[0] for d in dead]}" if dead else "")
    except Exception as e:
        check("T4 排雷规则覆盖率", False, f"异常 {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- T5
    print("\n[T5] 边界条件", flush=True)
    from vi_system.backtest import engine as bt
    for label, asof in [("数据窗口之前", "2010-05-15"), ("极早时点", "2013-05-15")]:
        try:
            scored, rejected, uni = bt.screen_at(st, asof, cfg, with_valuation=True)
            n = 0 if scored is None or scored.empty else len(scored)
            check(f"  {label} ({asof}) 不崩溃", True, f"宇宙 {len(uni)} / 候选 {n}")
        except Exception as e:
            check(f"  {label} ({asof}) 不崩溃", False, f"{type(e).__name__}: {e}")
    # 空 DataFrame 输入
    try:
        empty = pd.DataFrame()
        r = compute_metrics(empty, empty, empty)
        check("  空输入不崩溃", True, f"返回 {type(r).__name__}")
    except Exception as e:
        check("  空输入不崩溃", False, f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------------- T6
    print("\n[T6] 组合约束：权重上限 / 权重和", flush=True)
    try:
        asof = "2025-05-15"
        scored, _, _ = bt.screen_at(st, asof, cfg, with_valuation=True)
        if scored is None or scored.empty:
            check("T6 组合约束", False, "无候选，无法检验")
        else:
            pf = portfolio_mod.build_portfolio(scored, cfg)
            pcfg = cfg.section("portfolio")
            max_w = pcfg.get("max_weight", 0.10)
            w = pf["weight"]
            over = float(w.max()) > max_w + 1e-9
            ssum = float(w.sum())
            check("  单只权重不超上限", not over, f"最大 {w.max():.2%} / 上限 {max_w:.2%}")
            check("  权重和 <= 100%", ssum <= 1.0 + 1e-9, f"合计 {ssum:.2%}")
            check("  持仓数为正", len(pf) > 0, f"{len(pf)} 只")
            check("T6 组合约束", (not over) and ssum <= 1.0 + 1e-9 and len(pf) > 0)
    except Exception as e:
        check("T6 组合约束", False, f"异常 {type(e).__name__}: {e}")
        traceback.print_exc()

    # ---------------------------------------------------------------- T7
    print("\n[T7] 宇宙构成与幸存者偏差", flush=True)
    try:
        u_all = pd.read_parquet(ROOT / "data" / "real_universe" / "universe.parquet")
        px_codes = set(st.load_prices()["code"].unique())
        delisted = u_all[u_all["delist_date"].notna()]
        with_px = len(px_codes & set(delisted["code"]))
        print(f"    宇宙 {len(u_all)} 只，其中标注退市 {len(delisted)} 只", flush=True)
        print(f"    有行情的 {len(px_codes)} 只中，退市股占 {with_px} 只", flush=True)
        print(f"    宇宙来源: {u_all['source'].value_counts().to_dict()}", flush=True)
        check("  回测宇宙含退市股", with_px > 0, f"退市股有行情 {with_px} 只（退市股无行情则无法计入）")
        # 宇宙 = 2026 年指数成份快照，回测 2018 年等于「预知后来入选」→ 幸存者偏差
        check("  宇宙非当前指数快照", False,
              "宇宙取自 2026 年沪深300/中证500 成份快照，回测早期时点等于预知入选结果（幸存者偏差）")
    except Exception as e:
        check("T7 宇宙构成", False, f"异常 {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- 汇总
    n_ok = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 66, flush=True)
    print(f"QA 汇总：通过 {n_ok} / 失败 {len(results) - n_ok}（共 {len(results)} 项）", flush=True)
    if n_ok < len(results):
        print("\n失败项：", flush=True)
        for name, ok, detail in results:
            if not ok:
                print(f"  ❌ {name}  {detail}", flush=True)


if __name__ == "__main__":
    main()
