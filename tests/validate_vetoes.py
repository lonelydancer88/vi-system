"""排雷有效性验证（阶段二真正判据）。

对历史造假/退市股，用腾讯接口的真实财务数据（带 InfoPublDate），
在爆雷前的各个历史调仓日做 point-in-time 排雷，验证 L3 能否在爆雷前拦下它们、
被哪条规则拦下。

不依赖行情（退市股无行情），只用财务数据算指标 + 绝对阈值否决，
正是排雷层的设计意图。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
from vi_system.config import load_config
from vi_system.data.westock import WeStockFetcher, _norm_code
from vi_system.data.store import Store
from vi_system.pipeline.metrics import _metrics_for_code
from vi_system.pipeline.vetoes import apply_vetoes
from vi_system.data.schema import PLEDGE_RATIO

# (代码, 名称, 行业, 爆雷年, 爆雷事件简述)  —— 代码内部统一用 _norm_code 后的 sh/sz 前缀格式
FRAUD = [
    (_norm_code("600518.SH"), "康美药业", "医药生物", 2019, "300亿货币资金造假，2019-04曝光"),
    (_norm_code("002450.SZ"), "康得新",   "化工",     2019, "119亿造假，2019-01曝光"),
    (_norm_code("300104.SZ"), "乐视网",   "传媒",     2017, "资金链断裂/财务造假，2017起"),
    (_norm_code("002069.SZ"), "獐子岛",   "农林牧渔", 2014, "扇贝“跑路”，2014起财务操纵"),
    (_norm_code("002680.SZ"), "长生生物", "医药生物", 2018, "疫苗造假，2018-07曝光"),
    (_norm_code("600086.SH"), "东方金钰", "轻工制造", 2019, "存货/资金造假，2019曝光"),
    (_norm_code("600781.SH"), "辅仁药业", "医药生物", 2019, "资金占用/造假，2019曝光"),
]

ASOFS = [f"{y}-05-15" for y in range(2013, 2021)] + [f"{y}-09-15" for y in range(2013, 2021)]


def main():
    cfg = load_config()
    f = WeStockFetcher(timeout=120)

    # 抓取 7 只造假股财务（含质押）
    all_rows = []
    pledge_by_code = {}
    for code, name, ind, blow, _ in FRAUD:
        print(f"  抓取 {code} {name} 财务...", flush=True)
        fin = f.financials([code], num=52)
        all_rows.append(fin)
        try:
            pl = f.pledge([code])
            if not pl.empty:
                pledge_by_code[code] = pl
        except Exception:
            pass

    facts = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    for code, pl in pledge_by_code.items():
        facts = pd.concat([facts, pl], ignore_index=True)

    print(f"  [debug] facts 行数={len(facts)}, 代码数={facts['code'].nunique() if len(facts) else 0}, "
          f"年份={sorted(set(str(p)[:4] for p in facts['period'])) if len(facts) else 'EMPTY'}")

    tmp = Store(str(Path(tempfile.mkdtemp()) / "fraud"))
    tmp.save_facts(facts)

    # 逐股 × 逐历史时点 排雷
    records = []
    for code, name, ind, blow, event in FRAUD:
        for asof in ASOFS:
            panel = tmp.facts_asof(asof, periods=12)
            if panel.empty or "code" not in panel.columns:
                continue
            g = panel[panel["code"] == code]
            if g.empty:
                continue
            if code == FRAUD[0][0] and asof in ("2018-05-15", "2017-05-15", "2019-05-15"):
                print(f"  [dbg] {code} {asof}: panel{panel.shape} g{g.shape} "
                      f"periods={sorted(g['period'].astype(str))}")
            # 市值用账面净资产代理（退市股无行情）；仅影响估值/Altman 的市值分量
            eq = g.iloc[-1].get("total_equity")
            mktcap = float(eq) if pd.notna(eq) and eq and eq > 0 else 1.0
            m = _metrics_for_code(g, mktcap)
            if not m:
                continue
            m["code"] = code
            m["name"] = name
            m["industry"] = ind
            df = pd.DataFrame([m])
            passed, rejected = apply_vetoes(df, cfg)
            flagged = not rejected.empty
            rule = rejected.iloc[0]["rule_desc"] if flagged else "—"
            val = rejected.iloc[0]["value"] if flagged else None
            thr = rejected.iloc[0]["threshold"] if flagged else None
            phase = "爆雷前" if int(asof[:4]) < blow else ("爆雷当年" if int(asof[:4]) == blow else "爆雷后")
            records.append({
                "code": code, "name": name, "asof": asof, "phase": phase,
                "blow_year": blow, "flagged": flagged, "rule": rule,
                "value": val, "threshold": thr,
                "beneish_m": m.get("beneish_m"), "ocf_to_ni_5y": m.get("ocf_to_ni_5y"),
                "altman_z": m.get("altman_z"), "net_debt_ebitda": m.get("net_debt_ebitda"),
                "interest_coverage": m.get("interest_coverage"),
                "pledge_ratio": m.get("pledge_ratio"), "share_dilution_5y": m.get("share_dilution_5y"),
            })

    res = pd.DataFrame(records)
    print(f"  [debug] 生成的判定记录数={len(res)}")
    res.to_csv("/Users/hpl/WorkBuddy/2026-09-07-15-54-23/out/veto_validation_raw.csv",
               index=False, encoding="utf-8-sig")
    _report(res, FRAUD)


def _report(res: pd.DataFrame, FRAUD):
    if res.empty:
        print("⚠️ 没有任何有效判定记录（可能抓取被限流返回空）。请重试。")
        return
    lines = ["# 排雷有效性验证报告（L3 能否拦下历史爆雷股）", "",
             "> 方法：用腾讯接口真实财务数据（带 InfoPublDate），在爆雷前各历史调仓日做 "
             "point-in-time 排雷。退市股无行情，故仅用财务数据算指标 + 绝对阈值否决。", ""]
    # 汇总：每只股在“爆雷前”最后一次未被拦 / 首次被拦
    for code, name, ind, blow, event in FRAUD:
        sub = res[res["code"] == code].sort_values("asof")
        pre = sub[sub["phase"] == "爆雷前"]
        first_flag = pre[pre["flagged"]].head(1)
        n_pre = len(pre)
        n_flag_pre = int(pre["flagged"].sum())
        if not first_flag.empty:
            ff = first_flag.iloc[0]
            verdict = f"✅ 在爆雷前约 {blow - int(ff['asof'][:4])} 年（{ff['asof']}）已被「{ff['rule']}」拦下"
        elif n_pre > 0:
            verdict = f"❌ 爆雷前 {n_pre} 个时点均未拦下"
        else:
            verdict = "⚠️ 爆雷前无可测时点（上市/数据不足）"
        lines += [f"## {code} {name}（爆雷年 {blow}：{event}）",
                  f"- 结论：**{verdict}**", f"- 爆雷前时点 {n_pre} 个，拦下 {n_flag_pre} 个", ""]
        # 详细表
        lines += ["| 时点 | 阶段 | 是否拦下 | 触发规则 | Beneish M | OCF/NI(5y) | Altman Z | 净负债/EBITDA | 利息覆盖 | 质押率 | 股本稀释 |",
                  "|------|------|---------|---------|-----------|------------|----------|-------------|---------|--------|---------|"]
        for _, r in sub.iterrows():
            def fmt(x, d=2):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return "—"
                return f"{x:.{d}f}"
            lines.append(f"| {r['asof']} | {r['phase']} | {'✅' if r['flagged'] else '—'} | {r['rule']} | "
                         f"{fmt(r['beneish_m'])} | {fmt(r['ocf_to_ni_5y'])} | {fmt(r['altman_z'])} | "
                         f"{fmt(r['net_debt_ebitda'])} | {fmt(r['interest_coverage'])} | {fmt(r['pledge_ratio'])} | {fmt(r['share_dilution_5y'])} |")
        lines.append("")

    # 总判定
    pre_all = res[res["phase"] == "爆雷前"]
    caught = pre_all.groupby("code")["flagged"].any()
    n_caught = int(caught.sum())
    lines += ["---", "## 总判定",
              f"- 历史爆雷股样本：{len(FRAUD)} 只",
              f"- 在爆雷前至少一个时点被排雷拦下：**{n_caught}/{len(FRAUD)}** 只",
              f"- 爆雷前所有时点平均拦下率：{pre_all['flagged'].mean():.0%}", ""]
    lines += ["说明：Beneish M-Score > -1.78 预警财务操纵；OCF/NI(5y) 过低=纸面利润；",
              "Altman Z<1.8 仅对净借款人触发；质押率/商誉/股本稀释为辅助信号。",
              "腾讯接口不提供折旧/商誉/审计意见，相关分量缺失时按中性处理（不误判）。"]
    out = "/Users/hpl/WorkBuddy/2026-09-07-15-54-23/out/veto_validation.md"
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:40]))
    print(f"\n→ 完整报告: {out}")


if __name__ == "__main__":
    main()
