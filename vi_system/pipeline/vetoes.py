"""L3 排雷层：硬性一票否决。

设计原则：
  - **否决制而非打分制**。排雷的目标是"排除确定的坏"，不是给坏的程度排序。
  - **留痕优先于效率**。被否决的公司必须留下触发规则与实际数值 —— 半年后的复盘全靠它。
  - 行业特化：银行/保险/券商/地产走各自的规则表，通用规则对它们完全失效。
"""

from __future__ import annotations

import pandas as pd

from ..config import Config


# 规则名 → (指标列名, 比较方向, 配置键, 中文说明)
# direction: "max" 表示超过阈值即否决；"min" 表示低于阈值即否决
# 注意：Altman Z-Score 已从通用硬否决移除 —— 现代低杠杆/净现金公司的 Z 会系统性偏低
# （如格力 Z≈1.78），硬排除会误杀优质股。改为"净借款人"专属否决（见下方 altman 块）。
_RULE_SPEC = {
    "beneish_m_score":    ("beneish_m",         "max", "beneish_m_score",         "Beneish M-Score 偏高（财务操纵嫌疑）"),
    "audit_opinion":      ("audit_nonstd_3y",   "max", "audit_opinion",           "近3年出现非标审计意见"),
    "pledge_ratio":       ("pledge_ratio",      "max", "pledge_ratio",            "大股东质押比例过高"),
    "goodwill_to_equity": ("goodwill_to_equity","max", "goodwill_to_equity",      "商誉占净资产过高"),
    "ocf_to_ni_5y":       ("ocf_to_ni_5y",      "min", "ocf_to_ni_5y",            "5年经营现金流/净利润过低（纸面利润）"),
    "share_dilution_5y":  ("share_dilution_5y", "max", "share_dilution_5y",       "5年股本膨胀过多（稀释股东）"),
}

# 行业特化的补充规则（指标列名, 方向, 配置键后缀, 说明）
_EXTRA_SPEC = {
    "npl_ratio_max":           ("npl_ratio",          "max", "不良率过高"),
    "provision_coverage_min":  ("provision_coverage", "min", "拨备覆盖率不足"),
    "cet1_min":                ("cet1",               "min", "核心一级资本充足率不足"),
    "cash_to_short_debt_min":  ("cash_short_debt",    "min", "现金/短债比不足（地产）"),
}


def apply_vetoes(metrics: pd.DataFrame, cfg: Config):
    """对指标表施加一票否决。

    返回 (passed_df, rejected_df)
      passed_df   : 通过排雷的指标表（原列 + 无额外列）
      rejected_df : code / name / industry / rule / rule_desc / value / threshold
    """
    if metrics.empty:
        return metrics, pd.DataFrame()

    vcfg = cfg.section("vetoes")
    if not vcfg.get("enabled", True):
        return metrics, pd.DataFrame()

    rules_cfg = vcfg.get("rules", {})
    overrides = vcfg.get("industry_overrides", {})

    reject_rows = []
    mask_pass = pd.Series(True, index=metrics.index)

    for idx, row in metrics.iterrows():
        ind = row.get("industry")
        ov = overrides.get(ind, {}) if isinstance(overrides, dict) else {}
        skip = set(ov.get("skip", []) or [])
        extra = ov.get("extra", {}) or {}

        hit = None
        # ---------------- 通用规则
        for rule, (col, direction, cfg_key, desc) in _RULE_SPEC.items():
            if rule in skip:
                continue
            rc = rules_cfg.get(rule, {})
            if not rc.get("enabled", True):
                continue
            # 规则自身声明跳过的行业（如 altman 跳过金融）
            if ind in (rc.get("skip_industries") or []):
                continue
            if cfg_key == "audit_opinion":
                threshold, value = 0.5, row.get(col, 0.0)
            elif direction == "max":
                threshold = rc.get("max")
                value = row.get(col)
            else:
                threshold = rc.get("min")
                value = row.get(col)
            if threshold is None or value is None or pd.isna(value):
                continue
            breach = value > threshold if direction == "max" else value < threshold
            if breach:
                hit = (rule, desc, value, threshold)
                break

        # ---------------- 行业特化补充规则
        if hit is None:
            for key, (col, direction, desc) in _EXTRA_SPEC.items():
                if key not in extra:
                    continue
                threshold = extra[key]
                value = row.get(col)
                if value is None or pd.isna(value):
                    continue
                breach = value > threshold if direction == "max" else value < threshold
                if breach:
                    hit = (f"industry:{key}", desc, value, threshold)
                    break

        # ---------------- 杠杆（复合条件）
        if hit is None and "leverage" not in skip:
            lc = rules_cfg.get("leverage", {})
            if lc.get("enabled", True) and ind not in (lc.get("skip_industries") or []):
                nd = row.get("net_debt_ebitda")
                ic = row.get("interest_coverage")
                if pd.notna(nd) and nd > lc.get("net_debt_ebitda_max", 5.0):
                    hit = ("leverage", "净负债/EBITDA 过高", nd, lc.get("net_debt_ebitda_max"))
                elif pd.notna(ic) and ic < lc.get("interest_coverage_min", 2.0):
                    hit = ("leverage", "利息覆盖倍数不足", ic, lc.get("interest_coverage_min"))

        # ---------------- Altman Z（仅对净借款人触发）
        # Z<1.8 是经典破产临界，但对净现金/低杠杆公司会系统性偏低（非破产信号）。
        # 因此只在公司确为净借款人（net_debt_ebitda > 0）时才硬否决；否则仅作论文软报警。
        if hit is None and "altman_z_score" not in skip:
            az = rules_cfg.get("altman_z_score", {})
            if (az.get("enabled", True)
                    and ind not in (az.get("skip_industries") or [])
                    and ind not in (lc.get("skip_industries") or [])
                    and pd.notna(row.get("altman_z"))
                    and row.get("altman_z") < az.get("min", 1.8)):
                nd = row.get("net_debt_ebitda")
                if pd.notna(nd) and nd > 0:
                    hit = ("altman_z_score", "Altman Z-Score 过低（破产风险）",
                           row.get("altman_z"), az.get("min", 1.8))

        if hit is not None:
            mask_pass.loc[idx] = False
            reject_rows.append({
                "code": row.get("code"),
                "name": row.get("name"),
                "industry": ind,
                "rule": hit[0],
                "rule_desc": hit[1],
                "value": hit[2],
                "threshold": hit[3],
            })

    passed = metrics[mask_pass].reset_index(drop=True)
    rejected = pd.DataFrame(reject_rows)
    return passed, rejected


def rule_coverage(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """排雷规则健康度：每条规则所依赖字段的有效率。

    字段全缺失 ⇒ 该规则**形同虚设**（如接口不提供商誉、质押只有当前快照）。
    这种「静默失效」很危险 —— 使用者会以为有这条防线，实际它从未触发过。
    因此必须显式暴露，而不是悄悄跳过。
    """
    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=["rule", "field", "desc", "coverage", "dead"])

    rules_cfg = cfg.section("vetoes").get("rules", {})
    rows = []

    def _rate(cols: list[str]) -> float:
        present = [c for c in cols if c in metrics.columns]
        if not present:
            return 0.0
        # 复合规则：任一依赖字段有效即可能触发
        return float(metrics[present].notna().any(axis=1).mean())

    for rule, (col, _direction, _key, desc) in _RULE_SPEC.items():
        if not rules_cfg.get(rule, {}).get("enabled", True):
            continue
        rate = _rate([col])
        rows.append({"rule": rule, "field": col, "desc": desc,
                     "coverage": rate, "dead": rate == 0.0})

    for rule, cols in (("leverage", ["net_debt_ebitda", "interest_coverage"]),
                       ("altman_z_score", ["altman_z", "net_debt_ebitda"])):
        if not rules_cfg.get(rule, {}).get("enabled", True):
            continue
        rate = _rate(cols)
        rows.append({"rule": rule, "field": "|".join(cols), "desc": rule,
                     "coverage": rate, "dead": rate == 0.0})

    out = pd.DataFrame(rows)
    return out.sort_values("coverage").reset_index(drop=True)


def warn_dead_rules(metrics: pd.DataFrame, cfg: Config) -> list[str]:
    """检测并显式告警「形同虚设」的排雷规则。

    规则依赖的字段在数据里全缺失时，该规则永远无法触发 —— 使用者却会以为
    它有防线。这种「静默失效」比没有规则更危险（给人虚假安全感）。

    典型场景（已在本项目真实数据上确认）：
      - 商誉占净资产：接口不提供商誉字段 ⇒ 永久失效
      - 大股东质押：接口仅给当前快照，历史 period 时间戳被年报过滤丢弃 ⇒ 失效

    返回告警文案列表（空列表表示所有规则健康）。调用方负责把它打到屏幕 / 报告。
    """
    cov = rule_coverage(metrics, cfg)
    warnings = []
    if cov.empty:
        return warnings

    dead = cov[cov["dead"]]
    for _, r in dead.iterrows():
        warnings.append(
            f"⚠️ 排雷规则失效：{r['rule']}（依赖字段 {r['field']} 在数据集中 0 覆盖）"
            f" —— 该防线形同虚设，请勿依赖它排除风险。"
        )

    # 半失效：覆盖率极低（<5%）但仍偶有触发，单独提示而非判死
    weak = cov[(~cov["dead"]) & (cov["coverage"] < 0.05)]
    for _, r in weak.iterrows():
        warnings.append(
            f"⚠️ 排雷规则覆盖率极低：{r['rule']}（{r['field']} 仅 {r['coverage']:.1%} 覆盖）"
            f" —— 多数样本未经此规则检验，结论需打折。"
        )
    return warnings


def veto_summary(rejected: pd.DataFrame) -> pd.DataFrame:
    """按规则统计否决数量 —— 哪条规则最"杀人"，一眼可见。"""
    if rejected.empty:
        return pd.DataFrame(columns=["rule", "rule_desc", "n", "share"])
    g = (
        rejected.groupby(["rule", "rule_desc"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    g["share"] = g["n"] / g["n"].sum()
    return g.reset_index(drop=True)


def to_markdown(passed: pd.DataFrame, rejected: pd.DataFrame, asof) -> str:
    """生成排雷报告（Markdown）。"""
    lines = [
        f"# 排雷报告（L3）",
        "",
        f"**决策日**：{asof}　**通过**：{len(passed)}　**否决**：{len(rejected)}",
        "",
    ]
    if not rejected.empty:
        lines += ["## 否决原因分布", ""]
        s = veto_summary(rejected)
        lines += ["| 规则 | 说明 | 数量 | 占比 |",
                  "|------|------|------|------|"]
        for _, r in s.iterrows():
            lines.append(f"| {r['rule']} | {r['rule_desc']} | {r['n']} | {r['share']:.1%} |")
        lines += ["", "## 被否决公司明细（前 50）", "",
                  "| 代码 | 名称 | 行业 | 触发规则 | 实际值 | 阈值 |",
                  "|------|------|------|---------|--------|------|"]
        for _, r in rejected.head(50).iterrows():
            val = r["value"]
            val = f"{val:.3f}" if isinstance(val, float) else str(val)
            lines.append(
                f"| {r['code']} | {r['name']} | {r['industry']} | {r['rule_desc']} | {val} | {r['threshold']} |"
            )
    return "\n".join(lines)
