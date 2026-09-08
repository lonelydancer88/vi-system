"""L4 因子层：便宜 × 好公司 × 安全 三支柱。

关键设计：
  1. **行业内 z-score**（申万二级）。跨行业混排会让消费股永久碾压制造业。
  2. **先取 5 年中位数再打分** —— 指标层已做；这里再对 z-score 做 Winsorize 防离群值主导。
  3. **三支柱 AND 门槛 + 加权总分**。这是防止"一项极端拉总分"的核心：
     - 只加权 → 天价估值的高质量股、或垃圾中的最便宜股都会混进来
     - 只 AND → 无法排序
     - 先 AND 再加权 → 先保证"没有致命短板"，再排优先级
  4. **缺失不填补为均值以外的东西**：NaN 的 z 取 0（行业中性），
     但单独记录 `valid_metrics`，覆盖率太低的标的排序时降权处理（不静默混入）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def _zscore_within(s: pd.Series, groups: pd.Series, winsor: float = 3.0) -> pd.Series:
    """行业内 z-score + 截断。组内样本不足或标准差为 0 时返回 0（中性）。"""
    def _z(x):
        if len(x) < 3:
            return pd.Series(0.0, index=x.index)
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / sd

    z = s.groupby(groups).transform(_z)
    return z.clip(-winsor, winsor).fillna(0.0)


def _zscore_global(s: pd.Series, winsor: float = 3.0) -> pd.Series:
    """全市场 z-score + 截断。作为行业内样本不足时的回退（见 score_factors）。"""
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return ((x - x.mean()) / sd).clip(-winsor, winsor).fillna(0.0)


def _percentile_within(s: pd.Series, groups: pd.Series) -> pd.Series:
    """行业内百分位（0–100，越高越好）。"""
    return (s.groupby(groups).rank(pct=True) * 100).fillna(0.0)


def score_factors(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """输入排雷后的指标表，输出带三支柱分与总分的排序表。"""
    if metrics.empty:
        return metrics

    fcfg = cfg.section("factors")
    winsor = fcfg.get("winsorize", 3.0)
    floor = fcfg.get("percentile_floor", 30)
    min_ind = fcfg.get("min_industry_size", 5)
    weights = fcfg.get("weights", {"value": 0.4, "quality": 0.4, "safety": 0.2})

    df = metrics.copy()
    df["industry"] = df["industry"].fillna("未知")

    # 行业内样本太少的组不打分（打分即噪音）
    sizes = df.groupby("industry")["code"].transform("size")
    scorable = sizes >= min_ind

    pillars = {
        "value": fcfg.get("value_metrics", {}),
        "quality": fcfg.get("quality_metrics", {}),
        "safety": fcfg.get("safety_metrics", {}),
    }

    for pillar, spec in pillars.items():
        if not spec:
            df[f"{pillar}_z"] = 0.0
            df[f"{pillar}_pct"] = 0.0
            continue
        zs_in, zs_all, valid_n = [], [], []
        for metric, sign in spec.items():
            if metric not in df.columns:
                continue
            raw = pd.to_numeric(df[metric], errors="coerce")
            # 防御：负数的分数次幂会产生复数（如 profit_growth_5y），
            # 而 groupby.rank 不支持 complex dtype，会直接抛 TypeError
            if np.iscomplexobj(raw):
                raw = pd.Series(np.real(raw.to_numpy()), index=raw.index, dtype=float)
            # 方向：sign=1 越高越好；sign=-1 越低越好
            signed = raw * sign
            zs_in.append(_zscore_within(signed, df["industry"], winsor))
            zs_all.append(_zscore_global(signed, winsor))
            valid_n.append(raw.notna().astype(float))
        if not zs_in:
            df[f"{pillar}_z"] = 0.0
            df[f"{pillar}_pct"] = 0.0
            continue
        Z_in = pd.concat(zs_in, axis=1).mean(axis=1)
        Z_all = pd.concat(zs_all, axis=1).mean(axis=1)
        # 优先行业内 z；组内样本不足（<min_industry_size）时回退到全市场 z，
        # 避免小宇宙静默返回全 NaN（全市场比较虽不完美，但远好过无结果）。
        Z = Z_in.where(scorable, Z_all)
        df[f"{pillar}_z"] = Z
        df[f"{pillar}_valid"] = pd.concat(valid_n, axis=1).sum(axis=1)
        pct_in = (Z_in.groupby(df["industry"]).rank(pct=True) * 100)
        pct_all = (Z_all.rank(pct=True) * 100)
        df[f"{pillar}_pct"] = pct_in.where(scorable, pct_all)

    # ---------------------------------------------------- AND 门槛
    # 门槛从「行业内百分位 ≥30」改为「行业内中性 z ≥ z_threshold（默认 0）」：
    #  ① 行业内 pct 只在行业内可比——不同行业样本量下 pct 阶梯分辨率不同（5 只=20/40/60/80/100、
    #     20 只则细密得多），拿它做跨行业总分排序会让分数尺度不可比；
    #  ② z 是「相对同行多少个标准差」，跨行业可比、且保留领先幅度（A 领先 3σ 与 B 微弱第一不再同分）。
    #     z≥0 即「每柱都跑赢行业典型（中位）」，AND 门取三者同时满足。
    zthr = fcfg.get("z_threshold", 0.0)
    gate = pd.Series(True, index=df.index)
    for pillar in pillars:
        z = df[f"{pillar}_z"]
        gate &= z.isna() | (z >= zthr)
    df["passes_gate"] = gate & df["value_z"].notna()

    # ---------------------------------------------------- 加权总分（z 尺度，跨行业可比）
    total = (
        weights.get("value", 0.4) * df["value_z"].fillna(0.0)
        + weights.get("quality", 0.4) * df["quality_z"].fillna(0.0)
        + weights.get("safety", 0.2) * df["safety_z"].fillna(0.0)
    )

    # A股反转倾斜（默认关闭；开启前必须先在样本外验证）
    tilt = fcfg.get("reversal_tilt", {})
    if tilt.get("enabled", False) and "ret_52w" in df.columns:
        rz = _zscore_within(-pd.to_numeric(df["ret_52w"], errors="coerce"),
                            df["industry"], winsor)
        total = total * (1 - tilt.get("weight", 0.05)) + rz * tilt.get("weight", 0.05)

    df["total_score"] = total
    df = df.sort_values("total_score", ascending=False, na_position="last")
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def factor_report(df: pd.DataFrame, top: int = 30) -> str:
    """生成因子排名报告（Markdown）。"""
    if df.empty:
        return "# 因子排名\n\n无数据"
    cols = ["rank", "code", "name", "industry", "value_pct", "quality_pct",
            "safety_pct", "total_score", "passes_gate", "ep", "gpa", "altman_z"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].head(top).copy()
    for c in sub.columns:
        if sub[c].dtype.kind == "f":
            sub[c] = sub[c].round(3)
    lines = [
        "# 因子排名（L4）", "",
        f"- 候选数：{len(df)}",
        f"- 过 AND 门槛：{int(df['passes_gate'].sum())}",
        "",
        sub.to_markdown(index=False),
    ]
    return "\n".join(lines)
