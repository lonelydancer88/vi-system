"""L6 组合层：从"选出来的"到"买多少"。

三条约束的物理意义：
  - 单票上限（8%）：防止单只黑天鹅摧毁组合
  - 行业上限（25%）：价值策略天然会扎堆到银行地产，不设限等于裸赌一个宏观变量
  - 缓冲区：价值是慢因子，排名在第 11 名和第 12 名之间没有信息量，
    为这点差异付两次交易成本纯属浪费 —— 缓冲区能砍掉一半以上的无效换手
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def _apply_caps(w: pd.Series, industry: pd.Series,
                max_pos: float, max_ind: float, iters: int = 60) -> pd.Series:
    """迭代施加单票与行业上限。

    **关键：不做强行归一化。** 候选太少时（如只有 3 只通过门槛），
    硬性凑满 90% 仓位必然突破 8% 的单票上限 —— 那等于绕过了风控。
    正确做法是让投资不足的部分自动变成现金：宁可少买，不可突破上限。
    """
    w = w.astype(float).clip(lower=0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(iters):
        changed = False
        # --- 单票上限：超出的部分转移给还有空间的标的
        excess = (w - max_pos).clip(lower=0).sum()
        if excess > 1e-12:
            w = w.clip(upper=max_pos)
            head = (max_pos - w).clip(lower=0)
            if head.sum() > 1e-12:
                w = w + head / head.sum() * excess
            changed = True
        # --- 行业上限：整组等比压缩
        iw = w.groupby(industry).sum()
        over = iw > max_ind + 1e-9
        if over.any():
            scale = industry.map((max_ind / iw).where(over).reindex(iw.index)).fillna(1.0)
            w = w * scale
            changed = True
        if not changed:
            break
    return w.clip(lower=0)


def build_portfolio(
    scored: pd.DataFrame,
    cfg: Config,
    current_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """构建目标组合。

    scored          : 因子层输出（含 passes_gate / total_score / rank）
    current_weights : 当前持仓权重（index=code）。传入则启用缓冲区。
    """
    if scored.empty:
        return pd.DataFrame(columns=["code", "weight", "industry", "total_score"])

    pcfg = cfg.section("portfolio")
    lo, hi = pcfg.get("target_size", [20, 30])
    max_pos = pcfg.get("max_position", 0.08)
    max_ind = pcfg.get("max_industry", 0.25)
    min_cash = pcfg.get("min_cash", 0.10)
    buf = pcfg.get("buffer", {})
    sell_mult = buf.get("sell_rank_mult", 1.5)
    liq_floor = pcfg.get("liquidity_floor_amount", 0)

    df = scored.copy()
    df["rank_pct"] = df["rank"] / max(len(df), 1) * 100
    if liq_floor and "avg_amount_60d" in df.columns:
        df = df[df["avg_amount_60d"] >= liq_floor]

    cand = df[df["passes_gate"]].copy()
    # 估值回灌（买入侧）：L5 判为「已到卖点」的不作为新买入标的。
    # 三支柱排序只回答"便宜/好公司/安全"，不回答"价格是否已透支"；
    # 缺这一步会出现「好公司但已到卖点」仍被买入（如圆通速递）。
    # 仅作用于买入，不强制卖出已持仓 —— 退出由缓冲区与 L7 论文报警负责。
    if pcfg.get("exclude_at_sell_point", False) and "verdict" in cand.columns:
        cand = cand[cand["verdict"] != "已到卖点"]
    # 门即纪律（决策 A：允许不满仓）：无人过 AND 门槛时保持空仓，
    # 不兜底买未过门的头部——「候选不足」不是放松买入标准的理由。
    # （旧版此处为 cand = df.head(hi)，会在极端行情下悄悄凑仓，违背该决策。）
    if cand.empty:
        return pd.DataFrame(columns=["code", "weight", "industry", "total_score"])
    cand = cand.sort_values("total_score", ascending=False)
    ind_map = df.set_index("code")["industry"].to_dict()

    # ---------------------------------------------------------- 粘性持有 / 缓冲区
    # 设计（v1.4 新增 sticky_holding）：
    #   现持仓只要「仍通过 AND 门（passes_gate）且未到 L5 卖点」就作为 keeper 保留，
    #   不因排名小幅波动而被「挤出」。这是纯换手削减 —— keeper 集合是「旧缓冲区」的
    #   **超集**，所以换手只会降、不会升。只有当某槽位空缺（keeper 被清出/失败）或新候选
    #   分数高出最弱 keeper 达 sticky_margin 时，才允许新标的进入。
    #   旧缓冲区（rank<=sell_mult×hi）保留为 fallback（sticky_holding=false 时启用），
    #   以兼容不希望启用粘性持有的场景。
    sticky_on = bool(buf.get("sticky_holding", True))
    sticky_margin = float(buf.get("sticky_margin", 0.10))
    # 滞后带：粘性开启时对集中档再加绝对下限（hi+12）；关闭时退回 v1.3 原值 hi×sell_mult，
    # 保证 sticky_holding=false 时与历史基线完全一致（否则会残留更宽的滞后带）。
    if sticky_on:
        sell_cut = int(max(hi * sell_mult, hi + 12))
    else:
        sell_cut = int(hi * sell_mult)
    ranks = df.set_index("code")["rank"]
    keeper_codes: set[str] = set()
    if current_weights is not None and len(current_weights) > 0:
        cur = current_weights[current_weights > 0]
        rank_keep = {c for c in cur.index if c in ranks.index and ranks[c] <= sell_cut}
        if sticky_on:
            # 仍过 AND 门（且不在卖点；cand 已含 exclude_at_sell_point 过滤）的现持仓 → 必留
            gate_keep = set(cand[cand["code"].isin(cur.index)]["code"])
            keeper_codes = rank_keep | gate_keep
        else:
            keeper_codes = rank_keep

    # ---------------------------------------------------------- 行业配额式选股
    # 关键设计：行业上限在**选股阶段**用配额实现，而不是事后压缩权重。
    # 事后压缩会把超额权重变成现金 —— 那不是分散，那是变相空仓。
    per_ind = max(1, int(hi * max_ind))
    picked: list[str] = []
    cnt: dict[str, int] = {}

    def _try_add(code: str) -> bool:
        ind = ind_map.get(code)
        if cnt.get(ind, 0) >= per_ind:
            return False
        cnt[ind] = cnt.get(ind, 0) + 1
        picked.append(code)
        return True

    score_of = cand.set_index("code")["total_score"].to_dict()
    ranked = cand["code"].tolist()                     # 分数降序
    ranked_new = [c for c in ranked if c not in keeper_codes]

    # 1) 先放 keeper（按分数降序，保证填得满时优先保高分）
    for c in sorted(keeper_codes, key=lambda x: -score_of.get(x, -1e9)):
        if len(picked) >= hi:
            break
        _try_add(c)
    # 2) 补足到持股数下限（用新候选）
    for c in ranked_new:
        if len(picked) >= lo:
            break
        if c not in picked:
            _try_add(c)
    # 3) 仍有空间则继续用新候选加满
    for c in ranked_new:
        if len(picked) >= hi:
            break
        if c not in picked:
            _try_add(c)
    # 4) 边际刷新（仅 sticky_on 且 sticky_margin>0）：新候选分数高出最弱 keeper 达 margin 才替换。
    #    关闭粘性时整段跳过 —— 与 v1.3 行为完全一致。
    if sticky_on and sticky_margin > 0:
        for c in ranked_new:
            if c in picked:
                continue
            if len(picked) < hi:                        # 空槽兜底（前三步应已填满）
                _try_add(c)
                continue
            weak = min(picked, key=lambda x: score_of.get(x, -1e9))
            weak_rank = ranks.get(weak, 1e9)
            # 仅当最弱 keeper 已跌出滞后带（weak_rank>sell_cut）且新候选明显更优才替换：
            # 滞内 keeper 永不因排名噪声被「挤出」，只有真正退化的持仓才让位给更优标的。
            if weak_rank > sell_cut and score_of.get(c, -1e9) > score_of.get(weak, -1e9) + sticky_margin:
                picked.remove(weak)
                if not _try_add(c):                     # 行业上限挡住，回退并跳过此候选
                    picked.append(weak)
            else:
                break

    sel_df = df[df["code"].isin(picked)].copy()
    if sel_df.empty:
        return pd.DataFrame(columns=["code", "weight", "industry", "total_score"])

    # ---------------------------------------------------------- 权重：分数 × 流动性
    base = sel_df["total_score"].clip(lower=0).fillna(0) + 1e-6
    if "avg_amount_60d" in sel_df.columns:
        liq = np.sqrt(sel_df["avg_amount_60d"].clip(lower=1).fillna(1))
        liq = liq / liq.median() if liq.median() and np.isfinite(liq.median()) else liq * 0 + 1
        base = base * liq.clip(upper=3)

    w = pd.Series(base.values, index=sel_df["code"].values)
    ind = pd.Series(sel_df["industry"].values, index=sel_df["code"].values)
    w = _apply_caps(w, ind, max_pos, max_ind) * (1 - min_cash)

    name_map = sel_df.set_index("code")["name"].reindex(w.index).values
    out = pd.DataFrame({
        "code": w.index,
        "name": name_map,
        "weight": w.values,
        "industry": ind.reindex(w.index).values,
        "total_score": sel_df.set_index("code")["total_score"].reindex(w.index).values,
    }).sort_values("weight", ascending=False).reset_index(drop=True)
    return out


def turnover(old: pd.Series | None, new: pd.Series) -> float:
    """单边换手率：sum(|w_new - w_old|) / 2 的两倍口径，这里返回 sum|Δw|。"""
    if old is None or len(old) == 0:
        return float(new.sum())
    idx = old.index.union(new.index)
    a = old.reindex(idx).fillna(0.0)
    b = new.reindex(idx).fillna(0.0)
    return float((b - a).abs().sum())


def portfolio_report(pf: pd.DataFrame, cfg: Config) -> str:
    if pf.empty:
        return "# 组合\n\n无持仓"
    lines = [
        "# 组合（L6）", "",
        f"- 持仓数：{len(pf)}",
        f"- 现金比例：{max(0.0, 1 - pf['weight'].sum()):.1%}",
        "",
        "## 行业分布", "",
    ]
    iw = pf.groupby("industry")["weight"].sum().sort_values(ascending=False)
    lines += ["| 行业 | 权重 |", "|------|------|"]
    for k, v in iw.items():
        lines.append(f"| {k} | {v:.2%} |")
    lines += ["", "## 持仓明细", "",
              "| 代码 | 名称 | 行业 | 权重 | 总分 |", "|------|------|------|------|------|"]
    for _, r in pf.iterrows():
        name = r.get("name", "")
        lines.append(f"| {r['code']} | {name} | {r['industry']} | {r['weight']:.2%} | {r['total_score']:.1f} |")
    return "\n".join(lines)
