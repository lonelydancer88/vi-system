"""L8 治理层：决策日志与复盘。

核心原则：**复盘只评"流程是否被执行"，不评盈亏。**
赚钱的错误决策比亏钱的正确决策更危险 —— 它会强化一个坏流程。

每条决策必须记录触发它的规则版本号。没有版本号，半年后无法回答
"当时为什么把它筛掉了"，系统也就失去了学习能力。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import Config


class DecisionLog:
    """追加式决策日志（JSONL）。"""

    def __init__(self, path: str | Path, cfg: Config | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

    def add(
        self,
        code: str,
        name: str,
        action: str,                 # 买入 / 加仓 / 减仓 / 清仓 / 观察
        price: float,
        weight: float | None,
        scores: dict | None,         # {value_pct, quality_pct, safety_pct, total_score}
        thesis: str,                 # 200 字内：为什么买、什么条件下卖
        confidence: str = "中",      # 高 / 中 / 低
        rules_version: str | None = None,
        note: str = "",
    ) -> dict:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "code": code, "name": name, "action": action,
            "price": price, "weight": weight,
            "scores": scores or {},
            "thesis": thesis, "confidence": confidence,
            "rules_version": rules_version or (self.cfg.stamp if self.cfg else "unknown"),
            "note": note,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = [json.loads(l) for l in open(self.path, "r", encoding="utf-8") if l.strip()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "scores" in df.columns:
            sc = pd.json_normalize(df["scores"]).add_prefix("sc_")
            df = pd.concat([df.drop(columns=["scores"]), sc], axis=1)
        return df

    def classify(self, code: str, error_type: str, reason: str) -> None:
        """为某条决策补记错误分类（半年复盘时用）。"""
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "code": code, "action": "复盘归类",
            "error_type": error_type, "reason": reason,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


ERROR_TYPES = {
    "I": "买了坏公司（排雷层漏洞）",
    "II": "好公司买贵了（估值层门槛太松）",
    "III": "好公司没拿住（论文不够硬/仓位过重）",
    "IV": "该买没买（宇宙覆盖缺口/恐惧）",
}


def review_report(log: DecisionLog) -> str:
    """生成复盘报告骨架。

    刻意不计算盈亏 —— 复盘要回答的是"流程执行得怎么样"，
    把盈亏放进复盘表，人就会开始用结果倒推过程。
    """
    df = log.load()
    lines = ["# 复盘报告（L8）", "",
             "> 复盘只评流程执行，不评盈亏。"
             "赚钱的错误决策比亏钱的正确决策更危险 —— 它会强化一个坏流程。", ""]
    if df.empty:
        lines += ["暂无决策记录。"]
        return "\n".join(lines)

    trades = df[df["action"] != "复盘归类"]
    lines += [f"- 决策条数：{len(trades)}",
              f"- 涉及标的：{trades['code'].nunique()}",
              f"- 规则版本：{', '.join(sorted(set(trades['rules_version'].astype(str))))}",
              ""]

    # 论文完整性检查：thesis 太短 = 没想清楚就买了
    # 阈值 25 字：短于这个长度基本不可能同时说清"为什么买"和"什么条件下卖"
    if "thesis" in trades.columns:
        short = trades[trades["thesis"].fillna("").str.len() < 25]
        lines += [f"## 论文完整性", "",
                  f"- 论文过短（<25字）的决策：{len(short)} / {len(trades)}", ""]
        if len(short):
            lines += ["| 代码 | 动作 | 论文字数 |", "|------|------|------|"]
            for _, r in short.head(20).iterrows():
                lines.append(f"| {r['code']} | {r['action']} | {len(str(r['thesis']))} |")
            lines.append("")

    errs = df[df["action"] == "复盘归类"]
    lines += ["## 错误分类", "", "| 类型 | 说明 | 次数 |", "|------|------|------|"]
    if errs.empty:
        for k, v in ERROR_TYPES.items():
            lines.append(f"| {k} | {v} | 0 |")
    else:
        cnt = errs["error_type"].value_counts()
        for k, v in ERROR_TYPES.items():
            lines.append(f"| {k} | {v} | {int(cnt.get(k, 0))} |")
        lines += ["", "### 归因指向", ""]
        guide = {
            "I": "→ 检查 L3 排雷规则是否漏了这条路径",
            "II": "→ 检查 L5 估值门槛（buy_discount）是否太松",
            "III": "→ 论文写得不够硬，或仓位过重导致拿不住",
            "IV": "→ L1 宇宙覆盖有缺口，或当时被恐惧支配",
        }
        for k, n in cnt.items():
            if k in guide and n:
                lines.append(f"- **{k} 类 {n} 次**：{guide[k]}")
    return "\n".join(lines)
