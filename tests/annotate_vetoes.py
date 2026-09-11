"""把「复核判断」列整合进 L3 排雷主报告（out/vetoes-{asof}.md）。

直接复用 vi_system.pipeline.veto_judge 的同一套领域知识判断，保证与
`cli screen` 输出完全一致——也就是说无论走哪条路径，L3 报告都带「复核判断」列。

判断由大模型（领域知识）给出：
  ✅ 真雷   —— 否决正确，确属财务风险/破产/真稀释
  ⚠️ 误报   —— 模型/规则对该类公司的系统性误判，不应直接否掉
  🔍 需复核 —— 无法一刀切，需结合具体业务人工核对

用法：python3 tests/annotate_vetoes.py [asof]
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
sys.path.insert(0, str(ROOT))

from vi_system.config import load_config
from vi_system.data.store import Store
from vi_system.backtest import engine as bt
from vi_system.pipeline.veto_judge import veto_report_md, judge_column, label


def main():
    asof = sys.argv[1] if len(sys.argv) > 1 else "2026-09-09"
    cfg = load_config()
    st = Store(str(ROOT / "data" / "real_universe"))
    _, rejected, uni = bt.screen_at(st, asof, cfg, with_valuation=True)
    if rejected is None or rejected.empty:
        print("无否决记录")
        return

    # 写回主 L3 报告（带「复核判断」列）
    out = ROOT / "out" / f"vetoes-{asof}.md"
    out.write_text(veto_report_md(rejected), encoding="utf-8")

    # 顺带打印复核分布
    tmp = rejected.copy()
    tmp["j"] = judge_column(tmp)
    tmp["l"] = tmp["j"].map(label)
    dist = tmp["l"].value_counts().to_dict()
    print(f"→ 已写入 {out}")
    print(f"复核分布：真雷 {dist.get('真雷', 0)} / 误报 {dist.get('误报', 0)} / 需复核 {dist.get('需复核', 0)}")


if __name__ == "__main__":
    main()
