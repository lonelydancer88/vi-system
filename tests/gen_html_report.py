"""汇总报告生成器：把本轮回测/筛选/持仓/排雷产物合成一个自包含 HTML。

用法: python3 tests/gen_html_report.py [asof]
输出: out/report-<asof>.html (内嵌 nav-curve.png base64，可直接双击/预览)
"""
from __future__ import annotations
import base64
import io
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
ASOF = sys.argv[1] if len(sys.argv) > 1 else "2026-09-09"


def read_table(md_path: Path, table_idx: int = -1) -> pd.DataFrame:
    """从 markdown 取第 table_idx 个表格（默认最后一个），按 | 切分。"""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    blocks, cur = [], []
    for ln in lines:
        if ln.strip().startswith("|"):
            cur.append(ln.strip().strip("|"))
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    if not blocks:
        return pd.DataFrame()
    block = blocks[table_idx]
    rows = []
    for i, ln in enumerate(block):
        if i == 1 and set(ln.replace("-", "").replace(":", "").replace(" ", "")) == set():
            continue  # 分隔行
        cells = [c.strip() for c in ln.split("|")]
        rows.append(cells)
    if not rows:
        return pd.DataFrame()
    hdr, body = rows[0], rows[1:]
    return pd.DataFrame(body, columns=hdr)


def nav_b64() -> str:
    p = OUT / "nav-curve.png"
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


# ---- 回测关键指标（从 backtest-hs300.md 解析）
bt_lines = (OUT / "backtest-hs300.md").read_text(encoding="utf-8")
metrics = {}
for key in ["策略年化", "基准年化", "策略最大回撤", "基准最大回撤", "超额年化(对沪深300)",
            "信息比率(对沪深300)", "超额年化(对等权)", "信息比率(对等权)", "策略波动", "基准波动"]:
    m = re.search(re.escape(key) + r"\s*\|\s*([+\-]?[\d.%]+)", bt_lines)
    if m:
        metrics[key] = m.group(1)

# ---- 持仓 / 估值 / 排雷
port = read_table(OUT / f"portfolio-{ASOF}.md", table_idx=-1)   # 持仓明细
ind = read_table(OUT / f"portfolio-{ASOF}.md", table_idx=0)      # 行业分布
val = read_table(OUT / f"valuation-{ASOF}.md", table_idx=-1)     # 估值明细
vet = read_table(OUT / f"vetoes-{ASOF}.md", table_idx=-1)        # 排雷明细(含复核判断)

# 排雷汇总：按复核判断的前缀标签（✅真雷 / ⚠️误报 / 🔍需复核 / 其它）聚合
vet_summary = ""
if vet is not None and not vet.empty:
    col = vet.columns[-1]
    def _cls(s):
        if not isinstance(s, str):
            return "其它"
        if "真雷" in s:
            return "✅ 真雷"
        if "误报" in s:
            return "⚠️ 误报"
        if "需复核" in s:
            return "🔍 需复核"
        return "其它"
    counts = vet[col].map(_cls).value_counts().to_dict()
    vet_summary = " / ".join(f"{k}: {v}" for k, v in counts.items())

# ---- HTML
def df_to_html(df: pd.DataFrame, max_rows=None, cls="tbl") -> str:
    d = df if max_rows is None else df.head(max_rows)
    return d.to_html(index=False, classes=cls, border=0, escape=False)


nav_img = nav_b64()
holdings_html = df_to_html(port, cls="tbl")
ind_html = df_to_html(ind, cls="tbl")
val_html = df_to_html(val, max_rows=30, cls="tbl")
vet_html = df_to_html(vet, max_rows=60, cls="tbl")

m = metrics
html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>价值投资选股系统 · 综合报告 {ASOF}</title>
<style>
  :root {{ --bg:#f6f7f9; --card:#fff; --ink:#1f2937; --sub:#6b7280; --line:#e5e7eb;
          --up:#c0392b; --down:#27ae60; --accent:#2563eb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:28px 20px 60px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .meta {{ color:var(--sub); font-size:13px; margin-bottom:22px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:26px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }}
  .kpi .v {{ font-size:22px; font-weight:700; }}
  .kpi .l {{ font-size:12px; color:var(--sub); margin-top:2px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 20px; margin-bottom:22px; }}
  .card h2 {{ font-size:17px; margin:0 0 12px; }}
  img.curve {{ width:100%; border-radius:8px; border:1px solid var(--line); }}
  table.tbl {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.tbl th, table.tbl td {{ border:1px solid var(--line); padding:6px 9px; text-align:left; }}
  table.tbl th {{ background:#f0f4f8; font-weight:600; }}
  table.tbl tr:nth-child(even) td {{ background:#fafbfc; }}
  .note {{ font-size:12.5px; color:var(--sub); line-height:1.6; }}
  .warn {{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412;
          border-radius:10px; padding:12px 14px; font-size:13px; margin-bottom:22px; }}
  .tag-t {{ color:var(--up); font-weight:600; }}
  .tag-f {{ color:var(--down); font-weight:600; }}
</style></head>
<body><div class="wrap">
  <h1>价值投资选股系统 · 综合报告</h1>
  <div class="meta">截面日 {ASOF} ｜ 基准对照：沪深300（价格回报）& 等权全市场 ｜ 数据口径：point-in-time 财务 + 干净行情</div>

  <div class="grid">
    <div class="kpi"><div class="v">{m.get('策略年化','—')}</div><div class="l">策略年化</div></div>
    <div class="kpi"><div class="v">{m.get('超额年化(对沪深300)','—')}</div><div class="l">超额年化(沪深300)</div></div>
    <div class="kpi"><div class="v">{m.get('信息比率(对沪深300)','—')}</div><div class="l">信息比率(沪深300)</div></div>
    <div class="kpi"><div class="v">{m.get('策略最大回撤','—')}</div><div class="l">策略最大回撤</div></div>
    <div class="kpi"><div class="v">{m.get('基准最大回撤','—')}</div><div class="l">沪深300最大回撤</div></div>
  </div>

  <div class="card"><h2>净值曲线</h2>
    {f'<img class="curve" src="{nav_img}">' if nav_img else '<p class="note">nav-curve.png 缺失</p>'}
  </div>

  <div class="warn">排雷「复核判断」分布（L3 一票否决仍生效，本列仅供人工复核参考，不改变结果）：{vet_summary or '—'}<br>
  已知数据缺口：排雷规则 <code>goodwill_to_equity</code>（商誉/净资产）因数据源 0 覆盖而失效，该防线形同虚设，请勿依赖其排除风险。</div>

  <div class="card"><h2>持仓明细（L6 · 30 只）</h2>
    {holdings_html}
  </div>

  <div class="card"><h2>行业分布</h2>
    {ind_html}
  </div>

  <div class="card"><h2>估值与买卖点（L5）</h2>
    {val_html}
  </div>

  <div class="card"><h2>排雷明细（L3 · 含大模型复核判断）</h2>
    {vet_html}
  </div>

  <p class="note">报告由 tests/gen_html_report.py 自动汇总自 out/ 下本轮生成的 markdown 与 nav-curve.png。
  所有数字均来自系统计算产物，未做手抄转写。</p>
</div></body></html>"""

(OUT / f"report-{ASOF}.html").write_text(html, encoding="utf-8")
print(f"→ out/report-{ASOF}.html  ({len(html)/1024:.0f} KB, nav-embedded={bool(nav_img)})")
