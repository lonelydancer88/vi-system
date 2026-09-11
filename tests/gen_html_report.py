"""汇总报告生成器：把本轮回测/筛选/持仓/排雷产物合成一个自包含 HTML。

按「策略」组织：默认档 = 30只（无后缀文件名）；top3 / top5 集中组合各一区块。
每区块内顺序固定：持仓 → 回测 → 逐期明细（调仓动作表已含价格/收益/手数/占用，即原 trades 台账合并而来）。
顶部「五策略对比总览」= 3 个策略（top3 / top5 / 默认30只）+ 2 个基准（沪深300 / 等权全市场）。
折叠：策略整块、每张内容卡片、回测逐期明细，三层均 <details> 可折（默认展开；逐期明细默认收起）。

用法: python3 tests/gen_html_report.py [asof]
输出: out/report-<asof>.html (内嵌 nav-curve.png base64，可直接双击/预览)
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
ASOF = sys.argv[1] if len(sys.argv) > 1 else "2026-09-07"


# ---------------- markdown → html（自包含，无外部依赖） ----------------
def _inline(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+?)`", r"<code>\1</code>", s)
    return s


def _split_row(r: str) -> list[str]:
    r = r.strip().strip("|")
    return [c.strip() for c in r.split("|")]


def _render_table(rows: list[str]) -> str:
    if len(rows) < 2:
        return ""
    header = _split_row(rows[0])
    body = rows[2:]  # 跳过分隔行（index 1）
    th = "".join(f"<th>{_inline(c)}</th>" for c in header)
    trs = ""
    for b in body:
        cells = _split_row(b)
        cells += [""] * (len(header) - len(cells))
        trs += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells[: len(header)]) + "</tr>"
    return f'<table class="tbl"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def md_to_html(md: str) -> str:
    """把整篇 markdown 转 HTML：标题 / 表格 / 列表 / 引用 / 分隔线 / 加粗 / 行内代码。"""
    if not md:
        return '<p class="note">（文件缺失）</p>'
    lines = md.split("\n")
    out: list[str] = []
    table: list[str] = []
    n = len(lines)

    def flush_table():
        if table:
            out.append(_render_table(table))
            table.clear()

    i = 0
    while i < n:
        s = lines[i].strip()
        if s.startswith("|") and s.count("|") >= 2:
            table.append(s)
            i += 1
            continue
        flush_table()
        if s == "":
            i += 1
            continue
        if s == "---":
            out.append("<hr>")
            i += 1
            continue
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(buf))}</blockquote>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue
        if re.match(r"^[-*]\s+", s):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                it = lines[i].strip()
                items.append(_inline(it[it.index(" ") + 1:]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>")
            continue
        out.append(f"<p>{_inline(s)}</p>")
        i += 1
    flush_table()
    return "\n".join(out)


# ---------------- 指标解析 / 表格抽取 ----------------
METRIC_KEYS = ["策略年化", "策略波动", "策略最大回撤", "夏普",
               "超额年化(对沪深300)", "信息比率(对沪深300)", "胜率(对沪深300)"]


def parse_metrics(md: str) -> dict:
    d = {}
    for k in METRIC_KEYS:
        m = re.search(re.escape(k) + r"\s*\|\s*([+\-]?[\d.%]+)", md or "")
        if m:
            d[k] = m.group(1)
    return d


def parse_bench(md: str) -> dict:
    """从 backtest-hs300.md 首个表抽取两个基准的年化/波动/最大回撤。

    表形如：| 基准年化 | 14.23% | 3.24% |（列序：等权基准、沪深300 基准）
    """
    got = {"等权全市场": {}, "沪深300": {}}
    keymap = {"基准年化": "年化", "基准波动": "波动", "基准最大回撤": "回撤"}
    for ln in (md or "").splitlines():
        if not ln.strip().startswith("|"):
            continue
        c = _split_row(ln)
        if len(c) >= 3 and c[0] in keymap:
            got["等权全市场"][keymap[c[0]]] = c[1]
            got["沪深300"][keymap[c[0]]] = c[2]
    return got


def extract_regime(md: str) -> str:
    """从 backtest.md 抽取「分市场状态检验」表（hs300 版没有，避免重复）。"""
    if not md:
        return ""
    lines = md.split("\n")
    for idx, ln in enumerate(lines):
        if "市场状态" in ln or "分市场状态" in ln:
            tbl = []
            j = idx + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("|")):
                if lines[j].strip().startswith("|"):
                    tbl.append(lines[j].strip())
                j += 1
            if tbl:
                return _render_table(tbl)
    return ""


def read_md(name: str) -> str:
    p = OUT / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _color(v: str) -> str:
    if v.startswith("+"):
        return f'<span class="up">{v}</span>'
    if v.startswith("-"):
        return f'<span class="down">{v}</span>'
    return v


def nav_b64() -> str:
    p = OUT / "nav-curve.png"
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


# ---------------- 区块构件 ----------------
def card(title: str, md: str, anchor: str | None = None,
         open_: bool = True) -> str:
    """所有 card 统一可折叠：标题即 <summary>，默认展开（open_=False 则默认收起）。"""
    a = f' id="{anchor}"' if anchor else ""
    o = " open" if open_ else ""
    return (f'<details class="card"{a}{o}>'
            f'<summary class="card-h">{title}</summary>'
            f'<div class="fold-body">{md_to_html(md)}</div></details>')


def details(summary: str, md: str, scroll: bool = False) -> str:
    cls = ' class="scroll"' if scroll else ""
    return (f'<details><summary>{summary}</summary>'
            f'<div{cls}>{md_to_html(md)}</div></details>')


# ---------------- 读取全部产物 ----------------
nav_img = nav_b64()
bt_hs300 = {  # 无后缀 = 默认30只；top3/top5 带后缀
    "默认(30只)": read_md("backtest-hs300.md"),
    "top3": read_md("backtest-hs300-top3.md"),
    "top5": read_md("backtest-hs300-top5.md"),
}
bt_plain = {
    "默认(30只)": read_md("backtest.md"),
    "top3": read_md("backtest-top3.md"),
    "top5": read_md("backtest-top5.md"),
}
port = {
    "默认(30只)": read_md(f"portfolio-{ASOF}.md"),
    "top3": read_md(f"portfolio-top3-{ASOF}.md"),
    "top5": read_md(f"portfolio-top5-{ASOF}.md"),
}
rep = {
    "默认(30只)": read_md(f"回测报告-{ASOF}.md"),
    "top3": read_md(f"回测报告-top3-{ASOF}.md"),
    "top5": read_md(f"回测报告-top5-{ASOF}.md"),
}
valuation = read_md(f"valuation-{ASOF}.md")
vetoes = read_md(f"vetoes-{ASOF}.md")
advice = read_md(f"持仓建议-{ASOF}.md")

# 排雷「复核判断」分布
vet_summary = ""
if vetoes:
    rows = [ln for ln in vetoes.splitlines() if ln.strip().startswith("|")]
    if len(rows) >= 3:
        col = _split_row(rows[-1])
        last = col[-1] if col else ""

        def _cls(s):
            if "真雷" in s:
                return "✅ 真雷"
            if "误报" in s:
                return "⚠️ 误报"
            if "需复核" in s:
                return "🔍 需复核"
            return "其它"

        counts = {}
        for r in rows[2:]:
            c = _split_row(r)
            if not c:
                continue
            k = _cls(c[-1])
            counts[k] = counts.get(k, 0) + 1
        vet_summary = " / ".join(f"{k}: {v}" for k, v in counts.items())


# ---------------- 顶部对比总览（3 策略 + 2 基准） ----------------
BENCH = parse_bench(bt_hs300["默认(30只)"])


def _ov_strat(label, md):
    d = parse_metrics(md)
    g = lambda k: d.get(k, "—")
    return ("<tr><td>" + label + "</td><td>" + _color(g("策略年化")) + "</td><td>"
            + g("策略波动") + "</td><td>" + _color(g("策略最大回撤")) + "</td><td>"
            + g("夏普") + "</td><td>" + _color(g("超额年化(对沪深300)")) + "</td><td>"
            + g("信息比率(对沪深300)") + "</td><td>"
            + g("胜率(对沪深300)") + "</td></tr>")


def _ov_bench(label, d):
    g = lambda k: d.get(k, "—")
    return ('<tr class="bench"><td>' + label
            + '<span class="tag">基准</span></td><td>' + _color(g("年化"))
            + "</td><td>" + g("波动") + "</td><td>" + _color(g("回撤"))
            + "</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>")


ov_html = (
    '<table class="tbl"><thead><tr><th>策略 / 基准</th><th>年化</th><th>年化波动</th>'
    '<th>最大回撤</th><th>夏普</th><th>年化超额<br>(vs 沪深300)</th>'
    '<th>信息比率<br>(vs 沪深300)</th><th>胜率<br>(vs 沪深300)</th></tr></thead><tbody>'
    + _ov_strat("top3（集中 3 只）", bt_hs300["top3"])
    + _ov_strat("top5（集中 5 只）", bt_hs300["top5"])
    + _ov_strat("默认（30 只）", bt_hs300["默认(30只)"])
    + _ov_bench("沪深300（价格回报）", BENCH["沪深300"])
    + _ov_bench("等权全市场", BENCH["等权全市场"])
    + "</tbody></table>"
)


# ---------------- 三策略区块 ----------------
def strategy_block(anchor, title, tiers, open_: bool = True):
    """整个策略区块本身可折叠：标题 = <summary>，默认展开。"""
    o = " open" if open_ else ""
    return (
        f'<details class="strat" id="{anchor}"{o}>'
        f'<summary class="strat-h">{title}</summary>' + "".join(tiers) + "</details>"
    )


# 策略 A / B 共用构造器（集中组合）
def concentrated_block(anchor, title, key):
    parts = [
        card(f"{title} · 持仓明细（L6）", port[key]),
        card(f"{title} · 回测指标（含沪深300对比）",
             bt_plain[key] + "\n\n" + bt_hs300[key]),
        details(f"{title} · 回测逐期明细（调仓 / 买卖原因 / 价格·收益·手数）", rep[key], scroll=False),
    ]
    return strategy_block(anchor, title, parts)


block_a = concentrated_block("sec-a", "策略 A：top3 集中组合", "top3")
block_b = concentrated_block("sec-b", "策略 B：top5 集中组合", "top5")

# 策略 C：默认档（30只）
c_parts = [
    card("持仓明细（L6 · 30只）", port["默认(30只)"]),
    card("估值与买卖点（L5）", valuation),
    card("排雷明细（L3 · 含大模型复核判断）", vetoes),
    card("持仓建议（默认档 · 含手数）", advice),
    card("回测指标（含沪深300对比 + 分市场状态）",
         bt_hs300["默认(30只)"] + "\n\n## 分市场状态检验（来自 backtest.md）\n\n"
         + (extract_regime(bt_plain["默认(30只)"]) or "（无）")),
    details("回测逐期明细（调仓 / 买卖原因 / 价格·收益·手数）", rep["默认(30只)"], scroll=False),
]
block_c = strategy_block("sec-c", "策略 C：默认档（30只）", c_parts)

# ---------------- 组装 HTML ----------------
n_fold = (block_a + block_b + block_c).count("<details")
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
  .wrap {{ max-width:1100px; margin:0 auto; padding:28px 20px 60px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .meta {{ color:var(--sub); font-size:13px; margin-bottom:18px; }}
  .nav {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:10px 16px; margin-bottom:22px; font-size:14px; }}
  .nav a {{ color:var(--accent); text-decoration:none; margin-right:18px; font-weight:600; }}
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
  table.tbl tr.bench td {{ background:#f8fafc; color:var(--sub); }}
  table.tbl tr.bench td:first-child {{ color:var(--ink); font-weight:600; }}
  .tag {{ display:inline-block; margin-left:6px; padding:0 6px; border-radius:8px;
         background:#eef2ff; color:#4338ca; font-size:11px; font-weight:600; vertical-align:1px; }}
  .strat {{ border-left:4px solid var(--accent); padding-left:16px; margin:30px 0 10px; }}
  .strat-h {{ color:var(--accent); font-size:20px; margin:0 0 14px; }}
  .scroll {{ max-height:540px; overflow:auto; border:1px solid var(--line);
            border-radius:8px; padding:10px 14px; margin-top:8px; }}
  details {{ margin-bottom:16px; }}
  summary {{ cursor:pointer; font-weight:600; padding:8px 0; color:var(--ink);
            user-select:none; }}
  summary:hover {{ color:var(--accent); }}
  details:not(.card)[open] > summary {{ color:var(--accent); margin-bottom:6px; }}
  details.strat > summary {{ padding:2px 0; }}
  details.card > summary {{ font-size:17px; padding:2px 0; }}
  .fold-body {{ margin-top:10px; }}
  .toolbar {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap;
             margin:-10px 0 20px; font-size:12.5px; color:var(--sub); }}
  .toolbar button {{ font:inherit; font-size:13px; padding:6px 13px; border-radius:8px;
                    border:1px solid var(--line); background:var(--card);
                    color:var(--ink); cursor:pointer; font-weight:600; }}
  .toolbar button:hover {{ border-color:var(--accent); color:var(--accent); }}
  .note {{ font-size:12.5px; color:var(--sub); line-height:1.6; }}
  .warn {{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412;
          border-radius:10px; padding:12px 14px; font-size:13px; margin-bottom:22px; }}
  blockquote {{ background:#f8fafc; border-left:3px solid var(--line); margin:8px 0;
               padding:6px 12px; color:var(--sub); font-size:13px; }}
  .up {{ color:var(--up); font-weight:600; }}
  .down {{ color:var(--down); font-weight:600; }}
</style></head>
<body><div class="wrap">
  <h1>价值投资选股系统 · 综合报告</h1>
  <div class="meta">截面日 {ASOF} ｜ 基准对照：沪深300（价格回报）&amp; 等权全市场 ｜
  数据口径：point-in-time 财务 + 干净行情</div>

  <div class="nav">
    <a href="#sec-a">A · top3</a>
    <a href="#sec-b">B · top5</a>
    <a href="#sec-c">C · 默认(30只)</a>
  </div>

  <div class="toolbar">
    <button type="button" onclick="document.querySelectorAll('details').forEach(function(d){{d.open=true}})">全部展开</button>
    <button type="button" onclick="document.querySelectorAll('details').forEach(function(d){{d.open=false}})">全部折叠</button>
    <span>共 {n_fold} 个折叠区块（策略整块 / 每张内容卡片）—— 点任意标题展开，再点一下即收回。</span>
  </div>

  <div class="card"><h2>① 五策略对比总览（3 策略 vs 2 基准）</h2>
    <p class="note">年化 / 波动 / 最大回撤为全期口径（起点 2017-05-15，已剔除期初空仓期）。
    基准行不填超额 / 信息比率 / 胜率——这三列衡量的是「相对沪深300」，对基准自身无意义；
    沪深300 为价格回报口径（不含股息再投），系统性低估真实全收益约 2~3%/年。</p>
    {ov_html}</div>

  <div class="card"><h2>② 净值曲线</h2>
    {f'<img class="curve" src="{nav_img}">' if nav_img else '<p class="note">nav-curve.png 缺失</p>'}
  </div>

  <div class="warn">③ 排雷「复核判断」分布（L3 一票否决仍生效，本列仅供人工复核参考，不改变结果）：{vet_summary or '—'}<br>
  已知数据缺口：排雷规则 <code>goodwill_to_equity</code>（商誉/净资产）因数据源 0 覆盖而失效，该防线形同虚设，请勿依赖其排除风险。</div>

  {block_a}
  {block_b}
  {block_c}

  <p class="note">报告由 tests/gen_html_report.py 自动汇总自 out/ 下本轮生成的 markdown 与 nav-curve.png。
  所有数字均来自系统计算产物，未做手抄转写。默认档=30只（配置 [20,30]），以无后缀文件名承载。</p>
</div></body></html>"""

(OUT / f"report-{ASOF}.html").write_text(html, encoding="utf-8")
print(f"→ out/report-{ASOF}.html  ({len(html)/1024:.0f} KB, nav-embedded={bool(nav_img)})")
