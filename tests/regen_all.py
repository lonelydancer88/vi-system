#!/usr/bin/env python3
"""一键重生成所有报告：数据或策略改动后跑这一个脚本就够。

为什么需要它
------------
`out/` 下的产物由 8 个不同的入口生成，各自的截面日（asof）默认值互不相同
（有硬编码 2026-09-09/2026-05-15 的、有默认 2026-09-10 的、有从库里推的）。
分头手跑必然出现「有的报告停在昨天、有的停在今天」的口径漂移，以及漏生成。
本脚本统一从 `prices.parquet` 最大交易日推导 ASOF 并显式传给每一步。

用法
----
    python3 tests/regen_all.py                    # ASOF = 库内最新交易日
    python3 tests/regen_all.py --asof 2026-09-07  # 指定截面日
    python3 tests/regen_all.py --dry-run          # 只打印将要执行的命令
    python3 tests/regen_all.py --keep-going       # 某步失败时继续后续步骤

生成清单
--------
    0  预热 screen 缓存   data/real_universe/.screen_cache/
    1  回测各档           out/backtest.md、backtest-top{3,5}.md、
                          out/backtest-hs300.md、backtest-hs300-top{3,5}.md
                          （默认档 = 30只，用「无后缀」文件名承载，不再有 top30）
    2  回测报告           out/回测报告-{asof}.md（默认/30只）、
                          out/回测报告-top{3,5}-{asof}.md
                          （逐期「调仓动作」表已内嵌价格/收益/手数/占用列，
                          即原 trades 台账合并而来，不再单独生成 trades*.md）
    3  当前截面           out/vetoes-{asof}.md（已含「复核判断」列）、
                          out/valuation-{asof}.md、out/portfolio-{asof}.md
    4  持仓 / 集中组合     out/持仓建议-{asof}.md、out/portfolio-top{3,5}-{asof}.md
    5  净值曲线           out/nav-curve.png
    6  HTML 汇总          out/report-{asof}.html

有意排除的产物
--------------
    backtest-split.md   样本内/样本外验证，研究型产物（与主回测区间重叠、只反映
                        默认持仓口径）。需要时手动跑：cli backtest --split
    建仓清单-*.md       与「持仓建议」同组合，仅多实时行情价/涨跌列，且价格来自
                        qt.gtimg.cn（非 point-in-time、不可复现）。手动跑 gen_position_list.py
    trades*.md          其价格/收益/手数/占用列已并入「回测报告」的调仓动作表，
                        二者底层同源（trade_ledger），不再单独生成。
    reasons*.md         与 trades 台账底层同函数、信息零差。
    annotate_vetoes.py  cli screen 的 _veto_md 已原生输出「复核判断」列，该脚本冗余。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = "data/real_universe"
OUT = "out"
PY = sys.executable or "python3"

TIERS = (None, 3, 5)


def _asof_from_db(db: str) -> str:
    """ASOF = 库内最新交易日。所有报告统一以它为准，避免四套日期口径漂移。"""
    import pandas as pd

    p = pd.read_parquet(ROOT / db / "prices.parquet", columns=["date"])
    return str(pd.to_datetime(p["date"]).max().date())


def _tier_args(mh: int | None) -> list[str]:
    return [] if mh is None else ["--max-holdings", str(mh)]


def _tier_name(mh: int | None) -> str:
    return "默认" if mh is None else f"top{mh}"


def _p(*a, **kw):
    """带 flush 的 print —— 步骤 0 预热要跑 5 分钟，不 flush 会让日志长时间空白。"""
    print(*a, **kw, flush=True)


def run(cmd: list[str], label: str, results: list, env: dict,
        dry: bool, keep_going: bool) -> bool:
    _p(f"\n{'─' * 72}\n▶ {label}\n  $ {' '.join(str(c) for c in cmd)}")
    if dry:
        results.append((label, "DRY", 0.0))
        return True

    t0 = time.time()
    # stdin=DEVNULL 必给：本脚本常以脱离会话的方式后台运行，此时父进程的 fd 0
    # 可能是失效的描述符；一旦被继承，子解释器启动即报
    # "Fatal Python error: init_sys_streams ... Bad file descriptor"。
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    dt = time.time() - t0
    for ln in (r.stdout or "").strip().splitlines()[-10:]:
        _p("    " + ln)
    for ln in (r.stderr or "").strip().splitlines()[-10:]:
        _p("  ! " + ln)

    ok = r.returncode == 0
    _p(f"  {'✓' if ok else '✗'} {label}（{dt:.1f}s，exit={r.returncode}）")
    results.append((label, "OK" if ok else f"FAIL({r.returncode})", dt))
    if not ok and not keep_going:
        _p("\n[abort] 上一步失败，已中止。加 --keep-going 可跳过失败继续跑完。")
        sys.exit(1)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="一键重生成所有报告")
    ap.add_argument("--db", default=DB, help=f"数据目录（默认 {DB}）")
    ap.add_argument("--asof", default=None,
                    help="截面日 YYYY-MM-DD（默认取库内最新交易日）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--keep-going", action="store_true", help="某步失败时继续后续步骤")
    ap.add_argument("--skip-warm", action="store_true",
                    help="跳过步骤 0（缓存已新鲜时省 ~7 分钟；缓存不新鲜会退化为逐日重算）")
    args = ap.parse_args()

    db = args.db
    asof = args.asof or _asof_from_db(db)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cli = [PY, "-m", "vi_system.cli", "--db", db]
    res: list = []

    print(f"仓库 : {ROOT}", flush=True)
    print(f"数据 : {db}", flush=True)
    print(f"截面 : {asof}" + ("" if args.asof else "（推导自 prices.parquet 最新交易日）"), flush=True)
    if args.dry_run:
        print("模式 : dry-run（不实际执行）", flush=True)

    # ---- 0 预热 screen 缓存：把回测全期调仓日的截面一次算好落盘，
    #         否则 backtest 串行逐日重算会超时（见 screen_cache.py）
    if args.skip_warm:
        _p(f"\n{'─' * 72}\n▶ 0 预热 screen 缓存  —— 已跳过（--skip-warm）")
        res.append(("0 预热 screen 缓存", "SKIP", 0.0))
    else:
        run([PY, "tests/warm_screen_cache.py", "--db", db, "--asof", asof],
            "0 预热 screen 缓存", res, env, args.dry_run, args.keep_going)

    # ---- 1 回测各档（默认 / 3只 / 5只，各带沪深300基准对比；默认档即 30只、无后缀）
    for mh in TIERS:
        nm = _tier_name(mh)
        run(cli + ["backtest"] + _tier_args(mh) + ["--benchmark", "sh000300",
                                                   "--out", OUT],
            f"1 回测·{nm}", res, env, args.dry_run, args.keep_going)

    # ---- 2 回测报告（逐期调仓动作 / 持仓 / 买卖原因）
    # 默认档用 --holdings 0（= 配置 [20,30]，文件名无后缀）；3/5 档用 --holdings N
    for mh in (0, 3, 5):
        nm = "默认(30只)" if mh == 0 else f"top{mh}"
        run([PY, "tests/gen_backtest_report.py", "--db", db, "--holdings", str(mh)],
            f"2 回测报告·{nm}", res, env, args.dry_run, args.keep_going)

    # ---- 3 当前截面：vetoes（含「复核判断」列）/ valuation / portfolio —— HTML 依赖这三件
    run(cli + ["screen", "--asof", asof, "--top", "30", "--out", OUT],
        "3 当前截面 screen", res, env, args.dry_run, args.keep_going)

    # ---- 4 持仓建议（默认档）+ 集中组合（3只 / 5只）
    run([PY, "tests/gen_position_advice.py", "--asof", asof],
        "4 持仓建议", res, env, args.dry_run, args.keep_going)
    run([PY, "tests/gen_concentrated.py", "--db", db, "--asof", asof, "--n", "3,5"],
        "4 集中组合 top3/5", res, env, args.dry_run, args.keep_going)

    # ---- 5 净值曲线（只出 nav-curve.png，不含历史遗留的手数表）
    run([PY, "tests/regen_charts.py"], "5 净值曲线", res, env,
        args.dry_run, args.keep_going)

    # ---- 6 HTML 汇总（读 backtest-hs300.md + portfolio/valuation/vetoes-{asof}.md）
    run([PY, "tests/gen_html_report.py", asof], "7 HTML 汇总", res,
        env, args.dry_run, args.keep_going)

    # ---- 汇总
    _p(f"\n{'=' * 72}\n汇总")
    bad = [x for x in res if x[1] not in ("OK", "DRY", "SKIP")]
    for label, status, dt in res:
        _p(f"  {status:10s} {dt:7.1f}s  {label}")
    total = sum(x[2] for x in res)
    _p(f"\n共 {len(res)} 步，用时 {total:.1f}s，失败 {len(bad)} 步"
       + ("" if not bad else "：" + "、".join(x[0] for x in bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
