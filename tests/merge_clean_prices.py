"""把新浪/腾讯两源干净行情合并回 prices.parquet，并做双源交叉校验。

背景
----
westock 落库的 close_raw/close_adj 大面积损坏（负价、复权因子锚点错）。
本脚本用两个**互相独立**的干净源重建：
  - 新浪 `prices_sina_clean.parquet`（506 只）：raw + hfq因子
  - 腾讯 `prices_gtimg_clean.parquet`（635 只）：raw + hfq
两源并集 = 802/802 全覆盖。

取数原则
--------
**按票取舍、不按日期混** —— 单只票的 close_raw/close_adj 必须来自同一源，
否则复权因子与价格序列跨源错配，含分红收益会失真。优先新浪（主源），缺则腾讯。
跨股票混源不影响回测：收益率是每只票自身序列算的，只在票内比较。

输出
----
- prices.parquet：close_raw/close_adj 被干净值整体替换，mktcap 重算
- 完全无干净覆盖的票 → 隔离到 prices_excluded_final.parquet 并从宇宙移除
- 打印负价校验、中国动力 sanity、以及 339 只重叠票的双源收益比对

用法
----
  python tests/merge_clean_prices.py            # 试跑（只打印，不写盘）
  python tests/merge_clean_prices.py --apply    # 实际写回 prices
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

ROOT = "data/real_universe"
PRICES = os.path.join(ROOT, "prices.parquet")
SINA = os.path.join(ROOT, "prices_sina_clean.parquet")
GTIMG = os.path.join(ROOT, "prices_gtimg_clean.parquet")
EXCLUDED = os.path.join(ROOT, "prices_excluded_final.parquet")
BAK = os.path.join(ROOT, "prices.parquet.bak_preclean")

WIN_BEG = "2017-05-15"   # 回测窗口起点
WIN_END = "2026-05-15"
CMP_BEG = "2019-01-01"   # 双源比对窗口（避开早期上市/停牌差异）
FLOOR = "2014-01-01"     # 数据起点；更早的行是脏 westock 遗留，且不在任何回看窗口内


def _load(path: str, tag: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close_raw", "close_adj"])
    df = df[(df.close_raw > 0) & (df.close_adj > 0)]
    print(f"[load] {tag}: {len(df)} 行 / {df.code.nunique()} 只")
    return df[["code", "date", "close_raw", "close_adj"]]


def gtimg_gate(gt_only: pd.DataFrame) -> pd.DataFrame:
    """对只能用腾讯的票做复权体检，不过关的退化为保守口径。

    实测：腾讯 close_adj 约 16.6% 出现"含分红 < 价格"（物理上不可能，锚点错），
    与 westock 同源病灶；而价格收益两源完全一致，故只怀疑它的复权。
    判据（不依赖任何第三方）：gap = 含分红收益 − 价格收益
      - gap < 0           → 错误（分红不可能让收益变少）
      - gap > 150pp       → 错误（远超 2019→2026 合理分红贡献）
    不过关则令 close_adj = close_raw：含分红≈价格收益，回测退化为裸价收益
    （诚实保守，本项目既定约定，见 2026-09-10 日志）。
    """
    B, E = pd.Timestamp(CMP_BEG), pd.Timestamp(WIN_END)
    bad: list[str] = []
    ok: list[str] = []
    for c, gg in gt_only.groupby("code"):
        gg = gg[(gg.date >= B) & (gg.date <= E)].sort_values("date")
        if len(gg) < 100:
            continue
        pr = gg.close_raw.iloc[-1] / gg.close_raw.iloc[0] - 1
        tr = gg.close_adj.iloc[-1] / gg.close_adj.iloc[0] - 1
        gap = tr - pr
        (ok if -0.005 <= gap <= 1.5 else bad).append(c)
    bad = set(bad)
    print(f"[体检] 腾讯复权：通过 {len(ok)} 只，未通过 {len(bad)} 只"
          f"（判据 0 ≤ 含分红−价格 ≤ 150pp）")
    if not bad:
        return gt_only
    out = gt_only.copy()
    m = out.code.isin(bad)
    out.loc[m, "close_adj"] = out.loc[m, "close_raw"]   # 退化为保守口径
    print(f"[体检] 未通过票据已退化为 close_adj=close_raw（含分红=价格，不再倒置）")
    return out


def build_combined() -> pd.DataFrame:
    """按票取舍：优先新浪，缺则腾讯（先过体检）。返 (code,date,close_raw,close_adj,src)。"""
    sn = _load(SINA, "新浪")
    gt = _load(GTIMG, "腾讯")
    sn["src"] = "sina"
    gt["src"] = "gtimg"
    sina_codes = set(sn.code.unique())
    # 新浪未覆盖的票才用腾讯补齐（单票同源），且先过复权体检
    gt_only = gtimg_gate(gt[~gt.code.isin(sina_codes)])
    comb = pd.concat([sn, gt_only], ignore_index=True)
    dup = comb.duplicated(["code", "date"]).sum()
    if dup:
        comb = comb.drop_duplicates(["code", "date"], keep="first")
    print(f"[combine] 合并后 {len(comb)} 行 / {comb.code.nunique()} 只"
          f"（新浪 {len(sina_codes)} + 腾讯补 {gt_only.code.nunique()}，重复行 {dup}）")
    return comb


def cross_check(sn: pd.DataFrame, gt: pd.DataFrame) -> None:
    """对重叠票做双源比对：同窗口内 close_adj 总收益应一致（锚点不同但收益率同）。"""
    S, G = set(sn.code.unique()), set(gt.code.unique())
    overlap = S & G
    print(f"\n[交叉校验] 两源重叠 {len(overlap)} 只，比对窗口 {CMP_BEG}~{WIN_END}")
    diffs = []
    for c in sorted(overlap):
        a = sn[(sn.code == c) & (sn.date >= CMP_BEG) & (sn.date <= WIN_END)].sort_values("date")
        b = gt[(gt.code == c) & (gt.date >= CMP_BEG) & (gt.date <= WIN_END)].sort_values("date")
        m = a.merge(b, on="date", suffixes=("_s", "_g"))
        if len(m) < 100:
            continue
        rs = m.close_adj_s.iloc[-1] / m.close_adj_s.iloc[0] - 1
        rg = m.close_adj_g.iloc[-1] / m.close_adj_g.iloc[0] - 1
        diffs.append(abs(rs - rg))
    if not diffs:
        print("  无可比对样本")
        return
    d = pd.Series(diffs)
    print(f"  样本 {len(d)} 只：双源收益差 中位数={d.median()*100:.3f}pp "
          f"均值={d.mean()*100:.3f}pp 最大={d.max()*100:.2f}pp")
    print(f"  差<1pp 的占比 {(d < 0.01).mean()*100:.1f}%，差<5pp 占 {(d < 0.05).mean()*100:.1f}%")
    print("  （两源独立抓取，收益差在 1pp 内即说明数据可信）")


def main(apply: bool = False):
    sn = _load(SINA, "新浪")
    gt = _load(GTIMG, "腾讯")
    comb = build_combined()

    px = pd.read_parquet(PRICES)
    px["date"] = pd.to_datetime(px["date"])
    n0 = len(px)
    print(f"\n[prices] 当前 {len(px)} 行 / {px.code.nunique()} 只")

    # 完全无干净覆盖的票 → 隔离
    clean_codes = set(comb.code.unique())
    failed = set(px.code.unique()) - clean_codes
    if failed:
        exc = px[px.code.isin(failed)]
        if apply:
            exc.to_parquet(EXCLUDED, index=False)
        print(f"[isolate] 无干净覆盖 {len(failed)} 只 → {'已隔离' if apply else '待隔离'}: "
              f"{sorted(failed)[:10]}")
    else:
        print("[isolate] 全部票均有干净覆盖")

    merged = px.merge(comb[["code", "date", "close_raw", "close_adj"]],
                      on=["code", "date"], how="left", suffixes=("", "_new"))
    if failed:
        merged = merged[~merged.code.isin(failed)].copy()
    cov = merged.close_raw_new.notna()
    print(f"[merge] 对齐覆盖 {int(cov.sum())}/{len(merged)} 行 "
          f"（未覆盖 {int((~cov).sum())} 行保留原值）")
    merged.loc[cov, "close_raw"] = merged.loc[cov, "close_raw_new"]
    merged.loc[cov, "close_adj"] = merged.loc[cov, "close_adj_new"]
    merged = merged.drop(columns=["close_raw_new", "close_adj_new"])

    ts = pd.to_numeric(merged["total_share"], errors="coerce")
    merged["mktcap"] = merged["close_raw"] * ts

    # 2014 年前的行：干净源不覆盖（FLOOR 限制），保留的仍是脏 westock 值，
    # 且回测(2017-05起)与其回看窗口都不涉及 → 直接丢弃，换取全表零负价。
    old = int((merged.date < FLOOR).sum())
    if old:
        merged = merged[merged.date >= FLOOR].copy()
        print(f"[clean] 丢弃 {FLOOR} 之前 {old} 行（脏 westock 遗留，回测与回看窗口均不涉及）")

    # ---- 校验 ----
    neg = int(((merged.close_raw <= 0) | (merged.close_adj <= 0)).sum())
    win = merged[merged.date >= WIN_BEG]
    win_neg = int(((win.close_raw <= 0) | (win.close_adj <= 0)).sum())
    print(f"\n[validate] 全表负价行 {neg}（应 0）｜回测窗口内负价行 {win_neg}（应 0）")

    g = merged[merged.code == "sh600482"].sort_values("date")
    if not g.empty:
        b = g[g.date <= WIN_BEG].iloc[-1]
        s = g[g.date <= WIN_END].iloc[-1]
        pr = s.close_raw / b.close_raw - 1
        inc = s.close_adj / b.close_adj - 1
        print(f"[sanity] 中国动力 价格={pr*100:+.1f}% 含分红={inc*100:+.1f}% "
              f"→ {'✅ 含分红>价格' if inc > pr else '❌ 仍倒置'}")

    # 全市场含分红<价格的票占比（应极少，仅真·无分红且数据无误差时相等）
    inv = 0
    tot = 0
    for c, gg in merged[merged.date >= WIN_BEG].groupby("code"):
        gg = gg.sort_values("date")
        if len(gg) < 50:
            continue
        b, s = gg.iloc[0], gg.iloc[-1]
        tot += 1
        if s.close_adj / b.close_adj < s.close_raw / b.close_raw - 1e-9:
            inv += 1
    print(f"[sanity] 回测窗口内 含分红<价格 的票 {inv}/{tot} "
          f"（{inv/max(tot,1)*100:.1f}%，应接近 0）")

    cross_check(sn[sn.code.isin(set(sn.code) & set(gt.code))], gt)

    if not apply:
        print("\n[dry-run] 未写盘。确认无误后加 --apply")
        return
    if not os.path.exists(BAK):
        shutil.copy2(PRICES, BAK)
        print(f"\n[bak] {PRICES} -> {BAK}")
    merged.to_parquet(PRICES, index=False)
    print(f"[done] 写回 {PRICES}：{n0}→{len(merged)} 行，{merged.code.nunique()} 只")

    uni = os.path.join(ROOT, "universe.parquet")
    if os.path.exists(uni) and failed:
        u = pd.read_parquet(uni)
        u = u[~u.code.isin(failed)].copy()
        u.to_parquet(uni, index=False)
        print(f"[universe] → {len(u)} 行（移除隔离票）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写回 prices（默认只试跑）")
    main(ap.parse_args().apply)
