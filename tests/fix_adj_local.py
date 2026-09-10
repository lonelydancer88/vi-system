"""
fix_adj_local.py — 本地重建 close_adj（不依赖网络）

根因：westock 落库的 close_adj（腾讯 hfq）每天抖 1%~3%、且存在缓慢虚假漂移
（ratio=close_adj/close_raw 本应是「仅除权日跳变」的阶梯常数，实测却是
负一阶自相关的均值回复 + 缓慢漂移）。这导致：
  - 持仓建议里「含分红」列比「价格收益」还小（如中国动力 +75.7% < +80.5%）
  - 回测 NAV 每期被注入噪声

修复（两段式，纯本地，无需网络）：
  - 金标准票（prices_adj_factor.parquet 里 153 只，gtimg 同源自抓 qfq/raw）：
      clean_factor = K * gtimg_factor,  K = median(ratio / gtimg_factor)
      这样把干净的阶梯因子映射回 westock 量纲，含分红与真实分红一致。
  - 其余票：没有金标准，用全期中位数作常数因子（漂移/抖动一起消除），
      含分红 ≈ 价格收益（不再倒置），回测退化为裸价收益（诚实保守）。

锚点常数在收益率中抵消，不影响回测收益；只是消除了每日噪声与虚假漂移。

用法：
  python tests/fix_adj_local.py            # dry-run 校验，打印 before/after
  python tests/fix_adj_local.py --apply   # 覆盖 prices.parquet 的 close_adj
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

DB = "data/real_universe"
PRICES = os.path.join(DB, "prices.parquet")
GOLD = os.path.join(DB, "prices_adj_factor.parquet")  # 153 只金标准 gtimg 因子


def build_clean(px: pd.DataFrame, gold: pd.DataFrame | None):
    px = px.sort_values(["code", "date"]).copy()
    px["ratio"] = px["close_adj"] / px["close_raw"]

    has_gold = gold is not None and len(gold) > 0
    if has_gold:
        gold = gold.copy()
        gold["date"] = pd.to_datetime(gold["date"])
        gold = gold.sort_values(["code", "date"])
        gfac = gold.rename(columns={"factor": "gfactor"})
        px = px.merge(gfac[["code", "date", "gfactor"]], on=["code", "date"], how="left")
        px["gfactor"] = px.groupby("code")["gfactor"].transform(lambda s: s.ffill().bfill())
        # K = median(ratio / gfactor) 每票一个常数，把金标准因子映射回 westock 量纲
        px["K"] = px.groupby("code").apply(
            lambda g: pd.Series(np.nanmedian((g["ratio"] / g["gfactor"]).values)
                                if g["gfactor"].notna().any() else np.nan, index=g.index)
        ).reset_index(level=0, drop=True)
        gold_mask = px["gfactor"].notna() & px["K"].notna()
    else:
        gold_mask = pd.Series(False, index=px.index)

    # 金标准票：clean_factor = K * gfactor
    px["clean_factor"] = np.where(gold_mask, px["K"] * px["gfactor"], np.nan)

    # 非金标准票：全期中位数作常数因子
    med = px[~gold_mask].groupby("code")["ratio"].transform("median")
    px.loc[~gold_mask, "clean_factor"] = med

    # 任何残留 NaN 用全期中位数兜底
    px["clean_factor"] = px.groupby("code")["clean_factor"].transform(lambda s: s.ffill().bfill())
    px["clean_factor"] = px["clean_factor"].fillna(px["ratio"])
    px["close_adj_new"] = px["close_raw"] * px["clean_factor"]
    return px


def daily_jump_rate(series: pd.Series) -> float:
    r = series.dropna()
    j = (r / r.shift(1) - 1).abs().dropna()
    return float((j > 0.01).mean() * 100)


def rebal_dates():
    ds = [pd.Timestamp(y, m, 15) for y in range(2017, 2027) for m in (5, 9, 11)]
    return [d for d in ds if pd.Timestamp("2017-05-15") <= d <= pd.Timestamp("2026-09-15")]


def validate(px: pd.DataFrame):
    ds = rebal_dates()
    print("=== 校验（2017-05 之后回测区间）===")
    before_j, after_j = [], []
    for code, g in px.groupby("code"):
        g = g[g.date >= "2017-05-15"]
        if len(g) < 60:
            continue
        before_j.append(daily_jump_rate(g["close_adj"] / g["close_raw"]))
        after_j.append(daily_jump_rate(g["close_adj_new"] / g["close_raw"]))
    before_j = np.array(before_j); after_j = np.array(after_j)
    print(f"样本票: {len(before_j)} 只")
    print(f"close_adj/close_raw 日跳变>1% 占比: 修复前 {before_j.mean():.2f}% -> 修复后 {after_j.mean():.2f}%")
    print(f"中国动力: 修复前 {daily_jump_rate(px[(px.code=='sh600482')&(px.date>='2017-05-15')]['close_adj']/px[(px.code=='sh600482')&(px.date>='2017-05-15')]['close_raw']):.2f}%"
          f" -> 修复后 {daily_jump_rate(px[(px.code=='sh600482')&(px.date>='2017-05-15')]['close_adj_new']/px[(px.code=='sh600482')&(px.date>='2017-05-15')]['close_raw']):.2f}%")
    # 中国动力 含分红 vs 价格收益（正确口径 adj vs adj）
    g = px[px.code == "sh600482"].sort_values("date")
    b = g[g.date <= "2025-11-14"].iloc[-1]; s = g[g.date <= "2026-05-15"].iloc[-1]
    pr = s.close_raw / b.close_raw - 1
    inc = s.close_adj_new / b.close_adj_new - 1
    print(f"中国动力 含分红(修复后)={inc*100:+.1f}%  价格收益={pr*100:+.1f}%  -> {'OK(≥价格)' if inc>=pr else '仍倒置!'}")
    return before_j.mean(), after_j.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际覆盖 prices.parquet")
    args = ap.parse_args()

    px = pd.read_parquet(PRICES)
    gold = pd.read_parquet(GOLD) if os.path.exists(GOLD) else None
    print(f"读取 prices.parquet: {px.code.nunique()} 只, {len(px)} 行; 金标准: {gold.code.nunique() if gold is not None else 0} 只")

    px = build_clean(px, gold)
    b, a = validate(px)

    if not args.apply:
        print("\n[dry-run] 未写入。确认无误后加 --apply 执行。")
        return

    px["close_adj_old"] = px["close_adj"]
    px["close_adj"] = px["close_adj_new"]
    keep = [c for c in px.columns if c not in ("ratio", "gfactor", "K", "clean_factor", "close_adj_new")]
    px = px[keep]
    px.to_parquet(PRICES, index=False)
    print(f"\n[applied] 已覆盖 prices.parquet 的 close_adj（原值存于 close_adj_old 列）。")


if __name__ == "__main__":
    main()
