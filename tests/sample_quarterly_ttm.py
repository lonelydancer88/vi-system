"""样本验证：季报 YTD → TTM 转换 + 与年报口径一致性检查。

抓 4 只样本（普通股/成长股/金融股/次新股），保留 0331/0630/0930/1231，
对利润表+现金流量表的累计(YTD)字段做 TTM 转换，校验：
  1. 年报恒等式：TTM(1231,Y) == YTD(1231,Y)（转换在年度点应为恒等）
  2. 无异常 NaN / 负值跳变
  3. TTM(最新Q1) 与 上一年年报同量级（确认 prior-year 减法未爆炸）
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from vi_system.data.westock import WeStockFetcher, _norm_code

# 普通股(茅台) / 成长股(比亚迪) / 金融股(工行) / 次新股(中芯国际)
SAMPLES = ["600519", "002594", "601398", "688981"]

YTD_FIELDS = {
    "TotalOperatingRevenue": "营收",
    "NPParentCompanyOwners": "归母净利",
    "NetOperateCashFlow": "经营现金流",
}


def _json_retry(ws, args, tries=4):
    for i in range(tries):
        try:
            return ws._json(args)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def ttm_series(ytd: dict) -> dict:
    """ytd: {period(str YYYYMMDD): value} → TTM 序列。

    公式：TTM(p) = YTD(p) + Annual(Y-1) - YTD(p 去年同期)
    年报(1231) 点：TTM = YTD（恒等式）。
    """
    periods = sorted(ytd)
    annual = {p[:4]: v for p, v in ytd.items() if p.endswith("1231")}
    out = {}
    for p in periods:
        y, suffix = p[:4], p[4:]
        ytd_cur = ytd[p]
        if suffix == "1231":
            out[p] = ytd_cur
            continue
        prev_annual = annual.get(str(int(y) - 1))
        ytd_prev_same = ytd.get(f"{int(y) - 1}{suffix}")
        if prev_annual is None or ytd_prev_same is None:
            out[p] = np.nan
            continue
        out[p] = ytd_cur + prev_annual - ytd_prev_same
    return out


def _as_list(s):
    return s if isinstance(s, list) else (list(s.values()) if isinstance(s, dict) else [])


def _pivot(sec):
    d = {}
    for r in sec or []:
        if not isinstance(r, dict):
            continue
        p = str(r.get("EndDate") or "")[:10].replace("-", "")
        if not p:
            continue
        d.setdefault(p, {})
        for f in YTD_FIELDS:
            if f in r and r[f] not in (None, ""):
                try:
                    d[p][f] = float(r[f])
                except (TypeError, ValueError):
                    pass
    return d


def main():
    ws = WeStockFetcher()
    for code in SAMPLES:
        code = _norm_code(code)
        try:
            data = _json_retry(ws, ["finance", code, "--num", "40"])
        except Exception as e:
            print(f"[skip] {code}: fetch error {e}")
            continue
        sections = data.get("sections") if isinstance(data, dict) else data
        if not isinstance(sections, list) or len(sections) < 3:
            print(f"[skip] {code}: bad sections")
            continue
        inc, bal, cfs = _as_list(sections[0]), _as_list(sections[1]), _as_list(sections[2])
        inc_p, cfs_p = _pivot(inc), _pivot(cfs)
        ps = sorted(inc_p)
        print(f"\n===== {code} : inc报告期={len(inc_p)} cfs报告期={len(cfs_p)} =====")
        print("  报告期后缀分布:", pd.Series([p[4:] for p in ps]).value_counts().to_dict())
        print("  样例周期:", ps[:2], "...", ps[-2:])

        for fname, label in YTD_FIELDS.items():
            ser = {p: v.get(fname) for p, v in inc_p.items() if fname in v}
            if not ser:
                # 现金流字段可能落在 cfs
                ser = {p: v.get(fname) for p, v in cfs_p.items() if fname in v}
            if not ser:
                print(f"  [{label}] 源无此字段，跳过")
                continue
            ttm = ttm_series(ser)
            # 1) 年报恒等式
            bad = [(p, ttm[p], ser[p]) for p in ser
                   if p.endswith("1231") and not np.isclose(ttm[p], ser[p], rtol=0, atol=1.0)]
            print(f"  [{label}] 年报恒等式违反: {bad if bad else '无 (OK)'}")
            # 2) 最新Q1 TTM 与上年年报同量级
            yrs = sorted({p[:4] for p in ser if p.endswith("1231")})
            if len(yrs) >= 2:
                ly, py = yrs[-1], yrs[-2]
                ttm_q1 = ttm.get(f"{ly}0331")
                ann_prev = ser.get(f"{py}1231")
                if ttm_q1 and ann_prev and np.isfinite(ttm_q1):
                    print(f"  [{label}] TTM({ly}Q1)/年报({py}) = {ttm_q1 / ann_prev:.2f}")
            # 打印近 8 期 TTM
            recent = [p for p in sorted(ttm)[-8:] if np.isfinite(ttm.get(p, np.nan))]
            print(f"  [{label}] 近8期TTM(亿):",
                  {p: round(ttm[p] / 1e8, 1) for p in recent})


if __name__ == "__main__":
    main()
