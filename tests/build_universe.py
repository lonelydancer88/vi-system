"""构建真实大宇宙代码表。

来源：沪深300(sh000300) + 中证500(sz399905) 成份股（去重），
再注入历史造假/退市案例（实测腾讯接口仍有其财务数据，足以做 point-in-time 排雷）。

产物：
  data/real_universe/universe.parquet   # 宇宙表（code/name/industry/list_date/delimit/is_financial）
  data/real_universe/codes.json         # {all:[...], hs300:[...], zz500:[...], fraud:[...]}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/hpl/WorkBuddy/2026-09-07-15-54-23")
from vi_system.data.westock import BIN, _norm_code

ROOT = Path("/Users/hpl/WorkBuddy/2026-09-07-15-54-23/data/real_universe")
ROOT.mkdir(parents=True, exist_ok=True)

# 沪深300 + 中证500 指数代码（已验证 constituent 可枚举）
INDEX_CODES = {"hs300": "sh000300", "zz500": "sz399905"}


def index_constituents(index_code: str) -> list[str]:
    import subprocess
    out = subprocess.run([BIN, "index", "constituent", index_code, "--raw"],
                         capture_output=True, text=True, timeout=120).stdout
    try:
        arr = json.loads(out)
    except Exception:
        print(f"  [warn] {index_code} 解析失败: {out[:120]}")
        return []
    codes = []
    for r in arr:
        c = r.get("code")
        if c:
            codes.append(_norm_code(c))
    return codes


def batch_profile(codes: list[str], chunk=40) -> dict:
    """批量取 name/industry/listedDate。返回 {norm_code: {name,industry,listedDate}}。"""
    import subprocess
    out = {}
    for i in range(0, len(codes), chunk):
        seg = codes[i:i + chunk]
        raw = ",".join(seg)
        try:
            r = subprocess.run([BIN, "profile", raw, "--raw"],
                               capture_output=True, text=True, timeout=180).stdout
            obj = json.loads(r)
        except Exception as e:
            print(f"  [warn] profile 批 {i} 失败: {e}")
            continue
        # 批量返回结构：{"success":true,"data":[{symbol, data:{...}}]}
        data = obj.get("data") if isinstance(obj, dict) else None
        if isinstance(data, list):
            for item in data:
                sym = item.get("symbol")
                inner = item.get("data") or {}
                if sym:
                    out[_norm_code(sym)] = {
                        "name": inner.get("name"),
                        "industry": (inner.get("industry") or "未知").strip(),
                        "listedDate": inner.get("listedDate"),
                    }
        time.sleep(0.3)
    return out


# 历史造假 / 退市案例（代码，名称，行业，上市日，退市日；退市日 None 表示仍上市）
FRAUD = [
    ("600518.SH", "康美药业", "医药生物", "2001-03-19", None),       # 300亿造假，2019爆雷，重整后仍上市
    ("002450.SZ", "康得新",   "化工",     "2010-07-16", "2021-04-30"), # 119亿造假，2019爆雷
    ("300104.SZ", "乐视网",   "传媒",     "2010-08-12", "2020-05-21"), # 资金链断裂，2020退市
    ("002069.SZ", "獐子岛",   "农林牧渔", "2006-09-28", None),         # 扇贝跑路，财务操纵
    ("002680.SZ", "长生生物", "医药生物", "2012-06-05", "2019-11-27"), # 疫苗造假，2018爆雷
    ("600086.SH", "东方金钰", "轻工制造", "1997-06-06", "2021-03-16"), # 财务造假，退市
    ("600781.SH", "辅仁药业", "医药生物", "1996-12-18", "2023-05-22"), # 资金占用，退市
]


def main():
    print("[1/3] 枚举指数成份股 ...")
    idx_codes: dict[str, list[str]] = {}
    all_idx: list[str] = []
    for name, code in INDEX_CODES.items():
        cs = index_constituents(code)
        idx_codes[name] = cs
        all_idx += cs
        print(f"  {name}({code}): {len(cs)} 只")
    all_idx = sorted(set(all_idx))
    print(f"  去重后指数成份股: {len(all_idx)} 只")

    print("[2/3] 批量取简况（行业/上市日）...")
    profile = batch_profile(all_idx)
    print(f"  成功获取简况: {len(profile)} / {len(all_idx)}")

    rows = []
    for c in all_idx:
        p = profile.get(c, {})
        rows.append({
            "code": c,
            "name": p.get("name") or c,
            "industry": p.get("industry") or "未知",
            "list_date": pd.to_datetime(p.get("listedDate"), errors="coerce"),
            "delist_date": pd.NaT,
            "is_financial": (p.get("industry") or "") in ("银行", "保险", "证券", "多元金融"),
            "source": "index",
        })

    print("[3/3] 注入历史造假/退市案例 ...")
    fraud_codes = []
    for c, name, ind, ld, dd in FRAUD:
        c = _norm_code(c)  # 统一为 sh/sz 前缀格式，与 facts 一致
        fraud_codes.append(c)
        rows.append({
            "code": c, "name": name, "industry": ind,
            "list_date": pd.to_datetime(ld, errors="coerce"),
            "delist_date": pd.to_datetime(dd, errors="coerce") if dd else pd.NaT,
            "is_financial": ind in ("银行", "保险", "证券", "多元金融"),
            "source": "fraud_injected",
        })

    df = pd.DataFrame(rows)
    # 去重（以防指数重叠），保留 fraud 标记
    df = df.drop_duplicates(subset=["code"], keep="first")
    df.to_parquet(ROOT / "universe.parquet", index=False)
    print(f"  宇宙写入: {len(df)} 只  -> {ROOT / 'universe.parquet'}")

    codes_json = {
        "all": df["code"].tolist(),
        "hs300": idx_codes["hs300"],
        "zz500": idx_codes["zz500"],
        "fraud": fraud_codes,
    }
    (ROOT / "codes.json").write_text(json.dumps(codes_json, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"  codes.json 写入: {len(codes_json['all'])} 只全部 / 造假 {len(fraud_codes)} 只")
    print("完成。")


if __name__ == "__main__":
    main()
