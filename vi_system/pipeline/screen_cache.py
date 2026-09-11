"""screen_at 的持久化磁盘缓存。

为什么需要它（超时根因）：
    回测会对每个调仓日调用 screen_at(asof) 重跑 L1→L5，单次约 0.3~0.7s；
    27 个调仓日 + 30/5/3/equal 四档 + 指数基准 + trades/reasons 多次重跑，
    串行累计 ~40 分钟，必撞本环境「约 20 分钟硬杀长进程」的限制（实测被杀）。
    原 engine 的 _SCREEN_CACHE 只在单次进程内有效，跨进程/次重跑又从零算。

本缓存把每个 asof 的筛选结果落盘到
    <db>/.screen_cache/<key>.{scored,rejected,uni}.parquet
只要数据（facts/prices/universe 的 mtime）与配置（rules.yaml 指纹）不变，
后续任何进程/次都直接读盘，单档回测从 ~10min 降到秒级；
四档合计只需第一次 ~10min 的「真算」，其余三档读盘 ~1min。

缓存身份与 with_valuation 解耦：估值(formula L5)只给 scored 加列、不改变过门
结果与权重，因此统一缓存「带估值」版本。with_valuation=False 的调用方直接复用
同一份缓存（多几列无副作用）——否则两种标志各存一份、且预热只用 True，会导致
run_backtest(默认 False) 全 miss 而整盘重算（regen_charts 卡死的真实根因）。

失效策略：data_version 用三张主表的 mtime 拼 sha1；cfg_fp 用 rules.yaml 内容
的 sha1。任一变化即 key 变化，旧缓存自动失效、重新计算。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

_CACHE_SUB = ".screen_cache"


def _cfg_fingerprint(cfg) -> str:
    # 只取影响 L1→L5 筛选的配置指纹，刻意排除 portfolio 区块——
    # run_tier 会按持仓档位(max_holdings)改写 portfolio.target_size/max_position/
    # max_industry，但这些只影响「组合权重」步骤，不影响筛选本身。若把 portfolio
    # 也算进指纹，每个档位(30/5/3/equal)与每次 --holdings 覆盖都会触发全量重算
    # (~9 分钟)，这是 gen_backtest_report 卡死的真实根因。剔除后各档位共用同一
    # 份筛选缓存。
    try:
        d = cfg.to_dict()
        d.pop("portfolio", None)
        d.pop("backtest", None)  # 调仓月/日/成本只影响回测循环，不影响截面筛选
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
    except Exception:
        blob = repr(cfg)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _data_version(store) -> str:
    h = hashlib.sha1()
    for p in (store.facts_path, store.prices_path, store.universe_path):
        if Path(p).exists():
            h.update(f"{Path(p).name}:{Path(p).stat().st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()[:12]


def _key(store, asof, data_ver, cfg_fp) -> str:
    # 缓存身份不含 with_valuation（估值只加列、不影响过门/权重），
    # 统一缓存「带估值」版本，False 调用方直接复用。
    a = pd.Timestamp(asof).strftime("%Y%m%d")
    return f"{Path(store.root).name}_{a}_{data_ver}_{cfg_fp}"


def _legacy_key(store, asof, with_valuation, data_ver, cfg_fp) -> str:
    # 兼容旧版本：曾把 with_valuation 编入 key（预热只写 True -> 后缀 _1）
    a = pd.Timestamp(asof).strftime("%Y%m%d")
    return (
        f"{Path(store.root).name}_{a}_{int(bool(with_valuation))}"
        f"_{data_ver}_{cfg_fp}"
    )


def get_screen(store, asof, cfg, compute, with_valuation=True):
    """返回 (scored, rejected, uni)。优先读盘，未命中则 compute() 并落盘。

    with_valuation 仅用于 legacy 回退路径兼容旧缓存文件名；新缓存统一按
    「带估值」版本存储，该参数不再影响新 key。
    """
    cache_dir = Path(store.root) / _CACHE_SUB
    data_ver = _data_version(store)
    cfg_fp = _cfg_fingerprint(cfg)
    key = _key(store, asof, data_ver, cfg_fp)
    scored_p = cache_dir / f"{key}.scored.parquet"
    rejected_p = cache_dir / f"{key}.rejected.parquet"
    uni_p = cache_dir / f"{key}.uni.parquet"

    def _read(k):
        sp, rp, up = (
            cache_dir / f"{k}.scored.parquet",
            cache_dir / f"{k}.rejected.parquet",
            cache_dir / f"{k}.uni.parquet",
        )
        if sp.exists() and rp.exists() and up.exists():
            try:
                return pd.read_parquet(sp), pd.read_parquet(rp), pd.read_parquet(up)
            except Exception:
                return None
        return None

    hit = _read(key)
    if hit is None:  # 新 key 未命中 -> 试 legacy（只认带估值的 _1，避免缺列）
        lk = _legacy_key(store, asof, True, data_ver, cfg_fp)
        if lk != key:
            hit = _read(lk)
            if hit is not None:  # 提升到新 key，避免后续重复 legacy 查找
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    sc, rj, un = hit
                    sc.to_parquet(scored_p, index=False)
                    rj.to_parquet(rejected_p, index=False)
                    un.to_parquet(uni_p, index=False)
                except Exception:
                    pass
    if hit is not None:
        return hit

    sc, rj, un = compute()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sc.to_parquet(scored_p, index=False)
        rj.to_parquet(rejected_p, index=False)
        un.to_parquet(uni_p, index=False)
    except Exception:
        pass  # 落盘失败不影响主流程
    return sc, rj, un


def warm(store, asofs, cfg, with_valuation=True, log=None):
    """预热点：对给定 asof 列表依次 screen_at，结果落盘（供后续多档回测/报告秒级读取）。

    log: 可选 callable(str)，用于进度输出（后台运行时写日志）。
    """
    from ..backtest import engine as bt

    n = len(asofs)
    for i, a in enumerate(asofs, 1):
        try:
            bt.screen_at(store, a, cfg, with_valuation=with_valuation)
            if log:
                log(f"[{i}/{n}] 预热 {pd.Timestamp(a).date()} 完成")
        except Exception as e:  # 单个截面失败不阻断其余
            if log:
                log(f"[{i}/{n}] 预热 {pd.Timestamp(a).date()} 失败：{e}")


def clear(store) -> int:
    """清空本库 screen 缓存，返回删除文件数。"""
    cache_dir = Path(store.root) / _CACHE_SUB
    if not cache_dir.exists():
        return 0
    n = 0
    for f in cache_dir.glob("*.parquet"):
        try:
            f.unlink()
            n += 1
        except Exception:
            pass
    return n
