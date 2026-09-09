import sys, json, time, subprocess, datetime
sys.path.insert(0, '.')
import pandas as pd
from vi_system.data.store import Store

st = Store("data/real_universe")
cfg = json.load(open("data/real_universe/codes.json"))
ALL = set(cfg["all"])

def coverage():
    px = st.load_prices()
    have = set(px["code"].unique())
    missing = [c for c in ALL if c not in have]
    return len(have), len(missing)

def backfill_alive():
    out = subprocess.run(["pgrep", "-f", "fetch_prices_extra"],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())

print(f"[{datetime.datetime.now()}] monitor start; ALL={len(ALL)}", flush=True)
# 只要补齐进程还活着，就持续等待（每30s观测一次覆盖数）
while backfill_alive():
    have, miss = coverage()
    print(f"[{datetime.datetime.now()}] [alive] have={have} missing={miss}", flush=True)
    time.sleep(30)

print(f"[{datetime.datetime.now()}] backfill process gone, settling 45s...", flush=True)
time.sleep(45)   # 让最后一批 save 落盘
have, miss = coverage()
print(f"final coverage: have={have} missing={miss}", flush=True)

if have >= 780:
    print("Universe expanded, rerunning screen at 2026-09-07 ...", flush=True)
    subprocess.run([sys.executable, "-m", "vi_system.cli", "--db", "data/real_universe",
                    "screen", "--asof", "2026-09-07", "--out", "out", "--top", "30"],
                   check=False)
    print("screen done", flush=True)
else:
    print(f"coverage {have} below 780 threshold, skip screen rerun", flush=True)

with open("out/backfill_done.md", "w") as f:
    f.write("# 行情补齐监控报告\n\n")
    f.write(f"- 监控结束时间: {datetime.datetime.now()}\n")
    f.write(f"- prices 覆盖: {have} 只（目标 {len(ALL)} 只）\n")
    f.write(f"- 仍缺行情: {miss} 只\n")
    if have >= 780:
        f.write(f"- 宇宙已显著扩大，已重跑 screen（见 out/portfolio-2026-09-07.md 顶部「宇宙 N 只」）。\n")
    else:
        f.write(f"- 覆盖数 {have} 未达 780，未自动重跑 screen（补齐可能未完成，需重跑 fetch_prices_extra.py）。\n")
print("SUMMARY written", flush=True)
