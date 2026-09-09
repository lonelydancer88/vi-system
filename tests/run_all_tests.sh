#!/bin/bash
# 一键跑全部测试：冒烟 → 分层单测 → 正确性（真实库）。
# 任一失败即以非零码退出。
cd /Users/hpl/WorkBuddy/2026-09-07-15-54-23 || exit 1
set -o pipefail

echo "############ 1/3 smoke_test（端到端 invariants）"
python3 -m tests.smoke_test || { echo "[FAIL] smoke_test"; exit 1; }

echo
echo "############ 2/3 test_layers（L0~L8 分层单测，合成库）"
python3 -m tests.test_layers || { echo "[FAIL] test_layers"; exit 1; }

echo
echo "############ 2.5/3 test_quarterly（季报双轨 TTM 单测）"
python3 -m tests.test_quarterly || { echo "[FAIL] test_quarterly"; exit 1; }

echo
echo "############ 3/3 correctness_test（真实库不变量与回归锚点）"
python3 -m tests.correctness_test || { echo "[FAIL] correctness_test"; exit 1; }

echo
echo "ALL TESTS PASSED ✅"
