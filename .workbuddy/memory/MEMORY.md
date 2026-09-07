# 项目长期记忆

## 项目：vi_system 价值投资选股系统

位置：`/Users/hpl/WorkBuddy/2026-09-07-15-54-23`

### Git 仓库（2026-09-07 建立）
- **远程**：`git@github.com:lonelydancer88/vi-system`（**私有**）
- GitHub 账号：`lonelydancer88`，`gh` CLI 已登录（ssh 协议，token 含 repo 权限）
- 默认分支：`main`
- 提交身份：hupenglong / hupenglong@kangfuzi.cn

### 版本控制约定
- **`data/` 整个目录已 gitignore**（约 90MB parquet，不入库）。重建方式：
  - `tests/fetch_real.py` 重新抓取真实数据（财务 + 行情，带断点续传）
  - `tests/build_universe.py` 重建宇宙代码表
  - `python -m vi_system.cli demo` 生成合成库
- `out/*.log` 也已忽略（运行时日志）
- 已入库：源码、配置、测试脚本、README、设计文档、out 下的 md/csv 报告、`.workbuddy/memory/`

### CLI 用法要点
- `--db` 是**全局参数**，必须写在子命令**之前**：
  `python -m vi_system.cli --db data/real_universe backtest --split`
  写成 `cli backtest --db X` 会报 `unrecognized arguments`
- 真实数据库目录：`data/real_universe`（807 只财务 + 沪深300 行情）

### 数据源
- 腾讯自选股 `westock-data-skillhub`（npx 包，**无需 token**），适配器 `vi_system/data/westock.py`
- 财报带 `InfoPublDate`（公告日）→ 可对齐 point-in-time，这是排雷/回测无前视偏差的地基
- 已知缺口：不提供折旧/摊销、商誉、审计意见、有息负债明细

### 用户偏好
- 不喜欢围绕已有工具做"拼接式"方案，要独立调研后的第一性判断
- 催答型（"答案呢""继续"），倾向直接给结论，少铺垫
- 选型时倾向让他拍板（会明确选 A / 推荐项）
