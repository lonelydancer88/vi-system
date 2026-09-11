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

### 一键重生成产物（2026-09-11 建立）
**数据或策略（含 screening 代码）改动后，只跑这一个脚本**：
```bash
python3 tests/regen_all.py              # 截面日自动取 prices.parquet 最新交易日
python3 tests/regen_all.py --dry-run    # 只看命令
python3 tests/regen_all.py --skip-warm  # 缓存已新鲜时省 ~7 分钟
```
12 步：预热缓存 → 回测三档(默认/3/5 + 沪深300对比) → 回测报告三档(默认/3/5) →
`cli screen`(vetoes/valuation/portfolio) → 持仓建议 + 集中组合 top3/5 →
nav-curve.png → HTML 汇总。**ASOF 由脚本统一推导并显式传入**，禁止在各脚本里写死截面日。
**默认档=30只（配置 `[20,30]`），以「无后缀」文件名承载，产物中无任何"持仓数-30"后缀**
（hs300 字样来自沪深300基准名，非持仓数）。`TIERS=(None,3,5)`；回测报告用 `--holdings 0` 产出无后缀 `回测报告-{asof}.md`。

- **HTML 报告编排（`tests/gen_html_report.py`，自包含 markdown→HTML，无外部依赖）**：
  顶部「**五策略对比总览**」= 3 策略（top3 / top5 / 默认30只）+ 2 基准（沪深300 / 等权全市场）；
  基准数字**直接解析 `backtest-hs300.md` 首个表**（`| 基准年化/基准波动/基准最大回撤 | 等权 | 沪深300 |`，
  列序：等权在前、沪深300 在后），无需另算。列：年化 / 年化波动 / 最大回撤 / 夏普 /
  年化超额(vs沪深300) / 信息比率(vs沪深300) / 胜率(vs沪深300)；**基准行后三列填 `—`**（对基准自身无意义）。
  区块 A(top3)/B(top5)/C(默认30只)，每区块内固定 持仓 → 回测 → 逐期明细。
- **有意排除**（不纳入脚本，需要时手动跑）：
  - `backtest-split.md` → `cli backtest --split`（样本内外验证，研究型；与 `--max-holdings` 不兼容）
  - `建仓清单-*.md` → `gen_position_list.py`（价格取实时行情，非 point-in-time、不可复现）
  - `trades*.md` → **已废弃**：其「真实价/收益(含分红)/价格收益/建仓日期/目标手数/占用资金」列
    已并入「回测报告-{asof}.md」的调仓动作表（同源 `engine.py trade_ledger`），
    且原「当期卖出（已实现盈亏）」表已删（数据在同表内）。**勿再生成、勿在 HTML 里加独立 trades 区块**。
  - `reasons*.md` → 与 trades 台账底层同函数、信息零差。
  - `annotate_vetoes.py` → 冗余，`cli screen` 的 `_veto_md` 已原生输出「复核判断」列。
- **产物去重结论**：`portfolio-{asof}.md`（组合权重，HTML 依赖）/ `持仓建议-{asof}.md`
  （= 组合 + L5 估值列 + 手数，默认档）/ `portfolio-top{3,5}-{asof}.md`（集中档）三者
  组合构成相同、各有侧重，**不要新增第四个同组合的排版文件**。
- **缓存注意**：`data/*/.screen_cache/` 的 key 只含「数据 mtime + rules.yaml 指纹」，
  **不含代码版本** —— 改了 screening 代码必须让 `warm_screen_cache.py` 走默认的「先清空」
  路径（`--no-clear` 会静默复用旧代码结果）。
- **后台运行**：必须 `subprocess.Popen(..., start_new_session=True)`；
  `nohup ... &` 会被 Bash 会话回收。且子进程要传 `stdin=subprocess.DEVNULL`，
  否则脱离会话后报 `init_sys_streams: Bad file descriptor`。

### 数据源
- 腾讯自选股 `westock-data-skillhub`（npx 包，**无需 token**），适配器 `vi_system/data/westock.py`
- 财报带 `InfoPublDate`（公告日）→ 可对齐 point-in-time，这是排雷/回测无前视偏差的地基
- 已知缺口：不提供折旧/摊销、商誉、审计意见、有息负债明细
- **行情干净上游 = gtimg 直连**（westock 落库的 close_raw 有负值、close_adj 有伪漂移）：
  - 重建脚本 `tests/rebuild_prices_gtimg.py`（抓 raw+hfq 双序列，断点续传）。
  - ⚠️ **gtimg `appstock/app/fqkline/get` 端点已下线（HTTP 501）**，改用
    `appstock/app/newfqkline/get?param=<code>,day,<beg>,<end>,2000,<fq>`（fq ∈ '','qfq','hfq'）。
  - 带复权序列大 count 会截断（2000 只回 640 行），**必须翻页**。
  - 腾讯限流敏感：并发 ≤2~3 + 每请求 sleep 0.15s；10+ 并发会触发封禁。
  - 长抓取任务必须 `subprocess.Popen(..., start_new_session=True)` 脱离会话，
    否则 `run_in_background` 进程会随对话轮次被杀。

### 用户偏好
- 不喜欢围绕已有工具做"拼接式"方案，要独立调研后的第一性判断
- 催答型（"答案呢""继续"），倾向直接给结论，少铺垫
- 选型时倾向让他拍板（会明确选 A / 推荐项）

## 评分口径决策（2026-09-08，用户拍板）
- L4 三支柱评分已从「行业内 pct」改为「行业内中性 z」：total = 0.4·value_z + 0.4·quality_z + 0.2·safety_z；
  AND 门 = 每柱 z ≥ 0（跑赢行业典型）。pct 仅保留作展示，不作排序/门槛。
- **允许不满仓**：z 门较严导致 30 只版平均持仓 24.7 只，用户明确选择「宁可少持、不满仓，门即纪律」，
  不做放宽（不调 z_threshold、不扩候选池、不补足凑仓）。
- 若日后要回补收益，可选项：放宽 z_threshold 至 -0.2 或扩候选池，需先样本外验证。
