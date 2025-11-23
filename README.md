# TaxBrain: Modular Quantitative Analysis Framework

> **Note for Stripe / Compliance:** This repository hosts the source code for a technical software architecture tailored for data analysis and backtesting. This is a software product for developers and researchers. No financial services, custody of funds, or investment advice are provided here.

# TaxBrainV29 Strategy Architecture

TaxBrainV29 是一套基于 Freqtrade 的多代理量化交易策略，实现了信号、分层、财政、预约、风险、执行、持久化等职能解耦的“模块化作战架构”。该仓库整理自 V29.1 版本，保留并显式记录了原策略的五项关键修订：

1. **ADX 动态列名**：当 ADX_{cfg.adx_len} 缺失时回退到 ADX_20，避免指标列名硬编码。
2. **盈利周期清债**：CycleAgent 在 cycle_len_bars 达成且本周期盈利时，清空 debt_pool 与各交易对的 local_loss。
3. **timeframe/startup 同步**：TaxBrainV29 实例化时更新实例与类级别的 	imeframe / startup_candle_count。
4. **早锁盈三重兜底**：custom_stoploss 依次读取 	rade.custom_data → trade.user_data → ActiveTradeMeta，确保止盈数据可用。
5. **预约释放不回滚财政**：撤单/拒单/TTL 过期仅释放 ReservationAgent 的预约，不修改财政拨款的内存状态。

---

## 目录结构

`	ext
user_data/strategies/
├── TaxBrainV29.py           # 策略主路由，连接 Freqtrade hook 与各代理
├── config/
│   └── v29_config.py        # V29Config 数据类与 apply_overrides 帮助函数
└── agents/
    ├── analytics.py         # JSONL/CSV 监控日志采集
    ├── cycle.py             # finalize 节奏控制与盈利周期清债
    ├── execution.py         # 开/平仓、撤单事件处理
    ├── exit.py              # 自定义退出策略（ExitPolicyV29）
    ├── persist.py           # StateStore 持久化
    ├── reservation.py       # 风险预约池管理
    ├── risk.py              # 风险不变式校验
    ├── signal.py            # 指标计算与候选生成
    ├── sizer.py             # 仓位/风险计算（含 TARGET_RECOVERY）
    ├── tier.py              # TierPolicy 管理与候选过滤
    └── treasury.py          # 财政拨款计划（fast/slow 桶分配）

user_data/logs/                # AnalyticsAgent 输出的 jsonl/csv
user_data/taxbrain_v29_state.json  # StateStore 默认持久化路径（可配置）

tests/agents/                  # 针对各代理的单元测试
`

---

## 策略工作流概览

1. **初始化 (TaxBrainV29.__init__)**
   - 读取 strategy_params，通过 pply_overrides 创建 V29Config。
   - 同步 	imeframe 和 startup_candle_count 至实例与类属性。
   - 构建全局状态容器 GlobalState、权益提供器 EquityProvider、StateStore 以及各代理。

2. **指标计算 (populate_indicators)**
   - signal.compute_indicators 写入 EMA/RSI/ATR/ADX 等列。
   - signal.gen_candidates 生成 MRL/PBL/TRS 候选；	ier.filter_best 根据当前 TierPolicy 筛选。
   - cycle.maybe_finalize 检查是否需要对整个 bar 周期执行 inalize。

3. **下单前检查 (confirm_trade_entry, custom_stake_amount)**
   - 验证冷却与方向一致性，将信号元数据缓存至 _pending_entry_meta。
   - sizer.compute 结合 VaR、财政拨款与 CAP 约束计算仓位；
eservation.reserve 锁定风险额度。

4. **成交 & 撤单事件 (order_filled, order_cancelled, order_rejected)**
   - 开仓：execution.on_open_filled 创建 ActiveTradeMeta，释放预约，并写入 trade 自定义数据。
   - 平仓：execution.on_close_filled 回收风险、更新权益与冷却。
   - 撤/拒单：仅释放预约（V29.1 #5）。

5. **自定义止损/退出**
   - custom_stoploss：整合三重兜底与 ExitPolicy 的早锁盈距离。
   - custom_exit：exit.decide 综合止盈命中、信号翻转、风控降压与 ICU 倒计时。

6. **finalize 周期 (cycle.finalize)**
   - 推进 ar_tick、衰减痛感、构造 Treasury 快照。
   - 	reasury.plan 生成 fast/slow 拨款并写回状态。
   - 盈利周期清债逻辑（V29.1 #2）。
   - 
isk.check_invariants 校验 CAP、预约一致性；结果写入 Analytics 日志。
   - persist.save 落盘全局状态。

---

## 关键代理详解

| 代理 | 主要职责 | 输出 | 备注 |
| ---- | -------- | ---- | ---- |
| AnalyticsAgent | 记录 finalize/reservation/exit/invariant 事件 | user_data/logs/v29_analytics.YYYYMMDD.jsonl 与 29_analytics.csv | CSV 包含 ar_tick、pnl、cap_used_pct 等指标 |
| CycleAgent | 管理 bar 周期、盈利清债、风险日志与持久化 | AllocationPlan | 控制 orce_finalize_mult 超时触发 |
| ExecutionAgent | 处理成交/撤单事件，维护 ActiveTradeMeta 与预约池 | - | 确保止损/止盈信息写入 trade |
| ExitPolicyV29 | 判断自定义退出条件与早锁盈距离 | 退出原因字符串/止损距离 | 顺序：tp → ICU → flip → 风险降压 |
| ReservationAgent | 锁定/释放风险额度，维护 TTL | 预约快照 | 不回滚财政拨款 |
| RiskAgent | 校验组合 CAP、单票 CAP、TTL 递减、重复释放 | InvariantReport 字典 | 错误时写 WARN 日志 |
| Signal | 指标计算、候选生成 | 候选列表 | ADX 动态列处理 |
| SizerAgent | 计算名义仓位与风险需求 | (stake, risk, bucket) | 支持 TARGET_RECOVERY 模式 |
| TierAgent | 根据 TierPolicy 过滤候选 | 最优 Candidate | CLOSS 越高，策略越保守 |
| TreasuryAgent | 按 fast/slow 桶分配风险预算 | AllocationPlan | 包含疼痛加权、最小注入、CAP 修剪 |
| StateStore | 全局状态持久化 | JSON 文件 | 采用临时文件替换保证原子性 |

---

## 配置参数 (V29Config)

配置分组示例：

- **时间**：	imeframe, startup_candle_count
- **风险上限**：portfolio_cap_pct_base, drawdown_threshold_pct
- **财政拨款**：	reasury_fast_split_pct, ast_topK_squads, slow_universe_pct, min_injection_nominal_fast/slow
- **债务/衰减**：	ax_rate_on_wins, pain_decay_per_bar, clear_debt_on_profitable_cycle, cycle_len_bars
- **早锁盈**：reakeven_lock_frac_of_tp, reakeven_lock_eps_atr_pct
- **Finalize 节奏**：orce_finalize_mult, 
eservation_ttl_bars
- **指标长度**：ema_fast, ema_slow, 
si_len, tr_len, dx_len
- **运行参数**：dry_run_wallet_fallback, enforce_leverage

在 config.json 或 Freqtrade 命令行传入 strategy_params，由 pply_overrides 更新默认值。

---

## 日志与监控

AnalyticsAgent 默认写入：

- **JSONL** (user_data/logs/v29_analytics.YYYYMMDD.jsonl)
  - event=finalize：记录 pnl, debt_pool, cap_used_pct, 
eservations 等。
  - event=reservation：create/
elease/expire。
  - event=exit：退出原因。
  - event=invariant：风险检查结果。
- **CSV** (user_data/logs/v29_analytics.csv)
  - 方便导入 BI 或绘图工具分析 cap_used_pct、cycle_cleared 等 KPI。

---

## 状态持久化与恢复

- 状态文件默认保存到 user_data/taxbrain_v29_state.json。
- ot_start 钩子会尝试加载历史快照，恢复 GlobalState、EquityProvider、ReservationAgent。
- GlobalState.reset_cycle_after_restore() 确保恢复后重新计算周期基准。

---

## 风险不变式校验

RiskAgent.check_invariants 检查：

- 组合风险是否超过 portfolio_cap_pct * equity。
- 任何交易对风险是否超出 TierPolicy 的 per_pair_risk_cap_pct。
- 预约 TTL 是否递减、是否出现负值。
- 同一预约 ID 是否重复释放。

Violation 会记录在 Analytics 日志，同时打印 WARN，便于快速定位问题。

---

## 单元测试

测试文件位于 	ests/agents/，覆盖重点功能：

| 文件 | 测试内容 |
| ---- | -------- |
| 	est_cycle.py | 盈利周期清债、预约 TTL 推进、风险日志触发 |
| 	est_exit.py | ExitPolicy 各分支、custom_stoploss 的三重兜底与早锁盈 |
| 	est_reservation.py | 预约创建/释放/过期及日志记录 |
| 	est_sizer.py | 压力期抑制、TARGET_RECOVERY、CAP & min/max 限制 |
| 	est_timeframe_sync.py | timeframe/startup 同步行为 |
| 	est_treasury.py | 财政拨款的 squad 选择、最小注入、CAP 修剪 |

运行方式：

`ash
pytest tests/agents -q
`

	ests/conftest.py 提供了 freqtrade 与 pandas_ta 的轻量化桩，确保在没有实际依赖的环境下也能执行单元测试。

---

## 使用方式

1. **配置 Freqtrade**
   - 在 config.json 中设置 "strategy": "TaxBrainV29"。
   - 可通过 "strategy_params" 覆盖 V29Config 字段，例如：
     `json
     "strategy_params": {
       "timeframe": "5m",
       "startup_candle_count": 210,
       "adx_len": 21,
       "treasury_fast_split_pct": 0.35
     }
     `

2. **回测 / Dry-run**
   - 常规 Freqtrade 命令均可使用，例如：
     `ash
     freqtrade backtesting -c config.json -s TaxBrainV29 -i 5m --timerange 20240101-20241001
     `

3. **监控输出**
   - 运行期间关注 user_data/logs/v29_analytics.*。
   - cycle_cleared 字段可快速确认盈利周期清债是否触发。

4. **状态管理**
   - 若策略或容器意外退出，重启后会自动从 	axbrain_v29_state.json 恢复。
   - 如需重置状态，清空该文件即可。

---

## 开发与扩展建议

- 添加新指标：在 signal.compute_indicators 内补充列，同时更新候选生成逻辑。
- 新的仓位算法：在 SizerAgent.compute 中根据 	ier_pol.sizing_algo 扩展分支。
- 新财政模型：TreasuryAgent.plan 已预留 TODO 注释，可在 v29.2 里实现 ICU 专席与执行概率权重。
- 新日志：直接调用 AnalyticsAgent 的 log_debug/log_finalize 等接口。
- 在修改核心逻辑时，同步更新 	ests/agents/ 以覆盖新行为。

---

## 参考文档

- [Freqtrade Strategy Documentation](https://www.freqtrade.io/en/latest/strategy-customization/)
- AGENT.MD：原始多代理架构设计说明（本仓库中的 docstring 与 README 以此为准绳）。


## License & Commercial Use

This project is open-source under the MIT License for educational and research purposes.

**For Enterprise & Commercial Inquiries:**
We provide professional customization, strategy optimization, and dedicated support for the TaxBrain architecture. If you wish to use this software for commercial proprietary trading or require a commercial license without open-source restrictions, please contact us.