# TaxBrain V29 - 分布式微仓马丁系统

基于 Freqtrade 的多智能体策略框架，围绕分布式侦察、局部恢复与全局社会化风控构建。本文档聚焦核心交易哲学与架构要点，帮助快速对齐代码实现。

## 核心交易哲学
- 分布式微仓侦察（Distributed Scouting）：T0 微仓在 100+ 交易对撒网，用大数定律稀释单票黑天鹅。
- 局部马丁回本（Local Recovery）：单币亏损触发 T1-T5 递进加注，将 50% 胜率拉向确定性 98%+，优先在本地消化。
- 全局债务熔断（Socialized Risk）：击穿 Max Tier 即熔断，停止加注，坏账一次性冲入中央债务池。
- 动态分摊还债（Fluid Repayment）：中央债务按信号质量（score²）非线性分摊给当前最健康的“打工仔”，以高胜率修复。
- 概率优势优先（Alpha Gatekeeping）：有债务时收紧门槛，仅放行高胜率信号携带分摊额开仓；保持整体 P>51% 即可持续消债。

## 架构与职责
- Global backend：`global_backend_mode`=`redis` 或 `local`。Redis 共享 `GLOBAL_DEBT/GLOBAL_RISK_USED/SCORES_WINDOW` 与市场偏移，支持多实例一致的债务、容量与分位数门槛。
- Market sensing：`MarketSensor` 计算 BTC/ETH bias+vol，并写入 backend，影响拨款极坐标与 sizing 熵因子。
- Signal pipeline：`agents/signals/builder.py` 汇总注册信号，生成含 SL/TP/score/exit_profile/recipe/ATR 提示的 `Candidate`。
- Tiers & routing：`TierManager` 依 closs 选择 `TierPolicy`（冷却、caps、默认 exit_profile、单票独占）；`TierAgent` 按 min_raw_score/min_rr/min_edge 过滤。
- Gatekeeping：`TreasuryAgent.evaluate_signal_quality` 用统一 `min_score`（可关）判定放行；健康币在无债或 clean 状态可享低门槛。
- Treasury：`TreasuryAgent.plan` 读取 backend snapshot + 本地风险/预约，计算组合 CAP、可用 debt 注入、极坐标 K_long/K_short、theta、volatility、final_r。
- Sizing：`SizerAgent` 结合 tier caps、baseline/target recovery、central fluid（score² 注入）、exchange 最小额，输出 nominal stake 与实际风险；如启用 Redis，建仓前会以 CAP 上限调用 `add_risk_usage`。
- Reservation：`ReservationAgent` 先锁风险（TTL），成交/撤单/过期释放；同时同步 backend risk_used。
- Execution & exits：`ExecutionAgent` 负责下单元数据；`ExitFacade` 解析配置化 exit_profiles + ATR；`ExitPolicyV29` 处理锁盈、flip、risk-off、ICU。
- Cycle & persist：`CycleAgent` 每 bar 衰减债务/冷却/预约 TTL，刷新拨款，盈利周期可选清债；`RiskAgent` 校验组合/单票/预约一致性；`StateStore` 持久化到 JSON（回测/优化自动禁用）。

## 债务与恢复机制
- 本地债务：亏损累加到 `PairState.local_loss`；盈利优先清本地，再还中央（backend.repay_loss）。closs 升级驱动更严 tier/cooldown。
- 熔断与中央债：达到 Max Tier 的亏损转入 `debt_pool` 并 `backend.add_loss`，本地债务清零并重置 closs。
- 流动分摊：`SizerAgent` 将 score² 映射为中心注入（center_algo，默认 TARGET_RECOVERY），把中央债务分配给当前高分标的；本地债务用相同管线求解。
- 衰减与清零：`CycleAgent` 按 `risk.pain_decay_per_bar` 衰减中央债；若配置盈利周期清债，周期为正且盈利时一键归零。

## Gatekeeping 与容量控制
- 评分门槛：`risk.gatekeeping.min_score`（统一快/慢通道），拒绝则记录原因；无债且 clean closs 可用宽松策略（健康通道）。
- Tier 阈值：`TierPolicy` 控制 min_raw_score/min_rr/min_edge、冷却、per_pair_risk_cap_pct、max_stake_notional_pct、icu_force_exit_bars。
- 组合 CAP：`portfolio_cap_pct_base` 随债务率超过 `drawdown_threshold_pct` 折半；backend 风险占用用于多实例乐观锁。
- 预约 TTL：`reservation_ttl_bars` 超时自动释放，不回滚财政。

## 配置入口
- 配置定义：`user_data/strategies/config/v29_config.py`（字段含 docstring）；模板 `user_data/config_template_v29.json`。
- 关键参数：`system.global_backend_mode`/redis 连接；`risk.pain_decay_per_bar`、`drawdown_threshold_pct`、`portfolio_cap_pct_base`、`gatekeeping.min_score`；`trading.treasury.debt_pool_cap_pct`；`strategy.tier_routing/tiers/exit_profiles`；`sizing_algos`（BASE_ONLY/BASELINE/TARGET_RECOVERY）。
- 运行模式：回测/超参强制使用 local backend；实盘多实例需安装 `redis` 依赖并配置 redis host/port/db/namespace。

## 快速提示
- 启用 Redis：配置 `system.global_backend_mode="redis"`，并提供 `redis_host/redis_port/redis_db/redis_namespace`；失败仅降级日志，不中断。
- 日志与持久化：`user_data/logs` 存储 analytics、sizer/tier debug；状态文件 `taxbrain_v29_state.json` 可断点续跑。
- 指标/信号：如需裁剪信号或额外因子，可在 config 中 `enabled_signals`/`extra_signal_factors` 调整，builder 会自动计算依赖的指标与信息周期。
- Vectorized backtest with informative timeframes requires `merge_informative_into_base=true` for consistency.
