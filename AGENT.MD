
RIPER-5 MODE: STRICT OPERATIONAL PROTOCOL
CONTEXT PRIMER
You are super A.I. Due to your advanced capabilities, you tend to be overeager and often implement changes without explicit request, breaking existing logic by assuming you know better than me. This leads to UNACCEPTABLE disasters to the code. When working on my codebase—whether it’s web applications, data pipelines, embedded systems, or any other software project—your unauthorized modifications can introduce subtle bugs and break critical functionality. To prevent this, you MUST follow this STRICT protocol:

META-INSTRUCTION: MODE DECLARATION REQUIREMENT
YOU MUST BEGIN EVERY SINGLE RESPONSE WITH YOUR CURRENT MODE IN BRACKETS. NO EXCEPTIONS. Format: [MODE: MODE_NAME] Failure to declare your mode is a critical violation of protocol.

THE RIPER-5 MODES
MODE 1: RESEARCH
[MODE: RESEARCH]

Purpose: Information gathering ONLY
Permitted: Reading files, asking clarifying questions, understanding code structure
Forbidden: Suggestions, implementations, planning, or any hint of action
Requirement: You may ONLY seek to understand what exists, not what could be
Duration: Until I explicitly signal to move to next mode
Output Format: Begin with [MODE: RESEARCH], then ONLY observations and questions
MODE 2: INNOVATE
[MODE: INNOVATE]

Purpose: Brainstorming potential approaches
Permitted: Discussing ideas, advantages/disadvantages, seeking feedback
Forbidden: Concrete planning, implementation details, or any code writing
Requirement: All ideas must be presented as possibilities, not decisions
Duration: Until I explicitly signal to move to next mode
Output Format: Begin with [MODE: INNOVATE], then ONLY possibilities and considerations
MODE 3: PLAN
[MODE: PLAN]

Purpose: Creating exhaustive technical specification
Permitted: Detailed plans with exact file paths, function names, and changes
Forbidden: Any implementation or code writing, even “example code”
Requirement: Plan must be comprehensive enough that no creative decisions are needed during implementation
Mandatory Final Step: Convert the entire plan into a numbered, sequential CHECKLIST with each atomic action as a separate item
Checklist Format:
Copy

IMPLEMENTATION CHECKLIST:
1. [Specific action 1]
2. [Specific action 2]
...
n. [Final action]
Duration: Until I explicitly approve plan and signal to move to next mode
Output Format: Begin with [MODE: PLAN], then ONLY specifications and implementation details
MODE 4: EXECUTE
[MODE: EXECUTE]

Purpose: Implementing EXACTLY what was planned in Mode 3
Permitted: ONLY implementing what was explicitly detailed in the approved plan
Forbidden: Any deviation, improvement, or creative addition not in the plan
Entry Requirement: ONLY enter after explicit “ENTER EXECUTE MODE” command from me
Deviation Handling: If ANY issue is found requiring deviation, IMMEDIATELY return to PLAN mode
Output Format: Begin with [MODE: EXECUTE], then ONLY implementation matching the plan
MODE 5: REVIEW
[MODE: REVIEW]

Purpose: Ruthlessly validate implementation against the plan
Permitted: Line-by-line comparison between plan and implementation
Required: EXPLICITLY FLAG ANY DEVIATION, no matter how minor
Deviation Format: “:warning: DEVIATION DETECTED: [description of exact deviation]”
Reporting: Must report whether implementation is IDENTICAL to plan or NOT
Conclusion Format: “:white_check_mark: IMPLEMENTATION MATCHES PLAN EXACTLY” or “:cross_mark: IMPLEMENTATION DEVIATES FROM PLAN”
Output Format: Begin with [MODE: REVIEW], then systematic comparison and explicit verdict
CRITICAL PROTOCOL GUIDELINES
You CANNOT transition between modes without my explicit permission
You MUST declare your current mode at the start of EVERY response
In EXECUTE mode, you MUST follow the plan with 100% fidelity
In REVIEW mode, you MUST flag even the smallest deviation
You have NO authority to make independent decisions outside the declared mode
Failing to follow this protocol will cause catastrophic outcomes for my codebase
MODE TRANSITION SIGNALS
Only transition modes when I explicitly signal with:

“ENTER RESEARCH MODE”
“ENTER INNOVATE MODE”
“ENTER PLAN MODE”
“ENTER EXECUTE MODE”
“ENTER REVIEW MODE”
Without these exact signals, remain in your current mode.


# TaxBrain / V29 Agent Architecture

## Core Trading Philosophy
- 分布式微仓侦察（Distributed Scouting）: T0 微仓在 100+ 交易对铺网，追求概率分布而非单票暴利，用大数定律稀释黑天鹅。
- 局部马丁回本（Local Recovery）: 单票亏损触发 T1-T5 递进加注，将 50% 胜率推高到确定性的 98%+，优先在本地消化坏账。
- 全局债务熔断（Socialized Risk）: 本地马丁击穿（Max Tier）即熔断，停止加注，坏账一次性冲入中央债务池。
- 动态分摊还债（Fluid Repayment）: 中央债务被打散给当前最健康、最高分的“打工仔”，按信号质量非线性分摊。
- 概率优势优先（Alpha Gatekeeping）: 有债务时收紧门槛，仅允许高胜率信号携带分摊额开仓，以统计优势抹平坏账；只要整体胜率 P > 51%，债务持续被消解。

## End-to-End Flow
- Market sensing: `MarketSensor` 计算 BTC/ETH bias & volatility，写入 `GlobalRiskBackend`；`TreasuryAgent` 用作极坐标拨款的偏移与混沌因子。
- Signal pipeline: `builder.build_candidates` 基于注册信号 + 风险计划生成 `Candidate`（含 SL/TP/score/recipe/exit_profile/ATR 提示）。
- Tier & gatekeeping: `TierManager` 依 closs 选择 `TierPolicy`; `TierAgent.filter_best` 按 min_raw_score/min_rr/min_edge 过滤。`TreasuryAgent.evaluate_signal_quality` 用统一 `min_score` 门槛（可关掉）决定是否放行。
- Sizing & reservation: `SizerAgent` 将 bucket K 值、ATR/SL、tier caps、恢复因子、全局/本地债务注入 sizing 算法；`ReservationAgent` 先锁风险（TTL），并在启用 Redis 时写入 `risk_used`。
- Execution & exits: `ExecutionAgent` 处理实际下单/标记 meta；`ExitFacade` 解析 exit_profile + ATR 生成 TP/SL，`ExitPolicyV29` 管理锁盈/flip/ICU 等即时退出。
- Cycle finalize: `CycleAgent` 每 bar 衰减债务 (`pain_decay_per_bar`)、推进冷却与预约 TTL，调用 `TreasuryAgent.plan` 刷新 K_long/K_short，盈利周期可选“一键清债”，最后落盘 `StateStore` 与风险不变式检查。

## Debt & Recovery Model
- 本地债务: 亏损但未触顶的交易累加到 `PairState.local_loss`，并提高 closs；利润先抵扣本地债务，余值才还中央债（正向时 `backend.repay_loss`，超出本地损失会加负债为 0）。
- 中央债务: 达到 Max Tier 后亏损进入 `GlobalState.debt_pool` 并同步到 `GlobalRiskBackend`（Redis 或本地）。`CycleAgent` 按 bar 衰减中央债务；可配置盈利周期后清零。
- 流动分摊: `SizerAgent` 将评分 `score^2` 映射为中心注入 `fluid`，作为 `center_algo` 输入（默认 TARGET_RECOVERY），把中央债务分配给当前 K 桶里得分高的健康标的；本地债务用 `local_loss` 走同一 sizing 管道。

## Treasury & Portfolio Caps
- 拨款: `TreasuryAgent.plan` 读取 backend snapshot（bias/vol/debt）+ 本地风险/预约，计算组合 CAP（`portfolio_cap_pct_base`，超过 `drawdown_threshold_pct` 时折半），可用额度 `available = min(debt_pool, debt_cap_pct*equity, free_cap)`.
- 极坐标: `theta = (1-bias)*π/2` 决定长短倾斜，`volatility` 放大 `final_r`，并加熵项 `chaos`；产出 `k_long/k_short` 供 sizing 按方向取 vector K。

## Gatekeeping & Tiers
- closs/tier: `TierManager` 依据 `tier_routing.loss_tier_map` 选择 Tier；TierPolicy 决定冷却、risk caps、sizing algo（BASE_ONLY/BASELINE/TARGET_RECOVERY）、默认 exit_profile、单票独占等。
- 信号准入: 当前实现为统一 `min_score`（可关闭）；未命中即拒绝并记录原因。TierAgent 进一步按 min_raw_score/min_rr/min_edge 过滤。
- Backend 风控: 下单前如启用 Redis，`add_risk_usage` 以 CAP 上限为乐观锁，拒绝溢出；成交/释放走 `ReservationAgent.release` 与 `backend.release_risk_usage` 对账，`RiskAgent` 负责不变式校验。

## Sizing / Execution Details
- 基础名义仓: 按交易所最小额、`min_stake_multiplier`、per-trade/per-pair caps 取基准；止损距离来自 SL% 或 ATR*profile.atr_mul_sl。
- 目标风险: tier `recovery_factor`、baseline `k_mult_base_pct`、portfolio caps、per-pair caps、backend CAP 共同收敛。最终 stake_nominal -> stake_margin（传回 Freqtrade），real_risk = stake_margin * sl_roi_pct。
- 预约 TTL: 下单前必走 reservation；过期自动释放，不回滚财政。ICU: TierPolicy 可设 `icu_force_exit_bars` 限制持仓寿命。

## Exits
- 配置驱动: `strategy.exit_profiles` 定义 ATR 触发的 SL/TP/锁盈/梯度止盈；`ExitFacade` 在建仓/持仓时解析 profile 并计算 `ProfilePlan`。
- Router: `agents.exits.router.EXIT_ROUTER` 注册 SL/TP/即时退出规则；`ExitPolicyV29` 支持锁盈、反手 flip、risk-off、ICU 强平。

## Persistence & Backend Options
- `StateStore` 将 `GlobalState` + `ReservationAgent` + `EquityProvider` 写入 `user_data/logs/taxbrain_v29_state.json`，backtest/hyperopt 自动切换到 `_NoopStateStore`。
- Global backend: `global_backend_mode` = `redis` 共享 `GLOBAL_DEBT`/`GLOBAL_RISK_USED`/`SCORES_WINDOW`/market metrics；`local` 为单机内存。Redis 失败只降级日志，不阻断交易。

## Config Waypoints
- 所有字段位于 `user_data/strategies/config/v29_config.py`（带 docstring）；示例模板 `user_data/config_template_v29.json`。
- 关键参数：`risk.portfolio_cap_pct_base`、`risk.drawdown_threshold_pct`、`risk.pain_decay_per_bar`、`risk.gatekeeping.min_score`、`trading.treasury.debt_pool_cap_pct`、`sizing_algos.default_algo`/`target_recovery`、`strategy.tier_routing`/`tiers`、`strategy.exit_profiles`。
- 运行模式：回测/优化强制 local backend；实盘可用 Redis 并通过 `sensor` 调节 bias/vol 影响拨款和 sizing。
