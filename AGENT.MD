# TaxBrain: Modular Quantitative Analysis Framework

> **Note for Stripe / Compliance:** This repository hosts the source code for a technical software architecture tailored for data analysis and backtesting. This is a software product for developers and researchers. No financial services, custody of funds, or investment advice are provided here.
---

# AGENTS.md — TaxBrain/Codex 多代理架构规范（无版本约束）

> 适用对象：顶级架构师 / 量化研究员 / 程序员（Freqtrade + Python）
> 目标：将 **状态机（健康度/Tier）**、**财政税制（VaR 预算与分配）**、**预约（并发控制/TTL）**、**一次一拍 Finalize（账本推进/清债/不变式校验）** 以**多代理**形态标准化，支撑**多币种 7×24**稳定运行与持续演进。

---

## 0. 原则与术语

* **风险统一度量**：一切预算与占用都以 **VaR（名义×SL%)**表示，组合 CAP、单票 CAP 统一以 VaR 约束。
* **财政二级拨款**：**fast（集中火力）** + **slow（广域薄铺）**；疼痛加权优先回血。
* **预约先行**：下单前必须先占用 VaR；成交转正，撤/拒/超时自动释放。
* **一次一拍**：每根 bar（或超时）进行 **Finalize**：冷却/ICU 递减、债务衰减、财政重算、盈利周期清债、全局不变式校验。
* **出场安全阀**：止盈、保本抬止损（ATR% ε）、方向翻转、压力期 Risk-Off、ICU 倒计到零强平。
* **多币种一等公民**：**每个交易对独立 PairState**，但共享**组合层 CAP/财政池/债务池**。
* **可回复/可重放**：所有关键状态**快照化**，恢复后能继续推进而无账本漂移。

---

## 1. 顶层 Agent 图（职责单一，可替换）

```
Data/DP → SignalAgent → TierAgent → Router
                             │best
                             v
                TreasuryAgent (plan fast/slow) → SizerAgent → ReservationAgent
                                                             │ reservation_id
                                 ExecutionAgent <── fills/cancel/reject ──┐
                                          │                                 │
                                          v                                 │
                                      Ledger/State ──→ ExitAgent → RiskAgent│
                                          │                   │             │
                                          └──────── CycleAgent(Finalize) ───┘
                                                       │
                                                  PersistAgent
                                                       │
                                                  Analytics/Mon
```

> Router 通常由策略类承载（把 Freqtrade 钩子事件路由到各 Agent）。

---

## 2. 目录建议

```
user_data/strategies/
  ├─ TaxBrain.py                # Orchestrator + Router（与 Freqtrade 钩子对接）
  ├─ agents/
  │   ├─ signal.py              # SignalAgent
  │   ├─ tier.py                # TierAgent (+ TierPolicy/健康度状态机)
  │   ├─ treasury.py            # TreasuryAgent（VaR 预算、fast/slow 分配）
  │   ├─ sizer.py               # SizerAgent（名义/风险/桶位）
  │   ├─ reservation.py         # ReservationAgent（预约/TTL/释放）
  │   ├─ execution.py           # ExecutionAgent（交易所适配/统一回调）
  │   ├─ exit.py                # ExitAgent（tp/flip/riskoff/icu/锁盈）
  │   ├─ cycle.py               # CycleAgent（Finalize/清债/心跳）
  │   ├─ risk.py                # RiskAgent（不变式与审计）
  │   ├─ persist.py             # PersistAgent（快照/恢复/兼容字段）
  │   └─ analytics.py           # Analytics/Mon（指标、报表、告警）
  └─ config/
      └─ config.py              # 统一参数中心（可被 strategy_params 覆盖）
```

---

## 3. Agent 职责与接口（坚固的最小接口）

### 3.1 SignalAgent

* **入**：已含 EMA/RSI/ATR/ADX 的 K 线窗口（由 Router 调用 `populate_indicators` 后传递）。
* **出**：`List[Candidate]`（字段：direction/kind/sl_pct/tp_pct/raw_score/rr_ratio/win_prob/expected_edge/squad）。
* **要点**：

  * **ADX 列名随配置**：`ADX_{adx_len}` 不存在则回退常数（如 20）。
  * 风险与收益常用 **ATR% 归一化**，`expected_edge = win*tp - (1-win)*sl` 排序。

### 3.2 TierAgent（健康度状态机）

* **入**：`closs`、`List[Candidate]`。
* **出**：`best_candidate | None`。
* **策略**：按 `TierPolicy` 过滤（允许的 kind、`min_edge`、`min_rr_ratio`），管理 **冷却**与**ICU 倒计时**基线。

### 3.3 TreasuryAgent（财政/分配）

* **入**：组合权益 `equity`、`debt_pool`、各 Pair 的 `last_*`、CAP 参数、`local_loss` 疼痛。
* **出**：`AllocationPlan {fast_alloc_risk, slow_alloc_risk}`。
* **规则**：

  * 动态组合 CAP：当 `debt_pool/equity` 超阈，**CAP 降档**；压力期可**抑制 baseline VaR**。
  * fast：按 **squad 代表 Top-K** 等额；slow：前 M 覆盖。
  * **最小名义注入 → VaR 下限**（fast/slow 各自）；再与 **单票 CAP** 相交裁剪。
  * **疼痛加权**：`score *= 1 + min(cap, local_loss / (norm*equity))`，优先回血。

### 3.4 SizerAgent（头寸计算）

* **入**：`sl_pct/tp_pct/dir`、财政拨款、TierPolicy、组合/单票 CAP、交易所 min/max。
* **出**：`stake_nominal, real_risk, bucket`。
* **逻辑**：

  * **baseline VaR**：`k_mult_base_pct * equity`（压力期可归零）。
  * **Target-Recovery**：`risk_rec = (local_loss*recovery_factor)*(sl_pct/tp_pct)`。
  * 合并：`risk_wanted = max(base_or_rec, alloc_total)` → 与 **CAP** 逐级 min → `stake = VaR/sl_pct`。

### 3.5 ReservationAgent（预约/并发防抖）

* **入**：`pair, risk, bucket, ttl`。
* **出**：`reservation_id`（唯一）。
* **规则**：
  成交(open)→释放预约并把风险转入在市 VaR；撤/拒/TTL 到→释放；**不回灌财政表**（由下一拍财政重算）。

### 3.6 ExecutionAgent（下单/成交）

* **入**：`order request`（含 `reservation_id`、`stake_nominal`、`sl_pct/tp_pct`）。
* **出**：统一回调（`order_filled/cancelled/rejected`）。
* **职责**：对接交易所/合约规则（精度、最小名义、步进、强平/资金费率查询等）。

### 3.7 ExitAgent（出场）

* **触发**：在 `custom_stoploss/custom_exit` 由 Router 调用。
* **策略**：`tp_hit` / **早锁盈（保本抬止损，ATR% ε）** / `flip_dir` / 压力期 `risk_off` / `icu_timeout`。
* **保障**：`tp_pct` 必可取得（custom_data → user_data → active_meta 多层兜底）。

### 3.8 CycleAgent（Finalize/心跳）

* **触发**：每根 bar 或超时（`force_finalize_mult * timeframe_sec`）。
* **动作**：冷却&ICU 递减、债务衰减、财政重算、**盈利周期清债**（周期到且 `PnL ≥ 0` 时清空 `debt_pool` 与各 `local_loss`）、不变式校验、持久化。

### 3.9 RiskAgent（不变式）

* **校验**：

  * `Σ in-market VaR + reserved_portfolio_risk ≤ equity * dynamic_portfolio_cap_pct`
  * `∀pair: pair_risk ≤ equity * per_pair_risk_cap_pct`
  * 预约 TTL 无泄漏；重复释放为 0；单对并发不超限（若策略限定）。

### 3.10 PersistAgent（持久化）

* **对象**：`GlobalState`+`EquityProvider`；
* **要求**：**兼容旧字段**（如 `local_debt/global_loss_bucket`），恢复后**账本无漂移**。

### 3.11 Analytics/Mon（监控）

* **KPI**：`stress=debt/equity`、`risk_utilization`、`allocator_hit_rate`、`reservation_expiry_rate`、`recovery_efficiency`、`icu_trigger_rate`、`cycle_pnl`、最大回撤、Sharpe。
* **事件日志**：Finalize 摘要、财政分配、预约创建/释放、成交/撤/拒、退出理由、清债触发、CAP 告警。

---

## 4. Freqtrade 钩子映射（Router 速查）

| Hook                       | 流程                                                                                                | 产出/副作用                           |
| -------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------- |
| `bot_start`                | PersistAgent.load → CycleAgent 初始化周期锚点                                                            | 状态恢复、周期起点                        |
| `populate_indicators`      | SignalAgent → TierAgent → 更新 `PairState.last_*` → CycleAgent.maybe_finalize（含 TTL 回收/清债/财政重算/不变式） | 最新信号缓存、一次一拍、持久化                  |
| `populate_entry_trend`     | 写入最后一根 K 的方向位                                                                                     | `enter_long/enter_short`         |
| `confirm_trade_entry`      | 冷却/并发闸门 + 方向一致性                                                                                   | 允许/拒绝                            |
| `custom_stake_amount`      | SizerAgent → ReservationAgent（创建预约）                                                               | `stake_nominal`、`reservation_id` |
| `leverage`                 | 固定或策略返回                                                                                           | 一般为 1.0                          |
| `order_filled`             | 释放预约→记账→登记 active_trade（含 ICU 倒计）                                                                 | 在市 VaR 刷新、冷却刷新                   |
| `order_cancelled/rejected` | **仅释放预约**（不回灌财政）                                                                                  | 预约清理                             |
| `custom_stoploss`          | ExitAgent（保本抬止损或兜底 SL）；可从 DP 再取 ATR%                                                              | 负距离                              |
| `custom_exit`              | ExitAgent（`tp_hit/flip/risk_off/icu_timeout`）                                                     | 退出标签                             |

---

## 5. 核心 Schema（统一接口契约）

**Candidate**

```json
{"direction":"long|short","kind":"mean_rev_long|pullback_long|trend_short",
 "sl_pct":0.0123,"tp_pct":0.0246,"raw_score":0.31,"rr_ratio":2.0,
 "win_prob":0.62,"expected_edge":0.0074,"squad":"MRL|PBL|TRS"}
```

**AllocationPlan**

```json
{"fast_alloc_risk":{"BTC/USDT":45.0},"slow_alloc_risk":{"ETH/USDT":10.0}}
```

**Reservation**

```json
{"id":"pair:bucket:uuid","pair":"BTC/USDT","risk":12.34,"bucket":"fast|slow","ttl_bars":6}
```

---

## 6. 配置面板（可由 `strategy_params` 覆盖）

* **时间/周期**：`timeframe`, `startup_candle_count`, `force_finalize_mult`, `cycle_len_bars`, `clear_debt_on_profitable_cycle`
* **组合 CAP**：`portfolio_cap_pct_base`, `drawdown_threshold_pct`
* **财政**：`treasury_fast_split_pct`, `fast_topK_squads`, `slow_universe_pct`, `min_injection_nominal_fast/slow`
* **债务/衰减**：`tax_rate_on_wins`, `pain_decay_per_bar`
* **早锁盈**：`breakeven_lock_frac_of_tp`, `breakeven_lock_eps_atr_pct`
* **指标**：`ema_fast/ema_slow/rsi_len/atr_len/adx_len`（**ADX 列名与 adx_len 动态一致**）
* **运行**：`enforce_leverage`, `dry_run_wallet_fallback`

> **多币种注意**：`current_whitelist()` 为一次一拍的「全票清单」。新增/移除交易对时，以 **超时 finalize** 作为兜底，避免阻塞。

---

## 7. 不变式（必须长期为真）

1. **风险上限**：
   `Σ(pair_risk_open) + reserved_portfolio_risk ≤ equity * dynamic_portfolio_cap_pct`
   `∀pair: get_pair_risk(pair) ≤ equity * per_pair_risk_cap_pct(tier)`
2. **预约正确性**：
   同一 `reservation_id` 只能释放一次；TTL 到期必释放；撤/拒不回灌财政。
3. **并发约束**：
   每对并发订单数量遵循策略约束（当前为 ≤1）。
4. **类/实例一致性**：
   `self.timeframe == cfg.timeframe` 且 `startup_candle_count` 同步。

---

## 8. 开发/测试规范（Codex 友好）

* **单测**：
  `SizerAgent`（CAP/最小注入/压力期抑制）、`TreasuryAgent`（fast/slow/疼痛/裁剪）、`ReservationAgent`（预约→成交/撤/拒/TTL）、`CycleAgent`（衰减/清债/不变式）。
* **性质测试（Hypothesis）**：
  任意输入序列下，**组合风险不越 CAP**；快照→恢复→继续推进**无账本漂移**。
* **回测评估**（多币种）：
  `allocator_hit_rate / reservation_expiry_rate / recovery_efficiency / icu_trigger_rate / risk_utilization / cycle_pnl`。
* **日志基线**：
  Finalize 节拍、财政摘要、预约 create/release、成交/撤/拒、退出理由、清债触发、CAP 告警。

---

## 9. 运行与部署（最小流程）

1. 放置 `TaxBrain.py` 与 `agents/*`、`config/config.py` 到 `user_data/strategies/`。
2. `config.json`：

   * `"strategy": "TaxBrain"`
   * `"strategy_params"` 覆盖所需字段（含 `timeframe/startup_candle_count`，并在构造时同步到实例/类属性）。
3. **首跑检查**：
   ADX 动态列名生效；Finalize 周期稳定；预约 TTL 回收正常；盈利周期清债触发；不变式零告警。

---

## 10. 多币种特化建议（关键要稳）

* **PairRegistry**：维护活跃对清单，增删对时触发一次「安全 finalize」。
* **财政配额去重**：同一对**不得同时领 fast 与 slow** 两份（取最大桶位）；或按“快优先、慢补缺”合并。
* **相关性抑制（可选）**：对同簇（高相关币）设置同拍分配上限，降低系统性风险。
* **执行适配**：每对读取交易所精度/最小名义/步进；Sizer 末端落地到可成交名义并校正 VaR 偏差。

---

## 11. 未来演进（不改接口的增强路线）

* **ICU 专席**：财政 fast 预留 1 个 ICU 名额，按 `icu_bars_left` 加权优先级。
* **执行概率加权**：财政分配乘以近窗口 `enter` 命中率，减少“分配→进不去”的浪费。
* **多市场/多交易所**：ExecutionAgent 插拔式适配；风险账本支持 `market_id` 维度。
* **自适应清债阈值**：盈利周期清债由 `≥0` 升级为 `> ε * equity` 的死区，抗噪更稳。

---

## 12. 最小自检（伪码）

```python
cap = equity * dynamic_portfolio_cap_pct()
assert total_inmarket_var() + reserved_portfolio_risk <= cap + 1e-9
for pair in whitelist:
    assert get_pair_risk(pair) <= equity * tier_cap(pair) + 1e-9
for rid in reservations:
    assert not released_twice(rid)
```

---

**说明**：

* 本规范**不绑定版本号**，只约束**交易架构与系统理念**；换核、换库、换交易所均不破坏接口。
* 以 **多币种**为一等目标；财政-预约-状态机三件套是系统生命线；一次一拍是全局一致性的时钟。
* Codex 可按本文件逐 Agent 生成/重写，实现「高内聚、低耦合、可替换」的长期演进。
